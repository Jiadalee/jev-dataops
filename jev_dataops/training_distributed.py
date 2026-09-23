"""Single-node DDP LoRA SFT over one immutable prepared bundle.

The public launcher always selects NCCL/CUDA. Explicit Gloo execution exists for
offline CPU integration tests; it is never an automatic fallback for missing GPUs.
No process loads the full dataset, and only rank zero writes shared artifacts.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path


def _rank_batches(path: Path, rank: int, world_size: int, batch_size: int,
                  limit: int | None = None):
    """Disjoint, ordered streaming shards; no padded or duplicated records."""
    from .training import _rows

    batch = []
    for index, row in enumerate(_rows(path)):
        if limit is not None and index >= limit:
            break
        if index % world_size == rank:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _encode_rows(tokenizer, rows: list[dict], config: dict, device):
    """Use the existing SFT rendering/masking contract on each local microbatch."""
    from .training import _render

    ids, cuts = [], []
    for row in rows:
        full, prompt, templated = _render(tokenizer, row)
        sequence = tokenizer(full, truncation=True, max_length=config["max_seq_length"] - 1,
                             add_special_tokens=not templated)["input_ids"]
        ids.append(sequence + [tokenizer.eos_token_id])
        cut = 0
        if config["loss_mask"] == "answer" and prompt is not None:
            prompt_ids = tokenizer(prompt, add_special_tokens=not templated)["input_ids"]
            if sequence[:len(prompt_ids)] == prompt_ids and len(prompt_ids) < len(sequence):
                cut = len(prompt_ids)
        cuts.append(cut)
    batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    for index, cut in enumerate(cuts):
        labels[index, :cut] = -100
    batch["labels"] = labels
    stats = [sum(bool(cut) for cut in cuts), sum(cuts), int((labels[:, 1:] != -100).sum())]
    return {key: value.to(device) for key, value in batch.items()}, stats


def _loss_sum(model, batch):
    """Summed causal NLL is finite even on a rank with no supervised tokens."""
    import torch.nn.functional as functional

    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"][:, 1:].contiguous()
    return functional.cross_entropy(logits[:, :-1].contiguous().float().view(-1, logits.shape[-1]),
                                    labels.view(-1), ignore_index=-100, reduction="sum")


def _on_rank_zero(function, rank: int, dist):
    """Propagate rank-zero preparation/save failures before peers await a barrier."""
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {"value": function()}
        except Exception as exc:
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]["value"]


def run_distributed_sft(bundle: Path, output: Path, *, backend: str = "nccl") -> dict | None:
    import torch
    import torch.distributed as dist
    import transformers
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from .training_launch import _config, prepare_sft_bundle, write_sft_report

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (world < 2 or not 0 <= rank < world or rank != local_rank
            or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != world
            or int(os.environ.get("GROUP_RANK", "0")) != 0):
        raise ValueError("Use torchrun with at least two workers on exactly one node.")
    if backend == "nccl":
        if os.environ.get("JEV_TRAIN_DEVICE", "auto") not in ("auto", "cuda"):
            raise ValueError("Multi-GPU SFT requires JEV_TRAIN_DEVICE=auto or cuda.")
        if not dist.is_nccl_available() or not torch.cuda.is_available() or torch.cuda.device_count() < world:
            raise RuntimeError("NCCL and one visible CUDA device per worker are required.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif backend == "gloo":
        if not dist.is_gloo_available():
            raise RuntimeError("The explicit CPU test requires Gloo support.")
        torch.set_num_threads(1)
        device = torch.device("cpu")
    else:
        raise ValueError("Supported backends are nccl (GPU) and gloo (explicit CPU test).")
    directory, destination = Path(bundle).resolve(), Path(output).resolve()
    dist.init_process_group(backend, init_method="env://", timeout=timedelta(seconds=300))
    try:
        manifest, config, counts = _on_rank_zero(lambda: prepare_sft_bundle(directory, destination), rank, dist)
        if _config(manifest)["n_gpus"] != world:
            raise ValueError("torchrun worker count must equal the bundle's n_gpus.")
        global_batch = config["batch_size"]
        local_batch = global_batch // world
        usable_rows = counts["train"] // global_batch * global_batch
        planned_steps = min(config["max_steps"], config["epochs"] * (usable_rows // global_batch))
        set_seed(config["seed"])
        local_only = os.environ.get("JEV_MODEL_LOCAL_ONLY", "0") == "1"
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=False, local_files_only=local_only)
        if tokenizer.eos_token_id is None:
            raise ValueError("The configured causal tokenizer must define an EOS token.")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        bf16 = torch.tensor(int(backend == "nccl" and torch.cuda.is_bf16_supported()), dtype=torch.int32, device=device)
        # Mixed-generation hosts must still load identical parameter dtypes on
        # every rank for DDP broadcasts and gradient collectives to agree.
        dist.all_reduce(bf16, op=dist.ReduceOp.MIN)
        dtype = torch.bfloat16 if bf16.item() else torch.float32
        loading = {"trust_remote_code": False, "use_safetensors": True, "local_files_only": local_only}
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        loading["dtype" if version >= (4, 56) else "torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(config["base_model"], **loading)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model_limit = getattr(model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and config["max_seq_length"] > model_limit:
            raise ValueError(f"max_seq_length exceeds the model limit of {model_limit}.")
        model.to(device)

        def evaluate(split: str) -> tuple[float, int, int]:
            # No DDP forward here: ranks may have unequal batch counts, including
            # zero. The only collective is after every rank finishes its own shard.
            model.eval()
            loss_sum, token_count, row_count = 0.0, 0, 0
            with torch.no_grad():
                for rows in _rank_batches(destination / f"{split}.jsonl", rank, world, local_batch):
                    batch, stats = _encode_rows(tokenizer, rows, config, device)
                    loss_sum += float(_loss_sum(model, batch).detach().cpu())
                    token_count += stats[2]
                    row_count += len(rows)
            totals = torch.tensor([loss_sum, token_count, row_count], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            total_loss, tokens, records = totals.tolist()
            if not math.isfinite(total_loss) or tokens <= 0 or int(records) != counts[split]:
                raise ValueError(f"Invalid distributed {split} evaluation totals.")
            return total_loss / tokens, int(tokens), int(records)

        baseline_validation, validation_tokens, validation_rows = evaluate("validation")
        baseline_test, test_tokens, test_rows = evaluate("test")
        set_seed(config["seed"])
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
            r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05, bias="none"))
        ddp = DistributedDataParallel(model, device_ids=[local_rank] if backend == "nccl" else None,
                                      broadcast_buffers=False)
        # Different dropout streams, identical adapter initialization and updates.
        set_seed(config["seed"] + rank)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.0)
        warmup = max(1, planned_steps // 10)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup
            else max(0.0, (planned_steps - step) / max(1, planned_steps - warmup)))
        step, seen, skipped, masking, losses = 0, 0, 0, [0, 0, 0], []
        for epoch in range(config["epochs"]):
            ddp.train()
            for rows in _rank_batches(destination / "train.jsonl", rank, world, local_batch, usable_rows):
                if step >= config["max_steps"]:
                    break
                batch, stats = _encode_rows(tokenizer, rows, config, device)
                tokens = torch.tensor(stats[2], dtype=torch.int64, device=device)
                dist.all_reduce(tokens)
                if tokens.item() == 0:
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                summed_loss = _loss_sum(ddp, batch)
                finite = torch.isfinite(summed_loss).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("A worker returned a non-finite training loss.")
                # DDP averages gradients across ranks. Undo that averaging before
                # dividing by the global supervised token count, not by examples.
                (summed_loss * world / tokens).backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                schedule.step()
                aggregate_loss = summed_loss.detach().to(torch.float64)
                dist.all_reduce(aggregate_loss)
                step += 1
                seen += len(rows)
                masking = [a + b for a, b in zip(masking, stats)]
                if rank == 0:
                    event = {"step": step, "epoch": epoch + 1, "loss": aggregate_loss.item() / tokens.item(), "lr": learning_rate}
                    losses.append(event)
                    print(json.dumps({"stage": "training", **event}), flush=True)
            if step >= config["max_steps"]:
                break
        if step < 1:
            raise ValueError("No optimizer step was possible from the training tokens.")
        validation_loss, final_validation_tokens, _ = evaluate("validation")
        test_loss, final_test_tokens, _ = evaluate("test")
        if (validation_tokens, test_tokens) != (final_validation_tokens, final_test_tokens):
            raise RuntimeError("Held-out token counts changed during training.")
        stats_tensor = torch.tensor([seen, *masking], dtype=torch.int64, device=device)
        dist.all_reduce(stats_tensor)
        global_seen, masked_rows, prompt_tokens, learned_tokens = stats_tensor.tolist()

        def save():
            model.save_pretrained(destination / "model", safe_serialization=True)
            tokenizer.save_pretrained(destination / "model")
            report = {
                "mode": "llm_lora_ddp", "trainer": "huggingface", "model_type": "causal_lm_lora", "is_llm": True,
                "base_model": config["base_model"], "base_model_revision": getattr(model.config, "_commit_hash", None),
                "baseline_loss": baseline_test, "trained_loss": test_loss,
                "baseline_validation_loss": baseline_validation, "trained_validation_loss": validation_loss,
                "evaluation_tokens": {"validation": validation_tokens, "test": test_tokens},
                "evaluation_rows": {"validation": validation_rows, "test": test_rows},
                "steps": step, "training_record_visits": global_seen, "loss_history": losses,
                "metric_unit": "natural-log NLL per supervised causal tokenizer token",
                "device": "cuda" if backend == "nccl" else "cpu", "dtype": str(dtype).replace("torch.", ""),
                "gpu_validated": backend == "nccl", "chat_template": bool(tokenizer.chat_template), "loss_mask": config["loss_mask"],
                "masking": {"rows_with_masked_prompt": masked_rows, "prompt_tokens_masked": prompt_tokens, "tokens_learned": learned_tokens},
                "lora": {"r": config["lora_r"], "alpha": config["lora_alpha"], "dropout": 0.05, "target_modules": "all-linear"},
                "schedule": {"warmup_steps": warmup, "decay": "linear", "peak_learning_rate": config["learning_rate"]},
                "trainable_parameters": sum(p.numel() for p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "distributed": {"backend": backend, "world_size": world, "nodes": 1, "global_batch_size": global_batch,
                    "per_rank_batch_size": local_batch, "training_rows_per_epoch": usable_rows,
                    "dropped_training_rows_per_epoch": counts["train"] - usable_rows,
                    "training_record_visits_per_rank": seen, "evaluation_padding_duplicates": 0,
                    "skipped_batches_without_tokens": skipped, "gradient_reduction": "global supervised-token mean",
                    "sharding": "row_index modulo world_size; no replication within a global batch"},
                "limitations": ["One complete base-model replica must fit on each device; this is DDP, not model sharding.",
                    "Only single-node execution is supported. Row order is preserved; incomplete global training batches are dropped.",
                    "Sequence tails beyond max_seq_length are truncated. Masking follows the single-process SFT contract.",
                    "Held-out NLL does not establish task accuracy, fairness, factuality, or production readiness.",
                    "Explicit Gloo CPU runs validate distributed logic only; they do not validate GPU execution."],
            }
            return write_sft_report(directory, destination, manifest, config, counts, report)

        report = _on_rank_zero(save, rank, dist)
        return report if rank == 0 else None
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal single-node distributed SFT process")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    args = parser.parse_args()
    run_distributed_sft(args.bundle, args.output, backend=args.backend)


if __name__ == "__main__":
    main()
