# One dataset, multiple GPUs

Prepare your data once, export one bundle, then launch all GPU workers against
that shared directory. You do **not** upload a separate copy for each GPU or
manually divide records into GPU-specific files.

This integration supports **one Linux host with multiple NVIDIA GPUs**. SFT uses
PyTorch DistributedDataParallel (DDP); GRPO/PPO use the pinned verl environment.
Multi-node launch and managed cloud GPU provisioning are not included.

## Upload and prepare once

1. Upload JSONL or CSV through the **self-hosted workbench**, or give the original
   file to `jev-dataops run`. For export-only preparation, disable automatic
   training in the workbench or use `--trainer none` with the CLI.
2. Complete screening and inspect `screening/data_report.json` and `keep.jsonl`.
3. Export that retained file with the GPU count and global batch size below.
4. If preparation and training use different machines, transfer the **whole
   bundle directory once** to the GPU host. All workers on that host read it.
   Install JEV DataOps and the matching training dependencies on the GPU host.

The public [website guide](https://jev-dataops.vercel.app/guide/#training) lets
you choose 1, 2, 4, or 8 GPUs and copy the corresponding commands. It does not
connect to a GPU server. The public browser demo keeps files in the current tab
and is limited to 2 MiB / 1,000 rows. The self-hosted upload limit defaults to
1 GiB and is configurable with `JEV_MAX_UPLOAD_MB`; the CLI can read a local
file without a browser upload. Resumable uploads are not implemented.

## Choose a global batch size

| GPUs on one host | Example `--batch-size` | SFT records per GPU per step |
| --- | --- | --- |
| 1 | 4 | 4 |
| 2 | 8 | 4 |
| 4 | 16 | 4 |
| 8 | 32 | 4 |

`--batch-size` counts records across **all** SFT workers, or input prompts across
the verl job. It must be divisible by `--n-gpus`. SFT accepts global batches up
to 32; verl accepts up to 4,096. The retained training split must contain at
least one full global batch for multi-GPU SFT or RL. Lower batch size to reduce
memory use, keeping at least one record per GPU. For GRPO, `--rollout-n` creates
several responses per prompt and adds compute and memory requirements.

These numbers are configuration examples, not hardware performance promises.
Model size, sequence length, precision, GPU memory, and interconnect all matter.

## Example: four-GPU SFT

Use an existing completed screening output and a fresh bundle directory:

```bash
pip install -e '.[train]'

jev-dataops export-training \
  --input runs/example/screening/keep.jsonl \
  --output runs/sft-4gpu \
  --target sft \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --n-gpus 4 --batch-size 16

jev-dataops launch-training --bundle runs/sft-4gpu --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  jev-dataops launch-training --bundle runs/sft-4gpu --execute
```

For another Python environment, pass `--python /absolute/path/to/venv/bin/python`
to the launcher. See the [environment guide](TRAINING_ENVIRONMENTS.md) for
dependency setup. Local base-model paths must exist on the training host; use a
model ID or arrange the same absolute path when transferring bundles.

The multi-GPU launcher uses `torchrun` to start the number of workers requested by
`--n-gpus`, with one worker per GPU and NCCL to synchronize gradients. At least
that many GPUs must be visible before execution starts. Each worker trains on
a disjoint portion of the fixed training split;
the data is not re-split into train/test for each GPU.

DDP keeps a model replica on each GPU: the model and its per-worker training
state must fit on an individual card. GPU memory is not pooled. The current
multi-GPU SFT implementation drops the final incomplete global training batch
instead of repeating records to pad workers; reports expose this behavior.
Held-out evaluation covers the test records without padding duplicates, and
losses are aggregated by the number of predicted tokens. Only the primary
worker writes the shared model and reports.

## Example: four-GPU GRPO or PPO

Export requires `pip install -e '.[export]'`. Run the launcher from your JEV
DataOps environment, passing `--python` to select the **separate pinned verl
environment**. Follow [its installation steps](TRAINING_ENVIRONMENTS.md).
Each RL record needs the verified reference answer named by `--reward-field`.

```bash
jev-dataops export-training \
  --input runs/example/screening/keep.jsonl \
  --output runs/grpo-4gpu \
  --target verl-grpo \
  --reward-field ground_truth \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --n-gpus 4 --batch-size 16 --rollout-n 4

jev-dataops launch-training \
  --bundle runs/grpo-4gpu \
  --python /absolute/path/to/verl/.venv/bin/python --dry-run

CUDA_VISIBLE_DEVICES=0,1,2,3 jev-dataops launch-training \
  --bundle runs/grpo-4gpu \
  --python /absolute/path/to/verl/.venv/bin/python --execute
```

For PPO, export into a new directory with `--target verl-ppo`; PPO also runs a
critic. Both use one shared Parquet train split and one validation split. The
test split stays reserved for independent final evaluation. Validation batches
are bounded by the configured global batch size.

The current verl recipe uses FSDP for training and vLLM tensor parallel size 1
for rollout. Each rollout replica therefore still needs the model to fit on
one GPU. JEV ingestion, screening, and export use bounded processing; verl's
own Hugging Face dataset loader has separate host memory and cache requirements.

## Check a run

Inspect `launch_report.json`, `training.log`, and the target-specific reports
or checkpoints in the output directory. A dry run validates data checksums and
shows the launch command; it does not verify CUDA or claim training success.
Re-export into a fresh directory to change GPU count or objective. Keep the
source file, grouping identifiers, and seed unchanged to preserve split
assignments. A new GPU count changes execution, not the dataset partition.

The default one epoch and 20-step cap are small workflow checks. Increase them
deliberately after verifying a small run. CPU/Gloo smoke tests check distributed
logic; they do not establish CUDA compatibility or multi-GPU throughput for
your hardware.

Implementation references: [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html),
[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
and [verl v0.9.1](https://github.com/verl-project/verl/tree/v0.9.1).
