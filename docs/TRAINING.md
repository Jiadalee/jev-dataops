# Training and honest evaluation

The pipeline implements two real training modes. `demo` is a small UTF-8 byte-bigram
count model for checking the complete workflow on a CPU without model downloads.
It is **not an LLM, a JEV model, or evidence of LLM quality**. `huggingface` performs
actual causal-language-model fine-tuning with Transformers, PyTorch, and PEFT LoRA.
Both modes report observed baseline and trained losses; neither assumes improvement.

## Python API

```python
from pathlib import Path
from jev_dataops.training import train_and_evaluate

report = train_and_evaluate(
    Path("keep.jsonl"), Path("run/training"),
    {"trainer": "demo", "epochs": 1, "max_steps": 20, "seed": 42},
    progress=lambda event: print(event),
    cancelled=lambda: False,
)
```

The source is JSONL, one object per line. Supported text fields are `text`, textual
`messages` (arrays or CSV JSON-encoded arrays), `instruction`/`input`/`output`, and
`prompt`/`response`. Training uses the screening module's shared schema and selection
precedence: messages first, then text, then instruction/output, then prompt/response.
Messages are serialized with role labels for full-text training. Image/audio message content is unsupported.
A line is limited to 8 MiB and training text to 1,000,000 characters. The output
directory must not already contain training artifacts. Cancellation raises
`TrainingCancelled`; invalid inputs and failed training never produce a success report.

## Splits and leakage

Preparation streams records into a disk-backed SQLite index with an 8 MiB cache.
Records connected by either `group_id`, `conversation_id`, or identical text after
whitespace normalization remain together. This also joins duplicate content across
different declared groups, including transitive connections. Without supplied groups,
content hashes define the independent units. At least **six independent components**
are required. Validation and test each receive at least one component; training
receives the remainder. The default fractions are 15% and 15% of components, so
record percentages may differ when groups have unequal sizes.

Assignment is deterministic for the same content, grouping, seed, and fractions,
including reordered input. `_split_group` in the output records allows auditing.
The temporary SQLite index is removed afterward. Preparation uses disk proportional
to the data; the SQLite spool and materialized splits temporarily coexist. Exact
deduplication does not identify paraphrases, shared documents with different text,
temporal dependencies, or other semantic leakage. Supply meaningful group IDs and
use a genuinely independent external benchmark for production decisions.

## Demo mode

The count model has 256 UTF-8 byte symbols and an EOS symbol, plus a BOS context.
It learns bigram counts on the training split with additive smoothing of 0.5.
The untrained baseline is uniform over 257 symbols. All evaluation is performed
on the same held-out validation and test records before and after training.
`max_seq_length` limits UTF-8 bytes per record in this mode. `max_steps` limits
batches; `epochs` limits passes. `learning_rate` does not apply to count updates.
The saved `model/byte_bigram.json` contains all counts needed to reproduce the
reported likelihoods. Memory for model counts is fixed, independent of corpus size.

## Hugging Face LoRA mode

Install the optional training dependencies and configure the model as the server
operator before starting a run:

```bash
pip install '.[train]'
export JEV_BASE_MODEL=HuggingFaceTB/SmolLM2-135M
# Optional: load only previously downloaded or local model assets.
export JEV_MODEL_LOCAL_ONLY=1
# Optional: cpu, cuda, mps, or auto (auto selects CUDA when available, otherwise CPU).
export JEV_TRAIN_DEVICE=cpu
```

`JEV_BASE_MODEL` is the single model allowlist entry and defaults to
`HuggingFaceTB/SmolLM2-135M`, a small smoke-test model. An API-supplied `base_model`
must match it exactly. Operators can supply a local model directory. Without
`JEV_MODEL_LOCAL_ONLY=1`, the selected model/tokenizer may download from Hugging Face
when this training mode is explicitly run. Gated-model access and the model license
remain the operator's responsibility. No external training or inference API is used.

Model loading disables remote Python code (`trust_remote_code=False`) and requires
SafeTensors weights. Models requiring custom code, only pickle weights, no EOS token,
or unsupported LoRA layers fail explicitly. The process loads one full base model,
then trains LoRA adapters on its linear layers (`r=8`, alpha 16, dropout 0.05).
The complete model and activations must fit on the configured device; this runner
does not implement quantization, distributed training, or automatic OOM retries.

Training reads small batches from disk, preserves record order, pads within each
batch, truncates to the configured sequence length, and uses AdamW with gradient
clipping. Prompt and response tokens both contribute to causal SFT loss. Actual EOS
tokens remain supervised even when EOS is also used as the padding token. The
saved `model/` contains SafeTensors LoRA adapter weights and tokenizer assets; it
requires the original base model for inference. Full optimizer-state resumption,
assistant-only loss, chat-template training, and best-checkpoint selection are not
implemented. The final checkpoint is evaluated once; test data is never optimized on.

## Configuration and reports

| Key | Default | Accepted range |
| --- | --- | --- |
| `trainer` | `demo` | `demo`, `huggingface` |
| `epochs` | 1 | Integer 1–20 |
| `max_steps` | 20 | Integer 1–10,000 |
| `batch_size` | 4 | Integer 1–32 |
| `learning_rate` | 0.0002 | 0.0000001–0.01 |
| `max_seq_length` | 256 | Integer 16–4,096, also bounded by the model context |
| `seed` | 42 | Integer 0–2,147,483,647 |
| `validation_fraction` | 0.15 | 0.05–0.40 |
| `test_fraction` | 0.15 | 0.05–0.40 |

Artifacts are `train.jsonl`, `validation.jsonl`, `test.jsonl`, `training_report.json`,
`model_report.json`, and `model/`. The training report includes per-step losses,
the actual number of training record visits, compute settings, split/group counts,
and both validation and test before/after losses. Top-level `baseline_loss` and
`trained_loss` are **test-set token-weighted mean negative log likelihoods**;
`delta_loss = trained_loss - baseline_loss`, so a negative number indicates lower
loss. Perplexity is the exponential of mean loss; overflow is represented by `null`.
Demo byte metrics and LLM tokenizer metrics must not be compared directly. Validation
is measured for transparency; it is not used to tune settings in this implementation.
GPU kernels may prevent bitwise reproducibility despite the fixed seed.

The optional Hugging Face integration test creates a tiny randomly initialized GPT-2
and tokenizer entirely on disk, then performs an actual CPU LoRA update, evaluation,
and SafeTensors adapter save. This synthetic fixture tests the implementation, not
pretrained model quality or expected improvement; it does not download a model.

These metrics do not measure task correctness, factuality, safety, or downstream
business value. A dataset-quality score is also not a substitute for model quality.
Large datasets are streamed, but evaluation still reads the entire validation and
test splits both before and after training, and therefore has corresponding runtime.

Implementation follows the official [PEFT quicktour](https://huggingface.co/docs/peft/quicktour)
and [LoRA configuration reference](https://huggingface.co/docs/peft/en/package_reference/lora).
