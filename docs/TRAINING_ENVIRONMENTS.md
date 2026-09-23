# From screened data to SFT, GRPO, or PPO

JEV DataOps can prepare a training bundle on your data-processing machine and
launch it in a training environment you control. **Exporting a bundle does not
train a model.** A launch without `--execute` is a dry run; it shows the planned
command without starting training.

| Target | What learns | Training runtime | Main prerequisite |
| --- | --- | --- | --- |
| `sft` | A LoRA adapter learns from retained examples | This project's Hugging Face trainer | Compatible causal LM and enough memory |
| `verl-grpo` | A policy learns from groups of generated answers and rewards | Pinned external verl environment | Linux NVIDIA GPU host and verifiable reference answers |
| `verl-ppo` | A policy and critic learn from generated answers and rewards | Pinned external verl environment | Same as GRPO, plus memory for the critic |

The public Vercel website is a **static demonstration**, not a training server.
It does not provide a GPU, run Python training jobs, or keep your API credentials.
Use the locally hosted workbench or CLI to process data, then launch training on
your own machine. The built-in web workflow runs Demo or Hugging Face SFT; the
external verl recipes below use the CLI.

For one upload shared by 2, 4, or 8 GPUs, follow the [multi-GPU guide](MULTI_GPU.md).
Set `--n-gpus` at export and a global `--batch-size` divisible by that count.
Multi-GPU SFT uses DDP on one Linux/NVIDIA host; the web workbench's automatic
SFT path remains single-process. Export and launch through the CLI for multi-GPU
execution. The current launcher does not configure multiple machines.

## 1. Screen data before exporting

Run these commands from a checkout of this repository in a Python virtual
environment. The bundled arithmetic fixture uses synthetic, explicitly supplied
reference answers; it demonstrates the mechanics, not useful model capability.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[export]'

jev-dataops run \
  --input examples/rl_math.jsonl \
  --output /tmp/jev-math-screen-001 \
  --provider demo --trainer none
```

This is local rule screening. For real JEV selection, set the appropriate
`OPENROUTER_API_KEY` or `TYPESAFE_API_KEY` and choose `--provider openrouter` or
`--provider typesafe`. Check `screening/data_report.json`: screening must be
complete and training-ready before using `screening/keep.jsonl`. Inspect retained
and rejected samples, rather than treating a keep decision as factual verification.

Use fresh output directories for the recipes below. The input to `export-training`
is the retained JSONL file, not the original upload, review partition, or a model
checkpoint.

The `export` extra supplies PyArrow for the RL Parquet files; it does not install
PyTorch, download model weights, or provision a training environment. SFT-only
exports do not need PyArrow.

## 2. Choose the right data and reward

SFT can learn from text, instruction/output pairs, prompt/response pairs, or text
conversations. See [the supported formats](../README.md#data). For RL, each retained
record needs a usable prompt and an **explicit, independently checked reference**.
For example:

```jsonl
{"group_id":"math-001","prompt":"What is 7 + 5? Reply with the number only.","response":"12","ground_truth":"12"}
{"group_id":"math-002","prompt":"What is 9 - 3? Reply with the number only.","response":"6","ground_truth":"6"}
```

These two rows illustrate the schema; they are insufficient for the required
train/validation/test split. Use the bundled fixture for a complete export example.

`--reward-field ground_truth` names the field you intentionally supply for reward
calculation. A response suitable for SFT is not automatically a verified RL answer.
The exporter does not infer truth from arbitrary prose or turn a JEV confidence or
quality score into a reward. Fix missing or invalid references before export.
The field names `text`, `_state`, and `_split_group` are reserved for preparation;
store verified answers in `ground_truth` or another dedicated metadata field.

The included reward is bounded exact matching, intended for closed-answer tasks
such as short arithmetic answers or fixed labels. It does not verify open-ended
financial analysis, citations, medical claims, code correctness, or semantic
equivalence. A model can produce a correct explanation that fails an exact-match
check, or exploit a poorly designed reference. Review the reward and test concrete
good and bad completions before spending GPU time. A domain-specific reward needs
its own validation.

The callback normalizes Unicode to NFC and collapses whitespace, then compares
the entire generated text with the reference. Case, punctuation and numbers stay
significant: `12` matches `12`, while `The answer is 12.` does not. It returns
`score=acc=1.0` for a match and `0.0` otherwise; it does not execute generated code
or extract a final answer from an explanation. Empty, non-text or oversized
references/responses receive zero reward. This simple verifier is deliberately
limited; the shipped launcher accepts only the shipped reward implementation.

For RL export, the source still includes a final assistant response: the exporter
removes that final response from the rollout prompt. The prompt must contain a
nonempty user turn and end in a user or tool turn. Bare text is SFT-only. Earlier
conversation turns remain context; the `ground_truth` field is stored separately
for the verifier, not appended to the prompt.

Related records must share meaningful `group_id` or `conversation_id` values.
Export preserves train/validation/test isolation, including repeated prompts and
content groups, so answer variants for one prompt cannot quietly span partitions.
At least six independent groups are required after these connections are joined.
Keep the same source data, grouping and seed when comparing experiments; inspect
the exported split metadata. These checks do not discover semantic paraphrases.

Each bundle contains:

```text
bundle/
├── manifest.json              # Source hash, target, config, split counts and checksums
├── launcher.json              # Launcher settings
├── data/
│   ├── train.jsonl            # SFT; RL uses train.parquet
│   ├── validation.jsonl       # SFT; RL uses validation.parquet
│   └── test.jsonl             # SFT; RL uses test.parquet
└── rewards.py                 # RL bundles only: reference-answer reward
```

SFT JSONL contains `messages` or bare `text`, plus split identifiers. RL Parquet
contains chat-format `prompt`, `data_source`, `ability`,
`reward_model.ground_truth` and `extra_info` with split identifiers. `manifest.json`
has status `prepared`; that does not mean a training job completed. Preserve the
bundle as an experiment input and re-export when changing its data or settings.

## 3. Export and run SFT

Create the bundle without loading the model:

```bash
jev-dataops export-training \
  --input /tmp/jev-math-screen-001/screening/keep.jsonl \
  --output /tmp/jev-sft-bundle-001 \
  --target sft \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1 --max-steps 20 --batch-size 4 \
  --max-seq-length 1024 --loss-mask answer

jev-dataops launch-training \
  --bundle /tmp/jev-sft-bundle-001 \
  --output /tmp/jev-sft-run-001 --dry-run
```

On the training machine, install this project's training dependencies into its
own environment. If you exported elsewhere, copy the **whole bundle** first and
use its new local path.

```bash
pip install -e '.[train]'

jev-dataops launch-training \
  --bundle /tmp/jev-sft-bundle-001 \
  --output /tmp/jev-sft-run-001 --execute
```

`--python /absolute/path/to/venv/bin/python` can select another prepared interpreter
on the same machine. This is a local process launch, not SSH or a cloud job
submission service. The selected runtime must have this project and its SFT
dependencies installed.

SFT trains from the bundle's fixed training split and evaluates before and after
on its fixed held-out splits. It uses the built-in LoRA trainer's answer masking
and chat-template behavior; see [training details](TRAINING.md). Defaults are small
smoke-test settings. Review the maximum steps, epochs, batch size and sequence
length before execution; neither a successful export nor a short run means every
record has been trained on.

The result is a LoRA adapter, not a standalone full model. To initialize a later RL
run from SFT, load the adapter with its exact original base model and merge/save a
complete compatible model checkpoint first. Then use that model directory as the
RL bundle's `--base-model`. Do not point verl at an adapter-only `model/` directory.
PEFT documents the [merge-and-unload operation](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#merge-lora-weights-into-the-base-model).

For the SFT recipe above, run this in the SFT environment after training completes:

```python
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = "/tmp/jev-sft-run-001/model"
report = json.loads(Path("/tmp/jev-sft-run-001/training_report.json").read_text())
destination = Path("/tmp/jev-sft-merged-001")
if destination.exists():
    raise ValueError("Choose a new merged-model directory")

# Use the same base weights/revision recorded in the SFT training report.
base = AutoModelForCausalLM.from_pretrained(
    report["base_model"], revision=report.get("base_model_revision"),
    trust_remote_code=False, use_safetensors=True,
)
merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
merged.save_pretrained(destination, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter, trust_remote_code=False).save_pretrained(destination)
```

This operation needs enough host memory for the full model and may download its
original weights. In the RL export commands below, replace the model ID with
`--base-model /tmp/jev-sft-merged-001` to continue from this SFT result. On a different
host, copy the merged model and use its absolute path there. The bundle contains
model references, not model weights.

## 4. Prepare a separate verl environment

The generated RL recipe targets **verl 0.9.1**, with FSDP training and vLLM rollout.
Use a Linux NVIDIA GPU host. This is not a macOS/CPU training recipe, and model
size, rollout length, batch size and the PPO critic determine memory requirements.
No fixed GPU memory claim is made.

The v0.9.1 environment uses Python 3.12, PyTorch 2.11.0 CUDA 13 wheels,
Transformers 5.9.0 and vLLM 0.24.0. Prepare a compatible NVIDIA driver and follow
the pinned release's environment instructions. This project's SFT extra requires
Transformers `<5`, so **do not install `jev-dataops[train]` into the verl environment**.
These versions come from the release's [dependency configuration](https://github.com/verl-project/verl/blob/v0.9.1/pyproject.toml).

With `uv` installed, use a separate shell outside the active JEV virtual environment:

```bash
git clone --branch v0.9.1 --depth 1 https://github.com/verl-project/verl.git verl-0.9.1
cd verl-0.9.1
uv sync --python 3.12 --frozen --all-packages --extra vllm --extra fsdp

.venv/bin/python -c 'import torch, transformers, verl; print(torch.__version__, transformers.__version__); print("CUDA available:", torch.cuda.is_available())'
```

Resolve installation and GPU visibility problems before launching a job. The
release's [installation guide](https://github.com/verl-project/verl/blob/v0.9.1/docs/start/install.rst)
describes the `uv` workflow; the [v0.9.1 release](https://github.com/verl-project/verl/releases/tag/v0.9.1)
provides the pinned source. Do not silently substitute an unrelated latest version
and assume its configuration fields are compatible.

Return to your JEV shell for export/launch commands. Pass the absolute path of the
prepared verl interpreter through `--python`; both environments can coexist on the
GPU machine. Installation does not itself launch training.

## 5. Export and run GRPO or PPO

Export the same retained dataset for GRPO:

```bash
jev-dataops export-training \
  --input /tmp/jev-math-screen-001/screening/keep.jsonl \
  --output /tmp/jev-grpo-bundle-001 \
  --target verl-grpo \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --reward-field ground_truth \
  --learning-rate 0.000001 --epochs 1 --max-steps 20 \
  --batch-size 4 --n-gpus 1 --rollout-n 4 \
  --max-prompt-length 512 --max-response-length 512

jev-dataops launch-training \
  --bundle /tmp/jev-grpo-bundle-001 \
  --python /absolute/path/to/verl-0.9.1/.venv/bin/python \
  --output /tmp/jev-grpo-run-001 \
  --dry-run
```

For PPO, export a separate bundle with `--target verl-ppo`:

```bash
jev-dataops export-training \
  --input /tmp/jev-math-screen-001/screening/keep.jsonl \
  --output /tmp/jev-ppo-bundle-001 \
  --target verl-ppo \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --reward-field ground_truth \
  --learning-rate 0.000001 --epochs 1 --max-steps 20 \
  --batch-size 4 --n-gpus 1 \
  --max-prompt-length 512 --max-response-length 512
```

Inspect the dry-run plan and generated configuration, then explicitly launch on
the prepared GPU machine:

```bash
jev-dataops launch-training \
  --bundle /tmp/jev-grpo-bundle-001 \
  --python /absolute/path/to/verl-0.9.1/.venv/bin/python \
  --output /tmp/jev-grpo-run-001 \
  --execute
```

Use the PPO bundle path to launch PPO. Both algorithms enter
`python -m verl.trainer.main_ppo`; GRPO chooses `algorithm.adv_estimator=grpo`,
multiple rollouts per prompt, and no critic. PPO chooses GAE and an enabled critic.
The generated recipe wires an explicit reference-based custom reward into verl;
JEV is not called to score policy completions.

The examples use small execution budgets, not tuned hyperparameters. GRPO requires
at least two rollout responses per prompt. RL batch size must be divisible by the
GPU count and fit within the training split. The generated recipe runs on one
node; these flags do not configure a multi-node cluster. Prompts longer than the
configured limit fail rather than being silently truncated. A compatible chat
template and SafeTensors base-model weights are required.

Validation uses the configured global batch size, rather than loading the entire
validation split into one batch. verl's dataset loader maintains its own cache
and index; plan for its host-memory and disk requirements. The supplied vLLM
rollout configuration uses tensor parallel size 1, so each rollout replica must
fit on one GPU even when the training job uses several GPUs.

Only the training partition is used for updates. The validation partition is
available for in-training validation; **the test partition is excluded from the
verl training launch**. Preserve it for a final, separate evaluation with a fixed
prompt, decoding and scoring policy. This launcher does not automatically produce
an SFT-style before/after `model_report.json` for external RL runs. Training rewards
alone are not held-out evidence of improvement.

## 6. Read the launch result

A dry run verifies the bundle's files/checksums and prints the command, working
directory and result path. The selected interpreter must exist, but a dry run does
not check its installed GPU libraries, download a model, or start a process.
`--execute` performs runtime checks before launching. For RL it checks the pinned
versions and available CUDA devices, resolves the model, and checks its tokenizer.
Multi-GPU SFT checks Linux, NCCL, and the requested number of visible CUDA devices.
It may download model assets; `JEV_MODEL_LOCAL_ONLY=1` requires already cached or
local assets.

The command runs synchronously on the launch host. Logs are written to the result
directory, rather than streamed back into the hosted website. With `--output`
omitted, the default is `results/` inside the bundle. The result directory must not
already exist, including when retrying a failed run; supply a new `--output` path.

| Output | Purpose |
| --- | --- |
| `launch_started.json` | Actual command, runtime preflight and start time |
| `training.log` | Combined child-process stdout and stderr |
| `launch_report.json` | Process completion status, return code and timestamps |
| SFT `training_report.json`, `model_report.json`, `model/` | Measured losses, training details and LoRA adapter |
| RL `checkpoints/` | Checkpoints and artifacts written by verl |

Preflight errors appear before these output files are created. Once launched,
check `launch_report.json` and `training.log`; a prepared manifest or successful
export is not a completion signal. A completed external process still needs
checkpoint inspection and independent evaluation before you claim model gains.

## 7. Before scaling up

- Confirm screening completed and inspect the selection audit. Validate reference
  labels separately from the JEV screening decision.
- Inspect split counts and related-example grouping. Never reuse the test set as
  `data.val_files` or a training input.
- Check the launch plan's model, interpreter, output paths, algorithm and compute
  settings. Use a new bundle and output location for a changed experiment.
- Start with a short GPU run and inspect generated answers and rewards. All-zero
  or all-one rewards often point to a reference, format or task-difficulty issue.
- Keep the bundle, original screening report, environment versions, checkpoints
  and evaluation results together. Export metadata does not replace recording
  the actual GPU/runtime environment.

**Validation boundary:** local export, configuration and dry-run checks do not
establish GPU compatibility or model quality. The real verl GRPO/PPO GPU training
path has not been executed in this development environment. Validate it on the
target hardware before relying on the recipe for a production experiment.
