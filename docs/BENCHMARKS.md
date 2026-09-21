# Validation and scale checks

Measured on 2026-09-21. Results below are reproducible checks, not production capacity promises.

## 100,000-row local pipeline

```bash
python scripts/benchmark.py --rows 100000 --output /tmp/jev-benchmark
```

| Measurement | Result |
|---|---:|
| Environment | macOS arm64, Python 3.12.14 |
| Synthetic records | 100,000 |
| Input bytes | 14,166,670 |
| Rule screening + disk audit/cache | 93.306 s |
| Screening throughput | 1,071.7 rows/s |
| Split + demo train + held-out evaluation | 7.247 s |
| Peak process RSS (including native allocations) | 49.52 MiB |
| Train / validation / test records | 70,000 / 15,000 / 15,000 |
| Independent conversation groups | 50,000 |
| Demo training steps | 20 |

All records in this particular corpus pass the local rules. Deliberate invalid/duplicate cases are covered separately by tests and the 84-row bundled example. Screening uses four workers; disk speed affects the result. The demo trainer only visits up to 20 batches, while splitting and evaluation scan their full partitions.

### Grouped commits

The figures above were taken with one SQLite commit per row. On a Linux workstation with an ext4 root disk (Python 3.12.3), the same 100,000-row run took **867 s (115 rows/s)** that way; committing every 500 rows or once a second brought it to **21.3 s (4,700 rows/s)** with the same peak memory (35 → 38 MiB) and identical partitions. A crash now loses at most 500 cached decisions, which the retry re-asks for. Putting the cache in WAL mode with `synchronous=NORMAL` and running demo rules inline instead of on the worker pool brought screening to **16.3 s (6,100 rows/s)**; turning the journal off for the scratch split index, which is deleted when the split finishes, brought split + train + evaluate from 13.7 s to **6.2 s**.

## Live screening: gates and connection reuse

Measured on 2026-09-21 against OpenRouter, `~typesafe/jev-latest` resolving to `typesafe/jev-1.13-20260917`, on the bundled 84-row example.

| | before | after |
|---|---:|---:|
| keep / review / reject | 15 / 67 / 2 | 60 / 21 / 3 |
| wall time, 4 workers, 81 requests | 8.9 s | 6.6 s |
| provider cost recorded | — | $0.0023 |
| resolved model recorded | — | yes |

The example is synthetic and clean, so nearly every row should be kept. Before, the run confidence of 0.85 was applied to JEV's `confidence` field on every dimension, and that field sits near 0.5 on `quality` even when the probability on `good` is 0.7–0.9, which sent 80% of the rows to review. After, `quality` and `trainability` gate on the probability mass of the chosen decision class and `privacy` keeps the confidence gate.

On a 63-row labelled set (45 clean rows: 30 from the example plus 15 hand-written passages, instruction pairs and conversations; 18 rows that should be excluded: garbage, contradictions, HTML noise, an API key, an SSN, a password, a card number, a private key, unrelated answers, boilerplate, spam, an empty answer) both rule sets were applied to the same stored JEV responses:

| gating | clean rows kept | clean rows reviewed | clean rows rejected | bad rows rejected | bad rows kept |
|---|---:|---:|---:|---:|---:|
| before, confidence 0.85 | 18 / 45 | 27 | 0 | 18 / 18 | 0 |
| before, confidence 0.50 | 29 / 45 | 16 | 0 | 18 / 18 | 0 |
| after, confidence 0.85 | 37 / 45 | 8 | 0 | 18 / 18 | 0 |

JEV's own argmax was right on every bad row, so both rule sets exclude all of them; the difference is entirely in how many good rows reach the training set without a human reading them. The eight clean rows still reviewed have `P(good)` between 0.50 and 0.59.

## LoRA path

Same 80 retained example rows, `HuggingFaceTB/SmolLM2-135M-Instruct`, one RTX PRO 6000, 14 steps of batch 4 at sequence length 256. The adapters were then scored with one evaluator: answer-token NLL on the held-out test rows, rendered with the model's chat template, which is the format the model is served in.

| adapter | training wall | answer NLL (chat template) | answer NLL (raw `user:`/`assistant:` text) |
|---|---:|---:|---:|
| none | — | 3.52 | 3.82 |
| before: plain text, full-token loss, float32 | 9.5 s | 3.02 | 3.16 |
| after: chat template, answer-only loss, bfloat16 | 4.6 s | 2.59 | 3.23 |
| after with `loss_mask: full` | 4.5 s | 3.10 | 3.60 |

Each adapter does best in the format it was trained in; the one trained on the served format is ahead by 0.43 there and behind by 0.07 on the other. Fourteen steps on 56 synthetic rows do not make a useful model, and the greedy generations of all three adapters are alike; the table is about the training path, not the model. Peak GPU memory rose from 634 to 857 MiB because the chat template adds a system turn to every row and the loss upcasts logits to float32 over more tokens; the bfloat16 saving on the weights matters at sizes where weights, not activations, dominate.

**This does not measure JEV API latency/cost/accuracy, LLM GPU training speed, or billion-row operation.** Real provider performance depends on quotas and request size. Model memory is additional to dataset processing memory.

## Functional checks

- Automated tests cover streaming JSONL/CSV, malformed input/typed JEV responses, confidence routing, API retry budgets, durable cache reuse, upload/auth/origin protections, cancel/retry, and incomplete-run training blocking.
- Training tests include actual offline LoRA forward/backward, held-out evaluation and safetensors adapter saving with a local randomly initialized tiny GPT-2 and tokenizer. This validates the execution path, not useful model capability.
- Browser walkthrough: upload/preview, start a workflow, read counts/loss/split reports, and inspect available downloadable artifacts.
- Bundled synthetic example: 84 rows → 80 keep / 2 review / 2 reject; 56 train / 12 validation / 12 test. Demo byte-bigram test NLL is measured, and must not be compared with LLM tokenizer metrics.

Live JEV calls and large pretrained-model fine-tuning were not run for this release validation. Configure a provider key and a licensed model on the target training host to validate those operational dependencies.
