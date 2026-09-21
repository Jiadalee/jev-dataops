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

All records in this particular corpus pass the local rules. Deliberate invalid/duplicate cases are covered separately by tests and the 84-row bundled example. Screening uses four workers and per-record SQLite commits; disk speed affects the result. The demo trainer only visits up to 20 batches, while splitting and evaluation scan their full partitions.

**This does not measure JEV API latency/cost/accuracy, LLM GPU training speed, or billion-row operation.** Real provider performance depends on quotas and request size. Model memory is additional to dataset processing memory.

## Functional checks

- Automated tests cover streaming JSONL/CSV, malformed input/typed JEV responses, confidence routing, API retry budgets, durable cache reuse, upload/auth/origin protections, cancel/retry, and incomplete-run training blocking.
- Training tests include actual offline LoRA forward/backward, held-out evaluation and safetensors adapter saving with a local randomly initialized tiny GPT-2 and tokenizer. This validates the execution path, not useful model capability.
- Browser walkthrough: upload/preview, start a workflow, read counts/loss/split reports, and inspect available downloadable artifacts.
- Bundled synthetic example: 84 rows → 80 keep / 2 review / 2 reject; 56 train / 12 validation / 12 test. Demo byte-bigram test NLL is measured, and must not be compared with LLM tokenizer metrics.

Live JEV calls and large pretrained-model fine-tuning were not run for this release validation. Configure a provider key and a licensed model on the target training host to validate those operational dependencies.
