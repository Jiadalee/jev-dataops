# Public browser demo

**[Open the live demo](https://jev-dataops.vercel.app/demo/)**

Try the complete workflow on a small dataset without installing Python or configuring a model provider. Files are read and processed in the current browser tab. The demo does not upload dataset content to a server or call JEV, Hugging Face, or a cloud training service.

## First run

1. Click **Use example**, or choose a UTF-8 JSONL / CSV file.
2. Check the data preview. Leave **Auto-train and evaluate** enabled to try every stage.
3. Click **Start workflow**.
4. Inspect the keep / review / reject counts and held-out evaluation.
5. Download the reports, screened records, splits, and model weights before reloading or closing the tab.

The included synthetic example produces **80 kept, 2 review, and 2 rejected records**. The retained records become **56 train, 12 validation, and 12 test records**. Loss and perplexity are measured from an actual byte-bigram statistical model before and after training; this is not an LLM.

## Scope and limits

| Feature | Browser demo |
| --- | --- |
| Input | UTF-8 JSONL / CSV; up to 2 MiB and 1,000 records per file |
| Session | Up to 5 datasets and 10 runs; one active run at a time |
| Screening | Schema, length, exact duplicates, and a basic email review rule |
| Training | Local byte-bigram model; 1 epoch, up to 20 batches of 4 records |
| Sequence length | First 256 UTF-8 bytes per record, followed by an end token |
| Evaluation | Loss and perplexity on fixed, separate validation and test splits |
| Group isolation | Matching content, `group_id`, or `conversation_id` stays in the same split |
| Storage | Tab memory only; reloading clears all datasets, runs, and results |

Training requires at least **6 independent groups after screening**. If there are fewer, the screening artifacts remain available, but training stops with an explanation. You can disable auto-training to screen a small file.

The email rule is only a basic demonstration; it is not comprehensive privacy detection. Domain-specific JEV judgment, LoRA fine-tuning, large-scale processing, persistent jobs, and multi-user storage require the [self-hosted application](../README.md#quickstart). Domain selection is disabled in this demo because its local rules do not assess domain quality.

## Build and serve

From the repository root, using Python 3.10+:

```bash
python3 scripts/build_public_demo.py /tmp/jev-public-demo
python3 -m http.server 8768 --bind 127.0.0.1 --directory /tmp/jev-public-demo
```

Open <http://localhost:8768>. The builder reuses the workbench interface and installs `runtime.js` as a local adapter. It leaves the Python application unchanged. Deploy the generated directory at the **root of an HTTPS static site**; subpath hosting requires adapting the `/static/` asset URLs. HTTPS or localhost is required for the browser's SHA-256 API.

No server functions, API keys, external fonts, or model downloads are needed. The demo fetches only its own bundled example when **Use example** is clicked; uploaded file contents stay in memory. Downloaded results are saved by the browser to the visitor's device.

Run the dependency-free regression checks with Node.js 24:

```bash
node scripts/test_public_demo.mjs
```

Checks cover sample output parity, isolation of grouped data, real model metrics, JSONL / CSV edge cases, limits, cancellation, and downloadable artifacts.
