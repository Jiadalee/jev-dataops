# Public website

The standalone website is a static project site, separate from the Python
workbench and training services.

| Route | Content |
| --- | --- |
| `/` | Project overview, workflow, domain examples, and training paths |
| `/demo/` | Interactive browser-only demo |
| `/guide/` | Getting-started and training-bundle guide |
| `/metrics/` | Six domain metric packs, definitions, typed fields, and downloads |

## Build and preview locally

Run from the repository root. The website builder uses only Python's standard
library; no npm packages or Python package installation is required.

```bash
python3 scripts/build_website.py /tmp/jev-website
python3 -m http.server 8080 --bind 127.0.0.1 --directory /tmp/jev-website
```

Open `http://127.0.0.1:8080`, then follow the Demo and Guide links. Serve the build
from the domain root so that `/assets/` and `/static/` resolve correctly. Stop
the preview server with Ctrl+C.

Edit the landing page, guide, and shared assets in [`website/`](../website/).
The builder imports [`build_public_demo.py`](../scripts/build_public_demo.py)
and places that demo at `/demo/`, with its assets under `/static/`.

Use an empty output directory or a previous website build. The builder refuses
to overwrite unrelated files, including existing Python package artifacts in
`dist/`. Rebuilding a recognized website output replaces its generated files.

## Deploy with Vercel

The repository is linked to the Vercel project
`renagaos-projects/jev-dataops`. Select that existing project when deploying
this checkout. The live website is [jev-dataops.vercel.app](https://jev-dataops.vercel.app/).
Forks should import their own Vercel project; `.vercel/` linkage is local and is
not committed.

[`vercel.json`](../vercel.json) defines the deployment settings:

| Setting | Value |
| --- | --- |
| Framework preset | Other |
| Root directory | Repository root |
| Build command | `python3 scripts/build_website.py dist` |
| Output directory | `dist` |
| Dependency installation | None required |

A fresh checkout produces the website in `dist/`. With an authenticated Vercel
CLI and the existing project selected, deploy from the repository root:

```bash
vercel --prod
```

Verify `/`, `/guide/`, `/metrics/`, and `/demo/` on the reported production URL. The config
also sets trailing slashes and response security headers. No serverless function
or Python API is included.

## Deployment contents and boundaries

[`.vercelignore`](../.vercelignore) denies files by default and allows only the
website sources, the two builders, demo frontend/runtime files, the MIT license,
six public metric packs and their synthetic JSONL examples, and the synthetic demo dataset. It excludes credentials,
`.env` files, virtual environments, uploaded data, run outputs, model weights,
and the Python backend. Keep this allowlist explicit when adding public assets.

The public demo processes at most 1,000 rows or 2 MiB in the current browser tab.
It uses local rules and a byte-bigram model; it does not call JEV, collect API
keys, upload records to a training service, or run an LLM. Reloading clears its
data and results. Download artifacts before closing the tab.

Real screening and training run in an environment you operate. See the
[training environment guide](TRAINING_ENVIRONMENTS.md) for local SFT and
external verl GRPO/PPO execution. Vercel serves the website; it does not provide
the GPU training environment.

## Optional custom domain

After production deployment succeeds, add a domain in the existing Vercel
project's Domains settings and apply the DNS records Vercel provides. A custom
domain is optional. The website uses root-relative routes and needs no source
changes when its domain changes.

The metrics catalog fetches definitions from `/assets/metric-packs/` and offers
synthetic inputs from `/assets/metric-examples/`. The builder and Vercel allowlist
name each published file explicitly. Real evaluation inputs and reports remain
on the machine running the metrics CLI; the catalog does not upload or score them.
