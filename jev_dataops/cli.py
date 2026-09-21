"""Console entry points for the workbench and streaming batch mode."""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_env_file(path):
    """KEY=VALUE lines into the environment without overriding what is already set."""
    loaded = 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


class Progress:
    """One updating line on stderr; a plain line every few seconds when not a terminal."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.last = 0.0
        self.width = 0

    def screening(self, update):
        counts = update.get("counts", {})
        line = (f"screening {update.get('processed', 0):,} rows · keep {counts.get('keep', 0):,} "
                f"review {counts.get('review', 0):,} reject {counts.get('reject', 0):,}")
        if update.get("mode") == "jev_api":
            line += f" · {update.get('api_requests', 0):,} requests, {update.get('cache_hits', 0):,} cached"
        self._show(line)

    def training(self, event):
        stage = event.get("stage", "")
        if stage == "training":
            self._show(f"training step {event['step']}/{event['max_steps']} · loss {event['loss']:.4f}")
        elif stage:
            self._show(stage.replace("_", " ") + (f" · {event['message']}" if event.get("message") else ""), force=True)

    def _show(self, line, force=False):
        now = time.monotonic()
        if not force and now - self.last < (0.2 if self.tty else 5.0):
            return
        self.last = now
        if self.tty:
            self.stream.write("\r" + line.ljust(self.width))
            self.width = max(self.width, len(line))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def done(self):
        if self.tty and self.width:
            self.stream.write("\n")
            self.stream.flush()


def summarize(report, trained=None):
    counts = report["counts"]
    lines = [f"Screening {report['status']}: {counts['total']:,} rows → keep {counts['keep']:,}, review {counts['review']:,}, "
             f"reject {counts['reject']:,} ({counts['duplicates']:,} duplicates)"]
    if report.get("mode") == "jev_api":
        usage = report.get("usage", {})
        models = ", ".join(report.get("models", {})) or "n/a"
        lines.append(f"  {report['api_requests']:,} requests, {report['cache_hits']:,} cache hits, "
                     f"${usage.get('cost', 0):.4f}, model {models}")
        if report.get("unevaluated"):
            lines.append(f"  {report['unevaluated']:,} rows not evaluated ({report['unevaluated_fraction']:.1%})")
    if report.get("error_count"):
        lines.append(f"  {report['error_count']:,} row errors; first: {report['errors'][0]}")
    if report.get("notice"):
        lines.append("  " + report["notice"])
    if trained:
        lines.append(f"Training {trained['trainer']}: {trained['steps']} steps · test loss {trained['baseline_loss']:.4f} → "
                     f"{trained['trained_loss']:.4f} ({trained['delta_loss']:+.4f}) · {trained['metric_unit']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="jev-dataops")
    parser.add_argument("--env-file", type=Path, help="Load KEY=VALUE lines (e.g. OPENROUTER_API_KEY) into the environment first")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the upload workbench")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", default=".jev-dataops")
    run = commands.add_parser("run", help="Screen and optionally train a local dataset")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--provider", choices=["demo", "typesafe", "openrouter"], default="demo")
    run.add_argument("--model", help="Pinned Jev model id; the provider's floating alias by default")
    run.add_argument("--trainer", choices=["none", "demo", "huggingface"], default="demo")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--max-requests", type=int, default=1000)
    run.add_argument("--confidence", type=float, default=0.85)
    run.add_argument("--rubric", choices=["general", "finance", "code"], default="general")
    run.add_argument("--min-chars", type=int, default=8)
    run.add_argument("--max-chars", type=int, default=32000)
    run.add_argument("--timeout", type=int, default=20, help="Seconds per Jev request")
    training = run.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--max-steps", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=0.0002)
    training.add_argument("--max-seq-length", type=int, default=256)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--lora-r", type=int, default=8)
    training.add_argument("--lora-alpha", type=int, default=16)
    training.add_argument("--loss-mask", choices=["answer", "full"], default="answer",
                          help="answer: learn only the response tokens; full: plain causal SFT over prompt and response")
    run.add_argument("--quiet", action="store_true", help="No progress or summary on stderr; JSON report on stdout only")
    args = parser.parse_args()
    if args.env_file:
        try:
            load_env_file(args.env_file)
        except OSError as exc:
            parser.error(f"cannot read {args.env_file}: {exc.strerror}")
    if args.command == "serve":
        if args.host not in {"localhost", "127.0.0.1", "::1"} and not os.environ.get("JEV_API_TOKEN"):
            parser.error("Set JEV_API_TOKEN before binding a non-loopback address")
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port, workers=1, proxy_headers=False)
    else:
        from .screening import screen_dataset
        from .training import train_and_evaluate
        config = {key: value for key, value in vars(args).items() if value is not None and key not in {"env_file", "quiet", "command"}}
        config["base_model"] = os.environ.get("JEV_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M")
        progress = None if args.quiet else Progress()
        try:
            report = screen_dataset(args.input, args.output / "screening", config, progress=progress.screening if progress else None)
        except ValueError as exc:
            parser.exit(2, f"{exc}\n")
        finally:
            if progress:
                progress.done()
        if not report.get("complete", False):
            if progress:
                print(summarize(report), file=sys.stderr)
            parser.exit(1, "Screening incomplete; training was not started. See data_report.json.\n")
        if args.trainer != "none" and not report.get("training_ready", True):
            parser.exit(1, (report.get("notice") or "Too many rows were not evaluated; training was not started.") + "\n")
        trained = None
        if args.trainer != "none":
            try:
                trained = train_and_evaluate(args.output / "screening" / "keep.jsonl", args.output / "training", config,
                                             progress=progress.training if progress else None)
            finally:
                if progress:
                    progress.done()
        if progress:
            print(summarize(report, trained), file=sys.stderr)
        print(json.dumps(trained or report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
