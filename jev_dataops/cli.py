"""Console entry points for the workbench and streaming batch mode."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="jev-dataops")
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
    args = parser.parse_args()
    if args.command == "serve":
        if args.host not in {"localhost", "127.0.0.1", "::1"} and not os.environ.get("JEV_API_TOKEN"):
            parser.error("Set JEV_API_TOKEN before binding a non-loopback address")
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port, workers=1, proxy_headers=False)
    else:
        from .screening import screen_dataset
        from .training import train_and_evaluate
        config = {key: value for key, value in vars(args).items() if value is not None}
        config["base_model"] = os.environ.get("JEV_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M")
        report = screen_dataset(args.input, args.output / "screening", config)
        if not report.get("complete", False):
            parser.exit(1, "Screening incomplete; training was not started. See data_report.json.\n")
        if args.trainer != "none" and not report.get("training_ready", True):
            parser.exit(1, (report.get("notice") or "Too many rows were not evaluated; training was not started.") + "\n")
        if args.trainer != "none":
            report = train_and_evaluate(args.output / "screening" / "keep.jsonl", args.output / "training", config)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
