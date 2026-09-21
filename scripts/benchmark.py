"""Reproducible synthetic local throughput check, never a JEV API benchmark."""
import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_dataops.screening import screen_dataset
from jev_dataops.training import train_and_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 10:
        parser.error("Use at least 10 rows")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "synthetic.jsonl"
    with source.open("w") as stream:
        for i in range(args.rows):
            stream.write(json.dumps({"text": f"Synthetic support example {i}: open your account settings to update notification preferences.", "group_id": f"conversation-{i // 2}"}) + "\n")
    start = time.monotonic()
    report = screen_dataset(source, args.output / "screening", {"provider": "demo", "concurrency": 4})
    screening_seconds = time.monotonic() - start
    start = time.monotonic()
    trained = train_and_evaluate(args.output / "screening/keep.jsonl", args.output / "training", {"trainer": "demo", "max_steps": 20})
    training_seconds = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    result = {"kind": "synthetic_local_demo", "python": platform.python_version(), "os": platform.system(), "machine": platform.machine(), "rows": args.rows, "input_bytes": source.stat().st_size, "screening_seconds": round(screening_seconds, 3), "rows_per_second": round(args.rows / screening_seconds, 1), "split_train_evaluate_seconds": round(training_seconds, 3), "peak_process_rss_mib": round(rss_mib, 2), "counts": report["counts"], "split_counts": trained["split_counts"], "steps": trained["steps"], "limitations": "Local rules + byte-bigram only. No API/network/GPU throughput measured. Training capped at 20 steps; split construction and held-out evaluation cover their full partitions."}
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
