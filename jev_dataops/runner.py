"""Single-host background queue with durable state and explicit recovery."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid

from .store import Store, now


class Runner:
    def __init__(self, store: Store):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-worker")
        self.events = {}
        self.lock = threading.RLock()

    def submit(self, dataset, config, run_id=None):
        with self.lock:
            if len(self.events) >= 32:
                raise ValueError("Queue is full (32 jobs). Wait for an existing job to finish.")
            run_id = run_id or uuid.uuid4().hex
            if run_id in self.events:
                raise ValueError("This run is still stopping. Retry after its worker exits.")
            old = self.store.get("run", run_id)
            if old and old["status"] not in {"failed", "cancelled"}:
                raise ValueError("Only failed or cancelled runs can be retried")
            run = {"id": run_id, "dataset_id": dataset["id"], "name": dataset["name"], "status": "queued", "stage": "upload", "created_at": old["created_at"] if old else now(), "updated_at": now(), "config": config, "progress": {"processed": 0, "total": dataset["rows"]}, "counts": {}, "data_report": None, "model_report": None, "error": None, "logs": [], "artifacts": []}
            run["attempt"] = old.get("attempt", 1) + 1 if old else 1
            self.store.put("run", run)
            event = threading.Event()
            self.events[run_id] = event
            self.pool.submit(self._execute, run_id, dataset, config, event)
            return run

    def cancel(self, run_id):
        with self.lock:
            event = self.events.get(run_id)
            if event:
                event.set()
                self.store.log(run_id, "Cancellation requested; waiting for the current bounded operation.")
            return self.store.get("run", run_id)

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        self.pool.shutdown(wait=True, cancel_futures=False)

    def _execute(self, run_id, dataset, config, event):
        directory = self.store.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        terminal = {}
        def stage(value, message):
            self.store.update(run_id, stage=value)
            self.store.log(run_id, message)
        last_update = [0.0]
        def progress(update):
            if time.monotonic() - last_update[0] < 0.35 and not update.get("stage"):
                return
            last_update[0] = time.monotonic()
            fields = {"progress": {"total": dataset["rows"], **update}}
            if update.get("counts"):
                fields["counts"] = update["counts"]
            mapped_stage = {"splitting": "data_evaluation", "loading_model": "training", "evaluating_baseline": "training", "evaluating": "model_evaluation"}.get(update.get("stage"), update.get("stage"))
            if mapped_stage in {"training", "model_evaluation", "data_evaluation"}:
                fields["stage"] = mapped_stage
            self.store.update(run_id, **fields)
        try:
            from .screening import screen_dataset
            from .training import train_and_evaluate
            if event.is_set():
                raise InterruptedError()
            self.store.update(run_id, status="running")
            stage("screening", "Streaming screening started; keep / review / reject are recorded separately.")
            source = self.store.root / "datasets" / (dataset["id"] + dataset["suffix"])
            report = screen_dataset(source, directory / "screening", config, progress=progress, cancelled=event.is_set)
            self.store.update(run_id, counts=report["counts"], data_report=report)
            if event.is_set() or report.get("status") == "cancelled":
                raise InterruptedError()
            if not report.get("complete", report.get("status") == "complete"):
                raise ValueError("Screening incomplete. Training was not started. Inspect data_report and retry.")
            stage("data_evaluation", "Data report saved. Reviewing retained volume and preparing independent splits.")
            if config["auto_train"]:
                stage("training", "Training started: " + config["trainer"])
                attempt = self.store.get("run", run_id)["attempt"]
                model_report = train_and_evaluate(directory / "screening" / "keep.jsonl", directory / f"training-{attempt}", config, progress=progress, cancelled=event.is_set)
                if event.is_set():
                    raise InterruptedError()
                stage("model_evaluation", "Held-out baseline and trained-model evaluation saved.")
                self.store.update(run_id, model_report=model_report)
            terminal = {"status": "completed", "stage": "completed"}
            self.store.log(run_id, "Workflow completed. Review the evidence before promoting a model.")
        except InterruptedError:
            terminal = {"status": "cancelled", "error": "Cancelled by user. Retry is available."}
        except Exception as exc:
            # Expected local validation errors are actionable. Provider exceptions
            # and unexpected tracebacks must never expose credentials or records.
            message = str(exc)[:500] if isinstance(exc, ValueError) else "Operation failed (" + type(exc).__name__ + "). Check service configuration."
            terminal = {"status": "cancelled" if event.is_set() else "failed", "error": message}
            self.store.log(run_id, message)
        finally:
            artifacts = []
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".safetensors", ".txt", ".model", ".jinja", ".vocab", ".tiktoken"}:
                    relative = path.relative_to(directory).as_posix()
                    artifacts.append({"name": relative, "url": f"/api/runs/{run_id}/artifacts/{relative}", "size": path.stat().st_size})
            with self.lock:
                # Publish terminal status with the complete inventory because
                # clients can stop polling immediately after this write.
                self.store.update(run_id, artifacts=artifacts, **terminal)
                if self.events.get(run_id) is event:
                    self.events.pop(run_id, None)
