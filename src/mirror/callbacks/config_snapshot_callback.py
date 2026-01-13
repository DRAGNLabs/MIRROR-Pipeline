from pathlib import Path
import subprocess, json, datetime, socket, os
from mirror.callbacks.callback import Callback
from mirror.util import mirror_data_path

class ConfigSnapshotCallback(Callback):
    is_singleton = True

    def on_fit_start(self, *, fabric, model, optimizer, dataset, training_run_id: str, run_config_yaml: str):
        # Only run on rank 0
        
        if hasattr(fabric, "is_global_zero") and not fabric.is_global_zero:
            return

        safe_id = training_run_id.replace(":", "-")
        run_dir = Path(mirror_data_path) / "training_runs" / safe_id
        run_dir.mkdir(parents=True, exist_ok=True)

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip() != ""

        (run_dir / "run_config.yaml").write_text(run_config_yaml)

        meta = {
            "training_run_id": training_run_id,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git": {"commit": commit, "dirty": dirty},
        }

        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
