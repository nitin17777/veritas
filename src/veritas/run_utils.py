import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from veritas.config import ROOT_DIR, load_config


def start_run(
    config_path: str = "configs/default.yaml",
) -> tuple[str, Path, dict]:
    config = load_config(config_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    results_root = ROOT_DIR / config["paths"]["results_dir"]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Save the exact configuration used for this run.
    shutil.copy2(
        Path(config_path)
        if Path(config_path).is_absolute()
        else ROOT_DIR / config_path,
        run_dir / "config.yaml",
    )

    # Initialize logging for this run.
    log_file = run_dir / "run.log"
    logger = logging.getLogger("veritas")
    logger.setLevel(
        getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    )
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Started VERITAS run: %s", run_id)
    logger.info("Configuration saved to %s", run_dir / "config.yaml")

    # Reserve the predictions file for later pipeline modules.
    (run_dir / "predictions.jsonl").touch(exist_ok=True)

    return run_id, run_dir, config