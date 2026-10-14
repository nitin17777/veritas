from veritas.config import load_config
from veritas.run_utils import start_run

def test_default_config_loads():
    config = load_config("configs/default.yaml")

    assert config["project"]["name"] == "VERITAS"
    assert config["project"]["seed"] == 42


def test_start_run_creates_required_files():
    run_id, run_dir, config = start_run()

    assert run_id
    assert config["project"]["name"] == "VERITAS"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "predictions.jsonl").exists()