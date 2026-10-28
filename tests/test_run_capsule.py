from __future__ import annotations

import json

import pytest
from run_capsule import read_config_arg, run

CONFIG = "task: predict\nmodel: m\nvideos:\n  - asset: s\n    camera: BottomCamera\n"


def test_missing_assets_fail_with_run_json(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    (data / "m").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"not mounted.*\['s'\]"):
        run(["--config", CONFIG, "--data-dir", str(data), "--results-dir", str(results)])
    record = json.loads((results / "run.json").read_text())
    assert record["status"] == "failed"
    assert "not mounted" in record["error"]
    assert (results / "run_config.yaml").read_text().startswith("task: predict")


def test_bad_config_fails_with_run_json(tmp_path):
    with pytest.raises(ValueError):
        run(["--config", "task: nope", "--results-dir", str(tmp_path)])
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "failed"


def test_config_arg_text_or_file(tmp_path):
    assert read_config_arg(CONFIG, tmp_path) == CONFIG.strip()
    (tmp_path / "c.yaml").write_text(CONFIG)
    assert read_config_arg("c.yaml", tmp_path) == CONFIG
    with pytest.raises(FileNotFoundError):
        read_config_arg("missing.yaml", tmp_path)
    with pytest.raises(ValueError, match="empty"):
        read_config_arg("  ", tmp_path)
