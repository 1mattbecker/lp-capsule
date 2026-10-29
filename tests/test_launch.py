from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from run_config import load_config_text, parse_config

spec = importlib.util.spec_from_file_location(
    "launch", Path(__file__).parents[1] / "scripts/launch.py"
)
launch = importlib.util.module_from_spec(spec)
sys.modules["launch"] = launch
spec.loader.exec_module(launch)

BATCH = """
task: predict
model: 11111111-2222-3333-4444-555555555555
videos:
  - {asset: sess_a, camera: BottomCamera, windows: [[0, 10]]}
  - {asset: sess_a, camera: SideCameraLeft, windows: [[0, 10]]}
  - {asset: sess_b, camera: BottomCamera}
  - {asset: sess_c, camera: BottomCamera}
"""


class FakeAssets:
    def __init__(self):
        self.named = {
            "sess_a": "a" * 8 + "-0000-0000-0000-" + "0" * 12,
            "sess_b": "b" * 8 + "-0000-0000-0000-" + "0" * 12,
            "sess_c": "c" * 8 + "-0000-0000-0000-" + "0" * 12,
        }

    def get_data_asset(self, asset_id):
        return SimpleNamespace(id=asset_id, name="model_x")

    def search_data_assets(self, params):
        name = params.query.removeprefix("name:")
        results = [SimpleNamespace(id=self.named[name], name=name)] if name in self.named else []
        return SimpleNamespace(results=results)


def test_split_by_session():
    config = parse_config(load_config_text(BATCH))
    runs = launch.split_predict(config, 2)
    assert [[v.asset for v in r.videos] for r in runs] == [
        ["sess_a", "sess_a", "sess_b"],
        ["sess_c"],
    ]
    assert launch.run_label(runs[0]) == "sess_a+1"


def test_plan_resolves_ids_to_mount_names():
    client = SimpleNamespace(data_assets=FakeAssets())
    config = parse_config(load_config_text(BATCH))
    records = launch.plan(client, launch.split_predict(config, 1), [None] * 3)
    first = records[0]
    assert first["label"] == "sess_a"
    assert [a["name"] for a in first["assets"]] == ["model_x", "sess_a"]
    assert parse_config(load_config_text(first["config"])).model == "model_x"


def test_unknown_asset_fails():
    client = SimpleNamespace(data_assets=FakeAssets())
    with pytest.raises(LookupError, match="sess_z"):
        launch.resolve_asset(client, "sess_z")


def test_merge_manifest_keeps_finished(tmp_path):
    new = [
        {"label": "a", "status": "queued", "config": "c1"},
        {"label": "b", "status": "queued", "config": "c2"},
        {"label": "c", "status": "queued", "config": "c3"},
    ]
    old = [
        {"label": "a", "status": "completed", "config": "c1"},
        {"label": "b", "status": "failed", "config": "c2"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    merged = launch.merge_manifest(tmp_path, new, retry=False)
    assert [r["status"] for r in merged] == ["completed", "failed", "queued"]
    merged = launch.merge_manifest(tmp_path, new, retry=True)
    assert [r["status"] for r in merged] == ["completed", "queued", "queued"]
    old[0]["config"] = "changed"
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="config changed"):
        launch.merge_manifest(tmp_path, new, retry=False)


def test_labeled_frames_only_is_one_run_and_first_run_only():
    only = parse_config({"task": "predict", "model": "m", "labeled_frames": True})
    runs = launch.split_predict(only, 1)
    assert len(runs) == 1 and launch.run_label(runs[0]) == "m_labeled_frames"
    both = parse_config(load_config_text(BATCH) | {"labeled_frames": True})
    assert [r.labeled_frames for r in launch.split_predict(both, 1)] == [True, False, False]


TRAIN_BATCH = """
defaults:
  train:
    project: [sess_a/train, sess_b]
    base_config: sess_b
    overrides: {training: {max_epochs: 300, ckpt_every_n_epochs: 50}, model: {losses_to_use: []}}
runs:
  - name: exp_all
    task: train
    csv_file: train_all.csv
  - name: exp_ctx
    task: train
    csv_file: train_all.csv
    overrides: {model: {model_type: heatmap_mhcrnn}}
  - name: exp_eval
    task: predict
    model: sess_c
    checkpoint: last
    labeled_frames: [sess_a/test/t1.csv, sess_a/test/t2.csv]
"""


def test_batch_defaults_names_and_capture():
    configs, names = launch.read_batch(TRAIN_BATCH)
    assert names == ["exp_all", "exp_ctx", "exp_eval"]
    ctx = configs[1].overrides
    assert ctx["model"] == {"losses_to_use": [], "model_type": "heatmap_mhcrnn"}
    assert ctx["training"]["ckpt_every_n_epochs"] == 50
    assert configs[2].checkpoint == "last"
    client = SimpleNamespace(data_assets=FakeAssets())
    records = launch.plan(client, configs, names, capture_names=False)
    assert [r.get("capture") for r in records] == ["models", "models", None]
    assert [r.get("capture_name") for r in records] == ["exp_all", "exp_ctx", None]
    assert [a["name"] for a in records[2]["assets"]] == ["sess_c", "sess_a"]
    with pytest.raises(ValueError, match="repeat"):
        launch.read_batch(TRAIN_BATCH.replace("exp_ctx", "exp_all"))


def test_run_without_results_is_queued_again(tmp_path):
    comp = SimpleNamespace(
        id="c1",
        state=launch.ComputationState.Failed,
        end_status=launch.ComputationEndStatus.Failed,
        exit_code=None,
        has_results=False,
    )
    client = SimpleNamespace(computations=SimpleNamespace(get_computation=lambda _: comp))
    record = {"label": "r", "status": "running", "computation_id": "c1"}
    launch.advance(client, record, tmp_path, None)
    assert record["status"] == "queued" and record["attempts"] == 2
    assert record["not_before"] > launch.time.time() + 100
    assert [launch.retry_delay(n) for n in (2, 3, 6, 7)] == [120, 240, 1800, 1800]
    record.update(status="running", attempts=launch.MAX_ATTEMPTS)
    launch.advance(client, record, tmp_path, None)
    assert record["status"] == "failed"
