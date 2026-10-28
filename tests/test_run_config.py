from __future__ import annotations

import pytest
from run_config import (
    PredictConfig,
    TrainConfig,
    assets_of,
    dump,
    load_config_text,
    parse_config,
)

PREDICT = """
task: predict
model: ephys_BottomCamera_Pylon2Test2
videos:
  - asset: behavior_782394_2025-04-23_10-51-14
    camera: BottomCamera
    windows: [[455000, 455500], [120000, 120500]]
  - asset: behavior_791691_2025-06-01_10-00-00
    path: behavior-videos/BottomCamera/video.mp4
"""


def test_predict_defaults_and_sorted_windows():
    config = parse_config(load_config_text(PREDICT))
    assert isinstance(config, PredictConfig)
    assert (config.reader, config.precision, config.batch_size) == ("dali", "fp16", 0)
    first, second = config.videos
    assert first.windows == ((120000, 120500), (455000, 455500))
    assert first.label == "behavior_782394_2025-04-23_10-51-14_BottomCamera"
    assert second.windows is None
    assert second.label == "behavior_791691_2025-06-01_10-00-00_video"


def test_dump_round_trips():
    config = parse_config(load_config_text(PREDICT))
    assert parse_config(load_config_text(dump(config))) == config


def test_assets_once_in_order():
    config = parse_config(load_config_text(PREDICT))
    assert assets_of(config) == [
        "ephys_BottomCamera_Pylon2Test2",
        "behavior_782394_2025-04-23_10-51-14",
        "behavior_791691_2025-06-01_10-00-00",
    ]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"reader": "pynvvc"}, "reader"),
        ({"extra": 1}, "unknown keys"),
        ({"videos": []}, "non-empty"),
        ({"videos": [{"asset": "s", "camera": "C", "path": "p.mp4"}]}, "exactly one"),
        ({"videos": [{"asset": "s"}]}, "exactly one"),
        ({"videos": [{"asset": "s", "path": "/abs.mp4"}]}, "relative"),
        ({"videos": [{"asset": "a/b", "camera": "C"}]}, "asset .mount. name"),
        ({"videos": [{"asset": "s", "camera": "C", "windows": [[5, 5]]}]}, "empty"),
        ({"videos": [{"asset": "s", "camera": "C", "windows": [[0, 10], [5, 20]]}]}, "overlap"),
        ({"videos": [{"asset": "s", "camera": "C", "windows": [[0, 1.5]]}]}, "start, end"),
        ({"videos": [{"asset": "s", "camera": "C"}] * 2}, "more than once"),
        ({"batch_size": -1}, "batch_size"),
    ],
)
def test_predict_errors(change, message):
    raw = load_config_text(PREDICT) | change
    with pytest.raises(ValueError, match=message):
        parse_config(raw)


def test_train_assets_and_round_trip():
    raw = {
        "task": "train",
        "project": "pylon2test2_r01",
        "base_config": "ephys_BottomCamera_Pylon2Test2/training.config.yaml",
        "init_checkpoint": "ephys_BottomCamera_Pylon2Test2",
        "overrides": {"training": {"max_epochs": 100}},
    }
    config = parse_config(raw)
    assert isinstance(config, TrainConfig)
    assert assets_of(config) == ["pylon2test2_r01", "ephys_BottomCamera_Pylon2Test2"]
    assert parse_config(load_config_text(dump(config))) == config


def test_task_required():
    with pytest.raises(ValueError, match="task"):
        parse_config({"model": "m"})


def test_labeled_frames_without_videos():
    config = parse_config({"task": "predict", "model": "m", "labeled_frames": True})
    assert config.labeled_frames and config.videos == ()
    assert assets_of(config) == ["m"]
    assert parse_config(load_config_text(dump(config))) == config
    with pytest.raises(ValueError, match="labeled_frames"):
        parse_config({"task": "predict", "model": "m"})
