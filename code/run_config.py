"""The run config: what one worker run does, read from YAML.

One run is one config with ``task: predict`` or ``task: train``. Assets are
named by their mount name under ``/data``; the launcher mounts each asset at
its name, and turns asset ids in a batch into names before writing configs.
Parsing is strict (unknown keys fail) and fills defaults, so the config the
worker writes to ``/results`` is the full config the run used.

Predict::

    task: predict
    model: ephys_BottomCamera_Pylon2Test2
    reader: dali          # or opencv
    precision: fp16       # or fp32
    batch_size: 0         # 0 = the model config's
    videos:
      - asset: behavior_782394_2025-04-23_10-51-14
        camera: BottomCamera          # or path: behavior-videos/BottomCamera/video.mp4
        windows: [[120000, 120500]]   # [start, end) source frames; omit = whole video
    labeled_frames: false # true: also score the model on its own labeled frames

Train::

    task: train
    project: pylon2test2_r01          # CollectedData.csv, labeled-data/, videos/
    base_config: ephys_BottomCamera_Pylon2Test2   # a model asset, or <asset>/<path>.yaml
    init_checkpoint: ephys_BottomCamera_Pylon2Test2   # optional: fine-tune from it
    overrides: {training: {max_epochs: 100, min_epochs: 100}}

This module imports nothing heavy, so the launcher uses it too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any

import yaml

READERS = ("dali", "opencv")
PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class VideoSpec:
    """One video of one session asset, and the frames to predict."""

    asset: str
    camera: str | None = None
    # Relative to the asset; used instead of looking the camera up.
    path: str | None = None
    # Sorted, non-overlapping [start, end) source frames; None is the whole video.
    windows: tuple[tuple[int, int], ...] | None = None

    @property
    def label(self) -> str:
        """Name of this video's outputs: ``<asset>_<camera or file stem>``."""
        view = self.camera or PurePosixPath(self.path or "").stem
        return f"{self.asset}_{view}"


@dataclass(frozen=True)
class PredictConfig:
    """A predict run: one model over windows of one or more videos."""

    model: str
    videos: tuple[VideoSpec, ...]
    reader: str = "dali"
    precision: str = "fp16"
    batch_size: int = 0
    # Write diagnostics.json about the first window (see predict.diagnose).
    debug: bool = False
    # Also predict the model's own labeled frames and report pixel error per
    # keypoint and train/val/test split (see predict.check_labeled_frames).
    labeled_frames: bool = False
    task: str = "predict"


@dataclass(frozen=True)
class TrainConfig:
    """A train run: an LP project, a starting config, optionally a checkpoint."""

    project: str
    base_config: str
    init_checkpoint: str | None = None
    csv_file: str = "CollectedData.csv"
    # Merged into the base config, nested as the config is.
    overrides: dict[str, Any] = field(default_factory=dict)
    # Predict the project's videos/ after training (slow on long videos).
    predict_videos: bool = False
    task: str = "train"


RunConfig = PredictConfig | TrainConfig


def load_config_text(text: str) -> dict[str, Any]:
    """Parse YAML (or JSON) config text into a mapping."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"the config must be a YAML mapping, got {type(raw).__name__}")
    return raw


def _check_keys(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _name(value: Any, where: str) -> str:
    """Return a mount name: a non-empty string with no slash."""
    if not isinstance(value, str) or not value or "/" in value:
        raise ValueError(f"{where} must be an asset (mount) name, got {value!r}")
    return value


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be true or false, got {value!r}")
    return value


def _choice(value: Any, choices: tuple[str, ...], where: str) -> str:
    if value not in choices:
        raise ValueError(f"{where} must be one of {list(choices)}, got {value!r}")
    return str(value)


def parse_windows(raw: Any, where: str) -> tuple[tuple[int, int], ...] | None:
    """Return windows sorted by start, refusing empty, negative or overlapping ones."""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where} must be a non-empty list of [start, end) pairs")
    windows = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in item)
        ):
            raise ValueError(f"{where}: each window is [start, end), got {item!r}")
        start, end = item
        if start < 0 or end <= start:
            raise ValueError(f"{where}: window [{start}, {end}) is empty or negative")
        windows.append((start, end))
    windows.sort()
    for (_, end), (start, _) in pairwise(windows):
        if start < end:
            raise ValueError(f"{where}: windows overlap at frame {start}")
    return tuple(windows)


def parse_video(raw: Any, where: str) -> VideoSpec:
    """Parse one entry of ``videos``."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping with asset and camera or path")
    _check_keys(raw, {"asset", "camera", "path", "windows"}, where)
    camera, path = raw.get("camera"), raw.get("path")
    if (camera is None) == (path is None):
        raise ValueError(f"{where}: give exactly one of camera and path")
    for key, value in (("camera", camera), ("path", path)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{where}.{key} must be a non-empty string")
    if path is not None and (path.startswith("/") or ".." in PurePosixPath(path).parts):
        raise ValueError(f"{where}.path must be relative to the asset, got {path!r}")
    return VideoSpec(
        asset=_name(raw.get("asset"), f"{where}.asset"),
        camera=camera,
        path=path,
        windows=parse_windows(raw.get("windows"), f"{where}.windows"),
    )


def parse_predict(raw: dict[str, Any]) -> PredictConfig:
    """Parse a ``task: predict`` config."""
    _check_keys(
        raw,
        {"task", "model", "videos", "reader", "precision", "batch_size", "debug", "labeled_frames"},
        "predict",
    )
    labeled_frames = _flag(raw.get("labeled_frames", False), "labeled_frames")
    videos = raw.get("videos", [] if labeled_frames else None)
    if not isinstance(videos, list) or not (videos or labeled_frames):
        raise ValueError("predict: videos must be a non-empty list (or set labeled_frames)")
    specs = tuple(parse_video(v, f"videos[{i}]") for i, v in enumerate(videos))
    labels = [s.label for s in specs]
    repeated = sorted({x for x in labels if labels.count(x) > 1})
    if repeated:
        raise ValueError(f"predict: videos listed more than once: {repeated}")
    batch_size = raw.get("batch_size", 0)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
        raise ValueError(f"batch_size must be an integer >= 0, got {batch_size!r}")
    return PredictConfig(
        model=_name(raw.get("model"), "model"),
        videos=specs,
        reader=_choice(raw.get("reader", "dali"), READERS, "reader"),
        precision=_choice(raw.get("precision", "fp16"), PRECISIONS, "precision"),
        batch_size=batch_size,
        debug=_flag(raw.get("debug", False), "debug"),
        labeled_frames=labeled_frames,
    )


def parse_train(raw: dict[str, Any]) -> TrainConfig:
    """Parse a ``task: train`` config."""
    _check_keys(
        raw,
        {
            "task",
            "project",
            "base_config",
            "init_checkpoint",
            "csv_file",
            "overrides",
            "predict_videos",
        },
        "train",
    )
    base = raw.get("base_config")
    if not isinstance(base, str) or not base or base.startswith("/"):
        raise ValueError(
            f"base_config must be a model asset name or <asset>/<path>.yaml, got {base!r}"
        )
    init = raw.get("init_checkpoint")
    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a mapping, nested as the LP config is")
    csv_file = raw.get("csv_file", "CollectedData.csv")
    if not isinstance(csv_file, str) or "/" in csv_file:
        raise ValueError(f"csv_file must be a file name in the project, got {csv_file!r}")
    predict_videos = raw.get("predict_videos", False)
    if not isinstance(predict_videos, bool):
        raise ValueError("predict_videos must be true or false")
    return TrainConfig(
        project=_name(raw.get("project"), "project"),
        base_config=base,
        init_checkpoint=None if init is None else _name(init, "init_checkpoint"),
        csv_file=csv_file,
        overrides=overrides,
        predict_videos=predict_videos,
    )


def parse_config(raw: dict[str, Any]) -> RunConfig:
    """Parse a run config by its ``task``."""
    task = raw.get("task")
    if task == "predict":
        return parse_predict(raw)
    if task == "train":
        return parse_train(raw)
    raise ValueError(f"task must be predict or train, got {task!r}")


def base_config_asset(base_config: str) -> str:
    """Return the asset a ``base_config`` names: its first path component."""
    return PurePosixPath(base_config).parts[0]


def assets_of(config: RunConfig) -> list[str]:
    """Return the mount names a run needs, in order, each once."""
    if isinstance(config, PredictConfig):
        names = [config.model, *(v.asset for v in config.videos)]
    else:
        names = [config.project, base_config_asset(config.base_config)]
        if config.init_checkpoint:
            names.append(config.init_checkpoint)
    return list(dict.fromkeys(names))


def to_dict(config: RunConfig) -> dict[str, Any]:
    """Return a config as plain data, defaults filled, ready to dump as YAML."""
    if isinstance(config, PredictConfig):
        return {
            "task": "predict",
            "model": config.model,
            "reader": config.reader,
            "precision": config.precision,
            "batch_size": config.batch_size,
            **({"debug": True} if config.debug else {}),
            **({"labeled_frames": True} if config.labeled_frames else {}),
            "videos": [
                {
                    "asset": v.asset,
                    **({"camera": v.camera} if v.camera else {}),
                    **({"path": v.path} if v.path else {}),
                    **({"windows": [list(w) for w in v.windows]} if v.windows is not None else {}),
                }
                for v in config.videos
            ],
        }
    return {
        "task": "train",
        "project": config.project,
        "base_config": config.base_config,
        "init_checkpoint": config.init_checkpoint,
        "csv_file": config.csv_file,
        "overrides": config.overrides,
        "predict_videos": config.predict_videos,
    }


def dump(config: RunConfig) -> str:
    """Return a config as YAML text."""
    return yaml.safe_dump(to_dict(config), sort_keys=False, default_flow_style=None)
