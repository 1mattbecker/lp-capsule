"""Training or fine-tuning a Lightning Pose model on an LP project.

The project asset is an LP project folder: the labels CSV, ``labeled-data/``
and, for the semi-supervised losses, ``videos/``. The base config's paths are
pointed at it, ``overrides`` are merged in, and the project's keypoint names
are written in, so the output is a bundle :mod:`predict` reads as is:
``model/config.yaml``, the best checkpoint under ``model/tb_logs/``, and LP's
predictions and pixel errors on the labeled frames.

Fine-tuning sets LP's ``model.checkpoint``, which LP loads with
``strict=False``: an LP 1.6 checkpoint's head (``upsampling_layers.*``) would be
skipped without a word. So the init checkpoint is first loaded strictly in its
own architecture, its head renamed, checked against the new config, and saved
as a temporary checkpoint whose every key LP's model has.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd
from bundle import (
    csv_keypoint_names,
    file_sha256,
    find_bundle,
    find_config,
    keypoint_names,
    read_lp_config,
)
from run_config import TrainConfig

# Config entries an init checkpoint's architecture must share with the new one.
ARCHITECTURE = (
    "model.model_type",
    "model.backbone",
    "data.num_keypoints",
    "data.image_resize_dims.height",
    "data.image_resize_dims.width",
    "data.downsample_factor",
)
PIXEL_ERRORS = "predictions_pixel_error.csv"


def base_config_path(config: TrainConfig, data_dir: Path) -> tuple[Path, list[str] | None]:
    """Return the base LP config, and its model's keypoint names when known."""
    path = data_dir / config.base_config
    if path.is_file():
        return path, None
    found = find_config(path)
    raw = read_lp_config(found)
    assert raw is not None
    try:
        names, _ = keypoint_names(raw, path)
    except ValueError:
        names = None
    return found, names


def _select(cfg: Any, dotted: str) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, dotted)


def init_checkpoint(cfg: Any, asset: Path, names: list[str], work: Path) -> dict[str, Any]:
    """Write the init checkpoint LP will load (see the module docstring)."""
    import torch
    from omegaconf import OmegaConf, open_dict
    from predict import load_module

    init = find_bundle(asset)
    if list(init.keypoint_names) != names:
        raise ValueError(
            f"init_checkpoint keypoints {list(init.keypoint_names)} differ from the "
            f"project's {names}"
        )
    init_cfg = OmegaConf.load(init.config)
    with open_dict(init_cfg):
        init_cfg.data.keypoint_names = names
    differ = [k for k in ARCHITECTURE if _select(init_cfg, k) != _select(cfg, k)]
    if differ:
        raise ValueError(
            "init_checkpoint's architecture differs from the training config at "
            + ", ".join(f"{k} ({_select(init_cfg, k)} vs {_select(cfg, k)})" for k in differ)
        )
    # Strict, so every renamed key is one this architecture has.
    module = load_module(init_cfg, init.checkpoint)
    path = work / "init.ckpt"
    torch.save({"state_dict": module.state_dict()}, path)
    del module
    return {
        "asset": asset.name,
        "checkpoint": str(init.checkpoint.relative_to(asset)),
        "checkpoint_sha256": file_sha256(init.checkpoint),
        "path": str(path),
    }


def assemble_project(data_dir: Path, projects: tuple[str, ...], csv_file: str, work: Path) -> Path:
    """Return one LP data_dir for the project assets.

    A single asset is used as mounted. Several are merged into ``work/project`` by
    symlinks: every ``labeled-data/<folder>`` and ``videos/<file>`` of each, and
    ``csv_file`` from the first asset that has it. A name in two assets is an error.
    """
    roots = [data_dir / p for p in projects]
    if len(roots) == 1:
        return roots[0]
    merged = work / "project"
    for sub in ("labeled-data", "videos"):
        (merged / sub).mkdir(parents=True)
        for root in roots:
            for entry in sorted((root / sub).glob("*")) if (root / sub).is_dir() else []:
                link = merged / sub / entry.name
                if link.is_symlink():
                    raise ValueError(f"{sub}/{entry.name} is in more than one project asset")
                link.symlink_to(entry)
    with_csv = [root for root in roots if (root / csv_file).is_file()]
    if not with_csv:
        raise FileNotFoundError(f"no project asset has {csv_file}: {list(projects)}")
    (merged / csv_file).symlink_to(with_csv[0] / csv_file)
    return merged


def check_label_paths(project: Path, labels: Path) -> int:
    """Return the number of labeled frames, failing if any image is missing."""
    with labels.open(newline="") as f:
        paths = [row[0] for i, row in enumerate(csv.reader(f)) if i >= 3 and row]
    missing = [p for p in paths if not (project / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} labeled images missing under {project}, "
            f"e.g. {missing[:3]}"
        )
    return len(paths)


def build_cfg(config: TrainConfig, data_dir: Path, work: Path) -> tuple[Any, dict[str, Any]]:
    """Return the LP training config and facts for the run record."""
    from omegaconf import OmegaConf, open_dict

    project = assemble_project(data_dir, config.project, config.csv_file, work)
    labels = project / config.csv_file
    if not labels.is_file():
        raise FileNotFoundError(f"no labels CSV {labels}")
    n_labeled = check_label_paths(project, labels)
    names = csv_keypoint_names(labels)
    base, base_names = base_config_path(config, data_dir)
    if base_names is not None and base_names != names:
        raise ValueError(
            f"the project's keypoints {names} differ from the base model's "
            f"{base_names}; give a base_config with these keypoints"
        )
    cfg = OmegaConf.merge(OmegaConf.load(base), OmegaConf.create(config.overrides))
    videos = project / "videos"
    with open_dict(cfg):
        cfg.data.data_dir = str(project)
        cfg.data.csv_file = config.csv_file
        cfg.data.video_dir = str(videos)
        cfg.data.keypoint_names = names
        cfg.eval.test_videos_directory = str(videos)
        cfg.eval.predict_vids_after_training = config.predict_videos
        cfg.eval.save_vids_after_training = False
        cfg.model.checkpoint = None
        if not cfg.training.get("num_gpus"):
            import torch

            cfg.training.num_gpus = max(torch.cuda.device_count(), 1)
        cfg.training.num_workers = min(int(cfg.training.get("num_workers", 4)), os.cpu_count() or 1)
    if cfg.data.num_keypoints != len(names):
        raise ValueError(
            f"{labels} has {len(names)} keypoints, the config num_keypoints "
            f"{cfg.data.num_keypoints}"
        )
    losses = list(cfg.model.get("losses_to_use") or [])
    if losses and not any(videos.glob("*.mp4")):
        raise FileNotFoundError(
            f"losses {losses} train on unlabeled video, but {videos} has no .mp4"
        )
    info: dict[str, Any] = {
        "project": list(config.project),
        "labels": config.csv_file,
        "n_labeled_frames": n_labeled,
        "keypoint_names": names,
        "base_config": str(base.relative_to(data_dir)),
    }
    if config.init_checkpoint:
        info["init"] = init_checkpoint(cfg, data_dir / config.init_checkpoint, names, work)
        with open_dict(cfg):
            cfg.model.checkpoint = info["init"]["path"]
    return cfg, info


def pixel_error_summary(model_dir: Path) -> pd.DataFrame | None:
    """Return mean and median pixel error per keypoint and set, if LP wrote them.

    Units are pixels of the original labeled frames: LP maps predictions back
    from the resized frame before comparing them with the labels.
    """
    path = model_dir / PIXEL_ERRORS
    if not path.is_file():
        return None
    errors = pd.read_csv(path, index_col=0)
    by = errors.pop("set") if "set" in errors else pd.Series("all", index=errors.index)
    long = errors.assign(set=by).melt(id_vars="set", var_name="keypoint", value_name="px")
    summary = long.groupby(["set", "keypoint"], sort=False)["px"].agg(["mean", "median", "count"])
    return summary.reset_index()


def run_train(
    config: TrainConfig, data_dir: Path, results_dir: Path, work: Path, record: Any
) -> None:
    """Train, then check the output loads as a bundle and summarize its errors."""
    import torch.multiprocessing
    from lightning_pose.train import train
    from omegaconf import OmegaConf, open_dict

    # The default sharing strategy hits Docker's small /dev/shm (bus error).
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg, info = build_cfg(config, data_dir, work)
    record.update(train=info)
    model_dir = results_dir / "model"
    train(cfg, model_dir=model_dir)

    # The temporary init checkpoint is gone after the run; run.json names its source.
    saved = model_dir / "config.yaml"
    out_cfg = OmegaConf.load(saved)
    with open_dict(out_cfg):
        out_cfg.model.checkpoint = None
    OmegaConf.save(out_cfg, saved)

    bundle = find_bundle(model_dir)
    summary = pixel_error_summary(model_dir)
    if summary is not None:
        summary.to_csv(results_dir / "pixel_error_summary.csv", index=False)
    record.update(
        bundle={
            "config": str(bundle.config.relative_to(results_dir)),
            "checkpoint": str(bundle.checkpoint.relative_to(results_dir)),
            "checkpoint_sha256": file_sha256(bundle.checkpoint),
        },
        pixel_error_px_original_frame=(
            summary.to_dict(orient="records") if summary is not None else None
        ),
    )
