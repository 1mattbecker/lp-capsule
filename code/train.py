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

Context models read each labeled frame's neighbours t-2..t+2 from its folder.
Projects labeled from clips (DLC's ``labeled-data/<clip>/imgNNN.png`` for
``videos/<clip>.mp4``) often lack them; ``context_frames_from_videos`` writes
them from the clips (:func:`add_context`).
"""

from __future__ import annotations

import csv
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Self

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
# Mean |difference| (0-255) allowed between a labeled PNG and the same frame
# decoded from its clip: ffmpeg's and OpenCV's YUV->RGB differ by <= 3 on a few pixels.
CONTEXT_MAX_MEAN_DIFF = 0.5


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


def listing(root: Path, depth: int = 2, limit: int = 12) -> dict[str, Any] | str:
    """Return a folder's names, ``depth`` levels down, for error messages."""
    if not root.is_dir():
        return "missing"
    entries = sorted(root.iterdir())
    return {
        e.name + ("/" if e.is_dir() else ""): listing(e, depth - 1, limit)
        if e.is_dir() and depth > 1
        else ""
        for e in entries[:limit]
    }


def assemble_project(
    data_dir: Path, projects: tuple[str, ...], csv_file: str, work: Path, link: bool = False
) -> Path:
    """Return one LP data_dir for the project assets.

    A single asset is used as mounted, unless ``link``. Several are merged into
    ``work/project`` by symlinks: every ``labeled-data/<folder>`` and
    ``videos/<file>`` of each, and ``csv_file`` from the first asset that has it.
    A name in two assets is an error.
    """
    roots = [data_dir / p for p in projects]
    if len(roots) == 1 and not link:
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
        found = {p: listing(r if r.is_dir() else data_dir / Path(p).parts[0])
                 for p, r in zip(projects, roots)}  # fmt: skip
        raise FileNotFoundError(f"no project asset has {csv_file}; their contents: {found}")
    (merged / csv_file).symlink_to(with_csv[0] / csv_file)
    return merged


def check_label_paths(project: Path, labels: Path) -> int:
    """Return the number of labeled frames, failing if any image is missing."""
    paths = label_images(labels)
    missing = [p for p in paths if not (project / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} labeled images missing under {project}, "
            f"e.g. {missing[:3]}"
        )
    return len(paths)


def label_images(labels: Path) -> list[str]:
    """Return the image paths of an LP labels CSV."""
    with labels.open(newline="") as f:
        return [row[0] for i, row in enumerate(csv.reader(f)) if i >= 3 and row]


def _image_diff(a: Path, b: Path) -> float:
    import numpy as np
    from PIL import Image

    x, y = (np.asarray(Image.open(p).convert("RGB"), dtype=np.int16) for p in (a, b))
    return float(np.abs(x - y).mean()) if x.shape == y.shape else float("inf")


def _frame_number(image: str) -> int:
    """Return the frame number in an image name, ``img025.png`` -> 25."""
    digits = "".join(c for c in image.rsplit("/", 1)[-1] if c.isdigit())
    return int(digits)


def decoded_frames(mp4: Path) -> int:
    """Return the number of frames ffmpeg decodes from a video (not its index count)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip().split(",")[0])


def add_context(project: Path, labels: Path, work: Path) -> dict[str, Any]:
    """Write missing neighbours of the labeled PNGs in folders cut from a clip.

    ``project`` is a symlinked merge (:func:`assemble_project`). A folder
    ``labeled-data/<f>`` with ``videos/<f>.mp4`` becomes a real folder of links
    plus the new PNGs (``video_clips.add_context_frames``, frame = the PNG's
    digits). First each folder's first and last labeled frames are decoded from
    the clip and compared with the labeled PNGs, so a clip that is not the
    folder's source fails the run instead of training on wrong neighbours.

    Neighbours stop at the clip's last *decoded* frame: some clips' MP4 index
    lists more samples than decode (502 decoded, 591 indexed), and the labeled
    PNGs are numbered by decoded frame.
    """
    from aind_dynamic_foraging_behavior_video_analysis.video_clips import add_context_frames

    by_folder: dict[str, list[str]] = {}
    for image in label_images(labels):
        parts = image.split("/")
        if len(parts) == 3 and parts[2].endswith(".png"):
            by_folder.setdefault(parts[1], []).append(image)
    folders = {
        f: sorted(imgs, key=_frame_number)
        for f, imgs in by_folder.items()
        if (project / "videos" / f"{f}.mp4").is_file()
    }
    stage = work / "context"

    def one(folder: str) -> tuple[float, int]:
        root = stage / folder
        clip_dir = project / "videos"
        images = folders[folder]
        n_decoded = decoded_frames(clip_dir / f"{folder}.mp4")
        # The first and last labeled frames, decoded from the clip, against the PNGs.
        check = root / "check"
        (check / folder).mkdir(parents=True)
        (check / "CollectedData.csv").write_text(f"{images[0]}\n{images[-1]}\n")
        add_context_frames(check, clip_dir, offsets=(0,))
        diff = 0.0
        for image in dict.fromkeys((images[0], images[-1])):
            name = image.split("/")[-1]
            d = _image_diff(check / folder / name, project / image)
            if d > CONTEXT_MAX_MEAN_DIFF:
                raise ValueError(
                    f"{folder}/{name} differs from frame {name} of videos/{folder}.mp4 "
                    f"(mean |diff| {d:.2f}): not the folder's clip?"
                )
            diff = max(diff, d)
        # The folder as links, then each labeled frame's neighbours inside the clip.
        dest = root / "labeled" / folder
        dest.mkdir(parents=True)
        for entry in (project / "labeled-data" / folder).iterdir():
            (dest / entry.name).symlink_to(entry.resolve())
        written = 0
        for image in images:
            k = _frame_number(image)
            offsets = [d for d in (-2, -1, 1, 2) if 0 <= k + d < n_decoded]
            (root / "labeled" / "CollectedData.csv").write_text(image + "\n")
            written += len(add_context_frames(root / "labeled", clip_dir, offsets=offsets))
        return diff, written

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = dict(zip(folders, pool.map(one, folders)))
    for folder in folders:
        link = project / "labeled-data" / folder
        link.unlink()
        link.symlink_to(stage / folder / "labeled" / folder)
    return {
        "folders": len(folders),
        "frames": sum(len(v) for v in folders.values()),
        "written": sum(n for _, n in results.values()),
        "max_center_mean_abs_diff": max((d for d, _ in results.values()), default=None),
    }


class GpuMemory:
    """Sample ``nvidia-smi`` while training: the GPU, and the peak memory in use.

    ``torch.cuda.max_memory_allocated`` misses DALI's buffers (the unlabeled
    video batch, where 16 GB ran out before), so the whole device is sampled.
    """

    QUERY: ClassVar[list[str]] = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits", "-i", "0"]  # fmt: skip

    def __init__(self, every_s: float = 5.0) -> None:
        self.every_s = every_s
        self.name: str | None = None
        self.total_mib = 0
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            out = subprocess.run(
                self.QUERY, check=False, capture_output=True, text=True, timeout=10
            ).stdout
            name, total, used = (x.strip() for x in out.strip().split(","))
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        self.name, self.total_mib = name, int(total)
        self.peak_mib = max(self.peak_mib, int(used))

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            self._sample()

    def __enter__(self) -> Self:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_gib": round(self.total_mib / 1024, 2),
            "peak_used_gib": round(self.peak_mib / 1024, 2),
            "sampled_every_s": self.every_s,
        }


def build_cfg(config: TrainConfig, data_dir: Path, work: Path) -> tuple[Any, dict[str, Any]]:
    """Return the LP training config and facts for the run record."""
    from omegaconf import OmegaConf, open_dict

    project = assemble_project(
        data_dir, config.project, config.csv_file, work, link=config.context_frames_from_videos
    )
    labels = project / config.csv_file
    if not labels.is_file():
        raise FileNotFoundError(f"no labels CSV {labels}")
    n_labeled = check_label_paths(project, labels)
    context = add_context(project, labels, work) if config.context_frames_from_videos else None
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
        **({"context_frames": context} if context is not None else {}),
    }
    if config.init_checkpoint:
        info["init"] = init_checkpoint(cfg, data_dir / config.init_checkpoint, names, work)
        with open_dict(cfg):
            cfg.model.checkpoint = info["init"]["path"]
    return cfg, info


def pixel_error_summary(model_dir: Path, by_set: bool = True) -> pd.DataFrame | None:
    """Return mean and median pixel error per keypoint and set, if LP wrote them.

    With ``by_set`` False every frame is in one set, "all".

    Units are pixels of the original labeled frames: LP maps predictions back
    from the resized frame before comparing them with the labels.
    """
    path = model_dir / PIXEL_ERRORS
    if not path.is_file():
        return None
    errors = pd.read_csv(path, index_col=0)
    by = errors.pop("set") if "set" in errors else None
    if by is None or not by_set:
        by = pd.Series("all", index=errors.index)
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
    gpu = GpuMemory()
    try:
        with gpu:
            train(cfg, model_dir=model_dir)
    finally:
        # Also after an out-of-memory error: the peak says how far over it went.
        record.update(gpu=gpu.summary())

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
            # Every checkpoint kept (training.ckpt_every_n_epochs adds epoch=<n> ones);
            # predict's checkpoint: best | last | <name> picks among them.
            "checkpoints": sorted(p.name for p in model_dir.rglob("*.ckpt")),
        },
        pixel_error_px_original_frame=(
            summary.to_dict(orient="records") if summary is not None else None
        ),
    )
