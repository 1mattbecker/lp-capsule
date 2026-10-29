"""Running a Lightning Pose model over windows of videos.

Per window, :func:`video_clips.cut_clip` cuts a frame-exact temporary clip
(seeking by the MP4's own sample timestamps, with even timestamps out), LP
predicts the whole clip, and the rows are re-indexed to source frames. A whole
video is predicted as is, after a temporary uniform restamp if DALI reads it
and its timeline is notched. The output is keyed by source frame index only:
row ``i`` of the video is frame ``i`` of the MP4, also when frames were
dropped, and behavior time is joined downstream (``video_timing_qc``).

Model loading follows ``predict.load_model`` in
AllenNeuralDynamics/lightning-pose-inference-capsule (MIT): the module is built
from the checkpoint's hyperparameters without labeled data and loaded strictly.
Relies on Lightning Pose 2.4.3 internals, named where used.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
from aind_dynamic_foraging_behavior_video_analysis.video_clips import cut_clip
from aind_video_utils import read_mp4_frame_index, restamp_uniform
from bundle import Bundle, file_sha256, find_bundle
from run_config import PredictConfig, VideoSpec
from session import local_list_dir, locate_camera


@dataclass
class VideoResult:
    """One video's predictions and how they were made."""

    predictions: pd.DataFrame
    info: dict[str, Any]


def load_module(cfg: Any, checkpoint: Path) -> Any:
    """Return the config's LP module with the checkpoint's weights, loaded strictly.

    Building the network fetches backbone weights only for the checkpoint to
    overwrite them: ``pretrained`` gates the torchvision ones, and
    ``resnet50_animal_*`` backbones fetch theirs from OpenMMLab regardless, so
    that fetch is stubbed out. Checkpoints older than LP's head refactor
    (LP 1.6) name the head's layers ``upsampling_layers.*``; see
    :func:`rename_head_keys`.
    """
    import torch
    from lightning_pose.models import check_if_semi_supervised, get_model_class

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    n = hparams.get("num_keypoints")
    if n is not None and n != cfg.data.num_keypoints:
        raise ValueError(f"{checkpoint} has {n} keypoints, its config {cfg.data.num_keypoints}")
    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    module_class = get_model_class(cfg.model.model_type, semi_supervised)
    factories: dict[str, None] = {"loss_factory": None}
    if semi_supervised:
        factories["loss_factory_unsupervised"] = None
    with mock.patch.object(torch.hub, "load_state_dict_from_url", return_value={"state_dict": {}}):
        module = module_class(**(hparams | {"pretrained": False}), **factories)
    module.load_state_dict(rename_head_keys(ckpt["state_dict"]), strict=True)
    module.eval()
    return module


def rename_head_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Rename LP 1.6 ``upsampling_layers.*`` weights to LP 2's ``head.*``."""
    return {f"head.{k}" if k.startswith("upsampling_layers.") else k: v for k, v in state.items()}


def load_model(
    bundle: Bundle, batch_size: int, precision: str, work: Path, videos: bool = True
) -> tuple[Any, int]:
    """Return an LP ``Model`` holding the bundle's weights, and its batch size.

    The keypoint names are written into the config in memory: LP's prediction
    handler refuses a config without ``data.keypoint_names``. Without ``videos``
    (labeled frames only) context models load too: LP reads their neighbours
    from the labeled images' folders.
    """
    from lightning_pose.api.model import Model
    from lightning_pose.api.model_config import ModelConfig
    from omegaconf import open_dict

    config = ModelConfig.from_yaml_file(bundle.config)
    cfg = config.cfg
    if cfg.model.model_type != "heatmap" and videos:
        # Context models read neighbouring frames, which a clip's first and last
        # frames lack; not supported until a model needs it.
        raise NotImplementedError(f"model type {cfg.model.model_type} is unsupported")
    with open_dict(cfg):
        cfg.data.keypoint_names = list(bundle.keypoint_names)
        if batch_size:
            cfg.dali.base.predict.sequence_length = batch_size
    # The directory is only where LP would put outputs we never ask for; the
    # weights are set here, so LP never looks for a checkpoint in it.
    model = Model(work / "model", config, precision=precision)
    model.model = load_module(cfg, bundle.checkpoint)
    return model, int(cfg.dali.base.predict.sequence_length)


def predict_file(model: Any, mp4: Path, reader: str, out_dir: Path) -> pd.DataFrame:
    """Return LP's predictions for every frame of ``mp4``, rows 0..n-1."""
    result = model.predict_on_video_file(
        mp4, output_dir=out_dir, compute_metrics=False, reader=reader
    )
    return result.predictions


def reindex(predictions: pd.DataFrame, start: int, n_frames: int) -> pd.DataFrame:
    """Return a clip's predictions indexed by source frame ``start + row``."""
    if len(predictions) != n_frames:
        raise RuntimeError(
            f"got {len(predictions)} predictions for the {n_frames} frames from {start}"
        )
    out = predictions.copy()
    out.index = pd.RangeIndex(start, start + n_frames)
    return out


def video_path(session_dir: Path, spec: VideoSpec) -> Path:
    """Return the video a spec names, found by camera unless a path is given."""
    relative = spec.path or locate_camera(local_list_dir(session_dir), spec.camera or "")
    path = session_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"no video {path}")
    return path


def _stats(tensor: Any) -> dict[str, Any]:
    t = tensor.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
        "finite": bool(t.isfinite().all()),
    }


def _forward(module: Any, frames: Any) -> dict[str, Any]:
    """Run the module on frames as predict_step does; return heatmap and keypoint stats."""
    import torch

    with torch.no_grad():
        heatmaps = module(frames)
        keypoints, confidence = module.head.run_subpixelmaxima(heatmaps)
    return {
        "heatmaps": _stats(heatmaps),
        "keypoints_resized_frame": keypoints[0].reshape(-1, 2).tolist(),
        "confidence": confidence[0].tolist(),
    }


def diagnose(model: Any, clip: Path, reader: str) -> dict[str, Any]:
    """Facts about the first clip, to tell bad weights from bad input.

    The module runs on clip frame 0 twice: decoded by ffmpeg and preprocessed by
    hand (bilinear resize, ImageNet normalization), and as LP's reader delivers it.
    """
    import subprocess

    import numpy as np
    import torch
    from lightning_pose.data.video.factory import build_video_reader

    module = model.model
    device = next(module.parameters()).device
    cfg = model.cfg
    height, width = cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width
    out: dict[str, Any] = {
        "module_class": type(module).__name__,
        "module_training": bool(module.training),
        "device": str(device),
        "head_weight_abs_sum": float(module.head.upsampling_layers[1].weight.abs().sum()),
        "head_bias_abs_sum": float(module.head.upsampling_layers[1].bias.abs().sum()),
        "head_temperature": float(module.head.temperature),
        "resize_dims": [height, width],
    }
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,pix_fmt,color_range", "-of", "json", str(clip)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    out["clip_probe"] = probe.stdout
    stream = json.loads(probe.stdout)["streams"][0]
    w, h = stream["width"], stream["height"]
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    rgb = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy())
    image = rgb.permute(2, 0, 1)[None].float().to(device) / 255
    image = torch.nn.functional.interpolate(image, size=(height, width), mode="bilinear")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    manual = (image - mean) / std
    out["manual_input"] = _stats(manual)
    out["manual"] = _forward(module, manual)

    try:
        loader = build_video_reader(
            reader=reader,
            probe_video=str(clip),
            model_type="base",
            dali_config=cfg.dali,
            filenames=[str(clip)],
            resize_dims=[height, width],
            bbox_df=None,
        )
        frames = next(iter(loader))["frames"]
        out["reader_input"] = _stats(frames)
        first = frames[:1].float().to(device)
        out["reader_minus_manual_abs_mean"] = float((first - manual).abs().mean())
        out["reader"] = _forward(module, first)
    except Exception as e:  # noqa: BLE001 -- diagnostics must not fail the run
        out["reader_error"] = f"{type(e).__name__}: {e}"
    return out


def predict_video(
    model: Any,
    mp4: Path,
    spec: VideoSpec,
    reader: str,
    work: Path,
    diagnostics: Path | None = None,
) -> VideoResult:
    """Predict a video's windows, or the whole video, keyed by source frame.

    With ``diagnostics``, :func:`diagnose` the first window's clip and write the
    result there, with LP's own prediction of that clip's frame 0.
    """
    index = read_mp4_frame_index(mp4)
    info: dict[str, Any] = {
        "video": str(mp4),
        "n_frames_in_video": int(index.n_samples),
        "windows": [],
    }
    if spec.windows is None:
        source, notched_after = mp4, []
        if reader == "dali":
            # DALI finds frame k at k frame periods, so a notched timeline makes it
            # read wrong frames; a uniform restamp leaves the frames untouched.
            notches = index.timeline_notches()
            if not notches.is_uniform:
                source = restamp_uniform(mp4, work / "restamped.mp4")
                notched_after = notches.pts_after.tolist()
        t0 = time.time()
        frames = reindex(predict_file(model, source, reader, work / "preds"), 0, index.n_samples)
        info["restamped_after_frame_index"] = notched_after
        info["windows"].append(
            {"start": 0, "end": int(index.n_samples), "seconds": round(time.time() - t0, 1)}
        )
        if source != mp4:
            source.unlink()
        return VideoResult(frames, info)

    parts = []
    for start, end in spec.windows:
        t0 = time.time()
        clip = cut_clip(mp4, start, end - start, work / f"clip_{start:08d}.mp4", index=index)
        t1 = time.time()
        try:
            found = diagnose(model, clip, reader) if diagnostics and not parts else None
            parts.append(
                reindex(predict_file(model, clip, reader, work / "preds"), start, end - start)
            )
            if found is not None:
                found["lp_prediction_row0"] = parts[0].iloc[0].tolist()
                found["head_weight_abs_sum_after"] = float(
                    model.model.head.upsampling_layers[1].weight.abs().sum()
                )
                diagnostics.write_text(json.dumps(found, indent=2) + "\n")
        finally:
            clip.unlink(missing_ok=True)
        info["windows"].append(
            {
                "start": start,
                "end": end,
                "cut_seconds": round(t1 - t0, 1),
                "predict_seconds": round(time.time() - t1, 1),
            }
        )
    return VideoResult(pd.concat(parts), info)


def check_labeled_frames(
    model: Any,
    bundle: Bundle,
    labels_spec: bool | str,
    data_dir: Path,
    out_dir: Path,
    prefix: str = "",
) -> dict[str, Any]:
    """Predict labeled frames; write pixel errors to ``out_dir``, names prefixed.

    With ``labels_spec`` True the labels are the training CSV in the bundle
    (``data.csv_file``'s name), and LP rebuilds the training run's
    train/val/test split from the config's seeds. As ``<asset>/<labels.csv>``
    they are that CSV, its image paths relative to its folder, in one set "all".
    The semi-supervised losses are switched off for this, so LP builds a
    labeled-only data module; the network is unchanged.
    """
    import shutil

    from omegaconf import open_dict
    from train import pixel_error_summary

    cfg = model.cfg
    if isinstance(labels_spec, str):
        labels = data_dir / labels_spec
        images = labels.parent
        if not labels.is_file():
            raise FileNotFoundError(f"no labels CSV {labels}")
    else:
        images = bundle.root
        labels = bundle.root / Path(str(cfg.data.get("csv_file") or "CollectedData.csv")).name
        if not labels.is_file():
            found = sorted(bundle.root.glob("CollectedData*.csv"))
            if not found:
                raise FileNotFoundError(f"no labels CSV in {bundle.root}")
            labels = found[0]
    losses = list(cfg.model.get("losses_to_use") or [])
    with open_dict(cfg):
        cfg.model.losses_to_use = []
    try:
        result = model.predict_on_label_csv(
            labels,
            data_dir=images,
            compute_metrics=True,
            add_train_val_test_set=not isinstance(labels_spec, str),
        )
    finally:
        with open_dict(cfg):
            cfg.model.losses_to_use = losses
    lp_out = model.image_preds_dir() / labels.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv in lp_out.glob("predictions*.csv"):
        shutil.copy(csv, out_dir / f"{prefix}{csv.name}")
    # LP still splits other labels into train/val; that split means nothing here.
    summary = pixel_error_summary(lp_out, by_set=not isinstance(labels_spec, str))
    if summary is not None:
        summary.to_csv(out_dir / f"{prefix}pixel_error_summary.csv", index=False)
    return {
        "labels": str(labels.relative_to(data_dir)),
        "n_frames": len(result.predictions),
        "pixel_error_px_original_frame": (
            summary.to_dict(orient="records") if summary is not None else None
        ),
    }


def run_predict(
    config: PredictConfig, data_dir: Path, results_dir: Path, work: Path, record: Any
) -> None:
    """Predict every video in the config, writing one CSV per video."""
    bundle = find_bundle(data_dir / config.model, config.checkpoint)
    record.update(
        model={
            "asset": config.model,
            "config": str(bundle.config.relative_to(bundle.root)),
            "checkpoint": str(bundle.checkpoint.relative_to(bundle.root)),
            "checkpoint_choice": config.checkpoint,
            "checkpoint_sha256": file_sha256(bundle.checkpoint),
            "keypoint_names": list(bundle.keypoint_names),
            "keypoint_names_from": bundle.keypoint_source,
        },
        reader=config.reader,
        precision=config.precision,
    )
    model, batch_frames = load_model(
        bundle, config.batch_size, config.precision, work, videos=bool(config.videos)
    )
    record.update(batch_size=batch_frames)
    out_dir = results_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    for spec in config.videos:
        mp4 = video_path(data_dir / spec.asset, spec)
        debug = results_dir / "diagnostics.json" if config.debug and not videos else None
        result = predict_video(model, mp4, spec, config.reader, work, debug)
        csv = out_dir / f"{spec.label}.csv"
        result.predictions.to_csv(csv)
        videos.append(
            {
                "label": spec.label,
                "csv": str(csv.relative_to(results_dir)),
                "n_rows": len(result.predictions),
                **result.info,
            }
        )
        record.update(videos=videos)
    out = results_dir / "labeled_frames"
    if isinstance(config.labeled_frames, tuple):
        # Several label sets: each scored on its own, files named <csv stem>.<file>.
        checks = []
        for labels in config.labeled_frames:
            prefix = f"{Path(labels).stem}."
            checks.append(check_labeled_frames(model, bundle, labels, data_dir, out, prefix))
            record.update(labeled_frames=checks)
    elif config.labeled_frames:
        record.update(
            labeled_frames=check_labeled_frames(model, bundle, config.labeled_frames, data_dir, out)
        )
