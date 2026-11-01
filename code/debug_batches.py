"""Look at the batches LP trains on, instead of training (train ``debug_batches: N``).

Built for one question: do the unlabeled video frames DALI feeds the
semi-supervised losses match the labeled frames? Writes to
``<results>/debug_batches/``:

- ``<source>_<aug|noaug>.png``: frames of one batch, de-normalized, for labeled
  frames and for each group of unlabeled clips (``fip``: clips named
  ``*_f<frame>.mp4``; ``ephys``: the rest), with and without LP's augmentation;
- ``stats.json``:
  - per batch: shape and per-channel mean/std/min/max of the normalized tensor;
  - with an ``init_checkpoint``, that model's labeled pixel error and its
    confidence on each source (eval mode);
  - the labeled error again after BatchNorm statistics were updated by
    ``N`` augmented batches of each source in train mode, without gradients,
    which is what each training step's forward passes do to them.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

FIP_CLIP = re.compile(r"_f\d+\.mp4$")
N_LABELED_EVAL = 4  # labeled batches the error is measured on


def clip_groups(video_dir: Path, work: Path) -> dict[str, Path]:
    """Folders of symlinks, one per clip group that has clips: ``fip`` and ``ephys``."""
    groups: dict[str, Path] = {}
    for clip in sorted(video_dir.glob("*.mp4")):
        name = "fip" if FIP_CLIP.search(clip.name) else "ephys"
        folder = groups.setdefault(name, work / "clip_groups" / name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / clip.name).symlink_to(clip.resolve())
    return groups


def tensor_stats(x: Any) -> dict[str, Any]:
    """Shape and per-channel mean/std/min/max of a (N, C, H, W) tensor."""
    x = x.detach().float().cpu()
    per = x.transpose(0, 1).reshape(x.shape[1], -1)
    return {
        "shape": list(x.shape),
        "mean": per.mean(1).round(decimals=3).tolist(),
        "std": per.std(1).round(decimals=3).tolist(),
        "min": per.min(1).values.round(decimals=3).tolist(),
        "max": per.max(1).values.round(decimals=3).tolist(),
    }


def save_grid(x: Any, path: Path, n: int = 8) -> None:
    """Save the first ``n`` frames of a normalized (N, C, H, W) batch as one PNG row."""
    from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
    from PIL import Image

    x = x[:n].detach().float().cpu().numpy()
    mean = np.asarray(_IMAGENET_MEAN)[None, :, None, None]
    std = np.asarray(_IMAGENET_STD)[None, :, None, None]
    img = np.clip((x * std + mean) * 255, 0, 255).astype(np.uint8)
    row = np.concatenate(list(img.transpose(0, 2, 3, 1)), axis=1)
    Image.fromarray(row).save(path)


def data_module(cfg: Any, imgaug: str, video_dir: Path | None) -> Any:
    """LP's data module for ``cfg`` with one augmentation setting and clip folder."""
    from lightning_pose.data import get_data_module, get_dataset, get_imgaug_transform
    from omegaconf import open_dict

    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        cfg.training.imgaug = imgaug
        if video_dir is None:
            cfg.model.losses_to_use = []
        else:
            cfg.data.video_dir = str(video_dir)
    data_dir = Path(cfg.data.data_dir)
    transform = get_imgaug_transform(cfg=cfg, data_dir=data_dir)
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=transform)
    return get_data_module(cfg=cfg, dataset=dataset, video_dir=cfg.data.video_dir)


def labeled_batches(module: Any, cfg: Any, n: int) -> list[dict[str, Any]]:
    import torch

    loader = torch.utils.data.DataLoader(
        module.train_dataset,
        batch_size=int(cfg.training.train_batch_size),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    out = []
    for batch in loader:
        out.append(batch)
        if len(out) == n:
            break
    return out


def unlabeled_batches(module: Any, n: int) -> list[dict[str, Any]]:
    out = []
    for batch in module.unlabeled_dataloader:
        # To CPU: DALI may reuse its buffers, and N batches of frames would fill the GPU.
        out.append({k: v.cpu() if hasattr(v, "cpu") else v for k, v in batch.items()})
        if len(out) == n:
            break
    return out


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def labeled_error(model: Any, batches: list[dict[str, Any]], device: str) -> dict[str, float]:
    """Median pixel error (original frame) and median confidence on labeled batches."""
    import torch

    errors, confs = [], []
    with torch.no_grad():
        for batch in batches:
            out = model.get_loss_inputs_labeled(to_device(batch, device))
            pred = out["keypoints_pred"].reshape(len(batch["images"]), -1, 2)
            targ = out["keypoints_targ"].reshape(pred.shape)
            errors.append(torch.linalg.norm(pred - targ, dim=-1).flatten().cpu())
            confs.append(out["confidences"].flatten().cpu())
    err = torch.cat(errors)
    return {
        "median_px": round(float(err[~torch.isnan(err)].median()), 2),
        "median_confidence": round(float(torch.cat(confs).median()), 3),
    }


def confidence(model: Any, frames: list[Any], device: str) -> dict[str, float]:
    """Median and share > 0.9 of the model's confidence on unlabeled frames."""
    import torch

    confs = []
    with torch.no_grad():
        for x in frames:
            _, conf = model.head.run_subpixelmaxima(model.forward(x.to(device)))
            confs.append(conf.flatten().cpu())
    c = torch.cat(confs)
    return {"median_confidence": round(float(c.median()), 3),
            "share_over_0.9": round(float((c > 0.9).float().mean()), 3)}  # fmt: skip


def update_batchnorm(model: Any, inputs: list[Any], device: str) -> Any:
    """A copy of ``model`` whose BatchNorm statistics ``inputs`` updated as training would."""
    import torch

    model = copy.deepcopy(model)
    model.train()
    with torch.no_grad():
        for x in inputs:
            model.forward(x.to(device))
    model.eval()
    return model


def run_debug_batches(cfg: Any, n: int, init: dict[str, Any] | None, data_dir: Path,
                      results_dir: Path, work: Path) -> dict[str, Any]:  # fmt: skip
    """Write the batches and checks of the module docstring; return the stats."""
    import torch
    from bundle import find_bundle
    from omegaconf import OmegaConf, open_dict
    from predict import load_module

    out_dir = results_dir / "debug_batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = clip_groups(Path(cfg.data.video_dir), work)
    stats: dict[str, Any] = {"clips": {g: len(list(p.glob("*.mp4"))) for g, p in groups.items()},
                             "batches": {}}  # fmt: skip

    # frames[source][aug] = list of normalized (N, C, H, W) tensors
    frames: dict[str, dict[str, list[Any]]] = {}
    labeled: dict[str, list[dict[str, Any]]] = {}
    for aug, imgaug in (("aug", cfg.training.imgaug), ("noaug", "none")):
        module = data_module(cfg, imgaug, None)
        labeled[aug] = labeled_batches(module, cfg, max(n, N_LABELED_EVAL))
        frames.setdefault("labeled", {})[aug] = [b["images"] for b in labeled[aug]]
        for group, folder in groups.items():
            module = data_module(cfg, imgaug, folder)
            frames.setdefault(group, {})[aug] = [b["frames"] for b in unlabeled_batches(module, n)]
    for source, by_aug in frames.items():
        for aug, batches in by_aug.items():
            save_grid(batches[0], out_dir / f"{source}_{aug}.png")
            stats["batches"][f"{source}_{aug}"] = tensor_stats(torch.cat(batches))

    if init is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = find_bundle(data_dir / init["asset"])
        init_cfg = OmegaConf.load(bundle.config)
        with open_dict(init_cfg):
            init_cfg.data.keypoint_names = list(bundle.keypoint_names)
        model = load_module(init_cfg, bundle.checkpoint).to(device)
        eval_batches = labeled["noaug"][:N_LABELED_EVAL]
        checks: dict[str, Any] = {"model": init["asset"]}
        checks["labeled"] = labeled_error(model, eval_batches, device)
        checks["confidence"] = {
            source: confidence(model, by_aug["noaug"], device) for source, by_aug in frames.items()
        }
        # Train-mode forward passes of each source, then the labeled error again.
        checks["labeled_after_batchnorm_update"] = {
            source: labeled_error(
                update_batchnorm(model, by_aug["aug"], device), eval_batches, device
            )
            for source, by_aug in frames.items()
        }
        checks["batchnorm_update_batches"] = n
        stats["model_checks"] = checks
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
