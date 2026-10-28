"""Finding a model's config, checkpoint and keypoint names in any asset layout.

A model bundle is any folder holding:

- **one LP config**: a ``*.yaml`` with ``model:`` and ``data:`` mappings
  (``hparams.yaml`` and this capsule's ``run_config.yaml`` are skipped).
  Identical copies are fine; different ones are an error.
- **one checkpoint**: ``*.ckpt`` anywhere. Identical copies are fine
  (Pylon2 bundles hold the same file at the top and under ``outputs/tb_logs``);
  otherwise the one ``*-best.ckpt``.
- **keypoint names**: ``data.keypoint_names`` in the config, else the
  ``bodyparts`` row of a ``CollectedData*.csv`` beside it (our LP 1.6 models
  leave the names null), else ``data.keypoints``.

Folders of images and videos are not searched.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SKIP_CONFIGS = {"hparams.yaml", "run_config.yaml"}
# Folders that hold data, not model files; pruned from every search.
SKIP_DIRS = {"labeled-data", "videos", "image_preds", "video_preds", "labeled_videos"}
LABELS_GLOB = "CollectedData*.csv"


@dataclass(frozen=True)
class Bundle:
    """A model's files and keypoint names."""

    root: Path
    config: Path
    checkpoint: Path
    keypoint_names: tuple[str, ...]
    # Where the names came from: "config", a CSV path, or "data.keypoints".
    keypoint_source: str


def walk_files(root: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    """Yield files under ``root`` with these suffixes, skipping data folders."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(suffixes) and not name.startswith("."):
                yield Path(folder) / name


def read_lp_config(path: Path) -> dict[str, Any] | None:
    """Return a YAML file's contents if it is an LP config, else None."""
    try:
        raw = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, UnicodeDecodeError):
        return None
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("model"), dict)
        and isinstance(raw.get("data"), dict)
    ):
        return raw
    return None


def find_config(root: Path) -> Path:
    """Return the bundle's LP config, preferring ``training.config.yaml`` among copies."""
    found: dict[bytes, list[Path]] = {}
    for path in walk_files(root, (".yaml", ".yml")):
        if path.name in SKIP_CONFIGS or read_lp_config(path) is None:
            continue
        found.setdefault(path.read_bytes(), []).append(path)
    if not found:
        raise FileNotFoundError(f"no LP config (YAML with model: and data:) under {root}")
    if len(found) > 1:
        listed = ", ".join(str(p[0].relative_to(root)) for p in found.values())
        raise ValueError(f"different LP configs under {root}: {listed}")
    (copies,) = found.values()
    preferred = [p for p in copies if p.name == "training.config.yaml"]
    return (preferred or copies)[0]


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(root: Path) -> Path:
    """Return the bundle's checkpoint (see the module docstring)."""
    checkpoints = list(walk_files(root, (".ckpt",)))
    if not checkpoints:
        raise FileNotFoundError(f"no .ckpt under {root}")
    if len(checkpoints) == 1:
        return checkpoints[0]
    # Sizes first: hashing reads every file, which is slow on a mounted bucket.
    same_size = len({p.stat().st_size for p in checkpoints}) == 1
    if same_size and len({file_sha256(p) for p in checkpoints}) == 1:
        return min(checkpoints, key=lambda p: len(p.relative_to(root).parts))
    best = [p for p in checkpoints if p.name.endswith("-best.ckpt")]
    if len(best) == 1:
        return best[0]
    listed = ", ".join(str(p.relative_to(root)) for p in checkpoints)
    raise ValueError(f"several different checkpoints under {root}: {listed}")


def csv_keypoint_names(path: Path) -> list[str]:
    """Return the keypoints of an LP/DLC labels CSV, from its header rows only."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in (next(reader, None) for _ in range(3)):
            if row and row[0] == "bodyparts":
                return list(dict.fromkeys(row[1:]))
    raise ValueError(f"{path} has no bodyparts header row")


def keypoint_names(config: dict[str, Any], root: Path) -> tuple[list[str], str]:
    """Return the model's keypoint names and where they came from."""
    data = config["data"]
    names, source = data.get("keypoint_names"), "config"
    if not names:
        csvs = sorted(root.glob(LABELS_GLOB)) or sorted(root.glob(f"*/{LABELS_GLOB}"))
        found = {tuple(csv_keypoint_names(p)): p for p in csvs}
        if len(found) > 1:
            raise ValueError(f"label CSVs under {root} disagree on keypoints: {csvs}")
        if found:
            ((names, path),) = found.items()
            source = str(path.relative_to(root))
    if not names and data.get("keypoints"):
        names, source = data["keypoints"], "data.keypoints"
    if not names:
        raise ValueError(
            f"no keypoint names under {root}: the config has none and there is no "
            f"{LABELS_GLOB} beside it"
        )
    names = [str(n) for n in names]
    expected = data.get("num_keypoints")
    if expected is not None and len(names) != expected:
        raise ValueError(
            f"{len(names)} keypoint names from {source}, but data.num_keypoints is {expected}"
        )
    return names, source


def find_bundle(root: Path) -> Bundle:
    """Find a model bundle's files and keypoint names."""
    config = find_config(root)
    raw = read_lp_config(config)
    assert raw is not None
    names, source = keypoint_names(raw, root)
    return Bundle(
        root=root,
        config=config,
        checkpoint=find_checkpoint(root),
        keypoint_names=tuple(names),
        keypoint_source=source,
    )
