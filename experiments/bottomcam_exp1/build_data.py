"""Build the labels asset for the bottom-camera training experiment.

Writes ``<out>/bottomcam_exp1/`` (upload that folder as the data asset):

- ``train/``: ``train_all.csv``, ``train_ephys.csv``, ``train_fip.csv`` (image
  paths into the merged project: Pylon2Test2's clips, round-1 sessions, FIP
  sessions), ``labeled-data/`` for the FIP training sessions (with their ±2
  neighbours), ``videos/`` unlabeled 1 s FIP clips for the semi-supervised run.
- ``test/``: ``test_{ephys,fip}_{id,ood}.csv`` with copies of their images and
  ±2 neighbours under ``test/labeled-data/``, so a test CSV scores on its own.
- ``splits.csv``: every labeled frame, its domain, mouse, session and split.

Splits: OOD = every frame of the held-out mice; ephys ID = the round-1 holdout
frames of the other mice; FIP ID = ``FIP_ID_PER_SESSION`` random frames per
remaining FIP session. Pylon2Test2's own frames are always training.

    uv run python experiments/bottomcam_exp1/build_data.py --out ~/LPProjects/bottomcam_exp1/upload
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

R01 = Path.home() / "LPProjects/pylon2test2_r01/upload/pylon2test2_r01_labels"
FIP = Path.home() / "LPProjects/fip_r01/data"
FIP_MANIFEST = Path.home() / "LPProjects/fip_r01/manifest.csv"
EPHYS_OOD_MICE = {"752014", "791691"}
FIP_OOD_MICE = {"815334", "820688"}
FIP_ID_PER_SESSION = 2
SEED = 0
# Unlabeled FIP clips: as many as Pylon2Test2's ephys clips, as long.
N_CLIPS = 120
CLIP_FRAMES = 500
# Clips start inside this fraction of the video, this far from any labeled frame.
CLIP_SPAN = (0.1, 0.9)
CLIP_MIN_GAP = 2500
MOUSE = re.compile(r"behavior_(\d+)_")
FRAME = re.compile(r"^(?P<head>.*?)(?P<digits>\d+)\.(?P<ext>png|jpg)$")


def read_labels(path: Path, columns: pd.MultiIndex | None = None) -> pd.DataFrame:
    labels = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if columns is not None:
        if list(labels.columns.droplevel(0)) != list(columns.droplevel(0)):
            raise ValueError(f"{path} keypoints/coords differ from the round-1 CSV")
        labels.columns = columns
    return labels


def frame_table(index: pd.Index, domain: str, source: str) -> pd.DataFrame:
    folders = index.str.split("/").str[1]
    return pd.DataFrame(
        {
            "image": index,
            "domain": domain,
            "source": source,
            "session": folders,
            "mouse": folders.str.extract(MOUSE, expand=False),
        }
    )


def neighbours(image: str) -> list[str]:
    folder, name = image.rsplit("/", 1)
    m = FRAME.match(name)
    assert m, image
    k, width = int(m["digits"]), len(m["digits"])
    return [
        f"{folder}/{m['head']}{str(k + d).zfill(width)}.{m['ext']}"
        for d in (-2, -1, 0, 1, 2)
        if k + d >= 0
    ]


def copy_images(images: list[str], src_roots: list[Path], dest: Path) -> int:
    n = 0
    for image in images:
        for path in neighbours(image):
            src = next((r / path for r in src_roots if (r / path).is_file()), None)
            if src is None:
                if path == image:
                    raise FileNotFoundError(image)
                continue
            (dest / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / path)
            n += 1
    return n


def cut_fip_clips(sessions: list[str], labeled: pd.DataFrame, videos_dir: Path) -> pd.DataFrame:
    from aind_dynamic_foraging_behavior_video_analysis.video_clips import cut_clips
    from aind_video_utils import read_mp4_frame_index

    urls = pd.read_csv(FIP_MANIFEST).set_index("session")["video"]
    rng = np.random.default_rng(SEED)
    per = np.full(len(sessions), N_CLIPS // len(sessions))
    per[: N_CLIPS - per.sum()] += 1
    rows = []
    for session, n in zip(sessions, per):
        url = urls[session]
        index = read_mp4_frame_index(url)
        lo, hi = (int(f * index.n_samples) for f in CLIP_SPAN)
        taken = [
            int(FRAME.match(i.rsplit("/", 1)[1])["digits"])
            for i in labeled.loc[labeled.session == session, "image"]
        ]
        starts: list[int] = []
        while len(starts) < n:
            s = int(rng.integers(lo, hi - CLIP_FRAMES))
            if all(abs(s - t) > CLIP_MIN_GAP for t in taken + starts):
                starts.append(s)
        ranges = pd.DataFrame({"start_frame": sorted(starts), "n_frames": CLIP_FRAMES})
        out = cut_clips(url, ranges, videos_dir, f"{session}_BottomCamera")
        bad = out[~out.status.isin(["cut", "exists"])]
        if len(bad):
            raise RuntimeError(f"{session}: {bad.status.tolist()}")
        rows.append(out.assign(session=session))
        print(f"  {session}: {n} clips", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-clips", action="store_true", help="skip cutting FIP clips")
    args = parser.parse_args()
    root = args.out / "bottomcam_exp1"
    train_dir, test_dir = root / "train", root / "test"

    r01_train = read_labels(R01 / "CollectedData.train.csv")
    columns = r01_train.columns
    r01_hold = read_labels(R01 / "CollectedData.holdout.csv", columns)
    fip = read_labels(FIP / "CollectedData.csv", columns)

    # Pylon2Test2's folders are clips (`..._clip_<n>_<times>`); round 1's are sessions.
    is_p2t2 = r01_train.index.str.contains("_clip_")
    frames = pd.concat(
        [
            frame_table(r01_train.index[is_p2t2], "ephys", "pylon2test2"),
            frame_table(r01_train.index[~is_p2t2], "ephys", "r01_train"),
            frame_table(r01_hold.index, "ephys", "r01_holdout"),
            frame_table(fip.index, "fip", "fip_r01"),
        ],
        ignore_index=True,
    )
    frames["split"] = "train"
    ephys_ood = (frames.domain == "ephys") & frames.mouse.isin(EPHYS_OOD_MICE)
    frames.loc[ephys_ood, "split"] = "test_ephys_ood"
    frames.loc[~ephys_ood & (frames.source == "r01_holdout"), "split"] = "test_ephys_id"
    fip_ood = (frames.domain == "fip") & frames.mouse.isin(FIP_OOD_MICE)
    frames.loc[fip_ood, "split"] = "test_fip_ood"
    rest = frames[(frames.domain == "fip") & ~fip_ood]
    picked = rest.groupby("session", group_keys=False).sample(FIP_ID_PER_SESSION, random_state=SEED)
    frames.loc[picked.index, "split"] = "test_fip_id"

    labels = pd.concat([r01_train, r01_hold, fip])
    assert labels.index.is_unique
    root.mkdir(parents=True, exist_ok=True)
    frames.to_csv(root / "splits.csv", index=False)

    train = frames[frames.split == "train"]
    sets = {
        "train_ephys": train[train.domain == "ephys"],
        "train_fip": train[train.domain == "fip"],
        "train_all": train,
    }
    train_dir.mkdir(parents=True, exist_ok=True)
    for name, part in sets.items():
        labels.loc[part.image].to_csv(train_dir / f"{name}.csv")
    fip_train = sets["train_fip"]
    n = copy_images(list(fip_train.image), [FIP], train_dir)
    print(f"train: {', '.join(f'{k} {len(v)}' for k, v in sets.items())}; {n} FIP images")

    test_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test_ephys_id", "test_ephys_ood", "test_fip_id", "test_fip_ood"):
        part = frames[frames.split == split]
        labels.loc[part.image].to_csv(test_dir / f"{split}.csv")
        n = copy_images(list(part.image), [R01, FIP], test_dir)
        print(f"{split}: {len(part)} frames, {part.session.nunique()} sessions, {n} images")

    if not args.no_clips:
        sessions = sorted(fip_train.session.unique())
        fip_frames = frames[frames.domain == "fip"]
        clips = cut_fip_clips(sessions, fip_frames, train_dir / "videos")
        clips.to_csv(root / "fip_clips.csv", index=False)
        print(f"videos: {len(clips)} FIP clips of {CLIP_FRAMES} frames")


if __name__ == "__main__":
    main()
