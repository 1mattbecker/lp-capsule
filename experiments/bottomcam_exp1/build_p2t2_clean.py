"""Build ``p2t2_clean``: Pylon2Test2's labeled frames with their neighbours, and clean clips.

Pylon2Test2's clips were cut by stream copy: each MP4 indexes 591-743 samples
of which 502 display (an edit list hides the rest), and some have timestamp
gaps. aind-video-utils refuses to address frames in them (edit list not
frame-addressing-safe) and DALI misreads them. So, per clip, with
aind-video-utils:

- ``transcode_video(normalize_cfr=True)`` losslessly (temporary) gives a clip
  of exactly its displayed frames on a uniform timeline;
- every labeled PNG is checked against ``extract_frame_by_index`` of that
  clip (frame k of the clip is ``img<k>.png``), then the neighbours k-2..k+2
  the context model reads are written as PNGs beside it;
- ``transcode_video(normalize_cfr=True)`` at crf 18 gives ``videos/<clip>.mp4``,
  the unlabeled video of the semi-supervised losses.

Writes ``<out>/p2t2_clean/{labeled-data,videos}/`` and ``checks.csv``; upload
the ``p2t2_clean`` folder as the data asset.

    uv run python experiments/bottomcam_exp1/build_p2t2_clean.py --out ~/LPProjects
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from aind_video_utils import (
    EncodingProfile,
    extract_frame_by_index,
    read_mp4_frame_index,
    transcode_video,
)
from PIL import Image

SOURCE = (
    "https://aind-scratch-data.s3.amazonaws.com/matt.becker/lightning_pose_test/"
    "ephys_BottomCamera_Pylon2Test2"
)
LABELS = Path.home() / "LPProjects/pylon2test2_r01/upload/pylon2test2_r01_labels"
OFFSETS = (-2, -1, 1, 2)
# Mean |difference| (0-255) allowed between a labeled PNG and the same frame
# of the lossless transcode; measured <= 0.004, a wrong frame is >= 0.9.
MAX_DIFF = 0.05


def profile(*codec_params: str) -> EncodingProfile:
    """H.264 with no colour conversion, so pixels stay those of the source decode."""
    return EncodingProfile(
        video_filters="format=yuv420p",
        codec="libx264",
        pixel_format="yuv420p",
        container="mp4",
        codec_params=codec_params,
        output_flags=("-movflags", "+faststart"),
    )


LOSSLESS = profile("-preset", "fast", "-qp", "0")
CLEAN = profile("-preset", "slow", "-crf", "18")


def fetch(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)
    return path


def transcode(src: Path, dst: Path, codec: EncodingProfile) -> Path:
    with warnings.catch_warnings():
        # The notched source timeline is what normalize_cfr repairs.
        warnings.simplefilter("ignore")
        transcode_video(src, dst, profile=codec, normalize_cfr=True, auto_fix_colorspace=False)
    return dst


def one_clip(clip: str, images: list[str], out: Path) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        src = fetch(f"{SOURCE}/videos/{clip}.mp4", Path(tmp) / "src.mp4")
        lossless = transcode(src, Path(tmp) / "lossless.mp4", LOSSLESS)
        index = read_mp4_frame_index(lossless)
        if not index.is_frame_addressing_safe() or not index.timeline_notches().is_uniform:
            raise RuntimeError(f"{clip}: transcode is not a clean clip")
        folder = out / "labeled-data" / clip
        for image in images:
            name = image.rsplit("/", 1)[1]
            png = fetch(f"{SOURCE}/{image}", folder / name)
            k, width = int(name[3:-4]), len(name) - 7
            labeled = np.asarray(Image.open(png).convert("RGB"), dtype=np.int16)
            decoded = extract_frame_by_index(lossless, k, index=index)
            diff = float(np.abs(decoded.astype(np.int16) - labeled).mean())
            if diff > MAX_DIFF:
                raise RuntimeError(f"{image}: differs from frame {k} of its clip ({diff:.3f})")
            written = 0
            for d in OFFSETS:
                j = k + d
                path = folder / f"img{str(j).zfill(width)}.png"
                if 0 <= j < index.n_samples and not path.exists():
                    Image.fromarray(extract_frame_by_index(lossless, j, index=index)).save(path)
                    written += 1
            rows.append(
                {
                    "image": image,
                    "frame": k,
                    "clip_frames": index.n_samples,
                    "source_samples": read_mp4_frame_index(src).n_samples,
                    "labeled_vs_clip_mean_abs_diff": diff,
                    "neighbours_written": written,
                }
            )
        transcode(src, out / "videos" / f"{clip}.mp4", CLEAN)
    print(
        f"  {clip}: {len(images)} labeled, {sum(r['neighbours_written'] for r in rows)} neighbours",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="folder to create p2t2_clean in")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clips", type=int, help="only the first N clips (a test)")
    args = parser.parse_args()
    out = args.out / "p2t2_clean"
    # Pylon2Test2's labeled frames: the clip rows of the round-1 training CSV.
    index = pd.read_csv(LABELS / "CollectedData.train.csv", header=[0, 1, 2], index_col=0).index
    images = [i for i in index if "_clip_" in i]
    by_clip: dict[str, list[str]] = {}
    for image in images:
        by_clip.setdefault(image.split("/")[1], []).append(image)
    if args.clips:
        by_clip = dict(sorted(by_clip.items())[: args.clips])
    print(f"{sum(map(len, by_clip.values()))} labeled frames in {len(by_clip)} clips -> {out}")
    (out / "videos").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda c: one_clip(c, by_clip[c], out), sorted(by_clip)))
    checks = pd.DataFrame([r for rows in results for r in rows])
    checks.to_csv(out / "checks.csv", index=False)
    print(
        f"done: {len(checks)} frames, max labeled-vs-clip diff "
        f"{checks.labeled_vs_clip_mean_abs_diff.max():.4f}, "
        f"{checks.neighbours_written.sum()} neighbours, {len(by_clip)} clean clips"
    )


if __name__ == "__main__":
    main()
