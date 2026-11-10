"""Clips and trace figures of sampled big steps, to review video_metrics' step classifier.

For each keypoint group (face / tongue_paws) and step kind (motion, spike,
teleport), samples confident steps over the frozen threshold from one model's
predictions. Each sample gets:

- a slowed clip (``--fps``, default 25: about 20x slower than the 500 Hz video)
  of frames t-``--half``..t+``--half``, cropped around the keypoint's track,
  with the keypoint (red; hollow grey when likelihood <= 0.5), its last 8
  positions (yellow trail) and the other keypoints (small cyan dots); the
  border is red on t-1 and t, the step judged;
- a figure of x, y, step size (with the threshold) and likelihood over the
  same frames, t marked.

Writes ``<out>/index.html`` (open it in a browser) with clips and figures side
by side, and ``<out>/media/``. Run with lp-relabel's environment plus matplotlib:

    uv run --project ~/Documents/Code/lp-relabel --with matplotlib python \\
        experiments/bottomcam_exp2/step_review.py runs/fip_r01 \\
        --sessions batches/fip_r01.sessions.csv --out runs/exp2/step_review
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lp_relabel.frames import extract_with_context
from step_sheet import FACE, collect, video_url
from video_metrics import CONF, THRESH_PX, read_predictions

CROP, SCALE, TRAIL = 192, 2, 8
KINDS = ("teleport", "spike", "motion")


def crop_box(pred: pd.DataFrame, kp: str, t: int, size: tuple[int, int]) -> tuple[int, int]:
    """Top-left of a fixed crop centred on the keypoint's confident positions near t."""
    near = pred.loc[pred.index.intersection(range(t - 10, t + 11))]
    near = near[near[(kp, "likelihood")] > CONF]
    xy = near[[(kp, "x"), (kp, "y")]].median().to_numpy()
    left = int(np.clip(xy[0] - CROP / 2, 0, size[0] - CROP))
    top = int(np.clip(xy[1] - CROP / 2, 0, size[1] - CROP))
    return left, top


def draw_frame(im: Image.Image, pred: pd.DataFrame, kp: str, f: int, t: int, box) -> Image.Image:
    left, top = box
    tile = im.crop((left, top, left + CROP, top + CROP)).resize((CROP * SCALE,) * 2)
    d = ImageDraw.Draw(tile)

    def at(g: int, k: str) -> tuple[float, float]:
        return (pred.loc[g, (k, "x")] - left) * SCALE, (pred.loc[g, (k, "y")] - top) * SCALE

    if f in pred.index:
        for k in THRESH_PX:
            if k != kp:
                x, y = at(f, k)
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill="cyan")
        trail = [at(g, kp) for g in range(f - TRAIL, f + 1) if g in pred.index]
        if len(trail) > 1:
            d.line(trail, fill="yellow", width=2)
        x, y = at(f, kp)
        conf = pred.loc[f, (kp, "likelihood")]
        r = 6
        if conf > CONF:
            d.ellipse([x - r, y - r, x + r, y + r], outline="red", width=3)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], outline="grey", width=2)
        label = f"t{f - t:+d}" if f != t else "t"
        d.text((6, 6), f"{label}  p={conf:.2f}", fill="white")
    if f in (t - 1, t):
        d.rectangle([0, 0, tile.width - 1, tile.height - 1], outline="red", width=4)
    return tile


def write_clip(tiles: list[Image.Image], path: Path, fps: int, tmp: Path) -> None:
    for i, tile in enumerate(tiles):
        tile.save(tmp / f"f{i:04d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
        check=True,
    )  # fmt: skip
    for p in tmp.glob("f*.png"):
        p.unlink()


def write_figure(pred: pd.DataFrame, kp: str, t: int, half: int, title: str, path: Path) -> None:
    frames = [f for f in range(t - half, t + half + 1) if f in pred.index]
    w = pred.loc[frames]
    off = np.array(frames) - t
    x, y = w[(kp, "x")].to_numpy(), w[(kp, "y")].to_numpy()
    conf = w[(kp, "likelihood")].to_numpy()
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    ok = np.r_[False, (conf[1:] > CONF) & (conf[:-1] > CONF)]
    judged = np.where(ok, step, np.nan)  # steps the metric scores; others dotted
    fig, ax = plt.subplots(3, 1, figsize=(6, 5.2), sharex=True)
    lo = conf <= CONF  # hide parked low-confidence positions so the track is readable
    ax[0].plot(off, np.where(lo, np.nan, x - x[off == 0]), label="x")
    ax[0].plot(off, np.where(lo, np.nan, y - y[off == 0]), label="y")
    ax[0].set_ylabel("px from t\n(confident)")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(off, step, color="grey", ls=":", lw=1)
    ax[1].plot(off, judged, color="k")
    ax[1].set_ylim(0, 1.3 * max(np.nanmax(judged), THRESH_PX[kp]))
    ax[1].axhline(THRESH_PX[kp], color="tab:red", ls="--", lw=1, label="threshold")
    ax[1].set_ylabel("step (px)\ndotted: not scored")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[2].plot(off, conf, color="tab:green")
    ax[2].axhline(CONF, color="grey", ls="--", lw=1)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("likelihood")
    ax[2].set_xlabel("frame from t (500 Hz: 1 frame = 2 ms)")
    for a in ax:
        a.axvspan(-1, 0, color="tab:red", alpha=0.15)
    fig.suptitle(title.replace("  ", "\n", 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", type=Path, help="predict output: <session>/predictions/*.csv")
    ap.add_argument("--sessions", type=Path, required=True, help="fip_r01.sessions.csv")
    ap.add_argument("--n-sessions", type=int, default=8)
    ap.add_argument("--per-kind", type=int, default=6)
    ap.add_argument("--half", type=int, default=50, help="frames each side of t")
    ap.add_argument("--fps", type=int, default=25, help="clip playback rate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    table = pd.read_csv(args.sessions).sample(args.n_sessions, random_state=args.seed)
    camera = dict(zip(table.raw_session, table.bottom_source_camera))
    steps = collect(args.predictions, list(table.raw_session))
    steps["group"] = np.where(steps.keypoint.isin(FACE), "face", "tongue_paws")
    counts = steps.groupby(["group", "kind"]).size()
    picks = steps.sample(frac=1, random_state=args.seed).groupby(["group", "kind"])
    picks = picks.head(args.per_kind).reset_index(drop=True)
    picks["id"] = [f"{r.group}_{r.kind}_{i:02d}" for i, r in enumerate(picks.itertuples())]

    media = args.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for session, g in picks.groupby("session"):
            folder = tmp / session
            extract_with_context(video_url(session, camera[session]), g.frame, folder,
                                 context=args.half, quality=3)  # fmt: skip
            csv = args.predictions / session / "predictions" / f"{session}_BottomCamera.csv"
            pred = read_predictions(csv)
            for r in g.itertuples():
                window = range(r.frame - args.half, r.frame + args.half + 1)
                files = [folder / f"img{f:08d}.jpg" for f in window]
                with Image.open(files[0]) as im0:
                    box = crop_box(pred, r.keypoint, r.frame, im0.size)
                tiles = []
                for f, p in zip(window, files):
                    with Image.open(p) as im:
                        frame = im.convert("RGB")
                    tiles.append(draw_frame(frame, pred, r.keypoint, f, r.frame, box))
                write_clip(tiles, media / f"{r.id}.mp4", args.fps, tmp)
                title = f"{r.keypoint}  step {r.d:.1f} px  {r.kind}  {session} frame {r.frame}"
                write_figure(pred, r.keypoint, r.frame, args.half, title, media / f"{r.id}.png")
                print("wrote", r.id, flush=True)

    rows = []
    for group in ("tongue_paws", "face"):
        for kind in KINDS:
            sel = picks[(picks.group == group) & (picks.kind == kind)]
            n = counts.get((group, kind), 0)
            rows.append(f"<h2>{group}: {kind} ({n} steps over threshold in {args.n_sessions} "
                        f"sessions; {len(sel)} shown)</h2>")  # fmt: skip
            for r in sel.itertuples():
                thresh = THRESH_PX[r.keypoint]
                label = html.escape(
                    f"{r.keypoint}, step {r.d:.1f} px (threshold {thresh}), "
                    f"{r.session} frame {r.frame}"
                )
                rows.append(
                    f'<div class="row"><p>{r.id}: {label}</p>'
                    f'<video src="media/{r.id}.mp4" autoplay loop muted controls></video>'
                    f'<img src="media/{r.id}.png"></div>'
                )
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Step review</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:20px}}
.row{{display:flex;flex-wrap:wrap;align-items:flex-start;gap:12px;margin:0 0 24px}}
.row p{{width:100%;margin:0}} video{{width:384px}} img{{background:#fff}}</style></head><body>
<h1>Big steps by kind</h1>
<p>Clips: {2 * args.half + 1} frames around the step, played at {args.fps} fps (about
{round(500 / args.fps)}x slower). Red circle: the keypoint (grey when likelihood &le; {CONF});
yellow: its last {TRAIL} positions; cyan: other keypoints. Red border: frames t-1 and t, the
step judged. The classifier calls a step a <b>spike</b> when the next frame comes back,
<b>motion</b> when movement continues on either side, otherwise <b>teleport</b>. Spikes and
teleports count as glitches.</p>
{"".join(rows)}</body></html>"""
    (args.out / "index.html").write_text(page)
    print("wrote", args.out / "index.html")


if __name__ == "__main__":
    main()
