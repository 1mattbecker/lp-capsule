"""Contact sheets of sampled big steps per class, to check video_metrics' step classifier.

For each keypoint group (face: nose, jaw, spouts; tongue_paws: the rest) and
step kind (motion, spike, teleport), samples steps over the frozen threshold
from one model's predictions and draws frames t-2..t+2 cropped around the
keypoint, with its predicted position in each frame (red; t-1's in yellow).
Writes ``<out>/<group>_<kind>.png``.

Run with lp-relabel's environment (aind-video-utils, Pillow):

    uv run --project ~/Documents/Code/lp-relabel python \\
        experiments/bottomcam_exp2/step_sheet.py runs/fip_r01 \\
        --sessions batches/fip_r01.sessions.csv --out runs/exp2/step_sheets
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_metrics import CONF, THRESH_PX, classify, read_predictions  # noqa: E402

from lp_relabel.frames import extract_with_context  # noqa: E402

BUCKET = "https://aind-open-data.s3.amazonaws.com"
FACE = {"nose_tip", "jaw", "spout_r", "spout_l"}
CROP, SCALE = 128, 2


def video_url(session: str, camera: str) -> str:
    if camera == "BottomCamera":
        return f"{BUCKET}/{session}/behavior-videos/BottomCamera/video.mp4"
    return f"{BUCKET}/{session}/behavior-videos/{camera}.mp4"


def collect(pred_dir: Path, sessions: list[str]) -> pd.DataFrame:
    """Every confident step over the threshold, with its source frame and kind."""
    rows = []
    for session in sessions:
        csv = pred_dir / session / "predictions" / f"{session}_BottomCamera.csv"
        df = read_predictions(csv)
        frames = df.index.to_numpy()
        consecutive = np.r_[False, np.diff(frames) == 1]
        for kp, thresh in THRESH_PX.items():
            p = df[[(kp, "x"), (kp, "y")]].to_numpy(float)
            conf = df[(kp, "likelihood")].to_numpy() > CONF
            ok = consecutive & np.r_[False, conf[1:] & conf[:-1]]
            steps = classify(p, ok)
            steps = steps[steps.d > thresh]
            rows.append(steps.assign(session=session, keypoint=kp, frame=frames[steps.t]))
    return pd.concat(rows, ignore_index=True)


def row_image(pred: pd.DataFrame, kp: str, frame: int, jpgs: dict[int, Path], label: str):
    x, y = pred.loc[frame, (kp, "x")], pred.loc[frame, (kp, "y")]
    tiles = []
    for f in range(frame - 2, frame + 3):
        im = Image.open(jpgs[f]).convert("RGB")
        left, top = int(x - CROP / 2), int(y - CROP / 2)
        tile = im.crop((left, top, left + CROP, top + CROP)).resize((CROP * SCALE,) * 2)
        draw = ImageDraw.Draw(tile)
        for g, color in ((f - 1, "yellow"), (f, "red")):
            if g in pred.index:
                gx, gy = (pred.loc[g, (kp, c)] - o for c, o in (("x", left), ("y", top)))
                r = 4
                draw.ellipse([gx * SCALE - r, gy * SCALE - r, gx * SCALE + r, gy * SCALE + r],
                             outline=color, width=2)  # fmt: skip
        draw.text((4, 4), f"t{f - frame:+d}" if f != frame else "t", fill="cyan")
        tiles.append(tile)
    row = Image.new("RGB", (CROP * SCALE * 5, CROP * SCALE + 18), "black")
    for i, tile in enumerate(tiles):
        row.paste(tile, (i * CROP * SCALE, 18))
    ImageDraw.Draw(row).text((4, 2), label, fill="white")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", type=Path, help="predict output: <session>/predictions/*.csv")
    ap.add_argument("--sessions", type=Path, required=True, help="fip_r01.sessions.csv")
    ap.add_argument("--n-sessions", type=int, default=8)
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    table = pd.read_csv(args.sessions)
    table = table.sample(args.n_sessions, random_state=args.seed)
    camera = dict(zip(table.raw_session, table.bottom_source_camera))
    steps = collect(args.predictions, list(table.raw_session))
    steps["group"] = np.where(steps.keypoint.isin(FACE), "face", "tongue_paws")
    shuffled = steps.sample(frac=1, random_state=args.seed)
    picks = shuffled.groupby(["group", "kind"]).head(args.per_sheet)
    print(steps.groupby(["group", "kind"]).size().to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    preds = {}
    rows: dict[tuple[str, str], list] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for session, g in picks.groupby("session"):
            url = video_url(session, camera[session])
            jpgs = extract_with_context(url, g.frame, Path(tmp) / session, context=2, quality=3)
            folder = Path(tmp) / session
            pred = preds.setdefault(session, read_predictions(
                args.predictions / session / "predictions" / f"{session}_BottomCamera.csv"))  # fmt: skip
            for r in g.itertuples():
                files = {f: folder / f"img{f:08d}.jpg" for f in range(r.frame - 2, r.frame + 3)}
                assert jpgs and all(p.exists() for p in files.values())
                label = f"{r.keypoint}  step {r.d:.1f} px  {r.kind}  {session} frame {r.frame}"
                rows.setdefault((r.group, r.kind), []).append(
                    row_image(pred, r.keypoint, r.frame, files, label)
                )
    for (group, kind), images in rows.items():
        sheet = Image.new("RGB", (images[0].width, sum(i.height for i in images)), "black")
        y = 0
        for im in images:
            sheet.paste(im, (0, y))
            y += im.height
        sheet.save(args.out / f"{group}_{kind}.png")
        print("wrote", args.out / f"{group}_{kind}.png")


if __name__ == "__main__":
    main()
