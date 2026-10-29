"""Compare the experiment's models on the four test sets (after eval.yaml ran).

Reads ``<eval>/<run>/labeled_frames/<test set>.predictions_pixel_error.csv``
(LP's per-frame, per-keypoint pixel error, original-frame pixels; NaN where
the keypoint is not labeled) and writes to ``<eval>/compare/``:

- ``summary.csv``: per model, checkpoint and test set: frames, median and mean
  error over every labeled keypoint, the share over ``FAIL_PX``;
- ``per_keypoint.csv``: the median per keypoint as well;
- ``contrasts.csv``: the planned comparisons, paired by frame. A frame's error
  is its mean over its labeled keypoints; the difference is B - A (negative =
  B better), with a 95% interval from resampling sessions (frames of one
  session move together).

    uv run python experiments/bottomcam_exp1/compare.py runs/exp1/eval
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SETS = ("test_ephys_id", "test_ephys_ood", "test_fip_id", "test_fip_ood")
EPHYS, FIP = SETS[:2], SETS[2:]
FAIL_PX = 10.0
N_BOOT = 2000
SEED = 0
RUN = re.compile(r"^eval_(?P<model>.+?)(?:_(?P<ckpt>best|last))?$")
# (question, A, B, test sets); A and B are "<model>:<checkpoint>".
CONTRASTS = [
    ("adding FIP, ephys frames", "exp1_ephys", "exp1_all", EPHYS),
    ("adding ephys, FIP frames", "exp1_fip", "exp1_all", FIP),
    ("context model", "exp1_all", "exp1_all_ctx", SETS),
    ("semi-supervised losses", "exp1_all", "exp1_all_semi", SETS),
    ("fine-tune vs scratch", "exp1_all", "exp1_all_ft", SETS),
    ("adding FIP to a fine-tune", "exp1_ephys_ft", "exp1_all_ft", SETS),
    ("seed (noise floor)", "exp1_all", "exp1_all_seed1", SETS),
    ("all vs Pylon2Test2", "pylon2test2", "exp1_all", SETS),
]


def load(eval_dir: Path) -> pd.DataFrame:
    """Return long errors: model, ckpt, set, image, session, keypoint, px."""
    parts = []
    for run in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        m = RUN.match(run.name)
        if m is None:
            continue
        for test in SETS:
            path = run / "labeled_frames" / f"{test}.predictions_pixel_error.csv"
            if not path.is_file():
                continue
            errors = pd.read_csv(path, index_col=0).drop(columns="set", errors="ignore")
            long = errors.rename_axis("image").reset_index().melt("image", var_name="keypoint")
            parts.append(
                long.rename(columns={"value": "px"}).assign(
                    model=m["model"], ckpt=m["ckpt"] or "only", set=test
                )
            )
    if not parts:
        raise FileNotFoundError(f"no *.predictions_pixel_error.csv under {eval_dir}/*/")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["px"])
    df["session"] = df.image.str.split("/").str[1]
    return df


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "ckpt", "set"]
    summary = df.groupby(keys).agg(
        frames=("image", "nunique"),
        median_px=("px", "median"),
        mean_px=("px", "mean"),
        fail_frac=("px", lambda x: (x > FAIL_PX).mean()),
    )
    per_kp = df.pivot_table(index=keys, columns="keypoint", values="px", aggfunc="median")
    return summary.reset_index(), per_kp.reset_index()


def frame_errors(df: pd.DataFrame, model: str, ckpt: str, test: str) -> pd.Series:
    part = df[(df.model == model) & (df.ckpt == ckpt) & (df.set == test)]
    return part.groupby(["session", "image"]).px.mean()


def boot_ci(diff: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    by_session = [g.to_numpy() for _, g in diff.groupby(level="session")]
    means = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(by_session), len(by_session))
        means.append(np.concatenate([by_session[i] for i in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ckpts = {m: sorted(df.loc[df.model == m, "ckpt"].unique()) for m in df.model.unique()}
    rows = []
    for question, a, b, sets in CONTRASTS:
        if a not in ckpts or b not in ckpts:
            continue
        for ckpt in ("last", "best"):
            ca = ckpt if ckpt in ckpts[a] else ckpts[a][0]
            cb = ckpt if ckpt in ckpts[b] else ckpts[b][0]
            for test in sets:
                ea, eb = frame_errors(df, a, ca, test), frame_errors(df, b, cb, test)
                diff = (eb - ea).dropna()
                if diff.empty:
                    continue
                lo, hi = boot_ci(diff, rng)
                rows.append(
                    {
                        "question": question,
                        "A": f"{a}:{ca}",
                        "B": f"{b}:{cb}",
                        "set": test,
                        "frames": len(diff),
                        "A_mean_px": ea.loc[diff.index].mean(),
                        "B_mean_px": eb.loc[diff.index].mean(),
                        "diff_mean_px": diff.mean(),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "B_better_frac": (diff < 0).mean(),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("eval_dir", type=Path)
    args = parser.parse_args()
    df = load(args.eval_dir)
    out = args.eval_dir / "compare"
    out.mkdir(exist_ok=True)
    summary, per_kp = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    per_kp.to_csv(out / "per_keypoint.csv", index=False)
    table = contrasts(df)
    table.to_csv(out / "contrasts.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.round(2).to_string(index=False))
        print()
        print(table.round(2).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
