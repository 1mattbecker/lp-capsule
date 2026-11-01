"""Label-free metrics of each model's predictions on the fixed video windows.

Reads ``<runs>/<model>/<session>/predictions/<session>_BottomCamera.csv`` (the
launcher's predict output for video_windows.yaml, one folder per model) and
writes ``<runs>/compare/video_metrics.csv`` (long) and ``video_summary.csv``.

Per model, domain (FIP / ephys, from video_windows.csv) and keypoint:

- ``glitch_per_1k``: steps between consecutive frames, both confident
  (likelihood > ``CONF``), that are larger than the keypoint's frozen
  threshold (``THRESH_PX``) and do not look like motion: a spike (the next
  step comes back) or a teleport (no motion either side). Per 1000 confident
  steps. ``glitch2x_per_1k``: the same at twice the threshold.
- ``fast_motion_per_1k``: steps over the threshold that look like real motion
  (motion continues into the neighbouring steps). Reported, not an error.
- ``low_conf``: share of frames with likelihood <= ``CONF``.
- ``pca_px_median`` / ``pca_px_p95``: for the keypoints LP's pca_singleview
  loss uses, the distance from each confident frame's pose to its
  reconstruction by a PCA (99% of variance, as LP) fit to the training labels.

    uv run python experiments/bottomcam_exp2/video_metrics.py runs/exp2/video \\
        --labels <bottomcam_exp1>/train/train_all.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONF = 0.5
# Frozen thresholds (px, original frame): about each group's 99th-percentile
# confident step in Pylon2Test2's FIP predictions, rounded. Do not refit per model.
THRESH_PX = {
    "nose_tip": 3, "jaw": 3, "spout_r": 3, "spout_l": 3,
    "tongue_tip_right": 12, "tongue_tip_center": 12, "tongue_tip_left": 12,
    "pointer_finger_r": 12, "paw_wrist_r": 12, "pointer_finger_l": 12, "paw_wrist_l": 12,
}  # fmt: skip
PCA_KEYPOINTS = ["nose_tip", "jaw", "spout_r", "spout_l"]  # columns_for_singleview_pca 0, 1, 9, 10
PCA_VARIANCE = 0.99


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    return df


def classify(p: np.ndarray, ok: np.ndarray) -> pd.DataFrame:
    """Step sizes and kinds at each t with steps t-1->t, t->t+1, t+1->t+2 all valid.

    A step t-1 -> t of size d is a spike when the next step is over d/2 and
    lands within d/2 of t-1; continuous motion when it is not a spike, a
    neighbouring step is at least d/4, and t+1 lies at least d/2 from t-1;
    otherwise a teleport.
    """
    d = np.full(len(p), np.nan)
    d[1:] = np.linalg.norm(p[1:] - p[:-1], axis=1)
    d[1:][~ok[1:]] = np.nan
    t = np.arange(2, len(p) - 1)
    dt, dprev, dnext = d[t], d[t - 1], d[t + 1]
    back = np.linalg.norm(p[t + 1] - p[t - 1], axis=1)
    valid = ~np.isnan(dt) & ~np.isnan(dprev) & ~np.isnan(dnext)
    spike = (dnext > 0.5 * dt) & (back < 0.5 * dt)
    motion = ~spike & ((dprev >= 0.25 * dt) | (dnext >= 0.25 * dt)) & (back >= 0.5 * dt)
    kind = np.where(spike, "spike", np.where(motion, "motion", "teleport"))
    return pd.DataFrame({"t": t, "d": dt, "kind": kind})[valid]


def fit_pca(labels_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean and kept components of LP-style singleview PCA on the training labels."""
    df = read_predictions(labels_csv)
    x = df[[(k, c) for k in PCA_KEYPOINTS for c in ("x", "y")]].dropna().to_numpy(float)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = np.cumsum(s**2) / np.sum(s**2)
    n = int(np.searchsorted(var, PCA_VARIANCE) + 1)
    return mean, vt[:n]


def session_metrics(df: pd.DataFrame, pca: tuple[np.ndarray, np.ndarray]) -> list[dict]:
    """Counts per keypoint for one session's predictions (summed over sessions later)."""
    frames = df.index.to_numpy()
    consecutive = np.zeros(len(df), bool)
    consecutive[1:] = np.diff(frames) == 1  # windows are separate runs of frames
    rows = []
    for kp, thresh in THRESH_PX.items():
        p = df[[(kp, "x"), (kp, "y")]].to_numpy(float)
        conf = df[(kp, "likelihood")].to_numpy() > CONF
        ok = consecutive.copy()
        ok[1:] &= conf[1:] & conf[:-1]
        steps = classify(p, ok)
        big = steps[steps.d > thresh]
        glitch = big[big.kind != "motion"]
        rows.append({
            "keypoint": kp, "frames": len(df), "low_conf_frames": int((~conf).sum()),
            "confident_steps": len(steps), "glitches": len(glitch),
            "glitches_2x": int((glitch.d > 2 * thresh).sum()),
            "fast_motion": int((big.kind == "motion").sum()),
        })  # fmt: skip
    mean, comps = pca
    cols = [(k, c) for k in PCA_KEYPOINTS for c in ("x", "y")]
    x = df[cols].to_numpy(float)
    all_conf = (df[[(k, "likelihood") for k in PCA_KEYPOINTS]].to_numpy() > CONF).all(1)
    x = x[all_conf] - mean
    recon = x @ comps.T @ comps
    err = np.linalg.norm((x - recon).reshape(len(x), -1, 2), axis=2)  # (frames, keypoints)
    for i, kp in enumerate(PCA_KEYPOINTS):
        next(r for r in rows if r["keypoint"] == kp)["pca_err"] = err[:, i]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="folder with one predict output folder per model")
    ap.add_argument("--labels", type=Path, required=True, help="training labels for the PCA")
    ap.add_argument("--windows", type=Path, default=HERE / "video_windows.csv")
    args = ap.parse_args()
    domain = pd.read_csv(args.windows).set_index("session").domain
    pca = fit_pca(args.labels)

    rows = []
    for model_dir in sorted(p for p in args.runs.iterdir() if p.is_dir() and p.name != "compare"):
        for csv in sorted(model_dir.glob("*/predictions/*_BottomCamera.csv")):
            session = csv.name.removesuffix("_BottomCamera.csv")
            if session not in domain:
                continue
            for r in session_metrics(read_predictions(csv), pca):
                rows.append({"model": model_dir.name, "domain": domain[session], **r})
    if not rows:
        raise SystemExit(f"no predictions under {args.runs}")
    counts = pd.DataFrame(rows)

    def agg(g: pd.DataFrame) -> pd.Series:
        steps = g.confident_steps.sum()
        errs = [e for e in g.get("pca_err", []) if isinstance(e, np.ndarray)]
        err = np.concatenate(errs) if errs else np.array([])
        return pd.Series({
            "sessions": len(g),
            "glitch_per_1k": 1000 * g.glitches.sum() / max(steps, 1),
            "glitch2x_per_1k": 1000 * g.glitches_2x.sum() / max(steps, 1),
            "fast_motion_per_1k": 1000 * g.fast_motion.sum() / max(steps, 1),
            "low_conf": g.low_conf_frames.sum() / g.frames.sum(),
            "pca_px_median": np.median(err) if len(err) else np.nan,
            "pca_px_p95": np.percentile(err, 95) if len(err) else np.nan,
        })  # fmt: skip

    out = args.runs / "compare"
    out.mkdir(exist_ok=True)
    metrics = counts.groupby(["model", "domain", "keypoint"], sort=False).apply(agg).reset_index()
    metrics.to_csv(out / "video_metrics.csv", index=False)
    summary = metrics.pivot_table(
        index=["domain", "keypoint"], columns="model", values="glitch_per_1k", sort=False
    ).round(2)
    summary.to_csv(out / "video_summary.csv")
    print("glitches per 1000 confident steps\n", summary.to_string())


if __name__ == "__main__":
    main()
