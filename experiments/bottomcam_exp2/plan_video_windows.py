"""Pick the fixed video windows every exp2 model is scored on without labels.

FIP: per rig, one session that has labeled frames and one that has none,
always including the two OOD mice. Ephys: the OOD mice's sessions, some
sessions with labeled frames, and sessions with none. Two 10,000-frame (20 s)
windows per session, moved until they are at least ``--gap`` frames from every
labeled frame and every unlabeled training clip of exp1. Writes a predict
batch (``video_windows.yaml``) and its table (``video_windows.csv``).

    uv run python experiments/bottomcam_exp2/plan_video_windows.py \\
        --fip-sessions batches/fip_r01.sessions.csv \\
        --ephys-sessions <aind-sessions manifest>.csv \\
        --splits <bottomcam_exp1>/splits.csv --clips <bottomcam_exp1>/fip_clips.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OOD_FIP = {"815334", "820688"}
OOD_EPHYS = {"752014", "791691"}
WINDOW = 10_000
FRAME = re.compile(r"img0*(\d+)\.\w+$")
BUCKET = "https://aind-open-data.s3.amazonaws.com"


def n_frames(session: str) -> int:
    """Frames in a session's bottom-camera video (MP4 header over HTTPS), in
    either layout (as the capsule's session.locate_camera finds it)."""
    for rel in ("BottomCamera/video.mp4", "bottom_camera.mp4"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=nb_frames", "-of", "csv=p=0", f"{BUCKET}/{session}/behavior-videos/{rel}"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if out.returncode == 0:
            return int(out.stdout.strip())
    raise FileNotFoundError(f"{session}: no bottom-camera video")


def place(anchor: int, avoid: np.ndarray, gap: int, limit: int, taken: list) -> tuple[int, int]:
    """The window nearest ``anchor`` (in steps of ``gap``) clear of ``avoid`` and ``taken``."""
    for k in range(200):
        for sign in (1, -1):
            start = anchor - WINDOW // 2 + sign * k * gap
            end = start + WINDOW
            if start < 0 or end > limit:
                continue
            clear = not ((avoid > start - gap) & (avoid < end + gap)).any()
            if clear and all(end + gap <= s or start >= e + gap for s, e in taken):
                return start, end
    raise ValueError(f"no clear window near frame {anchor}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fip-sessions", type=Path, required=True)
    p.add_argument("--ephys-sessions", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--n-ephys-labeled", type=int, default=4, help="besides the OOD mice")
    p.add_argument("--n-ephys-unlabeled", type=int, default=6)
    p.add_argument("--gap", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    splits = pd.read_csv(args.splits)
    splits["frame"] = splits.image.str.extract(FRAME, expand=False).astype(float)
    clips = pd.read_csv(args.clips)
    avoid = {s: g.frame.dropna().to_numpy() for s, g in splits.groupby("session")}
    for s, g in clips.groupby("session"):
        span = np.concatenate([np.arange(a, a + n) for a, n in zip(g.start_frame, g.n_frames)])
        avoid[s] = np.concatenate([avoid.get(s, np.array([])), span])
    labeled = set(splits.session)

    fip = pd.read_csv(args.fip_sessions)
    fip["mouse"] = fip.subject.astype(str)
    fip["labeled"] = fip.raw_session.isin(labeled)
    # Per rig, one session with labels and one without, from mice not yet
    # picked where possible; then each OOD mouse's labeled session on top.
    picks: list[dict] = []
    order = fip[~fip.mouse.isin(OOD_FIP)].sample(frac=1, random_state=args.seed)
    for rig, g in order.groupby("rig", sort=True):
        for has_labels in (True, False):
            pool = g[g.labeled == has_labels]
            if not len(pool):
                continue
            fresh = pool[~pool.mouse.isin({p["mouse"] for p in picks})]
            row = (fresh if len(fresh) else pool).iloc[0]
            picks.append({"domain": "fip", "session": row.raw_session, "mouse": row.mouse,
                          "rig": rig, "labeled": has_labels,
                          "anchor": (row.start_frame + row.end_frame) // 2})  # fmt: skip
    for mouse in sorted(OOD_FIP):
        row = fip[(fip.mouse == mouse) & fip.labeled].sort_values("raw_session").iloc[-1]
        picks.append({"domain": "fip", "session": row.raw_session, "mouse": mouse,
                      "rig": row.rig, "labeled": True,
                      "anchor": (row.start_frame + row.end_frame) // 2})  # fmt: skip

    # Sessions with full Pylon2Test2 predictions, plus the labeled ephys sessions
    # (one OOD mouse has none of the former).
    ephys = pd.read_csv(args.ephys_sessions, dtype={"mouse": str})[["session", "mouse"]]
    named = splits[(splits.domain == "ephys") & splits.mouse.notna()]
    named = named.assign(mouse=named.mouse.astype(int).astype(str))[["session", "mouse"]]
    ephys = pd.concat([ephys, named]).drop_duplicates("session")
    ephys["labeled"] = ephys.session.isin(labeled)
    ood = ephys[ephys.mouse.isin(OOD_EPHYS) & ephys.labeled]
    rest = ephys[~ephys.mouse.isin(OOD_EPHYS) & ephys.labeled]
    none = ephys[~ephys.labeled]
    chosen = [ood]
    one_per_mouse = rest.drop_duplicates("mouse")
    chosen.append(one_per_mouse.sample(args.n_ephys_labeled, random_state=args.seed))
    # Unlabeled sessions spread over mice: one per mouse first.
    by_mouse = none.sample(frac=1, random_state=args.seed).drop_duplicates("mouse")
    chosen.append(by_mouse.head(args.n_ephys_unlabeled))
    for row in pd.concat(chosen).itertuples():
        picks.append({"domain": "ephys", "session": row.session, "mouse": row.mouse, "rig": "",
                      "labeled": bool(row.labeled), "anchor": None})  # fmt: skip

    rows = []
    for pick in picks:
        total = n_frames(pick["session"])
        if pick["anchor"] is None:  # ephys: 40% and 60% through the session
            anchors = [int(0.4 * total), int(0.6 * total)]
        else:  # FIP: 40 s either side of the labeling window's centre (mid-task)
            anchors = [pick["anchor"] - 20_000, pick["anchor"] + 20_000]
        taken: list[tuple[int, int]] = []
        for a in anchors:
            taken.append(place(a, avoid.get(pick["session"], np.array([])), args.gap, total, taken))
        windows = [(int(a), int(b)) for a, b in sorted(taken)]
        rows.append(pick | {"n_frames": total, "windows": windows,
                            "ood": pick["mouse"] in OOD_FIP | OOD_EPHYS})  # fmt: skip
    table = pd.DataFrame(rows).drop(columns="anchor")
    table.to_csv(HERE / "video_windows.csv", index=False)

    batch = {
        "task": "predict",
        "model": "ephys_BottomCamera_Pylon2Test2",
        "videos": [
            {
                "asset": r["session"],
                "camera": "BottomCamera",
                "windows": [list(w) for w in r["windows"]],
            }
            for r in rows
        ],
    }
    header = (
        "# Fixed video windows for exp2's label-free metrics (plan_video_windows.py;\n"
        "# table in video_windows.csv). `model` is replaced per run.\n"
    )
    text = yaml.safe_dump(batch, sort_keys=False, default_flow_style=None)
    (HERE / "video_windows.yaml").write_text(header + text)
    summary = table.groupby(["domain", "labeled"]).size()
    print(f"{len(table)} sessions, {len(table.mouse.unique())} mice\n{summary}")


if __name__ == "__main__":
    main()
