"""Finding a camera's video in a mounted session asset.

Two layouts, both relative to the session asset:

- the file standard's ``behavior-videos/<Camera>/video.mp4``;
- the legacy dynamic-foraging ``behavior/<behavior session>/behavior-videos/``
  ``<camera>.mp4`` (or loose ``behavior-videos/<camera>.mp4``), in snake case.

Adapted from ``session.locate_camera`` in
AllenNeuralDynamics/lightning-pose-inference-capsule (MIT), without frame times:
this capsule keys outputs by source frame index and takes no times.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

# Names of the entries directly under a folder relative to the session asset,
# empty where it does not exist.
ListDir = Callable[[str], list[str]]

# File-standard camera names to the legacy layout's.
LEGACY_CAMERAS = {
    "BodyCamera": "body_camera",
    "BottomCamera": "bottom_camera",
    "SideCameraLeft": "side_camera_left",
    "SideCameraRight": "side_camera_right",
}


def locate_camera(list_dir: ListDir, camera: str) -> str:
    """Return a camera's video, relative to the session, in either layout."""
    top = list_dir("behavior-videos")
    if camera in top:
        folder = f"behavior-videos/{camera}"
        if "video.mp4" not in list_dir(folder):
            raise FileNotFoundError(f"no {folder}/video.mp4")
        return f"{folder}/video.mp4"
    # Legacy videos sit in a behavior session's folder, or loose at the top; a
    # session can hold several behavior_* folders, not all with videos.
    legacy = {"behavior-videos": top} | {
        folder: list_dir(folder)
        for name in list_dir("behavior")
        if name.startswith("behavior_") and "behavior-videos" in list_dir(f"behavior/{name}")
        for folder in [f"behavior/{name}/behavior-videos"]
    }
    stem = LEGACY_CAMERAS.get(camera, camera)
    holding = [f for f, present in legacy.items() if f"{stem}.mp4" in present]
    if len(holding) > 1:
        raise FileNotFoundError(f"{camera} video in several folders: {holding}")
    if holding:
        return f"{holding[0]}/{stem}.mp4"
    cameras = {n for present in legacy.values() for n in present}
    raise FileNotFoundError(f"no {camera} video; present: {', '.join(sorted(cameras)) or 'none'}")


def local_list_dir(session_dir: Path) -> ListDir:
    """List folders of a session mounted at ``session_dir``."""

    def list_dir(relative: str) -> list[str]:
        folder = session_dir / relative
        return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []

    return list_dir
