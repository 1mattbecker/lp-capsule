"""The run record, ``run.json``: what ran, on what, and whether it finished.

The launcher reads ``status`` to tell a finished run from one Code Ocean calls
succeeded although the script failed, so ``status: ok`` is written last.
"""

from __future__ import annotations

import json
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "lightning-pose",
    "kornia",
    "torch",
    "nvidia-dali-cuda110",
    "aind-video-utils",
    "aind-dynamic-foraging-behavior-video-analysis",
)
RUN_FILE = "run.json"


def package_versions() -> dict[str, str | None]:
    """Return the installed version of each package that changes results."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def ffmpeg_version() -> str | None:
    """Return the first line of ``ffmpeg -version``, or None without ffmpeg."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return None
    return out.split("\n")[0].strip() or None


class RunRecord:
    """Collects a run's facts and writes them to ``run.json``."""

    def __init__(self, results_dir: Path, task: str) -> None:
        self.path = results_dir / RUN_FILE
        self.started = time.time()
        self.data: dict[str, Any] = {
            "task": task,
            "status": "running",
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "versions": {**package_versions(), "ffmpeg": ffmpeg_version()},
        }

    def update(self, **fields: Any) -> None:
        """Add fields to the record."""
        self.data.update(fields)

    def write(self, status: str, **fields: Any) -> None:
        """Write the record with a final status."""
        self.data.update(fields)
        self.data["status"] = status
        self.data["seconds"] = round(time.time() - self.started, 1)
        tmp = self.path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        tmp.replace(self.path)
