"""Shared fixtures: model bundles in the known layouts, and a small MP4."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"
PYLON2_NAMES = [
    "nose_tip",
    "jaw",
    "tongue_tip_right",
    "tongue_tip_center",
    "tongue_tip_left",
    "pointer_finger_r",
    "paw_wrist_r",
    "pointer_finger_l",
    "paw_wrist_l",
    "spout_r",
    "spout_l",
]


def write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


@pytest.fixture
def pylon2_bundle(tmp_path: Path) -> Path:
    """The Pylon2Test2 layout: names only in the CSV, the ckpt in two places."""
    root = tmp_path / "ephys_BottomCamera_Pylon2Test2"
    shutil.copy(FIXTURES / "pylon2test2.config.yaml", write(root / "training.config.yaml", ""))
    header = (FIXTURES / "pylon2test2.labels_header.csv").read_text()
    write(root / "CollectedData.csv", header + "labeled-data/a/img0.png" + ",1" * 22 + "\n")
    write(root / "CollectedData.masked.csv", header)
    ckpt = b"weights" * 10
    write(root / "epoch=209-step=17220-best.ckpt", ckpt)
    version = root / "outputs/tb_logs/test/version_0"
    write(version / "checkpoints/epoch=209-step=17220-best.ckpt", ckpt)
    write(version / "hparams.yaml", "num_keypoints: 11\nbackbone: resnet50_animal_ap10k\n")
    write(root / "labeled-data/a/img0.png", b"png")
    return root


def lp_config(names: list[str] | None, num_keypoints: int) -> str:
    return yaml.safe_dump(
        {
            "data": {"keypoint_names": names, "num_keypoints": num_keypoints},
            "model": {"model_type": "heatmap", "model_name": "test"},
        }
    )


@pytest.fixture
def anna_bundle(tmp_path: Path) -> Path:
    """Anna's layout: names in the config, two identical configs, ckpt in checkpoints/."""
    root = tmp_path / "anna_face_test"
    config = lp_config(["eye_top", "eye_right"], 2)
    write(root / "training.config.yaml", config)
    write(root / "super_updated_tube_wheel_face.config.yaml", config)
    write(root / "hparams.yaml", "num_keypoints: 2\n")
    write(root / "checkpoints/epoch=294-step=23895-best.ckpt", b"w")
    return root


@pytest.fixture(scope="session")
def mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 60-frame, 30 fps H.264 MP4 whose frames show their own number."""
    path = tmp_path_factory.mktemp("video") / "video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30:duration=2",
            "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path
