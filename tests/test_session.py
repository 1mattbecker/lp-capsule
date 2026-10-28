from __future__ import annotations

import pytest
from conftest import write
from session import local_list_dir, locate_camera


def test_file_standard_layout(tmp_path):
    write(tmp_path / "behavior-videos/BottomCamera/video.mp4", b"")
    write(tmp_path / "behavior-videos/BottomCamera/metadata.csv", "")
    assert (
        locate_camera(local_list_dir(tmp_path), "BottomCamera")
        == "behavior-videos/BottomCamera/video.mp4"
    )


def test_legacy_layout(tmp_path):
    write(tmp_path / "behavior/behavior_1_2024/behavior-videos/bottom_camera.mp4", b"")
    write(tmp_path / "behavior/behavior_0_2024/other.json", "")
    assert (
        locate_camera(local_list_dir(tmp_path), "BottomCamera")
        == "behavior/behavior_1_2024/behavior-videos/bottom_camera.mp4"
    )


def test_missing_camera_lists_present(tmp_path):
    write(tmp_path / "behavior-videos/side_camera_left.mp4", b"")
    with pytest.raises(FileNotFoundError, match="side_camera_left.mp4"):
        locate_camera(local_list_dir(tmp_path), "BottomCamera")
