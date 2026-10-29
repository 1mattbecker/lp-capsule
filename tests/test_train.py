"""Building the training config; training itself needs a GPU and LP."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES, PYLON2_NAMES, write
from run_config import TrainConfig
from train import build_cfg, pixel_error_summary


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "data/pylon2test2_r01"
    header = (FIXTURES / "pylon2test2.labels_header.csv").read_text()
    write(root / "CollectedData.csv", header + "labeled-data/a/img0.png" + ",1" * 22 + "\n")
    write(root / "videos/a.mp4", b"")
    write(root / "labeled-data/a/img0.png", b"png")
    return root


def test_build_cfg_points_at_project(project, pylon2_bundle, tmp_path):
    data = project.parent
    shutil.move(pylon2_bundle, data / pylon2_bundle.name)
    config = TrainConfig(
        project=project.name,
        base_config="ephys_BottomCamera_Pylon2Test2",
        overrides={"training": {"max_epochs": 5, "min_epochs": 5, "num_workers": 99}},
    )
    cfg, info = build_cfg(config, data, tmp_path)
    assert cfg.data.data_dir == str(project)
    assert cfg.data.csv_file == "CollectedData.csv"
    assert cfg.data.video_dir == str(project / "videos")
    assert list(cfg.data.keypoint_names) == PYLON2_NAMES
    assert cfg.training.max_epochs == 5
    assert cfg.training.num_workers <= 99
    assert cfg.eval.predict_vids_after_training is False
    assert cfg.model.checkpoint is None
    assert info["n_labeled_frames"] == 1
    assert info["base_config"] == "ephys_BottomCamera_Pylon2Test2/training.config.yaml"


def test_semi_supervised_needs_videos(project, pylon2_bundle):
    data = project.parent
    shutil.move(pylon2_bundle, data / pylon2_bundle.name)
    (project / "videos/a.mp4").unlink()
    config = TrainConfig(project=project.name, base_config="ephys_BottomCamera_Pylon2Test2")
    with pytest.raises(FileNotFoundError, match="unlabeled video"):
        build_cfg(config, data, data)


def test_keypoints_must_match_base(project, anna_bundle):
    data = project.parent
    shutil.move(anna_bundle, data / anna_bundle.name)
    config = TrainConfig(project=project.name, base_config="anna_face_test")
    with pytest.raises(ValueError, match="differ from the base model"):
        build_cfg(config, data, data)


def test_pixel_error_summary(tmp_path):
    write(
        tmp_path / "predictions_pixel_error.csv",
        ",a,b,set\nimg0,1.0,3.0,train\nimg1,3.0,5.0,train\nimg2,2.0,,test\n",
    )
    summary = pixel_error_summary(tmp_path).set_index(["set", "keypoint"])
    assert summary.loc[("train", "a"), "mean"] == 2.0
    assert summary.loc[("test", "a"), "count"] == 1
    assert summary.loc[("test", "b"), "count"] == 0


def test_project_assets_merged_with_label_check(project, pylon2_bundle, tmp_path):
    """Old labels and videos in the model asset, new ones in a second asset."""
    data = project.parent
    shutil.move(pylon2_bundle, data / pylon2_bundle.name)
    old = data / pylon2_bundle.name
    write(old / "videos/clip.mp4", b"")
    new = data / "r01_labels"
    header = (FIXTURES / "pylon2test2.labels_header.csv").read_text()
    write(
        new / "CollectedData.train.csv",
        header
        + "labeled-data/a/img0.png" + ",1" * 22 + "\n"
        + "labeled-data/sess/img00000007.jpg" + ",1" * 22 + "\n",
    )  # fmt: skip
    write(new / "labeled-data/sess/img00000007.jpg", b"jpg")
    config = TrainConfig(
        project=("r01_labels", old.name),
        base_config=old.name,
        csv_file="CollectedData.train.csv",
    )
    cfg, info = build_cfg(config, data, tmp_path)
    merged = tmp_path / "project"
    assert cfg.data.data_dir == str(merged)
    assert (merged / "labeled-data/a").resolve() == (old / "labeled-data/a").resolve()
    assert (merged / "labeled-data/sess").resolve() == (new / "labeled-data/sess").resolve()
    assert (merged / "videos/clip.mp4").is_symlink()
    assert info["n_labeled_frames"] == 2
    (new / "labeled-data/sess/img00000007.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="1 of 2 labeled images missing"):
        build_cfg(config, data, tmp_path / "again")


def test_pixel_error_summary_pooled(tmp_path):
    write(tmp_path / "predictions_pixel_error.csv", ",a,set\nimg0,1.0,train\nimg1,3.0,validation\n")
    summary = pixel_error_summary(tmp_path, by_set=False)
    assert summary.set.unique().tolist() == ["all"] and summary["mean"].iloc[0] == 2.0


def test_context_frames_from_clip(project, pylon2_bundle, mp4, tmp_path):
    """Neighbours of a clip's labeled PNG come from videos/<folder>.mp4."""
    import subprocess

    data = project.parent
    shutil.move(pylon2_bundle, data / pylon2_bundle.name)
    old = data / "clips"
    header = (FIXTURES / "pylon2test2.labels_header.csv").read_text()
    write(
        old / "CollectedData.csv",
        header
        + "labeled-data/c1/img10.png" + ",1" * 22 + "\n"
        + "labeled-data/c1/img58.png" + ",1" * 22 + "\n",
    )  # fmt: skip
    (old / "videos").mkdir(parents=True)
    shutil.copy(mp4, old / "videos/c1.mp4")
    png = old / "labeled-data/c1/img10.png"
    png.parent.mkdir(parents=True)

    def frame(k, out):
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4),
             "-vf", f"select=eq(n\\,{k})", "-frames:v", "1", "-fps_mode", "passthrough",
             str(out)],
            check=True,
        )  # fmt: skip

    frame(10, png)
    frame(58, png.with_name("img58.png"))
    config = TrainConfig(
        project=("clips",), base_config=pylon2_bundle.name, context_frames_from_videos=True
    )
    cfg, info = build_cfg(config, data, tmp_path / "w1")
    folder = Path(cfg.data.data_dir) / "labeled-data/c1"
    # The 60-frame clip ends at frame 59: frame 58 gets 56, 57 and 59 only.
    assert sorted(p.name for p in folder.iterdir()) == [
        "img08.png", "img09.png", "img10.png", "img11.png", "img12.png",
        "img56.png", "img57.png", "img58.png", "img59.png",
    ]  # fmt: skip
    assert not any((old / "labeled-data/c1").glob("img0*.png"))  # the asset is untouched
    assert info["context_frames"]["written"] == 7
    assert info["context_frames"]["max_center_mean_abs_diff"] == 0
    frame(20, png)  # labeled image no longer frame 10 of the clip
    with pytest.raises(ValueError, match="not the folder's clip"):
        build_cfg(config, data, tmp_path / "w2")
