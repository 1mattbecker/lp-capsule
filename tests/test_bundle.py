from __future__ import annotations

import pytest
from bundle import find_bundle, find_checkpoint, find_config
from conftest import PYLON2_NAMES, lp_config, write


def test_pylon2_layout(pylon2_bundle):
    bundle = find_bundle(pylon2_bundle)
    assert bundle.config.name == "training.config.yaml"
    # Identical copies: the shallower one.
    assert bundle.checkpoint == pylon2_bundle / "epoch=209-step=17220-best.ckpt"
    assert list(bundle.keypoint_names) == PYLON2_NAMES
    assert bundle.keypoint_source in ("CollectedData.csv", "CollectedData.masked.csv")


def test_anna_layout(anna_bundle):
    bundle = find_bundle(anna_bundle)
    assert bundle.config.name == "training.config.yaml"
    assert bundle.checkpoint.parent.name == "checkpoints"
    assert bundle.keypoint_names == ("eye_top", "eye_right")
    assert bundle.keypoint_source == "config"


def test_top_level_ckpt_without_best(tmp_path):
    root = tmp_path / "ephys_bottom-camera"
    write(root / "training.config.yaml", lp_config(["a", "b"], 2))
    write(root / "epoch=214-step=10535.ckpt", b"w")
    assert find_bundle(root).checkpoint.name == "epoch=214-step=10535.ckpt"


def test_different_configs_fail(anna_bundle):
    write(anna_bundle / "other.yaml", lp_config(["x", "y"], 2))
    with pytest.raises(ValueError, match="different LP configs"):
        find_config(anna_bundle)


def test_different_checkpoints_prefer_best(tmp_path):
    write(tmp_path / "epoch=1.ckpt", b"aaaa")
    best = write(tmp_path / "epoch=2-best.ckpt", b"bbbbbbbb")
    assert find_checkpoint(tmp_path) == best
    write(tmp_path / "sub/epoch=3-best.ckpt", b"cc")
    with pytest.raises(ValueError, match="several different"):
        find_checkpoint(tmp_path)


def test_choose_best_last_or_named(tmp_path):
    """A trained bundle: LP's best plus its every-n-epochs checkpoints."""
    ckpts = tmp_path / "model/tb_logs/test/version_0/checkpoints"
    best = write(ckpts / "epoch=4-step=450-best.ckpt", b"best")
    write(ckpts / "epoch=49-step=4500.ckpt", b"e49")
    last = write(ckpts / "epoch=99-step=9000.ckpt", b"e99!")
    assert find_checkpoint(tmp_path) == best
    assert find_checkpoint(tmp_path, "best") == best
    assert find_checkpoint(tmp_path, "last") == last
    assert find_checkpoint(tmp_path, "epoch=49-step=4500.ckpt").name == "epoch=49-step=4500.ckpt"
    with pytest.raises(FileNotFoundError, match="no epoch=1.ckpt"):
        find_checkpoint(tmp_path, "epoch=1.ckpt")


def test_last_prefers_every_n_copy_and_falls_back_to_best(tmp_path):
    best = write(tmp_path / "epoch=209-step=17220-best.ckpt", b"w")
    assert find_checkpoint(tmp_path, "last") == best
    copy = write(tmp_path / "epoch=209-step=17220.ckpt", b"w")
    assert find_checkpoint(tmp_path, "last") == copy
    write(tmp_path / "model.ckpt", b"x")
    assert find_checkpoint(tmp_path, "last") == copy


def test_same_size_different_content(tmp_path):
    write(tmp_path / "a.ckpt", b"aaaa")
    write(tmp_path / "b.ckpt", b"bbbb")
    with pytest.raises(ValueError, match="several different"):
        find_checkpoint(tmp_path)


def test_checkpoints_in_data_folders_ignored(tmp_path):
    write(tmp_path / "model.ckpt", b"w")
    write(tmp_path / "videos/stray.ckpt", b"x")
    assert find_checkpoint(tmp_path).name == "model.ckpt"


def test_names_must_match_num_keypoints(tmp_path):
    write(tmp_path / "config.yaml", lp_config(["a", "b", "c"], 2))
    write(tmp_path / "m.ckpt", b"w")
    with pytest.raises(ValueError, match="num_keypoints"):
        find_bundle(tmp_path)


def test_no_names_anywhere(tmp_path):
    write(tmp_path / "config.yaml", lp_config(None, 2))
    write(tmp_path / "m.ckpt", b"w")
    with pytest.raises(ValueError, match="no keypoint names"):
        find_bundle(tmp_path)


def test_trained_output_layout(tmp_path):
    """What the train task writes: model/config.yaml with names, ckpt under tb_logs."""
    model = tmp_path / "model"
    write(model / "config.yaml", lp_config(["a", "b"], 2))
    write(model / "tb_logs/test/version_0/checkpoints/epoch=9-step=90-best.ckpt", b"w")
    write(model / "tb_logs/test/version_0/hparams.yaml", "num_keypoints: 2\n")
    write(tmp_path / "run_config.yaml", "task: train\nproject: p\n")
    bundle = find_bundle(tmp_path)
    assert bundle.config == model / "config.yaml"
    assert bundle.keypoint_names == ("a", "b")
