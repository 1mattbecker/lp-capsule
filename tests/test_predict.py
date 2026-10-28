"""Clip cutting and re-indexing, with a stand-in for the LP model.

The stand-in predicts each frame's mean gray level as ``x``, so a prediction
row can be matched to the source frame it came from.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest
from aind_video_utils import read_mp4_frame_index
from predict import predict_video, reindex, rename_head_keys
from run_config import VideoSpec


def gray_means(mp4) -> np.ndarray:
    """Return each frame's mean gray level, decoded by ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "gray", "-fps_mode", "passthrough", "pipe:1"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, 48 * 64).mean(axis=1)


class FakeModel:
    """Predicts one keypoint per frame: x = the frame's mean gray level."""

    def __init__(self):
        self.videos = []

    def predict_on_video_file(self, video, output_dir, compute_metrics, reader):
        self.videos.append(video)
        x = gray_means(video)
        columns = pd.MultiIndex.from_product(
            [["scorer"], ["kp"], ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        values = np.stack([x, np.zeros_like(x), np.ones_like(x)], axis=1)

        class Result:
            predictions = pd.DataFrame(values, columns=columns)

        return Result()


def test_reindex_checks_count():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert list(reindex(df, 100, 3).index) == [100, 101, 102]
    with pytest.raises(RuntimeError, match="got 3 predictions"):
        reindex(df, 0, 4)


def test_windows_keyed_by_source_frame(mp4, tmp_path):
    spec = VideoSpec(asset="s", camera="C", windows=((5, 12), (40, 47)))
    model = FakeModel()
    result = predict_video(model, mp4, spec, "dali", tmp_path)
    frames = list(range(5, 12)) + list(range(40, 47))
    assert list(result.predictions.index) == frames
    # The clip re-encodes (CRF 18), so levels match closely, not exactly.
    source = gray_means(mp4)[frames]
    got = result.predictions[("scorer", "kp", "x")].to_numpy()
    assert np.abs(got - source).max() < 1.0
    assert [w["start"] for w in result.info["windows"]] == [5, 40]
    assert not list(tmp_path.glob("clip_*.mp4")), "clips are deleted"


def test_whole_video_uniform_is_not_restamped(mp4, tmp_path):
    model = FakeModel()
    result = predict_video(model, mp4, VideoSpec(asset="s", camera="C"), "dali", tmp_path)
    assert model.videos == [mp4]
    assert len(result.predictions) == read_mp4_frame_index(mp4).n_samples
    assert result.info["restamped_after_frame_index"] == []


def test_window_past_end_fails(mp4, tmp_path):
    spec = VideoSpec(asset="s", camera="C", windows=((55, 70),))
    with pytest.raises(ValueError, match="past the end"):
        predict_video(FakeModel(), mp4, spec, "opencv", tmp_path)


def test_rename_head_keys():
    state = {"upsampling_layers.0.weight": 1, "backbone.conv.weight": 2}
    renamed = rename_head_keys(state)
    assert renamed == {"head.upsampling_layers.0.weight": 1, "backbone.conv.weight": 2}
