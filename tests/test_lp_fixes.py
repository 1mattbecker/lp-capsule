"""lp_fixes against stand-ins for torch and LP (neither is installed for tests)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


@pytest.fixture
def fake_lp(monkeypatch):
    """Stub modules: a 2-batch labeled loader in model coords (384) of 768x384 frames."""
    batches = [
        {"keypoints": np.array([[192.0, 96.0]]), "frame": (768, 384)},
        {"keypoints": np.array([[0.0, 384.0], [384.0, 0.0]]), "frame": (768, 384)},
    ]

    class DataExtractor:
        def __init__(self, data_module, cond, extract_images, remove_augmentations):
            assert (cond, extract_images, remove_augmentations) == ("train", False, True)
            self.dataset_length = 3

        def get_loader(self):
            return "combined"

        @staticmethod
        def verify_labeled_loader(loader):
            assert loader == "combined"
            return batches

    def model_to_frame_batch(batch, keypoints, in_place=True):
        assert in_place is False
        w, h = batch["frame"]
        return keypoints * np.array([w / 384, h / 384])

    class KeypointPCA:
        def __init__(self, data_module):
            self.data_module = data_module

        def _get_data(self):
            raise AssertionError("not patched")

    modules = {
        "torch": types.SimpleNamespace(cat=lambda xs, dim: np.concatenate(xs, axis=dim)),
        "lightning_pose": types.ModuleType("lightning_pose"),
        "lightning_pose.utils": types.ModuleType("lightning_pose.utils"),
        "lightning_pose.utils.pca": types.SimpleNamespace(KeypointPCA=KeypointPCA),
        "lightning_pose.data": types.ModuleType("lightning_pose.data"),
        "lightning_pose.data.bboxes": types.SimpleNamespace(
            model_to_frame_batch=model_to_frame_batch
        ),
        "lightning_pose.data.extractor": types.SimpleNamespace(DataExtractor=DataExtractor),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["lightning_pose.utils"].pca = modules["lightning_pose.utils.pca"]
    import lp_fixes

    monkeypatch.setattr(lp_fixes, "APPLIED", [])
    return KeypointPCA, lp_fixes


def test_pca_fit_on_frame_coords(fake_lp):
    KeypointPCA, lp_fixes = fake_lp
    assert lp_fixes.apply() == ["pca_frame_coords"]
    assert lp_fixes.apply() == ["pca_frame_coords"]  # once
    pca = KeypointPCA(data_module="dm")
    pca._get_data()
    np.testing.assert_allclose(pca.data_arr, [[384.0, 96.0], [0.0, 384.0], [768.0, 0.0]])
