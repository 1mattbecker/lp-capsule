"""Fixes applied to Lightning Pose 2.4.3 at run time, each recorded in run.json.

``pca_frame_coords``: LP's PCA losses fit their PCA on the training labels as
the dataset returns them, in model coordinates (the resized image), but score
the unlabeled predictions after ``model_to_frame_batch`` has mapped them to
original-frame coordinates. Whenever the frame is not the resize size (here
720x540 vs 384x384) a correct pose lies off the fitted subspace: the
pca_singleview loss of Pylon2Test2's own predictions is about 13-20 px instead
of about 2, and the loss pulls the model away from the labels (exp1_all_semi
and the exp2 semi smoke run). The fix fits the PCA on the training labels in
original-frame coordinates, mapped with the same ``model_to_frame_batch``.
"""

from __future__ import annotations

from typing import Any

APPLIED: list[str] = []


def frame_coords_keypoints(data_module: Any) -> Any:
    """The training split's labels in original-frame coordinates, as LP's PCA fit takes them."""
    import torch
    from lightning_pose.data.bboxes import model_to_frame_batch
    from lightning_pose.data.extractor import DataExtractor

    extractor = DataExtractor(
        data_module=data_module, cond="train", extract_images=False, remove_augmentations=True
    )
    loader = extractor.verify_labeled_loader(extractor.get_loader())
    keypoints = [
        model_to_frame_batch(batch, batch["keypoints"], in_place=False) for batch in loader
    ]
    out = torch.cat(keypoints, dim=0)
    assert out.shape[0] == extractor.dataset_length, (out.shape, extractor.dataset_length)
    return out


def apply() -> list[str]:
    """Patch LP in this process; return the names of the fixes applied."""
    from lightning_pose.utils import pca

    if "pca_frame_coords" not in APPLIED:

        def _get_data(self: Any) -> None:
            self.data_arr = frame_coords_keypoints(self.data_module)

        pca.KeypointPCA._get_data = _get_data
        APPLIED.append("pca_frame_coords")
    return list(APPLIED)
