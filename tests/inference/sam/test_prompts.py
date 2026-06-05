"""Unit tests for the SAM prompt builders (CPU; no SAM)."""

import numpy as np
import pytest

from sleap_nn.inference.sam.prompts import (
    PROMPT_MODES,
    SamPrompt,
    box_prompt,
    centroid_prompt,
    crop_center_prompt,
    kpt_box,
    pose_prompt,
    prompt_for_instance,
    visible_keypoints,
)


def test_visible_keypoints_drops_non_finite():
    pts = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.inf], [5.0, 6.0]])
    vis = visible_keypoints(pts)
    assert vis.shape == (2, 2)
    np.testing.assert_allclose(vis, [[1.0, 2.0], [5.0, 6.0]])


def test_kpt_box_margin_and_clamp():
    # Two keypoints; side = 20 px each axis -> margin = max(15, 0.6*20) = 15.
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    box = kpt_box(pts, (100, 120))
    np.testing.assert_allclose(box, [0.0, 5.0, 45.0, 55.0])
    # Clamps to [0, w-1] x [0, h-1].
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 119 and box[3] <= 99


def test_kpt_box_single_point_uses_min_margin():
    # A degenerate (single-point) instance: side 0 -> margin = margin_min.
    pts = np.array([[50.0, 50.0]])
    box = kpt_box(pts, (200, 200))
    np.testing.assert_allclose(box, [35.0, 35.0, 65.0, 65.0])


def test_pose_prompt_points_and_box():
    pts = np.array([[10.0, 20.0], [30.0, 40.0], [np.nan, np.nan]])
    p = pose_prompt(pts, (100, 120))
    assert p.mode == "pose"
    # Non-finite row dropped -> 2 positive points, all label 1.
    assert p.point_coords.shape == (2, 2)
    assert p.point_labels.tolist() == [1, 1]
    # Box is present and equals the reject box.
    assert p.box is not None
    np.testing.assert_allclose(p.box, p.reject_box)


def test_pose_prompt_requires_visible_keypoints():
    with pytest.raises(ValueError):
        pose_prompt(np.array([[np.nan, np.nan]]), (50, 50))


def test_centroid_prompt_point_only_no_box():
    p = centroid_prompt(np.array([25.0, 35.0]), (100, 100))
    assert p.mode == "centroid"
    assert p.point_coords.shape == (1, 2)
    np.testing.assert_allclose(p.point_coords[0], [25.0, 35.0])
    # The box is NEVER passed to SAM in centroid mode (point is ambiguous).
    assert p.box is None
    # A reject box still exists for the candidate-rejection heuristic.
    assert p.reject_box.shape == (4,)


def test_centroid_prompt_reject_box_from_keypoints_when_given():
    kpts = np.array([[10.0, 20.0], [30.0, 40.0]])
    p = centroid_prompt(np.array([20.0, 30.0]), (100, 120), keypoints=kpts)
    np.testing.assert_allclose(p.reject_box, kpt_box(kpts, (100, 120)))


def test_box_prompt_box_only_no_points():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    p = box_prompt(pts, (100, 120))
    assert p.mode == "box"
    assert p.point_coords is None
    assert p.point_labels is None
    assert p.box is not None
    np.testing.assert_allclose(p.box, p.reject_box)


def test_crop_center_prompt_is_center_pixel():
    p = crop_center_prompt((64, 80))  # (h, w)
    assert p.mode == "crop_center"
    # Center pixel is (w/2, h/2) = (40, 32).
    np.testing.assert_allclose(p.point_coords[0], [40.0, 32.0])
    # Reject box is the full crop extent.
    np.testing.assert_allclose(p.reject_box, [0.0, 0.0, 79.0, 63.0])


def test_prompt_for_instance_pose_with_keypoints():
    p = prompt_for_instance(
        "pose", (100, 120), keypoints=np.array([[10.0, 20.0], [30.0, 40.0]])
    )
    assert p.mode == "pose"
    assert p.point_coords.shape == (2, 2)


def test_prompt_for_instance_pose_falls_back_to_centroid_point():
    # The L3 product rule: no visible keypoints -> center-point fallback.
    p = prompt_for_instance(
        "pose",
        (100, 120),
        keypoints=np.array([[np.nan, np.nan]]),
        centroid=np.array([50.0, 60.0]),
    )
    assert p.mode == "centroid"
    np.testing.assert_allclose(p.point_coords[0], [50.0, 60.0])


def test_prompt_for_instance_pose_no_fallback_raises():
    with pytest.raises(ValueError):
        prompt_for_instance("pose", (50, 50), keypoints=None, centroid=None)


def test_prompt_for_instance_centroid_means_keypoints_when_no_centroid():
    p = prompt_for_instance(
        "centroid", (100, 100), keypoints=np.array([[10.0, 20.0], [30.0, 40.0]])
    )
    np.testing.assert_allclose(p.point_coords[0], [20.0, 30.0])


def test_prompt_for_instance_crop_center_prefers_pose_when_local_keypoints():
    # crop_center with crop-local keypoints uses the richer pose prompt.
    p = prompt_for_instance(
        "crop_center", (64, 64), keypoints=np.array([[20.0, 20.0], [40.0, 40.0]])
    )
    assert p.mode == "pose"


def test_prompt_for_instance_crop_center_naive_when_no_keypoints():
    p = prompt_for_instance("crop_center", (64, 64), keypoints=None)
    assert p.mode == "crop_center"
    np.testing.assert_allclose(p.point_coords[0], [32.0, 32.0])


def test_prompt_for_instance_unknown_mode_raises():
    with pytest.raises(ValueError):
        prompt_for_instance("nope", (10, 10))


def test_prompt_modes_constant():
    assert PROMPT_MODES == ("pose", "centroid", "box", "crop_center")


def test_sam_prompt_is_a_dataclass():
    p = SamPrompt(
        point_coords=None,
        point_labels=None,
        box=np.zeros(4, np.float32),
        reject_box=np.zeros(4, np.float32),
        mode="box",
    )
    assert p.mode == "box"
