"""Tests for the native centroid-from-masks path.

Covers ``derive_centroids_from_masks`` (synthesize a single-node ``centroid`` pose from
each segmentation mask) and its wiring into ``ModelTrainer._setup_train_val_labels`` via
``data_config.centroids_from_masks`` — so a ``centroid`` model can train on mask-only
data with no offline preprocessing.
"""

import numpy as np
import pytest
import sleap_io as sio
from omegaconf import OmegaConf

from sleap_nn.data.custom_datasets import derive_centroids_from_masks
from tests.fixtures.datasets import make_seg_labels_from_slp


@pytest.fixture
def seg_labels(minimal_instance):
    """Mask-only labels (circular blobs around each instance).

    ``make_seg_labels_from_slp`` retains the source keypoint instances alongside the
    masks; this path is for mask-only data, so strip them (matching the real use case —
    and the fail-fast guard against clobbering real poses).
    """
    labels = make_seg_labels_from_slp(str(minimal_instance))
    for lf in labels:
        lf.instances = []
    labels.skeletons = []
    return labels


class TestDeriveCentroidsFromMasks:
    def test_adds_single_node_centroid_skeleton(self, seg_labels):
        n_masks = sum(len(lf.masks) for lf in seg_labels)
        assert n_masks > 0

        out = derive_centroids_from_masks(seg_labels, centering="com")

        assert [s.name for s in out.skeletons] == ["centroid"]
        assert [n.name for n in out.skeletons[0].nodes] == ["centroid"]
        n_inst = sum(len(lf.instances) for lf in out)
        assert n_inst == n_masks
        # One single-node instance per mask, back-linked.
        for lf in out:
            assert len(lf.instances) == len(lf.masks)
            for mask, inst in zip(lf.masks, lf.instances):
                assert inst.numpy().shape == (1, 2)
                assert mask.instance is inst

    def test_com_matches_blob_center(self, seg_labels):
        """The COM of a circular blob is its center (within ~1px)."""
        lf = seg_labels[0]
        from sleap_nn.inference.segmentation_convert import decode_mask_to_image_res

        expected = []
        for mask in lf.masks:
            mb = np.asarray(decode_mask_to_image_res(mask)) > 0.5
            ys, xs = np.where(mb)
            expected.append((xs.mean(), ys.mean()))

        derive_centroids_from_masks(seg_labels, centering="com")
        for (ex, ey), inst in zip(expected, seg_labels[0].instances):
            cx, cy = inst.numpy()[0]
            assert abs(cx - ex) < 1e-3 and abs(cy - ey) < 1e-3
            # Centroid lies inside the image.
            h, w = seg_labels.videos[0].shape[1:3]
            assert 0 <= cx < w and 0 <= cy < h

    def test_bbox_centering_routes_concave_mask_to_bbox_midpoint(self):
        """For a concave (L-shaped) mask, bbox midpoint != center-of-mass.

        Validates the ``centering='bbox'`` branch INSIDE derive_centroids_from_masks
        (not just the primitive), and that COM and bbox genuinely diverge.
        """
        video = sio.Video.from_filename("v.mp4")
        # L-shape: a tall left bar + a bottom bar. COM is pulled toward the mass; the
        # bbox midpoint is the center of the (full) bounding box.
        blob = np.zeros((40, 40), dtype=bool)
        blob[4:36, 4:10] = True  # tall left bar
        blob[30:36, 4:36] = True  # bottom bar

        def _one(centering):
            labels = sio.Labels(
                videos=[video],
                labeled_frames=[
                    sio.LabeledFrame(
                        video=video,
                        frame_idx=0,
                        masks=[sio.UserSegmentationMask.from_numpy(blob)],
                    )
                ],
            )
            derive_centroids_from_masks(labels, centering=centering)
            return labels[0].instances[0].numpy()[0]

        com = _one("com")
        bbox = _one("bbox")
        # bbox_center (upstream sleap-io #531) treats a pixel as a unit area, so the
        # midpoint of occupied columns/rows [4, 35] is (4 + 35 + 1) / 2 = 20.0 (the old
        # hand-rolled `(min+max)/2` gave 19.5; this is the deliberate +0.5px shift).
        assert abs(bbox[0] - 20.0) < 1e-6 and abs(bbox[1] - 20.0) < 1e-6
        # The COM is pulled off the bbox center by the asymmetric mass.
        assert abs(com[0] - bbox[0]) + abs(com[1] - bbox[1]) > 2.0

    def test_preserves_tracks(self, seg_labels):
        # Attach tracks to the masks, then confirm they ride onto the centroids.
        track = sio.Track(name="animal0")
        for lf in seg_labels:
            for mask in lf.masks:
                mask.track = track
        derive_centroids_from_masks(seg_labels, centering="com")
        for lf in seg_labels:
            for inst in lf.instances:
                assert inst.track is track

    def test_no_masks_is_noop(self, minimal_instance):
        labels = sio.load_slp(str(minimal_instance))  # keypoint-only, no masks
        orig_skel = labels.skeletons
        out = derive_centroids_from_masks(labels, centering="com")
        # Unchanged: no masks -> no synthesis.
        assert out.skeletons is orig_skel

    def test_invalid_centering_raises(self, seg_labels):
        with pytest.raises(ValueError, match="centering"):
            derive_centroids_from_masks(seg_labels, centering="diagonal")

    def test_mixed_pose_and_mask_frame_raises(self, seg_labels):
        # A frame carrying BOTH masks and real keypoint instances must fail fast — the
        # synthesized centroids would otherwise silently clobber the real poses. This
        # path is mask-only; mixed data should use the standard centroid path.
        pose_skel = sio.Skeleton(nodes=["a", "b"], name="pose")
        seg_labels[0].instances = [
            sio.Instance.from_numpy(
                np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=pose_skel
            )
        ]
        original = seg_labels[0].instances
        with pytest.raises(ValueError, match="mask-only"):
            derive_centroids_from_masks(seg_labels, centering="com")
        # Atomic: the real instances are untouched (guard runs before any mutation).
        assert seg_labels[0].instances is original


class TestTrainerWiring:
    """data_config.centroids_from_masks drives the synthesis in the trainer."""

    @staticmethod
    def _cfg(enabled, centering="com"):
        return OmegaConf.create(
            {
                "data_config": {
                    "centroids_from_masks": enabled,
                    "centroids_from_masks_centering": centering,
                    "user_instances_only": True,
                    "use_same_data_for_val": False,
                    "val_labels_path": ["dummy"],  # val provided -> no internal split
                    "split": None,
                }
            }
        )

    def _run(self, cfg, minimal_instance, mask_only=False):
        from sleap_nn.training.model_trainer import ModelTrainer

        # Independent train/val objects (the trainer mutates labels in place).
        train = make_seg_labels_from_slp(str(minimal_instance))
        val = make_seg_labels_from_slp(str(minimal_instance))
        if mask_only:
            for labels in (train, val):
                for lf in labels:
                    lf.instances = []
                labels.skeletons = []
        mt = ModelTrainer.__new__(ModelTrainer)
        mt.config = cfg
        mt.model_type = "centroid"
        mt._setup_train_val_labels(labels=[train], val_labels=[val])
        return mt

    def test_enabled_synthesizes_centroids(self, minimal_instance):
        mt = self._run(self._cfg(True), minimal_instance, mask_only=True)
        assert [s.name for s in mt.skeletons] == ["centroid"]
        assert sum(len(lf.instances) for lf in mt.train_labels[0]) == sum(
            len(lf.masks) for lf in mt.train_labels[0]
        )

    def test_disabled_leaves_mask_only_data_skeletonless(self, minimal_instance):
        # Mask-only labels (no retained pose) stay skeleton-less when the flag is off.
        mt = self._run(self._cfg(False), minimal_instance, mask_only=True)
        assert not mt.skeletons
