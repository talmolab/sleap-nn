"""Centroid-method knobs on the `embedding` head (#586, #671 follow-up).

`EmbeddingDataset` centers each re-ID crop on a pose centroid, and until now that
centroid was always the mean of visible nodes (or the `anchor_part` node). #738
gave every other head config `centroid_method` / `centroid_fallback` in sleap-io's
vocabulary; these are the same knobs on the embedding head, plus the parity that
makes them safe: `centroid_method: null` must reproduce today's crops exactly.
"""

import numpy as np
import pytest
import sleap_io as sio
from omegaconf import OmegaConf

from sleap_nn.config.utils import check_centroid_methods
from sleap_nn.data.custom_datasets import EmbeddingDataset

# Three nodes; deliberately asymmetric so the three reduce methods disagree:
#   center_of_mass -> (10 + 0 + 2) / 3 = 4.0 in x
#   bbox_center    -> (0 + 10) / 2     = 5.0 in x
#   anchor("a")    -> 10.0 in x
_SKEL = sio.Skeleton(nodes=["a", "b", "c"], name="s")
_POINTS = np.array([[10.0, 0.0], [0.0, 0.0], [2.0, 0.0]])

_COM_X = 4.0
_BBOX_X = 5.0
_ANCHOR_X = 10.0


def _head(**overrides):
    cfg = {
        "embedding_dim": 16,
        "output_stride": 16,
        "objective": {
            "positives": {"scope": "tracklet", "aug_views": 2},
            "negatives": {"sources": ["in_batch"], "restrict_same_video": True},
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _labels(skeleton=_SKEL, points=_POINTS):
    video = sio.Video.from_filename("a.mp4")
    track = sio.Track(name="t0")
    frame = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[sio.Instance.from_numpy(points, skeleton=skeleton, track=track)],
    )
    return sio.Labels(
        videos=[video],
        labeled_frames=[frame],
        skeletons=[skeleton],
        tracks=[track],
    )


def _dataset(head, labels=None):
    return EmbeddingDataset(
        labels=[labels if labels is not None else _labels()],
        crop_size=16,
        class_names=[],
        embedding_head_config=head,
        max_stride=16,
        id_scope="tracklet",
        cache_img=None,
    )


def _crop_center_x(head, labels=None):
    """The x of the one indexed instance's crop center."""
    dataset = _dataset(head, labels)
    assert len(dataset.mask_idx_list) == 1
    return float(dataset.mask_idx_list[0]["centroid"][0])


class TestEmbeddingCentroidMethod:
    def test_default_is_center_of_mass(self):
        """`centroid_method: null` with no anchor = the historical mean-of-visible."""
        head = _head()
        dataset = _dataset(head)
        assert (dataset.centroid_method, dataset.centroid_fallback) == (
            "center_of_mass",
            None,
        )
        assert _crop_center_x(head) == pytest.approx(_COM_X)

    def test_absent_keys_reproduce_the_default(self):
        """A config written before #586 has neither key and must be unchanged."""
        head = _head()
        assert "centroid_method" not in head
        assert _crop_center_x(head) == pytest.approx(_COM_X)

    @pytest.mark.parametrize(
        "method,expected_x",
        [
            ("center_of_mass", _COM_X),
            ("bbox_center", _BBOX_X),
        ],
    )
    def test_method_moves_the_crop_center(self, method, expected_x):
        """The knob is load-bearing: it actually relocates the crop."""
        assert _crop_center_x(_head(centroid_method=method)) == pytest.approx(
            expected_x
        )

    def test_geometric_median_is_accepted_and_differs_from_bbox(self):
        """The Weiszfeld median resolves and is not the bbox midpoint here."""
        head = _head(centroid_method="geometric_median")
        dataset = _dataset(head)
        assert dataset.centroid_method == "geometric_median"
        assert _crop_center_x(head) != pytest.approx(_BBOX_X)

    def test_anchor_part_still_wins(self):
        """The pre-existing `anchor_part` path is unchanged."""
        head = _head(anchor_part="a")
        dataset = _dataset(head)
        assert (dataset.centroid_method, dataset.centroid_fallback) == (
            "anchor",
            "center_of_mass",
        )
        assert _crop_center_x(head) == pytest.approx(_ANCHOR_X)

    def test_anchor_fallback_is_configurable(self):
        """`centroid_fallback` rides along with an anchor and is resolved."""
        dataset = _dataset(_head(anchor_part="a", centroid_fallback="bbox_center"))
        assert (dataset.centroid_method, dataset.centroid_fallback) == (
            "anchor",
            "bbox_center",
        )

    def test_unresolvable_anchor_degrades_to_the_fallback(self):
        """An `anchor_part` absent from the skeleton degrades, it does not raise.

        `EmbeddingDataset` resolves `anchor_ind` leniently (embedding labels often
        carry a partial skeleton), so the method must degrade in step with it --
        otherwise `generate_centroids` would be asked for an "anchor" with no index.
        """
        head = _head(anchor_part="nope", centroid_fallback="bbox_center")
        dataset = _dataset(head)
        assert dataset.anchor_ind is None
        assert (dataset.centroid_method, dataset.centroid_fallback) == (
            "bbox_center",
            None,
        )
        assert _crop_center_x(head) == pytest.approx(_BBOX_X)

    def test_contradictory_config_is_rejected_naming_the_head(self):
        """`anchor_part` + a non-anchor method fails at setup, not in a worker."""
        config = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "embedding": {
                            "embedding": {
                                "anchor_part": "a",
                                "centroid_method": "bbox_center",
                            }
                        }
                    }
                }
            }
        )
        with pytest.raises(ValueError, match=r"head_configs\.embedding\.embedding"):
            check_centroid_methods(config)

    def test_unknown_method_is_rejected(self):
        config = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "embedding": {"embedding": {"centroid_method": "centre"}}
                    }
                }
            }
        )
        with pytest.raises(ValueError, match="Unknown centroid_method"):
            check_centroid_methods(config)
