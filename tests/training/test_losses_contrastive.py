"""Tests for the embedding contrastive losses + mask builder.

Covers the NaN-prevention guards (empty positive/negative rows) and the
``build_contrastive_masks`` correctness contract (scope selection + the
``restrict_same_video`` "unknown cross-video pair" rule).
"""

import torch

from sleap_nn.training.losses import (
    build_contrastive_masks,
    get_contrastive_loss,
    infonce_loss,
    supcon_loss,
    triplet_loss,
)


def _normalized(b, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, d, generator=g)
    return torch.nn.functional.normalize(z, dim=1)


class TestEmptyRowGuards:
    """A batch item with no positives must not produce NaN/inf."""

    def test_supcon_no_positive_row_is_finite(self):
        z = _normalized(4)
        # No positives anywhere (e.g. all-singleton identities).
        pos = torch.zeros(4, 4, dtype=torch.bool)
        neg = ~torch.eye(4, dtype=torch.bool)
        loss = supcon_loss(z, pos, neg, temperature=0.1)
        assert torch.isfinite(loss).all()

    def test_infonce_no_positive_row_is_finite(self):
        z = _normalized(4)
        pos = torch.zeros(4, 4, dtype=torch.bool)
        neg = ~torch.eye(4, dtype=torch.bool)
        loss = infonce_loss(z, pos, neg, temperature=0.1)
        assert torch.isfinite(loss).all()

    def test_triplet_no_valid_row_is_finite(self):
        z = _normalized(4)
        pos = torch.zeros(4, 4, dtype=torch.bool)
        neg = ~torch.eye(4, dtype=torch.bool)
        loss = triplet_loss(z, pos, neg, margin=0.2)
        assert torch.isfinite(loss).all()

    def test_partial_positives_finite_and_nonzero(self):
        # Two pairs of identities: rows 0-1 positive, rows 2-3 positive.
        z = _normalized(4)
        pos = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool
        )
        neg = (~pos) & (~torch.eye(4, dtype=torch.bool))
        for name in ("supcon", "infonce", "triplet"):
            loss = get_contrastive_loss(name)(z, pos, neg)
            assert torch.isfinite(loss).all()


class TestBuildContrastiveMasks:
    def test_restrict_same_video_puts_cross_video_pair_in_neither_mask(self):
        # Two crops, DIFFERENT groups, DIFFERENT videos. With restrict_same_video, a
        # cross-video pair is "unknown" (could be the same animal) -> excluded from
        # negatives; different groups -> not a positive. So it lands in NEITHER mask.
        item = torch.tensor([0, 1])
        video = torch.tensor([0, 1])
        frame = torch.tensor([0, 0])
        group = torch.tensor([5, 9])  # different identities
        pos, neg = build_contrastive_masks(
            item,
            video,
            frame,
            group,
            positives_scope="global_id",
            restrict_same_video=True,
        )
        assert not bool(pos[0, 1]) and not bool(pos[1, 0])
        assert not bool(neg[0, 1]) and not bool(neg[1, 0])

    def test_global_id_cross_video_same_identity_is_positive(self):
        # global_id links identities ACROSS videos -> a cross-video same-id pair is a
        # positive even under restrict_same_video.
        item = torch.tensor([0, 1])
        video = torch.tensor([0, 1])
        frame = torch.tensor([0, 0])
        group = torch.tensor([5, 5])
        pos, _ = build_contrastive_masks(
            item,
            video,
            frame,
            group,
            positives_scope="global_id",
            restrict_same_video=True,
        )
        assert bool(pos[0, 1]) and bool(pos[1, 0])

    def test_global_id_groups_same_identity_as_positive(self):
        item = torch.tensor([0, 1, 2])
        video = torch.tensor([0, 0, 0])
        frame = torch.tensor([0, 1, 2])
        group = torch.tensor([5, 5, 9])  # 0 and 1 share identity
        pos, _ = build_contrastive_masks(
            item, video, frame, group, positives_scope="global_id"
        )
        assert bool(pos[0, 1]) and bool(pos[1, 0])
        assert not bool(pos[0, 2])
        # No self positives.
        assert not bool(pos[0, 0])

    def test_aug_view_positive_is_only_the_shared_item(self):
        # aug_view: positives are same ORIGINAL crop (shared item_id), NOT same group.
        item = torch.tensor([7, 7, 8])  # two views of item 7
        video = torch.tensor([0, 0, 0])
        frame = torch.tensor([0, 0, 1])
        group = torch.tensor([0, 0, 0])  # group is irrelevant for aug_view
        pos, _ = build_contrastive_masks(
            item, video, frame, group, positives_scope="aug_view"
        )
        assert bool(pos[0, 1]) and bool(pos[1, 0])  # the two views
        assert not bool(pos[0, 2]) and not bool(pos[2, 0])  # different item
