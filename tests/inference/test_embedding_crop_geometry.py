"""Crop geometry on the composed centroid + `embedding` inference path (#586).

`_build_topdown_embedding` builds the `CentroidCrop` that feeds the embedder. Its
crop geometry has to reproduce what the EMBEDDER was trained with -- the vectors
are only comparable to the training distribution if the crops are centered the
same way -- so both the anchor and the centroid method come off the embedding
model's own saved head config, not the centroid model's and not only the CLI.
"""

from __future__ import annotations

import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.inference import loaders as loaders_mod
from sleap_nn.inference.loaders import _build_topdown_embedding

_SKEL = sio.Skeleton(nodes=["head", "thorax", "abdomen"], name="fly")


def _emb_config(**head_overrides):
    head = {"embedding_dim": 8, "output_stride": 16}
    head.update(head_overrides)
    return OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "scale": 1.0,
                    "crop_size": 32,
                    "ensure_rgb": False,
                    "ensure_grayscale": True,
                },
                "skeletons": [
                    {
                        "nodes": [{"name": n} for n in _SKEL.node_names],
                        "edges": [],
                        "symmetries": [],
                        "name": _SKEL.name,
                    }
                ],
            },
            "model_config": {
                "backbone_config": {"unet": {"max_stride": 16}},
                "head_configs": {"embedding": {"embedding": head}},
            },
        }
    )


@pytest.fixture
def stub_embedding_ckpt(monkeypatch):
    """Return a factory that stubs the checkpoint load with an in-memory config."""

    def _install(emb_config):
        def fake_load(module_cls, ckpt_path, *, model_type, **kwargs):
            assert model_type == "embedding"
            return torch.nn.Identity(), emb_config, "unet"

        monkeypatch.setattr(loaders_mod, "_load_lightning_module", fake_load)

    return _install


def _build(emb_config, stub, anchor_part=None):
    stub(emb_config)
    assets = _build_topdown_embedding(
        None,  # no centroid model -> the GT-centroid crop path
        "unused/ckpt",
        device="cpu",
        backbone_ckpt_path=None,
        head_ckpt_path=None,
        peak_threshold=0.2,
        integral_refinement="integral",
        integral_patch_size=5,
        max_instances=None,
        return_confmaps=False,
        preprocess_config=OmegaConf.create(
            {"scale": None, "crop_size": None, "ensure_rgb": None, "max_height": None}
        ),
        anchor_part=anchor_part,
    )
    return assets.inference_model.centroid_crop


class TestEmbeddingCropGeometry:
    def test_default_is_center_of_mass(self, stub_embedding_ckpt):
        crop = _build(_emb_config(), stub_embedding_ckpt)
        assert crop.centroid_method == "center_of_mass"
        assert crop.anchor_ind is None

    def test_method_comes_from_the_embedding_head(self, stub_embedding_ckpt):
        crop = _build(
            _emb_config(centroid_method="bbox_center"),
            stub_embedding_ckpt,
        )
        assert crop.centroid_method == "bbox_center"

    def test_fallback_comes_from_the_embedding_head(self, stub_embedding_ckpt):
        crop = _build(
            _emb_config(anchor_part="thorax", centroid_fallback="bbox_center"),
            stub_embedding_ckpt,
        )
        assert crop.centroid_method == "anchor"
        assert crop.centroid_fallback == "bbox_center"

    def test_trained_anchor_is_honored_without_a_cli_override(
        self, stub_embedding_ckpt
    ):
        """The gap this closes: `EmbeddingDataset` centers training crops on the
        head config's `anchor_part`, so inference must too -- otherwise a model
        trained on thorax-centered crops is fed mean-of-visible ones."""
        crop = _build(_emb_config(anchor_part="thorax"), stub_embedding_ckpt)
        assert crop.anchor_ind == _SKEL.node_names.index("thorax")
        assert crop.centroid_method == "anchor"

    def test_cli_anchor_overrides_the_trained_one(self, stub_embedding_ckpt):
        crop = _build(
            _emb_config(anchor_part="thorax"),
            stub_embedding_ckpt,
            anchor_part="abdomen",
        )
        assert crop.anchor_ind == _SKEL.node_names.index("abdomen")
        assert crop.centroid_method == "anchor"

    def test_trained_anchor_absent_from_the_skeleton_degrades(
        self, stub_embedding_ckpt
    ):
        """Embedding labels routinely carry a partial skeleton; degrade, don't raise."""
        crop = _build(
            _emb_config(anchor_part="wing", centroid_fallback="bbox_center"),
            stub_embedding_ckpt,
        )
        assert crop.anchor_ind is None
        assert crop.centroid_method == "bbox_center"

    def test_cli_anchor_not_in_the_skeleton_still_raises(self, stub_embedding_ckpt):
        """An explicit override names a node by hand: a typo must not pass silently."""
        with pytest.raises(ValueError):
            _build(_emb_config(), stub_embedding_ckpt, anchor_part="wing")


# ── The composed centroid + embedding layer at its entry point (F10) ─────────
#
# `Predictor.from_model_paths([centroid_dir, embedding_dir])` builds a
# `TopDownEmbeddingLayer` (API-only: the CLI's fused route embeds through
# `embed_labels` instead). These drive it the way an API caller does.


def _embedding_dir(root, **preprocessing):
    """A tiny random-weights embedding model dir, with preprocessing overrides."""
    from tests.fixtures.model_ckpts import (
        _build_embedding_training_config,
        _write_embedding_model_dir,
    )

    config = _build_embedding_training_config()
    for key, value in preprocessing.items():
        config.data_config.preprocessing[key] = value
    torch.manual_seed(0)
    return _write_embedding_model_dir(root / "embedding_model", config)


def _frame(labels_path):
    """The first frame of ``labels_path`` as a ``(1, C, H, W)`` float tensor."""
    image = sio.load_slp(str(labels_path))[0].image
    return torch.from_numpy(image).permute(2, 0, 1)[None].float()


class TestComposedEmbeddingLayer:
    def test_crops_are_sized_like_the_embedder_was_trained(
        self, minimal_instance_centroid_ckpt, minimal_instance, tmp_path
    ):
        """`EmbeddingDataset` size-matches each frame to the EMBEDDER's saved
        max_height/max_width before cropping; the composed layer re-applied the
        CENTROID model's (384 here vs the embedder's 64), so it embedded crops at
        6x the scale the model was trained on."""
        from sleap_nn.data.instance_cropping import make_centered_bboxes
        from sleap_nn.data.resizing import apply_sizematcher
        from sleap_nn.inference.ops.crops import crop_bboxes
        from sleap_nn.inference.predictor import Predictor

        emb_dir = _embedding_dir(tmp_path, max_height=64, max_width=64)
        layer = Predictor.from_model_paths(
            [str(minimal_instance_centroid_ckpt), str(emb_dir)], device="cpu"
        ).layer
        image = _frame(minimal_instance)
        out = layer.predict(image)

        valid = out.instance_valid[0]
        centroids = out.pred_centroids[0][valid]  # original-image space
        assert len(centroids) > 0, "the centroid model found nothing to embed"

        # The embedder's training geometry: frame sized to ITS max_hw, crop
        # `crop_size` around the centroid in that sized space.
        sized, ratio = apply_sizematcher(image[0], 64, 64)
        crops = crop_bboxes(
            sized[None],
            make_centered_bboxes(centroids * ratio, 32, 32),
            torch.zeros(len(centroids), dtype=torch.long),
        )
        expected = layer.centered_instance_layer.predict(crops).pred_embeddings[:, 0]

        torch.testing.assert_close(out.pred_embeddings[0][valid], expected)

    def test_burn_in_embedder_warns(
        self, minimal_instance_centroid_ckpt, minimal_instance, tmp_path
    ):
        """This path never has masks, so a burn-in model runs off distribution;
        `embed_labels` warns about that case, this layer was silent."""
        from loguru import logger

        from sleap_nn.inference.predictor import Predictor

        messages = []
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            for burn_in in (False, True):
                emb_dir = _embedding_dir(tmp_path / str(burn_in), burn_in=burn_in)
                Predictor.from_model_paths(
                    [str(minimal_instance_centroid_ckpt), str(emb_dir)], device="cpu"
                ).layer.predict(_frame(minimal_instance))
                text = " ".join(" ".join(str(m) for m in messages).split())
                assert ("trained with mask burn-in" in text) is burn_in, text
        finally:
            logger.remove(handler_id)

    def test_filtered_detections_lose_their_vector_and_valid_flag(
        self, minimal_instance_centroid_ckpt, minimal_instance, tmp_path
    ):
        """A filter NaN'd the dropped slot's score and centroid but left its
        `pred_embeddings` row and `instance_valid=True` -- the fields a consumer
        of embedding outputs enumerates by -- so the detection survived."""
        from sleap_nn.inference.filters import FilterConfig
        from sleap_nn.inference.predictor import Predictor

        emb_dir = _embedding_dir(tmp_path)
        paths = [str(minimal_instance_centroid_ckpt), str(emb_dir)]
        video = sio.load_slp(str(minimal_instance)).video

        kept = Predictor.from_model_paths(paths, device="cpu").predict(
            video, make_labels=False
        )[0]
        n_kept = int(kept.instance_valid.sum())
        assert n_kept > 0, "nothing to filter"
        assert torch.isfinite(kept.pred_embeddings[kept.instance_valid]).all()

        # A score gate no centroid passes drops every detection.
        gate = float(kept.instance_scores[kept.instance_valid].max()) + 1.0
        dropped = Predictor.from_model_paths(
            paths, device="cpu", filter_config=FilterConfig(min_instance_score=gate)
        ).predict(video, make_labels=False)[0]
        assert torch.isnan(dropped.instance_scores).all()
        assert not dropped.instance_valid.any()
        assert torch.isnan(dropped.pred_embeddings).all()
