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
