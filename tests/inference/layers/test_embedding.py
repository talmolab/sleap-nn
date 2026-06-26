"""Tests for the embedding (re-ID) inference layers.

Covers the single-stage :class:`EmbeddingLayer` (precropped / mask-driven) and the
composed :class:`TopDownEmbeddingLayer` (centroid -> crop -> embed). The composed
layer is exercised at the ``_run_stage_2`` level with synthetic centroids so the
test is portable (no trained centroid model needed) and deterministic: it asserts
that per-instance embeddings ride on the supplied centroids, that empty slots are
NaN-padded, and that ``instance_valid`` enumerates the populated detections.
"""

import torch
from omegaconf import OmegaConf

from sleap_nn.inference.layers.backends.torch_backend import TorchBackend
from sleap_nn.inference.layers.centroid import CentroidLayer
from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
from sleap_nn.inference.layers.embedding import EmbeddingLayer, TopDownEmbeddingLayer
from sleap_nn.training.lightning_modules import EmbeddingLightningModule

_DIM = 16
_MAX_STRIDE = 16


def _make_embedding_layer():
    """Build a tiny (random-weights) EmbeddingLayer on CPU for layer tests."""
    backbone = OmegaConf.create(
        {
            "unet": {
                "in_channels": 1,
                "kernel_size": 3,
                "filters": 8,
                "filters_rate": 1.5,
                "max_stride": _MAX_STRIDE,
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": True,
                "stacks": 1,
                "convs_per_block": 2,
                "output_stride": 2,
            }
        }
    )
    objective = {
        "positives": {"scope": "global_id", "aug_views": 2},
        "negatives": {
            "sources": ["in_batch"],
            "exclude_same_track": True,
            "restrict_same_video": False,
        },
        "loss": {"name": "supcon", "temperature": 0.1, "margin": 0.2},
        "sampler": {"kind": "pk", "groups_per_batch": 2, "samples_per_group": 4},
        "use_projection": True,
        "projection_dim": _DIM,
    }
    heads = OmegaConf.create(
        {
            "embedding": {
                "embedding": {
                    "embedding_dim": _DIM,
                    "num_fc_layers": 1,
                    "num_fc_units": 32,
                    "pool": "gem",
                    "normalize": True,
                    "output_stride": _MAX_STRIDE,
                    "loss_weight": 1.0,
                    "freeze_backbone": False,
                    "objective": objective,
                }
            }
        }
    )
    module = EmbeddingLightningModule(
        model_type="embedding",
        backbone_type="unet",
        backbone_config=backbone,
        head_configs=heads,
        init_weights="xavier",
    ).eval()
    return EmbeddingLayer(
        backend=TorchBackend(model=module.model, device="cpu"),
        embedding_module=module,
        embedding_dim=_DIM,
        output_stride=_MAX_STRIDE,
        max_stride=_MAX_STRIDE,
        preprocess_config=PreprocessConfig(scale=1.0),
        postprocess_config=PostprocessConfig(),
    )


def _make_topdown_embedding_layer(emb_layer, crop_size=(64, 64)):
    """Compose ``emb_layer`` with a GT-centroid stage-1 into a TopDownEmbeddingLayer.

    Stage 1 is irrelevant for the ``_run_stage_2`` tests (we feed centroids
    directly); a GT-centroid CentroidLayer reusing the embedder backend is the
    cheapest valid stage-1 object.
    """
    centroid_layer = CentroidLayer(
        backend=emb_layer.backend,
        output_stride=1,
        max_stride=1,
        use_gt_centroids=True,
        preprocess_config=PreprocessConfig(scale=1.0),
        postprocess_config=PostprocessConfig(),
    )
    return TopDownEmbeddingLayer(
        centroid_layer=centroid_layer,
        centered_instance_layer=emb_layer,
        crop_size=crop_size,
    )


def test_embedding_layer_single_stage_shapes():
    """Single-stage EmbeddingLayer: each crop -> one L2-normalized vector."""
    layer = _make_embedding_layer()
    crops = torch.rand(3, 1, 64, 64)
    out = layer.predict(crops)
    assert out.pred_embeddings.shape == (3, 1, _DIM)
    # L2-normalized (the head's normalize=True).
    norms = out.pred_embeddings.squeeze(1).norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-3)
    # Each crop enumerates as exactly one valid detection.
    assert out.instance_valid.shape == (3, 1)
    assert bool(out.instance_valid.all())


def test_topdown_embedding_run_stage2_rides_on_centroids():
    """Composed stage 2: embeddings scatter onto the supplied valid centroids."""
    emb_layer = _make_embedding_layer()
    td = _make_topdown_embedding_layer(emb_layer)

    image = torch.rand(1, 1, 128, 128)
    centroids = torch.tensor([[[40.0, 50.0], [90.0, 70.0]]])  # (B=1, I=2, 2)
    centroid_vals = torch.tensor([[0.9, 0.8]])
    valid = torch.tensor([[True, False]])  # second slot is padding

    out = td._run_stage_2(
        image, centroids, centroid_vals, valid, eff_scale=torch.tensor([1.0])
    )
    emb = out.pred_embeddings
    assert emb.shape == (1, 2, _DIM)
    # Valid slot 0: finite + L2-normalized.
    assert bool(torch.isfinite(emb[0, 0]).all())
    assert torch.allclose(emb[0, 0].norm(), torch.tensor(1.0), atol=1e-3)
    # Invalid slot 1: NaN-padded so to_instances / n_instances compact it away.
    assert bool(torch.isnan(emb[0, 1]).all())
    # instance_valid enumerates only the populated slot.
    assert out.instance_valid.tolist() == [[True, False]]
    # Embeddings ride on the real centroids (image-space, eff_scale=1).
    assert torch.allclose(out.pred_centroids, centroids)


def test_topdown_embedding_run_stage2_all_invalid():
    """No valid centroids -> all-NaN embeddings + all-False instance_valid (no crash)."""
    emb_layer = _make_embedding_layer()
    td = _make_topdown_embedding_layer(emb_layer)

    image = torch.rand(2, 1, 96, 96)
    centroids = torch.zeros(2, 3, 2)
    centroid_vals = torch.zeros(2, 3)
    valid = torch.zeros(2, 3, dtype=torch.bool)

    out = td._run_stage_2(
        image, centroids, centroid_vals, valid, eff_scale=torch.ones(2)
    )
    assert out.pred_embeddings.shape == (2, 3, _DIM)
    assert bool(torch.isnan(out.pred_embeddings).all())
    assert not bool(out.instance_valid.any())


def test_topdown_embedding_run_stage2_eff_scale():
    """``eff_scale`` lifts the (sized-space) centroids back to image space."""
    emb_layer = _make_embedding_layer()
    td = _make_topdown_embedding_layer(emb_layer)

    image = torch.rand(1, 1, 128, 128)
    centroids = torch.tensor([[[40.0, 50.0]]])  # sized space
    valid = torch.tensor([[True]])
    out = td._run_stage_2(
        image, centroids, torch.tensor([[1.0]]), valid, eff_scale=torch.tensor([2.0])
    )
    # image-space centroid = sized / eff_scale.
    assert torch.allclose(out.pred_centroids, centroids / 2.0)
    assert bool(torch.isfinite(out.pred_embeddings[0, 0]).all())
