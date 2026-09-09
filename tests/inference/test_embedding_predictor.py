"""What an `embedding` predictor may and may not be asked for.

An embedding model emits appearance vectors, which have no `sio.Labels`
representation — so `make_labels=True` / `predict_to_file` used to bypass the
skeleton guard, find nothing to package, and hand back an EMPTY `Labels`,
indistinguishable from a model that predicted nothing.

Also pins the train/inference crop parity: `EmbeddingDataset` sizematches and
scales the frame BEFORE cropping, and the inference path dropped both, so it
embedded crops at a different animal scale than training produced.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.inference.predictor import Predictor


@pytest.fixture
def embedding_predictor(minimal_embedding_model_dir):
    return Predictor.from_model_paths([str(minimal_embedding_model_dir)], device="cpu")


def test_make_labels_is_refused(embedding_predictor, minimal_instance):
    """`make_labels=True` must say why, not return an empty Labels."""
    with pytest.raises(ValueError, match="not supported for an `embedding`"):
        embedding_predictor.predict(sio.load_slp(minimal_instance), make_labels=True)


def test_predict_to_file_is_refused(embedding_predictor, minimal_instance, tmp_path):
    """Same for the streaming-to-.slp entry point."""
    with pytest.raises(ValueError, match="not supported for an `embedding`"):
        embedding_predictor.predict_to_file(
            sio.load_slp(minimal_instance),
            (tmp_path / "out.slp").as_posix(),
        )


def test_refusal_points_at_the_supported_route(embedding_predictor, minimal_instance):
    """The message names the .h5 writer and the CLI flag."""
    with pytest.raises(ValueError) as excinfo:
        embedding_predictor.predict(sio.load_slp(minimal_instance), make_labels=True)

    message = str(excinfo.value)
    assert "predict_embeddings_to_h5" in message
    assert "--embeddings_path" in message


def test_pose_predictor_still_requires_a_skeleton(minimal_instance):
    """The skeleton guard the embedding branch sits next to is unchanged."""
    from sleap_nn.inference.layers.centroid import CentroidLayer  # noqa: F401

    predictor = Predictor.from_model_paths(
        ["tests/assets/model_ckpts/minimal_instance_centroid"], device="cpu"
    )
    predictor.skeleton = None
    with pytest.raises(ValueError, match="requires a skeleton"):
        predictor.predict(sio.load_slp(minimal_instance), make_labels=True)


# ─────────────────────────────────────────────────────────────────────────
# Train/inference crop parity
# ─────────────────────────────────────────────────────────────────────────
def test_inference_dataset_inherits_the_trained_crop_geometry(
    minimal_embedding_model_dir, monkeypatch, tmp_path
):
    """`max_hw` and `scale` must come off the saved training config.

    The fixture config sets max_height/max_width = 64; passing neither left the
    inference dataset at `(None, None)` with `scale=1.0`, so a frame that
    training would have sizematched to 64 px was cropped at full resolution.
    """
    from omegaconf import OmegaConf

    from sleap_nn.data import custom_datasets
    from sleap_nn.inference import embedding as embedding_module

    cfg = OmegaConf.load(minimal_embedding_model_dir / "training_config.yaml")
    assert cfg.data_config.preprocessing.max_height == 64
    cfg.data_config.preprocessing.scale = 0.5
    OmegaConf.save(cfg, minimal_embedding_model_dir / "training_config.yaml")

    captured = {}
    real_init = custom_datasets.EmbeddingDataset.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing the constructor arguments")

    monkeypatch.setattr(custom_datasets.EmbeddingDataset, "__init__", spy_init)

    # A tracked-mask .slp so the writer gets past its own checks before building
    # the dataset.
    video = sio.Video.from_filename("emb_parity.mp4")
    mask_arr = np.zeros((32, 32), dtype=bool)
    mask_arr[4:12, 4:12] = True
    mask = sio.UserSegmentationMask.from_numpy(mask_arr)
    mask.track = sio.Track("t0")
    mask.identity = sio.Identity(name="a0")
    labels = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0, instances=[], masks=[mask])
        ],
        videos=[video],
        skeletons=[],
        tracks=[mask.track],
    )
    slp_path = tmp_path / "tracked.slp"
    sio.save_slp(labels, slp_path.as_posix())

    with pytest.raises(RuntimeError, match="stop after capturing"):
        embedding_module.predict_embeddings_to_h5(
            model_paths=[str(minimal_embedding_model_dir)],
            data_path=slp_path.as_posix(),
            output_path=(tmp_path / "out.h5").as_posix(),
            device="cpu",
        )

    assert captured["max_hw"] == (64, 64)
    assert captured["scale"] == 0.5
