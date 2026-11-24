"""``upload_batches``: the one option that copies frames to the GPU before stage 1.

One mechanism, on the ``Predictor``: ``Predictor._upload_images`` uploads what the
provider (or ``PoseStream.push``) hands over, and the layer and a streamed part stage
both read the object it returns. ``frames_on_gpu`` is the option's former name for
top-down pipelines and stays as a deprecated alias on ``predict``, ``open_stream``,
``Predictor.from_model_paths`` and the CLI.

Everything here runs without a GPU: ``"meta"`` stands in for the CUDA device where a
tensor has to move. ``tests/inference/test_cuda.py`` does the real copies.
"""

import contextlib
import inspect
import json
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sleap_io as sio
import torch
from click.testing import CliRunner
from loguru import logger

from sleap_nn.inference.keypoint_crops import StreamingPartStage
from sleap_nn.inference.layers.backends import TorchBackend
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor, _torch_backends
from sleap_nn.inference.providers import NumpyProvider
from sleap_nn.inference.run import _build_predictor, open_stream, predict

RULE = {"center": ["A"], "reference": [["A"], ["B"]], "scale": 1.0, "min_half_side": 4}
NO_EFFECT = "upload_batches=True has no effect"


@contextlib.contextmanager
def _warnings():
    """The warnings logged inside the block, as a list of messages."""
    messages = []
    sink = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _n_alias(messages):
    """How many of ``messages`` are the alias's deprecation warning."""
    return sum("is a deprecated alias" in m and "frames_on_gpu" in m for m in messages)


class _Layer:
    """Remembers the frames ``predict`` is handed; one two-node instance per frame."""

    def __init__(self):
        self.seen = []

    def predict(self, image, **kwargs):
        self.seen.append(image)
        b = image.shape[0]
        return Outputs(
            pred_keypoints=torch.zeros(b, 1, 2, 2),
            pred_peak_values=torch.ones(b, 1, 2),
        )


class _Part:
    """A part stage that remembers the frames it is handed."""

    def __init__(self):
        self.seen = []

    def reset(self):
        pass

    def __call__(self, images, outputs, video_indices=None):
        self.seen.append(images)
        return outputs


def _uploading(**kwargs):
    """A predictor whose upload goes to the stand-in device (no GPU needed)."""
    kwargs.setdefault("layer", _Layer())
    predictor = Predictor(batch_size=2, device="cuda", upload_batches=True, **kwargs)
    assert predictor._upload_device == "cuda"
    predictor._upload_device = "meta"
    return predictor


# ─────────────────────────────────────────────────────────────────────────
# Off by default; one mechanism
# ─────────────────────────────────────────────────────────────────────────


def test_default_off():
    for fn in (predict, open_stream, Predictor.from_model_paths):
        params = inspect.signature(fn).parameters
        assert params["upload_batches"].default is False
        assert params["frames_on_gpu"].default is False
    predictor = Predictor(layer=_Layer())
    assert predictor.upload_batches is False and predictor._upload_device is None


def test_the_predictor_is_the_only_place_that_uploads():
    """No switch on the top-down layer, none on the predictor builder."""
    assert not hasattr(TopDownLayer, "frames_on_gpu")
    assert not hasattr(TopDownLayer, "_frames_device")
    assert "frames_on_gpu" not in inspect.signature(_build_predictor).parameters
    assert "upload_batches" in inspect.signature(_build_predictor).parameters


# ─────────────────────────────────────────────────────────────────────────
# Predictor._upload_images
# ─────────────────────────────────────────────────────────────────────────


def test_upload_images_moves_arrays_and_tensors():
    """Arrays and tensors go to the upload device; dtype and layout are kept."""
    predictor = _uploading()
    array = np.arange(2 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8, 1)
    channels_first = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    for images in (array, torch.from_numpy(array), channels_first):
        uploaded = predictor._upload_images(images)
        assert isinstance(uploaded, torch.Tensor)
        assert uploaded.device.type == "meta"
        assert uploaded.dtype == torch.uint8
        assert tuple(uploaded.shape) == tuple(images.shape)
        # A tensor already on the device is used where it is.
        assert predictor._upload_images(uploaded) is uploaded


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"device": "cuda"},
        {"device": "cuda", "upload_batches": False},
        {"device": "cpu", "upload_batches": True},
        {"upload_batches": True},
    ],
)
def test_upload_images_leaves_frames_alone_when_off_or_without_cuda(kwargs):
    predictor = Predictor(layer=_Layer(), **kwargs)
    array = np.zeros((2, 8, 8, 1), dtype=np.uint8)
    for images in (array, torch.from_numpy(array)):
        assert predictor._upload_images(images) is images


def test_other_inputs_pass_through():
    """Only arrays and tensors are uploaded; anything else is the layer's business."""
    frames = [np.zeros((8, 8, 1), dtype=np.uint8)]
    assert _uploading()._upload_images(frames) is frames


# ─────────────────────────────────────────────────────────────────────────
# Which device: the first model on CUDA, else none
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "backend_devices, predictor_device, expected",
    [
        (["cuda:1", "cuda:0"], "cuda", "cuda:1"),  # the first model's GPU
        (["cuda"], "cpu", "cuda"),  # the layer's device, not Predictor.device
        (["cpu", "cuda:1"], "cuda", "cuda:1"),  # stage 1 on the CPU, stage 2 on a GPU
        (["cpu", "cpu"], "cuda", None),  # no model on CUDA
        ([], "cuda:0", "cuda:0"),  # no PyTorch backend: Predictor.device
        ([], "cpu", None),
        ([], None, None),
    ],
)
def test_upload_device(monkeypatch, backend_devices, predictor_device, expected):
    import types

    import sleap_nn.inference.predictor as predictor_mod

    backends = [types.SimpleNamespace(device=d) for d in backend_devices]
    monkeypatch.setattr(predictor_mod, "_torch_backends", lambda layer: backends)
    with _warnings() as messages:
        predictor = Predictor(
            layer=_Layer(), device=predictor_device, upload_batches=True
        )
    assert predictor._upload_device == expected
    assert sum(NO_EFFECT in m for m in messages) == (expected is None)


def test_every_stage_of_a_checkpoint_pair_reports_the_build_device(
    minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt
):
    """Both stages of a top-down pair sit in a ``TorchBackend`` on ``device``.

    So on a CUDA build the upload device is that CUDA device (stage 1's backend
    comes first). A crop model run alone shares its backend with the GT stage 1.
    """
    pair = Predictor.from_model_paths(
        [
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
        ],
        device="cpu",
    )
    backends = _torch_backends(pair.layer)
    assert backends[0] is pair.layer.centroid_layer.backend
    assert backends[1] is pair.layer.centered_instance_layer.backend
    assert [b.device for b in backends] == ["cpu", "cpu"]

    alone = Predictor.from_model_paths(
        [str(minimal_instance_centered_instance_ckpt)], device="cpu"
    )
    assert _torch_backends(alone.layer) == [alone.layer.centered_instance_layer.backend]


def _network_only_export(ckpt_dir, tmp_path, monkeypatch):
    """A network-only export folder of ``ckpt_dir`` with a stand-in graph.

    The folder holds what the loader reads (the training config and the export
    metadata); the graph itself is replaced by a function, so no ONNX runtime and
    no GPU is needed to build the pipeline around it.
    """
    import sleap_nn.inference.predictor as predictor_mod

    export_dir = tmp_path / "network_only"
    export_dir.mkdir()
    shutil.copy(ckpt_dir / "training_config.yaml", export_dir / "training_config.yaml")
    (export_dir / "export_metadata.json").write_text(
        json.dumps({"model_type": "centered_instance", "network_only": True})
    )
    (export_dir / "model.onnx").write_bytes(b"")

    def graph(x):
        return {"CenteredInstanceConfmapsHead": torch.zeros(x.shape[0], 2, 8, 8)}

    monkeypatch.setattr(
        predictor_mod, "_build_export_backend", lambda runtime, path, device: graph
    )
    return export_dir


def test_network_only_exports_upload_to_the_build_device(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
    monkeypatch,
):
    """Network-only export folders given as model paths upload like checkpoints.

    An exported network stands where the checkpoint's module stands, inside a
    ``TorchBackend`` on the build device: alone (built for ``"cuda"`` here, which
    loads nothing onto a GPU) and next to a checkpoint stage 1.
    """
    from sleap_nn.inference.exported_network import ExportedNetwork

    export_dir = _network_only_export(
        minimal_instance_centered_instance_ckpt, tmp_path, monkeypatch
    )
    alone = Predictor.from_model_paths(
        [str(export_dir)], device="cuda", export_runtime="onnx", upload_batches=True
    )
    (backend,) = _torch_backends(alone.layer)
    assert isinstance(backend, TorchBackend)
    assert isinstance(backend.model, ExportedNetwork)
    assert backend.device == "cuda" and alone._upload_device == "cuda"

    mixed = Predictor.from_model_paths(
        [str(minimal_instance_centroid_ckpt), str(export_dir)],
        device="cpu",
        export_runtime="onnx",
    )
    backends = _torch_backends(mixed.layer)
    assert [type(b.model).__name__ for b in backends] == [
        "CentroidLightningModule",
        "ExportedNetwork",
    ]
    assert [b.device for b in backends] == ["cpu", "cpu"]


# ─────────────────────────────────────────────────────────────────────────
# One upload per batch; the part stage reads the layer's copy
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("upload", [False, True])
def test_part_stage_gets_the_object_the_layer_got(upload):
    layer, part = _Layer(), _Part()
    if upload:
        predictor = _uploading(layer=layer, part_stage=part)
    else:
        predictor = Predictor(layer=layer, batch_size=2, part_stage=part)
    images = np.zeros((4, 8, 8, 1), dtype=np.uint8)
    provider = NumpyProvider(images=images, batch_size=2)
    batches = []
    real_batch_images = Predictor._batch_images

    def spy(self, batch):
        batches.append(batch.images)
        return real_batch_images(self, batch)

    with patch.object(Predictor, "_batch_images", spy):
        predictor.predict(provider, make_labels=False)

    assert len(layer.seen) == len(part.seen) == len(batches) == 2
    for from_provider, for_layer, for_part in zip(batches, layer.seen, part.seen):
        assert for_part is for_layer
        if upload:
            assert for_layer.device.type == "meta"
            assert for_layer.dtype == torch.uint8 and for_layer.shape == (2, 8, 8, 1)
        else:
            assert for_layer is from_provider  # the provider's own array


# ─────────────────────────────────────────────────────────────────────────
# PoseStream: every frame goes through the predictor's upload
# ─────────────────────────────────────────────────────────────────────────


def _frame(minimal_instance):
    img = np.asarray(sio.load_slp(str(minimal_instance)).labeled_frames[0].image)
    return img[..., 0] if img.ndim == 3 else img


@pytest.mark.parametrize("pipeline", ["single_instance", "top_down", "top_down+part"])
def test_stream_uploads_each_frame_through_the_predictor(
    pipeline,
    minimal_instance,
    minimal_instance_single_instance_ckpt,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
):
    """The warm-up and every push hand the networks what ``_upload_images`` returns.

    The upload is stood in for by a copy of the frame, so the test can tell the
    uploaded object from the pushed one: each push uploads once, and the pose
    layer and the part stage both get that object.
    """
    from sleap_nn.inference.layers.single_instance import SingleInstanceLayer

    uploads, for_layers, for_part = [], [], []

    def upload(self, images):
        uploads.append(torch.as_tensor(images).clone())
        return uploads[-1]

    def spy_on(cls, name, seen):
        real = getattr(cls, name)

        def spy(self, images, *args, **kwargs):
            seen.append(images)
            return real(self, images, *args, **kwargs)

        monkeypatch.setattr(cls, name, spy)

    monkeypatch.setattr(Predictor, "_upload_images", upload)
    spy_on(StreamingPartStage, "__call__", for_part)
    if pipeline == "single_instance":
        spy_on(SingleInstanceLayer, "predict", for_layers)
        kw = dict(model_paths=[str(minimal_instance_single_instance_ckpt)])
    else:
        spy_on(TopDownLayer, "predict", for_layers)  # the pose and the part layer
        kw = dict(
            model_paths=[
                str(minimal_instance_centroid_ckpt),
                str(minimal_instance_centered_instance_ckpt),
            ],
            centroid_threshold=0.05,
            max_instances=2,
        )
    if pipeline == "top_down+part":
        kw.update(
            part_model_paths=[str(minimal_instance_centered_instance_ckpt)],
            part_crop_rule=RULE,
            part_peak_threshold=0.0,
            part_prefix="part/",
        )

    def uploaded(image):
        return any(image is u for u in uploads)

    frame = _frame(minimal_instance)
    with open_stream(
        device="cpu", peak_threshold=0.0, frame_shape=frame.shape, **kw
    ) as stream:
        # Warm-up: two runs of the whole pipeline, then one frame for the crops.
        assert len(uploads) == 3
        assert for_layers and all(uploaded(image) for image in for_layers)
        assert all(uploaded(image) for image in for_part)
        for pushed in (frame, torch.from_numpy(frame)):
            n = len(uploads)
            for_layers.clear()
            for_part.clear()
            poses = stream.push(pushed)
            assert len(uploads) == n + 1  # once per push
            assert for_layers and all(image is uploads[-1] for image in for_layers)
            if pipeline == "top_down+part":
                assert len(for_part) == 1 and for_part[0] is uploads[-1]
                assert stream.node_names == ["A", "B", "part/A", "part/B"]
            else:
                assert for_part == []
            assert poses.points.shape[1:] == (len(stream.node_names), 2)


@pytest.mark.parametrize("name", [None, "upload_batches", "frames_on_gpu"])
def test_open_stream_reaches_the_predictor(name, minimal_instance_single_instance_ckpt):
    """``open_stream(upload_batches=True)`` (or the alias) builds an uploading predictor."""
    factory = Predictor.from_model_paths
    kwargs = {} if name is None else {name: True}
    with (
        _warnings() as messages,
        patch.object(Predictor, "from_model_paths", wraps=factory) as spy,
        open_stream(
            [str(minimal_instance_single_instance_ckpt)], device="cpu", **kwargs
        ) as stream,
    ):
        assert stream._predictor.upload_batches is (name is not None)
        assert stream._predictor._upload_device is None  # no CUDA model here
    assert spy.call_args.kwargs["upload_batches"] is (name is not None)
    assert "frames_on_gpu" not in spy.call_args.kwargs
    assert _n_alias(messages) == (name == "frames_on_gpu")
    assert sum(NO_EFFECT in m for m in messages) == (name is not None)


# ─────────────────────────────────────────────────────────────────────────
# frames_on_gpu: a deprecated alias on every entry point that took it
# ─────────────────────────────────────────────────────────────────────────


def test_alias_on_from_model_paths(minimal_instance_single_instance_ckpt):
    """The alias turns ``upload_batches`` on, for any pipeline, and warns once."""
    with _warnings() as messages:
        predictor = Predictor.from_model_paths(
            [str(minimal_instance_single_instance_ckpt)],
            device="cpu",
            frames_on_gpu=True,
        )
    assert predictor.upload_batches is True
    assert _n_alias(messages) == 1
    assert "upload_batches" in next(m for m in messages if "deprecated alias" in m)
    # Without CUDA it warns, as upload_batches does.
    assert sum(NO_EFFECT in m for m in messages) == 1


def test_alias_on_predict():
    predictor = MagicMock()
    with (
        _warnings() as messages,
        patch.object(Predictor, "from_model_paths", return_value=predictor) as factory,
    ):
        predict("video.mp4", model_paths=["/m"], device="cpu", frames_on_gpu=True)
    assert factory.call_args.kwargs["upload_batches"] is True
    assert "frames_on_gpu" not in factory.call_args.kwargs
    assert _n_alias(messages) == 1


def test_alias_on_predict_with_an_export_dir_is_ignored_with_a_warning():
    """As ``upload_batches``: an exported model directory ignores it (no error)."""
    predictor = MagicMock()
    with (
        _warnings() as messages,
        patch.object(Predictor, "from_export_dir", return_value=predictor) as factory,
    ):
        predict("video.mp4", export_dir="/exp", device="cpu", frames_on_gpu=True)
    assert "upload_batches" not in factory.call_args.kwargs
    assert "frames_on_gpu" not in factory.call_args.kwargs
    assert _n_alias(messages) == 1
    assert sum("upload_batches is ignored" in m for m in messages) == 1


def _invoke_predict(*args):
    from sleap_nn.cli import cli

    return CliRunner().invoke(
        cli,
        ["predict", "--data_path", "/fake/path.mp4", "--model_paths", "/fake/model"]
        + list(args),
    )


@pytest.mark.parametrize("flag", ["--frames_on_gpu", "--frames-on-gpu"])
def test_alias_on_the_cli(flag):
    """The old flag reaches ``predict`` as ``upload_batches``, with one warning."""
    with (
        _warnings() as messages,
        patch(
            "sleap_nn.inference.run.predict", return_value=MagicMock()
        ) as mock_predict,
    ):
        result = _invoke_predict(flag)
    assert result.exit_code == 0, result.output
    assert mock_predict.call_args.kwargs["upload_batches"] is True
    assert "frames_on_gpu" not in mock_predict.call_args.kwargs
    assert _n_alias(messages) == 1


def test_alias_on_the_cli_stream_to_file_route(tmp_path):
    """``--stream-to-file`` builds its predictor itself: the alias reaches it too."""

    class _Built(Exception):
        pass

    with (
        _warnings() as messages,
        patch.object(Predictor, "from_model_paths", side_effect=_Built) as factory,
    ):
        result = _invoke_predict(
            "--stream-to-file", str(tmp_path / "out.slp"), "--frames_on_gpu"
        )
    assert isinstance(result.exception, _Built), result.output
    assert factory.call_args.kwargs["upload_batches"] is True
    assert "frames_on_gpu" not in factory.call_args.kwargs
    assert _n_alias(messages) == 1


def test_cli_old_flag_is_hidden():
    """``--frames_on_gpu`` still parses but is out of ``--help``."""
    from sleap_nn.cli import cli

    params = {p.name: p for p in cli.commands["predict"].params}
    old, new = params["frames_on_gpu"], params["upload_batches"]
    assert old.hidden and "deprecated" in old.help.lower()
    assert "--upload_batches" in old.help
    assert not new.hidden
    for covered in ("network-only", "--part_streaming", "not bit-identical"):
        assert covered in new.help
    help_text = CliRunner().invoke(cli, ["predict", "--help"]).output
    assert "--upload_batches" in help_text
    assert "frames_on_gpu" not in help_text and "frames-on-gpu" not in help_text
