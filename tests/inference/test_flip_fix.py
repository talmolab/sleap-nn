"""fix_flips: undo head-to-tail flips frame to frame (sleap_nn.inference.flip_fix)."""

import numpy as np
import pytest
import sleap_io as sio
import torch

from sleap_nn.inference.flip_fix import FlipFixer
from sleap_nn.inference.outputs import Outputs

NAMES = ["nose", "mid", "tail_base", "left", "right"]


def _animal(x, flipped=False, score=0.9):
    """One mouse along x at frame position ``x`` (nose ahead), 5 nodes."""
    pts = np.array(
        [[x + 20, 50], [x, 50], [x - 20, 50], [x, 44], [x, 56]], dtype=np.float32
    )
    if flipped:
        pts = pts[[2, 1, 0, 4, 3]]  # head and tail (and sides) swapped
    return pts, np.full(5, score, dtype=np.float32)


def _outputs(frames):
    """Outputs for frames given as lists of (points, scores) per instance."""
    n_inst = max(len(f) for f in frames)
    k = np.full((len(frames), n_inst, 5, 2), np.nan, np.float32)
    v = np.full((len(frames), n_inst, 5), np.nan, np.float32)
    for b, f in enumerate(frames):
        for i, (p, s) in enumerate(f):
            k[b, i], v[b, i] = p, s
    return Outputs(
        pred_keypoints=torch.from_numpy(k), pred_peak_values=torch.from_numpy(v)
    )


def _run(fixer, frames, fidx, vidx=None):
    out = _outputs(frames)
    fixer(out, np.asarray(fidx), None if vidx is None else np.asarray(vidx))
    return out.pred_keypoints.numpy()


def test_flip_turned_back_and_first_frame_left_alone():
    fixer = FlipFixer.for_skeleton(NAMES, ["nose", "tail_base"])
    frames = [[_animal(100 + 2 * f, flipped=(f == 3))] for f in range(5)]
    k = _run(fixer, frames, range(5))
    want3, _ = _animal(106)
    np.testing.assert_allclose(
        k[3, 0][[0, 2]], want3[[0, 2]], atol=1e-4
    )  # head, tail back
    np.testing.assert_array_equal(
        k[4, 0], frames[4][0][0]
    )  # the next frame is left as is
    assert fixer.counts == {"flips_undone": 1, "flips_kept": 0}
    first = FlipFixer(0, 2)  # a flipped first frame has nothing to compare with
    k = _run(first, [[_animal(100, flipped=True)]], [0])
    np.testing.assert_array_equal(k[0, 0], _animal(100, flipped=True)[0])
    assert first.counts["flips_undone"] == 0


def test_gap_restarts_and_batch_boundaries_do_not():
    frames = [[_animal(100 + 2 * f, flipped=(f == 3))] for f in range(5)]
    one = FlipFixer(0, 2)
    whole = _run(one, frames, range(5))
    split = FlipFixer(0, 2)
    a = _run(split, frames[:3], range(3))  # batch boundary between frames 2 and 3
    b = _run(split, frames[3:], range(3, 5))
    np.testing.assert_array_equal(np.concatenate([a, b]), whole)
    gap = FlipFixer(0, 2)
    k = _run(gap, [frames[0], frames[1], frames[3]], [0, 1, 3])  # frame 2 missing
    np.testing.assert_array_equal(k[2, 0], frames[3][0][0])  # not compared across a gap
    assert gap.counts["flips_undone"] == 0


def test_two_videos_are_independent():
    fixer = FlipFixer(0, 2)
    frames = [
        [_animal(100)],  # video 0, frame 0
        [_animal(300)],  # video 1, frame 0
        [_animal(102, flipped=True)],  # video 0, frame 1: flipped
        [_animal(302)],  # video 1, frame 1
    ]
    k = _run(fixer, frames, [0, 0, 1, 1], [0, 1, 0, 1])
    want, _ = _animal(102)
    np.testing.assert_allclose(k[2, 0][[0, 2]], want[[0, 2]], atol=1e-4)
    np.testing.assert_array_equal(k[3, 0], frames[3][0][0])


def test_higher_scoring_flip_is_kept_and_missing_or_unmatched_left_alone():
    fixer = FlipFixer(0, 2)
    frames = [
        [_animal(100, score=0.5)],
        [_animal(102, flipped=True, score=0.7)],  # head/tail now score 0.2 higher
    ]
    k = _run(fixer, frames, [0, 1])
    np.testing.assert_array_equal(k[1, 0], frames[1][0][0])
    assert fixer.counts == {"flips_undone": 0, "flips_kept": 1}
    fixer = FlipFixer(0, 2)
    p, s = _animal(102, flipped=True)
    p[0] = np.nan  # head missing
    far, sf = _animal(400, flipped=True)  # no overlap with any earlier animal
    k = _run(fixer, [[_animal(100), _animal(250)], [(p, s), (far, sf)]], [0, 1])
    np.testing.assert_array_equal(k[1, 0], p)
    np.testing.assert_array_equal(k[1, 1], far)
    assert fixer.counts["flips_undone"] == 0


def test_node_names_by_bare_name():
    fixer = FlipFixer.for_skeleton(
        ["skel/nose", "skel/tail_base"], ["nose", "tail_base"]
    )
    assert (fixer.head, fixer.tail) == (0, 1)
    with pytest.raises(ValueError, match="needs the nodes"):
        FlipFixer.for_skeleton(NAMES, ["snout", "tail_base"])
    with pytest.raises(ValueError, match="two node names"):
        FlipFixer.for_skeleton(NAMES, ["nose"])


def test_predict_with_fix_flips(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """predict(fix_flips=...) runs and records its counts; off by default."""
    from sleap_nn.inference.run import predict

    kw = dict(
        model_paths=[
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
        ],
        device="cpu",
        peak_threshold=0.0,
    )
    plain = predict(str(minimal_instance), **kw)
    fixed = predict(str(minimal_instance), fix_flips=["A", "B"], **kw)
    assert "fix_flips" not in plain.provenance
    assert fixed.provenance["fix_flips"]["nodes"] == ["A", "B"]
    a = np.stack([i.numpy() for lf in plain for i in lf.predicted_instances])
    b = np.stack([i.numpy() for lf in fixed for i in lf.predicted_instances])
    np.testing.assert_array_equal(a, b)  # one frame: nothing to compare with
    with pytest.raises(ValueError, match="needs the nodes"):
        predict(str(minimal_instance), fix_flips=["nose", "tail_base"], **kw)


def test_cli_fix_flips(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    base = [
        "predict",
        "-i",
        str(minimal_instance),
        "-m",
        str(minimal_instance_centroid_ckpt),
        "-m",
        str(minimal_instance_centered_instance_ckpt),
        "--device",
        "cpu",
        "-o",
        str(tmp_path / "out.slp"),
    ]
    r = CliRunner().invoke(cli, base + ["--fix_flips", "A,B"])
    assert r.exit_code == 0, r.output
    assert sio.load_slp(str(tmp_path / "out.slp")).provenance["fix_flips"]["nodes"] == [
        "A",
        "B",
    ]
    r = CliRunner().invoke(cli, base + ["--fix_flips", "A"])
    assert r.exit_code != 0 and "two node names" in r.output
