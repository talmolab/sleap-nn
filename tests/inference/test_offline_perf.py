"""Unit tests for offline inference performance optimizations."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# torch.compile Power-of-2 Bucket Ladder
# ─────────────────────────────────────────────────────────────────────────────


def test_torch_backend_power_of_two_bucket_ladder_and_burst_dropback(monkeypatch):
    """A single burst frame never locks padding permanently; >cap chunks cleanly."""
    import torch._dynamo

    from sleap_nn.inference.layers.backends.torch_backend import (
        COMPILE_BATCH_BUCKETS,
        TorchBackend,
    )

    class _LinearNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x.float())

    real_compile = torch.compile
    compiled_shapes = []

    def counting_backend(gm, example_inputs):
        compiled_shapes.append(
            next(
                tuple(t.shape)
                for t in example_inputs
                if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[-1] in (4, 16)
            )
        )
        return gm.forward

    def fake_compile(model, mode=None, dynamic=None):
        return real_compile(model, backend=counting_backend, dynamic=dynamic)

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", fake_compile)
    try:
        torch.manual_seed(0)
        net = _LinearNet()
        eager = TorchBackend(model=_LinearNet(), device="cpu")
        eager.model.load_state_dict(net.state_dict())
        compiled = TorchBackend(model=net, device="cpu", use_compile=True)
        assert compiled.compile_batch_buckets == COMPILE_BATCH_BUCKETS

        # Sequence with a big crowd burst (150 -> chunked 64+64+32, _padded_batch=64)
        # followed by small batches (5 -> 8, 1 -> 1) and another burst (270 -> chunked).
        seq = [(5, 8), (150, 64), (40, 64), (5, 8), (1, 1), (270, 64), (7, 8)]
        for n, expected_padded in seq:
            x = torch.randn(n, 1, 4, 4)
            out = compiled(x)["output"]
            assert out.shape == (n, 2, 4, 4)
            assert compiled._padded_batch == expected_padded
            torch.testing.assert_close(out, eager(x)["output"])

        # Only in-ladder buckets (<= 64) were compiled: 8 (for 5, 7), 64 (chunks),
        # 32 (for 150 % 64 = 22), 1 (for 1), and 16 (for 270 % 64 = 14).
        # No oversized 128/256/270 graph was ever compiled, and small batches
        # dropped cleanly back down to 8 and 1 after bursts!
        assert compiled_shapes == [
            (8, 1, 4, 4),
            (64, 1, 4, 4),
            (32, 1, 4, 4),
            (1, 1, 4, 4),
            (16, 1, 4, 4),
        ]
        assert torch._dynamo.config.cache_size_limit >= 64
        if hasattr(torch._dynamo.config, "recompile_limit"):
            assert torch._dynamo.config.recompile_limit >= 64

        # Verify default mode compiles with dynamic=True (single symbolic graph
        # across buckets >= 2, preventing mid-clip recompiles on tail/burst batches).
        torch._dynamo.reset()
        compiled_shapes.clear()
        dyn_compiled = TorchBackend(
            model=net, device="cpu", use_compile=True, compile_mode="default"
        )
        for n in [8, 4, 9, 24, 32, 150]:
            x = torch.randn(n, 1, 16, 16)
            out = dyn_compiled(x)["output"]
            assert out.shape == (n, 2, 16, 16)
            torch.testing.assert_close(out, eager(x)["output"])
        # Buckets 32 and 64 reuse the symbolic graph without recompiling
        # (CPU conv2d has an internal batch<=15 vs >=16 dispatch split, so <=2
        # graphs across 5 distinct bucket sizes {4, 8, 16, 32, 64}).
        assert len(compiled_shapes) <= 2
        assert (32, 1, 16, 16) not in compiled_shapes
        assert (64, 1, 16, 16) not in compiled_shapes
    finally:
        torch._dynamo.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Sync-Free Batched Preproc / NMS / Cropping / coord / PartStage
# ─────────────────────────────────────────────────────────────────────────────


def test_batched_sizematcher_and_squash_exact_parity():
    from sleap_nn.data.resizing import apply_sizematcher, apply_squash_resize
    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.base import InferenceLayer
    from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
    from sleap_nn.inference.outputs import Outputs

    class _NoopLayer(InferenceLayer):
        def postprocess(self, raw_out, info):
            return Outputs(preprocess_info=info)

    torch.manual_seed(0)
    imgs = torch.randint(0, 256, (6, 3, 180, 240), dtype=torch.uint8)

    for mode in ("letterbox", "squash"):
        layer = _NoopLayer(
            backend=TorchBackend(nn.Identity(), device="cpu"),
            preprocess_config=PreprocessConfig(
                max_height=128, max_width=160, resize_mode=mode
            ),
            postprocess_config=PostprocessConfig(),
            output_stride=1,
            max_stride=16,
        )
        batched, info = layer.preprocess(imgs)
        assert info.eff_scale_is_identity is False
        # Compare against per-frame reference loop
        per_frame = []
        for b in range(imgs.shape[0]):
            if mode == "letterbox":
                r, _ = apply_sizematcher(imgs[b], 128, 160, warn=False)
            else:
                r, _ = apply_squash_resize(imgs[b], 128, 160)
            per_frame.append(r)
        ref = torch.stack(per_frame, dim=0)
        torch.testing.assert_close(batched[:, 0], ref, atol=0, rtol=0)

    # Identity case sets eff_scale_is_identity=True
    id_layer = _NoopLayer(
        backend=TorchBackend(nn.Identity(), device="cpu"),
        preprocess_config=PreprocessConfig(max_height=180, max_width=240),
        postprocess_config=PostprocessConfig(),
        output_stride=1,
        max_stride=1,
    )
    _, id_info = id_layer.preprocess(imgs)
    assert id_info.eff_scale_is_identity is True


def test_vectorized_batched_nms_exact_parity_vs_per_frame():
    from torchvision.ops import batched_nms

    from sleap_nn.inference.bbox import decode_set, decode_set_keypoints

    torch.manual_seed(42)
    B, Q, C, K = 6, 40, 3, 4
    logits = torch.randn(B, Q, C + 1)
    # Create overlapping clusters of boxes so NMS actively suppresses duplicates
    centers = torch.randint(1, 5, (B, Q, 2)).float() * 0.2
    sizes = torch.empty(B, Q, 2).uniform_(0.12, 0.25)
    boxes = torch.cat([centers, sizes], dim=-1)
    kpts = torch.rand(B, Q, K, 2)
    kpt_logits = torch.randn(B, Q, K)
    img_hw = (576, 768)

    def _ref_decode_set(iou_thr):
        probs = logits[..., :-1].float().sigmoid()
        k = min(300, Q * C)
        scores, inds = probs.reshape(B, -1).topk(k, dim=1)
        classes = inds % C
        q_idx = inds // C
        bx = boxes.float().gather(1, q_idx[..., None].expand(B, k, 4))
        cx, cy, w, h = bx.unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        xyxy = xyxy * xyxy.new_tensor([img_hw[1], img_hw[0], img_hw[1], img_hw[0]])
        xyxy = torch.stack(
            [
                xyxy[..., 0].clamp(0, img_hw[1] - 1),
                xyxy[..., 1].clamp(0, img_hw[0] - 1),
                xyxy[..., 2].clamp(0, img_hw[1] - 1),
                xyxy[..., 3].clamp(0, img_hw[0] - 1),
            ],
            -1,
        )
        m = 20
        out_b = torch.full((B, m, 4), float("nan"))
        out_s = torch.full((B, m), float("nan"))
        out_c = torch.full((B, m), -1, dtype=torch.long)
        for i in range(B):
            keep = (scores[i] >= 0.25).nonzero().flatten()
            bxi = xyxy[i][keep]
            keep = keep[(bxi[:, 2] > bxi[:, 0]) & (bxi[:, 3] > bxi[:, 1])]
            if iou_thr is not None and keep.numel():
                nms = batched_nms(
                    xyxy[i][keep].float(),
                    scores[i][keep].float(),
                    classes[i][keep],
                    float(iou_thr),
                )
                keep = keep[nms]
            keep = keep[:m]
            keep = keep[scores[i][keep].argsort(descending=True)]
            n = keep.numel()
            out_b[i, :n] = xyxy[i][keep]
            out_s[i, :n] = scores[i][keep]
            out_c[i, :n] = classes[i][keep]
        return out_b, out_s, out_c

    for iou_thr in (None, 0.45, 0.7):
        got_b, got_s, got_c = decode_set(
            logits,
            boxes,
            img_hw,
            score_threshold=0.25,
            max_detections=20,
            iou_threshold=iou_thr,
        )
        ref_b, ref_s, ref_c = _ref_decode_set(iou_thr)
        torch.testing.assert_close(got_b, ref_b, equal_nan=True, atol=1e-5, rtol=0)
        torch.testing.assert_close(got_s, ref_s, equal_nan=True, atol=1e-6, rtol=0)
        assert torch.equal(got_c, ref_c)

    # Also verify decode_set_keypoints with NMS & empty-threshold case
    kb, ks, kc, kp, kv = decode_set_keypoints(
        logits,
        boxes,
        kpts,
        kpt_logits,
        img_hw,
        score_threshold=0.9999,
        max_detections=10,
        iou_threshold=0.5,
    )
    assert torch.isnan(ks).all() and (kc == -1).all()


def test_undo_eff_scale_identity_flag_and_fallback():
    from sleap_nn.inference.ops.coord import undo_eff_scale

    coords = torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]]])
    ones = torch.ones(2)
    scales = torch.tensor([0.5, 2.0])

    # Explicit True returns exact same tensor without touching eff_scale
    assert undo_eff_scale(coords, scales, eff_scale_is_identity=True) is coords
    # Explicit False divides cleanly
    out = undo_eff_scale(coords, scales, eff_scale_is_identity=False)
    torch.testing.assert_close(
        out, torch.tensor([[[20.0, 40.0]], [[15.0, 20.0]]])
    )
    # Fallback None on CPU preserves identity return
    assert undo_eff_scale(coords, ones) is coords


def test_crop_windows_vectorized_numerical_parity():
    import torch.nn.functional as F

    from sleap_nn.data.instance_cropping import (
        _frame_coverage,
        bbox_crop_windows,
        crop_hw,
        crop_windows,
        points_from_crop,
        points_to_crop,
    )

    def _ref_crop_windows(images, windows, sample_inds, crop_size):
        n, c = windows.shape[0], images.shape[1]
        h, w = images.shape[-2:]
        ch, cw = crop_hw(crop_size)
        out = torch.zeros(n, c, ch, cw, device=images.device)
        ix = torch.arange(cw, device=images.device, dtype=torch.float32) + 0.5
        iy = torch.arange(ch, device=images.device, dtype=torch.float32) + 0.5
        xs = windows[:, 0:1] + ix[None, :] * ((windows[:, 2:3] - windows[:, 0:1]) / cw)
        ys = windows[:, 1:2] + iy[None, :] * ((windows[:, 3:4] - windows[:, 1:2]) / ch)
        grid = torch.stack(
            [
                (2 * xs / w - 1)[:, None, :].expand(n, ch, cw),
                (2 * ys / h - 1)[:, :, None].expand(n, ch, cw),
            ],
            dim=-1,
        )
        frames = images.float()
        for b in torch.unique(sample_inds).tolist():
            rows = (sample_inds == b).nonzero(as_tuple=False).flatten()
            g = grid[rows].reshape(1, -1, cw, 2)
            sampled = F.grid_sample(
                frames[b : b + 1],
                g,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            out[rows] = sampled.reshape(c, len(rows), ch, cw).transpose(0, 1)
        cov_x = _frame_coverage(windows[:, 0], windows[:, 2], cw, w)
        cov_y = _frame_coverage(windows[:, 1], windows[:, 3], ch, h)
        return out * (cov_y[:, None, :, None] * cov_x[:, None, None, :])

    torch.manual_seed(0)
    for B in (1, 3, 8):
        imgs = torch.randint(0, 256, (B, 3, 128, 160), dtype=torch.uint8)
        raw_boxes = torch.tensor(
            [
                [-15.0, -10.0, 50.0, 60.0],
                [110.0, 90.0, 180.0, 145.0],
                [30.0, 20.0, 90.0, 80.0],
            ]
            * B
        )
        wins, scales, _ = bbox_crop_windows(
            raw_boxes, (48, 64), aspect="square", padding=0.2
        )
        sinds = torch.arange(B).repeat_interleave(3)
        got = crop_windows(imgs, wins, sinds, (48, 64))
        ref = _ref_crop_windows(imgs, wins, sinds, (48, 64))
        assert got.is_contiguous()
        assert (got - ref).abs().max().item() < 1e-3
        # UDP coordinate round-trip remains exact
        pts = torch.rand(len(wins), 5, 2) * 100.0
        rt = points_from_crop(points_to_crop(pts, wins, scales), wins, scales)
        torch.testing.assert_close(rt, pts, atol=1e-4, rtol=0)


def test_streaming_part_stage_tensor_parity_and_session_floor():
    import numpy as np

    from sleap_nn.inference.keypoint_crops import KeypointCropRule, StreamingPartStage
    from sleap_nn.inference.outputs import Outputs

    rule = KeypointCropRule(
        center=["tail_base", "spine"],
        reference=["ear_l", "ear_r"],
        reference_fallback=["nose"],
        scale=2.0,
        fallback="extent",
        fallback_scale=1.2,
        min_half_side=12.0,
        session_floor=1.0,
        session_min_boxes=3,
    )
    names = ["nose", "ear_l", "ear_r", "spine", "tail_base"]
    resolved = rule.resolve(names)

    # Batch covering all rule paths: normal ref, fallback ref, center fallback,
    # extent fallback, all-NaN empty slot, and curled short ref (floored by session_floor).
    kpts = torch.tensor(
        [
            [
                [[50, 50], [45, 55], [55, 55], [50, 75], [50, 100]],  # normal
                [
                    [50, 50],
                    [float("nan"), float("nan")],
                    [55, 55],
                    [50, 75],
                    [50, 100],
                ],  # ref_fallback
                [
                    [50, 50],
                    [45, 55],
                    [55, 55],
                    [50, 75],
                    [float("nan"), float("nan")],
                ],  # center[1]=spine
                [[float("nan")] * 2] * 5,  # empty slot
            ],
            [
                [
                    [50, 50],
                    [float("nan")] * 2,
                    [float("nan")] * 2,
                    [float("nan")] * 2,
                    [60, 60],
                ],  # fallback extent
                [[50, 50], [50, 51], [50, 51], [50, 52], [50, 52]],  # curled -> floored
                [[10, 10], [10, 10], [10, 10], [10, 10], [10, 10]],  # d==0 -> extent
                [[float("nan")] * 2] * 5,
            ],
        ],
        dtype=torch.float32,
    )
    inst_scores = torch.tensor(
        [[0.9, 0.8, float("nan"), float("nan")], [0.7, 0.6, 0.5, float("nan")]]
    )

    # Compare boxes_tensor (floor=0) directly against scalar rule.box
    center, raw_half, valid = rule.boxes_tensor(kpts, resolved)
    for b in range(kpts.shape[0]):
        for i in range(kpts.shape[1]):
            ref_box, _ = rule.box(kpts[b, i].numpy(), resolved, 0.0)
            if ref_box is None:
                assert not bool(valid[b, i])
            else:
                assert bool(valid[b, i])
                h = float(raw_half[b, i])
                cx, cy = float(center[b, i, 0]), float(center[b, i, 1])
                got_box = np.array([cx - h, cy - h, cx + h, cy + h], dtype=np.float32)
                np.testing.assert_allclose(got_box, ref_box, atol=1e-4)

    # Verify StreamingPartStage passes exact session-floored boxes & supports undo_last
    seen_boxes = []
    seen_scores = []

    class _FakePartLayer:
        def predict(self, images, frame_boxes=None, frame_box_scores=None):
            seen_boxes.append(frame_boxes.clone())
            seen_scores.append(frame_box_scores.clone())
            b, k = frame_boxes.shape[:2]
            return Outputs(
                pred_keypoints=torch.zeros(b, k, 2, 2, device=frame_boxes.device),
                pred_peak_values=torch.ones(b, k, 2, device=frame_boxes.device),
            )

    stage = StreamingPartStage(_FakePartLayer(), rule, names, n_part_nodes=2)
    out_in = Outputs(
        pred_keypoints=kpts,
        pred_peak_values=torch.ones_like(kpts[..., 0]),
        instance_scores=inst_scores,
    )
    merged = stage(
        torch.zeros(2, 3, 128, 128), out_in, video_indices=torch.tensor([0, 0])
    )
    assert merged.pred_keypoints.shape == (2, 4, 7, 2)
    # Curled mouse at (1, 1) had raw half clamped to min_half_side=12, then floored by session median!
    assert 0 in stage.last_floors and stage.last_floors[0] > 12.0
    curled_box = seen_boxes[0][1, 1]
    assert float((curled_box[2] - curled_box[0]) / 2) == pytest.approx(
        stage.last_floors[0], abs=1e-3
    )
    # NaN instance score on valid box (0, 2) defaults to 1.0
    assert float(seen_scores[0][0, 2]) == pytest.approx(1.0)
    assert torch.isnan(seen_scores[0][0, 3])
    # undo_last cleanly restores floor state
    stage.undo_last()
    assert len(stage.every) == 0 and stage.last_floors == {}
