"""Benchmark sub-pixel peak refinement and flip-test fusion on synthetic confmaps.

Renders sleap-nn confidence-map targets (Gaussians on the stride-``s`` grid, sampled
at input pixel ``i * stride``) for random keypoints, optionally degrades them to
look like predictions, decodes them with each refinement method, and reports the
localization error in input pixels. CPU only; about a minute at the default size.

Usage:
    uv run python scripts/benchmark_peak_refinement.py [--n 20000] [--seed 0]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sleap_nn.data.confidence_maps import make_confmaps
from sleap_nn.data.utils import make_grid_vectors
from sleap_nn.inference.ops.peaks import find_global_peaks, fuse_flip_confmaps

STRIDE = 4
SIGMA = 2.0  # confmap pixels (the training `sigma`)
H_OUT, W_OUT = 64, 48
H_IN, W_IN = H_OUT * STRIDE, W_OUT * STRIDE

CONDITIONS = {
    # name: (sigma scale, amplitude, noise std, per-pass location jitter in px)
    "clean": (1.0, 1.0, 0.0, 0.0),
    "blur": (1.25, 0.8, 0.0, 0.0),
    "blur+noise": (1.25, 0.8, 0.05, 0.0),
    "blur+noise+jitter": (1.25, 0.8, 0.05, 1.0),
}

METHODS = {
    "argmax": dict(refinement=None),
    "integral 5x5": dict(refinement="integral", integral_patch_size=5),
    "local (quarter offset)": dict(refinement="local"),
    "dark, no blur": dict(refinement="dark", dark_blur_sigma=0.0),
    "dark, blur 1": dict(refinement="dark", dark_blur_sigma=1.0),
    "dark, blur = sigma": dict(refinement="dark", dark_blur_sigma=SIGMA),
}


def blur(x, sigma):
    """Depthwise Gaussian blur of ``(S, C, H, W)`` maps (replicate padding)."""
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    s, c, h, w = x.shape
    y = F.pad(x.reshape(s * c, 1, h, w), (r, r, r, r), mode="replicate")
    y = F.conv2d(F.conv2d(y, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    return y.reshape(s, c, h, w)


def render(points, cond, gen):
    """Render ``(1, N, H_OUT, W_OUT)`` "predicted" confmaps for input-px points."""
    sig_scale, amp, noise, jitter = cond
    pts = points.clone()
    if jitter > 0:
        pts = pts + jitter * torch.randn(pts.shape, generator=gen)
    xv, yv = make_grid_vectors(H_IN, W_IN, STRIDE)
    cms = amp * make_confmaps(pts[None], xv, yv, SIGMA * sig_scale * STRIDE)
    if noise > 0:
        n = blur(torch.randn(cms.shape, generator=gen), 1.0)
        cms = cms + n / n.std() * noise
    return cms


def decode(cms, method):
    """Decode ``(1, N, H, W)`` confmaps to ``(N, 2)`` input-pixel peaks."""
    peaks, _ = find_global_peaks(cms, threshold=-1e9, **method)
    return peaks[0] * STRIDE


def main():
    """Run the benchmark and print a markdown table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    margin = 4 * SIGMA * STRIDE
    pts = torch.stack(
        [
            margin + torch.rand(args.n, generator=gen) * (W_IN - 2 * margin),
            margin + torch.rand(args.n, generator=gen) * (H_IN - 2 * margin),
        ],
        dim=-1,
    )
    pts_flip = pts.clone()
    pts_flip[:, 0] = (W_IN - 1) - pts_flip[:, 0]

    rows = []
    for cname, cond in CONDITIONS.items():
        cms = render(pts, cond, gen)
        cms_f = render(pts_flip, cond, gen)  # "prediction" on the mirrored input
        back = torch.flip(cms_f, dims=[-1])
        back_1px = torch.cat([back[..., :1], back[..., :-1]], dim=-1)
        fused = {
            "no flip": cms,
            "flip, no shift": 0.5 * (cms + back),
            "flip, 1-cell shift (HRNet)": 0.5 * (cms + back_1px),
            "flip, sub-pixel shift (sleap-nn)": fuse_flip_confmaps(
                cms, cms_f, output_stride=STRIDE
            ),
        }
        for mname, method in METHODS.items():
            for fname, fcms in fused.items():
                err = (decode(fcms, method) - pts).norm(dim=-1).numpy()
                rows.append((cname, mname, fname, err))

    print(
        f"N={args.n}, stride={STRIDE}, sigma={SIGMA} (confmap px), "
        f"confmap {H_OUT}x{W_OUT}, input {H_IN}x{W_IN}; error in input px"
    )
    print("| condition | method | flip | mean | median | p95 |")
    print("|---|---|---|---|---|---|")
    for cname, mname, fname, err in rows:
        print(
            f"| {cname} | {mname} | {fname} | {err.mean():.3f} | "
            f"{np.median(err):.3f} | {np.percentile(err, 95):.3f} |"
        )


if __name__ == "__main__":
    main()
