"""Miscellaneous utility functions for Inference modules."""

from omegaconf import OmegaConf
import sleap_io as sio
from sleap_io.io.skeleton import SkeletonYAMLDecoder
import torch


def get_skeleton_from_config(skeleton_config: OmegaConf):
    """Create Sleap-io Skeleton objects from config.

    Args:
        skeleton_config: OmegaConf object containing the skeleton config.

    Returns:
        Returns a list of `sio.Skeleton` objects created from the skeleton config
        stored in the `training_config.yaml`.

    """
    skeletons = []
    for skel_cfg in skeleton_config:
        skel = SkeletonYAMLDecoder().decode(dict(skel_cfg))
        skel.name = skel_cfg.name
        skeletons.append(skel)

    return skeletons


def interp1d(x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor) -> torch.Tensor:
    """Linear 1-D interpolation.

    Src: https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py

    Args:
        x : (N, ) or (D, N) Tensor.
        y : (N,) or (D, N) float Tensor. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Tensor. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.

    Returns:
        (P, ) or (D, P) Tensor.
    """
    # making the vectors at least 2D
    is_flat = {}
    v = {}
    eps = torch.finfo(y.dtype).eps
    for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
        assert len(vec.shape) <= 2, "interp1d: all inputs must be at most 2-D."
        if len(vec.shape) == 1:
            v[name] = vec[None, :]
        else:
            v[name] = vec
        is_flat[name] = v[name].shape[0] == 1
    device = y.device

    # Checking for the dimensions
    assert v["x"].shape[1] == v["y"].shape[1] and (
        v["x"].shape[0] == v["y"].shape[0]
        or v["x"].shape[0] == 1
        or v["y"].shape[0] == 1
    ), (
        "x and y must have the same number of columns, and either "
        "the same number of row or one of them having only one "
        "row."
    )

    if (v["x"].shape[0] == 1) and (v["y"].shape[0] == 1) and (v["xnew"].shape[0] > 1):
        # if there is only one row for both x and y, there is no need to
        # loop over the rows of xnew because they will all have to face the
        # same interpolation problem. We should just stack them together to
        # call interp1d and put them back in place afterwards.
        v["xnew"] = v["xnew"].contiguous().view(1, -1)

    # identify the dimensions of output
    D = max(v["x"].shape[0], v["xnew"].shape[0])
    shape_ynew = (D, v["xnew"].shape[-1])
    ynew = torch.zeros(*shape_ynew, device=device)

    # moving everything to the desired device in case it was not there
    # already (not handling the case things do not fit entirely, user will
    # do it if required.)
    for name in v:
        v[name] = v[name].to(device)

    # calling searchsorted on the x values.
    ind = ynew.long()

    # expanding xnew to match the number of rows of x in case only one xnew is
    # provided
    if v["xnew"].shape[0] == 1:
        v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

    # the squeeze is because torch.searchsorted does accept either an n-d tensor with
    # matching shapes for x and xnew or a 1d vector for x. Here we would
    # have (1,len) for x sometimes
    torch.searchsorted(v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind)

    # the `-1` is because searchsorted looks for the index where the values
    # must be inserted to preserve order. And we want the index of the
    # preceding value.
    ind -= 1
    # we clamp the index, because the number of intervals is x.shape-1,
    # and the left neighbour should hence be at most number of intervals
    # -1, i.e. number of columns in x -2
    ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

    # helper function to select stuff according to the found indices.
    def sel(name):
        if is_flat[name]:
            return v[name].contiguous().view(-1)[ind]
        return torch.gather(v[name], 1, ind)

    # assuming x are sorted in the dimension 1, computing the slopes for
    # the segments
    is_flat["slopes"] = is_flat["x"]
    # now we have found the indices of the neighbors, we start building the
    # output.
    v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
        eps + (v["x"][:, 1:] - v["x"][:, :-1])
    )

    # now build the linear interpolation
    ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

    if len(y.shape) == 1:
        ynew = ynew.view(-1)

    return ynew
