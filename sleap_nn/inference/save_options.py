"""File-layout and frame-format options for the ``.slp`` files inference writes.

Five options control how a predicted ``.slp`` is written. Each defaults to
``None``, which passes nothing to sleap-io, so a save without them is exactly the
save made before these options existed:

* ``slp_version``: ``"auto"``, ``"v2"`` or ``"v3"``, the file layout. ``"v2"`` is
  the SLP 2.x layout that older sleap-io (< 0.10) and SLEAP releases read;
  ``"v3"`` is the SLP 3.0 layout; ``"auto"`` leaves the choice to sleap-io.
* ``image_format``: ``"png"``, ``"jpg"`` (or ``"jpeg"``) or ``"hdf5"``, the format
  of embedded frames.
* ``image_quality``: JPEG quality (0-100) or PNG compression level (0-9) of
  re-encoded embedded frames.
* ``compression``: ``"gzip"`` or ``"lzf"`` compression of the table columns.
* ``unlabeled_as``: ``"error"`` or ``"absent"``, what the 2.x layout does with
  points that are not labeled, which it cannot store.

:func:`slp_save_kwargs` validates them and turns them into the keyword arguments
of ``sleap_io.save_slp`` (``Labels.save`` forwards them).
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Mapping, Optional

SAVE_OPTION_NAMES = (
    "slp_version",
    "image_format",
    "image_quality",
    "compression",
    "unlabeled_as",
)
SLP_VERSIONS = ("auto", "v2", "v3")
IMAGE_FORMATS = ("png", "jpg", "jpeg", "hdf5")
COMPRESSIONS = ("gzip", "lzf")
UNLABELED_AS = ("error", "absent")
_QUALITY_RANGES = {"jpg": (0, 100), "jpeg": (0, 100), "png": (0, 9)}


def _save_slp_parameters() -> frozenset:
    """Names of the keyword arguments the installed ``sleap_io.save_slp`` takes."""
    import sleap_io as sio

    return frozenset(inspect.signature(sio.save_slp).parameters)


def _sleap_io_version() -> str:
    import sleap_io as sio

    return str(getattr(sio, "__version__", "unknown"))


def _choice(name: str, value: Any, choices: tuple) -> str:
    text = str(value).strip().lower()
    if text not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}.")
    return text


def save_options_from(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """The five save options in ``mapping`` (missing ones as ``None``)."""
    return {name: mapping.get(name) for name in SAVE_OPTION_NAMES}


def validate_save_options(
    slp_version: Optional[str] = None,
    image_format: Optional[str] = None,
    image_quality: Optional[int] = None,
    compression: Optional[str] = None,
    unlabeled_as: Optional[str] = None,
) -> Dict[str, Any]:
    """Check the save options and return them normalized (``None`` where unset).

    Raises:
        ValueError: If a value is not one of its choices, ``image_quality`` is
            not an integer, is out of range for ``image_format``, or is given
            without ``image_format`` ``"png"`` or ``"jpg"``.
    """
    out: Dict[str, Any] = dict.fromkeys(SAVE_OPTION_NAMES)
    if slp_version is not None:
        out["slp_version"] = _choice("slp_version", slp_version, SLP_VERSIONS)
    if image_format is not None:
        out["image_format"] = _choice("image_format", image_format, IMAGE_FORMATS)
    if image_quality is not None:
        if isinstance(image_quality, bool) or not isinstance(image_quality, int):
            try:
                as_int = int(str(image_quality).strip())
            except ValueError:
                raise ValueError(
                    f"image_quality must be an integer; got {image_quality!r}."
                ) from None
        else:
            as_int = int(image_quality)
        bounds = _QUALITY_RANGES.get(out["image_format"])
        if bounds is None:
            raise ValueError(
                "image_quality needs image_format 'jpg' (quality 0-100) or 'png' "
                f"(compression level 0-9); got image_format={image_format!r}."
            )
        if not bounds[0] <= as_int <= bounds[1]:
            raise ValueError(
                f"image_quality for image_format {out['image_format']!r} must be "
                f"in {bounds[0]}-{bounds[1]}; got {as_int}."
            )
        out["image_quality"] = as_int
    if compression is not None:
        out["compression"] = _choice("compression", compression, COMPRESSIONS)
    if unlabeled_as is not None:
        out["unlabeled_as"] = _choice("unlabeled_as", unlabeled_as, UNLABELED_AS)
    return out


def slp_save_kwargs(
    slp_version: Optional[str] = None,
    image_format: Optional[str] = None,
    image_quality: Optional[int] = None,
    compression: Optional[str] = None,
    unlabeled_as: Optional[str] = None,
    *,
    embeds_frames: Optional[bool] = None,
) -> Dict[str, Any]:
    """The ``sleap_io.save_slp`` keyword arguments for the given save options.

    Only options that are set appear in the result, so with none set it is
    ``{}`` and the save is unchanged.

    ``slp_version`` is passed straight through when the installed sleap-io takes
    it. An older sleap-io that only has ``legacy_format`` gets ``v2`` as
    ``legacy_format=True``, ``v3`` as ``legacy_format=False`` and ``auto`` as
    nothing (sleap-io's own default layout).

    Args:
        slp_version: ``"auto"``, ``"v2"`` or ``"v3"``.
        image_format: ``"png"``, ``"jpg"``, ``"jpeg"`` or ``"hdf5"``.
        image_quality: JPEG quality (0-100) or PNG compression level (0-9).
        compression: ``"gzip"`` or ``"lzf"``.
        unlabeled_as: ``"error"`` or ``"absent"``.
        embeds_frames: Whether the save embeds frames. ``False`` warns when
            ``image_format`` or ``image_quality`` is set, since they then have no
            effect; ``None`` skips the check.

    Returns:
        Keyword arguments for ``sleap_io.save_slp`` / ``Labels.save``.

    Raises:
        ValueError: If a value is invalid (see :func:`validate_save_options`) or
            the installed sleap-io cannot take a set option.
    """
    opts = validate_save_options(
        slp_version=slp_version,
        image_format=image_format,
        image_quality=image_quality,
        compression=compression,
        unlabeled_as=unlabeled_as,
    )
    if all(v is None for v in opts.values()):
        return {}

    params = _save_slp_parameters()

    def _unsupported(name: str) -> ValueError:
        return ValueError(
            f"{name} is not supported by the installed sleap-io "
            f"({_sleap_io_version()}): its save_slp has no '{name}' option. "
            "Upgrade sleap-io or drop the option."
        )

    kwargs: Dict[str, Any] = {}
    version = opts.pop("slp_version")
    if version is not None:
        if "slp_version" in params:
            kwargs["slp_version"] = version
        elif "legacy_format" in params:
            if version == "v2":
                kwargs["legacy_format"] = True
            elif version == "v3":
                kwargs["legacy_format"] = False
        else:
            raise _unsupported("slp_version")
    for name, value in opts.items():
        if value is None:
            continue
        if name not in params:
            raise _unsupported(name)
        kwargs[name] = value

    if embeds_frames is False:
        set_image = [
            n for n in ("image_format", "image_quality") if opts.get(n) is not None
        ]
        if set_image:
            warnings.warn(
                f"{' and '.join(set_image)} apply only to embedded frames; this "
                "save embeds no frames, so they have no effect.",
                UserWarning,
                stacklevel=2,
            )
    return kwargs
