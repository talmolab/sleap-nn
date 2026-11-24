"""File-layout and frame-format options for the ``.slp`` files inference writes.

Five options control how a predicted ``.slp`` is written. Each defaults to
``None``, which passes nothing to sleap-io, so a save without them is exactly the
save made before these options existed:

* ``slp_version``: ``"auto"``, ``"v2"`` or ``"v3"``, the file layout. ``"v2"`` is
  the SLP 2.x layout that older sleap-io (< 0.10) and SLEAP releases read;
  ``"v3"`` is the SLP 3.0 layout. ``"auto"`` (sleap-io 0.10's default) writes
  the 2.x layout when the content fits and SLP 3.0 with a warning otherwise.
  For an older sleap-io snapshot without its own ``slp_version`` option,
  ``"auto"`` passes nothing so that snapshot's default applies.
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

import contextlib
import inspect
import re
import warnings
from typing import Any, Dict, Iterator, Mapping, Optional

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


def check_save_option_names(mapping: Mapping[str, Any]) -> None:
    """Refuse keys of ``mapping`` that are not save option names.

    Raises:
        ValueError: If ``mapping`` has a key outside :data:`SAVE_OPTION_NAMES`.
    """
    unknown = sorted(str(k) for k in mapping if k not in SAVE_OPTION_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown save option(s): {', '.join(unknown)}. The save options are "
            f"{', '.join(SAVE_OPTION_NAMES)}."
        )


def _name_list(names) -> str:
    """``"a"``, ``"a and b"``, ``"a, b and c"``."""
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def unused_options_message(names, applies_to: str, reason: str) -> str:
    """A grammatical "these options have no effect" message.

    ``"image_format applies only to embedded frames and has no effect: <reason>."``
    for one option, ``"... apply ... and have no effect ..."`` for several.

    Args:
        names: The option names that are set but unused.
        applies_to: What they apply to, e.g. ``"only to embedded frames"``.
        reason: Why that is not the case here, e.g. ``"this save embeds no
            frames"``.
    """
    names = list(names)
    if len(names) == 1:
        return f"{names[0]} applies {applies_to} and has no effect: {reason}."
    return f"{_name_list(names)} apply {applies_to} and have no effect: {reason}."


# sleap-io's refusal, on a 2.x-layout save, of points marked as not labeled. It
# names sleap-io's own `sio convert --unlabeled-as`; sleap-nn spells it
# differently, so `explain_unlabeled_points` rewords exactly this error.
_UNLABELED_POINTS_ERROR = re.compile(
    r"^(\d+) point\(s\) are unlabeled, but the legacy SLP layout can't store "
    r"unlabeled points"
)


@contextlib.contextmanager
def explain_unlabeled_points() -> Iterator[None]:
    """Reword sleap-io's error for unlabeled points in an SLP 2.x save.

    sleap-io refuses to write points marked as not labeled in the 2.x layout
    and suggests its own ``sio convert`` flag. Inside this context that one
    ``ValueError`` (matched by its message) is re-raised naming sleap-nn's
    ``--unlabeled_as absent`` / ``unlabeled_as="absent"``; every other
    exception propagates unchanged.

    Raises:
        ValueError: The reworded error, chained to sleap-io's.
    """
    try:
        yield
    except ValueError as e:
        match = _UNLABELED_POINTS_ERROR.match(str(e))
        if match is None:
            raise
        raise ValueError(
            f"{match.group(1)} point(s) are marked as not labeled, which the SLP 2.x "
            "layout (slp_version 'v2') cannot store. Pass --unlabeled_as absent "
            '(unlabeled_as="absent" in Python) to write them as hidden points, or '
            "save in the SLP 3.0 layout (--slp_version v3)."
        ) from e


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
    it (sleap-io 0.10's ``"auto"`` writes the 2.x layout when the content fits,
    else SLP 3.0 with a warning). An older sleap-io snapshot that only has
    ``legacy_format`` gets ``v2`` as ``legacy_format=True``, ``v3`` as
    ``legacy_format=False`` and ``auto`` as nothing.

    Args:
        slp_version: ``"auto"``, ``"v2"`` or ``"v3"``.
        image_format: ``"png"``, ``"jpg"``, ``"jpeg"`` or ``"hdf5"``.
        image_quality: JPEG quality (0-100) or PNG compression level (0-9).
        compression: ``"gzip"`` or ``"lzf"``.
        unlabeled_as: ``"error"`` or ``"absent"``.
        embeds_frames: Whether the save embeds frames. ``False`` warns when
            ``image_format`` or ``image_quality`` is set, since they then have no
            effect; ``None`` skips the check. (The command line checks this
            before any model loads; this warning is for Python callers.)

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
                unused_options_message(
                    set_image,
                    "only to embedded frames",
                    "this save embeds no frames",
                ),
                UserWarning,
            )
    return kwargs
