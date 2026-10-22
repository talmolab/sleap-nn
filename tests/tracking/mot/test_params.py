"""Tests for `sleap_nn.tracking.mot.params` (MOTParams presets and parsing)."""

import pytest

from sleap_nn.tracking.mot.params import (
    MOT_TRACKER_NAMES,
    MOTParams,
    parse_mot_param,
)


@pytest.mark.parametrize("name", MOT_TRACKER_NAMES)
def test_score_normalization_per_preset(name):
    """animal: scores normalized per video; paper: published thresholds on raw
    scores."""
    assert MOTParams.for_tracker(name).normalize_scores is True
    assert MOTParams.for_tracker(name, preset="paper").normalize_scores is False
    assert MOTParams.for_tracker(name, normalize_scores=False).normalize_scores is False
    assert MOTParams(tracker=name).normalize_scores is False


def test_sort_paper_preset_uses_published_max_age():
    """Reference SORT: max_age=1 (paper); the animal preset keeps 30."""
    assert MOTParams.for_tracker("sort", preset="paper").track_buffer == 1
    assert MOTParams.for_tracker("sort").track_buffer == 30
    for name in MOT_TRACKER_NAMES:
        if name != "sort":
            assert MOTParams.for_tracker(name, preset="paper").track_buffer == 30
        assert MOTParams.for_tracker(name).track_buffer == 30


@pytest.mark.parametrize(
    "name, paper, animal",
    [
        ("sort", "xysr", "xysr"),
        ("deepsort", "xyah", "xywh"),
        ("strongsort", "xyah", "xywh"),
        ("bytetrack", "xyah", "xywh"),
        ("botsort", "xywh", "xywh"),
        ("ocsort", "xysr", "xysr"),
        ("deepocsort", "xywh", "xywh"),
    ],
)
def test_kalman_per_preset(name, paper, animal):
    """The paper preset keeps each published filter; the animal preset drops the
    pedestrian xyah (a turning animal's aspect ratio breaks it)."""
    assert MOTParams.for_tracker(name, preset="paper").kalman == paper
    assert MOTParams.for_tracker(name).kalman == animal
    assert MOTParams.for_tracker(name, kalman="xyah").kalman == "xyah"


def test_body_size_field():
    """Optional positive body size (px); ints become floats."""
    assert MOTParams.for_tracker("sort").body_size is None
    params = MOTParams.for_tracker("sort", body_size=40)
    assert params.body_size == 40.0 and isinstance(params.body_size, float)
    assert params.as_dict()["body_size"] == 40.0
    for bad in (0, -5.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="body_size"):
            MOTParams.for_tracker("sort", body_size=bad)


def test_parse_new_fields():
    """--mot_param body_size / normalize_scores."""
    assert parse_mot_param("body_size=40") == ("body_size", 40.0)
    assert parse_mot_param("body_size=none") == ("body_size", None)
    assert parse_mot_param("normalize_scores=false") == ("normalize_scores", False)
    with pytest.raises(ValueError, match="number"):
        parse_mot_param("body_size=big")
    assert parse_mot_param("max_tracks=none") == ("max_tracks", None)
    assert parse_mot_param("max_tracks=3") == ("max_tracks", 3)
    with pytest.raises(ValueError, match="cannot be None"):
        parse_mot_param("track_buffer=none")


@pytest.mark.parametrize("bad", [0, -0.1, float("nan"), float("inf"), "wide"])
def test_oks_stddev_must_be_finite_and_positive(bad):
    """oks_stddev: a finite number > 0 (NaN and inf used to pass the `<= 0` check)."""
    with pytest.raises(ValueError, match="oks_stddev"):
        MOTParams.for_tracker("bytetrack", oks_stddev=bad)
    assert MOTParams.for_tracker("bytetrack", oks_stddev=0.05).oks_stddev == 0.05
