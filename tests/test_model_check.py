"""Is the chosen model compatible with spaCR and the chosen classes?"""
import os

import pytest

from spacr.model_check import (
    ModelReport, check_model, expected_channels, expected_classes,
    resolve_model_source,
)


def _classes(n=2):
    return {f"c{i}": {"column": "annot", "value": i} for i in range(n)}


# ---------------------------------------------------------------------------
# Which model is actually used
# ---------------------------------------------------------------------------

def test_a_custom_model_path_that_exists_supersedes_model_type(tmp_path):
    """"if custom model has a path and a model can be loaded from that path,
    this setting super seeds model type"."""
    path = tmp_path / "model.pth"
    path.write_bytes(b"x")
    kind, name = resolve_model_source({"model_type": "maxvit_t",
                                       "custom_model_path": str(path)})
    assert kind == "custom" and name == str(path)


def test_a_missing_custom_path_falls_back_to_model_type(tmp_path):
    """A path that is set and missing is a mistake, not a preference."""
    kind, name = resolve_model_source(
        {"model_type": "maxvit_t",
         "custom_model_path": str(tmp_path / "nope.pth")})
    assert kind == "builtin" and name == "maxvit_t"


def test_no_boolean_is_consulted(tmp_path):
    """The old custom_model flag could disagree with the path beside it."""
    path = tmp_path / "m.pth"
    path.write_bytes(b"x")
    kind, _ = resolve_model_source({"custom_model": False,
                                    "custom_model_path": str(path)})
    assert kind == "custom"


# ---------------------------------------------------------------------------
# What the dataset will hand it
# ---------------------------------------------------------------------------

def test_the_channel_count_comes_from_whichever_setting_is_set():
    assert expected_channels({"train_channels": [0, 1, 2]}) == 3
    assert expected_channels({"extract_channels": [0, 1]}) == 2
    assert expected_channels({}) is None


def test_the_class_count_comes_from_the_classes_dict():
    assert expected_classes({"classes": _classes(3)}) == 3
    assert expected_classes({}) is None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def test_no_model_at_all_says_what_to_set():
    report = check_model({"classes": _classes()})
    assert not report.ok
    assert "model_type" in report.summary()


def test_a_single_class_is_not_a_classifier():
    report = check_model({"model_type": "maxvit_t", "classes": _classes(1)})
    assert not report.ok
    assert "needs two" in report.summary()


def test_no_classes_defined_is_reported_rather_than_assumed():
    report = check_model({"model_type": "maxvit_t"})
    assert not report.ok
    assert "Classes" in report.summary()


def test_a_file_that_is_not_a_model_says_so(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a model")
    report = check_model({"custom_model_path": str(path),
                          "classes": _classes()})
    assert not report.ok
    assert "cannot be read" in report.summary()


def test_a_state_dict_is_told_apart_from_a_model(tmp_path):
    """It can be loaded INTO a model but is not one, and the difference is
    worth saying plainly rather than failing later."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pth"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, path)

    report = check_model({"custom_model_path": str(path),
                          "classes": _classes()})
    assert not report.ok
    assert "resume_checkpoint" in report.summary()


def test_a_head_that_does_not_match_the_classes_is_caught(tmp_path):
    """The failure that silently half-works: a two-class head on a
    three-class problem trains happily and is wrong about every object of the
    third class."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    path = tmp_path / "model.pth"
    torch.save(model, path)

    report = check_model({"custom_model_path": str(path),
                          "classes": _classes(3), "train_channels": [0, 1, 2]})
    assert not report.ok
    assert "2 output" in report.summary() and "3 classes" in report.summary()


def test_a_matching_model_passes(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))
    path = tmp_path / "model.pth"
    torch.save(model, path)

    report = check_model({"custom_model_path": str(path),
                          "classes": _classes(3), "train_channels": [0, 1, 2]})
    assert report.ok, report.summary()
    assert "model_type is not used" in report.summary()


def test_an_unusual_channel_count_is_a_note_not_a_problem(tmp_path):
    """Two channels is fine; the first layer is adapted. Saying so is useful,
    refusing would be wrong."""
    report = check_model({"model_type": "maxvit_t", "classes": _classes(2),
                          "train_channels": [0, 1]})
    assert "first layer" in report.summary() or report.ok


def test_the_check_never_raises(tmp_path):
    """It runs from a click, and a dialog that crashes the screen is a worse
    answer than one that says what is wrong."""
    for settings in ({}, {"custom_model_path": "/nope"},
                     {"model_type": None}, {"classes": "nonsense"}):
        try:
            report = check_model(settings)
        except Exception as exc:                       # pragma: no cover
            pytest.fail(f"check_model raised on {settings!r}: {exc}")
        assert isinstance(report, ModelReport)
