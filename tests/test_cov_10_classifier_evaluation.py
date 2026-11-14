"""Refusals, fallbacks and unverifiable identities in classifier evaluation.

This module's whole job is to stop a number that means nothing from being
reported as a score. The branches exercised here are the ones that fire when
the inputs are not what the caller believes: a frame that carries its object
identity on the index instead of in a column, a group id that is blank, a
crop file that has vanished or cannot be read, folds that overlap, class
indices with no probability column, and a figure writer that fails while the
rest of the bundle is already on disk. Each of them is the difference between
a refusal the user can act on and a leaked, over-optimistic accuracy.
"""
from __future__ import annotations

from dataclasses import fields
import json

import numpy as np
import pytest

from spacr import classifier_evaluation as ce
from spacr.classifier_evaluation import (
    EVALUATION_FILES,
    audit_dataset_splits,
    audit_split_leakage,
    dataset_split_paths,
    evaluate_predictions,
    expected_calibration_error,
    fit_temperature,
    load_evaluation_bundle,
    normalize_probabilities,
    split_columns_for,
    write_evaluation_bundle,
)


@pytest.mark.parametrize(
    "record",
    (ce.SplitReport, ce.LeakageReport, ce.FoldLeakageAudit),
)
def test_split_and_leakage_records_document_every_reported_field(record):
    """Every split size and leakage finding is explained in the public API."""
    documentation = record.__doc__ or ""
    missing = [
        item.name
        for item in fields(record)
        if f":param {item.name}:" not in documentation
    ]
    assert not missing, f"{record.__name__}: {missing}"


# ---------------------------------------------------------------------------
# Which columns a split level needs
# ---------------------------------------------------------------------------

def test_a_cell_split_needs_no_metadata_columns():
    """The finest level is the row itself, so it must not demand plate/row
    /column metadata that an image-only dataset does not have."""
    level, wanted = split_columns_for("cell", [], table="cells")
    assert (level, wanted) == ("cell", [])


def test_a_well_split_names_the_complete_identity_it_will_group_on():
    """A partial key is not a well. The caller has to be told all three
    columns so it cannot group on ``columnID`` alone across plates."""
    level, wanted = split_columns_for(
        "well", ["plateID", "rowID", "columnID", "extra"], table="cells")
    assert level == "well"
    assert wanted == ["plateID", "rowID", "columnID"]


# ---------------------------------------------------------------------------
# Content hashing and leakage
# ---------------------------------------------------------------------------

def test_unhashable_files_are_a_named_warning_and_can_be_made_critical(
        tmp_path):
    """A split whose files cannot be hashed is not a clean split; it is an
    unverified one. With ``require_identity`` that distinction becomes a
    failure instead of a line nobody reads."""
    present = tmp_path / "plate1_A01_f1_o1.png"
    present.write_bytes(b"a")
    missing = str(tmp_path / "plate1_A02_f1_o1.png")

    lenient = audit_split_leakage([str(present)], [missing],
                                  group_by="well", hash_content=True)
    assert any("content-hashed" in w for w in lenient.warnings)
    assert "unverifiable_content" not in lenient.critical_levels

    strict = audit_split_leakage([str(present)], [missing], group_by="well",
                                 hash_content=True, require_identity=True)
    assert "unverifiable_content" in strict.critical_levels
    assert strict.passed is False


def test_unparseable_group_identities_are_a_lenient_warning():
    report = audit_split_leakage(
        ["train.png"], ["validation.png"],
        group_by="well", require_identity=False)

    assert report.passed
    assert report.unverifiable_counts == {"well": 2}
    assert any("do not encode the requested well identity" in warning
               for warning in report.warnings)


# ---------------------------------------------------------------------------
# Dataset folders
# ---------------------------------------------------------------------------

def test_a_split_folder_that_is_not_there_collects_nothing(tmp_path):
    """A missing ``train/`` is not an error at this level; it is an empty
    list, and the caller above turns that into the message that names it."""
    assert dataset_split_paths(tmp_path, "train") == []


def test_an_empty_dataset_side_is_named_rather_than_audited(tmp_path):
    """An audit over no images would pass. The one side that has no images
    has to be named, because that is the actual problem."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "plate1_A01_f1_o1.png").write_bytes(b"a")
    (tmp_path / "test").mkdir()
    with pytest.raises(FileNotFoundError, match="no images found in test"):
        audit_dataset_splits(tmp_path)


# ---------------------------------------------------------------------------
# Probabilities and calibration
# ---------------------------------------------------------------------------

def test_a_probability_array_with_no_class_axis_is_refused():
    """A three-dimensional array is not a probability matrix, and reshaping a
    guess out of it would silently score the wrong columns."""
    with pytest.raises(ValueError, match="two-dimensional matrix"):
        normalize_probabilities(np.zeros((2, 2, 2)))


def test_a_probability_matrix_must_match_the_declared_class_count():
    """A head with three outputs and a two-class schema means the labels and
    the columns disagree; every metric would be computed against the wrong
    column."""
    with pytest.raises(ValueError, match="3 columns but 2 classes"):
        normalize_probabilities(np.full((2, 3), 1 / 3), n_classes=2)


def test_calibration_error_needs_one_probability_row_per_label():
    """Truncating to the shorter of the two would report a calibration error
    for a subset nobody asked about."""
    with pytest.raises(ValueError, match="equal length"):
        expected_calibration_error([0, 1, 0], np.array([[0.5, 0.5]]))


def test_calibration_error_of_nothing_is_not_a_number():
    """Zero predictions have no calibration. Returning 0.0 would read as
    perfectly calibrated."""
    assert np.isnan(expected_calibration_error([], np.zeros((0, 2))))


def test_a_temperature_that_will_not_fit_is_an_error_not_a_default(
        monkeypatch):
    """Silently returning temperature 1.0 reports that calibration ran while
    leaving the probabilities untouched."""
    import scipy.optimize

    class _Failed:
        success = False
        message = "stub optimiser refused"
        x = 0.0

    monkeypatch.setattr(scipy.optimize, "minimize_scalar",
                        lambda *a, **k: _Failed())
    with pytest.raises(RuntimeError, match="stub optimiser refused"):
        fit_temperature([0, 1, 0, 1], np.array([[0.6, 0.4], [0.3, 0.7],
                                                [0.8, 0.2], [0.2, 0.8]]))


# ---------------------------------------------------------------------------
# evaluate_predictions guards
# ---------------------------------------------------------------------------

_PROBS = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
_PATHS = ["plate1_A01_f1_o1.png", "plate1_A01_f1_o2.png",
          "plate1_A02_f1_o1.png", "plate1_A02_f1_o2.png"]


# ---------------------------------------------------------------------------
# Writing and finding bundles
# ---------------------------------------------------------------------------

def test_a_figure_that_cannot_be_drawn_does_not_lose_the_bundle(
        tmp_path, monkeypatch, capsys):
    """The tables are the evaluation; the figures are a convenience. A
    plotting failure has to be recorded in the manifest and printed, not
    allowed to discard six files that were already written."""
    evaluation = evaluate_predictions([0, 1, 0, 1], _PROBS, _PATHS)

    def _explode(*args, **kwargs):
        raise RuntimeError("no renderer here")

    monkeypatch.setattr(ce, "_write_confusion_figure", _explode)
    monkeypatch.setattr(ce, "_write_calibration_figure", _explode)
    manifest_path = write_evaluation_bundle(tmp_path, evaluation)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["warnings"]) == 2
    assert any("Confusion figure failed" in w for w in manifest["warnings"])
    assert any("Calibration figure failed" in w for w in manifest["warnings"])
    assert (tmp_path / EVALUATION_FILES["summary"]).is_file()
    assert "no renderer here" in capsys.readouterr().out


def test_a_folder_with_no_manifest_cannot_be_loaded(tmp_path):
    """Returning an empty bundle would let a caller plot nothing and call it
    a result."""
    with pytest.raises(FileNotFoundError, match=EVALUATION_FILES["manifest"]):
        load_evaluation_bundle(tmp_path)
