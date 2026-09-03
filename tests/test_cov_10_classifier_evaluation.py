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
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import classifier_evaluation as ce
from spacr.classifier_evaluation import (
    EVALUATION_FILES,
    LeakageError,
    audit_cv_folds,
    audit_dataset_splits,
    audit_split_leakage,
    calibration_table,
    cross_calibrate_probabilities,
    dataset_split_paths,
    evaluate_predictions,
    expected_calibration_error,
    find_evaluation_bundles,
    fit_temperature,
    grouped_split,
    load_evaluation_bundle,
    normalize_probabilities,
    split_columns_for,
    split_group_values,
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
        if f":ivar {item.name}:" not in documentation
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


def test_group_values_need_something_to_read_them_from():
    """With neither a frame nor paths there is no identity to verify, and
    inventing one row per index would make a leaking split look grouped."""
    with pytest.raises(ValueError, match="needs a frame or paths"):
        split_group_values(group_by="cell")


def test_a_blank_object_key_in_a_column_is_refused():
    """An empty ``prcfo`` is not an object identity. Accepting it would put
    several unrelated cells in one group."""
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1", "   "]})
    with pytest.raises(ValueError, match="no object identity"):
        split_group_values(group_by="cell", frame=frame, table="cells")


def test_an_object_key_carried_on_the_index_is_read_from_the_index():
    """Measurement frames are routinely indexed by ``prcfo`` rather than
    carrying it as a column. Both spellings have to give the same groups."""
    frame = pd.DataFrame({"value": [1.0, 2.0]},
                         index=pd.Index(["plate1_r1_c1_f1_o1",
                                         "plate1_r1_c1_f1_o2"], name="prcfo"))
    level, values = split_group_values(group_by="cell", frame=frame,
                                       table="cells")
    assert level == "cell"
    assert list(values) == ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"]


def test_a_blank_object_key_on_the_index_is_refused_too():
    """The index spelling gets the same check as the column spelling; a gap
    the column form rejects must not pass just because it moved."""
    frame = pd.DataFrame({"value": [1.0, 2.0]},
                         index=pd.Index(["plate1_r1_c1_f1_o1", " "],
                                        name="prcfo"))
    with pytest.raises(ValueError, match="no object identity"):
        split_group_values(group_by="cell", frame=frame, table="cells")


def test_a_well_split_reads_the_index_when_prcfo_is_not_a_column():
    """Grouping by well off an index-keyed frame has to parse the same
    identity; falling through to row numbers would defeat the grouping."""
    frame = pd.DataFrame({"value": [1.0, 2.0]},
                         index=pd.Index(["plate1_r1_c1_f1_o1",
                                         "plate1_r1_c1_f2_o2"], name="prcfo"))
    level, values = split_group_values(group_by="well", frame=frame,
                                       table="cells")
    assert level == "well"
    assert len(set(values)) == 1


def test_a_malformed_object_key_names_its_row_and_value():
    frame = pd.DataFrame({"prcfo": ["not-an-object"]})

    with pytest.raises(ValueError, match=r"row 0.*not-an-object"):
        split_group_values(group_by="well", frame=frame, table="cells")


def test_a_cell_split_uses_paths_or_row_ids_when_object_keys_are_absent():
    frame = pd.DataFrame({"value": [1.0, 2.0]})

    level, families = split_group_values(
        group_by="cell", frame=frame,
        paths=["a_aug1.png", "b_rot90.png"])
    fallback_level, row_ids = split_group_values(
        group_by="cell", frame=frame)

    assert level == fallback_level == "cell"
    assert families.tolist() == ["a", "b"]
    assert row_ids.tolist() == [0, 1]


def test_a_frame_of_crop_paths_supplies_the_groups_when_metadata_is_absent():
    """A png table has no plate/row/column columns, but every path encodes
    the well. Using it is what keeps an image-only split grouped."""
    frame = pd.DataFrame({"png_path": ["plate1_A01_f1_o1.png",
                                       "plate1_A01_f2_o2.png",
                                       "plate1_A02_f1_o1.png"]})
    level, values = split_group_values(group_by="well", frame=frame,
                                       table="pngs")
    assert level == "well"
    assert len(set(values)) == 2


# ---------------------------------------------------------------------------
# grouped_split
# ---------------------------------------------------------------------------

def test_one_labelled_cell_cannot_be_split():
    """There is no train/test division of a single row, and returning an
    empty test side would score a model against nothing."""
    with pytest.raises(ValueError, match="at least two labelled cells"):
        grouped_split(["w1"], [0], 0.5, group_by="cell")


def test_a_blank_group_identity_stops_a_grouped_split():
    """Independence is a claim about the groups. One row with no group id
    means the claim cannot be checked, so the split is refused instead of
    quietly treating the blank as its own well."""
    with pytest.raises(ValueError, match="no group identity"):
        grouped_split(["w1", "w2", "", "w2"], [0, 1, 0, 1], 0.5,
                      group_by="well")


def test_a_holdout_that_leaves_no_training_groups_falls_back_to_the_folds():
    """``GroupShuffleSplit`` refuses a fraction that consumes every group.
    That refusal must not end the search: the fold-based candidates already
    collected are still valid splits."""
    groups = ["w1", "w1", "w2", "w2", "w3", "w3", "w4", "w4"]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    train_idx, test_idx, report = grouped_split(groups, labels, 0.99,
                                                group_by="well")
    assert report.rule.startswith("StratifiedGroupKFold")
    assert len(train_idx) and len(test_idx)
    assert not (set(np.asarray(groups)[train_idx])
                & set(np.asarray(groups)[test_idx]))


def test_a_splitter_that_lets_groups_cross_is_caught_before_it_is_used(
        monkeypatch):
    """sklearn promises disjoint groups; this asserts spaCR does not take
    that promise on trust. A crossing split is what silently turns a leakage
    audit green, so it has to raise here rather than be reported as a
    well-grouped split."""
    import sklearn.model_selection as skms

    class _NoFolds:
        def __init__(self, *args, **kwargs):
            raise ValueError("no folds from this stub")

    class _Crossing:
        def __init__(self, *args, **kwargs):
            pass

        def split(self, indices, y, groups):
            yield np.array([0, 1, 2]), np.array([1, 2, 3])

    monkeypatch.setattr(skms, "StratifiedGroupKFold", _NoFolds)
    monkeypatch.setattr(skms, "GroupShuffleSplit", _Crossing)
    with pytest.raises(RuntimeError, match="crossed the train/test boundary"):
        grouped_split(["w1", "w1", "w2", "w2"], [0, 1, 0, 1], 0.5,
                      group_by="well")


# ---------------------------------------------------------------------------
# Content hashing and leakage
# ---------------------------------------------------------------------------

def test_a_path_that_is_not_a_file_is_reported_not_hashed():
    """A missing crop cannot be compared byte for byte. The audit has to know
    that, because "no hash" and "hash that matched nothing" are different
    claims about the split."""
    digest, error = ce._content_sha256("/no/such/crop/for/spacr.png")
    assert digest == ""
    assert "does not exist" in error


def test_a_file_that_cannot_be_read_is_reported_with_its_error(tmp_path):
    """An unreadable file exists but yields no digest. Letting the OSError
    escape would abort the whole audit over one bad permission bit."""
    blocked = tmp_path / "blocked.png"
    blocked.write_bytes(b"pixels")
    blocked.chmod(0o000)
    try:
        digest, error = ce._content_sha256(blocked)
    finally:
        blocked.chmod(0o600)
    assert digest == ""
    assert "PermissionError" in error


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
# Cross-validation fold audits
# ---------------------------------------------------------------------------

def _paths(n, plate="plate1"):
    return [f"{plate}_A{i + 1:02d}_f1_o1.png" for i in range(n)]


def test_a_fold_index_outside_the_sample_range_is_refused():
    """An index the paths do not have is a fold table built against another
    dataset. Ignoring it would audit a partition that is not the one used."""
    with pytest.raises(ValueError, match="out-of-range"):
        audit_cv_folds(_paths(2), [([0], [5])])


def test_a_fold_that_trains_and_validates_on_one_index_is_leakage():
    """The same row on both sides of one fold is the definition of the thing
    this audit exists to catch."""
    with pytest.raises(LeakageError, match="both train and validation"):
        audit_cv_folds(_paths(2), [([0, 1], [1])])


def test_labels_must_have_one_value_per_path():
    """A label vector of the wrong length silently mislabels every sample
    after the first missing one."""
    with pytest.raises(ValueError, match="one value per path"):
        audit_cv_folds(_paths(2), [([0], [1])], labels=[0])


def test_an_index_validated_twice_is_a_broken_partition():
    """Each sample must be held out exactly once. Validating one twice
    averages a sample into the score twice and shrinks the apparent
    variance."""
    audit = audit_cv_folds(_paths(3), [([1, 2], [0]), ([1], [0, 2])],
                           require_identity=False)
    assert "validation_membership_duplicate" in audit.critical_levels
    assert audit.passed is False


def test_related_crops_with_disagreeing_labels_are_reported(tmp_path):
    """An augmented copy carrying a different class than its original is a
    broken annotation. It has to be named before training, because the model
    is being taught both answers for one object."""
    paths = ["plate1_A01_f1_o1.png", "plate1_A01_f1_o1_aug1.png"]
    audit = audit_cv_folds(paths, [([1], [0]), ([0], [1])],
                           labels=[0, 1], require_identity=False)
    assert "conflicting_labels" in audit.critical_levels
    assert any("different class labels" in w for w in audit.warnings)


def test_paths_that_do_not_encode_the_group_are_counted_and_can_fail(tmp_path):
    """A filename with no well in it cannot be checked for well leakage. The
    count is always reported; ``require_identity`` decides whether an
    uncheckable split is allowed to pass."""
    paths = ["nowell.png", "alsonowell.png"]
    lenient = audit_cv_folds(paths, [([1], [0]), ([0], [1])],
                             group_by="well", require_identity=False)
    assert lenient.unverifiable_counts.get("well") == 2
    assert any("do not encode well identity" in w for w in lenient.warnings)

    strict = audit_cv_folds(paths, [([1], [0]), ([0], [1])],
                            group_by="well", require_identity=True)
    assert "unverifiable_well" in strict.critical_levels


def test_files_that_cannot_be_hashed_are_counted_and_can_fail():
    """The content check is the only one that survives a rename. A file it
    could not read is an unverified sample, and with ``require_identity`` an
    unverified sample fails the audit."""
    paths = ["plate1_A01_f1_o1.png", "plate1_A02_f1_o1.png"]
    lenient = audit_cv_folds(paths, [([1], [0]), ([0], [1])],
                             hash_content=True, require_identity=False)
    assert lenient.unverifiable_counts.get("content_sha256") == 2
    assert any("could not be hashed" in w for w in lenient.warnings)
    assert len(lenient.hash_errors) == 2

    with pytest.raises(LeakageError, match="unverifiable_content"):
        audit_cv_folds(paths, [([1], [0]), ([0], [1])], hash_content=True,
                       require_identity=True, raise_on_leakage=True)


def test_readable_distinct_fold_files_hash_and_serialize_cleanly(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first crop")
    second.write_bytes(b"second crop")

    audit = audit_cv_folds(
        [first, second], [([1], [0]), ([0], [1])],
        group_by="cell", hash_content=True)

    assert audit.hash_errors == []
    assert audit.to_dict()["passed"] is True


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


def test_a_label_with_no_probability_column_is_refused_by_the_calibrator():
    """Every fold's fit would fail, each would fall back to temperature 1.0,
    and the run would report calibrated probabilities that were never
    touched."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValueError, match="outside the 2 probability columns"):
        cross_calibrate_probabilities([0, 1, 0, 5], probs, [1, 1, 2, 2])


def test_the_calibrator_needs_one_fold_id_and_one_label_per_row():
    """Mismatched lengths mean some samples would be calibrated by a
    temperature fitted on themselves."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError, match="equal length"):
        cross_calibrate_probabilities([0, 1], probs, [1])


def test_calibration_switched_off_returns_a_normalized_copy():
    """"none" must still hand back a normalized matrix, and a distinct one:
    the caller may edit it, and mutating the input would corrupt the raw
    probabilities the bundle also reports."""
    probs = np.array([[0.5, 0.5], [0.25, 0.75]])
    calibrated, temperatures = cross_calibrate_probabilities(
        [0, 1], probs, [1, 2], method="none")
    assert temperatures == {}
    assert calibrated is not probs
    np.testing.assert_allclose(calibrated.sum(axis=1), [1.0, 1.0])


def test_an_unknown_calibration_method_is_refused_not_skipped():
    """Skipping an unrecognised method would report calibration as done."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError, match="'none' or 'temperature'"):
        cross_calibrate_probabilities([0, 1], probs, [1, 2],
                                      method="isotonic")


def test_cross_fitted_calibration_needs_more_than_one_fold():
    """With one fold the temperature would be fitted on the same predictions
    it is applied to, which is the bias cross-fitting exists to remove."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError, match="at least two folds"):
        cross_calibrate_probabilities([0, 1], probs, [1, 1])


def test_the_calibration_table_needs_matching_lengths():
    """A shorter label vector would silently bin the wrong rows."""
    with pytest.raises(ValueError, match="equal length"):
        calibration_table([0, 1, 0], np.array([[0.5, 0.5]]))


def test_the_calibration_table_refuses_a_label_with_no_column():
    """Such a label matches no class, so every observed frequency reads 0.0
    and the reliability curve looks catastrophically miscalibrated rather
    than wrong."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError, match="outside the 2 probability columns"):
        calibration_table([0, 7], probs)


def test_the_calibration_table_refuses_the_wrong_number_of_class_names():
    """Names are applied by position; one too few would relabel every column
    after it."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError, match="equal length"):
        calibration_table([0, 1], probs, classes=["only_one"])


# ---------------------------------------------------------------------------
# evaluate_predictions guards
# ---------------------------------------------------------------------------

_PROBS = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
_PATHS = ["plate1_A01_f1_o1.png", "plate1_A01_f1_o2.png",
          "plate1_A02_f1_o1.png", "plate1_A02_f1_o2.png"]


def test_evaluation_needs_a_path_for_every_prediction():
    """The per-plate table is built from the paths; one short and every plate
    after the gap is attributed to the wrong well."""
    with pytest.raises(ValueError, match="equal length"):
        evaluate_predictions([0, 1, 0, 1], _PROBS, _PATHS[:3])


def test_evaluation_refuses_class_names_that_do_not_match_the_columns():
    """Names are positional, so a wrong count renames the wrong classes in
    every table the bundle writes."""
    with pytest.raises(ValueError, match="equal length"):
        evaluate_predictions([0, 1, 0, 1], _PROBS, _PATHS, classes=["a"])


def test_evaluation_of_no_predictions_is_refused():
    """An empty bundle would be written with an accuracy of NaN and read as
    a run that completed."""
    with pytest.raises(ValueError, match="At least one prediction"):
        evaluate_predictions([], np.zeros((0, 2)), [])


def test_evaluation_refuses_a_label_outside_the_class_schema():
    """Clipping the label would move a sample into a class it is not in and
    change every confusion-matrix row."""
    with pytest.raises(ValueError, match="outside the class schema"):
        evaluate_predictions([0, 1, 0, 9], _PROBS, _PATHS)


def test_evaluation_needs_one_fold_id_per_prediction():
    """A short fold vector would calibrate some samples on themselves."""
    with pytest.raises(ValueError, match="fold_ids must have the same length"):
        evaluate_predictions([0, 1, 0, 1], _PROBS, _PATHS, fold_ids=[1, 1])


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


def test_a_manifest_named_directly_is_the_bundle_it_names(tmp_path):
    """Pointing at the manifest file itself is the natural thing to do from a
    file dialog, and it must not be read as "search inside this file"."""
    manifest = tmp_path / EVALUATION_FILES["manifest"]
    manifest.write_text("{}", encoding="utf-8")
    assert find_evaluation_bundles(manifest) == [manifest]


def test_any_other_file_in_a_bundle_points_at_its_folder(tmp_path):
    """Dropping ``summary.json`` on the loader should find the bundle beside
    it rather than reporting that nothing was found."""
    manifest = tmp_path / EVALUATION_FILES["manifest"]
    manifest.write_text("{}", encoding="utf-8")
    other = tmp_path / EVALUATION_FILES["summary"]
    other.write_text("{}", encoding="utf-8")
    assert find_evaluation_bundles(other) == [manifest]


def test_a_missing_evaluation_source_names_the_path(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        find_evaluation_bundles(missing)


def test_a_folder_with_no_manifest_cannot_be_loaded(tmp_path):
    """Returning an empty bundle would let a caller plot nothing and call it
    a result."""
    with pytest.raises(FileNotFoundError, match=EVALUATION_FILES["manifest"]):
        load_evaluation_bundle(tmp_path)
