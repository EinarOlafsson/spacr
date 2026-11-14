"""Table-driven tests of the guards in ``spacr.classifier_evaluation``.

The module exists to stop a number that means nothing from being reported as
a score, so every unit here is tested on what it refuses and on the exact
value it returns instead: the groups a split reads off a frame (column, index
or crop paths), the grouped train/test split, the content hash of a crop, the
cross-validation fold audit, cross-fitted calibration, the calibration table,
the guards of ``evaluate_predictions`` and how an evaluation bundle is found.
One general test per unit, each walking a case table with pytest's
``subtests`` fixture and asserting exact messages and values.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from spacr import classifier_evaluation as ce
from spacr.classifier_evaluation import (EVALUATION_FILES, LeakageError,
                                         audit_cv_folds, calibration_table,
                                         cross_calibrate_probabilities,
                                         evaluate_predictions,
                                         find_evaluation_bundles,
                                         grouped_split, split_group_values)
from tests.refused_access import refused

SEP = "\x1f"
PROBS4 = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
PATHS4 = ["plate1_A01_f1_o1.png", "plate1_A01_f1_o2.png",
          "plate1_A02_f1_o1.png", "plate1_A02_f1_o2.png"]
SWAP = [([1], [0]), ([0], [1])]


def _indexed(keys):
    return pd.DataFrame({"value": [1.0] * len(keys)},
                        index=pd.Index(keys, name="prcfo"))


def _well_paths(n):
    return [f"plate1_A{i + 1:02d}_f1_o1.png" for i in range(n)]


def test_split_group_values_reads_column_index_paths_or_rows(subtests):
    """Groups come from prcfo (column or index), crop paths or row ids; gaps refuse."""
    two = ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f2_o2"]
    cases = [
        ("cell_from_index", dict(group_by="cell", frame=_indexed(two[:1] + ["plate1_r1_c1_f1_o2"]), table="cells"),
         ("cell", ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"])),
        ("well_from_index", dict(group_by="well", frame=_indexed(two), table="cells"),
         ("well", [f"plate1{SEP}r1{SEP}c1"] * 2)),
        ("field_from_index", dict(group_by="field", frame=_indexed(two), table="cells"),
         ("field", [f"plate1{SEP}r1{SEP}c1{SEP}f1", f"plate1{SEP}r1{SEP}c1{SEP}f2"])),
        ("plate_from_column", dict(group_by="plate", table="cells",
                                   frame=pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1", "plate2_r1_c1_f2_o2"]})),
         ("plate", ["plate1", "plate2"])),
        ("cell_from_augmented_paths", dict(group_by="cell", frame=pd.DataFrame({"value": [1.0, 2.0]}),
                                           paths=["a_aug1.png", "b_rot90.png"]),
         ("cell", ["a", "b"])),
        ("cell_from_row_ids", dict(group_by="cell", frame=pd.DataFrame({"value": [1.0, 2.0]})),
         ("cell", [0, 1])),
        ("well_from_png_paths", dict(group_by="well", table="pngs", frame=pd.DataFrame(
            {"png_path": ["plate1_A01_f1_o1.png", "plate1_A01_f2_o2.png", "plate1_A02_f1_o1.png"]})),
         ("well", ["plate1_A01", "plate1_A01", "plate1_A02"])),
    ]
    for case_id, kwargs, (level, values) in cases:
        with subtests.test(msg=case_id):
            got_level, got = split_group_values(**kwargs)
            assert (got_level, got.tolist()) == (level, values)

    refusals = [
        ("nothing_to_read", dict(group_by="cell"), "split_group_values needs a frame or paths"),
        ("blank_key_in_column", dict(group_by="cell", table="cells",
                                     frame=pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1", "   "]})),
         "Cannot split cells by cell: at least one row has no object identity."),
        ("blank_key_on_index", dict(group_by="cell", table="cells",
                                    frame=_indexed(["plate1_r1_c1_f1_o1", " "])),
         "Cannot split cells by cell: at least one row has no object identity."),
        ("malformed_key", dict(group_by="well", table="cells",
                               frame=pd.DataFrame({"prcfo": ["not-an-object"]})),
         "Cannot split cells by well: prcfo at row 0 is not a valid object identity "
         "('not-an-object'): 'not-an-object' is not a prcfo: expected at least "
         "plate_row_column_field_object, got 1 part(s)."),
    ]
    for case_id, kwargs, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                split_group_values(**kwargs)


def test_grouped_split_keeps_groups_apart_or_refuses(subtests, monkeypatch):
    """A grouped split never lets a group cross; impossible inputs refuse by name."""
    cases = [
        # A 99% hold-out consumes every group, so the fold candidates win.
        ("holdout_too_large_falls_back_to_folds",
         (["w1", "w1", "w2", "w2", "w3", "w3", "w4", "w4"], [0, 1] * 4, 0.99), {"group_by": "well"},
         ([0, 1, 6, 7], [2, 3, 4, 5],
          ce.SplitReport(group_by="well", requested_fraction=0.99, cell_fraction=0.5,
                         group_fraction=0.5, train_cells=4, test_cells=4, train_groups=2,
                         test_groups=2, total_groups=4,
                         rule="StratifiedGroupKFold(2) over 4 well groups; no group appears on both sides"))),
        ("two_wells_half_each",
         (["A01", "A01", "A02", "A02"], [0, 1, 0, 1], 0.5), {"seed": 0},
         ([0, 1], [2, 3],
          ce.SplitReport(group_by="well", requested_fraction=0.5, cell_fraction=0.5,
                         group_fraction=0.5, train_cells=2, test_cells=2, train_groups=1,
                         test_groups=1, total_groups=2,
                         rule="StratifiedGroupKFold(2) over 2 well groups; no group appears on both sides"))),
    ]
    for case_id, args, kwargs, (train, test, report) in cases:
        with subtests.test(msg=case_id):
            got_train, got_test, got_report = grouped_split(*args, **kwargs)
            assert (got_train.tolist(), got_test.tolist(), got_report) == (train, test, report)

    refusals = [
        ("one_cell", (["w1"], [0], 0.5), {"group_by": "cell"},
         "a train/test split needs at least two labelled cells"),
        ("blank_group", (["w1", "w2", "", "w2"], [0, 1, 0, 1], 0.5), {"group_by": "well"},
         "Cannot split by well: at least one cell has no group identity, so independence cannot be verified."),
        ("more_groups_than_labels", (["A01", "A02", "A03"], [0, 1], 0.5), {},
         "group-aware splitting requires one group per label; the split has 3 groups for 2 labels"),
    ]
    for case_id, args, kwargs, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                grouped_split(*args, **kwargs)

    with subtests.test(msg="crossing_splitter_is_caught"):
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
            grouped_split(["w1", "w1", "w2", "w2"], [0, 1, 0, 1], 0.5, group_by="well")


def test_content_sha256_returns_exactly_one_of_digest_or_error(subtests, tmp_path):
    """A file hashes to its SHA-256; anything else is an empty digest plus a named error."""
    readable = tmp_path / "crop.png"
    readable.write_bytes(b"pixels")
    blocked = tmp_path / "blocked.png"
    blocked.write_bytes(b"pixels")
    cases = [
        ("readable_file", lambda: ce._content_sha256(readable),
         (hashlib.sha256(b"pixels").hexdigest(), "")),
        ("missing_file", lambda: ce._content_sha256("/no/such/crop/for/spacr.png"),
         ("", "/no/such/crop/for/spacr.png: file does not exist")),
        ("empty_path", lambda: ce._content_sha256(""), ("", "no file path given (empty path)")),
        ("directory", lambda: ce._content_sha256(tmp_path), ("", f"{tmp_path}: file does not exist")),
    ]
    for case_id, call, expected in cases:
        with subtests.test(msg=case_id):
            assert call() == expected

    with subtests.test(msg="unreadable_file"):
        with refused(blocked, 0o000, windows="read"):
            digest, error = ce._content_sha256(blocked)
        assert digest == ""
        assert error.startswith(f"{blocked}: PermissionError: ")

    with subtests.test(msg="caller_gets_one_error_per_unhashable_path"):
        sets, errors = ce._identity_sets_with_hashes(
            [str(tmp_path / "gone.png"), str(readable)], hash_content=True)
        assert sets["content_sha256"] == {hashlib.sha256(b"pixels").hexdigest()}
        assert errors == [f"{tmp_path / 'gone.png'}: file does not exist"]


def test_audit_cv_folds_reports_every_broken_partition(subtests, tmp_path):
    """Each fold defect is a named critical level, warning or refusal; clean folds pass."""
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first.write_bytes(b"first crop")
    second.write_bytes(b"second crop")
    shared_well = ["plate1_A01_f1_o1.png", "plate1_A01_f2_o2.png"]
    cases = [
        ("validated_twice", (_well_paths(3), [([1, 2], [0]), ([1], [0, 2])]),
         dict(require_identity=False),
         dict(critical_levels=["validation_membership_missing", "validation_membership_duplicate",
                               "exact", "augmentation_family", "object", "well"],
              passed=False, warnings=[])),
        ("augmented_copy_with_other_label",
         (["plate1_A01_f1_o1.png", "plate1_A01_f1_o1_aug1.png"], SWAP),
         dict(labels=[0, 1], require_identity=False),
         dict(critical_levels=["augmentation_family", "object", "well", "conflicting_labels"],
              passed=False,
              warnings=["Related crops carry different class labels; fix annotations before "
                        "training even when all copies happen to be in one fold."])),
        ("no_well_in_path_lenient", (["nowell.png", "alsonowell.png"], SWAP),
         dict(group_by="well", require_identity=False),
         dict(critical_levels=[], passed=True, unverifiable_counts={"well": 2},
              warnings=["2 sample(s) do not encode well identity."])),
        ("no_well_in_path_strict", (["nowell.png", "alsonowell.png"], SWAP),
         dict(group_by="well", require_identity=True),
         dict(critical_levels=["unverifiable_well"], passed=False)),
        ("unhashable_lenient", (_well_paths(2), SWAP), dict(hash_content=True, require_identity=False),
         dict(critical_levels=[], passed=True, unverifiable_counts={"content_sha256": 2},
              warnings=["2 sample(s) could not be hashed."],
              hash_errors=["plate1_A01_f1_o1.png: file does not exist",
                           "plate1_A02_f1_o1.png: file does not exist"])),
        ("shared_well_grouped_by_well", (shared_well, SWAP), dict(group_by="well"),
         dict(critical_levels=["well"], passed=False)),
        # An ungrouped audit still counts and shows the shared well, but does not fail on it.
        ("shared_well_ungrouped", (shared_well, SWAP), dict(group_by="cell"),
         dict(critical_levels=[], passed=True,
              overlap_counts={"exact": 0, "content_sha256": 0, "augmentation_family": 0,
                              "object": 0, "field": 0, "well": 1, "plate": 1})),
        ("readable_distinct_files", ([first, second], SWAP), dict(group_by="cell", hash_content=True),
         dict(critical_levels=[], passed=True, hash_errors=[], unverifiable_counts={})),
    ]
    for case_id, args, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            audit = audit_cv_folds(*args, **kwargs)
            assert {k: getattr(audit, k) for k in expected} == expected
            assert audit.to_dict()["passed"] is expected["passed"]

    with subtests.test(msg="ungrouped_examples_still_name_the_well"):
        audit = audit_cv_folds(shared_well, SWAP, group_by="cell")
        assert audit.examples["well"] == ["plate1_A01 -> folds 1,2"]

    refusals = [
        ("index_out_of_range", (_well_paths(2), [([0], [5])]), {}, ValueError,
         "fold 1 contains out-of-range indexes [5]"),
        ("train_and_validate_one_index", (_well_paths(2), [([0, 1], [1])]), {}, LeakageError,
         "fold 1 puts indexes in both train and validation: [1]"),
        ("labels_wrong_length", (_well_paths(2), [([0], [1])]), {"labels": [0]}, ValueError,
         "labels must have one value per path"),
        ("unhashable_strict_raises", (_well_paths(2), SWAP),
         dict(hash_content=True, require_identity=True, raise_on_leakage=True), LeakageError,
         "Cross-validation leakage audit failed: unverifiable_content"),
    ]
    for case_id, args, kwargs, error, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(error, match=f"^{re.escape(message)}$"):
                audit_cv_folds(*args, **kwargs)


def test_cross_calibrate_probabilities_refuses_or_returns_exact_matrices(subtests):
    """Calibration refuses stray labels, bad lengths, one fold and unknown methods."""
    probs = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
    cases = [
        ("none_returns_a_copy", ([0, 1], np.array([[0.5, 0.5], [0.25, 0.75]]), [1, 2]),
         {"method": "none"}, np.array([[0.5, 0.5], [0.25, 0.75]]), {}),
        ("empty_set_reaches_the_method_switch", ([], np.zeros((0, 2)), []),
         {"method": "none"}, np.zeros((0, 2)), {}),
        # Each fold's other side holds one class, so both fits fail and keep raw probabilities.
        ("unfittable_folds_keep_raw", ([0, 0, 1, 1], probs[[0, 2, 1, 3]], [0, 0, 1, 1]),
         {"method": "temperature"}, probs[[0, 2, 1, 3]], {"0": 1.0, "1": 1.0}),
    ]
    for case_id, args, kwargs, matrix, temperatures in cases:
        with subtests.test(msg=case_id):
            calibrated, got_temperatures = cross_calibrate_probabilities(*args, **kwargs)
            assert calibrated is not args[1]
            assert calibrated.shape == matrix.shape
            np.testing.assert_allclose(calibrated, matrix)
            assert got_temperatures == temperatures

    with subtests.test(msg="unfittable_fold_warnings_collected"):
        collected = []
        cross_calibrate_probabilities([0, 0, 1, 1], probs[[0, 2, 1, 3]], [0, 0, 1, 1],
                                      method="temperature", warnings_out=collected)
        tail = ("could not be temperature-calibrated from the other folds (ValueError: "
                "Temperature calibration needs at least two classes and two samples.); "
                "its raw probabilities were retained.")
        assert len(collected) == 2
        assert all(w.startswith("Held-out fold ") and w.endswith(tail) for w in collected)

    refusals = [
        ("label_without_column", ([0, 1, 0, 5], probs, [1, 1, 2, 2]), {},
         "y_true contains class indices [5] outside the 2 probability columns; "
         "calibration would silently do nothing."),
        ("label_without_column_method_none", ([0, 2], np.full((2, 2), 0.5), [0, 1]), {"method": "none"},
         "y_true contains class indices [2] outside the 2 probability columns; "
         "calibration would silently do nothing."),
        ("fold_ids_short", ([0, 1], probs[:2], [1]), {},
         "y_true, probabilities, and fold_ids must have equal length."),
        ("unknown_method", ([0, 1], probs[:2], [1, 2]), {"method": "isotonic"},
         "calibration_method must be 'none' or 'temperature', not 'isotonic'."),
        ("one_fold", ([0, 1], probs[:2], [1, 1]), {},
         "Cross-fitted calibration requires predictions from at least two folds."),
    ]
    for case_id, args, kwargs, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                cross_calibrate_probabilities(*args, **kwargs)


def test_calibration_table_and_evaluate_predictions_refuse_misaligned_inputs(subtests):
    """Lengths, class names and labels must agree with the probability columns."""
    two = np.array([[0.6, 0.4], [0.3, 0.7]])
    refusals = [
        ("table_lengths", lambda: calibration_table([0, 1, 0], np.array([[0.5, 0.5]])),
         "y_true and probabilities must have equal length."),
        ("table_stray_label", lambda: calibration_table([0, 7], two),
         "y_true contains class indices [7] outside the 2 probability columns."),
        ("table_class_names", lambda: calibration_table([0, 1], two, classes=["only_one"]),
         "classes and probability columns must have equal length."),
        ("eval_paths_short", lambda: evaluate_predictions([0, 1, 0, 1], PROBS4, PATHS4[:3]),
         "y_true, probabilities, and sample_paths must have equal length."),
        ("eval_class_names", lambda: evaluate_predictions([0, 1, 0, 1], PROBS4, PATHS4, classes=["a"]),
         "classes and probability columns must have equal length."),
        ("eval_empty", lambda: evaluate_predictions([], np.zeros((0, 2)), [], classes=["neg", "pos"]),
         "At least one prediction is required."),
        ("eval_label_outside_schema", lambda: evaluate_predictions([0, 1, 0, 9], PROBS4, PATHS4),
         "y_true contains a label outside the class schema."),
        ("eval_fold_ids_short", lambda: evaluate_predictions([0, 1, 0, 1], PROBS4, PATHS4, fold_ids=[1, 1]),
         "fold_ids must have the same length as y_true and probabilities."),
    ]
    for case_id, call, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                call()

    with subtests.test(msg="smallest_input_gets_its_plate_row"):
        result = evaluate_predictions([0], np.asarray([[0.9, 0.1]]), ["plate1_A01_f1_o1.png"],
                                      classes=["negative", "positive"],
                                      calibration_method="none", calibration_bins=2)
        per_plate = result["per_plate"]
        assert list(per_plate.columns)[:2] == ["plate", "n"]
        assert per_plate[["plate", "n", "accuracy", "mean_confidence"]].to_dict("list") == {
            "plate": ["plate1"], "n": [1], "accuracy": [1.0], "mean_confidence": [0.9]}


def test_find_evaluation_bundles_resolves_manifest_file_or_folder(subtests, tmp_path):
    """The manifest itself, or any file beside it, names the bundle; a missing path refuses."""
    manifest = tmp_path / EVALUATION_FILES["manifest"]
    manifest.write_text("{}", encoding="utf-8")
    summary = tmp_path / EVALUATION_FILES["summary"]
    summary.write_text("{}", encoding="utf-8")
    for case_id, source in (("manifest_named_directly", manifest),
                            ("sibling_file_points_at_folder", summary)):
        with subtests.test(msg=case_id):
            assert find_evaluation_bundles(source) == [manifest]

    with subtests.test(msg="missing_source"):
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError,
                           match=f"^{re.escape(f'Evaluation source does not exist: {missing}')}$"):
            find_evaluation_bundles(missing)
