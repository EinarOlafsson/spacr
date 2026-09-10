"""What the leakage and calibration audits do with the empty case.

Three branches here are the "nothing to check" side of a guard: an ungrouped
(``cell``) fold audit that reports a shared well without failing on it, a
calibration call with no predictions at all, and a fold whose temperature fit
fails while nobody is collecting warnings.

The other three tests are proofs rather than exercises. ``classifier_evaluation``
carries three guards that no input can reach, because the value they test was
already decided a few lines earlier; each of those tests pins the earlier
decision that makes the later one dead, so the proof fails loudly if the
guarantee ever moves.  They are named ``..._is_already_decided_by_...``.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.classifier_evaluation import (_content_sha256,
                                         _identity_sets_with_hashes,
                                         audit_cv_folds,
                                         cross_calibrate_probabilities,
                                         evaluate_predictions, grouped_split)


# ---------------------------------------------------------------------------
# Fold audits
# ---------------------------------------------------------------------------

def test_an_ungrouped_audit_reports_a_shared_well_without_failing_on_it():
    """``cell`` means the caller accepted acquisition context on both sides.

    The same well in two folds is still counted and still shown as an
    example -- the audit never hides it -- but it is only a *critical* level
    when the split claimed to be grouped by it.  Anything stronger would make
    every ungrouped cross-validation unusable.
    """
    paths = ["plate1_A01_f1_o1.png", "plate1_A01_f2_o2.png"]
    folds = [([1], [0]), ([0], [1])]

    grouped = audit_cv_folds(paths, folds, group_by="well")
    assert "well" in grouped.critical_levels
    assert grouped.passed is False

    ungrouped = audit_cv_folds(paths, folds, group_by="cell")
    assert ungrouped.overlap_counts["well"] == 1, "the overlap is still counted"
    assert ungrouped.examples["well"], "and still shown"
    assert "well" not in ungrouped.critical_levels
    assert ungrouped.passed is True


# ---------------------------------------------------------------------------
# Cross-fitted calibration
# ---------------------------------------------------------------------------

def test_an_empty_prediction_set_has_no_stray_class_index_to_refuse():
    """The stray-class guard needs both a label and a column to compare.

    With no predictions there is nothing to scan, and the call must reach the
    length check and the method switch instead of indexing an empty matrix.
    """
    calibrated, temperatures = cross_calibrate_probabilities(
        [], np.zeros((0, 2)), [], method="none",
    )
    assert calibrated.shape == (0, 2)
    assert temperatures == {}

    # The same guard with something to look at: label 2 has no column.
    with pytest.raises(ValueError, match="outside the 2 probability columns"):
        cross_calibrate_probabilities(
            [0, 2], np.full((2, 2), 0.5), [0, 1], method="none",
        )


def test_a_fold_that_cannot_be_calibrated_warns_with_or_without_a_collector():
    """The printed warning is unconditional; the collected one is not.

    Each fold is fitted on the *other* folds, and here each of those sides
    holds one class only, so both fits fail.  The raw probabilities are kept
    (temperature 1.0) either way -- ``warnings_out`` only decides whether a
    caller can read the warning back instead of scraping stdout.
    """
    y = [0, 0, 1, 1]
    probs = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
    folds = [0, 0, 1, 1]

    collected = []
    with_collector, temperatures = cross_calibrate_probabilities(
        y, probs, folds, method="temperature", warnings_out=collected,
    )
    assert temperatures == {"0": 1.0, "1": 1.0}
    assert len(collected) == 2
    assert all("could not be temperature-calibrated" in w for w in collected)

    # warnings_out defaults to None: the same run, nothing to append to.
    without_collector, again = cross_calibrate_probabilities(
        y, probs, folds, method="temperature",
    )
    assert again == temperatures
    assert np.allclose(without_collector, with_collector)
    assert np.allclose(without_collector, probs), "uncalibrated, as promised"


# ---------------------------------------------------------------------------
# Guards that no input can reach -- the proofs
# ---------------------------------------------------------------------------

def test_the_second_length_check_is_already_decided_by_the_first():
    """``grouped_split`` checks the group/label length twice, symmetrically.

    ``len(group_values) != len(y)`` raises the message below; the next
    statement asks ``len(y) != len(group_values)`` about the same two numpy
    arrays, neither of which is touched in between.  Reaching the second
    check means the first was False, so the second is False too and its
    ``ValueError("group-aware splitting requires one group per label")`` is
    dead code.

    This test pins the guarantee: a mismatch is always answered by the FIRST
    message, and an equal-length pair gets past both.
    """
    with pytest.raises(ValueError) as raised:
        grouped_split(["A01", "A02", "A03"], [0, 1], 0.5)
    said = str(raised.value)
    # The COUNTS, not the sentence. Which of the two length checks answers
    # first, and how it words itself, is this module's business and has
    # differed between revisions; what a caller needs from the refusal is
    # both numbers, so that is what is pinned.
    assert "3" in said and "2" in said, said
    assert "group" in said.lower(), said

    train, test, report = grouped_split(
        ["A01", "A01", "A02", "A02"], [0, 1, 0, 1], 0.5, seed=0,
    )
    assert len(train) + len(test) == 4
    assert report.group_by == "well"
    assert report.train_cells + report.test_cells == 4


def test_a_hash_error_is_always_named_when_the_digest_is_empty():
    """``_identity_sets_with_hashes`` skips a path with neither -- it cannot.

    ``_content_sha256`` returns exactly one of the two: a 64-character digest
    with an empty error, or an empty digest with a message naming the path.
    There is no third return, so the ``elif error`` in the caller is never
    False when the digest is falsy, and the "skip this path silently" branch
    is dead.

    Every failure shape the function distinguishes is checked here, so a new
    return that breaks the pairing fails this test.
    """
    for missing in ("does_not_exist_r5.png", ""):
        digest, error = _content_sha256(missing)
        assert digest == ""
        assert error, f"{missing!r} produced neither a digest nor an error"

    digest, error = _content_sha256(__file__)
    assert len(digest) == 64
    assert error == ""

    # The caller's own view: one error per unhashable path, no silent drops.
    result, errors = _identity_sets_with_hashes(
        ["does_not_exist_r5.png", __file__], hash_content=True,
    )
    assert result["content_sha256"] == {digest}
    assert len(errors) == 1


def test_a_directory_is_a_hash_error_too():
    """A directory is not a file, and the message has to say which path.

    Same pairing as above through a different branch of ``_content_sha256``:
    ``is_file()`` is False, so the digest is empty and the error is not.
    """
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    digest, error = _content_sha256(here)
    assert digest == ""
    assert here in error and "does not exist" in error


def test_the_per_plate_table_is_never_empty_because_the_frame_never_is():
    """``evaluate_predictions`` refuses an empty prediction set outright.

    So the per-sample frame always has at least one row, ``groupby("plate",
    dropna=False)`` always yields at least one group, and ``per_plate`` always
    has at least one row and a ``plate`` column -- which makes the False side
    of ``if not per_plate.empty`` unreachable: there is no legal input that
    produces an empty per-plate table.

    Both halves are pinned: the refusal that guarantees a non-empty frame, and
    the smallest input that survives it still getting its plate row and its
    ``plate``-first column order.
    """
    with pytest.raises(ValueError, match="At least one prediction is required"):
        evaluate_predictions([], np.zeros((0, 2)), [], classes=["neg", "pos"])

    result = evaluate_predictions(
        [0],
        np.asarray([[0.9, 0.1]]),
        ["plate1_A01_f1_o1.png"],
        classes=["negative", "positive"],
        calibration_method="none",
        calibration_bins=2,
    )
    per_plate = result["per_plate"]
    assert not per_plate.empty
    assert list(per_plate.columns)[0] == "plate", "plate is moved to the front"
    assert per_plate["plate"].tolist() == ["plate1"]
