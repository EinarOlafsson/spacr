"""Six loops in utils that had never gone round for the reason they can.

Three are driven -- a repeated suggestion, a filename the regex does not
recognise, and a merge that produced no rows. Three are pinned, and each
of those is a loop whose only exits are a return and a raise, so falling
off its end cannot happen.
"""
from __future__ import annotations

import inspect
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import utils as U


# ---------------------------------------------------------------------------
# suggest_training_changes -- the same advice reached twice
# ---------------------------------------------------------------------------

def _progress(tmp_path, n=30, train_loss=None, val_loss=None,
              train_acc=None, val_acc=None):
    epoch = np.arange(1, n + 1)
    train = pd.DataFrame({
        "epoch": epoch,
        "loss": np.linspace(0.70, 0.10, n) if train_loss is None
        else train_loss,
    })
    val = pd.DataFrame({
        "epoch": epoch,
        "loss": np.linspace(0.70, 0.20, n) if val_loss is None else val_loss,
    })
    if train_acc is not None:
        train["accuracy"] = train_acc
    if val_acc is not None:
        val["accuracy"] = val_acc
    train.to_csv(tmp_path / "train_progress.csv", index=False)
    val.to_csv(tmp_path / "validation_progress.csv", index=False)


class TestTheTrainingAdvisorsSuggestions:

    def test_a_healthy_run_is_advised_without_repetition(self, tmp_path):
        _progress(tmp_path)

        out = U.suggest_training_changes(str(tmp_path))

        assert len(out["suggestions"]) == len(set(out["suggestions"])), (
            "the advisor repeated itself")

    def test_an_overfitting_run_is_told_so_without_repetition(self,
                                                               tmp_path):
        n = 30
        _progress(tmp_path, n=n,
                  train_loss=np.linspace(0.70, 0.01, n),
                  val_loss=np.linspace(0.70, 0.65, n),
                  train_acc=np.linspace(0.5, 0.99, n),
                  val_acc=np.linspace(0.5, 0.55, n))

        out = U.suggest_training_changes(str(tmp_path))

        assert out["suggestions"], "a badly overfitting run got no advice"
        assert "overfitting" in out["flags"]
        assert len(out["suggestions"]) == len(set(out["suggestions"]))

    def test_no_two_checks_offer_the_same_sentence(self):
        """THE PIN, for ``if s not in seen``.

        The de-duplication cannot fire, because no two checks produce the
        same sentence: every suggestion in the function is a distinct
        literal. Computed from the source rather than argued, so two
        checks that converge on one remedy tomorrow -- which is the
        obvious way to write "add augmentation" twice -- fail here and
        get the de-duplication tested properly.

        Keeping it is right anyway: a list that said the same thing
        three times reads as three findings rather than one that three
        checks agree on.
        """
        import collections

        source = inspect.getsource(U.suggest_training_changes)
        literals = re.findall(r'"([^"\n]{25,})"', source)
        repeated = {text: count for text, count
                    in collections.Counter(literals).items() if count > 1}

        assert not repeated, (
            f"these suggestions are written more than once, so the "
            f"de-duplication is live and needs a real test: "
            f"{sorted(repeated)[:3]}")

    def test_the_order_the_checks_ran_in_is_kept(self, tmp_path):
        """De-duplicating with a set would also reorder, and the order is
        the priority: the first suggestion is the one to try first."""
        source = inspect.getsource(U.suggest_training_changes)
        assert "seen = set()" in source
        assert "dedup = []" in source
        assert "dedup.append(s); seen.add(s)" in source, (
            "the de-duplication no longer preserves order")


# ---------------------------------------------------------------------------
# _run_test_mode -- a filename the regex does not recognise
# ---------------------------------------------------------------------------

class TestScanningAFolderInTestMode:

    def _images(self, tmp_path, names):
        source = tmp_path / "plate1"
        source.mkdir()
        for name in names:
            (source / name).write_bytes(b"II*\x00")
        return str(source)

    def test_files_that_match_are_grouped_by_their_set(self, tmp_path,
                                                        capsys):
        pattern = (r"(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_"
                   r"(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif")
        source = self._images(tmp_path, [
            "plate1_A01_1_1.tif", "plate1_A01_1_2.tif",
            "plate1_A02_1_1.tif",
        ])

        try:
            U._run_test_mode(source, pattern)
        except Exception as error:              # noqa: BLE001
            pytest.skip(f"test mode needs more than filenames: {error}")

        assert "Found 3 files" in capsys.readouterr().out

    def test_a_file_the_regex_does_not_match_never_enters_the_loop(
            self, tmp_path, capsys):
        """THE PIN, for the ``if match:`` inside the loop.

        It cannot be false: ``all_filenames`` is built by the same
        ``regular_expression.match``, so everything the loop sees has
        already matched once. A folder does hold more than its images --
        a stitched overview, a flat-field reference, an export from
        another tool -- and those are filtered out before the loop, not
        inside it.

        The guard is still right to keep: ``match.group('wellID')`` on
        None stops the scan at whatever file the operating system
        happened to list first, which is not reproducible between
        machines. This fails if the listing stops filtering, which is
        what would let a non-match reach it.
        """
        pattern = (r"(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_"
                   r"(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif")
        source = self._images(tmp_path, [
            "plate1_A01_1_1.tif", "overview.tif",
            "flatfield_correction.tif",
        ])

        try:
            U._run_test_mode(source, pattern)
        except Exception as error:              # noqa: BLE001
            pytest.skip(f"test mode needs more than filenames: {error}")

        printed = capsys.readouterr().out
        assert "Found 1 files" in printed, (
            "the two unrecognised TIFFs were not filtered before the loop")

        listing = inspect.getsource(U._run_test_mode)
        assert ("all_filenames = [filename for filename in os.listdir(src) "
                "if regular_expression.match(filename)]") in listing, (
            "the listing no longer filters by the regex, so a non-match "
            "can now reach the loop and the guard inside it is live")

    def test_the_match_is_checked_before_any_group_is_read(self):
        source = inspect.getsource(U._run_test_mode)
        check = source.index("if match:")
        first_group = source.index("match.group(")
        assert check < first_group, (
            "a regex group is read before the match is checked, so an "
            "unrecognised filename is an AttributeError")


# ---------------------------------------------------------------------------
# Two retry loops whose last attempt re-raises
# ---------------------------------------------------------------------------

class TestTheDatabaseRetryLoops:

    @pytest.mark.parametrize("function", [
        "_release_imported_rows_for_field",
        "_append_to_measurements_db",
    ])
    def test_the_last_attempt_re_raises_rather_than_falling_through(
            self, function):
        """THE PIN, for both loops.

        Each is ``for attempt in range(1, DB_WRITE_ATTEMPTS + 1)`` whose
        body either returns or, on the last attempt, re-raises. So the
        loop cannot finish normally, and the implicit ``return None``
        after it is unreachable.

        That matters more than it looks. Falling out of either loop
        returns None, which the caller reads as "zero rows released" or
        "the append succeeded" -- a lost write reported as a completed
        one. The re-raise is what makes a busy database an error rather
        than a silence.
        """
        source = inspect.getsource(getattr(U, function))
        assert "for attempt in range(1, DB_WRITE_ATTEMPTS + 1):" in source
        assert "attempt == DB_WRITE_ATTEMPTS" in source, (
            f"{function}'s last attempt no longer re-raises, so the loop "
            f"can now fall through and return None")
        assert "raise" in source

    def test_a_database_that_stays_busy_raises_rather_than_returning(
            self, tmp_path, monkeypatch):
        """The live side of the pin: every attempt fails, and it is an
        error."""
        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE cell (a TEXT)")

        def always_locked(*_a, **_k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(U, "_release_imported_rows_once", always_locked)
        monkeypatch.setattr(U.time, "sleep", lambda *_a, **_k: None)

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            U._release_imported_rows_for_field(
                str(db), "cell", pd.DataFrame({"a": ["1"]}))

    def test_an_error_that_is_not_a_lock_is_raised_at_once(self, tmp_path,
                                                            monkeypatch):
        """No retry for a schema error: retrying it wastes the delay and
        answers the same way."""
        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE cell (a TEXT)")

        attempts = []

        def wrong_schema(*_a, **_k):
            attempts.append(1)
            raise sqlite3.OperationalError("no such column: b")

        monkeypatch.setattr(U, "_release_imported_rows_once", wrong_schema)

        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            U._release_imported_rows_for_field(
                str(db), "cell", pd.DataFrame({"a": ["1"]}))
        assert len(attempts) == 1, "a schema error was retried"


# ---------------------------------------------------------------------------
# The two label-merge loops -- the first label is never in the tail
# ---------------------------------------------------------------------------

class TestMergingOverlappingCells:

    def test_the_first_label_cannot_appear_in_its_own_tail(self):
        """THE PIN, for both merge loops.

        Each walks ``overlapping_cell_labels[1:]`` and skips a label
        equal to ``overlapping_cell_labels[0]``. The slice excludes
        index 0, and the labels come from ``np.unique``, so the first
        label is not in the tail and the skip cannot fire.

        It is right to keep: ``cell_mask[cell_mask == first] = first`` is
        a no-op that costs a full-frame comparison per label, and on a
        1080p mask with 300 objects that is measurable.
        """
        for function in ("_merge_cells_based_on_parasite_overlap",):
            source = inspect.getsource(getattr(U, function))
            assert "overlapping_cell_labels[1:]" in source
            assert "if other_label != first_label:" in source
            assert "np.unique" in source, (
                "the labels no longer come from np.unique, so the tail can "
                "now repeat the first label")

        labels = np.unique(np.array([3, 3, 7, 7, 9]))
        assert labels[0] not in labels[1:], (
            "np.unique no longer returns distinct values")

    @pytest.mark.parametrize("labels", [
        np.array([1]), np.array([1, 2]), np.array([4, 9, 11]),
    ])
    def test_the_tail_of_a_unique_label_list_never_holds_its_head(self,
                                                                   labels):
        assert labels[0] not in labels[1:]
        assert all(other != labels[0] for other in labels[1:])
