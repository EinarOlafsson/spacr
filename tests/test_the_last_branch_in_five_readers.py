"""Five more last branches, in the code that reads a screen off disk.

Four of the five are loop or guard arcs whose untaken side is "this item is
not the special one" -- an unprefixed file, a class the model never predicted,
a folder with nothing in it. A loop body that has only ever seen matching
items has never proved it can skip.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _v1_v2_bridge.v2_mask_source — arc 214 -> 216, a plain .npy name
# ---------------------------------------------------------------------------

def _v2_merged(tmp_path, names):
    """A v2 ``merged/`` folder holding ``names``, with a mask channel."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": ["dapi"], "mask_channels": ["cell"]}))
    for name in names:
        np.save(merged / name, np.zeros((4, 4, 2), dtype=np.uint16))
    return merged


def test_a_field_file_without_the_stack_prefix_keeps_its_whole_name(tmp_path):
    """The ``if field.startswith("stack_"):`` branch not taken.

    v2 writes ``stack_<field>.npy``, and the prefix is stripped so the field
    id matches what the rest of the pipeline calls it. A file that is NOT so
    named must keep its whole name -- stripping nothing is what makes the
    field id correct for a plate written by any other route, and truncating
    it would silently key the QC results under a name nothing else uses.
    """
    from spacr._v1_v2_bridge import v2_mask_source

    merged = _v2_merged(tmp_path, ["stack_plate1_A01_F001.npy",
                                   "plate1_A02_F001.npy"])
    out = v2_mask_source(merged, object_type="cell")

    assert set(out) == {"plate1_A01_F001", "plate1_A02_F001"}


def test_a_file_that_is_not_an_npy_is_skipped(tmp_path):
    """The ``continue`` above it, so the arc above is not the only skip."""
    from spacr._v1_v2_bridge import v2_mask_source

    merged = _v2_merged(tmp_path, ["stack_plate1_A01_F001.npy"])
    (merged / "notes.txt").write_text("not a field")

    assert set(v2_mask_source(merged, object_type="cell")) == {"plate1_A01_F001"}


# ---------------------------------------------------------------------------
# annotation.annotate — arc 259 -> 262, already joined and asked to be quiet
# ---------------------------------------------------------------------------

def test_a_second_annotation_pass_says_nothing_when_asked_to_be_quiet(capsys):
    """The ``if not quiet:`` branch not taken.

    ``annotate`` is called inside loops that run once per plate. The notice
    that a table is already annotated is useful once and is noise 1,536 times,
    which is exactly what ``quiet`` exists for -- and the quiet path had never
    been taken with something to be quiet ABOUT.
    """
    from spacr.annotation import annotate

    frame = pd.DataFrame({"gene_nr": ["TGGT1_231640", "TGGT1_231650"]})
    once = annotate(frame, quiet=True)
    capsys.readouterr()

    twice = annotate(once, quiet=True)            # everything already there
    printed = capsys.readouterr()

    assert printed.out == ""
    assert list(twice.columns) == list(once.columns)


def test_a_second_annotation_pass_explains_itself_when_not_quiet(capsys):
    """The taken side: the same call, allowed to speak."""
    from spacr.annotation import annotate

    frame = pd.DataFrame({"gene_nr": ["TGGT1_231640", "TGGT1_231650"]})
    once = annotate(frame, quiet=True)
    capsys.readouterr()

    annotate(once, quiet=False)
    printed = capsys.readouterr()

    assert "already on this table" in printed.out


# ---------------------------------------------------------------------------
# classifier_quality.discover_test_splits — arc 364 -> 361, an empty folder
# ---------------------------------------------------------------------------

def test_a_screen_folder_with_no_test_output_is_left_out_of_the_index(tmp_path):
    """The ``if found:`` branch not taken.

    A screen directory holds one folder per model, and a model that has been
    started but not yet evaluated has a folder and no test CSV. It must be
    ABSENT from the mapping rather than present with an empty value: the
    caller reads this as "which models can be compared", and an entry
    pointing at nothing is a comparison that fails later.
    """
    from spacr.classifier_quality import discover_test_splits

    (tmp_path / "model_a").mkdir()
    (tmp_path / "model_a" / "epoch3_test_predictions.csv").write_text("a,b\n1,2\n")
    (tmp_path / "model_b").mkdir()                       # started, not evaluated
    (tmp_path / "model_c").mkdir()
    (tmp_path / "model_c" / "notes.txt").write_text("nothing matching")

    found = discover_test_splits(str(tmp_path))

    assert set(found) == {"model_a"}
    assert found["model_a"].endswith("epoch3_test_predictions.csv")


def test_a_root_that_is_not_a_directory_is_refused(tmp_path):
    """The raise above the loop, which the empty-folder case must not reach."""
    from spacr.classifier_quality import discover_test_splits

    plain = tmp_path / "a_file"
    plain.write_text("")
    with pytest.raises(ValueError, match="not a directory"):
        discover_test_splits(str(plain))


# ---------------------------------------------------------------------------
# confusion.confusion_counts — arc 326 -> 325, a label outside the class list
# ---------------------------------------------------------------------------

def test_a_prediction_outside_the_given_classes_is_not_counted():
    """The ``if ... in matrix.index and ... in matrix.columns:`` not taken.

    When the caller supplies ``classes``, it is asking for a matrix of exactly
    those. A row naming something else must be dropped rather than added --
    ``matrix.at[...]`` on an absent label would GROW the frame, silently
    turning a requested 2x2 into a 3x3 that no longer matches the legend drawn
    beside it.
    """
    from spacr.confusion import (PREDICTED_COLUMN, TRUE_COLUMN,
                                 confusion_counts)

    predictions = pd.DataFrame({
        TRUE_COLUMN: ["pos", "neg", "pos", "unlabelled"],
        PREDICTED_COLUMN: ["pos", "neg", "neg", "pos"],
    })
    matrix = confusion_counts(predictions, classes=["pos", "neg"])

    assert list(matrix.index) == ["pos", "neg"]
    assert list(matrix.columns) == ["pos", "neg"]
    assert int(matrix.to_numpy().sum()) == 3          # the fourth row dropped
    assert matrix.at["pos", "pos"] == 1


def test_every_class_gets_a_row_even_if_the_model_never_chose_it():
    """The documented default, which the drop above must not be confused with."""
    from spacr.confusion import (PREDICTED_COLUMN, TRUE_COLUMN,
                                 confusion_counts)

    predictions = pd.DataFrame({TRUE_COLUMN: ["pos", "neg"],
                                PREDICTED_COLUMN: ["pos", "pos"]})
    matrix = confusion_counts(predictions)

    assert list(matrix.columns) == ["neg", "pos"]
    assert matrix.at["neg", "neg"] == 0


# ---------------------------------------------------------------------------
# errors.read_run_status — arc 724 -> 726, nothing to close
# ---------------------------------------------------------------------------

def test_a_database_that_never_opened_is_not_closed_in_the_finally(tmp_path,
                                                                   monkeypatch):
    """The ``if conn is not None:`` branch in the finally, not taken.

    ``conn`` is bound to None before the try, so a failure inside
    ``sqlite3.connect`` itself reaches the finally with nothing to close.
    Calling ``.close()`` on None there would replace a useful
    RunStatusUnreadable -- which tells the user the run may not have finished
    -- with an AttributeError that tells them nothing.
    """
    from spacr import database_concurrency
    from spacr import errors as errors_module

    db = tmp_path / "measurements.db"
    sqlite3.connect(str(db)).close()

    def refuse(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    # The connection is opened through spacr's own helper, not sqlite3
    # directly, so that is what has to refuse for `conn` to stay None.
    monkeypatch.setattr(database_concurrency, "connect", refuse)

    with pytest.raises(errors_module.RunStatusUnreadable) as excinfo:
        errors_module.read_run_status(str(db))

    # The message is the point: this outcome must never be folded into the
    # empty-list one, which would report an interrupted run as complete.
    assert "may not have finished" in str(excinfo.value)


def test_a_database_that_opens_is_closed_again(tmp_path):
    """The taken side of the same finally, so the arc above is a real pair."""
    from spacr import errors as errors_module

    db = tmp_path / "measurements.db"
    sqlite3.connect(str(db)).close()

    # No run-status table: a legitimate "no information" answer, reached
    # through the branch that DOES open and therefore does close a handle.
    assert errors_module.read_run_status(str(db)) == []
