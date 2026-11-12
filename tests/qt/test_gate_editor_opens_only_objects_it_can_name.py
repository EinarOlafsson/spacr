"""Open selection in Annotate either opens the right objects or says why not.

The Gate Editor sends the selected gate's members -- or, with no gate
selected, the shared selection -- to Annotate. Each way that can fail has to
end in a sentence on the screen rather than a silent no-op: a merged table
that cannot vouch for its images, a gate the table no longer carries the
column for, nothing selected, and nowhere registered to open objects.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.linked_selection import (                         # noqa: E402
    register_object_opener, unregister_object_opener,
)
from tests.qt.test_cov_gate_editor_screen import (              # noqa: E402,F401
    _objects, database, loaded, screen,
)


@pytest.fixture
def opener():
    received = []
    register_object_opener("annotate", received.append)
    try:
        yield received
    finally:
        unregister_object_opener("annotate", received.append)


def test_each_way_an_open_can_fail_is_said_on_the_screen(loaded):
    loaded.gates.tree.select("")
    loaded.gates.canvas.link.clear_selection()
    loaded._open_selection_in_annotate()
    assert loaded._source.text() == "Select a gate in the hierarchy first."

    loaded.gates.canvas.publish_selection(["p1_A_1_f1_o1", "p1_A_1_f1_o2"])
    loaded._open_selection_in_annotate()
    assert loaded._source.text().startswith("could not open those objects: ")

    loaded.gates.tree.select("big")
    loaded.gates._frame = loaded.gates._frame.drop(columns=["area"])
    loaded._open_selection_in_annotate()
    assert loaded._source.text().startswith("gate 'big' names column 'area'")

    loaded._merge_definition = {"name": "merged"}
    loaded._open_selection_in_annotate()
    assert loaded._source.text() == "This merge has no verified image/object provenance."


def test_the_shared_selection_is_opened_when_no_gate_is_selected(loaded, opener):
    loaded.gates.tree.select("")
    loaded.gates.canvas.publish_selection(["p1_A_1_f1_o3"])

    loaded._open_selection_in_annotate()

    assert len(opener) == 1
    assert list(opener[0].keys) == ["p1_A_1_f1_o3"]
    assert opener[0].reason == "selected in the Gate Editor"


def test_a_saved_table_keeps_unsaved_labels_only_when_they_can_still_be_saved(
        screen, database):
    import pandas as pd

    keyable = pd.DataFrame({"prcfo": ["p1_A_1_f1_o1"]})
    loose = pd.DataFrame({"wellID": ["A1"]})
    screen._unsaved = {"keyable": (keyable, {}, {}, 1, {}),
                       "loose": (loose, {}, {}, 1, {})}

    screen._on_annotated_table_saved(database, "loose", "cell")
    screen._on_annotated_table_saved(database, "keyable", "cell")

    assert list(screen._unsaved) == ["keyable"]
