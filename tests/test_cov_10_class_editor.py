"""The class editor when the definitions it is handed cannot be used.

The Classes setting decides what a classifier is trained on, and it arrives
from four places: a settings CSV that stored it as text, an older run that
stored a plain list, a table the user is looking at, and two fields the user
types into. The branches here are the ones where one of those hands over
something that is not a class -- a malformed entry, a value with no column, a
picker that cannot be built, an edit on a row that no longer exists. Each has
to leave the editor usable, because a class silently dropped here is a class
missing from every figure afterwards.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


from spacr.qt.widgets.class_editor import (              # noqa: E402
    ClassEditorWidget)


@pytest.fixture
def editor(qtbot):
    widget = ClassEditorWidget(frame=pd.DataFrame({"condition": ["nc", "pc"]}))
    qtbot.addWidget(widget)
    return widget


def test_a_picker_that_cannot_be_built_is_not_a_broken_editor(editor,
                                                              monkeypatch):
    """The SQL picker is an extra way to name a column; typing one still
    works without it. A failure to build it must not stop the settings panel
    that owns this widget from opening."""
    from spacr.qt.widgets import column_picker

    def _explode(*args, **kwargs):
        raise RuntimeError("no database picker here")

    monkeypatch.setattr(column_picker, "attach_column_picker", _explode)
    assert editor.attach_sql_picker(lambda: "/tmp/run") is None
    assert editor.column.isEditable()


