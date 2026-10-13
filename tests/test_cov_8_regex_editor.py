"""The import workbench hands its pattern back; it never saves one itself.

The regex editor is the single place a filename pattern is accepted from.
The workbench button opens a second, richer view of the same filenames, and
the contract between them is that the workbench RETURNS a pattern rather
than writing the setting -- two doors that both write ``custom_regex`` is
how they end up disagreeing about what it is. These tests drive the button's
three outcomes: a pattern chosen, the workbench cancelled, and a workbench
that was accepted with an empty box.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog          # noqa: E402

from spacr.qt.regex_editor import RegexEditorDialog   # noqa: E402


_SAMPLES = [
    "plate1_A01_f01_ch1.tif",
    "plate1_A01_f01_ch2.tif",
    "plate1_A02_f01_ch1.tif",
]
_STARTING = r"(?P<plateID>plate\d+)_(?P<wellID>[A-P]\d+)"


class _FakeWorkbench:
    """A stand-in workbench with one editable regex box."""

    def __init__(self, chosen):
        self.regex = _FakeLine(chosen)


class _FakeLine:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _FakeWorkbenchDialog:
    """Records how it was opened and answers with a canned result."""

    result = QDialog.Accepted
    chosen = ""
    seen = []

    def __init__(self, samples, regex, parent=None):
        type(self).seen.append((list(samples), regex))
        self.workbench = _FakeWorkbench(type(self).chosen)

    def exec(self):
        return type(self).result


@pytest.fixture
def editor(qtbot, monkeypatch):
    """A live editor whose workbench button opens the recorder above."""
    import spacr.qt.widgets.import_workbench as workbench_module

    _FakeWorkbenchDialog.seen = []
    monkeypatch.setattr(workbench_module, "ImportWorkbenchDialog",
                        _FakeWorkbenchDialog)
    dialog = RegexEditorDialog(_SAMPLES, _STARTING)
    qtbot.addWidget(dialog)
    return dialog


def test_a_pattern_chosen_in_the_workbench_lands_in_the_editor(editor):
    """What the workbench returns replaces what the editor was showing."""
    _FakeWorkbenchDialog.result = QDialog.Accepted
    _FakeWorkbenchDialog.chosen = r"(?P<plateID>\w+)_(?P<wellID>[A-P]\d+)_f"

    editor._on_workbench()

    assert editor._regex_input.text() == _FakeWorkbenchDialog.chosen
    assert _FakeWorkbenchDialog.seen == [(_SAMPLES, _STARTING)]


def test_a_cancelled_workbench_leaves_the_editor_alone(editor):
    """Cancelling the second view must not rewrite the first one's box."""
    _FakeWorkbenchDialog.result = QDialog.Rejected
    _FakeWorkbenchDialog.chosen = "something else entirely"

    editor._on_workbench()

    assert editor._regex_input.text() == _STARTING


def test_an_empty_workbench_answer_is_not_a_pattern(editor):
    """An accepted workbench with a blank box must not blank the editor."""
    _FakeWorkbenchDialog.result = QDialog.Accepted
    _FakeWorkbenchDialog.chosen = "   "

    editor._on_workbench()

    assert editor._regex_input.text() == _STARTING
