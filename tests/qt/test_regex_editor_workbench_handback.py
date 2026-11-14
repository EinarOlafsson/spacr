"""The import workbench hands its pattern back; it never saves one itself.

The regex editor is the single place a filename pattern is accepted from.
Its workbench button opens a second, richer view of the same filenames on
the editor's current (trimmed) pattern, and the contract is that the
workbench RETURNS a pattern rather than writing ``custom_regex`` -- two
doors that both write the setting is how they end up disagreeing.  Only an
accepted, non-blank answer replaces what the editor shows.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog  # noqa: E402

from spacr.qt.regex_editor import RegexEditorDialog  # noqa: E402

pytestmark = pytest.mark.qt

_SAMPLES = [
    "plate1_A01_f01_ch1.tif",
    "plate1_A01_f01_ch2.tif",
    "plate1_A02_f01_ch1.tif",
]
_STARTING = r"(?P<plateID>plate\d+)_(?P<wellID>[A-P]\d+)"


class _Line:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _Workbench:
    def __init__(self, chosen):
        self.regex = _Line(chosen)


def test_the_workbench_answer_lands_in_the_editor_only_when_it_is_a_pattern(
        qtbot, monkeypatch, subtests):
    """Each workbench outcome leaves the editor's box with the exact text."""
    import spacr.qt.widgets.import_workbench as workbench_module

    chosen = r"(?P<plateID>\w+)_(?P<wellID>[A-P]\d+)_f"
    padded = f" {_STARTING} "     # the workbench is opened on the trimmed text
    # case, dialog result, workbench text, editor text after
    cases = [
        ("an accepted pattern replaces the editor's", QDialog.Accepted,
         chosen, chosen),
        ("an accepted pattern is trimmed", QDialog.Accepted,
         f"  {chosen}  ", chosen),
        ("a cancelled workbench changes nothing", QDialog.Rejected,
         "something else entirely", padded),
        ("an accepted blank box is not a pattern", QDialog.Accepted, "   ",
         padded),
    ]
    for case, result, answer, expected in cases:
        with subtests.test(msg=case):
            opened = []

            class _Dialog:
                def __init__(self, samples, regex, parent=None):
                    opened.append((list(samples), regex, parent))
                    self.workbench = _Workbench(answer)

                def exec(self):
                    return result

            monkeypatch.setattr(workbench_module, "ImportWorkbenchDialog",
                                _Dialog)
            editor = RegexEditorDialog(_SAMPLES, _STARTING)
            qtbot.addWidget(editor)
            editor._regex_input.setText(padded)

            editor._on_workbench()

            assert opened == [(_SAMPLES, _STARTING, editor)]
            assert editor._regex_input.text() == expected
