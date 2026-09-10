"""The first-run setup screen (instruction 221).

THE INSTALLER IS THE WRONG PLACE and this supersedes that request: it runs
once, often unattended, sometimes by an administrator who is not the user,
and asks before the person has seen a screen of the thing they are
configuring. Every answer is a guess.

EVERY QUESTION HAS A WORKING DEFAULT, so the screen can be dismissed without
answering anything and nothing is worse for having been skipped.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QWidget)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A config dir of this test's own.

    WITHOUT THIS THE TEST ANSWERS THE SETUP SCREEN ON THE USER'S MACHINE --
    the preferences are real QSettings on a real path, and marking it
    answered here would stop it ever opening for them.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences

    importlib.reload(preferences)
    yield
    importlib.reload(preferences)


@pytest.fixture
def dialog(app):
    from spacr.qt.widgets.setup_dialog import SetupDialog

    return SetupDialog()


class TestItOpensOnWhatIsAlreadyTrue:

    def test_every_question_has_an_editor(self, dialog):
        from spacr.qt.setup_screen import questions

        assert set(dialog._editors) == {q[0] for q in questions()}

    def test_the_boxes_show_the_current_values(self, dialog):
        from spacr.qt.setup_screen import current

        now = current()
        for key, value in dialog.answers().items():
            assert now.get(key) == value, key

    def test_a_choice_is_a_combo_and_a_flag_is_a_check(self, dialog):
        assert isinstance(dialog._editors["theme"], QComboBox)
        assert isinstance(dialog._editors["hash_inputs"], QCheckBox)

    def test_the_theme_combo_carries_values_not_captions(self, dialog):
        """`theme_choices()` is (caption, value) while every other list is
        (value, caption); normalised at the source so the screen does not
        have to know which of its questions is back to front."""
        assert dialog.answers()["theme"] in {
            "dark", "light", "glass", "system",
            "cell:microtubules", "cell:filopodia"}


class TestEveryQuestionHasAWorkingDefault:

    def test_nothing_is_required(self, dialog):
        """The screen can be dismissed without answering anything."""
        answers = dialog.answers()
        assert all(v is not None for v in answers.values())

    def test_dismissing_still_marks_it_answered(self, dialog):
        """A user who closes it has CHOSEN the defaults, and reopening on
        every launch until they fill it in would make dismissing it
        impossible."""
        from spacr.qt.setup_screen import should_open

        assert should_open()
        dialog.reject()
        assert not should_open()

    def test_accepting_marks_it_answered_too(self, dialog):
        from spacr.qt.setup_screen import should_open

        dialog.accept()
        assert not should_open()

    def test_accepting_writes_the_answers(self, dialog):
        from spacr.qt import preferences

        box = dialog._editors["hash_inputs"]
        box.setChecked(not box.isChecked())
        wanted = box.isChecked()
        dialog.accept()
        assert preferences.get_hash_inputs() == wanted


class TestTheGrouping:
    """"ELEVEN QUESTIONS IS A LOT FOR A FIRST SCREEN, and the design has to
    answer for that rather than list them in a column"."""

    def test_the_questions_are_grouped(self):
        from spacr.qt.widgets.setup_dialog import GROUPS

        assert len(GROUPS) >= 3

    def test_look_comes_first(self):
        """Language and theme change what the NEXT screen looks like."""
        from spacr.qt.widgets.setup_dialog import GROUPS

        assert "language" in GROUPS[0][1] and "theme" in GROUPS[0][1]

    def test_every_grouped_key_is_a_real_question(self):
        from spacr.qt.setup_screen import questions
        from spacr.qt.widgets.setup_dialog import GROUPS

        asked = {q[0] for q in questions()}
        grouped = {k for _, keys in GROUPS for k in keys}
        assert grouped <= asked | {"module_visibility", "github_login"}

    def test_every_question_is_in_a_group(self):
        """A question with no group would be built and never shown."""
        from spacr.qt.setup_screen import questions
        from spacr.qt.widgets.setup_dialog import GROUPS

        grouped = {k for _, keys in GROUPS for k in keys}
        for key, *_ in questions():
            assert key in grouped, f"{key} is in no group"


class TestDecorationIsNotLoadBearing:
    """INVARIANTS 10. If the blur cannot be drawn, the dialog is a plain
    dialog with the same controls and the same answers."""

    def test_it_builds_with_no_parent_and_so_no_backdrop(self, dialog):
        assert dialog._backdrop_view is None
        assert dialog._editors

    def test_a_failed_grab_does_not_stop_it(self, app, qtbot, monkeypatch):
        from spacr.qt.widgets.setup_dialog import SetupDialog

        parent = QWidget()
        # OWNED BY qtbot, WHICH IS NOT TIDINESS. Left to the garbage
        # collector the parent's C++ widget is destroyed with the dialog
        # still wrapped in a Python cycle, and the wrapper is freed after
        # the object it points at -- which segfaults the next test's
        # setup rather than this one, in the session fixture that walks
        # the widget tree.
        qtbot.addWidget(parent)

        def boom():
            raise RuntimeError("no compositor")

        monkeypatch.setattr(parent, "grab", boom)
        built = SetupDialog(parent)
        qtbot.addWidget(built)
        assert built._backdrop_view is None
        assert built.answers()

    def test_the_card_still_reports_a_corner(self, dialog):
        """The accent is a pointer-position readout, and it answers before
        the pointer has moved."""
        assert dialog.card.corner() in ("topLeft", "topRight",
                                        "bottomLeft", "bottomRight")


class TestItAsksOnlyWhenItShould:

    def test_open_if_needed_returns_none_once_answered(self, app):
        from spacr.qt.setup_screen import current_version, mark_answered
        from spacr.qt.widgets.setup_dialog import open_setup_if_needed

        mark_answered(current_version())
        assert open_setup_if_needed(None) is None

    def test_the_caller_does_not_decide(self):
        """A screen each caller gated for itself is one that appears twice on
        one launch and never on another."""
        import inspect

        from spacr.qt.widgets import setup_dialog

        source = inspect.getsource(setup_dialog.open_setup_if_needed)
        assert "should_open" in source


class TestTheProviderQuestion:

    def test_it_removes_itself_when_nothing_is_installed(self, monkeypatch):
        """Asking somebody to choose between providers none of which are
        installed is asking a question with no true answers."""
        from spacr.qt import setup_screen

        monkeypatch.setattr(setup_screen, "_provider_choices", lambda: [])
        assert "ai_provider" not in {q[0] for q in setup_screen.questions()}

    def test_whatever_is_available_is_the_default(self, monkeypatch):
        """A machine with two CLIs today may have one tomorrow, and a pinned
        name that is gone is worse than no preference."""
        from spacr.qt import setup_screen

        monkeypatch.setattr(setup_screen, "_provider_choices",
                            lambda: [("", "whatever is available"),
                                     ("claude", "claude")])
        found = {q[0]: q for q in setup_screen.questions()}
        assert found["ai_provider"][4][0][0] == ""
