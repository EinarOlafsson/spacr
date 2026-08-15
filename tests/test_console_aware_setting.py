"""Console aware is one yes/no preference, not a mode selector.

Instruction 105, as refined by the maintainer: the three-mode combo
(Auto / Include / Off) that sat beside the chat input is replaced by a single
``Console aware`` setting in the AI settings, defaulting to yes.

The test that matters most here is
:func:`test_a_non_diagnostic_question_still_gets_the_console`. "Auto" decided
from the user's wording whether the model was allowed to see the console, and
being wrong in the "no" direction produces exactly the failure 105 exists to
end -- an assistant that cannot see the traceback on the screen in front of
it. Deleting Auto is the point of the change, so a test has to pin that it is
gone rather than hidden.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.console_panel import ConsolePanel
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def aware():
    """Restore whatever the developer's own setting was."""
    from spacr.qt.ai import settings as ai_settings
    original = ai_settings.get_console_aware()
    yield ai_settings
    ai_settings.set_console_aware(original)


def _with_output(panel):
    panel.append_stdout("Segmenting plate1...\nFound 12 cells\n")
    panel.append_error(
        "Traceback (most recent call last):\n  File x\nValueError: boom\n")
    return panel


def test_the_default_is_yes(aware):
    """The chat is switched on after something has gone wrong, so off by
    default would mean it cannot see the thing it was opened for."""
    from spacr.qt.ai.settings import _KEY_CONSOLE_AWARE, _settings

    _settings().remove(_KEY_CONSOLE_AWARE)
    assert aware.get_console_aware() is True


def test_the_setting_persists(aware):
    aware.set_console_aware(False)
    assert aware.get_console_aware() is False
    aware.set_console_aware(True)
    assert aware.get_console_aware() is True


def test_the_dropdown_and_status_label_are_gone(panel):
    """The input row is the input and nothing else."""
    assert not hasattr(panel, "_console_context_mode")
    assert not hasattr(panel, "_console_context_status")


def test_console_aware_on_attaches_the_traceback(panel, aware):
    aware.set_console_aware(True)
    context, label = _with_output(panel)._console_context_for_question(
        "what went wrong? explain based on the console")
    assert "ValueError: boom" in context
    assert "chars sent" in label


def test_console_aware_off_sends_nothing(panel, aware):
    aware.set_console_aware(False)
    context, label = _with_output(panel)._console_context_for_question(
        "what went wrong? explain based on the console")
    assert context == ""
    assert label == "Console context off"


def test_a_non_diagnostic_question_still_gets_the_console(panel, aware):
    """Auto is deleted, not hidden.

    Under Auto, a question that did not look diagnostic was answered without
    the console. That is a heuristic on the user's wording deciding whether
    the model may see the error -- silently, and sometimes wrongly.
    """
    aware.set_console_aware(True)
    context, _label = _with_output(panel)._console_context_for_question(
        "how do I set the diameter")
    assert "ValueError: boom" in context, (
        "a question that does not look diagnostic must still carry the "
        "console -- otherwise Auto has survived under another name")


def test_the_status_goes_on_the_message_not_a_shared_label(panel, aware):
    """Reported per message, where it stays true; a shared label is stale
    from the moment the next question is typed."""
    aware.set_console_aware(True)
    _context, label = _with_output(panel)._console_context_for_question("why?")
    # It is returned to the caller, which stamps it on the user bubble.
    assert isinstance(label, str) and label
    assert not hasattr(panel, "_console_context_status")
