"""Console-context sharing survives a QSettings backend that stores strings.

``QSettings`` does not promise to hand back the type it was given: the INI
backend on Linux and the registry backend on Windows both return the string
``"false"`` where a ``bool`` went in. A preference read that trusts the type
therefore reads every stored value as truthy, and the console-context toggle
in particular would then attach console output for a user who had turned it
off. :func:`spacr.qt.ai.settings.get_console_aware` normalises both spellings
and defaults to on only when nothing has been stored.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import settings as ai_settings          # noqa: E402

pytestmark = pytest.mark.qt


def _store(value):
    ai_settings._settings().setValue(ai_settings._KEY_CONSOLE_AWARE, value)


@pytest.mark.parametrize("stored,expected", [
    ("false", False), ("0", False), ("no", False), ("", False),
    ("true", True), ("1", True), ("yes", True), ("TRUE", True),
])
def test_a_string_shaped_preference_still_means_off_when_it_says_off(
        stored, expected):
    """A backend that stringifies the flag must not flip it to on."""
    _store(stored)

    assert ai_settings.get_console_aware() is expected


def test_a_real_boolean_is_returned_unchanged():
    """Where the backend does keep the type, nothing is reinterpreted."""
    ai_settings.set_console_aware(False)
    assert ai_settings.get_console_aware() is False

    ai_settings.set_console_aware(True)
    assert ai_settings.get_console_aware() is True


def test_console_context_is_on_until_somebody_turns_it_off():
    """Nothing stored means the default, and the default is on."""
    ai_settings._settings().remove(ai_settings._KEY_CONSOLE_AWARE)

    assert ai_settings.get_console_aware() is True
