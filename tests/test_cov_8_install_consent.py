"""Installer choices are applied once, whatever shape the profile is in.

The desktop installers write ``consent`` into a JSON profile that spaCR
reads on first launch. That file is produced outside the application, so
:func:`spacr.qt.install_consent.maybe_show_installer_consent` cannot assume
it holds a mapping -- a hand-edited or half-written profile has to fall back
to asking in-app rather than crashing the launch. Setup no longer offers
account sign-in, so even a profile that still asks for it opens no provider
setup dialog.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import install_consent as ic     # noqa: E402




@pytest.mark.parametrize("value,expected", [
    ("yes", True), ("ON", True), (" 1 ", True), ("true", True),
    ("no", False), ("0", False), ("", False), (None, False), (2, False),
])
def test_a_written_out_flag_reads_back_as_the_boolean_it_meant(value, expected):
    """QSettings and JSON both hand back strings; both must mean the flag."""
    assert ic._as_bool(value) is expected


def test_a_flag_that_is_already_a_bool_is_passed_straight_through():
    """The string path must not be able to change a real boolean."""
    assert ic._as_bool(True) is True
    assert ic._as_bool(False) is False


