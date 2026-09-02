"""The fetched folder ends up in `src`, whatever the shipped settings say.

Instruction 349. Reported 2026-09-02: "loade test images dosnt loade the right
path into src in mask generation it loads <src>".

THE ORDER WAS THE BUG. `_put_the_example_images_in_place` wrote the download
directory into the field and THEN called `apply_settings_that_came_with`,
which loads the CSV the example shipped and applies every key in it, `src`
included. `reanchor_example_paths` re-homes a recorded absolute path onto the
local folder, but a value it cannot resolve is deliberately left alone by
every branch -- and then wins, because it was written second.

These tests drive that exact sequence rather than asserting on the final
string alone: a test that only checked the field would pass against code that
never applied the shipped settings at all.
"""
from __future__ import annotations

import pytest


class _Field:
    """The one thing the code asks of the `src` widget."""

    def __init__(self):
        self.value = ""

    def setText(self, text):  # noqa: N802 - Qt name
        self.value = str(text)

    def text(self):
        return self.value


class _Console:
    def __init__(self):
        self.lines = []

    def append_stdout(self, text):
        self.lines.append(text)

    def append_notice(self, text, **kw):
        self.lines.append(text)


class _Screen:
    """Just enough of AppScreen to run the method under test."""

    app_key = "mask"

    def __init__(self, shipped):
        from types import SimpleNamespace

        self._field = _Field()
        self._settings_model = SimpleNamespace(_widgets={"src": self._field})
        self._console = _Console()
        self._shipped = shipped
        self.applied_with = None

    def apply_settings_that_came_with(self, folder):
        """Stand in for the real loader, which applies every shipped key."""
        self.applied_with = folder
        self._field.setText(self._shipped)
        return 1


@pytest.fixture
def put_in_place():
    from spacr.qt.screens.app_screen import AppScreen

    return AppScreen._put_the_example_images_in_place


@pytest.mark.parametrize("shipped", [
    "<src>",                              # the reported value
    "/home/carruthers/datasets/plate1",   # a real path from another machine
    "",                                   # an empty cell
])
def test_the_downloaded_folder_wins_the_src_field(put_in_place, tmp_path,
                                                  shipped):
    images = tmp_path / "toxo_mito"
    images.mkdir()
    screen = _Screen(shipped)

    result = put_in_place(screen, images, None)

    assert screen._field.text() == str(images)
    assert result["src"] == str(images)


def test_the_shipped_settings_are_still_applied(put_in_place, tmp_path):
    """The fix must not become "stop applying the example's settings".

    That would trade a wrong path for no configuration at all, which the
    example exists to provide. Asserted because writing `src` last is only
    correct while the settings are still applied FIRST.
    """
    images = tmp_path / "toxo_mito"
    images.mkdir()
    screen = _Screen("<src>")

    put_in_place(screen, images, None)

    assert screen.applied_with == images
