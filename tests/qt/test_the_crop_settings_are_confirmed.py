"""Two crop settings are worth a question before a twenty-minute run.

Both change what every downstream model sees, and neither is recoverable
without measuring the plate again:

* NORMALISING the crops rescales raw pixels before the annotation viewer or
  the training pipeline -- each of which normalises for itself -- ever sees
  them, so intensity differences between cells are flattened into one range
  and the crops on disk cannot be un-normalised;
* a BOUNDING BOX keeps whatever shares the rectangle with the object, so a
  classifier can learn the neighbourhood rather than the cell.

The dialog also says what the user does not have to decide yet: after Measure,
crops can be streamed from the merged arrays or from the database coordinates
without measuring again.

Reported 2026-09-01.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens.app_screen import AppScreen


class _Screen:
    """Just the two methods under test, off a real AppScreen."""

    app_key = "measure"
    _crop_choice_warnings = AppScreen._crop_choice_warnings
    _confirm_crop_choices = AppScreen._confirm_crop_choices


def _notes(settings):
    return _Screen()._crop_choice_warnings(settings)


def test_an_ordinary_run_is_not_interrupted():
    """Nothing to warn about must mean no dialog at all."""
    assert _notes({"normalize": False, "use_bounding_box": False}) == []


def test_normalising_is_warned_about():
    text = "\n".join(_notes({"normalize": True}))
    assert "NORMALISING" in text
    assert "cannot be un-normalised" in text


def test_the_warning_says_who_else_normalises():
    """Without that, "it loses information" reads as an argument for never
    normalising rather than for not doing it TWICE."""
    text = "\n".join(_notes({"normalize": True}))
    assert "annotation" in text and "training" in text


def test_the_bounding_box_is_warned_about():
    text = "\n".join(_notes({"use_bounding_box": True}))
    assert "BOUNDING BOX" in text
    assert "neighbourhood rather than the cell" in text


def test_both_appear_together_in_one_dialog():
    """One question, not two: they are set in the same panel and answered in
    the same breath."""
    notes = _notes({"normalize": True, "use_bounding_box": True})
    text = "\n".join(notes)
    assert "NORMALISING" in text and "BOUNDING BOX" in text


def test_the_streaming_reminder_rides_along():
    text = "\n".join(_notes({"normalize": True}))
    assert "streaming crops" in text or "streaming" in text
    assert "measurements.db" in text


def test_the_reminder_says_which_source_can_cut_to_the_object():
    """The difference decides which one the user should reach for, and it is
    not guessable: the database stores coordinates, so it can only ever give
    a rectangle."""
    text = "\n".join(_notes({"use_bounding_box": True}))
    assert "BOUNDING-BOX crops only" in text
    assert "cut to the object" in text


def test_no_reminder_when_there_is_nothing_to_remind_about():
    """A run with neither setting on gets no lecture."""
    assert _notes({}) == []


def test_saying_no_stops_the_run(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    screen = _Screen()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.No))
    assert screen._confirm_crop_choices({"normalize": True}) is False


def test_saying_yes_runs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    screen = _Screen()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    assert screen._confirm_crop_choices({"normalize": True}) is True


def test_nothing_to_warn_about_asks_nothing(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: pytest.fail("it asked with nothing to warn about")))
    assert _Screen()._confirm_crop_choices({}) is True


def test_the_default_button_is_no(monkeypatch):
    """A run started by habit must not be confirmed by habit."""
    from PySide6.QtWidgets import QMessageBox

    seen = {}

    def _question(parent, title, text, buttons, default=None):
        seen["default"] = default
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    _Screen()._confirm_crop_choices({"normalize": True})
    assert seen["default"] == QMessageBox.No


def test_run_actually_consults_it():
    """A source check: the warnings are inert unless _on_run calls them."""
    from pathlib import Path

    import spacr.qt.screens.app_screen as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'self.app_key == "measure" and not self._confirm_crop_choices(' \
        in source
    assert "cancelled_at_crop_warning" in source


def test_only_measure_is_asked():
    """Mask has no crop settings; asking there would be noise."""
    from pathlib import Path

    import spacr.qt.screens.app_screen as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert '_confirm_crop_choices' in source
    assert 'self.app_key == "measure" and not self._confirm_crop_choices' \
        in source
