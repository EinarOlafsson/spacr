"""One training button in Annotation, with both destinations behind it.

"Train CV" and "Train XG" sat side by side and read as two features rather than
as one decision. They are the same act -- take these annotations and train
something on them -- differing only in WHAT the model looks at: the images, or
the measured features. Two four-letter abbreviations left the user to work that
out.

Reported 2026-09-01: "in the annotation application there are two buttons for
training cs and ml these shpuld be one button".
"""
from __future__ import annotations

from pathlib import Path

import spacr.qt.screens.annotate as annotate


def _source() -> str:
    return Path(annotate.__file__).read_text(encoding="utf-8")


def test_there_is_one_button_not_two():
    source = _source()
    assert 'QPushButton("Train…")' in source
    assert 'QPushButton("Train CV")' not in source
    assert 'QPushButton("Train XG")' not in source


def test_both_destinations_are_still_reachable():
    """Merging the buttons must not remove a capability."""
    source = _source()
    assert "action_cv.triggered.connect(self._on_train_cv)" in source
    assert "action_xg.triggered.connect(self._on_train_xg)" in source


def test_the_choices_say_what_they_look_at():
    """"CV" and "XG" name the algorithm; what the user is choosing between is
    the images and the measured features."""
    source = _source()
    assert "On the images (CNN / Transformer)…" in source
    assert "On the measured features (XGBoost)…" in source


def test_both_handlers_still_exist():
    assert callable(getattr(annotate.AnnotateScreen, "_on_train_cv", None))
    assert callable(getattr(annotate.AnnotateScreen, "_on_train_xg", None))


def test_the_old_names_still_point_at_something():
    """Anything that enabled, disabled or clicked the two buttons must still
    have a widget to hold -- and flipping one must flip the button they
    became."""
    source = _source()
    assert "self._btn_train_cv = self._btn_train" in source
    assert "self._btn_train_xg = self._btn_train" in source
