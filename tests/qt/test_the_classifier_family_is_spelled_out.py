"""Classify's family switch says which two things it is choosing between.

Asked for on 2026-09-02: "in classify in classifier family spell out computer
vision and machine learning and change machine learning to Tabular Machine
Learning and cv to Computer vision (Torch)".

``cv`` and ``ml`` are abbreviations of abbreviations. A dropdown reading
``cv`` / ``ml`` asks the user to already know which of two whole disciplines
this module means, and the distinction that actually matters is what each one
READS: one is fed object crops through Torch, the other rows of measured
features. That is a choice about your data, not about your vocabulary.

The stored values do not move. ``spacr.classify`` dispatches on ``'cv'`` and
``'ml'``, every settings file already written carries them, and a caption is
not a value -- which is the trap this file pins, because Qt's
``setCurrentText`` matches the CAPTION and silently does nothing on a
non-editable combo when there is no match.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import settings_model as SM     # noqa: E402

#: What the two entries must read, in order.
EXPECTED = (("Computer Vision (Torch)", "cv"),
            ("Tabular Machine Learning", "ml"))


@pytest.fixture
def family(qapp):
    """The built classifier-family control from the Classify panel."""
    widgets = SM.SettingsWidgets("classify_merged")
    widgets.build_sections()
    control = widgets._widgets.get("classifier_family")
    if control is None:
        pytest.skip("this build has no classifier_family control")
    return widgets, control


def _entries(control):
    return tuple((control.itemText(i), control.itemData(i))
                 for i in range(control.count()))


def test_both_families_are_spelled_out(family):
    _widgets, control = family
    assert _entries(control) == EXPECTED


def test_no_caption_is_a_bare_abbreviation(family):
    """The whole request: not "cv", not "ml", not "CV (Torch)"."""
    _widgets, control = family
    captions = [text for text, _value in _entries(control)]
    assert not [c for c in captions if c.strip().lower() in {"cv", "ml"}]
    assert all(len(c.split()) >= 3 for c in captions), captions


def test_the_stored_values_did_not_move(family):
    """Every settings file already written goes on meaning what it meant."""
    _widgets, control = family
    assert [value for _text, value in _entries(control)] == ["cv", "ml"]


def test_the_panel_reads_back_the_value_and_not_the_caption(family):
    """A caption reaching `spacr.classify` would raise
    ClassifierFamilyError at run time, after the user had walked away."""
    widgets, control = family
    for index, (_caption, stored) in enumerate(EXPECTED):
        control.setCurrentIndex(index)
        assert widgets._read_widget(control) == stored


def test_a_settings_file_still_selects_by_its_stored_value(family):
    """`setCurrentText('ml')` is how a loaded settings file arrives, and Qt
    matches the caption -- so a caption that stops being its own value is
    exactly when this silently becomes a no-op."""
    _widgets, control = family
    control.setCurrentIndex(0)
    control.setCurrentText("ml")
    assert control.currentData() == "ml"
    control.setCurrentText("cv")
    assert control.currentData() == "cv"


def test_the_two_families_are_the_two_the_backend_dispatches_on(family):
    """A caption is cosmetic; the alphabet is not. If `spacr.classify` grows
    a third family, this control has to grow with it rather than offering
    two of three."""
    from spacr.classify import resolve_family

    _widgets, control = family
    for _caption, stored in _entries(control):
        assert resolve_family({"classifier_family": stored}) is not None
