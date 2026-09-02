"""Setting help appears once, at the bottom, and always fits its strip.

Asked for on 2026-09-01: "for the tooltips i dont need the popup box if the
tooltip is shown on the bottom of the window and the text needs to always fit
in the container", then "same for annotate and classify and probably more".

Hovering a setting wrote the bottom strip AND popped a sticky tooltip carrying
the same sentence, so the popup was a second copy drawn over the form being
read -- the same objection that moved the module blurbs to the bottom. The
strip is a fixed four lines, and anything longer was clipped mid-word by the
layout, which reads as a rendering fault rather than as "there is more".
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QWidget

from spacr.qt.screens.app_screen import HINT_STRIP_LINES

LONG = (
    "Folder the current step reads from and writes into: raw images for mask "
    "generation, the merged folder of npy stacks for measure, the plate root "
    "for dataset and regression steps, or the folder of fastq.gz reads for "
    "sequencing. Outputs such as stack, masks, measurements.db, datasets and "
    "results are all created inside it. A list of paths processes several "
    "plates in one run, and there is no usable default."
) * 3


@pytest.fixture(params=["mask", "classify_merged", "map_barcodes"])
def screen(qtbot, request):
    from spacr.qt.screens.app_screen import AppScreen
    s = AppScreen(request.param)
    qtbot.addWidget(s)
    s.resize(1200, 900)
    s.show()
    return s


def _lines_used(strip) -> int:
    metrics = QFontMetrics(strip.font())
    rect = metrics.boundingRect(0, 0, max(0, strip.width() - 8), 0,
                                Qt.TextWordWrap, strip.text())
    return max(1, round(rect.height() / max(1, metrics.lineSpacing())))


def test_hovering_a_setting_pops_no_tooltip(screen, monkeypatch):
    """One place, not two."""
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    shown = []
    monkeypatch.setattr(HoverTooltip, "show_for",
                        lambda self, *a, **k: shown.append(a))

    target = next((w for w in screen.findChildren(QWidget)
                   if w.property("settingKey")), None)
    if target is None:
        pytest.skip("this module renders no keyed setting rows")

    screen.eventFilter(target, QEvent(QEvent.Enter))
    assert shown == [], "the sticky popup was shown as well as the strip"


def test_hovering_a_setting_writes_the_strip(screen):
    target = next((w for w in screen.findChildren(QWidget)
                   if w.property("settingKey")), None)
    if target is None:
        pytest.skip("this module renders no keyed setting rows")

    before = screen._hint_strip.text()
    screen.eventFilter(target, QEvent(QEvent.Enter))
    assert screen._hint_strip.text() != before


def test_a_long_description_is_trimmed_to_fit(screen):
    """The container is fixed, so the text has to give."""
    screen._write_hint(LONG)
    assert _lines_used(screen._hint_strip) <= HINT_STRIP_LINES
    assert screen._hint_strip.text().endswith("…"), (
        "a trimmed hint should say it was trimmed")
    assert len(screen._hint_strip.text()) < len(LONG)


def test_a_short_description_is_not_trimmed(screen):
    """Not vacuous the other way: trimming must not fire on everything."""
    short = "Folder the step reads from."
    screen._write_hint(short)
    assert screen._hint_strip.text() == short
    assert not screen._hint_strip.text().endswith("…")


def test_the_whole_text_stays_reachable(screen):
    screen._write_hint(LONG)
    assert screen._hint_strip.toolTip() == LONG


def test_the_documentation_link_survives_losing_the_popup(screen):
    """The strip's own prompt promises a link, and it only lived in the popup.

    Suppressing the popup without moving the link would have removed the
    documentation route entirely.
    """
    screen._write_hint("Some description.", "https://example.invalid/api")
    text = screen._hint_strip.text()
    assert "<a href=" in text and "example.invalid" in text
    assert screen._hint_strip.openExternalLinks()


def test_the_link_costs_a_line_rather_than_overflowing(screen):
    """Adding a line to a fixed-height strip would push the body out of view."""
    screen._write_hint(LONG, "https://example.invalid/api")
    assert _lines_used(screen._hint_strip) <= HINT_STRIP_LINES


def test_leaving_a_setting_restores_the_prompt(screen):
    target = next((w for w in screen.findChildren(QWidget)
                   if w.property("settingKey")), None)
    if target is None:
        pytest.skip("this module renders no keyed setting rows")
    screen.eventFilter(target, QEvent(QEvent.Enter))
    screen.eventFilter(target, QEvent(QEvent.Leave))
    assert screen._hint_strip.text() == screen._default_hint()
