"""Every core module a user can open offers test data, or says why not.

Asked on 2026-09-01: "so all core modules have test data?". Checking it found
that Classify's control was keyed on ``classify``, which is not in ``APPS`` and
is not folded onto any host -- unreachable from the UI. The module a user
actually opens is ``classify_merged``, so the Classify screen had never shown
an example-data control at all.

That is the kind of gap a table in a commit message does not catch, so it is
asserted here instead.
"""
from __future__ import annotations

import pytest

from spacr.qt.app import APPS, SECTION_CORE
from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS

#: Core modules that fetch their own data through ``EXAMPLE_DATA_SECTIONS``.
FETCHES_ITS_OWN = ("mask", "measure", "classify_merged", "map_barcodes",
                   "regression")

#: Core modules that offer test data through their own bespoke screen rather
#: than the shared settings-section mechanism.
FETCHES_ELSEWHERE = ("annotate",)

#: Core modules that READ what an earlier module produced and have no dataset
#: of their own to fetch. Training Runs compares runs that Classify made;
#: Prediction Profiler and Investigate Hit read Regression's output. Listing
#: them here is a decision, not an omission -- a "download test data" button
#: on a viewer would have to fetch somebody else's module's output and then
#: pretend this module produced it.
READS_WHAT_EARLIER_MODULES_MAKE = ("train_compare", "profiler",
                                   "investigate_hit")


def _core_keys():
    return tuple(k for k, _n, _d, section in APPS if section == SECTION_CORE)


def test_every_core_module_is_accounted_for():
    """No core module is silently missing from all three lists."""
    known = set(FETCHES_ITS_OWN) | set(FETCHES_ELSEWHERE) | set(
        READS_WHAT_EARLIER_MODULES_MAKE)
    unaccounted = [k for k in _core_keys() if k not in known]
    assert not unaccounted, (
        f"core modules with no stated test-data story: {unaccounted}")


@pytest.mark.parametrize("key", FETCHES_ITS_OWN)
def test_it_is_keyed_on_a_module_that_exists(key):
    """The Classify defect exactly: a section keyed on an unreachable name.

    ``EXAMPLE_DATA_SECTIONS`` is matched against ``self.app_key``, so an entry
    naming a key no screen is built with is dead code that looks like a
    feature.
    """
    assert key in EXAMPLE_DATA_SECTIONS, f"{key} has no example-data section"
    assert key in _core_keys(), f"{key} is not a core module"


@pytest.mark.parametrize("key", FETCHES_ITS_OWN)
def test_the_control_is_actually_built(qtbot, key):
    """Built, not merely configured. The mapping being right is not the same
    as the button appearing."""
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(key)
    qtbot.addWidget(screen)
    labels = [b.text() for b in screen.findChildren(QPushButton)
              if b.text()]
    assert any("test data" in text.lower() or text in ("Score", "Count",
                                                       "Measurements (.db)",
                                                       "Image crops")
               for text in labels), (
        f"{key} builds no example-data control; found {labels[:12]}")


def test_annotate_offers_it_through_its_own_screen(qtbot):
    """Annotate is built from AnnotateScreen, not AppScreen, so it carries its
    own chooser rather than an EXAMPLE_DATA_SECTIONS entry."""
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    assert hasattr(screen, "_btn_test_data")
    assert "test data" in screen._btn_test_data.text().lower()


def test_the_section_each_control_lands_in_is_a_real_section(qtbot):
    """A section title that matches nothing silently drops the control.

    The dispatch is an exact string comparison against the section title, so a
    renamed or mistyped section removes the button with no error anywhere.
    """
    from spacr.qt.screens.app_screen import AppScreen

    for key in FETCHES_ITS_OWN:
        screen = AppScreen(key)
        qtbot.addWidget(screen)
        titles = {
            sec._header.text().replace("&&", "&")
            for sec in getattr(screen, "_settings_sections", [])
            if getattr(sec, "_header", None) is not None
        }
        wanted = EXAMPLE_DATA_SECTIONS[key].upper()
        assert any(wanted in t.upper() for t in titles), (
            f"{key}: no section matching {EXAMPLE_DATA_SECTIONS[key]!r}; "
            f"has {sorted(titles)}")
