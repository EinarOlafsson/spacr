"""Panel-against-page separation, and the guards on contributed QSS blocks.

Two things in :mod:`spacr.qt.theme` that only fail in ways nobody sees.

**A resting panel has to be distinguishable from the page behind it.** The
separation is small by design -- a panel that shouts is worse than one that
whispers -- and at 60 % opacity it is smaller still, which is exactly where a
palette edit stops being visible without anyone noticing. ``lightness`` is CIE
L* rather than a contrast ratio because a ratio of 1.1:1 means something very
different between two near-blacks and two near-whites, and the page/panel
question lives at both ends.

**A widget that contributes a QSS block can get it wrong.** A block that
raises, a block that returns something other than a string, a name claimed
twice, a name that is empty: every one of them has to cost that widget its
styling and nothing else. An exception escaping here leaves the *whole*
application unstyled -- black text on a black window -- because one widget had
a typo.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from spacr.qt import theme  # noqa: E402


@pytest.fixture()
def spare_block_name():
    """A registry name that is released again however the test ends."""
    name = "ATestOnlyBlockThatIsAlwaysReleased"
    theme.unregister_widget_qss(name)
    yield name
    theme.unregister_widget_qss(name)


# ---------------------------------------------------------------------------
# Panel against page
# ---------------------------------------------------------------------------

def test_lightness_spans_black_to_white():
    """L* is 0 at black and 100 at white, whatever the colour space beneath."""
    assert theme.lightness("#000000") == pytest.approx(0.0)
    assert theme.lightness("#ffffff") == pytest.approx(100.0)


def test_lightness_stays_linear_in_the_near_black_where_the_cube_root_fails():
    """Below the CIE knee L* is linear in luminance, not a cube root.

    The cube-root branch is what makes L* perceptually uniform; used all the
    way to zero it collapses the near-blacks together, which is precisely the
    range a dark theme's page and panel live in.
    """
    near_black = theme.lightness("#010101")

    assert 0.0 < near_black < 1.0
    # Twice the luminance is twice the L* in the linear region -- the cube
    # root would give roughly 1.26x.
    assert theme.lightness("#020202") == pytest.approx(2 * near_black, rel=0.02)


@pytest.mark.parametrize("name", theme.THEMES)
def test_every_resting_panel_separates_from_its_page(name):
    """One row per panel role per opacity, and all of them pass."""
    report = theme.page_separation_report(name)

    assert len(report) == len(theme.PAGE_PANEL_ROLES) * 2
    assert {row["role"] for row in report} == set(theme.PAGE_PANEL_ROLES)
    assert {row["opacity"] for row in report} == {1.0,
                                                  theme.PAGE_FADED_OPACITY}
    for row in report:
        assert row["page"] == theme.page_colour(name)
        assert row["ratio"] >= row["min_ratio"], row
        assert row["delta_lstar"] >= row["min_delta_lstar"], row
        assert row["passes"] is True
    assert theme.page_separation_failures(name) == []


def test_a_faded_panel_is_composited_over_the_page_it_sits_on():
    """The faded row is the panel *over the page*, not the panel alone.

    Compositing it over anything else would report a separation the user never
    sees.
    """
    report = theme.page_separation_report("dark")
    palette = theme.palette_for("dark")
    page = theme.page_colour("dark")

    for row in report:
        if row["opacity"] >= 1.0:
            assert row["panel"] == palette[row["role"]]
        else:
            assert row["panel"] == theme.composite(
                palette[row["role"]], theme.PAGE_FADED_OPACITY, page)


def test_a_failing_separation_is_reported_by_name(monkeypatch):
    """The failure line names the theme, the role, both colours and both
    limits, so a palette edit that closes the gap can be read without
    re-deriving it."""
    monkeypatch.setattr(theme, "PAGE_MIN_RATIO", 99.0)

    failures = theme.page_separation_failures("dark")

    assert failures, "an impossible minimum has to fail"
    assert all("dark: page (" in line for line in failures)
    assert any("surface" in line for line in failures)
    assert all("99.00:1" in line for line in failures)


# ---------------------------------------------------------------------------
# Guards on a contributed QSS block
# ---------------------------------------------------------------------------

def test_a_block_that_is_not_callable_is_refused_at_registration(
        spare_block_name):
    """Refused when it is registered, not when the next sheet is generated."""
    with pytest.raises(TypeError, match="is not callable"):
        theme.register_widget_qss(spare_block_name, "QWidget { color: red; }")

    assert spare_block_name not in theme.widget_qss_names()


def test_two_widgets_cannot_quietly_claim_one_block_name(spare_block_name):
    """A silent overwrite would drop the first widget's styling for good."""
    theme.register_widget_qss(spare_block_name,
                              lambda palette, opacity: "QWidget#A {}")

    with pytest.raises(ValueError, match="already registered"):
        theme.register_widget_qss(spare_block_name,
                                  lambda palette, opacity: "QWidget#B {}")

    assert "QWidget#A {}" in theme.stylesheet(theme="dark")


def test_a_deliberate_replacement_wins(spare_block_name):
    theme.register_widget_qss(spare_block_name,
                              lambda palette, opacity: "QWidget#A {}")
    theme.register_widget_qss(spare_block_name,
                              lambda palette, opacity: "QWidget#B {}",
                              replace=True)

    sheet = theme.stylesheet(theme="dark")

    assert "QWidget#B {}" in sheet
    assert "QWidget#A {}" not in sheet


def test_a_block_that_returns_something_other_than_a_string_is_dropped(
        spare_block_name, caplog):
    """The rest of the sheet survives, and the log says which block it was."""
    theme.register_widget_qss(spare_block_name,
                              lambda palette, opacity: {"color": "red"})

    with caplog.at_level(logging.ERROR, logger="spacr.qt.theme"):
        sheet = theme.stylesheet(theme="dark")

    assert theme._WIDGET_QSS_MARKER.format(name=spare_block_name) not in sheet
    assert "QWidget" in sheet, "the rest of the stylesheet is still there"
    messages = [record.getMessage() for record in caplog.records]
    assert any(spare_block_name in message and "expected str" in message
               for message in messages), messages
