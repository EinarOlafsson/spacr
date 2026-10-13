"""The chart/sidebar divider moves, at every size the window opens at.

Two arbitrary constants used to freeze it. A plain QPushButton's size hint
is its whole label and its policy treats that as a hard minimum, so
"Propagate automatic clusters" pinned the sidebar at 198 px; and the crop
preview carried a 120 px floor justified by "the controls under it start
to elide", which is what those controls are for.

Measured with both gone: the divider moves in both directions on a wide
window, and the sidebar can be widened at every size. Below about a
thousand pixels it cannot be narrowed further, and that is correct -- it
is already as narrow as a label beside a spin box can be, which is a
content floor rather than a number somebody chose.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QPushButton                # noqa: E402


@pytest.fixture()
def explorer(qtbot):
    from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

    made = ImageUmapExplorer()
    qtbot.addWidget(made)
    return made


def _drag(explorer, width, delta):
    """Resize to ``width``, push the divider by ``delta``, report if it moved."""
    explorer.resize(width, 760)
    explorer.show()
    splitter = explorer._body_splitter
    start = list(splitter.sizes())
    splitter.setSizes([start[0] + delta, max(0, start[1] - delta)])
    return start, list(splitter.sizes())


@pytest.mark.parametrize("width", [1400, 1000, 820, 700])
def test_the_sidebar_can_always_be_widened(explorer, width):
    start, after = _drag(explorer, width, -150)
    assert after != start, f"the divider is frozen at {width} px"


def test_the_sidebar_can_be_narrowed_when_it_has_width_to_give(explorer):
    start, after = _drag(explorer, 1400, 150)
    assert after != start
    assert after[1] < start[1]


def test_the_buttons_shorten_instead_of_holding_the_sidebar_open(explorer):
    """A button whose label is its hard minimum is a button that decides
    how wide the chart may be."""
    from spacr.qt.widgets.eliding import ElidingPushButton

    explorer.resize(1000, 760)
    explorer.show()
    for button in (explorer._apply_selected, explorer._apply_clusters):
        assert isinstance(button, ElidingPushButton), (
            f"{button.text()!r} cannot shorten itself")


def test_the_preview_gives_way_rather_than_setting_the_floor(explorer):
    from spacr.qt.widgets.umap_explorer import _ScaledPreview

    explorer.resize(1000, 760)
    explorer.show()
    side = explorer._body_splitter.widget(1)
    previews = side.findChildren(_ScaledPreview)
    assert previews, "no crop preview to check"
    floor = side.minimumSizeHint().width()
    for preview in previews:
        assert preview.minimumWidth() < floor, (
            "the preview is what stops the sidebar shrinking")


def test_what_stops_it_is_content_and_not_a_chosen_number(explorer):
    """The floor must be some row's real width, not a constant."""
    from PySide6.QtWidgets import QWidget

    explorer.resize(1000, 760)
    explorer.show()
    side = explorer._body_splitter.widget(1)
    floor = side.minimumSizeHint().width()
    pinned = [w for w in side.findChildren(QWidget)
              if w.minimumWidth() >= floor]
    assert not pinned, (
        f"these set the sidebar's floor by declaration: "
        f"{[type(w).__name__ for w in pinned]}")


def test_the_divider_says_what_it_is_for(explorer):
    """A 1 px line with no hover text is indistinguishable from an edge."""
    explorer.resize(1400, 760)
    explorer.show()
    handle = explorer._body_splitter.handle(1)
    assert handle is not None
    assert "Drag" in (handle.toolTip() or "")
