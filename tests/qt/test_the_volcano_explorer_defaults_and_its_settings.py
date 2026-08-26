"""The volcano explorer's publication defaults, its red labels and its folds.

Three claims, and every one of them is measured off the thing a user looks
at rather than off the dictionary that produced it:

* a freshly opened volcano is BLACK ON WHITE -- read out of the canvas's own
  pixel buffer, because a style dict saying ``"#FFFFFF"`` and a canvas
  painting the dark application panel through a transparent figure patch is
  exactly the disagreement that made this necessary;
* a broken setting costs the reader the SETTING and not the FIGURE -- the
  points stay on the canvas, the offending name is red in painted pixels,
  and the reason is on a line under the plot;
* the settings panel folds, and starts folded.

Every case drives the real controls and spins the event loop: the canvas
defers its draw through a timer it owns, so a probe that never processes
events measures the previous frame.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from spacr.qt.widgets.volcano_explorer import VolcanoExplorer
from spacr.volcano_style import (VolcanoStyle, page_ground, render_volcano,
                                 validate_style)


def _results(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "guide": [f"G{i:03d}" for i in range(n)],
        "gene": [f"TGGT1_{i // 3:06d}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(size=n),
        "adjusted_p_value": rng.random(n) * 0.5 + 1e-6,
    })


@pytest.fixture
def explorer(qt_theme_applied, qtbot):
    """A shown explorer with a drawn canvas, styled by the application.

    Shown and drawn on purpose: an unshown widget has no pixels, and the
    label colours are read out of painted ones.
    """
    widget = VolcanoExplorer(_results())
    qtbot.addWidget(widget)
    widget.resize(1300, 760)
    widget.show()
    qt_theme_applied.processEvents()
    widget._canvas.resize(720, 520)
    qt_theme_applied.processEvents()
    widget._canvas.draw()
    qt_theme_applied.processEvents()
    return widget


def _canvas_pixels(explorer) -> np.ndarray:
    """The canvas's own RGBA buffer, redrawn first."""
    explorer._canvas.draw()
    return np.asarray(explorer._canvas.buffer_rgba())


def _redness(widget) -> float:
    """How much of ``widget``'s painted ink is red, 0..1.

    Painted, not declared: a stylesheet string proves the rule was written,
    not that anything on screen changed colour.
    """
    image = widget.grab().toImage()
    width, height = image.width(), image.height()
    if width == 0 or height == 0:
        return 0.0
    red = ink = 0
    for y in range(height):
        for x in range(width):
            colour = image.pixelColor(x, y)
            r, g, b = colour.red(), colour.green(), colour.blue()
            if colour.alpha() < 128:
                continue
            spread = max(r, g, b) - min(r, g, b)
            if spread < 30:
                continue                      # background or grey text
            ink += 1
            if r > g + 40 and r > b + 40:
                red += 1
    return red / ink if ink else 0.0


def _type_into(field, text, app) -> None:
    """Type a value the way a user does: into the box, then Return."""
    field.setText("")
    QTest.keyClicks(field, text)
    QTest.keyClick(field, Qt.Key_Return)
    app.processEvents()


# ---------------------------------------------------------------------------
# 1. the defaults are publication black on white
# ---------------------------------------------------------------------------

def test_a_freshly_opened_volcano_draws_on_white(explorer):
    """Measured in the canvas's pixels.

    Before this, the explorer's figure patch carried alpha 0 so the dark
    application panel showed through the margins: grabbing the canvas
    answered (22, 23, 27) in the corner with black text drawn over it.
    """
    pixels = _canvas_pixels(explorer)
    corners = [pixels[2, 2], pixels[2, -3], pixels[-3, 2], pixels[-3, -3]]
    for corner in corners:
        assert tuple(int(v) for v in corner) == (255, 255, 255, 255), corners
    # And what the user actually sees, with the panel painted underneath.
    image = explorer._canvas.grab().toImage()
    assert image.pixelColor(3, 3).getRgb()[:3] == (255, 255, 255)


def test_the_axes_and_the_text_are_black(explorer):
    """The spines, the tick marks, the tick labels and both axis titles."""
    axis = explorer._panels[0]
    drawn = {name: spine.get_edgecolor()
             for name, spine in axis.spines.items() if spine.get_visible()}
    assert drawn, "the volcano drew no spines at all"
    for name, colour in drawn.items():
        assert colour[:3] == (0.0, 0.0, 0.0), (name, colour)
    assert axis.xaxis.label.get_color() == "#000000"
    assert axis.yaxis.label.get_color() == "#000000"
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        assert label.get_color() == "#000000"


def test_the_left_spine_is_black_in_pixels(explorer):
    """Read off the buffer, not off the artist: an edgecolor that never
    reached the renderer would satisfy the assertion above."""
    pixels = _canvas_pixels(explorer)
    height = pixels.shape[0]
    box = explorer._panels[0].get_window_extent()
    row = int(round(height - (box.y0 + box.height / 2)))
    window = pixels[row, int(box.x0) - 1:int(box.x0) + 3, :3]
    darkest = window.reshape(-1, 3).max(axis=1).min()
    assert darkest <= 60, (darkest, window)


def test_the_threshold_lines_are_black(explorer):
    """The significance horizontal and the zero vertical, in the style and
    then in the pixels they painted."""
    style = explorer.style()
    assert style.line_color == "#000000"
    assert style.zero_line_color == "#000000"

    axis = explorer._panels[0]
    level = -np.log10(style.alpha)
    pixels = _canvas_pixels(explorer)
    height = pixels.shape[0]
    box = axis.get_window_extent()
    _x, y_display = axis.transData.transform((0.0, level))
    row = int(round(height - y_display))
    # Inside the axes and clear of the spines, so the only near-black thing
    # that can be on this row is the significance line itself.
    strip = pixels[row, int(box.x0) + 12:int(box.x1) - 12, :3]
    assert strip.size, "the significance line fell outside the axes"
    assert strip.reshape(-1, 3).max(axis=1).min() <= 60


def test_the_style_defaults_are_black_on_white():
    """Without any Qt at all, so the values ship rather than being applied
    by the widget on the way past."""
    style = VolcanoStyle()
    assert style.screen_background == "#FFFFFF"
    assert style.axis_color == "#000000"
    assert style.line_color == "#000000"
    assert style.zero_line_color == "#000000"


# ---------------------------------------------------------------------------
# 1b. and the export still honours the global transparent default
# ---------------------------------------------------------------------------

def test_the_exported_ground_is_still_the_global_transparent_default():
    """The two grounds are told apart, and only the screen one is white."""
    from spacr.figure_style import GENERAL_DEFAULTS

    style = VolcanoStyle()
    assert style.background_color == GENERAL_DEFAULTS["background"] == "none"
    assert page_ground(style, screen=False) is None
    assert page_ground(style, screen=True) == "#FFFFFF"
    # A ground the user actually named wins on both routes.
    named = VolcanoStyle(background_color="#123456")
    assert page_ground(named, screen=False) == "#123456"
    assert page_ground(named, screen=True) == "#123456"


def test_the_export_render_leaves_the_figure_ground_alone():
    """The call the export path makes must not paint the screen's ground."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, 3))
    figure.patch.set_alpha(0.0)
    render_volcano(_results(), VolcanoStyle(), figure=figure)
    assert figure.patch.get_alpha() == 0.0
    # ...while the screen render does paint it.
    screen_figure = Figure(figsize=(4, 3))
    screen_figure.patch.set_alpha(0.0)
    render_volcano(_results(), VolcanoStyle(), figure=screen_figure,
                   screen=True)
    assert screen_figure.patch.get_alpha() == 1.0
    assert screen_figure.patch.get_facecolor()[:3] == (1.0, 1.0, 1.0)


def test_the_screens_ground_never_reaches_an_exported_file(explorer, tmp_path,
                                                           qt_theme_applied):
    """Driven through the panel and measured in the written file's pixels.

    A magenta screen ground is unmistakable: if section 1 had changed what
    gets exported, the file would be full of it.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    _type_into(explorer._controls["screen_background"], "#FF00FF",
               qt_theme_applied)
    on_screen = _canvas_pixels(explorer)
    magenta_on_screen = ((on_screen[..., 0] > 200) & (on_screen[..., 1] < 60)
                         & (on_screen[..., 2] > 200)).sum()
    assert magenta_on_screen > 1000, "the screen ground never took effect"

    written = explorer.export("png", str(tmp_path / "volcano.png"))
    assert written
    saved = np.asarray(Image.open(written).convert("RGBA"))
    magenta_in_file = ((saved[..., 0] > 200) & (saved[..., 1] < 60)
                       & (saved[..., 2] > 200)).sum()
    assert magenta_in_file == 0, magenta_in_file


def test_a_transparent_export_is_still_transparent(explorer, tmp_path,
                                                  qt_theme_applied):
    """Ticking the export's own transparency still writes a clear file.

    The white the explorer paints on screen must not have become the file's
    ground on the way past, and a file asked to be transparent is where that
    would show up as an opaque rectangle.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    explorer._controls["transparent"].setChecked(True)
    qt_theme_applied.processEvents()
    assert explorer.style().transparent is True
    written = explorer.export("png", str(tmp_path / "clear.png"))
    assert written
    saved = np.asarray(Image.open(written).convert("RGBA"))
    assert saved[0, 0, 3] == 0, saved[0, 0]
    assert saved[-1, -1, 3] == 0, saved[-1, -1]


# ---------------------------------------------------------------------------
# 2. a broken setting turns its label red and the plot stays
# ---------------------------------------------------------------------------

def _point_count(explorer) -> int:
    return sum(len(collection.get_offsets())
               for collection in explorer._panels[0].collections)


def test_one_bad_setting_keeps_the_figure_and_reddens_its_name(
        explorer, qt_theme_applied):
    before = _point_count(explorer)
    assert before == len(explorer.results())

    _type_into(explorer._controls["base_color"], "notacolour",
               qt_theme_applied)

    assert _point_count(explorer) == before, "the figure lost its points"
    assert set(explorer.problems()) == {"base_color"}
    label = explorer.label_for("base_color")
    assert _redness(label) > 0.5, _redness(label)


def test_the_reason_is_on_a_line_under_the_plot_not_over_it(
        explorer, qt_theme_applied):
    _type_into(explorer._controls["base_color"], "notacolour",
               qt_theme_applied)
    line = explorer._problem_line
    assert line.isVisible()
    assert "notacolour" in line.text()
    assert "Colour of non-significant points" in line.text()
    # Nothing was written over the figure.
    assert not [text.get_text() for text in explorer._panels[0].texts
                if "Cannot draw" in text.get_text()]
    # And it really is BELOW the canvas.
    assert line.mapTo(explorer, line.rect().topLeft()).y() > \
        explorer._canvas.mapTo(explorer, explorer._canvas.rect().center()).y()


def test_the_broken_setting_falls_back_to_the_last_value_that_drew(
        explorer, qt_theme_applied):
    """"With the last good value", measured off the drawn marks.

    Green is chosen, drawn, and then broken; the points must stay GREEN
    rather than snapping back to the shipped grey, which is what "the last
    good value" means and what a naive reset-to-default would fail.
    """
    import matplotlib.colors as mcolors

    _type_into(explorer._controls["base_color"], "#00FF00", qt_theme_applied)
    grey = mcolors.to_rgba("#B8BDC5")
    green = mcolors.to_rgba("#00FF00")
    drawn = explorer._panels[0].collections[0].get_facecolor()[0]
    assert tuple(round(float(v), 3) for v in drawn[:3]) == green[:3]

    _type_into(explorer._controls["base_color"], "notacolour",
               qt_theme_applied)
    drawn = explorer._panels[0].collections[0].get_facecolor()[0]
    assert tuple(round(float(v), 3) for v in drawn[:3]) == green[:3]
    assert tuple(round(float(v), 3) for v in drawn[:3]) != grey[:3]


def test_two_bad_settings_give_two_red_labels_and_neither_message_is_lost(
        explorer, qt_theme_applied):
    _type_into(explorer._controls["base_color"], "notacolour",
               qt_theme_applied)
    _type_into(explorer._controls["significant_color"], "alsobad",
               qt_theme_applied)

    assert set(explorer.problems()) == {"base_color", "significant_color"}
    for name in ("base_color", "significant_color"):
        assert _redness(explorer.label_for(name)) > 0.5, name
    text = explorer._problem_line.text()
    assert "notacolour" in text and "alsobad" in text
    assert _point_count(explorer) == len(explorer.results())


def test_correcting_the_settings_clears_the_red_and_the_line(
        explorer, qt_theme_applied):
    _type_into(explorer._controls["base_color"], "notacolour",
               qt_theme_applied)
    _type_into(explorer._controls["significant_color"], "alsobad",
               qt_theme_applied)
    assert len(explorer.problems()) == 2

    _type_into(explorer._controls["base_color"], "#B8BDC5", qt_theme_applied)
    assert set(explorer.problems()) == {"significant_color"}
    assert _redness(explorer.label_for("base_color")) < 0.1

    _type_into(explorer._controls["significant_color"], "#D55E00",
               qt_theme_applied)
    assert explorer.problems() == {}
    assert not explorer._problem_line.isVisible()
    for name in ("base_color", "significant_color"):
        assert _redness(explorer.label_for(name)) < 0.1, name


def test_a_broken_column_is_named_rather_than_replacing_the_plot(
        explorer, qt_theme_applied):
    """The case that used to blank the canvas: a style naming a column the
    results do not have."""
    explorer.set_style(VolcanoStyle(color_by="not_a_column"))
    qt_theme_applied.processEvents()
    assert set(explorer.problems()) == {"color_by"}
    assert _point_count(explorer) == len(explorer.results())
    assert _redness(explorer.label_for("color_by")) > 0.5


def test_validate_style_reports_every_fault_at_once():
    """Headless, so the rule that "a design holding one message drops the
    second" is pinned on the function and not only on the widget."""
    problems = validate_style(_results(), VolcanoStyle(
        base_color="notacolour", colormap="nope", marker="zz",
        line_style="wobbly", x_scale="banana", shape_by="missing"))
    assert set(problems) == {"base_color", "colormap", "marker",
                             "line_style", "x_scale", "shape_by"}
    assert all(isinstance(message, str) and message for message in
               problems.values())
    assert validate_style(_results(), VolcanoStyle()) == {}


def test_the_control_rule_complains_about_its_own_control(explorer,
                                                          qt_theme_applied):
    """'control' without a control column is a fault of the METHOD the user
    chose, so that is the name that goes red."""
    combo = explorer._controls["threshold_method"]
    combo.setCurrentIndex(combo.findData("control"))
    qt_theme_applied.processEvents()
    assert "threshold_method" in explorer.problems()
    assert "control" in explorer._problem_line.text()
    assert _point_count(explorer) == len(explorer.results())


# ---------------------------------------------------------------------------
# 3. the settings fold
# ---------------------------------------------------------------------------

def test_the_settings_panel_opens_with_every_section_closed(explorer):
    sections = explorer.sections()
    assert len(sections) >= 4, [s.title() for s in sections]
    assert [s.title() for s in sections if s.is_expanded()] == []


def test_a_folded_section_hides_the_controls_it_holds(explorer,
                                                      qt_theme_applied):
    """Measured on the widget, not on the section's own flag."""
    field = explorer._controls["marker_size"]
    assert not field.isVisible()
    explorer.section_for("marker_size").set_expanded(True)
    qt_theme_applied.processEvents()
    assert field.isVisible()


def test_the_constantly_reached_controls_are_never_folded_away(explorer):
    """The three that make up the threshold rule, plus the colour column.

    Those four are what the maintainer changes: the long note on
    ``VolcanoStyle.threshold_method`` is a measured comparison of the rules,
    and moving between them means touching the method, its multiplier and
    the significance level in one sitting. Recolouring by a covariate is the
    move the widget was written for.
    """
    for name in ("alpha", "threshold_method", "threshold_multiplier",
                 "color_by"):
        assert explorer.section_for(name) is None, name
        assert explorer._controls[name].isVisible(), name


def test_a_section_holding_a_red_label_opens_itself(explorer,
                                                    qt_theme_applied):
    section = explorer.section_for("base_color")
    assert not section.is_expanded()
    _type_into(explorer._controls["base_color"], "notacolour",
               qt_theme_applied)
    assert section.is_expanded()
    assert explorer.label_for("base_color").isVisible()


def test_every_setting_that_was_reachable_before_is_still_reachable(explorer):
    """The panel still offers every field of the style, and the right-click
    menu still agrees with it."""
    fields = {field.name for field in dataclasses.fields(VolcanoStyle)}
    assert explorer.panel_settings() == fields
    assert explorer.menu_settings() == fields
    # And each one has a name beside it, which is what a red label needs.
    assert set(explorer._labels) == fields


def test_every_setting_is_either_folded_or_permanently_visible(explorer):
    """No setting fell out of the panel while it was being regrouped."""
    fields = {field.name for field in dataclasses.fields(VolcanoStyle)}
    always = {name for name in fields if explorer.section_for(name) is None}
    folded = {name for name in fields if explorer.section_for(name)}
    assert always | folded == fields
    assert len(always) == 4, sorted(always)
