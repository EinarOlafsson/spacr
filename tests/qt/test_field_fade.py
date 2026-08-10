"""The field fade, measured in pixels.

Every claim the feature makes is an assertion on a rendered image here:
the ramp is the cubic, it accelerates to the right, the outline goes with
the fill, the text does not, the page-opacity slider cannot touch it, and
the preference genuinely restores the plain field.

Preference isolation, and why it is done this way: ``QSettings("spacr",
"qt")`` is a NativeFormat object and resolves to the real user config
whatever ``QSettings.setPath`` says, so a fixture that ``clear()``s it
deletes the developer's own spaCR settings. Nothing here builds that
object. :func:`spacr.qt.preferences._settings` is replaced with an
INI-format store under ``tmp_path``, and the very first assertion in the
fixture is that the path it resolves to really is inside ``tmp_path``.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint, QSettings
from PySide6.QtGui import QColor, QImage, QPainter, QRegion
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox,
                               QWidget)

from spacr.qt import preferences as prefs
from spacr.qt import theme as theme_mod
from spacr.qt.widgets import field_fade as ff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs_sandbox(qapp, tmp_path, monkeypatch):
    """A throwaway preferences store, plus a clean app afterwards."""
    store = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)

    # The guard this whole file depends on. If it ever fails, every test
    # below is writing to the real user's configuration.
    resolved = prefs._settings().fileName()
    assert str(tmp_path) in resolved, resolved

    # Complete only because tests/qt/conftest.py's
    # `_widget_qss_registrars_loaded` has already filled the registry. Taken
    # before that loader runs this is 19 blocks of 37, and the restore below
    # then deletes the other 18 for the rest of the process -- the loader
    # latches on `_QSS_REGISTRARS_LOADED` and cannot refill them.
    saved_qss = dict(theme_mod._WIDGET_QSS)
    saved_sheet = qapp.styleSheet()
    ff.invalidate_field_fade()
    prefs.set_theme_choice("dark")
    prefs.set_font_scale(1.0)
    yield store
    ff.uninstall_field_fade(qapp)
    ff.invalidate_field_fade()
    theme_mod._WIDGET_QSS.clear()
    theme_mod._WIDGET_QSS.update(saved_qss)
    qapp.setStyleSheet(saved_sheet)


def _apply(qapp):
    """Push the current preferences into the app, as a prefs save would."""
    prefs.apply_preferences_to_app(qapp)


def _render(widget) -> QImage:
    """Render ``widget`` onto transparency so its own alpha survives.

    ``DrawWindowBackground`` is deliberately off: it would fill the image
    with an opaque palette brush first and every alpha below would read
    255 no matter what the widget painted.
    """
    image = QImage(widget.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    widget.render(painter, QPoint(0, 0), QRegion(widget.rect()),
                  QWidget.RenderFlags(QWidget.DrawChildren))
    painter.end()
    return image


def _field(qtbot, factory=QLineEdit, width=240, height=32, css=""):
    widget = factory()
    if isinstance(widget, QComboBox):
        widget.addItem("chosen option")
    if css:
        widget.setStyleSheet(css)
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


#: Fields are 240px wide and sampled on two rows. Row 3 is pure fill:
#: below the 1px outline, above the 4px padding's first glyph row, and
#: 2.5px down the 4px corner arc, which by then has curved back to within
#: a pixel of the edge. Row 0 is the outline itself.
_FILL_ROW = 3
_BORDER_ROW = 0
_WIDTH = 240
#: Sampled x positions: clear of both corner arcs and of the right-hand
#: subcontrols (a combo's arrow, a spin box's buttons), so one set works
#: for every field type. The two ends are checked separately, below.
_XS = tuple(range(8, 232, 8))
#: The extremes the request is actually about — "0% to the left, 100% to
#: the right". 1 and 237 are the outermost columns the arcs leave fully
#: covered on row 3.
_LEFT_X, _RIGHT_X = 1, 237


def _fill_profile(image, row=_FILL_ROW):
    return [image.pixelColor(x, row).alpha() for x in _XS]


def _wanted(x, width=_WIDTH):
    """The alpha the curve asks for at column ``x``, 0-255.

    The gradient runs between the outline's centre-line, half a pixel
    inside each edge, so column ``x`` (centre ``x + 0.5``) sits at
    ``x / (width - 1)`` along it.
    """
    return 255.0 * theme_mod.field_fade_alpha(x / float(width - 1))


# ---------------------------------------------------------------------------
# The curve — no Qt needed
# ---------------------------------------------------------------------------

def test_the_curve_is_a_cubic_ease_in_on_transparency():
    assert theme_mod.FIELD_FADE_EXPONENT == 3.0
    assert theme_mod.field_fade_alpha(0.0) == 1.0
    assert theme_mod.field_fade_alpha(1.0) == 0.0
    # Out of range is clamped, not extrapolated into a negative alpha.
    assert theme_mod.field_fade_alpha(-2.0) == 1.0
    assert theme_mod.field_fade_alpha(7.0) == 0.0

    for i in range(101):
        t = i / 100.0
        assert theme_mod.field_fade_alpha(t) == pytest.approx(1.0 - t ** 3)

    # The property the request asks for: transparency grows FASTER on the
    # right. Compare the drop over each half.
    left_half = (theme_mod.field_fade_alpha(0.0)
                 - theme_mod.field_fade_alpha(0.5))
    right_half = (theme_mod.field_fade_alpha(0.5)
                  - theme_mod.field_fade_alpha(1.0))
    assert left_half == pytest.approx(0.125)
    assert right_half == pytest.approx(0.875)
    assert right_half > 6 * left_half

    # ...and monotonically, quarter by quarter, never flat.
    drops = [theme_mod.field_fade_alpha(a) - theme_mod.field_fade_alpha(b)
             for a, b in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))]
    assert all(later > earlier for earlier, later in zip(drops, drops[1:]))
    assert all(d > 0 for d in drops)


def test_the_sampled_profile_reaches_both_ends_and_only_falls():
    profile = theme_mod.field_fade_profile()
    assert len(profile) == theme_mod.FIELD_FADE_STOPS
    assert profile[0] == (0.0, 1.0)
    assert profile[-1] == (1.0, 0.0)
    alphas = [a for _, a in profile]
    assert all(b < a for a, b in zip(alphas, alphas[1:]))
    # The stop count has to be dense enough that the straight lines
    # between stops are the cubic to better than one 8-bit level.
    for (t0, a0), (t1, a1) in zip(profile, profile[1:]):
        mid = (t0 + t1) / 2.0
        assert abs((a0 + a1) / 2.0
                   - theme_mod.field_fade_alpha(mid)) < 1.0 / 255.0


# ---------------------------------------------------------------------------
# The rendered container
# ---------------------------------------------------------------------------

def test_a_rendered_field_follows_the_cubic_across_its_width(qtbot,
                                                             prefs_sandbox,
                                                             qapp):
    _apply(qapp)
    widget = _field(qtbot)
    image = _render(widget)
    alphas = _fill_profile(image)

    # "0% to the left, 100% to the right."
    assert image.pixelColor(_LEFT_X, _FILL_ROW).alpha() == 255
    assert image.pixelColor(_RIGHT_X, _FILL_ROW).alpha() <= 10

    # Monotonically thinner everywhere, and strictly so over the half
    # where the curve moves by more than one 8-bit level per step.
    assert all(b <= a for a, b in zip(alphas, alphas[1:]))
    half = alphas[len(alphas) // 2:]
    assert all(b < a for a, b in zip(half, half[1:]))

    # Every sample is the cubic, to the last representable level.
    for x, alpha in zip(_XS, alphas):
        assert abs(alpha - _wanted(x)) <= 1, (x, alpha, _wanted(x))

    # And it accelerates: the second half of the field loses far more
    # alpha than the first half does.
    mid = len(alphas) // 2
    assert (alphas[mid] - alphas[-1]) > 5 * (alphas[0] - alphas[mid])


def test_the_outline_fades_on_the_same_ramp_as_the_fill(qtbot, prefs_sandbox,
                                                        qapp):
    _apply(qapp)
    widget = _field(qtbot)
    image = _render(widget)

    border = [image.pixelColor(x, _BORDER_ROW).alpha() for x in _XS]
    fill = _fill_profile(image)

    # The top row really is the outline, not the fill: it is the theme's
    # border colour, and the fill row is not.
    palette = theme_mod.palette_for("dark")
    assert image.pixelColor(_XS[0], _BORDER_ROW).name() == palette["border"]
    assert image.pixelColor(_XS[0], _FILL_ROW).name() != palette["border"]

    # A QSS `border: 1px solid X` takes ONE colour, so an outline that
    # was not painted here would be flat 255 the whole way across. It
    # isn't: it starts solid, only ever thins, and is gone at the right.
    assert border[0] == 255
    assert all(b <= a for a, b in zip(border, border[1:]))
    assert image.pixelColor(_RIGHT_X, _BORDER_ROW).alpha() <= 12

    # And it is the SAME ramp, not merely some ramp. Row 0 is the 1px
    # stroke sitting over the top half of the fill's own edge pixel, so
    # what lands there is the ramp composited with half of itself —
    # a + a(1-a)/2 — computed from the fill measured at that same x.
    for x, (got, under) in zip(_XS, zip(border, fill)):
        a = under / 255.0
        want = 255.0 * (a + 0.5 * a * (1.0 - a))
        assert abs(got - want) <= 1, (x, got, want)

    # The bottom outline goes with it — the whole box, not just one edge.
    bottom = [image.pixelColor(x, widget.height() - 1).alpha() for x in _XS]
    assert all(abs(b - t) <= 1 for b, t in zip(bottom, border))


def test_the_text_stays_fully_opaque_all_the_way_across(qtbot, prefs_sandbox,
                                                        qapp):
    """The constraint that ruled out an opacity mask over the widget.

    Rendered at a size where a glyph stem covers whole pixels, so "fully
    opaque" is a measurable 255 rather than an antialiasing coin-flip.
    Asserted per 12px window across the ENTIRE width, including the last
    quarter where the container behind it has all but vanished.
    """
    _apply(qapp)
    widget = _field(qtbot, height=40,
                    css="font-size: 26px; font-weight: 700;")
    widget.setText("M" * 40)
    image = _render(widget)
    empty = _render(_field(qtbot, height=40))

    band = range(6, 34)
    text_max = [max(image.pixelColor(x, y).alpha() for y in band)
                for x in range(6, 234)]
    for start in range(0, len(text_max) - 11, 6):
        window = text_max[start:start + 12]
        assert max(window) == 255, (start + 6, max(window))

    # Meanwhile the container underneath that text is fading normally:
    # by the last quarter it is nearly gone, which is what makes the
    # assertion above worth making.
    container = _fill_profile(empty)
    assert container[0] == 255
    assert empty.pixelColor(_RIGHT_X, _FILL_ROW).alpha() <= 10


# ---------------------------------------------------------------------------
# The exemption
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.6, 1.0])
def test_page_opacity_does_not_reach_a_field(qtbot, prefs_sandbox, qapp,
                                             opacity):
    """The request's first clause, measured.

    "the fields should not be subject to the occupacy setting" — so the
    same field renders the same alpha profile at every slider position.
    """
    prefs.set_pane_opacity(1.0)
    _apply(qapp)
    reference = _fill_profile(_render(_field(qtbot)))

    prefs.set_pane_opacity(opacity)
    _apply(qapp)
    assert _fill_profile(_render(_field(qtbot))) == reference

    # ...while the slider is demonstrably still doing its job on the
    # surfaces it does own, so this is an exemption and not a dead knob.
    assert (theme_mod.panel_alpha("dark", "surface_alt", 0.0)
            != theme_mod.panel_alpha("dark", "surface_alt", 1.0))


# ---------------------------------------------------------------------------
# The preference
# ---------------------------------------------------------------------------

def test_it_is_on_by_default_and_round_trips(prefs_sandbox):
    assert prefs.DEFAULT_FIELD_FADE is True
    assert prefs.get_field_fade_enabled() is True
    prefs.set_field_fade_enabled(False)
    assert prefs.get_field_fade_enabled() is False
    assert ff.field_fade_enabled() is False
    prefs.set_field_fade_enabled(True)
    assert prefs.get_field_fade_enabled() is True
    assert ff.field_fade_enabled() is True


def test_turning_it_off_restores_the_plain_field(qtbot, prefs_sandbox, qapp):
    prefs.set_field_fade_enabled(True)
    _apply(qapp)
    faded = _fill_profile(_render(_field(qtbot)))
    assert faded[0] - faded[-1] > 200
    assert faded[0] == 255

    prefs.set_field_fade_enabled(False)
    _apply(qapp)
    plain_image = _render(_field(qtbot))
    plain = _fill_profile(plain_image)

    # No ramp at all: one flat alpha from end to end.
    assert len(set(plain)) == 1

    # And it is the stylesheet's own input fill again — the built-in
    # rule, at the page opacity, which is exactly the pre-existing look.
    prefs.set_pane_opacity(0.6)
    _apply(qapp)
    plain_dim = _fill_profile(_render(_field(qtbot)))
    want = theme_mod.panel_alpha("dark", "surface_alt", 0.6)
    assert len(set(plain_dim)) == 1
    assert abs(plain_dim[0] - round(255 * want)) <= 2

    # The QSS block is empty, not merely overridden.
    assert ff.field_fade_qss(theme_mod.palette_for("dark"), 0.6) == ""
    assert "registered widget QSS: FieldFade" not in qapp.styleSheet()


def test_the_preferences_dialog_carries_the_toggle(qtbot, prefs_sandbox,
                                                   qapp):
    """A preference the user cannot reach is not a preference."""
    from PySide6.QtWidgets import QDialogButtonBox
    from spacr.qt.widgets.toggle import Toggle

    _apply(qapp)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    toggles = [w for w in dialog.findChildren(Toggle)
               if w.objectName() == "FieldFadeEnabled"]
    assert len(toggles) == 1
    # It shows the stored value...
    assert toggles[0].isChecked() is True

    # ...and Save stores what it shows.
    toggles[0].setChecked(False)
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert prefs.get_field_fade_enabled() is False
    assert ff.field_fade_enabled() is False


# ---------------------------------------------------------------------------
# Which widgets it applies to
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [QLineEdit, QComboBox, QSpinBox,
                                     QDoubleSpinBox])
def test_every_single_line_field_type_fades(qtbot, prefs_sandbox, qapp,
                                            factory):
    _apply(qapp)
    image = _render(_field(qtbot, factory, height=48))
    alphas = _fill_profile(image)
    assert image.pixelColor(_LEFT_X, _FILL_ROW).alpha() == 255
    assert image.pixelColor(_RIGHT_X, _FILL_ROW).alpha() <= 10
    assert all(b <= a for a, b in zip(alphas, alphas[1:]))
    for x, alpha in zip(_XS, alphas):
        assert abs(alpha - _wanted(x)) <= 1, (factory, x, alpha)


def test_the_editor_inside_a_spin_box_does_not_fade_twice(qtbot,
                                                          prefs_sandbox):
    """A spin box embeds a QLineEdit. Two ramps in one control is a seam."""
    spin = _field(qtbot, QSpinBox)
    inner = spin.lineEdit()
    assert isinstance(inner, QLineEdit)
    assert ff.fades(spin) is True
    assert ff.fades(inner) is False

    combo = _field(qtbot, QComboBox)
    combo.setEditable(True)
    assert ff.fades(combo) is True
    assert ff.fades(combo.lineEdit()) is False


def test_a_widget_can_opt_out(qtbot, prefs_sandbox, qapp):
    _apply(qapp)
    widget = _field(qtbot)
    assert ff.fades(widget) is True
    widget.setProperty(ff.OPT_OUT_PROPERTY, True)
    assert ff.fades(widget) is False
    alphas = _fill_profile(_render(widget))
    assert len(set(alphas)) == 1


def test_an_item_views_cell_editor_is_not_a_field(qtbot, prefs_sandbox):
    """An editor laid over a data row has no room to trail off into."""
    from PySide6.QtWidgets import QAbstractItemView, QTableWidget

    table = QTableWidget(2, 2)
    table.setEditTriggers(QAbstractItemView.DoubleClicked)
    qtbot.addWidget(table)

    # An editor is parented to the VIEWPORT, which is what `fades` looks
    # for. Build one the same way an item delegate does.
    editor = QLineEdit(table.viewport())
    assert ff.fades(editor) is False
    # ...while a field merely sitting inside some other container does
    # fade, so the exclusion is about item views and not about nesting.
    assert ff.fades(_field(qtbot)) is True


def test_multi_line_editors_are_not_fields(qtbot, prefs_sandbox):
    from PySide6.QtWidgets import QPlainTextEdit, QTextEdit
    for factory in (QPlainTextEdit, QTextEdit):
        widget = factory()
        qtbot.addWidget(widget)
        assert ff.fades(widget) is False


# ---------------------------------------------------------------------------
# Theme switches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("choice,theme", [("dark", "dark"),
                                          ("light", "light"),
                                          ("cell:filopodia", "cell"),
                                          ("glass", "glass")])
def test_the_fade_survives_a_theme_switch(qtbot, prefs_sandbox, qapp,
                                          choice, theme):
    prefs.set_theme_choice(choice)
    _apply(qapp)
    assert prefs.resolve_effective_theme() == theme
    image = _render(_field(qtbot))
    alphas = _fill_profile(image)

    chrome = theme_mod.field_chrome(theme)
    fill_colour, fill_alpha = chrome["fill"]
    assert alphas[0] == round(255 * fill_alpha)
    assert image.pixelColor(_RIGHT_X, _FILL_ROW).alpha() <= 10
    assert all(b <= a for a, b in zip(alphas, alphas[1:]))
    # Painted in THIS theme's colour, not a frozen dark one.
    assert image.pixelColor(_XS[0], _FILL_ROW).name() == fill_colour
    assert chrome["radius"] == (10.0 if theme == "glass"
                                else float(theme_mod.RADIUS["sm"]))


def test_glass_keeps_its_translucent_rim_and_still_reaches_zero():
    chrome = theme_mod.field_chrome("glass")
    colour, alpha = chrome["border"]
    assert (colour, alpha) == ("#ffffff", 0.16)
    # The ramp multiplies the colour's own alpha, so Glass keeps its
    # material at the left edge and is still gone at the right.
    assert alpha * theme_mod.field_fade_alpha(0.0) == pytest.approx(0.16)
    assert alpha * theme_mod.field_fade_alpha(1.0) == 0.0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_applying_preferences_installs_the_hook_and_the_block(prefs_sandbox,
                                                              qapp):
    ff.uninstall_field_fade(qapp)
    theme_mod.unregister_widget_qss("FieldFade")
    assert "FieldFade" not in theme_mod.widget_qss_names()

    _apply(qapp)
    assert "FieldFade" in theme_mod.widget_qss_names()
    assert "registered widget QSS: FieldFade" in qapp.styleSheet()
    assert ff._filter is not None

    # Idempotent: applying preferences again does not stack filters.
    first = ff._filter
    _apply(qapp)
    assert ff._filter is first


def test_the_hook_is_a_no_op_without_a_qapplication(monkeypatch):
    from PySide6.QtWidgets import QApplication
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    saved, ff._filter = ff._filter, None
    try:
        assert ff.install_field_fade() is False
        assert ff.uninstall_field_fade() is False
        assert ff.repaint_fields() == 0
    finally:
        ff._filter = saved


def test_a_broken_palette_does_not_stop_a_field_drawing(qtbot, prefs_sandbox,
                                                        qapp, caplog):
    """A cosmetic effect must never be why a screen fails to paint."""
    import logging

    _apply(qapp)
    widget = _field(qtbot)

    def explode(*_args, **_kwargs):
        raise RuntimeError("no such palette role")

    with caplog.at_level(logging.ERROR):
        original = ff.paint_field_fade
        ff.paint_field_fade = explode
        try:
            image = _render(widget)
        finally:
            ff.paint_field_fade = original
    assert image.size() == widget.size()
    assert any("Field fade could not paint" in r.getMessage()
               for r in caplog.records)


def test_repaint_fields_reaches_every_live_field(qtbot, prefs_sandbox, qapp):
    _apply(qapp)
    made = [_field(qtbot, factory) for factory in
            (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)]
    assert ff.repaint_fields(qapp) >= len(made)
