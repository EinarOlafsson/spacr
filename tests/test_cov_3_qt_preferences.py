"""A preference read back as rubbish falls back to its default, silently.

The store is an INI file a user can hand-edit, a file that survives a spaCR
upgrade, and a file two versions of spaCR can disagree about. Every reader in
this module therefore has to answer with the default rather than raise, and
every writer has to clamp rather than refuse -- because the alternative is an
exception thrown from inside a repaint, which takes the window down over a
cosmetic setting.
"""
from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (                                 # noqa: E402
    QApplication, QDialog, QFormLayout, QLabel, QMessageBox, QWidget,
)

from spacr.qt import preferences as prefs                       # noqa: E402


@pytest.fixture()
def store():
    """The sandboxed QSettings the root conftest already redirected."""
    return prefs._settings()


def _write(store, key, value):
    store.setValue(key, value)
    store.sync()


# ---------------------------------------------------------------------------
# JSON-valued preferences
# ---------------------------------------------------------------------------


def test_a_section_layout_that_is_not_a_mapping_is_discarded(store):
    """A stored list would make every panel ask a list for its folded
    sections."""
    _write(store, prefs._KEY_SECTION_LAYOUT, "[1, 2]")

    assert prefs.get_section_layout("settings") == {}


def test_saving_a_section_layout_over_a_damaged_record_starts_fresh(store):
    """The panel being saved must land in the store even when what was
    there before cannot be read."""
    _write(store, prefs._KEY_SECTION_LAYOUT, "{not json")

    prefs.set_section_layout("settings", folded=("Advanced",), sizes=(3, 4))

    assert prefs.get_section_layout("settings") == {
        "folded": ["Advanced"], "sizes": [3, 4]}


def test_saving_a_section_layout_over_a_non_mapping_record_starts_fresh(store):
    """Valid JSON of the wrong shape, on the write path this time."""
    _write(store, prefs._KEY_SECTION_LAYOUT, "[1, 2]")

    prefs.set_section_layout("figures", folded=(), sizes=(1,))

    assert prefs.get_section_layout("figures")["sizes"] == [1]


# ---------------------------------------------------------------------------
# Numeric preferences
# ---------------------------------------------------------------------------

def test_an_unknown_figure_format_is_refused_by_name():
    """The list is in the message because the caller passing 'svg' has no
    other way to learn spaCR writes png and pdf."""
    with pytest.raises(ValueError) as excinfo:
        prefs.set_figure_format("svg")

    assert "'svg'" in str(excinfo.value)
    assert "png" in str(excinfo.value)


@pytest.mark.parametrize("getter,key,default_attr", [
    (prefs.get_figure_live_cache, prefs._KEY_FIG_LIVE_CACHE,
     "DEFAULT_FIG_LIVE_CACHE"),
    (prefs.get_figure_png_dpi, prefs._KEY_FIG_PNG_DPI, "DEFAULT_PNG_DPI"),
    (prefs.get_montage_columns, prefs._KEY_MONTAGE_COLUMNS,
     "DEFAULT_MONTAGE_COLUMNS"),
])
def test_a_non_numeric_stored_value_reads_as_its_default(store, getter, key,
                                                         default_attr):
    """Each of these is read on a hot path -- a repaint, a montage draw --
    where an exception costs the window, not the setting."""
    _write(store, key, "twelve")

    assert getter() == getattr(prefs, default_attr)


def test_a_figure_text_size_that_is_not_a_number_leaves_matplotlib_alone(
        store):
    """Zero is 'do not touch the font sizes', which is the only safe answer
    when the stored one cannot be read."""
    _write(store, prefs._KEY_FIG_TEXT_SIZE, "large")

    assert prefs.get_figure_text_size() == 0


def test_a_pane_opacity_that_is_not_a_number_reads_as_the_default(store):
    """Read on every stylesheet build."""
    _write(store, prefs._KEY_PANE_OPACITY, "mostly")

    assert prefs.get_pane_opacity() == prefs.DEFAULT_PANE_OPACITY_PCT / 100.0


def test_a_stored_figure_dynamic_flag_is_read_as_a_word_or_a_bool(store):
    """QSettings hands an INI value back as a string; treating 'true' as a
    non-empty string would make the flag impossible to turn off."""
    _write(store, prefs._KEY_FIG_DYNAMIC, "true")
    assert prefs.get_figure_dynamic() is True

    _write(store, prefs._KEY_FIG_DYNAMIC, "off")
    assert prefs.get_figure_dynamic() is False


def test_a_stored_verbose_flag_is_read_as_a_word_or_a_bool(store):
    """The same INI round-trip, for the logger switch."""
    _write(store, prefs._KEY_VERBOSE_LOG, "yes")
    assert prefs.get_verbose_logging() is True

    _write(store, prefs._KEY_VERBOSE_LOG, "no")
    assert prefs.get_verbose_logging() is False


@pytest.mark.parametrize("stored", ["quickly", "nan"])
def test_an_unreadable_spinner_delay_reads_as_the_default(store, stored):
    """A NaN gets past `float()` and then clamps to NaN, which makes the
    spinner never appear -- so NaN is checked separately from garbage."""
    _write(store, prefs._KEY_SPINNER_DELAY, stored)

    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY


@pytest.mark.parametrize("given", ["quickly", float("nan")])
def test_an_unreadable_spinner_delay_is_stored_as_the_default(given):
    """Setters clamp rather than refuse: a preferences dialog that raises on
    Save loses every other setting on the page too."""
    prefs.set_spinner_delay(given)

    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY


@pytest.mark.parametrize("setter,getter,default_attr", [
    (prefs.set_rim_length, prefs.get_rim_length, "DEFAULT_RIM_LENGTH"),
    (prefs.set_rim_lag, prefs.get_rim_lag, "DEFAULT_RIM_LAG"),
    (prefs.set_rim_period, prefs.get_rim_period, "DEFAULT_RIM_PERIOD"),
])
def test_an_unstorable_rim_setting_falls_back_to_its_default(setter, getter,
                                                             default_attr):
    """Each returns what it actually stored, so the dialog can show the
    value that took effect rather than the one it asked for."""
    assert setter("very long") == getattr(prefs, default_attr)
    assert getter() == getattr(prefs, default_attr)


def test_an_unreadable_rim_period_reads_as_the_default(store):
    """Read on every animation frame."""
    _write(store, prefs._KEY_RIM_PERIOD, "slowly")

    assert prefs.get_rim_period() == prefs.DEFAULT_RIM_PERIOD


def test_an_unreadable_workspace_copy_limit_reads_as_the_default(store):
    """The limit decides whether a file is copied into the workspace or
    linked; an exception here stops the run, not the copy."""
    from spacr.workspace import DEFAULT_COPY_LIMIT_MB

    _write(store, prefs._KEY_WORKSPACE_COPY_LIMIT, "big")

    assert prefs.get_workspace_copy_limit_mb() == float(DEFAULT_COPY_LIMIT_MB)
    assert prefs.set_workspace_copy_limit_mb("big") == float(
        DEFAULT_COPY_LIMIT_MB)


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure colour migrations
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ambient motion
# ---------------------------------------------------------------------------


def test_a_damaged_ambient_marker_does_not_stop_the_conversion(store):
    """Same rule as the figure colours: an unreadable marker means repeat
    the migration, not skip it."""
    _write(store, prefs._KEY_AMBIENT_SCALE, "later")
    _write(store, prefs._KEY_AMBIENT_BLUR, 4.0)

    prefs._migrate_ambient_motion()

    assert int(store.value(prefs._KEY_AMBIENT_SCALE)) == \
        prefs.AMBIENT_MOTION_SCALE


@pytest.mark.parametrize("stored", ["a lot", "nan"])
def test_an_unreadable_ambient_multiplier_reads_as_its_default(store, stored):
    """Read every animation frame."""
    _write(store, prefs._KEY_AMBIENT_SCALE, prefs.AMBIENT_MOTION_SCALE)
    _write(store, prefs._KEY_AMBIENT_RESOLUTION, stored)

    (low, high), default = prefs._ambient_ranges()[3]
    assert prefs.get_ambient_resolution() == default


@pytest.mark.parametrize("given", ["a lot", float("nan")])
def test_an_unstorable_ambient_multiplier_stores_its_default(given):
    """A slider handed a bad value writes the default rather than refusing
    to save the whole page."""
    prefs.set_ambient_resolution(given)

    (_low, _high), default = prefs._ambient_ranges()[3]
    assert prefs.get_ambient_resolution() == pytest.approx(default)


def test_an_unknown_starfield_direction_is_refused_with_the_list():
    """A direction nothing can draw would leave the starfield still."""
    with pytest.raises(ValueError) as excinfo:
        prefs.set_ambient_drift_direction("sideways")

    assert "'sideways'" in str(excinfo.value)


def test_the_minimum_visuals_are_applied_even_if_the_animation_will_not_stop(
        monkeypatch):
    """Switching to the low-power mode has to reach the resolution, density
    and per-paint effects even when the animation itself refuses."""
    def explode(_key):
        raise RuntimeError("no such animation")

    monkeypatch.setattr(prefs, "set_ambient_animation", explode)

    prefs._minimise_visuals()

    ranges = prefs._ambient_ranges()
    assert prefs.get_ambient_resolution() == pytest.approx(ranges[3][0][0])
    assert prefs.get_setting_animations_enabled() is False
    assert prefs.get_field_fade_enabled() is False


def test_a_stash_that_cannot_be_applied_is_reported_as_not_restored(store):
    """False is what tells the caller the user kept the minimums. Returning
    True would leave them believing their visuals came back."""
    _write(store, prefs._KEY_MODE_VISUAL_STASH,
           json.dumps({"ambient_resolution": "not a number"}))

    assert prefs._restore_visuals() is False


# ---------------------------------------------------------------------------
# Logging levels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pushing preferences into a running app
# ---------------------------------------------------------------------------

def test_applying_preferences_with_no_application_does_nothing(monkeypatch):
    """Called from module import paths that run headless; there is nothing
    to theme and nothing to complain about."""
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    assert prefs.apply_preferences_to_app() is None


def test_applying_preferences_survives_every_optional_subsystem_failing(
        qapp, monkeypatch):
    """The field fade, the console zoom, the verbose logger and the
    workspace push are all optional. A window that will not theme itself
    because one of them is missing is a window nobody can use."""
    def explode():
        raise RuntimeError("not available")

    monkeypatch.setattr(prefs, "apply_workspace_preference", explode)
    for name in ("spacr.qt.widgets.field_fade",
                 "spacr.qt.widgets.console_panel",
                 "spacr.qt.verbose_logger"):
        monkeypatch.setitem(sys.modules, name, None)

    prefs.apply_preferences_to_app(qapp)

    from spacr.qt import theme as _theme
    # THE SHEET IS ON THE WINDOWS, NOT ON THE APPLICATION, since 380's
    # last lever moved it there -- a preference save with four modules
    # open went from 10.2 s to 3.1 s. `qapp.styleSheet()` is empty by
    # design now.
    assert _theme.window_stylesheet(qapp), "the application was left unthemed"


def test_turning_the_field_fade_off_survives_a_missing_painter(monkeypatch):
    """The cache cannot be stale if it was never built, so a headless
    process must be able to store the preference anyway."""
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.field_fade", None)

    prefs.set_field_fade_enabled(False)

    assert prefs.get_field_fade_enabled() is False


# ---------------------------------------------------------------------------
# The resource buttons
# ---------------------------------------------------------------------------


def test_the_result_of_a_resource_action_is_shown_with_its_details(
        qapp, monkeypatch):
    """A cleanup that reports nothing looks like a cleanup that did not
    run, so the measured summary and the per-item detail both have to
    reach a dialog."""
    shown = {}

    class Result:
        def summary(self):
            return "Freed 12 MB"
        details = ("figures/a.png", "figures/b.png")

    def capture(self):
        shown["text"] = self.text()
        shown["details"] = self.detailedText()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", capture)

    prefs._show_resource_result("ram", Result())

    assert shown["text"] == "Freed 12 MB"
    assert "figures/a.png" in shown["details"]


# ---------------------------------------------------------------------------
# Explaining the rows
# ---------------------------------------------------------------------------

def test_a_form_row_with_no_label_widget_is_skipped(qapp):
    """A row built from a widget rather than a caption has no text to look
    a tip up by; skipping it is what keeps the pass from raising on the
    dialog's own layout rows."""
    dialog = QDialog()
    form = QFormLayout(dialog)
    form.addRow(QWidget(), QWidget())
    form.addRow(QLabel("Theme"), QWidget())

    explained = prefs.explain_every_row(dialog)

    assert explained >= 0
    dialog.deleteLater()


# ---------------------------------------------------------------------------
# Telling the cards
# ---------------------------------------------------------------------------
