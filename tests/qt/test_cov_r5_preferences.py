"""A preference store answering badly, and the dialog built on top of it.

The store is a file a user can hand-edit and a file an older spaCR wrote, so
every getter here has a path for "that is not a number", "that is not the
JSON I wrote" and "that key is from a build before this one". This file
drives those paths from the outside -- through the public getters and
setters, and through the Preferences dialog itself -- and pins what the user
is left with: a default they can see, a value clamped to what the renderer
can do, and a dropdown that still opens on a token no build of spaCR offers
any more.

Where a value cannot be read, the rule is the module's own: a preference
that cannot be read is a preference at its default, and never an exception
on the way to a window opening.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


@pytest.fixture
def store(qapp, tmp_path, monkeypatch):
    """A throwaway INI store, so no test touches the real preferences."""
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    assert str(tmp_path) in prefs._settings().fileName()
    return settings


# ---------------------------------------------------------------------------
# The Mandelbrot defaults, which are read late and may not be readable at all
# ---------------------------------------------------------------------------

def test_the_fractal_defaults_are_the_published_ones_or_the_written_fallback(
        monkeypatch):
    """`_mandelbrot_defaults` reads the renderer's own DEFAULTS, because a
    second copy of them here would drift from the numbers the maintainer
    actually runs. The fallback exists for the build where that module
    cannot be imported at all -- a headless install without the widget
    package -- and it must still carry the cost numbers, not a lighter
    preset, or the first impression is not the published pattern.
    """
    import sys

    live = prefs._mandelbrot_defaults()
    assert live["supersampling"] == 2 and live["render_scale"] == 1.0

    # The import inside the function is what fails on such a build.
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fractal_mandelbrot",
                        None)
    fallback = prefs._mandelbrot_defaults()
    assert fallback["supersampling"] == 2
    assert fallback["seconds_per_decade"] == 24.0
    assert fallback["precision_digits"] == 320


def test_the_lazy_defaults_load_once_through_whichever_read_comes_first(
        monkeypatch):
    """`_MANDEL_DEFAULTS` is a dict that fills itself on first read, so that
    importing this module headless does not pull QtWidgets in. Every read
    shape has to trigger that load -- ``in``, ``keys()`` and ``items()`` as
    much as ``[]`` -- or a caller who asks the wrong way sees an empty dict
    and silently gets no defaults at all.
    """
    calls = []

    def counted():
        calls.append(1)
        return {"supersampling": 3, "max_depth": 34.0}

    monkeypatch.setattr(prefs, "_mandelbrot_defaults", counted)

    for reader in (lambda d: "supersampling" in d,
                   lambda d: list(d.keys()),
                   lambda d: dict(d.items()),
                   lambda d: d["supersampling"],
                   lambda d: d.get("supersampling")):
        lazy = prefs._LazyDefaults()
        assert reader(lazy), reader
    assert len(calls) == 5, "one load per instance, whichever read came first"

    # Read twice through one instance and it loads once, not twice.
    calls.clear()
    once = prefs._LazyDefaults()
    assert "supersampling" in once and once.get("max_depth") == 34.0
    assert len(calls) == 1


def test_defaults_that_cannot_be_read_leave_an_empty_dict_not_an_exception(
        monkeypatch, caplog):
    """A failure part-way through the load must not leave it retrying on
    every read, and must not reach the caller: the fractal is decoration,
    and a decorative default is never a reason for a window not to open.
    """
    def broken():
        raise RuntimeError("the renderer package is half-installed")

    monkeypatch.setattr(prefs, "_mandelbrot_defaults", broken)
    lazy = prefs._LazyDefaults()

    with caplog.at_level("DEBUG", logger=prefs.LOG.name):
        assert lazy.get("supersampling", 7) == 7
    assert "could not read the fractal defaults" in caplog.text
    assert lazy._loaded, "a failed load is not retried on every read"
    assert list(lazy.keys()) == []


# ---------------------------------------------------------------------------
# The number fields, which take anything
# ---------------------------------------------------------------------------

def test_a_fractal_field_holding_nothing_at_all_is_named_as_not_a_number():
    """The fields are uncapped, so this function is the only place a value
    that cannot work is refused -- in words, at the point it is entered.
    NaN is the case that gets through ``float()`` and then compares false
    against every bound, so it needs its own answer rather than passing.
    """
    assert prefs.explain_a_fractal_number("supersampling",
                                          float("nan")) == \
        "supersampling: not a number."
    assert prefs.explain_a_fractal_number("supersampling", "two") == \
        "supersampling: 'two' is not a number."
    # A number inside the bounds gets no complaint at all.
    assert prefs.explain_a_fractal_number("supersampling", 2) == ""


# ---------------------------------------------------------------------------
# Safe mode: reads answer with the caller's default, writes go through
# ---------------------------------------------------------------------------

def test_safe_mode_forwards_a_removal_to_the_store_it_shadows(tmp_path):
    """Safe mode exists to CHANGE a saved value, so writes are not shadowed
    -- and clearing a key is a write. A removal that stopped at the shadow
    would leave the value that broke the launch in place, and the next
    ordinary start would die on it again.
    """
    real = QSettings(str(tmp_path / "real.ini"), QSettings.IniFormat)
    real.setValue(prefs._KEY_THEME, "dark")
    shadow = prefs._DefaultsForReadingRealForWriting(real)

    assert shadow.value(prefs._KEY_THEME, "system") == "system"  # not read
    shadow.remove(prefs._KEY_THEME)
    shadow.sync()

    assert real.value(prefs._KEY_THEME, "gone") == "gone"


# ---------------------------------------------------------------------------
# JSON preferences written by another build, or by hand
# ---------------------------------------------------------------------------

def test_the_default_graph_types_survive_a_dict_a_bad_blob_and_a_list(store):
    """The mapping is stored as JSON, but a QSettings backend may hand a
    dict straight back, and a hand-edited file may hold neither. Each has
    one honest answer: the mapping, or none -- never a half-read mapping
    that would draw the wrong graph first.
    """
    store.setValue(prefs._KEY_DEFAULT_GRAPH_TYPES,
                   {"xy": "scatter", "empty": ""})
    assert prefs.get_default_graph_types() == {"xy": "scatter"}
    assert prefs.get_default_graph_type("xy") == "scatter"

    store.setValue(prefs._KEY_DEFAULT_GRAPH_TYPES, '{"xy": "scatter"')
    assert prefs.get_default_graph_types() == {}

    store.setValue(prefs._KEY_DEFAULT_GRAPH_TYPES, '["xy", "scatter"]')
    assert prefs.get_default_graph_types() == {}

    # And the shape it actually writes still round-trips, so the three
    # refusals above are about the blob and not about the reader.
    prefs.set_default_graph_type("xy", "line")
    assert prefs.get_default_graph_type("xy") == "line"


def test_a_style_default_blob_that_is_not_an_object_stores_no_styles(store):
    """``{kind: {field: value}}`` is the only shape that means anything
    here. A JSON list parses and is still not that, so it is discarded
    rather than iterated into a mapping of index to character.
    """
    store.setValue(prefs._KEY_FIG_STYLE_DEFAULTS, '["dpi", 300]')
    assert prefs.get_figure_style_defaults() == {}

    store.setValue(prefs._KEY_FIG_STYLE_DEFAULTS, 'not json at all')
    assert prefs.get_figure_style_defaults() == {}

    store.setValue(prefs._KEY_FIG_STYLE_DEFAULTS,
                   json.dumps({"bar": {"dpi": 300}, "junk": 5}))
    assert prefs.get_figure_style_defaults() == {"bar": {"dpi": 300}}
    assert prefs.get_figure_style_default("bar") == {"dpi": 300}


def test_an_unknown_figure_save_mode_is_refused_by_name(store):
    """The three modes decide what a saved figure looks like on a page.
    Storing a fourth would leave the saver choosing at read time, so the
    setter names the value and the choices instead.
    """
    with pytest.raises(ValueError, match="unknown figure save mode 'inverse'"):
        prefs.set_figure_save_mode("inverse")

    prefs.set_figure_save_mode("Transparent ")
    assert prefs.get_figure_save_mode() == "transparent"


# ---------------------------------------------------------------------------
# A store written by an older build
# ---------------------------------------------------------------------------

def test_a_blur_of_zero_is_marked_migrated_without_inventing_a_resolution(
        store):
    """``ambient_blur`` used to be a buffer-resolution divisor, and the
    migration translates it. Zero is not a divisor -- ``1/0`` is what a
    naive translation would try -- so the value is left alone and only the
    marker is written, which is what stops the migration running again on
    every read.
    """
    store.setValue(prefs._KEY_AMBIENT_BLUR, 0.0)

    prefs._migrate_ambient_motion()

    assert int(store.value(prefs._KEY_AMBIENT_SCALE)) == \
        prefs.AMBIENT_MOTION_SCALE
    assert store.value(prefs._KEY_AMBIENT_RESOLUTION, None) is None

    # A blur that IS a divisor is translated, which is what says the branch
    # above is about the zero and not about the migration doing nothing.
    store.remove(prefs._KEY_AMBIENT_SCALE)
    store.setValue(prefs._KEY_AMBIENT_BLUR, 2.0)
    prefs._migrate_ambient_motion()
    assert float(store.value(prefs._KEY_AMBIENT_RESOLUTION)) == pytest.approx(
        0.5)


def test_a_store_that_cannot_be_written_still_answers_the_migration(
        monkeypatch, caplog):
    """The level is migrated on first read and written back so it is only
    computed once. A read-only store (a config directory somebody's backup
    tool made read-only) must still get the level -- the migration is
    idempotent, so the worst case of not storing it is doing it again.
    """
    class _ReadOnly:
        def __init__(self):
            self.values = {prefs._KEY_LAPTOP_MODE: "on"}

        def value(self, key, default=None, type=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            raise OSError("read-only file system")

        def sync(self):
            pass

    monkeypatch.setattr(prefs, "_settings", _ReadOnly)
    with caplog.at_level("DEBUG", logger=prefs.LOG.name):
        assert prefs.get_performance_level() == "laptop"
    assert "could not store the migrated performance level" in caplog.text


def test_a_mode_from_a_newer_build_still_renders_as_text(store):
    """The dropdown draws `mode_label` for whatever is stored. A stored
    value this build does not know -- a downgrade, a hand-edited file --
    has to render as itself, because a blank row is a row the user cannot
    tell from a missing setting.
    """
    known = next(iter(prefs.MODE_LABELS))
    assert prefs.mode_label(known) == prefs.MODE_LABELS[known]
    assert prefs.mode_label("hyperdrive") == "hyperdrive"


def test_a_log_level_nobody_recognises_is_dropped_not_stored_as_a_string(
        store):
    """The levels are stored as ``"INFO,WARNING"`` and read back into
    numbers. A token that is not a level name resolves to the string
    ``"Level NOTALEVEL"`` rather than to an int, and storing that would put
    a string where the logging module expects a number.
    """
    kept = prefs._parse_levels("INFO,NOTALEVEL,ERROR", fallback=())

    import logging
    assert kept == prefs._parse_levels("INFO,ERROR", fallback=())
    assert logging.INFO in kept and logging.ERROR in kept
    assert all(isinstance(level, int) for level in kept)


# ---------------------------------------------------------------------------
# The fractal settings, which are fields and not sliders
# ---------------------------------------------------------------------------

def test_a_fractal_number_that_is_not_one_reads_back_as_its_default(store):
    """Every fractal reader goes through one ``_number`` helper. A word, a
    missing value and a NaN all have to reach the published default: the
    renderer divides by several of these, and a NaN reaching it is a black
    screen with nothing said.
    """
    store.setValue(prefs._KEY_FRACTAL_SPEED, "quickly")
    speed = prefs.get_fractal_settings()["speed"]
    store.remove(prefs._KEY_FRACTAL_SPEED)
    assert speed == prefs.get_fractal_settings()["speed"]

    store.setValue(prefs._KEY_FRACTAL_SPEED, float("nan"))
    assert prefs.get_fractal_settings()["speed"] == speed

    store.setValue(prefs._KEY_FRACTAL_SPEED, 1.5)
    assert prefs.get_fractal_settings()["speed"] == 1.5


def test_a_stored_boolean_written_as_a_word_is_read_as_one(store):
    """An INI file hands every value back as a string, so ``bool("false")``
    is True and a switch the user turned off comes back on. Both switches
    that a fractal run reads have to parse the word.
    """
    store.setValue(prefs._KEY_FRACTAL_VARIABLE_SPEED, "false")
    store.setValue(prefs._KEY_FRACTAL_POINTER, "false")
    values = prefs.get_fractal_settings()
    assert values["variable_speed"] is False
    assert values["pointer_gravity"] is False

    store.setValue(prefs._KEY_FRACTAL_VARIABLE_SPEED, "on")
    store.setValue(prefs._KEY_FRACTAL_POINTER, "yes")
    values = prefs.get_fractal_settings()
    assert values["variable_speed"] is True
    assert values["pointer_gravity"] is True


def test_an_unknown_fractal_quality_is_refused_beside_the_ones_that_are_not(
        store):
    """Pattern, backend and quality are the three named choices, and each is
    checked against its own list: a quality of ``'best'`` would otherwise be
    stored and then silently ignored by the preset lookup, which is a
    setting that does nothing and says nothing.
    """
    with pytest.raises(ValueError, match="unknown fractal quality 'best'"):
        prefs.set_fractal_settings(quality="best")

    prefs.set_fractal_settings(quality="high")
    assert prefs.get_fractal_settings()["quality"] == "high"


def test_a_number_the_shader_cannot_use_is_moved_to_the_edge_it_can(store):
    """The fields keep the number they were given -- that is the point of
    them -- but the two ends that cannot work at all are moved. The shader's
    loop is bounded at 4096, so a larger iteration count would be silently
    ignored, and a count of zero draws nothing.
    """
    prefs.set_fractal_settings(max_iterations=9000)
    assert prefs.get_fractal_settings()["max_iterations"] == 4096

    prefs.set_fractal_settings(max_iterations=0)
    assert prefs.get_fractal_settings()["max_iterations"] == 1

    prefs.set_fractal_settings(max_iterations=2200)
    assert prefs.get_fractal_settings()["max_iterations"] == 2200


def test_a_steering_number_from_outside_its_range_reads_back_inside_it(store):
    """Steering is the one fractal number that is a proportion, so it has a
    ceiling as well as a floor and the reader holds both: a hand-edited 5
    is a steering strength of one, not of five.
    """
    store.setValue(prefs._KEY_FRACTAL_STEERING, 5.0)
    assert prefs.get_fractal_settings()["steering"] == 1.0

    store.setValue(prefs._KEY_FRACTAL_STEERING, -2.0)
    assert prefs.get_fractal_settings()["steering"] == 0.0

    store.setValue(prefs._KEY_FRACTAL_STEERING, 0.4)
    assert prefs.get_fractal_settings()["steering"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Laptop mode, the interface font, and the two cache numbers
# ---------------------------------------------------------------------------

def test_the_laptop_note_says_what_this_machine_will_do_before_it_is_saved(
        store):
    """Automatic is the choice whose label cannot state the outcome, so the
    note has to: it reports the MEASUREMENT, taken with the environment
    override cleared, because an override set for one launch is not this
    machine's own answer. The other two choices say what they turn down.
    """
    automatic = prefs.laptop_mode_note("automatic")
    on = prefs.laptop_mode_note("on")
    off = prefs.laptop_mode_note("off")

    assert automatic and automatic not in (on, off)
    assert "Turns down" in on and "same answer either way" in on
    assert off == ("Keeps the animation and the blur on, whatever this "
                   "machine is.")

    from spacr.qt.laptop_mode import what_it_turns_down
    for what, _cost in what_it_turns_down():
        assert what in on


def test_an_unknown_laptop_mode_is_refused_by_name(store):
    """Three choices, and the third is not "sometimes". A stored fourth
    would be read as "not on", which is a setting that silently means its
    opposite for the user who typed it.
    """
    with pytest.raises(ValueError, match="unknown laptop mode 'sometimes'"):
        prefs.set_laptop_mode("sometimes")

    prefs.set_laptop_mode("off")
    assert prefs.get_laptop_mode() == "off"


def test_the_font_weight_is_stored_even_when_no_application_is_running(
        store, monkeypatch):
    """The weight is applied to the running QApplication as well as saved,
    and a headless caller has none. Neither the missing application nor a
    theme module that raises may stop the preference being stored -- the
    next launch is what has to read it.
    """
    from spacr.qt import app as qt_app

    def explode(*_args, **_kwargs):
        raise RuntimeError("no font families are registered")

    monkeypatch.setattr(qt_app, "_use_open_sans", explode)
    prefs.set_interface_font_weight("Light")
    assert prefs.get_interface_font_weight() == "light"

    with pytest.raises(ValueError, match="unknown interface font weight"):
        prefs.set_interface_font_weight("thin")


def test_the_two_cache_numbers_fall_back_when_the_store_holds_a_word(store):
    """Both are read at cache-eviction time, in code with no user in front
    of it. A word where a number should be has to become the default rather
    than an exception thrown from a background sweep.
    """
    from spacr.qt.memory_budget import (DEFAULT_CACHE_CEILING_MB,
                                        DEFAULT_IDLE_MINUTES)

    store.setValue(prefs._KEY_IDLE_MINUTES, "quarter of an hour")
    store.setValue(prefs._KEY_CACHE_CEILING, "plenty")
    assert prefs.get_idle_minutes() == DEFAULT_IDLE_MINUTES
    assert prefs.get_cache_ceiling_mb() == DEFAULT_CACHE_CEILING_MB

    prefs.set_idle_minutes(3.0)
    prefs.set_cache_ceiling_mb(512)
    assert prefs.get_idle_minutes() == 3.0
    assert prefs.get_cache_ceiling_mb() == 512


# ---------------------------------------------------------------------------
# Leaving Extra Performance with only part of a stash
# ---------------------------------------------------------------------------

def test_a_stash_holding_one_setting_restores_that_one_and_says_it_did(
        store):
    """The stash is what Extra Performance recorded before turning the
    visuals down, and a stash written by an older build carries fewer keys
    than this one reads. Each is restored only if it is there, and the
    stash is cleared either way so leaving the mode twice cannot put back
    yesterday's values.
    """
    from spacr.qt.widgets.ambient import ANIMATION_CHOICES

    wanted = next(key for key in ANIMATION_CHOICES
                  if key != prefs.get_ambient_animation())
    store.setValue(prefs._KEY_MODE_VISUAL_STASH,
                   json.dumps({"ambient_animation": wanted}))

    assert prefs._restore_visuals() is True
    assert prefs.get_ambient_animation() == wanted
    assert store.value(prefs._KEY_MODE_VISUAL_STASH, None) in (None, "")

    # Nothing left to restore, and it says so rather than claiming it did.
    assert prefs._restore_visuals() is False


# ---------------------------------------------------------------------------
# Moving every tooltip into the hint strip
# ---------------------------------------------------------------------------

def test_a_tooltip_that_will_not_move_is_left_where_it_is(qapp):
    """The strip is the answer, not a second one, so every tooltip in the
    finished dialog is moved into it. A bar that refuses one, and a bar that
    raises on one, are both blemishes rather than reasons for Preferences
    not to open -- so the sweep goes on and reports only what it moved.
    """
    from PySide6.QtWidgets import QLabel, QWidget

    holder = QWidget()
    moved_it, refused_it, broke_on_it = (QLabel("a", holder),
                                         QLabel("b", holder),
                                         QLabel("c", holder))
    for widget, tip in ((moved_it, "moves"), (refused_it, "refused"),
                        (broke_on_it, "raises")):
        widget.setToolTip(tip)
    QLabel("no tooltip at all", holder)

    class _Bar:
        def __init__(self):
            self.seen = []

        def explain(self, widget):
            self.seen.append(widget.toolTip())
            if widget is broke_on_it:
                raise RuntimeError("the strip has gone away")
            return widget is moved_it

    bar = _Bar()
    moved = prefs._everything_explains_itself_in_the_strip(holder, bar)

    assert moved == 1
    assert sorted(bar.seen) == ["moves", "raises", "refused"]
    assert broke_on_it.toolTip() == "raises"


# ---------------------------------------------------------------------------
# The dialog, opened on a store this build no longer understands
# ---------------------------------------------------------------------------

@pytest.fixture
def spaceout(monkeypatch):
    """Turn the fractal backdrop on for one test, and OFF again after.

    It is a module-level flag, so a test that switches it on and walks away
    leaves the Fractal tab on the dialog for every test that follows.
    """
    from spacr.qt import theme

    monkeypatch.setattr(theme, "_SPACEOUT", True)


def _dialog(qtbot):
    """A built Preferences dialog, registered for teardown."""
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    return dialog


def _row(dialog, label_text):
    """The control on the row the user reads as ``label_text``."""
    from PySide6.QtWidgets import QFormLayout, QLabel

    for form in dialog.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            label = form.itemAt(index, QFormLayout.LabelRole)
            widget = label.widget() if label is not None else None
            if isinstance(widget, QLabel) and widget.text() == label_text:
                field = form.itemAt(index, QFormLayout.FieldRole)
                assert field is not None, label_text
                return field.widget()
    raise AssertionError(f"no row labelled {label_text!r}")


#: Each dropdown that walks its own items looking for a stored value: the
#: getter it reads, the row the user sees it on, and a token no build offers.
_POINTED_AT_A_STORED_VALUE = (
    ("get_theme_choice", "Theme", "space:nebula"),
    ("get_ambient_animation", "Animation", "lava lamp"),
    ("get_dock_mode", "App dock", "sideways"),
    ("get_color_blind_mode", "Colour-blind mode", "tetrachromacy"),
    ("get_figure_format", "Figure format", "bmp"),
    ("get_figure_png_dpi", "PNG resolution", 999),
    ("get_performance_level", "Performance", "hyperdrive"),
)


def test_a_stored_choice_this_build_does_not_offer_opens_on_the_first_row(
        store, qtbot, monkeypatch):
    """Every one of these dropdowns walks its own items looking for the
    stored value. A downgrade, a hand-edited INI or a preference this build
    dropped leaves a token none of them holds, and the dialog still has to
    open: on the first row, which is a value the user can see and change.
    Falling through with no current index would draw a blank row instead.
    """
    from PySide6.QtWidgets import QComboBox

    for getter, _label, unknown in _POINTED_AT_A_STORED_VALUE:
        monkeypatch.setattr(prefs, getter, lambda unknown=unknown: unknown)
    # Language is the same row and is checked here too, but by name: the
    # stored language is also what decides which language the dialog draws
    # its own row labels in, so it cannot be found by its label.
    monkeypatch.setattr(prefs, "get_language", lambda: "kl")

    dialog = _dialog(qtbot)

    for getter, label, unknown in _POINTED_AT_A_STORED_VALUE:
        combo = _row(dialog, label)
        assert combo.count() > 1, getter
        assert combo.currentIndex() == 0, getter
        assert combo.currentData() != unknown, getter

    language = dialog.findChild(QComboBox, "LanguagePreference")
    assert language.count() > 1 and language.currentIndex() == 0
    assert language.currentData() != "kl"


def test_a_stored_choice_the_build_does_offer_opens_on_that_row(
        store, qtbot, monkeypatch):
    """The counterpart, and the reason the test above is not vacuous: the
    same walk, over the same dropdowns, pointed at the LAST row each one
    holds. A dialog that always opened on row zero would pass the first
    test and lose the user's choice every time they opened Preferences.
    """
    plain = _dialog(qtbot)
    last = {}
    for getter, label, _unknown in _POINTED_AT_A_STORED_VALUE:
        combo = _row(plain, label)
        last[getter] = combo.itemData(combo.count() - 1)
    assert all(value is not None for value in last.values()), last

    for getter, value in last.items():
        monkeypatch.setattr(prefs, getter, lambda value=value: value)
    dialog = _dialog(qtbot)

    for getter, label, _unknown in _POINTED_AT_A_STORED_VALUE:
        combo = _row(dialog, label)
        assert combo.currentIndex() == combo.count() - 1, getter
        assert combo.currentData() == last[getter], getter


def test_a_stored_language_this_build_does_have_opens_on_its_own_row(
        store, qtbot, monkeypatch):
    """The counterpart for the Language row, which is the one row that has
    to be found by name: reading the stored language is also what decides
    which language the dialog's own labels are drawn in.
    """
    from PySide6.QtWidgets import QComboBox

    plain = _dialog(qtbot).findChild(QComboBox, "LanguagePreference")
    wanted = plain.itemData(plain.count() - 1)
    assert wanted and wanted != plain.itemData(0)

    monkeypatch.setattr(prefs, "get_language", lambda: wanted)
    pointed = _dialog(qtbot).findChild(QComboBox, "LanguagePreference")
    assert pointed.currentData() == wanted
    assert pointed.currentIndex() == pointed.count() - 1


def test_the_starfield_direction_row_points_at_the_stored_direction(
        store, qtbot, monkeypatch):
    """The drift direction is only meaningful for the starfield, so its row
    is built whatever the animation is and hidden when it does not apply.
    It still has to find the stored direction among its own items rather
    than showing the first one whatever is saved.
    """
    from PySide6.QtWidgets import QComboBox

    plain = _dialog(qtbot)
    combo = plain.findChild(QComboBox, "AmbientDriftDirection")
    assert combo is not None and combo.count() > 1
    wanted = combo.itemData(combo.count() - 1)

    monkeypatch.setattr(prefs, "get_ambient_drift_direction",
                        lambda: wanted)
    pointed = _dialog(qtbot).findChild(QComboBox, "AmbientDriftDirection")
    assert pointed.currentData() == wanted


def test_the_backend_note_says_which_renderer_this_machine_will_use(
        store, spaceout, qtbot, monkeypatch):
    """"Automatic" cannot be explained by its label: what it picks depends
    on whether VisPy imports and whether this session has a display at all.
    The note under the row is where the honest answer goes, and each of the
    three reasons has to produce a different sentence -- a single "the CPU
    renderer" would hide the two cases the user could act on.
    """
    from PySide6.QtWidgets import QComboBox, QLabel

    from spacr.qt.widgets import fractal_travel

    dialog = _dialog(qtbot)
    backend = dialog.findChild(QComboBox, "FractalBackend")
    note = dialog.findChild(QLabel, "FractalBackendNote")
    assert backend is not None and note is not None

    def choose(key):
        index = backend.findData(key)
        assert index >= 0, key
        backend.setCurrentIndex(index)

    monkeypatch.setattr(fractal_travel, "platform_can_do_opengl",
                        lambda: False)
    choose("gpu")
    assert "No usable display/OpenGL context" in note.text()

    monkeypatch.setattr(fractal_travel, "platform_can_do_opengl",
                        lambda: True)
    monkeypatch.setattr(fractal_travel, "gpu_is_available", lambda: False)
    choose("auto")
    assert "VisPy is not installed" in note.text()

    monkeypatch.setattr(fractal_travel, "gpu_is_available", lambda: True)
    choose("cpu")
    cpu_text = note.text()
    choose("auto")
    assert note.text().startswith("Automatic: this machine will use the")
    assert cpu_text != note.text() and "Automatic" not in cpu_text
    assert "renderer" in cpu_text


def test_saving_the_fractal_tab_stores_what_the_fields_hold(
        store, spaceout, qtbot):
    """The fractal rows are only written when the tab exists, so Save has to
    reach them under spaceout and leave them alone otherwise. Every number
    on the tab goes to the store in one call, because a partial write would
    leave the renderer holding a mixture of two settings.
    """
    from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QDoubleSpinBox,
                                   QSpinBox)

    dialog = _dialog(qtbot)
    quality = dialog.findChild(QComboBox, "FractalQuality")
    supersampling = dialog.findChild(QSpinBox, "FractalSupersampling")
    scale = dialog.findChild(QDoubleSpinBox, "FractalScale")
    assert None not in (quality, supersampling, scale)

    quality.setCurrentIndex(quality.findData("ultra"))
    supersampling.setValue(3)
    scale.setValue(1.5)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    saved = prefs.get_fractal_settings()
    assert saved["quality"] == "ultra"
    assert saved["supersampling"] == 3
    assert saved["scale"] == pytest.approx(1.5)


def test_a_number_the_renderer_cannot_use_is_said_in_words_on_save(
        store, spaceout, qtbot, monkeypatch):
    """The fields take anything -- that is deliberate -- so Save is where a
    number that cannot work is answered. It is stored as the nearest value
    that works AND the user is told, because a field that silently became
    something else is a control that lies about what it did.
    """
    from PySide6.QtWidgets import (QDialogButtonBox, QMessageBox, QSpinBox)

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *args, **kwargs: warned.append(args[2])))

    dialog = _dialog(qtbot)
    dialog.findChild(QSpinBox, "FractalSupersampling").setValue(0)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert len(warned) == 1
    assert "supersampling: 0 is too small (needs at least 1)" in warned[0]
    assert prefs.get_fractal_settings()["supersampling"] == 1

    # A number the renderer can use warns about nothing at all.
    warned.clear()
    again = _dialog(qtbot)
    again.findChild(QSpinBox, "FractalSupersampling").setValue(2)
    again.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert warned == []
    assert prefs.get_fractal_settings()["supersampling"] == 2


def test_quitting_from_preferences_works_with_no_run_registry_at_all(
        store, qtbot, monkeypatch):
    """The Quit button is the "this machine is not behaving" tool, so it has
    to work on a dialog with no owner window and therefore no register of
    running work: there is nothing to cancel, and the graceful path still
    has to close the window rather than stop at the missing registry.
    """
    from PySide6.QtWidgets import QPushButton

    from spacr.qt import shutdown

    asked = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *args, **kwargs: asked.append(kwargs) or "")
    monkeypatch.setattr(shutdown, "force_quit_now",
                        lambda: pytest.fail("the graceful path force-quit"))

    dialog = _dialog(qtbot)
    assert getattr(dialog.window(), "_runs", None) is None
    dialog.findChild(QPushButton, "QuitSpacrButton").click()

    assert len(asked) == 1 and asked[0]["what"] == "spaCR"
    assert not dialog.isVisible()


def test_reset_leaves_a_dropdown_alone_when_the_default_is_not_on_it(
        store, qtbot, monkeypatch):
    """Reset reads the real getters against an EMPTY store rather than a
    second copy of the defaults. A getter that answers with something the
    dropdown does not hold -- a build whose default moved on -- must leave
    the control where it is instead of clearing it to nothing.
    """
    from PySide6.QtWidgets import QPushButton

    dialog = _dialog(qtbot)
    dock = _row(dialog, "App dock")
    dock.setCurrentIndex(dock.count() - 1)
    chosen = dock.currentData()

    monkeypatch.setattr(prefs, "get_dock_mode", lambda: "sideways")
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert dock.currentData() == chosen

    # A default the dropdown does hold moves it, which is what says the
    # reset above did reach this control.
    monkeypatch.setattr(prefs, "get_dock_mode", lambda: dock.itemData(0))
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert dock.currentIndex() == 0


def test_a_backdrop_that_will_not_restart_does_not_lose_the_saved_numbers(
        store, spaceout, qtbot, monkeypatch, caplog):
    """The new numbers are pushed into the running backdrop and the dive is
    sent back to the surface, because a dive that resumed thirty decades
    down would make a changed starting scale look as though it did nothing.
    A backdrop that cannot be reached is a cosmetic loss -- the preferences
    are already written, and Save must not fail on top of it.
    """
    from PySide6.QtWidgets import QDialogButtonBox, QSpinBox

    from spacr.qt.widgets import fractal_travel

    def gone():
        raise RuntimeError("the backdrop widget has been destroyed")

    monkeypatch.setattr(fractal_travel, "apply_saved_controls", gone)

    dialog = _dialog(qtbot)
    dialog.findChild(QSpinBox, "FractalSupersampling").setValue(3)
    with caplog.at_level("DEBUG", logger=prefs.LOG.name):
        dialog.findChild(QDialogButtonBox).button(
            QDialogButtonBox.Save).click()

    assert "could not restart the dive" in caplog.text
    assert prefs.get_fractal_settings()["supersampling"] == 3


def test_the_font_weight_is_saved_when_there_is_no_application_to_apply_it_to(
        store, monkeypatch):
    """The setter is called from headless paths too -- a migration, a
    settings script -- where there is no QApplication to restyle. The store
    is what the next launch reads, so the write happens either way and the
    live restyle is the part that is skipped.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt import app as qt_app

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    monkeypatch.setattr(qt_app, "_use_open_sans",
                        lambda *a, **k: pytest.fail(
                            "restyled an application that is not running"))

    prefs.set_interface_font_weight("regular")
    assert prefs.get_interface_font_weight() == "regular"
