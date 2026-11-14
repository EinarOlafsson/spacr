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

import os

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
    assert str(tmp_path) in os.path.normpath(prefs._settings().fileName())
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

    # EVERY SHARED KEY, not three of them. The fallback is a hand-written
    # copy -- it has to be, since it runs where the renderer cannot be
    # imported -- and it had drifted on two keys before this compared them
    # all: supersampling and max_depth were still the lighter preset's.
    monkeypatch.undo()
    from spacr.qt.widgets.fractal_mandelbrot import DEFAULTS

    drift = {key: (DEFAULTS[key], fallback[key]) for key in fallback
             if key in DEFAULTS and DEFAULTS[key] != fallback[key]}
    assert not drift, f"the written fallback no longer matches DEFAULTS: {drift}"


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
    assert off == ("Keeps the animation at its chosen settings, whatever this "
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
