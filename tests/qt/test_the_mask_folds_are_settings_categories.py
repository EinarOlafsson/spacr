"""Timelapse and Motility fold into Mask Generation without opening anything.

Both are the mask pipeline with a flag on. ``preprocess_generate_masks_
timelapse`` forces ``timelapse=True`` and then calls
``preprocess_generate_masks`` unchanged. (The motility assay used to fold
here too and does not any more: it reads finished masks and writes a
measurements table, so it belongs to Measure. What remains here is
both true. So neither is a second screen -- what is theirs is a handful of
settings categories, and the button reveals them on the form the user is
already looking at.

What these tests protect:

* NOTHING OPENS. A window is the last resort for a fold and neither of
  these needs one, so a button that opened one would be a regression.
* NOTHING IS DUPLICATED. A setting has one control or it has two sources
  of truth; ``collect()`` is keyed on the setting name, so a second widget
  for ``src`` would silently replace the host's own.
* NOTHING IS LOST. Every setting the folded module's own screen resolves
  has to reach the run from here, and the gate has to be on when -- and
  only when -- the switch is.
* THE ASSAY BRINGS TRACKING WITH IT, because it runs inside the timelapse
  branch and a form offering it beside no tracking would describe a run
  that cannot happen.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from spacr.qt.screens import mask as mask_folds
from spacr.qt.screens import map_barcodes
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import resolve_default_settings
from spacr.qt.widgets.fold_strip import BUTTON_NAME, FoldStrip


@pytest.fixture
def host(qtbot, qt_theme_applied):
    """A Mask Generation screen with its two switches installed."""
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    strip = mask_folds.install_folds(screen)
    assert strip is not None, "Mask got no fold strip"
    return screen, strip


def _sections(screen, key):
    """The category cards one fold mounted on the host."""
    return mask_folds.fold_set(screen).folds[key].sections


# ---------------------------------------------------------------------------
# The switches
# ---------------------------------------------------------------------------

def test_the_series_module_is_a_switch_on_the_mask_masthead(host):
    """One checkable icon button, and no caption beside it."""
    screen, strip = host

    assert list(strip.keys()) == list(mask_folds.FOLDED_APPS)
    named = [button for button in strip.findChildren(QPushButton)
             if button.objectName() == BUTTON_NAME]
    assert len(named) == len(mask_folds.FOLDED_APPS)
    for button in named:
        assert button.isCheckable(), f"{button.app_key} is not a switch"
        assert not button.text(), f"{button.app_key} drew a caption"
        assert button.accessibleName()


def test_a_switch_carries_the_name_sentence_and_stage_its_tile_had(host):
    """The button must still be recognisable as the module it replaced.

    Asked of ``fold_description`` and not of ``app_stage``: the fold
    outlives its registry row, and once the row is dropped the registry
    answers "stable" with no name and no sentence for every one of these
    keys. What the button has to go on saying is what the TILE said.
    """
    _screen, strip = host

    for key in mask_folds.FOLDED_APPS:
        button = strip.button_for(key)
        name, description, stage = map_barcodes.fold_description(key)
        assert name and description
        assert button.toolTip() == f"{name}\n{description}"
        assert button.property("stage") == stage


def test_the_recorded_stage_is_real_and_agrees_with_the_assessment(host):
    """The fold fallback is the only record left of what the tile lit.

    Once the row is dropped neither ``APP_STAGE`` nor ``maturity`` says
    anything about the key -- ``app_stage`` answers "stable" for every
    module it has never heard of, which is why the fallback exists at
    all. What can still be checked is that the stage it records is one
    the theme has a colour for, and that it does not contradict
    ``maturity`` wherever maturity did assess the module: two tables
    saying different things about one module's maturity is the drift the
    fallback was written to avoid.
    """
    from spacr.qt import maturity
    from spacr.qt.theme import STAGE_HOVER

    for key in mask_folds.FOLDED_APPS:
        _name, _description, stage = map_barcodes.fold_description(key)
        assert stage in STAGE_HOVER, f"{key} records an unpaintable stage"
        assessed = (maturity.PROMOTIONS.get(key)
                      or maturity.AFFIRMED.get(key))
        if assessed is not None:
            assert stage == assessed[0], (
                f"{key}: the fold says {stage!r}, maturity says "
                f"{assessed[0]!r}")


def test_the_lit_state_is_the_stage_colour_the_tile_used(host):
    """A switch says it is on while nobody is touching it.

    Hover and pressed are momentary; being part of the run is not. The
    colour is read from the maturity table rather than retyped, so this
    asserts the module's own hue rather than a literal.
    """
    from spacr.qt.theme import STAGE_HOVER

    _screen, strip = host
    button = strip.button_for("timelapse")
    rule = button.styleSheet()

    assert ":checked" in rule
    assert STAGE_HOVER[map_barcodes.fold_description("timelapse")[2]] in rule


def test_pressing_a_switch_opens_no_window(host):
    """A window is the last resort, and neither of these needs one."""
    screen, strip = host
    before = set(QApplication.topLevelWidgets())

    strip.button_for("timelapse").setChecked(True)
    strip.button_for("timelapse").setChecked(True)

    assert set(QApplication.topLevelWidgets()) - before == set()
    assert not hasattr(screen, "_fold_openers")


# ---------------------------------------------------------------------------
# What the switch reveals
# ---------------------------------------------------------------------------

def test_the_categories_are_hidden_until_the_switch_is_pressed(host):
    """A Mask run is not a time series until somebody says so."""
    screen, strip = host

    for key in mask_folds.FOLDED_APPS:
        cards = _sections(screen, key)
        assert cards, f"{key} mounted no categories"
        assert not any(card.isVisibleTo(screen) for card in cards)

    strip.button_for("timelapse").setChecked(True)

    assert all(card.isVisibleTo(screen)
               for card in _sections(screen, "timelapse"))


def test_the_tracking_categories_are_the_ones_mask_does_not_have(host):
    """Only what is uniquely the folded module's is mounted.

    The two modules share the folder, the channels, the segmentation
    models, the filters and the outputs; a fold that re-mounted those
    would put a second control on the form for settings that already have
    one.
    """
    screen, _strip = host
    titles = {card.title() for card in _sections(screen, "timelapse")}

    assert titles == {"TRACKING SETUP", "TRACKING BACKENDS"}
    assert "timelapse_mode" in screen._settings_model._widgets
    assert "IMAGE PREPROCESSING" not in titles


def test_no_setting_gets_a_second_control(host):
    """One key, one widget: `collect()` is keyed on the setting name."""
    screen, _strip = host
    model = screen._settings_model
    folds = mask_folds.fold_set(screen)

    mounted = [key for fold in folds.folds.values()
               for key in fold.settings_keys]
    assert len(mounted) == len(set(mounted))
    for fold in folds.folds.values():
        for key in fold.settings_keys:
            # The host's map holds the widget the fold mounted, and the
            # fold's own model built exactly one of them.
            assert model._widgets[key] is fold.model._widgets[key]


def test_a_folded_setting_answers_into_the_host_hint_strip(host):
    """Hovering a tracking setting fills the strip every other one fills.

    A row wired to the module's own screen would answer into a screen
    nobody is looking at, so the label is filtered by the HOST and the
    hint is looked up in the host's settings model.
    """
    screen, _strip = host
    card = [c for c in _sections(screen, "timelapse")
            if c.title() == "TRACKING SETUP"][0]
    labels = {str(child.property("settingKey")): child
              for child in card.findChildren(QLabel)
              if child.property("settingKey")}
    assert "timelapse_objects" in labels

    screen._hint_strip.setText("")
    screen.eventFilter(labels["timelapse_objects"], QEvent(QEvent.Enter))

    expected = screen._settings_model.plain_tooltip_for("timelapse_objects")
    assert expected
    assert screen._hint_strip.text() == expected


# ---------------------------------------------------------------------------
# The panel the fold brought with it
# ---------------------------------------------------------------------------

def _preview(screen, key):
    """The folded module's attached preview host on ``screen``."""
    return getattr(screen, "_folded_previews", {}).get(key)


def _shown(widget):
    """Whether ``widget`` would be on screen if its window were."""
    return widget.isVisibleTo(widget.parentWidget())


def test_the_switch_brings_the_folded_modules_own_preview_with_it(host):
    """A fold must not cost the panel that shows what its settings do.

    Mask Generation's own preview segments one field with Cellpose and
    answers "did the mask come out right". Timelapse's answers "did the
    objects link up", which is the only question its settings are about,
    and it was built by a screen that the fold means nobody opens any
    more -- so without this the tracking settings would arrive with
    nothing to judge them by.
    """
    screen, strip = host
    preview = _preview(screen, "timelapse")

    assert preview is not None, "the track preview did not come with the fold"
    assert type(preview.panel).__name__ == "TimelapsePreviewPanel"
    assert preview.toggle.text() == "Track preview"


def test_the_track_preview_is_not_offered_until_tracking_is_on(host):
    """Mask Generation must not grow a track preview for an untracked run."""
    screen, strip = host
    preview = _preview(screen, "timelapse")

    assert not _shown(preview.toggle)
    assert not _shown(preview.card)

    strip.button_for("timelapse").setChecked(True)

    assert _shown(preview.toggle)
    # Offered, not opened: a preview costs a Cellpose pass over a field,
    # so it waits to be asked for like every other one.
    assert not _shown(preview.card)

    preview.toggle.setChecked(True)
    assert _shown(preview.card)


def test_switching_tracking_off_closes_the_track_preview(host):
    """A card left open would be showing tracks for an untracked run."""
    screen, strip = host
    preview = _preview(screen, "timelapse")
    strip.button_for("timelapse").setChecked(True)
    preview.toggle.setChecked(True)
    assert _shown(preview.card)

    strip.button_for("timelapse").setChecked(False)

    assert not preview.toggle.isChecked()
    assert not _shown(preview.card)
    assert not _shown(preview.toggle)


def test_the_hosts_own_preview_is_untouched_by_the_fold(host):
    """The Cellpose live preview is Mask's and stays Mask's."""
    screen, strip = host
    strip.button_for("timelapse").setChecked(True)

    assert screen._live_preview_card is not None
    assert screen._live_preview_card is not _preview(screen,
                                                      "timelapse").card


# ---------------------------------------------------------------------------
# What the run is handed
# ---------------------------------------------------------------------------

def test_the_gate_is_on_only_while_the_switch_is(host):
    """The pipeline flag follows the button, in both directions."""
    screen, strip = host

    assert screen._settings_model.collect()["timelapse"] is False

    strip.button_for("timelapse").setChecked(True)
    assert screen._settings_model.collect()["timelapse"] is True

    strip.button_for("timelapse").setChecked(False)
    assert screen._settings_model.collect()["timelapse"] is False


@pytest.mark.parametrize("key", ["timelapse"])
def test_the_run_is_handed_everything_the_module_would_have_handed_it(
        host, key):
    """No capability is lost: every one of the module's settings arrives.

    Including the ones neither screen draws a control for -- the folded
    module's own screen resolves them as defaults and passes them on, so
    the host has to as well or the pipeline falls back to a different
    default than the module would have.
    """
    screen, strip = host
    strip.button_for(key).setChecked(True)
    collected = screen._settings_model.collect()

    missing = sorted(set(resolve_default_settings(key)) - set(collected))
    assert missing == []


def test_a_tracking_backend_typed_here_reaches_the_run(host):
    """The mounted controls are read by the host's collect, not just shown."""
    screen, strip = host
    strip.button_for("timelapse").setChecked(True)

    assert screen._settings_model.set_value_for_key("timelapse_mode", "btrack")

    assert screen._settings_model.collect()["timelapse_mode"] == "btrack"


# ---------------------------------------------------------------------------
# A settings file written by the folded module
# ---------------------------------------------------------------------------

def test_loading_a_timelapse_settings_file_moves_the_switch(host):
    """The controls take their values; the gate has no control to take one.

    Driven through ``apply_settings_dict`` alone, which is the one door
    every settings file, demo layout and chained hand-off comes in by. It
    used to fill in every tracking knob and leave tracking switched off,
    because the gate has no widget for a bulk apply to land in -- calling
    ``sync_folds`` by hand in the test proved the mechanism and not the
    wiring.
    """
    screen, strip = host

    switched = screen.apply_settings_dict({"timelapse": "True",
                                            "timelapse_mode": "trackpy"})

    assert switched
    assert screen._settings_model.collect()["timelapse_mode"] == "trackpy"
    assert strip.button_for("timelapse").isChecked()
    assert screen._settings_model.collect()["timelapse"] is True


def test_the_console_says_which_module_the_file_switched_on(host):
    """Importing a settings CSV reports what it did, not what it ignored.

    The notice used to say tracking had been IGNORED and send the reader
    to a sidebar row -- true when tracking was a module of its own, and
    wrong in both halves now that it is a switch on this form.
    """
    screen, strip = host
    written = []
    screen._console.append_notice = (
        lambda text, **kw: written.append(text.format(**kw)))

    screen.apply_settings_dict({"timelapse": "True"})
    screen._warn_about_moved_settings({"timelapse": "True"})

    assert strip.button_for("timelapse").isChecked()
    said = " ".join(written)
    assert "switched Timelapse on" in said
    assert "ignored" not in said
    assert "sidebar" not in said


def test_a_screen_with_no_switch_says_the_flag_was_ignored(qtbot,
                                                             qt_theme_applied):
    """An honest report of the case the switch cannot answer.

    A Mask screen that never got its strip -- the window hook did not run,
    or the masthead could not carry one -- collects no `timelapse` key, so
    the flag really is dropped and saying it landed would be a lie.
    """
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    written = []
    screen._console.append_notice = (
        lambda text, **kw: written.append(text.format(**kw)))

    screen.apply_settings_dict({"timelapse": "True"})
    screen._warn_about_moved_settings({"timelapse": "True"})

    assert mask_folds.fold_set(screen) is None
    assert "was ignored" in " ".join(written)


def test_the_assay_is_reported_as_a_module_measure_opens(host):
    """The motility gate has no switch here, and the notice says where it is.

    The assay reads finished masks and writes a measurements table, so it
    folded onto Measure as a screen of its own rather than onto this form
    as a category -- and a CSV asking for it here is asking for something
    this screen does not collect.
    """
    screen, _strip = host
    written = []
    screen._console.append_notice = (
        lambda text, **kw: written.append(text.format(**kw)))

    screen.apply_settings_dict({"motility_analysis": "True"})
    screen._warn_about_moved_settings({"motility_analysis": "True"})

    said = " ".join(written)
    assert "motility_analysis=True was ignored" in said
    assert "Measure" in said
    assert "motility_analysis" not in screen._settings_model.collect()


def test_a_settings_file_without_the_gate_leaves_the_switch_alone(host):
    """A plain Mask settings file must not turn tracking on."""
    screen, strip = host
    strip.button_for("timelapse").setChecked(True)

    assert mask_folds.sync_folds(screen, {"src": "/tmp/plate"}) == ()

    assert not strip.button_for("timelapse").isChecked()
    assert screen._settings_model.collect()["timelapse"] is False


def test_syncing_a_screen_with_no_folds_is_a_no_op(qtbot):
    """Safe to call from the one place every screen's settings land in."""
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)

    assert mask_folds.sync_folds(screen, {"timelapse": True}) == ()


# ---------------------------------------------------------------------------
# Installing the strip
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_mask_gets_no_switches(qtbot, qt_theme_applied):
    """The seam that calls this can be wrong without consequence."""
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)

    assert mask_folds.install_folds(screen) is None


def test_installing_twice_does_not_add_a_second_strip(host):
    """A screen reached twice by the stack watcher keeps one strip."""
    screen, strip = host

    assert mask_folds.install_folds(screen) is strip


def test_a_screen_with_no_masthead_gets_no_switches(qtbot):
    """A strip needs somewhere to hang; a missing one costs the buttons."""
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    screen._header = None

    assert mask_folds.install_folds(screen) is None


def test_the_stack_walk_routes_a_mask_screen_here(qtbot, qt_theme_applied):
    """Mask is reached by the same walk every other host is reached by."""
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)

    strip = map_barcodes.install_folds_on(screen)

    assert isinstance(strip, FoldStrip)
    assert list(strip.keys()) == list(mask_folds.FOLDED_APPS)


def test_a_strip_that_cannot_be_built_never_takes_the_host_down(
        qtbot, qt_theme_applied, monkeypatch):
    """A screen without its switches is smaller; one that raises is gone."""
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)

    def explode(*_args, **_kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(mask_folds, "CategoryFoldSet", explode)

    assert mask_folds.install_folds(screen) is None


def test_a_module_with_nothing_of_its_own_gets_no_button(
        qtbot, qt_theme_applied):
    """A fold that adds no category would be a button revealing nothing.

    Mask folded into itself is the degenerate case: every setting is
    already on the form, so nothing mounts and the key is dropped from the
    set instead of drawing a switch over an empty section.
    """
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    folds = map_barcodes.CategoryFoldSet(screen, {"mask": ("timelapse",)})

    assert folds.mount() == ()
    assert folds.build_strip(screen) is None


def test_a_switch_with_an_unknown_maturity_invents_no_colour(
        qtbot, qt_theme_applied, monkeypatch):
    """A stage the table has never heard of leaves the shipped rules alone.

    The lit state is read from the maturity table, and a button that made
    one up would light in a colour no tile lights in.
    """
    from spacr.qt.widgets.fold_strip import FoldButton

    monkeypatch.setattr("spacr.qt.theme.STAGE_HOVER", {})

    button = FoldButton("timelapse", checkable=True)
    qtbot.addWidget(button)

    assert button.isCheckable()
    assert button.styleSheet() == ""
