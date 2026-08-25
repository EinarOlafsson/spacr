"""A folded module is a button on a host, and it is not also a tile.

``instructions/open/246_settings_appear_when_they_apply.txt`` §4 states the
bar in one line: *no key appears in both a host's fold table and the
registry*. A key in both draws a tile the maintainer has already been told
is gone, and the tile and the button then race for which one a user reaches
first -- the tile opens the module bare, the button opens it seeded by the
host.

ILLUMINATION IS THE FOLD THESE TESTS DRIVE. Its nine ``illumination_*``
keys were filed under Measure's "Illumination Correction" heading, which is
the switch thrown on the run whose numbers it changes, but a settings
category cannot ask for the one thing the module also does on its own:
estimate the field and write the QC figures WITHOUT measuring the plate --
minutes of work that decides whether hours of measuring are worth starting.
So the module keeps its own settings form and Run button as a folded page
on Measure, and only then loses its row. Both routes end in
:func:`spacr.illumination.prepare_illumination_correction`.

The two tests that read the ledgers rather than the screen --
:func:`test_no_key_is_both_a_fold_button_and_a_tile` and
:func:`test_the_folded_module_still_answers_everywhere_its_row_did` -- are
the acceptance bar, and they name what is still owed when they fail.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from importlib import import_module

from spacr.qt.screens import map_barcodes, measure
from spacr.qt.screens.app_screen import AppScreen


#: Every host that folds modules into itself, by the module that owns the
#: strip. Read from the registry's own table rather than typed here, so a
#: host added tomorrow is swept without an edit; ``make_masks`` names its
#: keys ``FOLD_ORDER`` and is added by hand for that reason alone.
def _folded_by_host():
    """``host module name`` → the keys it offers as buttons."""
    hosts = {}
    for _key, module_name in map_barcodes.FOLD_HOST_MODULES.items():
        module = import_module(f"spacr.qt.screens.{module_name}")
        keys = set(getattr(module, "FOLDED_APPS", ()) or ())
        keys |= set(getattr(module, "FOLD_ORDER", ()) or ())
        if keys:
            hosts[module_name] = keys
    for module_name in ("annotate", "make_masks"):
        module = import_module(f"spacr.qt.screens.{module_name}")
        keys = set(getattr(module, "FOLDED_APPS", ()) or ())
        keys |= set(getattr(module, "FOLD_ORDER", ()) or ())
        if keys:
            hosts.setdefault(module_name, set()).update(keys)
    return hosts


def _live_keys():
    """Every registered app key, self-registering screens included."""
    import spacr.qt.screens                     # noqa: F401 - registers rows
    from spacr.qt.app import APPS

    return {row[0] for row in APPS}


def test_no_key_is_both_a_fold_button_and_a_tile():
    """The one line §4 asks for, swept rather than listed by hand.

    Failing here does not mean a fold is broken; it means a fold is only
    half done. The message names the host, so the row to drop and the
    module that registers it can be found from the failure alone.
    """
    live = _live_keys()
    both = sorted(
        (key, host) for host, keys in _folded_by_host().items()
        for key in keys if key in live)
    assert not both, (
        "these keys are buttons on a host AND still draw a tile, so the "
        "same module has two front doors: "
        + ", ".join(f"{key} (folded into {host})" for key, host in both))


def test_illumination_is_not_a_tile():
    """The module the maintainer named: 'i still see illumination'."""
    assert "illumination" not in _live_keys()


def test_illumination_is_the_first_button_on_the_measure_masthead(
        qtbot, qt_theme_applied):
    """Strip order is pipeline order, and the button is the module's icon.

    The field is estimated and divided out before any intensity feature is
    measured, and the AnnData file is written from the tables afterwards, so
    a user reads the strip left to right in the order a plate goes through
    it.
    """
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)
    strip = measure.install_folds(screen)
    assert strip is not None

    assert list(strip.keys()) == ["illumination", "anndata_export",
                                  "motility"]
    button = strip.button_for("illumination")
    assert button.text() == "", "the fold button has a caption"
    assert not button.icon().isNull(), (
        "the Illumination icon has nowhere to go if the button has none")


def test_the_illumination_button_opens_the_module_itself(qtbot,
                                                         qt_theme_applied):
    """Not a summary of it: the settings form and the Run button.

    A fold that offered "estimate the field" with no keys would drop the
    estimator, the degree, the field budget, the dark offset and the QC
    switch -- every choice the module has.
    """
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)
    strip = measure.install_folds(screen)
    strip.button_for("illumination").click()

    opened = [o for o in screen._fold_openers if o.key == "illumination"]
    page = opened[0].window
    assert page is not None, "clicking the button opened nothing"
    qtbot.addWidget(page)
    assert isinstance(page, AppScreen)
    assert page.app_key == "illumination"
    # A PAGE beside the measure settings, not a window over them.
    assert not page.isWindow()
    assert screen._fold_pages.currentWidget() is page
    assert page._btn_run is not None
    assert len(page._settings_model.collect()) > 0


def test_the_page_and_the_measure_panel_offer_the_same_nine_keys(
        qtbot, qt_theme_applied):
    """Two doors, one set of knobs -- measured on both panels.

    Asserted as a set rather than by eye: a key that reached only one of
    the two would be a correction that behaved differently depending on
    which door was used to ask for it.
    """
    from spacr.illumination import illumination_settings

    expected = {key for key in illumination_settings({})
                if key.startswith("illumination_")}
    assert len(expected) == 9

    panels = {}
    for key in ("measure", "illumination"):
        screen = AppScreen(app_key=key)
        qtbot.addWidget(screen)
        panels[key] = {name for name in screen._settings_model._widgets
                       if name.startswith("illumination_")}
    assert panels["measure"] == expected
    assert panels["illumination"] == expected


def test_being_asked_to_run_with_the_switch_off_says_so(capsys, tmp_path):
    """A no-op Run button that prints nothing reads as a broken one.

    ``illumination_correction`` ships False, so pressing Run on the folded
    page before ticking it does nothing at all. It still does nothing --
    that is the contract every measure run depends on -- but it now names
    the switch that was not thrown instead of returning in silence.
    """
    from spacr.illumination import (illumination_settings,
                                    prepare_illumination_correction)
    from spacr import measure_hooks

    settings = illumination_settings({"src": str(tmp_path), "channels": [0]})
    assert settings["illumination_correction"] is False

    assert prepare_illumination_correction(settings) is None

    said = capsys.readouterr().out
    assert "illumination_correction" in said
    assert "OFF" in said
    # ...and it is still a no-op: nothing estimated, nothing installed.
    assert measure_hooks.preprocessing_hooks() == ()
    assert not list(tmp_path.iterdir())


def test_it_stays_silent_when_the_caller_asked_for_silence(capsys, tmp_path):
    """`verbose=False` means a batch of plates does not narrate itself."""
    from spacr.illumination import (illumination_settings,
                                    prepare_illumination_correction)

    settings = illumination_settings({"src": str(tmp_path), "channels": [0],
                                      "verbose": False})

    assert prepare_illumination_correction(settings) is None
    assert capsys.readouterr().out == ""


def test_the_folded_module_still_answers_everywhere_its_row_did():
    """Dropping a row drops five answers with it. Each has a new home.

    ``register_app`` fans one call out into the tables a running window
    reads, so a module that loses its row loses its Run button, its page
    header, its API link, its translated name and the maturity colour its
    button lights in -- all of them silently, and none of them at import.
    The homes are the ones Barcode QC and AnnData Export already use:

    * the pipeline entry in ``spacr.qt.bridge.resolve_pipeline_entry``;
    * the API doc module in ``settings_model._APP_API_MODULE``;
    * the header and the paragraph under it in ``app_screen.APP_TITLES``
      and ``APP_INTROS``;
    * the name, the sentence and the stage in
      ``map_barcodes.FOLD_FALLBACK``, which is what ``restate_fold_button``
      reads and the only carrier of the stage once the registry answers
      "stable" for the key.
    """
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    from spacr.qt.screens.settings_model import (_APP_API_MODULE,
                                                 resolve_default_settings)
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES

    key = "illumination"
    owed = []

    entry = resolve_pipeline_entry(key)
    if entry is None or not callable(entry):
        owed.append("bridge.resolve_pipeline_entry: the Run button on the "
                    "folded page reports 'Not runnable'")
    if key not in _APP_API_MODULE:
        owed.append("settings_model._APP_API_MODULE: the info link beside "
                    "the settings lands on the API index")
    if not APP_TITLES.get(key, "").strip():
        owed.append("app_screen.APP_TITLES: the page is headed 'Illumination' "
                    "rather than 'Illumination Correction'")
    if len(APP_INTROS.get(key, "")) <= 40:
        owed.append("app_screen.APP_INTROS: the page says 'Configure "
                    "settings, then press Run.' and nothing about the module")
    name, description, stage = map_barcodes.fold_description(key)
    if not (name and description and stage):
        owed.append("map_barcodes.FOLD_FALLBACK: the button has no tooltip "
                    "and lights stable-blue for a beta module")
    missing_languages = [code for code in VALID_LANGUAGE_CODES
                         if code != "en"
                         and not CATALOGS[code].get("Illumination", "").strip()]
    if missing_languages:
        owed.append("i18n._ROWS: the page masthead is English in a "
                    + "/".join(missing_languages) + " window")

    assert not owed, (
        "the illumination row is gone and these answers went with it:\n  - "
        + "\n  - ".join(owed))
    # The one answer that survived the row on its own, asserted so a change
    # to the settings seam cannot take it away unnoticed: `spacr.settings`
    # imports `spacr.illumination` at its own bottom, so the defaults are
    # registered before any panel asks for them.
    assert len(resolve_default_settings(key)) > 1


def test_the_headless_paths_never_went_through_the_row():
    """`spacr-run illumination` and pre-flight are untouched by the fold."""
    from spacr import cli
    from spacr.validate import APP_FUNCTIONS

    assert "illumination" in cli.MODULES
    assert cli.resolve_module("illumination") is not None
    assert APP_FUNCTIONS["illumination"] == (
        "spacr.illumination.prepare_illumination_correction")
