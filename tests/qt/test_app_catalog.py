"""The declared app catalog: the row without the screen.

Registration used to import 1,281 modules to learn some strings. These hold
the two halves of the replacement to account: the registry the launch builds
must be exactly what it was, and the screens behind it must NOT be imported
to build it. A change that keeps the first and loses the second has moved the
cost rather than removed it, and the module-count tests below are what say so.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from spacr.qt import SELF_REGISTERING_MODULES
from spacr.qt import app as app_mod
from spacr.qt.app_catalog import (DECLARED_APPS, DeclaredApp,
                                  LazyScreenFactory, declared_app,
                                  declared_for, register_declared)


@pytest.fixture(autouse=True)
def registered():
    """The registry a launched GUI has, not the one a bare import leaves.

    ``import spacr.qt.app`` fills in the rows named in its own table; the
    rest arrive when :func:`spacr.qt.register_self_registering_modules` walks
    ``SELF_REGISTERING_MODULES``, which is what ``run()`` does before the
    window is built. Every assertion here is about the registry the window
    reads, so it has to be that one. ``conftest``'s ``_restore_app_registry``
    puts it back afterwards.
    """
    import spacr.qt

    spacr.qt.register_self_registering_modules()


def _in_a_cold_process(body: str) -> dict:
    """Run ``body`` in a fresh interpreter and return the JSON it prints.

    A cold one, because everything here is about what a launch imports and
    this process has already imported most of it.
    """
    done = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True,
        timeout=600,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
             "CUDA_VISIBLE_DEVICES": ""})
    line = next((l for l in done.stdout.splitlines() if l.startswith("{")), "")
    assert line, f"the probe printed nothing usable:\n{done.stdout}\n{done.stderr}"
    return json.loads(line)


# ---------------------------------------------------------------------------
# the row is what the module says it is
# ---------------------------------------------------------------------------

_CONSTANTS = (
    ("APP_NAME", "name"),
    ("APP_DESCRIPTION", "desc"),
    ("APP_DESC", "desc"),
    ("APP_INTRO", "intro"),
    ("APP_CLI_NOTE", "cli_note"),
    ("APP_NAME_TRANSLATIONS", "translations"),
    ("APP_TRANSLATIONS", "translations"),
)


@pytest.mark.parametrize("row", DECLARED_APPS, ids=lambda r: r.key)
def test_the_module_agrees_with_the_row_declared_for_it(row: DeclaredApp):
    """Whatever the screen still calls its own name is the catalog's name.

    The strings MOVED into the catalog rather than being copied there, so a
    screen that defines ``APP_NAME`` now reads it back from its row. This
    asserts that is really what happened: a module that starts spelling its
    own name again — a merge that resurrects the literal, a copy-paste from a
    sibling — is a module with two names, and the tile and the screen header
    would stop agreeing without anything failing.
    """
    module = __import__(row.module, fromlist=["*"])
    for constant, field in _CONSTANTS:
        if not hasattr(module, constant):
            continue
        assert getattr(module, constant) == getattr(row, field), (
            f"{row.module}.{constant} is not the {field!r} declared for "
            f"{row.key!r} in spacr.qt.app_catalog")


@pytest.mark.parametrize("row", DECLARED_APPS, ids=lambda r: r.key)
def test_the_named_factory_exists_and_is_callable(row: DeclaredApp):
    """The row names a factory; the module must actually have one.

    A row holds the factory's NAME, which is what lets it be registered
    without importing anything — and also what lets a rename go unnoticed
    until a user clicks the tile and gets an empty screen. This is the check
    that makes the rename fail at test time instead.
    """
    module = __import__(row.module, fromlist=["*"])
    assert row.factory, f"{row.key} declares no factory"
    factory = getattr(module, row.factory, None)
    assert callable(factory), (
        f"{row.module} has no callable {row.factory!r}; the row in "
        f"spacr.qt.app_catalog names a factory that is not there")


def test_every_declared_section_and_stage_is_one_the_registry_knows():
    """The catalog spells the section out; the spelling has to be right.

    ``app_catalog`` cannot import ``app`` — ``app`` reads the table while it
    is being imported — so the sections and stages are literals rather than
    references, and a typo would be a section with one app in it that no tab
    shows.
    """
    for row in DECLARED_APPS:
        assert row.section in app_mod.SECTION_ORDER, (
            f"{row.key} declares section {row.section!r}, which is not in "
            f"SECTION_ORDER: {app_mod.SECTION_ORDER}")
        assert row.stage in app_mod.STAGES, (
            f"{row.key} declares stage {row.stage!r}, not one of "
            f"{app_mod.STAGES}")


def test_no_two_rows_claim_the_same_key_or_module():
    keys = [row.key for row in DECLARED_APPS]
    modules = [row.module for row in DECLARED_APPS]
    assert len(set(keys)) == len(keys), "duplicate key in DECLARED_APPS"
    assert len(set(modules)) == len(modules), "duplicate module in DECLARED_APPS"


# ---------------------------------------------------------------------------
# the registry the launch builds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", DECLARED_APPS, ids=lambda r: r.key)
def test_the_registry_row_is_the_declared_row(row: DeclaredApp):
    """Key, name, description, section, stage — the tile and the sidebar.

    The stage is asserted as a floor rather than an equality, because
    ``spacr.qt.maturity`` runs last and PROMOTES the apps whose evidence no
    longer matches what they declared. Promotion is the only direction it
    moves in, so "at least what the row declares" is the invariant; an
    equality here would fail the moment a screen earned its beta.
    """
    rows = [entry for entry in app_mod.APPS if entry[0] == row.key]
    assert len(rows) == 1, f"{row.key} appears {len(rows)} times in APPS"
    assert rows[0] == (row.key, row.name, row.desc, row.section)
    registered = app_mod.APP_STAGE.get(row.key)
    assert registered in app_mod.STAGES
    assert app_mod.STAGES.index(registered) >= app_mod.STAGES.index(row.stage), (
        f"{row.key} is registered as {registered!r}, earlier than the "
        f"{row.stage!r} its row declares — nothing demotes an app")


@pytest.mark.parametrize("row", DECLARED_APPS, ids=lambda r: r.key)
def test_the_metadata_fanned_out_is_the_declared_metadata(row: DeclaredApp):
    """The six tables ``register_app`` distributes into, from one row.

    ``title`` and ``intro`` carry ``register_app``'s own fallbacks — a row
    that gives neither gets the name and the description — so the expected
    value is the fallback, not the empty string.
    """
    meta = app_mod.APP_META[row.key]
    assert meta["name"] == row.name
    assert meta["title"] == (row.title or row.name)
    assert meta["intro"] == (row.intro or row.desc)
    assert meta["cli_note"] == row.cli_note
    assert meta["api_module"] == row.api_module
    assert meta["entry"] == row.entry
    assert meta["defaults_module"] == row.defaults_module
    assert meta["translations"] == row.translations


def test_the_gui_only_sentence_reaches_spacr_run():
    """``spacr-run <key>`` answers for a declared app it never imports.

    The PULL half of the metadata seam, and the half a lazy registration
    could quietly break: ``spacr.cli`` is not imported at launch, so nothing
    pushes into it — it collects what it missed when it is finally imported.
    """
    from spacr import cli

    for row in DECLARED_APPS:
        if not row.cli_note:
            continue
        assert cli.INTERACTIVE_ONLY.get(row.key) == row.cli_note, (
            f"spacr-run {row.key} does not print the note declared for it")


# ---------------------------------------------------------------------------
# ... and the screens it does NOT import to build it
# ---------------------------------------------------------------------------

_REGISTRATION_PROBE = r"""
import json, sys
from PySide6.QtWidgets import QApplication
QApplication([])
import spacr.qt as qt
before = set(sys.modules)
import spacr.qt.app
qt.register_self_registering_modules()
from spacr.qt.app_catalog import DECLARED_APPS
from spacr.qt.app import APPS
print(json.dumps({
    "imported_screens": sorted(row.module for row in DECLARED_APPS
                               if row.module in sys.modules),
    "heavy": sorted(p for p in ("pandas", "scipy", "sklearn", "torch")
                    if p in sys.modules),
    "modules": len(set(sys.modules) - before),
    "apps": len(APPS),
}))
"""


def test_registering_imports_none_of_the_screens_it_registers():
    """The whole point, asserted directly.

    Every declared module must still be absent from ``sys.modules`` after the
    launch has finished registering — the registry is built from the table,
    and the screen is imported when somebody opens it.
    """
    data = _in_a_cold_process(_REGISTRATION_PROBE)
    assert data["imported_screens"] == [], (
        "these declared screens were imported while registering, which is "
        f"the cost the catalog exists to remove: {data['imported_screens']}")


def test_registering_pulls_in_no_scientific_stack():
    """pandas, scipy and sklearn are not needed to learn an app's name.

    Named individually because each arrived by its own route and each was
    worth several hundred modules: scipy behind the Dose-Response fitter,
    sklearn behind the classifier metrics, pandas behind the feature
    dictionary.
    """
    data = _in_a_cold_process(_REGISTRATION_PROBE)
    assert data["heavy"] == [], (
        f"registering imported {data['heavy']}, which no registry row needs")


#: What ``import spacr.qt.app`` plus the launch registration may import.
#:
#: It was 1,281 before the catalog. The budget is deliberately far above what
#: it costs now and far below what it cost then: this is a ratchet against the
#: cost creeping back one module-level import at a time, not a target.
MODULE_BUDGET = 400


def test_the_registration_walk_stays_inside_its_module_budget():
    data = _in_a_cold_process(_REGISTRATION_PROBE)
    assert data["modules"] < MODULE_BUDGET, (
        f"importing spacr.qt.app and registering every app took "
        f"{data['modules']} modules, over the budget of {MODULE_BUDGET}. "
        f"Something on that path grew a module-level import of a heavy "
        f"library; find it with `python -X importtime`.")


def test_every_app_is_registered_by_the_time_the_window_reads_the_registry():
    """Laziness must not cost a single tile.

    The registry is read by ``MainWindow.__init__``; an app registered after
    that has no tile, no sidebar row and no shortcut. The count is checked in
    the same cold process that checks nothing was imported, so the two can
    never be true separately.
    """
    data = _in_a_cold_process(_REGISTRATION_PROBE)
    assert data["apps"] == len(app_mod.APPS)
    assert data["apps"] >= len(DECLARED_APPS)


# ---------------------------------------------------------------------------
# the stand-in
# ---------------------------------------------------------------------------

def test_registered_factory_hands_back_the_real_function():
    """``registered_factory`` resolves the stand-in, and caches the result.

    Everything downstream reads the factory's signature to decide what to
    pass it, so a stand-in must never be what they see.
    """
    row = declared_app("trellis")
    app_mod.APP_FACTORIES[row.key] = LazyScreenFactory(row.module, row.factory)
    resolved = app_mod.registered_factory(row.key)
    module = __import__(row.module, fromlist=["*"])
    assert resolved is getattr(module, row.factory)
    # Cached: the stand-in is gone from the table, not merely bypassed.
    assert app_mod.APP_FACTORIES[row.key] is resolved
    assert app_mod.registered_factory(row.key) is resolved


def test_a_stand_in_whose_module_is_broken_does_not_take_the_window_down():
    """An unimportable screen costs its own tile and nothing else."""
    app_mod.APP_FACTORIES["probe_key"] = LazyScreenFactory(
        "spacr.qt.no_such_screen_module", "make_screen")
    try:
        assert app_mod.registered_factory("probe_key") is None
    finally:
        app_mod.APP_FACTORIES.pop("probe_key", None)


def test_a_stand_in_passes_only_the_arguments_the_real_factory_declares():
    """The "take what you need" contract survives the indirection.

    ``register_app`` lets a factory declare ``app_key`` and/or ``host`` and
    promises it is given whichever it accepts. A stand-in forwarding
    ``**kwargs`` blindly would hand both to ``lambda: Screen()`` and raise
    ``TypeError`` on every open.
    """
    lazy = LazyScreenFactory("spacr.qt.app_catalog", "_probe_zero_arg")
    seen = {}

    def zero_arg():
        seen["called"] = True
        return "screen"

    lazy._resolved = zero_arg
    assert lazy(app_key="trellis", host=object()) == "screen"
    assert seen["called"]

    def takes_key(app_key=None):
        return app_key

    lazy._resolved = takes_key
    assert lazy(app_key="trellis", host=object()) == "trellis"

    def takes_everything(**kwargs):
        return sorted(kwargs)

    lazy._resolved = takes_everything
    assert lazy(app_key="trellis", host=None) == ["app_key", "host"]


def test_a_stand_in_says_what_it_is_before_and_after_resolving():
    lazy = LazyScreenFactory("spacr.qt.screens.trellis", "make_trellis_screen")
    assert "not imported" in repr(lazy)
    lazy.resolve()
    assert "resolved" in repr(lazy)


# ---------------------------------------------------------------------------
# registering from the row
# ---------------------------------------------------------------------------

def test_register_declared_is_idempotent():
    """Three paths reach the same registration; a duplicate key raises."""
    row = declared_app("outliers")
    assert register_declared(row.module) is None, (
        "the app is already registered, so a second call must be a no-op")
    module = __import__(row.module, fromlist=["*"])
    assert module.register() is False


def test_register_declared_puts_a_removed_row_back():
    row = declared_app("trellis")
    app_mod.unregister_app(row.key)
    try:
        added = register_declared(row.module)
        assert added == (row.key, row.name, row.desc, row.section)
        assert app_mod.APP_META[row.key]["translations"] == row.translations
        assert isinstance(app_mod.APP_FACTORIES[row.key], LazyScreenFactory), (
            "registering from the row must not import the screen")
    finally:
        app_mod.unregister_app(row.key)
        register_declared(row.module)


def test_register_declared_ignores_a_module_that_declares_nothing():
    assert declared_for("spacr.qt.maturity") is None
    assert register_declared("spacr.qt.maturity") is None


def test_an_override_places_the_app_somewhere_else():
    """``section``/``stage``/``key`` overrides, which two registrars expose."""
    row = declared_app("lineage")
    app_mod.unregister_app("lineage_probe")
    added = register_declared(row.module, key="lineage_probe",
                              section=app_mod.SECTION_DATA,
                              stage=app_mod.STAGE_BETA)
    try:
        assert added == ("lineage_probe", row.name, row.desc,
                         app_mod.SECTION_DATA)
        assert app_mod.APP_STAGE["lineage_probe"] == app_mod.STAGE_BETA
        # ... and the real row is untouched by the copy.
        assert app_mod.APP_STAGE["lineage"] == row.stage
    finally:
        app_mod.unregister_app("lineage_probe")


def test_the_self_registering_list_is_declared_rows_and_real_work():
    """Every entry is one or the other, and the ones left do real work.

    A module that is still imported at launch is a module whose registration
    is not just a row — it wraps another screen's factory, installs a hook,
    or reassesses what is already registered. If one of these ever becomes a
    plain ``register_app`` call, it belongs in the catalog.
    """
    does_real_work = {
        "spacr.qt.widgets.feature_dictionary",  # also registers its QSS block
        "spacr.qt.chaining",                    # wraps every ported factory
        "spacr.qt.prerun",                      # wraps two of them
        "spacr.qt.resource_cleanup",            # installs the pre-run hook
        "spacr.qt.maturity",                    # re-reads the whole registry
    }
    for name in SELF_REGISTERING_MODULES:
        declared = declared_for(name) is not None
        assert declared or name in does_real_work, (
            f"{name} is imported at launch and is not in the exception list; "
            f"either declare its row in spacr.qt.app_catalog or say here "
            f"what it does that a row cannot")
        assert not (declared and name in does_real_work), (
            f"{name} is both declared and listed as doing real work")


# ---------------------------------------------------------------------------
# the import-time costs that were pushed into the functions that need them
# ---------------------------------------------------------------------------

_IMPORT_PROBE = r"""
import json, sys
from PySide6.QtWidgets import QApplication
QApplication([])
import importlib
importlib.import_module(sys.argv[1])
print(json.dumps({"heavy": sorted(p for p in ("pandas", "scipy", "sklearn")
                                  if p in sys.modules)}))
"""


@pytest.mark.parametrize("module, forbidden", [
    ("spacr.feature_dict", ["pandas"]),
    ("spacr.qt.screens.classifier_evaluation", ["scipy", "sklearn"]),
    ("spacr.qt.widgets.control_chart", ["scipy"]),
])
def test_a_module_imported_for_its_stylesheet_stays_cheap(module, forbidden):
    """These three are imported at launch and did not need what they pulled.

    Each is on the launch path — two through ``theme.WIDGET_QSS_MODULES``,
    one behind the Feature Dictionary's registration — and each imported a
    scientific library at its top for something used in one function. The
    import moved into the function; this is what keeps it there.
    """
    done = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, module],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
             "CUDA_VISIBLE_DEVICES": ""})
    line = next((l for l in done.stdout.splitlines() if l.startswith("{")), "")
    assert line, f"{done.stdout}\n{done.stderr}"
    loaded = json.loads(line)["heavy"]
    clash = sorted(set(loaded) & set(forbidden))
    assert not clash, (
        f"importing {module} pulled in {clash}; it is on the launch path and "
        f"the import belongs in the function that uses it")


def test_the_normal_quartile_is_the_number_scipy_gave():
    """The stdlib replacement is not an approximation of the old constant.

    ``_NORMAL_QUARTILE`` seeds both robust sigma estimators, so a change in
    its last bits would move every control limit this module draws. It came
    from ``scipy.special.ndtri``; it comes from ``statistics.NormalDist`` now,
    and the two agree exactly — which is the only reason the swap was allowed.
    """
    from scipy.special import ndtri

    from spacr.qt.widgets import control_chart

    assert control_chart._NORMAL_QUARTILE == float(ndtri(0.75))


def test_c4_still_computes_the_published_constants():
    """The unbiasing constant, with its scipy import now inside the function."""
    from spacr.qt.widgets.control_chart import c4

    assert round(c4(2), 4) == 0.7979
    assert round(c4(5), 4) == 0.9400
    assert round(c4(10), 4) == 0.9727
    # Past the point where the gamma ratio overflows, the series takes over.
    assert c4(400) == pytest.approx(1.0 - 0.75 / 400)


def test_the_screens_package_imports_no_screens():
    """Importing one screen must not import nine.

    ``spacr/qt/screens/__init__.py`` used to import nine screens so their
    registry rows would exist. The rows exist without them now, and this
    package sits on the import path of every screen the window builds — so an
    import put back here is paid by everything.
    """
    data = _in_a_cold_process(r"""
import json, sys
from PySide6.QtWidgets import QApplication
QApplication([])
import spacr.qt.screens
print(json.dumps({
    "screens": sorted(n for n in sys.modules
                      if n.startswith("spacr.qt.screens.")),
    "heavy": sorted(p for p in ("pandas", "scipy", "sklearn")
                    if p in sys.modules),
}))
""")
    assert data["screens"] == [], (
        f"importing the screens package imported {data['screens']}")
    assert data["heavy"] == []
