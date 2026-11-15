"""Reading a declared app row, and registering it or a copy of it.

``spacr.qt.app_catalog`` declares most apps as data: a row names its key, its
module and a factory *path*, so registration imports nothing. Two units are
held to exact values here:

* ``declared_app`` / ``DeclaredApp``: the lookup fails loudly on a typo, a row
  prints as its key and module, its ``register_kwargs`` leave empty optional
  fields out (so ``register_app`` applies its fallbacks), and the factory
  stand-in imports once and keeps what it imported;
* ``register_declared``: idempotent per key, a ``None`` for a module with no
  row, and ``key`` / ``section`` / ``stage`` overrides that reach the copy
  without touching the declared row.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_mod  # noqa: E402
from spacr.qt.app_catalog import (  # noqa: E402
    DECLARED_APPS,
    DeclaredApp,
    LazyScreenFactory,
    declared_app,
    register_declared,
)


@pytest.fixture(autouse=True)
def registered():
    """The registry a launched GUI has (conftest restores it afterwards)."""
    import spacr.qt

    spacr.qt.register_self_registering_modules()


@pytest.fixture
def probe_keys():
    """Keys to unregister after the test, so no copy outlives it."""
    keys: list[str] = []
    yield keys
    for key in keys:
        app_mod.unregister_app(key)


def test_a_declared_row_is_found_printed_and_turned_into_kwargs(subtests):
    """Lookup, repr, kwargs and the lazy factory, each to its exact value."""
    with subtests.test(msg="an unknown key is a KeyError"):
        with pytest.raises(KeyError) as raised:
            declared_app("no_such_app_key")
        assert raised.value.args == ("no_such_app_key",)

    with subtests.test(msg="a known key is its own row"):
        row = declared_app("trellis")
        assert (row.key, row.module, row.name, row.factory, row.section) == (
            "trellis", "spacr.qt.screens.trellis", "Small Multiples",
            "make_trellis_screen", "Tools")
        assert all(declared_app(each.key) is each for each in DECLARED_APPS)

    with subtests.test(msg="a row prints as its key and module"):
        assert repr(declared_app("trellis")) == (
            "DeclaredApp('trellis' from 'spacr.qt.screens.trellis')")
        assert [repr(each) for each in DECLARED_APPS] == [
            f"DeclaredApp({each.key!r} from {each.module!r})"
            for each in DECLARED_APPS]

    common = dict(key="probe", name="Probe", desc="A probe row",
                  section=app_mod.SECTION_DATA, module="spacr.qt.screens.probe")
    kwargs_cases = [
        ("no factory and no optional field", {}, {}),
        ("only the populated optional fields travel",
         dict(stage="alpha", api_module="qt/screens/probe", title="",
              intro="", cli_note="", entry="", defaults_module=""),
         {"stage": "alpha", "api_module": "qt/screens/probe"}),
        ("every optional field travels when set",
         dict(stage="beta", title="T", intro="I", cli_note="C",
              api_module="A", entry="m:f", defaults_module="d",
              translations=["x", "y"]),
         {"stage": "beta", "title": "T", "intro": "I", "cli_note": "C",
          "api_module": "A", "entry": "m:f", "defaults_module": "d",
          "translations": ("x", "y")}),
    ]
    for case_id, extra, expected in kwargs_cases:
        with subtests.test(msg=f"kwargs: {case_id}"):
            assert DeclaredApp(**common, **extra).register_kwargs() == expected

    with subtests.test(msg="a factory path becomes an unimported stand-in"):
        kwargs = DeclaredApp(**common, factory="make_probe_screen",
                             stage="alpha").register_kwargs()
        assert sorted(kwargs) == ["factory", "stage"]
        stand_in = kwargs["factory"]
        assert type(stand_in) is LazyScreenFactory
        assert (stand_in.module, stand_in.attribute, stand_in._resolved) == (
            "spacr.qt.screens.probe", "make_probe_screen", None)
        assert repr(stand_in) == ("<lazy screen factory spacr.qt.screens.probe:"
                                  "make_probe_screen (not imported)>")

    with subtests.test(msg="a registered row with no title is headed by its name"):
        row = declared_app("lineage")
        kwargs = row.register_kwargs()
        assert "title" not in kwargs and "entry" not in kwargs
        assert kwargs["api_module"] == "qt/screens/lineage"
        assert kwargs["translations"] == row.translations
        assert kwargs["cli_note"] == row.cli_note
        assert app_mod.APP_META["lineage"]["title"] == row.name

    with subtests.test(msg="the stand-in resolves once and keeps it"):
        row = declared_app("trellis")
        lazy = LazyScreenFactory(row.module, row.factory)
        first = lazy.resolve()
        assert first is importlib.import_module(row.module).make_trellis_screen
        assert lazy.resolve() is first
        assert lazy._resolved is first
        assert repr(lazy) == ("<lazy screen factory spacr.qt.screens.trellis:"
                              "make_trellis_screen (resolved)>")


def test_registering_a_row_or_a_copy_of_it(subtests, probe_keys):
    """A second registration is a no-op; overrides reach only the copy."""
    trellis = declared_app("trellis")
    lineage = declared_app("lineage")

    with subtests.test(msg="a module with no row registers nothing"):
        before = list(app_mod.APPS)
        assert register_declared("spacr.qt.maturity") is None
        assert app_mod.APPS == before

    with subtests.test(msg="the declared key is already registered at launch"):
        before = list(app_mod.APPS)
        assert register_declared(trellis.module) is None
        assert app_mod.APPS == before

    with subtests.test(msg="a copy under a new key is added once"):
        probe_keys.append("trellis_probe")
        assert register_declared(trellis.module, key="trellis_probe") == (
            "trellis_probe", "Small Multiples", trellis.desc, "Tools")
        before = list(app_mod.APPS)
        assert register_declared(trellis.module, key="trellis_probe") is None
        assert app_mod.APPS == before
        assert app_mod.APP_STAGE["trellis_probe"] == trellis.stage

    with subtests.test(msg="a copy can move section and stage"):
        assert trellis.section != app_mod.SECTION_DATA
        probe_keys.append("trellis_elsewhere")
        stage_before = app_mod.APP_STAGE["trellis"]
        assert register_declared(trellis.module, key="trellis_elsewhere",
                                 section=app_mod.SECTION_DATA,
                                 stage=app_mod.STAGE_BETA) == (
            "trellis_elsewhere", "Small Multiples", trellis.desc, "Data")
        assert app_mod.APP_STAGE["trellis_elsewhere"] == "beta"
        assert app_mod.APP_META["trellis_elsewhere"]["name"] == "Small Multiples"
        assert app_mod.APP_STAGE["trellis"] == stage_before
        assert [entry for entry in app_mod.APPS if entry[0] == "trellis"] == [
            ("trellis", "Small Multiples", trellis.desc, "Tools")]

    with subtests.test(msg="no stage override keeps the declared stage"):
        probe_keys.extend(["lineage_plain", "lineage_promoted"])
        assert lineage.stage == "alpha"
        assert register_declared(lineage.module, key="lineage_plain") == (
            "lineage_plain", "Lineage", lineage.desc, "Data")
        promoted = register_declared(lineage.module, key="lineage_promoted",
                                     stage=app_mod.STAGE_BETA)
        assert promoted == ("lineage_promoted", "Lineage", lineage.desc, "Data")
        assert app_mod.APP_STAGE["lineage_plain"] == "alpha"
        assert app_mod.APP_STAGE["lineage_promoted"] == "beta"
        assert lineage.stage == "alpha"
        factory = app_mod.APP_FACTORIES["lineage_plain"]
        assert type(factory) is LazyScreenFactory
        assert (factory.module, factory.attribute) == (
            "spacr.qt.screens.lineage", "make_lineage_screen")
