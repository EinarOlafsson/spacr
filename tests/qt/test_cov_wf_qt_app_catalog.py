"""The catalog's edge rows: no factory, no readable signature, no row at all.

``tests/qt/test_app_catalog.py`` holds the catalog to the registry a real
launch builds. What it does not exercise is what the catalog does when a row
is not the ordinary one -- a row that declares no factory (the app takes the
generic settings screen), a resolved factory whose signature cannot be read,
a module with no row, and the ``key``/``section``/``stage`` overrides a second
copy of a screen is registered through. Each of those is a live path in
``spacr/qt/app_catalog.py`` that ends in a tile a user clicks, so each is
checked here against the value it actually produces rather than by inspection.
"""
from __future__ import annotations

import inspect

import pytest

from spacr.qt import app as app_mod
from spacr.qt.app_catalog import (DECLARED_APPS, DeclaredApp,
                                  LazyScreenFactory, declared_app,
                                  declared_for, register_declared)

#: A module the catalog deliberately declares no row for: its registration
#: does real work (it re-reads the whole registry), so it is imported at
#: launch instead of being a row. Asserted, not assumed, in the test below.
UNDECLARED_MODULE = "spacr.qt.maturity"


@pytest.fixture(autouse=True)
def registered():
    """The registry a launched GUI has, not the one a bare import leaves.

    ``import spacr.qt.app`` fills in only the rows named in its own table; the
    declared ones arrive when the launch walks
    ``SELF_REGISTERING_MODULES``. Every assertion here about ``APP_META`` or
    ``APP_STAGE`` is about the registry the window reads, so it has to be
    that one; ``conftest``'s ``_restore_app_registry`` puts it back after.
    """
    import spacr.qt

    spacr.qt.register_self_registering_modules()


@pytest.fixture
def probe_keys():
    """Hand back a list to fill with keys to unregister afterwards.

    A registration that outlives its test is a stray tile, a stray sidebar row
    and a stray Ctrl+N binding for everything that runs after it, which is the
    leak ``conftest``'s ``_restore_app_registry`` exists for. These tests
    register copies on purpose, so they clean up after themselves too.
    """
    keys: list[str] = []
    yield keys
    for key in keys:
        app_mod.unregister_app(key)


# ---------------------------------------------------------------------------
# the row itself
# ---------------------------------------------------------------------------

def test_a_row_names_its_key_and_its_module_when_it_is_printed():
    """A row in a traceback has to say which of the nineteen it is.

    ``DeclaredApp`` uses ``__slots__`` and defines no dataclass repr, so
    without its own ``__repr__`` every row prints as an anonymous
    ``<...DeclaredApp object at 0x...>``. The two facts that identify a row
    are its key and the module the screen lives in -- exactly what somebody
    reading a failed registration or a debugger frame needs in order to find
    the declaration to fix.
    """
    row = declared_app("trellis")
    text = repr(row)
    assert text == "DeclaredApp('trellis' from 'spacr.qt.screens.trellis')"
    # Every row prints, and each prints its own identity, not the class name.
    printed = {repr(each) for each in DECLARED_APPS}
    assert len(printed) == len(DECLARED_APPS)
    for each in DECLARED_APPS:
        assert each.key in repr(each) and each.module in repr(each)


def test_a_row_that_declares_no_factory_asks_for_no_factory():
    """The generic settings screen is what an app without a factory gets.

    ``register_app`` decides between the declared screen and the generic
    settings screen on whether it was PASSED a factory; a catalog that always
    sent the keyword -- ``factory=None``, or a stand-in for the empty string
    -- would either hand the registry a proxy that resolves to nothing or
    lose the generic-screen fallback altogether. The contrast is the point,
    so both shapes of row are built here and their kwargs compared.
    """
    common = dict(key="probe_generic", name="Probe", desc="A probe row",
                  section=app_mod.SECTION_DATA, stage="alpha",
                  api_module="qt/screens/probe")
    without = DeclaredApp(module="spacr.qt.screens.probe", **common)
    with_one = DeclaredApp(module="spacr.qt.screens.probe",
                           factory="make_probe_screen", **common)

    bare = without.register_kwargs()
    assert "factory" not in bare, (
        "a row with no factory must not send the keyword at all, or "
        "register_app cannot fall back to the generic settings screen")
    # ... and the optional fields it DOES declare still travel.
    assert bare == {"stage": "alpha", "api_module": "qt/screens/probe"}

    full = with_one.register_kwargs()
    stand_in = full["factory"]
    assert isinstance(stand_in, LazyScreenFactory)
    assert (stand_in.module, stand_in.attribute) == (
        "spacr.qt.screens.probe", "make_probe_screen")
    assert stand_in._resolved is None, (
        "building the kwargs must not import the screen module")


def test_an_empty_optional_field_is_left_out_so_register_app_can_fill_it():
    """``title`` and ``intro`` fall back to the name and the description.

    ``register_app`` applies those fallbacks only for arguments it was not
    given, so a catalog that passed every field unconditionally would register
    an empty header and an empty paragraph for the fourteen rows that declare
    neither. The row used here declares an ``api_module`` and no ``title``,
    which is both halves of the rule in one object.
    """
    row = declared_app("lineage")
    kwargs = row.register_kwargs()
    assert kwargs["api_module"] == "qt/screens/lineage"
    assert kwargs["translations"] == row.translations
    assert "title" not in kwargs and "entry" not in kwargs, (
        "empty fields must be omitted; the row declares neither")
    assert row.title == "" and row.entry == ""
    # The fields it does declare are there, so the omission above is the rule
    # being applied and not the whole dictionary going missing.
    assert kwargs["cli_note"] == row.cli_note and row.cli_note
    # The registry that was actually built agrees: the header is the name.
    assert app_mod.APP_META[row.key]["title"] == row.name


# ---------------------------------------------------------------------------
# the stand-in, when the factory is not an ordinary function
# ---------------------------------------------------------------------------

def test_a_stand_in_forwards_everything_only_to_a_factory_that_takes_it():
    """``**kwargs`` gets both arguments; a narrow factory gets only its own.

    ``register_app`` promises a factory is given whichever of ``app_key`` and
    ``host`` it declares. The two sides are asserted together because the
    filtering branch is only correct relative to the forwarding one: a
    stand-in that filtered everything would starve the ``**kwargs`` screens
    of the host they draw into, and one that forwarded everything would raise
    ``TypeError`` on every open of a zero-argument screen.
    """
    lazy = LazyScreenFactory("never.imported", "never_looked_up")
    host = object()

    def takes_everything(**kwargs):
        return kwargs

    lazy._resolved = takes_everything
    got = lazy(app_key="trellis", host=host)
    assert got == {"app_key": "trellis", "host": host}

    def takes_only_the_key(app_key=None):
        return {"app_key": app_key}

    lazy._resolved = takes_only_the_key
    assert lazy(app_key="trellis", host=host) == {"app_key": "trellis"}
    assert lazy.module == "never.imported", (
        "neither call may have imported anything: _resolved was set by hand")


def _unreadable_type_error():
    """A callable whose ``__signature__`` makes ``inspect`` raise TypeError."""
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return "screen"

    factory.__signature__ = 42  # not a Signature: inspect refuses it
    return factory, calls


def _unreadable_value_error():
    """A callable ``inspect`` reports as having an invalid signature."""
    calls = []

    class Screen:
        __signature__ = "not a signature at all"

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return "screen"

    return Screen(), calls


@pytest.mark.parametrize("make, expected", [
    (_unreadable_type_error, TypeError),
    (_unreadable_value_error, ValueError),
])
def test_a_factory_whose_signature_cannot_be_read_is_still_called(make,
                                                                  expected):
    """An uninspectable screen factory opens empty-handed instead of crashing.

    ``inspect.signature`` is not total: it raises on a C-level callable with
    no signature information and on an object whose ``__signature__`` is not a
    ``Signature`` -- which is what a factory wrapped by a decorator, a
    ``functools.partial`` over a builtin, or a Qt-bound callable can look
    like. The catalog treats "cannot tell" as "declares nothing" and calls the
    factory with no arguments, so opening the app is a screen built from
    defaults rather than a traceback out of the launcher.
    """
    factory, calls = make()
    with pytest.raises(expected):
        inspect.signature(factory)

    lazy = LazyScreenFactory("never.imported", "never_looked_up")
    lazy._resolved = factory
    assert lazy(app_key="trellis", host=object()) == "screen"
    assert calls == [{}], (
        "an unreadable signature declares nothing, so nothing is forwarded")

    # The same stand-in with a readable signature does forward, which is what
    # makes the empty call above a decision and not a dropped argument.
    def readable(app_key=None):
        return app_key

    lazy._resolved = readable
    assert lazy(app_key="trellis", host=object()) == "trellis"


def test_a_stand_in_resolves_once_and_keeps_what_it_imported():
    """A screen module is imported on the first open, never on the second.

    The stand-in exists so registration costs no imports; caching is what
    keeps the saving after the tile is clicked. A ``resolve`` that re-imported
    would also hand out a different function object each time, and
    ``registered_factory``'s "swap the proxy for the real callable" step would
    have nothing stable to install.
    """
    row = declared_app("trellis")
    lazy = LazyScreenFactory(row.module, row.factory)
    assert lazy._resolved is None
    first = lazy.resolve()
    module = __import__(row.module, fromlist=["*"])
    assert first is getattr(module, row.factory)
    assert lazy.resolve() is first
    assert lazy._resolved is first


# ---------------------------------------------------------------------------
# registering from a row
# ---------------------------------------------------------------------------

def test_a_module_with_no_row_registers_nothing_and_says_so():
    """``register_declared`` is safe to call for a module it knows nothing of.

    ``spacr.qt.register_self_registering_modules`` walks a list that mixes
    declared rows with modules that do real work at registration. Asking the
    catalog about one of the second kind must be a quiet ``None``, not a
    ``KeyError`` that takes the launch down before the window is built.
    """
    before = len(app_mod.APPS)
    assert declared_for(UNDECLARED_MODULE) is None, (
        f"{UNDECLARED_MODULE} is expected to do real work at registration")
    assert register_declared(UNDECLARED_MODULE) is None
    assert len(app_mod.APPS) == before

    # A module that DOES have a row answers with the row it appended, which
    # is what makes the None above a decision rather than a dead function.
    row = declared_for("spacr.qt.screens.trellis")
    assert row is not None and row.key == "trellis"


def test_registering_a_row_that_is_already_registered_changes_nothing(
        probe_keys):
    """Startup and a direct ``register()`` both reach the same row.

    Two paths register the same key -- the launch walk and the module's own
    ``register()`` -- so the second one has to be a no-op. If it appended
    instead, the app would have two tiles, two sidebar rows and two entries
    competing for one shortcut.
    """
    row = declared_app("trellis")
    probe_keys.append("trellis_probe")
    added = register_declared(row.module, key="trellis_probe")
    assert added == ("trellis_probe", row.name, row.desc, row.section)

    before = list(app_mod.APPS)
    assert register_declared(row.module, key="trellis_probe") is None
    assert app_mod.APPS == before, (
        "a second registration of the same key must leave the registry alone")


def test_a_copy_can_be_placed_in_another_section_at_another_stage(probe_keys):
    """The overrides a second copy of a screen is registered through.

    ``register_layer_viewer_app`` puts a second copy of one screen in a
    different section, and the maturity of a copy is not the maturity of the
    original. Both overrides have to reach the registry -- and neither may
    touch the declared row, or promoting a copy would silently promote the
    app the user actually launches.
    """
    row = declared_app("trellis")
    assert row.section != app_mod.SECTION_DATA, (
        "the override has to move the copy somewhere it was not already")
    probe_keys.append("trellis_elsewhere")
    stage_before = app_mod.APP_STAGE["trellis"]
    added = register_declared(row.module, key="trellis_elsewhere",
                              section=app_mod.SECTION_DATA,
                              stage=app_mod.STAGE_BETA)
    assert added == ("trellis_elsewhere", row.name, row.desc,
                     app_mod.SECTION_DATA)
    assert app_mod.APP_STAGE["trellis_elsewhere"] == app_mod.STAGE_BETA
    assert app_mod.APP_META["trellis_elsewhere"]["name"] == row.name
    # The declared row is untouched by the copy.
    assert app_mod.APP_STAGE["trellis"] == stage_before
    assert [entry for entry in app_mod.APPS if entry[0] == "trellis"] == [
        (row.key, row.name, row.desc, row.section)]


def test_a_copy_with_no_stage_override_keeps_the_declared_maturity(probe_keys):
    """No override means the row's own stage, not "unknown" and not beta.

    The stage decides whether the app is shown at all under the alpha/beta
    filter, so a copy registered with no explicit stage must inherit exactly
    what the row declares. Registered beside an explicit override so the
    difference between the two is asserted in one place.
    """
    row = declared_app("lineage")
    probe_keys.extend(["lineage_plain", "lineage_promoted"])
    plain = register_declared(row.module, key="lineage_plain")
    promoted = register_declared(row.module, key="lineage_promoted",
                                 stage=app_mod.STAGE_BETA)
    assert plain == ("lineage_plain", row.name, row.desc, row.section)
    assert promoted[0] == "lineage_promoted"
    assert app_mod.APP_STAGE["lineage_plain"] == row.stage
    assert app_mod.APP_STAGE["lineage_promoted"] == app_mod.STAGE_BETA
    assert row.stage != app_mod.STAGE_BETA, (
        "the two keys must differ, or the inherited stage proves nothing")
    # Neither copy imported the screen: both got the stand-in.
    assert isinstance(app_mod.APP_FACTORIES["lineage_plain"],
                      LazyScreenFactory)


def test_a_key_the_catalog_does_not_know_is_a_keyerror():
    """``declared_app`` is a lookup, and a typo in a key has to fail loudly.

    Screens call it to read back their own name and description; a miss that
    returned ``None`` would surface as a screen with an empty header much
    later, in the window, instead of at the call that misspelled the key.
    """
    with pytest.raises(KeyError):
        declared_app("no_such_app_key")
    assert declared_app("trellis").name == "Small Multiples"
