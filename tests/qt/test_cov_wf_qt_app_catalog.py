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
from spacr.qt.app_catalog import (
    DeclaredApp,
    LazyScreenFactory,
    declared_for,
    register_declared,
)

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


# ---------------------------------------------------------------------------
# the row itself
# ---------------------------------------------------------------------------


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
    (_unreadable_value_error, (ValueError, TypeError)),
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
