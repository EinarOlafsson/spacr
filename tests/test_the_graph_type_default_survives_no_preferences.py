"""Choosing a default graph type, and the token cleanup that must not raise.

Two small guards from different modules, both of the same kind: a fallback that
runs when the environment cannot answer, and a cleanup that must survive being
asked to remove something already gone.
"""
from __future__ import annotations

import builtins

import pytest


# ---------------------------------------------------------------------------
# graph_types.default_for — no Qt, no preferences
# ---------------------------------------------------------------------------

def test_a_shape_falls_back_when_preferences_cannot_be_read(monkeypatch):
    """Arc 199 -> 202: the comment's own case -- "a figure still has to be drawn".

    This runs headless in every batch job and every test, so the fallback is
    not the exceptional path, it is the usual one. Raising here would mean no
    figure at all rather than a figure in the default style.
    """
    from spacr import graph_types

    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        if "preferences" in (fromlist or ()) or name.endswith("preferences"):
            raise ImportError("no Qt preferences in this process")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)

    for shape in graph_types.DEFAULTS:
        assert graph_types.default_for(shape) == graph_types.DEFAULTS[shape]


def test_a_stored_preference_that_fits_the_shape_is_used(monkeypatch):
    """The taken side, so the fallback above is visibly conditional."""
    from spacr import graph_types
    from spacr.qt import preferences

    shape = next(iter(graph_types.DEFAULTS))
    # GRAPH_TYPES is (name, description) pairs; FITS is the shape's own list.
    choices = [name for name, _description in graph_types.GRAPH_TYPES
               if graph_types.fits(shape, name)]
    assert choices, "the shape must accept at least one graph type"

    monkeypatch.setattr(preferences, "get_default_graph_type",
                        lambda _shape: choices[-1])

    assert graph_types.default_for(shape) == choices[-1]


def test_a_stored_preference_that_does_not_fit_is_ignored(monkeypatch):
    """The ``fits`` check, which is what stops a saved preference breaking a panel.

    A graph type saved for one shape and read for another would draw the wrong
    thing, so the shape's own default wins over a stored value it cannot use.
    """
    from spacr import graph_types
    from spacr.qt import preferences

    shape = next(iter(graph_types.DEFAULTS))
    monkeypatch.setattr(preferences, "get_default_graph_type",
                        lambda _shape: "not_a_graph_type")

    assert graph_types.default_for(shape) == graph_types.DEFAULTS[shape]


def test_learning_the_setting_keys_asks_no_preference(monkeypatch):
    """Validation's key sweep must not open the Qt preference store.

    `validate._known_setting_keys` calls every settings default with `{}`
    to learn which KEYS exist. Since 293 three of those defaults ask
    `chosen_for` for a graph type, and asking imports PySide6, so validating
    a batch queue imported Qt (tests/test_batch.py, run 34961482728). That
    test checks in a subprocess, which coverage does not see, so the
    mechanism is pinned here in-process. The second half matters as much:
    the switch is scoped to the sweep, and an ordinary caller afterwards
    still gets the user's choice.
    """
    from spacr import graph_types, validate
    from spacr.qt import preferences

    asked = []

    def preference(shape):
        asked.append(shape)
        return "violin"

    monkeypatch.setattr(preferences, "get_default_graph_type", preference)
    monkeypatch.setattr(validate, "_KNOWN_KEYS_CACHE", None)

    assert "graph_type" in validate._known_setting_keys()
    assert asked == [], (
        f"the key sweep asked the preference store for {asked}; it only "
        f"needs key names, and asking imports Qt")

    assert graph_types.chosen_for("categorical_continuous") == "violin"
    assert asked == ["categorical_continuous"], (
        "after the sweep an ordinary caller no longer reads the preference")


def test_a_key_sweep_that_raises_part_way_gives_the_preference_back(monkeypatch):
    """The switch is restored when the sweep dies, not only when it finishes.

    The sweep calls every settings default, so it can be interrupted part-way
    by something its per-function ``except Exception`` does not catch. A switch
    left off would stop every later figure in that process reading the user's
    graph-type choice (293) -- silently, in the one process that already
    raised, where no test would look. Hence ``reset`` with the ``set`` token in
    a ``finally``, which this pins.
    """
    from spacr import graph_types, settings, validate
    from spacr.qt import preferences

    class _Interrupted(BaseException):
        """Not an Exception, so the sweep's per-function guard lets it out."""

    def set_default_that_is_interrupted(_settings=None):
        raise _Interrupted()

    asked = []
    monkeypatch.setattr(preferences, "get_default_graph_type",
                        lambda shape: asked.append(shape) or "violin")
    monkeypatch.setattr(validate, "_KNOWN_KEYS_CACHE", None)
    monkeypatch.setattr(settings, "set_default_that_is_interrupted",
                        set_default_that_is_interrupted, raising=False)

    with pytest.raises(_Interrupted):
        validate._known_setting_keys()

    assert graph_types._READ_THE_PREFERENCE_STORE.get() is True, (
        "the sweep raised and left the preference store switched off")
    assert graph_types.chosen_for("categorical_continuous") == "violin"
    assert asked == ["categorical_continuous"]


def test_an_unknown_shape_is_a_key_error():
    """The comment on line 194 says so explicitly, so it is pinned."""
    from spacr.graph_types import default_for

    with pytest.raises(KeyError):
        default_for("not_a_shape")


# ---------------------------------------------------------------------------
# cancellation.installed_token — removing a token that is already gone
# ---------------------------------------------------------------------------

def test_a_token_removed_from_under_the_context_does_not_raise():
    """Lines 115-116: ``delattr`` on an attribute someone else already cleared.

    The context restores thread-local state on exit. Another cleanup path --
    a worker shutting down, a nested context unwinding out of order -- can
    have removed it first, and raising in a ``finally`` would replace whatever
    error caused the unwind with an AttributeError about bookkeeping.
    """
    from spacr import cancellation

    token = cancellation.CancellationToken()

    with cancellation.installed_token(token):
        assert cancellation.current_token() is token
        # Simulate a competing cleanup that got there first.
        try:
            delattr(cancellation._LOCAL, "token")
        except AttributeError:
            pass

    assert cancellation.current_token() is None


def test_a_nested_token_restores_the_outer_one():
    """The ``else`` branch beside it, which is the ordinary nesting case."""
    from spacr import cancellation

    outer = cancellation.CancellationToken()
    inner = cancellation.CancellationToken()

    with cancellation.installed_token(outer):
        with cancellation.installed_token(inner):
            assert cancellation.current_token() is inner
        assert cancellation.current_token() is outer

    assert cancellation.current_token() is None
