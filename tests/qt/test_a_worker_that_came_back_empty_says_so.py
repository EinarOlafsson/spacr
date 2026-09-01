"""Four callbacks that receive nothing, and say so rather than crashing.

Instruction 288. Each of these runs on the GUI thread when a worker
finishes. A worker that failed hands back ``None``, and the callback has
to turn that into a sentence the user can read -- the alternative is an
AttributeError inside a slot, which surfaces as an unhandled exception in
the Qt event loop and leaves the screen showing the last thing it had.

All four carried a bare ``# pragma: no cover`` with no reason at all,
which is the kind most worth checking: nothing recorded why anybody
believed it was unreachable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# laptop_mode.total_memory_gib
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [AttributeError, ValueError, OSError])
def test_memory_that_cannot_be_read_is_none_not_a_crash(error, monkeypatch):
    """THE ARM. ``os.sysconf`` is absent on Windows (AttributeError) and
    refuses unknown names elsewhere."""
    from spacr.qt import laptop_mode

    def _refuse(_name):
        raise error("no sysconf here")

    monkeypatch.setattr(laptop_mode.os, "sysconf", _refuse)

    assert laptop_mode.total_memory_gib() is None


def test_memory_is_actually_read_when_it_can_be():
    """So the None above is about the failure, not about a function that
    always declines."""
    from spacr.qt import laptop_mode

    total = laptop_mode.total_memory_gib()
    if total is None:
        pytest.skip("this platform has no sysconf memory figures")
    assert total > 0.1, f"implausible memory figure: {total} GiB"


# ---------------------------------------------------------------------------
# The three worker callbacks
# ---------------------------------------------------------------------------

def test_a_hit_list_that_could_not_be_built_is_reported(qtbot):
    """THE ARM. `_on_hits_ready(None)` is what a failed build delivers."""
    from spacr.qt.screens.hit_list import HitListScreen

    screen = HitListScreen()
    qtbot.addWidget(screen)

    said = []
    screen._set_summary = lambda text, *, problem: said.append((text, problem))

    screen._on_hits_ready(None)

    assert said, "the failure was never reported to the user"
    text, problem = said[0]
    assert "could not be built" in text
    assert problem is True, "a failure was reported as ordinary news"
    assert screen._all is None


def test_a_hit_list_that_was_built_is_emitted(qtbot):
    """The other side: the arm above must not be the only path."""
    from spacr.qt.screens.hit_list import HitListScreen

    screen = HitListScreen()
    qtbot.addWidget(screen)

    emitted = []
    screen.hits_loaded.connect(emitted.append)
    screen._apply_filters = lambda: None

    sentinel = object()
    screen._on_hits_ready(sentinel)

    assert emitted == [sentinel], "a built hit list was not handed on"


def test_a_digest_that_could_not_be_built_is_reported(qtbot):
    from spacr.qt.screens.methods_export import MethodsExportScreen

    screen = MethodsExportScreen()
    qtbot.addWidget(screen)

    said = []
    screen._set_provenance = lambda text, *, problem: said.append(
        (text, problem))

    screen._on_digest_ready(None)

    assert said and "could not be built" in said[0][0]
    assert said[0][1] is True


def test_an_empty_digest_is_treated_as_no_digest(qtbot):
    """`if not digest` -- an empty dict is as useless as None, and
    saying nothing about it would leave the last digest on screen."""
    from spacr.qt.screens.methods_export import MethodsExportScreen

    screen = MethodsExportScreen()
    qtbot.addWidget(screen)

    said = []
    screen._set_provenance = lambda text, *, problem: said.append(
        (text, problem))

    screen._on_digest_ready({})

    assert said and said[0][1] is True


def test_a_draft_that_could_not_be_produced_is_reported(qtbot):
    from spacr.qt.screens.methods_export import MethodsExportScreen

    screen = MethodsExportScreen()
    qtbot.addWidget(screen)

    said = []
    screen._set_provenance = lambda text, *, problem: said.append(
        (text, problem))

    screen._on_draft_ready(None)

    assert said and "could not be produced" in said[0][0]
    assert said[0][1] is True


# ---------------------------------------------------------------------------
# manuscript.availability with a provider list that will not build
# ---------------------------------------------------------------------------

def test_a_provider_list_that_raises_still_gives_advice(monkeypatch):
    """THE ARM.

    ``availability`` only reaches ``list_providers`` when NOTHING is
    configured -- that is the branch which explains how to set a provider
    up. If the registry cannot be read, the advice above it is still
    worth printing: a traceback here would replace a paragraph of help
    with nothing, and the run's own numbers do not depend on any of it.

    ``configured_providers`` is stubbed empty because this machine has
    Claude configured, so the function returns before the guard.
    """
    from spacr.qt.ai import manuscript

    monkeypatch.setattr(manuscript, "configured_providers", lambda: ())

    asked = []

    def _refuse():
        asked.append(True)
        raise RuntimeError("the provider registry is unreadable")

    monkeypatch.setattr(manuscript, "list_providers", _refuse)

    answer = manuscript.availability()

    assert asked == [True], "the provider list was never asked for"
    assert answer.ok is False
    assert "No AI provider is configured" in answer.message, (
        f"the advice was lost with the provider list: {answer.message[:200]}")


def test_a_provider_list_that_works_is_used(monkeypatch):
    """So the arm above is about the failure, not a function that never
    consults the registry."""
    from spacr.qt.ai import manuscript

    monkeypatch.setattr(manuscript, "configured_providers", lambda: ())

    class _Provider:
        label = "Test Provider"
        cli_name = "testcli"
        login_command = "testcli login"

        @staticmethod
        def is_installed():
            return True

    monkeypatch.setattr(manuscript, "list_providers", lambda: [_Provider()])

    message = manuscript.availability().message

    assert "Test Provider" in message and "testcli" in message, (
        f"an installed provider was not named: {message[:200]}")


def test_a_configured_provider_short_circuits_before_the_advice(monkeypatch):
    """The path this machine actually takes, pinned so the two stubs
    above are visibly necessary rather than decorative."""
    from spacr.qt.ai import manuscript

    class _Ready:
        name = "claude"
        label = "Claude"

    monkeypatch.setattr(manuscript, "configured_providers",
                        lambda: [_Ready()])
    monkeypatch.setattr(
        manuscript, "list_providers",
        lambda: pytest.fail("the advice branch was reached anyway"))

    answer = manuscript.availability()

    assert answer.ok is True
    assert answer.providers == ("claude",)
