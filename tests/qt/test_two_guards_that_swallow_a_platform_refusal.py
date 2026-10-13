"""Two guards that absorb a refusal from below rather than passing it up.

Instruction 288.

``set_menu_role`` sets a Qt menu role so macOS moves Preferences and Quit
into the application menu. A binding that will not accept the role must
not stop the action being built -- the menu still works, it is just in
the ordinary place.

``_on_release`` removes the rubber-band patch it drew while dragging.
Matplotlib raises when an artist has already been removed or belongs to
a container that does not support removal, and a failed cleanup of a
temporary decoration must not lose the gate the user just drew.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.menus import set_menu_role


# ---------------------------------------------------------------------------
# set_menu_role
# ---------------------------------------------------------------------------

class _RefusingAction:
    """Stands in for a binding whose setMenuRole will not take."""

    def __init__(self):
        self.attempts = 0

    def setMenuRole(self, _role):        # noqa: N802 - Qt naming
        self.attempts += 1
        raise RuntimeError("this binding does not support menu roles")


class _AcceptingAction:
    def __init__(self):
        self.role = None

    def setMenuRole(self, role):         # noqa: N802 - Qt naming
        self.role = role


@pytest.mark.parametrize("role", ["none", "preferences", "quit", "about"])
def test_a_role_that_will_not_set_still_returns_the_action(role):
    """THE ARM. The action is the return value, so swallowing must not
    also swallow the action."""
    action = _RefusingAction()

    assert set_menu_role(action, role) is action
    assert action.attempts == 1, "the role was never attempted"


@pytest.mark.parametrize("role", ["none", "preferences", "quit", "about"])
def test_a_role_that_sets_is_actually_applied(role):
    """So the swallow above is not hiding a function that never sets
    anything."""
    action = _AcceptingAction()

    assert set_menu_role(action, role) is action
    assert action.role is not None


def test_an_unknown_role_is_refused_loudly():
    """The OTHER failure, and it must stay loud: an unknown role is a
    typo in spaCR's own code, not a platform saying no."""
    with pytest.raises(ValueError, match="unknown menu role"):
        set_menu_role(_AcceptingAction(), "not_a_role")


def test_the_refusal_names_what_it_would_have_accepted():
    """A refusal that does not say the alternatives makes the caller
    read the source."""
    with pytest.raises(ValueError) as caught:
        set_menu_role(_AcceptingAction(), "nonsense")
    message = str(caught.value)
    for role in ("about", "none", "preferences", "quit"):
        assert role in message


# ---------------------------------------------------------------------------
# the rubber-band patch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [ValueError, NotImplementedError])
def test_a_patch_that_will_not_be_removed_does_not_lose_the_gate(qtbot,
                                                                 error):
    """THE ARM, driven through the real ``_on_release``.

    Matplotlib raises ValueError when an artist is already gone and
    NotImplementedError for containers that do not support removal.
    Either way the drag is over and the reference has to be dropped, or
    the next drag draws over a stale patch that is never cleaned up.

    An earlier draft of this test asserted on the SOURCE instead of
    calling the method, and covered none of it -- the coverage JSON said
    so. Driving the method is the only thing that proves the arm runs.
    """
    from spacr.qt.widgets.gate_editor import GateCanvas

    canvas = GateCanvas()
    qtbot.addWidget(canvas)

    class _Stubborn:
        def __init__(self):
            self.asked = 0

        def remove(self):
            self.asked += 1
            raise error("this artist will not go")

    patch = _Stubborn()
    canvas._tool = "rect"                   # not "" and not POLYGON
    canvas._drag_patch = patch
    canvas._drag_origin = None

    class _Event:
        inaxes = None
        xdata = None
        ydata = None

    canvas._on_release(_Event())            # must not raise

    assert patch.asked == 1, "the patch was never asked to remove itself"
    assert canvas._drag_patch is None, (
        "the patch reference survived a failed removal; the next drag "
        "draws over a stale patch that is never cleaned up")


def test_a_patch_that_removes_cleanly_is_also_dropped(qtbot):
    """The ordinary path, so the arm above is not the only one that
    clears the reference."""
    from spacr.qt.widgets.gate_editor import GateCanvas

    canvas = GateCanvas()
    qtbot.addWidget(canvas)

    class _Willing:
        def __init__(self):
            self.asked = 0

        def remove(self):
            self.asked += 1

    patch = _Willing()
    canvas._tool = "rect"
    canvas._drag_patch = patch
    canvas._drag_origin = None

    class _Event:
        inaxes = None
        xdata = None
        ydata = None

    canvas._on_release(_Event())

    assert patch.asked == 1
    assert canvas._drag_patch is None
