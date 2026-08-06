"""Menu-bar roles, which are a macOS problem that looks like a spaCR bug.

macOS does not draw an application's menu bar inside its window. Qt moves it
to the system bar, and — this is the part that surprises people — it also
moves *individual actions between menus*, based on each action's
``QAction::menuRole``.

Qt assigns that role automatically by **pattern-matching the action's text**.
An action whose text contains ``about``, ``config``, ``options``, ``setup``,
``settings``, ``preferences``, ``quit`` or ``exit`` is claimed, removed from
the menu it was added to, and relocated to the application menu — the one
named after the running executable, which for spaCR is ``python`` because it
runs under the interpreter rather than as a bundled ``.app``.

That single behaviour produced both halves of a bug report that read like
two:

* **"Preferences and Quit don't show up."** They were moved out of the spaCR
  menu into the ``python`` menu. Nothing was hidden; they were relocated.
* **"The python menu's Preferences opens the module recipes window."**
  ``recipes.MENU_ACTION_TEXT`` is ``"Settings recipes…"``. It contains
  *settings*, so Qt gave it ``PreferencesRole`` too — and with two actions
  claiming one slot, the wrong one won.

**The rule: set the role on every menu action, explicitly, always.** Never
leave it to the text. An action that should not be relocated needs
``NoRole`` — not "no setMenuRole call", which is what lets Qt guess. Renaming
an action is otherwise enough to move it to a different menu on one platform,
which is not a connection anybody makes while renaming a menu item.

None of this is observable on Linux or Windows: Qt applies menu roles only on
macOS. So the tests assert that the roles are *set correctly*, which is
platform-independent and is the thing that was actually wrong. Where the item
lands still has to be confirmed on a Mac.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


def set_menu_role(action: Any, role: str = "none") -> Any:
    """Pin ``action``'s macOS menu role. Returns ``action``.

    :param action: the ``QAction``.
    :param role: ``'none'`` (default — stay where you were put),
        ``'preferences'``, ``'quit'`` or ``'about'``.
    :returns: the same action, for use inline.
    :raises ValueError: an unknown role name, because a typo here would
        silently restore the guessing this function exists to stop.
    """
    try:
        from PySide6.QtGui import QAction
    except Exception:                       # pragma: no cover - headless
        return action

    roles = {
        "none": QAction.MenuRole.NoRole,
        "preferences": QAction.MenuRole.PreferencesRole,
        "quit": QAction.MenuRole.QuitRole,
        "about": QAction.MenuRole.AboutRole,
    }
    key = str(role).strip().lower()
    if key not in roles:
        raise ValueError(
            f"unknown menu role {role!r}; expected one of {sorted(roles)}")
    try:
        action.setMenuRole(roles[key])
    except Exception:                       # pragma: no cover
        pass
    return action


def pin_menu_roles(actions: Iterable[Any],
                   preferences: Optional[Any] = None,
                   quit_action: Optional[Any] = None,
                   about: Optional[Any] = None) -> None:
    """Give every action in ``actions`` an explicit role.

    Everything gets ``NoRole`` except the three named, so adding a menu item
    later cannot accidentally claim the Preferences slot by being called
    something reasonable.

    :param actions: every action on the menu bar.
    :param preferences: the action that really is Preferences, if any.
    :param quit_action: the action that really is Quit, if any.
    :param about: the action that really is About, if any.
    """
    special = {id(a): name for a, name in (
        (preferences, "preferences"), (quit_action, "quit"), (about, "about"))
        if a is not None}
    for action in actions:
        if action is None:
            continue
        set_menu_role(action, special.get(id(action), "none"))
