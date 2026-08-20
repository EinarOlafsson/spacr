"""Assign explicit Qt menu roles for consistent cross-platform menus.

On macOS, Qt may relocate actions according to text such as "Preferences" or
"Quit". Explicit roles prevent translated or renamed actions from moving to
the wrong system-menu slot. Other platforms retain their normal menu layout.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


def set_menu_role(action: Any, role: str = "none") -> Any:
    """Assign a macOS menu role and return ``action``.

    Parameters
    ----------
    action : PySide6.QtGui.QAction
        Action whose role is assigned.
    role : {'none', 'preferences', 'quit', 'about'}, default='none'
        Target system-menu role. ``'none'`` keeps the action in its menu.

    Returns
    -------
    PySide6.QtGui.QAction
        The input action.

    Raises
    ------
    ValueError
        If ``role`` is not supported.
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
