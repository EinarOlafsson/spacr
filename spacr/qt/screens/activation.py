"""Activation-map workflow embedded in the Classify screen.

The ``activation`` workflow remains distinct from classifier training: it
uses its own settings form, Run action, console, and hyperparameter-search
key, and :func:`spacr.qt.bridge.resolve_pipeline_entry` dispatches it to
``generate_activation_map``. The Classify masthead opens that workflow as a
folded page so attribution results remain associated with the trained model
without combining the two pipeline runs.

:class:`ExplainNavigator` routes requests from the Explain CV Model page to
the same folded activation page. The aliases in :data:`NAV_KEYS` cover the
pipeline key, command alias, and navigation spelling without constructing a
second screen or duplicating its settings and job state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QWidget

LOG = logging.getLogger(__name__)

#: Registry key of the module folded onto Classify.
APP_KEY = "activation"

#: Every name a request for the activation maps may arrive under.
#:
#: ``activation_maps`` is what :mod:`spacr.qt.screens.model_explanation`
#: asks for, ``activation_map`` is the spelling :data:`spacr.cli.ALIASES`
#: maps onto the key, and ``activation`` is the key itself -- what a saved
#: run journal or ``spacr-qt activation`` names. One set rather than three
#: call sites deciding separately which spelling counts.
NAV_KEYS = frozenset({"activation", "activation_map", "activation_maps"})


def build(host_window: Optional[QWidget] = None) -> QWidget:
    """The module's own screen, wired the way its sidebar row wired it.

    :param host_window: the main window, when there is one to connect to.
    :returns: the settings-driven screen for :data:`APP_KEY`, with its
        host connections -- "Explain error" and "Run on a cluster" -- made.
    """
    from .map_barcodes import build_settings_screen

    return build_settings_screen(APP_KEY, host_window)


def opener_on(screen: Optional[QWidget]) -> Optional[Any]:
    """Return the activation-page opener registered on ``screen``, if present.

    The opener is owned by the host screen. Reusing it ensures that masthead
    actions and navigation requests address the same folded page and preserve
    its console, job runner, and settings state.

    :param screen: Candidate host screen.
    :returns: Matching fold opener, or ``None``.
    """
    for opener in getattr(screen, "_fold_openers", ()) or ():
        if getattr(opener, "key", "") == APP_KEY:
            return opener
    return None


def host_of(widget: Optional[QWidget]) -> Optional[QWidget]:
    """The fold host ``widget`` is sitting on, or None.

    Walks up from a folded page to the screen whose strip can open the
    activation maps. Derived from the widget tree rather than handed in,
    because the page is built before it is mounted -- and because a page
    that ended up in a window instead (the fold's last resort) has no such
    host above it, which is exactly what None says.
    """
    node = widget
    while node is not None:
        if opener_on(node) is not None:
            return node
        node = node.parent()
    return None


def open_page(host_screen: Optional[QWidget]) -> Optional[QWidget]:
    """Show the activation page on ``host_screen`` and raise it.

    :returns: the module's screen, or None when this host carries no
        activation fold.
    """
    opener = opener_on(host_screen)
    return opener.open() if opener is not None else None


def apply_seed(screen: Optional[QWidget], values: Dict[str, Any]) -> None:
    """Push ``values`` into ``screen``'s settings form.

    The rule ``MainWindow._on_train_requested`` applies, asked of the same
    function rather than written out a second time: a navigation that
    seeded differently from the sidebar's would be a second answer to one
    question. A key with no widget is skipped, as it is there.
    """
    model = getattr(screen, "_settings_model", None)
    if model is None or not values:
        return
    widgets = getattr(model, "_widgets", {})
    try:
        from ..app import MainWindow

        apply_value = MainWindow._apply_seed_value
    except Exception:                                        # noqa: BLE001
        LOG.debug("Could not read the seeding rule", exc_info=True)
        return
    for key, value in values.items():
        widget = widgets.get(key)
        if widget is None:
            continue
        try:
            apply_value(widget, value)
        except Exception:                                    # noqa: BLE001
            LOG.debug("Could not seed %s with %r", key, value, exc_info=True)


class ExplainNavigator:
    """The host handed to Classify's Explain CV page.

    Explain CV Model sends the user on in one place -- "Open Activation
    Maps" -- and does it by calling ``host._on_train_requested``. Standing
    in for the window there is what lets that button land on the page
    beside it instead of on a key nothing knows, and it costs the
    explanation screen nothing: anything that is not a request for the
    activation maps is forwarded to the real window unchanged.

    A plain object rather than a ``QObject``: it is reached by attribute
    access only, and the page holds it, so it lives exactly as long as the
    page that uses it.

    :param window: the main window, when it can be navigated through --
        see :func:`spacr.qt.screens.classify._navigable`.
    """

    def __init__(self, window: Optional[QWidget] = None) -> None:
        self.window = window
        #: The page this navigator was built for, once it exists.
        self.page: Optional[QWidget] = None

    def attach(self, page: QWidget) -> None:
        """Remember the page, so the host it lands on can be found later."""
        self.page = page

    def _on_train_requested(self, target_key: str,
                            seed: Optional[Dict[str, Any]] = None
                            ) -> Optional[QWidget]:
        """Answer a navigation, or hand it on to the window.

        :param target_key: the module the page asked for.
        :param seed: settings to push into it, as the window would.
        :returns: the screen that was opened, or None when nothing could
            answer -- which is what the page did before it had a host.
        """
        values = dict(seed or {})
        key = str(target_key)
        if key in NAV_KEYS:
            opened = open_page(host_of(self.page))
            if opened is not None:
                apply_seed(opened, values)
                return opened
            # No fold on this host -- the page is in a window of its own.
            # Ask the window for the module by the name the registry knows
            # it under rather than by the one that reaches nothing.
            key = APP_KEY
        window = self.window
        if window is None or not callable(
                getattr(window, "_on_train_requested", None)):
            return None
        return window._on_train_requested(key, values)
