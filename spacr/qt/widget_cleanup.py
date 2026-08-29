"""Ownership-scoped cleanup for Qt objects libraries leave parentless.

Qt normally destroys a widget tree with its parent. pyqtgraph context menus
are an exception: ``PlotItem`` and ``ViewBox`` intentionally create them as
top-level windows and keep them through Python references. Closing a screen
therefore used to leave hundreds of live menu widgets behind, all of which a
later palette change had to visit.

The helper below walks only graphics scenes owned by the widget being closed.
It never sweeps ``QApplication`` and never invokes Python's cycle collector,
so it cannot collect an unrelated live QThread wrapper.
"""
from __future__ import annotations

import sys
from typing import Any


def retire_pyqtgraph_menus(owner: Any) -> int:
    """Queue deletion of parentless pyqtgraph menus owned by ``owner``.

    :returns: number of distinct menu roots retired. ``0`` also covers an
        environment without pyqtgraph or an owner whose C++ object is gone.
    """
    try:
        from PySide6.QtWidgets import QGraphicsView, QWidget
    except ImportError:
        return 0

    try:
        graphics_views = owner.findChildren(QGraphicsView)
    except (AttributeError, RuntimeError):
        return 0
    if not graphics_views:
        return 0

    # Cleanup must not be the event that loads an optional plotting stack.
    # A real pyqtgraph view can only already exist after its package has been
    # imported; unrelated QGraphicsViews need no work from this helper.
    pyqtgraph = sys.modules.get("pyqtgraph")
    if pyqtgraph is None:
        return 0
    PlotItem = getattr(pyqtgraph, "PlotItem", ())
    ViewBox = getattr(pyqtgraph, "ViewBox", ())

    menus = set()

    def retire(menu) -> None:
        if menu is None or id(menu) in menus:
            return
        menus.add(id(menu))
        try:
            # ViewBoxMenu embeds spin boxes and combo-box popup views that Qt
            # promotes to top-level windows. Delete the whole QObject-owned
            # widget tree before deleting the menu root, not just submenus,
            # or those controls are reparented to ``None`` and survive it.
            for child in reversed(menu.findChildren(QWidget)):
                child.close()
                child.deleteLater()
            # pyqtgraph's generated ``Ui_Form`` control holders are ordinary
            # Python objects, not QObject children of ViewBoxMenu. Closing
            # the menu reparents a few of their editors/popups to ``None``;
            # retire those widgets while the menu still owns the holders.
            for controls in getattr(menu, "ctrl", ()):
                for value in vars(controls).values():
                    if not isinstance(value, QWidget):
                        continue
                    for child in reversed(value.findChildren(QWidget)):
                        child.close()
                        child.deleteLater()
                    value.close()
                    value.deleteLater()
            menu.close()
            menu.deleteLater()
        except RuntimeError:
            pass

    viewboxes = set()
    for graphics_view in graphics_views:
        try:
            scene = graphics_view.scene()
            items = list(scene.items()) if scene is not None else []
        except RuntimeError:
            continue
        for item in items:
            if isinstance(item, PlotItem):
                retire(getattr(item, "ctrlMenu", None))
                try:
                    item.ctrlMenu = None
                except (AttributeError, RuntimeError):
                    pass
        for item in items:
            if not isinstance(item, ViewBox) or id(item) in viewboxes:
                continue
            viewboxes.add(id(item))
            menu = getattr(item, "menu", None)
            if menu is None:
                continue
            try:
                item.close()
            except (KeyError, RuntimeError):
                pass
            retire(menu)
            try:
                item.menu = None
            except (AttributeError, RuntimeError):
                pass
    return len(menus)
