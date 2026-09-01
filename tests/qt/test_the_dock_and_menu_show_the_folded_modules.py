"""Every folded module is reachable from the dock and from the menu bar.

Instruction 330. Item 318 folded 33 modules onto 11 mastheads, and neither the
dock nor the menu showed them at all -- a user who wanted Volcano Explorer had
to know it lives on Regression. The counts here come from ``folded_children()``
rather than a list in the test, so a module folded later is covered without
anyone editing this file.
"""
from __future__ import annotations

import pytest

from spacr.qt.app import folded_children


@pytest.fixture
def window(qtbot):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_every_folded_module_has_a_dock_row(window):
    """The dock carries a row per folded child, indented under its host."""
    from PySide6.QtWidgets import QAbstractButton

    rows = {}
    for btn in window._sidebar.findChildren(QAbstractButton):
        key = str(btn.property("navKey") or "")
        if key and btn.property("isFoldChild"):
            rows[key] = str(btn.property("foldParent") or "")

    expected = {c: host for host, kids in folded_children().items() for c in kids}
    assert expected, "folded_children() is empty; this test would prove nothing"
    missing = sorted(set(expected) - set(rows))
    assert not missing, f"folded modules with no dock row: {missing}"
    wrong = {k: (rows[k], expected[k]) for k in expected if rows[k] != expected[k]}
    assert not wrong, f"rows indented under the wrong host: {wrong}"


def test_every_folded_module_has_a_menu_entry(window):
    """The menu bar carries the same second level."""
    keys = {str(a.property("moduleAppKey") or "")
            for a in window.findChildren(type(window.menuBar().actions()[0]))}
    expected = {c for kids in folded_children().values() for c in kids}
    assert expected
    missing = sorted(expected - keys)
    assert not missing, f"folded modules with no menu action: {missing}"


def test_the_children_start_collapsed_and_open_with_their_host(window):
    """33 rows always on screen would make the dock taller than the display."""
    sidebar = window._sidebar
    host, kids = next(iter(folded_children().items()))
    assert not sidebar.host_is_expanded(host), "a host starts collapsed"

    sidebar.expand_host(host)
    assert sidebar.host_is_expanded(host)

    # Opening a module with no children puts the dock back to one level.
    sidebar.expand_host("__home__")
    assert not sidebar.host_is_expanded(host)


def test_only_one_host_is_expanded_at_a_time(window):
    """Otherwise the dock grows by every host the user has ever pressed."""
    sidebar = window._sidebar
    hosts = [h for h, kids in folded_children().items() if kids][:2]
    if len(hosts) < 2:
        pytest.skip("needs two hosts with folded children")
    sidebar.expand_host(hosts[0])
    sidebar.expand_host(hosts[1])
    assert not sidebar.host_is_expanded(hosts[0])
    assert sidebar.host_is_expanded(hosts[1])


def test_a_folded_row_routes_through_open_module(window, monkeypatch):
    """A folded key does not name a screen.

    Navigating to it directly builds an orphan page with no sidebar row and no
    way back -- the second front door the fold exists to remove. The row must
    go through ``open_module``, which resolves the key to its host.
    """
    seen = []
    monkeypatch.setattr(window, "open_module", lambda k: seen.append(k) or k)
    # NOT connected here on purpose. MainWindow already connects this signal;
    # connecting again would fire twice and the test would pass on its own
    # wiring rather than on the window's -- which is exactly what it caught
    # the first time it was written.

    host, kids = next(iter(folded_children().items()))
    window._sidebar.fold_child_selected.emit(kids[0])
    assert seen == [kids[0]]
