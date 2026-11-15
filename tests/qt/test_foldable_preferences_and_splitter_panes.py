"""`make_foldable`: the fold's preference store, its listeners, and folded splitter panes.

A fold is remembered only when there is a key and only for a fold the user
made; an unreadable store opens the panel and an unwritable one still folds.
Listeners are added once (non-callables never) and a failing one does not
stop the fold.  A `CollapsibleSplitter` takes folded panes, opens a folded
focus pane when it is shown, and ignores a folder that outlived its pane.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

import spacr.qt.preferences as preferences                      # noqa: E402
from spacr.qt.widgets import collapsible_splitter as cs         # noqa: E402
from spacr.qt.widgets.foldable import make_foldable             # noqa: E402

pytestmark = pytest.mark.qt


def _explode(*_args, **_kwargs):
    raise RuntimeError("no preference store")


def test_make_foldable_remembers_folds_only_when_it_can_and_should(
        qtbot, monkeypatch, subtests):
    """Start state, preference writes, the on_change callback and listeners."""
    def panel():
        heading, body = QLabel("Console"), QWidget()
        qtbot.addWidget(heading)
        qtbot.addWidget(body)
        return heading, body

    for case, stored, key, start in (
            ("a stored fold is applied at build time", {"tests/console": True},
             "tests/console", True),
            ("an unreadable store opens the panel", _explode, "tests/console",
             False),
            ("a keyless panel never reads the store", _explode, "", False),
    ):
        with subtests.test(msg=case):
            with monkeypatch.context() as patch:
                patch.setattr(preferences, "get_folded_panels",
                              stored if callable(stored) else (lambda: stored))
                patch.setattr(preferences, "set_folded_panel",
                              lambda *a, **k: None)
                folder = make_foldable(*panel(), persist_key=key)
            assert folder.shut is start

    for key in ("", "regression.summary"):
        with subtests.test(msg=f"writes for key {key!r}: user folds only"):
            writes = []
            with monkeypatch.context() as patch:
                patch.setattr(preferences, "get_folded_panels", lambda: {})
                patch.setattr(preferences, "set_folded_panel",
                              lambda name, shut: writes.append((name, shut)))
                folder = make_foldable(*panel(), persist_key=key)
                folder.set_shut(True, by_user=False)
                assert writes == []
                folder.set_shut(False, by_user=True)
                folder.set_shut(True, by_user=True)
            assert writes == ([(key, False), (key, True)] if key else [])

    with subtests.test(msg="a keyless panel writes nothing even to a stand-in module"):
        wrote = []
        module = types.ModuleType("spacr.qt.preferences")
        module.set_folded_panel = lambda key, shut: wrote.append(key)
        module.get_folded_panels = lambda: {}
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "spacr.qt.preferences", module)
            make_foldable(*panel()).set_shut(True)
        assert wrote == []

    with subtests.test(msg="an unwritable store still folds and reports the change"):
        seen = []
        with monkeypatch.context() as patch:
            patch.setattr(preferences, "get_folded_panels", lambda: {})
            patch.setattr(preferences, "set_folded_panel", _explode)
            folder = make_foldable(*panel(), on_change=seen.append,
                                   persist_key="tests/console")
            assert folder.toggle() is True
        assert (folder.shut, seen) == (True, [True])

    with subtests.test(msg="a listener is added once, a non-callable never, a failing one is contained"):
        heading, body = panel()
        folder = make_foldable(heading, body)
        heard = []

        def listener(shut, by_user):
            heard.append((shut, by_user))

        def broken(_shut, _by_user):
            raise RuntimeError("listener fell over")

        folder.add_listener(broken)
        folder.add_listener(listener)
        folder.add_listener(listener)
        folder.add_listener("not callable")
        assert folder.toggle() is True
        assert heard == [(True, True)]
        assert folder.shut is True and body.isHidden()


def _pump(n=6):
    for _ in range(n):
        QApplication.processEvents()


def _headed(name="Pane"):
    pane = QWidget()
    heading, body = QLabel(name), QWidget()
    body.setMinimumHeight(120)
    column = QVBoxLayout(pane)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(heading)
    column.addWidget(body, 1)
    return pane, make_foldable(heading, body, name=name)


def test_a_collapsible_splitter_with_foldable_panes(qtbot, subtests):
    """Folding through the splitter: refusals, folded adds, focus panes, orphans, sizing."""
    def split():
        made = cs.CollapsibleSplitter(Qt.Vertical)
        qtbot.addWidget(made)
        made.resize(400, 600)
        return made

    with subtests.test(msg="what cannot be folded is refused"):
        splitter = split()
        splitter.add_pane(QWidget(), "Plain")
        assert splitter.set_collapsed("Plain", True) is False
        assert splitter.set_collapsed("nope", True) is False
        assert splitter.toggle_pane("nope") is False
        pane, folder = _headed()
        splitter.add_pane(pane, "Head", folder=folder)
        assert splitter.set_collapsed("Head", False) is True

    with subtests.test(msg="a folded pane can be added"):
        splitter = split()
        pane, folder = _headed()
        folder.toggle()
        splitter.add_pane(pane, "Head", folder=folder)
        assert splitter.is_collapsed("Head") is True

    with subtests.test(msg="a folder that outlived its pane is ignored"):
        splitter = split()
        pane, folder = _headed()
        splitter._panes.remove(splitter.add_pane(pane, "Head", folder=folder))
        before = splitter.sizes()
        assert folder.toggle() is True
        assert splitter.sizes() == before

    with subtests.test(msg="a folded focus pane opens when it is shown"):
        splitter = split()
        pane, folder = _headed()
        splitter.add_pane(pane, "Preview", folder=folder, focus=True)
        splitter.add_pane(QWidget(), "Rest")
        splitter.show()
        pane.hide()
        folder.toggle()
        pane.show()
        assert splitter.is_collapsed("Preview") is False

    with subtests.test(msg="the absorber takes what the fixed panes leave"):
        splitter = split()
        pane, folder = _headed()
        splitter.add_pane(pane, "Head", folder=folder)
        splitter.add_pane(QWidget(), "Fixed", stretch=0, extent=100)
        splitter.show()
        _pump()
        folder.toggle()
        _pump()
        sizes = splitter.sizes()
        assert sizes[0] > sizes[1]
        assert sum(sizes) <= 600

