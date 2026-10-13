"""Deferred calls bounded by a widget work on every supported PySide6.

``QTimer.singleShot(msec, owner, callable)`` is rejected by PySide6 6.6, the
oldest version spaCR supports, so the minimum-dependencies job failed wherever
it was used. :func:`spacr.qt._after` gives the same behaviour from a timer
parented to the owner, and no module may go back to the three-argument form.
"""

import ast
from pathlib import Path

from PySide6.QtWidgets import QWidget

import spacr
from spacr.qt import _after


def test_the_callback_runs_once_after_the_delay(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    calls = []
    _after(0, owner, lambda: calls.append(1))
    qtbot.waitUntil(lambda: calls == [1])
    qtbot.wait(30)
    assert calls == [1]


def test_a_deleted_owner_never_receives_the_call(qtbot):
    owner = QWidget()
    calls = []
    _after(20, owner, lambda: calls.append(1))
    owner.deleteLater()
    qtbot.wait(80)
    assert calls == []


def test_the_timer_is_a_child_of_the_owner(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    timer = _after(1000, owner, lambda: None)
    assert timer.parent() is owner
    assert timer.isSingleShot() and timer.isActive()


def _three_argument_single_shots():
    root = Path(spacr.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "singleShot"):
                continue
            if len(node.args) == 3 and not (
                isinstance(node.args[2], ast.Constant)
                and isinstance(node.args[2].value, (str, bytes))
            ):
                found.append(f"{path.relative_to(root)}:{node.lineno}")
    return found


def test_no_module_passes_an_owner_and_a_callable_to_single_shot():
    assert _three_argument_single_shots() == []
