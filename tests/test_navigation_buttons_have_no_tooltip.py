"""Navigation buttons carry no tooltip (the maintainer's rule, 2026-10-26).

Verbatim: "tooltips are not needed for button like next and previous or
any navigation buttons". A button whose label already says everything --
Next, Previous, Back, Continue, Finish, Done, Close, Cancel, OK, the
previous/next plate, page, image, field and 3D slice steppers, wizard step
navigation and pager arrows -- gets no ``setToolTip`` of its own, and the
application-wide tooltip filter does not lend it its parent's either.

Buttons that do work keep their tooltips, because those explain
consequences. That includes the few Cancel and Close buttons that also stop
a running job; they are named in :data:`WORK_STOPPING_DISMISSALS` so the
exception is explicit rather than a loophole.

The rule itself is :data:`spacr.qt.tooltip_policy._NAVIGATION_BUTTONS` and
:func:`spacr.qt.tooltip_policy._navigation_label`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spacr.qt.tooltip_policy import (_NAVIGATION_BUTTONS, _navigation_label)

QT_ROOT = Path(__file__).resolve().parents[1] / "spacr" / "qt"

WORK_STOPPING_DISMISSALS = frozenset({
    ("screens/annotate.py", "self._btn_suggest_cancel"),
    ("screens/map_barcodes.py", "self._button"),
    ("screens/measure_inputs.py", "self.close_button"),
    ("widgets/calibration_preview.py", "self.cancel"),
    ("widgets/cell_montage_view.py", "self._cancel"),
})
"""Cancel and Close buttons that stop running work and say so in a tooltip.

Each pair is the module, relative to ``spacr/qt``, and the receiver the
tooltip is set on. Every one of them stops or discards work already in
progress (a suggestion run, a barcode search, a measurement, a calibration
scan, a montage read), which a bare "Cancel" or "Close" does not tell.
"""


def _literal(node):
    """The string a label expression spells, through ``tr(...)``, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and node.args:
        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        if name in {"tr", "_tr", "translate"}:
            return _literal(node.args[0])
    return None


def _non_empty(node) -> bool:
    """Whether a ``setToolTip`` argument can be anything but empty."""
    text = _literal(node)
    return text is None or bool(text.strip())


def _navigation_receivers(scope, inherited):
    """Receivers in ``scope`` given a navigation label, by expression text."""
    found = dict(inherited)
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and node.value.args:
            callee = ast.unparse(node.value.func)
            label = _literal(node.value.args[0])
            if "button" in callee.casefold() and label is not None \
                    and _navigation_label(label) is not None:
                for target in node.targets:
                    found[ast.unparse(target)] = label
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple) \
                and isinstance(node.targets[0], ast.Tuple):
            for target, value in zip(node.targets[0].elts, node.value.elts):
                if isinstance(value, ast.Call) and value.args \
                        and "Button" in ast.unparse(value.func):
                    label = _literal(value.args[0])
                    if label is not None and _navigation_label(label) is not None:
                        found[ast.unparse(target)] = label
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "setText" and node.args:
            label = _literal(node.args[0])
            if label is not None and _navigation_label(label) is not None:
                found[ast.unparse(node.func.value)] = label
    return found


def _offences():
    """Every navigation button in spacr/qt that is handed a tooltip."""
    offences = []
    for path in sorted(QT_ROOT.rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        relative = path.relative_to(QT_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        members = {key: value for key, value in
                   _navigation_receivers(tree, {}).items()
                   if key.startswith("self.")}
        scopes = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for scope in scopes:
            receivers = _navigation_receivers(scope, members)
            for node in ast.walk(scope):
                if isinstance(node, ast.Call) and node.args \
                        and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "setToolTip":
                        receiver = ast.unparse(node.func.value)
                        if receiver in receivers and _non_empty(node.args[0]) \
                                and (relative, receiver) not in WORK_STOPPING_DISMISSALS:
                            offences.append(f"{relative}:{node.lineno} {receiver} "
                                            f"{receivers[receiver]!r}")
                    elif len(node.args) >= 2 and "button" in node.func.attr.casefold():
                        label, tip = _literal(node.args[0]), _literal(node.args[1])
                        receiver = ast.unparse(node.func)
                        if label is not None and _navigation_label(label) is not None \
                                and tip is not None and tip.strip() \
                                and (relative, receiver) not in WORK_STOPPING_DISMISSALS:
                            offences.append(f"{relative}:{node.lineno} {receiver} {label!r}")
                if isinstance(node, ast.Tuple) and len(node.elts) >= 3:
                    label, tip = _literal(node.elts[0]), node.elts[1]
                    if label is not None and _navigation_label(label) is not None \
                            and _literal(tip) is not None and _literal(tip).strip() \
                            and any(isinstance(item, (ast.Call, ast.Attribute, ast.Lambda))
                                    and _literal(item) is None for item in node.elts[2:]):
                        offences.append(f"{relative}:{node.lineno} {label!r} row")
    return sorted(set(offences))


def test_no_navigation_button_is_given_a_tooltip():
    assert _offences() == []


def test_the_exceptions_are_still_in_the_code():
    """A named exception that no longer matches anything must be removed."""
    for relative, receiver in WORK_STOPPING_DISMISSALS:
        text = (QT_ROOT / relative).read_text(encoding="utf-8")
        assert f"{receiver}.setToolTip(" in text or f"{receiver}(" in text, \
            (relative, receiver)


@pytest.mark.parametrize("label, expected", [
    ("Next", "next"), ("&Previous", "previous"), ("‹ Back", "back"),
    ("Next ›", "next"), ("◀", ""), ("▶", ""), ("Continue", "continue"),
    ("Finish", "finish"), ("Done", "done"), ("Close", "close"),
    ("Cancel", "cancel"), ("OK", "ok"), ("Previous plate", "previous plate"),
    ("Next page", "next page"), ("Next image", "next image"),
    ("Previous 3D slice", "previous 3d slice"),
    ("← Previous flagged field", "previous flagged field"),
    ("← All figures", "all figures"),
])
def test_navigation_labels_are_recognised(label, expected):
    assert _navigation_label(label) == expected


@pytest.mark.parametrize("label", [
    "Run", "Export…", "Measure", "Apply", "Add plates…", "Remove plate",
    "Cancel job", "Cancel loading or export", "▲", "▼", "↑", "↓", "",
])
def test_buttons_that_do_work_are_not_navigation(label):
    assert _navigation_label(label) is None


def test_the_rule_names_the_maintainers_list():
    for label in ("next", "previous", "back", "continue", "finish", "done",
                  "close", "cancel", "ok", "previous plate", "next plate",
                  "previous page", "next page", "previous image", "next image",
                  "previous 3d slice", "next 3d slice"):
        assert label in _NAVIGATION_BUTTONS


def test_a_navigation_button_does_not_borrow_its_parents_tooltip(qapp):
    from PySide6.QtWidgets import QPushButton, QWidget

    from spacr.qt.tooltip_policy import _is_navigation_button, tooltip_text_for

    host = QWidget()
    host.setToolTip("What this whole panel is for.")
    following = QPushButton("Next", host)
    work = QPushButton("Run", host)
    try:
        assert _is_navigation_button(following)
        assert not _is_navigation_button(work)
        assert tooltip_text_for(following) == ""
        assert tooltip_text_for(work) == "What this whole panel is for."
    finally:
        host.deleteLater()
