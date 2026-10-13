"""Fold strip host declarations read from odd sources, and badge edges."""
from __future__ import annotations

import ast
import sys
import types

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import fold_strip as fs  # noqa: E402

SOURCE = '''
from .sibling import CONST as ALIAS
from . import sibling
KEY_A = "a"
annotated: str
x, y = 1, 2
obj.attr = "nope"
FOLD_ORDER = (KEY_A, sibling.MISSING)
FOLDED_APPS = [KEY_A, sibling.CONST, "c", other.CONST]
FOLD_FALLBACK = {KEY_A: ("A", 1), unknown: 2, a.b.c: 5, "bad": make(), sibling.CONST: 3}
'''

SIBLING = '''
import os
annotated: int = 3
NUMBER = 4
CONST = "b"
'''


@pytest.fixture
def sources(monkeypatch):
    import spacr.qt.app as app

    trees = {"spacr.qt.host": ast.parse(SOURCE).body,
             "spacr.qt.sibling": ast.parse(SIBLING).body}
    monkeypatch.setattr(app, "_top_level_nodes", lambda name: trees.get(name))
    monkeypatch.setattr(fs, "_HOST_DECLARATION_CACHE", {})
    return trees


def test_a_host_without_source_has_no_declarations(sources):
    assert fs._host_declarations("spacr.qt.missing") is None
    assert fs._HOST_DECLARATION_CACHE["spacr.qt.missing"] is None


def test_declarations_skip_what_cannot_be_read_as_strings(sources):
    members, table = fs._host_declarations("spacr.qt.host")
    assert members == ()
    assert table == {"a": ("A", 1), "b": 3}


def test_a_sibling_constant_that_is_not_a_string_is_none(sources):
    assert fs._sibling_constant("spacr.qt.sibling", "NUMBER") is None
    assert fs._sibling_constant("spacr.qt.sibling", "CONST") == "b"


def test_a_badge_without_regression_qc_falls_back_to_literals(qtbot,
                                                              monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.regression_qc", None)
    button = fs.FoldButton.__new__(fs.FoldButton)
    assert fs.FoldButton._verdict_ink(button, "fail") == "#F85149"
    assert fs.FoldButton._verdict_ink(button, "odd") == ""


def test_scaling_a_strip_without_a_layout_does_nothing():
    strip = types.SimpleNamespace(layout=lambda: None)
    assert fs.FoldStrip._apply_icon_scale(strip, 1.0) is None


def test_a_verdict_with_no_shared_ink_uses_the_literal(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.regression_qc",
                        types.SimpleNamespace(_VERDICT_INK={}))
    button = fs.FoldButton.__new__(fs.FoldButton)
    assert fs.FoldButton._verdict_ink(button, "pass") == "#3FB950"


def test_a_verdict_without_ink_paints_no_dot(qtbot, monkeypatch):
    button = fs.FoldButton.__new__(fs.FoldButton)
    from PySide6.QtWidgets import QPushButton

    QPushButton.__init__(button, "x")
    qtbot.addWidget(button)
    button._verdict = "odd"
    monkeypatch.setattr(fs.FoldButton, "_verdict_ink", lambda self, level: "")
    assert not button.grab().isNull()
