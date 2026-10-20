"""An alternative version with no note of its own gets no tooltip.

The coverage gate named the branch of ``_mark_alternatives`` that skips such
a row (spacr/qt/screens/model_zoo.py, run 38151063008). The row is passed
over and the next alternative still gets its own reason.
"""
from __future__ import annotations

from types import SimpleNamespace

from spacr import model_zoo as mz
from spacr.qt.screens import model_zoo as screen


class _Combo:
    def __init__(self):
        self.items = []

    def setItemData(self, index, value, role):
        self.items.append((index, value))


def test_only_an_alternative_with_a_note_gets_a_tooltip():
    alternative = sorted(mz.ALTERNATIVE_MODEL_KEYS)[0]
    pairs = [
        ("v4", SimpleNamespace(key="toxoplasma_pv_v4", notes=("default",))),
        ("v5", SimpleNamespace(key=alternative, notes=())),
        ("v6", SimpleNamespace(key=alternative, notes=("why this one",))),
    ]
    combo = _Combo()

    screen._mark_alternatives(combo, pairs)

    assert [index for index, _value in combo.items] == [2]
    assert "why this one" in str(combo.items[0][1])
