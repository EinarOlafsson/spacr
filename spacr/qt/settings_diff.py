"""
Settings-diff viewer — spot what changed between two runs.

Given two settings dicts (or two run-folder paths, or two CSVs), show
a color-coded diff so users can immediately see which knobs moved
between "the run that worked" and "the run that didn't".

Public API::

    from spacr.qt.settings_diff import diff_settings, SettingsDiffDialog

    changes = diff_settings(a, b)      # → list of (key, a_val, b_val, kind)
    SettingsDiffDialog(a, b, parent).exec()

Diff kinds:
  ``"added"``   — key present in B but not A
  ``"removed"`` — key present in A but not B
  ``"changed"`` — key in both, value differs
  ``"same"``    — filtered out; only difference kinds are surfaced
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pure diff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiffRow:
    """One diff entry."""
    key:   str
    a_val: Any
    b_val: Any
    kind:  str   # "added" / "removed" / "changed"


def diff_settings(a: Dict[str, Any], b: Dict[str, Any]) -> List[DiffRow]:
    """Return the list of keys that differ between ``a`` and ``b``.

    Sorted alphabetically by key. ``same``-valued keys are omitted.

    :param a: baseline settings dict.
    :param b: comparison settings dict.
    :returns: list of :class:`DiffRow`.
    """
    a = a or {}
    b = b or {}
    keys = sorted(set(a) | set(b))
    out: List[DiffRow] = []
    for k in keys:
        av, bv, in_a, in_b = a.get(k), b.get(k), k in a, k in b
        if in_a and in_b:
            if _normalize(av) != _normalize(bv):
                out.append(DiffRow(k, av, bv, "changed"))
        elif in_a:
            out.append(DiffRow(k, av, None, "removed"))
        else:
            out.append(DiffRow(k, None, bv, "added"))
    return out


def _normalize(v: Any) -> Any:
    """Weakly-canonicalise a value so `"1"` and `1` compare equal, etc."""
    if isinstance(v, str):
        s = v.strip()
        # Bool
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # Int
        try:
            return int(s)
        except (ValueError, TypeError):
            pass
        # Float
        try:
            return float(s)
        except (ValueError, TypeError):
            pass
        return s
    return v


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class SettingsDiffDialog:
    """Deferred: real Qt dialog is built on demand so this module can
    be imported (and diff_settings called) without needing PySide6."""

    def __new__(cls, a, b, parent=None, a_label="A", b_label="B"):
        # Lazy build of the Qt dialog when actually invoked in a GUI.
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QLabel, QTableWidget,
            QTableWidgetItem, QVBoxLayout,
        )

        dlg = QDialog(parent)
        dlg.setWindowTitle(f"Settings diff — {a_label} → {b_label}")
        dlg.setMinimumSize(720, 480)
        layout = QVBoxLayout(dlg)

        rows = diff_settings(_load(a), _load(b))
        summary = QLabel(
            f"<b>{len(rows)} differences</b> "
            f"({sum(1 for r in rows if r.kind=='changed')} changed, "
            f"{sum(1 for r in rows if r.kind=='added')} added, "
            f"{sum(1 for r in rows if r.kind=='removed')} removed) "
            f"between <code>{a_label}</code> and <code>{b_label}</code>."
        )
        summary.setTextFormat(Qt.RichText)
        layout.addWidget(summary)

        table = QTableWidget(len(rows), 4, dlg)
        table.setHorizontalHeaderLabels(
            ["Key", a_label, b_label, "Change"]
        )
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Colour palette per kind — inline styles so it works in both
        # light/dark themes without extra QSS.
        colours = {
            "added":   "#144d1e",
            "removed": "#4d1414",
            "changed": "#494914",
        }
        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(r.key))
            table.setItem(i, 1, QTableWidgetItem(_render(r.a_val)))
            table.setItem(i, 2, QTableWidgetItem(_render(r.b_val)))
            table.setItem(i, 3, QTableWidgetItem(r.kind))
            for col in range(4):
                item = table.item(i, col)
                if item is not None:
                    item.setBackground(_qcolor(colours[r.kind]))
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 220)
        layout.addWidget(table, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        return dlg


def _render(v: Any) -> str:
    if v is None:
        return "—"
    return repr(v)


def _qcolor(hex_str: str):
    from PySide6.QtGui import QColor
    return QColor(hex_str)


def _load(source) -> Dict[str, Any]:
    """Accept a dict, a Path to a run folder, or a Path to settings.json/csv."""
    if isinstance(source, dict):
        return source
    p = Path(source)
    if p.is_dir():
        from ..run_journal import load_run_settings
        return load_run_settings(p)
    if p.suffix == ".json":
        import json
        return json.loads(p.read_text())
    if p.suffix == ".csv":
        import csv
        out: Dict[str, Any] = {}
        with open(p) as f:
            for row in csv.reader(f):
                if row and row[0] and row[0] != "Key":
                    out[row[0]] = row[1] if len(row) > 1 else ""
        return out
    raise ValueError(f"unsupported source for _load: {source!r}")
