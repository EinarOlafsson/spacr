"""
Settings-diff viewer — spot what changed between two runs.

Given two settings dicts (or two run-folder paths, or two CSVs), show
a color-coded diff so users can immediately see which knobs moved
between "the run that worked" and "the run that didn't".

Public API::

    from spacr.qt.settings_diff import (diff_settings, diff_settings_grouped,
                                        SettingsDiffDialog)

    changes = diff_settings(a, b)      # → list of (key, a_val, b_val, kind)
    grouped = diff_settings_grouped(a, b)   # the same, by settings category
    SettingsDiffDialog(a, b, parent).exec()

Diff kinds:

* ``"added"``   — key present in B but not A
* ``"removed"`` — key present in A but not B
* ``"changed"`` — key in both, value differs
* ``"same"``    — a key both runs set to the same value. Never returned by
  :func:`diff_settings`; :func:`diff_settings_grouped` carries it only when
  asked, because the default view of a 200-key settings dict has to be the
  handful that moved.

Grouping uses the same ``spacr.settings.categories`` map both GUIs group
their settings panels by, so "what changed" is read under the same
headings the user set the values under. A key nobody categorised lands in
:data:`UNCATEGORISED` rather than being dropped — an unclassified knob is
still a knob that moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


#: Heading for keys that appear in no ``spacr.settings.categories`` bucket.
#: The same word the Qt settings panel puts them under, so the diff and the
#: form that produced it do not disagree about where a key lives.
UNCATEGORISED = "Other"

#: category → keys, resolved once. ``None`` until the first lookup, so
#: importing this module never drags in :mod:`spacr.settings` (about a
#: second of imports) for a caller that only wants :func:`diff_settings`.
_CATEGORY_OF: Optional[Dict[str, str]] = None

#: Category display order, taken from ``spacr.settings.categories``.
_CATEGORY_ORDER: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pure diff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiffRow:
    """One diff entry."""
    key:   str
    a_val: Any
    b_val: Any
    kind:  str   # "added" / "removed" / "changed" / "same"

    @property
    def category(self) -> str:
        """The settings heading this key is grouped under."""
        return setting_category(self.key)


@dataclass(frozen=True)
class CategoryDiff:
    """Every row from one settings category, differences first.

    :param category: the heading, e.g. ``"Cellpose"``.
    :param rows: the rows that differ, sorted by key.
    :param same: the rows both runs agree on. Empty unless the caller
        asked for them; :attr:`n_same` counts them either way.
    :param n_same: how many keys in this category matched, whether or not
        ``same`` was populated. It is what lets the collapsed view say
        "Cellpose: 2 changed, 14 unchanged" without carrying 14 rows.
    """

    category: str
    rows: Tuple[DiffRow, ...] = ()
    same: Tuple[DiffRow, ...] = ()
    n_same: int = 0

    @property
    def n_changed(self) -> int:
        """Rows whose value moved."""
        return sum(1 for r in self.rows if r.kind == "changed")

    @property
    def n_added(self) -> int:
        """Keys only the second run set."""
        return sum(1 for r in self.rows if r.kind == "added")

    @property
    def n_removed(self) -> int:
        """Keys only the first run set."""
        return sum(1 for r in self.rows if r.kind == "removed")

    def __len__(self) -> int:
        """Number of differing rows."""
        return len(self.rows)


@dataclass(frozen=True)
class SettingsDiff:
    """The whole settings comparison, grouped by category.

    :param categories: one :class:`CategoryDiff` per heading that has
        something to show, in ``spacr.settings.categories`` order.
    :param include_same: whether unchanged keys were carried.
    :param n_same: how many keys both runs set to the same value, over
        the whole comparison. A stored count and not a sum over
        :attr:`categories`, because the default view omits the categories
        in which *nothing* differs — and "17 settings matched" is still
        true, and still worth saying, when none of those 17 has a row.
    """

    categories: Tuple[CategoryDiff, ...] = ()
    include_same: bool = False
    n_same: int = 0

    @property
    def rows(self) -> Tuple[DiffRow, ...]:
        """Every differing row, category order then key order."""
        return tuple(r for c in self.categories for r in c.rows)

    @property
    def n_changed(self) -> int:
        """Keys both runs set, to different values."""
        return sum(c.n_changed for c in self.categories)

    @property
    def n_added(self) -> int:
        """Keys only the second run set."""
        return sum(c.n_added for c in self.categories)

    @property
    def n_removed(self) -> int:
        """Keys only the first run set."""
        return sum(c.n_removed for c in self.categories)

    @property
    def identical(self) -> bool:
        """True when nothing differs at all."""
        return not any(c.rows for c in self.categories)

    def category(self, name: str) -> Optional[CategoryDiff]:
        """Return one category's block, or ``None`` if it has nothing."""
        for candidate in self.categories:
            if candidate.category == name:
                return candidate
        return None

    def summary(self) -> str:
        """One sentence: how much moved, and where."""
        if self.identical:
            return "Settings are identical."
        parts = []
        for label, count in (("changed", self.n_changed),
                             ("added", self.n_added),
                             ("removed", self.n_removed)):
            if count:
                parts.append(f"{count} {label}")
        where = ", ".join(c.category for c in self.categories if c.rows)
        return f"{'; '.join(parts)} in {where}."

    def __len__(self) -> int:
        """Number of differing rows."""
        return len(self.rows)


def diff_settings(a: Dict[str, Any], b: Dict[str, Any]) -> List[DiffRow]:
    """Return the list of keys that differ between ``a`` and ``b``.

    Sorted alphabetically by key. ``same``-valued keys are omitted.

    :param a: baseline settings dict.
    :param b: comparison settings dict.
    :returns: list of :class:`DiffRow`.
    """
    return [row for row in _all_rows(a, b) if row.kind != "same"]


def _all_rows(a: Dict[str, Any], b: Dict[str, Any]) -> List[DiffRow]:
    """Every key in either dict as a :class:`DiffRow`, ``"same"`` included."""
    a = a or {}
    b = b or {}
    keys = sorted(set(a) | set(b))
    out: List[DiffRow] = []
    for k in keys:
        av, bv, in_a, in_b = a.get(k), b.get(k), k in a, k in b
        if in_a and in_b:
            kind = "same" if _values_equal(av, bv) else "changed"
            out.append(DiffRow(k, av, bv, kind))
        elif in_a:
            out.append(DiffRow(k, av, None, "removed"))
        else:
            out.append(DiffRow(k, None, bv, "added"))
    return out


def diff_settings_grouped(a: Dict[str, Any], b: Dict[str, Any], *,
                          include_same: bool = False) -> SettingsDiff:
    """Diff two settings dicts and group the result by settings category.

    A spaCR run carries around two hundred keys, so an ungrouped diff of
    two runs that differ in one Cellpose knob and one plate-map column
    reads as an undifferentiated list. Grouping under the same headings
    the settings panel uses makes it answerable at a glance: *the change
    was in Cellpose*.

    :param a: baseline settings dict.
    :param b: comparison settings dict.
    :param include_same: also carry the keys both runs agree on, for the
        "show everything" toggle. Off by default — the point of the
        default view is that an unchanged setting is not in it.
    :returns: a :class:`SettingsDiff`. Categories with nothing to show
        are absent; with ``include_same`` that means categories neither
        run mentions at all.
    """
    rows = _all_rows(a, b)
    buckets: Dict[str, List[DiffRow]] = {}
    same: Dict[str, List[DiffRow]] = {}
    for row in rows:
        target = same if row.kind == "same" else buckets
        target.setdefault(row.category, []).append(row)

    blocks: List[CategoryDiff] = []
    for name in _ordered_categories(set(buckets) | set(same)):
        differing = tuple(buckets.get(name, ()))
        matching = tuple(same.get(name, ()))
        if not differing and not (include_same and matching):
            continue
        blocks.append(CategoryDiff(
            category=name,
            rows=differing,
            same=matching if include_same else (),
            n_same=len(matching),
        ))
    return SettingsDiff(categories=tuple(blocks), include_same=include_same,
                        n_same=sum(len(v) for v in same.values()))


def setting_category(key: str) -> str:
    """Return the settings heading ``key`` is grouped under.

    :param key: a settings key, e.g. ``"cell_diameter"``.
    :returns: the category name, or :data:`UNCATEGORISED` when the key is
        in no bucket (a plugin's key, a run-journal bookkeeping column, or
        a knob nobody has filed yet).
    """
    return _category_map().get(str(key), UNCATEGORISED)


def _category_map() -> Dict[str, str]:
    """key → category, built once from ``spacr.settings.categories``.

    Falls back to an empty map — everything :data:`UNCATEGORISED` — when
    :mod:`spacr.settings` cannot be imported, so a diff still renders in
    an environment where the heavy settings module is unavailable.
    """
    global _CATEGORY_OF, _CATEGORY_ORDER
    if _CATEGORY_OF is None:
        mapping: Dict[str, str] = {}
        order: List[str] = []
        try:
            from ..settings import categories as _categories
        except Exception:
            _categories = {}
        for name, keys in dict(_categories).items():
            order.append(str(name))
            for key in keys or ():
                # First bucket wins. `tests/test_settings_categories.py`
                # forbids a key appearing twice, but a plugin merging its
                # own categories in is not covered by that test, and a
                # silent overwrite would move the key under a heading the
                # settings panel does not put it under.
                mapping.setdefault(str(key), str(name))
        _CATEGORY_OF = mapping
        _CATEGORY_ORDER = tuple(order)
    return _CATEGORY_OF


def _ordered_categories(present: Sequence[str]) -> List[str]:
    """Sort category names into settings-panel order.

    Declared categories keep the order ``spacr.settings.categories``
    writes them in — the order the user reads them in the settings form.
    Anything else (a plugin heading, :data:`UNCATEGORISED`) follows
    alphabetically, with :data:`UNCATEGORISED` pinned last because "Other"
    is where you look when the answer was not under a real heading.
    """
    _category_map()
    present = set(present)
    ordered = [name for name in _CATEGORY_ORDER if name in present]
    rest = sorted(present - set(ordered) - {UNCATEGORISED})
    if UNCATEGORISED in present:
        rest.append(UNCATEGORISED)
    return ordered + rest


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two setting values structurally.

    Delegates to :func:`spacr.run_journal.values_equal` so this dialog and
    :func:`spacr.run_journal.diff_runs` can never disagree about what counts
    as a change. That matters in practice: the journal round-trips settings
    through CSV, so an older run stores ``channels`` as the string
    ``"[0, 1, 2]"`` while a newer one stores the list ``[0, 1, 2]``. The
    local normaliser below only handles str->int/float/bool, so it reported
    that pair as a change and the dialog showed differences that weren't real.

    Falls back to :func:`_normalize` if run_journal cannot be imported, so
    this module still works standalone.
    """
    try:
        from ..run_journal import values_equal
    except Exception:
        return _normalize(a) == _normalize(b)
    return values_equal(a, b)


def _normalize(v: Any) -> Any:
    """Weakly-canonicalise a value so `"1"` and `1` compare equal, etc.

    Retained as the offline fallback for :func:`_values_equal`.
    """
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
            tint = _qcolor(colours[r.kind])
            cells = (r.key, _render(r.a_val), _render(r.b_val), r.kind)
            for col, text in enumerate(cells):
                # Built and coloured in one pass. Setting the four cells
                # and then reading them back left a `table.item(...) is
                # None` branch that could not happen and was never tested.
                item = QTableWidgetItem(text)
                item.setBackground(tint)
                table.setItem(i, col, item)
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
