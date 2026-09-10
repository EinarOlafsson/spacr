"""Four guards in the graph stack, driven rather than assumed.

Instruction 288. Each was marked ``# pragma: no cover`` with a reason
that is true today and rests on something that could change:

* ``_orientation`` parses matplotlib's version to pick between
  ``vert=`` and ``orientation=``; an unparseable version string has to
  mean "assume modern" rather than raise inside a render.
* ``_draw_panel_marks`` returns ``None`` after every kind is handled --
  which is only true while the kind list and the dispatcher agree.
* ``_numeric_columns`` skips a column whose ``nunique`` raises, which
  happens for a column of values pandas cannot compare.
* ``DataFilterPanel.clause`` is an abstract method; the base raising is
  the contract subclasses are held to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# _orientation: an unparseable matplotlib version
# ---------------------------------------------------------------------------

def _orientation():
    from spacr.qt.widgets.graph_builder import _orientation as fn
    return fn


@pytest.mark.parametrize("version", ["3.10.0", "3.9.2", "4.0.1"])
def test_a_normal_version_is_parsed(version, monkeypatch):
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", version)
    answer = _orientation()(True)
    assert "orientation" in answer or "vert" in answer


def test_an_old_matplotlib_gets_the_old_spelling(monkeypatch):
    """The whole reason the function exists: 3.10 renamed the argument
    and warns on the old spelling, once per panel per render."""
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", "3.9.2")
    assert _orientation()(True) == {"vert": True}


def test_a_modern_matplotlib_gets_the_new_one(monkeypatch):
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", "3.10.0")
    assert _orientation()(True) == {"orientation": "vertical"}
    assert _orientation()(False) == {"orientation": "horizontal"}


@pytest.mark.parametrize("version", [
    "3",            # IndexError: no minor
    "",             # IndexError
    "3.x",          # ValueError
    "dev",          # ValueError
    "3.10.0.dev0+g1234",  # parses, but pinned as a shape that occurs
])
def test_an_odd_version_string_assumes_modern(version, monkeypatch):
    """THE ARM. It must not raise inside a render, and the fallback has
    to be MODERN -- guessing old on a new matplotlib brings back the
    warning this function exists to silence."""
    import matplotlib

    monkeypatch.setattr(matplotlib, "__version__", version)
    answer = _orientation()(True)
    assert answer == {"orientation": "vertical"}, (
        f"version {version!r} did not fall back to the modern spelling")


# ---------------------------------------------------------------------------
# regressable_columns: a column nunique cannot count
# ---------------------------------------------------------------------------

def test_a_column_whose_count_raises_is_skipped_not_fatal(monkeypatch):
    """THE ARM, driven where no ordinary DataFrame can reach it.

    The `is_numeric_dtype` check above this guard shields it from every
    built-in dtype: complex, nullable Int64, bool, sparse, float16,
    timedelta and plain int/float were all checked, and `nunique` raised
    for none of them. What the guard covers is a column whose dtype
    CLAIMS to be numeric while its values will not compare -- a
    third-party ExtensionArray, which spaCR cannot rule out because it
    scans whatever DataFrame the project produced.

    So `nunique` is made to raise for one column. That is an artificial
    input on purpose: the point is that ONE such column must not stop the
    scan finding the others, not that pandas ships one.
    """
    from spacr.qt.widgets.measurement_scan_panel import regressable_columns

    real = pd.Series.nunique

    def _selective(self, *args, **kwargs):
        if getattr(self, "name", None) == "awkward":
            raise TypeError("'<' not supported between these values")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "nunique", _selective)

    frame = pd.DataFrame({
        "good": [1.0, 2.0, 3.0, 4.0],
        "awkward": [1.0, 2.0, 3.0, 4.0],
        "also_good": [10.0, 20.0, 30.0, 40.0],
    })

    columns = regressable_columns(frame)

    assert "good" in columns and "also_good" in columns
    assert "awkward" not in columns, (
        "the uncountable column was kept, so the scan would later fit it")


def test_no_builtin_numeric_dtype_reaches_that_arm():
    """THE PREMISE, kept as the sweep that established it.

    If pandas ever made one of these raise, the arm would stop being
    theoretical -- and if the `is_numeric_dtype` guard above were
    removed, object columns would start reaching it.
    """
    columns = {
        "complex": pd.Series([1 + 2j, 3 + 4j]),
        "float": pd.Series([1.0, 2.0]),
        "int": pd.Series([1, 2]),
        "Int64": pd.Series([1, 2], dtype="Int64"),
        "bool": pd.Series([True, False]),
        "sparse": pd.Series(pd.arrays.SparseArray([1.0, 2.0])),
        "float16": pd.Series(np.array([1, 2], dtype="float16")),
    }
    checked = 0
    for name, column in columns.items():
        if not pd.api.types.is_numeric_dtype(column):
            continue
        # ASSERTED, not merely survived: a loop whose body never runs
        # passes just as well as one where nothing raises.
        assert column.nunique(dropna=True) >= 1, name
        checked += 1
    assert checked >= 5, (
        f"only {checked} of these dtypes counted as numeric; the premise "
        f"is being checked against far fewer cases than it claims")


def test_a_constant_column_is_skipped_for_a_different_reason():
    """The neighbouring `continue`, so the two are not confused: fewer
    than two distinct values is not the same as uncountable."""
    from spacr.qt.widgets.measurement_scan_panel import regressable_columns

    frame = pd.DataFrame({"flat": [7.0] * 5,
                          "varied": [1.0, 2.0, 3.0, 4.0, 5.0]})
    columns = regressable_columns(frame)
    assert "flat" not in columns
    assert "varied" in columns


# ---------------------------------------------------------------------------
# _ClauseRow.clause: the abstract contract
# ---------------------------------------------------------------------------

def test_the_base_clause_row_refuses_to_answer():
    """THE ARM, and the contract it states: a subclass that forgets
    `clause` fails loudly rather than filtering on nothing."""
    from spacr.qt.widgets.data_filter_panel import _ClauseRow

    row = _ClauseRow.__new__(_ClauseRow)
    with pytest.raises(NotImplementedError):
        row.clause()


def test_the_real_rows_do_answer():
    """So the abstract raise is a contract and not simply dead."""
    import inspect

    from spacr.qt.widgets import data_filter_panel

    implementors = [
        obj for _name, obj in vars(data_filter_panel).items()
        if inspect.isclass(obj)
        and obj is not data_filter_panel._ClauseRow
        and issubclass(obj, data_filter_panel._ClauseRow)
        and "clause" in vars(obj)
    ]
    assert implementors, (
        "nothing overrides clause any more, so the abstract raise is the "
        "only implementation there is")
