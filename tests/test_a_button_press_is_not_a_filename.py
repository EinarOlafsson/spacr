"""Qt's ``clicked``/``triggered`` bool must never be read as a path.

REPORTED BY THE MAINTAINER, 2026-08-20, pressing Export on a plot:

    TypeError: 'PySide6.QtGui.QImage.save' called with wrong argument types:
      PySide6.QtGui.QImage.save(bool)

`clicked` carries a `checked` bool. `FastPlotWidget.export(path=None)` takes
an optional first argument, so Qt handed the bool straight into it; `False is
None` is False, so the save dialog never opened and `False` travelled all the
way to QImage.save.

IT IS A CLASS OF BUG, NOT ONE BUG. An AST sweep of every
``clicked/triggered.connect(self.method)`` whose method takes an optional
first argument found eleven more sites. Most are harmless because they test
``if not path`` and a bool is falsy -- but two tested ``is None``, which a
bool passes straight through:

    FastPlotWidget.export          the one the maintainer hit
    CellMontageView.save           the montage's own Save figure button

Both now normalise a bool to None, and their connections pass no argument.
The normalising guard stays in the methods as well as at the seam: both are
public, and the next person to wire a button to one should not have to know
that Qt's signal carries an argument they do not want.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "spacr"

#: Signals whose first argument is a bool nobody asked for.
BOOL_SIGNALS = ("clicked", "triggered", "toggled")


def _optional_first_argument_methods(tree):
    """Methods whose first parameter after ``self`` has a default."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            positional = node.args.args[1:]
            if positional and len(node.args.defaults) >= len(positional):
                out[node.name] = node.lineno
    return out


def _bare_connections():
    """Every ``<bool signal>.connect(self.method)`` with no lambda."""
    found = []
    for path in ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):       # pragma: no cover
            continue
        takers = _optional_first_argument_methods(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute)
                    and func.attr == "connect"
                    and getattr(func.value, "attr", "") in BOOL_SIGNALS):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if (isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "self"
                    and arg.attr in takers):
                found.append((path.name, node.lineno, arg.attr))
    return found


class TestTheTwoThatWereBroken:
    """These tested ``is None``, so the bool sailed past the dialog."""

    def test_export_treats_a_bool_as_no_path(self):
        source = (ROOT / "qt" / "widgets" / "fast_plots.py").read_text()
        body = source[source.index("def export(self, path"):]

        assert "isinstance(path, bool)" in body[:2000], (
            "a button press must not be mistaken for a filename")

    def test_the_montage_save_treats_a_bool_as_no_path(self):
        source = (ROOT / "qt" / "widgets" / "cell_montage_view.py").read_text()
        body = source[source.index("def save(self, path"):]

        assert "isinstance(path, bool)" in body[:2000]

    @pytest.mark.parametrize("name,method", [
        ("fast_plots.py", "export"),
        ("cell_montage_view.py", "save"),
    ])
    def test_neither_is_connected_bare_any_more(self, name, method):
        bare = [hit for hit in _bare_connections()
                if hit[0] == name and hit[2] == method]

        assert not bare, (
            f"{name}:{method} is connected without a lambda, so Qt's bool "
            f"reaches its first argument again: {bare}")


class TestTheRestOfTheClass:
    """The sweep that found them, kept so the class cannot come back unseen."""

    def test_every_remaining_bare_connection_tolerates_a_bool(self):
        """A bare connection is allowed ONLY where a bool is harmless.

        Harmless means the method treats its argument falsily -- ``if not
        path`` or ``path or default`` -- so Qt's ``False`` behaves exactly
        like the default. Testing ``is None`` is what makes it a bug.
        """
        offenders = []
        for name, line, method in _bare_connections():
            path = next(ROOT.rglob(name))
            source = path.read_text(encoding="utf-8")
            marker = f"def {method}(self"
            if marker not in source:
                continue
            body = source[source.index(marker):][:2500]
            first = body.split("\n", 1)[0]
            argument = first.split("self,", 1)[-1].split(":")[0].split("=")[0]
            argument = argument.strip().strip(")").strip()
            if not argument:
                continue
            if f"{argument} is None" in body and "isinstance" not in body:
                offenders.append(f"{name}:{line} {method}({argument})")

        assert not offenders, (
            "these read Qt's checked bool as a real argument because they "
            f"test `is None`: {offenders}")


class TestTheOtherCrashTheSameSessionReported:
    """An all-missing measurement is an empty comparison, not a traceback.

    REPORTED 2026-08-20, opening Compare on the Cells tab:

        ValueError: Cannot set a DataFrame with multiple columns to the
        single column unit

    `build` drops rows whose value is not a number, and when that empties the
    frame `agg("_".join, axis=1)` returns an empty DATAFRAME of the key
    columns rather than a Series -- assigning which to one column raises.
    Every other empty case in `build` returns a Comparison carrying its
    reason, and so does this one now.
    """

    def test_a_measurement_with_no_numbers_returns_a_reason(self):
        import numpy as np
        import pandas as pd

        from spacr.gene_measurement_compare import build

        objects = pd.DataFrame({
            "plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
            "columnID": ["c1", "c2"], "cell_area": [np.nan, np.nan]})

        comparison = build(objects, "cell_area",
                           groups={"a": [0], "b": [1]}, level="well")

        assert len(comparison.frame) == 0
        assert "numeric" in comparison.note
        assert "cell_area" in comparison.note

    def test_text_that_is_not_a_number_is_the_same_case(self):
        import pandas as pd

        from spacr.gene_measurement_compare import build

        objects = pd.DataFrame({
            "plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
            "columnID": ["c1", "c2"], "cell_area": ["n/a", "missing"]})

        comparison = build(objects, "cell_area",
                           groups={"a": [0], "b": [1]}, level="well")

        assert len(comparison.frame) == 0 and comparison.note

    def test_a_plate_level_comparison_survives_it_too(self):
        import numpy as np
        import pandas as pd

        from spacr.gene_measurement_compare import build

        objects = pd.DataFrame({"plateID": ["p1", "p2"],
                                "cell_area": [np.nan, np.nan]})

        comparison = build(objects, "cell_area",
                           groups={"a": [0]}, level="plate")

        assert len(comparison.frame) == 0 and comparison.note

    def test_a_measurement_that_still_has_numbers_is_unaffected(self):
        import pandas as pd

        from spacr.gene_measurement_compare import build

        objects = pd.DataFrame({
            "plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
            "columnID": ["c1", "c2"], "cell_area": [10.0, 20.0]})

        comparison = build(objects, "cell_area",
                           groups={"a": [0], "b": [1]}, level="well")

        assert len(comparison.frame) == 2
