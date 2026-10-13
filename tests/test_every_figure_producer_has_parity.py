"""Every spaCR figure producer has N658 parity.

``tools/audit_figure_producers.py`` finds every place in ``spacr/`` that
makes a Matplotlib figure and records, for each, whether it registers its
tidy data and plot spec, how the user's figure settings reach it, which
writer saves it and where the right-click menu (Edit, Change graph type,
Statistics, Save zip) reaches it. This guard fails when a producer misses
any of them, so a new figure cannot arrive without the menu, the style or
its data. The dated audit is
``features/data/658_producer_parity_2026-10-10/producers.csv``.
"""
from __future__ import annotations

import functools
import pathlib
import textwrap

from tools import audit_figure_producers as audit


_SCAN = audit.scan


@functools.lru_cache(maxsize=1)
def _spacr_rows() -> tuple:
    """The audit of the real package, scanned once per session."""
    return tuple(_SCAN())


def _package(tmp_path, files: dict) -> pathlib.Path:
    """A throwaway package with ``files`` (relative path to source)."""
    root = tmp_path / "pkg"
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


def _by_function(rows) -> dict:
    return {row["function"]: row for row in rows}


def test_every_producer_has_parity():
    """No producer misses registration, style or the menu."""
    rows = _spacr_rows()
    missing = [row for row in rows if row["parity"] == "missing"]
    assert not missing, (
        "These figure producers miss N658 parity (registers / styled_by / "
        "menu is empty or 'no'). Register the data with "
        "spacr.figures.bundle._register_figure_data, show the figure through "
        "the figure queue, the shared menu canvas or "
        "figure_settings._show_rendered_figure_menu, and let "
        "save_figure or _apply_user_style style it:\n"
        + "\n".join(f"  spacr/{r['module']}:{r['line']} {r['function']} "
                    f"registers={r['registers']} styled={r['styled_by']!r} "
                    f"menu={r['menu']!r}" for r in missing))


def test_the_audit_sees_every_known_producer():
    """The scan is not silently empty: it sees the producers it audited."""
    rows = _spacr_rows()
    assert len(rows) >= 180
    seen = {(row["module"], row["function"]) for row in rows}
    for expected in [("qt/screens/hyperparam.py", "_trial_figure"),
                     ("qt/screens/hyperparam.py", "build_panel_figure"),
                     ("response_distribution.py", "panel"),
                     ("qt/screens/control_chart.py",
                      "ControlChartCanvas.__init__"),
                     ("plot.py", "plot_histogram")]:
        assert expected in seen, expected


def test_the_exemptions_are_the_registration_guards():
    """One list of figures that never reach a user as a graph."""
    import importlib
    import sys

    tests = str(pathlib.Path(__file__).resolve().parent)
    sys.path.insert(0, tests)
    try:
        guard = importlib.import_module("test_every_figure_registers_its_data")
    finally:
        sys.path.remove(tests)
    assert set(audit.EXEMPT) == set(guard.EXEMPT)
    for row in _spacr_rows():
        if row["parity"] == "exempt":
            assert row["reason"], row


def test_the_audit_flags_each_missing_part(tmp_path):
    """Unregistered, unstyled and menu-less producers are each caught."""
    root = _package(tmp_path, {
        "plots.py": """
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from .figures.bundle import _register_figure_data

            def bare(df):
                fig, ax = plt.subplots()
                ax.plot(df.x, df.y)
                return fig

            def registered(df):
                fig, ax = plt.subplots()
                _register_figure_data(fig, df, x="x", y="y")
                plt.show()

            def _hidden(df):
                fig = Figure()
                _register_figure_data(fig, df, x="x", y="y")
                return fig

            def implicit(df):
                import seaborn as sns
                sns.boxplot(data=df, x="g", y="v")
        """,
        "qt/view.py": """
            from matplotlib.figure import Figure
            from ..figures.bundle import _register_figure_data

            def figure_to_pixmap(fig):
                return fig

            def thumbnail(df):
                fig = Figure()
                _register_figure_data(fig, df, x="x", y="y")
                return figure_to_pixmap(fig)
        """,
    })
    rows = _by_function(audit.scan(root))
    assert rows["bare"]["parity"] == "missing"
    assert rows["bare"]["registers"] == "no"
    assert rows["registered"]["parity"] == "ok"
    assert rows["registered"]["menu"] == "figure queue"
    assert "figure queue render" in rows["registered"]["styled_by"]
    assert rows["_hidden"]["parity"] == "missing"
    assert rows["_hidden"]["styled_by"] == ""
    assert rows["implicit"]["family"] == "seaborn-implicit"
    assert rows["implicit"]["parity"] == "missing"
    assert rows["thumbnail"]["parity"] == "missing"
    assert rows["thumbnail"]["menu"] == ""


def test_the_audit_credits_each_route(tmp_path):
    """save_figure, the menu canvas and the rendered-picture menu count."""
    root = _package(tmp_path, {
        "writer.py": """
            from matplotlib.figure import Figure
            from .plot import save_figure as save_to
            from .figures.bundle import _register_figure_data

            def _draw(df, path):
                fig = Figure()
                _register_figure_data(fig, df, x="x", y="y")
                save_to(fig, path)
        """,
        "qt/widgets/graph_builder.py": """
            def _canvas_class():
                from ...figures.style import _style_new_content

                class Canvas:
                    def draw(self):
                        _style_new_content(self.figure)
                return Canvas
        """,
        "qt/screen.py": """
            from matplotlib.figure import Figure
            from .widgets.graph_builder import _canvas_class
            from ..figures.bundle import _register_figure_data

            class Screen:
                def __init__(self, df):
                    self.figure = Figure()
                    self.canvas = _canvas_class()(self.figure)

                def draw(self, df):
                    _register_figure_data(self.figure, df, x="x", y="y")
        """,
        "qt/pictures.py": """
            from matplotlib.figure import Figure
            from ..figures.bundle import _register_figure_data
            from ..figures.style import _apply_user_style

            def figure_to_pixmap(fig):
                return fig

            def thumbnail(df):
                fig = Figure()
                _register_figure_data(fig, df, x="x", y="y")
                _apply_user_style(fig)
                return figure_to_pixmap(fig)

            def menu(widget, fig):
                from .widgets.figure_settings import (
                    _show_rendered_figure_menu)
                _show_rendered_figure_menu(widget, fig, None, print)
        """,
    })
    rows = _by_function(audit.scan(root))
    assert rows["_draw"]["parity"] == "ok"
    assert rows["_draw"]["saved_by"] == "save_figure"
    assert rows["_draw"]["menu"] == "file only"
    assert rows["Screen.__init__"]["parity"] == "ok"
    assert rows["Screen.__init__"]["styled_by"] == "menu canvas draw"
    assert rows["thumbnail"]["parity"] == "ok"
    assert rows["thumbnail"]["menu"] == "rendered-picture menu"


def test_the_csv_writes_one_row_per_producer(tmp_path, monkeypatch):
    """The tool writes a header and one row per producer, and exits 0."""
    monkeypatch.setattr(audit, "scan", lambda: list(_spacr_rows()))
    out = tmp_path / "producers.csv"
    assert audit.main(["--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(audit.FIELDS)
    assert len(lines) - 1 == len(_spacr_rows())


def test_the_tool_fails_when_a_producer_misses_parity(tmp_path, monkeypatch):
    """A missing row makes the tool exit non-zero."""
    row = dict(_spacr_rows()[0], parity="missing")
    monkeypatch.setattr(audit, "scan", lambda: [row])
    assert audit.main(["--out", str(tmp_path / "p.csv")]) == 1
