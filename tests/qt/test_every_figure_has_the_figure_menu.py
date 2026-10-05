"""Every matplotlib canvas offers one figure menu: edit, retype, test, zip.

Right-click on any figure gives "Edit figure…", "Change graph type" (the
kinds the data fits, redrawn from the data the figure carries),
"Statistics…" (chosen from the data, overridable) and "Save figure (zip)…"
(image, data, one statistics CSV, the recipe and a script that re-creates
the figure).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("seaborn")

from matplotlib.figure import Figure  # noqa: E402

from spacr.figures import bundle  # noqa: E402
from spacr.figures.stats import _auto_statistics  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _groups(spec: dict, seed: int = 0) -> pd.DataFrame:
    """A tidy frame from ``{label: sampler(rng, n)}``."""
    rng = np.random.default_rng(seed)
    rows = [(label, float(v)) for label, draw in spec.items()
            for v in draw(rng)]
    return pd.DataFrame(rows, columns=["group", "value"])


def _quantiles(n):
    from scipy import stats
    return stats.norm.ppf((np.arange(n) + 0.5) / n)


def _normal(mean, sd=1.0, n=40):
    """Exactly normal values in random order, so the checks are not luck."""
    return lambda rng: rng.permutation(mean + sd * _quantiles(n))


def _skewed(mean, n=40):
    return lambda rng: rng.permutation(np.exp(mean + _quantiles(n)))


def _rows(table, stage):
    return table[table["test_stage"] == stage]


def _names(table, stage):
    return set(_rows(table, stage)["test_name"])


@pytest.fixture
def registered():
    frame = _groups({"ctrl": _normal(0), "drug": _normal(1),
                     "both": _normal(2)})
    figure = Figure(figsize=(4, 3), dpi=80)
    bundle._register_figure_data(figure, frame, x="group", y="value",
                                 kind="box", title="Dose")
    bundle._draw(figure, frame, figure._spacr_spec)
    return figure


class TestEveryCanvasCarriesTheMenu:

    def test_every_canvas_class_attaches_the_shared_menu(self):
        """A canvas subclass or bare construction anywhere under spacr/qt
        lives in a file that attaches the menu or builds it."""
        offenders = []
        for path in (ROOT / "spacr" / "qt").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            makes = re.search(r"class \w+\(FigureCanvasQTAgg\)|"
                              r"FigureCanvasQTAgg\(\w", text)
            if makes and "_attach_figure_menu" not in text and \
                    "build_figure_context_menu" not in text:
                offenders.append(str(path.relative_to(ROOT)))
        assert offenders == []

    def test_custom_menus_add_the_shared_entries(self):
        for name in ("spacr/qt/widgets/volcano_explorer.py",
                     "spacr/qt/screens/gate_editor.py"):
            assert "_add_figure_tools" in (ROOT / name).read_text(
                encoding="utf-8"), name

    @pytest.mark.parametrize("factory", ["graph_builder", "train_compare"])
    def test_a_right_click_opens_the_menu(self, qapp, monkeypatch,
                                          registered, factory):
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtWidgets import QApplication

        if factory == "graph_builder":
            from spacr.qt.widgets.graph_builder import _canvas_class
            canvas = _canvas_class()(registered)
        else:
            from spacr.qt.screens.train_compare import panel_canvas_class
            canvas = panel_canvas_class()(registered)
        shown = []
        monkeypatch.setattr(fs, "_exec_menu",
                            lambda menu, *a, **k: shown.append(menu))
        event = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(5, 5),
                                  QPoint(5, 5))
        QApplication.sendEvent(canvas, event)
        assert shown, "no menu opened"
        titles = [a.text() for a in shown[0].actions()]
        for expected in ("Edit figure…", "Statistics…",
                         "Save figure (zip)…", "Change graph type"):
            assert expected in titles
        canvas.deleteLater()

    def test_a_canvas_with_its_own_menu_keeps_it(self, qapp, monkeypatch,
                                                 registered):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.graph_builder import _canvas_class

        canvas = _canvas_class()(registered)
        canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        own, shown = [], []
        canvas.customContextMenuRequested.connect(own.append)
        monkeypatch.setattr(fs, "_exec_menu",
                            lambda menu, *a, **k: shown.append(menu))
        QApplication.sendEvent(canvas, QContextMenuEvent(
            QContextMenuEvent.Mouse, QPoint(3, 3), QPoint(3, 3)))
        assert own and not shown
        canvas.deleteLater()


class TestChangeGraphType:

    def test_only_the_kinds_the_data_fits_are_offered(self, qapp,
                                                      registered):
        from spacr.qt.widgets.figure_settings import build_figure_context_menu

        menu = build_figure_context_menu(None, registered)
        retype = next(a.menu() for a in menu.actions()
                      if a.menu() and a.menu().title() == "Change graph type")
        offered = {a.text() for a in retype.actions()}
        assert {"Box", "Violin", "Strip", "Swarm", "Bar", "Point",
                "Boxen"} <= offered
        assert "Scatter" not in offered and "Heatmap" not in offered

    @pytest.mark.parametrize("spec,expected", [
        ({"x": "a", "y": "b"}, {"scatter", "line", "hex", "kde", "reg"}),
        ({"y": "b"}, {"hist", "kde", "ecdf"}),
        ({"matrix": True}, {"heatmap", "clustermap"}),
    ])
    def test_the_families(self, spec, expected):
        frame = pd.DataFrame({"a": np.arange(20.0), "b": np.ones(20)})
        kinds = {k for k, _c in bundle._kinds_for(frame, spec)}
        assert expected <= kinds

    def test_retyping_redraws_from_the_stored_data(self, qapp, registered):
        from matplotlib.collections import PolyCollection

        from spacr.qt.widgets.figure_settings import _retype

        assert _retype(registered, "violin")
        assert registered._spacr_spec["kind"] == "violin"
        assert any(isinstance(c, PolyCollection)
                   for c in registered.axes[0].collections)
        assert _retype(registered, "swarm")
        offsets = sum(len(c.get_offsets())
                      for c in registered.axes[0].collections)
        assert offsets == 120

    def test_every_offered_kind_draws(self):
        rng = np.random.default_rng(1)
        frames = [
            (_groups({"a": _normal(0), "b": _normal(1)}),
             {"x": "group", "y": "value"}),
            (pd.DataFrame({"a": rng.normal(size=60),
                           "b": rng.normal(size=60)}), {"x": "a", "y": "b"}),
            (pd.DataFrame({"b": rng.normal(size=60)}), {"y": "b"}),
            (pd.DataFrame({"a": rng.choice(list("XY"), 60),
                           "b": rng.choice(list("PQ"), 60)}),
             {"x": "a", "y": "b"}),
            (pd.DataFrame(rng.normal(size=(6, 4)), columns=list("ABCD")),
             {"matrix": True}),
        ]
        figure = Figure()
        for frame, spec in frames:
            for kind, _caption in bundle._kinds_for(frame, spec):
                ax = bundle._draw(figure, frame, dict(spec, kind=kind))
                assert ax.has_data() or ax.images or ax.collections, kind


class TestStatisticsAreChosenFromTheData:

    def test_two_normal_groups_take_student_t(self):
        t = _auto_statistics(_groups({"a": _normal(0), "b": _normal(1)}),
                             "group", "value")
        assert list(t["test_stage"])[:3] == ["normality", "normality",
                                             "equal_variance"]
        assert _names(t, "equal_variance") == {"Bartlett"}
        assert _names(t, "pairwise") == {"Student's t"}

    def test_unequal_variance_takes_welch(self):
        t = _auto_statistics(_groups({"a": _normal(0, 1), "b": _normal(
            1, 5)}), "group", "value")
        assert _names(t, "pairwise") == {"Welch's t"}

    def test_skewed_groups_take_mann_whitney_and_levene(self):
        t = _auto_statistics(_groups({"a": _skewed(0), "b": _skewed(1)}),
                             "group", "value")
        assert _names(t, "equal_variance") == {"Levene (median-centred)"}
        assert _names(t, "pairwise") == {"Mann-Whitney U"}

    def test_three_normal_groups_take_anova_then_tukey(self):
        t = _auto_statistics(_groups({k: _normal(i) for i, k in
                                      enumerate("abc")}), "group", "value")
        assert _names(t, "omnibus") == {"one-way ANOVA"}
        assert _names(t, "pairwise") == {"Tukey HSD"}
        assert len(_rows(t, "pairwise")) == 3

    def test_three_unequal_groups_take_welch_anova_then_games_howell(self):
        t = _auto_statistics(_groups({k: _normal(i, 1 + 3 * i) for i, k in
                                      enumerate("abc")}), "group", "value")
        assert _names(t, "omnibus") == {"Welch's ANOVA"}
        assert _names(t, "pairwise") == {"Games-Howell"}

    def test_three_skewed_groups_take_kruskal_then_corrected_dunn(self):
        t = _auto_statistics(_groups({k: _skewed(i) for i, k in
                                      enumerate("abc")}), "group", "value",
                             correction="bonferroni")
        assert _names(t, "omnibus") == {"Kruskal-Wallis"}
        pairs = _rows(t, "pairwise")
        assert set(pairs["test_name"]) == {"Dunn"}
        assert set(pairs["correction"]) == {"bonferroni"}
        assert (pairs["p_adjusted"] >= pairs["p_value"]).all()

    def test_paired_two_groups(self):
        frame = _groups({"pre": _normal(0, n=30), "post": _normal(1, n=30)})
        frame["subject"] = list(range(30)) * 2
        t = _auto_statistics(frame, "group", "value", pair="subject")
        assert _names(t, "pairwise") == {"paired t"}
        skewed = _groups({"pre": _skewed(0, n=30), "post": _skewed(1, n=30)})
        skewed["subject"] = list(range(30)) * 2
        t = _auto_statistics(skewed, "group", "value", pair="subject")
        assert _names(t, "pairwise") == {"Wilcoxon signed-rank"}

    def test_repeated_measures_take_friedman(self):
        frame = _groups({k: _normal(i, n=30) for i, k in enumerate("abc")})
        frame["subject"] = list(range(30)) * 3
        t = _auto_statistics(frame, "group", "value", pair="subject")
        assert _names(t, "omnibus") == {"Friedman"}

    def test_two_measurements_take_pearson_or_spearman(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=60)
        t = _auto_statistics(pd.DataFrame({"x": x, "y": x + rng.normal(
            size=60)}), "x", "y")
        assert _names(t, "correlation") == {"Pearson"}
        x = rng.lognormal(size=60)
        t = _auto_statistics(pd.DataFrame({"x": x, "y": x ** 3}), "x", "y")
        assert _names(t, "correlation") == {"Spearman"}

    def test_counts_take_chi_square_or_fisher(self):
        rng = np.random.default_rng(3)
        big = pd.DataFrame({"a": rng.choice(list("XYZ"), 400),
                            "b": rng.choice(list("PQ"), 400)})
        t = _auto_statistics(big, "a", "b")
        assert _names(t, "contingency") == {"chi-square"}
        assert len(_rows(t, "pairwise")) == 3
        small = pd.DataFrame({"a": list("XXXXYYYY"), "b": list("PPPQQQQP")})
        assert _names(_auto_statistics(small, "a", "b"),
                      "contingency") == {"Fisher's exact"}

    def test_proportions_take_z_or_chi_square(self):
        rng = np.random.default_rng(4)
        two = pd.DataFrame({"g": np.repeat(list("ab"), 100),
                            "hit": (rng.random(200) < 0.4).astype(int)})
        assert _names(_auto_statistics(two, "g", "hit"),
                      "pairwise") == {"two-proportion z"}
        three = pd.DataFrame({"g": np.repeat(list("abc"), 100),
                              "hit": rng.random(300) < 0.4})
        t = _auto_statistics(three, "g", "hit")
        assert _names(t, "omnibus") == {"chi-square"}

    def test_the_user_can_override_and_it_is_recorded(self):
        t = _auto_statistics(_groups({"a": _normal(0), "b": _normal(1)}),
                             "group", "value", test="Mann-Whitney U")
        row = _rows(t, "pairwise").iloc[0]
        assert row["test_name"] == "Mann-Whitney U"
        assert row["chosen_by"] == "user"
        assert set(_rows(t, "normality")["chosen_by"]) == {"auto"}

    def test_the_table_has_every_column(self):
        t = _auto_statistics(_groups({"a": _normal(0), "b": _normal(1)}),
                             "group", "value")
        for column in ("test_stage", "test_name", "groups", "statistic",
                       "df", "p_value", "p_adjusted", "correction",
                       "effect_size", "n", "chosen_by", "reason"):
            assert column in t.columns

    def test_the_dialog_annotates_the_plot(self, qapp, registered):
        from spacr.qt.widgets.figure_settings import _StatisticsDialog

        dialog = _StatisticsDialog(registered)
        assert "one-way ANOVA" in dialog.report.toPlainText()
        dialog._apply()
        spec = registered._spacr_spec
        assert spec["stats"]["test"] is None
        assert spec["stats_note"].startswith("one-way ANOVA")
        assert spec["annotations"]
        assert any(t.get_gid() == "spacr-stats"
                   for t in registered.axes[0].texts)
        dialog.test.setCurrentIndex(dialog.test.findData("Kruskal-Wallis"))
        dialog._apply()
        assert registered._spacr_spec["stats"]["test"] == "Kruskal-Wallis"
        dialog.deleteLater()


class TestTheZip:

    def test_it_holds_everything_and_the_script_recreates_the_figure(
            self, qapp, registered, tmp_path):
        import matplotlib

        registered._spacr_spec["stats"] = {"test": "Kruskal-Wallis"}
        out = bundle._save_zip(registered, str(tmp_path / "fig"),
                               formats=["png", "pdf"], name="dose")
        assert out.endswith(".zip")
        names = set(zipfile.ZipFile(out).namelist())
        assert {"dose.png", "dose.pdf", "data.csv", "statistics.csv",
                "statistics.txt", "spec.json",
                "recreate_figure.py"} <= names
        folder = tmp_path / "unzipped"
        zipfile.ZipFile(out).extractall(folder)
        stats = pd.read_csv(folder / "statistics.csv")
        omnibus = stats[stats["test_stage"] == "omnibus"].iloc[0]
        assert omnibus["test_name"] == "Kruskal-Wallis"
        assert omnibus["chosen_by"] == "user"
        assert len(pd.read_csv(folder / "data.csv")) == 120
        spec = json.loads((folder / "spec.json").read_text())
        assert spec["kind"] == "box" and spec["x"] == "group"

        done = subprocess.run(
            [sys.executable, "recreate_figure.py"], cwd=folder,
            capture_output=True, text=True, timeout=240,
            env=dict(os.environ, MPLBACKEND="Agg"))
        assert done.returncode == 0, done.stderr[-2000:]
        recreated = matplotlib.image.imread(folder / "recreated.png")

        with matplotlib.rc_context(matplotlib.rcParamsDefault):
            figure = Figure()
            bundle._draw(figure, pd.read_csv(folder / "data.csv"), spec)
            figure.savefig(tmp_path / "here.png", dpi=spec["dpi"])
        here = matplotlib.image.imread(tmp_path / "here.png")
        assert recreated.shape == here.shape
        assert np.abs(recreated - here).mean() < 0.01

    def test_a_figure_with_no_data_still_saves(self, tmp_path):
        figure = Figure()
        figure.add_subplot(111).plot([1, 2, 3])
        out = bundle._save_zip(figure, str(tmp_path / "bare.zip"),
                               formats=["png"])
        names = set(zipfile.ZipFile(out).namelist())
        assert {"data.csv", "statistics.csv", "spec.json",
                "recreate_figure.py"} <= names
