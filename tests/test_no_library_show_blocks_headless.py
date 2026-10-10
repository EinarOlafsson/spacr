"""No spaCR library function may block on ``plt.show()`` headless.

A bare ``plt.show()`` under QtAgg with ``QT_QPA_PLATFORM=offscreen`` enters a
Qt event loop nothing can end, so a scripted run sits on its first figure.
Every library show goes through ``spacr.figure_sink._show_if_interactive``,
which still shows in a notebook, an interactive prompt and the GUI.

The guard is parsed, not grepped: docstrings across spacr/ mention
``plt.show()`` in prose, and a text search would trip on them.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from spacr import figure_sink

SPACR = Path(__file__).resolve().parents[1] / "spacr"
HELPER = SPACR / "figure_sink.py"


def _pyplot_aliases(tree):
    """Names bound to ``matplotlib.pyplot`` and to its ``show`` in a module."""
    modules, shows = {"plt", "pyplot"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib.pyplot":
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module == "matplotlib" and alias.name == "pyplot":
                    modules.add(alias.asname or "pyplot")
                if node.module == "matplotlib.pyplot" and alias.name == "show":
                    shows.add(alias.asname or "show")
    return modules, shows


def _bare_shows(path):
    """``(line, source)`` of every direct pyplot show call in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules, shows = _pyplot_aliases(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "show":
            owner = ast.unparse(func.value)
            if owner in modules or owner == "matplotlib.pyplot":
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(func, ast.Name) and func.id in shows:
            found.append((node.lineno, ast.unparse(node)))
    return found


def test_no_module_in_spacr_calls_pyplot_show_directly():
    offenders = []
    for path in sorted(SPACR.rglob("*.py")):
        if path == HELPER:
            continue
        for line, source in _bare_shows(path):
            offenders.append(f"{path.relative_to(SPACR.parent)}:{line}: {source}")
    assert not offenders, (
        "call spacr.figure_sink._show_if_interactive() instead of plt.show(), "
        "which hangs headless QtAgg runs:\n" + "\n".join(offenders))


def test_the_helper_is_the_one_place_that_shows():
    assert [source for _, source in _bare_shows(HELPER)] == [
        "pyplot.show(**kwargs)"]


def test_the_scan_sees_every_spelling(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(textwrap.dedent('''
        """plt.show() in prose is fine."""
        import matplotlib
        import matplotlib.pyplot as mp
        from matplotlib import pyplot as p2
        from matplotlib.pyplot import show as s
        mp.show()
        p2.show(block=False)
        matplotlib.pyplot.show()
        s()
        widget.show()
    '''), encoding="utf-8")
    assert [line for line, _ in _bare_shows(sample)] == [7, 8, 9, 10]


def test_kwargs_reach_a_captured_show(monkeypatch):
    seen = []
    monkeypatch.setattr(plt, "show", lambda **kwargs: seen.append(kwargs))
    assert figure_sink._show_if_interactive(block=False) is True
    assert seen == [{"block": False}]


def test_agg_skips_without_touching_show(monkeypatch):
    def show(*_args, **_kwargs):
        raise AssertionError("shown on Agg")
    show.__module__ = "matplotlib.pyplot"
    monkeypatch.setattr(plt, "show", show)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    assert figure_sink._show_if_interactive() is False


def test_representative_functions_return_under_offscreen_qtagg(tmp_path):
    """Real QtAgg, offscreen Qt, no MPLBACKEND=Agg, in a child with a timeout."""
    pytest.importorskip("PySide6")
    script = textwrap.dedent("""
        import sys
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        dst = sys.argv[1]
        calls = []
        real_show = plt.show

        from spacr import plot, sim, toxo
        from spacr.figure_sink import _can_show

        masks = np.zeros((16, 16), dtype=int)
        masks[2:6, 2:6] = 1
        masks[8:12, 8:14] = 2
        plot.visualize_masks(masks, masks, masks)
        plt.close("all")

        frame = pd.DataFrame({"value": np.linspace(0.0, 1.0, 40)})
        plot.plot_histogram(frame, "value", dst=dst)
        plt.close("all")

        genes = pd.DataFrame({"Gene ID": ["TGGT1_1", "TGGT1_2"],
                              "a": [1.0, 2.0], "b": [3.0, 0.5]})
        toxo.plot_gene_heatmaps(genes, ["1", "2"], ["a", "b"])
        plt.close("all")

        rng = np.random.default_rng(0)
        sweep = pd.DataFrame({
            "number_of_active_genes": rng.random(12),
            "avg_reads_per_gene": rng.random(12),
            "prauc": rng.random(12),
            "roc_auc": rng.random(12),
        })
        sim.plot_correlation_matrix(sweep, dst=dst)
        plt.close("all")

        assert plt.show is real_show
        print("backend", matplotlib.get_backend().lower(), "can_show", _can_show())
        print("returned")
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="")
    env.pop("MPLBACKEND", None)
    # As on CI: no display server, so Matplotlib accepts QtAgg only once a
    # QApplication exists, which is how spaCR's own figures run.
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    lines = done.stdout.strip().splitlines()
    assert lines[-1] == "returned"
    assert lines[-2] == "backend qtagg can_show False"
    assert (tmp_path / "value_histogram.pdf").exists()
