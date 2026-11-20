"""Figures use the same face whichever way spaCR is started.

The decode tuning curve pins its title at 9 pt and saves at 150 DPI, so a
title that looks a different size between ``spacr-run ops`` and a Python
driver can only come from a different face: the bundled Open Sans silently
falling back to another sans-serif. These tests pin the face, check that the
fonts ship, and check that a missing font is now said out loud.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=False)

from spacr import figure_font  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json, sys
import matplotlib
matplotlib.use("Agg")
{entry}
from matplotlib.figure import Figure
from spacr.figures.style import figure_style
with figure_style("print"):
    figure = Figure(figsize=(5.0, 3.8), dpi=150)
    title = figure.subplots().set_title("x", fontsize=9)
    figure.canvas.draw()
    print(json.dumps({{"face": title.get_fontname(), "size": title.get_size(),
                      "pixels": list(figure.canvas.get_width_height())}}))
"""


def _probe(entry):
    """Render the probe figure in a fresh interpreter after ``entry`` runs."""
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(entry=entry)],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT),
             "CUDA_VISIBLE_DEVICES": "", "QT_QPA_PLATFORM": "offscreen"})
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_bundled_faces_ship_and_resolve():
    assert figure_font.bundled_faces(), figure_font.font_dir()
    path = figure_font.require_figure_font()
    assert "OpenSans" in Path(path).name


def test_the_fonts_are_package_data():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "resources/font/**/*" in setup
    assert "recursive-include spacr/resources/font" in manifest


def test_the_driver_and_the_console_entry_point_draw_the_same_title():
    driver = _probe("import spacr")
    console = _probe("import spacr.cli")
    assert driver == console
    assert driver["face"] == figure_font.FAMILY
    assert driver["size"] == 9


def test_a_missing_font_is_logged_not_silent(monkeypatch, tmp_path, caplog):
    fake = SimpleNamespace(
        fontManager=SimpleNamespace(ttflist=[], addfont=lambda path: None))
    monkeypatch.setattr(figure_font, "_registered", False)
    monkeypatch.setattr(figure_font, "_resolved", False)
    monkeypatch.setattr(figure_font, "_figure_font_manager", lambda: fake)
    monkeypatch.setattr(figure_font, "font_dir", lambda: str(tmp_path))

    with caplog.at_level("WARNING", logger="spacr.figure_font"):
        assert figure_font.use_open_sans_for_figures() is False
    assert "Figures will not use Open Sans" in caplog.text
    assert str(tmp_path) in caplog.text

    with pytest.raises(RuntimeError, match="ships no Open Sans"):
        figure_font.require_figure_font()


def test_an_unresolved_open_sans_is_left_out_of_the_family_list(monkeypatch):
    import spacr.figure_font as module
    from spacr.figures import style

    monkeypatch.setattr(module, "use_open_sans_for_figures", lambda: False)
    assert figure_font.FAMILY not in style.rc("print")["font.sans-serif"]
    monkeypatch.setattr(module, "use_open_sans_for_figures", lambda: True)
    assert style.rc("print")["font.sans-serif"][0] == figure_font.FAMILY
