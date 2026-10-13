"""Molecular structures remain separate when a real canvas changes shape."""
from types import SimpleNamespace

import pandas as pd
import pytest


def test_structure_tiles_reflow_after_canvas_resize():
    pytest.importorskip("rdkit")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from spacr.sp_stats import _draw_hit_structures

    sar = pd.DataFrame({
        "compound": [
            "AR-12", "SU3327", "TG-101348", "NSC-632839", "puromycin",
            "ryuvidine", "AZD7762", "LDN-212854", "romidepsin", "UNC1999",
            "thiostrepton", "buparlisib", "anisomycin", "ponatinib",
            "NSC-663284", "BVT-948", "UNC2025", "delanzomib", "CYT-997",
            "oxibendazole", "homoharringtonine", "pyrrolidine-dithiocarbamate",
            "NVP-HSP990", "TG-02",
        ],
        "smiles": ["CC(=O)Oc1ccccc1C(=O)O"] * 24,
        "hit": [True] * 24, "smiles_valid": [True] * 24,
        "cluster": range(1, 25), "best_rank": range(1, 25),
        "potency": [-10.123] * 24,
    })
    sar.attrs["statistic"] = "robust_z"
    chemistry = SimpleNamespace(sar=sar, clustered=True)
    figure = Figure(figsize=(8, 3), dpi=100)
    canvas = FigureCanvasAgg(figure)
    assert _draw_hit_structures(figure, chemistry) == 24
    for width, height in [(8.6, 8), (12, 9.2), (10, 6)]:
        figure.set_size_inches(width, height)
        _draw_hit_structures(figure, chemistry)
        canvas.draw()
        axes = figure.axes
        for row in range(4):
            for column in range(5):
                left, right = axes[row * 6 + column:row * 6 + column + 2]
                assert left.get_window_extent().x1 <= right.get_window_extent().x0
        for column in range(6):
            for row in range(3):
                above, below = axes[row * 6 + column], axes[(row + 1) * 6 + column]
                assert below.get_window_extent().y1 <= above.get_window_extent().y0
        for axis in axes:
            image = axis.images[0]
            assert image.get_array().shape[:2] == (300, 300)
            assert axis.get_xlim() == (-.5, 299.5)
            assert axis.get_ylim() == (299.5, -.5)


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_structure_labels_follow_theme_and_preserve_explicit_ink(
        theme, monkeypatch, tmp_path):
    pytest.importorskip('rdkit')
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from PySide6.QtCore import QSettings

    from spacr.figures.style import INK_PRINT
    from spacr.qt import preferences
    from spacr.sp_stats import _draw_hit_structures

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'figure.ini'), QSettings.IniFormat))
    preferences.set_theme(theme)
    preferences.set_figure_colors_auto()
    chemistry = SimpleNamespace(clustered=True, sar=pd.DataFrame({
        'compound': ['aspirin'], 'smiles': ['CC(=O)Oc1ccccc1C(=O)O'],
        'hit': [True], 'smiles_valid': [True], 'cluster': [1],
        'best_rank': [1], 'potency': [-4.5],
    }))
    figure = Figure()
    _draw_hit_structures(figure, chemistry)
    wanted = '#000000' if theme == 'light' else '#ffffff'
    assert to_rgba(figure.axes[0].title.get_color()) == to_rgba(wanted)
    assert to_rgba(figure.axes[0].xaxis.label.get_color()) == to_rgba(wanted)
    _draw_hit_structures(figure, chemistry, target='print')
    assert to_rgba(figure.axes[0].title.get_color()) == to_rgba(INK_PRINT)
    preferences.set_figure_colors('auto', '#123456')
    for target in ['screen', 'print']:
        _draw_hit_structures(figure, chemistry, target=target)
        assert to_rgba(figure.axes[0].title.get_color()) == to_rgba('#123456')
