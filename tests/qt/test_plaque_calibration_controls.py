"""Editable per-well scale and time preserve the annotation under sorting."""
import numpy as np
from PySide6.QtCore import Qt
from spacr import plaque_papers as pp
from spacr.qt.widgets import plaque_preview as pv


def test_sorted_edit_changes_correct_well_and_clearing_restores_ruler(qtbot, monkeypatch):
    monkeypatch.setattr(pv, 'missing_papers_packages', lambda *a, **k: [])
    panel = pv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._annotations = [pp.Annotation(pp.Region(0,0,100,100)), pp.Annotation(pp.Region(120,0,220,100))]
    panel._automatic_scales = [pp._Scale(10, 'well'), pp._Scale(20, 'well')]
    panel._refresh_calibration_scales()
    panel._wells = {0: dict(count=1, mean_area=100, rows=[dict(area_px=100)])}
    panel._fill_table()
    panel._table.sortItems(0, Qt.DescendingOrder)
    row = panel._view_row(0)
    panel._table.item(row, pv.PIXELS_PER_UM_COLUMN).setText('0.05')
    assert panel._annotations[0].pixels_per_um == .05
    assert panel._annotations[1].pixels_per_um is None
    assert panel.plaque_table_rows()[0]['area_mm2'] == .04
    panel._table.item(panel._view_row(0), pv.FORMATION_HOURS_COLUMN).setText('168')
    assert panel.plaque_table_rows()[0]['formation_hours'] == 168
    panel._table.item(panel._view_row(0), pv.PIXELS_PER_UM_COLUMN).setText('nan')
    assert panel._annotations[0].pixels_per_um == .05
    panel._table.item(panel._view_row(0), pv.PIXELS_PER_UM_COLUMN).setText('')
    assert panel._annotations[0].pixels_per_um is None
    assert panel.plaque_table_rows()[0]['area_mm2'] == 1


def test_estimates_are_opt_in_saved_separately_and_recomputed(qtbot, monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(pv, 'missing_papers_packages', lambda *a, **k: [])
    panel = pv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.set_mode(pv.FIGURE_MODE)
    panel._figure = dict(path=str(tmp_path / 'figure.png'), regions=[pp.Region(0,0,100,100)])
    panel._annotations = [pp.Annotation(pp.Region(0,0,100,100))]
    panel._scales = [pp._Scale()]
    panel._wells = {0: dict(count=8, mean_area=100, rows=[dict(area_px=100)] * 8)}
    panel._fill_table()
    assert 'estimated_formation_hours' not in panel.table_rows()[0]
    panel._growth_btn.click()
    row = panel.table_rows()[0]
    assert row['estimated_formation_hours'] == 168
    assert row['formation_hours'] is None
    assert row['pixels_per_um'] is None
    assert json.loads(row['growth_estimate_provenance'])['experimental']
    panel._table.item(panel._view_row(0), pv.FORMATION_HOURS_COLUMN).setText('96')
    qtbot.wait(1)
    assert panel.table_rows()[0]['estimated_formation_hours'] is None
    assert panel.settings_for_propagation()['plaque_estimate_growth'] is True
    panel._growth_btn.click()
    assert 'estimated_formation_hours' not in panel.table_rows()[0]
