"""Saved plaque diagnostics use numerical model outputs and the actual mask."""
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from cellpose import dynamics

from spacr import plaque, plaque_papers, submodules
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


def _prediction():
    labels = np.zeros((20, 28), dtype=np.uint16)
    labels[4:16, 4:14] = 3
    labels[4:16, 14:24] = 41
    dense = np.where(labels == 41, 2, labels > 0).astype(np.int32)
    flow = dynamics.masks_to_flows_gpu(dense, device=torch.device('cpu')) * 5
    prob = np.where(labels == 3, 2.0, -1.0).astype(np.float32)
    return labels, [None, flow, prob], None


def test_flow_error_detects_a_merge_and_preserves_original_ids():
    output = _prediction()
    correct = plaque._plaque_segmentation_metrics(output[0], output)
    merged = plaque._plaque_segmentation_metrics((output[0] > 0).astype(np.int32), output)
    assert set(correct) == {3, 41}
    assert correct[3]['cell_probability_mean'] == 2.0
    assert correct[41]['cell_probability_mean'] == -1.0
    for row in correct.values():
        assert row['flow_error'] == pytest.approx(0, abs=1e-12)
        assert row['flow_alignment_mean'] == pytest.approx(1.0)
        assert 0 < row['flow_magnitude_mean'] <= 1
        assert row['flow_pixel_fraction'] == 1
        assert row['cell_probability_pixel_fraction'] == 1
    assert merged[1]['flow_error'] > 0.05
    assert merged[1]['flow_alignment_mean'] < 0.95


@pytest.mark.parametrize('flows', [None, [], np.zeros((20, 28, 3)),
                                 [None, np.zeros((2, 10, 14)), np.zeros((10, 14))]])
def test_missing_or_wrong_grid_outputs_are_null(flows):
    labels = _prediction()[0]
    rows = plaque._plaque_segmentation_metrics(labels, (labels, flows, None))
    assert set(rows) == {3, 41}
    assert all(value is None for row in rows.values() for value in row.values())


def test_partial_nonfinite_outputs_report_coverage_without_false_flow_error():
    labels, flows, styles = _prediction()
    flows[2][4, 4] = np.nan
    flows[1][:, 4, 4] = np.inf
    row = plaque._plaque_segmentation_metrics(labels, (labels, flows, styles))[3]
    assert row['cell_probability_pixel_fraction'] == pytest.approx(119 / 120)
    assert row['flow_pixel_fraction'] == pytest.approx(119 / 120)
    assert row['cell_probability_mean'] == 2
    assert np.isfinite(row['flow_magnitude_mean'])
    assert row['flow_error'] is None
    assert row['flow_alignment_mean'] is None


def test_an_unreadable_flow_does_not_discard_a_readable_probability():
    class LostTensor:
        def __array__(self, *args, **kwargs):
            raise RuntimeError('device no longer exists')

    labels, flows, styles = _prediction()
    flows[1] = LostTensor()
    row = plaque._plaque_segmentation_metrics(labels, (labels, flows, styles))[3]
    assert row['cell_probability_mean'] == 2
    assert row['flow_error'] is None


def _run(tmp_path, monkeypatch, output=None):
    output = _prediction() if output is None else output
    tifffile.imwrite(tmp_path / 'well.tif', np.zeros(output[0].shape, dtype=np.uint8))
    calls = []

    class Model:
        def eval(self, image, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
            check_cellpose_eval_call(image, channel_axis,
                                     require_channel_axis=False)
            calls.append({"channel_axis": channel_axis, **kwargs})
            return output

    monkeypatch.setattr(submodules, '_resolve_plaque_model', lambda *args, **kwargs: 'model')
    monkeypatch.setattr(submodules, '_plaque_cellpose_model', lambda path: Model())
    monkeypatch.setattr('spacr.utils.save_settings', lambda *args, **kwargs: None)
    settings = {'src': str(tmp_path), 'masks': True}
    submodules.analyze_plaques(settings)
    return settings, calls


def _saved(tmp_path):
    with sqlite3.connect(tmp_path / 'masks' / 'plaques_analysis.db') as db:
        return pd.read_sql('SELECT * FROM per_plaque ORDER BY plaque_id', db)


def test_database_csv_and_mask_only_rerun_keep_same_diagnostics(tmp_path, monkeypatch):
    settings, calls = _run(tmp_path, monkeypatch)
    saved = _saved(tmp_path)
    assert saved['plaque_id'].tolist() == [3, 41]
    assert saved['cell_probability_mean'].tolist() == [2, -1]
    assert np.allclose(saved['flow_error'], 0)
    csv = pd.read_csv(tmp_path / 'masks' / 'per_plaque.csv')
    pd.testing.assert_frame_equal(csv.fillna(''), saved.fillna(''), check_dtype=False)
    with sqlite3.connect(tmp_path / 'masks' / 'plaques_analysis.db') as db:
        details = pd.read_sql('SELECT * FROM details ORDER BY plaque_id', db)
    assert details['cell_probability_mean'].tolist() == [2, -1]
    assert len(calls) == 1
    assert calls[0]["channel_axis"] is MISSING_CHANNEL_AXIS
    submodules.analyze_plaques({**settings, 'masks': False})
    pd.testing.assert_frame_equal(_saved(tmp_path), saved)
    assert len(calls) == 1


def test_changed_mask_rejects_stale_metrics(tmp_path, monkeypatch, caplog):
    settings, _ = _run(tmp_path, monkeypatch)
    mask_path = tmp_path / 'masks' / 'well.tif'
    labels = tifffile.imread(mask_path)
    labels[4, 4] = 0
    tifffile.imwrite(mask_path, labels)
    submodules.analyze_plaques({**settings, 'masks': False})
    saved = _saved(tmp_path)
    assert saved[list(plaque._PLAQUE_METRIC_COLUMNS)].isna().all().all()
    assert 'Ignoring stale plaque diagnostics' in caplog.text


def test_old_masks_without_diagnostics_have_null_columns(tmp_path, monkeypatch):
    settings, _ = _run(tmp_path, monkeypatch)
    (tmp_path / 'masks' / 'well.diagnostics.csv').unlink()
    submodules.analyze_plaques({**settings, 'masks': False})
    assert _saved(tmp_path)[list(plaque._PLAQUE_METRIC_COLUMNS)].isna().all().all()


def test_no_plaques_still_saves_empty_tables_with_metric_columns(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, (np.zeros((20, 28), np.uint16), [], None))
    saved = _saved(tmp_path)
    assert saved.empty
    assert set(plaque._PLAQUE_METRIC_COLUMNS) <= set(saved.columns)
    csv = pd.read_csv(tmp_path / 'masks' / 'per_plaque.csv')
    assert csv.empty
    assert set(plaque._PLAQUE_METRIC_COLUMNS) <= set(csv.columns)


def test_failed_flow_comparison_retains_probability_and_magnitude(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError('comparison unavailable')

    monkeypatch.setattr(dynamics, 'flow_error', broken)
    output = _prediction()
    row = plaque._plaque_segmentation_metrics(output[0], output)[3]
    assert row['cell_probability_mean'] == 2
    assert row['flow_magnitude_mean'] > 0
    assert row['flow_error'] is None
    assert row['flow_alignment_mean'] is None
    assert 'could not be calculated' in caplog.text


def test_figure_workflow_saves_metrics_and_upgrades_old_database(tmp_path, monkeypatch):
    from PIL import Image
    import cellpose.models

    output = _prediction()
    Image.fromarray(np.zeros((*output[0].shape, 3), np.uint8)).save(tmp_path / 'figure.png')

    class Model:
        def __init__(self, **kwargs):
            pass

        def eval(self, image, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
            check_cellpose_eval_call(image, channel_axis,
                                     require_channel_axis=False)
            return output

    monkeypatch.setattr(cellpose.models, 'CellposeModel', Model)
    segment = plaque_papers._cellpose_segmenter('model', return_metrics=True)
    result = plaque_papers.measure_figure_folder(
        tmp_path, tmp_path / 'out', detector='unused', segmenter='unused',
        detect=lambda *args, **kwargs: [plaque.Well(0, 0, 28, 20)],
        read_text=lambda path: [], segment=segment)
    with sqlite3.connect(result['database']) as db:
        rows = pd.read_sql('SELECT * FROM plaques ORDER BY label', db)
    assert rows['label'].tolist() == [3, 41]
    assert rows['cell_probability_mean'].tolist() == [2, -1]
    assert np.allclose(rows['flow_error'], 0)
    old = tmp_path / 'old.db'
    with sqlite3.connect(old) as db:
        db.execute('CREATE TABLE plaques (region_id INTEGER, label INTEGER, area_px INTEGER)')
        db.execute('INSERT INTO plaques VALUES (5, 2, 100)')
    db = plaque_papers.open_database(old)
    row = db.execute('SELECT area_px, cell_probability_mean, flow_error FROM plaques').fetchone()
    db.close()
    assert row == (100, None, None)
