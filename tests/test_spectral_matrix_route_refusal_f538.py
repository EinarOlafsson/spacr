"""Saved direct-matrix options refuse unsupported routes before work or output."""

from copy import deepcopy
import json

import numpy as np
import pytest


REFUSAL = (
    'A supplied spectral matrix requires calibrated organized 2D cell V2 segmentation, '
    'without control wells or per-plate estimation; this processing path is not supported.'
)


@pytest.fixture
def saved_matrix_source(tmp_path):
    """Persist small real images, masks and a saved matrix option for refusal flows."""
    import tifffile

    source = tmp_path / 'plate1'
    merged = source / 'merged'
    merged.mkdir(parents=True)
    first = np.zeros((24, 24), dtype=np.float32)
    second = np.zeros_like(first)
    first[3:9, 3:9] = 120
    second[14:20, 14:20] = 80
    mask = np.zeros(first.shape, dtype=np.uint16)
    mask[2:10, 2:10] = 1
    mask[13:21, 13:21] = 2
    camera = np.array([[1., .25], [.5, 1.]])
    paths = {}
    for well, dyes in [('A01', (first, first * 0)),
                       ('B01', (second * 0, second)),
                       ('C01', (first, second))]:
        readings = np.stack(dyes, axis=-1) @ camera.T + 10
        path = merged / f'plate1_{well}_1.npy'
        np.save(path, np.dstack((readings, mask)).astype(np.uint16))
        paths[well] = path
    tifffile.imwrite(source / 'plate1_C01_1_0.tif', first.astype(np.uint16))
    native = tmp_path / 'native_field'
    native.mkdir()
    pixels = native / 'channel.npy'
    labels = native / 'cells.npy'
    np.save(pixels, first)
    np.save(labels, mask)
    calibration = tmp_path / 'measured_calibration.json'
    calibration.write_text(json.dumps(dict(
        version=1, axes='YX', pixel_size_um=[1., 1.], reference_channel=0,
        channels=[dict(channel=channel, shift_to_reference_px=[0, 0], psf=[[1.]],
                       psf_origin_px=[0, 0], psf_pixel_size_um=[1., 1.])
                  for channel in [0, 1]])))
    matrix_file = tmp_path / 'saved_matrix.json'
    matrix_file.write_text(json.dumps({'matrix': camera.tolist(), 'channels': [0, 1]}))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    original_paths = set(tmp_path.rglob('*'))
    return dict(source=source, fields=paths, pixels=pixels, labels=labels,
                matrix_file=matrix_file, calibration=calibration,
                before=before, original_paths=original_paths)


def _unchanged(sample, root):
    """Check exact persisted bytes and absence of new source or output paths."""
    assert set(root.rglob('*')) == sample['original_paths']
    assert all(path.read_bytes() == content for path, content in sample['before'].items())


@pytest.mark.parametrize('entry', ['v1', 'watch', 'cloud', 'timelapse', 'direct-watch', 'volume'])
def test_saved_matrix_refuses_unsupported_segmentation_before_dispatch(
        saved_matrix_source, tmp_path, monkeypatch, entry):
    """Reject the saved path before dry-run, acquisition or native volume branches."""
    import spacr.core as core
    import spacr.io as io
    import spacr.ome_zarr as cloud
    import spacr.validate as validate

    sample = saved_matrix_source
    settings = dict(src=str(sample['source']), unmix=True,
                    unmix_matrix_file=str(sample['matrix_file']), unmix_calibration='')

    def forbidden(*args, **kwargs):
        """Fail if an unsupported request reaches preflight or dispatch."""
        pytest.fail('saved direct matrix reached unsupported segmentation work')

    monkeypatch.setattr(validate, 'run_preflight', forbidden)
    monkeypatch.setattr(cloud, '_run_with_cloud_sources', forbidden)
    monkeypatch.setattr(core, '_watch_vendor_recipe', forbidden)
    monkeypatch.setattr(io, '_preprocess_mapped_volume_series', forbidden)
    if entry == 'v1':
        settings.update(pipeline_style='v1', dry_run=True, unmix=False)
        call = core.preprocess_generate_masks
    elif entry == 'watch':
        settings.update(pipeline_style='v2', watch_folder=True)
        call = core.preprocess_generate_masks
    elif entry == 'cloud':
        settings.update(pipeline_style='v2', src='s3://example-bucket/plate1',
                        unmix_calibration=str(sample['calibration']),
                        unmix_pixel_size_um=[1., 1.])
        call = core.preprocess_generate_masks
    elif entry == 'timelapse':
        settings.update(pipeline_style='v2', unmix=False)
        call = core.preprocess_generate_masks_timelapse
    elif entry == 'direct-watch':
        settings.update(unmix=False)
        call = core._watch_folder_and_analyse
    else:
        settings.update(z_stack=True, t_stack=True, z_segmentation_mode='volumetric')
        call = io.preprocess_img_data
    with pytest.raises(ValueError) as error:
        call(settings)
    assert str(error.value) == REFUSAL
    _unchanged(sample, tmp_path)


@pytest.mark.parametrize('entry', ['measure', 'field-table'])
def test_saved_matrix_refuses_ordinary_measure_before_outputs(
        saved_matrix_source, tmp_path, monkeypatch, entry):
    """Refuse hidden saved options before normal Measure stages any output."""
    import spacr.measure as measure
    import spacr.validate as validate

    sample = saved_matrix_source
    table = measure.FieldTable(rows=[measure.FieldRow(
        'field', channels={0: str(sample['pixels'])},
        masks={'cell': str(sample['labels'])})])
    settings = dict(src=str(sample['source']), unmix=False, unmix_calibration='',
                    unmix_matrix_file=str(sample['matrix_file']), dry_run=True)

    def forbidden(*args, **kwargs):
        """Fail if refusal becomes preflight or a hand-picked-array write."""
        pytest.fail('saved direct matrix reached ordinary Measure work')

    monkeypatch.setattr(validate, 'run_preflight', forbidden)
    monkeypatch.setattr(measure, 'write_field_table_project', forbidden)
    with pytest.raises(ValueError) as error:
        if entry == 'measure':
            measure.measure_crop(settings)
        else:
            measure.measure_from_field_table(table, settings, dst=str(tmp_path / 'features'))
    assert str(error.value) == REFUSAL
    _unchanged(sample, tmp_path)


def test_saved_matrix_live_preview_refuses_before_model_or_worker(
        saved_matrix_source, tmp_path, qtbot, monkeypatch):
    """The actual panel retains the hidden option and blocks request dispatch."""
    from spacr.qt.widgets import live_preview as lp

    sample = saved_matrix_source
    panel = lp.LivePreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    try:
        panel.apply_settings(dict(unmix=False, unmix_calibration='',
                                  unmix_matrix_file=str(sample['matrix_file']),
                                  cell_channel=0, organelle_channel=1))
        assert panel.load_image(sample['fields']['C01']), panel._status.text()
        raw = panel._image.copy()

        def forbidden(*args, **kwargs):
            """Fail before model resolution or creation of the preview worker."""
            pytest.fail('saved direct matrix reached live preview inference')

        monkeypatch.setattr(panel, '_model_for_this_pass', forbidden)
        monkeypatch.setattr(lp, '_PreviewWorker', forbidden)
        panel.run_preview()
        assert REFUSAL in panel._status.text()
        assert panel._settings['unmix_matrix_file'] == str(sample['matrix_file'])
        np.testing.assert_array_equal(panel._image, raw)
        _unchanged(sample, tmp_path)
    finally:
        panel.shutdown()


@pytest.mark.parametrize('checked', [False, True], ids=['raw-crops', 'unmixed-crops'])
def test_saved_matrix_measure_preview_invalidates_then_refuses_work_and_export(
        saved_matrix_source, tmp_path, qtbot, monkeypatch, checked):
    """A saved option clears real prior crops even when the unmix toggle is off."""
    from spacr.qt.widgets import measure_preview as mp

    sample = saved_matrix_source
    monkeypatch.setattr('spacr.qt.preferences._is_alpha_visible', lambda *args: True)
    panel = mp.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    try:
        panel.apply_settings(dict(
            unmix=True, unmix_controls='0:A01; 1:B01', unmix_background_percentile=5.,
            channels=[0, 1], cell_mask_dim=2, nucleus_mask_dim=None,
            pathogen_mask_dim=None, organelle_mask_dim=None,
            png_channel_mapping={'r': 0, 'g': 1, 'b': 0}, png_size=[16, 16],
            cell_min_size=1))
        for role, control in panel._mask_dims.items():
            control.setValue(2 if role == 'cell' else -1)
        panel.load_array(str(sample['fields']['C01']))
        qtbot.waitUntil(lambda: not panel._crop_running, timeout=10000)
        panel._unmix_btn.setChecked(checked)
        qtbot.waitUntil(lambda: not panel._crop_running, timeout=10000)
        assert panel._crops
        raw = panel._data.copy()
        crops = deepcopy(panel._crops)
        if checked:
            assert all('unmixing' in crop for crop in crops)
        else:
            assert all('unmixing' not in crop for crop in crops)

        def forbidden(*args, **kwargs):
            """Fail on a crop/export job or destination dialog after the saved edit."""
            pytest.fail('saved direct matrix reached Measure preview work')

        monkeypatch.setattr(panel._jobs, 'submit', forbidden)
        monkeypatch.setattr(mp.QFileDialog, 'getExistingDirectory', forbidden)
        panel.apply_settings({'unmix_matrix_file': str(sample['matrix_file'])})
        assert panel._unmix_settings['unmix_matrix_file'] == str(sample['matrix_file'])
        panel.run_preview()
        destination = tmp_path / 'unsupported-crops'
        panel._export_displayed_crops()
        panel._start_crop_export(destination)
        assert REFUSAL in panel._status.text()
        assert not panel._crops and not destination.exists()
        np.testing.assert_array_equal(panel._data, raw)
        _unchanged(sample, tmp_path)
    finally:
        panel.shutdown()
