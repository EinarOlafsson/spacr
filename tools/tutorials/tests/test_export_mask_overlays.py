"""Exercise actual plotting and positively establish every refusal premise."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_mask_overlays import LAYOUT, check_figure, source_files


@pytest.fixture
def plotted():
    from spacr.plot import plot_image_mask_overlay
    # Use pytest's temporary path indirectly through the actual API fixture below.
    return plot_image_mask_overlay


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / 'merged'; folder.mkdir()
    (folder / '.spacr_plane_layout.json').write_text(json.dumps(LAYOUT))
    array = np.zeros((24, 30, 7), dtype=np.uint16)
    for ch in range(4):
        array[..., ch] = np.arange(24 * 30, dtype=np.uint16).reshape(24, 30) * (ch + 1)
    array[7:16, 7:20, 4] = 1
    array[9:13, 10:15, 5] = 1
    array[12:15, 16:19, 6] = 1
    np.save(folder / 'field.npy', array)
    return folder, array


def make_figure(plotted, source):
    folder, array = source
    fig = plotted(str(folder / 'field.npy'), [0, 1, 2, 3], 1, 0, 2,
                  figuresize=2, percentiles=(1, 99), thickness=3, save_pdf=False)
    check_figure(array, fig)
    return fig


def test_real_api_matches_declared_channels_and_masks(plotted, source):
    fig = make_figure(plotted, source)
    try:
        result = check_figure(source[1], fig)
        assert result['channel_rgb_values_checked'] == 24 * 30 * 4 * 3
        assert result['counts'] == {'cell': 1, 'nucleus': 1, 'pathogen': 1}
        assert result['segmentation_accuracy_certified'] is False
    finally: plt.close(fig)


@pytest.mark.parametrize('kind', ['channel_pixel', 'channel_title', 'combined_background'])
def test_wrong_display_rejected_after_actual_api_positive(plotted, source, kind):
    fig = make_figure(plotted, source)
    try:
        if kind == 'channel_title':
            fig.axes[0].set_title('cell (channel 0)')
        else:
            index = 4 if kind == 'combined_background' else 0
            values = np.asarray(fig.axes[index].images[0].get_array()).copy()
            values[0, 0] = [.7, .2, .1]
            fig.axes[index].images[0].set_data(values)
        with pytest.raises(ValueError): check_figure(source[1], fig)
    finally: plt.close(fig)


def test_metadata_is_not_a_numeric_image(source):
    folder, _ = source
    marker, paths = source_files(folder)
    assert marker.name == '.spacr_plane_layout.json'
    assert [p.name for p in paths] == ['field.npy']
    (folder / 'notes.json').write_text('{"not":"an image"}')
    assert source_files(folder)[1] == paths


def test_wrong_layout_rejected_after_positive(source):
    folder, _ = source
    source_files(folder)
    wrong = dict(LAYOUT, mask_plane_order=['nucleus', 'cell', 'pathogen'])
    (folder / '.spacr_plane_layout.json').write_text(json.dumps(wrong))
    with pytest.raises(ValueError): source_files(folder)


def test_existing_array_link_rejected_after_positive(source):
    folder, _ = source
    source_files(folder)
    (folder / 'duplicate.npy').symlink_to(folder / 'field.npy')
    assert (folder / 'duplicate.npy').is_file()
    with pytest.raises(ValueError): source_files(folder)
