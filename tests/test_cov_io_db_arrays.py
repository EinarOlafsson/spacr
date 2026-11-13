"""Coverage for the SQLite / array-merging helpers in ``spacr.io``.

Targets ``_read_and_join_tables``, ``_save_settings_to_db``,
``_save_mask_timelapse_as_gif``, ``_save_object_counts_to_database``,
``_create_database``, ``save_object_mask``, ``_load_and_concatenate_arrays``
and ``_results_to_csv`` — in particular the branches that only fire when a
table is missing, a mask has no background label, tracks are overlaid on a
timelapse, or pathogen/organelle mask folders take part in the merge.

Everything here is CPU-only, offline and writes exclusively into ``tmp_path``.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


# ---------------------------------------------------------------------------
# _read_and_join_tables
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _save_settings_to_db
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _save_mask_timelapse_as_gif
# ---------------------------------------------------------------------------

def test_save_mask_timelapse_as_gif_overlays_tracks(tmp_path):
    """A non-None tracks_df draws one white polyline per track_id.

    The track-overlay branch inside ``_update`` is only reachable when
    ``tracks_df`` is not None, so this drives the animation with a real
    two-track frame table and asserts the GIF was written.
    """
    import matplotlib.colors as mcolors
    from PIL import Image
    from spacr.io import _save_mask_timelapse_as_gif

    masks = []
    for t in range(2):
        m = np.zeros((12, 12), np.uint16)
        m[2 + t:5 + t, 2:5] = 1
        m[7:10, 7:10] = 2
        masks.append(m)

    tracks_df = pd.DataFrame({
        'track_id': [1, 1, 2, 2],
        'x': [3.0, 4.0, 8.0, 8.0],
        'y': [3.0, 4.0, 8.0, 9.0],
    })
    out = tmp_path / 'tracks.gif'

    _save_mask_timelapse_as_gif(masks, tracks_df, str(out), cmap='viridis',
                                norm=mcolors.Normalize(vmin=0, vmax=2),
                                filenames=['frame0.npy', 'frame1.npy'])

    assert out.is_file() and out.stat().st_size > 0
    with Image.open(str(out)) as im:
        assert im.format == 'GIF'
        assert im.n_frames == len(masks)


# ---------------------------------------------------------------------------
# _save_object_counts_to_database
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _create_database
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# save_object_mask
# ---------------------------------------------------------------------------

def test_save_object_mask_uncompressed_preserves_labels(tmp_path):
    """compression='none' maps to no codec and labels survive verbatim."""
    from spacr.io import save_object_mask, _load_array_any

    mask = np.zeros((8, 8), np.int32)
    mask[1:4, 1:4] = 9
    mask[5:7, 5:7] = 65535
    out = save_object_mask(str(tmp_path), 'plate1_A01_f1.npy', mask,
                           compression='none')
    assert out == str(tmp_path / 'plate1_A01_f1.tif')
    back = _load_array_any(out)
    assert back.dtype == np.uint16
    assert np.array_equal(back, mask.astype(np.uint16))


# ---------------------------------------------------------------------------
# _load_and_concatenate_arrays
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _results_to_csv
# ---------------------------------------------------------------------------

def test_results_to_csv_writes_both_frames_with_index(tmp_path):
    """cells.csv / wells.csv are written under results/ and returned as-is."""
    from spacr.io import _results_to_csv

    cells = pd.DataFrame({'cell_area': [1.0, 2.0]}, index=['o1', 'o2'])
    wells = pd.DataFrame({'mean_area': [1.5]}, index=['plate1_A01'])

    out_cells, out_wells = _results_to_csv(str(tmp_path), cells, wells)

    assert out_cells is cells and out_wells is wells
    cells_csv = tmp_path / 'results' / 'cells.csv'
    wells_csv = tmp_path / 'results' / 'wells.csv'
    assert cells_csv.is_file() and wells_csv.is_file()

    back = pd.read_csv(cells_csv, index_col=0)
    assert list(back.index) == ['o1', 'o2']
    assert back['cell_area'].tolist() == [1.0, 2.0]
    back_wells = pd.read_csv(wells_csv, index_col=0)
    assert list(back_wells.index) == ['plate1_A01']
    assert back_wells['mean_area'].tolist() == [1.5]
