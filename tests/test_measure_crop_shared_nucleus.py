"""A nucleus label that spans two cells keeps its field and is flagged."""
import numpy as np
import pytest

from spacr.database_concurrency import connect


def _stack(split):
    """Two cells side by side and one nucleus label reaching into both.

    With ``split`` the nucleus is two disconnected pieces of one label, a
    large one in the left cell and a small one in the right cell, which the
    overlap merge cannot repair. Without it the nucleus is one blob that
    overlaps the left cell 71 percent and the right cell 29 percent, which
    the overlap merge folds into one cell as before.
    """
    cell = np.zeros((40, 40), np.uint16)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    cell[4:36, 2:19] = 1
    cell[4:36, 21:38] = 2
    if split:
        nucleus[10:20, 6:16] = 1
        nucleus[10:14, 24:28] = 1
    else:
        nucleus[10:20, 14:23] = 1
    nucleus[24:30, 26:32] = 2
    pathogen[26:28, 28:30] = 1
    rng = np.random.default_rng(0)
    planes = [rng.integers(50, 200, size=(40, 40)).astype(np.uint16)
              for _ in range(2)]
    return np.stack(planes + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


def _measure(tmp_path, split, **extra):
    from spacr.measure import measure_crop
    root = tmp_path / 'plate01'
    (root / 'merged').mkdir(parents=True)
    np.save(root / 'merged' / 'plate01_A01_1.npy', _stack(split))
    settings = {
        'src': str(root), 'cell_mask_dim': 2, 'nucleus_mask_dim': 3,
        'pathogen_mask_dim': 4, 'channels': [0, 1], 'crop_mode': ['cell'],
        'save_png': False, 'n_jobs': 1, 'cell_min_size': 0,
        'nucleus_min_size': 0, 'pathogen_min_size': 0, 'plot': False,
        'normalize': [1, 99], 'normalize_by': 'png', 'timelapse': False,
    }
    settings.update(extra)
    measure_crop(settings)
    return root / 'measurements' / 'measurements.db'


def _table(db, name):
    from spacr.tabular import _read_query
    conn = connect(str(db), readonly=True)
    try:
        return _read_query(conn, f'SELECT * FROM {name}', canonicalise=False)
    finally:
        conn.close()


def test_a_nucleus_in_two_cells_keeps_its_field_and_goes_to_the_larger_overlap(tmp_path, capsys):
    db = _measure(tmp_path, split=True)
    nuclei = _table(db, 'nucleus')
    cells = _table(db, 'cell')
    assert len(nuclei) == 2
    shared = nuclei[nuclei['nucleus_shared'].astype(bool)]
    assert len(shared) == 1
    assert len(cells) == 2
    assert cells['nucleus_shared'].astype(bool).sum() == 2
    assert 'shared by two cells' in capsys.readouterr().out


def test_the_existing_merge_of_cells_sharing_a_nucleus_is_unchanged(tmp_path):
    db = _measure(tmp_path, split=False)
    cells = _table(db, 'cell')
    nuclei = _table(db, 'nucleus')
    assert len(cells) == 1
    assert len(nuclei) == 2
    assert not nuclei['nucleus_shared'].astype(bool).any()
    assert not cells['nucleus_shared'].astype(bool).any()
