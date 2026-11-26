"""The fixed spanning rule ``'merge_safe'`` (item 1037, alpha).

``'merge'`` and ``'best_parent'`` stay exactly as on nightly 678e644366: the
golden hashes below were taken from that tree on synthetic masks.
"""
import hashlib

import numpy as np

from spacr import measure as M
from spacr import ops_large_sites, ops_sample
from spacr import settings as S

GOLDEN = {
    "best_parent": "4a1d945f9e886b38823b1aceace6aabd39ffbe2a8298d9e93eed86992a573e04",
    "merge": "b9d2710bd3f88b435e16d2c7714134d02d73eb39d1ae6e28d4baef35ca3475d6",
}


def _random_case(seed):
    r = np.random.default_rng(seed)
    cells = np.zeros((60, 60), np.int32)
    nuc = np.zeros((60, 60), np.int32)
    for i in range(12):
        y, x = r.integers(0, 50, 2)
        h, w = r.integers(8, 20, 2)
        cells[y:y + h, x:x + w] = i + 1
    for i in range(10):
        y, x = r.integers(0, 54, 2)
        h, w = r.integers(3, 9, 2)
        nuc[y:y + h, x:x + w] = i + 1
    return nuc, cells


def _hash(mode):
    h = hashlib.sha256()
    for seed in range(40):
        n, c = _random_case(seed)
        r = M._assign_nuclei_to_cells(n.copy(), c.copy(), spanning=mode)
        for a in r[:2]:
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr((sorted(r[2]), sorted(r[3]), r[4])).encode())
    return h.hexdigest()


def test_merge_and_best_parent_are_byte_identical_to_nightly():
    assert _hash("best_parent") == GOLDEN["best_parent"]
    assert _hash("merge") == GOLDEN["merge"]


def _two_cells_two_nuclei():
    cells = np.zeros((20, 40), np.int32)
    cells[2:18, 2:18] = 1
    cells[2:18, 18:38] = 2
    nuc = np.zeros_like(cells)
    nuc[6:14, 6:16] = 1
    nuc[6:14, 22:32] = 2
    nuc[8:12, 16:22] = 3
    return nuc, cells


def test_cells_with_their_own_nucleus_are_never_merged():
    n, c = _two_cells_two_nuclei()
    n[n == 3] = 0
    n[8:12, 14:22] = 1
    n[8:12, 22:26] = 2
    n2, c2, _sn, _sc, merged = M._assign_nuclei_to_cells(
        n.copy(), c.copy(), spanning="merge_safe")
    assert merged == 0
    assert set(np.unique(c2)) == {0, 1, 2}
    old = M._assign_nuclei_to_cells(n.copy(), c.copy(), spanning="merge")
    assert old[4] >= 1 or set(np.unique(old[1])) == {0, 1}


def test_a_nucleusless_piece_merges_into_the_nucleus_cell():
    cells = np.zeros((20, 30), np.int32)
    cells[2:18, 2:14] = 1
    cells[2:18, 14:28] = 2
    nuc = np.zeros_like(cells)
    nuc[6:14, 8:20] = 1
    nuc[6:14, 8:14] = 1
    nuc[6:14, 14:20] = 1
    n2, c2, _a, _b, merged = M._assign_nuclei_to_cells(
        nuc.copy(), cells.copy(), spanning="merge_safe")
    assert merged == 1
    assert set(np.unique(c2)) == {0, 1}
    assert (c2 > 0).sum() == (cells > 0).sum()
    assert (n2 > 0).sum() == (nuc > 0).sum()


def test_the_share_counts_in_cell_pixels_only():
    cells = np.zeros((20, 40), np.int32)
    cells[2:18, 2:14] = 1
    cells[2:18, 14:15] = 2
    nuc = np.zeros_like(cells)
    nuc[6:14, 4:34] = 1
    cells_before = cells.copy()
    n_safe, c_safe, _a, _b, merged_safe = M._assign_nuclei_to_cells(
        nuc.copy(), cells.copy(), spanning="merge_safe")
    assert merged_safe == 0
    assert (c_safe == cells_before).all()
    assert (n_safe[:, 14] == 0).all()
    assert (n_safe[6:14, 4:14] == 1).all()
    n_old, c_old, _a, _b, merged_old = M._assign_nuclei_to_cells(
        nuc.copy(), cells.copy(), spanning="merge")
    assert merged_old == 1


def test_the_mode_is_accepted_and_the_default_unchanged():
    assert M._spanning_mode("Merge_Safe") == "merge_safe"
    assert M._spanning_mode(None) == "best_parent"
    assert M._SPANNING_MODES[0] == "best_parent"


def test_ops_measure_and_large_site_pair_the_same():
    n, c = _two_cells_two_nuclei()
    cells = np.zeros((20, 30), np.int32)
    cells[2:18, 2:14] = 1
    cells[2:18, 14:28] = 2
    nuc = np.zeros_like(cells)
    nuc[6:14, 8:20] = 1
    direct = M._assign_nuclei_to_cells(nuc.copy(), cells.copy(),
                                       spanning="merge_safe")
    ops = ops_sample._pair_nuclei_with_cells(nuc, cells, spanning="merge_safe")
    site = ops_large_sites._site_pairing(nuc, cells, None, None,
                                         spanning="merge_safe")
    assert ops[2]["merged_cells"] == direct[4] == 1
    assert site[2]["merged_cells"] == 1
    assert (ops[0] == site[0]).all() and (ops[1] == site[1]).all()
    assert ops[1].max() == 1
    kept = ops_sample._pair_nuclei_with_cells(n, c, spanning="merge_safe")
    assert kept[2]["merged_cells"] == 0


def test_the_choice_is_registered_alpha_and_described():
    assert S._is_alpha("choices", "spanning_merge", "merge_safe")
    assert S._is_alpha("choices", "ops_ph_spanning_merge", "merge_safe")
    import pathlib
    root = pathlib.Path(S.__file__).parent
    for name in ("settings.py", "ops_settings.py"):
        assert "'merge_safe'" in (root / name).read_text()
