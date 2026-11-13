"""CPU coverage for the cell-mask adjustment block of spacr.utils.

Covers the nucleus-overlap merge branch and the shared-perimeter merge branch of
``_merge_cells_based_on_parasite_overlap``, the organelle / missing-file paths of
``process_mask_file_adjust_cell`` and ``adjust_cell_masks``, and the whole
``process_masks`` clustering routine (including its nested helpers).

Every mask here is hand-built so the expected merge outcome is known exactly
rather than merely "something changed".
"""
from __future__ import annotations


import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from tests import locale_cases


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# mask builders
# ---------------------------------------------------------------------------

def _stacked_cells(h=50, w=50):
    """Big top cell (rows 5-19) and a thin bottom cell (rows 20-25) touching it."""
    cell = np.zeros((h, w), np.uint16)
    cell[5:20, 5:45] = 1
    cell[20:26, 5:45] = 2
    return cell


# ---------------------------------------------------------------------------
# _merge_cells_based_on_parasite_overlap - nucleus-overlap merging (6196-6205)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _merge_cells_based_on_parasite_overlap - shared-perimeter merging (6219-6237)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# process_mask_file_adjust_cell
# ---------------------------------------------------------------------------

def _write_triple(tmp_path, name="f0.npy", with_organelle=False, n_files=1):
    """Write parasite/cell/nuclei (+organelle) folders and return their paths."""
    cell = _stacked_cells()
    nuc = np.zeros_like(cell)
    nuc[8:14, 10:16] = 1
    para = np.zeros_like(cell)
    para[18:22, 10:16] = 1
    org = np.zeros_like(cell)
    org[9:12, 20:23] = 1

    folders = {}
    for key, arr in (("parasite", para), ("cell", cell), ("nuclei", nuc)):
        d = tmp_path / key
        d.mkdir(exist_ok=True)
        for i in range(n_files):
            np.save(d / (name if n_files == 1 else f"f{i}.npy"), arr)
        folders[key] = str(d)
    if with_organelle:
        d = tmp_path / "organelle"
        d.mkdir(exist_ok=True)
        np.save(d / (name if n_files == 1 else "f0.npy"), org)
        folders["organelle"] = str(d)
    return folders


# ---------------------------------------------------------------------------
# adjust_cell_masks
# ---------------------------------------------------------------------------

def _adjustment_checks_later_array_before_changing_any_field_case(tmp_path, monkeypatch, role, damage, workers):
    """One case of :func:`test_adjustment_checks_later_array_before_changing_any_field`."""
    import spacr.utils as U

    folders = _write_triple(tmp_path, with_organelle=True, n_files=2)
    target = tmp_path / role / 'f1.npy'
    if damage == 'shape':
        np.save(target, np.zeros((12, 15), np.uint16))
    elif damage == 'object':
        np.save(target, np.array([[{'not': 'a label'}]], dtype=object))
    else:
        np.save(target, np.zeros((50, 50), np.uint16))
        target.write_bytes(target.read_bytes()[:-10])
    before = {path: path.read_bytes() for path in tmp_path.rglob('*.npy')}
    monkeypatch.setattr(U, 'process_mask_file_adjust_cell',
                        lambda *args, **kwargs: pytest.fail('adjustment already started'))
    monkeypatch.setattr(U, 'Pool',
                        lambda *args, **kwargs: pytest.fail('workers already started'))
    with pytest.raises(ValueError, match='f1.npy'):
        U.adjust_cell_masks(folders['parasite'], folders['cell'], folders['nuclei'],
                            organelle_folder=folders['organelle'], n_jobs=workers)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob('*.npy')}


def test_adjustment_checks_later_array_before_changing_any_field(subtests, tmp_path, monkeypatch):
    for label, case in locale_cases.cases(('role', ['parasite', 'cell', 'nuclei', 'organelle']), ('damage', ['shape', 'truncated', 'object']), ('workers', [1, 2])):
        with subtests.test(**label), locale_cases.monkeypatch_scope() as case_monkeypatch:
            _adjustment_checks_later_array_before_changing_any_field_case(tmp_path=locale_cases.case_dir(tmp_path, label), monkeypatch=case_monkeypatch, **case)


# ---------------------------------------------------------------------------
# process_masks
# ---------------------------------------------------------------------------

