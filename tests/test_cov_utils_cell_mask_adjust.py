"""CPU coverage for the cell-mask adjustment block of spacr.utils.

Covers the nucleus-overlap merge branch and the shared-perimeter merge branch of
``_merge_cells_based_on_parasite_overlap``, the organelle / missing-file paths of
``process_mask_file_adjust_cell`` and ``adjust_cell_masks``, and the whole
``process_masks`` clustering routine (including its nested helpers).

Every mask here is hand-built so the expected merge outcome is known exactly
rather than merely "something changed".
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# mask builders
# ---------------------------------------------------------------------------

def _side_by_side_cells(h=40, w=40):
    """Two cells touching along the column-20 boundary; no nuclei yet."""
    cell = np.zeros((h, w), np.uint16)
    cell[5:35, 5:20] = 1
    cell[5:35, 20:35] = 2
    return cell


def _stacked_cells(h=50, w=50):
    """Big top cell (rows 5-19) and a thin bottom cell (rows 20-25) touching it."""
    cell = np.zeros((h, w), np.uint16)
    cell[5:20, 5:45] = 1
    cell[20:26, 5:45] = 2
    return cell


# ---------------------------------------------------------------------------
# _merge_cells_based_on_parasite_overlap - nucleus-overlap merging (6196-6205)
# ---------------------------------------------------------------------------

def test_merge_on_nucleus_overlap_merges_when_all_fractions_above_threshold():
    """A nucleus split 50/50 over two cells merges them into one object."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = _side_by_side_cells()
    nuc = np.zeros_like(cell)
    nuc[15:25, 15:25] = 1                       # 5 columns in cell 1, 5 in cell 2
    para = np.zeros_like(cell)                  # no parasites -> parasite loop is a no-op

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, None, overlap_threshold=5, perimeter_threshold=30)

    assert out.dtype == np.uint16
    assert set(np.unique(out)) == {0, 1}, "the two cells were not merged into one label"
    # both original footprints survive and now carry the same id
    assert out[10, 8] == out[10, 30] == 1
    assert np.count_nonzero(out) == np.count_nonzero(cell)


def test_merge_on_nucleus_overlap_skipped_when_one_fraction_below_threshold():
    """A lopsided nucleus (91%/9%) leaves the cells separate at threshold 20."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = _side_by_side_cells()
    nuc = np.zeros_like(cell)
    nuc[15:25, 10:21] = 1                       # 10 cols in cell 1, 1 col in cell 2
    para = np.zeros_like(cell)

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, None, overlap_threshold=20, perimeter_threshold=30)

    assert set(np.unique(out)) == {0, 1, 2}
    assert out[10, 8] != out[10, 30], "cells merged despite a sub-threshold overlap"


def test_merge_on_nucleus_overlap_single_cell_nucleus_is_left_alone():
    """A nucleus wholly inside one cell never triggers the merge branch."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = _side_by_side_cells()
    nuc = np.zeros_like(cell)
    nuc[10:14, 8:12] = 1                        # entirely inside cell 1
    nuc[10:14, 26:30] = 2                       # entirely inside cell 2
    para = np.zeros_like(cell)

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, None, overlap_threshold=5, perimeter_threshold=30)

    assert set(np.unique(out)) == {0, 1, 2}
    assert out[10, 8] != out[10, 30]


# ---------------------------------------------------------------------------
# _merge_cells_based_on_parasite_overlap - shared-perimeter merging (6219-6237)
# ---------------------------------------------------------------------------

def test_nucleus_free_cell_merges_into_neighbour_with_large_shared_border():
    """A nucleus-free cell sharing ~45% of its perimeter is absorbed by the neighbour."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = _stacked_cells()
    nuc = np.zeros_like(cell)
    nuc[8:14, 10:16] = 1                        # only cell 1 has a nucleus
    para = np.zeros_like(cell)

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, None, overlap_threshold=5, perimeter_threshold=10)

    assert set(np.unique(out)) == {0, 1}
    assert out[10, 10] == out[22, 10] == 1
    assert np.count_nonzero(out) == np.count_nonzero(cell)


def test_nucleus_free_cell_kept_when_shared_border_below_perimeter_threshold():
    """Same geometry, but a 95% perimeter threshold blocks the merge."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = _stacked_cells()
    nuc = np.zeros_like(cell)
    nuc[8:14, 10:16] = 1
    para = np.zeros_like(cell)

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, None, overlap_threshold=5, perimeter_threshold=95)

    assert set(np.unique(out)) == {0, 1, 2}
    assert out[10, 10] != out[22, 10]


def test_isolated_nucleus_free_cell_has_no_neighbours_and_survives():
    """With no neighbouring cell the shared-border list is empty and nothing merges."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = np.zeros((40, 40), np.uint16)
    cell[5:15, 5:15] = 1
    cell[25:35, 25:35] = 2                      # far away, touches nothing
    nuc = np.zeros_like(cell)
    nuc[7:12, 7:12] = 1
    para = np.zeros_like(cell)

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, None, overlap_threshold=5, perimeter_threshold=30)

    assert set(np.unique(out)) == {0, 1, 2}
    assert out[30, 30] != 0 and out[30, 30] != out[10, 10]
    assert np.count_nonzero(out == out[30, 30]) == 100


def test_parasite_overlap_merge_with_organelle_mask_supplied():
    """The organelle branch is exercised and parasite-driven merging still fires."""
    from spacr.utils import _merge_cells_based_on_parasite_overlap

    cell = _side_by_side_cells()
    nuc = np.zeros_like(cell)
    nuc[10:14, 8:12] = 1
    para = np.zeros_like(cell)
    para[18:23, 17:23] = 1                      # straddles the cell-1 / cell-2 border
    organelle = np.zeros_like(cell)
    organelle[7:10, 7:10] = 1

    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, organelle, overlap_threshold=5, perimeter_threshold=30)

    assert set(np.unique(out)) == {0, 1}
    assert out[10, 8] == out[10, 30] == 1


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


def test_process_mask_file_adjust_cell_missing_cell_file_raises(tmp_path):
    from spacr.utils import process_mask_file_adjust_cell

    para_dir = tmp_path / "parasite"
    para_dir.mkdir()
    np.save(para_dir / "f0.npy", np.zeros((10, 10), np.uint16))
    empty_cell = tmp_path / "cell"
    empty_cell.mkdir()
    nuc_dir = tmp_path / "nuclei"
    nuc_dir.mkdir()
    np.save(nuc_dir / "f0.npy", np.zeros((10, 10), np.uint16))

    with pytest.raises(ValueError, match="f0.npy"):
        process_mask_file_adjust_cell("f0.npy", str(para_dir), str(empty_cell), str(nuc_dir))


def test_process_mask_file_adjust_cell_missing_nuclei_file_raises(tmp_path):
    from spacr.utils import process_mask_file_adjust_cell

    para_dir = tmp_path / "parasite"
    para_dir.mkdir()
    np.save(para_dir / "f0.npy", np.zeros((10, 10), np.uint16))
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    np.save(cell_dir / "f0.npy", np.zeros((10, 10), np.uint16))
    nuc_dir = tmp_path / "nuclei"
    nuc_dir.mkdir()

    with pytest.raises(ValueError):
        process_mask_file_adjust_cell("f0.npy", str(para_dir), str(cell_dir), str(nuc_dir))


def test_process_mask_file_adjust_cell_loads_organelle_mask(tmp_path, monkeypatch):
    """When an organelle file exists it is loaded and handed to the merger."""
    import spacr.utils as U

    folders = _write_triple(tmp_path, with_organelle=True)
    seen = {}
    original = U._merge_cells_based_on_parasite_overlap

    def spy(parasite_mask, cell_mask, nuclei_mask, organelle_mask, *a, **kw):
        seen["organelle"] = organelle_mask
        seen["parasite_sum"] = int(parasite_mask.sum())
        return original(parasite_mask, cell_mask, nuclei_mask, organelle_mask, *a, **kw)

    monkeypatch.setattr(U, "_merge_cells_based_on_parasite_overlap", spy)

    elapsed = U.process_mask_file_adjust_cell(
        "f0.npy", folders["parasite"], folders["cell"], folders["nuclei"],
        organelle_folder=folders["organelle"], overlap_threshold=5, perimeter_threshold=10)

    assert elapsed >= 0.0
    assert isinstance(seen["organelle"], np.ndarray)
    assert int(seen["organelle"].sum()) == 9          # the 3x3 organelle blob
    written = np.load(os.path.join(folders["cell"], "f0.npy"))
    assert written.dtype == np.uint16
    assert set(np.unique(written)) == {0, 1}          # nucleus-free cell absorbed


def test_process_mask_file_adjust_cell_organelle_folder_without_matching_file(tmp_path, monkeypatch):
    """An organelle folder lacking this file leaves organelle_mask as None."""
    import spacr.utils as U

    folders = _write_triple(tmp_path, with_organelle=True)
    os.remove(os.path.join(folders["organelle"], "f0.npy"))
    seen = {}
    original = U._merge_cells_based_on_parasite_overlap

    def spy(parasite_mask, cell_mask, nuclei_mask, organelle_mask, *a, **kw):
        seen["organelle"] = organelle_mask
        return original(parasite_mask, cell_mask, nuclei_mask, organelle_mask, *a, **kw)

    monkeypatch.setattr(U, "_merge_cells_based_on_parasite_overlap", spy)

    U.process_mask_file_adjust_cell(
        "f0.npy", folders["parasite"], folders["cell"], folders["nuclei"],
        organelle_folder=folders["organelle"])

    assert seen["organelle"] is None
    assert np.load(os.path.join(folders["cell"], "f0.npy")).ndim == 2


# ---------------------------------------------------------------------------
# adjust_cell_masks
# ---------------------------------------------------------------------------

def test_adjust_cell_masks_raises_on_file_count_mismatch(tmp_path):
    from spacr.utils import adjust_cell_masks

    para = tmp_path / "parasite"; para.mkdir()
    cell = tmp_path / "cell"; cell.mkdir()
    nuc = tmp_path / "nuclei"; nuc.mkdir()
    blank = np.zeros((10, 10), np.uint16)
    np.save(para / "f0.npy", blank)
    np.save(para / "f1.npy", blank)
    np.save(cell / "f0.npy", blank)
    np.save(nuc / "f0.npy", blank)

    with pytest.raises(ValueError, match="number of files"):
        adjust_cell_masks(str(para), str(cell), str(nuc), n_jobs=1)


def test_adjust_cell_masks_non_npy_files_are_ignored(tmp_path):
    """Stray non-.npy files do not count towards the per-folder file totals."""
    from spacr.utils import adjust_cell_masks

    folders = _write_triple(tmp_path, n_files=1)
    (tmp_path / "parasite" / "notes.txt").write_text("ignore me")

    adjust_cell_masks(folders["parasite"], folders["cell"], folders["nuclei"],
                      organelle_folder=None, overlap_threshold=5,
                      perimeter_threshold=10, n_jobs=1)

    out = np.load(os.path.join(folders["cell"], "f0.npy"))
    assert set(np.unique(out)) == {0, 1}


def test_adjust_cell_masks_zero_workers_runs_inline(tmp_path, monkeypatch):
    """n_jobs=0 is a request for stable inline work, not an automatic pool."""
    import spacr.utils as U

    folders = _write_triple(tmp_path, n_files=1)
    monkeypatch.setattr(
        U, "Pool",
        lambda *args, **kwargs: pytest.fail("inline work started a process pool"),
    )

    U.adjust_cell_masks(
        folders["parasite"],
        folders["cell"],
        folders["nuclei"],
        n_jobs=0,
    )

    out = np.load(os.path.join(folders["cell"], "f0.npy"))
    assert set(np.unique(out)) == {0, 1}


def test_adjust_cell_masks_warns_on_organelle_count_mismatch(tmp_path, capsys):
    """A short organelle folder prints a warning and processing continues."""
    from spacr.utils import adjust_cell_masks

    folders = _write_triple(tmp_path, n_files=2, with_organelle=True)  # 2 masks, 1 organelle

    adjust_cell_masks(folders["parasite"], folders["cell"], folders["nuclei"],
                      organelle_folder=folders["organelle"], overlap_threshold=5,
                      perimeter_threshold=10, n_jobs=1)

    out = capsys.readouterr().out
    assert "organelle mask count (1)" in out
    assert "does not match other masks (2)" in out
    for i in range(2):
        arr = np.load(os.path.join(folders["cell"], f"f{i}.npy"))
        assert set(np.unique(arr)) == {0, 1}


def test_adjust_cell_masks_missing_organelle_folder_is_dropped(tmp_path):
    """A non-existent organelle folder is silently ignored (no warning, no crash)."""
    from spacr.utils import adjust_cell_masks

    folders = _write_triple(tmp_path, n_files=1)
    ghost = str(tmp_path / "does_not_exist")

    adjust_cell_masks(folders["parasite"], folders["cell"], folders["nuclei"],
                      organelle_folder=ghost, overlap_threshold=5,
                      perimeter_threshold=10, n_jobs=1)

    assert not os.path.exists(ghost)
    assert set(np.unique(np.load(os.path.join(folders["cell"], "f0.npy")))) == {0, 1}


# ---------------------------------------------------------------------------
# process_masks
# ---------------------------------------------------------------------------

def _mask_two_small_one_big(shape=(40, 40)):
    m = np.zeros(shape, np.int32)
    m[2:7, 2:7] = 1          # area 25
    m[2:7, 12:17] = 2        # area 25
    m[20:34, 10:24] = 3      # area 196
    return m


def _rgb_image(shape=(40, 40)):
    return np.stack([np.full(shape, 100, np.uint16),
                     np.full(shape, 500, np.uint16)], axis=-1)


def _mask_image_folders(tmp_path, masks, images):
    mask_dir = tmp_path / "masks"; mask_dir.mkdir()
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    for i, (m, im) in enumerate(zip(masks, images)):
        np.save(mask_dir / f"f{i}.npy", m)
        np.save(img_dir / f"f{i}.npy", im)
    return str(mask_dir), str(img_dir)


def test_process_masks_keeps_only_the_majority_cluster(tmp_path):
    """The single large object is dropped; the two small ones keep their ids."""
    from spacr.utils import process_masks

    masks = [_mask_two_small_one_big() for _ in range(3)]
    images = [_rgb_image() for _ in range(3)]
    mask_dir, img_dir = _mask_image_folders(tmp_path, masks, images)

    process_masks(mask_dir, img_dir, channel=1, batch_size=2, n_clusters=2, plot=False)

    for i in range(3):
        out = np.load(os.path.join(mask_dir, f"f{i}.npy"))
        assert out.shape == (40, 40)
        assert out.dtype == np.int32
        assert set(np.unique(out)) == {0, 1, 2}, "large object was not removed"
        assert np.count_nonzero(out == 1) == 25
        assert np.count_nonzero(out == 2) == 25
        assert np.count_nonzero(out == 3) == 0
        # intensity images must be untouched
        assert np.array_equal(np.load(os.path.join(img_dir, f"f{i}.npy")), images[i])


def test_process_masks_batch_size_one_and_non_npy_files(tmp_path):
    """batch_size=1 yields one batch per file; stray non-.npy files are skipped."""
    from spacr.utils import process_masks

    masks = [_mask_two_small_one_big() for _ in range(3)]
    images = [_rgb_image() for _ in range(3)]
    mask_dir, img_dir = _mask_image_folders(tmp_path, masks, images)
    # a non-.npy file whose counterpart does not exist in the image folder:
    # np.load would explode on it if the batch reader did not filter by suffix
    (tmp_path / "masks" / "README.md").write_text("not a mask")

    process_masks(mask_dir, img_dir, channel=0, batch_size=1, n_clusters=2, plot=False)

    assert (tmp_path / "masks" / "README.md").read_text() == "not a mask"
    for i in range(3):
        out = np.load(os.path.join(mask_dir, f"f{i}.npy"))
        assert set(np.unique(out)) == {0, 1, 2}
        assert np.count_nonzero(out) == 50


def test_process_masks_plot_true_draws_pca_scatter(tmp_path, monkeypatch):
    """plot=True renders a 2-D PCA scatter of every object and calls plt.show once."""
    import spacr.utils as U

    masks = [_mask_two_small_one_big() for _ in range(2)]
    images = [_rgb_image() for _ in range(2)]
    mask_dir, img_dir = _mask_image_folders(tmp_path, masks, images)

    captured = {}

    def fake_show(*args, **kwargs):
        ax = plt.gca()
        captured["calls"] = captured.get("calls", 0) + 1
        captured["title"] = ax.get_title()
        captured["xlabel"] = ax.get_xlabel()
        captured["ylabel"] = ax.get_ylabel()
        captured["n_points"] = len(ax.collections[0].get_offsets())

    monkeypatch.setattr(U.plt, "show", fake_show)

    U.process_masks(mask_dir, img_dir, channel=1, batch_size=50, n_clusters=2, plot=True)

    assert captured["calls"] == 1
    assert captured["title"] == "Object Clustering"
    assert captured["xlabel"] == "PCA Component 1"
    assert captured["ylabel"] == "PCA Component 2"
    assert captured["n_points"] == 6              # 3 objects x 2 files
    assert set(np.unique(np.load(os.path.join(mask_dir, "f0.npy")))) == {0, 1, 2}


def test_process_masks_three_clusters_partitions_objects(tmp_path):
    """With n_clusters=3 the per-file majority cluster still survives cleaning."""
    from spacr.utils import process_masks

    def mask_with_sizes():
        m = np.zeros((60, 60), np.int32)
        m[2:7, 2:7] = 1        # 25
        m[2:7, 12:17] = 2      # 25
        m[10:20, 30:40] = 3    # 100
        m[30:55, 5:30] = 4     # 625
        return m

    masks = [mask_with_sizes() for _ in range(2)]
    images = [_rgb_image((60, 60)) for _ in range(2)]
    mask_dir, img_dir = _mask_image_folders(tmp_path, masks, images)

    process_masks(mask_dir, img_dir, channel=0, batch_size=50, n_clusters=3, plot=False)

    out = np.load(os.path.join(mask_dir, "f0.npy"))
    kept = set(np.unique(out)) - {0}
    assert kept == {1, 2}, f"expected the two small objects to be the majority cluster, got {kept}"
    assert np.count_nonzero(out) == 50


def test_process_masks_handles_non_contiguous_label_ids(tmp_path):
    """Masks whose labels are not 1..N (e.g. after any filtering step) must still work."""
    from spacr.utils import process_masks

    m = np.zeros((40, 40), np.int32)
    m[2:7, 2:7] = 2            # small
    m[20:34, 10:24] = 3        # big
    mask_dir, img_dir = _mask_image_folders(tmp_path, [m], [_rgb_image()])

    process_masks(mask_dir, img_dir, channel=0, batch_size=50, n_clusters=2, plot=False)

    out = np.load(os.path.join(mask_dir, "f0.npy"))
    assert set(np.unique(out)) - {0} in ({2}, {3})


def test_process_masks_handles_object_free_mask(tmp_path):
    """A field of view with no segmented objects must not abort the whole folder."""
    from spacr.utils import process_masks

    masks = [_mask_two_small_one_big(), np.zeros((40, 40), np.int32)]
    images = [_rgb_image(), _rgb_image()]
    mask_dir, img_dir = _mask_image_folders(tmp_path, masks, images)

    process_masks(mask_dir, img_dir, channel=0, batch_size=50, n_clusters=2, plot=False)

    assert np.count_nonzero(np.load(os.path.join(mask_dir, "f1.npy"))) == 0
    assert set(np.unique(np.load(os.path.join(mask_dir, "f0.npy")))) == {0, 1, 2}
