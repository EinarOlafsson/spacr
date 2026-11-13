"""Table-driven tests of the cell-mask adjustment and label merge/split helpers in ``spacr.utils``.

Covers the parasite / nucleus / shared-perimeter cell merger, its per-file and
per-folder drivers (``process_mask_file_adjust_cell`` and
``adjust_cell_masks``), ``process_masks`` clustering, ``_save_image``, and the
in-memory / on-disk field filter (``_process_single_fov_in_memory``,
``_process_single_fov``, ``merge_split_objects``).  Each unit gets one
general test that walks a case table with pytest's ``subtests`` fixture.
Every mask is hand-built so the expected result is known exactly.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

tifffile = pytest.importorskip("tifffile")

import spacr.utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


def _uniq(arr):
    """Sorted list of the distinct values of ``arr``."""
    return sorted(np.unique(arr).tolist())


def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    d = tmp_path / name
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# _merge_cells_based_on_parasite_overlap
# ---------------------------------------------------------------------------

def _side_by_side_cells(h=40, w=40):
    """Two cells touching along the column-20 boundary."""
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


def _pair_12x20(gap=False):
    cell = np.zeros((12, 20), np.uint16)
    cell[2:10, 1:9] = 1
    if gap:
        cell[2:10, 11:19] = 2
    else:
        cell[2:10, 9:17] = 2
    return cell


def _nuclei_12x20():
    nuc = np.zeros((12, 20), np.uint16)
    nuc[3:5, 2:4] = 1
    nuc[3:5, 12:14] = 2
    return nuc


def _merge_inputs(kind):
    """Return (parasite, cell, nuclei, organelle) for one merge scenario."""
    if kind in ("nucleus_even", "nucleus_lopsided", "nucleus_inside_each",
                "parasite_with_organelle"):
        cell = _side_by_side_cells()
        nuc = np.zeros_like(cell)
        para = np.zeros_like(cell)
        org = None
        if kind == "nucleus_even":
            nuc[15:25, 15:25] = 1                 # 5 columns in each cell
        elif kind == "nucleus_lopsided":
            nuc[15:25, 10:21] = 1                 # 10 columns vs 1
        elif kind == "nucleus_inside_each":
            nuc[10:14, 8:12] = 1
            nuc[10:14, 26:30] = 2
        else:
            nuc[10:14, 8:12] = 1
            para[18:23, 17:23] = 1                # straddles the border
            org = np.zeros_like(cell)
            org[7:10, 7:10] = 1
        return para, cell, nuc, org
    if kind == "stacked":
        cell = _stacked_cells()
        nuc = np.zeros_like(cell)
        nuc[8:14, 10:16] = 1                      # only the top cell has a nucleus
        return np.zeros_like(cell), cell, nuc, None
    if kind == "isolated":
        cell = np.zeros((40, 40), np.uint16)
        cell[5:15, 5:15] = 1
        cell[25:35, 25:35] = 2
        nuc = np.zeros_like(cell)
        nuc[7:12, 7:12] = 1
        return np.zeros_like(cell), cell, nuc, None
    if kind == "parasite_95_5":
        para = np.zeros((12, 20), np.uint16)
        para[5, 1:9] = 1
        para[6, 1:9] = 1
        para[7, 1:4] = 1
        para[5, 9] = 1                            # exactly 5% in the right cell
        return para, _pair_12x20(), _nuclei_12x20(), None
    if kind == "parasite_in_one_cell":
        para = np.zeros((12, 20), np.uint16)
        para[4:6, 3:7] = 1
        return para, _pair_12x20(), _nuclei_12x20(), None
    if kind == "no_shared_parasite":
        cell = np.zeros((12, 30), np.uint16)
        cell[2:10, 1:9] = 1
        cell[2:10, 21:29] = 2
        para = np.zeros((12, 30), np.uint16)
        para[4:8, 2:8] = 1
        empty = np.zeros((12, 30), np.uint16)
        return para, cell, empty, empty
    if kind == "three_apart_one_parasite":
        cell = np.zeros((12, 30), np.uint16)
        cell[2:10, 1:9] = 1
        cell[2:10, 11:19] = 2
        cell[2:10, 21:29] = 3
        para = np.zeros((12, 30), np.uint16)
        para[4:8, 1:29] = 1
        empty = np.zeros((12, 30), np.uint16)
        return para, cell, empty, empty
    if kind == "two_touching_one_parasite":
        para = np.zeros((12, 20), np.uint16)
        para[4:8, 3:15] = 1
        empty = np.zeros((12, 20), np.uint16)
        return para, _pair_12x20(), empty, empty
    raise AssertionError(kind)


def test_merge_cells_based_on_parasite_overlap_joins_only_split_cells(subtests):
    """Cells merge for a shared parasite/nucleus above threshold or a large shared border; else kept."""
    cases = [
        dict(id="nucleus_split_evenly", kind="nucleus_even",
             kw=dict(overlap_threshold=5, perimeter_threshold=30), merged=True),
        dict(id="nucleus_below_threshold", kind="nucleus_lopsided",
             kw=dict(overlap_threshold=20, perimeter_threshold=30), merged=False),
        dict(id="nucleus_in_one_cell_each", kind="nucleus_inside_each",
             kw=dict(overlap_threshold=5, perimeter_threshold=30), merged=False),
        dict(id="large_shared_border", kind="stacked",
             kw=dict(overlap_threshold=5, perimeter_threshold=10), merged=True),
        dict(id="border_below_perimeter_threshold", kind="stacked",
             kw=dict(overlap_threshold=5, perimeter_threshold=95), merged=False),
        dict(id="isolated_nucleus_free_cell", kind="isolated",
             kw=dict(overlap_threshold=5, perimeter_threshold=30), merged=False),
        dict(id="parasite_with_organelle", kind="parasite_with_organelle",
             kw=dict(overlap_threshold=5, perimeter_threshold=30), merged=True),
        # the cell holding 5% of the parasite does not trigger a second merge
        dict(id="parasite_split_95_5_merges_once", kind="parasite_95_5",
             kw=dict(overlap_threshold=5), merged=True),
        dict(id="parasite_in_one_cell", kind="parasite_in_one_cell", kw={}, merged=False),
        dict(id="no_shared_parasite", kind="no_shared_parasite",
             kw=dict(overlap_threshold=1, perimeter_threshold=1000), merged=False),
        # the LABEL is unified in place, then the return is relabelled by
        # connectivity: three blocks that do not touch come back as three.
        dict(id="three_apart_one_parasite", kind="three_apart_one_parasite",
             kw=dict(overlap_threshold=1, perimeter_threshold=1), merged=False,
             in_place_labels=[0, 1]),
        dict(id="two_touching_one_parasite", kind="two_touching_one_parasite",
             kw=dict(overlap_threshold=1, perimeter_threshold=1), merged=True),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            para, cell, nuc, org = _merge_inputs(case["kind"])
            original = cell.copy()
            out = U._merge_cells_based_on_parasite_overlap(para, cell, nuc, org,
                                                           **case["kw"])
            assert out.dtype == np.uint16
            if case["merged"]:
                expected = (original > 0).astype(np.uint16)
            else:
                expected = original
            np.testing.assert_array_equal(out, expected)
            assert np.count_nonzero(out) == np.count_nonzero(original)
            if "in_place_labels" in case:
                assert _uniq(cell) == case["in_place_labels"]


def test_merge_cells_depends_on_geometry_not_numeric_label_ids(subtests):
    """Renumbering the cells (sparse or reversed ids) gives the identical merge."""
    for reason in ("parasite", "nucleus", "perimeter"):
        for identities in ((7, 300), (300, 7)):
            with subtests.test(reason=reason, identities=identities):
                cells = _side_by_side_cells()
                nuclei = np.zeros_like(cells)
                parasites = np.zeros_like(cells)
                if reason == "parasite":
                    parasites[18:23, 17:23] = 1
                    nuclei[10:14, 8:12] = 1
                    nuclei[10:14, 26:30] = 2
                elif reason == "nucleus":
                    nuclei[15:25, 15:25] = 1
                else:
                    nuclei[10:14, 8:12] = 1
                sparse = np.zeros_like(cells)
                for old, new in enumerate(identities, 1):
                    sparse[cells == old] = new
                expected = U._merge_cells_based_on_parasite_overlap(
                    parasites, cells.copy(), nuclei, None, perimeter_threshold=10)
                actual = U._merge_cells_based_on_parasite_overlap(
                    parasites, sparse, nuclei, None, perimeter_threshold=10)
                assert _uniq(expected) == [0, 1]
                np.testing.assert_array_equal(actual, expected)
                assert np.count_nonzero(actual) == np.count_nonzero(cells)


# ---------------------------------------------------------------------------
# process_mask_file_adjust_cell / adjust_cell_masks
# ---------------------------------------------------------------------------

def _write_triple(root, name="f0.npy", with_organelle=False, n_files=1):
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
        d = root / key
        d.mkdir(exist_ok=True)
        for i in range(n_files):
            np.save(d / (name if n_files == 1 else f"f{i}.npy"), arr)
        folders[key] = str(d)
    if with_organelle:
        d = root / "organelle"
        d.mkdir(exist_ok=True)
        np.save(d / (name if n_files == 1 else "f0.npy"), org)
        folders["organelle"] = str(d)
    return folders


def _write_r4_set(root):
    """Two fields of separate (non-touching) cells with a matching organelle folder."""
    folders = {}
    for name in ("parasite", "cell", "nuclei", "organelle"):
        (root / name).mkdir()
        folders[name] = str(root / name)
    for stem in ("f1.npy", "f2.npy"):
        cells = np.zeros((12, 20), np.uint16)
        cells[2:10, 1:9] = 1
        cells[2:10, 11:19] = 2
        np.save(os.path.join(folders["cell"], stem), cells)
        parasites = np.zeros((12, 20), np.uint16)
        parasites[4:6, 3:7] = 1
        np.save(os.path.join(folders["parasite"], stem), parasites)
        np.save(os.path.join(folders["nuclei"], stem), _nuclei_12x20())
        np.save(os.path.join(folders["organelle"], stem), np.zeros((12, 20), np.uint16))
    return folders


def _snapshot(root):
    return {p: p.read_bytes() for p in root.rglob("*.npy")}


def test_process_mask_file_adjust_cell_merges_one_field_in_place(subtests, tmp_path,
                                                                 monkeypatch):
    """Loads the field's masks (organelle when present), merges, rewrites the cell mask; missing masks raise."""
    cases = [
        dict(id="missing_cell_file", missing="cell", raises="f0.npy"),
        dict(id="missing_nuclei_file", missing="nuclei", raises="f0.npy"),
        dict(id="organelle_loaded", organelle=True,
             kw=dict(overlap_threshold=5, perimeter_threshold=10), organelle_sum=9),
        dict(id="organelle_folder_lacks_file", organelle=True, drop_organelle=True,
             kw={}, organelle_sum=None),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            root = _case_dir(tmp_path, case["id"])
            if "missing" in case:
                for key in ("parasite", "cell", "nuclei"):
                    (root / key).mkdir()
                    if key != case["missing"]:
                        np.save(root / key / "f0.npy", np.zeros((10, 10), np.uint16))
                before = _snapshot(root)
                with pytest.raises(ValueError, match=case["raises"]):
                    U.process_mask_file_adjust_cell("f0.npy", str(root / "parasite"),
                                                    str(root / "cell"),
                                                    str(root / "nuclei"))
                assert _snapshot(root) == before
                continue
            folders = _write_triple(root, with_organelle=True)
            if case.get("drop_organelle"):
                os.remove(os.path.join(folders["organelle"], "f0.npy"))
            seen = {}
            original = U._merge_cells_based_on_parasite_overlap

            def spy(parasite_mask, cell_mask, nuclei_mask, organelle_mask, *a, **kw):
                seen["organelle"] = organelle_mask
                seen["parasite_sum"] = int(parasite_mask.sum())
                return original(parasite_mask, cell_mask, nuclei_mask, organelle_mask,
                                *a, **kw)

            with monkeypatch.context() as mp:
                mp.setattr(U, "_merge_cells_based_on_parasite_overlap", spy)
                elapsed = U.process_mask_file_adjust_cell(
                    "f0.npy", folders["parasite"], folders["cell"], folders["nuclei"],
                    organelle_folder=folders["organelle"], **case["kw"])
            assert elapsed >= 0.0
            assert seen["parasite_sum"] == 24
            if case["organelle_sum"] is None:
                assert seen["organelle"] is None
            else:
                assert isinstance(seen["organelle"], np.ndarray)
                assert int(seen["organelle"].sum()) == case["organelle_sum"]
            written = np.load(os.path.join(folders["cell"], "f0.npy"))
            assert written.ndim == 2
            assert written.dtype == np.uint16
            assert _uniq(written) == [0, 1]          # nucleus-free cell absorbed


def test_adjust_cell_masks_refuses_mismatched_folders_before_any_work(subtests, tmp_path,
                                                                      monkeypatch):
    """Different counts or names across the folders raise before a field is touched."""
    cases = [dict(id="count_mismatch", kind="count", workers=1)]
    for role in ("cell", "nuclei", "parasite"):
        for workers in (1, 2):
            cases.append(dict(id=f"names_{role}_{workers}", kind="names", role=role,
                              workers=workers))
    for case in cases:
        with subtests.test(case=case["id"]):
            root = _case_dir(tmp_path, case["id"])
            if case["kind"] == "count":
                for key in ("parasite", "cell", "nuclei"):
                    (root / key).mkdir()
                    np.save(root / key / "f0.npy", np.zeros((10, 10), np.uint16))
                np.save(root / "parasite" / "f1.npy", np.zeros((10, 10), np.uint16))
                folders = {k: str(root / k) for k in ("parasite", "cell", "nuclei")}
                match, frags = "number of files", []
            else:
                folders = _write_triple(root, n_files=2)
                (root / case["role"] / "f1.npy").rename(root / case["role"] / "other.npy")
                match, frags = "filenames do not match", ["f1.npy", "other.npy"]
            before = _snapshot(root)
            with monkeypatch.context() as mp:
                mp.setattr(U, "process_mask_file_adjust_cell",
                           lambda *a, **k: pytest.fail("adjustment already started"))
                mp.setattr(U, "Pool", lambda *a, **k: pytest.fail("workers already started"))
                with pytest.raises(ValueError, match=match) as raised:
                    U.adjust_cell_masks(folders["parasite"], folders["cell"],
                                        folders["nuclei"], n_jobs=case["workers"])
            for frag in frags:
                assert frag in str(raised.value)
            assert _snapshot(root) == before


def _counting_merger(mp, calls):
    real = U._merge_cells_based_on_parasite_overlap

    def counting(parasite, cell, *args, **kwargs):
        calls.append(cell.shape)
        return real(parasite, cell, *args, **kwargs)

    mp.setattr(U, "_merge_cells_based_on_parasite_overlap", counting)


def test_adjust_cell_masks_rewrites_each_field_once(subtests, tmp_path, monkeypatch, capsys):
    """Every field is merged in place (inline for n_jobs<=1), stray files and missing organelle folders ignored, re-runs skipped by ledger."""
    cases = [
        dict(id="non_npy_files_ignored", n_files=1, stray=True, n_jobs=1,
             kw=dict(organelle_folder=None, overlap_threshold=5, perimeter_threshold=10),
             want={"f0.npy": [0, 1]}),
        dict(id="zero_workers_inline", n_files=1, n_jobs=0, no_pool=True, kw={},
             want={"f0.npy": [0, 1]}),
        dict(id="organelle_count_mismatch_warns", n_files=2, organelle=True, n_jobs=1,
             kw=dict(overlap_threshold=5, perimeter_threshold=10), organelle_kw=True,
             want={"f0.npy": [0, 1], "f1.npy": [0, 1]},
             says=["organelle mask count (1)", "does not match other masks (2)"]),
        dict(id="missing_organelle_folder_dropped", n_files=1, ghost=True, n_jobs=1,
             kw=dict(overlap_threshold=5, perimeter_threshold=10),
             want={"f0.npy": [0, 1]}, silent=True),
        dict(id="matching_organelle_no_warning", r4=True, n_jobs=1, kw={},
             want={"f1.npy": [0, 1, 2], "f2.npy": [0, 1, 2]}, silent=True),
        dict(id="rerun_skips_adjusted", n_files=2, n_jobs=1, kw={}, runs=2,
             merges=[2, 2], ledger=True, want={"f0.npy": [0, 1], "f1.npy": [0, 1]}),
        dict(id="resegmented_is_adjusted_again", n_files=2, n_jobs=1, kw={}, runs=2,
             reseg="f1.npy", merges=[2, 3], ledger=True,
             want={"f0.npy": [0, 1], "f1.npy": [0, 1]}),
        dict(id="separate_output_writes_no_record", n_files=1, n_jobs=1, kw={},
             output=True, ledger=False, want={}),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            capsys.readouterr()
            root = _case_dir(tmp_path, case["id"])
            if case.get("r4"):
                folders = _write_r4_set(root)
            else:
                folders = _write_triple(root, n_files=case["n_files"],
                                        with_organelle=case.get("organelle", False))
            kw = dict(case["kw"])
            if case.get("stray"):
                (root / "parasite" / "notes.txt").write_text("ignore me", encoding="utf-8")
            if case.get("organelle_kw") or case.get("r4"):
                kw["organelle_folder"] = folders["organelle"]
            ghost = str(root / "does_not_exist")
            if case.get("ghost"):
                kw["organelle_folder"] = ghost
            if case.get("output"):
                kw["output_folder"] = str(root / "adjusted")
            calls = []
            snapshots = []
            with monkeypatch.context() as mp:
                if case.get("no_pool"):
                    mp.setattr(U, "Pool",
                               lambda *a, **k: pytest.fail("inline work started a pool"))
                _counting_merger(mp, calls)
                for run in range(case.get("runs", 1)):
                    if run == 1 and case.get("reseg"):
                        np.save(root / "cell" / case["reseg"], _stacked_cells())
                    U.adjust_cell_masks(folders["parasite"], folders["cell"],
                                        folders["nuclei"], n_jobs=case["n_jobs"], **kw)
                    snapshots.append({p.name: p.read_bytes()
                                      for p in (root / "cell").glob("*.npy")})
                    if "merges" in case:
                        assert len(calls) == case["merges"][run]
            out = capsys.readouterr().out
            for name, values in case["want"].items():
                assert _uniq(np.load(os.path.join(folders["cell"], name))) == values
            for frag in case.get("says", []):
                assert frag in out
            if case.get("silent"):
                assert "does not match other masks" not in out
            if case.get("ghost"):
                assert not os.path.exists(ghost)
            if case.get("runs") == 2 and not case.get("reseg"):
                assert snapshots[0] == snapshots[1]
            ledger = root / "cell" / U.ADJUSTED_CELLS_LEDGER
            if "ledger" in case:
                assert ledger.is_file() == case["ledger"]
            if case.get("output"):
                assert not (root / "adjusted" / U.ADJUSTED_CELLS_LEDGER).exists()
            if case.get("stray"):
                assert (root / "parasite" / "notes.txt").read_text(
                    encoding="utf-8") == "ignore me"


def _mask_set(root):
    """Two fields of parasite / cell / nuclei masks that merge cleanly."""
    folders = {}
    for kind in ("parasite", "cell", "nuclei"):
        (root / kind).mkdir()
        folders[kind] = str(root / kind)
    for index in range(2):
        cell = np.zeros((16, 16), np.uint16)
        cell[2:8, 2:8] = 1
        cell[2:8, 9:15] = 2
        nuclei = np.zeros((16, 16), np.uint16)
        nuclei[3:5, 3:5] = 1
        nuclei[3:5, 10:12] = 2
        parasite = np.zeros((16, 16), np.uint16)
        parasite[5:11, 5:12] = 1
        name = f"field{index}.npy"
        np.save(os.path.join(folders["cell"], name), cell)
        np.save(os.path.join(folders["nuclei"], name), nuclei)
        np.save(os.path.join(folders["parasite"], name), parasite)
    return folders


@pytest.mark.integration
def test_adjust_cell_masks_pool_writes_every_field_like_inline(subtests, tmp_path, capsys):
    """The process-pool path saves the same merged masks as the inline path."""
    results = {}
    for n_jobs in (1, 2):
        with subtests.test(n_jobs=n_jobs):
            capsys.readouterr()
            folders = _mask_set(_case_dir(tmp_path, f"jobs{n_jobs}"))
            before = [np.load(os.path.join(folders["cell"], f"field{i}.npy")).copy()
                      for i in range(2)]
            U.adjust_cell_masks(folders["parasite"], folders["cell"], folders["nuclei"],
                                n_jobs=n_jobs)
            after = [np.load(os.path.join(folders["cell"], f"field{i}.npy"))
                     for i in range(2)]
            assert all(np.array_equal(after[0], plane) for plane in after)
            assert len(np.unique(after[0])) <= len(np.unique(before[0]))
            assert "adjust_cell_masks" in capsys.readouterr().out
            results[n_jobs] = after
    assert all(np.array_equal(a, b) for a, b in zip(results[1], results[2]))


# ---------------------------------------------------------------------------
# process_masks
# ---------------------------------------------------------------------------

def _mask_two_small_one_big(shape=(40, 40)):
    m = np.zeros(shape, np.int32)
    m[2:7, 2:7] = 1
    m[2:7, 12:17] = 2
    m[20:34, 10:24] = 3
    return m


def _mask_four_sizes():
    m = np.zeros((60, 60), np.int32)
    m[2:7, 2:7] = 1
    m[2:7, 12:17] = 2
    m[10:20, 30:40] = 3
    m[30:55, 5:30] = 4
    return m


def _mask_non_contiguous():
    m = np.zeros((40, 40), np.int32)
    m[2:7, 2:7] = 2
    m[20:34, 10:24] = 3
    return m


def _rgb_image(shape=(40, 40)):
    return np.stack([np.full(shape, 100, np.uint16), np.full(shape, 500, np.uint16)],
                    axis=-1)


def test_process_masks_keeps_each_fields_majority_cluster(subtests, tmp_path, monkeypatch):
    """Objects are clustered on shape/intensity and only the majority cluster survives each field."""
    small = {1: 25, 2: 25, 3: 0}
    cases = [
        dict(id="majority_two_small", masks=[_mask_two_small_one_big() for _ in range(3)],
             shape=(40, 40), kw=dict(channel=1, batch_size=2, n_clusters=2),
             want={i: ([0, 1, 2], small) for i in range(3)}),
        dict(id="batch_size_one_and_readme", masks=[_mask_two_small_one_big()
                                                    for _ in range(3)],
             shape=(40, 40), kw=dict(channel=0, batch_size=1, n_clusters=2), readme=True,
             want={i: ([0, 1, 2], small) for i in range(3)}),
        dict(id="plot_pca_scatter", masks=[_mask_two_small_one_big() for _ in range(2)],
             shape=(40, 40), kw=dict(channel=1, batch_size=50, n_clusters=2, plot=True),
             want={0: ([0, 1, 2], {})}, plot_points=6),
        dict(id="three_clusters", masks=[_mask_four_sizes() for _ in range(2)],
             shape=(60, 60), kw=dict(channel=0, batch_size=50, n_clusters=3),
             want={0: ([0, 1, 2], {1: 25, 2: 25})}),
        dict(id="non_contiguous_ids", masks=[_mask_non_contiguous()], shape=(40, 40),
             kw=dict(channel=0, batch_size=50, n_clusters=2), one_of=[[0, 2], [0, 3]]),
        dict(id="object_free_field", masks=[_mask_two_small_one_big(),
                                            np.zeros((40, 40), np.int32)],
             shape=(40, 40), kw=dict(channel=0, batch_size=50, n_clusters=2),
             want={0: ([0, 1, 2], {}), 1: ([0], {})}),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            root = _case_dir(tmp_path, case["id"])
            mask_dir, img_dir = root / "masks", root / "imgs"
            mask_dir.mkdir()
            img_dir.mkdir()
            images = [_rgb_image(case["shape"]) for _ in case["masks"]]
            for i, (m, im) in enumerate(zip(case["masks"], images)):
                np.save(mask_dir / f"f{i}.npy", m)
                np.save(img_dir / f"f{i}.npy", im)
            if case.get("readme"):
                (mask_dir / "README.md").write_text("not a mask", encoding="utf-8")
            kw = dict(plot=False)
            kw.update(case["kw"])
            captured = {}

            def fake_show(*args, **kwargs):
                ax = plt.gca()
                captured["calls"] = captured.get("calls", 0) + 1
                captured["title"] = ax.get_title()
                captured["xlabel"] = ax.get_xlabel()
                captured["ylabel"] = ax.get_ylabel()
                captured["n_points"] = len(ax.collections[0].get_offsets())

            with monkeypatch.context() as mp:
                mp.setattr(U.plt, "show", fake_show)
                U.process_masks(str(mask_dir), str(img_dir), **kw)
            for i, (values, counts) in case.get("want", {}).items():
                out = np.load(mask_dir / f"f{i}.npy")
                assert out.shape == case["shape"]
                assert out.dtype == np.int32
                assert _uniq(out) == values
                for lbl, n in counts.items():
                    assert np.count_nonzero(out == lbl) == n
            if "one_of" in case:
                assert _uniq(np.load(mask_dir / "f0.npy")) in case["one_of"]
            for i, im in enumerate(images):
                assert np.array_equal(np.load(img_dir / f"f{i}.npy"), im)
            if case.get("readme"):
                assert (mask_dir / "README.md").read_text(encoding="utf-8") == "not a mask"
            if "plot_points" in case:
                assert captured == dict(calls=1, title="Object Clustering",
                                        xlabel="PCA Component 1", ylabel="PCA Component 2",
                                        n_points=case["plot_points"])
            else:
                assert captured == {}


# ---------------------------------------------------------------------------
# _save_image
# ---------------------------------------------------------------------------

def test_save_image_writes_npy_by_extension_and_tiff_otherwise(subtests, tmp_path):
    """.npy is written with numpy; any other suffix is written as TIFF; _load_image reads both back."""
    cases = [
        dict(id="npy_roundtrip", name="mask.npy", img=np.arange(12, dtype=np.uint16)
             .reshape(3, 4), loads=True),
        dict(id="tif_roundtrip", name="mask.tif", img=(np.arange(9, dtype=np.uint16) * 7)
             .reshape(3, 3), loads=True),
        dict(id="png_falls_back_to_tiff", name="mask.png",
             img=np.arange(4, dtype=np.uint16).reshape(2, 2), loads=False),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            path = str(_case_dir(tmp_path, case["id"]) / case["name"])
            U._save_image(path, case["img"])
            assert os.path.exists(path)
            back = U._load_image(path)
            if case["loads"]:
                assert back.shape == case["img"].shape
                assert back.dtype == np.uint16
                assert np.array_equal(back, case["img"])
            else:
                assert back is None
            if case["name"].endswith(".npy"):
                assert np.array_equal(np.load(path), case["img"])
            else:
                assert np.array_equal(tifffile.imread(path), case["img"])


# ---------------------------------------------------------------------------
# _process_single_fov_in_memory / _process_single_fov / merge_split_objects
# ---------------------------------------------------------------------------

def _two_touching_blocks(size=16):
    """Two 10x6 rectangles sharing a 10-px edge (labels 1 and 2), 120 px total."""
    m = np.zeros((size, size), dtype=np.uint16)
    m[2:12, 2:8] = 1
    m[2:12, 8:14] = 2
    return m


def _big_square_plus_speck():
    """A 30x30 square (label 1) and a 2x2 speck (label 2)."""
    m = np.zeros((60, 60), dtype=np.uint16)
    m[10:40, 10:40] = 1
    m[50:52, 50:52] = 2
    return m


def _in_memory_kwargs(**overrides):
    """Neutral (all-phases-off) kwargs for ``_process_single_fov_in_memory``."""
    kw = dict(mask=None, intensity_img=None, intensity_channel=None,
              do_perimeter_merge=False, perimeter_fraction=0.5, min_area=0, max_area=0,
              remove_border_objects=False)
    kw.update(overrides)
    return kw


def _blocks_plane(values, shape=(16, 16), dtype=np.float64):
    m = _two_touching_blocks()
    img = np.zeros(shape, dtype=dtype)
    for lbl, v in values.items():
        img[m == lbl] = v
    return img


def _blocks_stack(channels, ch, values, dtype=np.uint16, first=False):
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, channels), dtype=dtype)
    for lbl, v in values.items():
        raw[m == lbl, ch] = v
    return np.moveaxis(raw, -1, 0) if first else raw


def _mean_not_max():
    img = _blocks_plane({1: 5})
    img[2, 8] = 1200                     # label 2: mean 20, maximum 1200
    return img


def _short_axis_mask():
    m = np.zeros((3, 16), np.uint16)
    m[1, 2:6] = 1
    m[1, 10:14] = 2
    return m


def _short_axis_raw():
    m = _short_axis_mask()
    raw = np.zeros((3, 16, 6), np.uint16)
    raw[m == 1, 5] = 4
    raw[m == 2, 5] = 14
    return raw


def _square_mask():
    m = np.zeros((16, 16), np.uint16)
    m[3:9, 3:9] = 1
    return m


def _square_raw():
    raw = np.zeros((16, 16, 3), np.uint16)
    raw[..., 1] = 9
    return raw


def _touching_4x4():
    t = np.zeros((16, 16), np.uint16)
    t[2:6, 2:6] = 1
    t[2:6, 6:10] = 2
    return t


EXACT = 2**24 + 1   # float32 would round this below an inclusive bound


def test_process_single_fov_in_memory_merges_and_filters_one_field(subtests, capsys):
    """Perimeter merge, area and own-channel mean-intensity bounds give the exact label image; bad intensity input is refused."""
    blocks = _two_touching_blocks()
    only1 = (blocks == 1).astype(np.uint16)
    only2 = (blocks == 2).astype(np.uint16)
    cases = [
        dict(id="none_mask", kw=dict(mask=None), want=None),
        dict(id="empty_mask", kw=dict(mask=np.zeros((12, 12), np.int32),
                                      do_perimeter_merge=True, fov_index=7),
             want=np.zeros((12, 12), np.uint16), says=["empty mask"]),
        dict(id="perimeter_merge_with_progress",
             kw=dict(mask=blocks, do_perimeter_merge=True, perimeter_fraction=0.5,
                     fov_index=3, total_fovs=9, op_name="merge_cell"),
             want=(blocks > 0).astype(np.uint16), progress=(3, 9, "merge_cell")),
        dict(id="mean_not_maximum", kw=dict(mask=blocks, intensity_img=_mean_not_max(),
                                            min_intensity=10, max_intensity=25,
                                            fov_index=2),
             want=only2, says=["Intensity filter: removed 1"]),
        dict(id="channel_last_raw_precision",
             kw=dict(mask=blocks, intensity_channel=1, min_intensity=EXACT,
                     max_intensity=EXACT,
                     intensity_img=_blocks_stack(3, 1, {1: EXACT, 2: EXACT + 2},
                                                 np.uint32)),
             want=only1, raw_unchanged=True),
        dict(id="channel_last_out_of_bounds",
             kw=dict(mask=blocks, intensity_img=np.zeros((16, 16, 3), np.uint16),
                     intensity_channel=5, min_intensity=1),
             raises=(IndexError, "out of bounds")),
        dict(id="channel_first_refused",
             kw=dict(mask=blocks, intensity_channel=1, min_intensity=5,
                     intensity_img=_blocks_stack(3, 1, {1: 9, 2: 3}, first=True)),
             raises=(ValueError, "same shape")),
        dict(id="channel_first_moved_to_last",
             kw=dict(mask=blocks, intensity_channel=1, min_intensity=5,
                     intensity_img=_blocks_stack(3, 1, {1: 9, 2: 3})),
             want=only1),
        dict(id="short_spatial_axis_not_channels",
             kw=dict(mask=_short_axis_mask(), intensity_img=_short_axis_raw(),
                     intensity_channel=5, min_intensity=7),
             want=(_short_axis_mask() == 2).astype(np.uint16)),
        dict(id="six_channel_stack_last_axis",
             kw=dict(mask=blocks, intensity_channel=5, min_intensity=10,
                     max_intensity=12, intensity_img=_blocks_stack(6, 5, {1: 5, 2: 11})),
             want=only2),
        dict(id="six_channel_out_of_bounds",
             kw=dict(mask=blocks, intensity_img=np.zeros((16, 16, 6), np.uint16),
                     intensity_channel=10, min_intensity=1),
             raises=(IndexError, "out of bounds")),
        dict(id="multichannel_needs_explicit_channel",
             kw=dict(mask=_square_mask(), intensity_img=_square_raw(), min_intensity=8),
             raises=(ValueError, "explicit own-channel index")),
        dict(id="multichannel_explicit_channel",
             kw=dict(mask=_square_mask(), intensity_img=_square_raw(), intensity_channel=1,
                     min_intensity=8, max_intensity=10),
             want=_square_mask()),
        dict(id="area_filter_drops_speck", kw=dict(mask=_big_square_plus_speck(),
                                                   min_area=100),
             want=(_big_square_plus_speck() == 1).astype(np.uint16)),
        dict(id="2d_intensity_with_channel_0",
             kw=dict(mask=blocks, intensity_img=np.where(blocks == 2, 15.0, 5.0).astype(np.float32),
                     intensity_channel=0, min_intensity=10, max_intensity=20,
                     perimeter_fraction=0.0),
             want=only2),
        # a (0, 0) field: the zero label count returns before any merge or filter
        dict(id="zero_sized_field",
             kw=dict(mask=np.zeros((0, 0), np.uint16), do_perimeter_merge=True,
                     perimeter_fraction=0.1),
             want=np.zeros((0, 0), np.uint16), silent=["merge:"]),
        dict(id="zero_sized_field_with_bound",
             kw=dict(mask=np.zeros((0, 0), np.uint16), do_perimeter_merge=True,
                     perimeter_fraction=0.1, min_intensity=250),
             want=np.zeros((0, 0), np.uint16), silent=["Intensity filter:"]),
        dict(id="labels_present_merge_reports",
             kw=dict(mask=_touching_4x4(), do_perimeter_merge=True, perimeter_fraction=0.1,
                     intensity_img=np.full((16, 16), 300), min_intensity=250),
             want=(_touching_4x4() > 0).astype(np.uint16), says=["merge: 2 → 1 objects"]),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            capsys.readouterr()
            calls = []
            kw = _in_memory_kwargs(**case["kw"])
            if "progress" in case:
                kw["progress_callback"] = lambda i, t, d, n: calls.append((i, t, d, n))
            img = kw.get("intensity_img")
            raw_before = None if img is None else np.array(img, copy=True)
            if "raises" in case:
                exc, match = case["raises"]
                with pytest.raises(exc, match=match):
                    U._process_single_fov_in_memory(**kw)
                continue
            out = U._process_single_fov_in_memory(**kw)
            if case["want"] is None:
                assert out is None
                continue
            assert out.dtype == np.uint16
            assert out.shape == case["want"].shape
            np.testing.assert_array_equal(out, case["want"])
            printed = capsys.readouterr().out
            for frag in case.get("says", []):
                assert frag in printed
            for frag in case.get("silent", []):
                assert frag not in printed
            if "progress" in case:
                assert len(calls) == 1
                idx, total, duration, name = calls[0]
                assert (idx, total, name) == case["progress"]
                assert duration >= 0.0
            if case.get("raw_unchanged"):
                np.testing.assert_array_equal(img, raw_before)
    # the union-find the merge phase ends with cannot be built over an empty array,
    # which is why the zero-label early return above matters
    with subtests.test(case="union_find_refuses_empty"):
        with pytest.raises(ValueError):
            U._apply_union_find(np.zeros((0, 0), np.uint16), {})


def _write(path, arr):
    if str(path).endswith(".npy"):
        np.save(path, arr)
    else:
        tifffile.imwrite(str(path), arr)


def _read(path):
    return np.load(path) if str(path).endswith(".npy") else tifffile.imread(str(path))


def test_process_single_fov_rewrites_the_mask_file_with_the_filtered_field(subtests,
                                                                           tmp_path):
    """The field file is replaced by the in-memory filter's result; unreadable or refused input leaves it untouched."""
    blocks = _two_touching_blocks()
    only1 = (blocks == 1).astype(np.uint16)
    only2 = (blocks == 2).astype(np.uint16)
    merged = (blocks > 0).astype(np.uint16)
    occupied = np.zeros((16, 16), np.uint16)
    occupied[2:6, 2:6] = 1
    occupied[10:14, 10:14] = 2
    ch_first = np.zeros((3, 16, 16), np.float32)
    ch_first[0, blocks == 1] = 3
    ch_first[0, blocks == 2] = 7
    plane_2d = np.full(blocks.shape, 5.0, np.float32)
    plane_2d[blocks == 2] = 15
    cases = [
        dict(id="unreadable_mask", mask_name="mask.png", mask=b"junk", want=None),
        dict(id="npy_merge_with_progress", mask_name="mask.npy", mask=blocks,
             args=dict(perimeter_fraction=0.5),
             kw=dict(fov_index=4, total_fovs=5, op_name="merge_nucleus"),
             want=merged, progress=(4, 5, "merge_nucleus")),
        dict(id="empty_field_written_back", mask_name="empty.npy",
             mask=np.zeros((16, 16), np.uint16), args=dict(perimeter_fraction=0.5),
             kw=dict(fov_index=0, total_fovs=2, op_name="merge"),
             want=np.zeros((16, 16), np.uint16), progress=(0, 2, "merge")),
        dict(id="occupied_disjoint_field", mask_name="occupied.npy", mask=occupied,
             args=dict(perimeter_fraction=0.5),
             kw=dict(fov_index=1, total_fovs=2, op_name="merge"),
             want=occupied, progress=(1, 2, "merge")),
        dict(id="channel_first_intensity_refused", mask_name="mask.tif", mask=blocks,
             int_name="intensity.tif", intensity=ch_first,
             args=dict(intensity_channel=0, do_perimeter_merge=False,
                       perimeter_fraction=0.0),
             kw=dict(min_intensity=5), raises="same shape"),
        dict(id="channel_last_intensity", mask_name="mask.tif", mask=blocks,
             int_name="intensity.tif", intensity=np.moveaxis(ch_first, 0, -1),
             args=dict(intensity_channel=0, do_perimeter_merge=False,
                       perimeter_fraction=0.0),
             kw=dict(min_intensity=5), want=only2),
        dict(id="whole_plane_without_channel", mask_name="mask.tif", mask=blocks,
             int_name="intensity.tif", intensity=_blocks_plane({1: 10.0, 2: 200.0},
                                                               dtype=np.float32),
             args=dict(intensity_channel=None, do_perimeter_merge=False,
                       perimeter_fraction=0.0),
             kw=dict(min_intensity=100, max_intensity=300), want=only2),
        dict(id="unreadable_intensity_unused", mask_name="mask.tif", mask=blocks,
             int_name="intensity.png", intensity=b"junk",
             args=dict(intensity_channel=0, perimeter_fraction=0.5), kw={}, want=merged),
        dict(id="unreadable_intensity_needed", mask_name="mask.tif", mask=blocks,
             int_name="intensity.png", intensity=b"junk",
             args=dict(intensity_channel=0, perimeter_fraction=0.5),
             kw=dict(min_intensity=1), raises="intensity plane.*same shape"),
        dict(id="2d_intensity_channel_0", mask_name="mask.tif", mask=blocks,
             int_name="intensity.tif", intensity=plane_2d,
             args=dict(intensity_channel=0, do_perimeter_merge=False,
                       perimeter_fraction=0.0),
             kw=dict(min_intensity=10, max_intensity=20), want=only2, same_in_memory=True),
        dict(id="uint32_channel_last_exact", mask_name="mask.tif", mask=blocks,
             int_name="intensity.tif",
             intensity=_blocks_stack(3, 1, {1: EXACT, 2: EXACT + 2}, np.uint32),
             args=dict(intensity_channel=1, do_perimeter_merge=False,
                       perimeter_fraction=0.0),
             kw=dict(min_intensity=EXACT, max_intensity=EXACT), want=only1,
             same_in_memory=True),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            d = _case_dir(tmp_path, case["id"])
            mask_path = d / case["mask_name"]
            if isinstance(case["mask"], bytes):
                mask_path.write_bytes(case["mask"])
            else:
                _write(mask_path, case["mask"])
            int_path = None
            if "int_name" in case:
                int_path = d / case["int_name"]
                if isinstance(case["intensity"], bytes):
                    int_path.write_bytes(case["intensity"])
                else:
                    _write(int_path, case["intensity"])
            args = dict(intensity_channel=None, do_perimeter_merge=True,
                        perimeter_fraction=0.5, min_area=0, max_area=0,
                        remove_border_objects=False)
            args.update(case.get("args", {}))
            calls = []
            kw = dict(case.get("kw", {}))
            if "progress" in case:
                kw["progress_callback"] = lambda *a: calls.append(a)
            before = mask_path.read_bytes()
            int_before = int_path.read_bytes() if int_path else None
            call = (str(mask_path), None if int_path is None else str(int_path),
                    args["intensity_channel"], args["do_perimeter_merge"],
                    args["perimeter_fraction"], args["min_area"], args["max_area"],
                    args["remove_border_objects"])
            if "raises" in case:
                with pytest.raises(ValueError, match=case["raises"]):
                    U._process_single_fov(*call, **kw)
                assert mask_path.read_bytes() == before
                continue
            assert U._process_single_fov(*call, **kw) is None
            if case["want"] is None:
                assert mask_path.read_bytes() == before
                continue
            out = _read(mask_path)
            assert out.dtype == np.uint16
            np.testing.assert_array_equal(out, case["want"])
            assert sorted(os.listdir(d)) == sorted(
                p for p in (case["mask_name"], case.get("int_name")) if p)
            if int_before is not None:
                assert int_path.read_bytes() == int_before
            if "progress" in case:
                assert len(calls) == 1
                assert (calls[0][0], calls[0][1], calls[0][3]) == case["progress"]
            if case.get("same_in_memory"):
                expected = U._process_single_fov_in_memory(**_in_memory_kwargs(
                    mask=case["mask"], intensity_img=case["intensity"],
                    intensity_channel=args["intensity_channel"],
                    perimeter_fraction=0.0, **case["kw"]))
                np.testing.assert_array_equal(out, expected)


def test_merge_split_objects_filters_every_mask_in_the_folder_in_place(subtests, tmp_path):
    """Each .tif/.npy mask is filtered in place (with the matching intensity image); other files ignored."""
    blocks = _two_touching_blocks()
    merged = (blocks > 0).astype(np.uint16)
    raw = np.zeros((16, 16, 3), np.float32)
    raw[blocks == 1, 2] = 3
    raw[blocks == 2, 2] = 9
    speck = _big_square_plus_speck()
    cases = [
        dict(id="no_masks_noop", masks={}, others={"notes.txt": "no masks here"},
             kw={}, want={}, returns_none=True),
        dict(id="every_mask_in_place",
             masks={"A01_f01.tif": blocks, "A01_f02.npy": blocks},
             others={"README.md": "ignored by the extension filter"},
             kw=dict(perimeter_fraction=0.5, n_jobs=1, op_name="merge_cell"),
             want={"A01_f01.tif": merged, "A01_f02.npy": merged}, progress=True),
        dict(id="with_intensity_directory", masks={"A01_f01.tif": blocks},
             intensity={"A01_f01.tif": raw},
             kw=dict(intensity_channel=2, perimeter_fraction=0.0, min_intensity=6,
                     max_intensity=10, n_jobs=1),
             want={"A01_f01.tif": (blocks == 2).astype(np.uint16)}),
        dict(id="area_filter", masks={"d.tif": speck},
             kw=dict(perimeter_fraction=0.0, min_area=20, n_jobs=1),
             want={"d.tif": (speck == 1).astype(np.uint16)}, min_area=20),
    ]
    for case in cases:
        with subtests.test(case=case["id"]):
            root = _case_dir(tmp_path, case["id"])
            mask_dir = root / "masks"
            mask_dir.mkdir()
            for name, arr in case["masks"].items():
                _write(mask_dir / name, arr)
            for name, text in case.get("others", {}).items():
                (mask_dir / name).write_text(text, encoding="utf-8")
            kw = dict(case["kw"])
            if "intensity" in case:
                int_dir = root / "intensity"
                int_dir.mkdir()
                for name, arr in case["intensity"].items():
                    _write(int_dir / name, arr)
                kw["intensity_img_src"] = str(int_dir)
            calls = []
            if case.get("progress"):
                kw["progress_callback"] = lambda *a: calls.append(a)
            result = U.merge_split_objects(str(mask_dir), **kw)
            if case.get("returns_none"):
                assert result is None
            assert sorted(p.name for p in mask_dir.iterdir()) == sorted(
                list(case["masks"]) + list(case.get("others", {})))
            for name, want in case["want"].items():
                out = _read(mask_dir / name)
                assert out.dtype == np.uint16
                np.testing.assert_array_equal(out, want)
                if "min_area" in case:
                    _, counts = np.unique(out[out > 0], return_counts=True)
                    assert bool(np.all(counts >= case["min_area"]))
            for name, text in case.get("others", {}).items():
                assert (mask_dir / name).read_text(encoding="utf-8") == text
            if case.get("progress"):
                assert [c[0] for c in calls] == [0, 1]
                assert {c[1] for c in calls} == {2}
                assert {c[3] for c in calls} == {"merge_cell"}
