"""Item 965: R2 nuclei by default on the large-site path. Small nucleus
pieces are merged into the nucleus they touch, then one cell is seeded from
every nucleus; settings files stamped before 943 keep the old behaviour, and
Funk's phenotype path reads neither setting."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd

from spacr import ops_large_sites as ols
from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION
from tests.test_ops_large_sites_921 import _models, _site, _stand_in_masks, _write_well  # noqa: F401

KEYS = ("seg_cells_from_nucleus_seeds", "seg_merge_small_nuclei_frac")


def _toy():
    """The reference implementation's self-test image: five big nuclei, two
    small pieces touching a big one and one isolated small piece."""
    t = np.zeros((40, 30), np.int32)
    t[20:30, 2:12] = 6
    t[20:30, 16:26] = 7
    t[31:39, 2:12] = 8
    t[2:12, 2:12] = 1
    t[5:8, 12:14] = 2
    t[2:12, 16:26] = 3
    t[14:16, 16:18] = 4
    t[12:14, 20:22] = 5
    return t


def test_the_reference_self_test_passes():
    merged, info = ols.merge_small_nuclei(_toy(), 0.25)
    assert merged.dtype == np.int32 and merged.max() == 6
    assert info["n_merged"] == 2 and info["n_small"] == 3
    assert info["median_area_px"] == 90.0 and info["threshold_px"] == 22.5
    assert (merged[5:8, 12:14] == merged[3, 3]).all()
    assert (merged[12:14, 20:22] == merged[3, 20]).all()


def test_the_merge_never_deletes_and_keeps_an_isolated_piece():
    toy = _toy()
    merged, _ = ols.merge_small_nuclei(toy, 0.25)
    assert np.array_equal(merged > 0, toy > 0)
    assert merged[14, 16] > 0 and (merged == merged[14, 16]).sum() == 4


def test_a_chain_of_small_pieces_ends_on_the_kept_nucleus():
    lab = np.zeros((30, 60), np.int32)
    for k, x0 in enumerate((2, 14, 26, 38), start=1):
        lab[2:12, x0:x0 + 10] = k
    lab[20:28, 2:12] = 5
    lab[4:8, 48:50] = 6
    lab[4:6, 50:52] = 7
    merged, info = ols.merge_small_nuclei(lab, 0.25)
    assert info["n_merged"] == 2
    assert merged[4, 48] == merged[4, 50] == merged[5, 45]
    assert merged.max() == 5


def test_the_longest_border_wins_and_frac_zero_is_off():
    lab = np.zeros((20, 40), np.int32)
    lab[0:20, 0:10] = 1
    lab[0:20, 14:24] = 2
    lab[0:20, 28:38] = 3
    lab[5:7, 10:14] = 4
    lab[5:7, 13:14] = 4
    lab[4:8, 13:14] = 4
    merged, _ = ols.merge_small_nuclei(lab, 0.25)
    assert merged[5, 11] == merged[5, 15]
    same, info = ols.merge_small_nuclei(lab, 0)
    assert np.array_equal(same, lab) and info["n_merged"] == 0
    empty, info = ols.merge_small_nuclei(np.zeros((4, 4), np.int32), 0.25)
    assert not empty.any() and info["n_merged"] == 0


def test_the_relabel_is_deterministic_and_consecutive():
    toy = _toy()
    perm = np.array([0, 9, 3, 7, 1, 5, 2, 8, 4])
    first, _ = ols.merge_small_nuclei(perm[toy], 0.25)
    again, _ = ols.merge_small_nuclei(perm[toy], 0.25)
    assert np.array_equal(first, again)
    assert set(np.unique(first)) == set(range(first.max() + 1))


def test_the_large_site_defaults_are_r2_and_files_stamped_before_keep_the_old_values():
    assert SETTINGS_SCHEMA_VERSION >= 943
    assert DEFAULT_CHANGES[943] == {"seg_cells_from_nucleus_seeds": False,
                                    "seg_merge_small_nuclei_frac": 0.0}
    settings = ols.large_site_settings({})
    assert settings["seg_cells_from_nucleus_seeds"] is True
    assert settings["seg_merge_small_nuclei_frac"] == 0.25
    assert "seg_merge_small_nuclei_frac" in ols.LARGE_SITE_TOOLTIPS


def test_a_settings_file_stamped_942_resolves_to_the_old_large_site_defaults(tmp_path):
    from spacr.cli import MODULES, resolve_settings

    old = tmp_path / "old.json"
    old.write_text(json.dumps({"spacr_settings_version": 942,
                               "seg_preset": "cell_painting_phalloidin"}))
    resolved = ols.large_site_settings(resolve_settings(MODULES["ops_large_sites"], str(old)))
    assert [resolved[k] for k in KEYS] == [False, 0.0]
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"spacr_settings_version": 943,
                               "seg_preset": "cell_painting_phalloidin"}))
    resolved = ols.large_site_settings(resolve_settings(MODULES["ops_large_sites"], str(new)))
    assert [resolved[k] for k in KEYS] == [True, 0.25]


def test_the_site_merges_before_seeding_and_counts_the_pieces(monkeypatch):
    nuclei = np.zeros((40, 60), np.int32)
    nuclei[5:15, 5:15] = 1
    nuclei[5:15, 25:35] = 2
    nuclei[25:35, 5:15] = 3
    nuclei[8:11, 15:17] = 4
    cellprob = np.full(nuclei.shape, 1.0, np.float32)
    monkeypatch.setattr(ols, "segment_tiled", lambda image, *a, **k: nuclei.copy())
    monkeypatch.setattr(ols, "_site_cellprob", lambda image, *a, **k: cellprob)
    stack = np.zeros((40, 60, 2), np.float32)
    on = ols.large_site_settings({})
    n, c, counts = ols.segment_cell_painting_site(stack, ["CorrDNA", "CorrPhalloidin"], on,
                                                  (None, None))
    assert counts["nuclei"] == counts["paired"] == 3
    assert counts["merged_small_nuclei"] == 1 and counts["median_nucleus_area_px"] == 100.0
    assert n[9, 16] == n[9, 9] > 0
    off = dict(on, seg_merge_small_nuclei_frac=0.0)
    n, c, counts = ols.segment_cell_painting_site(stack, ["CorrDNA", "CorrPhalloidin"], off,
                                                  (None, None))
    assert counts["nuclei"] == counts["paired"] == 4
    assert "merged_small_nuclei" not in counts and "median_nucleus_area_px" not in counts


def test_the_run_records_the_merge_per_site_and_an_off_run_writes_the_old_table(tmp_path):
    stack, _ = _site()
    channels = ols.LARGE_SITE_DEFAULTS["seg_site_channels"]
    well, ref = _write_well(tmp_path, stack, channels)
    base = {"seg_preset": "cell_painting_phalloidin", "seg_sites_src": str(well),
            "seg_tile_px": 256, "seg_tile_overlap_px": 64, "seg_report_reference": str(ref),
            "seg_measure_band_px": 200, "seg_diameter_min_objects": 3}
    on = ols.run_large_site_segmentation(dict(base, seg_dst=str(tmp_path / "on")),
                                         models=_models(), available_bytes=2 ** 40)
    table = pd.read_csv(tmp_path / "on" / "sites.csv")
    assert {"merged_small_nuclei", "median_nucleus_area_px"} <= set(table.columns)
    assert on["merge_line"].startswith("R2: merged ") and "% of nuclei)" in on["merge_line"]
    off = ols.run_large_site_segmentation(
        dict(base, seg_dst=str(tmp_path / "off"), seg_cells_from_nucleus_seeds=False,
             seg_merge_small_nuclei_frac=0.0), models=_models(), available_bytes=2 ** 40)
    assert "merge_line" not in off
    assert "merged_small_nuclei" not in pd.read_csv(tmp_path / "off" / "sites.csv").columns


def test_funks_phenotype_path_reads_neither_setting():
    from spacr import ops_engine

    source = inspect.getsource(ops_engine._ph_segment_field)
    assert not any(key in source for key in KEYS)
    engine = inspect.getsource(ops_engine)
    # Items 1028/1029 add ops_ph_merge_small_nuclei_px, an opt-in OPS option of
    # the shared pairing (off by default); the large-site merge itself stays out.
    assert "merge_small_nuclei(" not in engine
    assert "seg_merge_small_nuclei_frac" not in engine
