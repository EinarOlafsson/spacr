"""Coverage-fill for spacr.spacr_cellpose — the CPU-reachable branches
(parse error paths + the mask-comparison helpers). The GPU model-build
functions are exercised by the @gpu suites.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import spacr_cellpose as SC


# ---------------------------------------------------------------------------
# parse_cellpose4_output — error branches
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_flows_wrong_type_raises(self):
        # flows not list/tuple → ValueError (line 32).
        with pytest.raises(ValueError):
            SC.parse_cellpose4_output(([np.zeros((4, 4))], "not-a-list"))

    def test_masks_length_undeterminable_raises(self):
        # masks with no len() → ValueError (lines 37-38).
        class _NoLen:
            pass
        with pytest.raises(ValueError):
            SC.parse_cellpose4_output((_NoLen(), [[], [], []]))

    def test_per_image_non_list_non_array_item(self):
        # A per-image flows item that is neither list nor ndarray → all
        # None branch (line 65).
        masks = [np.zeros((4, 4), dtype=np.int32)]
        flows = ["scalar-item"]   # len == num_images (1), item is str
        out = SC.parse_cellpose4_output((masks, flows))
        # returns (masks, f0, f1, f2, f3); f0 for the str item is None.
        assert out[0] is masks
        assert out[1] == [None]

    def test_unrecognized_structure_raises(self):
        # flows len != 4 and != num_images → ValueError (line 75).
        masks = [np.zeros((4, 4)), np.zeros((4, 4))]  # num_images = 2
        flows = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]  # len 3
        with pytest.raises(ValueError):
            SC.parse_cellpose4_output((masks, flows))

    def test_per_image_list_item_partial(self):
        # A per-image list item with <4 entries fills the rest with None.
        masks = [np.zeros((4, 4), dtype=np.int32)]
        flows = [[np.zeros((4, 4))]]   # one entry → f1,f2,f3 = None
        out = SC.parse_cellpose4_output((masks, flows))
        assert out[2] == [None]


# ---------------------------------------------------------------------------
# Mask-comparison helpers (CPU, synthetic .tif masks)
# ---------------------------------------------------------------------------

def _write_mask(path: Path, mask: np.ndarray):
    import tifffile
    tifffile.imwrite(str(path), mask.astype(np.uint16))


def _mask(size=32, blob=(8, 24)):
    m = np.zeros((size, size), dtype=np.uint16)
    m[blob[0]:blob[1], blob[0]:blob[1]] = 1
    return m


class TestCompareMask:
    def test_compare_mask_returns_metrics(self, tmp_path):
        d1 = tmp_path / "a"; d2 = tmp_path / "b"
        d1.mkdir(); d2.mkdir()
        _write_mask(d1 / "img.tif", _mask())
        _write_mask(d2 / "img.tif", _mask(blob=(9, 25)))
        out = SC.compare_mask(
            (str(tmp_path), "img.tif", [str(d1), str(d2)], ["a", "b"]))
        assert out is not None
        assert out["filename"] == "img.tif"
        # A jaccard key for the a-vs-b pair exists.
        assert any(k.startswith("jaccard_") for k in out)

    def test_compare_mask_missing_file_returns_none(self, tmp_path):
        d1 = tmp_path / "a"; d2 = tmp_path / "b"
        d1.mkdir(); d2.mkdir()
        _write_mask(d1 / "img.tif", _mask())
        # d2 lacks img.tif → None (line 373-374).
        out = SC.compare_mask(
            (str(tmp_path), "img.tif", [str(d1), str(d2)], ["a", "b"]))
        assert out is None


class TestSaveResultsAndFigure:
    def test_saves_csv_and_pdf(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        SC.save_results_and_figure(
            str(tmp_path), fig,
            [{"filename": "x", "jaccard_a_b": 0.9}])
        results_dir = tmp_path / "results"
        assert (results_dir / "results.csv").exists()
        assert (results_dir / "model_comparison_plot.pdf").exists()
        plt.close(fig)

    def test_accepts_dataframe(self, tmp_path):
        import matplotlib.pyplot as plt
        import pandas as pd
        fig, ax = plt.subplots()
        SC.save_results_and_figure(
            str(tmp_path), fig,
            pd.DataFrame([{"filename": "x", "ap_a_b": 0.5}]))
        assert (tmp_path / "results" / "results.csv").exists()
        plt.close(fig)


class TestCompareCellposeMasks:
    def test_end_to_end_two_conditions(self, tmp_path, monkeypatch):
        # Two sibling dirs with a common mask file → full compare +
        # plot + save. Patch multiprocessing to run in-process so the
        # coverage tracer sees compare_mask.
        d1 = tmp_path / "modelA"; d2 = tmp_path / "modelB"
        d1.mkdir(); d2.mkdir()
        _write_mask(d1 / "img.tif", _mask())
        _write_mask(d2 / "img.tif", _mask(blob=(9, 25)))

        class _SerialPool:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def map(self, fn, args): return [fn(a) for a in args]
        monkeypatch.setattr(SC, "Pool", _SerialPool)

        SC.compare_cellpose_masks(str(tmp_path), verbose=False, save=False)
        assert (tmp_path / "results" / "results.csv").exists()

    def test_verbose_renders_overlays(self, tmp_path, monkeypatch):
        d1 = tmp_path / "modelA"; d2 = tmp_path / "modelB"
        d1.mkdir(); d2.mkdir()
        _write_mask(d1 / "img.tif", _mask())
        _write_mask(d2 / "img.tif", _mask(blob=(10, 26)))

        class _SerialPool:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def map(self, fn, args): return [fn(a) for a in args]
        monkeypatch.setattr(SC, "Pool", _SerialPool)
        # Stub the overlay renderer so no windows / files are needed.
        import spacr.plot as PL
        monkeypatch.setattr(PL, "visualize_cellpose_masks",
                            lambda *a, **k: None)
        SC.compare_cellpose_masks(str(tmp_path), verbose=True, save=False)
        assert (tmp_path / "results" / "results.csv").exists()


def test_parse_batched_four_array_format():
    # Batched format: exactly 4 ndarray flows over the batch (lines 42-49).
    n = 2
    masks = np.zeros((n, 8, 8), dtype=np.int32)
    flow0 = np.zeros((n, 8, 8), dtype=np.float32)
    flow1 = np.zeros((3, n, 8, 8), dtype=np.float32)
    flow2 = np.zeros((n, 8, 8), dtype=np.float32)
    flow3 = np.zeros((n, 8, 8), dtype=np.float32)
    out = SC.parse_cellpose4_output((masks, [flow0, flow1, flow2, flow3]))
    m, f0, f1, f2, f3 = out
    assert len(f0) == n and len(f1) == n
