"""Item 1041: the faster f_* background, QC tables and barcode scan give the
numbers of the per-peak and per-read reference code they replaced."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from spacr import ops_decode
from spacr.ops_sbs import LibraryMatcher, called_bases

from test_ops_decode_qc_876_877 import _qc_fields


def _reference_background(stack, py, px, size):
    """The per-image whole-image filter the f_bg step used before item 1041."""
    return np.stack([np.stack([
        ndimage.minimum_filter(stack[i, k], size)[py, px]
        for k in range(stack.shape[1])], 1) for i in range(stack.shape[0])], 1)


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_the_background_at_the_peaks_equals_the_whole_image_filter(dtype):
    rng = np.random.default_rng(1)
    stack = (rng.random((3, 4, 90, 70)) * 5000).astype(dtype)
    py = rng.integers(4, 86, 300)
    px = rng.integers(4, 66, 300)
    got = ops_decode._local_minimum_at(stack, py, px, 9, ndimage)
    want = _reference_background(stack, py, px, 9)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.tobytes() == want.tobytes()


def test_the_background_falls_back_near_the_border_and_with_nan():
    rng = np.random.default_rng(2)
    stack = (rng.random((2, 4, 40, 40)) * 100).astype(np.float32)
    py, px = np.array([1, 20, 38]), np.array([2, 20, 39])
    got = ops_decode._local_minimum_at(stack, py, px, 9, ndimage)
    assert got.tobytes() == _reference_background(stack, py, px, 9).tobytes()
    stack[0, 1, 10:12, 10:12] = np.nan
    py, px = np.array([10, 20]), np.array([11, 20])
    got = ops_decode._local_minimum_at(stack, py, px, 9, ndimage)
    want = _reference_background(stack, py, px, 9)
    assert np.array_equal(got, want, equal_nan=True)


def test_the_packed_barcode_scan_equals_the_letter_by_letter_scan():
    rng = np.random.default_rng(3)
    library = ["".join(rng.choice(list("ACGT"), 11)) for _ in range(800)]
    matcher = LibraryMatcher(library)
    for _ in range(600):
        read = list(library[rng.integers(len(library))])
        for _ in range(rng.integers(0, 5)):
            read[rng.integers(11)] = rng.choice(list("ACGTNX"))
        read = "".join(read)
        letters = np.frombuffer(read.encode(), np.uint8)
        want = ((matcher.table != letters) & (letters != ord("N"))).sum(1)
        assert np.array_equal(matcher._scan_distance(read), want)
    assert matcher._packed is not False
    other = LibraryMatcher(["ACGT", "ACGN"])
    assert np.array_equal(other._scan_distance("ACGA"), [1, 1])
    assert other._packed is False


def test_the_false_call_table_numbers_equal_the_per_read_reference():
    guides, decoys, fields = _qc_fields(n_fields=4)
    z = fields[0]["z"]
    z["f_raw"] = np.array(z["f_raw"], np.float32)
    z["f_bg"] = np.array(z["f_bg"], np.float32)
    z["f_raw"][3, 2, :] = np.nan           # a cycle with no finite raw read
    z["f_bg"][5, 1, :] = np.nan
    z["f_bg"][6, 0, 1:] = np.nan           # a cycle with one finite value
    z["f_raw"][7, 4, 0] = np.inf
    reads, _ = ops_decode.false_call_tables(fields, guides, decoys, keep_reads=True)
    rule = ops_decode.QC_PERMISSIVE
    block = reads[reads["field"] == 0].reset_index(drop=True)
    selected = np.asarray(z["strength"]) > rule["threshold"]
    first, _, _ = called_bases(np.asarray(z["values"])[selected], normalise=True, gpu=False)
    _, _, margin = called_bases(first, compensate=True, method="argmax_median",
                                normalise=False, gpu=False)
    raw, background = np.asarray(z["f_raw"])[selected], np.asarray(z["f_bg"])[selected]
    assert len(block) == len(margin)
    for i in range(len(block)):
        m = margin[i]
        finite = np.isfinite(m).any()
        assert block.loc[i, "q"] == (float(np.nanmin(m)) if finite else 0.0)
        assert block.loc[i, "q_mean"] == (float(np.nanmean(m)) if finite else 0.0)
        for c in range(m.shape[0]):
            assert block.loc[i, f"margin_c{c + 1}"] == float(np.nan_to_num(m[c], nan=-1))
            r, b = raw[i, c], background[i, c]
            assert block.loc[i, f"raw_max_c{c + 1}"] == (
                float(np.nanmax(r)) if np.isfinite(r).any() else -1)
            assert block.loc[i, f"bg_c{c + 1}"] == (
                float(np.nanmean(b)) if np.isfinite(b).any() else -1)
        assert block.loc[i, "radius_in_tile"] == float(np.hypot(
            z["peaks"][selected][i, 0] - 740.0, z["peaks"][selected][i, 1] - 740.0))
