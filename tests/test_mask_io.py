"""Tests for spacr.mask_io — TIFF/npy read+write for Cellpose masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _fake_mask() -> np.ndarray:
    """A tiny synthetic labelled-mask array."""
    m = np.zeros((32, 32), dtype=np.uint16)
    m[6:10, 6:10] = 1
    m[15:20, 15:20] = 2
    return m


@pytest.mark.parametrize("name, kwargs, suffix", [
    pytest.param("foo", {}, ".tif", id="default_is_tiff"),
    pytest.param("foo", {"fmt": "npy"}, ".npy", id="explicit_npy"),
    # User passed .npy in the path: it wins over the default tif.
    pytest.param("foo.npy", {}, ".npy", id="extension_overrides_fmt"),
])
def test_save_mask_writes_the_format_it_was_asked_for(tmp_path: Path, name,
                                                      kwargs, suffix):
    from spacr.mask_io import save_mask
    p = save_mask(tmp_path / name, _fake_mask(), **kwargs)
    assert p.suffix == suffix
    assert p.is_file()


def test_load_mask_finds_tif_by_stem(tmp_path: Path):
    from spacr.mask_io import save_mask, load_mask
    mask = _fake_mask()
    save_mask(tmp_path / "bar", mask, fmt="tif")
    loaded = load_mask(tmp_path / "bar")   # no suffix
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, mask)


def test_load_mask_finds_npy_by_stem(tmp_path: Path):
    from spacr.mask_io import save_mask, load_mask
    mask = _fake_mask()
    save_mask(tmp_path / "baz", mask, fmt="npy")
    loaded = load_mask(tmp_path / "baz")
    assert np.array_equal(loaded, mask)


def test_load_mask_prefers_tif_when_both_exist(tmp_path: Path):
    from spacr.mask_io import save_mask, load_mask
    mask = _fake_mask()
    # Save TIFF with different content vs NPY so we can tell which
    # one the loader picked.
    save_mask(tmp_path / "clash", mask, fmt="tif")
    save_mask(tmp_path / "clash", mask * 2, fmt="npy")
    loaded = load_mask(tmp_path / "clash")
    # tif takes priority — content should match original mask, not *2
    assert loaded.max() == mask.max()


def test_load_mask_raises_when_missing(tmp_path: Path):
    from spacr.mask_io import load_mask
    with pytest.raises(FileNotFoundError):
        load_mask(tmp_path / "nothing_here")


def test_roundtrip_preserves_uint16(tmp_path: Path):
    from spacr.mask_io import save_mask, load_mask
    mask = (np.random.randint(0, 500, size=(64, 64))
             .astype(np.uint16))
    for fmt in ("tif", "npy"):
        p = save_mask(tmp_path / f"round_{fmt}", mask, fmt=fmt)
        loaded = load_mask(p)
        assert loaded.dtype == np.uint16
        assert np.array_equal(loaded, mask)
