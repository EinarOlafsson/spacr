"""Pairing and measuring one OPS phenotype field, on planted masks.

The masks are drawn here, so every count, label and centroid the functions
return is checked against what was drawn.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_sample import (SampleError, _measurement_settings,
                              measure_phenotype_field, pair_nuclei_with_cells)


def _two_cells():
    """Two square cells, each holding one square nucleus, and a two-channel
    image whose first channel is bright inside the nuclei."""
    cells = np.zeros((60, 60), np.int32)
    cells[2:28, 2:28] = 1
    cells[30:58, 30:58] = 2
    nuclei = np.zeros_like(cells)
    nuclei[8:18, 8:18] = 1
    nuclei[38:50, 38:50] = 2
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (60, 60, 2)).astype(np.float32)
    image[..., 0][nuclei > 0] += 5.0
    return nuclei, cells, image


def test_pairing_refuses_masks_of_different_shape():
    with pytest.raises(SampleError, match="differ in shape"):
        pair_nuclei_with_cells(np.zeros((4, 4), int), np.zeros((4, 5), int))


def test_pairing_with_no_overlap_drops_and_counts_everything():
    nuclei = np.zeros((10, 10), int)
    nuclei[0:2, 0:2] = 1
    nuclei[0:2, 4:6] = 2
    cells = np.zeros_like(nuclei)
    cells[6:9, 6:9] = 5
    paired_n, paired_c, counts = pair_nuclei_with_cells(nuclei, cells)
    assert paired_n.dtype == np.int32 and paired_c.dtype == np.int32
    assert not paired_n.any() and not paired_c.any()
    assert counts == {"nuclei": 2, "cells": 1, "paired": 0,
                      "nuclei_without_cell": 2, "cells_without_nucleus": 1,
                      "extra_nuclei_in_cell": 0}


def test_measurement_settings_place_the_masks_after_the_channels():
    settings = _measurement_settings(3, texture=False)
    assert settings["cell_mask_dim"] == 3
    assert settings["nucleus_mask_dim"] == 4
    assert settings["pathogen_mask_dim"] is None
    assert settings["channels"] == [0, 1, 2]
    assert settings["homogeneity"] is False
    assert _measurement_settings(1, texture=True)["homogeneity"] is True


def test_a_field_is_refused_when_the_image_is_not_over_the_masks():
    nuclei, cells, image = _two_cells()
    with pytest.raises(SampleError, match=r"must be \(H, W, C\)"):
        measure_phenotype_field(nuclei, cells, image[..., 0], ["DAPI"])
    with pytest.raises(SampleError, match=r"must be \(H, W, C\)"):
        measure_phenotype_field(nuclei, cells, image[:30], ["DAPI", "PHA"])


def test_a_field_is_refused_when_the_channel_names_do_not_match():
    nuclei, cells, image = _two_cells()
    with pytest.raises(SampleError, match="1 channel names for 2 channels"):
        measure_phenotype_field(nuclei, cells, image, ["DAPI"])


def test_a_field_without_nuclei_gives_an_empty_table():
    _nuclei, cells, image = _two_cells()
    out = measure_phenotype_field(np.zeros_like(cells), cells, image,
                                  ["DAPI", "PHA"])
    assert len(out) == 0
    assert list(out.columns) == ["label", "centroid_row", "centroid_col"]


def test_a_field_is_measured_per_cell_with_named_channels():
    nuclei, cells, image = _two_cells()
    out = measure_phenotype_field(nuclei, cells, image, ["DAPI", "Phalloidin"])
    assert out["label"].tolist() == [1, 2]
    assert np.allclose(out[["centroid_row", "centroid_col"]].to_numpy(),
                       [[12.5, 12.5], [43.5, 43.5]])
    columns = set(out.columns)
    assert {"cell_area", "nucleus_area", "cytoplasm_area"} <= columns
    assert not any("channel_" in c or "pathogen" in c for c in columns)
    assert not any(c.endswith("_cell_id") for c in columns)
    assert all(c in {"label", "centroid_row", "centroid_col"}
               or c.split("_", 1)[0] in {"cell", "nucleus", "cytoplasm"}
               for c in columns)
    assert out["nucleus_area"].tolist() == [100, 144]
    assert out["cell_area"].tolist() == [26 * 26, 28 * 28]
    assert out["cytoplasm_area"].tolist() == [26 * 26 - 100, 28 * 28 - 144]
    nucleus_dapi = out["nucleus_dapi_mean_intensity"].to_numpy()
    cytoplasm_dapi = out["cytoplasm_dapi_mean_intensity"].to_numpy()
    assert (nucleus_dapi > cytoplasm_dapi + 4.0).all()
    assert any("homogeneity" in c for c in columns)
    assert all(np.issubdtype(out[c].dtype, np.number) for c in out.columns)


def test_texture_can_be_left_out():
    nuclei, cells, image = _two_cells()
    out = measure_phenotype_field(nuclei, cells, image, ["DAPI", "PHA"],
                                  texture=False)
    assert out["label"].tolist() == [1, 2]
    assert not any("homogeneity" in c for c in out.columns)


def test_a_field_whose_nuclei_fill_their_cells_has_no_cytoplasm_rows():
    nuclei, cells, image = _two_cells()
    out = measure_phenotype_field(cells.copy(), cells, image, ["DAPI", "PHA"],
                                  texture=False)
    assert out["label"].tolist() == [1, 2]
    assert out["nucleus_area"].tolist() == out["cell_area"].tolist()
    cytoplasm = [c for c in out.columns if c.startswith("cytoplasm_")]
    assert not cytoplasm or out[cytoplasm].isna().all().all()
