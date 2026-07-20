"""
Tests using the public Hugging Face datasets the user maintains:

  einarolafsson/toxo_mito       real 4-channel CellVoyager microscopy
  einarolafsson/spacr_settings  spacr's reference settings CSVs

Every test here is marked @pytest.mark.network so it skips cleanly in
offline CI. The fixtures download deterministic slices (one field, four
channels; two CSVs) — total < 20 MB.
"""
from __future__ import annotations

import csv
import os
import re

import pytest


pytestmark = pytest.mark.network


# ---------------------------------------------------------------------------
# 1. The Hugging Face TIFFs match spacr's Yokogawa CellVoyager regex.
# ---------------------------------------------------------------------------

def test_hf_toxo_mito_field_downloads_four_channels(hf_toxo_mito_field):
    assert len(hf_toxo_mito_field["files"]) == 4
    for p in hf_toxo_mito_field["files"]:
        assert os.path.exists(p)
        # File is a TIFF (starts with II* or MM*).
        with open(p, "rb") as f:
            magic = f.read(4)
        assert magic[:2] in (b"II", b"MM"), f"{p} does not look like a TIFF"


def test_hf_toxo_mito_matches_cellvoyager_regex(hf_toxo_mito_field):
    from spacr.utils import _get_regex

    regex = re.compile(_get_regex("cellvoyager", "tif", None))
    for p in hf_toxo_mito_field["files"]:
        fname = os.path.basename(p)
        m = regex.match(fname)
        assert m is not None, f"CellVoyager regex failed on real HF file {fname!r}"
        assert m.group("plateID") == hf_toxo_mito_field["plate"]
        assert m.group("wellID") == hf_toxo_mito_field["well"]
        assert m.group("fieldID") == hf_toxo_mito_field["field"]


def test_hf_toxo_mito_metadata_extractor_groups_by_channel(hf_toxo_mito_field):
    from spacr.utils import _get_regex, _extract_filename_metadata

    src = hf_toxo_mito_field["src"]
    regex = re.compile(_get_regex("cellvoyager", "tif", None))
    filenames = sorted(os.listdir(src))
    grouped = _extract_filename_metadata(
        filenames, src, regex, metadata_type="cellvoyager"
    )
    # One well, one field → the number of groups equals the number of channels
    # in the download (4).
    assert len(grouped) == 4
    # _extract_filename_metadata normalizes numeric channels via
    # _safe_int_convert, so "01" becomes "1".
    channels_seen = {k[3] for k in grouped.keys()}
    assert channels_seen == {"1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# 2. The reference settings CSVs load and expose the columns spacr expects.
# ---------------------------------------------------------------------------

def test_hf_spacr_settings_csv_files_download(hf_spacr_settings):
    for name, path in hf_spacr_settings.items():
        assert os.path.exists(path)
        with open(path, "r") as f:
            head = f.readline()
        # spacr's own save_settings uses 'Key' / 'Value' columns.
        assert "Key" in head and "Value" in head, (
            f"{name}: unexpected CSV header {head!r}"
        )


def test_hf_gen_masks_settings_loadable_via_spacr(hf_spacr_settings):
    """spacr.utils.load_settings should ingest the reference settings CSV."""
    from spacr.utils import load_settings

    path = hf_spacr_settings["gen_masks_settings.csv"]
    loaded = load_settings(path, setting_key="Key", setting_value="Value")
    assert loaded is not None
    # The gen_masks settings should include the standard channel wiring.
    if hasattr(loaded, "keys"):
        keys = list(loaded.keys())
    else:  # pandas DataFrame path
        keys = list(loaded.index)
    common = {"src", "cell_channel", "nucleus_channel", "pathogen_channel"}
    matched = common & set(keys)
    assert matched, (
        f"expected at least one of {common} in gen_masks_settings, got {keys[:20]}"
    )


# ---------------------------------------------------------------------------
# 3. GPU + real HF data: run CellposeSAM on one real channel image.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_cellposesam_on_hf_toxo_mito_field(hf_toxo_mito_field):
    """Run CellposeSAM on one real HF-hosted CellVoyager TIFF."""
    import numpy as np
    import torch
    try:
        from tifffile import imread
        from cellpose.models import CellposeModel
    except Exception as e:
        pytest.skip(f"cellpose or tifffile missing: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")

    # Pick the first (nucleus channel, C01 by cellvoyager convention).
    path = next(p for p in hf_toxo_mito_field["files"] if "C01" in os.path.basename(p))
    img = imread(path)
    if img.ndim > 2:
        img = img[..., 0]
    assert img.ndim == 2

    device = torch.device("cuda:0")
    model = CellposeModel(gpu=True, pretrained_model='cpsam', device=device)
    result = model.eval(img.astype(np.float32), diameter=30)
    masks = result[0]
    assert masks.shape == img.shape
    assert masks.dtype.kind in "iu"
    assert masks.min() >= 0
    # For a real microscopy image with visible nuclei, we should find at
    # least one object.
    n_objs = len([i for i in np.unique(masks) if i != 0])
    assert n_objs >= 1, "CellposeSAM found zero objects in a real nucleus image"
