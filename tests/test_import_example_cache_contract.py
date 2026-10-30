"""Cached import variants keep their real images, masks and measurement paths."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import import_examples as ix


def test_default_import_cache_resolves_real_members_and_refuses_unknown_variants(
        tmp_path, monkeypatch):
    """A misspelled chooser key cannot produce plausible paths into another variant."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = ix.import_example_folder()
    assert root == tmp_path / ".cache/spacr/example_data/plate1/import_example"
    assert not root.exists()
    variant = ix.import_variant("cellvoyager")
    folder = root / "variants" / variant.key
    image = folder / "plate1" / variant.example
    image.parent.mkdir(parents=True)
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    tifffile.imwrite(image, pixels)
    mask_folder = folder / "masks" / "cell"
    mask_folder.mkdir(parents=True)
    tifffile.imwrite(mask_folder / image.name, np.ones_like(pixels))
    database = folder / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame({"file_name": [image.name], "object_label": [1],
                      "area": [20]}).to_sql("cell", connection, index=False)
    manifest = root / "manifest.csv"
    pd.DataFrame({"variant": [variant.key],
                  "path": [str(image.relative_to(root))]}).to_csv(manifest, index=False)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    inputs = ix.variant_inputs(root, variant.key)

    assert ix.is_present(root, variant.key)
    assert Path(inputs["images"]) / variant.example == image
    assert inputs["masks"] == {"cell": str(mask_folder)}
    assert inputs["measurements"] == str(database)
    assert inputs["metadata_type"] == "cellvoyager"
    assert inputs["route"] == "import" and inputs["metadata_well_map"] == ""
    np.testing.assert_array_equal(tifffile.imread(image), pixels)
    with pytest.raises(KeyError, match="no import test variant named 'cellvoyager-typo'") as error:
        ix.variant_inputs(root, "cellvoyager-typo")
    assert all(entry.key in str(error.value) for entry in ix.IMPORT_VARIANTS)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
