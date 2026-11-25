"""ph_rebuild: ownership, links, gene and phenotype tables from stored masks.

The paper session rebuilds A1 in a new run folder whose ``phenotype_masks``
and ``decode_cache`` are hard links into the original run, so a write in
place there would change the original. These tests run ``ph_rebuild`` on
such a folder and check that the linked files stay byte-identical, and that
ph_tables fills the gene of a linked cell without an aggregate cell table.
"""
from __future__ import annotations

import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_engine
from tests.test_ops_phenotype_run_v2 import (PLATE, SETTINGS, measured,  # noqa: F401
                                             models)

LINKED = ("phenotype_masks", "decode_cache")


def _settings(tmp_path, run, **extra):
    return {**SETTINGS, "dst_root": str(run),
            "phenotype_source": str(tmp_path / "tiles"),
            "ops_design": str(tmp_path / "design.csv"), **extra}


def _digests(root):
    """``{relative path: (sha256, inode)}`` of every file under ``root``."""
    out = {}
    for folder, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            out[os.path.relpath(path, root)] = (digest, os.stat(path).st_ino)
    return out


def _copy_with_links(source, target):
    """``source`` copied to ``target``, the masks and decode cache hard-linked."""
    os.makedirs(target)
    for name in os.listdir(source):
        here = os.path.join(source, name)
        if name != "A1":
            (shutil.copytree if os.path.isdir(here) else shutil.copy2)(
                here, os.path.join(target, name))
            continue
        os.makedirs(os.path.join(target, "A1"))
        for part in os.listdir(here):
            inner = os.path.join(here, part)
            out = os.path.join(target, "A1", part)
            if part in LINKED:
                shutil.copytree(inner, out, copy_function=os.link)
            elif os.path.isdir(inner):
                shutil.copytree(inner, out)
            else:
                shutil.copy2(inner, out)


def test_ph_rebuild_leaves_hard_linked_masks_and_decode_cache_byte_identical(measured):
    run = measured / "run"
    ops_engine.run_ops(_settings(measured, run),
                       phases=("ph_masks", "merge", "aggregate", "ph_tables"))
    cache = run / "A1" / "decode_cache"
    cache.mkdir()
    np.savez(cache / "site0000.npz", reads=np.arange(12, dtype=np.uint16))
    before = {part: _digests(run / "A1" / part) for part in LINKED}
    assert all(before.values())
    rebuilt = measured / "rebuilt"
    _copy_with_links(str(run), str(rebuilt))
    for part in LINKED:
        assert _digests(rebuilt / "A1" / part) == before[part]
    out = ops_engine.run_ops(_settings(measured, rebuilt), phases=("ph_rebuild",))
    assert out["wells"]["A1"]["ph_ownership"]["rule"] == "nearest_centre"
    assert "aggregate" in out and "ph_tables" in out
    for part in LINKED:
        assert _digests(run / "A1" / part) == before[part]
        assert _digests(rebuilt / "A1" / part) == before[part]


def test_ph_tables_fills_the_gene_of_a_linked_cell_without_an_aggregate_table(measured):
    run = measured / "run"
    ops_engine.run_ops(_settings(measured, run), phases=("ph_masks", "merge"))
    assert not os.path.isfile(ops_engine._ph_cells_table_path(str(run), PLATE))
    out = ops_engine.run_ops(_settings(measured, run), phases=("ph_tables",))
    assert out["ph_tables"]["genotype_table"] == "ops_merge+ops_barcodes+ops_design"
    keys = ds.dataset(os.path.join(out["ph_tables"]["store"], "tables",
                                   "cell__keys"), partitioning="hive").to_table(
        ).to_pandas()
    linked = keys[keys["object_id"].notna()]
    assert len(linked) and linked["gene"].notna().all()
    design = pd.read_csv(measured / "design.csv")
    assert set(linked["gene"]) <= set(design["gene_symbol"])
    assert keys.loc[keys["object_id"].isna(), "gene"].isna().all()
