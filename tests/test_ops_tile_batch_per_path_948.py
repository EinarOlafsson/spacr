"""Settings version 948 (TASK X): the OPS Cellpose tile batch per path.

OPS Cellpose now runs through the shared mask tile batch
(``mask_eval_batch_size``; 0 is one tile per field in the resident batch).
The phenotype path always ran at 8 tiles (``ops_ph_batch_size``) and the
stitched sequencing nuclei and Spatial Mask Generation at 16. The tile batch
can move mask pixels, so a settings file written before 948 must keep those
values: ``ops_ph_tile_batch``, ``ops_sbs_tile_batch`` and
``spatial_tile_batch`` get them from ``DEFAULT_CHANGES``, and an old
``ops_ph_batch_size`` moves to ``ops_ph_tile_batch`` only, so it never changes
the sequencing path.
"""
import csv

import numpy as np
import pytest

from spacr import cli
from spacr import object as objects
from spacr import ops_engine, ops_settings
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            SETTINGS_VERSION_KEY)

NIGHTLY = {"ops_ph_tile_batch": 8, "ops_sbs_tile_batch": 16, "spatial_tile_batch": 16}


def _resolved(tmp_path, module, rows):
    """The settings ``spacr-run <module> --settings <file>`` hands the pipeline."""
    path = tmp_path / "settings.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return cli.resolve_settings(cli.resolve_module(module), str(path))


def _tiles(monkeypatch, settings, site, field_count=3):
    """The tile batch one OPS site hands Cellpose."""
    seen = []

    def evaluate(_model, _image, **keywords):
        """Record the Cellpose keywords without running a model."""
        seen.append(keywords)
        return "labels"

    monkeypatch.setattr(objects, "_mask_eval", evaluate)
    ops_engine._ops_mask_eval(object(), object(), settings, site=site,
                              field_count=field_count)
    return seen[0]["batch_size"]


def test_the_row_holds_the_tile_batches_each_path_ran_at():
    assert DEFAULT_CHANGES[948] == NIGHTLY
    assert SETTINGS_SCHEMA_VERSION >= 948
    assert ops_settings.OPS_DEFAULTS["ops_ph_tile_batch"] == 0
    assert ops_settings.OPS_DEFAULTS["ops_sbs_tile_batch"] == 0
    assert ops_settings.SPATIAL_MASK_DEFAULTS["spatial_tile_batch"] == 0


@pytest.mark.parametrize("stamp", [None, 945, 947])
def test_an_old_ops_file_reproduces_the_nightly_tile_batch_of_each_path(
        tmp_path, monkeypatch, stamp):
    """THE CASE THIS EXISTS FOR: phenotype 8, sequencing and stitched 16."""
    rows = [["plate", "P1"]] + ([[SETTINGS_VERSION_KEY, stamp]] if stamp else [])
    settings = _resolved(tmp_path, "ops", rows)
    assert settings["ops_ph_tile_batch"] == 8 and settings["ops_sbs_tile_batch"] == 16
    assert settings["mask_eval_batch_size"] == 0
    for fields in (1, 3, 50):
        assert _tiles(monkeypatch, settings, "ops_phenotype", fields) == 8
        assert _tiles(monkeypatch, settings, "ops_stitched", fields) == 16


def test_an_old_ops_ph_batch_size_sets_only_the_phenotype_path(tmp_path, monkeypatch):
    settings = _resolved(tmp_path, "ops", [[SETTINGS_VERSION_KEY, 945],
                                           ["ops_ph_batch_size", 4]])
    assert "ops_ph_batch_size" not in settings
    assert _tiles(monkeypatch, settings, "ops_phenotype") == 4
    assert _tiles(monkeypatch, settings, "ops_stitched") == 16


def test_an_old_spatial_file_keeps_sixteen_tiles(tmp_path, monkeypatch):
    settings = _resolved(tmp_path, "spatial_masks", [[SETTINGS_VERSION_KEY, 945]])
    assert settings["spatial_tile_batch"] == 16
    assert "ops_sbs_tile_batch" not in settings
    assert _tiles(monkeypatch, settings, "ops_spatial", 1) == 16


def test_a_new_file_takes_the_shared_rule_and_can_pin_each_path(tmp_path, monkeypatch):
    new = _resolved(tmp_path, "ops", [[SETTINGS_VERSION_KEY, SETTINGS_SCHEMA_VERSION]])
    assert new["ops_ph_tile_batch"] == 0 and new["ops_sbs_tile_batch"] == 0
    assert _tiles(monkeypatch, new, "ops_phenotype", 3) == 3
    assert _tiles(monkeypatch, new, "ops_stitched", 3) == 3
    shared = dict(new, mask_eval_batch_size=12)
    assert _tiles(monkeypatch, shared, "ops_phenotype") == 12
    assert _tiles(monkeypatch, shared, "ops_stitched") == 12
    pinned = _resolved(tmp_path, "ops", [[SETTINGS_VERSION_KEY, SETTINGS_SCHEMA_VERSION],
                                         ["ops_ph_tile_batch", 8],
                                         ["ops_sbs_tile_batch", 16],
                                         ["mask_eval_batch_size", 4]])
    assert _tiles(monkeypatch, pinned, "ops_phenotype") == 8
    assert _tiles(monkeypatch, pinned, "ops_stitched") == 16


def test_the_phenotype_field_segments_both_roles_at_its_own_tile_batch(monkeypatch):
    """Nuclei and cells of one field both run at ``ops_ph_tile_batch``."""
    seen = []

    def evaluate(_model, image, **keywords):
        """Planted labels; records the tile batch of each role."""
        seen.append(keywords["batch_size"])
        return np.zeros(image.shape[:2], np.int32), None, None

    monkeypatch.setattr(objects, "_mask_eval", evaluate)
    stack = np.zeros((8, 8, 2), np.float32)
    settings = {"ops_ph_cell_channel": "AF750", "ops_ph_downsample": 1,
                "ops_ph_tile_batch": 8, "ops_sbs_tile_batch": 16}
    ops_engine._ph_segment_field(stack, [ops_engine._NUCLEAR, "AF750"],
                                 (object(), object()), settings, field_count=3)
    assert seen == [8, 8]


@pytest.mark.parametrize("value", [-1, 2.5, "many", True])
def test_a_tile_batch_that_is_not_a_whole_number_is_refused(value):
    with pytest.raises(ValueError, match="ops_ph_tile_batch must be 0"):
        ops_engine._ops_tile_batch({"ops_ph_tile_batch": value}, "ops_phenotype", 1)
