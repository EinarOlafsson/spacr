"""Item 967 default: the spacer-prefix minority-read rule is on at k = 3 (schema 945).

Maintainer decision 2026-11-19. Settings files stamped before 945 keep the
rule off (k = 0) and write what the decode wrote before, byte for byte; a
file stamped 945 or later, or an explicit ``ops_prefix_minority_min_reads``,
decides it otherwise.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr import ops_prefix as P
from tests.test_ops_read_mode_952 import _prefix

KEY = "ops_prefix_minority_min_reads"

#: A small tuning grid, so the synthetic well tunes in seconds.
GRID = {"ops_read_threshold": [10, 5000], "ops_prefix_min_read_quality": [0, 0.5]}


def _same_outputs(one, two):
    """Assert two decode folders hold the same files, byte for byte.

    A per-well tuning record names its own CSV by absolute path, so each
    folder's path is replaced by one placeholder before the bytes are compared.
    """
    names = sorted(os.listdir(one))
    assert names == sorted(os.listdir(two))
    for name in names:
        with open(os.path.join(one, name), "rb") as a, open(os.path.join(two, name), "rb") as b:
            left = a.read().replace(os.fsencode(one), b"<run>")
            right = b.read().replace(os.fsencode(two), b"<run>")
        assert left == right, name


def _resolved(tmp_path, name, stamp, overrides=()):
    """The settings ``spacr-run`` hands the OPS engine for a file stamped ``stamp``.

    The file holds the synthetic well of :func:`tests.test_ops_read_mode_952._prefix`
    (written by one plain decode first) under recipe_simple classes, so the
    minority-read rule can change calls.
    """
    from spacr import cli

    root = tmp_path / "data"
    if not root.exists():
        _prefix(tmp_path, "seed")
    settings = {"spacr_settings_version": stamp, "ops_layout": "cpg_cropped",
                "ops_library_mode": "spacer_prefix",
                "cpg_root": str(root / "images_corrected_cropped"),
                "cpg_phenotype_channels": ["CorrDNA"],
                "ops_library": str(tmp_path / "Barcodes.csv"),
                "ops_prefix_masks": str(root / "masks" / "{plate}_{well}_{site}_cells.tiff"),
                "ops_prefix_quality_method": "recipe_simple",
                "ops_prefix_tuning_grid": GRID, "n_workers": 1,
                "dst_root": str(tmp_path / name)}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return cli.resolve_settings(cli.resolve_module("ops"), str(path), list(overrides))


def test_the_new_default_and_the_schema_row():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION

    assert OPS_DEFAULTS[KEY] == 3
    assert SETTINGS_SCHEMA_VERSION >= 945
    assert DEFAULT_CHANGES[945] == {KEY: 0}
    assert "Default 3" in OPS_TOOLTIPS[KEY]


def test_the_stamp_decides_the_rule():
    from spacr.settings import _fold_older_defaults

    for stamp in (0, 942, 944):
        folded = {}
        assert _fold_older_defaults(folded, stamp)[KEY] == (0, 945)
        assert folded[KEY] == 0
    current = {}
    _fold_older_defaults(current, 945)
    assert KEY not in current
    kept = {KEY: 3}
    _fold_older_defaults(kept, 944)
    assert kept[KEY] == 3


def test_resolved_files_take_the_default_of_their_era(tmp_path):
    assert _resolved(tmp_path, "s944", 944)[KEY] == 0
    assert _resolved(tmp_path, "s945", 945)[KEY] == 3
    assert _resolved(tmp_path, "s944k3", 944, [f"{KEY}=3"])[KEY] == 3
    assert _resolved(tmp_path, "s945k0", 945, [f"{KEY}=0"])[KEY] == 0


def test_an_old_stamp_and_k0_write_what_the_decode_wrote(tmp_path):
    old = P.run_prefix_decode(_resolved(tmp_path, "old", 944))
    off = P.run_prefix_decode(_resolved(tmp_path, "off", 945, [f"{KEY}=0"]))
    _same_outputs(old["folder"], off["folder"])
    assert "minority_rescued" not in old
    saved = json.load(open(os.path.join(old["folder"], "quality_classes.json")))
    assert KEY not in saved["settings"]


def test_the_new_default_is_the_k3_decode(tmp_path):
    new = P.run_prefix_decode(_resolved(tmp_path, "new", 945))
    k3 = P.run_prefix_decode(_resolved(tmp_path, "k3", 944, [f"{KEY}=3"]))
    _same_outputs(new["folder"], k3["folder"])
    saved = json.load(open(os.path.join(new["folder"], "quality_classes.json")))
    assert saved["settings"][KEY] == 3
    assert "CP9_W1" in new["minority_rescued"]


def test_weighted_majority_needs_the_rule_off():
    base = {"ops_library": "x.csv", "ops_prefix_quality_method": "weighted_majority"}
    with pytest.raises(ValueError, match="set ops_prefix_minority_min_reads to 0"):
        P.prefix_settings({**base, KEY: 3})
    assert P.prefix_settings({**base, KEY: 0})[KEY] == 0
