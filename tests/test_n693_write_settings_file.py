"""Item 693: ``spacr.cli.write_settings_file``, the inverse of the loader.

The writer is what the GUI's Export settings… calls. Its CSV is the layout
``spacr.utils.save_settings`` writes, and both loaders read it back.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from spacr.cli import (
    _settings_csv_cell,
    _settings_json_value,
    load_settings_file,
    write_settings_file,
)


def test_cells_match_save_settings_except_exponent_floats():
    """``str`` of each value, ``None`` empty, exponent floats positional."""
    assert _settings_csv_cell(None) == ""
    assert _settings_csv_cell(True) == "True"
    assert _settings_csv_cell(3) == "3"
    assert _settings_csv_cell(0.5) == "0.5"
    assert _settings_csv_cell(1.5e-07) == "1.5e-07"
    assert _settings_csv_cell(1e-05) == "0.00001"
    assert _settings_csv_cell(1e20) == "100000000000000000000.0"
    assert _settings_csv_cell(float("inf")) == "inf"
    assert _settings_csv_cell([1, "a"]) == "[1, 'a']"
    assert _settings_csv_cell("/data/plate 1") == "/data/plate 1"


def test_json_values_keep_types_or_fall_back_to_repr():
    """Containers recurse, numpy scalars unwrap, the rest is a repr."""
    class Stubborn:
        def item(self):
            raise ValueError("no single item")

        def __repr__(self):
            return "<stubborn>"

    assert _settings_json_value({"a": (1, {2})}) == {"a": [1, [2]]}
    assert _settings_json_value(np.int64(4)) == 4
    assert _settings_json_value(Stubborn()) == "<stubborn>"
    assert _settings_json_value(object).startswith("<class")


def test_the_file_round_trips_through_both_loaders(tmp_path):
    """Key,Value header, dict order, every value type, JSON beside it."""
    from spacr.utils import load_settings

    settings = {"src": "/mnt/a b/plate,1", "tiny": 1e-05, "big": 1e20,
                "ratio": 0.5, "none": None, "channels": [0, 1],
                "pair": (1, "a"), "nested": {"a": 1}, "plot": False,
                "test_mode": True, "n": 3, "name": "hello"}
    target = tmp_path / "deep" / "measure_settings.csv"
    written = write_settings_file(settings, target)
    assert written == [str(target), str(target.with_suffix(".json"))]
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Key", "Value"]
    assert [row[0] for row in rows[1:]] == list(settings)
    assert load_settings_file(str(target)) == settings
    assert load_settings(str(target)) == settings
    assert json.loads(target.with_suffix(".json").read_text()) == json.loads(
        json.dumps(settings))


def test_export_refuses_redirected_staging_without_replacing_either_existing_file(tmp_path, monkeypatch):
    import errno
    import os

    import pytest

    from spacr import tabular

    target = tmp_path / 'settings.csv'
    write_settings_file({'src': '/original/plate', 'threshold': 1}, target)
    unrelated = tmp_path / 'unrelated.csv'
    unrelated.write_bytes(b'private unrelated table\x00\xff')
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    resolve = tabular.resolve_path
    redirected = []

    def changed_table_destination(path, *args, **kwargs):
        if '.spacr-settings-export-' in os.fspath(path):
            redirected.append(path)
            return str(unrelated)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(tabular, 'resolve_path', changed_table_destination)
    with pytest.raises(OSError) as caught:
        write_settings_file({'src': '/new/plate', 'threshold': 2}, target)
    assert caught.value.errno == errno.EINVAL
    assert len(redirected) == 1
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original
    assert load_settings_file(str(target)) == {'src': '/original/plate', 'threshold': 1}
