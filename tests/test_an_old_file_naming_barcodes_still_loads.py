"""A barcode-mapping settings file that names `barcodes` still loads.

Instruction 364 retired `barcodes` on 2026-09-19, at the maintainer's
decision that day: "Retire both" (with `Toxoplasma`). It was declared only
by `get_map_barcodes_default_settings`, which no pipeline calls, so no run
ever read the value; `grna_csv`, `column_csv` and `row_csv` were always the
keys that worked. It had been held since 2026-09-14 because a human-reviewed
zh_CN translation was pinned to its tooltip. That record is WITHDRAWN AND
RECORDED AS WITHDRAWN in its own file, which the last test here checks.

A withdrawn key has no successor, so an old file must: load, keep the key
where the pre-flight can name it, be told the value has no effect, and run
exactly as it always did.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from spacr.cli import load_settings_file
from spacr.settings import (bundled_barcode_path, categories, expected_types,
                            get_map_barcodes_default_settings,
                            set_default_generate_barecode_mapping, tooltips)
from spacr.validate import (ERROR, RETIRED_SETTINGS, WARNING,
                            _check_retired_keys, validate_settings)

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = (ROOT / "docs" / "i18n" / "reviewed" / "runtime" / "zh_CN"
            / "2026-08-14-tail-000-020.json")


def _old_file(tmp_path, barcodes):
    """A map_barcodes settings CSV as one saved before 2026-09-19."""
    path = tmp_path / "map_barcodes_settings.csv"
    rows = [("Key", "Value"), ("src", str(tmp_path)),
            ("barcodes", barcodes), ("chunk_size", "12345")]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def test_an_old_file_loads_and_keeps_its_other_values(tmp_path):
    old = str(tmp_path / "my_plate_barcodes.csv")
    settings = set_default_generate_barecode_mapping(
        load_settings_file(str(_old_file(tmp_path, old))))
    assert settings["chunk_size"] == 12345, "the file did not load"
    assert settings["barcodes"] == old, (
        "a withdrawn key keeps its place so the pre-flight can name it")
    for key, kind in (("grna_csv", "grna"), ("column_csv", "column"),
                      ("row_csv", "row")):
        assert settings[key] == bundled_barcode_path(kind), (
            f"{key} changed: a key no run read cannot change what runs")


def test_the_pre_flight_warns_and_does_not_refuse(tmp_path):
    settings = set_default_generate_barecode_mapping(
        load_settings_file(str(_old_file(tmp_path, "/x/barcodes.csv"))))
    about_it = [p for p in validate_settings(settings, "map_barcodes")
                if p.setting == "barcodes"]
    assert len(about_it) == 1, [p.message for p in about_it]
    assert about_it[0].severity == WARNING
    assert not [p for p in about_it if p.severity == ERROR]
    assert "no longer a spaCR setting" in about_it[0].message
    assert "no effect" in about_it[0].fix


def test_the_doctor_names_no_replacement():
    """Pointing at `grna_csv` would imply `barcodes` had been doing something."""
    assert RETIRED_SETTINGS["barcodes"] == ""
    problem = _check_retired_keys({"barcodes": "/x.csv"})[0]
    assert "renamed" not in problem.message


def test_the_key_is_gone_from_every_table_that_declares_a_setting():
    assert "barcodes" not in expected_types
    assert "barcodes" not in tooltips
    assert "barcodes" not in get_map_barcodes_default_settings({})
    assert not any("barcodes" in keys for keys in categories.values()
                   if isinstance(keys, (list, tuple)))
    from spacr.qt.screens import settings_model

    assert "barcodes" not in settings_model.PATH_LIST_KEYS
    assert "barcodes" not in settings_model.PATH_LIST_TITLES
    assert "barcodes" not in settings_model.PATH_LIST_SINGLE_KEYS


def test_the_reviewed_translation_is_withdrawn_and_recorded():
    """Never deleted silently: the file says what left, when and why."""
    payload = json.loads(REVIEWED.read_text(encoding="utf-8"))
    keys = {(r["table"], r["key"]) for r in payload["records"]}
    assert ("setting_tooltips", "barcodes") not in keys
    note = payload["review_notes"]["barcodes"]
    assert note.startswith("WITHDRAWN 2026-09-19")
    assert "Retire both" in note
    assert "4c3125b0f047a36722c65324a73341a7ceadd869f97ffbd9d0fe80610e96b1b4" in note
    assert "旧版条形码映射辅助函数" in note, "the reviewed translation was not kept"
