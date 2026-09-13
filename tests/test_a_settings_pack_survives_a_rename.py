"""A settings pack written against an older spaCR keeps its renamed keys.

THE PACK ON HUGGING FACE IS OLDER THAN THE PACKAGE and always will be. It is
published beside a dataset, not rebuilt with every rename, so the keys in it
are whatever spaCR called them on the day it was written. 317 is the item
about this and the maintainer's words are "these might need to be reformated
if they are not compatible with the new version".

WHAT WENT WRONG. `settings_pack.PACK_RENAMES` is a per-app table and is empty
for both `mask` and `measure`, so the reader had no rename to apply and
reported every renamed key as DROPPED -- while `spacr.settings` knew exactly
what each had become. Measured 2026-09-13 on a four-key pack: applied 2,
renamed 0, dropped 4, and all four resolvable through
`surviving_setting_name`.

That is the THIRD rename table in this package to disagree with the others.
364 found `RENAMED_SETTINGS` and `RETIRED_SETTINGS` disagreeing -- one decides
what the run does, the other what the doctor says -- and nothing checked that
they agreed. This is the same shape one level out.

WHY A SPLIT IS STILL DROPPED. `surviving_setting_name` can return several
names where one setting became many. A single value cannot be sent to all of
them without inventing a meaning for it, so those stay with `PACK_RENAMES`,
which is curated and can say what was intended. Silence is recoverable; a
number on the wrong control looks deliberate.
"""
from __future__ import annotations

import csv
import pathlib

import pytest

from spacr.qt.settings_pack import PACK_RENAMES, settings_from_pack
from spacr.settings import expected_types, surviving_setting_name

#: Keys 391 renamed, with what they became. Spelled out rather than derived,
#: so this test fails if either side moves rather than following it.
RENAMED_BY_391 = {
    "cell_FT": "cell_flow_threshold",
    "cell_CP_prob": "cell_cellprob_threshold",
    "nucleus_FT": "nucleus_flow_threshold",
    "pathogen_CP_prob": "pathogen_cellprob_threshold",
}


def _pack(tmp_path, app_key, rows):
    """Write a settings pack in the shipped format: key,value and no header."""
    path = tmp_path / f"{app_key}_settings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for key, value in rows.items():
            writer.writerow([key, value])
    return tmp_path


def test_the_old_names_are_still_the_names_this_test_is_about():
    """If 391's renames are undone, the rest of this file proves nothing."""
    for old, new in RENAMED_BY_391.items():
        assert old not in expected_types, f"{old} is live again; rewrite this test"
        assert new in expected_types, f"{new} is not a setting any more"
        assert surviving_setting_name(old) == (new,)


def test_a_pack_using_the_old_names_keeps_its_values(tmp_path):
    """The whole point: nothing the package can migrate is dropped."""
    rows = dict(RENAMED_BY_391)
    rows.update({"cell_FT": "0.4", "cell_CP_prob": "0.0",
                 "nucleus_FT": "0.35", "pathogen_CP_prob": "-1.0",
                 "verbose": "True"})
    directory = _pack(tmp_path, "mask", rows)

    settings, report = settings_from_pack("mask", str(directory),
                                          src=str(directory))

    assert not report.dropped, (
        f"a pack lost keys the package can migrate: {sorted(report.dropped)}")
    assert dict(report.renamed) == RENAMED_BY_391
    assert settings["cell_flow_threshold"] == 0.4
    assert settings["cell_cellprob_threshold"] == 0.0
    assert settings["nucleus_flow_threshold"] == 0.35
    assert settings["pathogen_cellprob_threshold"] == -1.0


def test_a_key_the_package_cannot_place_is_still_dropped(tmp_path):
    """Dropping is the right answer when there is no single destination.

    Silence about a key is recoverable -- the default stands and the report
    names what was dropped. A value written to a control it does not belong
    to is not: it reads as a deliberate choice.
    """
    directory = _pack(tmp_path, "mask", {"a_setting_that_never_existed": "1"})
    _settings, report = settings_from_pack("mask", str(directory),
                                           src=str(directory))
    assert report.dropped == ["a_setting_that_never_existed"]
    assert not report.renamed


def test_the_per_app_table_still_wins(tmp_path):
    """`PACK_RENAMES` is curated and is consulted first, by construction.

    It can say "this changed MEANING, do not carry the value" where the
    mechanical resolver only knows the name changed. Asserting the order
    keeps that possible.
    """
    assert "mask" in PACK_RENAMES, "the per-app table lost its mask entry"
    # The mask table is empty today; the ordering is what this pins.
    source = pathlib.Path(
        __import__("spacr.qt.settings_pack", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    per_app = source.index("moved = renames.get(key)")
    package = source.index("_package_renames(key)")
    assert per_app < package, "the per-app table must be consulted first"
