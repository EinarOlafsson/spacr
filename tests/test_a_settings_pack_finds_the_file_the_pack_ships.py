"""The migrating pack reader can actually read the pack that is published.

ITEM 317, PART 3, CORRECTED. The note for 317 said "the unsafe reader is
gone and its successor had a hole I closed tonight". The second half is
right -- `settings_pack.PACK_RENAMES` was empty and the package's own
rename table is now consulted after it. The first half is INVERTED. The
successor never replaced anything, because on the pack that is actually
published it read nothing at all:

  * it opened ``<app>_settings.csv`` and the pack ships
    ``gen_masks_settings.csv`` / ``crop_measure_settings.csv``;
  * the shipped files open with a ``Key,Value`` header row, which arrived
    as a setting named ``Key`` and was reported to the user as dropped;
  * ``channels,"[0, 1, 2, 3]"`` arrived as an eleven-character STRING over
    a list default, and was counted as applied -- a wrong value reported as
    a right one.

MEASURED 2026-09-14 on ``~/datasets/settings/gen_masks_settings.csv``, 59
non-blank lines, one of them the header. Before: applied 0, renamed 0,
dropped 0, malformed 0 -- a silent miss, so the caller applied plain
defaults and said nothing. After: 32 applied + 9 renamed + 17 dropped = 58,
every row accounted for, and 10 of the 17 are live settings this form does
not carry rather than settings this build has lost.

MEANWHILE THE BLIND READER IS STILL LIVE, and this file does not fix that:
`AppScreen.apply_settings_that_came_with` (spacr/qt/screens/app_screen.py)
is what every "Load test data..." button goes through, and it reports a
COUNT without naming anything that did not land. It is now possible for it
to use this module; doing so is somebody's next change, not this one's.
"""
from __future__ import annotations

import csv
import pathlib

import pytest

from spacr.qt.settings_pack import (PACK_FILES, PackReport, _HEADER_ROWS,
                                    _pack_candidates, _pack_path, read_pack,
                                    settings_from_pack)

#: A pack, as published: a header row, keys 391 renamed, keys that are gone,
#: keys that are live settings elsewhere but not on the mask form, and a
#: bracketed list.
SHIPPED = """Key,Value
src,/somebody/elses/disk
channels,"[0, 1, 2, 3]"
cell_FT,100
timelapse,False
workers,28
"""


def _write(folder, name, text=SHIPPED):
    path = pathlib.Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return str(folder)


# ---------------------------------------------------------------------------
# The file the pack ships
# ---------------------------------------------------------------------------

def test_the_published_spelling_is_found(tmp_path):
    """The whole point: the reader opens the file that exists.

    ``mask_settings.csv`` has never been published. Looking only for it made
    every real pack a silent miss.
    """
    _write(tmp_path, "gen_masks_settings.csv")
    assert pathlib.Path(_pack_path("mask", str(tmp_path))).name == \
        "gen_masks_settings.csv"
    values, malformed = read_pack("mask", str(tmp_path))
    assert malformed == 0
    assert values["cell_FT"] == 100


def test_the_measure_spelling_is_found_too(tmp_path):
    """The pack carries ``crop_measure_settings.csv`` for the measure stage."""
    _write(tmp_path, "crop_measure_settings.csv", "Key,Value\ncell_mask_dim,4\n")
    assert pathlib.Path(_pack_path("measure", str(tmp_path))).name == \
        "crop_measure_settings.csv"
    assert read_pack("measure", str(tmp_path))[0] == {"cell_mask_dim": 4}


def test_the_project_spelling_still_wins_when_both_are_there(tmp_path):
    """``<app>_settings.csv`` is what this project would write today."""
    _write(tmp_path, "gen_masks_settings.csv", "Key,Value\ncell_FT,1\n")
    _write(tmp_path, "mask_settings.csv", "Key,Value\ncell_FT,2\n")
    assert _pack_candidates("mask")[0] == "mask_settings.csv"
    assert read_pack("mask", str(tmp_path))[0]["cell_FT"] == 2


def test_a_pack_with_nothing_for_this_app_is_still_not_an_error(tmp_path):
    assert _pack_path("mask", str(tmp_path)) is None
    assert read_pack("mask", str(tmp_path)) == ({}, 0)


def test_the_report_says_which_file_it_read(tmp_path):
    """A pack that was never found and a clean pack both used to report
    nothing at all, and the caller could not tell them apart."""
    _write(tmp_path, "gen_masks_settings.csv")
    _settings, report = settings_from_pack(
        "mask", str(tmp_path), defaults={"channels": [0]})
    assert report.source == "gen_masks_settings.csv"

    _empty, missing = settings_from_pack(
        "mask", str(tmp_path / "nowhere"), defaults={"channels": [0]})
    assert missing.source == ""


def test_the_names_agree_with_the_screens_table():
    """Two tables naming the same files, pinned to each other.

    `AppScreen._EXAMPLE_SETTINGS_FILES` is the list the example buttons use
    and `PACK_FILES` is the copy this module keeps so that reading a CSV
    does not import a Qt screen. 364 is the item about two rename tables
    that disagreed because nothing compared them; this is the same shape one
    level out, so it is compared.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    shipped = AppScreen._EXAMPLE_SETTINGS_FILES
    assert shipped, "the screens' table is empty; one of these two moved"
    for app_key, names in shipped.items():
        known = set(_pack_candidates(app_key))
        missing = [name for name in names if name not in known]
        assert not missing, (
            f"{app_key}: the example buttons open {missing} and the pack "
            f"reader does not; add them to settings_pack.PACK_FILES")


def test_the_header_spellings_agree_with_the_screens_columns():
    """Every column pair the import path accepts is a header row here."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    for key_col, value_col in AppScreen._CSV_COLUMNS:
        pair = (key_col.strip().lower(), value_col.strip().lower())
        assert pair in _HEADER_ROWS, (
            f"{pair} is a header the importer accepts and the pack reader "
            f"would read as a setting")


# ---------------------------------------------------------------------------
# The header row is not a setting
# ---------------------------------------------------------------------------

def test_the_header_row_is_not_read_as_a_setting(tmp_path):
    """``Key,Value`` is a column heading, and was reported as a lost setting."""
    _write(tmp_path, "gen_masks_settings.csv")
    values, malformed = read_pack("mask", str(tmp_path))
    assert "Key" not in values and "key" not in values
    assert malformed == 0

    _settings, report = settings_from_pack(
        "mask", str(tmp_path), defaults={"channels": [0]})
    assert "Key" not in report.dropped
    assert "Key" not in report.summary()


def test_a_row_called_key_further_down_is_still_a_setting(tmp_path):
    """A heading is a heading because of where it is.

    Skipping ``key,value`` wherever it appeared would silently discard a
    setting of that name instead of reporting it.
    """
    _write(tmp_path, "gen_masks_settings.csv",
           "Key,Value\ncell_FT,100\nkey,value\n")
    values, _malformed = read_pack("mask", str(tmp_path))
    assert values["key"] == "value"
    _settings, report = settings_from_pack(
        "mask", str(tmp_path), defaults={"cell_flow_threshold": 0.4})
    assert "key" in report.dropped


# ---------------------------------------------------------------------------
# A bracketed cell is a container
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell,expected", [
    ('"[0, 1, 2, 3]"', [0, 1, 2, 3]),
    ("['cell']", ["cell"]),
    ('"(1, 2)"', (1, 2)),
])
def test_a_bracketed_cell_becomes_the_container_it_looks_like(
        tmp_path, cell, expected):
    """``channels`` is ``list`` in ``expected_types``.

    Handing the form the STRING "[0, 1, 2, 3]" replaced a real list default
    with text and counted it as applied, which is worse than dropping it:
    the report said it worked.
    """
    _write(tmp_path, "gen_masks_settings.csv", f"Key,Value\nchannels,{cell}\n")
    settings, report = settings_from_pack(
        "mask", str(tmp_path), defaults={"channels": [9]})
    assert settings["channels"] == expected
    assert not isinstance(settings["channels"], str)
    assert report.applied == ["channels"]


def test_a_bracket_that_is_not_a_container_stays_text(tmp_path):
    """Unparseable is not a reason to lose the value."""
    _write(tmp_path, "gen_masks_settings.csv", "Key,Value\ncmap,[inferno\n")
    assert read_pack("mask", str(tmp_path))[0]["cmap"] == "[inferno"


# ---------------------------------------------------------------------------
# The report says true things
# ---------------------------------------------------------------------------

def test_a_live_setting_this_form_lacks_is_not_called_gone(tmp_path):
    """"this version has no such setting" was untrue of ten keys.

    `timelapse` and nine others are live settings that the mask form does
    not carry. Telling a user the version dropped them sends them looking
    for a version that has them.
    """
    from spacr.settings import expected_types
    assert "timelapse" in expected_types, (
        "timelapse is no longer a setting; pick another live key for this test")

    _write(tmp_path, "gen_masks_settings.csv", "Key,Value\ntimelapse,False\n")
    _settings, report = settings_from_pack(
        "mask", str(tmp_path), defaults={"channels": [0]})

    assert report.dropped == ["timelapse"], "it still did not land"
    assert report.elsewhere == ["timelapse"]
    text = report.summary()
    assert "timelapse (settings this build has, but not on this form)" in text
    assert "this version has no such setting" not in text


def test_a_key_that_is_gone_everywhere_is_still_called_gone(tmp_path):
    """The other sentence, so the split is a split and not a rename."""
    _write(tmp_path, "gen_masks_settings.csv", "Key,Value\nworkers,28\n")
    from spacr.settings import expected_types
    assert "workers" not in expected_types, (
        "workers is live again; pick another retired key for this test")

    _settings, report = settings_from_pack(
        "mask", str(tmp_path), defaults={"channels": [0]})
    assert report.dropped == ["workers"]
    assert report.elsewhere == []
    assert "workers (this version has no such setting)" in report.summary()


def test_what_is_named_elsewhere_is_also_named_dropped(tmp_path):
    """`elsewhere` is a subset, never a fourth bucket.

    A caller that asks "did anything fail to land?" of `dropped` alone must
    still get the whole answer -- `app.py` asks exactly that.

    THE SUBSET IS ASSERTED OF A REPORT THE CODE BUILT, not of one this test
    wrote down: ``set(["b"]) <= set(["a", "b"])`` on a hand-made PackReport
    is a fact about the literal and no change to the module can redden it.
    The two keys below are the two cases at once -- `timelapse` is live on
    another form, `workers` is gone from the build (each is pinned to its
    own side by the two tests above) -- so the split is exercised rather
    than the dataclass.
    """
    _write(tmp_path, "gen_masks_settings.csv",
           "Key,Value\ntimelapse,False\nworkers,28\n")
    _settings, built = settings_from_pack(
        "mask", str(tmp_path), defaults={"channels": [0]})
    assert sorted(built.dropped) == ["timelapse", "workers"]
    assert built.elsewhere == ["timelapse"]
    assert set(built.elsewhere) < set(built.dropped)

    report = PackReport(dropped=["a", "b"], elsewhere=["b"])
    assert "a (this version has no such setting)" in report.summary()
    assert "b (settings this build has, but not on this form)" in \
        report.summary()


# ---------------------------------------------------------------------------
# Every row of a real published file is accounted for
# ---------------------------------------------------------------------------

REAL_PACK = (pathlib.Path(__file__).resolve().parents[1]
             / "spaCR_settings" / "1_generate_masks_settings.csv")


def test_the_shipped_sample_is_still_where_this_test_reads_it():
    """If it moved, the accounting test below proves nothing."""
    assert REAL_PACK.is_file(), f"{REAL_PACK} is gone; repoint or drop this test"


def test_every_row_of_a_real_pack_is_accounted_for(tmp_path):
    """applied + renamed + dropped == the rows read. No silent third state.

    A reader that loses a row without counting it is the failure this whole
    module is about, one level down: the report is only worth something if
    the parts sum to the file.
    """
    _write(tmp_path, "gen_masks_settings.csv",
           REAL_PACK.read_text(encoding="utf-8"))

    with (tmp_path / "gen_masks_settings.csv").open(newline="",
                                                    encoding="utf-8") as fh:
        rows = [row for row in csv.reader(fh)
                if row and not row[0].lstrip().startswith("#")]
    body = rows[1:]                       # row 0 is the Key,Value header
    assert rows[0] == ["Key", "Value"], "the sample stopped carrying a header"
    assert len({row[0].strip() for row in body}) == len(body), (
        "the sample now repeats a key; the sum below would not be a sum")

    raw, malformed = read_pack("mask", str(tmp_path))
    assert malformed == 0
    assert len(raw) == len(body), "a row went missing between file and reader"

    _settings, report = settings_from_pack("mask", str(tmp_path),
                                           src=str(tmp_path))
    accounted = len(report.applied) + len(report.renamed) + len(report.dropped)
    assert accounted == len(raw), (
        f"{len(raw)} rows read but {accounted} accounted for: "
        f"applied {len(report.applied)}, renamed {len(report.renamed)}, "
        f"dropped {len(report.dropped)}")
    assert set(report.elsewhere) <= set(report.dropped)
    assert report.renamed, "a pack this old must have renamed keys"
