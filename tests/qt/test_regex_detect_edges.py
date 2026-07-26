"""Edge / failure paths of :mod:`spacr.qt.regex_detect`.

The happy paths live in ``tests/qt/test_regex_detect.py``. This file
covers what happens when the sample is empty, unparseable, only
partially matched, or when the synthesiser has to invent a pattern —
including the two hit-count / template-literal bugs fixed alongside it.

Pure-python, no Qt, no I/O.
"""
from __future__ import annotations

import re

import pytest

from spacr.qt import regex_detect as rd


YOKO = [
    "plate1_A01_T0001F001L01A01Z01C01.tif",
    "plate1_A01_T0001F001L01A01Z01C02.tif",
    "plate1_B03_T0001F002L01A01Z01C01.tif",
]


# ---------------------------------------------------------------------------
# validate_records
# ---------------------------------------------------------------------------

def test_validate_records_on_empty_list_says_nothing_matched():
    assert rd.validate_records([]) == ["No filenames matched the regex."]
    # The message must not depend on the channel mode — an empty parse is
    # an empty parse either way.
    assert rd.validate_records([], multichannel=False) == [
        "No filenames matched the regex."
    ]


def test_validate_records_singlechannel_demands_a_field_id():
    records = [rd.MetadataRecord("a.tif", {"plateID": "p1", "wellID": "A01"})]
    warnings = rd.validate_records(records, multichannel=False)
    assert len(warnings) == 1
    assert "fieldID" in warnings[0]
    assert "single-channel" in warnings[0]
    # plateID was captured, so the soft plate warning must NOT fire.
    assert not any("plateID" in w for w in warnings)


def test_validate_records_ignores_groups_that_matched_empty_text():
    """A group present but empty is not a captured field.

    ``(?P<chanID>\\d*)`` happily matches the empty string; treating that
    as "chanID captured" would silence the warning the user needs.
    """
    records = [rd.MetadataRecord("a.tif", {"plateID": "p1", "chanID": "",
                                           "fieldID": "001"})]
    warnings = rd.validate_records(records, multichannel=True)
    assert [w for w in warnings if "chanID" in w], warnings


def test_validate_records_clean_multichannel_parse_has_no_warnings():
    records, missed = rd.apply_regex(YOKO, rd.YOKOGAWA)
    assert len(records) == 3 and missed == []
    assert rd.validate_records(records, multichannel=True) == []


# ---------------------------------------------------------------------------
# auto_detect_regex
# ---------------------------------------------------------------------------

def test_auto_detect_on_empty_sample_returns_the_empty_sentinel():
    assert rd.auto_detect_regex([]) == (None, "empty", 0)


def test_auto_detect_skips_a_builtin_that_does_not_compile(monkeypatch):
    """A malformed entry in BUILTIN_REGEXES must be stepped over, not raise."""
    monkeypatch.setattr(rd, "BUILTIN_REGEXES", {
        "broken": r"(?P<oops>",           # unterminated group
        "canonical": rd.CANONICAL,
    })
    files = ["p1_A01_F001_C01.tif", "p1_A01_F002_C01.tif"]
    pattern, label, hits = rd.auto_detect_regex(files)
    assert label == "canonical"
    assert pattern == rd.CANONICAL
    assert hits == 2


def test_auto_detect_keeps_a_builtin_that_matches_half_the_sample():
    """>=50 % is the documented bar for preferring a built-in."""
    files = [YOKO[0], "notes.txt"]
    pattern, label, hits = rd.auto_detect_regex(files)
    assert label == "cellvoyager"
    assert pattern == rd.CELLVOYAGER
    assert hits == 1


def test_auto_detect_falls_back_to_best_builtin_when_synthesis_gives_up():
    """No extension anywhere → the synthesiser returns None."""
    files = ["noextension", "alsonoextension"]
    assert rd._synthesise_regex(files) is None
    pattern, label, hits = rd.auto_detect_regex(files)
    assert hits == 0
    assert pattern in rd.BUILTIN_REGEXES.values()


def test_auto_detect_reports_the_true_number_of_synthesised_matches():
    """BUG (fixed): the synthesised branch returned ``n`` unconditionally.

    ``a.txt``/``b.txt`` produce a regex anchored on image extensions, so
    it matches nothing — yet auto_detect_regex claimed 2/2 and the regex
    editor printed "matched 2/2 sampled filenames" over an empty table.
    """
    files = ["a.txt", "b.txt"]
    pattern, label, hits = rd.auto_detect_regex(files)
    real = sum(1 for f in files if re.compile(pattern).match(f))
    assert real == 0
    assert hits == 0, "auto_detect_regex over-reported the match count"


def test_auto_detect_prefers_a_builtin_the_synthesiser_cannot_beat():
    """Synthesis works off one template file, so it can score *worse*
    than the sub-50 % built-in it was supposed to replace. When it does,
    the built-in must be handed back."""
    files = ["a.txt", YOKO[0], "zz.txt"]
    pattern, label, hits = rd.auto_detect_regex(files)
    assert label == "cellvoyager"
    assert hits == 1
    assert sum(1 for f in files if re.compile(pattern).match(f)) == 1


def test_auto_detect_degrades_gracefully_if_synthesis_emits_junk(monkeypatch):
    """The synthesiser's output is compiled before use; a regression that
    made it emit an uncompilable pattern must not blow up the caller."""
    monkeypatch.setattr(rd, "_synthesise_regex", lambda files: r"(?P<x>")
    files = ["a.txt", YOKO[0], "zz.txt"]
    pattern, label, hits = rd.auto_detect_regex(files)
    assert label == "cellvoyager"
    assert hits == 1
    re.compile(pattern)      # whatever came back must be usable


@pytest.mark.parametrize("files", [
    ["IMG_0001.tif", "IMG_0002.tif", "IMG_0003.tif"],
    ["myrun_W1_F01_C01.tif", "myrun_W1_F02_C01.tif", "myrun_W2_F03_C02.tif"],
    ["run5_T0001F001L01A01Z01C01.tif", "run5_T0001F002L01A01Z01C01.tif"],
])
def test_auto_detect_hit_count_always_matches_reality(files):
    pattern, label, hits = rd.auto_detect_regex(files)
    real = sum(1 for f in files if re.compile(pattern).match(f))
    assert hits == real, f"{label} claimed {hits}/{len(files)}, really {real}"


def test_synthesised_regex_generalises_a_bare_digit_run():
    """BUG (fixed): a numeric token was escaped literally into the regex.

    ``IMG_0001.tif`` yielded ``(?P<plateID>...)_0001\\.(?:tif|...)$``,
    which matched the one template file and nothing else in the folder.
    """
    files = ["IMG_0001.tif", "IMG_0002.tif", "IMG_0003.tif"]
    pattern, label, hits = rd.auto_detect_regex(files)
    assert label == "synthesised"
    assert "0001" not in pattern, pattern
    rx = re.compile(pattern)
    assert [f for f in files if rx.match(f)] == files
    assert rx.match("IMG_0002.tif").group("plateID") == "IMG"


# ---------------------------------------------------------------------------
# _synthesise_regex
# ---------------------------------------------------------------------------

def test_synthesise_regex_on_empty_input_is_none():
    assert rd._synthesise_regex([]) is None


def test_synthesise_regex_needs_an_extension():
    assert rd._synthesise_regex(["plate1_A01_F001_C01"]) is None


def test_synthesise_regex_splits_a_packed_yokogawa_token():
    """`T0001F001L01A01Z01C01` is one token — every prefix gets a group."""
    files = ["run5_T0001F001L01A01Z01C01.tif",
             "run5_T0002F003L01A01Z04C02.tif"]
    pattern = rd._synthesise_regex(files)
    m = re.compile(pattern).match(files[1])
    assert m is not None, pattern
    assert m.groupdict() == {
        "plateID": "run5", "timeID": "0002", "fieldID": "003",
        "laserID": "01", "AID": "01", "sliceID": "04", "chanID": "02",
    }


def test_synthesise_regex_keeps_only_the_first_of_a_repeated_prefix():
    """A second `C` run must stay ungrouped — duplicate group names are a
    re.error, which would make the synthesised pattern uncompilable."""
    files = ["run5_T0001F001C01Z01C02.tif", "run5_T0009F001C07Z01C08.tif"]
    pattern = rd._synthesise_regex(files)
    assert pattern.count("(?P<chanID>") == 1, pattern
    m = re.compile(pattern).match(files[1])
    assert m is not None
    assert m.group("chanID") == "07"      # first C wins
    assert m.group("sliceID") == "01"


def test_synthesise_regex_reuses_a_prefix_that_appears_in_two_tokens():
    """A plate barcode like `F0012` eats the `fieldID` name, so the later
    real field token has to fall back to an unnamed `F\\d+`.

    The heuristic cannot tell the two apart — but it must still emit a
    *compilable* regex that matches the whole folder (a second
    ``(?P<fieldID>...)`` would be a re.error), and the resulting
    "no plateID captured" warning is what pushes the user to edit it.
    """
    files = ["F0012_A01_F0003_C01.jpg", "F0012_A02_F0007_C02.jpg"]
    pattern = rd._synthesise_regex(files)
    assert pattern.count("(?P<fieldID>") == 1
    assert "_F\\d+_" in pattern, pattern
    rx = re.compile(pattern)
    assert [f for f in files if rx.match(f)] == files
    assert rx.match(files[1]).groupdict() == {
        "fieldID": "0012", "wellID": "A02", "chanID": "02",
    }
    records, missed = rd.apply_regex(files, pattern)
    assert missed == []
    assert any("plateID" in w for w in rd.validate_records(records))


def test_synthesise_regex_escapes_a_token_it_cannot_interpret():
    files = ["run5_A01_scan(2)_C01.tif", "run5_A02_scan(2)_C03.tif"]
    pattern = rd._synthesise_regex(files)
    assert r"scan\(2\)" in pattern, pattern
    rx = re.compile(pattern)
    assert [f for f in files if rx.match(f)] == files
    assert rx.match(files[1]).group("wellID") == "A02"
    assert rx.match(files[1]).group("chanID") == "03"


def test_synthesise_regex_varies_an_unknown_letter_prefix():
    """`W1` must become `W\\d+`, otherwise only well W1 would ever match."""
    files = ["myrun_W1_F01_C01.tif", "myrun_W2_F03_C02.tif"]
    pattern = rd._synthesise_regex(files)
    rx = re.compile(pattern)
    assert [f for f in files if rx.match(f)] == files


# ---------------------------------------------------------------------------
# tabulate_records
# ---------------------------------------------------------------------------

def test_tabulate_records_on_empty_input_explains_itself():
    assert rd.tabulate_records([]) == \
        "(no records — regex did not match any files)"


def test_tabulate_records_head_mode_keeps_input_order():
    records = [rd.MetadataRecord(f"f{i:02d}.tif", {"fieldID": str(i)})
               for i in range(10)]
    out = rd.tabulate_records(records, max_rows=3, random_sample=False)
    body = out.splitlines()[2:]
    assert len(body) == 3
    assert [ln.split()[1] for ln in body] == ["f00.tif", "f01.tif", "f02.tif"]
    assert [ln.split()[0] for ln in body] == ["0", "1", "2"]


def test_tabulate_records_random_mode_is_seed_stable():
    records = [rd.MetadataRecord(f"f{i:02d}.tif", {"fieldID": str(i)})
               for i in range(30)]
    a = rd.tabulate_records(records, max_rows=5, seed=7)
    b = rd.tabulate_records(records, max_rows=5, seed=7)
    c = rd.tabulate_records(records, max_rows=5, seed=8)
    assert a == b
    assert a != c


def test_tabulate_records_marks_a_field_the_regex_did_not_capture():
    records = [rd.MetadataRecord("a.tif", {"wellID": "A01", "chanID": "01"}),
               rd.MetadataRecord("b.tif", {"wellID": "A02", "chanID": None})]
    out = rd.tabulate_records([records[0]], columns=["wellID", "plateID"])
    header, rule, row = out.splitlines()
    assert header.split() == ["wellID", "plateID", "filename"]
    assert row.split() == ["A01", "—", "a.tif"]


def test_tabulate_records_orders_columns_by_known_fields_then_extras():
    records = [rd.MetadataRecord("a.tif", {"chanID": "01", "zzz": "9",
                                           "plateID": "p1", "wellID": "A01"})]
    header = rd.tabulate_records(records).splitlines()[0].split()
    assert header == ["plateID", "wellID", "chanID", "zzz", "filename"]


def test_tabulate_records_pads_every_column_to_the_widest_cell():
    records = [rd.MetadataRecord("short.tif", {"plateID": "p"}),
               rd.MetadataRecord("a_much_longer_name.tif",
                                 {"plateID": "plate_number_one"})]
    lines = rd.tabulate_records(records).splitlines()
    # header, rule and both body rows are the same width
    assert len({len(ln) for ln in lines}) == 1
