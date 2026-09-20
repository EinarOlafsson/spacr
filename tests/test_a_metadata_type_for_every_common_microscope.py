"""The filename conventions spaCR reads, and the evidence behind each one.

`metadata_type` used to name two microscopes and both of them were Yokogawas.
Instruction 441 replaced the if/elif chain in `spacr.utils._get_regex` with a
TABLE -- `spacr.regex_infer._METADATA_CONVENTIONS` -- one record per
convention, carrying the vendor, the instrument family, real example
filenames, what each named group means, where it was sourced, and whether it
is confirmed or provisional.

THREE PROPERTIES ARE WORTH A TEST AND THEY ARE NOT THE OBVIOUS ONE.

* THE OLD FOUR MUST NOT MOVE. `cellvoyager`, `cq1`, `auto` and `custom` are
  written into settings CSVs that are years old and into six other modules'
  docstrings. Their patterns are pinned here CHARACTER FOR CHARACTER,
  including the bare `.` before the extension -- which matches any character
  and is a latent bug, and is kept anyway, because changing it is a separate
  decision from adding microscopes.

* NO TWO CONVENTIONS MAY CLAIM THE SAME FILENAME. A pattern that matches a
  name it was not written for does not fail: it assigns a well, a field and a
  channel, all of them wrong, and the run finishes with a measurements
  database that is wrong rather than empty. One real instance was caught
  while the table was being written -- EVOS reuses Cellomics' whole
  `<well>f<field>d<dye>` tail, so `scan_R_p2_z1_0_B03f01d0.tif` matched
  `arrayscan` as well as `evos` until a negative lookahead was added.

* NO CAPTURING GROUP MAY BE OPTIONAL. `spacr.utils._extract_filename_metadata`
  reads `match.group('fieldID')[0]` to decide whether to strip zero padding.
  A group that did not participate returns None and that subscript raises
  TypeError, which nothing up the stack catches. A group that matched the
  empty string raises IndexError, which IS caught -- and silently skips the
  file. Both are worse than not matching at all.
"""
from __future__ import annotations

import re

import pytest

from spacr.regex_infer import (_METADATA_CONVENTIONS, _metadata_autodetect,
                               _metadata_convention,
                               _metadata_convention_example,
                               _metadata_convention_keys,
                               _metadata_convention_menu,
                               _metadata_parse_report, _metadata_pattern)


#: The four arms `_get_regex` shipped before instruction 441, exactly as they
#: were, for `img_format='tif'`. Copied from the source at 03cfdbf02.
ORIGINAL_PATTERNS = {
    "cellvoyager": (
        "(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
        "L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*).tif"),
    "cq1": (
        "W(?P<wellID>.*)F(?P<fieldID>.*)T(?P<timeID>.*)Z(?P<sliceID>.*)"
        "C(?P<chanID>.*).tif"),
    "auto": (
        "(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
        "L(?P<laserID>.*)C(?P<chanID>.*).tif"),
}

#: The only pair of conventions allowed to claim the same filename, and the
#: reason it is allowed. `auto` and `cellvoyager` are BOTH the Yokogawa
#: grammar written entirely in `.*`, differing only in whether the L block is
#: followed by A and Z. A cellvoyager name therefore satisfies `auto` as well,
#: and neither pattern may be tightened: both are pinned above. Recorded as a
#: known overlap rather than hidden, so that a NEW overlap fails.
ALLOWED_OVERLAPS = {("auto", "cellvoyager")}


def _extension_of(filename: str) -> str:
    """The filename's extension without its dot, as a run would compute it."""
    return filename.rsplit(".", 1)[-1].lower()


@pytest.mark.parametrize("metadata_type", sorted(ORIGINAL_PATTERNS))
def test_the_three_built_in_patterns_are_unchanged_character_for_character(
        metadata_type):
    """The pin. A settings file from 2023 must get the pattern it got then."""
    from spacr.utils import _get_regex

    assert _get_regex(metadata_type, "tif") == ORIGINAL_PATTERNS[
        metadata_type]


def test_the_custom_arm_still_wraps_the_users_expression_the_same_way():
    """Including for `custom_regex=None`, which produced the literal '(None)'.

    That looks like a bug and is behaviour: `f"({custom_regex}).{img_format}"`
    formats None as the string 'None', and a settings file that chose
    `custom` without supplying a pattern has always got `(None).tif` and the
    zero matches that follow. Reproduced rather than fixed, because the four
    original arms are pinned and this is one of them.
    """
    from spacr.utils import _get_regex

    assert _get_regex("custom", "tif", r"(?P<wellID>[A-H]\d\d)") == (
        r"((?P<wellID>[A-H]\d\d)).tif")
    assert _get_regex("custom", "tif") == "(None).tif"


def test_the_image_format_is_interpolated_exactly_as_it_always_was():
    """Verbatim, dot and all -- `auto` ignores it, and `None` means tif.

    A caller passing '.tif' rather than 'tif' is not hypothetical:
    `tests/test_utils_training_advice.py` does, and
    `spacr.io.preprocess_img_data` passes a bare extension. The old arms
    interpolated whatever arrived, so the pinned four still do.
    """
    from spacr.utils import _get_regex

    assert _get_regex("cq1", None).endswith(".tif")
    assert _get_regex("cq1", ".tif").endswith("C(?P<chanID>.*)..tif")
    assert _get_regex("cellvoyager", "png").endswith(".png")
    assert _get_regex("auto", "png").endswith(".tif")


def test_an_unknown_convention_is_refused_with_the_whole_vocabulary():
    """The list IS the documentation at the moment it is needed."""
    from spacr.utils import _get_regex

    with pytest.raises(ValueError) as excinfo:
        _get_regex("nikon", "tif")

    message = str(excinfo.value)
    assert "nikon" in message
    for key in _metadata_convention_keys():
        assert repr(key) in message


@pytest.mark.parametrize("record", _METADATA_CONVENTIONS,
                         ids=[r["key"] for r in _METADATA_CONVENTIONS])
def test_every_convention_parses_every_example_it_claims(record):
    """The examples are the evidence; a record whose own examples fail is a
    guess with a citation attached."""
    for example in record["examples"]:
        pattern = _metadata_pattern(record["key"], _extension_of(example),
                                    custom_regex=r"(?P<wellID>[A-H]\d\d)")
        assert re.compile(pattern).match(example), (
            f"{record['key']} does not parse its own example {example!r}")


@pytest.mark.parametrize("record", _METADATA_CONVENTIONS,
                         ids=[r["key"] for r in _METADATA_CONVENTIONS])
def test_every_example_fills_every_group_the_importer_reads(record):
    """No group may be absent, None, or empty -- see this file's header."""
    required = {"wellID", "fieldID", "chanID"}
    must_not_be_empty = required | {"timeID", "sliceID"}
    for example in record["examples"]:
        pattern = _metadata_pattern(record["key"], _extension_of(example),
                                    custom_regex=r"(?P<wellID>[A-H]\d\d)")
        match = re.compile(pattern).match(example)
        groups = match.groupdict()
        if record["key"] != "custom":
            assert required <= set(groups), (
                f"{record['key']} must capture {sorted(required)}; "
                f"{example!r} captured {sorted(groups)}")
        assert not [name for name, value in groups.items() if value is None], (
            f"{record['key']}: a group did not participate on {example!r} -- "
            f"_extract_filename_metadata raises TypeError on that")
        assert not [name for name, value in groups.items()
                    if value == "" and name in must_not_be_empty], (
            f"{record['key']}: a required group matched the empty string on "
            f"{example!r} -- the importer skips the file and says nothing "
            f"useful about why")


def test_no_convention_claims_a_filename_written_by_another():
    """Two patterns matching one name is how a plate is silently mislabelled."""
    overlaps = set()
    for record in _METADATA_CONVENTIONS:
        if record["key"] == "custom":
            continue
        compiled = {}
        for other in _METADATA_CONVENTIONS:
            if other["key"] == record["key"]:
                continue
            for example in other["examples"]:
                extension = _extension_of(example)
                if extension not in compiled:
                    compiled[extension] = re.compile(
                        _metadata_pattern(record["key"], extension))
                if compiled[extension].match(example):
                    overlaps.add((record["key"], other["key"]))
    assert overlaps == ALLOWED_OVERLAPS, (
        "conventions that claim each other's filenames: "
        f"{sorted(overlaps - ALLOWED_OVERLAPS)}; "
        f"overlaps that stopped happening: "
        f"{sorted(ALLOWED_OVERLAPS - overlaps)}")


@pytest.mark.parametrize("record", _METADATA_CONVENTIONS,
                         ids=[r["key"] for r in _METADATA_CONVENTIONS])
def test_every_record_carries_the_evidence_for_itself(record):
    """Vendor, instrument, source, status, and enough example filenames.

    CONFIRMED NEEDS TWO EXAMPLES AND PROVISIONAL NEEDS ONE. One filename
    proves a pattern can parse something; two proves it parses a FAMILY, and
    that is the difference between a convention and a screenshot. Where only
    one could be sourced -- EVOS, ScanR, Cytation -- the record says
    provisional and the dropdown shows it, which is the honest outcome and
    not a failure.
    """
    for field in ("key", "label", "vendor", "instrument", "pattern",
                  "groups", "source", "status"):
        assert record.get(field), f"{record['key']} has no {field}"
    assert record["status"] in ("confirmed", "provisional")
    if record["key"] == "custom":
        return
    floor = 2 if record["status"] == "confirmed" else 1
    assert len(record["examples"]) >= floor, (
        f"{record['key']} is {record['status']} with "
        f"{len(record['examples'])} example(s); it needs {floor}")
    for name in re.compile(record["pattern"].replace("{ext}", "tif")
                           ).groupindex:
        assert name in record["groups"], (
            f"{record['key']} captures {name} and does not say what it means")


@pytest.mark.parametrize(
    "record", [r for r in _METADATA_CONVENTIONS
               if r["key"] not in ORIGINAL_PATTERNS and r["key"] != "custom"],
    ids=[r["key"] for r in _METADATA_CONVENTIONS
         if r["key"] not in ORIGINAL_PATTERNS and r["key"] != "custom"])
def test_a_new_pattern_prefers_a_character_class_to_a_bare_dot_star(record):
    """`(?P<wellID>.*)` is what made the old arms grab the wrong half of a name.

    The four originals are exempt: they are pinned. Everything added since
    names what it expects, so a name with one more underscore than the author
    imagined FAILS TO MATCH -- which is visible -- instead of matching wrong.
    """
    assert ".*)" not in record["pattern"], (
        f"{record['key']} captures a greedy .* ; use a character class")


def test_the_dropdown_is_grouped_by_vendor_and_stores_the_same_values():
    """The menu the panel draws, and the keys a settings CSV holds."""
    menu = _metadata_convention_menu()
    vendors = [vendor for vendor, _rows in menu]
    assert len(vendors) == len(set(vendors)), "a vendor appears twice"
    assert "Yokogawa" in vendors
    assert {"Zeiss", "Nikon", "Leica", "Thermo Fisher"} <= set(vendors)
    listed = [key for _vendor, rows in menu for key, _label, _status in rows]
    assert sorted(listed) == sorted(_metadata_convention_keys()), (
        "the menu and the table must offer the same conventions")
    assert set(ORIGINAL_PATTERNS) | {"custom"} <= set(listed)
    assert vendors[0] == "Yokogawa", "the default convention leads the menu"
    assert listed[0] == "cellvoyager"
    assert vendors[-1] == "spaCR", (
        "'auto' and 'custom' are not microscopes and come last")


def test_the_example_filename_is_the_first_one_the_record_carries():
    """What the row shows under the menu."""
    assert _metadata_convention_example("cq1") == "W0262F0001T0001Z000C2.tif"
    assert _metadata_convention_example("custom") == ""
    assert _metadata_convention_example("no such convention") == ""


def test_the_parse_report_counts_and_names_the_first_failure():
    """The readout behind 'Test on my folder'."""
    names = ["r01c01f01p01-ch1sk1fk1fl1.tiff",
             "r01c01f01p01-ch2sk1fk1fl1.tiff",
             "a_name_from_somewhere_else.tiff"]
    assert _metadata_parse_report(names, "opera_phenix", "tiff") == (
        2, 3, "a_name_from_somewhere_else.tiff")
    assert _metadata_parse_report(names, "cq1", "tiff") == (
        0, 3, "r01c01f01p01-ch1sk1fk1fl1.tiff")


def test_a_pattern_that_does_not_compile_reports_zero_rather_than_raising():
    """A custom_regex is a thing the user is still typing."""
    assert _metadata_parse_report(["a.tif"], "custom", "tif", "(") == (
        0, 1, "a.tif")
    assert _metadata_parse_report(["a.tif"], "no such convention", "tif") == (
        0, 1, "a.tif")


def test_autodetect_ranks_and_offers_but_names_nothing_it_cannot_judge():
    """`custom` parses whatever was last typed and `auto` renames first.

    Neither is evidence about a folder, so neither is ever the answer the
    offer presents -- and the offer is a ranking rather than an assignment,
    because a setting changed without being asked for is the same defect as
    a setting left wrong.
    """
    names = ["Plate1_A01_s1_w1.TIF", "Plate1_A01_s1_w2.TIF",
             "Plate1_B02_s2_w1.TIF", "nothing_parses_this.TIF"]
    ranked = _metadata_autodetect(names, "TIF")
    assert ranked, "the ImageXpress plate should be recognised"
    assert ranked[0][:3] == ("imagexpress", 3, 4)
    assert not [key for key, _m, _t in ranked if key in ("auto", "custom")]
    assert not _metadata_autodetect(["nothing_parses_this.tif"], "tif")


def test_validate_knows_every_convention_the_dropdown_offers():
    """Otherwise the run advice tells a user their good plate parses nothing.

    `spacr.validate._candidate_patterns` sweeps `METADATA_REGEXES`; a
    `metadata_type` missing from it leaves `raw_channels` None and raises a
    WARNING naming three conventions at a user who chose a fourth.
    """
    from spacr.validate import METADATA_REGEXES

    missing = [record["key"] for record in _METADATA_CONVENTIONS
               if record["key"] != "custom"
               and record["key"] not in METADATA_REGEXES]
    assert not missing, f"validate cannot see: {missing}"
    for key, pattern in METADATA_REGEXES.items():
        re.compile(pattern)
        assert "{ext}" not in pattern, f"{key} kept its placeholder"


def test_the_convention_table_reaches_the_settings_spec_menu():
    """The panel and the table are one list, not two."""
    from spacr.settings_spec import _metadata_type_choices

    choices = _metadata_type_choices()
    assert sorted(value for value, _label in choices) == sorted(
        _metadata_convention_keys())
    assert choices[0][0] == "cellvoyager", (
        "the default must be the first thing a plain dropdown shows")
    provisional = {record["key"] for record in _METADATA_CONVENTIONS
                   if record["status"] == "provisional"}
    for value, label in choices:
        assert ("provisional" in label) == (value in provisional)
