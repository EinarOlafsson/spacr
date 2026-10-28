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

* REQUIRED IMAGE IDENTIFIERS MAY NOT BE OPTIONAL. `spacr.utils._extract_filename_metadata`
  reads `match.group('fieldID')[0]` to decide whether to strip zero padding.
  A group that did not participate returns None and that subscript raises
  TypeError, which nothing up the stack catches. A group that matched the
  empty string raises IndexError, which IS caught -- and silently skips the
  file. Both are worse than not matching at all. A documented absent EVOS Z
  plane is explicitly preserved as None; it does not identify a synthetic Z0.
"""
from __future__ import annotations

import os
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
    """Required groups participate; only the documented EVOS slice may be absent."""
    required = {"wellID", "fieldID", "chanID"}
    must_not_be_empty = required | {"timeID", "sliceID"}
    for example in record["examples"]:
        pattern = _metadata_pattern(record["key"], _extension_of(example),
                                    custom_regex=r"(?P<wellID>[A-H]\d\d)")
        match = re.compile(pattern).match(example)
        groups = match.groupdict()
        absent = ({"sliceID"} if record["key"] == "evos" and example ==
                  "image_0002_Top Slide_R_p00_0_A01f00d0.TIF" else set())
        if absent:
            assert groups["sliceID"] is None
        if record["key"] != "custom":
            assert required <= set(groups), (
                f"{record['key']} must capture {sorted(required)}; "
                f"{example!r} captured {sorted(groups)}")
        assert not [name for name, value in groups.items()
                    if value is None and name not in absent], (
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
    that is the difference between a convention and a screenshot. EVOS has
    two exact vendor-documented literals, including the no-Z save dialog.
    ScanR's two acquisition-user stems omit extensions; both runnable
    examples explicitly disclose an inferred TIFF suffix.
    Cytation also remains provisional because its additional upstream
    real-world-derived names disclose sanitization. The dropdown shows this
    evidence limit rather than treating a matching example as confirmation.
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


def test_cytation_real_world_derived_examples_keep_their_provenance_caveat():
    """Upstream sanitized acquisition names remain provisional evidence."""
    record = _metadata_convention("cytation")
    names = (
        "A1_01_1_1_Phase Contrast_001.tif",
        "A1_01_1_2_Phase Contrast_001.tif",
    )
    assert set(names) <= set(record["examples"])
    assert "A1ROI1_02_1_1_Bright Field_High Contrast_003.tif" in record["examples"]
    assert record["status"] == "provisional"
    assert "f5440e9996908f2a108560961635ac9bc6693e3c" in record["source"]
    assert "BioTekTest.java" in record["source"]
    assert "sanitization" in record["source"]
    assert "not unchanged raw filenames" in record["source"]
    for field, name in enumerate(names, start=1):
        match = re.fullmatch(_metadata_pattern("cytation", "tif"), name)
        assert match is not None
        assert match.group("wellID") == "A1"
        assert int(match.group("fieldID")) == field
        assert match.group("chanID") == "Phase Contrast"
        assert match.group("timeID") == "001"


def test_scanr_user_stems_use_the_vendor_documented_tiff_suffix():
    """Vendor grammar confirms the suffix without fabricating acquired names."""
    record = _metadata_convention("scanr")
    stems = (
        "A1--W00001--P00001--Z00000--T00000--Alexa 488",
        "A1--W00001--P00001--Z00000--T00000--mCherry1",
    )
    assert record["examples"] == tuple(stem + ".tif" for stem in stems)
    assert record["status"] == "provisional"
    assert record["ext"] == "suffix"
    assert "https://forum.knime.com/t/combining-images-to-stack/2119" in record["source"]
    assert "forum.image.sc thread 14872" in record["source"]
    assert "Both cited stems omit the file extension" in record["source"]
    assert "vendor-documented .tif suffix" in record["source"]
    assert "scanR-Acquisition31_en_00__1_.pdf" in record["source"]
    assert "printed page 99" in record["source"]
    assert "well comment" in record["groups"]["wellID"]
    assert "not observed " in record["source"]
    assert "complete filenames" in record["source"]
    assert "ONE literal filename" not in record["source"]
    for stem, channel in zip(stems, ("Alexa 488", "mCherry1")):
        assert stem in record["source"]
        pattern = _metadata_pattern("scanr", "tif")
        assert re.fullmatch(pattern, stem) is None
        match = re.fullmatch(pattern, stem + ".tif")
        assert match is not None
        assert match.groupdict() == {
            "wellID": "A1", "fieldID": "00001", "sliceID": "00000",
            "timeID": "00000", "chanID": channel,
        }


@pytest.mark.parametrize("channel,identifier", [
    (1, "836B5662-3526-47FE-A453-0D1775E08D17"),
    (1, "6DE341F2-7E20-4CE1-859F-E368584F24DB"),
    (10, "836B5662-3526-47FE-A453-0D1775E08D17"),
    (12, "12345678-1234-1234-1234-123456789012"),
    (123, "abcdef12-abcd-1234-abcd-123456789abc"),
    (12, ""),
])
def test_imagexpress_wavelength_ends_before_a_fixed_uuid(channel, identifier):
    """Above-nine names are constructed controls, not acquired-file claims."""
    from spacr.utils import _extract_filename_metadata

    name = f"Plate_A01_s2_w{channel}{identifier}.tif"
    pattern = re.compile(_metadata_pattern("imagexpress", "tif"))
    match = pattern.fullmatch(name)
    assert match is not None
    assert match.groupdict() == {
        "plateID": "Plate", "wellID": "A01", "fieldID": "2",
        "chanID": str(channel), "AID": identifier,
    }
    assert _extract_filename_metadata([name], "/data/Plate", pattern,
                                      "imagexpress") == {
        ("Plate", "A01", "2", str(channel), None, None): [os.path.join("/data/Plate", name)],
    }
    for record in _METADATA_CONVENTIONS:
        if record["key"] not in {"imagexpress", "custom"}:
            assert re.compile(_metadata_pattern(record["key"], "tif")).fullmatch(
                name) is None


@pytest.mark.parametrize("suffix", [
    "12_thumb_836B5662-3526-47FE-A453-0D1775E08D17",
    "1A36B566-3526-47FE-A453-0D1775E08D17",
    "12A36B56622-3526-47FE-A453-0D1775E08D17",
    "12836B5662-3526-47FE-A453-0D1775E08D1",
    "12836B5662-3526-47FE-A453-0D1775E08D177",
    "12836B5662-3526-47FE-A453-0D1775E08D1G",
    "12836B5662_3526_47FE_A453_0D1775E08D17",
    "12arbitrary_suffix",
])
def test_imagexpress_refuses_malformed_uuid_and_thumbnail_suffixes(suffix):
    """A malformed acquisition suffix must not become channel digits."""
    pattern = re.compile(_metadata_pattern("imagexpress", "tif"))
    assert pattern.fullmatch(f"Plate_A01_s2_w{suffix}.tif") is None


INCELL_FULL_FORM_VENDOR_NAMES = (
    ("A_01_f1_c1_x405_mBlue_z1_t1.tif", "A_01", 1, 1),
    ("A_01_f1_c2_x488_mGreen_z1_t1.tif", "A_01", 1, 2),
    ("A_01_f1_c3_x561_mOrange_z1_t1.tif", "A_01", 1, 3),
    ("A_01_f2_c1_x405_mBlue_z1_t1.tif", "A_01", 2, 1),
    ("A_01_f2_c2_x488_mGreen_z1_t1.tif", "A_01", 2, 2),
    ("A_01_f2_c3_x561_mOrange_z1_t1.tif", "A_01", 2, 3),
    ("A_02_f1_c1_x405_mBlue_z1_t1.tif", "A_02", 1, 1),
    ("A_02_f1_c2_x488_mGreen_z1_t1.tif", "A_02", 1, 2),
    ("A_02_f1_c3_x561_mOrange_z1_t1.tif", "A_02", 1, 3),
    ("A_02_f2_c1_x405_mBlue_z1_t1.tif", "A_02", 2, 1),
    ("A_02_f2_c2_x488_mGreen_z1_t1.tif", "A_02", 2, 2),
    ("A_02_f2_c3_x561_mOrange_z1_t1.tif", "A_02", 2, 3),
)


@pytest.mark.parametrize("name,well,field,channel", INCELL_FULL_FORM_VENDOR_NAMES)
def test_incell_full_form_vendor_literals_keep_every_one_based_axis(
        name, well, field, channel, tmp_path):
    """Read literal names from Cytiva's 7.2 release notes, printed page 7."""
    from spacr.convert import _parse_by_convention
    from spacr.regex_infer import _metadata_match, _metadata_zero_based
    from spacr.utils import _extract_filename_metadata

    assert _metadata_zero_based("incell_full_form") == ()
    found = _metadata_match(name, "incell_full_form")
    assert found == {"wellID": well, "fieldID": str(field),
                     "chanID": str(channel), "sliceID": "1", "timeID": "1"}
    assert _parse_by_convention(name, "incell_full_form") == {
        "plate": None, "well": well.replace("_", ""), "well_token": well,
        "field": str(field), "field_number": field,
        "channel": f"C{channel}", "z": 1, "t": 1,
    }
    grouped = _extract_filename_metadata(
        [name], str(tmp_path), re.compile(_metadata_pattern("incell_full_form")),
        "incell_full_form")
    assert grouped == {
        (tmp_path.name, well.replace("_", ""), str(field), str(channel), "1", "1"):
            [str(tmp_path / name)]}


@pytest.mark.parametrize("name", [
    "A_01_f0_c1_x405_mBlue_z1_t1.tif",
    "A_01_f1_c0_x405_mBlue_z1_t1.tif",
    "A_01_f1_c1_x405_mBlue_z0_t1.tif",
    "A_01_f1_c1_x405_mBlue_z1_t0.tif",
    "A_00_f1_c1_x405_mBlue_z1_t1.tif",
    "A_01_f1_c1_x_mBlue_z1_t1.tif",
    "A_01_f1_c1_x405_m_z1_t1.tif",
    "A_01_f1_c1_x405_mBlue_z1_t1.tif.bak",
    "A - 01(fld 2 wv Blue - FITC).tif",
])
def test_incell_full_form_refuses_missing_or_zero_based_dimensions(name):
    """The vendor's positive indices never become guessed default planes."""
    from spacr.regex_infer import _metadata_match

    assert _metadata_match(name, "incell_full_form") is None


def test_incell_full_form_is_vendor_documented_and_offered_without_legacy_changes():
    """An explicit optional preference is separate from legacy IN Cell names."""
    record = _metadata_convention("incell_full_form")
    assert record["examples"] == tuple(row[0] for row in INCELL_FULL_FORM_VENDOR_NAMES)
    assert record["status"] == "confirmed"
    assert "INCell_Release_Notes_V7.2.pdf" in record["source"]
    assert "printed pages 6-7" in record["source"]
    assert "optional TIFF naming preference" in record["source"]
    assert "not generated acquisition fixtures" in record["source"]
    assert ("incell_full_form", 12, 12) in _metadata_autodetect(
        [row[0] for row in INCELL_FULL_FORM_VENDOR_NAMES])
    rows = dict(_metadata_convention_menu())["GE / Cytiva"]
    assert ("incell_full_form", record["label"], "confirmed") in rows
    assert _metadata_convention("incell")["examples"][0] == (
        "A - 01(fld 2 wv Blue - FITC).tif")


def test_incell_full_form_real_tiffs_reach_mask_without_identity_or_pixel_changes(tmp_path):
    """Synthetic TIFF pixels use the vendor's twelve literal documented names."""
    import numpy as np
    import tifffile

    from spacr.io import _rename_and_organize_image_files
    from spacr.utils import _map_wells

    source = tmp_path / "incell_run"
    source.mkdir()
    originals = {}
    for name, well, field, channel in INCELL_FULL_FORM_VENDOR_NAMES:
        value = int(well[-2:]) * 100 + field * 10 + channel
        path = source / name
        tifffile.imwrite(path, np.full((5, 7), value, dtype=np.uint16))
        originals[name] = path.read_bytes()
    assert _rename_and_organize_image_files(
        str(source), _metadata_pattern("incell_full_form"),
        metadata_type="incell_full_form", save_original_images=True) == 3
    stacks = list((source / "stack").glob("*.npy"))
    assert len(stacks) == 4
    identities = set()
    for path in stacks:
        plate, row, column, field, time, _ = _map_wells(path.name, timelapse=True)
        identities.add((plate, row, column, field, time))
        column_number, field_number = int(column[1:]), int(field[1:])
        expected = np.stack([
            np.full((5, 7), column_number * 100 + field_number * 10 + channel,
                    dtype=np.uint16) for channel in (1, 2, 3)], axis=2)
        assert np.array_equal(np.load(path), expected)
    assert identities == {
        ("incell_run", "r1", "c1", "f1", "t1"),
        ("incell_run", "r1", "c1", "f2", "t1"),
        ("incell_run", "r1", "c2", "f1", "t1"),
        ("incell_run", "r1", "c2", "f2", "t1"),
    }
    assert {path.name: path.read_bytes() for path in (source / "orig").iterdir()} == originals
