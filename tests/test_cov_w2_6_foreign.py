"""Somebody else's schema, and the four ways mapping it goes silently wrong.

The one mistake this module exists to prevent is a number that is wrong by
a factor of a few hundred and looks completely plausible -- their ``Area``
in micrometres squared written into a column labelled pixels squared. So
most of these tests are about the refusals: what the importer will NOT do
without being told the pixel size, the unit, or which table to read.

Every table here is a real file (CSV, TSV, Excel, Parquet, SQLite) read by
the real reader.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import foreign
from spacr.errors import ConfigurationError


# --------------------------------------------------------------------------
# turning an arbitrary header into a column name
# --------------------------------------------------------------------------

@pytest.mark.parametrize("header,expected", [
    ("Area (µm²)", "area_um2"),
    ("Mean Intensity", "mean_intensity"),
    ("  spaced  out  ", "spaced_out"),
    ("!!!", "column"),
    ("2ndChannel", "c2ndchannel"),
    ("ImageNumber", "imagenumber"),
])
def test_a_header_becomes_a_safe_lower_case_identifier(header, expected):
    """Deliberately lossy, and reversible only through the map file, which
    records the original source next to every target."""
    assert foreign._sanitise_column(header) == expected


def test_a_name_already_taken_gets_a_number_rather_than_overwriting():
    taken = {"foreign_area", "foreign_area_2"}
    assert foreign._unique("foreign_area", taken) == "foreign_area_3"
    assert foreign._unique("foreign_perimeter", taken) == "foreign_perimeter"


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

@pytest.mark.parametrize("written,normalised", [
    ("µm²", "um^2"), ("um**2", "um^2"), (" PX ", "px"),
    ("micron", "um"), ("counts", "counts"), (None, ""),
])
def test_a_unit_is_normalised_but_never_blanked(written, normalised):
    """An unknown unit is kept: blanking it would turn 'we do not know what
    this is' into 'it is in spaCR's units'."""
    assert foreign._norm_unit(written) == normalised


@pytest.mark.parametrize("unit,family,power", [
    ("um", "metric", 1), ("um^2", "metric", 2), ("um^3", "metric", 3),
    ("px", "pixel", 1), ("px^2", "pixel", 2),
    ("", None, 0), ("counts", "other", 0), ("au", "other", 0),
])
def test_a_unit_is_placed_in_a_family_or_declared_to_be_neither(unit, family,
                                                                power):
    assert foreign._unit_family(unit) == (family, power)


def test_a_unit_nobody_declared_says_so_in_the_plan():
    assert foreign._pretty_unit("") == "not declared"
    assert foreign._pretty_unit("um^2") == "um^2"


# --------------------------------------------------------------------------
# what counts as one of spaCR's own names
# --------------------------------------------------------------------------

def test_a_column_spacr_writes_is_recognised_as_one():
    assert foreign.is_spacr_name("cell_area") is True
    assert foreign.is_spacr_name("their_own_odd_column") is False


def test_a_name_the_parser_chokes_on_is_simply_not_spacrs(monkeypatch):
    """A crash here would refuse an import over a column name."""
    from spacr import feature_dict as fdict

    def _explode(_name):
        raise RuntimeError("the grammar fell over")

    monkeypatch.setattr(fdict, "parse_column", _explode)
    assert foreign.is_spacr_name("cell_area") is False


# --------------------------------------------------------------------------
# stems and keys
# --------------------------------------------------------------------------

def test_a_field_stem_produces_the_same_keys_a_native_run_would():
    stem = foreign._stem_of("plate1", "A01", 3)
    assert stem == "plate1_A01_3"
    plate, row_id, column_id, field, prc, prcf = foreign._keys_of_stem(stem)
    assert (plate, field) == ("plate1", 3)
    assert prc.startswith("plate1_")
    assert prcf.startswith(prc)


# --------------------------------------------------------------------------
# a mapping's factor -- and every reason there is not one
# --------------------------------------------------------------------------

def _mapping(**kwargs):
    kwargs.setdefault("source", "Area")
    kwargs.setdefault("target", "foreign_area")
    return foreign.ColumnMap(**kwargs)


def test_a_literal_factor_is_taken_at_its_word():
    assert _mapping(transform="*0.65").resolve(None) == (0.65, "")
    assert _mapping(transform="/1000").resolve(None) == (0.001, "")
    assert _mapping(transform="*0.65").is_literal is True
    assert _mapping(transform="area").is_literal is False


def test_dividing_by_zero_is_not_a_unit_conversion():
    mapping = _mapping(transform="/0")
    assert mapping.literal_factor is None
    factor, reason = mapping.resolve(0.65)
    assert factor is None
    assert "unknown transform" in reason


def test_an_identity_copy_needs_nothing_at_all():
    assert _mapping().resolve(None) == (1.0, "")


def test_declaring_a_conversion_and_then_not_converting_is_refused():
    """It would copy the values unchanged into a column labelled as
    converted, which is the shape of the whole bug."""
    factor, reason = _mapping(transform="identity", unit_in="um^2",
                              unit_out="px^2").resolve(0.65)
    assert factor is None
    assert 'transform is "identity"' in reason


def test_a_transform_nobody_defined_lists_the_ones_that_exist():
    factor, reason = _mapping(transform="log10").resolve(0.65)
    assert factor is None
    assert "unknown transform" in reason
    for name in foreign.TRANSFORMS:
        assert name in reason


def test_a_conversion_with_no_unit_in_is_not_guessable():
    factor, reason = _mapping(transform="area", unit_out="px^2").resolve(
        0.65)
    assert factor is None
    assert "needs unit_in" in reason


def test_a_unit_that_is_neither_length_nor_area_cannot_be_converted():
    factor, reason = _mapping(transform="area", unit_in="counts",
                              unit_out="px^2").resolve(0.65)
    assert factor is None
    assert "has no meaning" in reason


def test_a_unit_out_that_is_not_a_pixel_or_micrometre_is_refused():
    factor, reason = _mapping(transform="area", unit_in="um^2",
                              unit_out="counts").resolve(0.65)
    assert factor is None
    assert "needs unit_out to be a pixel or micrometre" in reason


def test_converting_a_unit_into_its_own_family_has_nothing_to_convert():
    factor, reason = _mapping(transform="area", unit_in="um^2",
                              unit_out="um^2").resolve(0.65)
    assert factor is None
    assert "nothing to convert" in reason


def test_a_power_mismatch_between_transform_and_units_is_refused():
    factor, reason = _mapping(transform="area", unit_in="um",
                              unit_out="px").resolve(0.65)
    assert factor is None
    assert "power-2 conversion" in reason


def test_an_unknown_pixel_size_never_silently_becomes_one():
    """The value is not multiplied by 1.0 and pretended to be pixels."""
    factor, reason = _mapping(transform="area", unit_in="um^2",
                              unit_out="px^2").resolve(None)
    assert factor is None
    assert "no pixel size was given" in reason


def test_a_pixel_size_that_is_not_a_number_is_refused_by_name():
    factor, reason = _mapping(transform="area", unit_in="um^2",
                              unit_out="px^2").resolve("half a micron")
    assert factor is None
    assert "is not a number" in reason


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_a_pixel_size_that_is_not_positive_and_finite_is_refused(bad):
    factor, reason = _mapping(transform="area", unit_in="um^2",
                              unit_out="px^2").resolve(bad)
    assert factor is None
    assert "must be a positive number" in reason


def test_micrometres_are_divided_by_the_pixel_size_to_the_right_power():
    """px = um / (um per px), squared for an area."""
    factor, reason = _mapping(transform="area", unit_in="um^2",
                              unit_out="px^2").resolve(0.5)
    assert reason == ""
    assert factor == pytest.approx(1.0 / 0.25)
    factor, _ = _mapping(transform="length", unit_in="um",
                         unit_out="px").resolve(0.5)
    assert factor == pytest.approx(2.0)


def test_pixels_are_multiplied_by_the_pixel_size_in_the_other_direction():
    factor, reason = _mapping(transform="volume", unit_in="px^3",
                              unit_out="um^3").resolve(0.5)
    assert reason == ""
    assert factor == pytest.approx(0.125)


# --------------------------------------------------------------------------
# a mapping as a row of the reviewable file
# --------------------------------------------------------------------------

def test_a_row_with_no_source_column_is_refused():
    """Every row must name the foreign column it describes."""
    with pytest.raises(ConfigurationError, match='no "source"'):
        foreign.ColumnMap.from_row({"target": "foreign_area"})


def test_a_row_round_trips_through_the_file_shape():
    mapping = _mapping(transform="area", unit_in="um^2", unit_out="px^2",
                       note="read from the header")
    again = foreign.ColumnMap.from_row(mapping.to_row())
    assert again == mapping


def test_blank_and_missing_cells_read_as_empty_not_as_nan():
    """A pandas NaN written into `target` would become the string 'nan' and
    then a column called nan."""
    mapping = foreign.ColumnMap.from_row(
        {"source": "Area", "target": float("nan"), "transform": None,
         "unit_in": "  um^2  "})
    assert mapping.target == ""
    assert mapping.transform == "identity"
    assert mapping.unit_in == "um^2"


# --------------------------------------------------------------------------
# proposing a mapping
# --------------------------------------------------------------------------

@pytest.fixture
def foreign_table():
    return pd.DataFrame({
        "ImageNumber": ["fov01", "fov01", "fov02"],
        "ObjectNumber": [1, 2, 1],
        "Area (µm²)": [10.0, 20.0, 30.0],
        "Perimeter_px": [4.0, 5.0, 6.0],
        "MeanIntensity": [100.0, 110.0, 120.0],
        "Treatment": ["dmso", "dmso", "drug"],
    })


def test_nothing_is_ever_proposed_onto_one_of_spacrs_own_names(foreign_table):
    """Their Area is not spaCR's cell_area: different segmentation,
    different definition, different unit."""
    proposals = foreign.infer_column_map(foreign_table)
    for mapping in proposals:
        assert mapping.target.startswith(foreign.FOREIGN_PREFIX)
        assert not foreign.is_spacr_name(mapping.target)


def test_the_join_keys_are_left_out_of_the_proposal(foreign_table):
    proposals = foreign.infer_column_map(foreign_table)
    sources = {m.source for m in proposals}
    assert "ImageNumber" not in sources
    assert "ObjectNumber" not in sources
    assert "Area (µm²)" in sources


def test_a_unit_in_the_header_becomes_a_declared_conversion(foreign_table):
    proposals = {m.source: m for m in foreign.infer_column_map(foreign_table)}
    area = proposals["Area (µm²)"]
    assert (area.transform, area.unit_in, area.unit_out) == ("area", "um^2",
                                                             "px^2")
    assert "needs a pixel size" in area.note


def test_a_column_already_in_pixels_is_copied_unchanged(foreign_table):
    proposals = {m.source: m for m in foreign.infer_column_map(foreign_table)}
    perimeter = proposals["Perimeter_px"]
    assert perimeter.transform == "identity"
    assert perimeter.unit_in == perimeter.unit_out == "px"
    assert "copied unchanged" in perimeter.note


def test_a_name_that_suggests_a_geometry_but_declares_no_unit_says_so():
    frame = pd.DataFrame({"MajorAxisLength": [1.0], "Label": [1]})
    proposals = {m.source: m for m in foreign.infer_column_map(frame)}
    note = proposals["MajorAxisLength"].note
    assert "no unit is declared" in note
    assert "transform=length" in note


def test_a_plain_column_says_what_to_do_if_it_needs_scaling(foreign_table):
    proposals = {m.source: m for m in foreign.infer_column_map(foreign_table)}
    assert "declare unit_in/unit_out" in proposals["Treatment"].note


def test_a_micrometre_column_with_no_geometry_hint_takes_its_power():
    frame = pd.DataFrame({"Something (um^3)": [1.0]})
    mapping, = foreign.infer_column_map(frame)
    assert mapping.transform == "volume"
    assert mapping.unit_out == "px^3"


def test_two_columns_that_sanitise_the_same_do_not_share_a_target():
    frame = pd.DataFrame({"Mean Intensity": [1.0], "mean_intensity": [2.0]})
    targets = [m.target for m in foreign.infer_column_map(frame)]
    assert len(set(targets)) == 2


def test_the_caller_may_name_the_keys_and_skip_more_columns(foreign_table):
    proposals = foreign.infer_column_map(
        foreign_table, image_key="Treatment", label_key="ObjectNumber",
        skip=["MeanIntensity"])
    sources = {m.source for m in proposals}
    assert sources == {"ImageNumber", "Area (µm²)", "Perimeter_px"}


def test_a_key_is_found_by_a_suffix_when_the_whole_name_does_not_match():
    frame = pd.DataFrame({"cp_imagenumber": ["a"], "the_objectnumber": [1],
                          "Area": [1.0]})
    proposals = {m.source for m in foreign.infer_column_map(frame)}
    assert proposals == {"Area"}


def test_a_table_with_no_recognisable_keys_proposes_every_column():
    frame = pd.DataFrame({"alpha": [1], "beta": [2]})
    assert len(foreign.infer_column_map(frame)) == 2


# --------------------------------------------------------------------------
# the reviewable file
# --------------------------------------------------------------------------

def test_the_map_file_says_what_a_reader_is_allowed_to_edit(tmp_path,
                                                            foreign_table):
    path = tmp_path / "column_map.csv"
    foreign.save_column_map(foreign.infer_column_map(foreign_table), path)
    text = path.read_text()
    assert text.startswith("# spaCR foreign column map")
    assert "leave \"target\" empty" in text
    assert "identity | length | area | volume" in text


def test_an_edited_map_comes_back_exactly_as_edited(tmp_path, foreign_table):
    path = tmp_path / "column_map.csv"
    foreign.save_column_map(foreign.infer_column_map(foreign_table), path)
    loaded = foreign.load_column_map(path)
    assert [m.source for m in loaded] == [
        "Area (µm²)", "Perimeter_px", "MeanIntensity", "Treatment"]
    assert loaded[0].transform == "area"


def test_a_map_file_that_is_not_there_says_which_one(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        foreign.load_column_map(tmp_path / "absent.csv")


def test_a_map_file_with_nothing_in_it_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("# only a comment\n")
    with pytest.raises(ConfigurationError, match="no rows"):
        foreign.load_column_map(path)


def test_a_csv_that_is_not_a_column_map_names_what_is_missing(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("gene,value\nA,1\n")
    with pytest.raises(ConfigurationError, match="missing column"):
        foreign.load_column_map(path)


def test_a_source_column_mapped_twice_is_refused(tmp_path):
    """Which one applies would be undefined."""
    path = tmp_path / "column_map.csv"
    path.write_text("source,target,transform,unit_in,unit_out,note\n"
                    "Area,foreign_area,identity,,,\n"
                    "Area,cell_area,area,um^2,px^2,\n")
    with pytest.raises(ConfigurationError, match="twice"):
        foreign.load_column_map(path)


def test_a_blank_line_in_the_middle_of_a_map_is_skipped(tmp_path):
    path = tmp_path / "column_map.csv"
    path.write_text("source,target,transform,unit_in,unit_out,note\n"
                    "Area,foreign_area,identity,,,\n"
                    ",,,,,\n"
                    "Perimeter,foreign_perimeter,identity,,,\n")
    assert [m.source for m in foreign.load_column_map(path)] == [
        "Area", "Perimeter"]


# --------------------------------------------------------------------------
# reading their measurement table
# --------------------------------------------------------------------------

@pytest.fixture
def rows():
    return pd.DataFrame({"ImageNumber": ["fov01", "fov02"],
                         "ObjectNumber": [1, 1],
                         "Area": [10.0, 20.0]})


def test_a_dataframe_is_copied_rather_than_shared(rows):
    out = foreign.read_measurements(rows)
    out.loc[0, "Area"] = -1
    assert rows.loc[0, "Area"] == 10.0


def test_a_table_that_is_not_there_says_which_one(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        foreign.read_measurements(tmp_path / "absent.csv")


def test_a_format_with_no_reader_lists_the_ones_there_are(tmp_path):
    path = tmp_path / "results.mat"
    path.write_bytes(b"")
    with pytest.raises(ConfigurationError, match="no reader for"):
        foreign.read_measurements(path)


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx", ".parquet"])
def test_every_supported_table_format_reads_to_the_same_frame(tmp_path, rows,
                                                              suffix):
    path = tmp_path / f"results{suffix}"
    if suffix == ".csv":
        rows.to_csv(path, index=False)
    elif suffix == ".tsv":
        rows.to_csv(path, sep="\t", index=False)
    elif suffix == ".xlsx":
        rows.to_excel(path, index=False)
    else:
        rows.to_parquet(path, index=False)
    out = foreign.read_measurements(path)
    assert list(out.columns) == list(rows.columns)
    assert len(out) == 2


def test_a_sqlite_source_with_one_table_needs_no_name(tmp_path, rows):
    path = tmp_path / "results.db"
    connection = sqlite3.connect(path)
    try:
        rows.to_sql("objects", connection, index=False)
    finally:
        connection.close()
    assert len(foreign.read_measurements(path)) == 2


def test_a_sqlite_source_with_several_tables_must_be_told_which(tmp_path,
                                                                rows):
    """Picking one at random is how you import the wrong 40 000 rows."""
    path = tmp_path / "results.db"
    connection = sqlite3.connect(path)
    try:
        rows.to_sql("cells", connection, index=False)
        rows.to_sql("nuclei", connection, index=False)
    finally:
        connection.close()
    with pytest.raises(ConfigurationError, match="name the one to import"):
        foreign.read_measurements(path)
    assert len(foreign.read_measurements(path, table="nuclei")) == 2


def test_a_sqlite_table_that_is_not_there_lists_the_ones_that_are(tmp_path,
                                                                  rows):
    path = tmp_path / "results.db"
    connection = sqlite3.connect(path)
    try:
        rows.to_sql("cells", connection, index=False)
    finally:
        connection.close()
    with pytest.raises(ConfigurationError, match="has no table"):
        foreign.read_measurements(path, table="nuclei")


def test_a_table_that_will_not_parse_says_so_with_its_path(tmp_path):
    path = tmp_path / "results.parquet"
    path.write_bytes(b"this is not parquet")
    with pytest.raises(ConfigurationError,
                       match="could not be read as a measurement table"):
        foreign.read_measurements(path)


def test_every_user_table_in_a_database_is_listed_in_schema_order(tmp_path,
                                                                  rows):
    path = tmp_path / "results.db"
    connection = sqlite3.connect(path)
    try:
        rows.to_sql("cells", connection, index=False)
        rows.to_sql("nuclei", connection, index=False)
        connection.execute("CREATE VIEW both AS SELECT * FROM cells")
    finally:
        connection.close()
    assert foreign._sqlite_tables(path) == ["cells", "nuclei", "both"]


# --------------------------------------------------------------------------
# pairing images with masks
# --------------------------------------------------------------------------

def test_a_pairing_is_only_ok_when_nothing_was_left_over():
    """A converter that quietly kept the intersection would produce a
    smaller, perfectly consistent, wrong experiment."""
    assert foreign.PairingReport().ok is True
    assert foreign.PairingReport(
        images_without_masks=[("/a.tif", "cell")]).ok is False
    assert foreign.PairingReport(
        masks_without_images=[("/m.tif", "cell")]).ok is False
    assert foreign.PairingReport(
        unreadable_masks=[("/m.tif", "not an image")]).ok is False


def test_a_stack_of_masks_is_read_down_to_its_first_plane(tmp_path):
    """Their mask folder routinely holds a one-plane stack rather than a
    plain 2-D image, and that is still one label image."""
    import tifffile

    path = tmp_path / "stack_mask.tif"
    plane = np.array([[0, 1], [2, 0]], dtype=np.uint16)
    tifffile.imwrite(path, np.stack([plane, plane * 2]))
    assert np.array_equal(foreign._read_mask(str(path)), plane)


def test_something_that_is_not_a_label_image_at_all_is_refused_by_shape(
        tmp_path):
    import tifffile

    path = tmp_path / "line.tif"
    tifffile.imwrite(path, np.zeros((1, 5), dtype=np.uint8))
    with pytest.raises(ConfigurationError, match="must be a 2-D label image"):
        foreign._read_mask(str(path))


def test_only_the_non_zero_labels_of_a_mask_are_its_objects():
    mask = np.array([[0, 1, 1], [0, 2, 0], [3, 3, 0]])
    assert foreign._mask_labels(mask) == (1, 2, 3)
    assert foreign._mask_labels(np.zeros((3, 3))) == ()


# --------------------------------------------------------------------------
# the join report
# --------------------------------------------------------------------------

def test_an_empty_join_matches_everything_it_was_given():
    report = foreign.JoinReport()
    assert report.match_rate == 1.0
    assert report.rows_unmatched == 0


def test_the_unmatched_row_count_is_the_honest_headline_number():
    report = foreign.JoinReport(rows_total=100, rows_matched=60)
    assert report.rows_unmatched == 40
    assert report.match_rate == pytest.approx(0.6)


def test_every_kind_of_failure_is_counted_separately():
    report = foreign.JoinReport(
        unresolved_fields=[("fov99", 3), ("fov98", 2)],
        rows_no_object=[("plate1_A01_1", 4)],
        objects_unmeasured=[("plate1_A01_1", 5)])
    assert report.n_unresolved == 5
    assert report.n_no_object == 4
    assert report.n_objects_unmeasured == 5


def test_the_join_summary_names_every_failure_and_caps_the_lists():
    report = foreign.JoinReport(
        image_key="ImageNumber", label_key="ObjectNumber",
        rows_total=100, rows_matched=50,
        unresolved_fields=[(f"fov{i:02d}", 1) for i in range(12)],
        rows_no_object=[(f"plate1_A{i:02d}_1", 1) for i in range(12)],
        objects_unmeasured=[(f"plate1_B{i:02d}_1", 1) for i in range(12)],
        ambiguous_keys=["fov1", "fov2"],
        examples=["fov01 label 7"])
    text = report.summary()
    assert "ImageNumber" in text and "ObjectNumber" in text
    assert "… and 2 " in text
    assert text.count("… and 2 ") == 3          # one per capped list
    assert "image-key spelling(s)" in text
    assert "e.g. fov01 label 7" in text


def test_a_single_field_table_says_it_has_no_image_column():
    report = foreign.JoinReport(object_type="nucleus")
    assert "(single field — no image column)" in report.key_description
    assert "(row order)" in report.key_description


# --------------------------------------------------------------------------
# which image a row is talking about
# --------------------------------------------------------------------------

def _map_row(**kwargs):
    row = {"plate": "plate1", "well": "A01", "field": 1,
           "source": "/data/raw/fov01_C1.tif",
           "source_relpath": "raw/fov01_C1.tif",
           "source_field": "fov01",
           "target": "plate1_A01_T0001F001L01C01.tif"}
    row.update(kwargs)
    return row


def test_a_row_answers_to_every_spelling_of_its_image():
    aliases = set(foreign._field_aliases(_map_row()))
    assert "raw/fov01_C1.tif" in aliases
    assert "/data/raw/fov01_C1.tif" in aliases
    assert "fov01_C1.tif" in aliases
    assert "fov01" in aliases
    assert "plate1_A01_1" in aliases


def test_a_row_with_nothing_but_a_plate_well_and_field_still_has_a_stem():
    aliases = foreign._field_aliases(
        {"plate": "plate1", "well": "A01", "field": 2})
    assert aliases == ["plate1_A01_2"]


def test_a_spelling_that_would_point_at_two_fields_is_dropped():
    """Picking one is how half an import lands in the wrong well."""
    rows = [_map_row(field=1, source="/data/shared.tif",
                     source_relpath="shared.tif", source_field="fov01"),
            _map_row(field=2, source="/data/shared.tif",
                     source_relpath="shared.tif", source_field="fov02")]
    index, ambiguous = foreign._build_field_index(rows)
    assert "shared.tif" in ambiguous
    assert "shared.tif" not in index
    assert index["fov01"] == "plate1_A01_1"


def test_an_image_key_is_resolved_however_it_was_written():
    index, _ambiguous = foreign._build_field_index([_map_row()])
    for written in ("raw/fov01_C1.tif", "raw\\fov01_C1.tif",
                    "/data/raw/fov01_C1.tif", "FOV01", "fov01_C1",
                    "plate1_A01_1"):
        assert foreign._resolve_field(written, index) == "plate1_A01_1", written


def test_an_image_key_that_names_nothing_resolves_to_nothing():
    index, _ambiguous = foreign._build_field_index([_map_row()])
    assert foreign._resolve_field(None, index) is None
    assert foreign._resolve_field("   ", index) is None
    assert foreign._resolve_field("fov99", index) is None


def test_a_unit_nobody_has_an_alias_for_is_kept_verbatim():
    """Blanking it would turn 'we do not recognise this' into 'no unit'."""
    assert foreign._norm_unit("furlongs") == "furlongs"
    assert foreign._unit_family("furlongs") == ("other", 0)


def test_two_unrecognised_units_that_differ_still_declare_a_conversion():
    """`counts -> au` is a conversion somebody has to explain, even though
    this module has no opinion about either unit."""
    assert _mapping(unit_in="counts", unit_out="au").declares_conversion \
        is True
    assert _mapping(unit_in="counts", unit_out="counts").declares_conversion \
        is False


def test_a_column_named_for_a_volume_is_guessed_to_be_one():
    assert foreign._column_family_hint("Cell Volume") == "volume"
    assert foreign._column_family_hint("Nucleus Area") == "area"
    assert foreign._column_family_hint("Feret Diameter") == "length"
    assert foreign._column_family_hint("Treatment") == "identity"


# --------------------------------------------------------------------------
# applying a resolved mapping
# --------------------------------------------------------------------------

def _resolved(factor, target="foreign_area", unit="px^2", calibrated=True,
              status="mapped", **mapping_kwargs):
    return foreign.ResolvedColumn(
        mapping=_mapping(**mapping_kwargs), target=target, factor=factor,
        calibrated=calibrated, unit=unit, status=status)


def test_a_factor_of_one_leaves_the_values_alone():
    values = pd.Series([1.0, 2.0, 3.0])
    out = _resolved(1.0).apply(values)
    assert out is values
    assert _resolved(None).apply(values) is values


def test_a_factor_multiplies_every_value():
    out = _resolved(4.0).apply(pd.Series([1.0, 2.0]))
    assert list(out) == [4.0, 8.0]


def test_a_column_of_names_is_passed_through_however_the_factor_reads():
    """Multiplying a string column of treatment names by 0.65 is not a unit
    conversion, it is a crash."""
    values = pd.Series(["dmso", "drug"])
    assert _resolved(0.65).apply(values) is values


def test_the_provenance_row_records_what_was_actually_applied():
    resolution = _resolved(2.0, unit="px^2", calibrated=True,
                           transform="area", unit_in="um^2", unit_out="px^2")
    record = resolution.to_record("foreign_cell")
    assert record["table"] == "foreign_cell"
    assert record["column"] == "foreign_area"
    assert record["source_column"] == "Area"
    assert record["transform"] == "area"
    assert record["factor"] == 2.0
    assert record["unit"] == "px^2"
    assert record["calibrated"] == 1


def test_an_uncalibrated_column_records_the_unit_it_really_is_in():
    """The flag that stops a um^2 column being read as px^2."""
    resolution = _resolved(None, unit="um^2", calibrated=False,
                           status="uncalibrated", transform="area",
                           unit_in="um^2", unit_out="px^2")
    record = resolution.to_record("foreign_cell")
    assert record["calibrated"] == 0
    assert record["unit"] == "um^2"
    assert record["factor"] is None
    assert resolution.source == "Area"


# --------------------------------------------------------------------------
# settling every collision, deterministically
# --------------------------------------------------------------------------

def _resolve(maps, unmapped=(), um_per_px=0.5, prefix=foreign.FOREIGN_PREFIX,
             on_conflict="refuse", allow_spacr_targets=False):
    return foreign._resolve_columns(list(maps), list(unmapped), um_per_px,
                                    prefix, on_conflict, allow_spacr_targets)


def test_a_mapping_with_no_target_is_imported_under_the_prefix():
    """Dropping a column the user cared about, quietly, is the failure mode
    this exists to prevent."""
    resolved, conflicts, warnings = _resolve([_mapping(source="Area",
                                                       target="")])
    assert resolved[0].target == f"{foreign.FOREIGN_PREFIX}area"
    assert resolved[0].status == "unmapped"
    assert "rather than dropped" in resolved[0].reason
    assert conflicts == [] and warnings == []


def test_a_reserved_key_column_is_never_written_over_at_any_setting():
    """A foreign value there does not corrupt a measurement, it corrupts the
    index every table is joined on -- so the rename happens even where the
    caller opted into spaCR's own names."""
    reserved = foreign.RESERVED_COLUMNS[0]
    for mode in foreign.ON_CONFLICT:
        resolved, conflicts, warnings = _resolve(
            [_mapping(source="Their Key", target=reserved)],
            on_conflict=mode, allow_spacr_targets=True)
        assert resolved[0].target != reserved
        assert resolved[0].target.startswith(foreign.FOREIGN_PREFIX)
        assert resolved[0].status == "renamed"
        assert conflicts[0].kind == "reserved"
        assert conflicts[0].blocking is (mode != "rename")
        assert warnings and "was renamed" in warnings[0]


def test_writing_into_one_of_spacrs_own_names_is_refused_by_default():
    resolved, conflicts, warnings = _resolve(
        [_mapping(source="Area", target="cell_area")])
    assert conflicts[0].kind == "spacr_name"
    assert conflicts[0].blocking is True
    assert "allow_spacr_targets=True" in conflicts[0].detail
    assert resolved[0].target.startswith(foreign.FOREIGN_PREFIX)


def test_renaming_turns_that_refusal_into_a_warning():
    resolved, conflicts, warnings = _resolve(
        [_mapping(source="Area", target="cell_area")], on_conflict="rename")
    assert conflicts[0].blocking is False
    assert warnings and "was renamed" in warnings[0]
    assert resolved[0].status == "renamed"


def test_opting_in_writes_their_values_into_spacrs_slot():
    """A decision with a name on it."""
    resolved, conflicts, _warnings = _resolve(
        [_mapping(source="Area", target="cell_area")],
        allow_spacr_targets=True)
    assert resolved[0].target == "cell_area"
    assert resolved[0].status == "mapped"
    assert [c for c in conflicts if c.blocking] == []


def test_two_sources_cannot_share_one_target():
    """Two different measurements cannot share one column."""
    resolved, conflicts, _warnings = _resolve([
        _mapping(source="Area", target="foreign_area"),
        _mapping(source="Area2", target="foreign_area"),
    ])
    assert conflicts[0].kind == "duplicate_target"
    assert conflicts[0].blocking is True
    assert resolved[0].target != resolved[1].target


def test_a_duplicate_target_can_be_renamed_instead_of_refused():
    resolved, conflicts, warnings = _resolve([
        _mapping(source="Area", target="foreign_area"),
        _mapping(source="Area2", target="foreign_area"),
    ], on_conflict="rename")
    assert conflicts[0].blocking is False
    assert any("was renamed" in w for w in warnings)
    assert len({r.target for r in resolved}) == 2


def test_a_column_that_cannot_be_converted_is_moved_under_the_prefix():
    """The values are stored unconverted and say so, rather than being
    multiplied by 1.0 and labelled as pixels."""
    resolved, _conflicts, warnings = _resolve(
        [_mapping(source="Area", target="cell_area", transform="area",
                  unit_in="um^2", unit_out="px^2")],
        um_per_px=None, allow_spacr_targets=True)
    resolution = resolved[0]
    assert resolution.calibrated is False
    assert resolution.factor is None
    assert resolution.unit == "um^2"
    assert resolution.target.startswith(foreign.FOREIGN_PREFIX)
    assert resolution.status == "uncalibrated"
    assert any("UNCALIBRATED" in w for w in warnings)
    assert any("calibrated = 0" in w for w in warnings)


def test_a_column_already_under_the_prefix_keeps_its_name_when_uncalibrated():
    resolved, _conflicts, _warnings = _resolve(
        [_mapping(source="Area", target="foreign_area", transform="area",
                  unit_in="um^2", unit_out="px^2")], um_per_px=None)
    assert resolved[0].target == "foreign_area"
    assert resolved[0].calibrated is False


def test_a_column_literally_named_like_one_of_spacrs_is_flagged_not_blocked():
    """The two are not the same measurement, and a reader has to be told."""
    resolved, conflicts, _warnings = _resolve(
        [_mapping(source="cell_area", target="")])
    shadow, = [c for c in conflicts if c.kind == "shadows_spacr"]
    assert shadow.blocking is False
    assert "not the same measurement" in shadow.detail
    assert resolved[0].target == f"{foreign.FOREIGN_PREFIX}cell_area"


def test_a_column_nobody_mapped_at_all_is_still_imported():
    resolved, _conflicts, _warnings = _resolve([], unmapped=["Treatment"])
    resolution = resolved[0]
    assert resolution.target == f"{foreign.FOREIGN_PREFIX}treatment"
    assert resolution.status == "unmapped"
    assert resolution.factor == 1.0
    assert resolution.calibrated is False
    assert "nothing is known about its unit" in resolution.reason


def test_the_same_mappings_always_resolve_the_same_way():
    """'run_import applies exactly what was saved' is only checkable if the
    derivation is deterministic."""
    maps = [_mapping(source="Area", target="cell_area"),
            _mapping(source="Area2", target="cell_area")]
    first = _resolve(maps, unmapped=["Treatment"])
    again = _resolve(maps, unmapped=["Treatment"])
    assert [r.target for r in first[0]] == [r.target for r in again[0]]
    assert [str(c) for c in first[1]] == [str(c) for c in again[1]]
    assert first[2] == again[2]


def test_a_conflict_prints_what_it_is_about():
    conflict = foreign.Conflict("spacr_name", "Area", "cell_area",
                                "it means something else")
    assert str(conflict) == ("[spacr_name] 'Area' -> 'cell_area': "
                             "it means something else")


# --------------------------------------------------------------------------
# which folders hold which masks
# --------------------------------------------------------------------------

def test_no_mask_folder_at_all_is_allowed():
    assert foreign._mask_folders(None) == {}


def test_one_folder_on_its_own_is_taken_as_the_cell_masks():
    assert foreign._mask_folders("/data/masks") == {"cell": "/data/masks"}
    assert foreign._mask_folders(Path("/data/masks")) == {
        "cell": "/data/masks"}


def test_mask_folders_come_back_in_the_order_a_merged_array_holds_them():
    from spacr import crops as cropping

    given = {"pathogen": "/p", "cell": "/c", "nucleus": "/n"}
    out = foreign._mask_folders(given)
    assert list(out) == [name for name in cropping.MASK_PLANE_ORDER
                         if name in given]


def test_mask_folders_may_be_given_as_pairs():
    out = foreign._mask_folders([("cell", "/c"), ("nucleus", "/n")])
    assert out["cell"] == "/c" and out["nucleus"] == "/n"


def test_an_object_type_with_no_plane_in_a_merged_array_is_refused():
    with pytest.raises(ConfigurationError, match="Unknown mask object type"):
        foreign._mask_folders({"mitochondrion": "/m"})
