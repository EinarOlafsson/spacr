"""The plate designer: does it place wells correctly, and does it warn?

The placement tests plant a known layout and assert the exact wells come back.
The warning tests build a design whose flaw is known by construction -- every
control on the edge, a control confined to one column, one replicate -- and
assert the specific finding fires, and equally that it does *not* fire for a
design that does not have that flaw. A checker that warns about everything is
read as a disclaimer and skipped.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.plate_layout import (  # noqa: E402
    EDGE_LEAVE_EMPTY, EDGE_USE, PLATE_FORMATS,
    ROLE_NEGATIVE, ROLE_POSITIVE, ROLE_TREATMENT,
    Condition, PlateDesign, assign_wells, check_design, format_findings,
    is_edge, plate_shape, to_settings_fragment, write_design,
)


def _keys(findings) -> set:
    return {finding.key for finding in findings}


def _design(**over) -> PlateDesign:
    base = dict(
        plate_id="plate1", plate_format=96,
        conditions=(
            Condition("dmso", 6, ROLE_NEGATIVE),
            Condition("pyrimethamine", 6, ROLE_POSITIVE),
            Condition("drug_a", 12),
            Condition("drug_b", 12),
        ),
        layout="random", edge_policy=EDGE_USE, seed=1,
    )
    base.update(over)
    return PlateDesign(**base)


# -- geometry ---------------------------------------------------------------


def test_the_plate_formats_are_the_real_ones():
    assert plate_shape(96) == (8, 12)
    assert plate_shape(384) == (16, 24)
    assert plate_shape(1536) == (32, 48)
    for wells, (rows, columns) in PLATE_FORMATS.items():
        assert rows * columns == wells


def test_an_unknown_format_is_refused():
    with pytest.raises(ValueError, match="not a known plate"):
        plate_shape(100)
    with pytest.raises(ValueError, match="not a known plate"):
        PlateDesign(plate_format=100)


def test_the_edge_is_the_outer_ring():
    for row, column, expected in ((1, 1, True), (1, 6, True), (8, 12, True),
                                  (4, 1, True), (4, 12, True),
                                  (2, 2, False), (4, 6, False), (7, 11, False)):
        assert is_edge(row, column, 8, 12) is expected
    # 36 of a 96-well plate.
    edges = sum(is_edge(r, c, 8, 12)
                for r in range(1, 9) for c in range(1, 13))
    assert edges == 36


# -- placement --------------------------------------------------------------


def test_the_ids_are_the_ones_the_pipeline_parses_out_of_file_names():
    """The whole reason for the export: the map must join to measurements
    without a translation step."""
    from spacr.schema import parse_field_stem

    table = assign_wells(_design(layout="row"))
    first = table.iloc[0]
    parsed = parse_field_stem(f"{first['plateID']}_{first['well']}_1")
    assert parsed.plateID == first["plateID"]
    assert parsed.rowID == first["rowID"]
    assert parsed.columnID == first["columnID"]


def test_a_row_layout_fills_row_by_row():
    design = _design(layout="row", conditions=(Condition("a", 12),
                                               Condition("b", 12)))
    table = assign_wells(design)
    a_wells = sorted(table.loc[table["condition"] == "a", "well"])
    assert a_wells == [f"A{n:02d}" for n in range(1, 13)]
    b_wells = sorted(table.loc[table["condition"] == "b", "well"])
    assert b_wells == [f"B{n:02d}" for n in range(1, 13)]


def test_a_column_layout_fills_column_by_column():
    design = _design(layout="column", conditions=(Condition("a", 8),
                                                  Condition("b", 8)))
    table = assign_wells(design)
    a_wells = sorted(table.loc[table["condition"] == "a", "well"])
    assert a_wells == [f"{letter}01" for letter in "ABCDEFGH"]


def test_a_random_layout_is_reproducible_from_its_seed():
    first = assign_wells(_design(seed=7))
    second = assign_wells(_design(seed=7))
    pd.testing.assert_frame_equal(first, second)
    other = assign_wells(_design(seed=8))
    assert not first["condition"].equals(other["condition"])


def test_leaving_the_edge_empty_uses_only_the_interior():
    design = _design(edge_policy=EDGE_LEAVE_EMPTY)
    table = assign_wells(design)
    assert not table["is_edge"].any()
    assert design.wells_available == 6 * 10


def test_every_replicate_gets_its_own_well():
    design = _design()
    table = assign_wells(design)
    assert len(table) == design.wells_requested
    assert table["well"].nunique() == len(table)
    for condition in design.conditions:
        block = table.loc[table["condition"] == condition.name]
        assert len(block) == condition.replicates
        assert sorted(block["replicate"]) == list(
            range(1, condition.replicates + 1))


def test_a_design_that_does_not_fit_is_refused_not_truncated():
    """A silently dropped replicate is a plate that does not match its map."""
    design = _design(conditions=(Condition("a", 200),))
    with pytest.raises(ValueError, match="only 96 usable"):
        assign_wells(design)
    assert "does_not_fit" in _keys(check_design(design))


def test_an_empty_design_is_an_empty_table_not_a_crash():
    table = assign_wells(_design(conditions=()))
    assert table.empty
    assert list(table.columns)[:3] == ["plateID", "well", "rowID"]
    assert "no_conditions" in _keys(check_design(_design(conditions=())))


def test_a_condition_with_no_replicates_is_refused_at_construction():
    with pytest.raises(ValueError, match="not in the experiment"):
        Condition("a", 0)
    with pytest.raises(ValueError, match="needs a name"):
        Condition("  ", 3)
    with pytest.raises(ValueError, match="role"):
        Condition("a", 3, "control")


# -- the edge warning, which is the point -----------------------------------


def test_controls_confined_to_the_edge_are_warned_about():
    """The named requirement. A column layout with controls first puts every
    one of them in column 1, which is the plate edge."""
    design = _design(layout="column",
                     conditions=(Condition("neg", 8, ROLE_NEGATIVE),
                                 Condition("drug", 24)))
    table = assign_wells(design)
    controls = table.loc[table["role"] == ROLE_NEGATIVE]
    assert bool(controls["is_edge"].all()), "the fixture must be edge-only"

    findings = check_design(design, table)
    assert "controls_all_on_edge" in _keys(findings)
    message = next(f.message for f in findings
                   if f.key == "controls_all_on_edge")
    assert "evaporat" in message
    assert "illumination correction" in message, (
        "the warning should say why this codebase knows about edge effects")


def test_controls_spread_through_the_interior_are_not_warned_about():
    """The complement, and the more important half: a checker that fires on a
    good design is one nobody reads."""
    design = _design(edge_policy=EDGE_LEAVE_EMPTY, seed=3)
    findings = check_design(design)
    assert "controls_all_on_edge" not in _keys(findings)
    assert "controls_mostly_on_edge" not in _keys(findings)


def test_a_control_confined_to_one_column_is_warned_about_separately():
    """Edge and confounding are different problems: a control in the middle
    column is not on an edge and is still confounded with position."""
    design = _design(layout="column",
                     conditions=(Condition("filler", 40),
                                 Condition("neg", 8, ROLE_NEGATIVE)))
    table = assign_wells(design)
    controls = table.loc[table["role"] == ROLE_NEGATIVE]
    assert not controls["is_edge"].all(), "these should not all be edge wells"

    findings = check_design(design, table)
    assert "negative_control_in_one_column" in _keys(findings)
    assert "controls_all_on_edge" not in _keys(findings)


def test_a_control_confined_to_one_row_is_warned_about():
    design = _design(layout="row",
                     conditions=(Condition("filler", 12),
                                 Condition("neg", 12, ROLE_NEGATIVE)))
    findings = check_design(design)
    assert "negative_control_in_one_row" in _keys(findings)


def test_a_block_layout_is_told_it_confounds_position():
    findings = check_design(_design(layout="block"))
    assert "block_layout_confounds_position" in _keys(findings)
    assert "block_layout_confounds_position" not in _keys(
        check_design(_design(layout="random")))


# -- the other findings -----------------------------------------------------


def test_a_missing_negative_control_is_a_warning_and_a_missing_positive_a_note():
    findings = check_design(_design(conditions=(Condition("a", 12),)))
    by_key = {f.key: f for f in findings}
    assert by_key["no_negative_control"].severity == "warn"
    assert by_key["no_positive_control"].severity == "note"
    assert "no_negative_control" not in _keys(check_design(_design()))


def test_a_single_replicate_condition_is_named():
    findings = check_design(_design(conditions=(
        Condition("neg", 6, ROLE_NEGATIVE), Condition("lonely", 1))))
    message = next(f.message for f in findings if f.key == "single_replicate")
    assert "lonely" in message
    assert "single_replicate" not in _keys(check_design(_design()))


def test_leaving_the_edge_empty_says_what_it_cost():
    findings = check_design(_design(edge_policy=EDGE_LEAVE_EMPTY))
    message = next(f.message for f in findings if f.key == "edge_left_empty")
    assert "36 of 96" in message


def test_findings_are_ordered_worst_first():
    findings = check_design(_design(
        layout="column",
        conditions=(Condition("neg", 8, ROLE_NEGATIVE),
                    Condition("lonely", 1))))
    order = ["error", "warn", "note"]
    positions = [order.index(f.severity) for f in findings]
    assert positions == sorted(positions)


def test_format_findings_says_so_when_there_is_nothing_to_say():
    assert "No problems" in format_findings([])
    assert format_findings(check_design(_design())).strip()


# -- export -----------------------------------------------------------------


def test_a_whole_row_layout_exports_as_settings_the_pipeline_reads():
    design = _design(layout="row", conditions=(
        Condition("dmso", 12, ROLE_NEGATIVE), Condition("drug", 12)))
    fragment = to_settings_fragment(design)
    assert fragment["expressible"] is True
    assert fragment["settings"]["treatments"] == ["dmso", "drug"]
    assert fragment["settings"]["treatment_plate_metadata"] == [["r1"], ["r2"]]
    assert fragment["settings"]["negative_control"] == "dmso"


def test_the_exported_settings_are_what_annotate_conditions_consumes():
    """Round trip: feed the fragment to the real annotator and check the
    wells come back labelled the way the design said."""
    from spacr.utils import annotate_conditions

    design = _design(layout="row", conditions=(
        Condition("dmso", 12, ROLE_NEGATIVE), Condition("drug", 12)))
    table = assign_wells(design)
    fragment = to_settings_fragment(design, table)

    measurements = table[["rowID", "columnID"]].copy()
    annotated = annotate_conditions(
        measurements,
        treatments=fragment["settings"]["treatments"],
        treatment_loc=fragment["settings"]["treatment_plate_metadata"])
    assert list(annotated["treatment"]) == list(table["condition"])


def test_a_randomised_layout_says_it_cannot_be_expressed_as_settings():
    """The honest refusal. annotate_conditions addresses wells by row and
    column id only, and a randomised design has no such expression -- so the
    long-form table is the artifact, and saying so beats an approximation
    that would mislabel wells."""
    fragment = to_settings_fragment(_design(layout="random", seed=1))
    assert fragment["expressible"] is False
    assert "plate_map.csv" in fragment["reason"]
    assert "rowID" in fragment["reason"]
    assert fragment["settings"] == {}


def test_write_design_leaves_three_files_that_read_back(tmp_path):
    design = _design()
    paths = write_design(design, tmp_path)
    assert set(paths) == {"plate_map", "design", "settings"}
    for path in paths.values():
        assert path.exists()

    table = pd.read_csv(paths["plate_map"])
    assert len(table) == design.wells_requested
    assert set(["plateID", "rowID", "columnID", "condition", "role",
                "replicate", "is_edge"]).issubset(table.columns)

    record = json.loads(paths["design"].read_text(encoding="utf-8"))
    assert record["plate_id"] == "plate1"
    assert record["seed"] == 1
    assert record["layout"] == "random"
    assert [c["name"] for c in record["conditions"]] == [
        c.name for c in design.conditions]
    # The warnings that were shown go on the record, not only on a screen.
    assert isinstance(record["findings"], list)


def test_the_written_record_is_enough_to_regenerate_the_layout(tmp_path):
    """A design nobody can reproduce is a design nobody can check."""
    original = _design(seed=11)
    paths = write_design(original, tmp_path)
    record = json.loads(paths["design"].read_text(encoding="utf-8"))

    rebuilt = PlateDesign(
        plate_id=record["plate_id"],
        plate_format=record["plate_format"],
        conditions=tuple(Condition(**c) for c in record["conditions"]),
        layout=record["layout"],
        edge_policy=record["edge_policy"],
        seed=record["seed"])
    pd.testing.assert_frame_equal(
        assign_wells(rebuilt),
        pd.read_csv(paths["plate_map"]),
        check_dtype=False)
