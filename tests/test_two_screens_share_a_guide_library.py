"""Two screens that share a guide library stack into ONE frame.

Instruction 122, parts 1 and 2:

    "these screens share many of the same grnas and many measurements. so if a
     user say ran 2 screens they could theoretically easily plug in the
     independent variables and then the measurements table and regress
     anything. problem here would be: 1. that the screens would bothe have
     plates1,2,3,4 sh that would have to be fixed somewow, e.g. with a new
     metadataitem callsed screenID."

Both screens have ``plate1``..``plate4``. The answer is NOT to rewrite the
plate id -- ``on_collision='qualify'`` already does that and it hides the
screen inside a string nobody can block on. The answer is a first-class
``screenID`` column beside ``plateID``, and the ``prc`` grammar untouched.

The two tests that matter most here are the mirror image of each other:

* :func:`test_two_screens_owning_the_same_plate_names_is_not_a_collision` --
  once the screen is in the frame, ``plate1`` in both screens is NORMAL. A
  merge that still calls it a clash has made the feature unusable.
* :func:`test_the_same_plate_twice_inside_one_screen_is_still_refused` -- the
  real error did not go away. It became "duplicate identity WITHIN a screen",
  and it must still fire, because pooling two ``plate1``s of ONE experiment is
  exactly as silent and exactly as wrong as it ever was.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import schema
from spacr.multi_database import (
    SOURCE_COLUMN, MergeRefused, describe_merge, read_merged,
)


def _screen(tmp_path, name, *, plates=("plate1", "plate2", "plate3", "plate4"),
            rows_per_plate=3, extra=None, screen_column=None, table="cell"):
    """One screen's measurement database: four plates, a few wells each."""
    frames = []
    for plate in plates:
        frame = pd.DataFrame({
            "plateID": [plate] * rows_per_plate,
            "rowID": [f"r{i + 1}" for i in range(rows_per_plate)],
            "columnID": ["c1"] * rows_per_plate,
            "cell_area": range(rows_per_plate),
        })
        if extra:
            frame[extra] = 1.0
        if screen_column is not None:
            frame[schema.SCREEN_KEY] = screen_column
        frames.append(frame)
    path = tmp_path / f"{name}.db"
    with sqlite3.connect(str(path)) as db:
        pd.concat(frames, ignore_index=True).to_sql(table, db, index=False)
    return str(path)


@pytest.fixture()
def two_screens(tmp_path):
    """The maintainer's case: two screens, each with plate1..plate4."""
    return [_screen(tmp_path, "screen_a"),
            _screen(tmp_path, "screen_b", extra="cell_perimeter")]


# --------------------------------------------------------------------------- #
#  Part 1 -- screenID is a column, and prc is untouched
# --------------------------------------------------------------------------- #

def test_the_prc_grammar_did_not_grow_a_component():
    """Growing `prc` would re-key every measurement ever written.

    The key columns and the composed keys are exactly what they were; the
    screen rides ALONGSIDE them. If this test ever fails, every database on
    disk has to be rewritten before it can be opened again.
    """
    assert schema.FIELD_KEY_COLUMNS == (
        "plateID", "rowID", "columnID", "fieldID")
    assert schema.WELL_KEY_COLUMNS == ("plateID", "rowID", "columnID")
    assert schema.SCREEN_KEY not in schema.FIELD_KEY_COLUMNS
    assert schema.SCREEN_KEY not in schema.WELL_KEY_COLUMNS

    assert schema.compose_prc("plate1", "r1", "c1") == "plate1_r1_c1"
    assert schema.compose_prcf("plate1", "r1", "c1", "f2") == "plate1_r1_c1_f2"
    assert schema.parse_prcf("plate1_r1_c1_f2").plateID == "plate1"


def test_the_screened_identity_is_the_screen_plus_the_old_key():
    """"the identity becomes (screenID, plateID, rowID, columnID, ...)" --
    as a SEPARATE tuple, so a caller that wants the old one still has it."""
    assert schema.SCREEN_KEY == "screenID"
    assert schema.SCREENED_WELL_KEY_COLUMNS == (
        "screenID", "plateID", "rowID", "columnID")
    assert schema.SCREENED_FIELD_KEY_COLUMNS == (
        "screenID", "plateID", "rowID", "columnID", "fieldID")


def test_a_row_with_no_screen_is_a_single_screen_project():
    """Default it, do not demand it, or every existing project stops opening."""
    assert schema.screen_id(None) == schema.DEFAULT_SCREEN
    assert schema.screen_id("") == schema.DEFAULT_SCREEN
    assert schema.screen_id("   ") == schema.DEFAULT_SCREEN
    assert schema.screen_id("tsg101") == "tsg101"
    assert schema.screen_id(" tsg101 ") == "tsg101"
    # Idempotent: running a defaulted frame through again does not rename it.
    assert schema.screen_id(schema.screen_id(None)) == schema.DEFAULT_SCREEN


def test_every_spelling_of_the_screen_column_becomes_one_column():
    """A user's metadata CSV says `screen`; spaCR says `screenID`. Two
    spellings of one dimension is two dimensions, and neither one blocks."""
    for spelling in ("screen", "screenID", "screenid", "ScreenID",
                     "screen_id", "screen_name"):
        assert schema.canonical_column_name(spelling) == schema.SCREEN_KEY


def test_adding_the_screen_column_leaves_an_existing_one_alone():
    """A frame that already knows its screen is not relabelled behind the
    user's back -- that would silently move rows between experiments."""
    frame = pd.DataFrame({"plateID": ["plate1", "plate1"],
                          schema.SCREEN_KEY: ["kd", "kd"]})
    same = schema.add_screen_column(frame)
    assert list(same[schema.SCREEN_KEY]) == ["kd", "kd"]

    bare = schema.add_screen_column(pd.DataFrame({"plateID": ["plate1"]}))
    assert list(bare[schema.SCREEN_KEY]) == [schema.DEFAULT_SCREEN]

    named = schema.add_screen_column(
        pd.DataFrame({"plateID": ["plate1"]}), screen="ko")
    assert list(named[schema.SCREEN_KEY]) == ["ko"]


def test_a_blank_screen_value_inside_a_frame_is_defaulted_not_left_empty():
    """An empty screen id is not an identity. Left blank it groups with every
    other blank row, which is the pooling this whole module refuses."""
    frame = pd.DataFrame({"plateID": ["plate1"] * 3,
                          schema.SCREEN_KEY: ["kd", None, ""]})
    filled = schema.add_screen_column(frame)
    assert list(filled[schema.SCREEN_KEY]) == [
        "kd", schema.DEFAULT_SCREEN, schema.DEFAULT_SCREEN]


# --------------------------------------------------------------------------- #
#  Part 2 -- the screen-aware merge
# --------------------------------------------------------------------------- #

def test_two_screens_owning_the_same_plate_names_is_not_a_collision(
        two_screens):
    """THE ACCEPTANCE TEST. Two screens, each with plate1..plate4, one frame,
    no collision reported, screenID telling them apart."""
    plan = describe_merge(two_screens, "cell", screens=["kd", "ko"])

    assert not plan.has_collisions, plan.colliding_plates
    # And it is not silence: the plan SAYS the plates are shared, because a
    # user who did not mean to give two screens the same plate names still
    # needs to see it.
    assert set(plan.shared_plates_across_screens) == {
        "plate1", "plate2", "plate3", "plate4"}

    merged = read_merged(two_screens, "cell", screens=["kd", "ko"])

    assert sorted(merged[schema.SCREEN_KEY].unique()) == ["kd", "ko"]
    assert sorted(merged["plateID"].unique()) == [
        "plate1", "plate2", "plate3", "plate4"]
    assert len(merged) == 24


def test_the_plate_id_is_not_rewritten_by_the_screen_aware_path(two_screens):
    """Qualification hides the screen inside the plate id. The screen-aware
    path must leave `plate1` spelled `plate1` -- nothing on disk moves."""
    merged = read_merged(two_screens, "cell", screens=["kd", "ko"])
    assert set(merged["plateID"]) == {"plate1", "plate2", "plate3", "plate4"}


def test_the_screen_is_a_dimension_you_can_block_on(two_screens):
    """Not a string to be parsed back apart: you can group on it, colour by
    it, facet on it and put it in a formula as it stands."""
    merged = read_merged(two_screens, "cell", screens=["kd", "ko"])

    per_screen = merged.groupby(schema.SCREEN_KEY).size().to_dict()
    assert per_screen == {"kd": 12, "ko": 12}
    # A crosstab of screen against plate is the shape a blocking factor has:
    # every plate is present in every screen, with no name mangling.
    table = pd.crosstab(merged[schema.SCREEN_KEY], merged["plateID"])
    assert table.shape == (2, 4)
    assert (table.to_numpy() == 3).all()


def test_a_per_screen_row_count_survives_the_merge(two_screens):
    """The anti-pooling test of instruction 109, now per SCREEN.

    Rows in must equal rows out for each screen. Every other failure in this
    file is loud; pooling two screens is silent, and every per-well number
    computed afterwards describes an experiment that never happened.
    """
    before = {}
    for label, path in zip(("kd", "ko"), two_screens):
        with sqlite3.connect(path) as db:
            before[label] = int(
                db.execute("SELECT COUNT(*) FROM cell").fetchone()[0])

    merged = read_merged(two_screens, "cell", screens=["kd", "ko"])
    after = merged.groupby(schema.SCREEN_KEY).size().to_dict()

    assert after == before
    assert len(merged) == sum(before.values())


def test_the_same_plate_twice_inside_one_screen_is_still_refused(tmp_path):
    """The real error did not go away, it got sharper.

    Two databases that are BOTH the kd screen and both contain plate1 are two
    halves of one experiment claiming one identity. Merging them computes
    every per-well number over two plates at once.
    """
    paths = [_screen(tmp_path, "part_one", plates=("plate1",)),
             _screen(tmp_path, "part_two", plates=("plate1",))]

    plan = describe_merge(paths, "cell", screens=["kd", "kd"])
    assert plan.has_collisions
    assert "plate1" in plan.colliding_plates

    with pytest.raises(MergeRefused) as caught:
        read_merged(paths, "cell", screens=["kd", "kd"], plan=plan)
    message = str(caught.value)
    assert "plate1" in message
    assert "kd" in message


def test_no_screens_at_all_is_still_refused_exactly_as_before(tmp_path):
    """A single-screen project has not changed. Two databases sharing plate1
    with no screen given are the collision instruction 109 refuses."""
    paths = [_screen(tmp_path, "runA", plates=("plate1",)),
             _screen(tmp_path, "runB", plates=("plate1",))]

    assert describe_merge(paths, "cell").has_collisions
    with pytest.raises(MergeRefused):
        read_merged(paths, "cell")


def test_a_single_screen_project_opens_with_no_screen_id_anywhere(tmp_path):
    """"A single-screen project opens and runs exactly as before, with no
    screenID anywhere in its settings." No screens= argument, no screenID
    column in the database, and the merge still works."""
    paths = [_screen(tmp_path, "solo", plates=("plate1", "plate2"))]

    plan = describe_merge(paths, "cell")
    assert not plan.has_collisions
    assert not plan.shared_plates_across_screens

    merged = read_merged(paths, "cell")
    assert len(merged) == 6
    # The dimension exists so downstream code never branches on its absence...
    assert schema.SCREEN_KEY in merged.columns
    # ...but it holds ONE value, so nothing about the analysis changes.
    assert merged[schema.SCREEN_KEY].nunique() == 1
    assert merged[schema.SCREEN_KEY].iloc[0] == schema.DEFAULT_SCREEN


def test_a_screen_id_already_in_the_database_is_believed(tmp_path):
    """A database that already carries screenID -- because it was exported
    from an earlier merge -- keeps its own labels rather than being
    overwritten with a guess."""
    paths = [_screen(tmp_path, "stored_a", plates=("plate1",),
                     screen_column="kd"),
             _screen(tmp_path, "stored_b", plates=("plate1",),
                     screen_column="ko")]

    plan = describe_merge(paths, "cell")
    assert not plan.has_collisions, plan.colliding_plates

    merged = read_merged(paths, "cell")
    assert sorted(merged[schema.SCREEN_KEY].unique()) == ["kd", "ko"]


def test_a_stored_screen_id_that_repeats_is_a_collision_again(tmp_path):
    """Two databases that both say they are the kd screen and both hold
    plate1 collide on the stored labels, with nothing passed in."""
    paths = [_screen(tmp_path, "stored_a", plates=("plate1",),
                     screen_column="kd"),
             _screen(tmp_path, "stored_b", plates=("plate1",),
                     screen_column="kd")]

    assert describe_merge(paths, "cell").has_collisions
    with pytest.raises(MergeRefused):
        read_merged(paths, "cell")


def test_an_explicit_screen_label_overrides_a_stored_one(tmp_path):
    """The caller is looking at the files. When they say which screen a
    database is, that is the answer -- and the stored column is replaced, not
    left to contradict it."""
    path = _screen(tmp_path, "mislabelled", plates=("plate1",),
                   screen_column="wrong")
    merged = read_merged([path], "cell", screens=["right"])
    assert merged[schema.SCREEN_KEY].unique().tolist() == ["right"]


def test_one_screen_label_per_database_is_required_when_given(two_screens):
    """A screens= list that does not line up with the paths would silently
    label the wrong database."""
    with pytest.raises(MergeRefused):
        describe_merge(two_screens, "cell", screens=["only_one"])
    with pytest.raises(MergeRefused):
        read_merged(two_screens, "cell", screens=["a", "b", "c"])


def test_screens_can_be_given_as_a_mapping_from_path(two_screens):
    """The GUI holds a row per file; a dict is the shape it has."""
    merged = read_merged(
        two_screens, "cell",
        screens={two_screens[0]: "kd", two_screens[1]: "ko"})
    assert sorted(merged[schema.SCREEN_KEY].unique()) == ["kd", "ko"]


def test_qualifying_is_still_available_for_callers_who_want_it(tmp_path):
    """"keep qualify for callers who genuinely want it". Unchanged."""
    paths = [_screen(tmp_path, "runA", plates=("plate1",)),
             _screen(tmp_path, "runB", plates=("plate1",))]
    merged = read_merged(paths, "cell", on_collision="qualify")
    plates = sorted(merged["plateID"].unique())
    assert len(plates) == 2, plates
    assert all("plate1" in plate for plate in plates)


def test_provenance_survives_the_screen(two_screens):
    """screenID is the experiment; source_database is the file. They answer
    different questions and both have to be there."""
    merged = read_merged(two_screens, "cell", screens=["kd", "ko"])
    assert SOURCE_COLUMN in merged.columns
    assert schema.SCREEN_KEY in merged.columns
    assert sorted(merged[SOURCE_COLUMN].unique()) == ["screen_a", "screen_b"]


# --------------------------------------------------------------------------- #
#  "the user must be told which measurements were dropped"
# --------------------------------------------------------------------------- #

def test_the_dropped_measurements_are_named_before_the_merge(two_screens):
    """That set IS the analysis they are about to run."""
    plan = describe_merge(two_screens, "cell", screens=["kd", "ko"])
    assert plan.dropped_columns == ("cell_perimeter",)
    assert "cell_perimeter" in plan.describe()


def test_common_is_the_default_for_a_wide_measurement_table(two_screens):
    """The measurement tables are wide and differ between versions, so the
    safe answer is the intersection -- and it is what you get without asking."""
    merged = read_merged(two_screens, "cell", screens=["kd", "ko"])
    assert "cell_perimeter" not in merged.columns


def test_the_merge_reports_the_dropped_measurements_to_the_caller(two_screens):
    """Not only in the plan: the frame that comes back says what it cost, and
    a caller can be handed the list directly."""
    told = []
    merged = read_merged(two_screens, "cell", screens=["kd", "ko"],
                         report=told.append)

    assert merged.attrs["dropped_columns"] == ("cell_perimeter",)
    assert any("cell_perimeter" in message for message in told), told


def test_union_drops_nothing_and_says_so(two_screens):
    """The other half of the choice: keep everything, and report an empty
    dropped set rather than no dropped set, so a caller never branches."""
    told = []
    merged = read_merged(two_screens, "cell", screens=["kd", "ko"],
                         columns="union", report=told.append)
    assert "cell_perimeter" in merged.columns
    assert merged.attrs["dropped_columns"] == ()
    assert not told
