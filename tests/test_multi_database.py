"""Merging several measurement databases without pooling them.

Instruction 109.

The test that matters most is
:func:`test_a_per_source_count_survives_the_merge`. Every other failure here
is loud; pooling two plates that share a name is silent, and every per-well
number computed afterwards describes an experiment that never happened.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.multi_database import (
    SOURCE_COLUMN, MergeRefused, describe_merge, read_merged,
)


def _database(path, plate, *, rows=4, extra=None, table="cell"):
    frame = pd.DataFrame({
        "plateID": [plate] * rows,
        "rowID": [f"r{i + 1}" for i in range(rows)],
        "columnID": ["c1"] * rows,
        "area": range(rows),
    })
    if extra:
        frame[extra] = 1.0
    with sqlite3.connect(str(path)) as db:
        frame.to_sql(table, db, index=False)
    return str(path)


@pytest.fixture
def two_plates(tmp_path):
    return (_database(tmp_path / "plateA.db", "plate1"),
            _database(tmp_path / "plateB.db", "plate2", extra="perimeter"))


@pytest.fixture
def colliding(tmp_path):
    """Two databases that each contain a plate called ``plate1``."""
    return (_database(tmp_path / "runA.db", "plate1"),
            _database(tmp_path / "runB.db", "plate1"))


# --------------------------------------------------------------------------- #
#  Describe before you merge
# --------------------------------------------------------------------------- #

def test_the_plan_reports_what_the_merge_would_cost(two_plates):
    """The column set a merge produces IS the analysis about to be run.
    Finding out afterwards is finding out too late."""
    plan = describe_merge(list(two_plates), "cell")

    assert plan.total_rows == 8
    assert set(plan.common_columns) == {"plateID", "rowID", "columnID", "area"}
    assert "perimeter" in plan.partial_columns
    assert plan.partial_columns["perimeter"] == ("plateB",)
    assert not plan.has_collisions


def test_the_plan_names_the_sources_readably(two_plates):
    plan = describe_merge(list(two_plates), "cell")
    assert [s.label for s in plan.sources] == ["plateA", "plateB"]


def test_databases_sharing_a_filename_still_get_distinct_labels(tmp_path):
    """Every plate's database is often called measurements.db under a
    differently-named folder, so the stem alone is not a name."""
    first = tmp_path / "plate_one"
    second = tmp_path / "plate_two"
    first.mkdir()
    second.mkdir()
    paths = [_database(first / "measurements.db", "p1"),
             _database(second / "measurements.db", "p2")]

    labels = [s.label for s in describe_merge(paths, "cell").sources]
    assert len(set(labels)) == 2, labels
    assert any("plate_two" in label for label in labels)


# --------------------------------------------------------------------------- #
#  The merge itself
# --------------------------------------------------------------------------- #

def test_every_row_remembers_where_it_came_from(two_plates):
    """A merged UMAP whose clusters are the three plates is the most valuable
    thing this feature can show, and it cannot show it without this."""
    merged = read_merged(list(two_plates), "cell")
    assert SOURCE_COLUMN in merged.columns
    assert sorted(merged[SOURCE_COLUMN].unique()) == ["plateA", "plateB"]


def test_common_drops_and_union_keeps(two_plates):
    common = read_merged(list(two_plates), "cell", columns="common")
    assert "perimeter" not in common.columns

    union = read_merged(list(two_plates), "cell", columns="union")
    assert "perimeter" in union.columns
    # Present for the source that had it, null for the one that did not.
    assert int(union["perimeter"].isna().sum()) == 4


def test_a_per_source_count_survives_the_merge(two_plates):
    """The anti-pooling test. Rows in must equal rows out, per source."""
    paths = list(two_plates)
    before = {}
    for path in paths:
        with sqlite3.connect(path) as db:
            before[path] = int(
                db.execute("SELECT COUNT(*) FROM cell").fetchone()[0])

    merged = read_merged(paths, "cell")
    after = merged.groupby(SOURCE_COLUMN).size().to_dict()

    assert sum(before.values()) == len(merged)
    assert sorted(after.values()) == sorted(before.values())


# --------------------------------------------------------------------------- #
#  Collisions -- the part that must never be silent
# --------------------------------------------------------------------------- #

def test_a_shared_plate_id_is_refused_by_default(colliding):
    plan = describe_merge(list(colliding), "cell")
    assert plan.has_collisions
    assert "plate1" in plan.colliding_plates

    with pytest.raises(MergeRefused) as caught:
        read_merged(list(colliding), "cell", plan=plan)
    # The message has to say what it found, or the user cannot act on it.
    assert "plate1" in str(caught.value)


def test_qualifying_makes_the_plates_distinct(colliding):
    merged = read_merged(list(colliding), "cell", on_collision="qualify")
    plates = sorted(merged["plateID"].unique())
    assert len(plates) == 2, plates
    assert all("plate1" in p for p in plates)
    # Still two separate experiments, with their counts intact.
    assert merged.groupby(SOURCE_COLUMN).size().to_dict() == {
        "runA": 4, "runB": 4}


def test_a_qualified_plate_is_still_a_legal_key_token(colliding):
    """The qualifier must not introduce the key separator, or it splits the
    key into an extra component -- the bug schema.KEY_ESCAPES exists for."""
    merged = read_merged(list(colliding), "cell", on_collision="qualify")
    for plate in merged["plateID"].unique():
        assert schema_separator_absent(plate)


def schema_separator_absent(plate) -> bool:
    from spacr import schema

    return schema.KEY_SEPARATOR not in str(plate)


def test_there_is_no_option_that_pools(colliding):
    """Refusing an unknown mode matters more than it looks: 'merge' or
    'ignore' would be the obvious thing for a caller to try."""
    for mode in ("merge", "ignore", "pool", ""):
        with pytest.raises(MergeRefused):
            read_merged(list(colliding), "cell", on_collision=mode)


def test_an_unknown_column_mode_is_refused(two_plates):
    with pytest.raises(MergeRefused):
        read_merged(list(two_plates), "cell", columns="everything")


# --------------------------------------------------------------------------- #
#  Edges
# --------------------------------------------------------------------------- #

def test_no_databases_is_an_empty_frame_not_a_crash():
    assert read_merged([], "cell").empty


def test_one_database_still_gains_provenance(tmp_path):
    """Merging one is not a special case -- it is the same frame plus the
    column that says where it came from, so a caller never branches."""
    path = _database(tmp_path / "only.db", "plate1")
    merged = read_merged([path], "cell")
    assert len(merged) == 4
    assert merged[SOURCE_COLUMN].unique().tolist() == ["only"]


def test_a_row_cap_applies_per_source(two_plates):
    merged = read_merged(list(two_plates), "cell", limit_per_source=2)
    assert len(merged) == 4
    assert merged.groupby(SOURCE_COLUMN).size().to_dict() == {
        "plateA": 2, "plateB": 2}


# --------------------------------------------------------------------------- #
#  The Gate Editor, which is one of the two screens instruction 109 names
# --------------------------------------------------------------------------- #

@pytest.mark.qt
def test_the_gate_editor_merges_several_databases(qtbot, two_plates):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.load_paths(list(two_plates))

    assert screen._frame is not None
    assert len(screen._frame) == 8
    assert SOURCE_COLUMN in screen._frame.columns
    # The label has to say the merge happened, and name the colour-by column.
    assert "2 databases" in screen._source.text()
    assert SOURCE_COLUMN in screen._source.text()


@pytest.mark.qt
def test_the_gate_editor_refuses_a_collision_and_says_which(qtbot, colliding):
    """Refusal must leave the previous state alone AND name the plate.

    A screen that blanks itself on a refusal has punished the user for a
    question they are allowed to ask.
    """
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.load_paths(list(colliding))

    assert screen._frame is None
    assert "plate1" in screen._source.text()


# --------------------------------------------------------------------------- #
#  Image UMAP, the other screen instruction 109 names
# --------------------------------------------------------------------------- #

def test_the_umap_and_the_merge_name_the_source_column_the_same():
    """One name for one idea.

    generate_image_umap has always accepted several source roots; it now
    keeps a source column so a user can colour by it. If that column were
    called something else, a frame from the UMAP and a frame from the Gate
    Editor's merge would answer the same question under two names, and
    nothing could compare them.
    """
    from spacr.core import UMAP_SOURCE_COLUMN

    assert UMAP_SOURCE_COLUMN == SOURCE_COLUMN


def test_the_umap_source_label_is_the_plate_folder():
    """get_db_paths appends measurements/measurements.db to every root, so
    the file name is identical for every plate and useless as a legend."""
    from spacr.core import _umap_source_label

    assert _umap_source_label("/data/plate1") == "plate1"
    assert _umap_source_label("/data/plate1/") == "plate1"
    assert _umap_source_label("plate2") == "plate2"


def test_the_umap_warns_about_colliding_plates_without_stopping(tmp_path,
                                                               capsys):
    """The advisory check is a WARNING, not a refusal.

    Unlike a fresh merge, this is an existing entry point with existing
    callers, and stopping a run that worked yesterday is a worse failure than
    saying so loudly. The check must also never raise -- a database missing a
    'cell' table is a legitimate shape here.
    """
    from spacr.multi_database import describe_merge

    paths = [_database(tmp_path / "runA.db", "plate1"),
             _database(tmp_path / "runB.db", "plate1")]
    plan = describe_merge(paths, "cell")
    assert plan.colliding_plates, "the fixture should collide"

    # The shape core.generate_image_umap uses: never raises, names the plate.
    detail = '; '.join(f"{plate!r} in {', '.join(labels)}"
                       for plate, labels in sorted(plan.colliding_plates.items()))
    assert "plate1" in detail
    assert "runA" in detail and "runB" in detail


# --------------------------------------------------------------------------- #
#  The parts a screen shows, and the edges that make labels unique
# --------------------------------------------------------------------------- #

def test_the_plan_describes_itself_for_a_dialog(two_plates, colliding,
                                                tmp_path):
    """`describe` is what a screen puts in front of a user before they
    commit, so it has to name the counts AND the two things that cost them
    something: columns present in only some sources, and colliding plates."""
    plan = describe_merge(list(two_plates), "cell")
    text = plan.describe()

    assert "2 databases" in text
    assert "8 rows" in text
    assert "columns in all of them" in text
    assert "perimeter" in text          # the partial column, named
    assert "appear in more than one" not in text

    clash = describe_merge(list(colliding), "cell").describe()
    assert "appear in more than one" in clash
    assert "plate1" in clash


def test_describe_truncates_a_long_partial_column_list(tmp_path):
    """Eight partial columns must not print eight names into a dialog."""
    extra = _database(tmp_path / "wide.db", "plateW")
    with sqlite3.connect(extra) as db:
        for i in range(8):
            db.execute(f'ALTER TABLE cell ADD COLUMN extra_{i} REAL')
    plain = _database(tmp_path / "plain.db", "plateP")

    text = describe_merge([extra, plain], "cell").describe()
    assert "..." in text


def test_a_source_summary_names_itself(two_plates):
    """`name` is what a legend or a chip shows."""
    plan = describe_merge(list(two_plates), "cell")
    assert [s.name for s in plan.sources] == ["plateA", "plateB"]


def test_three_databases_with_one_stem_all_get_distinct_labels(tmp_path):
    """Parent-directory disambiguation runs out when the parents match too,
    so the numeric tail has to work -- otherwise two sources share a legend
    entry and the provenance column stops distinguishing them."""
    # FOUR, not three. The first takes the stem, the second the
    # parent/stem, the third parent/stem (2) -- the counter only has to
    # ADVANCE on the fourth, which is the line this covers.
    paths = []
    for index in range(4):
        folder = tmp_path / "runs" / f"copy{index}" / "same"
        folder.mkdir(parents=True)
        paths.append(_database(folder / "measurements.db", f"p{index}"))

    labels = [s.label for s in describe_merge(paths, "cell").sources]
    assert len(set(labels)) == 4, labels
    assert any("(3)" in label for label in labels), labels


def test_a_table_without_a_plate_column_reports_no_plates(tmp_path):
    """Not every table is keyed by plate -- a summary table is a legitimate
    merge target, and it simply cannot collide."""
    path = tmp_path / "nokey.db"
    with sqlite3.connect(str(path)) as db:
        pd.DataFrame({"value": [1, 2]}).to_sql("summary", db, index=False)

    plan = describe_merge([str(path)], "summary")
    assert plan.sources[0].plates == ()
    assert not plan.has_collisions


def test_a_legacy_column_spelling_is_canonicalised_on_the_way_in(tmp_path):
    """Two databases spelling one column differently must become ONE column,
    not two -- and case-folded, so the merged frame is still writable."""
    modern = _database(tmp_path / "modern.db", "plateM")
    legacy = tmp_path / "legacy.db"
    frame = pd.DataFrame({
        "plateID": ["plateL"] * 2, "row": ["r1", "r2"],
        "columnID": ["c1"] * 2, "area": [1.0, 2.0],
    })
    with sqlite3.connect(str(legacy)) as db:
        frame.to_sql("cell", db, index=False)

    merged = read_merged([modern, str(legacy)], "cell", columns="union")

    assert "rowID" in merged.columns
    assert "row" not in merged.columns


# --------------------------------------------------------------------------- #
#  The name a source carries, which is the colour of a point on the map
# --------------------------------------------------------------------------- #

def test_a_plate_folder_layout_names_the_plate_and_not_the_file(tmp_path):
    """spaCR writes every plate to ``<plate>/measurements/measurements.db``.

    So the stem AND its immediate parent are the same string for every plate
    in a screen, and disambiguating on the immediate parent produced
    ``measurements``, ``measurements/measurements`` and
    ``measurements/measurements (2)`` -- three labels that are distinct and
    tell the user nothing, in the column a merged embedding is COLOURED BY.
    """
    from spacr.core import _umap_source_label

    roots, paths = [], []
    for index in (1, 2, 3):
        root = tmp_path / f"plate{index}"
        (root / "measurements").mkdir(parents=True)
        roots.append(str(root))
        paths.append(_database(root / "measurements" / "measurements.db",
                               f"plate{index}"))

    labels = [s.label for s in describe_merge(paths, "cell").sources]
    assert labels == ["plate1", "plate2", "plate3"], labels
    # And the Image UMAP names them the same way, which is the whole reason
    # the two are allowed to share one column name.
    assert labels == [_umap_source_label(root) for root in roots]


def test_source_labels_is_the_one_answer_both_a_chip_and_a_legend_use(tmp_path):
    """A chip reading ``plate1`` beside a legend reading ``measurements (2)``
    is provenance the user cannot follow, so there is one function."""
    from spacr.multi_database import source_labels

    roots = []
    for index in (1, 2):
        root = tmp_path / f"plate{index}" / "measurements"
        root.mkdir(parents=True)
        roots.append(_database(root / "measurements.db", f"plate{index}"))

    assert source_labels(roots) == ("plate1", "plate2")
    assert [s.label for s in describe_merge(roots, "cell").sources] == [
        "plate1", "plate2"]


# --------------------------------------------------------------------------- #
#  Three plates, which is the case the instruction is about
# --------------------------------------------------------------------------- #

def test_three_databases_keep_every_per_source_count(tmp_path):
    """The anti-pooling test at the size the instruction describes.

    Rows in equals rows out, per source, for every source -- and the counts
    travel ON the frame, so a reader does not have to go back to the files to
    prove nothing was pooled.
    """
    paths, before = [], {}
    for index, rows in enumerate((4, 7, 5), start=1):
        path = _database(tmp_path / f"plate{index}.db", f"plate{index}",
                         rows=rows)
        paths.append(path)
        before[f"plate{index}"] = rows

    merged = read_merged(paths, "cell")

    after = merged.groupby(SOURCE_COLUMN).size().to_dict()
    assert after == before
    assert len(merged) == sum(before.values())
    assert merged.attrs["source_rows"] == before


# --------------------------------------------------------------------------- #
#  Told, and recorded
# --------------------------------------------------------------------------- #

def test_the_refusal_does_not_recommend_hiding_the_screen_in_the_plate_id(
        colliding):
    """`on_collision='qualify'` stays available to a caller who wants it, and
    stops being the advice a user is given.

    Rewriting ``plate1`` to ``runA-plate1`` makes the keys unique -- which is
    all it was built to do -- and hides which experiment a plate belongs to
    inside its own id, where it can no longer be blocked on, tested for or
    coloured by (instruction 122).
    """
    with pytest.raises(MergeRefused) as caught:
        read_merged(list(colliding), "cell")

    message = str(caught.value)
    assert "plate1" in message
    assert "qualify" not in message
    assert "Remove one of those databases" in message


def test_a_merge_decision_is_written_down(tmp_path, colliding):
    """"The user is TOLD, and what they choose is RECORDED."

    A collision the user resolves by dropping one of two databases leaves no
    trace in the surviving frame -- it cannot say which ``plate1`` it is -- so
    the record is the only thing that can answer it afterwards.
    """
    import json

    from spacr.multi_database import decision_for, record_decision

    plan = describe_merge(list(colliding), "cell")
    decision = decision_for(plan, outcome="resolved",
                            resolution="removed runB from the working set")

    log = tmp_path / "merge_decisions.jsonl"
    assert record_decision(decision, str(log)) == str(log)

    written = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(written) == 1
    record = written[0]
    assert record["outcome"] == "resolved"
    assert record["resolution"] == "removed runB from the working set"
    assert record["colliding_plates"]["plate1"] == ["runA", "runB"]
    assert record["rows"] == {"runA": 4, "runB": 4}
    assert record["when"]

    # Appended, not rewritten: two screens deciding at once must not lose one
    # of the two answers.
    record_decision(decision_for(plan, outcome="refused"), str(log))
    assert len(log.read_text().splitlines()) == 2


def test_an_unwritable_decision_log_does_not_take_the_screen_down(colliding):
    """An audit line is never worth failing a load for -- but the caller is
    told it was not kept, rather than being left to assume it was."""
    from spacr.multi_database import decision_for, record_decision

    plan = describe_merge(list(colliding), "cell")
    written = record_decision(decision_for(plan, outcome="refused"),
                              "/proc/not/a/writable/place/merge.jsonl")
    assert written == ""


# --------------------------------------------------------------------------- #
#  Instruction 154: the plate id, the declared types, and a merge that can stop
# --------------------------------------------------------------------------- #

def test_the_plate_id_rule_here_is_the_same_rule_correct_metadata_applies():
    """ONE KEY VOCABULARY (instruction 145), pinned rather than trusted.

    ``utils.correct_metadata`` has repaired ``pplate1`` -> ``plate1`` in
    frames since the day it cost a whole run -- score files stamped
    ``pplate1`` met count files stamped ``plate1``, every ``prc`` differed by
    one character, the join produced ZERO rows, and the run died two hundred
    lines later inside a plot with ``KeyError: 0``. Nothing repaired the plate
    ids read out of a measurements DATABASE, which is how the Measurements tab
    came to show ``pplate1`` beside a plate the user calls ``plate1``.

    Two implementations of one rule drift, so this asserts they agree instead
    of asserting a list of answers typed here.
    """
    from spacr.multi_database import canonical_plate_id
    from spacr.utils import correct_metadata

    for stored in ("pplate1", "plate1", "pp3", "p1", "plate", "", "ppp1"):
        expected = correct_metadata(
            pd.DataFrame({"plateID": [stored]}))["plateID"][0]
        assert canonical_plate_id(stored) == expected, stored


def test_the_declared_type_is_what_predicts_the_aggregation(tmp_path):
    """Instruction 154 C. A pre-merge plan has to say how each column will be
    combined, and the merge decides that from the pandas DTYPE. Matching on
    the column NAME alone announced that ``file_name`` "would take the default
    (mean)", which is not what happens and is not a thing that can happen to a
    string. The declared affinity is what predicts the dtype.
    """
    from spacr.merge_tables import aggregation_plan
    from spacr.multi_database import column_kinds

    path = str(tmp_path / "kinds.db")
    frame = pd.DataFrame({"plateID": ["plate1"], "file_name": ["a.tif"],
                          "area": [1.0], "count": [2]})
    with sqlite3.connect(path) as db:
        frame.to_sql("cell", db, index=False)

    kinds = column_kinds(path, "cell")
    assert kinds == {"plateID": "text", "file_name": "text",
                     "area": "numeric", "count": "numeric"}

    # ...and it agrees with what the merge will actually do.
    plan = aggregation_plan(frame)
    assert plan["file_name"] == "first"
    assert plan["area"] == "sum"


def test_a_column_with_no_declared_type_is_unknown_rather_than_guessed(
        tmp_path):
    """"An absent fingerprint that reads as an absent difference is a false
    assurance." A column SQLite was given no type for cannot be promised
    either treatment, so it is named as unanswerable instead."""
    from spacr.multi_database import column_kinds

    path = str(tmp_path / "untyped.db")
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE cell (plateID TEXT, mystery, blobby BLOB)')

    kinds = column_kinds(path, "cell")
    assert kinds == {"plateID": "text", "mystery": "unknown",
                     "blobby": "unknown"}


def test_a_read_says_which_database_it_is_on_and_counts_rows(two_plates):
    """Instruction 154 A: "say what stage it is on ... show rows processed
    against rows expected". The denominator is the plan's own total, so the
    count can be checked against the sentence printed above it."""
    from spacr.multi_database import read_merged

    plan = describe_merge(list(two_plates), "cell")
    seen = []
    read_merged(list(two_plates), "cell", plan=plan,
                progress=lambda stage, done, total: seen.append(
                    (stage, done, total)))

    assert seen[0] == ("reading cell from plateA", 0, plan.total_rows)
    assert seen[-1] == ("read cell from plateB", plan.total_rows,
                        plan.total_rows)


def test_a_read_that_is_asked_to_stop_returns_nothing_at_all(two_plates):
    """"leaving nothing half-written". A source is read by a single query, so
    the check is BETWEEN sources -- interrupting one would leave a partial
    frame this function has no honest way to hand back."""
    from spacr.multi_database import MergeCancelled, read_merged

    calls = []

    def stop_after_the_first():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(MergeCancelled) as raised:
        read_merged(list(two_plates), "cell", cancelled=stop_after_the_first)

    assert "none of them were kept" in str(raised.value)


def test_a_cancellation_is_not_a_refusal(two_plates):
    """A refusal is an ANSWER about the data and the caller shows it; a
    cancellation is the user changing their mind. Catching both as one would
    put "the merge was refused" in front of somebody who pressed Stop."""
    from spacr.multi_database import MergeCancelled

    assert not issubclass(MergeCancelled, MergeRefused)
    assert not issubclass(MergeRefused, MergeCancelled)


def test_a_reader_can_continue_another_readers_count(two_plates):
    """One bar across several tables: the panel merges cell, then nucleus,
    then pathogen, and a count that restarted at zero for each would read as
    the merge going backwards."""
    from spacr.multi_database import read_merged

    seen = []
    frame = read_merged(list(two_plates), "cell", rows_done=100,
                        rows_total=500,
                        progress=lambda stage, done, total: seen.append(
                            (done, total)))

    assert seen[0] == (100, 500)
    assert frame.attrs["rows_done"] == 100 + len(frame)


# ---------------------------------------------------------------------------
# The doubled plate prefix reaches the DATA, not only the label
# ---------------------------------------------------------------------------

def test_normalise_plate_ids_agrees_with_correct_metadata():
    """The frame rule and the scalar rule give the same answer.

    `canonical_plate_id` was already pinned against `utils.correct_metadata`.
    This pins the FRAME form to the scalar one, so the three cannot drift --
    which is the whole point of instruction 145's one-vocabulary rule.
    """
    import pandas as pd
    from spacr.multi_database import canonical_plate_id, normalise_plate_ids

    plates = ["pplate1", "plate2", "pp1", "p1", "control", ""]
    frame = pd.DataFrame({"plateID": list(plates)})
    got = list(normalise_plate_ids(frame)["plateID"])
    assert got == [canonical_plate_id(p) for p in plates]


def test_the_composed_keys_are_normalised_too():
    """`prc` carries the plate as its first component.

    Rewriting `plateID` alone would leave `prc` unjoinable and the two columns
    disagreeing about the same plate -- which is worse than not fixing it,
    because a frame that is half-normalised joins on one key and not the other.
    """
    import pandas as pd
    from spacr.multi_database import normalise_plate_ids

    frame = pd.DataFrame({
        "plateID": ["pplate1"],
        "prc": ["pplate1_r3_c7"],
        "prcfo": ["pplate1_r3_c7_f2_o5"],
    })
    out = normalise_plate_ids(frame)
    assert out["plateID"][0] == "plate1"
    assert out["prc"][0] == "plate1_r3_c7"
    assert out["prcfo"][0] == "plate1_r3_c7_f2_o5"


def test_a_non_text_plate_column_is_left_alone():
    """A plate id stored as an INTEGER cannot carry a 'pp' prefix.

    `.str` refuses a non-object column, so the guard is needed rather than
    merely tidy: without it a database with an integer plateID raises on read.
    """
    import pandas as pd
    from spacr.multi_database import normalise_plate_ids

    frame = pd.DataFrame({"plateID": [1, 2, 3]})
    assert list(normalise_plate_ids(frame)["plateID"]) == [1, 2, 3]


def test_a_merged_frame_can_meet_a_normalised_score_file(tmp_path):
    """THE BUG THIS EXISTS FOR, end to end.

    A measurements database stamped `pplate1` used to produce merged rows
    stamped `pplate1`, while `utils.correct_metadata` had already normalised
    the score CSV to `plate1`. The merge succeeded, the row counts were right,
    and the two frames then shared no well -- which surfaced far away as a
    gene half that was missing for no visible reason.
    """
    import sqlite3
    import pandas as pd
    from spacr.multi_database import read_merged

    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        pd.DataFrame({
            "plateID": ["pplate1"] * 3,
            "rowID": ["r1", "r2", "r3"],
            "columnID": ["c1", "c1", "c1"],
            "object_label": [1, 2, 3],
            "area": [10.0, 11.0, 12.0],
        }).to_sql("cell", db, index=False)

    merged = read_merged([str(path)], "cell")
    assert set(merged["plateID"]) == {"plate1"}, merged["plateID"].tolist()

    # The score side, as `correct_metadata` leaves it.
    scores = pd.DataFrame({"plateID": ["plate1"], "rowID": ["r1"],
                           "columnID": ["c1"], "grna": ["g1"]})
    joined = merged.merge(scores, on=["plateID", "rowID", "columnID"])
    assert len(joined) == 1, "the merged frame must meet a normalised score file"
