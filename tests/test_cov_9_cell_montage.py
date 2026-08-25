"""What the montage does when the files, the counts or the route fall short.

A montage is a picture of cells labelled with a guide nobody sequenced them
for, so every fallback in this module is a place where a wrong picture could
be drawn with a caption that still reads as if it worked. These tests hold
the line at each of them: an unreadable sweep grid means "no sweep", not a
crash; a count table with no fraction to normalise against keeps the raw
fraction rather than inventing a factor; a route with no mask plane offers
bounding boxes and says so instead of quietly serving one.
"""
from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spacr import cell_montage as cm
from spacr.cell_montage import (
    CROP_SHAPES,
    CropSourceChoice,
    EFFECTS_GRID_FILE,
    MontageError,
    RouteRequirements,
    effects_from_results,
    effects_grid_from_results,
    montage_route_requirements,
    normalised_share,
    resolve_montage_crop_source,
    write_effects_grid,
)

WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")
PER_WELL = 8


# ---------------------------------------------------------------------------
# Screen fixtures
# ---------------------------------------------------------------------------

def _objects(wells=WELLS, per_well=PER_WELL):
    """The per-object frame a montage selects from."""
    rows = []
    for index, well in enumerate(wells):
        row_id, column_id = well.split("_")
        for label in range(1, per_well + 1):
            rows.append({
                "prc": f"plate1_{well}",
                "plateID": "plate1", "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "object_label": label,
                "pred": round(0.05 + 0.9 * (label - 1) / (per_well - 1), 4),
                "area": 100.0 + label + 10 * index,
                "perimeter": 40.0 + 2 * label,
            })
    return pd.DataFrame(rows)


def _counts(fractions=None, cell_count=PER_WELL):
    """A ``regression_data.csv``-shaped count frame."""
    if fractions is None:
        fractions = {
            "r1_c1": {"GRA14_1": 0.25, "GRA14_2": 0.25, "OTHER_1": 0.5},
            "r1_c2": {"GRA14_1": 0.125, "OTHER_1": 0.875},
            "r1_c3": {"GRA14_2": 0.5, "OTHER_1": 0.5},
            "r1_c4": {"OTHER_1": 1.0},
        }
    genes = {"GRA14_1": "GRA14", "GRA14_2": "GRA14", "OTHER_1": "OTHER"}
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id,
                "grna": guide, "gene": genes[guide], "fraction": fraction,
                "cell_count": cell_count, "pred": 0.5,
            })
    return pd.DataFrame(rows)


def _corrupt_csv(path):
    """A file pandas cannot tokenise: a quoted field that is never closed."""
    path.write_text('a,b\n"1,2\n', encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# The persisted sweep grid
# ---------------------------------------------------------------------------

def test_no_path_at_all_means_no_sweep_grid():
    """An empty path is "this run has no folder", not a file to look for.

    The montage is usable without a run folder; asking the filesystem about
    the empty string would raise on the one code path that exists to say the
    grid is simply absent.
    """
    assert effects_grid_from_results("") is None
    assert effects_grid_from_results(None) is None


def test_a_grid_with_no_rows_is_not_a_grid(tmp_path):
    """A header-only grid names measurements but scores no guide.

    Multivariate selection indexes the grid by guide; an empty one would make
    every posterior fall back to the prior while the caption still claimed a
    sweep had been used.
    """
    (tmp_path / EFFECTS_GRID_FILE).write_text("guide,area\n", encoding="utf-8")

    assert effects_grid_from_results(str(tmp_path)) is None


def test_a_grid_that_cannot_be_filed_does_not_fail_the_sweep(tmp_path):
    """A sweep that produced its answer has not failed because filing failed.

    The grid is a convenience for a later session. Raising here would throw
    away a completed sweep over a directory that could not be created.
    """
    blocking_file = tmp_path / "results"
    blocking_file.write_text("not a directory", encoding="utf-8")
    effects = pd.DataFrame({"area": [0.5]}, index=["GRA14_1"])

    assert write_effects_grid(effects, str(blocking_file / "run")) == ""


# ---------------------------------------------------------------------------
# The per-guide effects table
# ---------------------------------------------------------------------------

def test_effects_come_back_empty_when_there_is_no_results_file(tmp_path):
    """A path that names no file yields no effects rather than raising.

    Attribution degrades to the single-score picker without them; the montage
    still draws, which is the behaviour a missing optional input owes.
    """
    assert effects_from_results("") == {}
    assert effects_from_results(str(tmp_path / "absent.csv")) == {}


def test_an_unreadable_results_table_yields_no_effects(tmp_path):
    """A corrupt results CSV means "no effects", not a traceback.

    The caller's own message says what attribution costs without them, and a
    half-parsed table would be worse than none.
    """
    assert effects_from_results(_corrupt_csv(tmp_path / "results.csv")) == {}


def test_a_results_table_with_no_effect_column_yields_no_effects(tmp_path):
    """A table that names guides but no coefficient cannot supply effects.

    Guessing a numeric column would attribute cells on whatever number
    happened to be there -- a p-value, a read count -- under the name of an
    effect size.
    """
    path = tmp_path / "results.csv"
    pd.DataFrame({"guide": ["GRA14_1"], "pvalue": [0.01]}).to_csv(
        path, index=False)

    assert effects_from_results(str(path)) == {}


# ---------------------------------------------------------------------------
# Normalising a guide's share of its well
# ---------------------------------------------------------------------------

def test_a_well_of_unusable_fractions_normalises_against_nothing():
    """Fractions that are not numbers give no total to divide by.

    Inventing one would be arithmetic on no evidence, so the guide keeps its
    own fraction and the factor stays exactly 1.
    """
    share, factor = normalised_share(["not a number", None], 0.25)

    assert share == pytest.approx(0.25)
    assert factor == 1.0


def test_a_guide_whose_own_fraction_is_not_a_number_shows_nothing():
    """A non-finite fraction cannot be multiplied into a count of cells.

    Passing it through would make ``round(n x fraction)`` NaN and the montage
    would either draw nothing or raise deep inside the count rule; zero with
    a factor of 1 is the honest answer.
    """
    assert normalised_share([0.25, 0.75], float("nan")) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Route requirements
# ---------------------------------------------------------------------------

def _merged_source(mask_dims, kind="merged"):
    return SimpleNamespace(kind=kind,
                           spec=SimpleNamespace(channels=(0, 1, 2),
                                                mask_dims=dict(mask_dims)))


def test_an_unknown_crop_shape_is_named_rather_than_silently_unavailable():
    """A shape nobody offers must say it is not a shape at all.

    ``why_not`` writes the sentence a greyed control carries; a blank one
    would leave a disabled button with no explanation next to it.
    """
    requirements = RouteRequirements(route="merged-bbox", shapes=("bbox",))

    message = requirements.why_not("banana")
    assert "'banana'" in message
    assert str(list(CROP_SHAPES)) in message


def test_a_route_that_is_missing_something_says_so_on_its_status_line():
    """The status line has to carry the missing piece, not just the route name.

    A route described only by its name reads as working; the whole point of
    the requirements check is that the tab can show what is absent before
    anything is cut.
    """
    requirements = RouteRequirements(route="merged-mask",
                                     shapes=("object", "bbox"),
                                     missing=("a bounding box",))

    described = requirements.describe()
    assert "MISSING" in described
    assert "a bounding box" in described


def test_the_png_route_needs_the_path_of_every_crop():
    """Exported PNGs are found by path, and an object table without one is stuck.

    Without ``png_path`` there is nothing to open, so it is reported up front
    rather than as tens of thousands of missing files later.
    """
    source = SimpleNamespace(kind="png", spec=None)
    objects = pd.DataFrame({"object_label": [1, 2]})

    requirements = montage_route_requirements(source, objects)

    assert requirements.route == "png"
    assert any("png_path" in item for item in requirements.missing)


def test_cytoplasm_needs_the_cell_plane_and_something_to_subtract():
    """Cytoplasm is derived, so one mask plane is not enough for it.

    With only the cell plane the subtraction has no second operand, and a
    "cytoplasm" crop would in fact be the whole cell under another name.
    """
    only_cell = montage_route_requirements(
        _merged_source({"cell": 4}), pd.DataFrame({"object_label": [1]}),
        object_type="cytoplasm", channels=[0, 1, 2])

    assert only_cell.route == "merged-bbox"
    assert "cytoplasm is derived" in only_cell.detail

    both = montage_route_requirements(
        _merged_source({"cell": 4, "nucleus": 5}),
        pd.DataFrame({"object_label": [1]}),
        object_type="cytoplasm", channels=[0, 1, 2])

    assert both.route == "merged-mask"
    assert both.offers("object")


def test_a_merged_route_with_no_mask_plane_can_only_cut_boxes():
    """No mask plane means no object outline, whatever the object table holds.

    Offering an object-shaped crop here would hand back a bounding box drawn
    as though it followed the cell's edge.
    """
    requirements = montage_route_requirements(
        _merged_source({"nucleus": 5}), pd.DataFrame({"object_label": [1]}),
        object_type="cell", channels=[0, 1, 2])

    assert requirements.route == "merged-bbox"
    assert requirements.shapes == ("bbox",)
    assert "no cell mask plane" in requirements.detail


def test_a_boxless_table_on_a_maskless_route_has_nothing_to_cut():
    """Without a mask plane AND without a box there is no crop at all.

    This is the one merged case that cannot draw anything, so it has to
    arrive as a missing requirement rather than as an empty montage.
    """
    requirements = montage_route_requirements(
        _merged_source({}), pd.DataFrame({"some_measurement": [1.0]}),
        object_type="cell", channels=[0, 1, 2])

    assert requirements.route == "merged-bbox"
    assert any("bounding box" in item for item in requirements.missing)
    assert not requirements.satisfied


def test_a_choice_with_no_requirements_adds_no_caption_lines():
    """A choice made before the route was checked has nothing to declare.

    Emitting a "crop route:" line with nothing after it would put an empty
    promise in the caption.
    """
    choice = CropSourceChoice(source=None, kind="", reason="not checked",
                              available=False, requirements=None)

    assert choice.requirement_notes() == ()


# ---------------------------------------------------------------------------
# One well's guide fractions
# ---------------------------------------------------------------------------

def test_a_count_table_with_no_guide_column_offers_no_fractions():
    """Without a guide column there is nothing in the well to attribute to.

    An empty mapping sends every picker back to rank, which is the honest
    fallback; guessing a column would attribute cells to whatever identifier
    happened to be first.
    """
    counts = _counts().drop(columns=["grna"])

    assert cm._well_guide_fractions(counts, "plate1_r1_c1", ["prc"],
                                    "grna", "fraction") == {}


def test_an_unusable_count_table_offers_no_fractions():
    """A count table that cannot even be labelled by well yields nothing.

    The attribution is optional and the montage still has to draw, so the
    failure comes back as "no fractions here" rather than as an exception in
    the middle of the well loop.
    """
    assert cm._well_guide_fractions("not a table", "plate1_r1_c1", ["prc"],
                                    "grna", "fraction") == {}


# ---------------------------------------------------------------------------
# Sudoku, and every way a screen cannot support it
# ---------------------------------------------------------------------------

def _sudoku_frame(**over):
    frame = pd.DataFrame({
        "prc": ["plate1_r1_c1"] * 4 + ["plate1_r1_c2"] * 4,
        "pred": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "area": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
    })
    for key, value in over.items():
        frame[key] = value
    return frame


def test_sudoku_says_so_when_no_cell_carries_a_score():
    """Sudoku propagates a score across wells, so with none there is nothing.

    Returning an empty assignment instead of ``None`` would leave the montage
    silently unringed with a caption that named sudoku as the picker.
    """
    notes = []
    frame = _sudoku_frame()
    frame["pred"] = "unscored"

    result = cm._sudoku_calls(frame, _counts(), ["prc"], "grna", "fraction",
                              "pred", "GRA14_1", notes)

    assert result is None
    assert any("no cell carries a classification score" in n for n in notes)


def test_sudoku_says_so_when_the_objects_carry_no_measurement():
    """The graph is built on the measurements, never on the score itself.

    The anchors are chosen BY the score, so a graph built on it would place
    every high-scoring cell beside every guide's anchors and affirm all of
    them. With no other numeric column there is no graph to build.
    """
    notes = []
    frame = _sudoku_frame().drop(columns=["area"])

    result = cm._sudoku_calls(frame, _counts(), ["prc"], "grna", "fraction",
                              "pred", "GRA14_1", notes)

    assert result is None
    assert any("no numeric measurement" in n for n in notes)


def test_sudoku_says_so_when_no_well_constrains_the_propagation():
    """Guide fractions are the constraint; without them propagation is free.

    An unconstrained propagation would assign every cell to whichever guide
    the graph happened to favour, which is a picture of the clustering rather
    than of the screen.
    """
    notes = []
    counts = _counts().drop(columns=["fraction"])

    result = cm._sudoku_calls(_sudoku_frame(), counts, ["prc"], "grna",
                              "fraction", "pred", "GRA14_1", notes)

    assert result is None
    assert any("no well has a guide fraction" in n for n in notes)


def test_sudoku_names_the_guides_it_looked_for_when_it_finds_none():
    """A guide in none of these wells cannot be propagated to any cell.

    The note repeats what was searched for, because the usual cause is a
    gene-level name being compared against guide-level fractions -- which
    matches nothing and would otherwise look like an empty screen.
    """
    notes = []

    result = cm._sudoku_calls(_sudoku_frame(), _counts(), ["prc"], "grna",
                              "fraction", "pred", "GRA14", notes)

    assert result is None
    assert any("looked for GRA14" in n for n in notes)


def test_a_screen_sudoku_cannot_run_on_falls_back_to_rank_and_says_so(
        monkeypatch):
    """A picker that cannot run says so and the montage still draws.

    The alternative is an empty montage with sudoku named in the caption,
    which reads as "no cell matched this guide" rather than "the picker never
    ran".
    """
    import spacr.sudoku as sudoku_module

    def _explode(*args, **kwargs):
        raise RuntimeError("the graph would not build")

    monkeypatch.setattr(sudoku_module, "sudoku", _explode)

    plan = cm.select_montage(_objects(), _counts(), "GRA14_1", 0.2,
                             level="grna", picking="sudoku")

    assert plan.n_objects > 0
    joined = " ".join(list(plan.notes) + [w.note for w in plan.wells])
    assert "sudoku could not run" in joined
    assert "fell back to rank" in joined or "fell back to rank" in joined


# ---------------------------------------------------------------------------
# select_montage: the optional steps that must never stop a montage
# ---------------------------------------------------------------------------

def test_an_exclusion_list_that_cannot_be_resolved_does_not_stop_the_montage(
        monkeypatch):
    """Excluding guides is a filter, and a filter that fails leaves the data.

    The montage still draws from the full count table. What must not happen is
    the run ending inside an optional filter, which would make a contaminant
    list impossible to experiment with.
    """
    import spacr.read_background as read_background

    def _explode(*args, **kwargs):
        raise RuntimeError("the exclusion list is unreadable")

    monkeypatch.setattr(read_background, "resolve_exclusions", _explode)

    plan = cm.select_montage(_objects(), _counts(), "GRA14_1", 0.2,
                             level="grna", exclude_grnas=["OTHER_1"])

    assert plan.n_objects > 0
    assert not any("excluded" in note for note in plan.notes)


def test_a_count_table_that_will_not_total_keeps_the_raw_fraction(monkeypatch):
    """With no well total there is nothing to normalise against.

    Normalisation divides a guide's fraction by what survived filtering; when
    that sum cannot be computed the raw fraction is used and the factor stays
    1, because a made-up denominator would inflate the count of cells shown.
    """
    real_labels = cm._well_labels

    def _fail_on_counts(frame, keys):
        if "grna" in getattr(frame, "columns", ()):
            raise RuntimeError("this table cannot be labelled by well")
        return real_labels(frame, keys)

    monkeypatch.setattr(cm, "_well_labels", _fail_on_counts)

    plan = cm.select_montage(_objects(), _counts(), "GRA14_1", 0.2,
                             level="grna", normalise_fraction=True)

    first = next(w for w in plan.wells if w.well.endswith("r1_c1"))
    assert first.n_expected == cm.objects_to_show(PER_WELL, first.fraction)


def test_a_preflight_that_raises_is_not_the_reason_a_montage_does_not_draw(
        monkeypatch):
    """The attribution pre-flight is a courtesy, not a precondition.

    It answers "can this guide be attributed at all" before any cell is
    attributed. Letting it fail the run would turn an advisory into a
    blocker.
    """
    import spacr.guide_attribution as guide_attribution

    def _explode(*args, **kwargs):
        raise RuntimeError("the pre-flight cannot be computed")

    monkeypatch.setattr(guide_attribution, "preflight", _explode)

    plan = cm.select_montage(
        _objects(), _counts(), "GRA14_1", 0.2, level="grna",
        picking="attributed", effects={"GRA14_1": 0.2, "OTHER_1": -0.1})

    assert plan.n_objects >= 0
    assert not any("pre-flight" in note.lower() for note in plan.notes)


# ---------------------------------------------------------------------------
# select_montage: the multivariate picker
# ---------------------------------------------------------------------------

def _effects_grid():
    return pd.DataFrame({"area": [0.9, -0.4, -0.5], "perimeter": [0.3, 0.1, -0.2]},
                        index=["GRA14_1", "GRA14_2", "OTHER_1"])


def test_the_multivariate_picker_reads_every_swept_measurement():
    """Option C decides from the measurements the sweep scored, not the score.

    One effect per measurement per guide is what separates two guides that
    share a well; collapsing to the single classification score throws away
    exactly the evidence the sweep was run to produce.
    """
    plan = cm.select_montage(
        _objects(), _counts(), "GRA14_1", 0.2, level="grna",
        picking="multivariate", effects_grid=_effects_grid(),
        effects={"GRA14_1": 0.2, "GRA14_2": 0.1, "OTHER_1": -0.1})

    joined = " ".join(w.note for w in plan.wells)
    assert "multivariate over" in joined
    assert "independent one(s)" in joined


def test_a_single_guide_well_cannot_be_told_apart_multivariately():
    """With one guide in the well there is nothing to distinguish it from.

    A posterior is a comparison; against nothing it returns the prior, so the
    picker says which fallback it took rather than reporting a multivariate
    result that was never computed.
    """
    counts = _counts({"r1_c1": {"GRA14_1": 1.0}, "r1_c2": {"OTHER_1": 1.0}})

    plan = cm.select_montage(
        _objects(), counts, "GRA14_1", 0.2, level="grna",
        picking="multivariate", effects_grid=_effects_grid())

    joined = " ".join(w.note for w in plan.wells)
    assert "this well has one guide" in joined


# ---------------------------------------------------------------------------
# Building fractions from the count CSVs the fit read
# ---------------------------------------------------------------------------

def _count_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_an_unreadable_count_file_is_named_and_the_rest_are_still_read(tmp_path):
    """One corrupt count CSV must not take the other plates down with it.

    The plate label comes from WHICH file a table is, so a skipped file keeps
    its place in the numbering; and the file that could not be read is named,
    because "no fractions" without a filename is unactionable.
    """
    bad = _corrupt_csv(tmp_path / "plate1.csv")
    good = _count_csv(tmp_path / "plate2.csv", [
        {"rowID": "r1", "columnID": "c1", "grna": "GRA14_1", "count": 30},
        {"rowID": "r1", "columnID": "c1", "grna": "OTHER_1", "count": 70},
    ])

    fractions = cm.fractions_from_counts([bad, good])

    assert set(fractions["grna"]) == {"GRA14_1", "OTHER_1"}
    assert float(fractions.loc[fractions["grna"] == "GRA14_1",
                               "fraction"].iloc[0]) == pytest.approx(0.3)
    assert fractions["prc"].str.startswith("plate2").all(), (
        "the second file keeps the second plate's label")


def test_every_count_file_being_unreadable_says_which_ones(tmp_path):
    """With nothing readable the refusal has to list what it tried.

    "The counts could not be built" alone leaves a user unable to tell a
    wrong file from a differently spelled header.
    """
    bad = _corrupt_csv(tmp_path / "plate1.csv")

    with pytest.raises(MontageError) as excinfo:
        cm.fractions_from_counts([bad])
    assert "plate1.csv" in str(excinfo.value)


def test_a_header_that_cannot_be_canonicalised_is_still_read(tmp_path, monkeypatch):
    """Canonicalisation is a convenience; the aliases below still apply.

    If the shared column-name corrector cannot run, a count table spelled the
    ordinary way must still be read rather than reported as missing 'grna'.
    """
    import spacr.schema as schema_module

    def _explode(_frame):
        raise RuntimeError("the corrector is unavailable")

    monkeypatch.setattr(schema_module, "correct_metadata_column_names", _explode)
    path = _count_csv(tmp_path / "plate1.csv", [
        {"rowID": "r1", "columnID": "c1", "grna": "GRA14_1", "count": 40},
        {"rowID": "r1", "columnID": "c1", "grna": "OTHER_1", "count": 60},
    ])

    fractions = cm.fractions_from_counts([path])

    assert float(fractions.loc[fractions["grna"] == "GRA14_1",
                               "fraction"].iloc[0]) == pytest.approx(0.4)


def test_the_guide_and_count_columns_are_accepted_under_their_usual_aliases(
        tmp_path):
    """A count table is the same identifier the rest of spaCR already accepts.

    Refusing ``guide`` and ``read_count`` here would make this module stricter
    than every other reader in the project, over a header the user cannot
    change without editing the regression's own input.
    """
    path = _count_csv(tmp_path / "plate1.csv", [
        {"rowID": "r1", "columnID": "c1", "guide": "GRA14_1", "read_count": 25},
        {"rowID": "r1", "columnID": "c1", "guide": "OTHER_1", "read_count": 75},
    ])

    fractions = cm.fractions_from_counts([path])

    assert set(fractions["grna"]) == {"GRA14_1", "OTHER_1"}
    assert float(fractions.loc[fractions["grna"] == "GRA14_1",
                               "fraction"].iloc[0]) == pytest.approx(0.25)


def test_counts_that_name_no_well_are_refused_with_what_they_would_need(
        tmp_path):
    """A fraction is per well, so a table with no well is not a count table.

    Pooling every row into one well would still produce fractions that sum to
    1, which is the corruption that stays invisible in every later check.
    """
    path = _count_csv(tmp_path / "plate1.csv", [
        {"grna": "GRA14_1", "count": 25},
        {"grna": "OTHER_1", "count": 75},
    ])

    with pytest.raises(MontageError) as excinfo:
        cm.fractions_from_counts([path])
    message = str(excinfo.value)
    assert "name no well" in message
    assert "rowID" in message


# ---------------------------------------------------------------------------
# Whatever a caller offered as scores
# ---------------------------------------------------------------------------

def test_no_scores_and_an_empty_score_frame_are_both_no_score_table():
    """A frame with no rows joins to nothing and must not look like a table.

    Returned as-is it would make the caller report "the loaded score table has
    no column that joins" instead of "no scores were offered" -- and those two
    messages send a user to different places.
    """
    assert cm._read_scores(None) is None
    assert cm._read_scores(pd.DataFrame()) is None


def test_score_paths_that_are_not_paths_are_skipped():
    """A list of scores can hold anything a caller had lying around.

    An entry that is not a path at all, and one that names no file, are both
    "nothing here" rather than reasons to fail before the montage is drawn.
    """
    assert cm._read_scores([12345]) is None
    assert cm._read_scores(["/definitely/not/a/file.csv"]) is None


def test_an_unreadable_score_file_is_skipped_not_fatal(tmp_path):
    """A corrupt score CSV leaves the montage to say it found no scores.

    Raising here would report a parser error where the caller has a specific,
    actionable message about classification not being merged.
    """
    assert cm._read_scores([_corrupt_csv(tmp_path / "scores.csv")]) is None


# ---------------------------------------------------------------------------
# Loading the per-object rows
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(tmp_path):
    """A plate whose png_list has no score column, and its score CSV.

    The crop files are deliberately absent, so the path re-rooting has
    nothing to resolve -- the state a screen is in on a machine that never
    held its pixels.
    """
    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    crops = plate / "data" / "w" / "cell_png"
    rows = []
    for index in range(4):
        name = f"plate1_r1_c{index}_f1_{index}.png"
        rows.append({"png_path": str(crops / name), "file_name": name,
                     "plateID": "plate1", "rowID": "r1",
                     "columnID": f"c{index}", "fieldID": "f1",
                     "object_label": index})
    db = plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows).to_sql("png_list", conn, index=False)
    scores = pd.DataFrame([{"path": row["png_path"], "pred": 0.1 * (i + 1),
                            "cv_predictions": i % 2}
                           for i, row in enumerate(rows)])
    csv = tmp_path / "plate1_scores.csv"
    scores.to_csv(csv, index=False)
    return {"db": str(db), "csv": str(csv)}


def test_the_scores_taken_from_a_loaded_csv_are_announced(screen, capsys):
    """A montage that borrowed its scores says so, and says nothing was written.

    The scores came from the regression's own input rather than from the
    database, and a user comparing this montage against the database later
    has to know which numbers they are looking at.
    """
    frame = cm.load_montage_objects(screen["db"], scores=[screen["csv"]],
                                    verbose=True)

    assert frame["pred"].notna().sum() == 4
    printed = capsys.readouterr().out
    assert "score(s) from the loaded score table" in printed
    assert "was not modified" in printed


def test_crop_paths_that_resolve_nowhere_are_reported(screen, capsys):
    """A column where nothing resolved is a route absent from this machine.

    Said out loud when asked for it, because the alternative is a montage of
    dead paths that draws nothing and blames the crops.
    """
    cm.load_montage_objects(screen["db"], scores=[screen["csv"]], verbose=True)

    assert "png_path" in capsys.readouterr().out


def test_an_uncanonicalisable_object_table_is_still_loaded(screen, monkeypatch):
    """Column canonicalisation is a repair, not a requirement.

    A table already spelled the canonical way needs no repair, so a corrector
    that cannot run must not stop the rows being read.
    """
    import spacr.multi_database as multi_database
    import spacr.schema as schema_module

    def _explode(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(schema_module, "correct_metadata_column_names", _explode)
    monkeypatch.setattr(multi_database, "normalise_plate_ids", _explode)

    frame = cm.load_montage_objects(screen["db"], scores=[screen["csv"]])

    assert len(frame) == 4
    assert frame["pred"].notna().sum() == 4


# ---------------------------------------------------------------------------
# Choosing the crop source
# ---------------------------------------------------------------------------

def test_a_merged_folder_given_as_a_list_still_finds_the_run_it_belongs_to(
        tmp_path, monkeypatch):
    """The channel question is answered from the run, not from the folder given.

    ``src`` arrives as a list as often as a string, and often points at
    ``merged/`` rather than the plate. Failing to walk back to the plate would
    make every such run report "no channel list", which is the caption saying
    default planes were drawn when the run had recorded its own.
    """
    import spacr.crops as crops_module

    root = tmp_path / "plate1"
    (root / "merged").mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    np.save(root / "merged" / "plate1_r1_c1_f1.npy",
            np.zeros((16, 16, 7), np.uint16))
    with sqlite3.connect(root / "measurements" / "measurements.db") as conn:
        conn.execute("CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")

    def _explode(_db):
        raise crops_module.CropError("the settings table cannot be read")

    monkeypatch.setattr(crops_module, "crop_settings_from_db", _explode)

    choice = resolve_montage_crop_source({"src": [str(root / "merged")]},
                                         objects=pd.DataFrame(
                                             {"object_label": [1]}))

    assert choice.available
    assert any("no channel list" in note
               for note in choice.requirement_notes())
