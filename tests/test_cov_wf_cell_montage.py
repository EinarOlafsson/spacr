"""The montage's quieter branches: fallbacks, exclusions and re-reads.

Every path here is one where the montage keeps drawing after something it
wanted was not there -- a count table with no guide column, a well whose only
guide was excluded as a contaminant, a picker that cannot run on a well with
one guide, a run folder with no ``measurements.db``. Each of them is a place
where a wrong picture could be drawn under a caption that still reads as if
everything worked, so what is asserted here is the caption as much as the
cells: which wells contributed, what arithmetic is quoted against each, and
what the run says it could not do.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import cell_montage as cm
from spacr import schema

PER_WELL = 6

#: Which gene each guide in these fixtures belongs to.
GENE_OF = {"GENE_1": "GENE", "GENE_2": "GENE", "OTHER_1": "OTHER"}


def _objects(wells, per_well=PER_WELL):
    """The per-object frame a montage selects from, one plate, evenly scored."""
    rows = []
    for index, well in enumerate(wells):
        row_id, column_id = well.split("_")
        for label in range(1, per_well + 1):
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id, "fieldID": "f1",
                "object_label": label,
                "pred": round(0.05 + 0.9 * (label - 1) / (per_well - 1), 4),
                "area": 100.0 + label + 10 * index,
                "perimeter": 40.0 + 2 * label,
            })
    return pd.DataFrame(rows)


def _counts(fractions, gene_of=None, cell_count=PER_WELL):
    """A ``regression_data.csv``-shaped frame: one row per well and guide."""
    genes = dict(gene_of or GENE_OF)
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


def _note_for(plan, well):
    """The caption line the plan recorded against one well."""
    return next(w.note for w in plan.wells if w.well == well)


# ---------------------------------------------------------------------------
# Reading a coefficient's level off the count data
# ---------------------------------------------------------------------------

def test_a_gene_only_count_table_still_names_the_level():
    """Not every count table carries a guide column, and gene fits use them.

    A gene-level regression can be handed a table aggregated to genes. Looking
    for ``grna`` first and refusing when it is absent would make every such
    coefficient unclickable, because the level is what decides whether the
    montage sums a gene's guides or takes one.
    """
    gene_only = pd.DataFrame({"gene": ["GENE", "OTHER"],
                              "prc": ["plate1_r1_c1", "plate1_r1_c2"],
                              "fraction": [0.4, 0.6]})

    assert cm.coefficient_level(gene_only, "GENE") == "gene"
    assert cm.coefficient_level(gene_only, "OTHER") == "gene"


def test_a_table_naming_neither_guides_nor_genes_is_refused_with_its_size():
    """A coefficient the counts have never seen must be named, not guessed at.

    The wrong table -- a results file, an aggregate -- is the usual cause, and
    the refusal has to carry the row count so the reader can see it looked at
    a real table and still found nothing, rather than silently drawing a
    montage of every well.
    """
    wrong_table = pd.DataFrame({"feature": ["GENE"] * 3,
                                "coefficient": [0.4, 0.1, -0.2]})

    with pytest.raises(cm.CoefficientNotFound) as raised:
        cm.coefficient_level(wrong_table, "GENE")

    message = str(raised.value)
    assert "'GENE'" in message
    assert "3 rows" in message
    # The same name against a table that does carry it resolves normally.
    assert cm.coefficient_level(_counts({"r1_c1": {"GENE_1": 1.0}}),
                                "GENE") == "gene"


# ---------------------------------------------------------------------------
# The attribution pre-flight, and what excluding a guide does to it
# ---------------------------------------------------------------------------

def test_a_well_emptied_by_an_exclusion_does_not_stop_the_preflight():
    """One contaminated well must not silence the answer for the others.

    ``exclude_grnas`` removes rows from the count table, and a well whose only
    guide was excluded then has no fractions at all. The pre-flight has to walk
    past it and still report on the wells that do have guides -- otherwise
    excluding one contaminant would remove the caption line that says whether
    the coefficient can be attributed anywhere.
    """
    fractions = {"r1_c1": {"GENE_1": 1.0},
                 "r1_c2": {"GENE_1": 0.4, "OTHER_1": 0.6}}

    plan = cm.select_montage(
        _objects(("r1_c1", "r1_c2")), _counts(fractions), "GENE_1", 0.4,
        picking="attributed", effects={"GENE_1": 0.5, "OTHER_1": -0.3},
        exclude_grnas=["GENE_1"])

    assert any("excluded 1 guide(s)" in note for note in plan.notes)
    assert any("GENE_1" in note and "no well carries it" in note
               for note in plan.notes)
    assert [w.well for w in plan.wells] == ["plate1_r1_c1", "plate1_r1_c2"]
    assert plan.n_objects == 10


def test_excluding_every_guide_leaves_the_preflight_nothing_to_say():
    """A pre-flight is a courtesy; with no fractions left it must stay silent.

    Reporting "cannot be attributed" when the count table has been emptied by
    the user's own exclusion list would read as a statement about the guide
    rather than about the exclusion, and the montage still draws either way.
    """
    fractions = {"r1_c1": {"GENE_1": 1.0},
                 "r1_c2": {"GENE_1": 0.5, "OTHER_1": 0.5}}
    objects = _objects(("r1_c1", "r1_c2"))
    effects = {"GENE_1": 0.5, "OTHER_1": -0.3}

    kept = cm.select_montage(objects, _counts(fractions), "GENE_1", 0.4,
                             picking="attributed", effects=effects)
    stripped = cm.select_montage(objects, _counts(fractions), "GENE_1", 0.4,
                                 picking="attributed", effects=effects,
                                 exclude_grnas=["GENE_1", "OTHER_1"])

    assert any("can be attributed in 2 of its 2 wells" in note
               for note in kept.notes)
    assert not any("attributed in" in note for note in stripped.notes)
    assert stripped.notes == (
        "excluded 2 guide(s) named by exclude_grnas before any fraction was "
        "formed",)
    assert stripped.n_objects == 9


# ---------------------------------------------------------------------------
# Pickers that cannot run on the well in front of them
# ---------------------------------------------------------------------------

def test_a_well_holding_one_guide_is_decided_by_rank_not_by_attribution():
    """A posterior is a comparison, so one guide in a well has no rival.

    Running the attribution there would return the prior for every cell and
    "attribute" the whole well on no evidence. The well is decided by score
    instead, and its caption quotes the fraction arithmetic that really made
    the choice -- while a well with two guides still shows the attribution's
    own count.
    """
    fractions = {"r1_c1": {"GENE_1": 1.0},
                 "r1_c2": {"GENE_1": 0.5, "OTHER_1": 0.5}}

    plan = cm.select_montage(
        _objects(("r1_c1", "r1_c2")), _counts(fractions), "GENE_1", 0.4,
        picking="attributed", effects={"GENE_1": 0.5, "OTHER_1": -0.3})

    lone = _note_for(plan, "plate1_r1_c1")
    shared = _note_for(plan, "plate1_r1_c2")
    assert lone.startswith("round(1 x 6) = 6")
    assert "attributed chose" not in lone
    assert "attributed chose 3 of 6 classified cell(s)" in shared
    assert [w.n_selected for w in plan.wells] == [6, 3]


def test_a_gene_level_multivariate_montage_still_draws_its_wells():
    """The sweep grid scores guides; a gene coefficient is not one of them.

    Option C returns a posterior per guide, so a gene-level montage finds its
    own name nowhere in that answer. It must fall through to the cells the
    fraction supports rather than return an empty montage, which is what a
    reader would read as "this gene has no cells".
    """
    fractions = {"r1_c1": {"GENE_1": 0.4, "OTHER_1": 0.6},
                 "r1_c2": {"GENE_2": 0.5, "OTHER_1": 0.5},
                 "r1_c3": {"OTHER_1": 1.0}}
    grid = pd.DataFrame({"area": [0.6, 0.4, -0.5],
                         "perimeter": [0.3, 0.2, -0.2]},
                        index=["GENE_1", "GENE_2", "OTHER_1"])

    plan = cm.select_montage(
        _objects(("r1_c1", "r1_c2", "r1_c3")), _counts(fractions), "GENE", 0.4,
        picking="multivariate", effects_grid=grid,
        effects={"GENE_1": 0.5, "GENE_2": 0.4, "OTHER_1": -0.3})

    assert [w.well for w in plan.wells] == ["plate1_r1_c1", "plate1_r1_c2"]
    assert [w.n_selected for w in plan.wells] == [2, 3]
    assert plan.n_objects == 5
    assert not any("GENE: no well carries it" in note for note in plan.notes)
    assert any(note.startswith("GENE_1 can be attributed")
               for note in plan.notes)
    assert any(note.startswith("GENE_2 can be attributed")
               for note in plan.notes)
    for well in plan.wells:
        assert str(well.n_selected) in well.note


# ---------------------------------------------------------------------------
# The montage cap
# ---------------------------------------------------------------------------

def test_the_cap_only_annotates_the_wells_it_actually_trimmed():
    """A well that gave nothing was not trimmed, and must not say it was.

    The cap rewrites each well's caption with what it took away. A well whose
    fraction already rounded to zero kept exactly what it had -- zero -- and
    appending "trimmed from 0 to 0" would invent a loss and hide the real
    reason that well is empty.
    """
    fractions = {"r1_c1": {"GENE_1": 1.0, "OTHER_1": 0.0},
                 "r1_c2": {"GENE_1": 0.02, "OTHER_1": 0.98}}

    plan = cm.select_montage(_objects(("r1_c1", "r1_c2")), _counts(fractions),
                             "GENE_1", 0.4, cap=2)

    assert plan.capped
    assert (plan.n_before_cap, plan.n_objects) == (6, 2)
    trimmed = _note_for(plan, "plate1_r1_c1")
    untouched = _note_for(plan, "plate1_r1_c2")
    assert "trimmed by the montage cap from 6 to 2" in trimmed
    assert "trimmed" not in untouched
    assert "rounds to zero" in untouched


# ---------------------------------------------------------------------------
# Reading the objects out of a measurements database
# ---------------------------------------------------------------------------

@pytest.fixture()
def screen(tmp_path):
    """A plate whose png_list has no score column, plus a score CSV for it."""
    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    crops = plate / "data" / "w" / "cell_png"
    crops.mkdir(parents=True)
    rows = []
    for i in range(4):
        name = f"plate1_r1_c{i}_f1_{i}.png"
        (crops / name).write_bytes(b"x")
        rows.append({"png_path": str(crops / name), "file_name": name,
                     "plateID": "plate1", "rowID": "r1", "columnID": f"c{i}",
                     "fieldID": "f1", "object_label": i})
    db = plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows).to_sql("png_list", conn, index=False)

    csv = tmp_path / "plate1_dv.csv"
    pd.DataFrame([{"path": r["png_path"], "pred": 0.1 * (i + 1)}
                  for i, r in enumerate(rows)]).to_csv(csv, index=False)
    return {"db": str(db), "csv": str(csv)}


def test_an_offer_of_scores_that_holds_no_table_still_refuses_by_name(screen):
    """"I gave you the scores" and "the scores are unreadable" differ.

    An empty or unreadable list of score files leaves the montage with no
    score at all, and the refusal must say the offered table was looked at --
    otherwise the user is told to run Classify while holding the CSVs that
    would have worked.
    """
    with pytest.raises(cm.MissingScores) as raised:
        cm.load_montage_objects(screen["db"], scores=[])

    message = str(raised.value)
    assert "The loaded score table has no column that joins to it either." \
        in message
    assert "No score table was offered" not in message
    # The same database with a readable score CSV loads instead of refusing.
    loaded = cm.load_montage_objects(screen["db"], scores=[screen["csv"]])
    assert loaded["pred"].notna().sum() == 4


def test_a_well_key_the_database_already_recorded_is_left_alone(tmp_path):
    """The counts join on the ``prc`` the run wrote, not on a fresh one.

    ``prc`` is only composed when the table has none. Recomposing over a value
    the database already carries would rewrite the exact string the count data
    is keyed by, and every well would then report that no object came from it.
    """
    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    crops = plate / "data" / "w" / "cell_png"
    crops.mkdir(parents=True)
    rows = []
    for i in range(4):
        name = f"plate1_r1_c{i}_f1_{i}.png"
        (crops / name).write_bytes(b"x")
        rows.append({"png_path": str(crops / name), "file_name": name,
                     "plateID": "plate1", "rowID": "r1", "columnID": f"c{i}",
                     "fieldID": "f1", "object_label": i,
                     "prc": f"plate1_rALT_c{i}", "pred": 0.1 * (i + 1)})
    db = plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows).to_sql("png_list", conn, index=False)

    frame = cm.load_montage_objects(str(db))

    assert frame["prc"].tolist() == [f"plate1_rALT_c{i}" for i in range(4)]
    assert schema.compose_prc("plate1", "r1", "c0") == "plate1_r1_c0"
    assert "plate1_r1_c0" not in set(frame["prc"])


# ---------------------------------------------------------------------------
# Choosing the crop source when the run folder is all there is
# ---------------------------------------------------------------------------

def test_a_run_with_no_measurements_database_says_its_channels_are_assumed(
        tmp_path):
    """A folder of ``merged/*.npy`` and nothing else recorded no channels.

    The crop spec always produces some planes, so an unrecorded run silently
    draws 2,1,0 and looks like a deliberate choice. With no ``measurements.db``
    to read ``png_dims`` back from there is nothing that could have declared
    them, and the caption has to say so.
    """
    root = tmp_path / "plate1"
    (root / "merged").mkdir(parents=True)
    np.save(root / "merged" / "plate1_r1_c1_f1.npy",
            np.zeros((8, 8, 5), np.uint16))
    assert not (root / "measurements" / "measurements.db").exists()

    choice = cm.resolve_montage_crop_source(
        {"src": [str(root / "merged")]},
        objects=pd.DataFrame({"object_label": [1]}))

    assert choice.available
    assert choice.requirements.route == "merged-mask"
    assumed = [note for note in choice.requirement_notes()
               if "no channel list" in note]
    assert len(assumed) == 1
    assert "[2, 1, 0]" in assumed[0]
