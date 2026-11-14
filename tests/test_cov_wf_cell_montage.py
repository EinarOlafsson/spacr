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

import pandas as pd
import pytest

from spacr import cell_montage as cm
from spacr import schema

PER_WELL = 6

#: Which gene each guide in these fixtures belongs to.
GENE_OF = {"GENE_1": "GENE", "GENE_2": "GENE", "OTHER_1": "OTHER"}


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
# Reading the objects out of a measurements database
# ---------------------------------------------------------------------------

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
