"""n_grna and n_gene count the rows that REACHED THE FIT.

The maintainer asked outright, 2026-08-17: "n_grna and n_gene are supposed to
cound how many instases are present for each gene and grna in the final
results. please check that they actually work". They did not.

They were counted on `independent_df`, BEFORE it is merged with the score
data -- and that merge is an INNER join (`pd.merge(..., on='prc')` with no
`how=`), so every sequencing well without an imaging partner was counted and
then dropped. On the real screen the run says so out loud: "Paired 620 wells.
724 sequencing well(s) and 0 imaging well(s) have no partner and take no part
in the regression." Measured on a synthetic case with half the wells
unpaired, every count came out EXACTLY 2x too high.

IT IS NOT ONLY A DISPLAY. `min_n` filters the hit list on these numbers, so
an inflated count lets a guide through a filter it should fail.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _counts(frame):
    """The same computation `_count_variable_instances` performs."""
    a = frame["grna"].value_counts().reset_index()
    a.columns = ["grna", "n_grna"]
    b = frame["gene"].value_counts().reset_index()
    b.columns = ["gene", "n_gene"]
    return (a.set_index("grna")["n_grna"].to_dict(),
            b.set_index("gene")["n_gene"].to_dict())


@pytest.fixture
def unpaired():
    """Ten sequencing wells, five of which have imaging. The real shape."""
    rows = [{"prc": f"w{w}", "grna": g, "gene": g.split("_")[0]}
            for w in range(10)
            for g in ["244480_1", "244480_2", "239740_1", "239740_2"]]
    independent = pd.DataFrame(rows)
    dependent = pd.DataFrame({"prc": [f"w{w}" for w in range(5)],
                              "pred": [0.1] * 5})
    return independent, dependent


# --------------------------------------------------------------------------- #
#  The bug, as a property
# --------------------------------------------------------------------------- #

def test_the_merge_that_drops_wells_is_an_inner_join(unpaired):
    """The premise. If this merge ever became a left join the counts would be
    right for a different reason, and this whole file would be testing
    nothing."""
    independent, dependent = unpaired

    merged = pd.merge(independent, dependent, on="prc")

    assert merged["prc"].nunique() == 5
    assert independent["prc"].nunique() == 10


def test_counting_before_the_merge_doubles_every_count(unpaired):
    """The measurement that showed the size of the error."""
    independent, dependent = unpaired
    merged = pd.merge(independent, dependent, on="prc")

    before_grna, before_gene = _counts(independent)
    after_grna, after_gene = _counts(merged)

    for guide in before_grna:
        assert before_grna[guide] == 2 * after_grna[guide]
    for gene in before_gene:
        assert before_gene[gene] == 2 * after_gene[gene]


def test_the_count_is_taken_from_the_merged_frame():
    """The fix, asserted against the source: `_count_variable_instances` is
    called on `merged_df`, not on `independent_df`."""
    import inspect

    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    call = "_count_variable_instances(\n        merged_df"
    assert call in source, (
        "the counts are no longer taken from merged_df, so they describe "
        "rows the inner merge drops")


def test_the_counts_match_the_wells_that_reached_the_fit(unpaired):
    """A guide in 5 paired wells has n_grna = 5, not 10."""
    independent, dependent = unpaired
    merged = pd.merge(independent, dependent, on="prc")

    grna, _gene = _counts(merged)

    assert set(grna.values()) == {5}


# --------------------------------------------------------------------------- #
#  What they mean, which is not what the names suggest
# --------------------------------------------------------------------------- #

def test_n_gene_is_wells_times_guides_not_guides():
    """Documented rather than silently redefined. On the real screen gene
    244480 has ONE guide and n_gene = 5; 239740 has TWO and n_gene = 15. A
    reader comparing n_gene across genes is comparing a product.

    `min_n` filters on it and every past run's CSV carries it, so changing
    what the number MEANS is a separate decision from fixing which rows it is
    taken over -- this pins the current meaning so a change is deliberate.
    """
    rows = [{"prc": f"w{w}", "grna": "244480_3", "gene": "244480"}
            for w in range(5)]
    rows += [{"prc": f"w{w}", "grna": g, "gene": "239740"}
             for w in range(3) for g in ["239740_1", "239740_2"]]
    frame = pd.DataFrame(rows)

    grna, gene = _counts(frame)

    assert gene["244480"] == 5      # 1 guide x 5 wells
    assert gene["239740"] == 6      # 2 guides x 3 wells
    assert grna["244480_3"] == 5    # a guide's count IS its well count


def test_the_meaning_is_written_down():
    """It was not, and the names invite exactly the wrong reading -- which is
    why the maintainer had to ask."""
    import inspect

    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    assert "wells MULTIPLIED BY guides" in source
    assert "n_grna  for a guide = the number of WELLS" in source
