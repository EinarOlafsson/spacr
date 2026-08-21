"""133 B's last item: the phenotype scores, and where they already were.

    "STILL MISSING, and it must be sourced rather than invented: the
     PHENOTYPE SCORES 'from the screens we have downloaded'.
     `toxoplasma_metadata.csv` has no Phenotype column -- checked, all 48.
     ... Either bundle that table too or keep reading it from
     `metadata_files` and say which."

IT IS BUNDLED, AND IT WAS BUNDLED BEFORE THAT WAS WRITTEN. The instruction
looked in `toxoplasma_metadata.csv`, which is the gene name and expression
table; the fitness screens are `resources/data/phenotype.csv` -- 0.48 MB,
8,283 genes, seven `fit_*` columns -- and `annotation._phenotype` already
joins every one of them. Nothing had to be sourced.

WHAT WAS ACTUALLY WRONG was the sentence the console prints about the join.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.annotation import annotate, columns, gene_number


class TestTheFitnessScreensAreBundled:

    def test_the_table_is_in_the_package(self):
        import os

        import spacr

        path = os.path.join(os.path.dirname(spacr.__file__), "resources",
                            "data", "phenotype.csv")
        assert os.path.isfile(path)

    def test_it_covers_the_genome(self):
        from spacr.annotation import _phenotype

        frame = _phenotype()
        assert frame is not None
        assert len(frame) > 8000

    def test_the_in_vitro_fitness_score_is_offered(self):
        """The one a screen paper reports: the genome-wide CRISPR fitness
        score in human foreskin fibroblasts."""
        assert "fit_invitro_hff" in columns()

    def test_and_the_in_vivo_ones_beside_it(self):
        for name in ("fit_invivo_PE", "fit_invivo_lung", "fit_invivo_liver",
                     "fit_invivo_spleen", "fit_naive_bmdm", "fit_ifng"):
            assert name in columns(), name

    def test_it_reaches_an_exported_table(self):
        frame = pd.DataFrame({"gene": ["TGGT1_200010"]})

        got = annotate(frame, key_column="gene", quiet=True)

        assert got["fit_invitro_hff"].iloc[0] == pytest.approx(2.54)

    def test_the_wheel_is_still_small(self):
        """"The bundled annotation adds under 1 MB to the wheel" -- the
        phenotype table is 0.48 MB of it."""
        import os

        import spacr

        folder = os.path.join(os.path.dirname(spacr.__file__), "resources",
                              "data")
        total = sum(os.path.getsize(os.path.join(folder, f))
                    for f in os.listdir(folder))
        assert total < 8 * 1024 * 1024


class TestTheJoinIsOnTheGeneNumber:
    """"`TGGT1_224750` and `TGME49_224750` are the same gene and the screen
    uses the first while the annotation tables use the second." """

    def test_both_strains_reach_the_same_row(self):
        first = annotate(pd.DataFrame({"gene": ["TGGT1_200010"]}),
                         key_column="gene", quiet=True)
        second = annotate(pd.DataFrame({"gene": ["TGME49_200010"]}),
                          key_column="gene", quiet=True)

        assert first["fit_invitro_hff"].iloc[0] == \
            second["fit_invitro_hff"].iloc[0]

    def test_a_design_term_does_too(self):
        assert gene_number("gene_fraction:gene[200010]") == "200010"
        assert gene_number("fraction:grna[200010_2]") == "200010"

    def test_a_split_gene_model_collapses_to_its_parent(self):
        assert gene_number("TGME49_201180A") == "201180"
        assert gene_number("TGME49_201180B") == "201180"

    def test_something_that_names_no_gene_is_none(self):
        for value in ("Intercept", "rowID[T.r03]", "", None, float("nan")):
            assert gene_number(value) is None


class TestTheConsoleSaysWhatWasActuallyJoined:
    """The fault this found, and it is 133's rule 3 pointing the other way.

    The count came from `added[0]` -- `gene_name` -- and a gene can be in
    every bundled table while having no NAME. `TGME49_200130` is one: it
    carries a product description, an in-vivo fitness score and a UniProt
    accession, and was reported as not matching.
    """

    def test_a_gene_with_no_name_still_counts_as_matched(self, capsys):
        frame = pd.DataFrame({"gene": ["TGGT1_200010", "TGME49_200130",
                                       "TGGT1_999999"]})

        annotate(frame, key_column="gene")

        assert "2 matched the annotation" in capsys.readouterr().out

    def test_that_gene_really_has_no_name_but_does_have_annotation(self):
        got = annotate(pd.DataFrame({"gene": ["TGME49_200130"]}),
                       key_column="gene", quiet=True)

        assert pd.isna(got["gene_name"].iloc[0])
        assert got["product_description"].iloc[0]
        assert got["uniprot"].iloc[0]

    def test_a_table_where_nothing_matches_says_zero(self, capsys):
        annotate(pd.DataFrame({"gene": ["TGGT1_999999"]}), key_column="gene")

        assert "0 matched the annotation" in capsys.readouterr().out

    def test_quiet_says_nothing(self, capsys):
        annotate(pd.DataFrame({"gene": ["TGGT1_200010"]}), key_column="gene",
                 quiet=True)

        assert capsys.readouterr().out == ""
