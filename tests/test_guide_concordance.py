"""Whether a gene's signal is its guides agreeing, or one guide carrying it.

The distinction that reorders the TSG101 hit list: gene 244480 outranks EAF1
on a single surviving guide, while EAF1's three guides agree without any of
them being significant alone.
"""
import numpy as np
import pandas as pd
import pytest


def _frame(rows):
    return pd.DataFrame(rows, columns=["feature", "coefficient", "p_value"])


class TestGuideSupport:

    def test_a_single_guide_gene_is_flagged(self):
        """Its gene-level p IS that guide's p, so it is not extra evidence."""
        from spacr.guide_concordance import guide_support

        support = guide_support(_frame([
            ("fraction:grna[244480_3]", 2.0, 1.6e-12),
            ("gene_fraction:gene[244480]", 2.0, 1.6e-12),
        ]))
        row = support.loc["244480"]
        assert row["single_guide"]
        assert row["n_guides"] == 1
        assert row["gene_p"] == row["best_guide_p"]

    def test_agreement_without_individual_significance(self):
        """EAF1's actual shape: three guides, none significant, all agreeing."""
        from spacr.guide_concordance import guide_support

        support = guide_support(_frame([
            ("fraction:grna[225160_1]", 0.4, 0.51),
            ("fraction:grna[225160_2]", 0.6, 0.14),
            ("fraction:grna[225160_3]", 0.5, 0.27),
            ("gene_fraction:gene[225160]", 0.5, 4.6e-08),
        ]))
        row = support.loc["225160"]
        assert not row["single_guide"]
        assert row["n_guides"] == 3
        assert row["n_guides_significant"] == 0
        assert row["concordance"] == 1.0

    def test_guides_disagreeing_in_direction_are_visible(self):
        """No p-value threshold reveals this, and it is the strongest
        argument that a hit is noise."""
        from spacr.guide_concordance import guide_support

        support = guide_support(_frame([
            ("fraction:grna[313330_1]", +1.0, 4.2e-05),
            ("fraction:grna[313330_2]", -1.2, 1.5e-05),
            ("gene_fraction:gene[313330]", -0.1, 8.6e-06),
        ]))
        assert support.loc["313330"]["concordance"] == 0.5

    def test_direction_is_taken_from_the_mean_not_the_biggest_guide(self):
        """Letting the strongest guide define 'correct' hides disagreement."""
        from spacr.guide_concordance import guide_support

        support = guide_support(_frame([
            ("fraction:grna[1_1]", -5.0, 0.001),   # one large negative
            ("fraction:grna[1_2]", +0.1, 0.9),
            ("fraction:grna[1_3]", +0.1, 0.9),
        ]))
        # The mean is negative, so only the large guide agrees with it.
        assert support.loc["1"]["n_same_direction"] == 1

    def test_an_empty_or_odd_table_returns_an_empty_frame(self):
        from spacr.guide_concordance import guide_support

        assert not len(guide_support(pd.DataFrame()))
        assert not len(guide_support(None))
        assert not len(guide_support(pd.DataFrame({"other": [1]})))


class TestTheReport:

    def test_it_names_the_single_guide_hits(self):
        from spacr.guide_concordance import concordance_report

        text = concordance_report(_frame([
            ("fraction:grna[244480_3]", 2.0, 1.6e-12),
            ("gene_fraction:gene[244480]", 2.0, 1.6e-12),
        ]))
        assert "SINGLE GUIDE" in text

    def test_a_control_in_the_hit_list_is_called_out(self):
        """The most useful line in the report, and the easiest to miss when
        it is just another six-digit number."""
        from spacr.guide_concordance import concordance_report

        text = concordance_report(
            _frame([("fraction:grna[233460_1]", 1.0, 0.002),
                    ("gene_fraction:gene[233460]", 1.0, 0.002)]),
            controls={"233460": "negative"})
        assert "NEGATIVE CONTROL" in text

    def test_nothing_significant_says_so(self):
        from spacr.guide_concordance import concordance_report

        text = concordance_report(_frame([
            ("fraction:grna[1_1]", 0.1, 0.9),
            ("gene_fraction:gene[1]", 0.1, 0.9),
        ]))
        assert "No gene reached" in text

    def test_no_guide_terms_at_all_is_reported_not_guessed(self):
        from spacr.guide_concordance import concordance_report

        assert "unknown" in concordance_report(
            _frame([("Intercept", 1.0, 0.01)]))


class TestAnnotating:

    def test_the_columns_are_joined_without_touching_the_original(self):
        from spacr.guide_concordance import annotate_results

        frame = _frame([
            ("fraction:grna[1_1]", 1.0, 0.01),
            ("fraction:grna[1_2]", 1.0, 0.02),
            ("gene_fraction:gene[1]", 1.0, 0.001),
        ])
        before = list(frame.columns)
        out = annotate_results(frame)
        assert list(frame.columns) == before, "the caller's table was rewritten"
        assert "n_guides" in out.columns
        assert out.loc[0, "n_guides"] == 2
