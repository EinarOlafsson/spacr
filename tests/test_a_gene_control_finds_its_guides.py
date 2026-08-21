"""195: the default control is the GENE, and the resolver can find it.

    "default for controlls in regression should be 000000"

AND THE BIOLOGY, corrected by the maintainer: "233460 is the negative
control, 000000 is the non cutting control." They are different things --
233460 is a real gene knocked out and expected to show nothing; 000000 binds
without cutting and is the empirical null every threshold and baseline is
measured against. This is about the second.

THE DEFAULT COULD NOT SIMPLY BE CHANGED, because the resolver could not find
the gene. `matches()` with no gene column asked whether a guide name STARTS
`000000_`, and the guides of a real count table start `TGGT1_000000_` -- the
organism prefix `resolve_control` has already measured and is carrying on
the spec, ignored at the point of comparison. 184 recorded "all four
spellings reach the same 28 guides", and that was measured WITH a gene column
beside the guides; this path -- the one a count table actually takes -- was
never exercised on prefixed names.

AND SELECTING NOTHING IS NOT LOUD: the thresholds fall back, the baseline
sits at zero, and the run finishes.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.control_names import matches, resolve_control, rows_for

#: A library spelled the way `process_reads` writes one.
PREFIXED = ([f"TGGT1_000000_{i}" for i in (1, 3, 11, 22)]
            + [f"TGGT1_{g}_{i}" for g in ("233460", "239740") for i in (1, 2)])

#: The same library with the prefix already dropped, as a design term has it.
BARE = [n[len("TGGT1_"):] for n in PREFIXED]


class TestAGeneFindsEveryGuideOfIt:

    @pytest.mark.parametrize("library", [PREFIXED, BARE])
    def test_the_gene_spelling(self, library):
        mask, _said = rows_for("000000", pd.Series(library), names=library)

        assert sorted(set(pd.Series(library)[mask.to_numpy()])) == \
            sorted(n for n in library if "000000" in n)

    @pytest.mark.parametrize("library", [PREFIXED, BARE])
    def test_the_prefixed_gene_spelling(self, library):
        mask, _said = rows_for("TGGT1_000000", pd.Series(library),
                               names=library)

        assert int(mask.sum()) == 4

    @pytest.mark.parametrize("library", [PREFIXED, BARE])
    def test_a_guide_finds_exactly_itself(self, library):
        mask, _said = rows_for("000000_11", pd.Series(library), names=library)

        assert int(mask.sum()) == 1

    @pytest.mark.parametrize("library", [PREFIXED, BARE])
    def test_a_prefixed_guide_does_too(self, library):
        mask, _said = rows_for("TGGT1_000000_11", pd.Series(library),
                               names=library)

        assert int(mask.sum()) == 1

    def test_all_four_spellings_agree(self):
        """184's own claim, on the library shape that broke it."""
        series = pd.Series(PREFIXED)
        genes = {int(rows_for(s, series, names=PREFIXED)[0].sum())
                 for s in ("000000", "TGGT1_000000")}
        guides = {int(rows_for(s, series, names=PREFIXED)[0].sum())
                  for s in ("000000_11", "TGGT1_000000_11")}

        assert genes == {4}
        assert guides == {1}

    def test_it_does_not_reach_another_gene(self):
        mask, _said = rows_for("000000", pd.Series(PREFIXED), names=PREFIXED)

        assert not any("233460" in n
                       for n in pd.Series(PREFIXED)[mask.to_numpy()])

    def test_a_gene_column_still_wins_where_there_is_one(self):
        """The path 184 measured. It must not have changed."""
        guides = pd.Series(["TGGT1_000000_1", "TGGT1_233460_1"])
        genes = pd.Series(["000000", "233460"])
        spec = resolve_control("000000", names=list(guides))

        assert list(matches(spec, guides, genes)) == [True, False]


class TestTheDefault:

    def test_controls_is_the_gene(self):
        from spacr.settings import get_perform_regression_default_settings

        assert get_perform_regression_default_settings({})["controls"] == \
            ["000000"]

    def test_the_negative_control_is_untouched(self):
        """A different question, and its answer was already right."""
        from spacr.settings import get_perform_regression_default_settings

        assert str(get_perform_regression_default_settings({})
                   ["negative_control"]) == "233460"

    def test_the_old_thirty_name_list_still_loads(self):
        """A settings CSV written before this must keep meaning what it
        meant."""
        from spacr.control_names import resolve_controls

        old = [f"000000_{i}" for i in (1, 3, 11, 22)]
        specs = resolve_controls(old, names=PREFIXED)

        from spacr.control_names import GUIDE

        assert len(specs) == 4
        assert all(s.level == GUIDE for s in specs)

    def test_and_selects_the_same_guides(self):
        series = pd.Series(PREFIXED)
        old = [f"000000_{i}" for i in (1, 3, 11, 22)]

        picked = set()
        for name in old:
            mask, _said = rows_for(name, series, names=PREFIXED)
            picked |= set(series[mask.to_numpy()])

        gene, _said = rows_for("000000", series, names=PREFIXED)
        assert picked == set(series[gene.to_numpy()])


@pytest.mark.slow
def test_the_default_resolves_on_the_example_screen():
    """The reported case: 30 guides, on the tables it was measured against."""
    import glob

    from spacr.cell_montage import fractions_from_counts
    from spacr.example_data import cache_folder
    from spacr.settings import get_perform_regression_default_settings

    files = sorted(glob.glob(f"{cache_folder()}/*unique_combinations.csv"))
    if len(files) != 4:
        pytest.skip("the example screen is not downloaded")

    guides = fractions_from_counts(files)["grna"].astype(str)
    names = [str(g) for g in guides.unique()]
    default = get_perform_regression_default_settings({})["controls"]

    mask, said = rows_for(default[0], guides, names=names)
    assert len(set(guides[mask.to_numpy()])) == 30, said
