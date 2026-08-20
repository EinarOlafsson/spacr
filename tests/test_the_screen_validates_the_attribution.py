"""Opt-in: instructions 172 and 173, validated on a real pooled screen.

There is no built-in path here. One environment variable names a screen
folder and it is the only way in::

    SPACR_SCREEN=/somewhere/tsg101_screen \\
        pytest -m slow tests/test_the_screen_validates_the_attribution.py -s

The folder must hold, for each plate N:

    plateN_dv.csv                            per-cell classification scores
    plateN/measurements/measurements.db      the measurements
    claude/plate_N_unique_combinations.csv   the per-well gRNA read counts

Set but not a directory is a FAILURE, not a skip -- the same rule
`test_e2e_real_dataset` keeps, and for the same reason: a typo that skipped
would report green for nothing.

WHAT THIS IS FOR. Instructions 172 and 173 each end in a list headed HOW TO
KNOW IT WORKED, and most of that list cannot be answered by a fixture. The
normalisation factor is only interesting if real count tables really do fall
to half after `fraction_threshold`; the permutation is only a test if the
guides really are as thin in the tail as the instruction says. These are
those checks, run against the screen they were written about.

MEASURED ON THE MAINTAINER'S FOUR PLATES, 2026-08-20. Every number the
assertions below use came from that run and is quoted where it is used, so a
later reader can tell a threshold that was measured from one that was picked.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

pd = pytest.importorskip("pandas")

PLATES = (1, 2, 3, 4)
VARIABLE = "SPACR_SCREEN"


def _root() -> Path:
    raw = os.environ.get(VARIABLE)
    if not raw:
        pytest.skip(f"set {VARIABLE} to a pooled-screen folder to run this")
    root = Path(raw)
    if not root.is_dir():
        raise AssertionError(
            f"{VARIABLE}={raw!r} is not a directory. You opted in, so this is "
            "a failure rather than a skip.")
    return root


@pytest.fixture(scope="module")
def root() -> Path:
    return _root()


@pytest.fixture(scope="module")
def counts(root):
    """Per-well gRNA fractions at the default `fraction_threshold`."""
    from spacr.ml import process_reads

    frames = []
    for n in PLATES:
        path = root / "claude" / f"plate_{n}_unique_combinations.csv"
        if not path.exists():
            pytest.skip(f"no count table at {path}")
        frame = process_reads(path, 0.02, f"plate{n}")
        frame["plate"] = n
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def raw_counts(root):
    """The same tables with no threshold at all."""
    from spacr.ml import process_reads

    frames = []
    for n in PLATES:
        path = root / "claude" / f"plate_{n}_unique_combinations.csv"
        if not path.exists():
            pytest.skip(f"no count table at {path}")
        frames.append(process_reads(path, None, f"plate{n}"))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def cells(root):
    """Per-cell classification scores, with `pathogen_area` joined on.

    THREE SPELLINGS IN ONE SCREEN, which is instruction 145's whole point
    seen in the wild: plate1_dv.csv says `col` and plates 2-4 say `column`;
    plate1's database says rowID/columnID and plates 2-4 say
    row_name/column_name. spaCR's own normaliser folds all of them, so this
    reads through it rather than carrying a spelling table.
    """
    from spacr.schema import correct_metadata_column_names

    frames = []
    for n in PLATES:
        dv_path = root / f"plate{n}_dv.csv"
        db_path = root / f"plate{n}" / "measurements" / "measurements.db"
        if not dv_path.exists() or not db_path.exists():
            pytest.skip(f"plate {n} is not laid out as this test expects")
        dv = correct_metadata_column_names(pd.read_csv(dv_path))
        dv["prc"] = dv["prc"].str.replace(r"^p", "", regex=True)
        dv["plate"] = n
        with sqlite3.connect(db_path) as con:
            png = correct_metadata_column_names(
                pd.read_sql("SELECT * FROM png_list", con))
            pathogen = correct_metadata_column_names(
                pd.read_sql("SELECT * FROM pathogen", con))
        keys = ["rowID", "columnID", "fieldID", "cell_id"]
        pathogen["cell_id"] = "o" + pathogen["cell_id"].astype("Int64").astype(str)
        # a cell can hold several pathogens; its burden is their sum
        area = pathogen.groupby(keys, as_index=False)["pathogen_area"].sum()
        png = png.merge(area, on=keys, how="left")
        frames.append(dv.dropna(subset=["pred"]).merge(
            png[["file_name", "pathogen_area"]],
            left_on="path", right_on="file_name", how="left"))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Instruction 172 -- the normalisation is not a no-op                          #
# --------------------------------------------------------------------------- #

class TestTheNormalisationIsReal:
    """The premise instruction 172 rests on, re-measured on the screen."""

    def test_the_raw_tables_sum_to_exactly_one(self, raw_counts):
        summed = raw_counts.groupby("prc")["fraction"].sum()
        assert np.allclose(summed, 1.0, atol=1e-9), (
            "a raw count table whose fractions do not sum to 1 means the well "
            "key and the read counts did not line up")

    def test_the_threshold_halves_them(self, counts):
        """Measured: median 0.5526, minimum 0.1515 across 1,536 wells."""
        summed = counts.groupby("prc")["fraction"].sum()
        assert summed.median() < 0.8, (
            f"the filtered fractions sum to a median of {summed.median():.4f}; "
            "if that were near 1 the normalisation would be the no-op "
            "instruction 172 argues it is not")
        assert summed.min() < 0.5

    def test_the_factor_reaches_several_fold(self, counts):
        """Measured: up to 6.60x, so an un-normalised share is far too small."""
        summed = counts.groupby("prc")["fraction"].sum()
        assert (1.0 / summed).max() > 3.0

    def test_no_well_is_normalised_above_one(self, counts):
        from spacr.cell_montage import normalised_share

        summed = counts.groupby("prc")["fraction"].sum()
        for _, row in counts.sample(2000, random_state=0).iterrows():
            share, factor = normalised_share(
                [summed[row["prc"]]], float(row["fraction"]))
            assert 0.0 <= share <= 1.0
            assert factor >= 1.0 - 1e-9


@pytest.fixture(scope="module")
def shares(counts, cells):
    if True:
        from spacr.cell_montage import normalised_share, objects_to_show

        classified = cells.groupby("prc").size()
        summed = counts.groupby("prc")["fraction"].sum()
        rows = []
        for _, row in counts.iterrows():
            well = row["prc"]
            n = int(classified.get(well, 0))
            share, factor = normalised_share(
                [summed[well]], float(row["fraction"]))
            rows.append((well, n, share, factor,
                         objects_to_show(n, share),
                         objects_to_show(n, float(row["fraction"]))))
        return pd.DataFrame(rows, columns=[
            "prc", "n", "share", "factor", "x", "x_unnormalised"])


class TestTheCountRule:
    """HOW MANY -- x, driven through the shipped arithmetic."""

    def test_a_well_never_promises_more_cells_than_it_has(self, shares):
        assert not (shares.x > shares.n).any()

    def test_the_normalisation_shows_substantially_more_cells(self, shares):
        """Measured: 226,444 cells against 155,919 -- 45% more."""
        have = shares[shares.n > 0]
        assert have.x.sum() > have.x_unnormalised.sum() * 1.2

    def test_it_also_rescues_wells_that_would_have_shown_nothing(self, shares):
        """Measured: 153 guide-well pairs round to zero, against 270."""
        have = shares[shares.n > 0]
        assert (have.x == 0).sum() < (have.x_unnormalised == 0).sum()

    def test_the_worked_example_from_the_instruction(self):
        """fraction 0.2 among fractions summing to 0.5, 100 cells -> 40."""
        from spacr.cell_montage import normalised_share, objects_to_show

        share, factor = normalised_share([0.5], 0.2)
        assert factor == pytest.approx(2.0)
        assert objects_to_show(100, share) == 40


# --------------------------------------------------------------------------- #
# Instruction 173 -- the attribution                                           #
# --------------------------------------------------------------------------- #

def _effects(counts, cells, alpha=1.0):
    """One effect per guide, ridge at well level with plate blocked out."""
    well = cells.groupby(["prc", "plate"])["pred"].mean().reset_index()
    well["y"] = well.groupby("plate")["pred"].transform(lambda s: s - s.mean())
    wide = counts.pivot_table(index="prc", columns="grna", values="fraction",
                              aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(well["prc"]).fillna(0.0)
    X, y = wide.to_numpy(dtype=float), well["y"].to_numpy(dtype=float)
    n, p = X.shape
    if p >= n:
        beta = X.T @ np.linalg.solve(X @ X.T + alpha * np.eye(n), y)
    else:
        beta = np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)
    return pd.Series(beta, index=wide.columns)


def _attribute(counts, cells, effects, wells=None):
    from spacr.guide_attribution import attribute_well

    per_plate = cells.groupby("plate")["pred"].agg(["median", "std"])
    guides_of = {k: dict(zip(v["grna"], v["fraction"]))
                 for k, v in counts.groupby("prc")}
    out = []
    for well, block in cells.groupby("prc"):
        if wells is not None and well not in wells:
            continue
        fractions = guides_of.get(well)
        if not fractions or len(fractions) < 2:
            continue
        plate = block["plate"].iloc[0]
        calls = attribute_well(
            block["pred"].to_numpy(), fractions,
            {g: float(effects.get(g, 0.0)) for g in fractions},
            centre=float(per_plate.loc[plate, "median"]),
            scale=float(per_plate.loc[plate, "std"]) or 1.0)
        out.append(pd.DataFrame({
            "prc": well, "plate": plate,
            "pathogen_area": block["pathogen_area"].to_numpy(),
            "grna": [c.guide for c in calls],
            "entropy": [c.entropy for c in calls]}))
    return pd.concat(out, ignore_index=True)


@pytest.fixture(scope="module")
def effects(counts, cells):
    return _effects(counts, cells)


class TestThePermutation:
    """"Structure surviving a permutation is structure the method invented.\""""

    def _movement(self, counts, cells, effects, limit=200):
        """Mean total distance from each cell's posterior to its prior."""
        from spacr.guide_attribution import normalise_fractions, posterior

        per_plate = cells.groupby("plate")["pred"].agg(["median", "std"])
        guides_of = {k: dict(zip(v["grna"], v["fraction"]))
                     for k, v in counts.groupby("prc")}
        moved = []
        for well, block in list(cells.groupby("prc"))[:limit]:
            fractions = guides_of.get(well)
            if not fractions or len(fractions) < 2:
                continue
            plate = block["plate"].iloc[0]
            priors = normalise_fractions(fractions)
            r, names = posterior(
                block["pred"].to_numpy(), priors,
                {g: float(effects.get(g, 0.0)) for g in fractions},
                centre=float(per_plate.loc[plate, "median"]),
                scale=float(per_plate.loc[plate, "std"]) or 1.0)
            flat = np.array([priors[g] for g in names])
            moved.append(float(np.abs(r - flat).sum(axis=1).mean()))
        return float(np.mean(moved))

    def test_equal_effects_collapse_the_posterior_onto_the_prior(
            self, counts, cells, effects):
        """By construction, and it came back EXACTLY zero on the screen.

        Not approximately: if a guide cannot be told from any other guide,
        the evidence term is identical for every one of them and the
        posterior is the prior. A solver that moved it at all would be
        inventing a distinction the likelihood does not contain.
        """
        flat = pd.Series(0.0, index=effects.index)
        assert self._movement(counts, cells, flat) == pytest.approx(0.0,
                                                                    abs=1e-12)

    def test_shuffling_the_guide_labels_collapses_it_too(
            self, counts, cells, effects):
        """Measured: 0.0824 real against 0.0065..0.0089 shuffled -- 10x."""
        real = self._movement(counts, cells, effects)
        nulls = []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            nulls.append(self._movement(counts, cells, pd.Series(
                rng.permutation(effects.to_numpy()), index=effects.index)))
        assert real > 3 * max(nulls), (
            f"real movement {real:.6f} against shuffled {nulls} -- the "
            "attribution is not reading the effects it claims to")


@pytest.fixture(scope="module")
def ceiling(counts, cells, effects):
    if True:
        from spacr.guide_attribution import attributable, normalise_fractions

        scale = cells.groupby("plate")["pred"].std().to_dict()
        centre = cells.groupby("plate")["pred"].median().to_dict()
        rows = []
        for well, block in counts.groupby("prc"):
            priors = normalise_fractions(
                dict(zip(block["grna"], block["fraction"])))
            plate = int(block["plate"].iloc[0])
            for guide, prior in priors.items():
                # THE COMPETITION, not a flat stand-in. A ceiling computed
                # against a competitor with no effect is the generous
                # reading, and on this screen the generous reading is the
                # one that says "impossible" about guides the run then calls.
                others = [(float(effects.get(other, 0.0)), weight)
                          for other, weight in priors.items()
                          if other != guide]
                can, best = attributable(
                    float(effects.get(guide, 0.0)),
                    float(scale.get(plate, 1.0)) or 1.0, prior,
                    others=others, centre=float(centre.get(plate, 0.0)))
                rows.append((well, guide, prior, best, can))
        return pd.DataFrame(rows, columns=["prc", "grna", "prior",
                                           "best", "can"])


class TestWhichGuidesCanBeAttributedAtAll:
    """The pre-flight instruction 173 asks for BEFORE anything is assigned."""

    def test_most_guides_can_never_be_called(self, ceiling):
        """Measured: 827 of 6,266 guide-well pairs, or 13.2%.

        This is the number a user needs before they do cell-level work, and
        it is arithmetic rather than sample size -- no amount of extra cells
        moves it.
        """
        assert 0.0 < ceiling.can.mean() < 0.5

    def test_the_ceiling_predicts_what_the_run_actually_does(
            self, counts, cells, effects, ceiling):
        """The pre-flight said 0 controls callable; 0 controls were called.

        Measured on the maintainer's screen, where the whole non-targeting
        series -- 30 guides, `TGGT1_000000_*` -- tops out at a best possible
        posterior of 0.5449 against a threshold of 0.55. It misses by 0.005,
        and it SHOULD: a non-targeting control has no effect to attribute on,
        so a method that confidently attributed cells to one would be
        inventing the structure this whole module exists to avoid.
        """
        callable_now = set(ceiling.loc[ceiling.can, "grna"])
        assigned = _attribute(counts, cells, effects)
        actually = set(assigned.loc[assigned.grna != "ambiguous", "grna"])
        assert actually <= callable_now, (
            "a guide was attributed that the pre-flight said could never be: "
            f"{sorted(actually - callable_now)[:5]}")


class TestTheIndependentMeasurement:
    """`pathogen_area` never entered the attribution, and the two groups
    being compared sit in the SAME well, so plate and well batch are gone."""

    def test_the_score_and_the_area_are_unrelated_within_a_well(self, cells):
        """The check that keeps this from being circular.

        Across the whole screen the two correlate at +0.26, which is a
        BETWEEN-well effect: wells with more recruitment hold bigger
        pathogens. Within a well it is +0.04, so a within-well contrast on
        `pathogen_area` is very nearly independent of the score the
        attribution was made from.
        """
        per_well = []
        for _, block in cells.groupby("prc"):
            if len(block) > 8 and block["pred"].std() and \
                    block["pathogen_area"].std():
                per_well.append(np.corrcoef(
                    block["pred"], block["pathogen_area"])[0, 1])
        assert abs(float(np.nanmedian(per_well))) < 0.15

    def test_a_gene_level_contrast_survives_only_without_permutation(
            self, counts, cells, effects):
        """Measured: 3 genes testable and all three above |t| = 5, against
        ZERO genes even testable under any of five permutations."""
        def genes(effect_map):
            assigned = _attribute(counts, cells, effect_map)
            assigned = assigned[assigned.grna != "ambiguous"].copy()
            assigned["gene"] = assigned.grna.map(
                lambda g: str(g).rsplit("_", 1)[0])
            rows = []
            for _, block in assigned.groupby("prc"):
                if block.gene.nunique() < 2:
                    continue
                total, n = block.pathogen_area.sum(), len(block)
                grouped = block.groupby("gene")["pathogen_area"].agg(
                    ["mean", "size"])
                for gene, r in grouped.iterrows():
                    if r["size"] < 6 or n - r["size"] < 6:
                        continue
                    rest = (total - r["mean"] * r["size"]) / (n - r["size"])
                    rows.append((gene, r["mean"] - rest))
            frame = pd.DataFrame(rows, columns=["gene", "diff"])
            out = []
            for gene, g in frame.groupby("gene"):
                sd = g["diff"].std(ddof=1)
                if len(g) >= 5 and sd and np.isfinite(sd):
                    out.append(abs(g["diff"].mean() / (sd / np.sqrt(len(g)))))
            return out

        real = genes(effects)
        assert real, "no gene reached the contrast at all on the real effects"
        assert max(real) > 3.0
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            null = genes(pd.Series(rng.permutation(effects.to_numpy()),
                                   index=effects.index))
            assert not null or max(null) < max(real), (
                "a permuted screen produced a contrast as strong as the real "
                "one, so the contrast is not evidence of anything")


class TestHeldOutPlates:
    """Estimate the effects on three plates and attribute the fourth."""

    def test_the_attribution_transfers_to_a_plate_it_never_saw(
            self, counts, cells):
        """Measured: 45-59% of cells called on the held-out plate, with
        90-93% of that plate's guides seen during training."""
        for held in PLATES:
            train_counts = counts[counts.plate != held]
            train_cells = cells[cells.plate != held]
            if not len(train_cells) or not len(train_counts):
                pytest.skip("the screen does not carry four plates")
            effects = _effects(train_counts, train_cells)
            wells = set(cells.loc[cells.plate == held, "prc"])
            assigned = _attribute(counts, cells, effects, wells=wells)
            called = float((assigned.grna != "ambiguous").mean())
            assert 0.1 < called < 0.95, (
                f"plate {held} called {called:.1%} of its cells from effects "
                "fitted without it")
            seen = set(train_counts.grna)
            here = counts.loc[counts.plate == held, "grna"]
            assert float(np.mean([g in seen for g in here])) > 0.5
