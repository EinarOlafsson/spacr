"""Sweeping the DEPENDENT variable: which measurement has genes with an effect.

Instruction 122 part 3, in the maintainer's words:

    "doing a sweep on these screen data of which measurements have genes with
     an effect size. so instead of a paramiter search a serch for which
     measurement has genes with clear effect sizes (one or several)"

THE TRAP THIS FILE EXISTS FOR. A measurement scan is a multiple-testing
problem ACROSS measurements. spaCR measures hundreds of features per object;
scan 500 and some look clear by chance, and they look exactly as convincing as
the real ones, because the per-measurement FDR was computed WITHIN each run
and knows nothing about the other 499.

So :func:`test_a_scan_over_permuted_guides_finds_nothing` is the test that
proves the correction is real, and it is built the way the instruction asks:
the data are the REAL measurements, and only the guide labels are permuted. If
that test ever passes for the wrong reason -- because the scan returns nothing
ever -- :func:`test_a_planted_effect_survives_both_corrections` fails, so the
pair has to hold together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import schema


# --------------------------------------------------------------------------- #
#  A screen shaped like the real one
# --------------------------------------------------------------------------- #

def _wells(rng, *, screens=("kd", "ko"), plates=("plate1", "plate2"),
           genes=24, wells_per_gene=6):
    """One row per well, a gene per well, two screens sharing the library."""
    rows = []
    for screen in screens:
        for plate in plates:
            for gene in range(genes):
                for replicate in range(wells_per_gene // len(plates)):
                    rows.append({
                        schema.SCREEN_KEY: screen,
                        "plateID": plate,
                        "rowID": f"r{(gene % 8) + 1}",
                        "columnID": f"c{replicate + 1}",
                        "gene": f"gene{gene:03d}",
                    })
    frame = pd.DataFrame(rows)
    return frame


def _add_noise_measurements(frame, rng, n, prefix="cell_noise"):
    for index in range(n):
        frame[f"{prefix}_{index}"] = rng.normal(size=len(frame))
    return frame


@pytest.fixture()
def screen_frame():
    """Real-shaped wells with 40 pure-noise measurements and no signal."""
    rng = np.random.default_rng(11)
    frame = _wells(rng)
    # A real screen effect that has nothing to do with the guides: the two
    # screens sit at different levels. Blocking on screenID must absorb it.
    offset = (frame[schema.SCREEN_KEY] == "ko").to_numpy().astype(float) * 3.0
    frame = _add_noise_measurements(frame, rng, 40)
    for column in [c for c in frame.columns if c.startswith("cell_noise")]:
        frame[column] = frame[column] + offset
    return frame


# --------------------------------------------------------------------------- #
#  The scan returns a table, and it is ranked by EFFECT
# --------------------------------------------------------------------------- #

def test_the_scan_says_how_many_measurements_it_looked_at(screen_frame):
    """Without the denominator the reader cannot judge anything else on the
    page: two hits out of three and two out of five hundred are not the same
    result."""
    from spacr.measurement_scan import scan_measurements

    result = scan_measurements(screen_frame)

    assert result.n_measurements_scanned == 40
    assert len(result.rows) == 40
    assert "40" in result.describe()


def test_the_table_is_ranked_by_effect_size_not_by_p_value(screen_frame):
    """"RANK BY EFFECT SIZE, NOT BY P-VALUE." With two screens' worth of
    wells a trivial effect is significant, so the P value is not the ordering
    the maintainer asked for.

    The two orderings coincide in a perfectly balanced design, so the frame
    here is the one a merged screen actually produces: databases written by
    different spaCR versions do not fill the same columns, so a measurement is
    fitted on the wells that have it and the wells differ per measurement.
    That is exactly when a large effect measured on few wells and a tiny one
    measured on many change places.
    """
    from spacr.measurement_scan import scan_measurements

    frame = screen_frame.copy()
    noise = [c for c in frame.columns if c.startswith("cell_noise")]
    for index, column in enumerate(noise):
        holes = 12 * (index % 8)
        if holes:
            frame.loc[frame.index[:holes], column] = np.nan

    table = scan_measurements(frame).frame()

    effects = table["effect_size"].abs().to_numpy()
    assert (np.diff(effects) <= 1e-12).all(), "not sorted by effect size"
    # And it really is a different order from the P value, or the assertion
    # above would be satisfied by an accident.
    assert not table["p_value"].is_monotonic_increasing


def test_every_row_carries_both_numbers(screen_frame):
    """"the report must show both numbers ... A measurement that survives
    within-run FDR and fails across-scan correction is the single most
    important thing this feature can tell a user, and the easiest for it to
    hide.\""""
    from spacr.measurement_scan import scan_measurements

    table = scan_measurements(screen_frame).frame()

    for column in ("within_run_q", "across_scan_q",
                   "survives_within_run", "survives_across_scan"):
        assert column in table.columns, column
    # The across-scan number is never the more permissive of the two.
    assert (table["across_scan_q"] >= table["within_run_q"] - 1e-12).all()


def test_it_names_the_gene_that_carried_the_measurement(screen_frame):
    """"which measurement has genes with clear effect sizes (one or several)"
    -- a measurement with no gene attached to it is not an answer."""
    from spacr.measurement_scan import scan_measurements

    table = scan_measurements(screen_frame).frame()
    assert table["top_gene"].notna().all()
    assert set(table["top_gene"]).issubset(set(screen_frame["gene"]))


# --------------------------------------------------------------------------- #
#  THE PROOF: pure noise survives nothing
# --------------------------------------------------------------------------- #

def test_a_scan_over_permuted_guides_finds_nothing(screen_frame):
    """THE TEST THAT PROVES THE CORRECTION IS REAL.

    Built the way the instruction asks: the measurements are the real ones,
    every correlation and every screen offset intact, and ONLY the guide
    assignment is permuted. Any measurement that survives the across-scan
    correction here is a false discovery by construction.
    """
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(3)
    permuted = screen_frame.copy()
    permuted["gene"] = rng.permutation(permuted["gene"].to_numpy())

    result = scan_measurements(permuted)

    assert result.n_measurements_scanned == 40
    assert result.surviving() == (), [
        row.measurement for row in result.surviving()]


@pytest.mark.slow
def test_over_many_permutations_the_across_scan_rate_is_the_one_promised():
    """The measured version of the claim, and the reason the feature exists.

    Repeat the permuted scan many times over fresh noise and count how often
    it produces a "hit". The two rates are not close to each other:

    * the ACROSS-SCAN correction fires on about 5% of pure-noise scans, which
      is the alpha it promises -- under the global null Benjamini-Hochberg's
      false-discovery rate is the family-wise rate;
    * the WITHIN-RUN correction alone fires on the great majority of the SAME
      scans, because each of the 40 measurements gets its own 5%.

    So a scan reporting only the within-run number would have shown the user a
    convincing measurement in most runs over data with no signal in it at all.
    That gap is the whole statistical content of this instruction.
    """
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(0)
    base = _wells(rng)
    offset = (base[schema.SCREEN_KEY] == "ko").to_numpy().astype(float) * 3.0

    trials = 60
    across = within = 0
    for _ in range(trials):
        frame = base.copy()
        for index in range(40):
            frame[f"cell_noise_{index}"] = rng.normal(size=len(frame)) + offset
        frame["gene"] = rng.permutation(frame["gene"].to_numpy())
        result = scan_measurements(frame)
        across += bool(result.surviving())
        within += any(row.survives_within_run for row in result.rows)

    assert across / trials <= 0.15, f"{across}/{trials} across-scan"
    assert within / trials >= 0.5, f"{within}/{trials} within-run"
    assert within > across * 3


def test_a_planted_effect_survives_both_corrections(screen_frame):
    """The other half of the pair. A scan that never returns anything would
    pass the noise test for the wrong reason."""
    from spacr.measurement_scan import scan_measurements

    frame = screen_frame.copy()
    hit = frame["gene"] == "gene003"
    frame["cell_real_signal"] = (
        np.random.default_rng(5).normal(size=len(frame)) + hit * 4.0)

    result = scan_measurements(frame)
    survivors = [row.measurement for row in result.surviving()]

    assert survivors == ["cell_real_signal"], survivors
    top = result.frame().iloc[0]
    assert top["measurement"] == "cell_real_signal"
    assert top["top_gene"] == "gene003"
    assert top["effect_size"] > 1.0


def test_a_measurement_can_survive_within_run_and_fail_across_the_scan(
        screen_frame):
    """The single most important thing this feature can say.

    With 40 pure-noise measurements a few will clear a within-run FDR by
    chance. None of them may clear the across-scan correction, and the table
    has to show BOTH facts on the same row rather than reporting the flattering
    one.
    """
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(3)
    permuted = screen_frame.copy()
    permuted["gene"] = rng.permutation(permuted["gene"].to_numpy())

    table = scan_measurements(permuted, alpha=0.2).frame()
    misleading = table[table["survives_within_run"]
                       & ~table["survives_across_scan"]]

    assert not misleading.empty, (
        "the fixture must produce at least one within-run hit, or this test "
        "proves nothing")
    assert not table["survives_across_scan"].any()


# --------------------------------------------------------------------------- #
#  Correlated measurements, and the correction that is not too harsh
# --------------------------------------------------------------------------- #

def test_the_effective_number_of_tests_is_below_the_column_count():
    """"area, perimeter, and equivalent-diameter are one thing measured three
    ways", so the effective number of independent tests is far below the
    column count and a naive Bonferroni over 500 correlated columns is too
    harsh in the other direction."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(19)
    frame = _wells(rng)
    # Twelve columns that are really four things measured three ways.
    for thing in range(4):
        base = rng.normal(size=len(frame))
        for spelling in range(3):
            frame[f"cell_thing{thing}_{spelling}"] = (
                base + rng.normal(scale=0.01, size=len(frame)))

    result = scan_measurements(frame)

    assert result.n_measurements_scanned == 12
    # Twelve columns, four things. The Li & Ji estimator is deliberately not
    # asserted to hit 4 exactly: its f(x) = I(x>=1) + (x - floor(x)) steps at
    # every integer eigenvalue, so a block whose eigenvalue is 2.49 rather
    # than 3.00 -- which is what four finite samples of 288 wells give -- adds
    # 1.49 rather than 1. What has to hold is the direction and the order of
    # magnitude: far below the column count, and near the real dimensionality.
    assert 4.0 <= result.effective_n_tests <= 7.0, result.effective_n_tests
    assert result.effective_n_tests < result.n_measurements_scanned


def test_the_correction_used_is_stated_not_assumed():
    """"Say which correction is used and why." A report that does not name
    its correction cannot be checked by the person reading it."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(23)
    frame = _add_noise_measurements(_wells(rng), rng, 5)
    result = scan_measurements(frame)

    text = result.describe()
    assert result.within_run_method == "fdr_bh"
    assert result.across_scan_method == "fdr_bh"
    assert "fdr_bh" in text or "Benjamini" in text
    assert "effective" in text.lower()


def test_there_is_not_a_second_correction_implementation():
    """spacr/multiple_testing.py already owns every correction spaCR offers.
    A private Benjamini-Hochberg here is how a GUI dropdown and an analysis
    end up meaning different things by one word."""
    from spacr import measurement_scan, multiple_testing

    assert measurement_scan.adjust_p_values is multiple_testing.adjust_p_values


def test_an_effective_bonferroni_is_available_and_is_kinder_than_the_naive_one():
    """The other direction of the same trap: Bonferroni over correlated
    columns overcorrects, so the scan can divide by the EFFECTIVE number of
    tests instead of the column count."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(29)
    frame = _wells(rng)
    for thing in range(3):
        base = rng.normal(size=len(frame))
        for spelling in range(4):
            frame[f"cell_thing{thing}_{spelling}"] = (
                base + rng.normal(scale=0.01, size=len(frame)))

    effective = scan_measurements(frame, across_scan_method="bonferroni_effective")
    naive = scan_measurements(frame, across_scan_method="bonferroni")

    assert effective.effective_n_tests < 12
    assert (effective.frame()["across_scan_q"].to_numpy()
            <= naive.frame()["across_scan_q"].to_numpy() + 1e-12).all()


# --------------------------------------------------------------------------- #
#  The screen is a blocking factor, which is what part 1 was for
# --------------------------------------------------------------------------- #

def test_the_screen_is_blocked_on_by_default(screen_frame):
    """"screenID is available as a blocking factor / fixed effect in the
    regression, not merely as a label."

    The fixture puts a large offset between the two screens. Blocked, it is
    absorbed; unblocked it inflates every residual and buries the signal.
    """
    from spacr.measurement_scan import scan_measurements

    frame = screen_frame.copy()
    hit = frame["gene"] == "gene003"
    frame["cell_real_signal"] = (
        np.random.default_rng(5).normal(size=len(frame)) + hit * 1.2)

    blocked = scan_measurements(frame)
    unblocked = scan_measurements(frame, block_columns=())

    assert schema.SCREEN_KEY in blocked.block_columns
    assert unblocked.block_columns == ()

    def _effect(result):
        row, = [r for r in result.rows if r.measurement == "cell_real_signal"]
        return row.effect_size

    assert _effect(blocked) > _effect(unblocked)


def test_a_single_screen_project_scans_with_no_screen_column_at_all():
    """A project that has never heard of screenID must still scan. The
    blocking factor is simply absent, not an error."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(31)
    frame = _wells(rng, screens=("only",)).drop(columns=[schema.SCREEN_KEY])
    frame = _add_noise_measurements(frame, rng, 4)

    result = scan_measurements(frame)
    assert result.n_measurements_scanned == 4
    assert result.block_columns == ()


def test_a_constant_block_is_dropped_rather_than_making_the_fit_singular():
    """One screen means the screen column carries no information. Including
    it would make the design rank-deficient; excluding it is the same model."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(37)
    frame = _wells(rng, screens=("solo",))
    frame = _add_noise_measurements(frame, rng, 4)

    result = scan_measurements(frame)
    assert result.n_measurements_scanned == 4
    assert all(np.isfinite(row.p_value) for row in result.rows)


# --------------------------------------------------------------------------- #
#  What the scan refuses to guess at
# --------------------------------------------------------------------------- #

def test_metadata_columns_are_not_scanned_as_measurements():
    """rowID and columnID are the design, not the dependent variable.
    Regressing the guides on the row index is not an analysis."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(41)
    frame = _add_noise_measurements(_wells(rng), rng, 3)
    frame["well_count"] = 1

    scanned = {row.measurement for row in scan_measurements(frame).rows}
    assert scanned == {"cell_noise_0", "cell_noise_1", "cell_noise_2"}


def test_a_constant_measurement_is_skipped_and_said_so():
    """A column with no variance has no effect size. Reporting it as zero
    would rank it alongside the measurements that were genuinely flat."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(43)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    frame["cell_flat"] = 7.0

    result = scan_measurements(frame)
    assert result.n_measurements_scanned == 2
    assert "cell_flat" in result.skipped
    assert "varian" in result.skipped["cell_flat"]
    assert "cell_flat" in result.describe()


def test_a_measurement_with_missing_wells_is_fitted_on_the_wells_it_has():
    """Databases from different spaCR versions do not measure the same
    things, so a merged frame has holes. Dropping the whole measurement would
    throw away the comparison the user came for."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(47)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    frame.loc[frame.index[:20], "cell_noise_0"] = np.nan

    result = scan_measurements(frame)
    row, = [r for r in result.rows if r.measurement == "cell_noise_0"]
    assert row.n_wells == len(frame) - 20
    assert np.isfinite(row.p_value)


def test_a_frame_with_no_genes_is_refused_not_silently_empty():
    """A scan with no independent variable is not an empty result, it is a
    caller that has not passed the screen in."""
    from spacr.measurement_scan import ScanRefused, scan_measurements

    rng = np.random.default_rng(53)
    frame = _add_noise_measurements(_wells(rng), rng, 2).drop(columns=["gene"])

    with pytest.raises(ScanRefused) as caught:
        scan_measurements(frame)
    assert "gene" in str(caught.value)


def test_one_gene_is_refused_because_there_is_nothing_to_contrast():
    """Every well the same gene means the gene term is the intercept."""
    from spacr.measurement_scan import ScanRefused, scan_measurements

    rng = np.random.default_rng(59)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    frame["gene"] = "gene000"

    with pytest.raises(ScanRefused):
        scan_measurements(frame)


def test_control_genes_become_the_baseline_the_effects_are_measured_from():
    """A screen's effect size is "against the controls", not "against
    whichever gene sorted first".

    The library here moves together and the control does not -- a plate
    effect, a transfection effect, the ordinary shape of a screen. Told which
    gene is the control, the scan reports the library as the thing that moved.
    Not told, it reports THE CONTROL as the hit, which is the wrong answer
    with a perfectly convincing effect size on it.
    """
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(61)
    frame = _wells(rng)
    frame["cell_signal"] = rng.normal(size=len(frame))
    frame.loc[frame["gene"] != "gene023", "cell_signal"] += 5.0

    told, = scan_measurements(frame, control_genes=["gene023"]).rows
    assert told.top_gene != "gene023"
    assert told.effect_size > 1.0

    not_told, = scan_measurements(frame).rows
    assert not_told.top_gene == "gene023"


def test_naming_a_control_that_is_not_in_the_frame_falls_back_quietly():
    """A settings CSV lists the lab's standard controls; this screen did not
    plate them. That is not a reason to refuse the scan."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(67)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    result = scan_measurements(frame, control_genes=["not_plated_here"])
    assert result.n_measurements_scanned == 2


def test_every_gene_being_a_control_is_refused():
    """A baseline with nothing to compare against it is not a screen."""
    from spacr.measurement_scan import ScanRefused, scan_measurements

    rng = np.random.default_rng(71)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    controls = sorted(frame["gene"].unique())

    with pytest.raises(ScanRefused) as caught:
        scan_measurements(frame, control_genes=controls)
    assert "control" in str(caught.value)


def test_asking_for_a_measurement_that_is_not_there_is_refused():
    """Silently scanning three of the four columns a caller named would
    report a smaller family than the one they meant to test."""
    from spacr.measurement_scan import ScanRefused, scan_measurements

    rng = np.random.default_rng(73)
    frame = _add_noise_measurements(_wells(rng), rng, 2)

    named = scan_measurements(frame, measurements=["cell_noise_0"])
    assert named.n_measurements_scanned == 1

    with pytest.raises(ScanRefused) as caught:
        scan_measurements(frame, measurements=["cell_noise_0", "cell_ghost"])
    assert "cell_ghost" in str(caught.value)


def test_a_frame_with_nothing_numeric_in_it_is_refused():
    """Identity columns are excluded on purpose, so a frame of nothing but
    identity has no response -- and an empty table would read as "no
    measurement had an effect"."""
    from spacr.measurement_scan import ScanRefused, scan_measurements

    rng = np.random.default_rng(79)
    with pytest.raises(ScanRefused) as caught:
        scan_measurements(_wells(rng))
    assert "numeric" in str(caught.value)


def test_a_text_column_is_not_a_measurement():
    """A user's annotation column is not a dependent variable, and it is not
    an error either -- it is simply not something to regress."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(83)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    frame["annotation"] = "looked fine"

    scanned = {row.measurement for row in scan_measurements(frame).rows}
    assert scanned == {"cell_noise_0", "cell_noise_1"}


def test_a_measurement_present_in_a_handful_of_wells_is_skipped_and_named():
    """One database in the merge measured it, and only for a few wells. The
    scan says so instead of reporting an effect fitted on nothing."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(89)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    frame["cell_barely_there"] = np.nan
    frame.loc[frame.index[:2], "cell_barely_there"] = [1.0, 2.0]
    # Four wells, four different genes: enough wells to try, not enough to
    # leave a single degree of freedom once the design has taken its share.
    frame["cell_thin"] = np.nan
    sparse = frame.index[[0, 3, 6, 9]]
    frame.loc[sparse, "cell_thin"] = rng.normal(size=4)
    assert frame.loc[sparse, "gene"].nunique() == 4, "fixture assumption"

    result = scan_measurements(frame)

    assert result.skipped["cell_barely_there"] == "too few wells with a value"
    assert result.skipped["cell_thin"] == "not enough wells left for the design"
    assert result.n_measurements_scanned == 2


def test_a_measurement_the_design_explains_exactly_has_no_estimable_effect():
    """A column that IS the gene assignment, with no noise at all, leaves no
    residual to measure an effect against. Reporting an infinite effect size
    would put it at the top of the table for ever."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(97)
    frame = _add_noise_measurements(_wells(rng), rng, 2)
    frame["cell_is_the_design"] = [
        float(gene[-3:]) for gene in frame["gene"]]

    result = scan_measurements(frame)
    assert result.skipped["cell_is_the_design"] == "no estimable gene effect"
    assert result.n_measurements_scanned == 2


def test_a_scan_where_nothing_could_be_measured_is_an_empty_table_not_a_crash():
    """Every measurement skipped is a legitimate answer and has to come back
    in the same shape as any other, or every caller branches."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(101)
    frame = _wells(rng)
    frame["cell_flat_a"] = 1.0
    frame["cell_flat_b"] = 2.0

    result = scan_measurements(frame)
    assert result.rows == ()
    assert result.n_measurements_scanned == 0
    assert result.surviving() == ()
    assert set(result.skipped) == {"cell_flat_a", "cell_flat_b"}
    assert list(result.frame().columns)[:2] == ["measurement", "n_wells"]
    assert result.frame().empty
    assert np.isnan(result.effective_n_tests)
    assert "0 measurement" in result.describe()


def test_a_scan_where_every_fit_fails_still_returns_a_table():
    """The other empty path: the columns varied, so they were kept, and then
    none of them could be fitted."""
    from spacr.measurement_scan import scan_measurements

    rng = np.random.default_rng(103)
    frame = _wells(rng)
    frame["cell_is_the_design"] = [float(g[-3:]) for g in frame["gene"]]

    result = scan_measurements(frame)
    assert result.rows == ()
    assert result.skipped == {"cell_is_the_design": "no estimable gene effect"}


# --------------------------------------------------------------------------- #
#  The two statistics, on their own
# --------------------------------------------------------------------------- #

def test_simes_over_nothing_is_not_a_p_value():
    """A family where every test failed is not a family that passed."""
    from spacr.measurement_scan import simes_p_value

    assert np.isnan(simes_p_value([]))
    assert np.isnan(simes_p_value([np.nan, np.nan]))
    # One test: Simes is that test.
    assert simes_p_value([0.03]) == pytest.approx(0.03)
    # Uniform P values give back roughly the smallest times the family size.
    assert simes_p_value([0.5, 0.5, 0.001]) == pytest.approx(0.003)
    # It never exceeds 1, and it does not need clipping to manage it: the
    # last term is m * p_(m) / m, which is the largest P value itself.
    assert simes_p_value([0.9, 0.9, 0.9]) == pytest.approx(0.9)
    assert simes_p_value([0.99, 0.995, 1.0]) <= 1.0


def test_the_effective_test_count_answers_the_degenerate_shapes():
    """Called on the way to a report, so it must never be the thing that
    raises."""
    from spacr.measurement_scan import effective_number_of_tests

    assert np.isnan(effective_number_of_tests(np.empty((5, 0))))
    assert np.isnan(effective_number_of_tests(np.arange(5)))
    assert effective_number_of_tests(np.arange(5.0).reshape(5, 1)) == 1.0
    # A constant column makes its correlations undefined; it is dropped, and
    # one usable column left is one test.
    # ONE constant column must not take its neighbours down with it: its
    # correlations are undefined, so it is dropped and the rest still count.
    rng = np.random.default_rng(109)
    poisoned = np.column_stack([rng.normal(size=200), np.ones(200)])
    assert effective_number_of_tests(poisoned) == 1.0
    assert effective_number_of_tests(
        np.column_stack([rng.normal(size=(200, 3)), np.ones(200)])
    ) == pytest.approx(3, abs=1.0)
    # Two columns measured on wells that never overlap have no correlation to
    # read; counting them as separate tests is the conservative answer.
    disjoint = np.full((10, 2), np.nan)
    disjoint[:5, 0] = rng.normal(size=5)
    disjoint[5:, 1] = rng.normal(size=5)
    assert effective_number_of_tests(disjoint) == pytest.approx(2, abs=0.01)
    # Nothing usable at all: every column constant.
    assert np.isnan(effective_number_of_tests(np.ones((6, 3))))
    # Independent columns are their own count.
    rng = np.random.default_rng(107)
    assert effective_number_of_tests(
        rng.normal(size=(400, 4))) == pytest.approx(4, abs=1.0)
