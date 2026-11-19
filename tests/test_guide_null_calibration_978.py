"""Guide-null inference preserves guide blocks and calibrates across independent screens.

The marginal experiment was preregistered after the earlier conditional
fixed-bank experiment failed; that result remains in the audit. Its Student-t
intervals use independent screen rates, not pooled within-screen Bernoulli trials.
The original two-sided acceptance result is retained beside the no-excess-error
criterion authorized in HOME_TASKS version 41 on 2026-11-19 at 18:15.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr import gene_measurement_compare as G
from spacr.ops_settings import OPS_DEFAULTS
from spacr.power_ops import OPS_POWER_DESIGN


SEED = 978
ALPHA = 0.05
CALIBRATION_SCREENS = 50
NULL_GENES_PER_PROFILE = 10
GUIDE_COUNTS = (2, 4, 8)


def _rng(*stream):
    """Use a predeclared independent stream without searching or tuning seeds."""
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=stream))


def _small_screen():
    """Build unequal two-guide genes and enough control guides for three separate banks."""
    rng = _rng(10)
    blocks, labels, guides = [], [], []
    for number in range(48):
        block = rng.normal(0, 0.6, (1, 4)) + rng.normal(0, 1, (40, 4))
        blocks.append(block)
        labels.extend(["nontargeting"] * len(block))
        guides.extend([f"NT_{number:02d}"] * len(block))
    for gene, profile in (("ASYMMETRIC", (18, 6)), ("BALANCED", (12, 12)),
                          ("SHORT", (8, 4))):
        for number, count in enumerate(profile):
            block = rng.normal(0, 0.6, (1, 4)) + rng.normal(0, 1, (count, 4))
            blocks.append(block)
            labels.extend([gene] * count)
            guides.extend([f"{gene}_{number}"] * count)
    return np.concatenate(blocks), np.asarray(labels), np.asarray(guides)


@pytest.fixture
def small_test():
    """Use a small explicit guide-null fit for deterministic sampling contracts."""
    cells, labels, guides = _small_screen()
    test = G.EnergyTest(cells, labels, guides, mode="guide_null", seed=SEED,
                        n_ref=320, n_null=39, n_holdout=20)
    return test, cells, labels, guides


def test_reference_null_and_holdout_use_disjoint_whole_guides(small_test):
    """Unsampled cells of reference guides must not leak back into either comparison bank."""
    test, cells, labels, guides = small_test
    all_controls = set(guides[labels == "nontargeting"])
    null_guides, holdout_guides = (set(half) for half in test._halves)
    pooled_guides = set(test._pools.index)
    reserved_guides = all_controls - pooled_guides
    reference_rows = test._reference_indices

    assert len(reference_rows) == len(np.unique(reference_rows)) == 320
    assert set(guides[reference_rows]) <= reserved_guides
    assert reserved_guides.isdisjoint(null_guides | holdout_guides)
    assert null_guides.isdisjoint(holdout_guides)
    assert null_guides | holdout_guides == pooled_guides
    assert reserved_guides | pooled_guides == all_controls
    assert len(reserved_guides) >= len(all_controls) // 3
    expected_rest = np.flatnonzero(np.isin(guides, list(pooled_guides)))
    np.testing.assert_array_equal(np.sort(test.rest), expected_rest)
    np.testing.assert_array_equal(test.reference.reference, cells[reference_rows])

    result = test.score(13)
    names = result["table"].gene
    diagnostic_guides = set(names[names.str.startswith("NTguide_")].str[len("NTguide_"):])
    assert diagnostic_guides == holdout_guides
    assert {row["nt_guides"] for row in result["summary"]} == {len(holdout_guides)}


@pytest.mark.parametrize("profile", [(18, 6), (12, 12), (8, 4), (5,)])
def test_pseudo_genes_match_each_guide_count_without_replacement(small_test, profile):
    """Equal total sizes do not permit unequal per-guide sampling profiles to be merged."""
    test, cells, _, guides = small_test
    before = cells.copy()
    for half in test._halves:
        actual = test._pseudo_gene(np.random.default_rng(91), half, profile)
        again = test._pseudo_gene(np.random.default_rng(91), half, profile)
        names, counts = np.unique(guides[actual], return_counts=True)
        assert len(actual) == sum(profile)
        assert len(np.unique(actual)) == len(actual)
        assert set(names) <= set(half)
        assert tuple(sorted(counts.tolist(), reverse=True)) == profile
        assert test._profile(actual) == profile
        np.testing.assert_array_equal(actual, again)
    np.testing.assert_array_equal(cells, before)


@pytest.mark.parametrize("unsupported", ["too_many_cells", "too_many_guides"])
def test_unsupported_profiles_are_refused_instead_of_truncated(small_test, unsupported):
    """A control bank cannot supply a smaller or repeated-guide substitute for a requested gene."""
    test, _, _, _ = small_test
    half = test._halves[0]
    profile = (41,) if unsupported == "too_many_cells" else (2,) * (len(half) + 1)
    with pytest.raises(ValueError, match="(?i)(guide|profile|cells)"):
        test._pseudo_gene(np.random.default_rng(17), half, profile)


def test_shared_prefix_statistics_equal_direct_energy_without_repeating_pairwise_work(
        small_test, monkeypatch):
    """Several exact count vectors reuse the same distance work and retain the energy formula."""
    from scipy.spatial import distance

    test, _, _, _ = small_test
    rng = np.random.default_rng(602)
    names = test._halves[0][:3]
    groups = [rng.permutation(test._pools[name])[:count]
              for name, count in zip(names, (20, 12, 6))]
    profiles = np.array([(20, 12, 6), (18, 6, 4), (12, 12, 6), (8, 6, 4), (3, 2, 2)])
    before = [rows.copy() for rows in groups]
    original = distance.cdist
    calls = []

    def counted(left, right, *args, **kwargs):
        """Count pairwise kernels without changing their values or allocating extra matrices."""
        calls.append((left.shape, right.shape))
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(distance, "cdist", counted)
    single = test._prefix_energies(groups, profiles[:1])
    single_calls = len(calls)
    assert single_calls > 0
    calls.clear()
    actual = test._prefix_energies(groups, profiles)
    assert len(calls) <= single_calls
    expected = np.asarray([
        test.reference.energy(test._rows(np.concatenate(
            [rows[:count] for rows, count in zip(groups, profile)])))
        for profile in profiles])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(single, actual[:1], rtol=1e-12, atol=1e-12)
    for rows, original_rows in zip(groups, before):
        np.testing.assert_array_equal(rows, original_rows)


def test_score_uses_each_actual_profile_after_the_cell_cap(small_test, monkeypatch):
    """The scorer must not replace a short or unequal-guide gene by the cap or median profile."""
    test, _, _, guides = small_test
    seen = []
    original = test._guide_permutation_test

    def capture(index, rng):
        """Record actual scored cell groups without altering their guide-level inference."""
        seen.append(test._profile(index))
        return original(index, rng)

    monkeypatch.setattr(test, "_guide_permutation_test", capture)
    result = test.score(13, subsample_seed=2)
    rng = np.random.default_rng(SEED + 100 + 2)
    expected = []
    for _, indices in test.genes.items():
        chosen = rng.choice(indices, 13, replace=False) if len(indices) > 13 else indices
        _, counts = np.unique(guides[chosen], return_counts=True)
        expected.append(tuple(sorted(counts.tolist(), reverse=True)))
    assert seen[:len(expected)] == expected
    assert sum(expected[-1]) == 12
    table = result["table"].set_index("gene")
    assert table.loc["ASYMMETRIC", "cells"] == table.loc["BALANCED", "cells"] == 13
    assert table.loc["SHORT", "cells"] == 12


def test_scalar_null_remains_an_exact_size_diagnostic_mixture(small_test, monkeypatch):
    """The frozen power caller still receives two arrays from null(n), each drawn at exactly n cells."""
    test, _, _, guides = small_test
    original = test._pseudo_gene
    observed = []

    def capture(rng, half, profile):
        """Check actual sampled row counts while preserving the sampler and its random stream."""
        rows = original(rng, half, profile)
        _, counts = np.unique(guides[rows], return_counts=True)
        observed.append((tuple(profile), tuple(sorted(counts.tolist(), reverse=True)), len(rows)))
        return rows

    monkeypatch.setattr(test, "_pseudo_gene", capture)
    null, holdout = test.null(13)
    assert null.shape == (test.n_null, len(G.ETEST_METRICS))
    assert holdout.shape == (test.n_holdout, len(G.ETEST_METRICS))
    assert np.isfinite(null).all() and np.isfinite(holdout).all()
    assert len(observed) == test.n_null + test.n_holdout
    assert all(sum(requested) == actual_size == 13 and requested == actual
               for requested, actual, actual_size in observed)


def test_unsupported_heldout_guide_keeps_energy_and_is_excluded_from_call_denominator(
        small_test, monkeypatch):
    """An oversized optional diagnostic stays visible without diluting the scored-guide rate."""
    original, cells, labels, guides = small_test
    oversized = original._halves[1][0]
    cells = np.concatenate([cells, np.zeros((20, cells.shape[1]))])
    labels = np.concatenate([labels, np.repeat("nontargeting", 20)])
    guides = np.concatenate([guides, np.repeat(oversized, 20)])
    test = G.EnergyTest(cells, labels, guides, mode="guide_null", seed=SEED,
                        n_ref=320, n_null=39, n_holdout=20)
    assert oversized in test._halves[1]
    assert max(map(len, test._pools.loc[test._halves[0]])) == 40

    original_score = test._guide_permutation_test

    def all_supported_significant(index, rng):
        """Give every supported row a call so an invalid diagnostic cannot silently dilute rates."""
        observed, pvalues, metadata = original_score(index, rng)
        if metadata["permutation_supported"]:
            pvalues = np.full(len(G.ETEST_METRICS), 0.01)
        return observed, pvalues, metadata

    monkeypatch.setattr(test, "_guide_permutation_test", all_supported_significant)
    with pytest.warns(UserWarning, match="(?i)unsupported") as warnings:
        result = test.score(None)
    assert sum("unsupported" in str(warning.message).lower() for warning in warnings) == 1
    table = result["table"].set_index("gene")
    row = table.loc[f"NTguide_{oversized}"]
    assert row["cells"] == 60
    for metric in G.ETEST_METRICS:
        assert np.isfinite(row[f"e_{metric}"])
        assert np.isnan(row[f"p_e_{metric}"])
        assert np.isnan(row[f"q_e_{metric}"])
    for summary in result["summary"]:
        assert summary["nt_guides"] == len(test._halves[1])
        assert summary["nt_guides_scored"] == len(test._halves[1]) - 1
        assert summary["nt_guides_unscored"] == 1
        assert summary["nt_guide_p05_pct"] == 100.0


def test_identical_cells_are_upper_tail_ties_with_pvalue_one():
    """Floating point addition order must not turn identical controls into significant genes."""
    cells, labels, guides = _small_screen()
    cells[:] = np.array([0.1, 0.2, 0.3, 0.4])
    test = G.EnergyTest(cells, labels, guides, mode="guide_null", seed=SEED,
                        n_ref=320, n_null=39, n_holdout=20)
    result = test.score(None)
    table = result["table"]
    genes = ~table.gene.str.startswith("NTguide_")
    for metric in G.ETEST_METRICS:
        assert (table[f"p_e_{metric}"] == 1.0).all()
        assert (table.loc[genes, f"q_e_{metric}"] == 1.0).all()
    for summary in result["summary"]:
        assert summary["nt_pseudo_p05_pct"] == 0.0
        assert summary["nt_guide_p05_pct"] == 0.0


def test_energy_reference_excludes_self_diagonals_before_either_metric(monkeypatch):
    """Even nonzero computed self distances must not enter either unbiased within-set term."""
    rng = _rng(11)
    cells, reference = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
    cross = np.square(cells[:, None, :] - reference[None, :, :]).sum(axis=2)
    within_cells = np.square(cells[:, None, :] - cells[None, :, :]).sum(axis=2)
    within_reference = np.square(reference[:, None, :] - reference[None, :, :]).sum(axis=2)
    expected = []
    for distances in ((cross, within_cells, within_reference),
                      tuple(np.sqrt(matrix) for matrix in (cross, within_cells, within_reference))):
        between, left, right = distances
        expected.append(2 * between.mean() - left.sum() / (7 * 6) - right.sum() / (9 * 8))
    original = G._squared_distances
    contaminated = []

    def add_diagonal_residual(left, right):
        """Amplify self-distance residue so accidental inclusion cannot pass as roundoff."""
        distances = original(left, right)
        if left is right:
            np.fill_diagonal(distances, 7.0)
            contaminated.append(len(left))
        return distances

    monkeypatch.setattr(G, "_squared_distances", add_diagonal_residual)
    actual = G.EnergyReference(reference).energy(cells)
    assert contaminated == [9, 7]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_permutation_ties_exclude_pooled_self_distance_residuals(monkeypatch):
    """Identical cells remain upper-tail ties after contaminated pooled self-distances are removed."""
    cells = np.tile([0.1, 0.2, 0.3, 0.4], (24, 1))
    labels = np.array(["nontargeting"] * 16 + ["TIED"] * 8)
    guides = np.repeat(["NT0", "NT1", "NT2", "NT3", "TIED0", "TIED1"], 4)
    contaminated = []

    def tied_distances_with_diagonal_residual(left, right):
        """Keep identical-cell distances exact while injecting visible self-distance contamination."""
        distances = np.square(left[:, None, :] - right[None, :, :]).sum(axis=2)
        if left is right:
            np.fill_diagonal(distances, 7.0)
            contaminated.append(len(left))
        return distances

    monkeypatch.setattr(G, "_squared_distances", tied_distances_with_diagonal_residual)
    test = G.EnergyTest(cells, labels, guides, mode="permutation", seed=SEED,
                        n_ref=8, permutations=39)
    row = test.score(None)["table"].set_index("gene").loc["TIED"]
    assert contaminated == [8, 8, 16]
    for metric in G.ETEST_METRICS:
        assert row[f"e_{metric}"] == 0.0
        assert row[f"p_e_{metric}"] == 1.0
        assert row[f"q_e_{metric}"] == 1.0


def _discrete_guide_test(query_means, control_means, query_cells=3, n_null=1000):
    """Build constant guide blocks with a separate fixed reference and a known finite control pool."""
    control_count = len(control_means)
    nt_guides = 3 * control_count
    blocks = [np.full((nt_guides * 3, 1), -1.0)]
    blocks.extend(np.full((query_cells, 1), value, dtype=float) for value in query_means)
    cells = np.concatenate(blocks)
    labels = np.array(["nontargeting"] * (nt_guides * 3) + ["TARGET"] * (len(query_means) * query_cells))
    guides = np.concatenate([
        np.repeat([f"NT_{number:04d}" for number in range(nt_guides)], 3),
        np.repeat([f"TARGET_{number:02d}" for number in range(len(query_means))], query_cells)])
    options = dict(mode="guide_null", seed=SEED, n_ref=3, n_null=n_null, n_holdout=0)
    layout = G.EnergyTest(cells, labels, guides, **options)
    assert len(layout._halves[0]) == control_count
    for name, value in zip(layout._halves[0], control_means):
        cells[layout._pools[name]] = value
    return G.EnergyTest(cells, labels, guides, **options)


def test_exact_whole_guide_permutations_have_the_direct_finite_ranks():
    """Every choice of two query guides among four exchangeable blocks has its exact energy rank."""
    from itertools import combinations

    values = np.array([0.0, 1.0, 4.0, 10.0])
    ranks = []
    for chosen in combinations(range(4), 2):
        remaining = [number for number in range(4) if number not in chosen]
        test = _discrete_guide_test(values[list(chosen)], values[remaining])
        index = test.genes["TARGET"]
        query_names = np.unique(test._guides[index])
        groups = [index[test._guides[index] == name] for name in query_names]
        groups.extend(test._pools[name] for name in test._halves[0])
        direct = np.array([
            test.reference.energy(test._rows(np.concatenate([groups[number] for number in assignment])))
            for assignment in combinations(range(4), 2)])
        expected = (direct >= direct[0]).mean(axis=0)
        with pytest.warns(UserWarning, match="(?i)resolution"):
            row = test.score(None)["table"].set_index("gene").loc["TARGET"]
        assert int(row["permutation_space"]) == 6
        assert row["permutation_draws"] == 5
        assert row["permutation_resolution"] == pytest.approx(1 / 6)
        assert row["permutation_exact"] and row["permutation_supported"]
        actual = np.array([row[f"p_e_{metric}"] for metric in G.ETEST_METRICS])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose([row[f"e_{metric}"] for metric in G.ETEST_METRICS],
                                   direct[0], rtol=1e-12, atol=1e-12)
        ranks.append(actual)
    np.testing.assert_allclose(np.sort(ranks, axis=0),
                               np.repeat((np.arange(1, 7) / 6)[:, None], 2, axis=1),
                               rtol=0, atol=1e-15)


def test_single_guide_permutation_keeps_all_controls_and_its_exact_resolution():
    """A large eligible bank must not be silently capped at 128 control guides."""
    test = _discrete_guide_test([100.0], np.zeros(300))
    row = test.score(None)["table"].set_index("gene").loc["TARGET"]
    assert int(row["permutation_space"]) == 301
    assert len(test._guide_permutation_bank((3,))["rows"]) == 300
    assert row["permutation_draws"] == 300
    assert row["permutation_exact"] and row["permutation_supported"]
    assert row["permutation_resolution"] == pytest.approx(1 / 301)
    for metric in G.ETEST_METRICS:
        assert row[f"p_e_{metric}"] == pytest.approx(1 / 301)


@pytest.mark.parametrize("query_count,control_count,space", [(8, 1, 9), (20, 2, 231)])
def test_more_query_guides_than_controls_still_have_finite_permutation_support(
        query_count, control_count, space):
    """NT-only pseudo-gene capacity must not discard valid query-inclusive assignments."""
    import warnings

    test = _discrete_guide_test(np.arange(query_count) + 10.0, np.arange(control_count))
    with warnings.catch_warnings(record=True) as reported:
        warnings.simplefilter("always")
        row = test.score(None)["table"].set_index("gene").loc["TARGET"]
    assert int(row["permutation_space"]) == space
    assert len(test._guide_permutation_bank((3,) * query_count)["rows"]) == control_count
    assert row["permutation_draws"] == space - 1
    assert row["permutation_exact"] and row["permutation_supported"]
    assert row["permutation_resolution"] == pytest.approx(1 / space)
    for metric in G.ETEST_METRICS:
        assert row[f"p_e_{metric}"] == pytest.approx(1 / space)
    if space == 9:
        assert any("resolution" in str(warning.message).lower() for warning in reported)


def test_monte_carlo_assignments_report_add_one_resolution(monkeypatch):
    """Unexhausted spaces use at least 1000 legal whole-guide draws and the add-one denominator."""
    test = _discrete_guide_test(np.arange(4) + 10.0, np.arange(16), n_null=39)
    original = test._guide_assignment_energies
    seen = []

    def inspect(groups, assignments, bank):
        """Verify the actual public scoring path keeps four intact blocks in every Monte Carlo draw."""
        assert len(groups) == 20
        assert all(len(group) == 3 and len(np.unique(test._guides[group])) == 1 for group in groups)
        assert len({test._guides[group[0]] for group in groups}) == len(groups)
        assert assignments.shape == (1001, 4)
        assert ((assignments >= 0) & (assignments < 20)).all()
        assert (np.diff(np.sort(assignments, axis=1), axis=1) > 0).all()
        for query in range(4):
            assert np.any(assignments[1:] == query)
        seen.append(assignments.shape)
        return original(groups, assignments, bank)

    monkeypatch.setattr(test, "_guide_assignment_energies", inspect)
    row = test.score(None)["table"].set_index("gene").loc["TARGET"]
    assert seen == [(1001, 4)]
    assert int(row["permutation_space"]) == 4845
    assert row["permutation_draws"] == 1000
    assert not row["permutation_exact"]
    assert row["permutation_resolution"] == pytest.approx(1 / 1001)
    for metric in G.ETEST_METRICS:
        assert 1 / 1001 <= row[f"p_e_{metric}"] <= 1
        assert 1001 * row[f"p_e_{metric}"] == pytest.approx(round(1001 * row[f"p_e_{metric}"]))


def test_unequal_profiles_permute_intact_query_and_control_guide_blocks(small_test, monkeypatch):
    """Assignments retain each guide's exact cells and include query blocks in alternative selections."""
    test, cells, _, guides = small_test
    index = test.genes["ASYMMETRIC"]
    before = cells.copy()
    pool_names = tuple(test._pools.index)
    original = test._guide_assignment_energies
    query_in_alternatives = []

    def inspect(groups, assignments, bank):
        """Inspect selected guide blocks while preserving their actual energy computation."""
        assert tuple(sorted(map(len, groups[:2]), reverse=True)) == (18, 6)
        assert set(np.concatenate(groups[:2])) == set(index)
        for group in groups:
            assert len(np.unique(group)) == len(group)
            assert len(np.unique(guides[group])) == 1
        assert len({guides[group[0]] for group in groups}) == len(groups)
        for group in groups[2:]:
            assert guides[group[0]] in test._halves[0]
        sizes = np.array([len(group) for group in groups])
        assert (np.sort(sizes[assignments], axis=1) == [6, 18]).all()
        assert (assignments[:, 0] != assignments[:, 1]).all()
        alternatives = np.any(np.sort(assignments, axis=1) != [0, 1], axis=1)
        query_in_alternatives.append(bool(np.any(alternatives & np.any(assignments < 2, axis=1))))
        return original(groups, assignments, bank)

    monkeypatch.setattr(test, "_guide_assignment_energies", inspect)
    test._guide_permutation_test(index, np.random.default_rng(SEED))
    assert any(query_in_alternatives)
    assert tuple(test._pools.index) == pool_names
    np.testing.assert_array_equal(cells, before)


def test_identity_only_required_profile_keeps_energy_and_reports_p_one():
    """A gene without any exchangeable control retains its statistic and explicit resolution limit."""
    test = _discrete_guide_test([100.0], [0.0, 1.0], query_cells=4)
    with pytest.warns(UserWarning, match="(?i)(lack matched|default to one)"):
        row = test.score(None)["table"].set_index("gene").loc["TARGET"]
    assert int(row["permutation_space"]) == 1
    assert row["permutation_draws"] == 0
    assert row["permutation_resolution"] == 1.0
    assert row["permutation_exact"] and not row["permutation_supported"]
    for metric in G.ETEST_METRICS:
        assert np.isfinite(row[f"e_{metric}"]) and row[f"e_{metric}"] > 0
        assert row[f"p_e_{metric}"] == row[f"q_e_{metric}"] == 1.0


def _marginal_screen(screen):
    """Generate one independent fitting screen and fresh query genes from preregistered streams."""
    rng = _rng(20, screen, 0)
    fitting = rng.normal(0, 0.6, (160, 1, 4)) + rng.normal(0, 1, (160, 80, 4))
    blocks = [fitting.reshape(-1, 4)]
    labels = [np.repeat("nontargeting", 160 * 80)]
    guides = [np.repeat([f"FIT_{number:03d}" for number in range(160)], 80)]
    null_names, planted_names = {}, {}
    for count in GUIDE_COUNTS:
        null_rng = _rng(20, screen, 1, count)
        null = (null_rng.normal(0, 0.6, (NULL_GENES_PER_PROFILE, count, 1, 4))
                + null_rng.normal(0, 1, (NULL_GENES_PER_PROFILE, count, 12, 4)))
        names = [f"NULL_K{count:02d}_{number:02d}" for number in range(NULL_GENES_PER_PROFILE)]
        null_names[count] = names
        for name, values in zip(names, null):
            blocks.append(values.reshape(-1, 4))
            labels.append(np.repeat(name, count * 12))
            guides.append(np.repeat([f"{name}_G{number:02d}" for number in range(count)], 12))
        planted_rng = _rng(20, screen, 2, count)
        planted = (planted_rng.normal(0, 0.6, (count, 1, 4))
                   + planted_rng.normal(0, 1, (count, 12, 4)) + 2.0)
        name = f"PLANTED_K{count:02d}"
        planted_names[count] = name
        blocks.append(planted.reshape(-1, 4))
        labels.append(np.repeat(name, count * 12))
        guides.append(np.repeat([f"{name}_G{number:02d}" for number in range(count)], 12))
    return np.concatenate(blocks), np.concatenate(labels), np.concatenate(guides), null_names, planted_names


def _marginal_acceptance(evidence):
    """Evaluate both acceptance criteria on existing evidence without fitting or sampling."""
    alpha = evidence["alpha"]
    primary = evidence["primary_metric"]
    profiles = evidence["profiles"]
    outcomes = {}
    for count in evidence["guides_per_gene"]:
        metrics = profiles[count] if count in profiles else profiles[str(count)]
        selected = metrics[primary]
        lower, upper = selected["student_t_95_ci"]
        planted = all(value < alpha for value in selected["planted_raw_p"])
        original_rate = lower <= alpha <= upper
        revised_rate = lower <= alpha
        outcomes[str(count)] = {
            "preregistered_two_sided_rate_pass": original_rate,
            "preregistered_low_miss": upper < alpha,
            "preregistered_high_miss": lower > alpha,
            "revised_no_excess_rate_pass": revised_rate,
            "all_planted_primary_raw_p_detected": planted,
            "preregistered_pass": original_rate and planted,
            "revised_pass": revised_rate and planted,
        }
    return {
        "revision_authorized_at": "2026-11-19 18:15",
        "revision_authority": "Maintainer authorization in HOME_TASKS.md version 41",
        "preregistered_criterion": "Each primary 95% interval contains alpha and every planted primary raw p < alpha",
        "revised_criterion": "Each primary lower 95% bound <= alpha and every planted primary raw p < alpha",
        "primary_metric": primary,
        "profiles": outcomes,
        "preregistered_pass": all(row["preregistered_pass"] for row in outcomes.values()),
        "revised_pass": all(row["revised_pass"] for row in outcomes.values()),
    }


def test_default_guide_null_calibrates_across_independent_screens_and_finds_planted_effects(
        record_property):
    """Fifty independent screen rates estimate marginal error without treating shared fits as independent."""
    from scipy.stats import t

    mode = OPS_DEFAULTS["ops_etest_mode"]
    primary = OPS_POWER_DESIGN["etest_metric"]
    assert mode == OPS_POWER_DESIGN["etest_mode"] == "guide_null"
    assert primary == "euclidean"
    screens = []
    for screen in range(CALIBRATION_SCREENS):
        cells, labels, guides, null_names, planted_names = _marginal_screen(screen)
        test = G.EnergyTest(cells, labels, guides, mode=mode, seed=SEED)
        assert len(test._reference_indices) == 2000
        assert test.n_null == 1000 and test.n_holdout == 500
        query_ids = set(guides[labels != "nontargeting"])
        assert set(guides[test._reference_indices]).isdisjoint(query_ids)
        assert set(test._pools.index).isdisjoint(query_ids)
        assert set(test._halves[0]).isdisjoint(test._halves[1])
        table = test.score(None)["table"].set_index("gene")
        result = {"screen": screen, "profiles": {}}
        for count in GUIDE_COUNTS:
            names = null_names[count]
            assert (table.loc[names, "cells"] == 12 * count).all()
            metrics = {}
            for metric in G.ETEST_METRICS:
                pvalues = table.loc[names, f"p_e_{metric}"].to_numpy(float)
                assert np.isfinite(pvalues).all() and ((pvalues > 0) & (pvalues <= 1)).all()
                calls = int(np.count_nonzero(pvalues < ALPHA))
                metrics[metric] = {
                    "calls": calls, "rate": calls / NULL_GENES_PER_PROFILE,
                    "planted_raw_p": float(table.loc[planted_names[count], f"p_e_{metric}"]),
                    "planted_bh_q": float(table.loc[planted_names[count], f"q_e_{metric}"]),
                }
            result["profiles"][count] = metrics
        screens.append(result)
        record_property(f"guide_null_marginal_screen_{screen:02d}", json.dumps(result, sort_keys=True))
    critical = float(t.ppf(0.975, CALIBRATION_SCREENS - 1))
    profiles = {}
    for count in GUIDE_COUNTS:
        metrics = {}
        for metric in G.ETEST_METRICS:
            outcomes = [screen["profiles"][count][metric] for screen in screens]
            rates = np.asarray([row["rate"] for row in outcomes])
            mean = float(rates.mean())
            half_width = critical * float(rates.std(ddof=1)) / np.sqrt(CALIBRATION_SCREENS)
            metrics[metric] = {
                "calls": sum(row["calls"] for row in outcomes),
                "draws": CALIBRATION_SCREENS * NULL_GENES_PER_PROFILE,
                "mean_screen_call_rate": mean, "screen_call_rates": rates.tolist(),
                "student_t_95_ci": [mean - half_width, mean + half_width],
                "planted_raw_p": [row["planted_raw_p"] for row in outcomes],
                "planted_bh_q": [row["planted_bh_q"] for row in outcomes],
            }
        profiles[count] = metrics
    evidence = {
        "seed": SEED, "screen_indices": list(range(CALIBRATION_SCREENS)),
        "fitting_stream": "(20, screen, 0)", "null_stream": "(20, screen, 1, K)",
        "planted_stream": "(20, screen, 2, K)", "guides_per_gene": list(GUIDE_COUNTS),
        "fitting_nt_guides": 160, "fitting_cells_per_guide": 80, "query_cells_per_guide": 12,
        "null_genes_per_screen_and_profile": NULL_GENES_PER_PROFILE,
        "guide_sd": 0.6, "cell_sd": 1.0, "dimensions": 4, "planted_shift": 2.0,
        "mode": mode, "energy_test_seed": SEED, "n_ref": 2000, "n_null": 1000, "n_holdout": 500,
        "alpha": ALPHA, "primary_metric": primary,
        "interval_scope": "Approximate Student-t interval across 50 independent screen rates, df=49",
        "planted_criterion": "Every planted primary raw p < alpha; BH q reported separately for 33 genes per screen",
        "profiles": profiles,
    }
    evidence["acceptance"] = _marginal_acceptance(evidence)
    record_property("guide_null_marginal_calibration", json.dumps(evidence, sort_keys=True))
    assert evidence["acceptance"]["revised_pass"], evidence
