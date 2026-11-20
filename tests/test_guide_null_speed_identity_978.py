"""Compare guide-distance optimizations with the immutable pre-U distance method.

The oracle freezes EnergyTest._guide_pair_sums from a8e413fbf9 and the
pre-thread _guide_permutation_test and _score_guides from the vector-v4
checkpoint. Those bodies preserve the S benchmark sampling and scoring
order independently of the new planner and workers. The oracle deliberately
avoids new distance caches. These are deterministic identity checks, not
another calibration run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as G


SEED = 978


class _BaselineEnergyTest(G.EnergyTest):
    """Freeze pre-thread scoring, RNG consumption and the original distance kernel."""

    def _guide_permutation_test(self, index, rng):
        """Test one exact post-cap profile by query-inclusive whole-guide permutations."""
        from spacr.qt.i18n import tr

        codes, names = pd.factorize(self._guides[index], sort=True)
        query = [np.asarray(index)[codes == number] for number in range(len(names))]
        query.sort(key=len, reverse=True)
        profile = tuple(len(rows) for rows in query)
        bank = self._guide_permutation_bank(profile)
        if not bank["supported"]:
            observed = self._prefix_energies([index], [[len(index)]])[0]
            return observed, np.ones(2), {"permutation_space": "1", "permutation_draws": 0,
                "permutation_resolution": 1.0, "permutation_exact": True,
                "permutation_supported": False, "matched_controls": 0, "stratum_counts": ()}
        strata = tuple((np.r_[np.flatnonzero(np.asarray(profile) == item["cells"]),
                              np.asarray(item["nt_indices"], dtype=np.int64) + len(query)], item["query_count"])
                       for item in bank["strata"])
        assignments, metadata = self._guide_permutation_assignments(strata, rng)
        selections = np.vstack([np.arange(len(query)), assignments])
        values, scales = self._guide_assignment_energies(tuple(query) + bank["rows"], selections, bank)
        if not np.isfinite(values).all():
            raise ValueError(tr("Guide-null statistics must be finite; repair the embedding before scoring."))
        operations = 4 * (len(self.reference.reference) + len(index) + len(query) ** 2
                          + self.reference.reference.shape[1]) + 8
        error = operations * np.finfo(np.float64).eps
        gamma = error / (1 - error)
        tolerance = gamma * (scales[1:] + scales[0])
        pvalues = (1 + (values[1:] >= values[0] - tolerance).sum(axis=0)) / (1 + len(assignments))
        metadata["matched_controls"] = len(bank["rows"])
        metadata["stratum_counts"] = tuple((item["cells"], item["query_count"], len(item["nt_indices"]))
                                            for item in bank["strata"])
        return values[0], pvalues, metadata

    def _score_guides(self, n, subsample_seed):
        """Score whole-guide permutations and retain finite-denominator control diagnostics."""
        import warnings
        from spacr.qt.i18n import tr

        if n is not None and (not isinstance(n, (int, np.integer)) or n < 2):
            raise ValueError(tr("Guide-null scoring needs an integer cell cap of at least two, or None for all cells."))
        draw = np.random.default_rng(self.seed + 100 + int(subsample_seed))
        heldout_names = set(self._halves[1])
        items = [(name, index, True) for name, index in self.genes.items()]
        items.extend((f"NTguide_{name}", index, False) for name, index in self.control_guides.items()
                     if name in heldout_names)
        permute = np.random.default_rng(self.seed + 300 + int(subsample_seed))
        rows, row_pvalues, metadata, gene_profiles, gene_flags = [], [], [], [], []
        for name, index, is_gene in items:
            if n is not None and len(index) > n:
                index = draw.choice(index, n, replace=False)
            if len(index) < 2 or is_gene and len(index) < self.min_cells:
                continue
            if is_gene:
                gene_profiles.append(self._profile(index))
            values, pvalues, details = self._guide_permutation_test(index, permute)
            if not is_gene and not details["permutation_supported"]:
                pvalues = np.full(2, np.nan)
            rows.append((name, len(index), values[0], values[1]))
            row_pvalues.append(pvalues)
            metadata.append(details)
            gene_flags.append(is_gene)
        if not gene_profiles:
            raise ValueError(tr("No gene has enough cells for guide-null scoring; provide genes meeting min_cells."))
        is_gene = np.asarray(gene_flags, dtype=bool)
        supported = np.array([item["permutation_supported"] for item in metadata], dtype=bool)
        resolutions = np.array([item["permutation_resolution"] for item in metadata])
        unsupported_genes = int((is_gene & ~supported).sum())
        unscored = int((~is_gene & ~supported).sum())
        low_resolution = supported & (resolutions >= 0.05)
        if unsupported_genes:
            warnings.warn(tr(
                "P-values default to one for {count} genes whose exact guide profiles lack matched NT controls; "
                "provide more eligible control guides.").format(count=unsupported_genes),
                UserWarning, stacklevel=2)
        if low_resolution.any():
            warnings.warn(tr(
                "Whole-guide permutation resolution cannot attain p < 0.05 for {count} scored rows "
                "(minimum attainable p {minimum:.6g} to {maximum:.6g}); see permutation_resolution "
                "and provide more independent matched NT guides.")
                .format(count=int(low_resolution.sum()), minimum=float(resolutions[low_resolution].min()),
                        maximum=float(resolutions[low_resolution].max())),
                UserWarning, stacklevel=2)
        if unscored:
            warnings.warn(tr(
                "P-values are unavailable for {count} held-out NT guides with unsupported cell counts; "
                "their statistics remain visible. Provide matched control guides for those diagnostics.")
                .format(count=unscored), UserWarning, stacklevel=2)
        draw = np.random.default_rng(self.seed + 200 + int(subsample_seed))
        held_pvalues = np.full((self.n_holdout, 2), np.nan)
        held_low_resolution = 0
        for number in range(self.n_holdout):
            profile = gene_profiles[int(draw.integers(len(gene_profiles)))]
            try:
                self._guide_eligibility(profile, self._halves[1])
            except ValueError:
                continue
            index = self._pseudo_gene(draw, self._halves[1], profile)
            _, pvalues, details = self._guide_permutation_test(index, permute)
            if details["permutation_supported"]:
                held_pvalues[number] = pvalues
                held_low_resolution += details["permutation_resolution"] >= 0.05
        held_supported = np.isfinite(held_pvalues[:, 0])
        held_unscored = int((~held_supported).sum())
        if held_unscored:
            warnings.warn(tr(
                "P-values are unavailable for {count} held-out pseudo-genes with unsupported guide profiles; "
                "provide matched held-out controls.").format(count=held_unscored),
                UserWarning, stacklevel=2)
        row_pvalues = np.asarray(row_pvalues)
        table = pd.DataFrame(rows, columns=["gene", "cells", "e_sqeuclidean", "e_euclidean"])
        for key in ("permutation_space", "permutation_draws", "permutation_resolution",
                    "permutation_exact", "permutation_supported"):
            table[key] = [item[key] for item in metadata]
        null_cells = int(np.median(table.loc[is_gene, "cells"]))
        summary = []
        for number, metric in enumerate(G.ETEST_METRICS):
            column = f"e_{metric}"
            table[f"p_{column}"] = row_pvalues[:, number]
            table.loc[is_gene, f"q_{column}"] = G._bh(table.loc[is_gene, f"p_{column}"])
            q = table.loc[is_gene, f"q_{column}"]
            held = (100 * float((held_pvalues[held_supported, number] < 0.05).mean())
                    if held_supported.any() else float("nan"))
            nt_p = row_pvalues[~is_gene, number]
            nt_p = nt_p[np.isfinite(nt_p)]
            summary.append({
                "n": "all" if n is None else str(n), "seed": int(subsample_seed), "distance": metric,
                "genes": int(is_gene.sum()), "hits_q05": int((q < 0.05).sum()),
                "hit_pct": 100 * float((q < 0.05).mean()), "nt_pseudo_p05_pct": held,
                "nt_guide_p05_pct": 100 * float((nt_p < 0.05).mean()) if len(nt_p) else float("nan"),
                "nt_guides": int((~is_gene).sum()),
                "nt_guides_scored": len(nt_p), "nt_guides_unscored": unscored,
                "genes_unsupported": unsupported_genes,
                "genes_low_resolution": int((is_gene & low_resolution).sum()),
                "nt_guides_low_resolution": int((~is_gene & low_resolution).sum()),
                "nt_pseudo_scored": int(held_supported.sum()), "nt_pseudo_unscored": held_unscored,
                "nt_pseudo_low_resolution": int(held_low_resolution),
                "genes_p05_pct": 100 * float((table.loc[is_gene, f"p_{column}"] < 0.05).mean()),
                "null_cells": null_cells})
        return {"table": table, "summary": summary}

    def _guide_pair_sums(self, groups, pairs, bank, query_count, local):
        """Compute only requested guide-pair sums using bounded direct subtraction.

        Pairs with identical block sizes share vectorized distance work.
        Large individual rectangles use the bounded cdist prefix reducer.
        NT-only sums can be reused across genes; conservative per-entry
        accounting includes dictionary, key, scalar and array overhead.
        """
        from spacr.qt.i18n import tr

        total = len(groups)
        result = np.empty((len(pairs), 2), dtype=np.float64)
        missing = {}
        for number, encoded in enumerate(pairs):
            left, right = divmod(int(encoded), total)
            nt_key = (left - query_count, right - query_count)
            value = bank["pairs"].get(nt_key) if left >= query_count else local.get(int(encoded))
            if value is None:
                missing.setdefault((len(groups[left]), len(groups[right])), []).append((number, left, right))
            else:
                result[number] = value
        dimension = self.reference.reference.shape[1]
        for (left_count, right_count), requests in missing.items():
            width = max(1, 1048576 // max(1, left_count * right_count * dimension))
            for start in range(0, len(requests), width):
                selected = requests[start:start + width]
                if left_count * right_count * dimension > 1048576:
                    values = np.array([self._prefix_distance_sums(
                        groups[left], groups[right], np.array([left_count]), np.array([right_count]))[0]
                        for _, left, right in selected])
                else:
                    left_rows = np.stack([groups[left] for _, left, _ in selected])
                    right_rows = np.stack([groups[right] for _, _, right in selected])
                    left = self._rows(left_rows)
                    right = self._rows(right_rows)
                    differences = left[:, :, None, :] - right[:, None, :, :]
                    squared = np.einsum("bijd,bijd->bij", differences, differences)
                    if not np.isfinite(squared).all():
                        raise ValueError(tr("Guide-null distances must be finite; repair nonfinite or overflowing embedding values."))
                    values = np.empty((len(selected), 2))
                    values[:, 0] = squared.sum(axis=(1, 2))
                    np.sqrt(squared, out=squared)
                    values[:, 1] = squared.sum(axis=(1, 2))
                for (number, left, right), value in zip(selected, values):
                    result[number] = value
                    retained = (float(value[0]), float(value[1]))
                    if left >= query_count:
                        if (bank["cached"] and len(bank["pairs"]) < 100000
                                and self._permutation_bank_bytes + 320 <= 64 * 1024 ** 2):
                            bank["pairs"][(left - query_count, right - query_count)] = retained
                            bank["bytes"] += 320
                            self._permutation_bank_bytes += 320
                    elif len(local) < 100000:
                        local[left * total + right] = retained
        return result


def _heterogeneous_screen(dimensions=104, include_eviction_profiles=False):
    """Use unequal NT capacities and several distinct query profiles in fixed float32 data."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(30,)))
    blocks, labels, guides = [], [], []
    for number in range(48):
        count = (9, 12, 18, 24)[number % 4]
        block = rng.normal(0, 0.6, (1, dimensions)) + rng.normal(0, 1, (count, dimensions))
        blocks.append(block)
        labels.extend(["nontargeting"] * count)
        guides.extend([f"NT_{number:02d}"] * count)
    profiles = [
        ("A_EQUAL_TWO", (6, 6), 0.0),
        ("B_EQUAL_FOUR", (6, 6, 6, 6), 0.0),
        ("C_UNEQUAL", (18, 6), 0.0),
        ("D_UNEQUAL_FOUR", (12, 12, 6, 6), 0.0),
        ("E_SHIFTED", (6, 6, 6, 6), 2.0),
        ("F_FIXED_STRATUM", (30, 6), 0.0),
        ("G_UNSUPPORTED", (30, 30), 0.0),
    ]
    if include_eviction_profiles:
        profiles.extend((f"PROFILE_{count:02d}", (count, 2), 0.0) for count in range(3, 13))
    for gene, profile, shift in profiles:
        for number, count in enumerate(profile):
            block = (rng.normal(0, 0.6, (1, dimensions))
                     + rng.normal(0, 1, (count, dimensions)) + shift)
            blocks.append(block)
            labels.extend([gene] * count)
            guides.extend([f"{gene}_{number}"] * count)
    return np.concatenate(blocks).astype(np.float32), np.asarray(labels), np.asarray(guides)


def _fit_pair(screen, **overrides):
    """Fit independent current and frozen kernels over the same immutable input arrays."""
    cells, labels, guides = screen
    cells.flags.writeable = False
    options = dict(mode="guide_null", seed=SEED, n_ref=64, n_null=39,
                   n_holdout=4, min_guide_cells=3)
    options.update(overrides)
    return (G.EnergyTest(cells, labels, guides, **options),
            _BaselineEnergyTest(cells, labels, guides, **options))


def _score_with_warnings(test, cap=None, subsample_seed=0):
    """Capture diagnostic warnings as part of the observable scoring result."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = test.score(cap, subsample_seed=subsample_seed)
    return result, tuple((item.category, str(item.message)) for item in caught)


def _assert_score_identity(actual, expected):
    """Require exact p/q, metadata, ordering and diagnostics, with tight energy agreement."""
    energies = [f"e_{metric}" for metric in G.ETEST_METRICS]
    pd.testing.assert_frame_equal(actual["table"].drop(columns=energies),
                                  expected["table"].drop(columns=energies),
                                  check_exact=True)
    np.testing.assert_allclose(actual["table"][energies].to_numpy(),
                               expected["table"][energies].to_numpy(),
                               rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(pd.DataFrame(actual["summary"]),
                                  pd.DataFrame(expected["summary"]), check_exact=True)


def _assert_permutation_identity(actual, expected):
    """Keep both raw p-values and finite-space metadata exact for a single query."""
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


@pytest.mark.parametrize("workers", [1, 4], ids=["forced-serial", "forced-parallel"])
def test_fixed_seed_scores_match_frozen_kernel_across_profiles_caps_and_reuse(monkeypatch, workers):
    """Serial and parallel scoring preserve the frozen RNG, rows, warnings and repeated results."""
    screen = _heterogeneous_screen()
    before = screen[0].copy()
    actual, baseline = _fit_pair(screen)

    def selected_workers(job_count):
        """Exercise the requested path independently of machine load and BLAS configuration."""
        return workers

    monkeypatch.setattr(actual, "_guide_worker_count", selected_workers)
    first = None
    for cap, subsample_seed in ((None, 0), (13, 5), (None, 0)):
        result, reported = _score_with_warnings(actual, cap, subsample_seed)
        expected, old_reported = _score_with_warnings(baseline, cap, subsample_seed)
        _assert_score_identity(result, expected)
        assert reported == old_reported
        if cap is None:
            table = result["table"].set_index("gene")
            assert table.loc["A_EQUAL_TWO", "permutation_exact"]
            assert not table.loc["B_EQUAL_FOUR", "permutation_exact"]
            assert table.loc["B_EQUAL_FOUR", "permutation_draws"] == 1000
            assert table.loc["B_EQUAL_FOUR", "permutation_resolution"] == 1 / 1001
            assert table.loc["F_FIXED_STRATUM", "permutation_supported"]
            assert not table.loc["G_UNSUPPORTED", "permutation_supported"]
            for metric in G.ETEST_METRICS:
                assert table.loc["G_UNSUPPORTED", f"p_e_{metric}"] == 1.0
                assert table.loc["G_UNSUPPORTED", f"q_e_{metric}"] == 1.0
            if first is None:
                first = result
            else:
                _assert_score_identity(result, first)
    np.testing.assert_array_equal(screen[0], before)


@pytest.mark.parametrize("workers", [1, 4], ids=["forced-serial", "forced-parallel"])
def test_skipped_heldout_profiles_keep_frozen_nan_denominators_and_warnings(monkeypatch, workers):
    """Unsupported real genes stay visible while every unavailable pseudo-gene stays unscored."""
    actual, baseline = _fit_pair(_heterogeneous_screen(dimensions=7), n_holdout=5)
    for test in (actual, baseline):
        test.genes = test.genes.loc[["G_UNSUPPORTED"]]
        test.control_guides = test.control_guides.iloc[:0]

    def selected_workers(job_count):
        """Force both scheduler paths even for the single unsupported gene."""
        return workers

    monkeypatch.setattr(actual, "_guide_worker_count", selected_workers)
    result, reported = _score_with_warnings(actual)
    expected, old_reported = _score_with_warnings(baseline)
    _assert_score_identity(result, expected)
    assert reported == old_reported
    assert result["table"]["gene"].tolist() == ["G_UNSUPPORTED"]
    assert not result["table"].iloc[0]["permutation_supported"]
    for metric in G.ETEST_METRICS:
        assert result["table"].iloc[0][f"p_e_{metric}"] == 1
        assert result["table"].iloc[0][f"q_e_{metric}"] == 1
    for summary in result["summary"]:
        assert summary["genes_unsupported"] == 1
        assert summary["nt_pseudo_scored"] == 0
        assert summary["nt_pseudo_unscored"] == 5
        assert np.isnan(summary["nt_pseudo_p05_pct"])
    assert any("held-out pseudo-genes" in message for _, message in reported)


@pytest.mark.parametrize("fail_first", [False, True], ids=["ordered-results", "worker-error"])
def test_parallel_runner_orders_results_bounds_plans_and_owns_private_caches(monkeypatch, fail_first):
    """Delayed first work bounds lookahead, preserves RNG order and joins isolated workers on error."""
    from concurrent.futures import Future, ThreadPoolExecutor
    import threading

    actual, _ = _fit_pair(_heterogeneous_screen(dimensions=2), n_holdout=0)
    actual._guide_permutation_bank((6, 6))
    first, second = map(int, actual._half_positions[0][:2])
    actual._guide_distance_geometry(first, second, False)
    dense = actual._guide_cell_geometry()
    dense_before = dense.copy()
    parent_banks = dict(actual._permutation_banks)
    parent_geometry = dict(actual._guide_geometry_cache)
    parent_profile_bytes = actual._permutation_bank_bytes
    parent_geometry_bytes = actual._guide_geometry_bytes
    lock = threading.Lock()
    release_first = threading.Event()
    four_started = threading.Event()
    later_finished = threading.Event()
    waiting_for_first = threading.Event()
    contexts, planned, plan_threads, completed, results, errors = [], [], [], [], [], []
    workers_seen, future_tags, consumed = set(), {}, set()
    accounting = {"submitted": 0, "consumed": 0, "peak": 0}
    original_context = G.EnergyTest._guide_context
    original_submit = ThreadPoolExecutor.submit
    original_result = Future.result

    def selected_workers(job_count):
        """Exercise four real worker threads without depending on host CPU or native limits."""
        return 4

    def capture_context(test, geometry_limit, profile_limit):
        """Retain each independent planner or worker context for ownership and budget assertions."""
        context = original_context(test, geometry_limit, profile_limit)
        with lock:
            contexts.append(context)
        return context

    def plan(test, index, rng):
        """Consume exactly one main-thread random draw per plan before any worker evaluation."""
        with lock:
            plan_threads.append(threading.get_ident())
        test._eligibility_sets["planner-owner"] = id(test)
        return {"index": index, "draw": float(rng.random())}

    def evaluate(test, prepared):
        """Force later work to finish first and optionally fail the blocked first result."""
        number = int(prepared["index"][0])
        with lock:
            workers_seen.add(threading.get_ident())
            if len(workers_seen) == 4:
                four_started.set()
        test._eligibility_sets["worker-owner"] = id(test)
        assert four_started.wait(10), "four requested worker threads did not start"
        if number == 0:
            assert release_first.wait(10), "test coordinator did not release the first result"
            if fail_first:
                raise ValueError("deliberate guide worker failure")
        with lock:
            completed.append(number)
        if number == 1:
            later_finished.set()
        return np.array([float(number), float(number)]), np.ones(2), {"draw": prepared["draw"]}

    def submit(executor, function, *args, **kwargs):
        """Count submitted but not yet consumed futures, including completed buffered results."""
        future = original_submit(executor, function, *args, **kwargs)
        with lock:
            future_tags[future] = int(args[0]["index"][0])
            accounting["submitted"] += 1
            accounting["peak"] = max(accounting["peak"],
                                      accounting["submitted"] - accounting["consumed"])
        return future

    def result(future, *args, **kwargs):
        """Signal the ordered wait and record consumption even when a worker raises."""
        with lock:
            tracked = future in future_tags
            first_result = tracked and future_tags[future] == 0
        if first_result:
            waiting_for_first.set()
        try:
            return original_result(future, *args, **kwargs)
        finally:
            with lock:
                if tracked and future not in consumed:
                    consumed.add(future)
                    accounting["consumed"] += 1

    def jobs():
        """Reject unbounded lazy consumption while the first result remains blocked."""
        for number in range(12):
            if number >= 8:
                assert release_first.is_set(), "runner consumed more than eight outstanding jobs"
            with lock:
                planned.append(number)
            yield number, np.array([number, number], dtype=np.int64)

    def consume():
        """Collect ordered runner output or its original exception on one coordinator thread."""
        try:
            results.extend(actual._guide_run_plans(jobs(), np.random.default_rng(1978), 12))
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(actual, "_guide_worker_count", selected_workers)
    monkeypatch.setattr(G.EnergyTest, "_guide_context", capture_context)
    monkeypatch.setattr(G.EnergyTest, "_guide_permutation_plan", plan)
    monkeypatch.setattr(G.EnergyTest, "_guide_evaluate_plan", evaluate)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    monkeypatch.setattr(Future, "result", result)
    coordinator = threading.Thread(target=consume, daemon=True)
    coordinator.start()
    try:
        assert waiting_for_first.wait(10), "runner did not wait for its first ordered result"
        assert later_finished.wait(10), "later work could not complete while the first result waited"
        with lock:
            assert planned == list(range(8))
            assert accounting["peak"] == 8
            assert 1 in completed and 0 not in completed
    finally:
        release_first.set()
        four_started.set()
        coordinator.join(10)
    assert not coordinator.is_alive(), "runner did not join its worker pool"
    assert len(workers_seen) == 4
    assert set(plan_threads) == {coordinator.ident}
    assert accounting["peak"] <= 8
    assert all(future.done() for future in future_tags)
    if fail_first:
        assert len(errors) == 1 and isinstance(errors[0], ValueError)
        assert str(errors[0]) == "deliberate guide worker failure"
        assert results == [] and planned == list(range(8))
    else:
        assert errors == []
        assert [tag for tag, _, _ in results] == list(range(12))
        assert [count for _, count, _ in results] == [2] * 12
        assert completed.index(1) < completed.index(0)
        expected_draws = np.random.default_rng(1978).random(12)
        for (tag, _, (energy, pvalues, metadata)), draw in zip(results, expected_draws, strict=True):
            np.testing.assert_array_equal(energy, [tag, tag])
            np.testing.assert_array_equal(pvalues, [1, 1])
            assert metadata["draw"] == draw
    assert len(contexts) == 5
    for attribute in ("_permutation_banks", "_guide_geometry_cache", "_eligibility_sets", "_nulls"):
        owners = [id(getattr(context, attribute)) for context in contexts + [actual]]
        assert len(set(owners)) == 6, attribute
    assert all(context._guide_cell_distances is dense for context in contexts)
    assert actual._guide_cell_bytes + parent_geometry_bytes + sum(
        context._guide_geometry_limit for context in contexts) <= 512 * 1024 ** 2
    assert parent_profile_bytes + sum(context._guide_profile_limit for context in contexts) <= 64 * 1024 ** 2
    assert actual._permutation_bank_bytes == parent_profile_bytes
    assert actual._guide_geometry_bytes == parent_geometry_bytes
    assert actual._permutation_banks.keys() == parent_banks.keys()
    assert actual._guide_geometry_cache.keys() == parent_geometry.keys()
    assert all(actual._permutation_banks[key] is value for key, value in parent_banks.items())
    assert all(actual._guide_geometry_cache[key] is value for key, value in parent_geometry.items())
    assert "planner-owner" not in actual._eligibility_sets
    assert "worker-owner" not in actual._eligibility_sets
    np.testing.assert_array_equal(dense, dense_before)


def test_profile_eviction_recreates_identical_rows_distances_and_rng_continuation():
    """Cross-profile cache reuse and eviction cannot substitute cells or change Monte Carlo draws."""
    actual, baseline = _fit_pair(_heterogeneous_screen(dimensions=7, include_eviction_profiles=True),
                                 n_holdout=0)
    profile = (6, 6, 6, 6)
    index = actual.genes["B_EQUAL_FOUR"]
    rng = np.random.default_rng(1978)
    old_rng = np.random.default_rng(1978)
    initial = actual._guide_permutation_test(index, rng)
    old_initial = baseline._guide_permutation_test(index, old_rng)
    _assert_permutation_identity(initial, old_initial)
    assert rng.bit_generator.state == old_rng.bit_generator.state
    assert not initial[2]["permutation_exact"]
    retained = actual._permutation_banks[profile]
    old_retained = baseline._permutation_banks[profile]
    original_rows = tuple(rows.copy() for rows in retained["rows"])

    for count in range(3, 13):
        query = actual.genes[f"PROFILE_{count:02d}"]
        result = actual._guide_permutation_test(query, rng)
        expected = baseline._guide_permutation_test(query, old_rng)
        _assert_permutation_identity(result, expected)
        assert rng.bit_generator.state == old_rng.bit_generator.state

    assert profile not in actual._permutation_banks
    assert profile not in baseline._permutation_banks
    assert not retained["cached"] and not old_retained["cached"]
    rng = np.random.default_rng(1978)
    old_rng = np.random.default_rng(1978)
    repeated = actual._guide_permutation_test(index, rng)
    old_repeated = baseline._guide_permutation_test(index, old_rng)
    _assert_permutation_identity(repeated, old_repeated)
    _assert_permutation_identity(repeated, initial)
    assert rng.bit_generator.state == old_rng.bit_generator.state
    for rows, original in zip(actual._permutation_banks[profile]["rows"], original_rows):
        np.testing.assert_array_equal(rows, original)


def test_cell_geometry_preserves_rectangle_order_and_reuses_cells_across_profiles(monkeypatch):
    """Unsorted, reversed and diagonal rectangles reuse exact geometry without another cell read."""
    actual, baseline = _fit_pair(_heterogeneous_screen(dimensions=7), n_holdout=0)
    names = tuple(reversed(actual._halves[0][-4:]))
    first_positions = ([5, 0, 3], [6, 1, 4, 2, 0], [7, 0, 2], [8, 2, 6, 0, 4])
    next_positions = ([8, 1, 6, 2, 4], [7, 2, 0], [6, 3, 8, 0, 1], [5, 1, 4])
    first_groups = tuple(actual._pools[name][positions]
                         for name, positions in zip(names, first_positions))
    next_groups = tuple(actual._pools[name][positions]
                        for name, positions in zip(names[1:] + names[:1], next_positions))
    pairs = np.arange(len(names) ** 2)
    expected = baseline._guide_pair_sums(first_groups, pairs,
                                         {"pairs": {}, "cached": False}, 0, {})
    result = actual._guide_pair_sums(first_groups, pairs,
                                     {"pairs": {}, "cached": False}, 0, {})
    np.testing.assert_array_equal(result, expected)
    distances = actual._guide_cell_distances
    assert distances is not None
    assert distances.shape == (actual._guide_cell_size, actual._guide_cell_size)
    assert actual._guide_cell_bytes >= distances.nbytes
    assert actual._guide_cell_bytes + getattr(actual, "_guide_geometry_bytes", 0) <= 512 * 1024 ** 2
    before = distances.copy()
    expected = baseline._guide_pair_sums(next_groups, pairs,
                                         {"pairs": {}, "cached": False}, 0, {})

    def reject_cell_read(index):
        """Fail if a warm cell lookup recomputes geometry from any embedding cells."""
        raise AssertionError("cell geometry should reuse previously read cells")

    with monkeypatch.context() as patch:
        patch.setattr(actual, "_rows", reject_cell_read)
        result = actual._guide_pair_sums(next_groups, pairs,
                                         {"pairs": {}, "cached": False}, 0, {})
    np.testing.assert_array_equal(result, expected)
    assert actual._guide_cell_distances is distances
    np.testing.assert_array_equal(distances, before)


def test_compact_pair_cache_reuses_exact_values_and_refuses_an_exhausted_budget(monkeypatch):
    """Bulk pair hits preserve values and order without geometry reads or over-budget allocation."""
    actual, baseline = _fit_pair(_heterogeneous_screen(dimensions=7), n_holdout=0)
    bank = actual._guide_permutation_bank((12, 6))
    groups = (actual.genes["A_EQUAL_TWO"][:6],) + bank["rows"]
    count, total = len(bank["rows"]), len(groups)
    left, right = np.indices((count, count))
    pairs = ((left + 1) * total + right + 1).ravel()
    expected = baseline._guide_pair_sums(groups, pairs,
                                         {"pairs": {}, "cached": False}, 1, {})
    budget_before, bank_before = actual._permutation_bank_bytes, bank["bytes"]
    result = actual._guide_pair_sums(groups, pairs, bank, 1, {})
    np.testing.assert_array_equal(result, expected)
    retained = bank["pair_values"]
    assert bank["pairs"] == {}
    assert bank["bytes"] == bank_before + retained.nbytes + 512
    assert actual._permutation_bank_bytes == budget_before + retained.nbytes + 512
    assert actual._permutation_bank_bytes <= 64 * 1024 ** 2
    before = retained.copy()
    order = np.r_[np.arange(len(pairs) - 1, -1, -1), np.arange(0, len(pairs), 3)]

    def reject_geometry(*args, **kwargs):
        """Fail if a compact pair hit reaches any cell or distance-geometry reader."""
        raise AssertionError("compact pair hits must not read geometry or cells")

    with monkeypatch.context() as patch:
        for name in ("_rows", "_guide_cell_geometry", "_guide_distance_geometry", "_prefix_distance_sums"):
            patch.setattr(actual, name, reject_geometry)
        result = actual._guide_pair_sums(groups, pairs[order], bank, 1, {})
    np.testing.assert_array_equal(result, expected[order])
    np.testing.assert_array_equal(retained, before)
    assert actual._permutation_bank_bytes == budget_before + retained.nbytes + 512

    fallback = actual._guide_permutation_bank((6, 6))
    groups = fallback["rows"]
    pairs = np.arange(len(groups) ** 2)
    expected = baseline._guide_pair_sums(groups, pairs,
                                         {"pairs": {}, "cached": False}, 0, {})
    bank_before = fallback["bytes"]
    limit = 64 * 1024 ** 2
    original_full = np.full
    pair_shape = (len(groups), len(groups), 2)

    def reject_pair_allocation(shape, fill_value, *args, **kwargs):
        """Reject the compact pair table while allowing unrelated indexing scratch arrays."""
        if isinstance(shape, (tuple, list)) and tuple(shape) == pair_shape:
            raise AssertionError("an exhausted profile budget must not allocate compact pair storage")
        return original_full(shape, fill_value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(actual, "_permutation_bank_bytes", limit)
        patch.setattr(np, "full", reject_pair_allocation)
        assert actual._guide_pair_cache(fallback, len(groups)) is None
        result = actual._guide_pair_sums(groups, pairs, fallback, 0, {})
        assert actual._permutation_bank_bytes == limit
    np.testing.assert_array_equal(result, expected)
    assert "pair_values" not in fallback
    assert fallback["pairs"] == {} and fallback["bytes"] == bank_before


def _numerical_screen(count, dimensions, tied):
    """Make dyadic guide blocks for exact translations, or identical nonbinary cells for ties."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(31, count, dimensions)))
    cells = rng.integers(-32, 33, size=(8 * count, dimensions)).astype(np.float64) / 16
    cells += np.repeat(np.arange(8) / 4, count)[:, None]
    if tied:
        cells[:] = np.resize(np.array([0.1, 0.2, 0.3]), dimensions)
    labels = np.array(["nontargeting"] * (6 * count) + ["TARGET"] * (2 * count))
    guides = np.repeat([f"NT_{number}" for number in range(6)] + ["TARGET_0", "TARGET_1"], count)
    return cells, labels, guides


@pytest.mark.parametrize("count,dimensions,tied", [
    pytest.param(8, 5, False, id="broadcast-translated"),
    pytest.param(128, 65, False, id="bounded-rectangle-translated"),
    pytest.param(8, 5, True, id="nonbinary-identical-ties"),
])
def test_large_common_offsets_preserve_both_metric_ranks_and_energy(count, dimensions, tied):
    """A large coordinate origin cannot replace stable differences with cancelling Gram products."""
    cells, labels, guides = _numerical_screen(count, dimensions, tied)
    translated = cells + np.resize(np.array([2.0 ** 30, -2.0 ** 30]), dimensions)
    original_result = None
    for values in (cells, translated):
        actual, baseline = _fit_pair((values, labels, guides), n_ref=count,
                                     min_guide_cells=2, n_holdout=2)
        result, reported = _score_with_warnings(actual)
        expected, old_reported = _score_with_warnings(baseline)
        _assert_score_identity(result, expected)
        assert reported == old_reported
        assert result["table"].permutation_exact.all()
        if original_result is None:
            original_result = result
        else:
            _assert_score_identity(result, original_result)
        if tied:
            for metric in G.ETEST_METRICS:
                assert (result["table"][f"p_e_{metric}"] == 1.0).all()
                assert (result["table"][f"e_{metric}"] == 0.0).all()
                gene = result["table"].set_index("gene").loc["TARGET"]
                assert gene[f"q_e_{metric}"] == 1.0


@pytest.mark.parametrize("nonfinite", [
    pytest.param(np.nan, id="unused-nan"),
    pytest.param(np.inf, id="unused-infinity"),
])
def test_geometry_prefetch_ignores_nonfinite_cells_outside_the_sample(nonfinite):
    """Only sampled NT cells affect validity, even when their full guide enters the cache."""
    cells, labels, guides = _heterogeneous_screen(dimensions=7)
    layout, _ = _fit_pair((cells, labels, guides), n_holdout=0)
    bank = layout._guide_permutation_bank((6, 6))
    sampled = bank["rows"][0]
    name = guides[sampled[0]]
    full_guide = layout._pools[name]
    unused = np.setdiff1d(full_guide, sampled)[0]
    assert unused not in layout._reference_indices
    contaminated = cells.copy()
    contaminated[unused, 0] = nonfinite
    actual, baseline = _fit_pair((contaminated, labels, guides), n_holdout=0)
    rng = np.random.default_rng(2978)
    old_rng = np.random.default_rng(2978)
    result = actual._guide_permutation_test(actual.genes["A_EQUAL_TWO"], rng)
    expected = baseline._guide_permutation_test(baseline.genes["A_EQUAL_TWO"], old_rng)
    _assert_permutation_identity(result, expected)
    assert rng.bit_generator.state == old_rng.bit_generator.state
    assert np.isfinite(result[0]).all() and np.isfinite(result[1]).all()
    distances = actual._guide_cell_distances
    if distances is not None:
        assert not np.isfinite(distances).all()
    else:
        position = actual._pools.index.get_loc(name)
        containing = [matrix for key, matrix in actual._guide_geometry_cache.items()
                      if position in key[:2]]
        assert containing
        assert any(not np.isfinite(matrix[0]).all() for matrix in containing)

    groups = (np.array([unused]), np.array([sampled[0]]))
    for test in (actual, baseline):
        with pytest.raises(ValueError, match="distances must be finite"):
            test._guide_pair_sums(groups, np.array([0, 1]),
                                  {"pairs": {}, "cached": False}, 0, {})


def test_oversized_full_guide_pair_uses_only_the_small_sampled_rectangle(monkeypatch):
    """An over-budget full-guide pair must fall back before allocating or reading that pair."""
    actual, baseline = _fit_pair(_numerical_screen(6000, 2, False),
                                 n_ref=2, n_holdout=0, min_guide_cells=2)
    left, right = actual._halves[0]
    groups = (actual._pools[left][[17, 2, 33]],
              actual._pools[right][[31, 4, 45, 9, 1]])
    pairs = np.array([0, 1, 3])
    expected = baseline._guide_pair_sums(groups, pairs,
                                         {"pairs": {}, "cached": False}, 0, {})
    original_empty = np.empty
    original_rows = actual._rows
    allocation_sizes, row_counts = [], []

    def bounded_empty(shape, *args, **kwargs):
        """Reject full-guide allocations before memory is reserved, while allowing sampled work."""
        size = int(np.prod(shape))
        assert size <= 4096, f"unexpected full-guide allocation: {shape}"
        allocation_sizes.append(size)
        return original_empty(shape, *args, **kwargs)

    def sampled_rows(index):
        """Reject full-guide prefetches before materializing their embedding rows."""
        count = np.asarray(index).size
        assert count <= 32, f"unexpected full-guide row read: {count}"
        row_counts.append(count)
        return original_rows(index)

    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", bounded_empty)
        patch.setattr(actual, "_rows", sampled_rows)
        result = actual._guide_pair_sums(groups, pairs,
                                         {"pairs": {}, "cached": False}, 0, {})
    np.testing.assert_array_equal(result, expected)
    assert allocation_sizes and row_counts
    assert actual._guide_geometry_cache == {}
    assert actual._guide_geometry_bytes == 0
