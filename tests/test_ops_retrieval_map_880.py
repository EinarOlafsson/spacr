"""Item 880: gene-gene retrieval mAP with multi-label references (copairs style)."""
import hashlib

import numpy as np
import pandas as pd

from spacr import sp_stats as S


def _labelled(seed=0, n=300, d=20, groups=30):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, groups, n)
    X = rng.standard_normal((n, d)) + 1.5 * rng.standard_normal((groups, d))[lab]
    X[5] = X[6]
    return lab, X


def test_ap_is_profile_average_precision_on_single_label_data():
    lab, X = _labelled()
    meta = pd.DataFrame({"g": lab.astype(str)})
    ref = S._profile_average_precision(meta, X, ["g"], (), (), ["g"])
    pos = lab[:, None] == lab[None, :]
    np.fill_diagonal(pos, False)
    ap, npos, ntot = S._pair_average_precision(S.profile_cosine(X), pos, np.ones(len(lab), bool))
    ok = np.isfinite(ref.average_precision.to_numpy())
    assert ok.sum() > 250
    assert np.allclose(ap[ok], ref.average_precision.to_numpy()[ok], atol=1e-6)
    assert (npos[ok] == ref.n_pos_pairs[ok]).all() and (ntot[ok] == ref.n_total_pairs[ok]).all()
    assert np.isnan(ap[~ok]).all()


def test_a_tie_ranks_the_partner_first():
    sims = np.array([[0, 0.5, 0.5], [0.5, 0, 0.1], [0.5, 0.1, 0]], np.float32)
    pos = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], bool)
    ap, npos, ntot = S._pair_average_precision(sims, pos, np.ones(3, bool))
    assert ap[0] == 1.0 and npos.tolist() == [1, 0, 1] and ntot.tolist() == [2, 2, 2]
    assert np.isnan(ap[1]) and ap[2] == 1.0


def test_random_profiles_score_the_random_expectation(tmp_path):
    lab, _ = _labelled()
    pos = lab[:, None] == lab[None, :]
    np.fill_diagonal(pos, False)
    names = np.array([f"G{i}" for i in range(len(lab))])
    X = np.random.default_rng(1).standard_normal((len(lab), 20))
    table = S.pair_map(S.profile_cosine(X), names, pos, null_size=500, null_cache=str(tmp_path))
    assert list(table.columns) == ["gene", "ap", "n_pos", "n_total", "nap", "expected_ap", "p", "q"]
    assert abs(table.ap.mean() - table.expected_ap.mean()) < 0.02
    assert (table.p < 0.05).mean() < 0.12 and (table.q >= table.p - 1e-12).all()
    assert table.p.min() >= 1 / 501
    shapes = set(zip(table.n_pos, table.n_total))
    assert len(list(tmp_path.iterdir())) == len(shapes)
    summary = S.summarize_pair_map(table, n_boot=50)
    assert summary["queries"] == len(table) and summary["mAP_lo"] <= summary["mAP"] <= summary["mAP_hi"]
    assert set(summary) >= {"frac_sig_q05", "frac_sig_lo", "frac_sig_hi", "mmAP_normalised"}


def test_null_follows_the_shape_seed_and_reads_back_from_its_cache(tmp_path):
    S._PAIR_NULLS.clear()
    seed = int(hashlib.md5(f"{(3, 40)}".encode()).hexdigest()[:8], 16) % 8096
    want = np.sort(S._null_average_precision(200, 3, 40, seed))
    got = S._pair_null(3, 40, 200, str(tmp_path))
    assert np.array_equal(got, want)
    assert (tmp_path / "n200_3_40.npy").is_file()
    S._PAIR_NULLS.clear()
    np.save(tmp_path / "n200_3_40.npy", np.zeros(200, np.float32))
    assert not S._pair_null(3, 40, 200, str(tmp_path)).any()
    S._PAIR_NULLS.clear()
    assert np.array_equal(S._pair_null(3, 40, 200), want)
    S._PAIR_NULLS.clear()


def test_controls_and_the_universe_never_take_part():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 4))
    names = np.array(["A", "B", "C", "D", "nontargeting_1", "E"])
    pos = np.zeros((6, 6), bool)
    for a, b in ((0, 1), (0, 4), (2, 3), (3, 5)):
        pos[a, b] = pos[b, a] = True
    controls = np.array([False] * 4 + [True, False])
    table = S.pair_map(S.profile_cosine(X), names, pos, controls=controls, pvalues=False)
    assert table.gene.tolist() == ["A", "B", "C", "D", "E"]
    assert table.n_total.tolist() == [4] * 5 and table.n_pos.tolist() == [1, 1, 1, 2, 1]
    assert "p" not in table
    inner = S.pair_map(S.profile_cosine(X), names, pos, controls=controls,
                       universe=np.array([True, True, True, True, True, False]), pvalues=False)
    assert inner.gene.tolist() == ["A", "B", "C", "D"] and inner.n_total.tolist() == [3] * 4
    empty = S.pair_map(S.profile_cosine(X), names, np.zeros((6, 6), bool))
    assert empty.empty and list(empty.columns)[-2:] == ["p", "q"]


def test_bootstrap_interval():
    assert S.bootstrap_ci([np.nan]) == (np.nan, np.nan) or all(np.isnan(S.bootstrap_ci([np.nan])))
    lo, hi = S.bootstrap_ci(np.arange(100.0), n_boot=200, seed=3)
    assert 40 < lo < 49.5 < hi < 60
    assert S.bootstrap_ci(np.arange(100.0), n_boot=200, seed=3) == (lo, hi)
    assert np.allclose(S.profile_cosine(np.array([[1.0, 0], [0, 0], [2, 0]])),
                       [[1, 0, 1], [0, 0, 0], [1, 0, 1]])
