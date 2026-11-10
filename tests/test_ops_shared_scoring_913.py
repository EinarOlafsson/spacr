"""Item 913: shared copairs / EFAAR scoring of OPS gene profiles and clusterings.

copairs and efaar_benchmarking are an optional extra. When they are not
installed, small stand-ins with the same call signatures (an exact average
precision, a seeded permutation null, a tail-recall benchmark) let the
scoring step's own logic run: planted gene groups must be active and
consistent, non-targeting pseudo-genes must not, shuffled labels fall to
the null, and an input with clusters only is listed as not scorable for
activity. With the real packages installed, the planted run is skipped (their EFAAR
sources name real genes) and the rest run on them.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from spacr import ops_scoring as S

_COPAIRS = '''
import numpy as np
import pandas as pd


def _ap(sim_row, positive):
    order = np.argsort(-sim_row, kind="stable")
    rel = positive[order]
    if not rel.any():
        return np.nan
    hits = np.cumsum(rel)
    return float((hits[rel] / (np.flatnonzero(rel) + 1)).mean())
'''

_MATCHING = '''
import numpy as np


def assign_reference_index(meta, index, reference_col):
    meta = meta.copy()
    meta[reference_col] = -1
    meta.loc[index, reference_col] = np.arange(len(index))
    return meta
'''

_MAP = '''
import numpy as np
import pandas as pd
from copairs import _ap


def _cos(x):
    x = np.asarray(x, float)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x @ x.T


def average_precision(meta, feats, pos_sameby, pos_diffby, neg_sameby, neg_diffby, batch_size=0,
                      progress_bar=False):
    sim = _cos(feats)
    key = pos_sameby[0]
    ctrl = meta[neg_diffby[0]].to_numpy(bool)
    labels = meta[key].astype(str).to_numpy()
    out = meta.copy()
    ap, npos, ntot = [], [], []
    for i in range(len(meta)):
        if ctrl[i]:
            ap.append(np.nan); npos.append(0); ntot.append(0)
            continue
        cand = np.flatnonzero((np.arange(len(meta)) != i) & ((labels == labels[i]) | ctrl))
        pos = labels[cand] == labels[i]
        ap.append(_ap(sim[i, cand], pos)); npos.append(int(pos.sum())); ntot.append(len(cand))
    out["average_precision"] = ap
    out["n_pos_pairs"] = npos
    out["n_total_pairs"] = ntot
    return out


def mean_average_precision(ap, sameby, null_size, threshold, seed=0, progress_bar=False, max_workers=None,
                           cache_dir=None):
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(seed)
    rows = []
    for name, d in ap.dropna(subset=["average_precision"]).groupby(sameby[0]):
        m = float(d.average_precision.mean())
        npos, ntot = int(d.n_pos_pairs.iloc[0]), int(d.n_total_pairs.iloc[0])
        null = []
        for _ in range(null_size):
            rel = np.zeros(ntot, bool)
            rel[rng.choice(ntot, max(npos, 1), replace=False)] = True
            hits = np.cumsum(rel)
            null.append((hits[rel] / (np.flatnonzero(rel) + 1)).mean())
        p = (np.sum(np.asarray(null) >= m) + 1) / (null_size + 1)
        rows.append({sameby[0]: name, "mean_average_precision": m, "p_value": p})
    out = pd.DataFrame(rows)
    out["corrected_p_value"] = multipletests(out.p_value, method="fdr_bh")[1] if len(out) else []
    out["below_p"] = out.p_value < threshold
    out["below_corrected_p"] = out.corrected_p_value < threshold
    return out
'''

_MULTILABEL = '''
import numpy as np
import pandas as pd
from copairs import _ap
from copairs.map.map import _cos


def average_precision(meta, feats, pos_sameby, pos_diffby, neg_sameby, neg_diffby, multilabel_col,
                      progress_bar=False):
    sim = _cos(feats)
    labels = meta[multilabel_col].tolist()
    rows = []
    for i, mine in enumerate(labels):
        others = np.r_[0:i, i + 1:len(labels)]
        for c in mine:
            pos = np.array([c in labels[j] for j in others])
            rows.append({multilabel_col: c, "average_precision": _ap(sim[i, others], pos),
                         "n_pos_pairs": int(pos.sum()), "n_total_pairs": len(others)})
    return pd.DataFrame(rows)
'''

_MAP_INIT = '''
from copairs.map.map import average_precision, mean_average_precision
from copairs.map import multilabel
'''

_COMPUTE = '''
import numpy as np


def ap_contiguous(rel, counts):
    out, confs, start = [], [], 0
    for n in counts:
        r = rel[start:start + n].astype(bool)
        start += n
        hits = np.cumsum(r)
        out.append(float((hits[r] / (np.flatnonzero(r) + 1)).mean()) if r.any() else 0.0)
        confs.append((int(r.sum()), int(n)))
    return np.array(out), np.array(confs)


def get_null_dists(confs, null_size, seed, cache_dir=None, progress_bar=False):
    rows = []
    for npos, ntot in confs:
        rng = np.random.default_rng((seed, int(npos), int(ntot)))
        draws = []
        for _ in range(null_size):
            r = np.zeros(ntot, bool)
            r[rng.choice(ntot, max(int(npos), 1), replace=False)] = True
            hits = np.cumsum(r)
            draws.append((hits[r] / (np.flatnonzero(r) + 1)).mean())
        rows.append(draws)
    return np.array(rows)
'''

_EFAAR_CONST = '''
BENCHMARK_DATA_DIR = "fake"
BENCHMARK_SOURCES = ["CORUM", "StringDB"]
RECALL_PERC_THRS = [(0.05, 0.95), (0.1, 0.9)]
'''

_EFAAR_BENCH = '''
import os
import numpy as np
import pandas as pd
from efaar_benchmarking import constants as cst


def get_benchmark_relationships(data_dir, src):
    return pd.read_parquet(os.environ["FAKE_EFAAR_PAIRS"]).assign(source=src)


def known_relationship_benchmark(bunch, col, log_stats=False):
    X = bunch.features.to_numpy(float)
    genes = list(bunch.metadata[col])
    x = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = x @ x.T
    null = np.sort(S[np.triu_indices(len(S), 1)])
    index = {g: i for i, g in enumerate(genes)}
    rows = []
    for src in cst.BENCHMARK_SOURCES:
        rel = get_benchmark_relationships(cst.BENCHMARK_DATA_DIR, src)
        q = S[rel.entity1.map(index).to_numpy(), rel.entity2.map(index).to_numpy()]
        left = np.searchsorted(null, q, side="left") / len(null)
        right = np.searchsorted(null, q, side="right") / len(null)
        row = {"source": src, "null_distribution_size": len(null), "query_distribution_size": len(q)}
        for lo, hi in cst.RECALL_PERC_THRS:
            row[f"recall_{lo}_{hi}"] = float(((right <= lo) | (left >= hi)).mean())
        rows.append(row)
    return pd.DataFrame(rows)
'''

GROUPS = 6
PER = 4


@pytest.fixture
def packages(tmp_path, monkeypatch):
    """The real scoring packages, or stand-ins with their signatures."""
    pairs = pd.DataFrame([(f"G{g}_{a}", f"G{g}_{b}") for g in range(GROUPS) for a in range(PER)
                          for b in range(a + 1, PER)], columns=["entity1", "entity2"])
    pairs.to_parquet(tmp_path / "pairs.parquet")
    monkeypatch.setenv("FAKE_EFAAR_PAIRS", str(tmp_path / "pairs.parquet"))
    try:
        importlib.import_module("copairs")
        importlib.import_module("efaar_benchmarking")
        real = True
    except ImportError:
        real = False
    if real:
        yield "real"
        return
    root = tmp_path / "fakes"
    files = {"copairs/__init__.py": _COPAIRS, "copairs/matching.py": _MATCHING,
             "copairs/map/__init__.py": _MAP_INIT, "copairs/map/map.py": _MAP,
             "copairs/map/multilabel.py": _MULTILABEL, "copairs/compute.py": _COMPUTE,
             "efaar_benchmarking/__init__.py": "", "efaar_benchmarking/constants.py": _EFAAR_CONST,
             "efaar_benchmarking/benchmarking.py": _EFAAR_BENCH}
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m.startswith(("copairs", "efaar_benchmarking"))]:
        monkeypatch.delitem(sys.modules, name)
    yield "fake"
    for name in [m for m in sys.modules if m.startswith(("copairs", "efaar_benchmarking"))]:
        sys.modules.pop(name, None)


def _references(folder):
    """A CORUM file whose complexes are the planted groups."""
    folder.mkdir()
    rows = [(f"cx{g}", f"G{g}_{k}") for g in range(GROUPS) for k in range(PER)]
    table = pd.DataFrame(rows, columns=["group", "gene_name"])
    table.to_csv(folder / "corum_group_benchmark.tsv", sep="\t", index=False)
    return str(folder)


def _inputs(tmp_path):
    """Planted gene groups as gene rows, guide rows and clusters, with NT rows."""
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (GROUPS, 8))
    genes = [f"G{g}_{k}" for g in range(GROUPS) for k in range(PER)]
    gene_rows = np.vstack([centres[g] + rng.normal(0, 0.1, 8) for g in range(GROUPS) for _ in range(PER)])
    nt = [f"nontargeting_{i}" for i in range(16)]
    nt_rows = rng.normal(0, 0.05, (16, 8))
    table = pd.DataFrame(np.vstack([gene_rows, nt_rows]), columns=[f"f{i}" for i in range(8)])
    table.insert(0, "gene", genes + nt)
    table.to_parquet(tmp_path / "genes.parquet")
    guides = []
    for i, g in enumerate(genes):
        for k in range(4):
            guides.append([f"{g}_s{k}", g, *(gene_rows[i] + rng.normal(0, 0.1, 8))])
    for i in range(24):
        guides.append([f"nt{i}", f"nontargeting_{i % 16}", *rng.normal(0, 0.05, 8)])
    guide_table = pd.DataFrame(guides, columns=["guide", "gene"] + [f"f{i}" for i in range(8)])
    guide_table["n_cells"] = 30
    guide_table.to_parquet(tmp_path / "guides.parquet")
    pd.DataFrame({"gene": genes, "cluster": [g // PER for g in range(len(genes))]}).to_parquet(
        tmp_path / "clusters.parquet")
    pd.DataFrame({"gene": genes, "cluster": rng.integers(0, 3, len(genes))}).to_parquet(
        tmp_path / "noisy.parquet")
    return genes


def test_planted_groups_score_and_controls_do_not(tmp_path, packages):
    if packages == "real":
        pytest.skip("the real EFAAR sources hold real gene symbols, not the planted ones; the real packages are "
                    "checked against the reference bundle instead")
    genes = _inputs(tmp_path)
    settings = {"dst_root": str(tmp_path / "run"), "ops_reference_dir": _references(tmp_path / "refs"),
                "ops_scoring_null_size": 200, "ops_scoring_references": ["CORUM"],
                "ops_scoring_inputs": {"spaCR": {"genes": str(tmp_path / "genes.parquet"),
                                                 "guides": str(tmp_path / "guides.parquet"),
                                                 "clusters": str(tmp_path / "clusters.parquet")},
                                       "Other": {"genes": str(tmp_path / "genes.parquet"),
                                                 "clusters": str(tmp_path / "noisy.parquet")}},
                "ops_scoring_min_complex": 3}
    from spacr.ops_engine import run_ops

    result = run_ops(settings, phases=["score"])
    folder = result["folder"]
    activity = pd.read_csv(os.path.join(folder, "activity_map.csv"))
    real = activity[activity.set == "spaCR genes"]
    assert len(real) == len(genes) and real.below_corrected_p.all()
    assert not activity[activity.set.str.startswith("NT pseudo")].below_corrected_p.any()
    scorable = pd.read_csv(os.path.join(folder, "scorable.csv"))
    other = scorable[(scorable.input == "Other") & (scorable.metric == "activity")]
    assert set(other.scorable) == {"not scorable"} and other.reason.str.contains("activity|guide").all()
    complexes = pd.read_csv(os.path.join(folder, "consistency_complex_map.csv"))
    planted = complexes[(complexes.rep == "spaCR profile") & (complexes.control == "none")]
    assert planted.mean_average_precision.min() > 0.9
    shuffled = complexes[(complexes.rep == "spaCR profile") & (complexes.control == "shuffled gene labels")]
    assert shuffled.mean_average_precision.mean() < planted.mean_average_precision.mean() - 0.3
    assert (complexes.control == "NT pseudo-complexes").any()
    pairs = pd.read_csv(os.path.join(folder, "consistency_pairs_summary_by_seed.csv"))
    real_pairs = pairs[(pairs.rep == "spaCR profile") & (pairs.control == "none")].mAP.iloc[0]
    assert real_pairs > 0.9
    assert pairs[(pairs.rep == "spaCR profile") & (pairs.control == "shuffled gene labels")].mAP.iloc[0] < 0.6
    recall = pd.read_csv(os.path.join(folder, "efaar_recall_summary.csv"))
    hit = recall[(recall.rep == "spaCR profile") & (recall.control == "none")].recall_0p05_0p95.mean()
    null = recall[(recall.rep == "spaCR profile") & (recall.control == "shuffled gene labels")].recall_0p05_0p95.mean()
    assert hit > 3 * 0.10 and null < hit / 2
    tests = pd.read_csv(os.path.join(folder, "efaar_mcnemar.csv"))
    assert set(zip(tests.a, tests.b)) == {("spaCR clusters", "Other clusters"), ("spaCR profile", "Other profile")}
    summary = pd.read_csv(os.path.join(folder, "summary_main.csv"))
    assert {"spaCR", "Other"} <= set(summary.pipeline)
    stats = json.load(open(os.path.join(folder, "stats.json")))
    assert {"activity", "complex_profile", "complex_clusters", "retrieval_profile", "efaar_clusters",
            "efaar_profile_10pct"} <= set(stats)
    setup = json.load(open(os.path.join(folder, "setup.json")))
    assert "same shared genes" in setup["matching"] and setup["genes"] == len(genes)
    assert os.path.isfile(os.path.join(folder, "fig_shared_profile.png"))
    assert os.path.isfile(os.path.join(folder, "fig_shared_clusters.png"))


def test_inputs_are_checked(tmp_path):
    _inputs(tmp_path)
    with pytest.raises(ValueError, match="needs columns"):
        S.load_scoring_inputs({"x": {"clusters": str(tmp_path / "genes.parquet")}})
    with pytest.raises(ValueError, match="no profile"):
        S.load_scoring_inputs({"x": {"genes": str(tmp_path / "genes.parquet")}}, genes=["MISSING"])
    with pytest.raises(ValueError, match="no cluster"):
        S.load_scoring_inputs({"x": {"clusters": str(tmp_path / "clusters.parquet")}}, genes=["MISSING"])
    data = S.load_scoring_inputs({"a": {"genes": str(tmp_path / "genes.parquet"),
                                        "guides": str(tmp_path / "guides.parquet")}})
    rows, nt = data["profiles"]["a"]
    assert rows.shape == (GROUPS * PER + 16, 8) and len(nt) == 16
    assert list(data["genes"]) == sorted(data["genes"])
    from spacr.ops_engine import run_ops

    with pytest.raises(ValueError, match="ops_scoring_inputs"):
        run_ops({"dst_root": str(tmp_path)}, phases=["score"])
    with pytest.raises(ValueError, match="dst_root"):
        S.score_ops_profiles({"ops_scoring_inputs": {"a": {"genes": "x"}}})
    with pytest.raises(ValueError, match="on its own"):
        run_ops({"dst_root": str(tmp_path)}, phases=["score", "stitch"])


def test_scorability_table_names_every_gap():
    data = {"guides": {}, "profiles": {"p": None}, "clusters": {"c": None, "p": None}}
    table = S.scorability_table(data, ["p", "c"])
    profile_activity = table[(table.input == "p") & (table.level == "profile") & (table.metric == "activity")]
    assert profile_activity.reason.iloc[0] == "no guide-level profiles"
    c_profile = table[(table.input == "c") & (table.level == "profile") & (table.metric == "efaar")]
    assert c_profile.scorable.iloc[0] == "not scorable" and "clusters only" in c_profile.reason.iloc[0]
    assert (table[(table.level == "clusters") & (table.metric == "efaar")].scorable == "yes").all()


def test_helpers():
    rows = S.onehot_rows(["a", "b", "a"], eps=0.0)
    assert rows.tolist() == [[1, 0], [0, 1], [1, 0]]
    assert S.is_nt(["NonTargeting_1", "ntpseudo_2", "GENE"]).tolist() == [True, True, False]
    assert set(S._versions()) == set(S.SCORING_PACKAGES)


def test_missing_package_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "copairs", None)
    with pytest.raises(ImportError, match="spacr\\[scoring\\]"):
        S._require("copairs")
