"""Simulated optical pooled screens with planted effects of known size.

spaCR's other power code (:mod:`spacr.power_simulate`, :mod:`spacr.power_model`)
plans arrayed and well-level pooled screens. An optical pooled screen (OPS)
has another structure: every cell is genotyped from its own barcode reads,
some cells get no call and some get the wrong one, the phenotype of a cell is
a many-dimensional profile, guides are nested in genes and cells sit in
fields and wells. This module simulates such a screen with known effects and
runs spaCR's own OPS analysis steps on it, so a design can be scored by what
it finds: the share of real effects detected, the false-positive rate, the
smallest effect detected at 80 % power, the complexes the clustering
recovers and the conditional information the information screen can see.

The layers of one simulated screen:

- library: genes, guides per gene, non-targeting (NT) guides, a guide
  abundance skew (Dirichlet, :func:`spacr.power_simulate.rdirichlet_stable`)
  and guide efficacy (inactive guides and a per-guide knockout penetrance,
  :func:`spacr.power_simulate.rbeta_mean_variance`);
- cells: cells per gene at the calibrated call rate, scaled by the design's
  call rate, assigned to guides by abundance, to wells and fields;
- genotyping: a share of the called cells carries a wrong guide (the decoy
  false discovery rate, FDR);
- phenotype: NT cells resampled from a real run (``'empirical'``) or a
  fitted multivariate t with well and field offsets (``'parametric'``);
- planted biology: a share of genes with a mean, variance or subpopulation
  effect on a few feature directions, a penetrance, and complexes of genes
  sharing an effect direction.

:func:`calibrate_from_run` fills every nuisance parameter from a real OPS run
and writes each value it uses to a JSON file. :func:`scan_ops_designs` sweeps
a grid of designs and caches each finished point, and
:func:`run_ops_power_sim` is the ``power_sim`` phase an OPS settings file
runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .power_simulate import rbeta_mean_variance, rdirichlet_stable, resolve_rng

#: The control prefix of simulated NT cells.
_CONTROL = "nontargeting"

#: The calibration file format version.
_CALIBRATION_SCHEMA = 1

#: Complex sizes used when a calibration has no CORUM table, the size
#: distribution of CORUM complexes with two to ten members.
_DEFAULT_COMPLEX_SIZES = (2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 8, 10)

#: Every design key with its default; the calibration replaces the ``None`` ones.
OPS_POWER_DESIGN: Dict[str, Any] = {
    "n_genes": None,
    "guides_per_gene": None,
    "nt_guides": None,
    "cells_per_gene": None,
    "call_rate": None,
    "fdr_pct": None,
    "wells": None,
    "fields_per_well": None,
    "guide_alpha": None,
    "inactive_guide_fraction": 0.0,
    "ko_penetrance": 1.0,
    "ko_penetrance_sd": 0.0,
    "hit_fraction": 0.1,
    "effect_type": "mean",
    "effect_size": 1.0,
    "effect_dims": 5,
    "effect_block": "any",
    "penetrance": 1.0,
    "complex_fraction": 0.5,
    "complex_noise": 0.3,
    "mode": "empirical",
    "bandwidth": 0.25,
    "batch_scale": 1.0,
    "info_new_dims": 8,
    "info_permutations": 100,
    "info_null": "guide",
    "etest_metric": "euclidean",
    "etest_mode": "guide_null",
    "etest_reference_cells": 2000,
    "etest_null": 1000,
    "etest_holdout": 500,
    "cluster_resolution": 16.0,
    "cluster_min_cells": 10,
}

#: The default sweep: cells per gene at Funk et al. 2022 plate A scale.
DEFAULT_OPS_POWER_GRID: Dict[str, List[Any]] = {
    "cells_per_gene": [10, 25, 50, 100, 200, 500, 1000],
    "effect_size": [0.1, 0.2, 0.3, 0.5, 1.0],
}


def _sha256(path: str) -> str:
    """The sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: str) -> Dict[str, Any]:
    """Where an input came from: path, size and modification time."""
    stat = os.stat(path)
    return {"path": os.path.abspath(path), "bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    """The 5, 25, 50, 75 and 95 % quantiles and the mean."""
    values = np.asarray(values, float)
    if not len(values):
        return {}
    out = {f"q{q}": float(np.quantile(values, q / 100)) for q in (5, 25, 50, 75, 95)}
    out["mean"] = float(values.mean())
    return out


def _guide_alpha(counts: np.ndarray) -> float:
    """The Dirichlet concentration per guide that gives the counts' spread.

    With shares from a symmetric Dirichlet of concentration ``a`` and a
    multinomial draw of the cells, ``var(count) ~ mu + mu**2 / a``.
    """
    counts = np.asarray(counts, float)
    mu, var = counts.mean(), counts.var(ddof=1) if len(counts) > 1 else 0.0
    if var <= mu or mu <= 0:
        return 1e6
    return float(mu * mu / (var - mu))


def _read_meta(path: str, columns: Sequence[str]) -> pd.DataFrame:
    """The columns of a cells_meta table that it has."""
    from .tabular import read_table

    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    return read_table(path, canonicalise=False, report=None, columns=[c for c in columns if c in names])


def _funnel_call_rate(run_dir: str) -> Optional[Dict[str, Any]]:
    """The call rate in the run's yield funnel tables, when it has any.

    :returns: ``{"value", "source", "linked", "with_guide"}`` from every
        ``funnel*.csv`` with ``linked`` and ``with_guide`` columns, or None.
    """
    from .tabular import read_table

    linked = called = 0
    files = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("funnel") and name.endswith(".csv"):
            table = read_table(os.path.join(run_dir, name), canonicalise=False, report=None)
            if {"linked", "with_guide"} <= set(table.columns):
                linked += int(table["linked"].sum())
                called += int(table["with_guide"].sum())
                files.append(_source(os.path.join(run_dir, name)))
    if not linked:
        return None
    return {"value": called / linked, "source": "funnel: with_guide / linked", "linked": linked,
            "with_guide": called, "files": files}


def _corum_sizes(path: Optional[str], genes: Sequence[str]) -> Dict[str, Any]:
    """The sizes of CORUM complexes among the screened genes.

    :param path: a ``gene_name, group`` table; None uses
        :data:`_DEFAULT_COMPLEX_SIZES`.
    :param genes: the screened genes.
    """
    if not path:
        return {"sizes": list(_DEFAULT_COMPLEX_SIZES), "source": "default"}
    from .tabular import read_table

    table = read_table(path, canonicalise=False, report=None)
    table = table[table["gene_name"].astype(str).isin(set(genes))]
    sizes = table.groupby("group")["gene_name"].nunique()
    sizes = sorted(int(s) for s in sizes if 2 <= s <= 50)
    return {"sizes": sizes or list(_DEFAULT_COMPLEX_SIZES), "source": _source(path)}


def _real_etest(cells: np.ndarray, labels: np.ndarray, guides: np.ndarray, settings: Mapping[str, Any],
                seed: int) -> Dict[str, Any]:
    """The real run's E-test in both null modes: hit share, NT error rates and the hits' effect sizes."""
    from .gene_measurement_compare import EnergyTest

    out: Dict[str, Any] = {}
    for mode in ("empirical_null", "guide_null"):
        test = EnergyTest(cells, labels, guides, control=_CONTROL, n_ref=int(settings["etest_reference_cells"]),
                          n_null=int(settings["etest_null"]), n_holdout=int(settings["etest_holdout"]), seed=seed,
                          mode=mode)
        result = test.score(None, 0)
        table = result["table"]
        genes = table[~table["gene"].str.startswith("NTguide_")]
        null = test.null(int(np.median(test.genes.map(len))))[0]
        hits = genes[genes["q_e_euclidean"] < 0.05]
        summary = {row["distance"]: {k: row[k] for k in ("genes", "hits_q05", "hit_pct", "nt_pseudo_p05_pct",
                                                        "nt_guide_p05_pct", "nt_guides", "genes_p05_pct",
                                                        "null_cells")}
                   for row in result["summary"]}
        norm2 = np.clip((hits["e_sqeuclidean"].to_numpy() - float(np.median(null[:, 0]))) / 2.0, 0.0, None)
        out[mode] = {"summary": summary, "hit_norm2": norm2.tolist(),
                     "hit_e_euclidean_ratio": (hits["e_euclidean"].to_numpy()
                                               / float(np.median(null[:, 1]))).tolist()}
    out["hit_norm2"] = out["guide_null"]["hit_norm2"]
    return out


def calibrate_from_run(run_dir: str, out: Optional[str] = None, *, phenotype_dir: Optional[str] = None,
                       call_rate: Optional[float] = None, call_rate_source: str = "",
                       fdr_pct: Optional[float] = None, fdr_source: str = "",
                       corum: Optional[str] = None, max_pool_cells: int = 50000, seed: int = 0,
                       real_etest: bool = True, report: Optional[Callable[[str], None]] = None
                       ) -> Dict[str, Any]:
    """Fill every nuisance parameter of the simulated optical pooled screen from a real run.

    The run folder holds ``cells_meta.parquet``, one row per genotyped cell
    with ``gene_symbol_0``, ``cell_barcode_0`` and, when present, ``well``
    and ``site``: the table the OPS ``etest`` phase reads. From it come the
    cells per gene and per guide, the guides per gene, the NT guides and the
    NT share of cells, the guide-abundance skew and the wells. The phenotype
    comes from ``cells_E.npy`` and ``cells_meta.parquet`` of
    ``phenotype_dir`` (the run folder when None): the NT cells' mean,
    covariance, multivariate t degrees of freedom, between-well and
    between-field variance and between-guide variance (corrected for each
    guide's sampling noise), and a pool of NT cells (at most
    ``max_pool_cells``, drawn with ``default_rng(seed)``) for the empirical
    mode. The call rate is read from the run's ``funnel*.csv`` tables
    (``with_guide / linked``) when not given.

    Every value used is written to ``out`` (JSON) and the NT pool to
    ``<out without .json>_pool.npz``, whose sha256 the JSON records.

    :param run_dir: the run folder.
    :param out: the calibration JSON; nothing is written when None.
    :param phenotype_dir: the folder of the per-cell embedding.
    :param call_rate: the genotyping call rate (0 to 1), overriding the
        funnel tables.
    :param call_rate_source: where ``call_rate`` comes from.
    :param fdr_pct: the decoy FDR of the calls, in percent; None records 0.
    :param fdr_source: where ``fdr_pct`` comes from.
    :param corum: a ``gene_name, group`` CORUM table for the complex sizes.
    :param max_pool_cells: the NT pool's cap.
    :param seed: the pool and E-test seed.
    :param real_etest: also score the real run with the E-test, in its
        ``empirical_null`` and ``guide_null`` modes, for the NT
        false-positive rates and the hits' effect sizes the validation
        compares with.
    :param report: called with progress lines.
    :returns: the calibration.
    :raises ValueError: when a table or column is missing.
    """
    say = report or (lambda text: None)
    meta_path = os.path.join(run_dir, "cells_meta.parquet")
    if not os.path.isfile(meta_path):
        raise ValueError(f"calibrate_from_run needs cells_meta.parquet in {run_dir}")
    meta = _read_meta(meta_path, ["gene_symbol_0", "cell_barcode_0", "well", "site"])
    for column in ("gene_symbol_0", "cell_barcode_0"):
        if column not in meta.columns:
            raise ValueError(f"{meta_path} lacks the column {column}")
    gene = meta["gene_symbol_0"].astype(str).to_numpy()
    guide = meta["cell_barcode_0"].astype(str).to_numpy()
    nt = np.char.startswith(gene.astype("U"), _CONTROL)
    per_gene = pd.Series(gene[~nt]).value_counts()
    per_guide = pd.Series(guide[~nt]).value_counts()
    guides_per_gene = pd.Series(guide[~nt]).groupby(gene[~nt]).nunique()
    nt_guides = pd.Series(guide[nt]).value_counts()
    wells = sorted(meta["well"].astype(str).unique()) if "well" in meta.columns else []
    say(f"calibration: {len(gene)} cells, {len(per_gene)} genes, {len(nt_guides)} NT guides")
    funnel = _funnel_call_rate(run_dir)
    if call_rate is not None:
        rate = {"value": float(call_rate), "source": call_rate_source or "given"}
    elif funnel is not None:
        rate = funnel
    else:
        rate = {"value": 1.0, "source": "no funnel table: every simulated cell is called"}
    calibration: Dict[str, Any] = {
        "schema": _CALIBRATION_SCHEMA,
        "sources": {"run_dir": os.path.abspath(run_dir), "cells_meta": _source(meta_path)},
        "cells": int(len(gene)),
        "targeting_cells": int((~nt).sum()),
        "nt_cells": int(nt.sum()),
        "nt_ratio": float(nt.sum() / max((~nt).sum(), 1)),
        "n_genes": int(len(per_gene)),
        "cells_per_gene": _quantiles(per_gene.to_numpy()),
        "cells_per_gene_counts": sorted(int(c) for c in per_gene.to_numpy()),
        "cells_per_guide": _quantiles(per_guide.to_numpy()),
        "guides_per_gene": {"median": float(guides_per_gene.median()),
                            "counts": {str(k): int(v) for k, v in guides_per_gene.value_counts().sort_index().items()}},
        "guide_alpha": _guide_alpha(per_guide.to_numpy()),
        "nt_guides": int(len(nt_guides)),
        "nt_cells_per_guide": _quantiles(nt_guides.to_numpy()),
        "wells": wells,
        "call_rate": rate,
        "fdr_pct": {"value": float(fdr_pct or 0.0), "source": fdr_source or ("given" if fdr_pct else "none given")},
    }
    from . import __version__

    calibration["spacr_version"] = __version__
    calibration["complexes"] = _corum_sizes(corum, per_gene.index.tolist())
    phenotype_dir = phenotype_dir or run_dir
    matrix = os.path.join(phenotype_dir, "cells_E.npy")
    if os.path.isfile(matrix):
        calibration.update(_calibrate_phenotype(phenotype_dir, out, max_pool_cells, seed, real_etest, say))
    else:
        say(f"calibration: no cells_E.npy in {phenotype_dir}; parametric settings left unset")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(calibration, handle, indent=1, sort_keys=True)
    return calibration


def _t_dof(mahalanobis2: np.ndarray, dims: int) -> float:
    """Multivariate t degrees of freedom matching the Mahalanobis kurtosis.

    For a d-dimensional t with ``nu`` degrees of freedom and unit scale
    matrix, ``E[m**2] / E[m]**2 = (d + 2) / d * (nu - 2) / (nu - 4)``.
    """
    m = np.asarray(mahalanobis2, float)
    ratio = float((m ** 2).mean() / m.mean() ** 2) * dims / (dims + 2)
    if ratio <= 1.0 + 1e-6:
        return 1e6
    return float(max(4.0 + 2.0 / (ratio - 1.0), 4.1))


def _calibrate_phenotype(folder: str, out: Optional[str], max_pool_cells: int, seed: int, real_etest: bool,
                         say: Callable[[str], None]) -> Dict[str, Any]:
    """The phenotype part of :func:`calibrate_from_run`."""
    matrix = os.path.join(folder, "cells_E.npy")
    meta_path = os.path.join(folder, "cells_meta.parquet")
    meta = _read_meta(meta_path, ["gene_symbol_0", "cell_barcode_0", "well", "site"])
    cells = np.load(matrix, mmap_mode="r")
    if len(cells) != len(meta):
        raise ValueError(f"{matrix} has {len(cells)} rows and {meta_path} {len(meta)}")
    gene = meta["gene_symbol_0"].astype(str).to_numpy()
    guide = meta["cell_barcode_0"].astype(str).to_numpy()
    nt_index = np.flatnonzero(np.char.startswith(gene.astype("U"), _CONTROL))
    x = np.asarray(cells[nt_index], np.float64)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    dims = x.shape[1]
    inverse = np.linalg.pinv(cov)
    centred = x - mean
    m2 = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    nu = _t_dof(m2, dims)
    out_ph: Dict[str, Any] = {
        "phenotype_sources": {"cells_E": _source(matrix), "cells_meta": _source(meta_path)},
        "dims": int(dims),
        "nt_mean": mean.tolist(),
        "nt_cov": cov.tolist(),
        "t_dof": nu,
        "t_scale_factor": float((nu - 2) / nu) if nu < 1e5 else 1.0,
    }
    for level in ("well", "site"):
        if level in meta.columns:
            keys = meta[level].astype(str).to_numpy()[nt_index]
            if level == "site" and "well" in meta.columns:
                keys = np.char.add(meta["well"].astype(str).to_numpy()[nt_index].astype("U"), keys.astype("U"))
            frame = pd.DataFrame(centred).groupby(keys).mean()
            out_ph[f"between_{level}_var"] = (frame.var(axis=0, ddof=1).fillna(0.0).to_numpy().tolist()
                                             if len(frame) > 1 else [0.0] * dims)
            out_ph[f"{level}s_seen"] = int(len(frame))
    nt_guide = guide[nt_index]
    by_guide = pd.DataFrame(centred).groupby(nt_guide)
    n_g = by_guide.size().to_numpy(float)
    keep = n_g >= 2
    guide_means = by_guide.mean().to_numpy()[keep]
    noise = (by_guide.var(ddof=1).to_numpy()[keep] / n_g[keep, None]).mean(axis=0)
    raw = guide_means.var(axis=0, ddof=1) - noise
    clipped = np.clip(raw, 0.0, None)
    total = max(float(raw.sum()), 0.0)
    out_ph["between_guide_var_raw"] = raw.tolist()
    out_ph["between_guide_var"] = (clipped * total / clipped.sum() if clipped.sum() > 0 else clipped).tolist()
    rng = np.random.default_rng(int(seed))
    pick = nt_index if len(nt_index) <= max_pool_cells else np.sort(rng.choice(nt_index, max_pool_cells, replace=False))
    pool = np.asarray(cells[pick], np.float32)
    pool_guides = guide[pick]
    out_ph["pool_cells"] = int(len(pool))
    out_ph["pool_guides"] = int(len(np.unique(pool_guides)))
    out_ph["pool_seed"] = int(seed)
    if out:
        pool_path = os.path.splitext(out)[0] + "_pool.npz"
        np.savez(pool_path, rows=pool, guides=pool_guides.astype("U"))
        out_ph["pool_file"] = os.path.abspath(pool_path)
        out_ph["pool_sha256"] = _sha256(pool_path)
    else:
        out_ph["_pool"] = (pool, pool_guides)
    say(f"calibration: NT pool {len(pool)} cells in {out_ph['pool_guides']} guides, {dims} dims, t dof {nu:.1f}")
    if real_etest:
        settings = {k: OPS_POWER_DESIGN[k] for k in ("etest_reference_cells", "etest_null", "etest_holdout")}
        say("calibration: scoring the real run with the E-test")
        out_ph["real_etest"] = _real_etest(np.asarray(cells), gene, guide, settings, int(seed))
        out_ph["real_etest"]["settings"] = settings
    return out_ph


def load_calibration(path: str) -> Dict[str, Any]:
    """Read a calibration JSON written by :func:`calibrate_from_run`.

    The NT pool is loaded from its ``_pool.npz`` and checked against the
    sha256 the JSON records.

    :param path: the JSON, or a run folder to calibrate on the fly (nothing
        written).
    :returns: the calibration.
    :raises ValueError: when the pool file changed since the calibration.
    """
    if os.path.isdir(path):
        return calibrate_from_run(path)
    with open(path, encoding="utf-8") as handle:
        calibration = json.load(handle)
    pool_file = calibration.get("pool_file")
    if pool_file:
        if not os.path.isabs(pool_file) or not os.path.isfile(pool_file):
            pool_file = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(pool_file))
        if _sha256(pool_file) != calibration.get("pool_sha256"):
            raise ValueError(f"{pool_file} changed since the calibration was written")
        with np.load(pool_file) as data:
            calibration["_pool"] = (data["rows"], data["guides"])
    return calibration


def resolve_design(calibration: Mapping[str, Any], design: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One full design: the defaults, then the calibration, then ``design``.

    :param calibration: from :func:`calibrate_from_run`.
    :param design: the design keys to set (see :data:`OPS_POWER_DESIGN`).
    :returns: every design key with a value.
    :raises ValueError: for an unknown key or effect type.
    """
    design = dict(design or {})
    unknown = sorted(set(design) - set(OPS_POWER_DESIGN))
    if unknown:
        raise ValueError(f"unknown OPS power design keys {unknown}; choose from {sorted(OPS_POWER_DESIGN)}")
    out = dict(OPS_POWER_DESIGN)
    fill = {
        "n_genes": calibration.get("n_genes"),
        "guides_per_gene": int(round((calibration.get("guides_per_gene") or {}).get("median") or 4)),
        "nt_guides": calibration.get("nt_guides"),
        "cells_per_gene": (calibration.get("cells_per_gene") or {}).get("q50"),
        "call_rate": (calibration.get("call_rate") or {}).get("value", 1.0),
        "fdr_pct": (calibration.get("fdr_pct") or {}).get("value", 0.0),
        "wells": max(len(calibration.get("wells") or []), 1),
        "fields_per_well": 0,
        "guide_alpha": calibration.get("guide_alpha", 1e6),
    }
    for key, value in fill.items():
        if out[key] is None:
            out[key] = value
    out.update({k: v for k, v in design.items() if v is not None})
    if out["effect_type"] not in ("mean", "variance", "subpopulation"):
        raise ValueError(f"effect_type must be mean, variance or subpopulation; got {out['effect_type']!r}")
    if out["mode"] not in ("empirical", "parametric"):
        raise ValueError(f"mode must be empirical or parametric; got {out['mode']!r}")
    if out["effect_block"] not in ("any", "new"):
        raise ValueError(f"effect_block must be any or new; got {out['effect_block']!r}")
    for key in ("n_genes", "guides_per_gene", "nt_guides", "wells", "effect_dims"):
        out[key] = int(out[key])
    return out


def _design_key(design: Mapping[str, Any]) -> str:
    """A digest of a design's values, the cache key of its rows."""
    payload = json.dumps({k: design[k] for k in sorted(design)}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


#: Design keys that change only the analysis, not the simulated screen.
_ANALYSIS_KEYS = ("etest_metric", "etest_mode", "etest_reference_cells", "etest_null", "etest_holdout",
                  "cluster_resolution", "cluster_min_cells", "info_permutations", "info_null")


def _rng_for(design: Mapping[str, Any], seed: int) -> np.random.Generator:
    """The generator of one replicate: the seed and the design's simulation keys.

    The analysis keys are left out, so every scorer setting sees the same
    screen.
    """
    keyed = {k: v for k, v in design.items() if k not in _ANALYSIS_KEYS}
    return resolve_rng(seed=np.random.SeedSequence([int(seed), int(_design_key(keyed), 16)]).generate_state(1)[0])


def _phenotype_base(calibration: Mapping[str, Any], design: Mapping[str, Any], n: int, cell_guide: np.ndarray,
                    n_guides: int, well: np.ndarray, field: np.ndarray, rng: np.random.Generator,
                    guide_offsets: Optional[np.ndarray] = None) -> np.ndarray:
    """The unperturbed phenotype of every simulated cell.

    Every simulated guide gets a normal offset with the NT guides'
    between-guide variance, corrected for their sampling noise
    (``between_guide_var``). Empirical: the
    within-guide residuals of the pooled NT cells, every cell drawn from the
    whole pool with a variance-preserving smoothed bootstrap,
    ``(r + h z) / sqrt(1 + h**2)`` with ``z`` drawn from the within-guide
    covariance and ``h`` the bandwidth. Resampled cells are never exact
    copies, and the means and shapes of the pool's few cells per guide do
    not act as guide effects (drawing a guide's cells from one real NT
    guide made null genes reach p < 0.05 three to six times too often). Parametric: a multivariate
    t with the NT mean, covariance and degrees of freedom, plus normal well
    and field offsets with the calibrated between-well and between-field
    variances times ``batch_scale``.
    """
    mean = np.asarray(calibration["nt_mean"], float)
    cov = np.asarray(calibration["nt_cov"], float)
    dims = len(mean)
    chol = np.linalg.cholesky(cov + 1e-9 * np.trace(cov) / dims * np.eye(dims))
    tau = np.sqrt(np.asarray(calibration.get("between_guide_var") or np.zeros(dims), float))
    if design["mode"] == "empirical":
        rows, guides = calibration["_pool"]
        names, inverse = np.unique(guides, return_inverse=True)
        rows = np.asarray(rows, float)
        sums = np.zeros((len(names), dims))
        np.add.at(sums, inverse, rows)
        residuals = rows - (sums / np.bincount(inverse, minlength=len(names))[:, None])[inverse]
        within = np.cov(residuals, rowvar=False)
        chol_w = np.linalg.cholesky(within + 1e-9 * np.trace(within) / dims * np.eye(dims))
        pick = rng.integers(0, len(rows), size=n)
        h = float(design["bandwidth"])
        offsets = (guide_offsets if guide_offsets is not None else rng.standard_normal((n_guides, dims)) * tau)
        z = rng.standard_normal((n, dims)) @ chol_w.T
        base = mean + offsets[cell_guide] + (residuals[pick] + h * z) / math.sqrt(1.0 + h * h)
        return base.astype(np.float32)
    nu = float(calibration.get("t_dof") or 1e6)
    scale = float(calibration.get("t_scale_factor") or 1.0)
    z = rng.standard_normal((n, dims)) @ chol.T * math.sqrt(scale)
    if nu < 1e5:
        z = z / np.sqrt(rng.chisquare(nu, size=n) / nu)[:, None]
    base = mean + z + (guide_offsets if guide_offsets is not None
                       else rng.standard_normal((n_guides, dims)) * tau)[cell_guide]
    for level, keys, count in (("well", well, int(design["wells"])),
                               ("site", field, int(design["wells"]) * max(int(design["fields_per_well"]), 1))):
        var = calibration.get(f"between_{level}_var")
        if var is None or (level == "site" and not int(design["fields_per_well"])):
            continue
        offsets = rng.standard_normal((count, dims)) * np.sqrt(np.asarray(var, float) * float(design["batch_scale"]))
        base = base + offsets[keys]
    return base.astype(np.float32)


def _effect_plan(calibration: Mapping[str, Any], d: Mapping[str, Any], rng: np.random.Generator,
                 n_genes: int, dims: int) -> Dict[str, Any]:
    """The planted biology of one screen: hit genes, complexes, sizes and directions.

    The draws are those :func:`simulate_ops_screen` makes after the cells'
    phenotype, in the same order.
    """
    sd = np.sqrt(np.diag(np.asarray(calibration["nt_cov"], float)))
    mean = np.asarray(calibration["nt_mean"], float)
    n_hits = int(round(float(d["hit_fraction"]) * n_genes))
    hit_genes = np.sort(rng.choice(n_genes, n_hits, replace=False)) if n_hits else np.array([], int)
    k = min(int(d["effect_dims"]), dims)
    block = np.arange(dims - int(d["info_new_dims"]), dims) if d["effect_block"] == "new" else np.arange(dims)
    k = min(k, len(block))
    complex_of = np.full(n_genes, -1)
    in_complex = rng.permutation(hit_genes)[: int(round(float(d["complex_fraction"]) * n_hits))]
    sizes = list((calibration.get("complexes") or {}).get("sizes") or _DEFAULT_COMPLEX_SIZES)
    start, label = 0, 0
    while start < len(in_complex) - 1:
        size = int(sizes[rng.integers(0, len(sizes))])
        members = in_complex[start:start + size]
        if len(members) >= 2:
            complex_of[members] = label
            label += 1
        start += size
    shared = {c: rng.standard_normal(k) for c in range(label)}
    shared_dims = {c: np.sort(rng.choice(block, k, replace=False)) for c in range(label)}
    size_of = np.zeros(n_genes)
    direction: Dict[int, tuple] = {}
    norm2 = (calibration.get("real_etest") or {}).get("hit_norm2") or []
    for g in hit_genes:
        c = complex_of[g]
        if c >= 0:
            v = shared[c] + float(d["complex_noise"]) * rng.standard_normal(k)
            touched = shared_dims[c]
        else:
            v = rng.standard_normal(k)
            touched = np.sort(rng.choice(block, k, replace=False))
        v = v / max(np.linalg.norm(v), 1e-12) * math.sqrt(k)
        if d["effect_size"] == "calibrated" and len(norm2):
            magnitude = float(norm2[rng.integers(0, len(norm2))])
            s = math.sqrt(magnitude / max(float((sd[touched] ** 2).sum()), 1e-12))
        else:
            s = float(d["effect_size"])
        size_of[g] = s
        direction[int(g)] = (touched, np.abs(v), np.sign(v))

    hit_mask = np.zeros(n_genes, bool)
    hit_mask[hit_genes] = True
    return {"sd": sd, "mean": mean, "hit_genes": hit_genes, "hit_mask": hit_mask, "complex_of": complex_of,
            "size_of": size_of, "direction": direction}


def _guide_knockout(d: Mapping[str, Any], n_target_guides: int, rng: np.random.Generator) -> np.ndarray:
    """The knockout penetrance of every targeting guide (0 for inactive guides)."""
    guide_ko = np.ones(n_target_guides)
    inactive = rng.random(n_target_guides) < float(d["inactive_guide_fraction"])
    if float(d["ko_penetrance_sd"]) > 0 and 0 < float(d["ko_penetrance"]) < 1:
        guide_ko = rbeta_mean_variance(n_target_guides, float(d["ko_penetrance"]),
                                       float(d["ko_penetrance_sd"]) ** 2, rng=rng)
    else:
        guide_ko = guide_ko * float(d["ko_penetrance"])
    guide_ko[inactive] = 0.0
    return guide_ko


def _apply_effects(cells: np.ndarray, d: Mapping[str, Any], plan: Mapping[str, Any], true_gene: np.ndarray,
                   cell_guide: np.ndarray, guide_ko: np.ndarray, rng: np.random.Generator) -> None:
    """Shift the responding cells of every hit gene in place, as the plan says."""
    sd, mean, direction = plan["sd"], plan["mean"], plan["direction"]
    hit_mask, size_of = plan["hit_mask"], plan["size_of"]
    affected = np.flatnonzero((true_gene >= 0) & hit_mask[np.clip(true_gene, 0, None)])
    p_respond = float(d["penetrance"]) * guide_ko[cell_guide[affected]]
    responders = affected[rng.random(len(affected)) < p_respond]
    for g, rows in pd.Series(responders).groupby(true_gene[responders]):
        touched, weight, sign = direction[int(g)]
        rows = rows.to_numpy()
        shift = size_of[int(g)] * sd[touched] * weight * sign
        block_rows = cells[np.ix_(rows, touched)].astype(float)
        if d["effect_type"] == "mean":
            block_rows = block_rows + shift
        elif d["effect_type"] == "variance":
            block_rows = mean[touched] + (block_rows - mean[touched]) * np.sqrt(1.0 + size_of[int(g)] * weight)
        else:
            block_rows = mean[touched] + shift + 0.5 * (block_rows - mean[touched])
        cells[np.ix_(rows, touched)] = block_rows.astype(np.float32)


def simulate_ops_screen(calibration: Mapping[str, Any], design: Optional[Mapping[str, Any]] = None,
                        seed: int = 0) -> Dict[str, Any]:
    """Draw one simulated optical pooled screen with planted effects.

    :param calibration: from :func:`calibrate_from_run` or
        :func:`load_calibration`; the empirical mode needs its NT pool.
    :param design: the design (see :func:`resolve_design`).
    :param seed: the replicate seed; with the design's values it fixes every
        draw, so the same design and seed give the same simulated screen.
    :returns: ``{"cells", "labels", "guides", "wells", "truth", "design"}``:
        the per-cell phenotype (float32), each cell's observed gene and
        guide (NT cells ``nontargeting``), its well, and one row per gene
        with ``gene``, ``hit``, ``complex``, ``effect_size`` and ``cells``.
    """
    d = resolve_design(calibration, design)
    rng = _rng_for(d, seed)
    dims = int(calibration["dims"])
    n_genes, per_gene, n_nt = d["n_genes"], d["guides_per_gene"], d["nt_guides"]
    n_target_guides = n_genes * per_gene
    n_guides = n_target_guides + n_nt
    guide_gene = np.concatenate([np.repeat(np.arange(n_genes), per_gene), np.full(n_nt, -1)])
    cells_per_gene = float(d["cells_per_gene"]) * float(d["call_rate"]) / float(
        (calibration.get("call_rate") or {}).get("value", 1.0) or 1.0)
    n_target = int(round(cells_per_gene * n_genes))
    screen_genes = int(calibration.get("n_genes") or n_genes)
    n_ntc = int(round(cells_per_gene * screen_genes * float(calibration.get("nt_ratio", 0.02))))
    shares = rdirichlet_stable(float(d["guide_alpha"]), n_target_guides, rng=rng)
    counts = rng.multinomial(n_target, shares)
    if n_nt:
        counts = np.concatenate([counts, rng.multinomial(n_ntc, rdirichlet_stable(float(d["guide_alpha"]), n_nt, rng=rng))])
    cell_guide = np.repeat(np.arange(n_guides), counts)
    n = len(cell_guide)
    well = rng.integers(0, int(d["wells"]), size=n)
    field = well * max(int(d["fields_per_well"]), 1) + rng.integers(0, max(int(d["fields_per_well"]), 1), size=n)
    cells = _phenotype_base(calibration, d, n, cell_guide, n_guides, well, field, rng)

    plan = _effect_plan(calibration, d, rng, n_genes, dims)
    hit_genes, complex_of, size_of = plan["hit_genes"], plan["complex_of"], plan["size_of"]
    guide_ko = _guide_knockout(d, n_target_guides, rng)
    true_gene = guide_gene[cell_guide]
    hit_mask = plan["hit_mask"]
    _apply_effects(cells, d, plan, true_gene, cell_guide, guide_ko, rng)

    observed = cell_guide.copy()
    wrong = rng.random(n) < float(d["fdr_pct"]) / 100.0
    if wrong.any():
        weights = counts / counts.sum()
        observed[wrong] = rng.choice(n_guides, int(wrong.sum()), p=weights)
    gene_names = np.array([f"G{g:05d}" for g in range(n_genes)])
    obs_gene = guide_gene[observed]
    labels = np.where(obs_gene >= 0, gene_names[np.clip(obs_gene, 0, None)], _CONTROL)
    guide_names = np.where(obs_gene >= 0, np.char.add(np.char.add(labels.astype("U"), "_"),
                                                        (observed % max(per_gene, 1)).astype("U")),
                           np.char.add("NT", (observed - n_target_guides).astype("U")))
    truth = pd.DataFrame({"gene": gene_names, "hit": hit_mask, "complex": complex_of, "effect_size": size_of,
                          "cells": np.bincount(obs_gene[obs_gene >= 0], minlength=n_genes)})
    return {"cells": cells, "labels": labels.astype(str), "guides": guide_names.astype(str), "wells": well,
            "truth": truth, "design": d}


def nominal_calibration(dims: int = 24, n_genes: int = 1000, guides_per_gene: int = 4, nt_guides: int = 50,
                        cells_per_gene: float = 100.0, call_rate: float = 0.8, fdr_pct: float = 2.0,
                        pool_cells: int = 20000, seed: int = 0) -> Dict[str, Any]:
    """A calibration that needs no real run, for benchmarks and design scans.

    It has every key :func:`simulate_ops_screen` reads, with nominal values:
    a correlated NT covariance, a multivariate t, between-guide, between-well
    and between-field variances, a pool of NT cells drawn from it and a
    moderately skewed guide library. A real calibration
    (:func:`calibrate_from_run`) replaces it wherever the numbers matter.

    :param dims: the dimensions of the latent phenotype profile.
    :param n_genes: the genes of the library.
    :param guides_per_gene: the guides per gene.
    :param nt_guides: the NT guides.
    :param cells_per_gene: the median genotyped cells per gene.
    :param call_rate: the genotyping call rate.
    :param fdr_pct: the decoy FDR of the calls, in percent.
    :param pool_cells: the size of the NT pool of the empirical mode.
    :param seed: the seed of the covariance and the pool.
    :returns: the calibration.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4242]))
    a = rng.standard_normal((dims, dims))
    cov = a @ a.T / dims + 0.5 * np.eye(dims)
    chol = np.linalg.cholesky(cov)
    pool = rng.standard_normal((pool_cells, dims)) @ chol.T
    names = np.array([f"{_CONTROL}_{i % max(nt_guides, 1)}" for i in range(pool_cells)])
    ones = np.ones(dims)
    return {"schema": _CALIBRATION_SCHEMA, "source": "nominal", "dims": int(dims), "nt_mean": np.zeros(dims).tolist(),
            "nt_cov": cov.tolist(), "between_guide_var": (0.05 * ones).tolist(),
            "between_well_var": (0.03 * ones).tolist(), "between_site_var": (0.02 * ones).tolist(),
            "t_dof": 8.0, "t_scale_factor": 0.9, "n_genes": int(n_genes),
            "guides_per_gene": {"median": float(guides_per_gene)}, "nt_guides": int(nt_guides),
            "cells_per_gene": {"q50": float(cells_per_gene)}, "call_rate": {"value": float(call_rate)},
            "fdr_pct": {"value": float(fdr_pct)}, "wells": ["A01"], "guide_alpha": 50.0, "nt_ratio": 0.05,
            "complexes": {"sizes": list(_DEFAULT_COMPLEX_SIZES)}, "_pool": (pool, names)}


#: The object tables of a streamed screen and the object class each column name starts with.
_STREAM_TABLES = ("cell", "nucleus", "cytoplasm")

#: The columns every table of a streamed screen carries beside its features.
_STREAM_KEYS = ("object_key", "timeID", "fieldID", "object_label", "rowID", "columnID", "gene", "guide",
                "condition", "prcfo")

#: Cells x features held at once by one batch of the streamed writer.
_STREAM_CELLS = 40_000_000


def _stream_features(n_features: int) -> Dict[str, List[Dict[str, Any]]]:
    """The feature columns of each table: A1's names and recorded kinds, evenly thinned.

    ``n_features`` columns are taken from the recorded dtype map's 2,798
    A1 columns, each object class in its A1 share and evenly spaced, so
    the names, the float16/float32/integer mix and the family structure are
    those of a real run; beyond 2,798 the classes get ``<class>_extra_<k>``
    float32 columns.
    """
    from . import store

    recorded = store._dtype_map().get("columns") or {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    held = {t: sorted(n for n in recorded if n.split("_", 1)[0] == t) for t in _STREAM_TABLES}
    total = sum(len(v) for v in held.values())
    left = int(n_features)
    for position, table in enumerate(_STREAM_TABLES):
        last = position == len(_STREAM_TABLES) - 1
        take = left if last else int(round(n_features * len(held[table]) / total))
        left -= take
        names = held[table]
        picked = ([names[int(i)] for i in np.linspace(0, len(names) - 1, min(take, len(names))).round()]
                  if names else [])
        picked = list(dict.fromkeys(picked))
        specs = [{"name": n, "kind": recorded[n]["kind"], "dtype": recorded[n]["dtype"]} for n in picked]
        specs += [{"name": f"{table}_extra_{k}", "kind": "float", "dtype": "float32"}
                  for k in range(max(take - len(specs), 0))]
        out[table] = specs
    return out


def _stream_arrow(name: str, values: np.ndarray, spec: Mapping[str, Any]):
    """One feature column as Arrow in the type the source holds it in."""
    import pyarrow as pa

    if spec["kind"] == "int":
        return pa.array(np.clip(np.rint(values), 0, 200).astype(np.uint8))
    return pa.array(values.astype(np.float32))


def write_ops_screen(calibration: Mapping[str, Any], project: str, design: Optional[Mapping[str, Any]] = None,
                     seed: int = 0, *, plates: int = 1, wells: int = 8, fields: int = 10,
                     cells_per_field: int = 125, n_features: int = 300, batch_cells: int = 25_000,
                     only: Optional[Sequence[str]] = None, run_id: Optional[str] = None,
                     row_copy: bool = False) -> Dict[str, Any]:
    """Stream a simulated optical pooled screen into spaCR's store, well by well.

    The cells of :func:`simulate_ops_screen`'s model (guides in genes,
    planted hit genes and complexes, the genotyping error rate, a latent
    phenotype from the calibrated NT cells) are generated one well at a
    time, in batches of at most ``batch_cells`` cells, and appended to
    :func:`spacr.store.writer` with the recorded dtype map, so a screen of
    100 million cells needs the memory of one batch. The latent profile is
    mixed into ``n_features`` columns carrying A1's column names, kinds and
    dtypes (bounded shape-like features, log-normal intensity-like ones and
    a few small counts), split into ``cell``, ``nucleus`` and
    ``cytoplasm`` tables that share ``object_key`` and row order.

    Every table carries ``object_key`` (:func:`spacr.ops_run_infra.integer_cell_key`),
    ``timeID``, ``fieldID``, ``object_label``, ``rowID``, ``columnID``,
    ``gene``, ``guide`` (as observed, with the genotyping error),
    ``condition`` and ``prcfo``; ``nucleus`` and ``cytoplasm`` add
    ``parent_cell_key``. The library, the planted effects and the feature
    mixing come from ``seed`` and the design; each well has its own stream
    from the seed, so the same arguments give the same bytes, and writing a
    subset of wells with ``only`` gives those wells exactly as writing all
    of them does. The manifests are left for :func:`spacr.store.register`.

    :param calibration: from :func:`calibrate_from_run`,
        :func:`load_calibration` or :func:`nominal_calibration`.
    :param project: the store project folder.
    :param design: the design (see :func:`resolve_design`).
    :param seed: the screen seed.
    :param plates: the plates.
    :param wells: the wells per plate (24 per row, from A01).
    :param fields: the fields per well.
    :param cells_per_field: the cells per field.
    :param n_features: the feature columns over the three tables.
    :param batch_cells: the most cells generated at once (lowered so that
        a batch holds at most 40 million values).
    :param only: ``"plate01/A01"`` style names of the wells to write; None
        writes every well.
    :param run_id: the run id the store records.
    :param row_copy: also write the row-major copy of the float columns.
    :returns: ``{"wells": [[plate, well], ...], "cells", "features", "truth"}``
        with the per-gene truth table (``gene``, ``hit``, ``complex``,
        ``effect_size``).
    """
    import pyarrow as pa
    from contextlib import ExitStack

    from . import store
    from .ops_run_infra import integer_cell_key

    d = resolve_design(calibration, design)
    keyed = {k: v for k, v in d.items() if k not in _ANALYSIS_KEYS}
    stream = [int(seed), int(_design_key(keyed), 16)]
    plan_rng = np.random.default_rng(np.random.SeedSequence(stream + [1]))
    dims = int(calibration["dims"])
    n_genes, per_gene, n_nt = d["n_genes"], d["guides_per_gene"], d["nt_guides"]
    n_target_guides = n_genes * per_gene
    n_guides = n_target_guides + n_nt
    guide_gene = np.concatenate([np.repeat(np.arange(n_genes), per_gene), np.full(n_nt, -1)])
    nt_share = float(calibration.get("nt_ratio", 0.02))
    share = rdirichlet_stable(float(d["guide_alpha"]), n_target_guides, rng=plan_rng) * (1.0 - nt_share)
    if n_nt:
        share = np.concatenate([share, rdirichlet_stable(float(d["guide_alpha"]), n_nt, rng=plan_rng) * nt_share])
    cumulative = np.cumsum(share / share.sum())
    plan = _effect_plan(calibration, d, plan_rng, n_genes, dims)
    guide_ko = _guide_knockout(d, n_target_guides, plan_rng)
    tau = np.sqrt(np.asarray(calibration.get("between_guide_var") or np.zeros(dims), float))
    offsets = plan_rng.standard_normal((n_guides, dims)) * tau
    specs = _stream_features(n_features)
    width = sum(len(v) for v in specs.values())
    loading = plan_rng.standard_normal((dims, width))
    loading /= np.linalg.norm(loading, axis=0)
    kinds = np.array([s["kind"] == "int" for v in specs.values() for s in v])
    bounded = np.array([s["dtype"] == "float16" for v in specs.values() for s in v])
    mean_latent = np.asarray(calibration["nt_mean"], float)
    sd_latent = np.sqrt(np.diag(np.asarray(calibration["nt_cov"], float)))
    gene_names = np.array([f"G{g:05d}" for g in range(n_genes)])
    batch = max(1000, min(int(batch_cells), _STREAM_CELLS // max(width, 1)))
    grid = []
    for p in range(int(plates)):
        for i in range(int(wells)):
            r, c = i // 24 + 1, i % 24 + 1
            grid.append((p, f"plate{p + 1:02d}", f"{chr(64 + r)}{c:02d}", r, c, i))
    if only is not None:
        wanted = set(only)
        grid = [g for g in grid if f"{g[1]}/{g[2]}" in wanted]
    ranges = {s["name"]: ((0.0, 200.0) if s["kind"] == "int" else (-0.7, 1.7)) for v in specs.values() for s in v}
    per_well = int(fields) * int(cells_per_field)
    chunk_design = dict(d, wells=1, fields_per_well=int(fields))
    for p, plate, well, r, c, i in grid:
        rng = np.random.default_rng(np.random.SeedSequence(stream + [2, p, i]))
        with ExitStack() as stack:
            writers = {}
            for start in range(0, per_well, batch):
                m = min(batch, per_well - start)
                index = np.arange(start, start + m)
                field = (index // int(cells_per_field) + 1).astype(np.uint32)
                label = (index % int(cells_per_field) + 1).astype(np.uint32)
                key = integer_cell_key(r, c, field, label)
                cell_guide = np.minimum(np.searchsorted(cumulative, rng.random(m)), n_guides - 1)
                latent = _phenotype_base(calibration, chunk_design, m, cell_guide, n_guides, np.zeros(m, int),
                                         field.astype(int) - 1, rng, guide_offsets=offsets)
                true_gene = guide_gene[cell_guide]
                _apply_effects(latent, d, plan, true_gene, cell_guide, guide_ko, rng)
                observed = cell_guide.copy()
                wrong = rng.random(m) < float(d["fdr_pct"]) / 100.0
                if wrong.any():
                    observed[wrong] = np.minimum(np.searchsorted(cumulative, rng.random(int(wrong.sum()))),
                                                 n_guides - 1)
                obs_gene = guide_gene[observed]
                genes = np.where(obs_gene >= 0, gene_names[np.clip(obs_gene, 0, None)], _CONTROL)
                guides = np.where(obs_gene >= 0, np.char.add(np.char.add(genes.astype("U"), "_"),
                                                              (observed % max(per_gene, 1)).astype("U")),
                                  np.char.add("NT", (observed - n_target_guides).astype("U")))
                z = (latent.astype(np.float32) - mean_latent.astype(np.float32)) / sd_latent.astype(np.float32)
                mixed = z @ loading.astype(np.float32)
                noise = rng.standard_normal(mixed.shape, dtype=np.float32)
                x = (np.sqrt(0.5) * mixed + np.sqrt(0.5) * noise).astype(np.float32)
                values = np.where(bounded, 0.5 + 0.15 * np.clip(x, -8, 8), np.exp(0.5 * x + 6.0))
                values = np.where(kinds, np.clip(np.rint(np.exp(0.5 * x + 1.5)), 0, 200), values)
                base = {"object_key": pa.array(key, pa.int64()), "timeID": pa.array(np.zeros(m, np.uint16)),
                        "fieldID": pa.array(field), "object_label": pa.array(label),
                        "rowID": pa.array(np.full(m, r, np.uint8)), "columnID": pa.array(np.full(m, c, np.uint8)),
                        "gene": pa.array(genes.astype(str), pa.string()),
                        "guide": pa.array(guides.astype(str), pa.string()),
                        "condition": pa.array([f"treat{(r * 24 + c) % 4}"] * m, pa.string()),
                        "prcfo": pa.array([f"{plate}_{r}_{c}_{f}_{lb}"
                                           for f, lb in zip(field.tolist(), label.tolist())], pa.string())}
                offset = 0
                for table in _STREAM_TABLES:
                    columns = dict(base)
                    if table != "cell":
                        columns["parent_cell_key"] = base["object_key"]
                    for j, spec in enumerate(specs[table]):
                        columns[spec["name"]] = _stream_arrow(spec["name"], values[:, offset + j], spec)
                    offset += len(specs[table])
                    frame = pa.table(columns)
                    if table not in writers:
                        writers[table] = stack.enter_context(store.writer(
                            project, table, plate=plate, well=well, run_id=run_id,
                            sort_by=("object_key", "timeID"), key_columns=_STREAM_KEYS, row_copy=row_copy,
                            schema=frame.schema, ranges=ranges, narrow=True))
                    writers[table].append(frame)
    truth = pd.DataFrame({"gene": gene_names, "hit": plan["hit_mask"], "complex": plan["complex_of"],
                          "effect_size": plan["size_of"]})
    return {"wells": [[g[1], g[2]] for g in grid], "cells": len(grid) * per_well, "features": width,
            "truth": truth}


def _wilson(k: int, n: int) -> tuple:
    """The 95 % Wilson interval of ``k`` successes in ``n``."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z, p = 1.959963984540054, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p)))


def _score_etest(screen: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """Per-gene E-tests against NT with BH, as the OPS etest phase runs them."""
    from .gene_measurement_compare import EnergyTest

    d = screen["design"]
    nt = int(np.sum(screen["labels"] == _CONTROL))
    n_ref = min(int(d["etest_reference_cells"]), max(nt // 2, 2))
    test = EnergyTest(screen["cells"], screen["labels"], screen["guides"], control=_CONTROL, n_ref=n_ref,
                      n_null=int(d["etest_null"]), n_holdout=int(d["etest_holdout"]), seed=int(seed),
                      mode=str(d["etest_mode"]))
    result = test.score(None, 0)
    table = result["table"]
    metric = f"e_{d['etest_metric']}"
    genes = table[~table["gene"].str.startswith("NTguide_")].merge(screen["truth"][["gene", "hit"]], on="gene")
    detected = genes[f"q_{metric}"] < 0.05
    hits, nulls = genes["hit"].to_numpy(), ~genes["hit"].to_numpy()
    summary = next(row for row in result["summary"] if row["distance"] == d["etest_metric"])
    tp, fp = int((detected & hits).sum()), int((detected & nulls).sum())
    planted = int(screen["truth"]["hit"].sum())
    return {"etest_power": tp / planted if planted else float("nan"), "etest_detected": tp, "etest_planted": planted,
            "etest_null_genes": int(nulls.sum()), "etest_null_q05": fp,
            "etest_fpr_null_genes": fp / max(int(nulls.sum()), 1),
            "etest_fdp": fp / max(tp + fp, 1), "nt_pseudo_p05_pct": summary["nt_pseudo_p05_pct"],
            "nt_guide_p05_pct": summary["nt_guide_p05_pct"], "etest_reference_cells": n_ref,
            "_table": genes}


def _guide_agreement(screen: Mapping[str, Any]) -> Dict[str, Any]:
    """The mean pairwise cosine of a gene's guide medians, minus the NT median."""
    cells, labels, guides = screen["cells"], screen["labels"], screen["guides"]
    nt = labels == _CONTROL
    centre = np.median(cells[nt], axis=0)
    frame = pd.DataFrame(cells[~nt] - centre)
    medians = frame.groupby(guides[~nt]).median()
    owner = pd.Series(labels[~nt]).groupby(guides[~nt]).first().reindex(medians.index)
    unit = medians.to_numpy() / np.clip(np.linalg.norm(medians.to_numpy(), axis=1, keepdims=True), 1e-12, None)
    values = {}
    for gene, index in pd.Series(np.arange(len(owner))).groupby(owner.to_numpy()):
        rows = unit[index.to_numpy()]
        if len(rows) >= 2:
            sim = rows @ rows.T
            values[gene] = float(sim[np.triu_indices(len(rows), 1)].mean())
    truth = screen["truth"].set_index("gene")["hit"]
    agree = pd.Series(values)
    hit = truth.reindex(agree.index).fillna(False).astype(bool)
    return {"guide_agreement_hits": float(agree[hit].mean()) if hit.any() else float("nan"),
            "guide_agreement_null": float(agree[~hit].mean()) if (~hit).any() else float("nan")}


def _score_clustering(screen: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """Hybrid profiles, PHATE + Leiden and complex recovery against the planted complexes.

    The gene rows are the 10 % trimmed means of each gene's cells (the
    profile phase's aggregation), embedded with
    :func:`spacr.ops_reference.hybrid_profile_rows` and clustered with
    :func:`spacr.ops_clusters.phate_leiden`; recovery is scored with
    :func:`spacr.ops_clusters.group_enrichment` on the planted complexes, and
    on the same complexes with shuffled members (the control).
    """
    from .ops_clusters import group_enrichment, phate_leiden
    from .ops_reference import hybrid_profile_rows
    from .ops_tvn import _trimmed_means

    d = screen["design"]
    labels = screen["labels"]
    columns = [f"PC_{j}" for j in range(screen["cells"].shape[1])]
    frame = pd.DataFrame(screen["cells"], columns=columns)
    frame["gene"] = labels
    grouped = frame.groupby("gene", sort=True)
    sizes = grouped.size()
    means = _trimmed_means(grouped, frame, columns)
    genes = means.reset_index().rename(columns={"index": "gene"})
    genes.insert(1, "n_cells", sizes.reindex(means.index).to_numpy())
    genes = genes[genes["n_cells"] > int(d["cluster_min_cells"])].reset_index(drop=True)
    rows, _ = hybrid_profile_rows(genes, seed=int(seed))
    names = rows["gene"].astype(str).tolist()
    embedded = rows[[c for c in rows.columns if c.startswith("E_")]].to_numpy(float)
    clusters = phate_leiden(embedded, [float(d["cluster_resolution"])], seed=42 + int(seed), n_jobs=1)
    labels_c = clusters[float(d["cluster_resolution"])]
    truth = screen["truth"]
    planted = truth[truth["complex"] >= 0]
    groups = pd.DataFrame({"gene_name": planted["gene"], "group": "C" + planted["complex"].astype(str)})
    rng = np.random.default_rng(1000 + int(seed))
    pool = np.array([g for g in names if g != _CONTROL])
    shuffled = groups.assign(gene_name=rng.choice(pool, len(groups), replace=False)) if len(groups) <= len(pool) else groups
    out = {"clusters": int(len(np.unique(labels_c))), "complexes_planted": int(groups["group"].nunique())}
    for tag, table in (("", groups), ("_shuffled", shuffled)):
        if not len(table):
            out[f"complex_recovery{tag}"] = float("nan")
            out[f"cluster_enriched_pct{tag}"] = float("nan")
            continue
        share, per_cluster = group_enrichment(table, names, labels_c, control_prefix=_CONTROL)
        found = set()
        for text in per_cluster["enriched"]:
            found.update(t for t in str(text).split("; ") if t)
        out[f"complex_recovery{tag}"] = len(found) / table["group"].nunique()
        out[f"cluster_enriched_pct{tag}"] = 100 * share
    where = {g: c for g, c in zip(names, labels_c)}
    pairs_true = {tuple(sorted((a, b))) for _, m in groups.groupby("group") for a in m.gene_name for b in m.gene_name if a < b}
    members = [g for g in planted["gene"] if g in where]
    pairs_found = {tuple(sorted((a, b))) for i, a in enumerate(members) for b in members[i + 1:] if where[a] == where[b]}
    hit_names = set(truth.loc[truth["hit"], "gene"])
    by_cluster = pd.Series(list(where.keys())).groupby(pd.Series(list(where.values())))
    co = set()
    for _, group in by_cluster:
        g = sorted(x for x in group if x in hit_names)
        co.update((a, b) for i, a in enumerate(g) for b in g[i + 1:])
    tp = len(pairs_true & co)
    precision = tp / len(co) if co else float("nan")
    recall = tp / len(pairs_true) if pairs_true else float("nan")
    out["pair_precision"] = precision
    out["pair_recall"] = recall
    out["pair_f1"] = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    out["planted_pairs_coclustered"] = len(pairs_found)
    return out


def conditional_information(old: np.ndarray, new: np.ndarray, labels: Sequence, *, ridge: float = 1e-6) -> float:
    """Gaussian conditional mutual information ``I(gene; new | old)``, in bits.

    ``1/2 log det S(new | old) - 1/2 log det S(new | old, gene)``: the
    Schur complement of the total covariance of ``[old, new]`` against that
    of the pooled within-gene covariance (the MANOVA / Wilks form of item
    964's screen), with ``ridge`` times the mean variance on the diagonal so
    a singular block stays finite.

    :param old: cells by old features.
    :param new: cells by new features.
    :param labels: each cell's gene.
    :param ridge: the ridge share.
    :returns: the added gene information in bits (biased upward by about
        ``groups * new_dims / (2 n ln 2)``; compare with a permutation null).
    """
    x = np.hstack([np.asarray(old, float), np.asarray(new, float)])
    codes = pd.factorize(np.asarray(labels))[0]
    return _cmi(x, codes, np.asarray(old).shape[1], ridge)


def _cmi(x: np.ndarray, codes: np.ndarray, n_old: int, ridge: float) -> float:
    """:func:`conditional_information` on a stacked matrix and integer codes."""
    n, dims = x.shape
    groups = int(codes.max()) + 1
    mu = x.mean(axis=0)
    xc = x - mu
    total = xc.T @ xc / (n - 1)
    sums = np.zeros((groups, dims))
    np.add.at(sums, codes, xc)
    counts = np.bincount(codes, minlength=groups).astype(float)
    between = (sums / np.clip(counts, 1, None)[:, None]).T @ sums
    within = (xc.T @ xc - between) / max(n - groups, 1)
    eps = ridge * float(np.trace(total)) / dims

    def schur(m: np.ndarray) -> np.ndarray:
        """The covariance of the new block given the old block."""
        m = m + eps * np.eye(dims)
        oo, on, nn = m[:n_old, :n_old], m[:n_old, n_old:], m[n_old:, n_old:]
        return nn - on.T @ np.linalg.solve(oo, on)

    return float(0.5 * (np.linalg.slogdet(schur(total))[1] - np.linalg.slogdet(schur(within))[1]) / math.log(2))


def _score_information(screen: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """The conditional-information screen of the last ``info_new_dims`` dims, with a permutation null.

    Interim scorer before item 964's ``spacr.ops_feature_info.screen`` lands:
    the same Gaussian conditional mutual information on the CPU, against
    ``info_permutations`` label permutations (``default_rng(seed + 5)``).
    With ``info_null`` ``'guide'`` (the default) the genes are permuted over
    whole guides, so each guide keeps its cells and the null carries the
    guide-to-guide variation; ``'cell'`` permutes the cells' gene labels,
    which counts guide-to-guide variation as gene information.
    """
    d = screen["design"]
    k = int(d["info_new_dims"])
    x = np.asarray(screen["cells"], float)
    n_old = x.shape[1] - k
    codes = pd.factorize(screen["labels"])[0]
    guide_codes, guide_index = np.unique(np.asarray(screen["guides"]), return_inverse=True)
    guide_gene = np.zeros(len(guide_codes), int)
    guide_gene[guide_index] = codes
    observed = _cmi(x, codes, n_old, 1e-6)
    rng = np.random.default_rng(int(seed) + 5)
    if str(d["info_null"]) == "cell":
        null = np.array([_cmi(x, rng.permutation(codes), n_old, 1e-6) for _ in range(int(d["info_permutations"]))])
    else:
        null = np.array([_cmi(x, rng.permutation(guide_gene)[guide_index], n_old, 1e-6)
                         for _ in range(int(d["info_permutations"]))])
    return {"info_bits": observed, "info_null_mean": float(null.mean()), "info_null_q95": float(np.quantile(null, 0.95)),
            "info_bits_adjusted": observed - float(null.mean()), "info_detected": bool(observed > np.quantile(null, 0.95))}


def score_ops_screen(screen: Mapping[str, Any], scorers: Sequence[str] = ("etest",), seed: int = 0) -> Dict[str, Any]:
    """Score one simulated screen with spaCR's OPS analysis steps.

    :param screen: from :func:`simulate_ops_screen`.
    :param scorers: any of ``'etest'`` (per-gene E-tests against NT with BH,
        :class:`spacr.gene_measurement_compare.EnergyTest` in the design's
        ``etest_mode``, plus the guide agreement), ``'clustering'`` (Hybrid profiles, PHATE + Leiden and the
        planted complexes recovered; needs the ``clustering`` extra) and
        ``'information'`` (the conditional-information screen of the last
        ``info_new_dims`` feature directions).
    :param seed: the scorers' seed.
    :returns: one flat row of metrics; a scorer whose optional packages are
        missing gives ``<scorer>_status`` ``'unavailable'``.
    :raises ValueError: for an unknown scorer.
    """
    unknown = sorted(set(scorers) - {"etest", "clustering", "information"})
    if unknown:
        raise ValueError(f"unknown OPS power scorers {unknown}; choose from etest, clustering, information")
    row: Dict[str, Any] = {}
    if "etest" in scorers:
        result = _score_etest(screen, seed)
        result.pop("_table")
        row.update(result)
        row.update(_guide_agreement(screen))
    if "clustering" in scorers:
        try:
            row.update(_score_clustering(screen, seed))
            row["clustering_status"] = "ok"
        except ImportError:
            row["clustering_status"] = "unavailable"
    if "information" in scorers:
        row.update(_score_information(screen, seed))
    return row


def _design_points(grid: Any, effect_types: Sequence[str]) -> List[Dict[str, Any]]:
    """The design points of a grid: a dict (its Cartesian product) or a list of dicts (their union)."""
    import itertools

    grids = grid if isinstance(grid, (list, tuple)) else [grid]
    points: List[Dict[str, Any]] = []
    for one in grids:
        one = dict(one or {})
        if "effect_type" not in one:
            one["effect_type"] = list(effect_types)
        axes = {k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in one.items()}
        for values in itertools.product(*axes.values()):
            point = dict(zip(axes.keys(), values))
            if point not in points:
                points.append(point)
    return points


_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(calibration: Mapping[str, Any]) -> None:
    """Give a sweep worker process its calibration (a spawned process starts empty)."""
    _WORKER_STATE["calibration"] = calibration


def _run_point(task: tuple) -> Dict[str, Any]:
    """Simulate and score one design point and replicate (a sweep worker)."""
    import time

    point, seed, scorers, threads = task
    calibration = _WORKER_STATE["calibration"]
    started = time.perf_counter()
    from contextlib import nullcontext

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        threadpool_limits = None
    limiter = threadpool_limits(int(threads)) if threadpool_limits and threads else nullcontext()
    with limiter:
        try:
            screen = simulate_ops_screen(calibration, point, seed)
            simulated = time.perf_counter() - started
            row = {**{f"design_{k}": v for k, v in point.items()}, "seed": int(seed),
                   "design_key": _design_key(screen["design"]), "status": "ok",
                   "cells": int(len(screen["labels"])),
                   "cells_per_gene_median": float(screen["truth"]["cells"].median())}
            row.update(score_ops_screen(screen, scorers, seed))
            row["simulate_s"] = round(simulated, 1)
            row["seconds"] = round(time.perf_counter() - started, 1)
        except Exception as exc:  # noqa: BLE001 - a failed point is recorded as failed
            row = {**{f"design_{k}": v for k, v in point.items()}, "seed": int(seed), "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
    return row


def _calibration_key(calibration: Mapping[str, Any]) -> str:
    """Hash the calibration values and every control-pool value for resume identity.

    Pools calibrated from a run can hold guide names in object arrays.
    Encode the names as strings so process-specific object addresses do not
    change the identity of an unchanged pool.

    :param calibration: from :func:`load_calibration` or :func:`calibrate_from_run`.
    :returns: the SHA256 content digest.
    """
    values = {key: value for key, value in calibration.items() if key != "_pool"}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode("utf-8"))
    for array in calibration.get("_pool", ()):
        array = np.asarray(array)
        if array.dtype.hasobject:
            array = array.astype(str)
        array = np.ascontiguousarray(array)
        digest.update(json.dumps([array.dtype.str, array.shape]).encode("utf-8"))
        digest.update(array.data)
    return digest.hexdigest()


def scan_ops_designs(calibration: Mapping[str, Any], grid: Any = None, seeds: Sequence[int] = (0, 1, 2), *,
                     effect_types: Sequence[str] = ("mean",), scorers: Sequence[str] = ("etest",),
                     progress_file: Optional[str] = None, workers: int = 1, threads: int = 2,
                     report: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Sweep OPS screen designs: simulate, score and cache every point and seed.

    :param calibration: from :func:`load_calibration`.
    :param grid: design keys to values; a dict sweeps their Cartesian
        product, a list of dicts the union of the sweeps;
        :data:`DEFAULT_OPS_POWER_GRID` when None.
    :param seeds: the replicate seeds of every point.
    :param effect_types: the effect types swept when the grid has no
        ``effect_type``.
    :param scorers: see :func:`score_ops_screen`.
    :param progress_file: a TSV each finished row is appended to; a rerun
        skips rows keyed by the calibration contents, explicit and resolved
        design values, seed and scorers, so a stopped sweep resumes.
        Invalid designs are still recorded as failed rows.
    :param workers: processes run at once.
    :param threads: BLAS threads per process.
    :param report: called with one line per finished row.
    :returns: one row per point and seed, in grid then seed order; with
        ``progress_file``, the rows as stored there, so a resumed sweep and
        an uninterrupted one return the same values.
    """
    say = report or (lambda text: None)
    points = _design_points(DEFAULT_OPS_POWER_GRID if grid is None else grid, effect_types)
    tag = ",".join(sorted(scorers))
    calibration_key = _calibration_key(calibration)
    tasks, keys = [], []
    for point in points:
        try:
            resolved = resolve_design(calibration, point)
        except (TypeError, ValueError, OverflowError):
            resolved = point
        for seed in seeds:
            keys.append(hashlib.sha1(json.dumps([calibration_key, point, resolved, int(seed), tag],
                                               sort_keys=True, default=str)
                                     .encode("utf-8")).hexdigest()[:16])
            tasks.append((point, int(seed), tuple(scorers), int(threads)))
    done: Dict[str, Dict[str, Any]] = {}
    if progress_file and os.path.isfile(progress_file):
        from .tabular import read_table

        previous = read_table(progress_file, canonicalise=False, report=None, sep="\t")
        for record in previous.to_dict("records"):
            done[str(record["run_key"])] = {k: v for k, v in record.items() if not (isinstance(v, float) and math.isnan(v))}
    todo = [i for i, key in enumerate(keys) if key not in done]
    say(f"OPS power sweep: {len(points)} designs x {len(seeds)} seeds, {len(todo)} to run")
    _WORKER_STATE["calibration"] = calibration

    def finish(i: int, row: Dict[str, Any]) -> None:
        """Keep one finished row and append it to the progress file."""
        row = {"run_key": keys[i], **row}
        done[keys[i]] = row
        if progress_file:
            _append_row(progress_file, row)
        say(f"OPS power {sum(key in done for key in keys)}/{len(keys)}: {row.get('status')} {row.get('seconds', '')} s")

    if int(workers) > 1 and len(todo) > 1:
        from .resource_log import _guard_workers

        workers = max(1, int(_guard_workers("ops_power_sim", int(workers), _screen_bytes(calibration, points))))
    if int(workers) > 1 and len(todo) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(int(workers), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(calibration,)) as pool:
            for i, row in zip(todo, pool.map(_run_point, [tasks[i] for i in todo])):
                finish(i, row)
    else:
        for i in todo:
            finish(i, _run_point(tasks[i]))
    _WORKER_STATE.clear()
    if progress_file:
        from .tabular import read_table

        stored = read_table(progress_file, canonicalise=False, report=None, sep="\t")
        stored = stored.drop_duplicates("run_key", keep="last").set_index("run_key", drop=False)
        return stored.loc[keys].reset_index(drop=True)
    return pd.DataFrame([done[key] for key in keys])


def _screen_bytes(calibration: Mapping[str, Any], points: Sequence[Mapping[str, Any]]) -> int:
    """About the peak memory of one simulated screen: the largest point's cells, four float64 copies."""
    largest = 0
    for point in points:
        try:
            d = resolve_design(calibration, point)
        except ValueError:
            continue
        per_gene = float(d["cells_per_gene"]) * float(d["call_rate"]) / float(
            (calibration.get("call_rate") or {}).get("value", 1.0) or 1.0)
        genes = int(calibration.get("n_genes") or d["n_genes"])
        cells = per_gene * (int(d["n_genes"]) + genes * float(calibration.get("nt_ratio", 0.02)))
        largest = max(largest, int(cells))
    return largest * int(calibration.get("dims") or 1) * 8 * 4


def _append_row(path: str, row: Mapping[str, Any]) -> None:
    """Append one row to a TSV, widening the header when the row has new columns."""
    from .tabular import read_table, write_table

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frame = pd.DataFrame([row])
    if os.path.isfile(path):
        frame = pd.concat([read_table(path, canonicalise=False, report=None, sep="\t"), frame], ignore_index=True)
    write_table(frame, path, canonicalise=False, sep="\t")


def power_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Power per design with its 95 % interval, from a sweep's rows.

    :param rows: from :func:`scan_ops_designs`.
    :returns: one row per design (every ``design_*`` column but the seed):
        ``replicates``, ``power`` (detected over planted, pooled over the
        seeds) with its Wilson interval, the seeds' mean null-gene and NT
        pseudo-gene false-positive rates and, when scored, the complex
        recovery and the information screen's detection share.
    """
    ok = rows[rows["status"] == "ok"]
    design_columns = [c for c in ok.columns if c.startswith("design_") and c != "design_key"]
    out = []
    for values, group in ok.groupby(design_columns, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(design_columns, values))
        row["replicates"] = int(len(group))
        if "etest_detected" in group:
            k, n = int(group["etest_detected"].sum()), int(group["etest_planted"].sum())
            low, high = _wilson(k, n)
            row.update({"power": k / n if n else float("nan"), "power_low": low, "power_high": high,
                        "fpr_null_genes": float(group["etest_fpr_null_genes"].mean()),
                        "nt_pseudo_p05_pct": float(group["nt_pseudo_p05_pct"].mean()),
                        "nt_guide_p05_pct": float(group["nt_guide_p05_pct"].mean())})
        for column in ("complex_recovery", "complex_recovery_shuffled", "pair_f1", "info_bits_adjusted"):
            if column in group:
                row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        if "info_detected" in group:
            row["info_detected_share"] = float(group["info_detected"].astype(str).str.lower().eq("true").mean())
        out.append(row)
    return pd.DataFrame(out)


def minimum_detectable_effect(summary: pd.DataFrame, x: str = "design_effect_size", target: float = 0.8,
                              by: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The smallest effect detected at ``target`` power, by linear interpolation along ``x``.

    :param summary: from :func:`power_summary`.
    :param x: the effect column.
    :param target: the power to reach.
    :param by: the columns that define one curve; every other ``design_*``
        column when None.
    :returns: one row per curve with ``mde`` (NaN when the curve never
        reaches ``target``, and the smallest ``x`` when it starts above it).
    """
    by = list(by) if by is not None else [c for c in summary.columns if c.startswith("design_") and c != x]
    out = []
    groups = summary.groupby(by, dropna=False, sort=True) if by else [((), summary)]
    for values, curve in groups:
        values = values if isinstance(values, tuple) else (values,)
        curve = curve.sort_values(x)
        xs = pd.to_numeric(curve[x], errors="coerce").to_numpy(float)
        ys = curve["power"].to_numpy(float)
        mde = float("nan")
        for i in range(len(xs)):
            if ys[i] >= target:
                if i == 0:
                    mde = xs[0]
                else:
                    mde = xs[i - 1] + (target - ys[i - 1]) * (xs[i] - xs[i - 1]) / max(ys[i] - ys[i - 1], 1e-12)
                break
        out.append({**dict(zip(by, values)), "mde": mde, "target_power": target})
    return pd.DataFrame(out)


def _figure(summary: pd.DataFrame, rows: pd.DataFrame, x: str, path: str) -> Optional[str]:
    """Power along ``x``: bars of the pooled power, the seeds as jittered points, an 80 % line."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES, figure_style, resolve_ink, resolve_label_ground
    from .plot import _checked_savefig

    if "power" not in summary or x not in summary:
        return None
    colours = ["#1f6f8b", "#2a9d8f", "#4a7fc1", "#1f4e79", "#76c7b7"]
    others = [c for c in summary.columns if c.startswith("design_") and c != x and summary[c].nunique() > 1]
    curves = list(summary.groupby(others, dropna=False, sort=True)) if others else [((), summary)]
    levels = sorted(summary[x].unique(), key=lambda v: float(v) if str(v).replace(".", "", 1).isdigit() else str(v))
    width = 0.8 / max(len(curves), 1)
    rng = np.random.default_rng(0)
    ok = rows[rows["status"] == "ok"]
    position = {v: i for i, v in enumerate(levels)}
    with figure_style("print"):
        figure = Figure(figsize=(7.5, 4.0))
        ax = figure.subplots()
        ink = resolve_ink("print")
        for j, (values, curve) in enumerate(curves):
            values = values if isinstance(values, tuple) else (values,)
            offset = (j - (len(curves) - 1) / 2) * width
            xs = np.array([position[v] for v in curve[x]]) + offset
            colour = colours[j % len(colours)]
            label = ", ".join(f"{c.replace('design_', '')}={v}" for c, v in zip(others, values)) or "power"
            ax.bar(xs, curve["power"], width * 0.9, color=colour, alpha=0.6, label=label)
            ax.errorbar(xs, curve["power"], yerr=[(curve["power"] - curve["power_low"]).clip(lower=0),
                                                  (curve["power_high"] - curve["power"]).clip(lower=0)],
                        fmt="none", ecolor=ink, capsize=2, lw=0.8)
            mask = np.ones(len(ok), bool)
            for c, v in zip(others, values):
                mask &= (ok[c].astype(str) == str(v)).to_numpy()
            seeds = ok[mask]
            sx = np.array([position.get(v, np.nan) for v in seeds[x]]) + offset
            sy = seeds["etest_detected"] / seeds["etest_planted"].clip(lower=1)
            ax.scatter(sx + rng.uniform(-width / 4, width / 4, len(sx)), sy, s=10, color=colour, alpha=0.6)
        ax.axhline(0.8, color=ROLES["reference"], lw=0.8, ls="--")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(v) for v in levels])
        ax.set_xlabel(x.replace("design_", "").replace("_", " "))
        ax.set_ylabel("power (planted genes at q < 0.05)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        _register_figure_data(figure, lambda: summary.assign(**{x: summary[x].astype(str)}), x=x, y="power",
                              kind="bar", order=[str(v) for v in levels])
        figure.tight_layout()
        _checked_savefig(figure, path, dpi=150, bbox_inches="tight", facecolor=resolve_label_ground("print"))
    return path


def _list_setting(value: Any, default: Sequence[Any]) -> List[Any]:
    """A list setting given as a list, a JSON text or comma-separated text."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return list(json.loads(text))
        return [item.strip() for item in text.split(",") if item.strip()]
    return list(value)


def _grid_setting(value: Any) -> Any:
    """``ops_power_grid``: a dict, a list of dicts, JSON text or a JSON file; None is the default grid."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_OPS_POWER_GRID
    if isinstance(value, str):
        text = value.strip()
        if os.path.isfile(text):
            with open(text, encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(text)
    return value


def run_ops_power_sim(settings: Mapping[str, Any], report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The OPS ``power_sim`` phase: calibrate, sweep the designs and write the power tables.

    Reads ``dst_root``, ``ops_power_calibration_run`` (a calibration JSON
    from :func:`calibrate_from_run`, or a run folder calibrated on the fly),
    ``ops_power_mode``, ``ops_power_grid``, ``ops_power_sim_seeds``,
    ``ops_power_effect_types``, ``ops_power_scorers``,
    ``ops_power_sim_workers`` and ``ops_blas_threads``. Everything goes to
    ``<dst_root>/results/ops_power_sim``: ``designs.tsv`` (every point and
    seed, the resume cache), ``power_rows.csv`` (the same rows, sorted),
    ``power_summary.csv``, ``power_mde.csv``, ``power_curve.png`` and
    ``runtime.json`` (the settings and the calibration's sha256). The same
    settings file gives the same tables.

    :param settings: the OPS settings.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs"}``.
    :raises ValueError: when ``dst_root`` or the calibration is missing.
    """
    from .ops_settings import ops_defaults
    from .tabular import write_table

    settings = ops_defaults(dict(settings))
    say = report or (lambda text: None)
    if not settings.get("dst_root"):
        raise ValueError("the power_sim phase needs dst_root, the folder its results go to")
    source = settings.get("ops_power_calibration_run")
    if not source or not os.path.exists(str(source)):
        raise ValueError(f"ops_power_calibration_run must be a calibration JSON or a run folder; got {source!r}")
    calibration = load_calibration(str(source))
    grid = _grid_setting(settings.get("ops_power_grid"))
    mode = str(settings.get("ops_power_mode") or "empirical")
    grids = grid if isinstance(grid, (list, tuple)) else [grid]
    grid = [{"mode": mode, **dict(g)} if "mode" not in g else dict(g) for g in grids]
    seeds = [int(s) for s in _list_setting(settings.get("ops_power_sim_seeds"), [0, 1, 2])]
    effect_types = [str(e) for e in _list_setting(settings.get("ops_power_effect_types"), ["mean"])]
    scorers = [str(s) for s in _list_setting(settings.get("ops_power_scorers"), ["etest"])]
    folder = os.path.join(str(settings["dst_root"]), "results", "ops_power_sim")
    os.makedirs(folder, exist_ok=True)
    progress = os.path.join(folder, "designs.tsv")
    rows = scan_ops_designs(calibration, grid, seeds, effect_types=effect_types, scorers=scorers,
                            progress_file=progress, workers=int(settings.get("ops_power_sim_workers") or 1),
                            threads=int(settings.get("ops_blas_threads") or 2), report=say)
    rows = rows.drop(columns=[c for c in ("seconds", "simulate_s") if c in rows.columns])
    outputs = {"designs": progress}
    outputs["rows"] = write_table(rows, os.path.join(folder, "power_rows.csv"), canonicalise=False)
    summary = power_summary(rows)
    outputs["summary"] = write_table(summary, os.path.join(folder, "power_summary.csv"), canonicalise=False)
    if "design_effect_size" in summary and "power" in summary:
        numeric = summary[pd.to_numeric(summary["design_effect_size"], errors="coerce").notna()]
        outputs["mde"] = write_table(minimum_detectable_effect(numeric), os.path.join(folder, "power_mde.csv"),
                                     canonicalise=False)
    x = next((c for c in ("design_effect_size", "design_cells_per_gene") if c in summary
              and summary[c].nunique() > 1), None)
    if x:
        figure = _figure(summary, rows, x, os.path.join(folder, "power_curve.png"))
        if figure:
            outputs["figure"] = figure
    keys = sorted(k for k in settings if k.startswith("ops_power_") or k in ("dst_root", "ops_blas_threads"))
    record = {"settings": {k: settings[k] for k in keys}, "calibration": os.path.abspath(str(source)),
              "calibration_sha256": _sha256(str(source)) if os.path.isfile(str(source)) else None,
              "outputs": outputs}
    with open(os.path.join(folder, "runtime.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str, sort_keys=True)
    return {"folder": folder, "outputs": outputs}
