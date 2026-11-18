"""How much gene information new per-cell features add beyond existing ones (item 964, alpha).

The screen asks, for each new feature column and each family of columns,
how much a cell's values tell about which gene it carries that the old
features do not already tell. It uses Gaussian conditional mutual
information from covariances, so every column is screened at once, on the
GPU when there is one and with the same maths on the CPU otherwise.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["screen"]

#: log(2), to turn natural-log information into bits.
_LN2 = math.log(2.0)


class _NumpyMath:
    """The CPU arithmetic of the screen: numpy, float64."""

    name = "cpu"

    def asarray(self, x):
        """``x`` as a float64 array."""
        return np.asarray(x, dtype=np.float64)

    def indices(self, x):
        """``x`` as an int64 index array."""
        return np.asarray(x, dtype=np.int64)

    def group_sums(self, index, n_groups: int, x):
        """Row sums of ``x`` per group in ``index``, ``(n_groups, d)``."""
        from scipy import sparse

        one_hot = sparse.csr_matrix(
            (np.ones(len(index)), (np.asarray(index), np.arange(len(index)))),
            shape=(n_groups, len(index)))
        return np.asarray(one_hot @ x, dtype=np.float64)

    def counts(self, index, n_groups: int):
        """How many rows each group holds."""
        return np.bincount(np.asarray(index), minlength=n_groups).astype(np.float64)

    def numpy(self, x):
        """``x`` back on the host."""
        return np.asarray(x)

    def solve(self, a, b):
        """``a^-1 b``."""
        return np.linalg.solve(a, b)

    def logdet(self, a) -> float:
        """log det of a symmetric positive definite matrix."""
        sign, value = np.linalg.slogdet(a)
        return float(value) if sign > 0 else float("nan")

    def eye(self, n: int):
        """The identity."""
        return np.eye(n)

    def outer_sum(self, a, b):
        """``a.T @ b``."""
        return a.T @ b


class _TorchMath:
    """The GPU arithmetic of the screen: torch, float64, on one device."""

    def __init__(self, device: str):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.name = str(self.device)

    def asarray(self, x):
        """``x`` as a float64 tensor on the device."""
        return self.torch.as_tensor(np.asarray(x), dtype=self.torch.float64,
                                    device=self.device)

    def indices(self, x):
        """``x`` as an int64 index tensor on the device."""
        return self.torch.as_tensor(np.asarray(x), dtype=self.torch.int64,
                                    device=self.device)

    def group_sums(self, index, n_groups: int, x):
        """Row sums of ``x`` per group in ``index``, by scatter-add."""
        out = self.torch.zeros((n_groups, x.shape[1]), dtype=x.dtype,
                               device=x.device)
        return out.index_add_(0, index, x)

    def counts(self, index, n_groups: int):
        """How many rows each group holds."""
        return self.torch.bincount(index, minlength=n_groups).to(
            self.torch.float64)

    def numpy(self, x):
        """``x`` back on the host."""
        return x.detach().cpu().numpy()

    def solve(self, a, b):
        """``a^-1 b``."""
        return self.torch.linalg.solve(a, b)

    def logdet(self, a) -> float:
        """log det of a symmetric positive definite matrix."""
        sign, value = self.torch.linalg.slogdet(a)
        return float(value) if float(sign) > 0 else float("nan")

    def eye(self, n: int):
        """The identity."""
        return self.torch.eye(n, dtype=self.torch.float64, device=self.device)

    def outer_sum(self, a, b):
        """``a.T @ b``."""
        return a.T @ b


def _math_for(device: str):
    """The arithmetic for ``device``: ``'auto'``, ``'cpu'`` (numpy), a torch device, or ``'torch:<device>'``."""
    device = str(device or "auto").strip().lower()
    if device == "cpu":
        return _NumpyMath()
    if device.startswith("torch:"):
        return _TorchMath(device.split(":", 1)[1])
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return _TorchMath("cuda")
        except ImportError:
            pass
        return _NumpyMath()
    return _TorchMath(device)


def _conditional(m, xp, target: Sequence[int], given: Sequence[int],
                 ridge: float):
    """The covariance of ``target`` given ``given`` within ``m``, ``(t, t)``.

    :param m: a scatter or covariance matrix over all columns.
    :param target: the columns asked about.
    :param given: the conditioning columns.
    :param ridge: added to the diagonal of the ``given`` block.
    """
    t_idx = xp.indices(target)
    block = m[t_idx][:, t_idx]
    if not len(given):
        return block
    g_idx = xp.indices(given)
    cc = m[g_idx][:, g_idx] + ridge * xp.eye(len(given))
    ct = m[g_idx][:, t_idx]
    return block - xp.outer_sum(ct, xp.solve(cc, ct))


def _column_bits(total, within, xp, target: Sequence[int],
                 given: Sequence[int], ridge: float):
    """Added bits, conditional variances and unique fractions of single columns.

    :returns: ``(bits, total conditional variance, unique fraction)``, one
        value per column of ``target``, as host arrays.
    """
    t_idx = xp.indices(target)
    out = []
    for m in (total, within):
        diagonal = m[t_idx, t_idx]
        if len(given):
            g_idx = xp.indices(given)
            cc = m[g_idx][:, g_idx] + ridge * xp.eye(len(given))
            ct = m[g_idx][:, t_idx]
            diagonal = diagonal - (ct * xp.solve(cc, ct)).sum(0)
        out.append(xp.numpy(diagonal))
    t_cond, w_cond = (np.maximum(v, 0.0) + ridge for v in out)
    raw = xp.numpy(total[t_idx, t_idx])
    return (0.5 * np.log(t_cond / w_cond) / _LN2, t_cond,
            np.clip((t_cond - ridge) / np.maximum(raw, 1e-300), 0.0, None))


def _block_bits(total, within, xp, target: Sequence[int],
                given: Sequence[int], ridge: float) -> float:
    """Added bits of a block of columns given ``given``: the Wilks form."""
    eye = ridge * xp.eye(len(target))
    t_cond = _conditional(total, xp, target, given, ridge) + eye
    w_cond = _conditional(within, xp, target, given, ridge) + eye
    return 0.5 * (xp.logdet(t_cond) - xp.logdet(w_cond)) / _LN2


def _codes(values) -> Tuple[np.ndarray, np.ndarray]:
    """Integer codes of ``values`` and the levels they index."""
    levels, codes = np.unique(np.asarray(values).astype(str),
                              return_inverse=True)
    return codes.astype(np.int64), levels


def _permuted_units(unit_gene: np.ndarray, unit_block: np.ndarray,
                    rng) -> np.ndarray:
    """``unit_gene`` shuffled among the units of each block (each well)."""
    out = unit_gene.copy()
    for block in np.unique(unit_block):
        where = np.flatnonzero(unit_block == block)
        out[where] = unit_gene[where[rng.permutation(len(where))]]
    return out


def _size_matched_partners(sizes: np.ndarray, owner: np.ndarray, rng,
                           width: int = 20) -> np.ndarray:
    """A partner for every row: a row of another owner and similar size.

    Rows are sorted by ``sizes`` and shuffled within bins of ``width``; a
    row paired with its own owner is swapped with another row of the bin
    until none is, so a null predictor rests on about as many cells as the
    observed one and never comes from the same gene.

    :param sizes: the cell count behind each row's predictor.
    :param owner: each row's gene.
    :returns: the partner index of every row.
    """
    order = np.argsort(sizes, kind="stable")
    partner = np.arange(len(sizes))
    starts = list(range(0, len(order), width))
    if len(starts) > 1 and len(order) - starts[-1] < 2:
        starts = starts[:-1]
    for k, start in enumerate(starts):
        stop = starts[k + 1] if k + 1 < len(starts) else len(order)
        members = order[start:stop]
        if len(members) < 2:
            continue
        shuffled = members[rng.permutation(len(members))]
        for _attempt in range(20):
            clash = np.flatnonzero(owner[shuffled] == owner[members])
            if not len(clash):
                break
            for i in clash:
                j = int(rng.integers(len(members)))
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        partner[members] = shuffled
    return partner


def _scatter(xtx, sums, n: float):
    """The scatter matrix about the mean."""
    return xtx - (sums[:, None] * sums[None, :]) / n


def _within_from(xtx, group_sums, group_counts, xp):
    """The pooled within-group scatter: ``X'X - sum_g S_g S_g' / n_g``."""
    keep = group_counts > 0
    scaled = group_sums[keep] / group_counts[keep][:, None]
    return xtx - xp.outer_sum(group_sums[keep], scaled)


def _residual_scatter(xtx, sums, predictors, counts, xp):
    """Scatter of ``x - p`` where every row of group k is predicted by ``p_k``.

    :param sums: per-group row sums ``S_k``.
    :param predictors: per-group predictors ``p_k``.
    :param counts: per-group row counts.
    """
    cross = xp.outer_sum(sums, predictors)
    return xtx - cross - cross.T + xp.outer_sum(
        predictors * counts[:, None], predictors)


def _items(names: Sequence[str], old_n: int,
           families: Optional[Mapping[str, Sequence[str]]]
           ) -> List[Tuple[str, str, List[int]]]:
    """The screened items: every new column, then every family.

    :returns: ``(kind, name, column indices in [old, new])`` triples; the
        family ``all_new`` holds every new column.
    """
    index = {name: old_n + i for i, name in enumerate(names)}
    items = [("column", name, [index[name]]) for name in names]
    declared = dict(families or {})
    declared.setdefault("all_new", list(names))
    for family, members in declared.items():
        cols = [index[m] for m in members if m in index]
        if cols:
            items.append(("family", str(family), cols))
    return items


def _evaluate(total, within, xp, items, given, ridge) -> np.ndarray:
    """Added bits of every item given ``given``."""
    out = np.empty(len(items))
    columns = [i for i, (kind, _n, _c) in enumerate(items) if kind == "column"]
    if columns:
        target = [items[i][2][0] for i in columns]
        out[columns] = _column_bits(total, within, xp, target, given, ridge)[0]
    for i, (kind, _name, cols) in enumerate(items):
        if kind == "family":
            out[i] = _block_bits(total, within, xp, cols, given, ridge)
    return out


def screen(old, new, cells=None, *, label: str = "gene", guide: str = "sgRNA",
           control: str = "nontargeting", well: str = "well",
           field: str = "site", id_columns: Optional[Sequence[str]] = None,
           families: Optional[Mapping[str, Sequence[str]]] = None,
           permutations: int = 500, perm_unit: str = "field", seed: int = 0,
           ridge: float = 1e-6, greedy_threshold: float = 0.0,
           mlp: bool = False, device: str = "auto",
           chunk: int = 200_000) -> Dict[str, Any]:
    """Screen new feature columns for gene information the old ones lack (item 964).

    :param old: the existing per-cell features, a DataFrame, already
        normalised (robust z or TVN against the control cells).
    :param new: the candidate per-cell features, normalised the same way.
    :param cells: the per-cell labels (``label``, ``guide``, ``well``,
        ``field``) when neither feature table carries them.
    :param label: the gene column; every control guide counts as one gene.
    :param guide: the guide column.
    :param control: the prefix of control genes.
    :param well: the well column; permutations stay within it, and held-out
        estimates split by it (by ``field`` in a one-well table).
    :param field: the field (site) column.
    :param id_columns: the columns the tables are joined on; by default the
        columns among ``plate``, ``well``, ``site`` and ``label`` that every
        table has (``label`` only when it is not the gene column).
    :param families: ``{family: [new columns]}``, screened as blocks; the
        family ``all_new`` is always added.
    :param permutations: gene-label permutations of the null.
    :param perm_unit: ``'field'`` permutes the genes of gene-by-field units
        within each well, ``'cell'`` the genes of single cells.
    :param seed: the seed of the permutations and the guide null.
    :param ridge: added to every conditioning block, times the mean variance.
    :param greedy_threshold: the smallest added bits greedy selection accepts.
    :param mlp: also fit a small network, from old and from old + new, and
        report the held-out cross-entropy difference in nats.
    :param device: ``'auto'`` (the GPU when there is one), ``'cpu'`` or a
        torch device.
    :param chunk: rows per accumulation step.
    :returns: ``table`` (a DataFrame, one row per column and per family),
        ``null`` (bits per permutation and item), ``greedy`` (the ranked
        list), ``summary`` and ``device``.

    THE QUANTITY is I(gene; new | old) for Gaussian data: half the log
    ratio of the determinants of the total and the within-gene covariance
    of the new columns, both conditioned on the old columns (Schur
    complements). A column that the old features predict has 0 added bits,
    and so has noise, up to the null. The unique fraction is 1 - R2 of a
    column on the old features.

    NULL, HELD-OUT AND GUIDE CHECKS. The permutations change only the
    group sums, so they are cheap. Observed minus the null mean is the
    corrected estimate; the interval is that value +- 1.96 null SDs. The
    held-out estimate predicts each cell by its gene's mean on the other
    wells. The guide-replicated estimate predicts each cell by the other
    guides of its gene, against the same predictors given to guides of other
    genes with as many cells behind them.
    """
    import pandas as pd

    started = time.perf_counter()
    xp = _math_for(device)
    frames = [f for f in (old, new, cells) if f is not None]
    if id_columns is None:
        id_columns = [c for c in ("plate", "well", "site", "label")
                      if c != label and all(c in f.columns for f in frames)]
    old_cols = [c for c in old.columns if c not in id_columns
                and pd.api.types.is_numeric_dtype(old[c])
                and c not in (label, guide, well, field)]
    new_cols = [c for c in new.columns if c not in id_columns
                and pd.api.types.is_numeric_dtype(new[c])
                and c not in (label, guide, well, field) and c not in old_cols]
    if not new_cols:
        raise ValueError("the new table has no numeric feature columns")
    table = old[list(id_columns) + old_cols + [c for c in (label, guide, well, field)
                                              if c in old.columns
                                              and c not in id_columns]]
    table = table.merge(new[list(id_columns) + new_cols], on=list(id_columns),
                        how="inner")
    if cells is not None:
        meta = [c for c in (label, guide, well, field)
                if c in cells.columns and c not in table.columns]
        table = table.merge(cells[list(id_columns) + meta],
                            on=list(id_columns), how="inner")
    for column in (label, well):
        if column not in table.columns:
            raise ValueError(f"no {column!r} column in the tables")
    values = table[old_cols + new_cols].to_numpy(np.float64)
    finite = np.isfinite(values).all(1)
    dropped = int((~finite).sum())
    table, values = table.loc[finite].reset_index(drop=True), values[finite]
    gene_names = table[label].astype(str).to_numpy()
    gene_names = np.where(np.char.startswith(gene_names.astype(str), control),
                          control, gene_names)
    gene, gene_levels = _codes(gene_names)
    wells, _ = _codes(table[well])
    fields, _ = _codes(table[field] if field in table.columns else table[well])
    n, d = values.shape
    p = len(old_cols)
    n_genes = len(gene_levels)

    unit_key = (np.arange(n) if perm_unit == "cell"
                else gene.astype(np.int64) * (int(fields.max()) + 1) + fields)
    _levels, unit_first, unit = np.unique(unit_key, return_index=True,
                                          return_inverse=True)
    unit = unit.astype(np.int64)
    unit_gene = gene[unit_first]
    unit_well = wells[unit_first]
    n_units = len(unit_first)

    split = wells if len(np.unique(wells)) > 1 else fields
    fold = np.zeros(n, np.int64)
    fold[np.isin(split, np.unique(split)[1::2])] = 1

    guides_present = guide in table.columns
    if guides_present:
        guide_code, guide_levels = _codes(table[guide])
        guide_gene = np.zeros(len(guide_levels), np.int64)
        guide_gene[guide_code] = gene
        per_gene = np.bincount(guide_gene, minlength=n_genes)
        multi = per_gene >= 2
        multi[gene_levels == control] = False
        in_multi = multi[gene]
    xtx = xp.asarray(np.zeros((d, d)))
    sums = xp.asarray(np.zeros(d))
    unit_sums = xp.asarray(np.zeros((n_units, d)))
    fold_xtx = [xp.asarray(np.zeros((d, d))) for _ in range(2)]
    fold_gene = [xp.asarray(np.zeros((n_genes, d))) for _ in range(2)]
    if guides_present:
        guide_sums = xp.asarray(np.zeros((len(guide_levels), d)))
        multi_xtx = xp.asarray(np.zeros((d, d)))
        multi_sums = xp.asarray(np.zeros(d))
    for start in range(0, n, int(chunk)):
        stop = min(n, start + int(chunk))
        x = xp.asarray(values[start:stop])
        xtx = xtx + xp.outer_sum(x, x)
        sums = sums + x.sum(0)
        unit_sums = unit_sums + xp.group_sums(
            xp.indices(unit[start:stop]), n_units, x)
        for f in (0, 1):
            rows = np.flatnonzero(fold[start:stop] == f)
            if len(rows):
                xf = x[xp.indices(rows)]
                fold_xtx[f] = fold_xtx[f] + xp.outer_sum(xf, xf)
                fold_gene[f] = fold_gene[f] + xp.group_sums(
                    xp.indices(gene[start:stop][rows]), n_genes, xf)
        if guides_present:
            guide_sums = guide_sums + xp.group_sums(
                xp.indices(guide_code[start:stop]), len(guide_levels), x)
            rows = np.flatnonzero(in_multi[start:stop])
            if len(rows):
                xm = x[xp.indices(rows)]
                multi_xtx = multi_xtx + xp.outer_sum(xm, xm)
                multi_sums = multi_sums + xm.sum(0)

    total = _scatter(xtx, sums, float(n))
    scale = float(np.mean(xp.numpy(total).diagonal())) / n
    lam = float(ridge) * scale * n
    unit_index = xp.indices(unit_gene)
    gene_counts = xp.counts(xp.indices(gene), n_genes)
    gene_sums = xp.group_sums(unit_index, n_genes, unit_sums)
    within = _within_from(xtx, gene_sums, gene_counts, xp)
    items = _items(new_cols, p, families)
    given = list(range(p))
    observed = _evaluate(total, within, xp, items, given, lam)
    col_idx = [i for i, it in enumerate(items) if it[0] == "column"]
    _bits, _t, unique = _column_bits(total, within, xp,
                                     [items[i][2][0] for i in col_idx], given,
                                     lam)

    rng = np.random.default_rng(seed)
    unit_sizes = np.bincount(unit, minlength=n_units).astype(np.float64)
    null = np.empty((int(permutations), len(items)))
    for j in range(int(permutations)):
        shuffled = _permuted_units(unit_gene, unit_well, rng)
        perm_sums = xp.group_sums(xp.indices(shuffled), n_genes, unit_sums)
        perm_counts = xp.asarray(np.bincount(shuffled, weights=unit_sizes,
                                             minlength=n_genes))
        null[j] = _evaluate(total, _within_from(xtx, perm_sums, perm_counts, xp),
                            xp, items, given, lam)
    null_mean = null.mean(0) if len(null) else np.zeros(len(items))
    null_sd = null.std(0, ddof=1) if len(null) > 1 else np.zeros(len(items))
    null_lo = np.percentile(null, 2.5, 0) if len(null) else null_mean
    null_hi = np.percentile(null, 97.5, 0) if len(null) else null_mean

    heldout = np.zeros(len(items))
    fold_counts = [xp.asarray(np.bincount(gene[fold == f], minlength=n_genes))
                   for f in (0, 1)]
    for f in (0, 1):
        other = 1 - f
        if not (fold == f).any() or not (fold == other).any():
            continue
        train_n = xp.numpy(fold_counts[other])
        grand = fold_gene[other].sum(0) / float(train_n.sum())
        safe = xp.asarray(np.where(train_n > 0, train_n, 1.0))
        present = xp.asarray((train_n > 0).astype(np.float64))[:, None]
        train_means = (present * fold_gene[other] / safe[:, None]
                       + (1.0 - present) * grand[None, :])
        t_test = _scatter(fold_xtx[f], fold_gene[f].sum(0),
                          float((fold == f).sum()))
        e_test = _residual_scatter(fold_xtx[f], fold_gene[f], train_means,
                                   fold_counts[f], xp)
        heldout += 0.5 * _evaluate(t_test, e_test, xp, items, given, lam)

    guide_bits = np.full(len(items), np.nan)
    guide_null = np.full(len(items), np.nan)
    if guides_present and in_multi.any():
        g_counts = np.bincount(guide_code, minlength=len(guide_levels)).astype(float)
        g_sums_host = guide_sums
        gene_total = xp.group_sums(xp.indices(guide_gene), n_genes, g_sums_host)
        gene_n = np.bincount(guide_gene, weights=g_counts, minlength=n_genes)
        rows = np.flatnonzero(multi[guide_gene] & (gene_n[guide_gene] - g_counts > 0))
        r = xp.indices(rows)
        own_gene = xp.indices(guide_gene[rows])
        rest_n = gene_n[guide_gene[rows]] - g_counts[rows]
        predictors = (gene_total[own_gene] - g_sums_host[r]) / xp.asarray(rest_n)[:, None]
        m_n = float(g_counts[rows].sum())
        t_multi = _scatter(multi_xtx, multi_sums, float(in_multi.sum()))
        e_multi = _residual_scatter(multi_xtx, g_sums_host[r], predictors,
                                    xp.asarray(g_counts[rows]), xp)
        if m_n:
            guide_bits = _evaluate(t_multi, e_multi, xp, items, given, lam)
        nulls = []
        for _j in range(int(permutations)):
            partner = _size_matched_partners(rest_n, guide_gene[rows], rng)
            e_null = _residual_scatter(multi_xtx, g_sums_host[r],
                                       predictors[xp.indices(partner)],
                                       xp.asarray(g_counts[rows]), xp)
            nulls.append(_evaluate(t_multi, e_null, xp, items, given, lam))
        if nulls:
            guide_null = np.mean(nulls, 0)

    greedy: List[Dict[str, Any]] = []
    pool = [i for i, it in enumerate(items) if it[0] == "family" and it[1] != "all_new"]
    if not pool:
        pool = list(col_idx)
    chosen: List[int] = []
    step_given = list(given)
    while pool and len(greedy) < 100:
        gains = _evaluate(total, within, xp, [items[i] for i in pool],
                          step_given, lam) - null_mean[pool]
        best = pool[int(np.argmax(gains))]
        gain = float(np.max(gains)) + float(null_mean[best])
        if gain - null_mean[best] <= max(float(greedy_threshold),
                                         null_hi[best] - null_mean[best]):
            break
        greedy.append({"rank": len(greedy) + 1, "item": items[best][1],
                       "kind": items[best][0], "gain_bits": float(gain),
                       "gain_bits_corrected": float(gain - null_mean[best])})
        chosen.append(best)
        step_given = step_given + [c for c in items[best][2] if c not in step_given]
        pool.remove(best)
    rank = {g["item"]: g["rank"] for g in greedy}
    gain_of = {g["item"]: g["gain_bits"] for g in greedy}

    unique_by_col = dict(zip([items[i][1] for i in col_idx], unique))
    rows_out = []
    for i, (kind, name, cols) in enumerate(items):
        members = [new_cols[c - p] for c in cols]
        rows_out.append({
            "kind": kind, "name": name, "n_columns": len(cols),
            "unique_fraction": float(np.mean([unique_by_col[m] for m in members])),
            "added_bits": float(observed[i]), "null_mean": float(null_mean[i]),
            "null_low": float(null_lo[i]), "null_high": float(null_hi[i]),
            "added_bits_corrected": float(observed[i] - null_mean[i]),
            "ci_low": float(observed[i] - null_mean[i] - 1.96 * null_sd[i]),
            "ci_high": float(observed[i] - null_mean[i] + 1.96 * null_sd[i]),
            "heldout_bits": float(heldout[i]),
            "guide_bits": float(guide_bits[i]),
            "guide_null_mean": float(guide_null[i]),
            "guide_bits_corrected": float(guide_bits[i] - guide_null[i]),
            "greedy_rank": rank.get(name), "greedy_gain_bits": gain_of.get(name)})
    out_table = pd.DataFrame(rows_out)
    result: Dict[str, Any] = {
        "table": out_table, "null": null, "greedy": greedy,
        "items": [(k, nme) for k, nme, _c in items], "device": xp.name,
        "summary": {
            "cells": int(n), "cells_dropped_nonfinite": dropped,
            "old_columns": p, "new_columns": len(new_cols), "genes": int(n_genes),
            "units": int(n_units), "perm_unit": perm_unit,
            "permutations": int(permutations), "ridge": float(ridge),
            "ridge_absolute": lam, "seed": int(seed), "device": xp.name,
            "heldout_split": well if len(np.unique(wells)) > 1 else field,
            "seconds": round(time.perf_counter() - started, 2)}}
    if mlp:
        result["mlp"] = _mlp_check(values, p, gene, fold, seed)
    return result


def _mlp_check(values: np.ndarray, p: int, gene: np.ndarray,
               fold: np.ndarray, seed: int, epochs: int = 30) -> Dict[str, float]:
    """Held-out cross-entropy of gene from old, then old + new (nats per cell).

    The difference is a variational lower bound on the added information.
    """
    import torch

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_genes = int(gene.max()) + 1
    out = {}
    for name, cols in (("old", slice(0, p)), ("old_new", slice(None))):
        x = torch.as_tensor(values[:, cols], dtype=torch.float32, device=device)
        y = torch.as_tensor(gene, device=device)
        train = torch.as_tensor(fold == 0, device=device)
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(128, n_genes)).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = torch.nn.CrossEntropyLoss()
        for _epoch in range(epochs):
            for batch in torch.split(torch.randperm(int(train.sum()), device=device), 4096):
                rows = torch.nonzero(train).flatten()[batch]
                optimiser.zero_grad()
                loss(model(x[rows]), y[rows]).backward()
                optimiser.step()
        with torch.no_grad():
            out[name] = float(loss(model(x[~train]), y[~train]))
    out["added_nats"] = out["old"] - out["old_new"]
    return out


def _read_table(path: str, columns=None):
    """A parquet or CSV table."""
    import pandas as pd

    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _families(value) -> Optional[Dict[str, List[str]]]:
    """``ops_feature_screen_families``: a dict, a JSON file, or a CSV of column,family."""
    if not value:
        return None
    if isinstance(value, Mapping):
        return {str(k): list(v) for k, v in value.items()}
    path = str(value)
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            return {str(k): list(v) for k, v in json.load(handle).items()}
    frame = _read_table(path)
    return {str(k): list(v) for k, v in
            frame.groupby("family")["column"].apply(list).items()}


def _plot(table, null, items, path: str) -> None:
    """Ranked bars of corrected bits with the permutation values as points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [i for i, (kind, _n) in enumerate(items) if kind == "family"]
    shown = families if families else list(range(len(items)))
    order = sorted(shown, key=lambda i: -table.loc[i, "added_bits_corrected"])[:30]
    fig, ax = plt.subplots(figsize=(max(4.0, 0.45 * len(order) + 1.5), 3.6))
    rng = np.random.default_rng(0)
    for x, i in enumerate(order):
        ax.bar(x, table.loc[i, "added_bits_corrected"], color="#1f4e79", alpha=0.6)
        values = null[:, i] - table.loc[i, "null_mean"]
        ax.scatter(x + rng.uniform(-0.25, 0.25, len(values)), values, s=4,
                   color="#2a9d8f", alpha=0.6, linewidths=0)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([table.loc[i, "name"] for i in order], rotation=60,
                       ha="right", fontsize=7)
    ax.set_ylabel("added gene information (bits, null-corrected)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _run_feature_screen(settings: Mapping[str, Any], report=None) -> Dict[str, Any]:
    """The ``ops_feature_screen`` phase of :func:`spacr.ops_engine.run_ops`.

    :param settings: read for the ``ops_feature_screen_*`` settings and
        ``dst_root``.
    :returns: the summary, the greedy list and the output paths.
    """
    say = report or (lambda text: None)
    old_path = settings.get("ops_feature_screen_old")
    new_path = settings.get("ops_feature_screen_new")
    if not old_path or not new_path:
        raise ValueError("ops_feature_screen needs ops_feature_screen_old and "
                         "ops_feature_screen_new")
    cells_path = settings.get("ops_feature_screen_cells")
    ids = settings.get("ops_feature_screen_id_columns")
    ids = ([c.strip() for c in ids.split(",") if c.strip()]
           if isinstance(ids, str) else (list(ids) if ids else None))
    result = screen(
        _read_table(old_path), _read_table(new_path),
        _read_table(cells_path) if cells_path else None,
        label=str(settings.get("ops_feature_screen_label") or "gene"),
        guide=str(settings.get("ops_feature_screen_guide") or "sgRNA"),
        control=str(settings.get("ops_control_prefix") or "nontargeting"),
        id_columns=ids,
        families=_families(settings.get("ops_feature_screen_families")),
        permutations=int(settings.get("ops_feature_screen_permutations") or 0),
        perm_unit=str(settings.get("ops_feature_screen_perm_unit") or "field"),
        seed=int(settings.get("ops_feature_screen_seed") or 0),
        ridge=float(settings.get("ops_feature_screen_ridge") or 0.0),
        greedy_threshold=float(settings.get("ops_feature_screen_greedy_threshold") or 0.0),
        mlp=bool(settings.get("ops_feature_screen_mlp", False)),
        device=str(settings.get("ops_feature_screen_device") or "auto"))
    destination = str(settings.get("ops_feature_screen_dst") or os.path.join(
        str(settings.get("dst_root") or "."), "ops_feature_screen"))
    os.makedirs(destination, exist_ok=True)
    csv_path = os.path.join(destination, "feature_information.csv")
    result["table"].to_csv(csv_path, index=False)
    chart = os.path.join(destination, "feature_information.png")
    _plot(result["table"], result["null"], result["items"], chart)
    summary = {**result["summary"], "greedy": result["greedy"],
               "mlp": result.get("mlp"), "table": csv_path, "chart": chart}
    with open(os.path.join(destination, "feature_information.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    say(f"ops_feature_screen: {summary['new_columns']} new columns on "
        f"{summary['cells']} cells, {len(result['greedy'])} chosen, "
        f"{summary['seconds']} s on {summary['device']}")
    return summary
