"""Typical variation normalisation (TVN) of optical pooled screen profiles.

TVN turns per-cell features into components in which the non-targeting
(control) cells of every batch have the same spread in every direction. It is
the profile normalisation of Brieflow (``workflow/lib/aggregate/align.py``,
Cheeseman lab, copyright 2025 Massachusetts Institute of Technology, MIT
licence). The functions here are a clean reimplementation of its
``centerscale_by_batch``, ``embed_by_pca`` and ``tvn_on_controls`` steps,
written so that the result can be checked value for value against it.

The steps, with the batch being the well:

1. columns with more than ``max_nonfinite`` non-finite values are dropped and
   the remaining gaps are filled with the column median;
2. every batch is centred and scaled on all of its cells;
3. a PCA keeping ``pca_variance`` of the variance is fitted on a random
   sample of ``pca_sample`` cells (seed ``seed``) and every cell is projected;
4. the projection is centred and scaled on the control cells, rotated onto
   the principal axes of the control cells, and every axis is scaled to unit
   variance on the control cells of each batch (the whitening);
5. CORAL aligns the control covariance of every batch to the pooled one
   (both regularised by ``coral_ridge`` times the identity).

Two implementations are given. :func:`tvn_in_memory` holds every cell at once,
as Brieflow does. :func:`fit_tvn_streamed` reads one batch at a time and keeps
only the control cells and a PCA sample, so a plate or a pool of plates never
sits in memory; :meth:`TVNModel.transform` then applies the fitted model batch
by batch.

THE WHITENING MAKES THE LOW-VARIANCE TAIL SENSITIVE TO THE PCA FIT SAMPLE.
Step 4 divides every axis by its control spread, so an axis that holds almost
no control variance is blown up to the same weight as the first one. Which
weak directions survive step 3 depends on the cells in the PCA sample, so two
fits that differ only in that sample (or in the order the batches are read)
agree on the strong axes but not on the tail. On plate A of Funk et al. 2022
the in-memory and the streamed fits give gene-pair cosines that correlate at
only r = 0.82, while their STRING AUPRC is the same. ``whiten_epsilon`` is
added to every axis's control variance at both scaling steps
(``x / sqrt(var + epsilon)`` instead of ``x / sqrt(var)``), which shrinks the
weak axes: at 3, the default, the two fits correlate at r = 0.93 and the
STRING AUPRC is at or above Brieflow's. ``whiten_epsilon=0`` is Brieflow's TVN
exactly, the setting that reproduces the plate A pilot. ``components`` keeps
only the leading control axes; it did not help on plate A and is off by
default.

This is the cell-level TVN of an optical pooled screen.
:func:`spacr.batch_correction.spherize_features` is the sphering of
well-level profiles that pycytominer calls TVN; it is a different step.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: TVN settings and their defaults. Brieflow's values, except
#: ``whiten_epsilon``: 3 rather than Brieflow's 0 (see the module notes).
TVN_DEFAULTS: Dict[str, Any] = {
    "pca_variance": 0.99,
    "pca_sample": 100000,
    "seed": 0,
    "components": None,
    "whiten_epsilon": 3.0,
    "coral_ridge": 0.5,
    "max_nonfinite": 0.05,
}

Components = Optional[Union[int, float]]


def _check_options(pca_variance, pca_sample, components, whiten_epsilon,
                   coral_ridge, max_nonfinite) -> None:
    """Reject option values that cannot work, with the reason."""
    if not 0 < float(pca_variance) <= 1:
        raise ValueError(f"pca_variance must be in (0, 1]; got {pca_variance}")
    if int(pca_sample) < 2:
        raise ValueError(f"pca_sample must be at least 2; got {pca_sample}")
    if components is not None:
        if isinstance(components, bool) or float(components) <= 0:
            raise ValueError(
                f"components must be a positive int, a fraction in (0, 1) or "
                f"None; got {components}")
        if float(components) >= 1 and float(components) != int(components):
            raise ValueError(
                f"components above 1 must be a whole number; got {components}")
    if float(whiten_epsilon) < 0:
        raise ValueError(f"whiten_epsilon must be 0 or more; got {whiten_epsilon}")
    if float(coral_ridge) < 0:
        raise ValueError(f"coral_ridge must be 0 or more; got {coral_ridge}")
    if not 0 <= float(max_nonfinite) <= 1:
        raise ValueError(f"max_nonfinite must be in [0, 1]; got {max_nonfinite}")


def _component_arg(components: Components):
    """The ``n_components`` argument of the control PCA."""
    if components is None:
        return None
    value = float(components)
    return value if value < 1 else int(value)


def _matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """A real fractional power of a symmetric positive definite matrix."""
    from scipy import linalg

    return np.real(linalg.fractional_matrix_power(matrix, power))


def _centerscale_by_batch(values: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Centre and scale every batch on all of its cells, in place."""
    from sklearn.preprocessing import StandardScaler

    for batch in pd.unique(batches):
        rows = batches == batch
        values[rows, :] = StandardScaler(copy=False).fit_transform(values[rows, :])
    return values


def _scale_rows(values: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    """``values`` centred and scaled on ``reference``; ``epsilon`` shrinks weak axes."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(copy=False).fit(reference)
    if epsilon <= 0:
        return scaler.transform(values)
    scale = np.sqrt(scaler.var_ + epsilon)
    return ((values - scaler.mean_) / scale).astype(values.dtype, copy=False)


def tvn_on_controls(embedding: np.ndarray, control: np.ndarray,
                    batches: Optional[np.ndarray] = None, *,
                    components: Components = None, whiten_epsilon: float = 3.0,
                    coral_ridge: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Whiten an embedding on its control cells and align batches with CORAL.

    :param embedding: one row per cell, for example PCA components.
    :param control: a boolean array over the rows; True marks a control cell.
    :param batches: the batch of every row (the well), or None for one batch.
    :param components: keep this many leading control axes (an int), or the
        axes holding this fraction of the control variance (a float below
        1). None keeps them all, as Brieflow does.
    :param whiten_epsilon: added to every axis's control variance before it
        is divided out, both when the embedding is first scaled on the
        control cells and when each batch is whitened, so weak axes are
        shrunk rather than blown up. 0 is Brieflow's whitening; the default,
        3, is the value that stabilised plate A of Funk et al. 2022. It is
        not the ``tvn_epsilon`` of
        :func:`spacr.batch_correction.correct_batch_effects` (default 1e-6,
        added to eigenvalues of a correlation matrix): a different knob on
        a different scale.
    :param coral_ridge: the multiple of the identity added to every
        covariance before CORAL.
    :returns: ``(embedding, info)``. ``info`` lists the axes kept and the
        batches CORAL skipped because they had fewer control cells than
        axes or a singular covariance.
    :raises ValueError: when there are no control cells.

    The steps follow Brieflow's ``tvn_on_controls``: centre and scale on all
    control cells, rotate onto the principal axes of the control cells,
    centre and scale every axis on each batch's control cells, then CORAL.
    """

    control = np.asarray(control, bool)
    if not control.any():
        raise ValueError("TVN needs control cells; none are marked")
    x = _scale_rows(embedding, embedding[control], float(whiten_epsilon))
    x = _control_rotation(x[control], components).transform(x)
    info: Dict[str, Any] = {"axes": int(x.shape[1]), "coral_skipped": []}
    if batches is None:
        return _scale_rows(x, x[control], float(whiten_epsilon)), info
    batches = np.asarray(batches)
    for batch in pd.unique(batches):
        rows = batches == batch
        reference = rows & control
        if reference.sum() == 0:
            continue
        x[rows] = _scale_rows(x[rows], x[reference], float(whiten_epsilon))
    d = x.shape[1]
    ridge = float(coral_ridge) * np.eye(d)
    target = _matrix_power(np.cov(x[control], rowvar=False, ddof=1) + ridge, 0.5)
    for batch in pd.unique(batches):
        rows = batches == batch
        reference = rows & control
        if reference.sum() < d:
            info["coral_skipped"].append(str(batch))
            continue
        source = _matrix_power(np.cov(x[reference], rowvar=False, ddof=1) + ridge, -0.5)
        if not np.all(np.isfinite(source)):
            info["coral_skipped"].append(str(batch))
            continue
        x[rows] = np.matmul(x[rows], source)
        x[rows] = np.matmul(x[rows], target)
    return x, info


def _control_basis(fit: np.ndarray, *, center: bool = True, scale: bool = False,
                   sign: str = "none") -> Dict[str, np.ndarray]:
    """The eigenbasis of the reference (control) rows: the core of every TVN in spaCR.

    The OPS TVN (:func:`tvn_on_controls`, :func:`fit_tvn_streamed`), the
    well-level sphering of :func:`spacr.batch_correction.spherize_features`
    (``correct_batch_effects(method="tvn")``) and the Measure profiles all
    take their whitening directions from here.

    :param fit: the reference rows, float64.
    :param center: subtract their mean first.
    :param scale: divide by their standard deviation (``ddof=1``) first, the
        correlation variant.
    :param sign: ``"none"`` keeps ``numpy.linalg.eigh``'s signs;
        ``"sklearn"`` flips every eigenvector so its largest-magnitude entry
        is positive, scikit-learn's PCA convention, so a projection equals
        ``PCA().fit(fit).transform`` up to rounding.
    :returns: ``mean``, ``sd`` (the reference spread, NaN-free; only
        computed with ``scale``), ``inv_sd``, ``eigvals`` (descending,
        clipped at 0) and ``eigvecs`` (columns).
    """
    mean = fit.mean(axis=0) if center else np.zeros(fit.shape[1])
    centered = fit - mean
    inv_sd = np.ones(fit.shape[1])
    sd = None
    if scale:
        sd = centered.std(axis=0, ddof=1)
        with np.errstate(divide="ignore"):
            inv_sd = 1.0 / sd
        centered = centered * inv_sd
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    if sign == "sklearn":
        pick = np.argmax(np.abs(eigvecs), axis=0)
        flip = np.sign(eigvecs[pick, np.arange(eigvecs.shape[1])])
        flip[flip == 0] = 1
        eigvecs = eigvecs * flip
    elif sign != "none":
        raise ValueError(f"sign must be 'none' or 'sklearn'; got {sign!r}")
    return {"mean": mean, "sd": sd, "inv_sd": inv_sd, "eigvals": eigvals, "eigvecs": eigvecs}


@dataclass
class _Rotation:
    """A centring plus a projection onto leading eigenvectors (a PCA without sklearn)."""

    mean_: np.ndarray
    components_: np.ndarray

    @property
    def n_components_(self) -> int:
        """The number of axes kept."""
        return int(self.components_.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Centre and project ``x``, in its own float precision."""
        x = np.asarray(x)
        out = (x.astype(np.float64) - self.mean_) @ self.components_.T
        return out.astype(np.float32 if x.dtype == np.float32 else np.float64, copy=False)


def _control_rotation(fit: np.ndarray, components: "Components") -> _Rotation:
    """The control PCA of TVN, from :func:`_control_basis`.

    ``components`` None keeps every axis (at most one fewer than the rows,
    as scikit-learn's PCA), an int that many, a fraction below 1 the
    fewest axes whose share of the variance reaches it.
    """
    basis = _control_basis(np.asarray(fit, np.float64), sign="sklearn")
    eigvals, eigvecs = basis["eigvals"], basis["eigvecs"]
    keep = min(len(eigvals), max(len(fit), 1))
    value = _component_arg(components)
    if isinstance(value, int):
        keep = min(value, keep)
    elif isinstance(value, float):
        ratio = np.cumsum(eigvals) / max(eigvals.sum(), 1e-300)
        keep = int(np.searchsorted(ratio, value, side="right")) + 1
        keep = min(keep, len(eigvals))
    return _Rotation(mean_=basis["mean"], components_=eigvecs[:, :keep].T)


def _screen_columns(values: np.ndarray, max_nonfinite: float) -> np.ndarray:
    """The columns with at most ``max_nonfinite`` non-finite values."""
    return np.isfinite(values).mean(0) >= 1 - float(max_nonfinite)


def tvn_in_memory(values: np.ndarray, batches: Sequence, control: Sequence, *,
                  pca_variance: float = 0.99, pca_sample: int = 100000,
                  seed: int = 0, components: Components = None,
                  whiten_epsilon: float = 3.0, coral_ridge: float = 0.5,
                  max_nonfinite: float = 0.05, copy: bool = True
                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """TVN with every cell in memory, as Brieflow's ``align.py`` computes it.

    :param values: one row per cell, one column per feature, worked on in
        float32, which is the precision Brieflow works in.
    :param batches: the batch of every cell (the well).
    :param control: True for every control (non-targeting) cell.
    :param pca_variance: the variance the first PCA keeps.
    :param pca_sample: the cells the first PCA is fitted on, drawn at random
        without replacement. All cells when there are fewer.
    :param seed: the seed of that draw.
    :param components: passed to :func:`tvn_on_controls`.
    :param whiten_epsilon: passed to :func:`tvn_on_controls`.
    :param coral_ridge: passed to :func:`tvn_on_controls`.
    :param max_nonfinite: columns with a larger share of non-finite values
        are dropped; the gaps left in the others are filled with the column
        median.
    :param copy: False lets a float32 ``values`` be overwritten instead of
        copied, which halves the memory a whole plate needs.
    :returns: ``(embedding, info)``: float32 TVN components, one row per
        cell, and ``info`` with the columns kept, the PCA size and the
        :func:`tvn_on_controls` notes.
    :raises ValueError: for an option out of range, mismatched lengths or no
        control cells.

    The PCA sample is centred and scaled on itself, batch by batch, before
    the fit, which is what the plate A pilot did. Everything after that sees
    every cell. The result therefore depends on ``pca_sample`` and ``seed``
    through the PCA axes; see the module notes on the low-variance tail.
    """
    _check_options(pca_variance, pca_sample, components, whiten_epsilon,
                   coral_ridge, max_nonfinite)
    batches = np.asarray(batches)
    control = np.asarray(control, bool)
    if len(control) != len(batches):
        raise ValueError("values, batches and control must have one entry per cell")
    embedding, info = _in_memory_embedding(values, batches, pca_variance=pca_variance,
                                           pca_sample=pca_sample, seed=seed,
                                           max_nonfinite=max_nonfinite, copy=copy)
    embedding, notes = tvn_on_controls(
        embedding, control, batches, components=components,
        whiten_epsilon=whiten_epsilon, coral_ridge=coral_ridge)
    return embedding.astype(np.float32, copy=False), {**info, **notes}


def _in_memory_embedding(values, batches: np.ndarray, *, pca_variance: float,
                         pca_sample: int, seed: int, max_nonfinite: float,
                         copy: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Steps 1 to 3 of :func:`tvn_in_memory`: screen, fill, centre, PCA."""
    from sklearn.decomposition import PCA

    x = np.array(values, dtype=np.float32) if copy else np.asarray(values, dtype=np.float32)
    if not len(x) == len(batches):
        raise ValueError("values, batches and control must have one entry per cell")
    keep = _screen_columns(x, max_nonfinite)
    if not keep.all():
        x = np.ascontiguousarray(x[:, keep])
    for j in range(x.shape[1]):
        column = x[:, j]
        bad = ~np.isfinite(column)
        if bad.any():
            column[bad] = np.median(column[~bad])
    spread = x.std(0)
    kept = np.flatnonzero(keep)
    if (spread == 0).any():
        x = np.ascontiguousarray(x[:, spread > 0])
        kept = kept[spread > 0]
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(x), size=min(int(pca_sample), len(x)),
                                replace=False))
    pca = PCA(n_components=float(pca_variance)).fit(
        _centerscale_by_batch(x[sample].copy(), batches[sample]))
    x = _centerscale_by_batch(x, batches)
    embedding = _project(pca, x)
    del x
    info = {"mode": "in_memory", "columns_kept": kept.tolist(),
            "pca_components": int(pca.n_components_), "pca_sample": int(len(sample)),
            "seed": int(seed)}
    return embedding, info


def _project(pca, x: np.ndarray, rows: int = 500000) -> np.ndarray:
    """``pca.transform(x)`` a block of rows at a time, to bound the copy it makes."""
    if len(x) <= rows:
        return pca.transform(x)
    out = np.empty((len(x), pca.n_components_), dtype=x.dtype)
    for start in range(0, len(x), rows):
        out[start:start + rows] = pca.transform(x[start:start + rows])
    return out


@dataclass
class TVNModel:
    """A TVN fitted batch by batch by :func:`fit_tvn_streamed`.

    :param columns: the input columns kept, as indices.
    :param medians: the fill value of every kept column.
    :param pca: the first PCA.
    :param control_mean: the control centre after the first PCA.
    :param control_scale: the control spread after the first PCA.
    :param control_pca: the PCA on the control cells.
    :param batch_stats: every batch's control centre and whitening scale.
    :param coral: every batch's CORAL matrix, or None where it was skipped.
    :param info: what the fit kept and skipped.
    """

    columns: np.ndarray
    medians: np.ndarray
    pca: Any
    control_mean: np.ndarray
    control_scale: np.ndarray
    control_pca: Any
    batch_stats: Dict[str, Tuple[np.ndarray, np.ndarray]]
    coral: Dict[str, Optional[np.ndarray]]
    info: Dict[str, Any] = field(default_factory=dict)

    def standardise(self, values: np.ndarray) -> np.ndarray:
        """Kept columns, gaps filled, centred and scaled on this batch.

        :param values: one batch's cells, every input column.
        :returns: float32 values ready for the first PCA.
        """
        x = np.array(values, dtype=np.float32)[:, self.columns]
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.broadcast_to(self.medians, x.shape)[bad]
        mean, spread = x.mean(0), x.std(0)
        spread[spread == 0] = 1
        return (x - mean) / spread

    def transform(self, values: np.ndarray, batch: Any) -> np.ndarray:
        """TVN components of one batch's cells.

        :param values: the batch's cells, every input column, the same cells
            the fit saw for this batch.
        :param batch: the batch name the fit used.
        :returns: float32 TVN components, one row per cell.
        """
        e = self.pca.transform(self.standardise(values)).astype(np.float32)
        return self._from_embedding(e, batch)

    def _from_embedding(self, e: np.ndarray, batch: Any) -> np.ndarray:
        """Steps 4 and 5 applied to a batch's first-PCA embedding (float32)."""
        e = e.astype(np.float64)
        e = self.control_pca.transform((e - self.control_mean) / self.control_scale)
        key = str(batch)
        if key in self.batch_stats:
            mean, scale = self.batch_stats[key]
            e = (e - mean) / scale
        if self.coral.get(key) is not None:
            e = e @ self.coral[key]
        return e.astype(np.float32)


def fit_tvn_streamed(batches: Sequence[Any],
                     read: Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]], *,
                     pca_variance: float = 0.99, pca_sample: int = 100000,
                     seed: int = 0, components: Components = None,
                     whiten_epsilon: float = 3.0, coral_ridge: float = 0.5,
                     max_nonfinite: float = 0.05) -> TVNModel:
    """Fit TVN one batch at a time, so the cells never sit in memory together.

    :param batches: the batch names (wells), in reading order.
    :param read: ``read(batch) -> (values, control, use)``. ``values`` holds
        every cell of the batch and every input column; ``control`` marks
        its control cells and ``use`` the cells to normalise (for example
        interphase cells). Column screening and fill medians look at every
        cell; everything else at the ``use`` cells only.
    :param pca_variance: the variance the first PCA keeps.
    :param pca_sample: the size of the first PCA's fit sample. Each batch
        contributes in proportion to its ``use`` cells.
    :param seed: the seed of the sample.
    :param components: passed on as in :func:`tvn_on_controls`.
    :param whiten_epsilon: as in :func:`tvn_on_controls`.
    :param coral_ridge: as in :func:`tvn_on_controls`.
    :param max_nonfinite: as in :func:`tvn_in_memory`.
    :returns: the fitted :class:`TVNModel`; apply it with
        :meth:`TVNModel.transform`, batch by batch.
    :raises ValueError: for an option out of range or no control cells.

    The batches are read three times: once for the column screen, once for
    the PCA sample, and once to project the control cells. Each batch is
    centred and scaled on all of its own cells before the sample is drawn,
    so the streamed fit differs from :func:`tvn_in_memory` in the PCA
    sample only. Batches with no control cells are left unwhitened.
    """
    from sklearn.decomposition import PCA

    _check_options(pca_variance, pca_sample, components, whiten_epsilon,
                   coral_ridge, max_nonfinite)
    batches = list(batches)
    nonfinite, total, medians = None, 0, []
    for batch in batches:
        values, _control, _use = read(batch)
        x = np.asarray(values, dtype=np.float32)
        counts, middle = np.zeros(x.shape[1], np.int64), np.full(x.shape[1], np.nan, np.float32)
        for j in range(x.shape[1]):
            finite = np.isfinite(x[:, j])
            counts[j] = len(x) - int(finite.sum())
            if finite.any():
                middle[j] = np.median(x[finite, j])
        nonfinite = counts if nonfinite is None else nonfinite + counts
        total += len(x)
        medians.append(middle)
    keep = nonfinite / max(total, 1) <= float(max_nonfinite)
    model = TVNModel(columns=np.flatnonzero(keep),
                     medians=np.nanmedian(np.vstack(medians), axis=0)[keep],
                     pca=None, control_mean=np.zeros(0), control_scale=np.zeros(0),
                     control_pca=None, batch_stats={}, coral={})
    sizes = [int(np.asarray(read(batch)[2], bool).sum()) for batch in batches]
    grand = max(sum(sizes), 1)
    rng = np.random.default_rng(seed)
    sample = []
    for batch, size in zip(batches, sizes):
        values, _control, use = read(batch)
        x = model.standardise(np.asarray(values)[np.asarray(use, bool)])
        take = min(int(round(int(pca_sample) * size / grand)), len(x))
        sample.append(x[rng.choice(len(x), take, replace=False)])
    sample = np.vstack(sample)
    sample_size = len(sample)
    model.pca = PCA(n_components=float(pca_variance), random_state=seed).fit(sample)
    del sample
    controls, control_batch = [], []
    for batch in batches:
        values, control, use = read(batch)
        use = np.asarray(use, bool)
        control = np.asarray(control, bool)[use]
        e = model.pca.transform(model.standardise(np.asarray(values)[use])).astype(np.float32)
        controls.append(e[control])
        control_batch.append(np.full(int(control.sum()), str(batch), dtype=object))
    nt = np.vstack(controls).astype(np.float64)
    nt_batch = np.concatenate(control_batch)
    if not len(nt):
        raise ValueError("TVN needs control cells; none are marked")
    skipped, d = _fit_control_stage(model, nt, nt_batch, components=components,
                                    whiten_epsilon=whiten_epsilon, coral_ridge=coral_ridge)
    model.info = {"mode": "streamed", "columns_kept": model.columns.tolist(),
                  "pca_components": int(model.pca.n_components_),
                  "pca_sample": int(sample_size),
                  "seed": int(seed), "axes": int(d), "coral_skipped": skipped}
    return model


def _fit_control_stage(model: "TVNModel", nt: np.ndarray, nt_batch: np.ndarray, *,
                       components: Components, whiten_epsilon: float,
                       coral_ridge: float) -> Tuple[list, int]:
    """Steps 4 and 5 of the streamed fit, from the control cells' first-PCA
    embedding: fills ``model``'s control centre, control PCA, whitening and
    CORAL, and returns the batches CORAL skipped and the axis count."""

    model.control_mean, model.control_scale = nt.mean(0), nt.std(0)
    model.control_scale[model.control_scale == 0] = 1
    if whiten_epsilon > 0:
        model.control_scale = np.sqrt(model.control_scale ** 2 + float(whiten_epsilon))
    nt = (nt - model.control_mean) / model.control_scale
    model.control_pca = _control_rotation(nt, components)
    nt = model.control_pca.transform(nt)
    names = np.unique(nt_batch)
    for name in names:
        rows = nt_batch == name
        mean, spread = nt[rows].mean(0), nt[rows].std(0)
        spread[spread == 0] = 1
        scale = np.sqrt(spread ** 2 + float(whiten_epsilon)) if whiten_epsilon > 0 else spread
        model.batch_stats[str(name)] = (mean, scale)
    whitened = np.vstack([(nt[nt_batch == n] - model.batch_stats[str(n)][0])
                          / model.batch_stats[str(n)][1] for n in names])
    d = whitened.shape[1]
    ridge = float(coral_ridge) * np.eye(d)
    target = _matrix_power(np.cov(whitened, rowvar=False, ddof=1) + ridge, 0.5)
    skipped = []
    for name in names:
        v = (nt[nt_batch == name] - model.batch_stats[str(name)][0]) / model.batch_stats[str(name)][1]
        if len(v) < d:
            model.coral[str(name)] = None
            skipped.append(str(name))
            continue
        source = _matrix_power(np.cov(v, rowvar=False, ddof=1) + ridge, -0.5)
        if np.all(np.isfinite(source)):
            model.coral[str(name)] = source @ target
        else:
            model.coral[str(name)] = None
            skipped.append(str(name))
    return skipped, d


def tvn_profiles(cells: pd.DataFrame, embedding: np.ndarray, *, guide: str = "sgRNA",
                 gene: str = "gene", prefix: str = "PC_",
                 level: str = "guides") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Guide and gene profiles from per-cell TVN components.

    :param cells: one row per cell with the ``guide`` and ``gene`` columns,
        in the row order of ``embedding``.
    :param embedding: the TVN components, one row per cell.
    :param guide: the guide column.
    :param gene: the gene column.
    :param prefix: the component column prefix (``PC_0``, ``PC_1``, ...).
    :param level: ``'guides'`` or ``'cells'``, as in
        :func:`spacr.gene_measurement_compare.aggregate_genes`.
    :returns: ``(guides, genes)``: a guide is the median of its cells and a
        gene the median of its guides (or of its cells), as for robust z
        profiles.
    """
    from .gene_measurement_compare import aggregate_genes, aggregate_guides

    columns = [f"{prefix}{j}" for j in range(embedding.shape[1])]
    frame = pd.concat([cells[[guide, gene]].reset_index(drop=True),
                       pd.DataFrame(embedding, columns=columns)], axis=1)
    guides = aggregate_guides(frame, columns, guide=guide, gene=gene)
    return guides, aggregate_genes(guides, columns, level=level, cells=frame,
                                   gene=gene)


def gene_pair_cosines(profiles: pd.DataFrame, columns: Sequence[str],
                      key: str = "gene") -> pd.Series:
    """Cosine similarity of every pair of rows, keyed by the two names.

    :param profiles: one row per gene (or guide).
    :param columns: the profile columns.
    :param key: the name column.
    :returns: the upper-triangle cosines, indexed by ``(name_a, name_b)``
        in the table's row order. Two TVN fits can be compared by
        correlating the cosines they give for the same pairs.
    """
    x = profiles[list(columns)].to_numpy(float)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    upper = np.triu_indices(len(x), 1)
    names = profiles[key].to_numpy()
    index = pd.MultiIndex.from_arrays([names[upper[0]], names[upper[1]]])
    return pd.Series((x @ x.T)[upper], index=index)


def _tvn_setting(settings: Mapping[str, Any], key: str, name: str,
                 cast: Callable[[Any], Any]) -> Any:
    """One TVN option from its setting, or its default when unset.

    :param settings: the run settings.
    :param key: the setting, such as ``ops_tvn_seed``.
    :param name: the option's key in :data:`TVN_DEFAULTS`.
    :param cast: turns the setting's value into the option's type.
    :returns: the cast value, or the default for a missing or blank setting.
    """
    value = settings.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return TVN_DEFAULTS[name]
    return cast(value)


def _tvn_components(value: Any) -> Union[int, float]:
    """The ``components`` option from a setting value.

    :param value: a number, or its text.
    :returns: an int count of components when at least 1, else the float
        share of variance to keep.
    """
    number = float(value)
    return int(number) if number >= 1 else number


def _read_well(well: Any, *, table_of: Mapping[Any, Tuple[int, np.ndarray]],
               cache: Dict[Any, np.ndarray], sources, read_columns,
               features: Sequence[str], rows: np.ndarray, offsets: np.ndarray,
               control: np.ndarray, use: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One well's feature values for the streamed TVN, keeping one well cached.

    :param well: the well.
    :param table_of: per well, its cell table's index in ``sources`` and the
        mask of the design cells it holds.
    :param cache: holds the last well read; cleared before another is read.
    :param sources: ``(path, names, rows)`` per well's cell table.
    :param read_columns: ``read_columns(sources, columns)`` reads columns.
    :param features: the feature columns.
    :param rows: every design cell's row in the stacked tables.
    :param offsets: the first stacked row of each cell table, and the total.
    :param control: the control cells.
    :param use: the cells to profile.
    :returns: ``(values, control, use)`` for the well's design cells.
    """
    i, inside = table_of[well]
    if well not in cache:
        cache.clear()
        frame = read_columns([sources[i]], list(features))
        frame = frame.iloc[rows[inside] - offsets[i]]
        cache[well] = frame.to_numpy(np.float32)
        del frame
    return cache[well], control[inside], use[inside]


def tvn_options(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """TVN keyword arguments from ``ops_tvn_*`` settings.

    :param settings: read for ``ops_tvn_pca_sample``, ``ops_tvn_seed``,
        ``ops_tvn_components`` and ``ops_tvn_whiten_epsilon``.
    :returns: keyword arguments for :func:`tvn_in_memory` and
        :func:`fit_tvn_streamed`; a missing or empty setting takes its
        default from :data:`TVN_DEFAULTS`.
    """
    return {"pca_sample": _tvn_setting(settings, "ops_tvn_pca_sample", "pca_sample", int),
            "seed": _tvn_setting(settings, "ops_tvn_seed", "seed", int),
            "components": _tvn_setting(settings, "ops_tvn_components", "components",
                                       _tvn_components),
            "whiten_epsilon": _tvn_setting(settings, "ops_tvn_whiten_epsilon",
                                           "whiten_epsilon", float)}



#: Where the TVN can run: every cell at once, or one well at a time.
TVN_MODES = ("in_memory", "streamed")


def _aggregate_tvn(sources, rows: np.ndarray, cells: pd.DataFrame, use: np.ndarray,
                   control: np.ndarray, features: Sequence[str],
                   settings: Mapping[str, Any], read_columns, block: int = 48
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """TVN guide and gene tables for the OPS aggregate phase.

    :param sources: ``(path, names, rows)`` per well's cell table, in the
        order ``read_columns`` stacks them.
    :param rows: every design cell's row in the stacked tables.
    :param cells: the design cells, with ``well``, ``sgRNA`` and ``gene``.
    :param use: the cells to profile (interphase cells when the run pools
        interphase only).
    :param control: the control cells.
    :param features: the feature columns.
    :param settings: read for ``ops_tvn_mode``, ``ops_aggregate_level`` and
        the ``ops_tvn_*`` options (:func:`tvn_options`); with
        ``_keep_embedding`` the returned info also holds the profiled cells
        (``_cells``) and their components (``_embedding``).
    :param read_columns: ``read_columns(sources, columns)`` reads columns.
    :param block: columns read at a time by the in-memory mode.
    :returns: ``(guides, genes, info)``.

    The cells are taken in the order of the cell tables, well by well, so
    the PCA sample is the same whichever order the joins left them in.
    """
    mode = str(settings.get("ops_tvn_mode") or "streamed").strip().lower()
    if mode not in TVN_MODES:
        raise ValueError(f"ops_tvn_mode must be one of {list(TVN_MODES)}; got {mode}")
    options = tvn_options(settings)
    order = np.argsort(rows, kind="stable")
    rows, use, control = rows[order], np.asarray(use, bool)[order], np.asarray(control, bool)[order]
    cells = cells.iloc[order].reset_index(drop=True)
    wells = cells["well"].astype(str).to_numpy()
    offsets = np.cumsum([0] + [int(s[2]) for s in sources])
    if mode == "in_memory":
        picked = rows[use]
        x = np.empty((len(picked), len(features)), np.float32)
        for start in range(0, len(features), block):
            names = list(features[start:start + block])
            x[:, start:start + len(names)] = read_columns(sources, names).to_numpy(
                np.float32)[picked]
        embedding, info = tvn_in_memory(x, wells[use], control[use], copy=False, **options)
        del x
    else:
        table_of = {}
        for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            inside = (rows >= start) & (rows < stop)
            if inside.any():
                table_of[wells[inside][0]] = (i, inside)
        names = [w for w in pd.unique(wells) if w in table_of]
        cache: Dict[str, np.ndarray] = {}
        read = partial(_read_well, table_of=table_of, cache=cache, sources=sources,
                       read_columns=read_columns, features=features, rows=rows,
                       offsets=offsets, control=control, use=use)
        model = fit_tvn_streamed(names, read, **options)
        parts = []
        for well in names:
            values, _c, keep = read(well)
            parts.append(model.transform(values[keep], well))
        cache.clear()
        index = np.concatenate([np.flatnonzero(table_of[w][1] & use) for w in names])
        embedding = np.empty((int(use.sum()), parts[0].shape[1]), np.float32)
        position = np.empty(len(cells), np.int64)
        position[np.flatnonzero(use)] = np.arange(int(use.sum()))
        embedding[position[index]] = np.vstack(parts)
        info = model.info
    guides, genes = tvn_profiles(cells.loc[use], embedding,
                                 level=str(settings.get("ops_aggregate_level")
                                           or "guides"))
    info = {**info, "options": options, "cells": int(use.sum())}
    if settings.get("_keep_embedding"):
        info["_cells"] = cells.loc[use].reset_index(drop=True)
        info["_embedding"] = embedding
    return guides, genes, info


#: The profile recipe of :func:`profile_cell_tables` and its defaults: no
#: outlier filter, median imputation, the per-well standardised PCA, gene =
#: median of its guides, non-targeting guides pooled four at a time into
#: pseudo-genes.
PROFILE_RECIPE: Dict[str, Any] = {"lof": 0, "impute": "median", "pca": "c0", "agg": "guide", "nt": "pseudo"}

#: The channels whose mean intensities the outlier filter looks at.
PROFILE_CHANNELS = ("dapi", "gfp", "a594", "af750")

_RECIPE_CHOICES = {"lof": (0, 1), "impute": ("median", "knn"), "pca": ("c0", "brieflow"),
                   "agg": ("guide", "gene"), "nt": ("pseudo", "barcode")}


def profile_recipe(value: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A profile recipe with its defaults filled in and checked.

    :param value: any of the :data:`PROFILE_RECIPE` keys. ``lof`` 1 drops
        the 1 % of cells a local outlier factor finds on the mean
        intensities; ``impute`` ``'knn'`` fills gaps by five nearest
        neighbours, as Brieflow does; ``pca`` ``'brieflow'`` fits the PCA on
        100,000 cells standardised by well with Brieflow's sampling; ``agg``
        ``'gene'`` takes the median of all of a gene's cells; ``nt``
        ``'barcode'`` keeps each non-targeting guide as its own row.
    :returns: the recipe.
    :raises ValueError: for an unknown key or choice.
    """
    recipe = dict(PROFILE_RECIPE)
    unknown = set(value or {}) - set(PROFILE_RECIPE)
    if unknown:
        raise ValueError(f"ops_profile_recipe has unknown keys {sorted(unknown)}; use {list(PROFILE_RECIPE)}")
    recipe.update(value or {})
    for key, choices in _RECIPE_CHOICES.items():
        if recipe[key] not in choices:
            raise ValueError(f"ops_profile_recipe[{key!r}] must be one of {list(choices)}; got {recipe[key]!r}")
    return recipe


def _knn_impute(features: pd.DataFrame, batch_size: int = 1000, sample_size: int = 10000) -> pd.DataFrame:
    """Fill gaps by five nearest neighbours in batches, as Brieflow's filter does.

    Each batch of rows with a gap is imputed together with a random sample
    of complete rows (``np.random.seed(42)``), on the columns that have gaps.
    """
    from sklearn.impute import KNNImputer

    gaps = features.columns[features.isna().any()].tolist()
    if not gaps:
        return features
    for column in gaps:
        if pd.api.types.is_integer_dtype(features[column]):
            features[column] = features[column].astype("float64")
    has_gap = features[gaps].isna().any(axis=1).to_numpy()
    gap_rows, full_rows = np.flatnonzero(has_gap), np.flatnonzero(~has_gap)
    positions = [features.columns.get_loc(c) for c in gaps]
    np.random.seed(42)
    for start in range(0, len(gap_rows), batch_size):
        batch = gap_rows[start:start + batch_size]
        sample = np.random.choice(full_rows, size=min(sample_size, len(full_rows)), replace=False)
        values = KNNImputer(n_neighbors=5).fit_transform(
            features.iloc[np.concatenate([batch, sample]), positions].to_numpy())
        features.iloc[batch, positions] = values[:len(batch)]
    return features


def _profile_well(path: str, plate: str, well: str, features: Sequence[str], recipe: Mapping[str, Any],
                  columns: Optional[List[str]], control_prefix: str
                  ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """One well's interphase cells, filtered and imputed, ready for the PCA.

    Mitotic cells are dropped; non-finite values become gaps; columns with
    5 % or more gaps and cells with 1 % or more are dropped (Brieflow's
    ``missing_values_filter``); remaining gaps are imputed; with ``lof``
    the outlier filter runs; the columns are aligned to the first well's.

    :returns: ``(meta, values, columns)``: ``meta`` has ``plate``, ``well``,
        ``gene_symbol_0`` (control genes suffixed with their guide),
        ``cell_barcode_0`` and ``batch_values`` (plate_well).
    """
    from .tabular import read_table

    cells = read_table(path, canonicalise=False, report=None,
                       columns=["well", "sgRNA", "gene", "mitotic"] + list(features))
    cells = cells[~cells.mitotic].drop(columns="mitotic").reset_index(drop=True)
    meta = pd.DataFrame({"plate": plate, "well": well, "gene_symbol_0": cells.gene.astype(str),
                         "cell_barcode_0": cells.sgRNA.astype(str)})
    x = cells[list(features)].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    del cells
    if x.isna().any().any():
        x = x.drop(columns=x.columns[x.isna().mean() >= 0.05].tolist())
        keep = x.isna().sum(axis=1) / x.shape[1] < 0.01
        x, meta = x.loc[keep], meta.loc[keep]
        if recipe["impute"] == "knn":
            x = _knn_impute(x)
    if recipe["impute"] == "median":
        x = x.fillna(x.median())
    if recipe["lof"]:
        from sklearn.neighbors import LocalOutlierFactor

        intensity = [c for c in features if any(c.endswith(f"_{ch}_mean_intensity") for ch in PROFILE_CHANNELS)]
        inlier = LocalOutlierFactor(contamination=0.01, n_jobs=2).fit_predict(x[[c for c in intensity if c in x]])
        meta, x = meta[inlier == 1], x[inlier == 1]
    meta, x = meta.reset_index(drop=True), x.reset_index(drop=True)
    if columns is not None:
        x = x.reindex(columns=columns)
        x = x.fillna(x.median()).fillna(0.0)
    meta["batch_values"] = meta["plate"].astype(str) + "_" + meta["well"].astype(str)
    control = meta["gene_symbol_0"].astype(str).str.startswith(control_prefix)
    meta.loc[control, "gene_symbol_0"] = (meta.loc[control, "gene_symbol_0"].astype(str) + "_"
                                          + meta.loc[control, "cell_barcode_0"].astype(str))
    return meta, x.to_numpy(), list(x.columns)


def _profile_embedding(values: List[np.ndarray], metas: List[pd.DataFrame], recipe: Mapping[str, Any]
                       ) -> Tuple[np.ndarray, int]:
    """The per-cell PCA of every well, fitted on a 100,000-cell sample.

    ``'c0'`` standardises each well on its own cells and samples each well
    in proportion to its size (``default_rng(0)``; every cell when there
    are fewer than 100,000); ``'brieflow'`` samples
    the pooled cells with ``np.random.seed(0)`` and centres and scales by
    well, as Brieflow's aggregate does. ``'c0'`` scales ``values`` in place.

    :returns: ``(components, n_components)``; single precision.
    """
    from sklearn.decomposition import PCA

    sizes = [len(m) for m in metas]
    if recipe["pca"] == "brieflow":
        np.random.seed(0)
        pick = np.sort(np.random.choice(sum(sizes), size=min(100000, sum(sizes)), replace=False))
        offsets = np.cumsum([0] + sizes)
        sample, sample_meta = [], []
        for i, x in enumerate(values):
            k = pick[(pick >= offsets[i]) & (pick < offsets[i + 1])] - offsets[i]
            sample.append(x[k])
            sample_meta.append(metas[i].iloc[k].reset_index(drop=True))
        batches = pd.concat(sample_meta, ignore_index=True)["batch_values"].to_numpy()
        pca = PCA(n_components=0.99).fit(_centerscale_by_batch(np.vstack(sample).copy(), batches))
        parts = [pca.transform(_centerscale_by_batch(x.copy(), m["batch_values"].to_numpy())).astype(np.float32)
                 for x, m in zip(values, metas)]
    else:
        rng = np.random.default_rng(0)
        take = [min(s, int(round(100000 * s / sum(sizes)))) for s in sizes]
        scaled = []
        for x in values:
            mu, sd = x.mean(0), x.std(0)
            sd[sd == 0] = 1
            x -= mu
            x /= sd
            scaled.append(x.astype(np.float32, copy=False))
        pca = PCA(n_components=0.99, random_state=0).fit(
            np.vstack([z[rng.choice(len(z), k, replace=False)] for z, k in zip(scaled, take)]))
        parts = [pca.transform(z).astype(np.float32) for z in scaled]
        del scaled
    return np.vstack(parts), int(pca.n_components_)


def nt_pseudo_genes(guides: Sequence[str], genes: Sequence[str], control_guides: Sequence[str], *,
                    size: int = 4, seed: int = 0, control_prefix: str = "nontargeting") -> np.ndarray:
    """Pool non-targeting guides into pseudo-genes of ``size`` guides.

    The sorted control guides are permuted with ``default_rng(seed)`` and
    taken ``size`` at a time as ``NTpseudo_00``, ``NTpseudo_01``, ...; a
    control row whose guide is not listed becomes ``nontargeting_drop``.

    :param guides: the guide of every row.
    :param genes: the gene of every row.
    :param control_guides: every control guide of the screen.
    :param size: guides per pseudo-gene.
    :param seed: the permutation seed.
    :param control_prefix: the control gene prefix.
    :returns: the gene names with control rows renamed.
    """
    pool = np.array(sorted(set(map(str, control_guides))))
    order = np.argsort(np.random.default_rng(seed).permutation(len(pool)))
    names = {g: "NTpseudo_%02d" % (k // size) for g, k in zip(pool, order)}
    return np.array([names.get(str(s), "nontargeting_drop") if str(g).startswith(control_prefix) else g
                     for s, g in zip(guides, genes)])


def _cell_tables(value: Any, plate: str, wells: Optional[Sequence[str]]) -> List[Tuple[str, str, str]]:
    """``(plate, well, path)`` of every stored cell table to profile."""
    import glob

    folders = value if isinstance(value, Mapping) else {plate or "A": value}
    out = []
    for name, folder in sorted(folders.items()):
        if not folder or not os.path.isdir(str(folder)):
            raise ValueError(f"ops_profile_cells: {folder!r} is not a folder of raw_<well>.parquet cell tables")
        found = {os.path.basename(p)[4:-8]: p for p in glob.glob(os.path.join(str(folder), "raw_*.parquet"))}
        chosen = sorted(found) if not wells else [str(w) for w in wells]
        missing = [w for w in chosen if w not in found]
        if missing or not chosen:
            raise ValueError(f"plate {name}: no stored cell table for wells {missing or 'any'} in {folder}")
        out += [(str(name), w, found[w]) for w in chosen]
    return out


def profile_cell_tables(settings: Mapping[str, Any], report: Optional[Callable[[str], None]] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Gene profiles from stored per-well cell tables, without any image step.

    The cells of every well are filtered and imputed (:func:`_profile_well`),
    embedded by a PCA keeping 99 % of the variance
    (:func:`_profile_embedding`), normalised by TVN on the control cells
    with the well as batch (:func:`tvn_on_controls`, whitening epsilon
    ``ops_tvn_whiten_epsilon``; 0 is Brieflow's TVN and is computed at
    single precision as Brieflow does, a positive value at double), and
    reduced to guides (median of cells) and genes (median of guides, or of
    cells), keeping rows with more than ``ops_profile_min_cells`` cells. A
    hits table can then keep only active genes and the control rows.

    :param settings: read for ``ops_profile_cells`` (a folder of
        ``raw_<well>.parquet`` tables with ``well``, ``sgRNA``, ``gene``,
        ``mitotic`` and feature columns, or ``{plate: folder}``),
        ``ops_profile_wells``, ``plate``, ``ops_profile_recipe``
        (:func:`profile_recipe`), ``ops_tvn_whiten_epsilon``,
        ``ops_profile_min_cells``, ``ops_profile_active_filter``
        (``{"hits_table": path, "fdr": 0.05}``: genes with any ``*_fdr``
        column below ``fdr``), ``ops_profile_shrinkage`` (``'eb_nt'``:
        :func:`spacr.gene_measurement_compare.shrink_gene_profiles`, with
        ``ops_profile_shrinkage_prior_df``), ``ops_profile_weighting``
        (``'contrastive_wiener'``:
        :func:`spacr.gene_measurement_compare.contrastive_weighting`) and
        ``ops_control_prefix``. Shrinkage and weighting run on every gene
        before the active-gene filter, as in the aggregate phase.
    :param report: called with progress lines.
    :returns: ``(genes, guides, info)``: ``genes`` has ``gene``,
        ``n_cells`` and ``PC_*`` (``w_*`` once weighted); ``guides`` has
        ``guide``, ``gene``, ``n_cells`` and the same columns; ``info``
        counts cells and components, times each step and, when used,
        records the shrinkage or weighting.
    :raises ValueError: for a bad setting, a missing table, an unknown
        shrinkage or weighting, or both asked for at once.
    """
    import time

    from .ops_engine import _ph_feature_columns
    from .tabular import read_table, table_columns

    shrinkage = str(settings.get("ops_profile_shrinkage") or "").strip().lower() or None
    weighting = str(settings.get("ops_profile_weighting") or "").strip().lower() or None
    if shrinkage not in (None, "eb_nt"):
        raise ValueError(f"ops_profile_shrinkage must be empty or eb_nt; got {shrinkage!r}")
    if weighting not in (None, "contrastive_wiener"):
        raise ValueError(f"ops_profile_weighting must be empty or contrastive_wiener; got {weighting!r}")
    if shrinkage and weighting:
        raise ValueError("ops_profile_weighting and ops_profile_shrinkage both remove what the control "
                         "guides explain; stacked, no direction is left, so choose one")
    say = report or (lambda text: None)
    recipe = profile_recipe(settings.get("ops_profile_recipe"))
    prefix = str(settings.get("ops_control_prefix") or "nontargeting")
    epsilon = float(settings["ops_tvn_whiten_epsilon"])
    min_cells = int(settings.get("ops_profile_min_cells", 10))
    tables = _cell_tables(settings.get("ops_profile_cells"), str(settings.get("plate") or ""),
                          settings.get("ops_profile_wells"))
    features = _ph_feature_columns(table_columns(tables[0][2], canonicalise=False))
    timing: Dict[str, Any] = {"plates": sorted({t[0] for t in tables})}
    started = time.time()
    metas, values, columns, control_guides = [], [], None, set()
    for plate, well, path in tables:
        listed = read_table(path, canonicalise=False, report=None, columns=["sgRNA", "gene"])
        control_guides |= set(listed.sgRNA[listed.gene.astype(str).str.startswith(prefix)].astype(str))
        meta, x, cols = _profile_well(path, plate, well, features, recipe, columns, prefix)
        columns = columns or cols
        metas.append(meta)
        values.append(x)
        say(f"profile: {plate} {well}: {len(meta)} interphase cells")
    timing["load_and_filter_s"] = round(time.time() - started, 1)
    started = time.time()
    embedding, n_components = _profile_embedding(values, metas, recipe)
    del values
    meta = pd.concat(metas, ignore_index=True)
    control = meta.gene_symbol_0.str.startswith(prefix).to_numpy()
    batches = meta.batch_values.to_numpy()
    if epsilon > 0:
        embedding = tvn_on_controls(embedding.astype(np.float64), control, batches, whiten_epsilon=epsilon)[0]
    else:
        embedding = tvn_on_controls(embedding, control, batches, whiten_epsilon=0.0)[0]
    timing["pca_tvn_s"] = round(time.time() - started, 1)
    started = time.time()
    pcs = [f"PC_{j}" for j in range(embedding.shape[1])]
    guide = np.where(control, meta.gene_symbol_0.str.split("_", n=1).str[1], meta.cell_barcode_0).astype(str)
    gene = np.where(control, prefix, meta.gene_symbol_0).astype(str)
    if recipe["nt"] == "pseudo":
        gene = nt_pseudo_genes(guide, gene, sorted(control_guides), control_prefix=prefix)
    else:
        gene = np.where(control, np.char.add(f"{prefix}_", guide), gene)
    cells = pd.DataFrame(embedding.astype(np.float32), columns=pcs)
    cells["guide"], cells["gene"] = guide, gene
    cells = cells[cells.gene != "nontargeting_drop"]
    grouped = cells.groupby(["guide", "gene"])
    guides = grouped[pcs].median()
    guides.insert(0, "n_cells", grouped.size())
    guides = guides.reset_index()
    if recipe["agg"] == "gene":
        grouped = cells.groupby("gene")
        genes = grouped[pcs].median()
        genes.insert(0, "n_cells", grouped.size())
    else:
        grouped = guides.groupby("gene")
        genes = grouped[pcs].median()
        genes.insert(0, "n_cells", grouped.n_cells.sum())
    genes = genes[genes.n_cells > min_cells].reset_index()
    adjusted: Dict[str, Any] = {}
    if shrinkage:
        genes, guides, adjusted["shrinkage"] = _profile_shrink(genes, guides, pcs, prefix, settings)
    if weighting:
        genes, guides, adjusted["weighting"] = _profile_weight(genes, guides, pcs, prefix)
    active = settings.get("ops_profile_active_filter")
    if active:
        hits = read_table(active["hits_table"], canonicalise=False, report=None)
        fdr = [c for c in hits.columns if c.endswith("_fdr")]
        keep = set(hits.gene[(hits[fdr] < float(active.get("fdr", 0.05))).any(axis=1)])
        is_control = genes.gene.str.lower().str.startswith((prefix.lower(), "ntpseudo"))
        genes = genes[genes.gene.isin(keep) | is_control].reset_index(drop=True)
        is_control = guides.gene.str.lower().str.startswith((prefix.lower(), "ntpseudo"))
        guides = guides[guides.gene.isin(keep) | is_control].reset_index(drop=True)
    timing["aggregate_s"] = round(time.time() - started, 1)
    info = {"cells": int(len(cells)), "pcs": n_components, "tvn_dims": int(embedding.shape[1]),
            "features": len(columns), "recipe": recipe, "tvn_whiten_epsilon": epsilon, "runtime": timing,
            **adjusted}
    return genes, guides, info


def _profile_shrink(genes: pd.DataFrame, guides: pd.DataFrame, pcs: Sequence[str], prefix: str,
                    settings: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Shrink the gene profiles toward the controls by their guides' reliability (item 902).

    :param genes: the gene rows, ``gene``, ``n_cells`` and ``pcs``.
    :param guides: the guide rows, ``guide``, ``gene``, ``n_cells`` and ``pcs``.
    :param pcs: the profile columns.
    :param prefix: the control gene prefix.
    :param settings: read for ``ops_profile_shrinkage_prior_df`` (4 when empty).
    :returns: ``(genes, guides, record)``: the shrunk gene rows in the order
        of ``genes``, the guide rows unchanged, and the shrinkage parameters
        with the median shrink factor.
    """
    from .gene_measurement_compare import shrink_gene_profiles

    prior = settings.get("ops_profile_shrinkage_prior_df")
    prior = 4.0 if prior is None else float(prior)
    shrunk, factor, params = shrink_gene_profiles(guides, pcs, control_prefix=prefix, prior_df=prior)
    out = genes[["gene", "n_cells"]].merge(shrunk[["gene", *pcs]], on="gene", how="inner")
    return out, guides, {**params, "median_factor": float(factor.median())}


def _profile_weight(genes: pd.DataFrame, guides: pd.DataFrame, pcs: Sequence[str], prefix: str
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Weight the gene and guide profiles by signal over guide noise (item 907).

    :param genes: the gene rows, ``gene``, ``n_cells`` and ``pcs``.
    :param guides: the guide rows, ``guide``, ``gene``, ``n_cells`` and ``pcs``.
    :param pcs: the profile columns.
    :param prefix: the control gene prefix.
    :returns: ``(genes, guides, record)``: both tables in the weighted space
        (``w_*`` columns) and the directions kept, ``kbar`` and the number of
        control guides.
    :raises ValueError: when fewer than two directions rise above the noise.
    """
    from .gene_measurement_compare import contrastive_weighting

    weighted_genes, weighted_guides, info = contrastive_weighting(genes, guides, pcs, control_prefix=prefix)
    record = {"directions_kept": int(info["directions_kept"]), "kbar": float(info["kbar"]),
              "nt_guides": int(info["nt_guides"])}
    return weighted_genes, weighted_guides, record


def _profile_covariates(settings: Mapping[str, Any], guides: pd.DataFrame) -> Dict[str, float]:
    """Cells per guide of every gene: from ``ops_eval_covariates`` or the guide table."""
    from .tabular import read_table

    path = settings.get("ops_eval_covariates")
    if path:
        table = read_table(path, canonicalise=False, report=None)
        return dict(zip(table.gene, table.cells_per_guide.astype(float)))
    grouped = guides.groupby("gene").n_cells
    return (grouped.sum() / grouped.size()).astype(float).to_dict()


#: The environment variables that set the thread count of the BLAS and
#: OpenMP libraries numpy, scipy and scikit-learn load.
_BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _pinned_blas(pin: bool):
    """Hold the BLAS and OpenMP thread pools to one thread while the block runs.

    A matrix product summed over several threads may add in another order
    from one run to the next, so the last bits of a PCA or a covariance can
    differ; one thread fixes the order. threadpoolctl limits the pools
    already loaded; without it the thread-count variables are set, which
    only reaches libraries loaded afterwards. Both are restored on exit.

    :param pin: False leaves every pool as it is.
    :returns: a context manager that yields ``"threadpoolctl"``,
        ``"environment"`` or None (nothing pinned).
    """
    if not pin:
        yield None
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            yield "threadpoolctl"
        return
    saved = {name: os.environ.get(name) for name in _BLAS_THREAD_VARIABLES}
    try:
        for name in _BLAS_THREAD_VARIABLES:
            os.environ[name] = "1"
        yield "environment"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_profile_phases(settings: Mapping[str, Any], phases: Sequence[str] = ("profile", "evaluate"),
                       report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The OPS profile path: stored cell tables -> gene profiles -> metrics.

    ``profile`` runs :func:`profile_cell_tables` and writes ``genes.parquet``
    and ``guides.parquet``; ``evaluate`` embeds the gene rows with more than
    ``ops_profile_min_cells`` cells (:func:`spacr.ops_reference.embed_gene_profiles`)
    and writes ``metrics.csv`` (:func:`spacr.ops_reference.evaluate_gene_profiles`),
    ``metrics.extra.csv`` (:func:`spacr.ops_reference.specificity_metrics`)
    and ``per_gene.csv.gz``. Everything goes to
    ``<dst_root>/ops_profile/<ops_profile_run>``, with ``runtime.json``
    holding the settings, counts and seconds per step.

    :param settings: the OPS settings; missing keys take the defaults of
        :func:`spacr.ops_settings.ops_defaults`. Read here: ``dst_root``,
        ``ops_profile_run`` (the plates when empty), ``ops_profile_similarity``
        (empty, ``'cosine'`` or ``'gene_cosine'``), ``ops_profile_min_cells``, ``ops_reference_dir``,
        ``ops_eval_covariates`` (a table with ``gene`` and
        ``cells_per_guide``; the guide table's own cells per guide when
        empty), ``ops_deterministic_blas`` (True holds the BLAS and OpenMP
        pools to one thread, so a re-run of the same settings file gives
        bit-identical tables), the ``ops_eval_*`` settings and those of
        :func:`profile_cell_tables`.
    :param phases: ``"profile"``, ``"evaluate"`` or both.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "info"}``.
    :raises ValueError: for a missing ``dst_root``, an unknown similarity,
        or an ``evaluate`` with no profile table to read.
    """
    import json
    import time

    from . import ops_reference
    from .ops_settings import ops_defaults
    from .tabular import read_table, write_table

    settings = ops_defaults(dict(settings))
    say = report or (lambda text: None)
    if not settings.get("dst_root"):
        raise ValueError("the profile phases need dst_root, the folder their results go to")
    if str(settings.get("ops_profile_similarity") or "cosine") not in ("cosine", "gene_cosine"):
        raise ValueError("the profile phases compare gene profiles by cosine: ops_profile_similarity must be "
                         f"empty, cosine or gene_cosine; got {settings['ops_profile_similarity']!r}")
    cells = settings.get("ops_profile_cells")
    if settings.get("ops_profile_run"):
        run = str(settings["ops_profile_run"])
    elif isinstance(cells, Mapping):
        run = "".join(sorted(map(str, cells)))
    else:
        run = str(settings.get("plate") or "A")
    folder = os.path.join(str(settings["dst_root"]), "ops_profile", run)
    os.makedirs(folder, exist_ok=True)
    outputs: Dict[str, str] = {}
    info: Dict[str, Any] = {}
    genes_path, guides_path = os.path.join(folder, "genes.parquet"), os.path.join(folder, "guides.parquet")
    with _pinned_blas(bool(settings.get("ops_deterministic_blas", True))) as pinned:
        if "profile" in phases:
            genes, guides, info = profile_cell_tables(settings, report=say)
            outputs["genes"] = write_table(genes, genes_path, canonicalise=False)
            outputs["guides"] = write_table(guides, guides_path, canonicalise=False)
            say(f"profile {run}: {len(genes)} genes, {len(guides)} guides from {info['cells']} cells")
        if "evaluate" in phases:
            if not os.path.isfile(genes_path):
                raise ValueError(f"evaluate needs the profile table {genes_path}; run the profile phase first")
            started = time.time()
            genes = read_table(genes_path, canonicalise=False, report=None)
            guides = read_table(guides_path, canonicalise=False, report=None)
            genes = genes[genes.n_cells > int(settings["ops_profile_min_cells"])]
            columns = [c for c in genes.columns if c.startswith(("PC_", "w_"))]
            names = genes.gene.to_numpy()
            pcs = ops_reference.embed_gene_profiles(genes[columns].to_numpy(float))
            fitness = _profile_covariates(settings, guides)
            reference_dir = settings.get("ops_reference_dir")
            metrics, per_gene = ops_reference.evaluate_gene_profiles(
                names, pcs, fitness, reference_dir, settings, arm=run,
                null_cache=os.path.join(str(settings["dst_root"]), "ops_profile", "null_cache"))
            extra = ops_reference.specificity_metrics(names, pcs, fitness, reference_dir, settings, arm=run)
            for name, table in (("metrics.csv", metrics), ("metrics.extra.csv", extra),
                                ("per_gene.csv.gz", per_gene)):
                outputs[name] = write_table(table, os.path.join(folder, name), canonicalise=False)
            info.setdefault("runtime", {})["metrics_s"] = round(time.time() - started, 1)
            info["references"] = ops_reference.reference_record(
                sorted({r for r in per_gene.reference.unique()}), reference_dir)
            say(f"evaluate {run}: {int(metrics.genes.iloc[0])} genes scored")
    info["blas_threads"] = pinned
    keys = sorted(k for k in settings if k.startswith(("ops_profile_", "ops_eval_", "ops_reference_"))
                  or k in ("ops_tvn_whiten_epsilon", "plate", "ops_control_prefix", "ops_deterministic_blas"))
    record = {"settings": {k: settings[k] for k in keys}, "phases": list(phases), "outputs": outputs, **info}
    with open(os.path.join(folder, "runtime.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    return {"folder": folder, "outputs": outputs, "info": info}
