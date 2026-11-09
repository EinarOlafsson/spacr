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
                 gene: str = "gene", prefix: str = "PC_") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Guide and gene profiles from per-cell TVN components.

    :param cells: one row per cell with the ``guide`` and ``gene`` columns,
        in the row order of ``embedding``.
    :param embedding: the TVN components, one row per cell.
    :param guide: the guide column.
    :param gene: the gene column.
    :param prefix: the component column prefix (``PC_0``, ``PC_1``, ...).
    :returns: ``(guides, genes)``: a guide is the median of its cells and a
        gene the median of its guides, as for robust z profiles.
    """
    from .gene_measurement_compare import aggregate_genes, aggregate_guides

    columns = [f"{prefix}{j}" for j in range(embedding.shape[1])]
    frame = pd.concat([cells[[guide, gene]].reset_index(drop=True),
                       pd.DataFrame(embedding, columns=columns)], axis=1)
    guides = aggregate_guides(frame, columns, guide=guide, gene=gene)
    return guides, aggregate_genes(guides, columns)


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
    :param settings: read for ``ops_tvn_mode`` and the ``ops_tvn_*``
        options (:func:`tvn_options`).
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
    guides, genes = tvn_profiles(cells.loc[use], embedding)
    info = {**info, "options": options, "cells": int(use.sum())}
    return guides, genes, info
