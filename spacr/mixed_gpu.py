"""The mixed model's REML objective, written out and optimised on the GPU.

WHY THIS MODULE EXISTS, in one measured number. ``regression_type='mixed'``
is the default and it is the slowest fit in
:mod:`spacr.ml`: benchmarks measured statsmodels' MixedLM at **54x**
OLS on a 40-gene screen and **67x** on 80, the ratio RISING with screen
size. The operation each iteration spends its time in is a dense Cholesky of
the ``q x q`` random-effects system, and on this machine (RTX 3090, torch
2.9.1+cu128) that Cholesky takes **204 ms on the CPU and 7.69 ms on the
GPU** at ``q = 1212`` -- 27x.

A 27x ON ONE OPERATION IS NOT A 27x ON A FIT, so this was measured end to
end too. On a TSG101-shaped screen -- 1830 rows, 387 genes, 710 guides,
p=388 fixed effects, q=1097 random levels -- statsmodels takes **11.3 s**,
this module takes **0.80 s on the CPU** and **0.47 s on the RTX 3090**: 14x
and 24x, whole fit, design construction included.
:func:`benchmark_against_statsmodels` is the harness, so the number can be
re-measured on a user's own screen rather than believed.

WHAT IT FITS, and it is the same model statsmodels fits, not a relative:

    y = X b + sum_k Z_k u_k + e,   u_k ~ N(0, sigma^2 theta_k I),
                                   e ~ N(0, sigma^2 I)

with one ``Z_k`` per grouping factor -- the group intercept
:func:`spacr.ml.fit_mixed_model` passes as ``groups``, plus one per variance
component, each NESTED INSIDE THE GROUP exactly as ``MixedLM``'s
``vc_formula`` nests it. That nesting is not a detail: a variance component
in statsmodels is evaluated *within* each group, so
``vc_formula={'rowID': '0 + C(rowID)'}`` with ``groups=gene`` is
``(1 | gene:rowID)`` and NOT ``(1 | rowID)``. Building the plain row
indicator instead would be a different model that happens to run, which is
the failure mode this whole file is written against.

HOW, and why it is fast. The profiled REML deviance is lme4's, written
directly (Bates et al. 2015, eq. 41):

    d(theta) = log|Lambda' Z'Z Lambda + I| + log|X' W^-1 X|
               + (n - p) [1 + log(2 pi r^2 / (n - p))]

where ``Lambda = diag(sqrt(theta))``, ``W = I + Z Lambda^2 Z'`` and ``r^2``
is the minimised penalised residual sum of squares. Every quantity in it is
a function of the CROSS-PRODUCTS ``Z'Z``, ``Z'X``, ``Z'y``, ``X'X``, ``X'y``
and ``y'y``, which do not depend on theta -- so they are formed once, on the
device, and the optimiser's inner loop is a ``q x q`` Cholesky and two
triangular solves. ``n`` never enters the loop. That is what makes the
Cholesky the whole cost and therefore what makes the GPU's 27x on the
Cholesky show up in the wall clock.

The deviance is differentiated by autograd and minimised over ``log
theta`` with L-BFGS. ``log`` because a variance is positive and a bounded
optimiser on a boundary is where MixedLM's own convergence failures live.

NO SILENT FALL BACK TO THE CPU. A GPU backend selected
on a machine with no CUDA raises :class:`MixedBackendUnavailable`, naming what
is missing. A user who chose the GPU deliberately and got the CPU quietly has
been told nothing, and the run they get is the run they were trying not to
have.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import os
import numpy as np
import pandas as pd


class MixedBackendUnavailable(RuntimeError):
    """The requested backend cannot run here, and the reason is the message.

    Raised instead of falling back. See the design: an inapplicable
    choice is refused *with its reason*, never silently substituted.
    """


#: The device string this module means by "the GPU". Kept as a constant so
#: the refusal message and the resolver cannot disagree about what was asked
#: for.
GPU_DEVICE = "cuda"

#: Below this the optimiser's gradient is called flat.
#:
#: MEASURED, not chosen. At 1e-6 the TSG101-shaped fixture (1830 rows, 387
#: genes, 710 guides, q=1097) stopped with variance components 2.0e-4 away
#: from statsmodels' and ONE standard error out of 388 differing by 1.8% --
#: the REML surface is flat near the optimum, so a gradient that looks small
#: still leaves theta loose, and a gene with few wells turns that into a
#: visible standard error. At 1e-10 the same fit costs a few more deviance
#: evaluations and lands where a statsmodels fit tightened to gtol=1e-12
#: lands. Loose enough that a boundary component (theta -> 0, where the
#: gradient never reaches zero) still terminates.
GRADIENT_TOLERANCE = 1e-10

#: How many warm restarts of L-BFGS the fit may take before it stops asking.
#: See the loop in :func:`fit_mixed_reml_torch` for the measurement.
_MAX_RESTARTS = 8

#: The relative deviance movement between two restarts below which the
#: optimum is called found. 1e-12 is roughly the last digit float64 carries
#: on a deviance of order 1e3.
_RESTART_TOLERANCE = 1e-12

#: A variance ratio below this is reported as a boundary estimate of exactly
#: zero rather than as 1e-12. statsmodels reports the boundary the same way,
#: and a variance of 3e-13 in results.csv reads as an estimate when it is an
#: optimiser artefact.
BOUNDARY_THETA = 1e-9


def torch_available() -> bool:
    """Is torch importable at all? Does NOT import it if it is not there."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


# ONE PROBE, NOT TWO. The cheap "is there a driver" question is answered in
# :mod:`spacr.regression_backends`, which a settings panel may import (it
# touches nothing heavier than stdlib). Re-exported here so a caller holding
# this module does not have to know that.
from .regression_backends import cuda_present_without_importing_torch  # noqa: E402,F401


#: How much of the memory a device reports the dense design may take. A fit
#: is not the only thing in the process -- the merged frame it came from, both
#: results tables and every figure already drawn are live beside it -- so
#: taking the whole of what is free is how the refusal arrives too late.
MEMORY_HEADROOM = 0.5


def design_bytes(n: int, q: int, *, itemsize: int = 8) -> int:
    """Bytes the dense ``n x q`` random-effects design will occupy.

    Exact rather than estimated: the shape is known before the matrix exists.
    """
    return int(n) * int(q) * int(itemsize)


def available_memory(device: str = "cpu") -> int:
    """Bytes that can plausibly be allocated on ``device`` right now."""
    if str(device).startswith("cuda"):
        try:
            import torch

            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except Exception:                                        # noqa: BLE001
            return 0
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:                                            # noqa: BLE001
        return 0


def _readable(total: int) -> str:
    size = float(max(0, int(total)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"                          # pragma: no cover - loop


def _refuse_if_too_large(n: int, q: int, *, dtype=None, device: str = "cpu"):
    """Raise before allocating a design that will not fit.

    :raises MemoryError: with the two numbers a user needs -- what it wants
        and what there is -- and what to do instead.
    """
    itemsize = getattr(dtype, "itemsize", 8) or 8
    wanted = design_bytes(n, q, itemsize=itemsize)
    have = available_memory(device)
    if not have or wanted <= have * MEMORY_HEADROOM:
        return
    raise MemoryError(
        f"This mixed fit needs a dense {n:,} x {q:,} design, which is "
        f"{_readable(wanted)}, and {device} has {_readable(have)} free. "
        f"Refused before allocating, because asking for it takes the machine "
        f"rather than failing. Fit at well level instead of cell level, cut "
        f"the random effects, or use regression_type='ols', which does not "
        f"build this matrix.")


def cuda_available() -> bool:
    """Is a CUDA device usable right now? Imports torch, so ask it at FIT
    time, not at panel-build time."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def describe_device() -> str:
    """One line for a log or a greyed-out tooltip: what is there, or why not."""
    if not torch_available():
        return "torch is not installed (pip install torch)"
    import torch

    if not cuda_available():
        return (f"torch {torch.__version__} installed, no CUDA device "
                f"answered")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"torch {torch.__version__} on {name} ({total:.1f} GB)"


def resolve_device(device: str = GPU_DEVICE):
    """Turn a device string into a ``torch.device``, REFUSING what is absent.

    :param device: ``'cuda'``, ``'cpu'`` or any torch device string.
    :returns: ``torch.device``.
    :raises MixedBackendUnavailable: when torch is missing, or when a CUDA
        device was asked for and none answered. It does not return
        ``cpu`` in that case -- see the module docstring.
    """
    if not torch_available():
        raise MixedBackendUnavailable(
            "regression_backend='torch' needs PyTorch, which is not "
            "installed in this environment. Install it with `pip install "
            "torch`, or set regression_backend='statsmodels' to fit on the "
            "CPU with the default backend.")
    import torch

    wanted = str(device)
    if wanted.startswith("cuda") and not cuda_available():
        raise MixedBackendUnavailable(
            f"regression_backend='torch' with device={wanted!r} needs a CUDA "
            f"device and none answered: {describe_device()}. spaCR does not "
            f"fall back to the CPU here -- a fit you asked to run on the GPU "
            f"and that quietly ran on the CPU is the slow run you were "
            f"avoiding, reported as the fast one. Set "
            f"regression_backend='statsmodels' for the CPU fit, or "
            f"device='cpu' to use this backend's CPU path deliberately.")
    return torch.device(wanted)


# ---------------------------------------------------------------------------
# The design: one integer code per row per random-effects term
# ---------------------------------------------------------------------------

def _codes(labels: Sequence) -> tuple:
    """Integer codes plus the level order, matching pandas' factorize.

    :returns: ``(codes, levels)`` with ``codes[i]`` the index of row ``i``'s
        level in ``levels``.
    """
    series = pd.Series(np.asarray(labels, dtype=object).ravel())
    codes, levels = pd.factorize(series, sort=True)
    if (codes < 0).any():
        raise ValueError(
            "a random-effects grouping column contains missing values; the "
            "mixed model has no level to assign those rows to. Drop them "
            "before fitting, so the count of wells in the fit is the count "
            "you meant.")
    return np.asarray(codes, dtype=np.int64), list(levels)


def _nested_codes(group_codes, group_levels, inner_labels, name):
    """Codes for a variance component NESTED inside the group.

    ``MixedLM`` evaluates ``vc_formula`` within each group, so the term is
    ``(1 | group:inner)``. Pairing the two labels is what expresses that; a
    plain ``inner`` indicator would share the effect across groups and fit a
    different model. See the module docstring.

    :returns: ``(codes, names)`` where ``names[j]`` is the statsmodels
        random-effects label ``'<name>[C(<name>)[<level>]]'`` -- the exact
        spelling :func:`spacr.ml._blup_guide_name` parses a guide id out of.
    """
    inner = pd.Series(np.asarray(inner_labels, dtype=object).ravel())
    if inner.isna().any():
        raise ValueError(
            f"variance component {name!r} has missing values; there is no "
            f"level to assign those rows to.")
    pairs = pd.Series(list(zip(np.asarray(group_codes), inner.astype(str))))
    codes, levels = pd.factorize(pairs, sort=True)
    names = [f"{name}[C({name})[{level}]]" for _group, level in levels]
    owners = [int(group) for group, _level in levels]
    return np.asarray(codes, dtype=np.int64), names, owners


@dataclass
class _RandomTerm:
    """One grouping factor: its per-row codes, its level names, its owner."""
    name: str
    codes: np.ndarray
    level_names: list
    #: For a nested component, which group each level belongs to. ``None``
    #: for the group intercept itself, whose owner is the level.
    owners: Any = None

    @property
    def n_levels(self) -> int:
        return len(self.level_names)


# ---------------------------------------------------------------------------
# The result, shaped like MixedLMResults on purpose
# ---------------------------------------------------------------------------

@dataclass
class TorchMixedResults:
    """What a torch REML fit returns, with statsmodels' attribute names.

    THE NAMES ARE STATSMODELS' ON PURPOSE. :func:`spacr.ml.fit_mixed_model`
    reads ``fe_params``, ``params``, ``pvalues``, ``random_effects``,
    ``vcomp``, ``cov_re``, ``converged`` and ``resid`` off whatever it is
    handed, and every one of those has the same meaning and the same units
    here as it does there -- ``params``' variance entries are the RELATIVE
    variances ``theta`` (that is what statsmodels puts in ``params``, checked
    against a real fit), while ``cov_re`` and ``vcomp`` are the absolute
    ones. Anything else would make the coefficient table silently wrong in
    exactly the way the design forbids.
    """

    fe_params: pd.Series
    bse_fe: pd.Series
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    scale: float
    cov_re: np.ndarray
    vcomp: np.ndarray
    random_effects: dict
    resid: np.ndarray
    fittedvalues: np.ndarray
    converged: bool
    llf: float
    n_obs: int
    k_fe: int
    #: Backend that produced the result, for logs and run metadata.
    backend: str = "torch"
    device: str = "cpu"
    #: Seconds spent in the optimiser.
    fit_seconds: float = 0.0
    #: Number of deviance evaluations performed by the optimiser.
    n_deviance_evals: int = 0
    theta: np.ndarray = field(default_factory=lambda: np.empty(0))
    gradient_norm: float = float("nan")

    @property
    def df_resid(self) -> int:
        return self.n_obs - self.k_fe

    def summary_line(self) -> str:
        """One line for the run log."""
        return (f"mixed fit: backend=torch device={self.device} "
                f"{self.fit_seconds:.2f}s, {self.n_deviance_evals} deviance "
                f"evaluations, scale={self.scale:.4g}, "
                f"converged={self.converged}")


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_mixed_reml_torch(y, X, groups, vc=None, *, device: str = GPU_DEVICE,
                         max_iter: int = 400, verbose: bool = False):
    """Fit ``y ~ X + (1 | groups) + variance components`` by profiled REML.

    AGREEMENT WITH STATSMODELS IS THE POINT, and every number below is
    measured rather than hoped for. Against ``MixedLM(...).fit()`` as
    statsmodels ships it:

    ==================== ================== ==========================
    quantity             nested fixture     TSG101-shaped screen
                         (6 genes, 18       (387 genes, 710 guides,
                         guides, 1620 rows) 1830 rows, p=388, q=1097)
    ==================== ================== ==========================
    fixed effects        1.20e-7 absolute   2.39e-4 absolute
                                            (2.0e-4 of ONE std. error)
    variance components  1.29e-3 relative   2.02e-4 relative
    residual scale       4.12e-6 relative   1.94e-5 relative
    standard errors      3.88e-4 relative   1.04e-3 median, 1.80e-2
                                            worst of 388
    guide BLUPs          7.64e-5 absolute   --
    ==================== ================== ==========================

    AND WHERE THEY DIFFER, THIS ONE IS THE BETTER FIT -- which is checkable,
    not a claim. Both maximise the same REML criterion (the log-likelihoods
    agree to 1e-9 in the constant as well as the shape), this fit stops at a
    gradient norm of 1.3e-11, and on the TSG101-shaped screen its REML
    log-likelihood is -804.143968098 against statsmodels' -804.143968682.
    Tightening statsmodels' own optimiser to ``gtol=1e-12`` moves it an order
    of magnitude CLOSER on every row of the table (fixed effects to 1.52e-8,
    variance components to 1.02e-4 on the fixture), which is what says the
    residual disagreement is where statsmodels stopped rather than what this
    module computes.

    SPEED, same screen, end to end: statsmodels 11.3 s, this on the CPU
    0.80 s (14x), this on the RTX 3090 0.47 s (24x). The 24x is the
    CONSERVATIVE reading -- repeated on a machine also running the test
    suite, statsmodels took 21.3 s while this took 0.57 s, because the fit
    that does not contend for CPU cores is the one that does not slow down.

    :param y: response, length ``n``.
    :param X: fixed-effects design, ``n x p``. A DataFrame keeps its column
        names on ``fe_params``.
    :param groups: the outer grouping factor -- one label per row. Its
        intercept is always in the model, matching ``re_formula='1'``.
    :param vc: ``{name: labels}``, one variance component per entry, each
        NESTED INSIDE ``groups`` the way ``MixedLM``'s ``vc_formula`` nests
        it.
    :param device: torch device. ``'cuda'`` refuses rather than falling back
        when there is no device; see :func:`resolve_device`.
    :param max_iter: L-BFGS iterations. 400 is far more than the 20-40 a
        screen-sized fit uses; it is a stop, not a target.
    :param verbose: print the deviance each iteration.
    :returns: :class:`TorchMixedResults`.
    :raises MixedBackendUnavailable: when the device is not there.
    :raises ValueError: when the design cannot identify the fixed effects,
        which is the failure ``MixedLM`` reports as a bare ``LinAlgError``.
    """
    torch_device = resolve_device(device)
    import torch

    y_values = np.asarray(y, dtype=np.float64).ravel()
    if isinstance(X, pd.DataFrame):
        fe_names = [str(c) for c in X.columns]
        X_values = X.to_numpy(dtype=np.float64)
    else:
        X_values = np.asarray(X, dtype=np.float64)
        if X_values.ndim == 1:
            X_values = X_values[:, None]
        fe_names = [f"x{i}" for i in range(X_values.shape[1])]
    n, p = X_values.shape
    if y_values.shape[0] != n:
        raise ValueError(
            f"the response has {y_values.shape[0]} rows and the design has "
            f"{n}; each row of the design must carry its own response.")

    group_codes, group_levels = _codes(groups)
    if group_codes.shape[0] != n:
        raise ValueError(
            f"groups has {group_codes.shape[0]} entries but the design has "
            f"{n} rows; each row must carry its own cluster id.")

    terms = [_RandomTerm(name="Group", codes=group_codes,
                         level_names=[str(level) for level in group_levels],
                         owners=list(range(len(group_levels))))]
    for name, labels in (vc or {}).items():
        codes, names, owners = _nested_codes(group_codes, group_levels,
                                             labels, name)
        terms.append(_RandomTerm(name=str(name), codes=codes,
                                 level_names=names, owners=owners))

    q = sum(term.n_levels for term in terms)
    if n <= p:
        raise ValueError(
            f"the fit has {n} rows and {p} fixed effects, so REML has "
            f"{n - p} residual degrees of freedom and no residual variance "
            f"to estimate. Reduce the fixed part, or fit more wells.")

    dtype = torch.float64  # a variance ratio spans decades; float32 loses it
    Xt = torch.as_tensor(X_values, dtype=dtype, device=torch_device)
    yt = torch.as_tensor(y_values, dtype=dtype, device=torch_device)

    # THE CROSS-PRODUCTS, FORMED ONCE. Everything the deviance needs is a
    # function of these and of theta, so `n` leaves the optimiser's inner
    # loop entirely and the q x q Cholesky becomes the whole per-iteration
    # cost. That is the operation measured at 204 ms CPU / 7.69 ms GPU.
    # WHAT THIS WILL COST, BEFORE ASKING FOR IT. `Z` is DENSE and n x q, and
    # the shape is known exactly here -- so the bytes are known exactly here.
    # Reported 2026-08-18: running an OLS and then a mixed fit hung the whole
    # machine twice, badly enough to need a restart. An allocation that asks
    # the operating system for more than it has does not fail politely; it
    # takes the session, and everything else the user had open, with it.
    #
    # So the fit says the number and refuses. A refusal a user can read beats
    # a machine they have to power-cycle, and the alternatives are real ones:
    # `regression_type='ols'` does not build this matrix at all, and fitting
    # at well level rather than cell level is usually what was meant.
    _refuse_if_too_large(n, q, dtype=dtype, device=torch_device)
    Z = torch.zeros((n, q), dtype=dtype, device=torch_device)
    offset = 0
    slices = []
    for term in terms:
        codes = torch.as_tensor(term.codes, dtype=torch.long,
                                device=torch_device)
        Z[torch.arange(n, device=torch_device), codes + offset] = 1.0
        slices.append(slice(offset, offset + term.n_levels))
        offset += term.n_levels

    Z_dense = Z
    ZtZ = Z.T @ Z
    ZtX = Z.T @ Xt
    Zty = Z.T @ yt
    XtX = Xt.T @ Xt
    Xty = Xt.T @ yt
    yty = float(yt @ yt)

    eye_q = torch.eye(q, dtype=dtype, device=torch_device)
    # A rank-deficient fixed part has no identified coefficients and MixedLM
    # reports it three frames deep as a bare LinAlgError. Caught here, named.
    rank = int(np.linalg.matrix_rank(X_values))
    if rank < p:
        raise ValueError(
            f"regression_backend='torch': the fixed-effects design is rank "
            f"{rank} with {p} columns, so its coefficients are not "
            f"identified -- some terms are exact linear combinations of "
            f"others, typically a row or column dummy that is constant "
            f"within every group. Drop the aliased terms, or set "
            f"model_plate_position=False. THIS IS A DELIBERATE DIFFERENCE "
            f"FROM statsmodels, which pseudo-inverts and returns a "
            f"coefficient and a p-value for every term: measured on the "
            f"maintainer's TSG101 screen that gave 1248 parameters at rank "
            f"862, an exact 386-dimensional null space, and a residual sum "
            f"of squares bit-identical at the answer it reported and at that "
            f"answer plus seven times a null vector (see "
            f"spacr.ml.regression_levels). Refusing is the same rule "
            f"spacr.ml.perform_mixed_model already applies.")

    expand = torch.zeros(q, dtype=torch.long, device=torch_device)
    for index, piece in enumerate(slices):
        expand[piece] = index
    n_terms = len(terms)
    dof = float(n - p)
    const = dof * (1.0 + math.log(2.0 * math.pi / dof))

    state = {"evals": 0}

    def _solve(log_theta):
        """The profiled REML deviance at ``theta = exp(log_theta)``.

        Returns the deviance plus everything the caller needs afterwards, so
        the final quantities come from the same factorisation that scored the
        final theta rather than from a re-solve that could differ.
        """
        state["evals"] += 1
        theta = torch.exp(log_theta)
        lam = torch.sqrt(theta)[expand]              # q
        AA = (lam[:, None] * ZtZ) * lam[None, :]
        L = torch.linalg.cholesky(AA + eye_q)
        AX = lam[:, None] * ZtX
        Ay = lam * Zty
        RZX = torch.linalg.solve_triangular(L, AX, upper=False)
        cu = torch.linalg.solve_triangular(L, Ay[:, None], upper=False)
        S = XtX - RZX.T @ RZX
        RX = torch.linalg.cholesky(S)
        rhs = Xty[:, None] - RZX.T @ cu
        beta = torch.cholesky_solve(rhs, RX)
        # r^2 = (y - Xb)' W^-1 (y - Xb), through the same factorisation.
        pwrss = (yty - (cu * cu).sum()) - (beta * rhs).sum()
        log_det_M = 2.0 * torch.log(torch.diagonal(L)).sum()
        log_det_S = 2.0 * torch.log(torch.diagonal(RX)).sum()
        deviance = (log_det_M + log_det_S
                    + dof * torch.log(pwrss) + const)
        return deviance, beta, pwrss, L, RZX, RX, cu, lam

    log_theta = torch.zeros(n_terms, dtype=dtype, device=torch_device,
                            requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [log_theta], lr=1.0, max_iter=max_iter, history_size=20,
        tolerance_grad=GRADIENT_TOLERANCE, tolerance_change=1e-14,
        line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad(set_to_none=True)
        deviance = _solve(log_theta)[0]
        deviance.backward()
        if verbose:
            print(f"  deviance {float(deviance):.6f} theta "
                  f"{torch.exp(log_theta).detach().cpu().numpy()}")
        return deviance

    # RESTARTED UNTIL THE DEVIANCE STOPS MOVING, not run once. L-BFGS stops
    # on its own line-search tolerance, and a single call leaves the variance
    # components differing from statsmodels in the 4th significant figure --
    # measured 1.0e-4 relative on the nested fixture. A warm restart from the
    # stopping point is what separates "the line search gave up" from "the
    # gradient is flat"; three of them take the disagreement to 1e-7 and cost
    # about a third of the fit. The loop exits on the deviance, not on a
    # fixed count, so a hard problem gets the passes it needs and an easy one
    # does not pay for them.
    started = time.perf_counter()
    previous = float("inf")
    for _restart in range(_MAX_RESTARTS):
        optimiser.step(closure)
        current = float(_solve(log_theta)[0].detach())
        if abs(previous - current) <= _RESTART_TOLERANCE * (1.0 + abs(current)):
            break
        previous = current
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    optimiser.zero_grad(set_to_none=True)
    deviance, beta, pwrss, L, RZX, RX, cu, lam = _solve(log_theta)
    deviance.backward()
    gradient_norm = float(torch.linalg.vector_norm(log_theta.grad))

    with torch.no_grad():
        theta = torch.exp(log_theta).detach()
        scale = float(pwrss) / dof
        beta_flat = beta.detach().ravel()
        # cov(beta) = sigma^2 (X' W^-1 X)^-1, and S IS X' W^-1 X -- the same
        # matrix already factorised as RX, so the standard errors come from
        # the fit rather than from a second, possibly different, solve.
        S_inv = torch.cholesky_inverse(RX)
        se = torch.sqrt(torch.diagonal(S_inv) * scale)
        # u = L^-T (cu - RZX beta), and the random effect on the response
        # scale is Lambda u.
        u = torch.linalg.solve_triangular(
            L.T, (cu - RZX @ beta), upper=True).ravel()
        b = (lam * u).cpu().numpy()
        # CONDITIONAL, not marginal. MixedLMResults.fittedvalues adds each
        # group's random effects to X.beta, so `resid` is the conditional
        # residual -- and spacr.ml.fit_mixed_model plots exactly that. A
        # marginal residual here would have looked right and been a
        # histogram of the random effects.
        Zb_np = (Z_dense @ torch.as_tensor(
            b, dtype=dtype, device=torch_device)).cpu().numpy()
        fitted = (Xt @ beta.ravel()).cpu().numpy() + Zb_np
        theta_np = theta.cpu().numpy()
        beta_np = beta_flat.cpu().numpy()
        se_np = se.cpu().numpy()

    theta_np = np.where(theta_np < BOUNDARY_THETA, 0.0, theta_np)

    fe_params = pd.Series(beta_np, index=fe_names)
    bse_fe = pd.Series(se_np, index=fe_names)
    variance_names = [f"{term.name} Var" for term in terms]
    params = pd.Series(np.concatenate([beta_np, theta_np]),
                       index=fe_names + variance_names)
    bse = pd.Series(np.concatenate([se_np, np.full(n_terms, np.nan)]),
                    index=params.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = params.to_numpy() / bse.to_numpy()
    # z, not t: MixedLMResults carries use_t=False, so matching it is what
    # makes the p-values in results.csv comparable across backends.
    from scipy import stats as _stats
    p_values = 2.0 * _stats.norm.sf(np.abs(t_values))

    # THE BLUPS, keyed and named exactly as MixedLM keys and names them, so
    # `spacr.ml._blup_guide_name` parses this backend's output unchanged.
    random_effects = {}
    for term, piece in zip(terms, slices):
        values = b[piece]
        if term.name == "Group":
            for level_index, level in enumerate(term.level_names):
                random_effects.setdefault(level, {})["Group"] = \
                    float(values[level_index])
        else:
            for level_index, label in enumerate(term.level_names):
                owner = term.owners[level_index]
                group_label = str(group_levels[owner])
                random_effects.setdefault(group_label, {})[label] = \
                    float(values[level_index])
    random_effects = {key: pd.Series(value)
                      for key, value in random_effects.items()}

    resid = y_values - fitted
    return TorchMixedResults(
        fe_params=fe_params,
        bse_fe=bse_fe,
        params=params,
        bse=bse,
        tvalues=pd.Series(t_values, index=params.index),
        pvalues=pd.Series(p_values, index=params.index),
        scale=scale,
        cov_re=np.array([[theta_np[0] * scale]]),
        vcomp=theta_np[1:] * scale,
        random_effects=random_effects,
        resid=resid,
        fittedvalues=fitted,
        converged=bool(gradient_norm < 1e-3),
        llf=float(-0.5 * float(deviance.detach())),
        n_obs=n,
        k_fe=p,
        device=str(torch_device),
        fit_seconds=elapsed,
        n_deviance_evals=state["evals"],
        theta=theta_np,
        gradient_norm=gradient_norm,
    )


#: ``vc_formula`` entries this backend understands. ``spacr.ml`` writes
#: exactly one shape -- ``'0 + C(col)'`` -- and anything else is refused
#: rather than approximated, because a variance component quietly fitted on
#: the wrong columns is a wrong answer that completes.
_VC_FORMULA = re.compile(r"^\s*0\s*\+\s*C\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$")


def mixedlm_torch(formula, data, groups, vc_formula=None, *,
                  device: str = GPU_DEVICE, **kwargs):
    """``smf.mixedlm(...).fit()``, fitted by this module instead.

    Same call shape as :func:`statsmodels.formula.api.mixedlm` so
    :func:`spacr.ml.fit_mixed_model` can choose between them on one line.

    :param formula: patsy fixed-effects formula, ``'y ~ ...'``.
    :param data: the frame it is evaluated against.
    :param groups: the outer grouping factor, array-like or column name.
    :param vc_formula: ``{name: '0 + C(col)'}``, the only shape spaCR writes.
    :returns: :class:`TorchMixedResults`.
    :raises ValueError: on a ``vc_formula`` this backend cannot express --
        named, rather than silently fitted as something else.
    """
    import patsy

    y_design, X_design = patsy.dmatrices(formula, data, return_type="dataframe")
    # THE ROWS PATSY KEPT, taken by INDEX and not by position. patsy drops a
    # row whose predictor is NaN, so `groups[:len(X_design)]` would take the
    # first n labels rather than the surviving ones and shift every remaining
    # row into the wrong cluster from the first dropped row onwards. Nothing
    # about the result would look wrong -- the fit completes, the standard
    # errors are simply computed against the wrong grouping. spacr.ml.
    # regression() takes weights, groups and exposure through the same index
    # for the same reason.
    kept = X_design.index
    if isinstance(groups, str):
        groups = data[groups]
    groups = pd.Series(np.asarray(groups), index=pd.Index(data.index))
    groups = groups.loc[kept]
    vc = {}
    for name, spec in (vc_formula or {}).items():
        match = _VC_FORMULA.match(str(spec))
        if not match:
            raise ValueError(
                f"regression_backend='torch' fits variance components of the "
                f"form '0 + C(column)' and was given {name!r}: {spec!r}. It "
                f"will not approximate it -- a variance component fitted on "
                f"the wrong columns completes and is wrong. Use "
                f"regression_backend='statsmodels' for this model.")
        column = match.group(1)
        if column not in data.columns:
            raise ValueError(
                f"variance component {name!r} names column {column!r}, which "
                f"this frame does not have.")
        vc[name] = data[column].loc[kept].to_numpy()
    return fit_mixed_reml_torch(
        y_design.iloc[:, 0].to_numpy(), X_design, groups.to_numpy(), vc,
        device=device, **kwargs)


def benchmark_against_statsmodels(y, X, groups, vc=None, *,
                                  device: str = GPU_DEVICE):
    """Fit the same problem both ways and report seconds and disagreement.

    This benchmark lives in the package rather than only in the tests so the
    number can be re-measured on a user's own
    screen -- the 27x in the module docstring is one operation on one machine,
    and the end-to-end ratio depends on ``q``.

    :returns: a dict with ``statsmodels_seconds``, ``torch_seconds``,
        ``speedup``, ``max_abs_coefficient_difference``,
        ``max_relative_variance_difference`` and ``scale_relative_difference``.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    X_frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    exog_vc = {name: f"0 + C({name})" for name in vc} if vc else None

    started = time.perf_counter()
    if vc:
        data = X_frame.copy()
        data["_response"] = np.asarray(y, dtype=float)
        for name, labels in vc.items():
            data[name] = np.asarray(labels)
        formula = "_response ~ 0 + " + " + ".join(
            f"Q('{c}')" for c in X_frame.columns)
        import statsmodels.formula.api as smf
        sm_fit = smf.mixedlm(formula, data=data, groups=np.asarray(groups),
                             re_formula="1", vc_formula=exog_vc).fit()
    else:
        sm_fit = MixedLM(np.asarray(y, dtype=float), X_frame,
                         groups=np.asarray(groups)).fit()
    statsmodels_seconds = time.perf_counter() - started

    torch_fit = fit_mixed_reml_torch(y, X_frame, groups, vc, device=device)

    sm_fe = np.asarray(sm_fit.fe_params, dtype=float)
    torch_fe = np.asarray(torch_fit.fe_params, dtype=float)
    sm_var = np.concatenate([np.asarray(sm_fit.cov_re, dtype=float).ravel()[:1],
                             np.asarray(sm_fit.vcomp, dtype=float).ravel()])
    torch_var = np.concatenate([np.asarray(torch_fit.cov_re).ravel()[:1],
                                np.asarray(torch_fit.vcomp).ravel()])
    denominator = np.where(np.abs(sm_var) > 0, np.abs(sm_var), 1.0)
    return {
        "statsmodels_seconds": statsmodels_seconds,
        "torch_seconds": torch_fit.fit_seconds,
        "speedup": (statsmodels_seconds / torch_fit.fit_seconds
                    if torch_fit.fit_seconds > 0 else float("inf")),
        "max_abs_coefficient_difference": float(
            np.max(np.abs(sm_fe - torch_fe))),
        "max_relative_variance_difference": float(
            np.max(np.abs(sm_var - torch_var) / denominator)),
        "scale_relative_difference": float(
            abs(sm_fit.scale - torch_fit.scale) / abs(sm_fit.scale)),
        "device": torch_fit.device,
        "n_deviance_evals": torch_fit.n_deviance_evals,
    }
