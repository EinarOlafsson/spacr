"""The two exact-NUTS backends, driven against stand-in samplers.

numpyro and pymc are optional dependencies and neither is installed in
spaCR's test environment, so :func:`spacr.power_model._fit_numpyro_nuts`
and :func:`spacr.power_model._fit_pymc_nuts` have never run here at all.
The module's own docstring says as much and asks that the first real fit
be checked against torch.

What can be pinned without the dependency is the half of those functions
that is spaCR's and not the sampler's: the priors it builds, the offset
and centring it puts into the linear predictor, the knobs it forwards,
and the result it assembles afterwards. These tests install minimal
stand-in ``jax``/``numpyro``/``pymc`` modules that record what they were
asked for and hand back fixed draws, then check that

* the horseshoe reaches the sampler with the ``df_*``/``scale_slab``/
  ``tau0`` the caller asked for, and the Poisson rate really is
  ``Ntotal * exp(intercept + Xcentred @ beta)``;
* ``max_tree_depth``, ``target_accept``, warmup/draws/chains and the seed
  are forwarded rather than defaulted;
* an unidentified gene comes back ``NaN`` and a divergent transition
  comes back ``converged=False``, on both backends.

Also here: the zero-fill path for a screen table that reports imaged
cells per gene rather than per well, and the header check that makes the
two column guards in ``scan_parameters``' resume block unreachable.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import power_model as pm


# --------------------------------------------------------------------------
# a design with one gene that cannot be identified
# --------------------------------------------------------------------------

#: Four wells, three genes. Gene ``C`` takes a constant fifth of every
#: well's reads, so its log10expression column has no contrast and
#: ``prepare_model_data`` flags it unidentified.
TIDY = pd.DataFrame({
    "well": ["w1"] * 3 + ["w2"] * 3 + ["w3"] * 3 + ["w4"] * 3,
    "gene": ["A", "B", "C"] * 4,
    "positive": [1, 2, 0, 3, 1, 0, 2, 2, 1, 0, 4, 1],
    "n_reads_per_gene_per_well": [100, 300, 100,
                                  300, 100, 100,
                                  200, 200, 100,
                                  50, 350, 100],
    "imaging_n_cells_per_gene_per_well": [20, 20, 10] * 4,
})


@pytest.fixture
def model_data():
    return pm.prepare_model_data(TIDY)


def _fake_module(name: str, **attributes) -> types.ModuleType:
    """A module object real enough for ``import`` and for ``find_spec``.

    ``_module_installed`` probes with :func:`importlib.util.find_spec`,
    which raises ``ValueError`` for a ``sys.modules`` entry whose
    ``__spec__`` is ``None``; the spec below is what makes the fake read
    as installed rather than as broken.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _Prior:
    """One recorded prior: which distribution, with which parameters."""

    def __init__(self, kind, args, kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.shape = ()
        self.masked = None

    def expand(self, shape):
        self.shape = tuple(shape)
        return self

    def mask(self, flag):
        self.masked = flag
        return self


def _distribution(kind):
    def make(*args, **kwargs):
        return _Prior(kind, args, kwargs)
    return make


#: Fixed "draws" the stand-in samplers return for each latent. ``lam`` and
#: ``tau`` are negative on purpose: the model takes their absolute value,
#: and a fake that only ever returned positives could not tell whether it
#: still did.
_LATENTS = {"z": 0.5, "lam": -2.0, "tau": -0.25, "c2": 4.0, "intercept": 0.7}


def _expected_beta(n_genes: int) -> np.ndarray:
    """``beta = z * tau * lam_tilde`` for the fixed latents above."""
    z = np.full(n_genes, _LATENTS["z"])
    lam = abs(_LATENTS["lam"])
    tau = abs(_LATENTS["tau"])
    c2 = _LATENTS["c2"]
    lam_tilde = np.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
    return z * tau * lam_tilde


# --------------------------------------------------------------------------
# numpyro stand-in
# --------------------------------------------------------------------------

class _NumpyroRecorder:
    """What the model asked the sampler for, and what the sampler did."""

    def __init__(self, beta_draws, intercept_draws, n_divergent):
        self.beta_draws = beta_draws
        self.intercept_draws = intercept_draws
        self.n_divergent = n_divergent
        self.priors = {}
        self.deterministic = {}
        self.observed = None
        self.rate = None
        self.kernel_kwargs = None
        self.mcmc_kwargs = None
        self.seed = None
        self.group_by_chain = None

    def sample(self, name, prior, obs=None):
        self.priors[name] = prior
        if obs is not None:
            self.observed = np.asarray(obs)
            self.rate = np.asarray(prior.args[0], dtype=np.float64)
            return obs
        return np.full(prior.shape, _LATENTS[name], dtype=np.float64)

    def determine(self, name, value):
        self.deterministic[name] = np.asarray(value, dtype=np.float64)
        return value


def _install_numpyro(monkeypatch, *, beta_draws, intercept_draws,
                     n_divergent=0) -> _NumpyroRecorder:
    """Put stand-in ``jax`` and ``numpyro`` modules in ``sys.modules``."""
    recorder = _NumpyroRecorder(beta_draws, intercept_draws, n_divergent)

    class _NUTS:
        def __init__(self, model, **kwargs):
            self.model = model
            recorder.kernel_kwargs = kwargs

    class _MCMC:
        def __init__(self, kernel, **kwargs):
            self.kernel = kernel
            recorder.mcmc_kwargs = kwargs

        def run(self, key):
            recorder.seed = key
            self.kernel.model()          # the model body is the thing under test

        def get_samples(self):
            return {"beta": recorder.beta_draws,
                    "intercept": recorder.intercept_draws}

        def get_extra_fields(self, group_by_chain=True):
            recorder.group_by_chain = group_by_chain
            return {"diverging": np.ones(recorder.n_divergent)}

    jnp = _fake_module(
        "jax.numpy", asarray=np.asarray, log=np.log, abs=np.abs,
        sqrt=np.sqrt, exp=np.exp, float64=np.float64,
    )
    jax = _fake_module(
        "jax", numpy=jnp,
        random=types.SimpleNamespace(PRNGKey=lambda seed: ("key", seed)),
    )
    distributions = _fake_module(
        "numpyro.distributions",
        Normal=_distribution("Normal"), StudentT=_distribution("StudentT"),
        InverseGamma=_distribution("InverseGamma"),
        Poisson=_distribution("Poisson"),
    )
    infer = _fake_module("numpyro.infer", MCMC=_MCMC, NUTS=_NUTS)
    numpyro = _fake_module(
        "numpyro", sample=recorder.sample, deterministic=recorder.determine,
        distributions=distributions, infer=infer,
    )
    for name, module in (("jax", jax), ("jax.numpy", jnp),
                         ("numpyro", numpyro),
                         ("numpyro.distributions", distributions),
                         ("numpyro.infer", infer)):
        monkeypatch.setitem(sys.modules, name, module)
    return recorder


def test_the_numpyro_backend_hands_back_a_fit_that_names_itself(
        monkeypatch, model_data):
    """``backend="numpyro"`` runs the NUTS path and says so on the result."""
    draws = np.arange(15, dtype=np.float64).reshape(5, 3)
    recorder = _install_numpyro(
        monkeypatch, beta_draws=draws.copy(),
        intercept_draws=np.linspace(-1.0, 1.0, 5),
    )

    fit = pm.fit_model(model_data, backend="numpyro", seed=7,
                       n_warmup=11, n_samples=5, n_chains=2,
                       max_tree_depth=9)

    assert (fit.backend, fit.requested_backend, fit.method) == (
        "numpyro", "numpyro", "nuts")
    assert fit.draws.shape == (5, 3)
    assert fit.converged is True
    assert fit.seed == 7
    # gene C has a constant covariate column: NaN, not a shrunk-to-zero
    # "not a hit". A and B come back exactly as sampled.
    assert list(model_data.unidentified_genes) == ["C"]
    assert np.isnan(fit.draws[:, 2]).all()
    assert np.array_equal(fit.draws[:, :2], draws[:, :2])
    assert fit.intercept_draws.tolist() == pytest.approx(
        np.linspace(-1.0, 1.0, 5).tolist())
    assert fit.diagnostics["n_divergent"] == 0
    assert fit.diagnostics["n_unidentified"] == 1
    assert fit.diagnostics["beta_scale"] == "per unit log10expression"
    assert (fit.diagnostics["n_warmup"], fit.diagnostics["n_samples"],
            fit.diagnostics["n_chains"]) == (11, 5, 2)
    assert fit.diagnostics["tau0"] == pytest.approx(
        pm._horseshoe_global_scale(3, 4, None, None))
    assert fit.diagnostics["seconds"] >= 0.0
    # the sampler knobs are forwarded, not silently defaulted
    assert recorder.kernel_kwargs == {"max_tree_depth": 9}
    assert recorder.mcmc_kwargs == {"num_warmup": 11, "num_samples": 5,
                                    "num_chains": 2, "progress_bar": False}
    assert recorder.seed == ("key", 7)
    assert recorder.group_by_chain is False


def test_the_numpyro_model_is_the_horseshoe_poisson_with_the_offset(
        monkeypatch, model_data):
    """The priors and the linear predictor, as the sampler sees them."""
    recorder = _install_numpyro(
        monkeypatch, beta_draws=np.zeros((4, 3)),
        intercept_draws=np.zeros(4),
    )

    pm.fit_model(model_data, backend="numpyro", df_local=8.0, df_global=2.0,
                 df_slab=6.0, scale_slab=3.0, scale_global=0.02,
                 n_warmup=1, n_samples=4, n_chains=1)

    priors = recorder.priors
    assert priors["z"].kind == "Normal" and priors["z"].shape == (3,)
    assert priors["lam"].kind == "StudentT"
    assert priors["lam"].args == (8.0, 0.0, 1.0)      # df_local
    assert priors["lam"].shape == (3,)
    # lam is sampled under mask(False): it is a half-Student-t taken by
    # absolute value, so its density must not be counted twice.
    assert priors["lam"].masked is False
    assert priors["tau"].args == (2.0, 0.0, 0.02)     # df_global, tau0
    assert priors["c2"].args == (3.0, 3.0 * 3.0 ** 2)  # df_slab/2, slab scale

    baseline = model_data.Npositive.sum() / model_data.Ntotal.sum()
    assert priors["intercept"].args == pytest.approx(
        (3.0, float(np.log(baseline)), 2.5))

    beta = recorder.deterministic["beta"]
    assert beta == pytest.approx(_expected_beta(3))

    # the response is the well's positives, and the rate carries the
    # log(Ntotal) offset on the *centred* design.
    design, _, _, _ = pm._prepare_design(model_data, False)
    expected_rate = model_data.Ntotal * np.exp(
        _LATENTS["intercept"] + design @ beta)
    assert recorder.observed.tolist() == model_data.Npositive.tolist()
    assert recorder.rate == pytest.approx(expected_rate)
    # centring, not the raw covariate: an uncentred design would give a
    # visibly different rate here.
    raw_rate = model_data.Ntotal * np.exp(
        _LATENTS["intercept"] + model_data.log10expression @ beta)
    assert not np.allclose(recorder.rate, raw_rate)


def test_a_divergent_numpyro_transition_is_not_a_converged_fit(
        monkeypatch, model_data):
    """Divergences are counted and reported, and they cost convergence."""
    recorder = _install_numpyro(
        monkeypatch, beta_draws=np.zeros((6, 3)),
        intercept_draws=np.zeros(6), n_divergent=3,
    )

    diverged = pm.fit_model(model_data, backend="numpyro", standardize=True)
    assert diverged.converged is False
    assert diverged.diagnostics["n_divergent"] == 3
    assert diverged.diagnostics["beta_scale"] == (
        "per standard deviation of log10expression")

    recorder.n_divergent = 0
    clean = pm.fit_model(model_data, backend="numpyro", standardize=True)
    assert clean.converged is True
    assert clean.diagnostics["n_divergent"] == 0


# --------------------------------------------------------------------------
# pymc stand-in
# --------------------------------------------------------------------------

class _Stacked:
    """Enough of an ``xarray.DataArray`` for the two lines that use one."""

    def __init__(self, values):
        self._values = values
        self.stacked = None
        self.transposed = None

    def stack(self, **kwargs):
        self.stacked = kwargs
        return self

    def transpose(self, *args):
        self.transposed = args
        return self

    def to_numpy(self):
        return self._values


class _PyMCRecorder:
    def __init__(self, beta_draws, intercept_draws, n_divergent):
        self.beta_draws = beta_draws
        self.intercept_draws = intercept_draws
        self.n_divergent = n_divergent
        self.priors = {}
        self.deterministic = {}
        self.observed = None
        self.rate = None
        self.sample_kwargs = None
        self.entered = False


def _install_pymc(monkeypatch, *, beta_draws, intercept_draws,
                  n_divergent=0) -> _PyMCRecorder:
    """Put a stand-in ``pymc`` module in ``sys.modules``."""
    recorder = _PyMCRecorder(beta_draws, intercept_draws, n_divergent)

    def _latent(name, *args, **kwargs):
        recorder.priors[name] = (args, kwargs)
        shape = kwargs.get("shape")
        # pymc's Half* distributions are positive by construction, so the
        # stand-in returns magnitudes; the numpyro model takes the absolute
        # value itself and is handed the signed draw instead.
        value = abs(_LATENTS[name])
        return np.full(shape, value, dtype=np.float64) if shape else value

    class _Model:
        def __enter__(self):
            recorder.entered = True
            return self

        def __exit__(self, *exc):
            return False

    def _deterministic(name, value):
        recorder.deterministic[name] = np.asarray(value, dtype=np.float64)
        return value

    def _poisson(name, mu=None, observed=None):
        recorder.rate = np.asarray(mu, dtype=np.float64)
        recorder.observed = np.asarray(observed)

    def _sample(**kwargs):
        recorder.sample_kwargs = kwargs
        return types.SimpleNamespace(
            posterior={"beta": _Stacked(recorder.beta_draws),
                       "intercept": _Stacked(recorder.intercept_draws)},
            sample_stats={"diverging": _Stacked(
                np.ones(recorder.n_divergent))},
        )

    pymc = _fake_module(
        "pymc", Model=_Model, Normal=_latent, HalfStudentT=_latent,
        InverseGamma=_latent, StudentT=_latent, Deterministic=_deterministic,
        Poisson=_poisson, sample=_sample,
        math=types.SimpleNamespace(sqrt=np.sqrt, dot=np.dot, exp=np.exp),
    )
    monkeypatch.setitem(sys.modules, "pymc", pymc)
    return recorder


def test_the_pymc_backend_hands_back_a_fit_that_names_itself(
        monkeypatch, model_data):
    """``backend="pymc"`` runs the third branch of ``fit_model``."""
    draws = np.arange(12, dtype=np.float64).reshape(4, 3)
    recorder = _install_pymc(
        monkeypatch, beta_draws=draws.copy(),
        intercept_draws=np.array([[0.1, 0.2], [0.3, 0.4]]),
    )

    fit = pm.fit_model(model_data, backend="pymc", seed=5, n_warmup=13,
                       n_samples=4, n_chains=3, target_accept=0.95,
                       df_local=7.0, scale_global=0.03)

    assert (fit.backend, fit.requested_backend, fit.method) == (
        "pymc", "pymc", "nuts")
    assert recorder.entered is True
    assert fit.draws.shape == (4, 3)
    assert np.isnan(fit.draws[:, 2]).all()
    assert np.array_equal(fit.draws[:, :2], draws[:, :2])
    # the intercept comes out of a (chain, draw) array and is flattened
    assert fit.intercept_draws.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert fit.converged is True
    assert fit.diagnostics["n_divergent"] == 0
    assert fit.diagnostics["n_unidentified"] == 1
    assert fit.diagnostics["tau0"] == 0.03
    assert (fit.diagnostics["n_warmup"], fit.diagnostics["n_samples"],
            fit.diagnostics["n_chains"]) == (13, 4, 3)
    assert fit.diagnostics["beta_scale"] == "per unit log10expression"
    assert recorder.sample_kwargs == {
        "draws": 4, "tune": 13, "chains": 3, "random_seed": 5,
        "target_accept": 0.95, "progressbar": False}
    # the horseshoe knobs reach pymc's own distributions
    assert recorder.priors["lam"][1]["nu"] == 7.0
    assert recorder.priors["tau"][1]["sigma"] == 0.03
    assert recorder.priors["intercept"][1]["nu"] == 3.0
    assert recorder.deterministic["beta"] == pytest.approx(_expected_beta(3))
    expected_rate = model_data.Ntotal * np.exp(
        _LATENTS["intercept"] + pm._prepare_design(model_data, False)[0]
        @ recorder.deterministic["beta"])
    assert recorder.rate == pytest.approx(expected_rate)
    assert recorder.observed.tolist() == model_data.Npositive.tolist()


def test_a_divergent_pymc_transition_is_not_a_converged_fit(
        monkeypatch, model_data):
    """Same rule as numpyro, through pymc's ``sample_stats``."""
    recorder = _install_pymc(
        monkeypatch, beta_draws=np.zeros((2, 3)),
        intercept_draws=np.zeros(2), n_divergent=4,
    )

    diverged = pm.fit_model(model_data, backend="pymc", standardize=True)
    assert diverged.converged is False
    assert diverged.diagnostics["n_divergent"] == 4
    assert diverged.diagnostics["beta_scale"] == (
        "per standard deviation of log10expression")

    recorder.n_divergent = 0
    clean = pm.fit_model(model_data, backend="pymc", standardize=True)
    assert clean.converged is True
    assert clean.diagnostics["n_divergent"] == 0


def test_auto_prefers_numpyro_over_pymc_when_both_are_installed(
        monkeypatch, model_data):
    """The 'auto' order in ``resolve_backend`` is the order it fits in."""
    _install_pymc(monkeypatch, beta_draws=np.zeros((2, 3)),
                  intercept_draws=np.zeros(2))
    _install_numpyro(monkeypatch, beta_draws=np.zeros((2, 3)),
                     intercept_draws=np.zeros(2))

    fit = pm.fit_model(model_data, backend="auto", n_samples=2, n_chains=1)
    assert fit.backend == "numpyro"
    assert fit.requested_backend == "auto"
    assert fit.method == "nuts"


# --------------------------------------------------------------------------
# zero-filling a screen that counts cells per gene, not per well
# --------------------------------------------------------------------------

def test_a_gap_is_zero_filled_when_the_table_has_no_per_well_cell_count():
    """``fill_missing=True`` on a table with only per-gene imaging counts.

    The per-well column gets a special fill (from the well's own rows,
    because zero-filling a well total would make the well disagree with
    itself); a table that does not have that column takes the plain
    zero-fill for everything, and the missing pair then contributes no
    reads and no cells at all.
    """
    tidy = pd.DataFrame({
        "well": ["w1", "w1", "w2"],
        "gene": ["A", "B", "A"],
        "positive": [1.0, 2.0, 3.0],
        "n_reads_per_gene_per_well": [400.0, 600.0, 500.0],
        "imaging_n_cells_per_gene_per_well": [30.0, 70.0, 40.0],
    })
    assert "imaging_n_cells_per_well" not in tidy.columns

    with pytest.raises(pm.PowerFitError, match="grid is incomplete"):
        pm.prepare_model_data(tidy)

    data = pm.prepare_model_data(tidy, fill_missing=True)
    assert list(data.wells) == ["w1", "w2"]
    assert list(data.genes) == ["A", "B"]
    # w2/B was never measured: no positives, and no cells added to w2's total.
    assert data.Npositive.tolist() == [3, 3]
    assert data.Ntotal.tolist() == [100, 40]
    # ... and no reads either, so its read fraction is the pseudocount alone.
    assert data.log10expression[1, 1] == pytest.approx(
        np.log10(pm.EXPRESSION_PSEUDOCOUNT))
    assert data.log10expression[1, 0] == pytest.approx(
        np.log10(1.0 + pm.EXPRESSION_PSEUDOCOUNT))


# --------------------------------------------------------------------------
# the resume block's column guards
# --------------------------------------------------------------------------

def test_a_progress_file_missing_a_result_column_is_rejected_by_its_header(
        tmp_path):
    """Why ``scan_parameters``' two per-column guards cannot fail.

    Lines 2011 and 2013 re-check that ``run_key`` and the other string
    columns are present in a progress file that has already been read.
    They cannot be false: line 1988 has just compared the file's whole
    column list against ``names + _SCAN_RESULT_COLUMNS`` for equality and
    raised on any difference, and every name those guards look for is in
    ``_SCAN_RESULT_COLUMNS``. This test drives that equality check with a
    header that is missing exactly the column the first guard looks for.
    """
    assert {"run_key", "backend", "method", "status", "seed_channel",
            "reason", "error"} <= set(pm._SCAN_RESULT_COLUMNS)

    columns = ["n_wells_per_screen"] + [c for c in pm._SCAN_RESULT_COLUMNS
                                        if c != "run_key"]
    path = tmp_path / "scan.tsv"
    path.write_text("\t".join(columns) + "\n"
                    + "\t".join(["1"] * len(columns)) + "\n")

    def _never_called(**_kwargs):
        raise AssertionError("the header check must fire before any fit")

    with pytest.raises(pm.PowerFitError, match="has columns"):
        pm.scan_parameters(n_wells_per_screen=[4], backend="torch",
                           progress_file=str(path), simulate_fn=_never_called)

    # and the same file with the full header is accepted by that check --
    # it gets past it to the resume=False refusal further down.
    full = ["n_wells_per_screen"] + list(pm._SCAN_RESULT_COLUMNS)
    path.write_text("\t".join(full) + "\n"
                    + "\t".join(["1"] * len(full)) + "\n")
    with pytest.raises(pm.PowerFitError, match="resume=False"):
        pm.scan_parameters(n_wells_per_screen=[4], backend="torch",
                           progress_file=str(path), resume=False,
                           simulate_fn=_never_called)
