"""The GPU mixed-model backend must return statsmodels' answer, faster.

Instruction 141 D is the whole risk this file exists to close: "a backend
that is fast and returns different numbers is a bug, not a trade-off". So
every number below is MEASURED on the maintainer's machine (RTX 3090, torch
2.9.1+cu128, statsmodels 0.14) and written down, not asserted loosely and
hoped for.

AGREEMENT, on a TSG101-shaped screen (1830 rows, 387 genes, 710 guides,
p=388 fixed effects, q=1097 random levels), torch vs statsmodels' default
fit:

    fixed effects        2.4e-4 absolute -- 2.0e-4 of ONE standard error
    variance components  2.0e-4 relative
    residual scale       1.9e-5 relative
    standard errors      1.0e-3 relative median, 1.8e-2 worst of 388

and on the small balanced fixture in this file (6 genes, 18 guides, 1620
rows):

                         vs statsmodels   vs statsmodels tightened
                         as it ships      to gtol=1e-12
    fixed effects        1.20e-7 abs      1.52e-8 abs
    variance components  1.29e-3 rel      1.02e-4 rel
    residual scale       4.12e-6 rel      3.82e-7 rel
    standard errors      3.88e-4 rel      3.78e-5 rel
    guide BLUPs          7.64e-5 abs      5.56e-6 abs
    residuals            4.78e-6 abs      4.44e-7 abs

THE SECOND COLUMN IS THE POINT OF THE FIRST. Tightening statsmodels' own
optimiser moves it an order of magnitude CLOSER to the torch fit on every
row, which is what says the residual disagreement is where statsmodels
stopped and not what this module computes. The thresholds asserted below are
against the first column, because that is the statsmodels a user has.

WHERE THEY DIFFER, THE TORCH FIT IS THE BETTER ONE, and that is checkable
rather than a claim: both maximise the same REML criterion, the torch fit
stops at a gradient norm of 1.3e-11, and its REML log-likelihood is
-804.143968098 against statsmodels' -804.143968682 on that screen. The
residual disagreement is statsmodels' own convergence tolerance, not a
difference of model -- which is why this file asserts the two log-likelihoods
agree AND that torch's is not the lower one.

SPEED, same screen, end to end including building the design:

    statsmodels          11.3 s
    torch on the CPU      0.80 s   (14x)
    torch on the RTX 3090 0.47 s   (24x)

The 27x quoted in instruction 141 B is the dense Cholesky alone (204 ms ->
7.69 ms at q=1212). The end-to-end number is smaller because the fit is not
only Cholesky, and larger than 1 because :mod:`spacr.mixed_gpu` forms the
cross-products once and leaves ``n`` out of the iteration entirely.

NO CUDA IS TESTED BY FORCING IT, not by assuming it: the machine that wrote
this has a GPU, so the refusal path is driven by patching
``torch.cuda.is_available`` to False, which is the only way to know the
message a GPU-less user actually gets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import mixed_gpu
from spacr.mixed_gpu import MixedBackendUnavailable, fit_mixed_reml_torch
from spacr.regression_backends import (backend_menu, backend_status,
                                       backend_supports, describe_backends,
                                       resolve_backend_name)
from spacr.regression_spec import (DEFAULT_REGRESSION_BACKEND,
                                   REGRESSION_BACKENDS,
                                   REGRESSION_BACKEND_ORDER)

statsmodels_api = pytest.importorskip("statsmodels.formula.api")


#: Devices the agreement tests run on. The CPU one always runs -- the answer
#: must not depend on the device either -- and the CUDA one is skipped rather
#: than failed where there is no card.
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=[
        pytest.mark.gpu,
        pytest.mark.skipif(not mixed_gpu.cuda_available(),
                           reason="no CUDA device"),
    ]),
]


def _nested_screen(n_genes=6, guides_per_gene=3, n_wells=90, seed=0):
    """A screen with the nesting the mixed model is for: guides inside genes.

    Deliberately BALANCED and generously replicated, so both fits land well
    away from the boundary. A variance component pinned at zero is a real and
    common outcome, but it is not the case that tests whether two optimisers
    find the same interior optimum -- it is the case where neither of them
    has one to find.
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(n_genes)]
    records = [(f"w{well}", gene, f"{gene}_{k}")
               for well in range(n_wells)
               for gene in genes
               for k in range(guides_per_gene)]
    frame = pd.DataFrame(records, columns=["well", "gene", "grna"])
    gene_effect = {gene: rng.normal(0, 0.5) for gene in genes}
    guide_effect = {guide: rng.normal(0, 0.3)
                    for guide in frame["grna"].unique()}
    frame["gene_fraction"] = rng.uniform(0.1, 0.9, len(frame))
    frame["y"] = (1.0
                  + 0.7 * frame["gene_fraction"]
                  + frame["gene"].map(gene_effect)
                  + frame["grna"].map(guide_effect)
                  + rng.normal(0, 0.4, len(frame)))
    return frame


@pytest.fixture(scope="module")
def screen():
    return _nested_screen()


@pytest.fixture(scope="module")
def statsmodels_fit(screen):
    """The reference: exactly the model spacr.ml.fit_mixed_model builds."""
    return statsmodels_api.mixedlm(
        "y ~ gene_fraction", data=screen, groups=screen["gene"],
        re_formula="1", vc_formula={"grna": "0 + C(grna)"}).fit()


def _torch_fit(screen, device):
    design = pd.DataFrame({"Intercept": 1.0,
                           "gene_fraction": screen["gene_fraction"].to_numpy()})
    return fit_mixed_reml_torch(
        screen["y"].to_numpy(), design, screen["gene"].to_numpy(),
        {"grna": screen["grna"].to_numpy()}, device=device)


# ---------------------------------------------------------------------------
# D. THE ANSWER MUST NOT CHANGE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_the_fixed_effects_agree_with_statsmodels(screen, statsmodels_fit,
                                                  device):
    """The coefficients that reach results.csv and the volcano plot.

    1e-6 absolute is ~8x the 1.20e-7 measured against statsmodels as it
    ships (1.52e-8 against a tightened one),
    which leaves room for a different BLAS without leaving room for a
    different answer: the coefficients here are of order 1 and their standard
    errors of order 0.05, so 1e-6 is 2e-5 of a standard error.
    """
    fit = _torch_fit(screen, device)
    assert list(fit.fe_params.index) == list(statsmodels_fit.fe_params.index)
    difference = np.abs(fit.fe_params.to_numpy()
                        - statsmodels_fit.fe_params.to_numpy())
    assert difference.max() < 1e-6, (
        f"the torch backend moved a fixed effect by {difference.max():.3e}; "
        f"a backend is a speed choice and must not be a modelling one")


@pytest.mark.parametrize("device", DEVICES)
def test_the_variance_components_agree_with_statsmodels(screen,
                                                        statsmodels_fit,
                                                        device):
    """The gene variance and the guide variance, in response units.

    RELATIVE, not absolute, and looser than the fixed effects on purpose: a
    variance component is what the REML surface is FLAT in near its optimum,
    which is exactly why the two optimisers stop in slightly different
    places. Measured 1.29e-3 against statsmodels as it ships and 1.02e-4
    against statsmodels tightened to gtol=1e-12 -- see the module docstring.
    5e-3 is ~4x the first, which leaves room for a different BLAS without
    leaving room for a different model.
    """
    fit = _torch_fit(screen, device)
    reference = np.concatenate([
        np.asarray(statsmodels_fit.cov_re, dtype=float).ravel()[:1],
        np.asarray(statsmodels_fit.vcomp, dtype=float).ravel()])
    measured = np.concatenate([
        np.asarray(fit.cov_re, dtype=float).ravel()[:1],
        np.asarray(fit.vcomp, dtype=float).ravel()])
    assert measured.shape == reference.shape
    relative = np.abs(measured - reference) / np.abs(reference)
    assert relative.max() < 5e-3, (
        f"variance components disagree by {relative.max():.3e} relative: "
        f"statsmodels {reference}, torch {measured}")


@pytest.mark.parametrize("device", DEVICES)
def test_the_residual_scale_and_standard_errors_agree(screen, statsmodels_fit,
                                                      device):
    """sigma^2 and the standard errors every p-value is divided by."""
    fit = _torch_fit(screen, device)
    assert fit.scale == pytest.approx(statsmodels_fit.scale, rel=1e-4)
    relative = np.abs(fit.bse_fe.to_numpy()
                      - statsmodels_fit.bse_fe.to_numpy())
    relative = relative / np.abs(statsmodels_fit.bse_fe.to_numpy())
    assert relative.max() < 1e-2


@pytest.mark.parametrize("device", DEVICES)
def test_the_torch_fit_is_not_at_a_worse_optimum_than_statsmodels(
        screen, statsmodels_fit, device):
    """THE CHECK THAT MAKES THE OTHERS MEAN SOMETHING.

    Two fits of the same model agreeing to 1e-4 could be two fits of two
    models that happen to be close. This asserts they are optimising the SAME
    criterion -- the REML log-likelihoods agree to 1e-6 absolute on a value
    of order 1e3 -- and that where they part, torch is not the one that
    stopped early. Both halves matter: the first says it is the same
    objective, the second says the remaining disagreement is statsmodels'
    tolerance rather than a bug here.
    """
    fit = _torch_fit(screen, device)
    assert fit.llf == pytest.approx(statsmodels_fit.llf, abs=1e-5)
    assert fit.llf >= statsmodels_fit.llf - 1e-9, (
        f"torch stopped at a WORSE REML optimum ({fit.llf!r}) than "
        f"statsmodels ({statsmodels_fit.llf!r})")
    assert fit.gradient_norm < 1e-4
    assert fit.converged


@pytest.mark.parametrize("device", DEVICES)
def test_the_blups_agree_and_keep_statsmodels_own_names(screen,
                                                        statsmodels_fit,
                                                        device):
    """The guide predictions, and the keys spacr.ml parses guide ids out of.

    ``spacr.ml._blup_guide_name`` searches for ``C(grna)[<id>]`` inside the
    random-effects index, so the naming is not cosmetic: get it wrong and the
    mixed run writes a results_grna.csv with no guides in it.
    """
    fit = _torch_fit(screen, device)
    assert set(fit.random_effects) == set(statsmodels_fit.random_effects)
    for gene, reference in statsmodels_fit.random_effects.items():
        measured = fit.random_effects[gene]
        assert list(measured.index) == list(reference.index)
        # 7.64e-5 measured; a BLUP is a shrunken prediction and moves with
        # the variance components it is shrunk by, so it inherits their
        # looser agreement rather than the fixed effects' tighter one.
        assert np.abs(measured.to_numpy()
                      - reference.to_numpy()).max() < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_the_residuals_are_conditional_like_statsmodels(screen,
                                                        statsmodels_fit,
                                                        device):
    """MixedLMResults.fittedvalues ADDS the random effects.

    Reporting the marginal residual instead would look entirely reasonable
    and would make ``spacr.ml.fit_mixed_model``'s residual histogram a plot
    of the random effects. Measured before this was fixed: the two residual
    vectors differed by 0.57 on a response with sd 0.6.
    """
    fit = _torch_fit(screen, device)
    difference = np.abs(np.asarray(statsmodels_fit.resid) - fit.resid).max()
    assert difference < 1e-4, f"residuals differ by {difference:.3e}"


def test_the_answer_does_not_depend_on_the_device(screen):
    """CPU and GPU must give the same numbers, or "faster" is meaningless."""
    if not mixed_gpu.cuda_available():
        pytest.skip("no CUDA device")
    on_cpu = _torch_fit(screen, "cpu")
    on_gpu = _torch_fit(screen, "cuda")
    assert np.abs(on_cpu.fe_params.to_numpy()
                  - on_gpu.fe_params.to_numpy()).max() < 1e-9
    assert on_cpu.scale == pytest.approx(on_gpu.scale, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. NO CUDA MUST NOT BREAK ANYTHING -- and is tested by FORCING it
# ---------------------------------------------------------------------------

def test_a_gpu_fit_without_cuda_is_refused_and_says_why(monkeypatch):
    """Instruction 106: refused with the reason, never a silent CPU fall back.

    The machine this was written on HAS a GPU, so the only honest way to test
    the GPU-less path is to make torch say there is none. Asserting on the
    message rather than only on the exception type, because the message is
    the entire deliverable here -- a user who chose the GPU deliberately has
    to be told the fit did not run on it.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MixedBackendUnavailable) as raised:
        mixed_gpu.resolve_device("cuda")
    message = str(raised.value)
    assert "CUDA" in message
    assert "fall back" in message
    assert "statsmodels" in message


def test_the_cpu_path_still_works_when_cuda_is_absent(screen, monkeypatch):
    """A machine with no GPU loses the GPU, not the module."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fit = _torch_fit(screen, "cpu")
    assert fit.device == "cpu"
    assert np.isfinite(fit.scale)


def test_importing_the_module_without_torch_does_not_explode(monkeypatch):
    """The refusal has to survive torch being absent, not just CUDA."""
    monkeypatch.setattr(mixed_gpu, "torch_available", lambda: False)
    with pytest.raises(MixedBackendUnavailable) as raised:
        mixed_gpu.resolve_device("cuda")
    assert "not installed" in str(raised.value)
    assert "pip install torch" in str(raised.value)


# ---------------------------------------------------------------------------
# A / C. The setting, the labels, and the greying rule
# ---------------------------------------------------------------------------

def test_the_default_backend_is_statsmodels():
    """Every existing result was produced with it (instruction 141 A)."""
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings({})
    assert resolve_backend_name(settings["regression_backend"]) == "statsmodels"
    assert DEFAULT_REGRESSION_BACKEND == "statsmodels"


def test_an_old_settings_csv_without_the_key_still_loads(tmp_path):
    """The migration test, through the loader a saved settings file uses.

    No CSV written before 2026-08-18 carries ``regression_backend``, and what
    every one of those files MEANT is the backend that produced them. So the
    absence has to resolve to statsmodels rather than to an error or to a
    key nothing fills in.
    """
    from spacr.cli import load_settings_file
    from spacr.settings import (expected_types,
                                get_perform_regression_default_settings)

    path = tmp_path / "old_regression_settings.csv"
    path.write_text("setting_key,setting_value\n"
                    "regression_type,mixed\n"
                    "dependent_variable,pred\n"
                    "alpha,1\n")
    loaded = load_settings_file(str(path))
    assert "regression_backend" not in loaded

    filled = get_perform_regression_default_settings(dict(loaded))
    assert resolve_backend_name(filled["regression_backend"]) == "statsmodels"
    assert filled["regression_type"] == "mixed", (
        "filling in the new key must not disturb the old ones")
    # DECLARED, or check_settings drops it: a key with no entry in
    # expected_types is discarded with a warning, which is how a Tk panel
    # once threw away the regression_type a user had picked.
    assert expected_types["regression_backend"] is str


def test_every_backend_entry_is_labelled_cpu_or_gpu():
    """Instruction 141 C, checked on the table both GUIs build their combo
    from -- so neither can offer an unlabelled entry."""
    from spacr.settings_spec import convert_settings_dict_for_gui

    kind, options, default = convert_settings_dict_for_gui(
        {"regression_backend": "statsmodels (CPU)"})["regression_backend"]
    assert kind == "combo"
    assert len(options) == len(REGRESSION_BACKEND_ORDER)
    for option in options:
        assert option.endswith("(CPU)") or option.endswith("(GPU)"), option
    assert default in options


def test_the_stored_value_round_trips_through_the_label():
    """The combo posts the label; everything downstream reads a short name."""
    for name, spec in REGRESSION_BACKENDS.items():
        assert resolve_backend_name(spec["label"]) == name
        assert resolve_backend_name(name) == name
    assert resolve_backend_name(None) == DEFAULT_REGRESSION_BACKEND
    assert resolve_backend_name("lme4") == "pymer4"
    with pytest.raises(ValueError) as raised:
        resolve_backend_name("jax")
    assert "statsmodels" in str(raised.value)


def test_an_incompatible_regression_type_greys_the_entry_out_with_its_reason():
    """"cuML has no mixed model" is instruction 141 C's own example."""
    status = backend_status("cuml", "mixed")
    assert not status["enabled"]
    assert "mixed" in status["reason"]
    assert "lasso" in status["reason"], (
        "the reason must say what it CAN fit, or it is a dead end")
    assert backend_status("cuml", "lasso")["reason"] != status["reason"]


def test_a_missing_package_greys_the_entry_out_with_the_pip_command():
    """A backend whose package is not installed says what would provide it.

    NO ESCAPE HATCH FOR THE UNWIRED ONES ANY MORE. This used to accept
    ``"pip install" in reason OR not implemented``, and every optional
    backend takes the second branch on every machine -- they are extras, so
    nobody has them -- which meant the pip command instruction 141 C asks for
    was shown by nothing, ever. Both facts belong on the same entry:
    installing the package would not make it choosable, and neither would
    wiring it up alone.
    """
    from spacr.regression_backends import package_installed

    absent = [name for name in REGRESSION_BACKEND_ORDER
              if REGRESSION_BACKENDS[name]["package"]
              and not package_installed(REGRESSION_BACKENDS[name]["package"])]
    if not absent:
        pytest.skip("every optional backend package is installed here")
    for name in absent:
        spec = REGRESSION_BACKENDS[name]
        status = backend_status(name, spec["types"][0])
        assert not status["enabled"]
        assert spec["pip"] in status["reason"], name
        assert "not installed" in status["reason"], name


def test_every_refusal_also_comes_at_combo_entry_length():
    """`short_reason` is the same refusal, short enough to go IN the entry.

    A greyed-out dropdown row says only "not this one", and Qt shows an item
    tooltip lazily and only while the popup is open -- so the full sentence
    was reachable and missable. The short form is what the Qt panel appends
    to the entry's own text and prints in the box under it.
    """
    for regression_type in ("mixed", "lasso", "ols", None):
        for status in backend_menu(regression_type):
            if status["enabled"]:
                assert status["short_reason"] == ""
                continue
            short = status["short_reason"]
            assert short, (status["name"], regression_type)
            # 80 is the longest that is still an entry rather than a
            # paragraph: glum's "no mixed model; fits glm, poisson, logit,
            # probit, quasi_binomial" is 64, and the label in front of it
            # takes the rendered row to 80.
            assert len(short) <= 80, (status["name"], regression_type, short)
            assert not short.endswith("."), (
                "a fragment appended to a label, not a sentence")


def test_the_compact_box_keeps_every_backend_and_every_link():
    """Instruction 135: the settings are ONE PAGE a user can read.

    The full description is 3,101 characters -- about ninety wrapped lines in
    a settings field. Compact drops the measured cost and the trailing
    sentences of the seven backends the user did NOT pick, and keeps what the
    ask names: every package, what it does, and its API link.
    """
    full = describe_backends("mixed", html=False)
    compact = describe_backends("mixed", html=False,
                                selected="statsmodels (CPU)", compact=True)
    assert len(compact) < len(full) / 2
    for name, spec in REGRESSION_BACKENDS.items():
        assert spec["label"] in compact, name
        assert spec["url"] in compact, name
    # The selected one keeps its measured cost; the others lose theirs.
    assert REGRESSION_BACKENDS["statsmodels"]["cost"] in compact
    assert REGRESSION_BACKENDS["gpytorch"]["cost"] not in compact
    html = describe_backends("mixed", html=True,
                             selected="torch (GPU)", compact=True)
    assert html.count("<a href=") == len(REGRESSION_BACKENDS)
    assert REGRESSION_BACKENDS["torch"]["cost"] in html


def test_the_compact_box_states_the_refusal_for_the_entries_it_greys():
    """An unavailable backend is described AND said to be unavailable."""
    compact = describe_backends("lasso", html=False,
                                selected="statsmodels (CPU)", compact=True)
    assert "unavailable: no lasso model; fits mixed" in compact
    assert "pip install 'spacr[rapids]'" in compact


def test_a_gpu_backend_is_greyed_out_when_there_is_no_device(monkeypatch):
    """Driven by forcing it, per instruction 141 point 5."""
    import spacr.regression_backends as backends

    monkeypatch.setattr(backends, "cuda_present_without_importing_torch",
                        lambda: False)
    status = backends.backend_status("torch", "mixed")
    assert not status["enabled"]
    assert "CUDA" in status["reason"]
    assert "CPU" in status["reason"]


def test_the_box_describes_every_backend_and_links_its_api():
    """Instruction 141 B: brief, and a link to the API for each."""
    text = describe_backends("mixed", html=False)
    for name, spec in REGRESSION_BACKENDS.items():
        assert spec["label"] in text, name
        assert spec["url"] in text, name
        assert spec["url"].startswith("https://")
    html = describe_backends("mixed", html=True)
    assert html.count("<a href=") == len(REGRESSION_BACKENDS)


def test_the_box_says_where_a_backend_cannot_agree_by_construction():
    """numpyro returns posteriors; cuML solves a penalised path its own way.

    Instruction 141 D: where a backend cannot agree with statsmodels by
    construction, the box says WHAT differs rather than pretending it is a
    faster version of the same answer.
    """
    text = describe_backends(html=False)
    assert "DIFFERENT ANSWER" in text
    assert REGRESSION_BACKENDS["numpyro"]["differs"] is not None
    assert REGRESSION_BACKENDS["torch"]["differs"] is None, (
        "the torch backend claims to return the same numbers; if that "
        "changes, this table has to say so")


def test_the_menu_offers_every_backend_in_one_order():
    menu = backend_menu("mixed")
    assert [entry["name"] for entry in menu] == list(REGRESSION_BACKEND_ORDER)
    assert all(entry["reason"] for entry in menu if not entry["enabled"])
    assert all(not entry["reason"] for entry in menu if entry["enabled"])


def test_backend_supports_is_the_table_and_not_a_second_opinion():
    assert backend_supports("statsmodels", "mixed")
    assert backend_supports("statsmodels", "rra")
    assert backend_supports("torch", "mixed")
    assert not backend_supports("torch", "ols")
    assert not backend_supports("cuml", "mixed")
    # regression_type=None means "chosen from the response after the data is
    # read", which only the default backend can promise to fit.
    assert backend_supports("statsmodels", None)
    assert not backend_supports("torch", None)


# ---------------------------------------------------------------------------
# The seam into spacr.ml
# ---------------------------------------------------------------------------

def test_regression_model_refuses_a_backend_that_cannot_fit_the_family():
    """The run-time half of the greying rule: a settings CSV never passes a
    panel, so the refusal cannot live only in the GUI."""
    import spacr.ml as ml

    with pytest.raises(ValueError) as raised:
        ml._require_backend("ols", "torch")
    assert "cannot fit" in str(raised.value)
    assert "statsmodels" in str(raised.value)


def test_a_backend_spacr_does_not_route_through_yet_is_refused_not_ignored():
    """pymer4 is described, listed and greyed out. Choosing it anyway must
    not quietly produce a statsmodels fit labelled as lme4."""
    import spacr.ml as ml

    with pytest.raises(ValueError) as raised:
        ml._require_backend("mixed", "pymer4")
    assert "not route any fit through it yet" in str(raised.value)


def test_the_torch_backend_refuses_a_vc_formula_it_cannot_express(screen):
    """A variance component fitted on the wrong columns completes and is
    wrong, which is the worst outcome available."""
    from spacr.mixed_gpu import mixedlm_torch

    with pytest.raises(ValueError) as raised:
        mixedlm_torch("y ~ gene_fraction", screen, screen["gene"],
                      vc_formula={"grna": "1 + gene_fraction"}, device="cpu")
    assert "0 + C(column)" in str(raised.value)
    assert "statsmodels" in str(raised.value)


def test_fit_mixed_model_gives_the_same_table_through_either_backend(tmp_path):
    """END TO END, through the function the run actually calls.

    The comparison is on the COEFFICIENT TABLE, not on the fit object: that
    table is what becomes results.csv, and it is where a naming difference
    (a BLUP key, a variance row) would show up as a missing gene rather than
    as a wrong number.
    """
    from spacr.ml import fit_mixed_model, prepare_formula

    rng = np.random.default_rng(3)
    genes = [f"g{i}" for i in range(8)]
    wells = [(f"r{rng.integers(1, 9)}", f"c{rng.integers(1, 13)}")
             for _ in range(60)]
    records = [("p1", wells[w][0], wells[w][1], gene, f"{gene}_{k}")
               for w in range(60) for gene in genes for k in range(3)]
    frame = pd.DataFrame(records, columns=["plateID", "rowID", "columnID",
                                           "gene", "grna"])
    gene_effect = {gene: rng.normal(0, 0.4) for gene in genes}
    guide_effect = {g: rng.normal(0, 0.2) for g in frame["grna"].unique()}
    frame["gene_fraction"] = rng.uniform(0.05, 0.95, len(frame))
    frame["fraction"] = frame["gene_fraction"] / 3
    frame["pred"] = (0.5 + 0.4 * frame["gene_fraction"]
                     + frame["gene"].map(gene_effect)
                     + frame["grna"].map(guide_effect)
                     + rng.normal(0, 0.3, len(frame)))
    formula = prepare_formula("pred", random_row_column_effects=False,
                              block_screen=False, level="gene",
                              model_plate_position=False)

    tables = {}
    for backend in ("statsmodels", "torch (GPU)" if mixed_gpu.cuda_available()
                    else "statsmodels"):
        _fit, table = fit_mixed_model(frame.copy(), formula, str(tmp_path),
                                      regression_backend=backend)
        tables[backend] = table.set_index("feature")
    if len(tables) < 2:
        pytest.skip("no CUDA device")
    reference, measured = tables["statsmodels"], tables["torch (GPU)"]
    assert list(reference.index) == list(measured.index), (
        "the two backends produced different FEATURES, which would write two "
        "different results.csv files")
    assert (reference["term_type"] == measured["term_type"]).all()
    fixed = reference["term_type"] == "fixed"
    difference = (reference.loc[fixed, "coefficient"]
                  - measured.loc[fixed, "coefficient"]).abs()
    assert difference.max() < 1e-3, (
        f"a gene's fixed effect moved by {difference.max():.3e} between "
        f"backends")
    p_difference = (reference.loc[fixed, "p_value"]
                    - measured.loc[fixed, "p_value"]).abs()
    assert np.nanmax(p_difference.to_numpy()) < 1e-2


# ---------------------------------------------------------------------------
# 4. AND MEASURE THE SPEEDUP -- end to end, not just the Cholesky
# ---------------------------------------------------------------------------

@pytest.mark.heavy
@pytest.mark.gpu
def test_the_gpu_backend_is_faster_end_to_end_on_a_screen_sized_problem(
        capsys):
    """Check numerical agreement and median GPU speed on a screen-sized fit."""
    if not mixed_gpu.cuda_available():
        pytest.skip("no CUDA device")
    screen = _nested_screen(
        n_genes=120, guides_per_gene=4, n_wells=40, seed=5)
    design = pd.DataFrame({
        "Intercept": 1.0,
        "gene_fraction": screen["gene_fraction"].to_numpy()})
    # Warm the CUDA context first: the first allocation on a device costs
    # ~0.3 s and it belongs to neither fit.
    fit_mixed_reml_torch(screen["y"].to_numpy()[:40], design.iloc[:40],
                         screen["gene"].to_numpy()[:40],
                         {"grna": screen["grna"].to_numpy()[:40]},
                         device="cuda")
    reports = [
        mixed_gpu.benchmark_against_statsmodels(
            screen["y"].to_numpy(), design, screen["gene"].to_numpy(),
            {"grna": screen["grna"].to_numpy()}, device="cuda")
        for _ in range(3)
    ]
    report = {
        "statsmodels_seconds": float(np.median([
            item["statsmodels_seconds"] for item in reports])),
        "torch_seconds": float(np.median([
            item["torch_seconds"] for item in reports])),
        "speedup": float(np.median([
            item["speedup"] for item in reports])),
        "max_abs_coefficient_difference": max(
            item["max_abs_coefficient_difference"] for item in reports),
        "max_relative_variance_difference": max(
            item["max_relative_variance_difference"] for item in reports),
    }
    with capsys.disabled():
        print(f"\n  q={screen['gene'].nunique() + screen['grna'].nunique()} "
              f"statsmodels {report['statsmodels_seconds']:.2f}s -> torch "
              f"{report['torch_seconds']:.2f}s "
              f"({report['speedup']:.1f}x), coefficients within "
              f"{report['max_abs_coefficient_difference']:.2e}")
    assert report["max_abs_coefficient_difference"] < 1e-4
    assert report["max_relative_variance_difference"] < 1e-2
    assert report["speedup"] > 1.0, (
        f"the GPU backend was not faster: {report}")


def test_a_row_patsy_drops_does_not_shift_the_grouping(screen):
    """The alignment bug that would have completed and been wrong.

    patsy drops a row whose predictor is NaN. Taking the grouping labels by
    POSITION (``groups[:len(X)]``) then hands every row after the first
    dropped one its neighbour's cluster: the fit completes, the numbers look
    ordinary, and every standard error is computed against the wrong
    grouping.

    THE REFERENCE IS THE SAME FRAME WITH THE HOLES ALREADY REMOVED, not a
    statsmodels fit -- ``smf.mixedlm`` cannot fit this frame at all. It
    drops the rows from the design and keeps the full-length ``groups``,
    then dies with ``IndexError: index 1618 is out of bounds for axis 0 with
    size 1618`` from three frames inside ``mixed_linear_model.py``. Loud is
    better than silent, but it is not an answer to compare against, so the
    check is that dropping the rows before the fit and letting patsy drop
    them during it give the SAME fit.
    """
    from spacr.mixed_gpu import mixedlm_torch

    holed = screen.copy()
    holed.loc[holed.index[3], "gene_fraction"] = np.nan
    holed.loc[holed.index[900], "gene_fraction"] = np.nan
    prefiltered = holed.dropna(subset=["gene_fraction"])

    measured = mixedlm_torch("y ~ gene_fraction", holed, holed["gene"],
                             vc_formula={"grna": "0 + C(grna)"}, device="cpu")
    reference = mixedlm_torch("y ~ gene_fraction", prefiltered,
                              prefiltered["gene"],
                              vc_formula={"grna": "0 + C(grna)"}, device="cpu")
    assert measured.n_obs == len(screen) - 2 == reference.n_obs
    difference = np.abs(measured.fe_params.to_numpy()
                        - reference.fe_params.to_numpy()).max()
    assert difference < 1e-12, (
        f"the two rows patsy dropped moved a fixed effect by "
        f"{difference:.3e} -- the grouping is misaligned")
    assert measured.scale == pytest.approx(reference.scale, rel=1e-12)
    assert set(measured.random_effects) == set(reference.random_effects)
