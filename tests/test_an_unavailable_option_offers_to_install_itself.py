"""Instruction 158, the half that needs no screen.

WHAT IS BEING PROTECTED. The install offer has THREE answers and the
expensive mistake is collapsing them to two: "pip install cuml" on the
interpreter spaCR is usually run on either fails, or succeeds at breaking the
install -- and the second is worse, because the user asked for a faster lasso
and got a numpy major upgrade under an image-analysis stack. So the tests
below check that cuML on Python 3.10 answers "somewhere else" and RUNS
NOTHING, that pymer4 answers "not by installing a Python package", and that a
plan touching numpy is refused until a second confirmation has named the move.
"""
from __future__ import annotations

import json

import pytest

from spacr import gpu_reduce, regression_backends as backends, updater


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------

#: A pip 25.3 `--report -` document, in the shape pip ACTUALLY emits it:
#: pip's own progress before it and a "Would install ..." line after it.
#: Trimmed from a real run against this environment on 2026-08-18.
_PIP_CHATTER = (
    "Collecting cuml-cu12\n"
    "  Using cached cuml_cu12-26.2.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    ".metadata (6.1 kB)\n"
    "Requirement already satisfied: rich in /env/lib/python3.10/site-packages"
    " (from cuml-cu12) (13.7.1)\n"
)
_PIP_TRAILER = "\nWould install cuml-cu12-26.2.0 numpy-2.2.6 treelite-4.7.0\n"


def _report(*packages) -> str:
    payload = {
        "version": "1",
        "pip_version": "25.3",
        "install": [{"metadata": {"name": name, "version": version}}
                    for name, version in packages],
        "environment": {"python_version": "3.10.19"},
    }
    return _PIP_CHATTER + json.dumps(payload, indent=2) + _PIP_TRAILER


class _Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_the_pip_report_is_found_between_pips_own_chatter():
    """`json.loads` on the whole stream CANNOT parse this, and did not.

    Measured 2026-08-18 against pip 25.3: `--report -` writes the document to
    stdout with progress lines before it and a "Would install ..." summary
    after it. A whole-string `json.loads` raises on the trailing data, and the
    plan for a resolve that SUCCEEDED was reported as "the packaging tool
    produced no readable plan" -- which routes to the refusal branch and would
    have made every install unreachable.
    """
    changes = updater._parse_pip_report(_report(("cuml-cu12", "26.2.0")))
    assert changes is not None
    assert [c.name for c in changes] == ["cuml-cu12"]


def test_a_dry_run_names_what_moves_and_what_is_added(monkeypatch):
    monkeypatch.setattr(updater, "installed_version",
                        lambda name: {"numpy": "1.26.4"}.get(str(name)))
    monkeypatch.setattr(updater, "pip_available", lambda: True)
    stdout = _report(("cuml-cu12", "26.2.0"), ("numpy", "2.2.6"))
    result = updater.dry_run_install(
        "cuml-cu12", runner=lambda *a, **k: _Completed(stdout=stdout))
    assert result.ok
    assert [c.name for c in result.additions] == ["cuml-cu12"]
    assert [c.describe() for c in result.moves] == ["numpy 1.26.4 -> 2.2.6"]
    assert "numpy 1.26.4 -> 2.2.6" in result.summary()


def test_a_numpy_move_is_refused_until_a_second_confirmation_names_it(
        monkeypatch):
    """The rule instruction 158 states outright, checked on the sentence."""
    monkeypatch.setattr(updater, "installed_version",
                        lambda name: {"numpy": "1.26.4"}.get(str(name)))
    monkeypatch.setattr(updater, "pip_available", lambda: True)
    stdout = _report(("cuml-cu12", "26.2.0"), ("numpy", "2.2.6"))
    result = updater.dry_run_install(
        "cuml-cu12", runner=lambda *a, **k: _Completed(stdout=stdout))
    decision = updater.install_decision(result)
    assert decision['needs_second_confirmation'] is True
    # NAMING what moves, not "are you sure".
    assert "numpy 1.26.4 -> 2.2.6" in decision['headline']


@pytest.mark.parametrize("package", ["numpy", "torch", "pandas",
                                     "scikit-learn"])
def test_every_protected_package_triggers_the_second_confirmation(
        monkeypatch, package):
    monkeypatch.setattr(updater, "installed_version",
                        lambda name: "1.0" if str(name) == package else None)
    monkeypatch.setattr(updater, "pip_available", lambda: True)
    stdout = _report(("something", "1.0"), (package, "2.0"))
    result = updater.dry_run_install(
        "something", runner=lambda *a, **k: _Completed(stdout=stdout))
    assert updater.install_decision(result)['needs_second_confirmation']


def test_an_additive_plan_needs_only_the_one_confirmation(monkeypatch):
    monkeypatch.setattr(updater, "installed_version", lambda name: None)
    monkeypatch.setattr(updater, "pip_available", lambda: True)
    result = updater.dry_run_install(
        "glum", runner=lambda *a, **k: _Completed(stdout=_report(
            ("glum", "3.4.0"), ("tabmat", "4.0.0"))))
    decision = updater.install_decision(result)
    assert decision['allowed'] and not decision['needs_second_confirmation']


def test_a_dry_run_that_did_not_answer_forbids_the_install(monkeypatch):
    """Unknown consequences are not "probably fine".

    The whole reason the dry run is mandatory is that the interesting case is
    invisible until the resolver has spoken.
    """
    monkeypatch.setattr(updater, "pip_available", lambda: True)
    result = updater.dry_run_install(
        "cuml-cu12",
        runner=lambda *a, **k: _Completed(stderr="ResolutionImpossible",
                                          returncode=1))
    assert result.ok is False
    assert updater.install_decision(result)['allowed'] is False


def test_uv_dry_run_output_is_understood_too(monkeypatch):
    """The desktop installers' venv has no pip; uv is the only tool there."""
    monkeypatch.setattr(updater, "pip_available", lambda: False)
    monkeypatch.setattr(updater, "find_uv", lambda: "/opt/spacr/bootstrap/uv")
    monkeypatch.setattr(updater, "installed_version", lambda name: None)
    text = (" + cuml-cu12==26.2.0\n"
            " - numpy==1.26.4\n"
            " + numpy==2.2.6\n")
    result = updater.dry_run_install(
        "cuml-cu12", runner=lambda *a, **k: _Completed(stdout=text))
    assert result.ok
    assert [c.describe() for c in result.moves] == ["numpy 1.26.4 -> 2.2.6"]
    assert result.protected_moves


def test_the_dry_run_and_the_install_use_the_same_tool(monkeypatch):
    """A report produced by one resolver and an install run by another is a
    report about a different question."""
    monkeypatch.setattr(updater, "pip_available", lambda: False)
    monkeypatch.setattr(updater, "find_uv", lambda: "/opt/spacr/bootstrap/uv")
    assert updater.dry_run_command("glum")[0] == "/opt/spacr/bootstrap/uv"
    assert (updater.install_requirement_command("glum")[0]
            == "/opt/spacr/bootstrap/uv")
    monkeypatch.setattr(updater, "pip_available", lambda: True)
    assert updater.dry_run_command("glum")[1:3] == ["-m", "pip"]
    assert updater.install_requirement_command("glum")[1:3] == ["-m", "pip"]


def test_nothing_may_be_run_for_an_offer_that_is_not_an_install():
    """`InstallOffer.command` is the gate, and it is closed on two of three."""
    for field in ("requirement", "recipe", "runs_anything"):
        assert f":param {field}:" in (updater.InstallOffer.__doc__ or "")
    assert updater.offer_elsewhere("t", "m", "r").command is None
    assert updater.offer_impossible("t", "m").command is None
    assert updater.offer_ready("t", "m").command is None
    assert updater.offer_install("t", "m", "glum").command is not None


# ---------------------------------------------------------------------------
# Which answer each backend gets
# ---------------------------------------------------------------------------

def test_cuml_on_this_interpreter_is_answered_elsewhere_and_runs_nothing(
        monkeypatch):
    monkeypatch.setattr(backends, "_cuml_python_supported", lambda: False)
    monkeypatch.setattr(backends, "package_installed",
                        lambda name: name != "cuml")
    offer = backends.backend_install_offer("cuml", "lasso")
    assert offer.action == "elsewhere"
    assert offer.command is None and offer.runs_anything is False
    assert "conda create" in offer.as_text()


def test_cuml_on_a_supported_interpreter_becomes_installable(monkeypatch):
    monkeypatch.setattr(backends, "_cuml_python_supported", lambda: True)
    monkeypatch.setattr(backends, "package_installed",
                        lambda name: name != "cuml")
    offer = backends.backend_install_offer("cuml", "lasso")
    assert offer.action == "install"
    assert offer.requirement == "cuml-cu12"


def test_pymer4_is_not_possible_by_installing_a_python_package(monkeypatch):
    monkeypatch.setattr(backends, "package_installed",
                        lambda name: name != "pymer4")
    offer = backends.backend_install_offer("pymer4", "mixed")
    assert offer.action == "impossible"
    assert offer.command is None
    assert "R" in offer.message


def test_the_requirement_handed_to_the_resolver_is_not_the_prose():
    """`REGRESSION_BACKENDS['pymer4']['pip']` reads
    "pip install pymer4  (plus R, rpy2, lme4)". Handed to pip verbatim that
    is four requirements and a syntax error, which is why
    `BACKEND_REQUIREMENTS` exists."""
    from spacr.regression_spec import REGRESSION_BACKENDS
    for name, requirement in backends.BACKEND_REQUIREMENTS.items():
        assert " " not in requirement, name
        assert "pip install" not in requirement, name
    assert "(" in str(REGRESSION_BACKENDS['pymer4']['pip'])


def test_every_greyed_backend_has_an_offer_that_is_not_ready():
    """The panel shows `enabled` and the offer side by side. A greyed entry
    whose offer said "ready" would be a control disagreeing with itself."""
    for regression_type in (None, 'mixed', 'ols', 'lasso'):
        for entry in backends.availability_entries(regression_type):
            if not entry['enabled']:
                assert entry['offer'].action != 'ready', (
                    regression_type, entry['key'])


def test_an_installed_backend_with_no_device_is_not_offered_an_install(
        monkeypatch):
    monkeypatch.setattr(backends, "package_installed", lambda name: True)
    monkeypatch.setattr(backends, "cuda_present_without_importing_torch",
                        lambda: False)
    offer = backends.backend_install_offer("torch", "mixed")
    assert offer.action == "impossible"
    assert "nvidia-smi" in offer.message


def test_a_family_mismatch_is_not_an_install_problem():
    offer = backends.backend_install_offer("glum", "mixed")
    assert offer.action == "impossible"
    assert "change the regression type" in offer.message.lower()


def test_auto_regression_type_says_so_rather_than_offering_a_package():
    offer = backends.backend_install_offer("glum", None)
    assert offer.action == "impossible"
    assert "auto" in offer.message


# ---------------------------------------------------------------------------
# The recipes instruction 158 D asks the API documentation to carry
# ---------------------------------------------------------------------------

def test_the_pymer4_recipe_is_the_real_one_in_order():
    recipe = backends.install_recipe("pymer4")
    for step in ("r-base", "lme4", "lmerTest", "rpy2", "pip install pymer4"):
        assert step in recipe, step
    # THE ORDER IS THE RECIPE. Installing pymer4 first is exactly the
    # failure the recipe exists to prevent: the wheel declares no
    # dependencies, so pip reports success and the import then dies on rpy2.
    steps = ("conda install -c conda-forge r-base", "install.packages",
             "pip install rpy2", "pip install pymer4")
    # `rindex`: the prose above the steps quotes `pip install pymer4` as the
    # thing that does NOT work on its own, so the first occurrence is the
    # warning and the last is the step.
    positions = [recipe.rindex(step) for step in steps]
    assert positions == sorted(positions), list(zip(steps, positions))
    assert "HEAVIER ASK" in recipe


def test_the_cuml_recipe_names_the_environment_and_what_it_costs():
    recipe = backends.install_recipe("cuml")
    assert "python=3.11" in recipe
    assert "pandas 2.3.3 -> 3.0.3" in recipe
    assert "cuda-toolkit 13.0.3 -> 12.9.2" in recipe
    assert "SELECT DIFFERENT VARIABLES" in recipe


@pytest.mark.parametrize("name", ["numpyro", "gpytorch"])
def test_the_sampler_recipes_say_different_answer_before_they_say_install(
        name):
    """"A user installing them to go faster has misunderstood what they are
    for", so the difference is the FIRST thing the recipe says."""
    recipe = backends.install_recipe(name)
    assert recipe.index("DIFFERENT QUESTION") < recipe.index("pip install")


def test_every_recipe_reaches_the_rendered_block():
    text = backends.describe_install_recipes()
    for name in backends.INSTALL_RECIPES:
        assert backends.REGRESSION_BACKENDS[name]['label'] in text
    html = backends.describe_install_recipes(html=True)
    assert "<b>" in html and "&lt;" not in html.split("<b>")[0]


def test_a_backend_with_no_recipe_answers_empty():
    assert backends.install_recipe("statsmodels") == ""
    assert backends.install_recipe("not-a-backend") == ""


# ---------------------------------------------------------------------------
# The Image UMAP asks the same question in the same words
# ---------------------------------------------------------------------------

def test_the_image_umap_answers_in_the_shared_shape(monkeypatch):
    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: False)
    monkeypatch.setattr(gpu_reduce, "python_supported", lambda: False)
    entry = gpu_reduce.availability_entry()
    assert set(entry) == {'key', 'title', 'reason', 'url', 'enabled', 'offer'}
    assert entry['enabled'] is False
    assert entry['offer'].action == "elsewhere"
    assert isinstance(entry['offer'], updater.InstallOffer)


def test_the_image_umap_and_the_backend_picker_share_one_vocabulary(
        monkeypatch):
    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: False)
    monkeypatch.setattr(gpu_reduce, "python_supported", lambda: False)
    monkeypatch.setattr(backends, "_cuml_python_supported", lambda: False)
    monkeypatch.setattr(backends, "package_installed",
                        lambda name: name != "cuml")
    umap = gpu_reduce.availability_entry()['offer']
    backend = backends.backend_install_offer("cuml", "lasso")
    assert umap.action == backend.action
    # And the SAME recipe string, so the two cannot drift.
    assert umap.recipe == backend.recipe == backends.INSTALL_RECIPES['cuml']


@pytest.mark.parametrize("action,plan", [
    ("ready", "ready"), ("impossible", "no_device"),
    ("elsewhere", "wrong_python"), ("install", "install"),
])
def test_every_install_plan_maps_to_an_offer(monkeypatch, action, plan):
    monkeypatch.setattr(gpu_reduce, "install_plan",
                        lambda: {"action": plan, "message": "m"})
    offer = gpu_reduce.install_offer()
    assert offer.action == action
    assert (offer.command is not None) == (action == "install")


def test_the_module_docstring_carries_every_command_the_gui_shows():
    """The recipes reach ``docs/source/api``, and the two copies agree.

    Instruction 158 D asks the API documentation to carry the recipes.
    ``docs/source/api`` is built by sphinx-autoapi, which publishes a module
    DOCSTRING verbatim and renders a module-level dict as a truncated repr --
    so a recipe that lived only in :data:`INSTALL_RECIPES` would be in the GUI
    and not in the docs. Both copies exist; this is what stops them drifting.
    """
    doc = backends.__doc__ or ""
    commands = set()
    for recipe in backends.INSTALL_RECIPES.values():
        for line in recipe.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            command = stripped.split("#")[0].strip()
            if command.startswith(("pip install", "conda ", "R -e")):
                commands.add(command)
    assert commands, "no commands were extracted; the test is not checking"
    missing = sorted(c for c in commands if c not in doc)
    assert not missing, missing
