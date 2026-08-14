"""cuML as an extra, with the CPU path kept. Instruction 86.

    "Implement cuML as an extra for image UMAP and anything else it would
     help, but only as an extra so the user must explicitly ask for cuML and
     this would mandate python be 3.11 or 3.12"

Every test here runs WITHOUT cuML installed, which is the state of every CI
cell and of the development machine -- so what is actually pinned is the
promise that matters most: absent, misbehaving or switched off, the extra
costs nothing.
"""
from __future__ import annotations

import pytest

from spacr.gpu_reduce import (
    ACCELERATED,
    ENV_FLAG,
    backend_for,
    describe,
    make_reducer,
    rapids_available,
)


# ---------------------------------------------------------------------------
# It is opt-in, twice over
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ACCELERATED)
def test_the_cpu_path_is_the_default_for_every_method(method):
    """An existing caller keeps the CPU path, and its reproducibility."""
    assert backend_for(method) == "cpu"


def test_asking_for_the_gpu_without_one_still_gives_the_cpu():
    assert backend_for("umap", prefer_gpu=True) in ("cpu", "cuml")
    if not rapids_available():
        assert backend_for("umap", prefer_gpu=True) == "cpu"


def test_a_method_cuml_does_not_implement_stays_on_the_cpu():
    assert backend_for("agglomerative", prefer_gpu=True) == "cpu"


def test_the_environment_flag_can_force_the_cpu(monkeypatch):
    """The escape hatch for "the GPU answer looks wrong" that does not
    require uninstalling anything."""
    monkeypatch.setenv(ENV_FLAG, "0")
    assert rapids_available() is False
    assert backend_for("umap", prefer_gpu=True) == "cpu"


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_every_spelling_of_off_is_honoured(monkeypatch, value):
    monkeypatch.setenv(ENV_FLAG, value)
    assert rapids_available() is False


# ---------------------------------------------------------------------------
# The CPU estimators are real, and are the ones spaCR already used
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,expected", [
    ("pca", "PCA"), ("tsne", "TSNE"), ("dbscan", "DBSCAN"),
    ("kmeans", "KMeans"),
])
def test_each_method_builds_its_sklearn_estimator(method, expected):
    estimator, backend = make_reducer(method)
    assert type(estimator).__name__ == expected
    assert backend == "cpu"


def test_umap_builds_the_umap_learn_reducer():
    estimator, backend = make_reducer("umap", n_neighbors=5)
    assert type(estimator).__name__ == "UMAP"
    assert backend == "cpu"


def test_the_shared_parameter_names_carry_through():
    """What makes ONE call site serve both backends."""
    estimator, _ = make_reducer("umap", n_neighbors=7, min_dist=0.3)
    assert estimator.n_neighbors == 7
    assert estimator.min_dist == pytest.approx(0.3)


def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(ValueError, match="umap"):
        make_reducer("not-a-method")


def test_the_backend_is_returned_not_only_logged():
    """So a caller can record WHICH backend produced a figure."""
    _estimator, backend = make_reducer("pca", n_components=2)
    assert backend in ("cpu", "cuml")


# ---------------------------------------------------------------------------
# What it says when it is not there
# ---------------------------------------------------------------------------

def test_describe_says_how_to_get_it():
    text = describe()
    if not rapids_available():
        assert "rapids" in text.lower()


def test_describe_names_the_interpreter_constraint():
    """3.11/3.12 is cuML's own declaration, and an installer or user hitting
    it needs to know why rather than reading a resolver error."""
    if not rapids_available():
        assert "3.11" in describe() or "disabled" in describe()


# ---------------------------------------------------------------------------
# The packaging promise
# ---------------------------------------------------------------------------

def test_rapids_is_an_extra_and_not_a_dependency():
    """As a dependency it would drop Python 3.9, 3.10, 3.13 and 3.14."""
    from pathlib import Path

    setup = Path(__file__).resolve().parents[1] / "setup.py"
    text = setup.read_text(encoding="utf-8")
    assert "'rapids':" in text
    # cuml must appear ONLY inside extras, never in install_requires.
    install_requires = text.split("install_requires")[1].split("extras_require")[0]
    assert "cuml" not in install_requires


def test_the_extra_carries_the_interpreter_marker():
    """Without it, `pip install spacr[rapids]` on 3.13 produces a resolver
    error the user cannot read."""
    from pathlib import Path

    setup = Path(__file__).resolve().parents[1] / "setup.py"
    text = setup.read_text(encoding="utf-8")
    # The REQUIREMENT line, not the comment above it that explains why the
    # marker is there -- a substring search finds the prose first.
    line = [l for l in text.splitlines()
            if "cuml-cu12" in l and "python_version" in l][0]
    assert 'python_version >= "3.11"' in line
    assert 'python_version < "3.13"' in line


def test_the_conda_forge_recipe_does_not_list_cuml():
    """An extra is invisible to conda-forge, which is the point."""
    from pathlib import Path

    recipe = (Path(__file__).resolve().parents[1]
              / "conda-forge" / "recipe" / "recipe.yaml")
    if not recipe.is_file():
        pytest.skip("no conda-forge recipe in this checkout")
    assert "cuml" not in recipe.read_text(encoding="utf-8")
