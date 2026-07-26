"""Simulation figures must not land in the process's current directory.

Five call sites passed ``src='figures'`` -- a RELATIVE path -- so every
simulator run dropped a ``figures/`` tree into whatever directory the process
was launched from. A test run leaked ``figures/feature_importance`` into the
repo working tree, which is how this was found.
"""
import os
import pytest


def test_default_is_unchanged_so_existing_callers_still_work():
    from spacr.sim import _figures_dst
    assert _figures_dst() == 'figures'
    assert _figures_dst(None) == 'figures'


def test_an_explicit_destination_wins(tmp_path):
    from spacr.sim import _figures_dst
    assert _figures_dst(str(tmp_path)) == str(tmp_path)
    # a Path, not just a str
    assert _figures_dst(tmp_path) == str(tmp_path)


def test_the_environment_variable_redirects_the_default(tmp_path, monkeypatch):
    from spacr.sim import _figures_dst
    monkeypatch.setenv('SPACR_SIM_FIGURES', str(tmp_path))
    assert _figures_dst() == str(tmp_path)


def test_an_explicit_destination_beats_the_environment(tmp_path, monkeypatch):
    from spacr.sim import _figures_dst
    monkeypatch.setenv('SPACR_SIM_FIGURES', str(tmp_path / 'env'))
    assert _figures_dst(str(tmp_path / 'arg')) == str(tmp_path / 'arg')


@pytest.mark.parametrize("name", [
    "plot_correlation_matrix",
    "plot_feature_importance",
    "calculate_permutation_importance",
    "plot_partial_dependences",
    "generate_shap_summary_plot",
])
def test_every_plotting_entry_point_accepts_a_destination(name):
    """A caller must be able to say where figures go, at every site."""
    import inspect
    import spacr.sim as sim
    fn = getattr(sim, name)
    params = inspect.signature(fn).parameters
    assert 'dst' in params, f"{name} cannot be told where to write"
    assert params['dst'].default is None, f"{name} changed its default"


def test_no_call_site_still_hard_codes_the_relative_path():
    """The literal must be gone from the module body, not merely shadowed."""
    import inspect
    import spacr.sim as sim
    src = inspect.getsource(sim)
    assert "src='figures'" not in src
    assert 'src="figures"' not in src
