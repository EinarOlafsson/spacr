"""The ``spacr`` package facade — lazy submodules and the frozen-bundle floor.

``import spacr`` must stay cheap, so every submodule and the API facade are
reached through ``__getattr__``. That makes this module the one place where
"the name exists" and "the name imports" can disagree, and both halves are
asserted here:

* the four facade names resolve through :mod:`spacr.api`;
* every name in the inventory is importable, so a module that was added to
  the directory but never wired up is caught here rather than by a user;
* the inventory itself is the DIRECTORY widened by the documented tuple, and
  it falls back to the tuple alone when there is no directory to read --
  which is what a PyInstaller bundle looks like from the inside.
"""
from __future__ import annotations


import pytest

import spacr


def test_the_version_is_a_real_version_string():
    assert spacr.__version__
    assert spacr.__version__[0].isdigit()


def test_the_facade_names_come_from_the_api_module():
    from spacr import api

    for name in ("MaskConfig", "MeasureConfig", "run_mask", "run_measure"):
        assert getattr(spacr, name) is getattr(api, name)


def test_a_lazy_submodule_is_the_imported_module():
    schema = spacr.schema
    import spacr.schema as direct

    assert schema is direct


def test_a_name_that_is_neither_says_so():
    with pytest.raises(AttributeError, match="no attribute 'not_a_module'"):
        spacr.not_a_module


def test_tab_completion_offers_the_facade_and_every_submodule():
    listed = set(dir(spacr))
    assert {"MaskConfig", "run_measure"} <= listed
    assert {"schema", "crop_source", "curation"} <= listed
    assert "__version__" in listed


def test_the_inventory_is_the_directory_widened_by_the_documented_tuple():
    """A hand-kept list of the files in its own directory drifted four times."""
    on_disk = spacr._submodules_on_disk()
    assert "crop_source" in on_disk
    assert "__init__" not in on_disk
    assert set(spacr._DOCUMENTED_SUBMODULES) <= set(spacr._SUBMODULES)
    assert on_disk <= set(spacr._SUBMODULES)


def test_a_bundle_with_no_directory_falls_back_to_the_documented_names(
        monkeypatch):
    """PyInstaller keeps the modules in an archive; there is nothing to scan."""
    def no_directory(_path):
        raise OSError("not a directory in a frozen bundle")

    monkeypatch.setattr(spacr._os, "listdir", no_directory)
    assert spacr._submodules_on_disk() == frozenset()


def test_download_models_is_imported_only_when_it_is_called(monkeypatch):
    """Keeping it lazy is what keeps ``import spacr`` cheap."""
    from spacr import utils

    seen = {}
    monkeypatch.setattr(
        utils, "download_models",
        lambda repo_id, retries, delay: seen.update(
            repo=repo_id, retries=retries, delay=delay) or "downloaded")

    assert spacr.download_models(repo_id="someone/models", retries=2,
                                 delay=0) == "downloaded"
    assert seen == {"repo": "someone/models", "retries": 2, "delay": 0}


def test_fonttools_is_pinned_at_warning_by_importing_spacr():
    """Forty INFO lines per saved figure would bury the run's own output."""
    import logging

    for name in ("fontTools", "fontTools.subset", "fontTools.ttLib"):
        assert logging.getLogger(name).level == logging.WARNING


def test_every_name_the_package_offers_can_actually_be_imported():
    """"The name exists" and "the name imports" must not disagree."""
    import importlib

    for name in spacr._DOCUMENTED_SUBMODULES:
        assert importlib.util.find_spec(f"spacr.{name}") is not None, name
