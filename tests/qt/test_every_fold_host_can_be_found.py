"""Every screen that folds a module must be identifiable as its host.

`app.folded_children()` builds the ONE host-to-children map that the dock,
the spaCR menu and the generated API page all read. It identifies a host by
`APP_KEY` or `HOST_KEY`, and a module that declares `FOLDED_APPS` without
either is silently skipped -- `FOLDED_APPS` is still read by the fold STRIP,
which is handed its key by the screen, so the button appears and only the
nesting goes missing.

That is what happened to Annotate: it declared `FOLDED_APPS = ("agreement",)`
and no key, so Annotator Agreement had a button on the masthead and no nested
dock row, no menu entry under Annotate, and no line on the API page.

THE FAILURE IS SILENT BY CONSTRUCTION, which is why this is a sweep and not
a test for that one screen. `folded_children` never raises -- a navigation
aid must not be able to stop the window being built -- so a host that drops
out looks exactly like a host with no children.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")


def _fold_host_modules():
    from spacr.qt.app import _EXTRA_FOLD_HOSTS
    from spacr.qt.widgets.fold_strip import FOLD_HOST_MODULES

    return tuple(FOLD_HOST_MODULES) + tuple(_EXTRA_FOLD_HOSTS)


@pytest.mark.parametrize("module_name", _fold_host_modules())
def test_a_host_that_folds_something_declares_its_key(module_name):
    module = importlib.import_module(module_name)
    folded = (getattr(module, "FOLDED_APPS", None)
              or getattr(module, "FOLD_ORDER", None) or ())
    if not folded:
        pytest.skip(f"{module_name} folds nothing")

    key = getattr(module, "APP_KEY", None) or getattr(module, "HOST_KEY", None)
    assert key, (
        f"{module_name} folds {tuple(folded)} and declares neither APP_KEY "
        f"nor HOST_KEY, so `folded_children()` skips it -- the buttons still "
        f"appear and the nesting silently does not")


def test_every_folding_host_reaches_the_map():
    """The same claim from the other end, so a change to how the map is
    built cannot pass the test above while dropping a host."""
    from spacr.qt.app import folded_children

    found = folded_children()
    for module_name in _fold_host_modules():
        module = importlib.import_module(module_name)
        folded = (getattr(module, "FOLDED_APPS", None)
                  or getattr(module, "FOLD_ORDER", None) or ())
        if not folded:
            continue
        key = (getattr(module, "APP_KEY", None)
               or getattr(module, "HOST_KEY", None))
        assert str(key) in found, f"{module_name} is missing from the map"
        assert found[str(key)] == tuple(str(k) for k in folded)


def test_annotate_hosts_annotator_agreement():
    """The specific regression, named so it cannot come back unnoticed."""
    from spacr.qt.app import folded_children

    assert folded_children().get("annotate") == ("agreement",)
