"""Every registered module has a route, and the API says which.

The home screen draws twenty-one tiles out of forty-four registered
modules. The other twenty-three are reached from a host's masthead, from
the Help menu, or from the command palette -- and the failure this file
prevents is a module that is registered, shipped, documented and
reachable by NONE of them.

That is not hypothetical. Instruction 318 removed twenty-three tiles, and
between the removal and the buttons landing, ten modules had no route at
all. Nothing failed: they built, they were translated, their API pages
generated, and a user simply could not open them.

The API's own reference is generated from the same tables the application
walks, so it cannot claim a fold the GUI does not install.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[1]
FOLDS_PAGE = ROOT / "docs" / "source" / "_generated" / "folded_modules.rst"


@pytest.fixture(scope="module")
def routes():
    """``(every key, tiled, folded, help)`` after full registration."""
    import spacr.qt
    from spacr.qt.app import APPS, _HELP_MODULES, tiled_apps
    from spacr.qt.screens.map_barcodes import FOLD_HOST_MODULES

    spacr.qt.register_self_registering_modules()
    folded = {}
    for host_key, module_name in FOLD_HOST_MODULES.items():
        module = import_module(f"spacr.qt.screens.{module_name}")
        for key in getattr(module, "FOLDED_APPS", ()):
            folded[key] = host_key
    return (
        {row[0] for row in APPS},
        {row[0] for row in tiled_apps()},
        folded,
        {key for key, *_rest in _HELP_MODULES} | {"feature_dict"},
    )


def test_no_registered_module_is_unreachable(routes):
    """A module with no tile, no host and no Help entry cannot be opened.

    ``feature_dict`` is counted as a Help entry: it registers its own
    action from ``spacr/qt/widgets/feature_dictionary.py`` rather than
    from app.py's table, so it is reachable but not listed there.
    """
    every, tiled, folded, helped = routes
    stranded = sorted(every - tiled - set(folded) - helped)
    assert not stranded, (
        f"these modules are registered but nothing opens them: {stranded}")


def test_a_module_is_not_offered_two_ways(routes):
    """A tile AND a fold button for the same module is the thing 318
    removed: two doors into one screen, and no way to tell which is
    canonical."""
    _every, tiled, folded, _helped = routes
    both = sorted(tiled & set(folded))
    assert not both, f"these have a tile and a fold button: {both}"


def test_the_api_page_names_every_folded_module(routes):
    """The generated reference is complete.

    Without it, the API index draws only the tiled modules and the other
    twenty-three cannot be reached from it at all.
    """
    _every, _tiled, folded, _helped = routes
    text = FOLDS_PAGE.read_text(encoding="utf-8")
    from spacr.qt.screens.map_barcodes import fold_description

    missing = [key for key in folded
               if (fold_description(key)[0] or key) not in text]
    assert not missing, (
        f"the API's fold reference does not name: {sorted(missing)}. "
        "Re-run packaging/generate_readme_visuals.py")


def test_the_api_page_names_the_host_of_each_fold(routes):
    """Naming the module is not enough -- it must say what opens it.

    "Investigate Hit exists" and "Regression opens Investigate Hit" are
    different facts, and only the second one lets a reader find it.
    """
    _every, _tiled, folded, _helped = routes
    from spacr.qt.app import APPS

    names = {row[0]: row[1] for row in APPS}
    text = FOLDS_PAGE.read_text(encoding="utf-8")
    for host in {names.get(h, h) for h in folded.values()}:
        assert f"**{host}** opens" in text, (
            f"the fold reference never says what {host} opens")
