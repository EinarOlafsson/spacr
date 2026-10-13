"""A stylesheet that cannot be registered costs a panel its look, not the run.

Instruction 288. Seven modules register a widget stylesheet AT IMPORT
TIME, each wrapped in the same try/except:

    try:
        from .theme import register_widget_qss as _register_widget_qss
        _register_widget_qss(NAME, _qss, replace=True)
    except Exception:
        LOG.debug("could not register the ... QSS", exc_info=True)

The ``try`` is taken in every real launch; the ``except`` is the arm no
test reached, and it is the one that matters. An exception at import time
does not cost a panel its background -- it stops the module importing at
all, which takes down whatever imports it. On this import path that is
the application.

Driven by reloading each module with ``register_widget_qss`` raising, and
asserting the module still comes back with its stylesheet name intact.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

#: (module, the constant naming its stylesheet, or None when the module
#: passes a literal instead of a constant)
REGISTERING_MODULES = [
    ("spacr.qt.settings_search", "BAR_NAME"),
    ("spacr.qt.shortcuts", "OVERLAY_NAME"),
    ("spacr.qt.recipes", "RECIPE_BUTTON_NAME"),
    ("spacr.qt.prerun", "QSS_NAME"),
    ("spacr.qt.screens.map_barcodes", "PAGES_NAME"),
    ("spacr.qt.widgets.gate_editor", "QSS_NAME"),
    ("spacr.qt.screens.settings_model", None),
]


def test_the_list_matches_what_the_package_actually_does():
    """The sweep is worthless if a module drops out of it silently."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "spacr" / "qt"
    found = {
        f"spacr.qt.{path.relative_to(root).with_suffix('')}".replace("/", ".")
        for path in root.rglob("*.py")
        if "_register_widget_qss(" in path.read_text(encoding="utf-8")
    }
    listed = {name for name, _const in REGISTERING_MODULES}
    assert listed <= found, (
        f"these no longer register a stylesheet: {listed - found}")


@pytest.mark.parametrize("module_name,constant", REGISTERING_MODULES)
def test_a_refusing_theme_still_lets_the_module_import(module_name, constant,
                                                       monkeypatch):
    """THE ARM.

    An exception here does not cost a panel its background -- it stops
    the module importing, and on this path that is the application.
    """
    import spacr.qt.theme as theme

    asked = []

    def _refuse(*_args, **_kwargs):
        asked.append(True)
        raise RuntimeError("the theme registry is not ready")

    monkeypatch.setattr(theme, "register_widget_qss", _refuse)

    module = importlib.reload(importlib.import_module(module_name))

    assert asked, f"{module_name} never registered a stylesheet"
    if constant is not None:
        assert getattr(module, constant, None), (
            f"{module_name} imported but lost {constant}")


@pytest.mark.parametrize("module_name,constant", REGISTERING_MODULES)
def test_a_working_theme_registers_the_sheet(module_name, constant,
                                             monkeypatch):
    """So the arm above is about the failure, not about a module that
    never registers anything."""
    import spacr.qt.theme as theme

    registered = []

    def _record(name, qss, replace=False):
        registered.append((name, replace))

    monkeypatch.setattr(theme, "register_widget_qss", _record)

    module = importlib.reload(importlib.import_module(module_name))

    assert registered, f"{module_name} registered nothing"
    names = [name for name, _replace in registered]
    if constant is not None:
        assert getattr(module, constant) in names, (
            f"{module_name} registered {names}, not its own {constant}")
    else:
        # A module that passes a literal still has to register SOMETHING
        # under a name of its own rather than an empty string.
        assert all(name for name in names), (
            f"{module_name} registered a stylesheet with no name")
    assert all(replace for _name, replace in registered), (
        "the sheet is registered without replace=True, so a reload "
        "stacks a second copy on the first")


@pytest.fixture(autouse=True)
def _leave_the_modules_as_they_were():
    """Put each module back exactly as it was, the SAME objects included.

    These tests reload with a stubbed theme. Reloading once more afterwards
    -- what this fixture used to do -- re-registers the stylesheet, but it
    also mints a new class for every class in the module, and every module
    that had already done ``from .settings_model import _ScalarEdit`` keeps
    the old one. For the rest of the process the two disagree:
    ``isinstance(widget, _ScalarEdit)`` is False for a widget the screen
    built, and a test that monkeypatches the module patches objects the
    screen no longer calls. A serial ``pytest tests/qt`` failed four tests
    in ``test_cov_app_screen.py`` that way (features/new/47).

    So the namespace is restored rather than re-executed. The stylesheets
    need nothing: both stubs stand in for ``register_widget_qss``, so the
    real registry still holds what the original import put there.
    """
    saved = {}
    for module_name, _constant in REGISTERING_MODULES:
        module = importlib.import_module(module_name)
        saved[module_name] = (module, dict(vars(module)))
    yield
    for module, namespace in saved.values():
        vars(module).clear()
        vars(module).update(namespace)
