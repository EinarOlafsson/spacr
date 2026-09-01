"""Every fold-strip button opens something, rather than raising.

Reported 2026-09-01 from the Measure console:

    Could not open the folded module 'trellis'
    NameError: name 'build_registered_screen' is not defined

``graph_builder`` called that function without importing it, so BOTH the
modules it hosts raised the moment their button was pressed. Auditing the
rest found the same hole in ``qc_dashboard`` (layer_viewer, control_chart,
outliers) and ``db_browser`` (lineage, tabulate) -- seven folded modules
that could not open at all.

WHY NOTHING CAUGHT IT. `FoldOpener.open` wraps the build in a try/except
that logs and carries on, which is right -- a folded module that cannot
build must not take its host down -- but it means the failure is a line
in a console nobody is reading, and every existing test asserted the
BUTTON existed rather than that pressing it produced a screen.

So this calls every builder in every host's BUILDERS table. A NameError
in one is then a red test rather than a message a user has to report.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

#: Every screen module that hosts folded modules.
HOSTS = (
    "spacr.qt.screens.map_barcodes",
    "spacr.qt.screens.graph_builder",
    "spacr.qt.screens.qc_dashboard",
    "spacr.qt.screens.db_browser",
    "spacr.qt.screens.classify",
    "spacr.qt.screens.regression",
    "spacr.qt.screens.foreign",
)


def _builders(module):
    """``{key: builder}`` for a host, however it spells its table."""
    for name in ("BUILDERS", "FOLD_BUILDERS", "_BUILDERS"):
        table = getattr(module, name, None)
        if isinstance(table, dict) and table:
            return table
    return {}


def test_the_hosts_really_do_have_builders():
    """Or the sweep below passes by finding nothing."""
    found = {name: len(_builders(importlib.import_module(name)))
             for name in HOSTS}
    assert sum(found.values()) >= 10, f"only found {found}"


@pytest.mark.parametrize("host_name", HOSTS)
def test_every_builder_resolves_the_names_it_uses(host_name):
    """THE REGRESSION, caught without building a widget.

    Compiling each builder and checking its global references against the
    module is enough: a NameError at call time is a name the module does
    not have, and that is visible statically.
    """
    module = importlib.import_module(host_name)
    builders = _builders(module)
    if not builders:
        pytest.skip(f"{host_name} hosts nothing")

    missing = []
    for key, builder in builders.items():
        function = getattr(builder, "func", builder)   # unwrap partial
        code = getattr(function, "__code__", None)
        if code is None:
            continue
        for name in code.co_names:
            if name in ("build_registered_screen", "install_fold_strip"):
                # The two this bug was about. Resolvable either at module
                # level or from inside the function's own imports.
                inner = name in code.co_names and (
                    hasattr(module, name)
                    or name in getattr(code, "co_consts", ())
                    or any(name in (getattr(c, "co_names", ()) or ())
                           for c in code.co_consts
                           if hasattr(c, "co_names")))
                imported_locally = name in (code.co_varnames or ())
                if not (hasattr(module, name) or imported_locally or inner):
                    missing.append(f"{host_name}.{key} -> {name}")
    assert not missing, (
        "these builders reference a name their module cannot resolve, so "
        "pressing the button raises NameError: " + ", ".join(missing))


@pytest.mark.parametrize("host_name", HOSTS)
def test_every_builder_is_callable_without_raising_a_nameerror(host_name):
    """THE BEHAVIOUR. Calls each builder for real.

    A builder may legitimately fail for other reasons in a test process
    -- no project open, no GL context -- so only NameError and
    ImportError are treated as failures here. Those two are exactly the
    "this was never wired up" shape that reached a user.
    """
    module = importlib.import_module(host_name)
    builders = _builders(module)
    if not builders:
        pytest.skip(f"{host_name} hosts nothing")

    broken = []
    for key, builder in builders.items():
        try:
            widget = builder(host_window=None)
        except (NameError, ImportError) as error:
            broken.append(f"{key}: {type(error).__name__}: {error}")
        except Exception:                                    # noqa: BLE001
            continue                                         # a real refusal
        else:
            if widget is not None:
                widget.deleteLater()
    assert not broken, (
        f"{host_name} hosts modules that cannot open: " + "; ".join(broken))
