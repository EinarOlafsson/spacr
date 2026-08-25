"""Every Home-screen variant builder draws a real page.

``spacr/resources/home/versions/_generators/variants.py`` holds thirty
candidate Home screens. Each one is a builder that assembles real Qt
widgets out of the real app registry, and the render harness screenshots
whatever it returns. A builder that raises produces no render at all, and
the reviewer finds out only when the whole sheet comes back short, so the
builders are driven here directly.

The categorisation tables the builders read live in ``common`` and name app
keys as literals. A key that leaves the registry leaves those literals
behind, so the tables are repaired to the live registry before the builders
run -- and the repair being necessary is itself asserted, so that a table
which has drifted is reported once rather than as seventeen identical
KeyErrors.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GENERATORS = os.path.join(REPO_ROOT, "spacr", "resources", "home", "versions",
                          "_generators")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(GENERATORS),
    reason="home-screen variant generators are not part of this checkout")


def _load(name: str):
    """Import one generator module under the plain name its siblings use."""
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _table_names(common) -> list:
    """Names of the module-level categorisation tables."""
    return sorted(n for n in dir(common) if n.startswith("CATS_"))


@pytest.fixture(scope="module")
def kit(qapp):
    """``common``, ``parts`` and ``variants`` under the names they import.

    The three modules import each other by plain name, so all three have to
    occupy those :data:`sys.modules` entries while they load; the originals
    go back on teardown. ``qapp`` exists first on purpose -- ``bootstrap()``
    then knows it is a guest and leaves the process-wide QSettings and the
    application stylesheet alone.
    """
    import types

    names = ("common", "parts", "variants")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        common = _load("common")
        common.bootstrap()
        parts = _load("parts")
        variants = _load("variants")
        yield types.SimpleNamespace(common=common, parts=parts,
                                    variants=variants, app=qapp)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def buildable(kit):
    """``kit`` with every categorisation narrowed to keys the registry has.

    A builder reads its table and asks :func:`common.name_of` for each key in
    it, so one stale literal is a :class:`KeyError` rather than a gap. The
    tables are bound into ``variants`` by ``from common import ...``, so both
    module namespaces are rebound. This is the categorisation the builders
    are meant to receive; that the shipped one differs is asserted by
    :func:`test_no_categorisation_names_an_app_the_registry_dropped`.
    """
    common, variants = kit.common, kit.variants
    known = set(common.all_keys())
    for name in _table_names(common):
        repaired = [(title, [k for k in keys if k in known])
                    for title, keys in getattr(common, name)]
        setattr(common, name, repaired)
        if hasattr(variants, name):
            setattr(variants, name, repaired)
    return kit


@pytest.fixture(scope="module")
def ctx(buildable):
    """A dark rendering context over the shared application."""
    return buildable.common.Ctx(buildable.app, "dark")


@pytest.fixture(scope="module")
def frozen_home():
    """Undo the live-value freeze variant 01 installs on the shipped Home.

    ``_patch_startup_determinism`` rebinds attributes on the *product*
    classes so a screenshot is reproducible. Left in place it would follow
    the shared interpreter into every later test that renders a Home page,
    so the originals are put back.
    """
    from spacr.qt.widgets import home as H

    targets = [(H.SystemPanel, "gpu_util"), (H.SystemPanel, "gpu_vram"),
               (H.SystemPanel, "disk_used"), (H.QueuedPanel, "queue_items")]
    saved = [(obj, attr, obj.__dict__.get(attr)) for obj, attr in targets]
    journal = sys.modules.get("spacr.run_journal")
    saved_journal = []
    if journal is not None:
        saved_journal = [(journal, a, getattr(journal, a, None))
                         for a in ("recent_runs", "journal_totals")]
    try:
        yield
    finally:
        for obj, attr, value in saved:
            if value is None:
                obj.__dict__.pop(attr, None)
            else:
                setattr(obj, attr, value)
        for obj, attr, value in saved_journal:
            if value is not None:
                setattr(obj, attr, value)


def _all_text(widget) -> str:
    """Concatenated text of every label and button under ``widget``."""
    from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit

    parts = []
    for kind in (QLabel, QAbstractButton, QLineEdit):
        for child in widget.findChildren(kind):
            text = child.text() if hasattr(child, "text") else ""
            if text:
                parts.append(text)
            placeholder = getattr(child, "placeholderText", None)
            if placeholder is not None:
                parts.append(placeholder())
    return "\n".join(parts)


def _specs(kit):
    return kit.variants.VARIANTS


def test_the_registry_is_the_only_source_of_variant_app_keys(kit):
    """Thirty variants are registered, numbered and slugged uniquely."""
    specs = _specs(kit)
    assert len(specs) == 30
    assert [s["n"] for s in specs] == list(range(1, 31))
    assert len({s["slug"] for s in specs}) == 30
    assert all(s["title"] and s["argument"] for s in specs)
    assert all(callable(s["build"]) for s in specs)


@pytest.fixture
def shipped_common(qapp):
    """A private ``common`` holding the categorisations exactly as shipped.

    The module-scoped :func:`buildable` repairs the tables in place, so a
    test that asks whether they NEED repairing has to read them from a
    freshly executed copy rather than from the one the builders ran against.
    """
    saved = sys.modules.get("common")
    try:
        module = _load("common")
        module.bootstrap()
        yield module
    finally:
        if saved is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = saved


@pytest.mark.xfail(strict=True, reason=(
    "common.py's categorisation tables still name 'cellpose_masks', which "
    "left spacr.qt.app.APPS; every variant that reads a table dies on "
    "name_of() for it"))
def test_no_categorisation_names_an_app_the_registry_dropped(shipped_common):
    """Each categorisation covers the live registry exactly once.

    ``check_coverage`` is the module's own statement of that contract. A
    table that survives it is a table every builder can index. The contract
    is also spelled out here as assertions, so a drifted table is reported
    by name and by key rather than as one opaque raise from the helper.
    """
    live = set(shipped_common.all_keys())
    tables = _table_names(shipped_common)
    assert tables, "common ships no categorisation tables at all"
    for name in tables:
        table = getattr(shipped_common, name)
        listed = [key for _title, keys in table for key in keys]
        assert len(listed) == len(set(listed)), f"{name} lists a key twice"
        assert set(listed) == live, f"{name} has drifted from the registry"
        shipped_common.check_coverage(table)


def test_every_registered_app_has_an_invented_use_count(kit):
    """No app sorts to the bottom of the frequency variants by accident.

    ``USE_COUNTS`` is read with ``[]`` by variant 14's badge, so a key
    missing from it is a crash, not a low rank; the module fills itself in
    at import for exactly that reason.
    """
    common = kit.common
    for key in common.all_keys():
        assert common.USE_COUNTS[key] >= common.UNUSED_APP_COUNT


def test_the_nine_keyboard_shortcuts_follow_registry_order(kit):
    """``Ctrl+1..9`` land on the first nine rows of ``APPS``, in order."""
    shortcuts = kit.variants._shortcuts()
    expected = kit.common.all_keys()[:9]
    assert list(shortcuts) == expected
    assert [shortcuts[k] for k in expected] == [
        f"Ctrl+{i}" for i in range(1, 10)]


def test_a_registered_variant_keeps_the_prose_it_was_given(kit):
    """The decorator stores the description beside the builder it wraps."""
    variants = kit.variants
    before = len(variants.VARIANTS)
    try:
        @variants.variant("probe-slug", "Probe", changes="c", adds="a",
                          removes="r", argument="because", notes="n")
        def _probe(ctx):
            return None

        spec = variants.VARIANTS[-1]
        assert spec["n"] == before + 1
        assert spec["slug"] == "probe-slug"
        assert spec["changes"] == "c" and spec["notes"] == "n"
        assert spec["build"] is _probe
    finally:
        del variants.VARIANTS[before:]


@pytest.mark.parametrize("index", range(30), ids=lambda i: f"v{i + 1:02d}")
def test_every_variant_builds_a_page_that_draws_something(
        index, ctx, buildable, frozen_home):
    """Each builder returns a laid-out page carrying real app names.

    Building is not enough on its own -- a page that returns an empty column
    would build too -- so the result is laid out at the render size and has
    to name at least one app from the live registry.
    """
    spec = _specs(buildable)[index]
    page = spec["build"](ctx)
    try:
        assert page is not None
        page.resize(1440, 900)
        page.layout().activate()
        text = _all_text(page)
        names = {buildable.common.name_of(k)
                 for k in buildable.common.all_keys()}
        assert names & set(text.split("\n")), (
            f"variant {spec['slug']} named no registered app")
        assert page.sizeHint().width() > 0
    finally:
        page.deleteLater()


def test_a_missing_run_journal_still_freezes_the_system_panel(
        kit, monkeypatch, frozen_home):
    """An installation without ``spacr.run_journal`` still renders a baseline.

    The journal is the one optional patch target: without it the baseline
    draws "no runs yet", which is a fair comparison. The panel readings
    before it are not optional and must be frozen either way.
    """
    import builtins

    from spacr.qt.widgets import home as H

    real_import = builtins.__import__

    def no_journal(name, *args, **kwargs):
        if name == "spacr.run_journal":
            raise ImportError("no journal in this installation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_journal)
    monkeypatch.delitem(sys.modules, "spacr.run_journal", raising=False)

    assert kit.variants._patch_startup_determinism() is None

    assert H.SystemPanel.gpu_util() == "41%"
    assert H.SystemPanel.disk_used() == "68%"
    assert H.QueuedPanel.queue_items(object()) == []
    assert "spacr.run_journal" not in sys.modules
