"""Drive every Home-screen variant builder through a real layout.

``spacr/resources/home/versions/_generators/variants.py`` holds thirty
builders. Each one assembles a candidate Home screen out of the shipped
Qt widgets and the live app registry, and each one is only exercised by
actually building it -- there is no shorter path to its body.

Seventeen of the thirty cannot be built from the categorisation tables
``common`` ships, because those tables name two app keys the registry no
longer defines. That is a defect in the tables, not in the builders: a
builder's job is to lay out whatever ``(title, keys)`` table it is
handed, and it does that correctly for every table whose keys exist.
:func:`_registry_consistent` therefore drops the dangling keys before the
builders run, which is precisely the invariant
``common.check_coverage`` demands of a categorisation. The unrepaired
case is asserted separately, and marked ``xfail(strict=True)`` so that
repairing the tables turns it into a failure that announces itself.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip("PySide6")

GENERATORS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "spacr", "resources", "home", "versions", "_generators")

#: The categorisation tables the builders read as module globals.
CAT_TABLES = ("CATS_BROAD3", "CATS_STAGE5", "CATS_NARROW8",
              "CATS_QUESTIONS", "CATS_INTENT4")


def _load(name: str):
    """Import one generator module under its plain name."""
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen(qapp):
    """The generator modules, loaded under the plain names they import.

    ``parts``, ``variants`` and ``render`` reach for each other as
    top-level modules, so they have to occupy those ``sys.modules``
    entries while they load; whatever was there before is put back.

    Depending on ``qapp`` means a QApplication already exists when
    ``common.bootstrap()`` runs, which keeps ``common`` a guest: it then
    refuses to redirect the process-wide QSettings path or to restyle an
    application it does not own.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
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


@pytest.fixture
def frozen_home(gen):
    """Undo the module-level patching ``_patch_startup_determinism`` does.

    The function replaces methods on the shipped ``SystemPanel`` and
    ``QueuedPanel`` and on ``spacr.run_journal``. Those writes outlive
    the test that triggered them, so every one of them is captured and
    restored -- otherwise the first variant built here decides what the
    rest of the Qt session reads for GPU load and run history.
    """
    from spacr.qt.widgets import home as H
    targets = [(H.SystemPanel, "gpu_util"), (H.SystemPanel, "gpu_vram"),
               (H.SystemPanel, "disk_used"), (H.QueuedPanel, "queue_items")]
    try:
        import spacr.run_journal as journal
    except Exception:
        journal = None
    else:
        targets += [(journal, "recent_runs"), (journal, "journal_totals")]
    saved = [(obj, name, obj.__dict__.get(name, _MISSING))
             for obj, name in targets]
    yield journal
    for obj, name, value in saved:
        if value is _MISSING:
            obj.__dict__.pop(name, None)
        else:
            setattr(obj, name, value)


class _Missing:
    pass


_MISSING = _Missing()


def _registry_consistent(cats, live):
    """``cats`` with every key the app registry no longer defines removed."""
    return [(title, [key for key in keys if key in live])
            for title, keys in cats]


@pytest.fixture
def buildable(gen, monkeypatch, frozen_home):
    """A context whose categorisation tables match the live registry."""
    live = set(gen.common.all_keys())
    for name in CAT_TABLES:
        repaired = _registry_consistent(getattr(gen.variants, name), live)
        monkeypatch.setattr(gen.variants, name, repaired)
    return gen.common.Ctx(gen.app, "dark")


def _texts(widget):
    """Every non-empty string any descendant widget displays."""
    from PySide6.QtWidgets import QAbstractButton, QLabel
    out = []
    for child in widget.findChildren(QLabel):
        if child.text():
            out.append(child.text())
    for child in widget.findChildren(QAbstractButton):
        if child.text():
            out.append(child.text())
    return out


def _variant_numbers(gen):
    return [entry["n"] for entry in gen.variants.VARIANTS]


@pytest.mark.parametrize("number", range(1, 31))
def test_every_variant_builds_a_page_that_shows_real_app_names(
        gen, buildable, number):
    """Each builder returns a laid-out widget quoting the real registry.

    The assertion is deliberately about registry text rather than widget
    counts: a page that built but drew none of the app names it was
    given has failed at the one job every variant shares.
    """
    entries = {entry["n"]: entry for entry in gen.variants.VARIANTS}
    assert number in entries, f"variant {number} is not registered"
    entry = entries[number]

    page = entry["build"](buildable)

    from PySide6.QtWidgets import QWidget
    assert isinstance(page, QWidget)
    shown = " \n".join(_texts(page))
    assert shown.strip(), f"variant {number} drew no text at all"
    names = [gen.common.name_of(key) for key in gen.common.all_keys()]
    hits = [name for name in names if name in shown]
    assert len(hits) >= 3, (
        f"variant {number} ({entry['slug']}) shows only {hits} of the "
        f"{len(names)} registered app names")
    page.deleteLater()


def test_the_registry_is_thirty_numbered_variants(gen):
    """Numbers run 1..30 with no gap, and slug and title are unique."""
    entries = gen.variants.VARIANTS
    assert _variant_numbers(gen) == list(range(1, len(entries) + 1))
    assert len(entries) == 30
    slugs = [entry["slug"] for entry in entries]
    titles = [entry["title"] for entry in entries]
    assert len(set(slugs)) == len(slugs)
    assert len(set(titles)) == len(titles)
    for entry in entries:
        assert callable(entry["build"])
        for field in ("changes", "adds", "removes", "argument"):
            assert entry[field].strip(), f"{entry['slug']} has an empty {field}"


def test_shortcuts_are_ctrl_one_to_nine_in_registry_order(gen):
    """``Ctrl+N`` lands on the registry's Nth app, not a section's Nth."""
    shortcuts = gen.variants._shortcuts()
    keys = gen.common.all_keys()[:9]
    assert list(shortcuts) == keys
    assert [shortcuts[key] for key in keys] == [
        f"Ctrl+{i + 1}" for i in range(len(keys))]


def test_startup_determinism_freezes_gpu_disk_queue_and_journal(
        gen, frozen_home):
    """The live readings a screenshot must not depend on are pinned."""
    journal = frozen_home
    gen.variants._patch_startup_determinism()

    from spacr.qt.widgets import home as H
    assert H.SystemPanel.gpu_util() == "41%"
    assert H.SystemPanel.gpu_vram() == "14.9 / 24 GB"
    assert H.SystemPanel.disk_used() == "68%"
    assert H.QueuedPanel.queue_items(object()) == []

    if journal is not None:
        runs = journal.recent_runs(limit=2)
        assert [run["app_key"] for run in runs] == [
            row[0] for row in gen.common.MOCK["recent"][:2]]
        assert {run["status"] for run in runs} <= {"success", "error"}
        assert journal.journal_totals()["total_runs"] == 148


def test_startup_determinism_survives_a_missing_run_journal(
        gen, frozen_home, monkeypatch):
    """No run journal is a fair baseline, not a crash.

    The import is blocked at ``builtins.__import__`` so the real
    ``except ImportError: return`` runs -- the panels are still frozen,
    and the function comes back without touching the journal.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "spacr.run_journal" or name.startswith("spacr.run_journal."):
            raise ImportError("no run journal in this installation")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "spacr.run_journal", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    gen.variants._patch_startup_determinism()

    from spacr.qt.widgets import home as H
    assert H.SystemPanel.disk_used() == "68%"


def test_variant_decorator_appends_its_prose_and_returns_the_builder(gen):
    """Registering a variant keeps the function callable and numbers it."""
    before = len(gen.variants.VARIANTS)
    try:
        @gen.variants.variant("probe-slug", "Probe",
                              changes="c", adds="a", removes="r",
                              argument="why", notes="n")
        def _probe(ctx):
            return "built"

        assert _probe(None) == "built"
        entry = gen.variants.VARIANTS[-1]
        assert entry["n"] == before + 1
        assert entry["slug"] == "probe-slug"
        assert entry["notes"] == "n"
        assert entry["build"] is _probe
    finally:
        del gen.variants.VARIANTS[before:]


@pytest.mark.xfail(strict=True, reason="common's categorisation tables name "
                                       "app keys the registry dropped")
def test_every_variant_builds_from_the_shipped_categorisations(gen,
                                                               frozen_home):
    """The thirty variants build from the tables ``common`` ships.

    They do not: ``CATS_BROAD3``/``CATS_STAGE5``/``CATS_NARROW8``/
    ``CATS_QUESTIONS`` still list ``cellpose_masks``, which was folded
    into ``train_cellpose``, and none of them lists ``parameter_sweep``
    in a way the registry agrees with -- so ``common.name_of`` raises
    ``KeyError`` and seventeen builders die. Fixing the tables makes
    this pass, and a strict xfail turns that into a failure that says so.
    """
    ctx = gen.common.Ctx(gen.app, "dark")
    built = 0
    for entry in gen.variants.VARIANTS:
        page = entry["build"](ctx)
        # A builder that returned an empty shell would raise nothing and
        # render a blank sheet, so each page is asked for its words.
        assert " \n".join(_texts(page)).strip(), \
            f"variant {entry['slug']} drew no text at all"
        page.deleteLater()
        built += 1
    assert built == len(gen.variants.VARIANTS)
