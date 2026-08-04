"""Guards for the thirty Home-screen candidates under
``spacr/resources/home/versions``.

These are *review artefacts*, not shipped UI — nothing under
``spacr/qt/`` imports them, and ``MANIFEST.in`` prunes the whole
directory out of the sdist. What the tests protect is that the
artefacts stay honest:

* every categorisation a variant proposes covers every real app
  exactly once, so a reviewer is never shown a screen that quietly
  drops an app;
* every rendered PNG exists, is exactly 1440x900, and is not a
  solid-colour "the widget never laid out" render;
* the generator still builds all thirty pages out of real widgets with
  no elided or clipped text — text clipping is the defect the
  home-screen rework was raised to fix;
* every "N apps" the generator writes is read from the registry, never
  typed. The prose said "29 apps" for long enough that five more apps
  were registered without a single test noticing.

Why the layout audit runs in a subprocess
-----------------------------------------
The only faithful way to measure these pages is the way ``render.py``
does it: apply the theme stylesheet to the **QApplication**, then build
the widgets, so every tile computes its size post-polish. Neither half
of that is available in-process here. ``QApplication.setStyleSheet``
re-polishes every top-level widget including ones other tests are
tearing down (that is a segfault, and :meth:`common.Ctx.apply_theme`
refuses to do it as a guest), and applying the stylesheet to the page
*after* its children exist gives different — wrong — numbers, because
``QStyleSheetStyle`` re-applies ``QPushButton { min-height: 22px }``
over the sizes the tiles had already fixed. Measured: variant 04 reads
1054 px that way against a clean 900 in a real render. So the audit gets
its own process, which owns its own QApplication and can do it properly.
"""
from __future__ import annotations



import importlib.util
import json
import os
import re
import subprocess
import sys
import types

import pytest

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VERSIONS = os.path.join(REPO_ROOT, "spacr", "resources", "home", "versions")
GENERATORS = os.path.join(VERSIONS, "_generators")

CANVAS = (1440, 900)
N_VARIANTS = 30

#: Variants that are *expected* to need a scrollbar at 1440x900: the
#: shipped baseline (01), the other one that draws the real Sidebar
#: (25), and the deliberately maximal control (30). Everything else
#: fitting without one is the finding these renders exist to make.
SCROLLBARS_ALLOWED = {1, 25, 30}

#: The variants that do NOT fit 1440x900 at the reference zoom, measured
#: at a registry of forty-nine apps. Twenty-one of the thirty are clean and
#: this is the record of the other nine.
#:
#: It is a measurement, not a permission. ``test_no_variant_clips_elides_or_
#: overflows`` compares the audit against this table with ``==``, so all
#: three things fail: a defect appearing in a clean variant, a listed one
#: getting worse, and a listed one getting BETTER. The last is the point —
#: a fix has to delete its line here, which is what stops a known-red
#: ledger from becoming a place defects go to be forgotten.
#:
#: Why these nine are recorded rather than fixed. They are a review
#: surface: thirty candidate home screens rendered from the real widgets so
#: a human can pick one, and nothing in ``_generators/`` is installed into
#: the app. Each entry is a design decision that a person has to take, and
#: v02's own comment in ``variants.py`` spells its afternoon of measurement
#: out: the fourteen — now nineteen — elided names are not a consequence of
#: the overflow, the cause is the 190 px tile, 190 is already the widest a
#: seven-column grid allows, six columns would need a row the page has no
#: room for, and shrinking the icon to 26 px leaves a caption with a bullet
#: beside it. There is no tuning left; the surface either shows fewer apps,
#: gets a taller canvas, or accepts elision with tooltips.
#:
#: The registry going from thirty-four to forty-nine apps is what did this.
#: Every count below is a fact about that growth against a fixed 1440x900,
#: and the three shapes it takes are: names too long for a tile (elided),
#: a description given fewer pixels of height than it needs (clipped), and
#: a page taller than the canvas (overflow).
KNOWN_LAYOUT_DEFECTS: dict = {
    # Five bands of seven, all five now wrapping to a second row.
    2:  {"elided": 19, "overflow": 1},
    3:  {"elided": 5},
    # Not names: ten one-line descriptions given 6-9 px of a 15 px need.
    4:  {"clipped": 10},
    5:  {"elided": 6},
    13: {"clipped": 1},
    17: {"overflow": 1},
    20: {"elided": 1, "overflow": 1},
    28: {"elided": 4, "overflow": 1},
    30: {"elided": 4},
}


def _load(name: str, module_name: str):
    """Import one generator module under an explicit module name."""
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen(qapp):
    """The four generator modules, loaded under their plain names.

    ``parts``, ``variants`` and ``render`` import each other by plain
    name, so they have to occupy those entries in ``sys.modules`` while
    they load; the originals are restored on teardown.

    Depends on pytest-qt's ``qapp`` so a QApplication always exists
    *before* :func:`common.bootstrap` runs. That keeps ``common`` a
    guest, which is what stops it redirecting the process-wide
    ``QSettings`` path and restyling an application it does not own —
    both of which have taken this suite down before.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    names = ("common", "parts", "variants", "render")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        common = _load("common", "common")
        common.bootstrap()
        parts = _load("parts", "parts")
        variants = _load("variants", "variants")
        render = _load("render", "render")
        yield types.SimpleNamespace(common=common, parts=parts,
                                    variants=variants, render=render,
                                    app=qapp)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def gen_common(gen):
    """Just the ``common`` module — most tests need nothing else."""
    return gen.common


@pytest.fixture
def ctx(gen):
    """A dark rendering context.

    ``apply_theme()`` is deliberately never called: see the module
    docstring. Structural assertions (what text a widget shows, how many
    cells a grid has) do not depend on the stylesheet; pixel geometry
    does, and that is measured in the subprocess instead.
    """
    return gen.common.Ctx(gen.app, "dark")


@pytest.fixture
def sandbox(gen, tmp_path, monkeypatch):
    """Point the generator's output paths at ``tmp_path``.

    Every writer in ``render`` resolves its destination through
    ``common.versions_dir()`` / ``common.here()`` at call time, so
    redirecting those two is enough — and it is essential:
    ``_prune_stale_dirs`` deletes directories under ``versions_dir()``,
    which unredirected is the checked-in set of ninety renders.
    """
    versions = tmp_path / "versions"
    here = versions / "_generators"
    here.mkdir(parents=True)
    monkeypatch.setattr(gen.common, "versions_dir", lambda: str(versions))
    monkeypatch.setattr(gen.common, "here", lambda: str(here))
    # Belt and braces: a redirect that silently failed would make the
    # tests below delete the reviewer's renders.
    assert gen.render._audit_path().startswith(str(tmp_path))
    assert str(tmp_path) in gen.render.variant_dir({"n": 1, "slug": "x"})
    return versions


@pytest.fixture
def frozen_home_panels():
    """Undo the product-class monkeypatching variant 01 does.

    ``variants._patch_startup_determinism`` overwrites
    ``spacr.qt.widgets.home.SystemPanel``'s readings and
    ``spacr.run_journal``'s functions *globally* so the baseline render
    is reproducible. That is correct in the standalone generator, which
    exits afterwards, and a leak inside pytest, where the next test to
    build a HomePage would silently be told the GPU is at 41%. Snapshot
    and restore around anything that builds variant 01.
    """
    from spacr.qt.widgets import home as H
    import spacr.run_journal as J
    saved = [
        (H.SystemPanel, "gpu_util", H.SystemPanel.__dict__["gpu_util"]),
        (H.SystemPanel, "gpu_vram", H.SystemPanel.__dict__["gpu_vram"]),
        (H.SystemPanel, "disk_used", H.SystemPanel.__dict__["disk_used"]),
        (H.QueuedPanel, "queue_items",
         H.QueuedPanel.__dict__["queue_items"]),
        (J, "recent_runs", J.recent_runs),
        (J, "journal_totals", J.journal_totals),
    ]
    try:
        yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


# ---------------------------------------------------------------------------
# The categorisations
# ---------------------------------------------------------------------------

def test_every_categorisation_covers_every_app(gen_common):
    """No proposed grouping may drop, duplicate or invent an app key.

    The bands are measured here rather than handed to
    :func:`common.check_coverage` and left at that. ``check_coverage``
    returns ``None``, so a version of it that stopped raising — or one
    whose ``all_keys()`` drifted away from the registry — would leave a
    body of bare calls green while every variant quietly lost an app.
    The last block is the other half of the same worry: the checker
    itself has to reject each of the three defects it names.
    """
    from spacr.qt.app import APPS
    registry = {key for key, *_rest in APPS}
    assert registry, "the app registry is empty"

    tables = [(name, getattr(gen_common, name))
              for name in ("CATS_BROAD3", "CATS_STAGE5", "CATS_NARROW8",
                           "CATS_QUESTIONS", "CATS_INTENT4")]
    tables.append(("cats_current()", gen_common.cats_current()))
    for name, cats in tables:
        placed = [key for _title, keys in cats for key in keys]
        assert len(placed) == len(set(placed)), (
            f"{name} lists a key twice: "
            f"{sorted(k for k in set(placed) if placed.count(k) > 1)}")
        assert set(placed) == registry, (
            f"{name} misses {sorted(registry - set(placed))} and invents "
            f"{sorted(set(placed) - registry)}")
        assert gen_common.check_coverage(cats) is None

    # The checker the generators call at build time has to be able to
    # fail, or none of the above is worth anything to them.
    good = [(title, list(keys)) for title, keys in tables[0][1]]
    victim = sorted(registry)[0]
    dropped = [(title, [k for k in keys if k != victim]) for title, keys in good]
    with pytest.raises(AssertionError, match="not categorised"):
        gen_common.check_coverage(dropped)
    duplicated = [(title, list(keys)) for title, keys in good]
    duplicated[0][1].append(duplicated[0][1][0])
    with pytest.raises(AssertionError, match="duplicate keys"):
        gen_common.check_coverage(duplicated)
    invented = [(title, list(keys)) for title, keys in good]
    invented[0][1].append("no_such_app")
    with pytest.raises(AssertionError, match="unknown keys"):
        gen_common.check_coverage(invented)


def test_orderings_are_permutations_of_the_real_registry(gen_common):
    """Frequency / alphabetical / pinned-first are re-orderings, not edits."""
    from spacr.qt.app import APPS
    keys = set(gen_common.all_keys())
    # Derived, not a literal: the registry grew a Replication Assay and a
    # hardcoded 29 turns "an app was added" into "the variant harness is
    # broken", which is the wrong thing to be told.
    assert keys == {k for k, *_rest in APPS}
    for order in (gen_common.by_frequency(), gen_common.alphabetical(),
                  gen_common.pinned_first()):
        assert len(order) == len(keys)
        assert set(order) == keys
    assert gen_common.pinned_first()[:len(gen_common.PINNED)] == \
        list(gen_common.PINNED)
    counts = [gen_common.USE_COUNTS[k] for k in gen_common.by_frequency()]
    assert counts == sorted(counts, reverse=True)


def test_use_counts_cover_every_app(gen_common):
    """The frequency-ordered variants need a count for every app.

    ``by_frequency`` sorts on ``USE_COUNTS.get(key, 0)``, so a missing
    app does not raise — it silently sinks to the bottom of every
    frequency-ordered variant, which is a wrong screen rather than a
    failed run.
    """
    assert set(gen_common.USE_COUNTS) == set(gen_common.all_keys())
    assert all(count > 0 for count in gen_common.USE_COUNTS.values())


def test_n_apps_is_read_from_the_registry(gen_common):
    """``n_apps()`` must track ``APPS``, not a number typed here."""
    from spacr.qt.app import APPS
    assert gen_common.n_apps() == len(APPS)


def test_the_three_late_apps_are_categorised(gen_common):
    """Regression guard for the gap this file exists to have caught.

    Classifier Evaluation, Distributed Jobs and Run History shipped in
    ``spacr.qt.app.APPS`` and appeared in none of the home generators,
    so every proposed home screen was a screen three apps were missing
    from. ``check_coverage`` catches that generically; this names them,
    because the generic failure sat red without being acted on.
    """
    late = {"classifier_evaluation", "distributed_jobs", "run_history"}
    assert late <= set(gen_common.all_keys())
    assert late <= set(gen_common.USE_COUNTS)
    for name in ("CATS_BROAD3", "CATS_STAGE5", "CATS_NARROW8",
                 "CATS_QUESTIONS", "CATS_INTENT4"):
        placed = {k for _title, keys in getattr(gen_common, name)
                  for k in keys}
        assert late <= placed, f"{name} is missing {sorted(late - placed)}"


def test_no_stage_band_exceeds_the_seven_column_grid_by_more_than_a_row(
        gen_common):
    """``CATS_STAGE5`` bands are drawn seven-wide; a band may wrap ONCE.

    Variants 02 and 23 lay every band out as one seven-column grid, so a
    band of eight takes a second row and a band of fifteen would take
    three. Five bands of seven was thirty-five slots for a registry of
    thirty-four; Illumination, Barcode QC, Layer Viewer and Graph
    Builder took it to thirty-eight, and Power / Design, AnnData Export
    and Run Compare to forty-two.

    Neither way out was ever available. A sixth band is not: variants 13,
    15 and 16 lay these out as exactly five columns and solve the gap
    between them from that count, which is asserted below so the next
    person to reach for it finds out here. A wider grid is not either:
    at eight columns the tile is 166 px, and at that width thirty-four
    of the names elide however small the font is set.

    So the cap is the number of tiles that fits in TWO rows of seven,
    and it was written as ``8`` while eight was the largest band there
    was. Forty-two apps do not go into five bands of eight — nine is the
    arithmetic floor — so it became nine, and Pipeline Graph, Hit List,
    Prediction Profiler and Methods & Results take the registry to
    forty-six, whose floor is TEN. Ten is still one wrapped row (seven,
    then three) and not two. Fourteen is where a third row starts; the
    cap stays below it, so that filling a band remains a decision
    somebody takes rather than a page that quietly gets taller.

    The cap moving is not the same as the bands drifting, which is what
    the floor below is for: the number is always the SMALLEST that can
    hold the registry, so slack cannot accumulate quietly, and raising it
    forces the four new apps to be filed rather than piled into the
    fallback band.

    Curate and Lineage then took the registry to forty-nine and this went
    red at "Report has 12", which is the failure working exactly as it was
    written to: they had been piled into the fallback band rather than
    filed. The cap did NOT move again and nothing left Report. Neither app
    belonged there — fixing a mask by hand is producing a mask, and a
    containment tree is a measurement — so they went to Segment and
    Measure, the two bands that still had room. Forty-nine over five is
    ten, so the floor below is met exactly with the cap where it was. A
    fallback overflow is usually this: not a band that is too small, but a
    key filed nowhere.
    """
    assert len(gen_common.CATS_STAGE5) == 5
    for title, keys in gen_common.CATS_STAGE5:
        assert len(keys) <= 10, (
            f"{title} has {len(keys)} apps, which is more than the one "
            f"wrapped row a seven-column grid may take")
    # ...and the floor is real: any cap below it would be unsatisfiable.
    total = sum(len(keys) for _title, keys in gen_common.CATS_STAGE5)
    assert -(-total // 5) == 10, (
        f"{total} apps over five bands no longer needs a ten-wide band; "
        f"tighten the cap above rather than leaving the slack unused")


def test_check_coverage_names_what_is_wrong(gen_common):
    """Its three failure modes have to be distinguishable in the message."""
    real = gen_common.all_keys()
    with pytest.raises(AssertionError, match="duplicate keys"):
        gen_common.check_coverage([("a", real), ("b", [real[0]])])
    with pytest.raises(AssertionError, match="unknown keys"):
        gen_common.check_coverage([("a", real + ["not_an_app"])])
    with pytest.raises(AssertionError, match="keys not categorised"):
        gen_common.check_coverage([("a", real[:-1])])
    # The whole registry in one bucket is legal — this checks coverage,
    # not whether the grouping is a good one.
    gen_common.check_coverage([("everything", real)])


def test_cats_current_is_the_shipped_grouping(gen_common):
    from spacr.qt.app import SECTIONS
    cats = gen_common.cats_current()
    assert [title for title, _keys in cats] == [s for s in SECTIONS]
    for title, keys in cats:
        for key in keys:
            assert gen_common.app_map()[key][3] == title


def test_paths_resolve_to_the_checkout(gen_common):
    assert os.path.isdir(gen_common.here())
    assert os.path.samefile(gen_common.versions_dir(), VERSIONS)
    assert os.path.isfile(os.path.join(gen_common.repo_root(), "setup.py"))


def test_available_themes_always_offers_dark_and_light(gen_common):
    themes = gen_common.available_themes()
    assert themes[:2] == ("dark", "light")
    assert len(set(themes)) == len(themes)


# ---------------------------------------------------------------------------
# The rendering context
# ---------------------------------------------------------------------------

def test_ctx_caches_icons_and_pixmaps(gen, ctx):
    """The icon pass is a PIL+numpy re-ink per PNG; thirty variants
    cannot each pay for it."""
    key = gen.common.all_keys()[0]
    assert ctx.icon(key) is ctx.icon(key)
    assert ctx.pixmap(key, 32) is ctx.pixmap(key, 32)
    assert ctx.pixmap(key, 32) is not ctx.pixmap(key, 48)
    assert not ctx.pixmap(key, 32).isNull()
    assert ctx.pixmap(key, 32).size().width() == 32


def test_ctx_icon_honours_the_apps_own_override_table(gen, ctx):
    """Same rules as the app: overrides and forced glyphs both resolve."""
    import spacr.qt.app as A
    for key in list(A._ICON_OVERRIDES)[:2]:
        if key in gen.common.app_map():
            assert not ctx.icon(key).isNull(), f"no icon for {key}"


def test_ctx_icon_uses_a_glyph_for_forced_keys(gen, ctx, monkeypatch):
    """``_FORCE_GLYPH`` is empty today, so drive the branch directly.

    An app in that set must get the qtawesome glyph rather than the
    re-inked PNG — the same rule the app applies. Untested, the branch
    would only be discovered the first time a key was added to the set.
    """
    import spacr.qt.app as A
    from PySide6.QtGui import QIcon
    from spacr.qt import iconset
    seen = []

    def fake_icon(key, theme=None):
        seen.append((key, theme))
        return QIcon()

    monkeypatch.setattr(A, "_FORCE_GLYPH", {"mask"})
    monkeypatch.setattr(gen.common, "_ICON_CACHE", {})
    monkeypatch.setattr(iconset, "icon", fake_icon)
    monkeypatch.setattr(iconset, "app_icon",
                        lambda *_a, **_k: pytest.fail(
                            "a forced-glyph key must not go through app_icon"))
    ctx.icon("mask")
    assert seen == [("mask", ctx.theme)]


def test_ctx_logo_is_scaled_and_cached(ctx):
    logo = ctx.logo(84)
    if logo is None:
        pytest.skip("logo_spacr.png not in this checkout")
    assert max(logo.width(), logo.height()) == 84
    assert ctx.logo(84) is logo


def test_ctx_logo_is_none_when_the_wordmark_is_missing(gen, ctx, tmp_path,
                                                       monkeypatch):
    """A checkout without the logo must degrade, not raise: two variants
    draw it and the other twenty-eight must still render."""
    monkeypatch.setattr(gen.common, "repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(gen.common, "_LOGO_CACHE", {})
    assert ctx.logo(84) is None


def test_ctx_logo_is_none_when_the_wordmark_will_not_decode(gen, ctx,
                                                            tmp_path,
                                                            monkeypatch):
    """Present but unreadable is the worse case: ``QPixmap`` returns a
    null pixmap rather than raising, and scaling one gives a 0x0 image
    that lands in the render as an invisible hole."""
    icons = tmp_path / "spacr" / "resources" / "icons"
    icons.mkdir(parents=True)
    (icons / "logo_spacr.png").write_bytes(b"this is not a PNG")
    monkeypatch.setattr(gen.common, "repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(gen.common, "_LOGO_CACHE", {})
    assert ctx.logo(84) is None


def test_available_themes_survives_a_theme_module_that_raises(gen,
                                                              monkeypatch):
    """``space`` is another effort's work in progress; probing for it
    must never be able to take the generator down."""
    import spacr.qt.theme as T
    monkeypatch.setattr(T, "palette_for",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            RuntimeError("no palettes here")))
    assert gen.common.available_themes() == ("dark", "light")


def test_ctx_qss_differs_between_themes(gen, ctx):
    qss = ctx.qss()
    assert "QWidget" in qss or "QPushButton" in qss
    assert gen.common.Ctx(gen.app, "light").qss() != qss


def test_apply_theme_as_a_guest_never_touches_the_application(gen, ctx):
    """The segfault guard: as a guest, styling goes to the target only.

    ``QApplication.setStyleSheet`` re-polishes every top-level widget,
    including ones another test is mid-teardown on. ``common`` is a
    guest here (pytest-qt owns the app), so ``apply_theme`` must leave
    the application's stylesheet exactly as it found it.
    """
    from PySide6.QtWidgets import QWidget
    before = gen.app.styleSheet()
    target = QWidget()
    try:
        ctx.apply_theme(target)
        assert target.styleSheet() == ctx.qss()
        assert gen.app.styleSheet() == before
        ctx.apply_theme()          # no target: must be a complete no-op
        assert gen.app.styleSheet() == before
    finally:
        target.deleteLater()


def test_apply_theme_as_the_owner_styles_the_application(gen, monkeypatch):
    """The other half of the branch, driven against a stand-in.

    The standalone generator owns its QApplication and *does* restyle it
    application-wide. That path cannot be run against the real
    QApplication from inside pytest — restyling it is the segfault the
    guard above exists to prevent — so it is driven against a recorder
    that stands in for the application.
    """
    import spacr.qt.theme as T
    palettes = []
    monkeypatch.setattr(T, "apply_qpalette",
                        lambda app, theme: palettes.append(theme))
    monkeypatch.setattr(gen.common, "_WE_OWN_THE_APP", True)

    class FakeApp:
        sheet = None

        def setStyleSheet(self, qss):      # noqa: N802 (Qt casing)
            self.sheet = qss

    fake = FakeApp()
    ctx = gen.common.Ctx(fake, "dark")
    ctx.apply_theme()
    assert palettes == ["dark"]
    assert fake.sheet == ctx.qss()


# ---------------------------------------------------------------------------
# The parts the variants are assembled from
# ---------------------------------------------------------------------------

def _texts(widget) -> list:
    """Every string a widget tree shows, elided ones unabridged."""
    from PySide6.QtWidgets import QLabel, QPushButton
    from spacr.qt.widgets.eliding import ElidingLabel, ElidingPushButton
    out = []
    for child in widget.findChildren(QLabel):
        out.append(child.full_text() if isinstance(child, ElidingLabel)
                   else child.text())
    for child in widget.findChildren(QPushButton):
        out.append(child.full_text() if isinstance(child, ElidingPushButton)
                   else child.text())
    return [t for t in out if t]


def test_elide_to_lines_respects_its_line_budget(gen):
    from PySide6.QtGui import QFont
    font = QFont()
    font.setPixelSize(12)
    long = ("Register tiles into one stitched canvas, written "
            "incrementally so a 20000x20000 mosaic never has to fit in "
            "RAM, which is the whole point of the thing.")
    one = gen.parts.elide_to_lines(long, font, 120, 1)
    assert one != long and one.endswith("…")
    assert gen.parts.line_count(one, font, 120) == 1
    two = gen.parts.elide_to_lines(long, font, 120, 2)
    assert gen.parts.line_count(two, font, 120) == 2
    assert len(two) > len(one)
    # Text that already fits comes back untouched, ellipsis-free.
    assert gen.parts.elide_to_lines("Mask", font, 120, 2) == "Mask"


def test_elide_to_lines_degenerate_inputs_are_pass_through(gen):
    """Zero width / zero lines / empty text must not raise or elide."""
    from PySide6.QtGui import QFont
    font = QFont()
    font.setPixelSize(12)
    assert gen.parts.elide_to_lines("Measure", font, 0, 2) == "Measure"
    assert gen.parts.elide_to_lines("Measure", font, 120, 0) == "Measure"
    assert gen.parts.elide_to_lines("", font, 120, 2) == ""
    assert gen.parts.line_count("", font, 120) == 1
    assert gen.parts.line_count("Measure", font, 0) == 1


def test_wrapped_reserves_its_budget_only_when_asked(gen, ctx):
    short = "One line."
    loose = gen.parts.wrapped(ctx, short, 260, 3)
    tight = gen.parts.wrapped(ctx, short, 260, 3, reserve=True)
    assert loose.height() < tight.height(), \
        "without reserve=True a one-line string must not hold three lines"
    assert loose.text() == short


def test_text_label_upper_and_colour_reach_the_widget(gen, ctx):
    lbl = gen.parts.text_label(ctx, "stages", size=11, upper=True,
                               color="#ff0000", tracking="2px")
    assert lbl.text() == "STAGES"
    assert "#ff0000" in lbl.styleSheet()
    assert "letter-spacing: 2px" in lbl.styleSheet()
    assert "font-size: 11px" in lbl.styleSheet()


def test_htile_shows_the_real_name_and_blurb(gen, ctx):
    """The tile is the app's own ``HTile``, carrying registry text."""
    tile = gen.parts.htile(ctx, "classifier_evaluation", width=260,
                           icon_px=40)
    assert gen.common.name_of("classifier_evaluation") in _texts(tile)
    assert gen.common.blurb_of("classifier_evaluation") in tile.toolTip()
    assert tile.width() == 260
    assert tile.height() == gen.parts.htile_height(40)


def test_htile_grid_lays_keys_out_in_rows_of_cols(gen, ctx):
    keys = gen.common.all_keys()[:7]
    grid = gen.parts.htile_grid(ctx, keys, cols=3, width=200, icon_px=32)
    layout = grid.layout()
    assert layout.count() == len(keys)
    positions = [layout.getItemPosition(i)[:2] for i in range(layout.count())]
    assert positions[:4] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert positions[-1] == (2, 0)


def test_big_tile_grid_draws_names_blurbs_and_badges(gen, ctx):
    keys = ["mask", "measure"]
    grid = gen.parts.big_tile_grid(ctx, keys, cols=2, width=280, height=170,
                                   blurb_lines=2,
                                   badges={"mask": "412 runs"})
    shown = _texts(grid)
    assert gen.common.name_of("mask") in shown
    assert gen.common.name_of("measure") in shown
    assert "412 runs" in shown


def test_dense_row_carries_name_blurb_badge_and_shortcut(gen, ctx):
    row = gen.parts.DenseRow(ctx, "mask", width=520, badge="412",
                             shortcut="Ctrl+1")
    shown = _texts(row)
    assert gen.common.name_of("mask") in shown
    assert "412" in shown
    assert "Ctrl+1" in shown
    assert gen.common.blurb_of("mask") in row.toolTip()


def test_dense_row_without_blurb_room_drops_the_blurb(gen, ctx):
    """A row too narrow for a description must omit it, not clip it."""
    narrow = gen.parts.DenseRow(ctx, "mask", width=200, name_width=136)
    wide = gen.parts.DenseRow(ctx, "mask", width=560, name_width=136)
    assert len(_texts(narrow)) < len(_texts(wide))


def test_dense_list_is_one_row_per_key(gen, ctx):
    keys = gen.common.CATS_NARROW8[0][1]
    lst = gen.parts.dense_list(ctx, keys, width=420)
    rows = lst.findChildren(gen.parts.DenseRow)
    assert len(rows) == len(keys)


def test_fixed_button_survives_a_restyle(gen):
    """The trap that squashed two variants: the app QSS carries
    ``QPushButton { min-height: 22px }`` and ``QStyleSheetStyle``
    re-applies it over ``setFixedSize`` on polish. Reporting the size
    through ``sizeHint`` is what survives."""
    from PySide6.QtCore import QSize
    button = gen.parts.FixedButton(116, 96)
    button.setStyleSheet("QPushButton { min-height: 22px; }")
    button.ensurePolished()
    assert button.sizeHint() == QSize(116, 96)
    assert button.minimumSizeHint() == QSize(116, 96)


def test_page_assembles_chrome_rail_aside_and_footer(gen, ctx):
    from PySide6.QtWidgets import QLabel, QWidget
    page = gen.parts.Page(ctx, margins=(10, 10, 10, 10))
    rail, aside = QWidget(), QWidget()
    page.add_rail(rail)
    page.add_aside(aside)
    page.body.addWidget(QLabel("body content"))
    same = page.finish(footer=gen.parts.hint_bar(ctx), status="Rendering")
    assert same is page
    shown = _texts(page)
    assert "body content" in shown
    assert "Rendering" in shown, "finish(status=) must reach the status bar"
    assert any("spaCR" in t for t in shown), "menu strip missing"
    assert page.middle.itemAt(0).widget() is rail
    assert page.middle.itemAt(page.middle.count() - 1).widget() is aside


def test_page_without_chrome_has_no_status_bar(gen, ctx):
    page = gen.parts.Page(ctx, chrome=False).finish(status="Ready")
    assert "Ready" not in _texts(page)


@pytest.mark.parametrize("builder", [
    "resume_banner", "recent_runs_strip", "recent_runs_list",
    "project_status_strip", "system_panel", "whats_new_panel",
    "quick_start", "start_run_panel", "pinned_row", "queue_panel",
])
def test_every_proposed_panel_builds_and_shows_content(gen, ctx, builder):
    """The panels spaCR does not have yet are real widgets, not mockups.

    Each must render text — a panel that builds but shows nothing is a
    painted screenshot in disguise, which is the thing this whole
    exercise refuses to produce.
    """
    widget = getattr(gen.parts, builder)(ctx)
    shown = _texts(widget)
    assert shown, f"{builder} rendered no text at all"
    assert any(len(t) > 3 for t in shown)


def test_panels_quote_the_mock_and_never_live_state(gen, ctx):
    """Fixed literals only: a home screen that renders differently every
    run cannot be reviewed."""
    MOCK = gen.common.MOCK
    assert MOCK["project"] in _texts(gen.parts.project_status_strip(ctx))
    queue_text = " ".join(_texts(gen.parts.queue_panel(ctx)))
    for plate, pipeline, state in MOCK["queue"]:
        assert plate in queue_text and pipeline in queue_text
        assert state in queue_text
    recent = " ".join(_texts(gen.parts.recent_runs_list(ctx)))
    for key, plate, when, _ok, _elapsed in MOCK["recent"]:
        assert gen.common.name_of(key) in recent
        assert plate in recent and when in recent
    resume_key, resume_plate, _when = MOCK["last_run"]
    banner = " ".join(_texts(gen.parts.resume_banner(ctx, width=900)))
    assert gen.common.name_of(resume_key) in banner
    assert resume_plate in banner


def test_stat_row_and_headers_render_their_arguments(gen, ctx):
    stats = gen.parts.stat_row(ctx, [("148", "runs"), ("12", "plates")])
    shown = _texts(stats)
    assert "148" in shown and "RUNS" in shown       # upper=True
    assert "12" in shown and "PLATES" in shown
    # cat_header sets the heading in caps; plain_header does not.
    head = _texts(gen.parts.cat_header(ctx, "Segment", note="7 apps"))
    assert "SEGMENT" in head and "7 apps" in head
    plain = _texts(gen.parts.plain_header(ctx, "Measure", "7 apps"))
    assert "Measure" in plain and "7 apps" in plain


def test_top_bar_and_hero_show_their_subtitle(gen, ctx):
    bar = _texts(gen.parts.top_bar(ctx, subtitle="34 apps",
                                   actions=(("Search…", False),)))
    assert "34 apps" in bar and "Search…" in bar
    assert _texts(gen.parts.hero(ctx))
    assert _texts(gen.parts.hero(ctx, compact=True))


def test_cat_rail_marks_the_selected_category(gen, ctx):
    from PySide6.QtWidgets import QListWidget
    titles = [t for t, _keys in gen.common.CATS_STAGE5]
    counts = [len(k) for _t, k in gen.common.CATS_STAGE5]
    rail = gen.parts.cat_rail(ctx, titles, selected=2, header="Stages",
                              counts=counts)
    lst = rail.findChild(QListWidget)
    assert lst is not None
    assert lst.count() == len(titles)
    assert lst.currentRow() == 2
    assert titles[2] in lst.item(2).text()


def test_search_box_and_chip_and_kbd_carry_their_text(gen, ctx):
    box = gen.parts.search_box(ctx, "Search 34 apps")
    assert box.placeholderText() == "Search 34 apps"
    on = gen.parts.chip(ctx, "All", on=True)
    off = gen.parts.chip(ctx, "Models")
    assert on.text() == "All" and off.text() == "Models"
    # Selection is carried by objectName so the page stylesheet can
    # colour it; a chip that looked identical either way is not a chip.
    assert on.objectName() != off.objectName()
    assert gen.parts.kbd(ctx, "Ctrl+K").text() == "Ctrl+K"


def test_scroll_area_scrolls_on_exactly_one_axis(gen, ctx):
    """Both axes free is how a variant hides content in two directions
    at once, which no reviewer would spot in a screenshot."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
    vertical = gen.parts.scroll_area(QLabel("x" * 200))
    assert vertical.widget() is not None
    assert vertical.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert vertical.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    horizontal = gen.parts.scroll_area(QLabel("y" * 200), horizontal=True)
    assert horizontal.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert horizontal.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded


def test_htile_min_width_table_still_holds(gen, ctx):
    """The five-column constraint, re-measured rather than trusted.

    ``HTILE_MIN_WIDTH`` records how narrow a tile may be before the
    longest app name elides. If the registry gains a longer name the
    table is stale, and every variant built on it starts eliding
    silently — which is exactly what "Classifier Evaluation" did.
    """
    # Pin zoom to 1.0 for this measurement. The test builds widgets at
    # EXPLICIT pixel sizes and asks whether the text fits; the zoom preference
    # scales the stylesheet's fonts on top of that, so at the 150% default the
    # label renders half again larger inside a box pinned at its unscaled
    # size. The product is fine — a real HomePage at 150% has zero elided
    # labels, because its tile widths go through `scaled_px`.
    #
    # Done inline rather than as an autouse fixture: the variant generator
    # copies this module into a standalone script and runs it without pytest,
    # so a module-level `@pytest.fixture` becomes a NameError there.
    from spacr.qt import preferences as _prefs
    _original_zoom = _prefs.get_font_scale()
    _prefs.set_font_scale(1.0)
    try:
        from spacr.qt.widgets.eliding import ElidingLabel
        longest = max(gen.common.all_keys(),
                      key=lambda k: len(gen.common.name_of(k)))
        for name_px, by_icon in gen.parts.HTILE_MIN_WIDTH.items():
            if not name_px:
                continue           # 0 means "the shipped 17 px subtitle size"
            for icon_px, width in by_icon.items():
                tile = gen.parts.htile(ctx, longest, width=width,
                                       icon_px=icon_px, name_px=name_px)
                tile.show()
                try:
                    gen.app.processEvents()
                    for label in tile.findChildren(ElidingLabel):
                        assert not label.is_elided(), (
                            f"{gen.common.name_of(longest)!r} elides at the "
                            f"recorded minimum {width} px "
                            f"(name {name_px} px, icon {icon_px} px)")
                finally:
                    tile.hide()
                    tile.deleteLater()


    # ---------------------------------------------------------------------------
    # The audit
    # ---------------------------------------------------------------------------

    finally:
        _prefs.set_font_scale(_original_zoom)
def _shown(widget, app):
    widget.resize(*CANVAS)
    widget.show()
    for _ in range(3):
        app.processEvents()
    return widget


def _drop(widget, app):
    from PySide6.QtCore import QEvent
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()
    # processEvents() does not drain DeferredDelete, and thirty pages of
    # a few thousand widgets each add up fast.
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_audit_reports_a_clean_page_clean(gen, ctx):
    """Including the two kinds of label that carry no measurable text.

    An empty spacer label reports a sizeHint bigger than the zero width
    it was given, and a hidden label has no geometry at all; flagging
    either fills the report with noise that hides the real findings.
    """
    from PySide6.QtWidgets import QLabel
    page = gen.parts.Page(ctx)
    page.body.addWidget(QLabel("plenty of room"))
    blank = QLabel("")
    blank.setFixedWidth(0)
    page.body.addWidget(blank)
    hidden = QLabel("a very long string in a box of no width at all")
    hidden.setFixedWidth(2)
    page.body.addWidget(hidden)
    hidden.hide()
    page.body.addStretch(1)
    page.finish()
    _shown(page, gen.app)
    try:
        assert gen.render.audit(page) == {"elided": [], "clipped": [],
                                          "scrollbars": [], "overflow": []}
    finally:
        _drop(page, gen.app)


def test_audit_catches_clipped_elided_and_overflowing_text(gen, ctx):
    from PySide6.QtWidgets import QLabel
    from spacr.qt.widgets.eliding import ElidingLabel
    page = gen.parts.Page(ctx)
    clipped = QLabel("a name far too long for the box it was given")
    clipped.setFixedWidth(20)
    page.body.addWidget(clipped)
    elided = ElidingLabel("Annotator Agreement and then some more text")
    elided.setFixedWidth(30)
    page.body.addWidget(elided)
    from spacr.qt.widgets.eliding import ElidingPushButton
    button = ElidingPushButton("Distributed Jobs, a name for a wide button")
    button.setFixedWidth(30)
    page.body.addWidget(button)
    tall = QLabel("tall")
    tall.setFixedHeight(CANVAS[1] + 200)
    page.body.addWidget(tall)
    page.finish()
    _shown(page, gen.app)
    try:
        report = gen.render.audit(page)
        assert any("far too long" in entry for entry in report["clipped"])
        assert any("Annotator Agreement" in entry
                   for entry in report["elided"])
        assert any("Distributed Jobs" in entry
                   for entry in report["elided"]), \
            "an elided button is as unreadable as an elided label"
        assert report["overflow"] and "px of height" in report["overflow"][0]
    finally:
        _drop(page, gen.app)


def test_audit_reports_a_layout_wider_than_the_canvas(gen, ctx):
    from PySide6.QtWidgets import QLabel
    page = gen.parts.Page(ctx, chrome=False)
    wide = QLabel("wide")
    wide.setFixedWidth(CANVAS[0] + 400)
    page.body.addWidget(wide)
    page.finish()
    _shown(page, gen.app)
    try:
        assert any("px of width" in entry
                   for entry in gen.render.audit(page)["overflow"])
    finally:
        _drop(page, gen.app)


def test_audit_ignores_wrapped_and_pixmap_labels(gen, ctx):
    """A word-wrapped label is *meant* to be narrower than its text, and
    an icon label has no text at all. Flagging either buries the real
    findings."""
    from PySide6.QtWidgets import QLabel
    page = gen.parts.Page(ctx)
    page.body.addWidget(gen.parts.wrapped(ctx, "a long sentence that has "
                                          "to wrap over several lines to "
                                          "fit", 120, 3))
    icon = QLabel()
    icon.setPixmap(ctx.pixmap("mask", 32))
    icon.setFixedWidth(4)
    page.body.addWidget(icon)
    page.body.addStretch(1)
    page.finish()
    _shown(page, gen.app)
    try:
        assert not gen.render.audit(page)["clipped"]
    finally:
        _drop(page, gen.app)


def test_audit_reports_a_visible_scrollbar(gen, ctx):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
    page = gen.parts.Page(ctx)
    inner = QLabel("\n".join(["row"] * 400))
    area = gen.parts.scroll_area(inner)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    page.body.addWidget(area)
    page.finish()
    _shown(page, gen.app)
    try:
        assert gen.render.audit(page)["scrollbars"]
    finally:
        _drop(page, gen.app)


def test_audit_tolerates_a_widget_with_no_layout(gen):
    from PySide6.QtWidgets import QWidget
    bare = QWidget()
    try:
        assert gen.render.audit(bare)["overflow"] == []
    finally:
        bare.deleteLater()


# ---------------------------------------------------------------------------
# The audit cache and VARIANTS.md
# ---------------------------------------------------------------------------

def test_audit_cache_round_trips_through_json(gen, sandbox):
    reports = {(1, "dark"): {"elided": ["Annotator Agreement"],
                             "clipped": [], "scrollbars": [], "overflow": []},
               (30, "light"): {"elided": [], "clipped": [],
                               "scrollbars": ["Vertical"], "overflow": []}}
    gen.render.save_audit(reports)
    assert gen.render.load_audit() == reports, \
        "the (int, str) key has to survive a JSON round trip"
    with open(gen.render._audit_path(), encoding="utf-8") as fh:
        assert set(json.load(fh)) == {"1|dark", "30|light"}


def test_load_audit_returns_empty_for_missing_and_corrupt_files(gen, sandbox):
    assert gen.render.load_audit() == {}
    with open(gen.render._audit_path(), "w", encoding="utf-8") as fh:
        fh.write("{not json at all")
    assert gen.render.load_audit() == {}, \
        "a corrupt cache must degrade to 'not re-rendered', not crash"


def test_audit_sentence_distinguishes_its_four_cases(gen):
    spec = {"n": 7}
    clean = {"elided": [], "clipped": [], "scrollbars": [], "overflow": []}
    dirty = dict(clean, elided=["a", "b"])
    themes = ("dark", "light")
    assert gen.render._audit_sentence(spec, themes, {}) == \
        "not re-rendered in this pass."
    assert gen.render._audit_sentence(
        spec, themes, {(7, "dark"): clean, (7, "light"): clean}) == \
        gen.render._CLEAN
    both = gen.render._audit_sentence(
        spec, themes, {(7, "dark"): dirty, (7, "light"): dirty})
    assert both.startswith("every theme") and "elided (2)" in both
    split = gen.render._audit_sentence(
        spec, themes, {(7, "dark"): dirty, (7, "light"): clean})
    assert "dark:" in split and "light: clean" in split


#: Number words the prose actually reaches for, so a count spelled out
#: is checked as hard as one written in digits. The first sweep that
#: derived the app counts left "the whole 29-app taxonomy" and
#: "Twenty-seven of the thirty variants" standing, because the guard
#: only looked for ``<digits><space><noun>``.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty-one": 21, "twenty-two": 22, "twenty-three": 23,
    "twenty-four": 24, "twenty-five": 25, "twenty-six": 26,
    "twenty-seven": 27, "twenty-eight": 28, "twenty-nine": 29,
    "thirty": 30, "thirty-one": 31, "thirty-two": 32, "thirty-three": 33,
    "thirty-four": 34, "thirty-five": 35, "thirty-six": 36,
    "thirty-seven": 37, "thirty-eight": 38, "thirty-nine": 39, "forty": 40,
}

#: ``<count><separator><noun>``. The separator is a space *or a hyphen*
#: ("29-app taxonomy"), and the count may be digits or a number word.
_COUNT_RE = re.compile(
    r"\b(\d+|[A-Za-z]+(?:-(?:one|two|three|four|five|six|seven|eight|nine))?)"
    r"[\s-](apps?|app rows|items|tiles|variants)\b")


def _counts_in(text: str):
    """``{noun: {numbers the prose puts in front of it}}``.

    Words that are not numbers ("the apps", "five-app") are skipped —
    only a token that resolves to an integer is a claim about a count.
    """
    found = {}
    for raw, noun in _COUNT_RE.findall(text):
        value = (int(raw) if raw.isdigit()
                 else _NUMBER_WORDS.get(raw.lower()))
        if value is None:
            continue
        found.setdefault("variants" if noun == "variants" else "apps",
                         set()).add(value)
    return found


def test_write_markdown_never_types_an_app_count(gen, sandbox):
    """Every "N apps" in the document must be the live registry size.

    This is the regression that let three shipped apps go missing: the
    prose said "29 apps" while ``APPS`` held 34, and nothing compared
    the two.

    The guard reads spelled-out and hyphenated counts too. Its first
    version did not, and "the whole 29-app taxonomy collapses into five
    closed accordion rows" survived the sweep that was supposed to have
    removed exactly that sentence.
    """
    whole = gen.common.n_apps()
    # Every number the prose is allowed to put in front of "apps",
    # each one derived from the registry rather than agreed with it:
    allowed = {
        whole,                                       # "all N apps"
        whole - len(gen.common.core_keys()),         # v18's "More tools (N)"
        whole - len(gen.common.PINNED),              # v12's "everything else"
        whole - 8,                                   # v06 shows eight
    }
    # A variant is also allowed to state the size of one of its own
    # bands ("Nine apps, and a door to the other 25", "two apps in the
    # Measure column"). Those are facts about a category table, so the
    # table is what says whether they are still true; nothing here has
    # to agree with a literal.
    for table in (gen.common.CATS_STAGE5, gen.common.CATS_BROAD3,
                  gen.common.CATS_NARROW8, gen.common.CATS_QUESTIONS,
                  gen.common.CATS_INTENT4):
        allowed |= {len(keys) for _title, keys in table}
    allowed.add(len(gen.common.PINNED))
    # The point of the set is that every member of it is DERIVED from the
    # live registry, so none of them can drift the way a typed literal
    # did. This used to be spelled `29 not in allowed` -- the exact stale
    # number the file was written about. That sentinel had to go, and its
    # going is the same lesson twice: with 38 apps and 9 Core ones, "the
    # other 29" is a true sentence, so the guard was itself pinned to a
    # registry size.
    assert whole in allowed
    assert allowed and all(0 < n <= whole for n in allowed), (
        f"a number the prose may print is not a size anything has: "
        f"{sorted(allowed)} against {whole} apps")
    path = gen.render.write_markdown(gen.variants.VARIANTS,
                                     ("dark", "light"),
                                     gen.render.load_audit(), 1356, 850)
    text = open(path, encoding="utf-8").read()
    counted = _counts_in(text).get("apps", set())
    assert counted, "no app count in the document at all — check the regex"
    assert whole in counted, \
        "nothing in the document states the real number of apps"
    stale = sorted(counted - allowed)
    assert not stale, (
        f"app counts in VARIANTS.md that the registry cannot produce: "
        f"{stale}. Either the prose typed a number, or it derived a new "
        f"one that belongs in `allowed` above.")


def test_write_markdown_never_types_a_variant_count(gen, sandbox):
    """Same rule for "N variants", which is where the sweep went wrong.

    ``render.py`` finding 2 was rewritten to stop hardcoding the app
    count and typed "Twenty-seven of the thirty variants below fit
    1440x900 with no scrollbar at all" into the replacement — a number
    no test could contradict, in the paragraph whose whole purpose is to
    report what the audit measured.
    """
    total = len(gen.variants.VARIANTS)
    reports = {(spec["n"], "dark"): {} for spec in gen.variants.VARIANTS}
    for number in (1, 25, 30):
        reports[(number, "dark")] = {"scrollbars": ["QScrollArea"]}
    path = gen.render.write_markdown(gen.variants.VARIANTS, ("dark",),
                                     reports, 1356, 850)
    text = open(path, encoding="utf-8").read()
    counted = _counts_in(text).get("variants", set())
    assert counted, "no variant count in the document — check the regex"
    allowed = {total, total - 3, 3}
    stale = sorted(counted - allowed)
    assert not stale, (
        f"variant counts in VARIANTS.md the audit cannot produce: {stale}")
    assert f"{total - 3} of the {total} variants" in text


def test_the_scrollbar_finding_is_counted_from_the_audit(gen, sandbox):
    """Change what the audit says and the sentence must change with it."""
    specs = gen.variants.VARIANTS
    total = len(specs)

    clean = {(spec["n"], "dark"): {} for spec in specs}
    text = _md_with(gen, specs, clean)
    assert f"All {total} variants below fit 1440x900" in text

    one_bad = dict(clean)
    one_bad[(4, "dark")] = {"scrollbars": ["QScrollArea"]}
    text = _md_with(gen, specs, one_bad)
    assert f"{total - 1} of the {total} variants below fit" in text
    assert "04 (" in text, "the sentence must name which variant scrolls"

    # A `--only` run measured almost nothing: it must say so rather than
    # divide by a total it never looked at.
    partial = {(4, "dark"): {"scrollbars": ["QScrollArea"]}}
    text = _md_with(gen, specs, partial)
    assert f"only audited 1 of the {total} variants" in text
    assert f"{total - 1} of the {total} variants below fit" not in text


def _md_with(gen, specs, reports):
    path = gen.render.write_markdown(specs, ("dark",), reports, 1356, 850)
    return open(path, encoding="utf-8").read()


def test_write_markdown_documents_every_variant(gen, sandbox):
    path = gen.render.write_markdown(gen.variants.VARIANTS, ("dark",),
                                     gen.render.load_audit(), 1356, 850)
    text = open(path, encoding="utf-8").read()
    for spec in gen.variants.VARIANTS:
        assert f"### {spec['n']:02d} · {spec['title']}" in text
        assert spec["changes"] in text
        assert spec["argument"] in text
    assert "the `space` palette was not available" in text, \
        "a themes tuple without space has to say so"
    assert "min-height: 22px" in text, "the outro's QSS trap note was dropped"
    # The proposed sidebar diff names the apps that fall off the bottom;
    # they are read from the registry, not typed.
    for key in gen.common.all_keys()[-3:]:
        assert gen.common.name_of(key) in text


def test_write_markdown_notes_the_space_theme_when_it_rendered(gen, sandbox):
    path = gen.render.write_markdown(gen.variants.VARIANTS,
                                     ("dark", "light", "space"),
                                     {}, 1356, 850)
    assert ", plus `space.png`" in open(path, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# Sheet, self-check and housekeeping
# ---------------------------------------------------------------------------

def _fake_png(path, size=CANVAS, noisy=True):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if noisy:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    else:
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_self_check_separates_good_missing_and_blank_renders(gen, sandbox,
                                                             capsys):
    Image = pytest.importorskip("PIL.Image")
    specs = [{"n": 1, "slug": "good"}, {"n": 2, "slug": "blank"},
             {"n": 3, "slug": "gone"}]
    _fake_png(os.path.join(gen.render.variant_dir(specs[0]), "dark.png"))
    _fake_png(os.path.join(gen.render.variant_dir(specs[1]), "dark.png"),
              noisy=False)
    result = gen.render.self_check(specs, ("dark",))
    by_dir = {os.path.basename(os.path.dirname(r["path"])): r
              for r in result["rows"]}
    assert by_dir["v01_good"]["ok"]
    assert by_dir["v01_good"]["unique"] > 64
    assert not by_dir["v02_blank"]["ok"]
    assert by_dir["v02_blank"]["why"] == "blank or wrong size"
    assert by_dir["v03_gone"]["why"] == "missing"
    assert result["sheet_ok"] is False
    gen.render._report_check(result)          # must not raise on failures
    assert "FAIL" in capsys.readouterr().out
    # A noisy but wrong-sized render is caught too.
    Image.new("RGB", (100, 100), "white").save(
        os.path.join(gen.render.variant_dir(specs[0]), "dark.png"))
    assert not gen.render.self_check(specs[:1], ("dark",))["rows"][0]["ok"]


def test_build_sheet_writes_one_numbered_grid(gen, sandbox):
    Image = pytest.importorskip("PIL.Image")
    specs = [{"n": n, "slug": f"s{n}", "title": f"Variant {n}"}
             for n in range(1, 7)]
    for spec in specs[:3]:
        _fake_png(os.path.join(gen.render.variant_dir(spec), "dark.png"))
    out = gen.render.build_sheet(specs, cols=3, thumb_w=120)
    assert os.path.isfile(out)
    with Image.open(out) as sheet:
        # 3 columns of 120 px thumbs, 4 gutters of 16; two rows deep.
        assert sheet.size[0] == 3 * 120 + 4 * 16
        assert sheet.size[1] > 2 * int(round(120 * CANVAS[1] / CANVAS[0]))
    assert gen.render._sheet_font(15) is not None


def test_prune_stale_dirs_removes_only_unregistered_variants(gen, sandbox):
    specs = [{"n": 1, "slug": "keep"}]
    keep = gen.render.variant_dir(specs[0])
    os.makedirs(keep)
    ghost = os.path.join(str(sandbox), "v99_ghost")
    os.makedirs(ghost)
    private = os.path.join(str(sandbox), "_generators")
    loose = os.path.join(str(sandbox), "notes.md")
    open(loose, "w").close()
    gen.render._prune_stale_dirs(specs)
    assert os.path.isdir(keep)
    assert not os.path.exists(ghost)
    assert os.path.isdir(private), "underscore folders are not variants"
    assert os.path.isfile(loose)


def test_prune_stale_dirs_is_a_no_op_without_a_versions_dir(gen, tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """A missing versions dir means *nothing happens* — not "mkdir it".

    The second half is the control: the identical call against a root
    that does exist deletes ``v01_ghost``, so the silence above is a
    property of the missing directory and not of a pruner that never
    prunes.
    """
    missing = tmp_path / "nope"
    ghost = tmp_path / "v01_ghost"
    ghost.mkdir()

    monkeypatch.setattr(gen.common, "versions_dir", lambda: str(missing))
    assert gen.render._prune_stale_dirs([]) is None
    assert not missing.exists(), "the pruner created the directory it skipped"
    assert [p.name for p in tmp_path.iterdir()] == ["v01_ghost"]
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(gen.common, "versions_dir", lambda: str(tmp_path))
    gen.render._prune_stale_dirs([])
    assert not ghost.exists()
    assert "pruned stale v01_ghost" in capsys.readouterr().out


def test_variant_dir_is_the_numbered_slug_folder(gen):
    path = gen.render.variant_dir({"n": 4, "slug": "eight-narrow"})
    assert os.path.basename(path) == "v04_eight-narrow"
    assert os.path.samefile(os.path.dirname(path), VERSIONS)


def test_measure_sidebar_measures_the_scrolled_content(gen):
    """Finding 1, re-measured every run instead of quoted.

    The number has to come from the rows *inside* the scroll area. The
    ``QScrollArea`` these renders argued for landed in
    ``spacr.qt.app.Sidebar``, which makes the widget's own minimum about
    85 px — a title over a collapsible viewport — so measuring the
    outer layout would report "the sidebar fits in 85 px" and quietly
    turn finding 1 into nonsense.
    """
    need, avail = gen.render.measure_sidebar(gen.app)
    assert avail == CANVAS[1] - 26 - 24
    from spacr.qt.app import APPS
    # One row per app plus five headings plus Home, at 20 px a row, is a
    # deliberately generous floor: the point is that it is nothing like
    # the ~85 px the outer layout reports.
    assert need > 20 * len(APPS), (
        f"measure_sidebar reports {need} px for {len(APPS)} app rows — it "
        "is measuring the viewport, not the rows inside it")
    assert need > avail, (
        "the sidebar's rows now fit without scrolling — finding 1 in "
        "VARIANTS.md is stale")


def test_main_check_and_md_only_never_render(gen, sandbox, capsys):
    """The two cheap CLI paths. Neither may write a PNG."""
    assert gen.render.main(["--check", "--themes", "dark"]) == 0
    assert "self-check" in capsys.readouterr().out
    assert gen.render.main(["--md-only", "--themes", "dark"]) == 0
    assert os.path.isfile(os.path.join(str(sandbox), "VARIANTS.md"))
    assert not [name
                for _root, _dirs, files in os.walk(str(sandbox))
                for name in files if name.endswith(".png")]


def test_render_one_writes_a_1440x900_png_and_its_audit(gen, ctx, sandbox):
    """The single-render path, on the cheapest variant in the set."""
    Image = pytest.importorskip("PIL.Image")
    spec = gen.variants.VARIANTS[17]          # 18 core-nine-only
    out = os.path.join(gen.render.variant_dir(spec), "dark.png")
    report = gen.render.render_one(gen.app, spec, "dark", out)
    assert set(report) == {"elided", "clipped", "scrollbars", "overflow"}
    with Image.open(out) as im:
        assert im.size == CANVAS
    assert os.path.getsize(out) > 1000, "a PNG that small drew nothing"


def test_render_one_raises_rather_than_losing_a_render(gen, ctx, sandbox,
                                                       monkeypatch):
    """A save that fails must stop the run, not leave a hole in the set.

    ``QPixmap.save`` returns False instead of raising, so without the
    explicit check a full render would report thirty successes and write
    twenty-nine files.
    """
    from PySide6.QtGui import QPixmap
    monkeypatch.setattr(QPixmap, "save", lambda *_a, **_k: False)
    spec = gen.variants.VARIANTS[17]
    with pytest.raises(RuntimeError, match="could not write"):
        gen.render.render_one(gen.app, spec, "dark",
                              os.path.join(gen.render.variant_dir(spec),
                                           "dark.png"))


def test_main_renders_only_what_it_was_asked_for(gen, sandbox, capsys):
    """The full pipeline — render, sheet, markdown, self-check — over
    one variant in one theme, so the expensive path is still exercised."""
    assert gen.render.main(["--only", "18", "--themes", "dark"]) == 0
    out = capsys.readouterr().out
    assert "variants: 1 of 30" in out
    assert "v18 dark" in out
    written = {os.path.relpath(os.path.join(root, name), str(sandbox))
               for root, _dirs, files in os.walk(str(sandbox))
               for name in files if name.endswith(".png")}
    assert written == {os.path.join("v18_core-nine-only", "dark.png"),
                       "_sheet.png"}
    assert os.path.isfile(os.path.join(str(sandbox), "VARIANTS.md"))
    # A partial run keeps every variant's prose but only re-audits the
    # one it rendered.
    assert gen.render.load_audit().keys() == {(18, "dark")}
    text = open(os.path.join(str(sandbox), "VARIANTS.md"),
                encoding="utf-8").read()
    assert "not re-rendered in this pass." in text


def test_patch_startup_determinism_survives_a_missing_run_journal(
        gen, frozen_home_panels, monkeypatch):
    """The baseline must still render when the journal cannot be imported.

    Its GPU/disk readings are the ones that matter for reproducibility;
    an absent ``spacr.run_journal`` is a reason to draw "no runs yet",
    not to lose variant 01 entirely — which is exactly what an import
    outside the ``try`` did when ``spacr.qt.screens.startup`` was
    deleted.
    """
    from spacr.qt.widgets import home as H
    monkeypatch.setitem(sys.modules, "spacr.run_journal", None)
    gen.variants._patch_startup_determinism()
    assert H.SystemPanel.gpu_util() == "41%"
    assert H.SystemPanel.disk_used() == "68%"


# ---------------------------------------------------------------------------
# The rendered artefacts already on disk
# ---------------------------------------------------------------------------

def _variant_dirs():
    if not os.path.isdir(VERSIONS):
        return []
    return sorted(
        os.path.join(VERSIONS, n) for n in os.listdir(VERSIONS)
        if n.startswith("v") and os.path.isdir(os.path.join(VERSIONS, n)))


def test_thirty_variant_folders_exist():
    dirs = _variant_dirs()
    if not dirs:
        pytest.skip("variants have not been rendered")
    assert len(dirs) == N_VARIANTS
    numbers = sorted(int(os.path.basename(d)[1:3]) for d in dirs)
    assert numbers == list(range(1, N_VARIANTS + 1))


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_png_is_the_right_size_and_not_blank(theme):
    """A solid-colour render means the widget never laid out."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    dirs = _variant_dirs()
    if not dirs:
        pytest.skip("variants have not been rendered")
    for folder in dirs:
        path = os.path.join(folder, f"{theme}.png")
        assert os.path.isfile(path), f"missing {path}"
        with Image.open(path) as im:
            assert im.size == CANVAS, f"{path} is {im.size}, want {CANVAS}"
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        assert float(arr.std()) > 3.0, f"{path} is near-uniform"
        # Pack RGB into one int32 before counting, the way
        # render.self_check does: np.unique(axis=0) over 1.3 M rows takes
        # seconds per image, and there are sixty images — it was thirty
        # seconds per theme of this file's runtime on its own.
        packed = ((arr[..., 0].astype(np.int32) << 16)
                  | (arr[..., 1].astype(np.int32) << 8)
                  | arr[..., 2].astype(np.int32))
        assert int(np.unique(packed).size) > 64, \
            f"{path} has almost no distinct colours"


def test_contact_sheet_and_markdown_exist():
    if not _variant_dirs():
        pytest.skip("variants have not been rendered")
    assert os.path.isfile(os.path.join(VERSIONS, "_sheet.png"))
    md = os.path.join(VERSIONS, "VARIANTS.md")
    assert os.path.isfile(md)
    text = open(md, encoding="utf-8").read()
    for n in range(1, N_VARIANTS + 1):
        assert f"### {n:02d} ·" in text, f"variant {n} missing from VARIANTS.md"


# ---------------------------------------------------------------------------
# The generator still runs
# ---------------------------------------------------------------------------

def test_the_variant_set_is_thirty_uniquely_slugged_pages(gen):
    assert len(gen.variants.VARIANTS) == N_VARIANTS
    assert [s["n"] for s in gen.variants.VARIANTS] == \
        list(range(1, N_VARIANTS + 1))
    assert len({s["slug"] for s in gen.variants.VARIANTS}) == N_VARIANTS
    for spec in gen.variants.VARIANTS:
        for field in ("title", "changes", "adds", "removes", "argument"):
            assert spec[field].strip(), f"v{spec['n']:02d} has an empty {field}"
        assert callable(spec["build"])


def test_shortcuts_map_ctrl_1_to_9_onto_the_core_pipeline(gen):
    shortcuts = gen.variants._shortcuts()
    assert list(shortcuts.values()) == [f"Ctrl+{i}" for i in range(1, 10)]
    from spacr.qt.app import APPS, SECTION_CORE
    assert list(shortcuts) == [k for k, _n, _d, s in APPS
                               if s == SECTION_CORE][:9]


def test_every_variant_builds_and_draws_real_registry_text(
        gen, ctx, frozen_home_panels):
    """All thirty, in process. Structure only — see the module docstring
    for why the geometry is measured elsewhere.

    Each page has to show app names that came out of ``APPS``: a variant
    that builds an empty shell passes a "does it raise" smoke test and
    fails a reviewer.
    """
    names = {gen.common.name_of(k) for k in gen.common.all_keys()}
    for spec in gen.variants.VARIANTS:
        page = spec["build"](ctx)
        try:
            _shown(page, gen.app)
            assert set(_texts(page)) & names, \
                f"v{spec['n']:02d} {spec['slug']} names no registered app"
        finally:
            _drop(page, gen.app)


def test_variants_that_claim_to_show_everything_do(gen, ctx):
    """The dense / A-to-Z / illustrated layouts promise every app.

    Their whole argument is "nothing is hidden", so an app missing from
    one of them makes the argument false — which is precisely what three
    uncategorised apps did to them.
    """
    names = {gen.common.name_of(k) for k in gen.common.all_keys()}
    for number in (13, 22, 23):
        spec = gen.variants.VARIANTS[number - 1]
        page = spec["build"](ctx)
        try:
            _shown(page, gen.app)
            missing = names - set(_texts(page))
            assert not missing, \
                f"v{number:02d} {spec['slug']} omits {sorted(missing)}"
        finally:
            _drop(page, gen.app)


_AUDIT_DRIVER = r'''
"""Build every variant in a process that owns its own QApplication."""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])          # the _generators directory
sys.path.insert(0, sys.argv[2])          # the repo root
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pin the zoom before any widget is built. The audit measures variants at
# their RECORDED widths, so it must run at the reference scale; the shipped
# default of 150% would elide every long name inside a box recorded at 100%.
try:
    from spacr.qt import preferences as _prefs
    _prefs.set_font_scale(float(os.environ.get("SPACR_AUDIT_FONT_SCALE", "1.0")))
except Exception:
    pass

import common
import render

app = common.bootstrap()
import variants

from PySide6.QtCore import QEvent



ctx = common.Ctx(app, "dark")
# Owning the application is the whole point: this applies the theme
# stylesheet before a single tile is constructed, which is the only
# order in which the tiles report the sizes a real render gives them.
ctx.apply_theme()
out = {}
for spec in variants.VARIANTS:
    page = spec["build"](ctx)
    page.resize(common.CANVAS_W, common.CANVAS_H)
    page.show()
    for _ in range(4):
        app.processEvents()
    report = render.audit(page)
    page.hide()
    page.setParent(None)
    page.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    out[spec["n"]] = {k: v for k, v in report.items() if v}
print("<<<AUDIT>>>" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def subprocess_audit(tmp_path_factory):
    """Every variant's layout audit, measured in its own process."""
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    driver = tmp_path_factory.mktemp("home-audit") / "audit_variants.py"
    driver.write_text(_AUDIT_DRIVER, encoding="utf-8")
    # The audit measures variant layouts at their RECORDED widths, so it has
    # to run at the reference zoom. The subprocess reads the real preference,
    # and at the 150% default every long app name elides inside a box recorded
    # at 100% — a property of the audit's fixed widths, not of the shipped
    # page, which has zero elided labels at 150% because its tile widths go
    # through `scaled_px`. Its own QSettings scope, so nothing the user has
    # set is touched.
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg",
               SPACR_AUDIT_FONT_SCALE="1.0")
    proc = subprocess.run(
        [sys.executable, str(driver), GENERATORS, REPO_ROOT],
        capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, (
        "the variant generator no longer runs:\n" + proc.stderr[-4000:])
    marker = proc.stdout.rindex("<<<AUDIT>>>") + len("<<<AUDIT>>>")
    return {int(n): flags
            for n, flags in json.loads(proc.stdout[marker:]).items()}


def test_the_generator_builds_all_thirty(subprocess_audit):
    assert sorted(subprocess_audit) == list(range(1, N_VARIANTS + 1))


def test_no_variant_clips_elides_or_overflows(subprocess_audit):
    """The defect the home-screen rework was raised to fix.

    Every one of the thirty, in the theme and the widget order a real
    render uses — not the two that happened to be sampled before.

    It asserted zero everywhere, which is what it should assert and what
    it did for as long as thirty-four apps fitted. The registry is at
    forty-nine and nine of the thirty do not fit any more, so a bare
    "assert nothing is wrong" stopped on the first of them and said
    nothing about the other twenty-nine — a red test that measured one
    variant. :data:`KNOWN_LAYOUT_DEFECTS` is that measurement written
    down for all thirty instead, compared with ``==`` so that a defect
    appearing, worsening OR being fixed all fail here.

    Nothing is excused by being listed. See the note on the table for why
    these nine are a design decision rather than a defect to tune away.
    """
    # Pin zoom to 1.0 for this measurement. The test builds widgets at
    # EXPLICIT pixel sizes and asks whether the text fits; the zoom preference
    # scales the stylesheet's fonts on top of that, so at the 150% default the
    # label renders half again larger inside a box pinned at its unscaled
    # size. The product is fine — a real HomePage at 150% has zero elided
    # labels, because its tile widths go through `scaled_px`.
    #
    # Done inline rather than as an autouse fixture: the variant generator
    # copies this module into a standalone script and runs it without pytest,
    # so a module-level `@pytest.fixture` becomes a NameError there.
    from spacr.qt import preferences as _prefs
    _original_zoom = _prefs.get_font_scale()
    _prefs.set_font_scale(1.0)
    try:
        measured = {}
        for number, flags in sorted(subprocess_audit.items()):
            counts = {defect: len(flags[defect])
                      for defect in ("elided", "clipped", "overflow")
                      if flags.get(defect)}
            if counts:
                measured[number] = counts

        # The whole picture at once, so the message names every variant
        # that moved rather than the lowest-numbered one.
        assert measured == KNOWN_LAYOUT_DEFECTS, (
            "the variant layouts moved.\n"
            f"  measured: {measured}\n"
            f"  recorded: {KNOWN_LAYOUT_DEFECTS}\n"
            "A new entry, or a bigger count, means a layout stopped "
            "fitting 1440x900 — decide what that variant does about it. "
            "A smaller count or a vanished entry means one was FIXED: "
            "delete or lower its line here in the same commit, or the "
            "record stops being one.")

        # The other half of "nothing is excused by being listed": twenty-one
        # variants have no line in the table and carry no defect at all,
        # which is the property the test was written for and still holds.
        assert len(measured) == 9 and N_VARIANTS - len(measured) == 21
    finally:
        _prefs.set_font_scale(_original_zoom)


def test_only_the_documented_variants_need_a_scrollbar(subprocess_audit):
    """"Twenty-seven of the thirty fit 1440x900 with no scrollbar at
    all" is the finding these renders exist to make. It stops being true
    silently otherwise."""
    scrolling = {n for n, flags in subprocess_audit.items()
                 if flags.get("scrollbars")}
    assert scrolling <= SCROLLBARS_ALLOWED, \
        f"new scrollbars in {sorted(scrolling - SCROLLBARS_ALLOWED)}"
