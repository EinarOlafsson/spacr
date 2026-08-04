"""The thirty candidate Home screens.

Each builder returns a real, laid-out :class:`~parts.Page` widget. The
render harness grabs it at 1440x900 per theme. Nothing here is wired
into the app.

Every variant draws its apps from the real registry
(:func:`common.apps`); where a variant deliberately drops apps off the
home surface, the ``removes`` note names them.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import common
import parts
from common import (
    CATS_BROAD3,
    CATS_INTENT4,
    CATS_NARROW8,
    CATS_QUESTIONS,
    CATS_STAGE5,
    MOCK,
    PINNED,
    USE_COUNTS,
    Ctx,
    alphabetical,
    by_frequency,
    cats_current,
    name_of,
)
from parts import (
    DenseRow,
    Page,
    big_tile_grid,
    cat_header,
    cat_rail,
    chip,
    dense_list,
    hero,
    hint_bar,
    htile_grid,
    kbd,
    panel,
    pinned_row,
    plain_header,
    project_status_strip,
    queue_panel,
    quick_start,
    real_sidebar,
    recent_runs_list,
    recent_runs_strip,
    resume_banner,
    scroll_area,
    search_box,
    start_run_panel,
    stat_row,
    system_panel,
    text_label,
    transparent,
    whats_new_panel,
    wrapped,
)

#: Body margins every variant uses, so "what fits above the fold" is
#: comparable between renders.
MARGINS = (28, 20, 28, 16)
CONTENT_W = 1440 - MARGINS[0] - MARGINS[2]

VARIANTS: List[dict] = []


def variant(slug: str, title: str, *, changes: str, adds: str,
            removes: str, argument: str, notes: str = ""):
    """Register a variant builder plus the prose that describes it."""
    def deco(fn: Callable[[Ctx], QWidget]):
        VARIANTS.append({
            "n": len(VARIANTS) + 1,
            "slug": slug,
            "title": title,
            "changes": changes,
            "adds": adds,
            "removes": removes,
            "argument": argument,
            "notes": notes,
            "build": fn,
        })
        return fn
    return deco


def _shortcuts() -> Dict[str, str]:
    """Ctrl+1..9 as the app actually assigns them (the nine core apps)."""
    return {k: f"Ctrl+{i + 1}"
            for i, k in enumerate(common.core_keys()[:9])}


# ---------------------------------------------------------------------------
# 01 — the control
# ---------------------------------------------------------------------------

def _patch_startup_determinism() -> None:
    """Freeze the live values the shipped Home screen reads.

    The shipped :class:`spacr.qt.widgets.home.HomePage` polls the GPU,
    the disk, the run journal and the plate queue. A screenshot of that
    differs on every machine and every hour, and a screen that renders
    differently every run cannot be reviewed. Patched in this process
    only — no product file is touched.

    **Only the run journal is optional.** An earlier draft of this
    docstring claimed "every patch target is looked up defensively";
    that was not true and is not wanted. ``spacr.qt.widgets.home`` and
    its two panel classes are the very things variant 01 renders — v01
    goes on to call ``spacr.qt.app.make_home_page()``, which imports the
    same module — so a rename there cannot be survived by this function
    anyway, and pretending otherwise would only move the traceback
    somewhere less informative. They are imported and assigned
    unguarded, and the generator is meant to die loudly if they move.

    The journal is different: it is a separate module, an installation
    may legitimately not have one, and a baseline that draws "no runs
    yet" is still a fair comparison. Its ``import`` alone sits in the
    ``try`` — the assignments after it are module attribute writes that
    cannot fail, and leaving them inside would let a future edit throw
    into a bare ``except`` and freeze nothing without saying so.
    """
    from spacr.qt.widgets import home as H
    # Staticmethods: assign plain functions, not lambdas taking self.
    H.SystemPanel.gpu_util = staticmethod(lambda: "41%")
    H.SystemPanel.gpu_vram = staticmethod(lambda: "14.9 / 24 GB")
    H.SystemPanel.disk_used = staticmethod(lambda: "68%")
    # An empty queue is what a fresh install shows, and it is the only
    # queue state that does not depend on the reviewer's ~/.spacr.
    H.QueuedPanel.queue_items = lambda self: []
    try:
        import spacr.run_journal as J
    except Exception:
        return
    J.recent_runs = lambda limit=10: [
        {"app_key": k, "status": "success" if ok else "error",
         "elapsed_s": 1324 if ok else 207,
         "start_utc": "2026-07-25T14:22:10Z", "dir": "/tmp"}
        for k, _p, _w, ok, _e in MOCK["recent"][:limit]
    ]
    J.journal_totals = lambda: {
        "total_runs": 148, "mask_runs": 52, "measure_runs": 47,
        "classify_runs": 23, "models_recorded": 12,
    }


@variant(
    "baseline-today", "Baseline — the Home screen as it ships today",
    changes="Nothing. This is the shipped screen, rendered with the same "
            "harness as every other variant so the comparison is fair: "
            "the real Sidebar plus the real HomePage, built through "
            "`spacr.qt.app.make_home_page()` — the same call MainWindow "
            "makes — so the bands, stages, notes and icon provider are "
            "the shipped ones rather than a re-assembly.",
    adds="Nothing.",
    removes="Nothing.",
    # Do not re-add "the last apps are cut off with no way to scroll to
    # them". That was true of the pre-QScrollArea Sidebar and is now
    # contradicted by finding 1 in VARIANTS.md, three paragraphs above
    # where this text lands — one artefact cannot say both.
    argument="It is the thing every other variant has to beat, and it "
            "shows its problem at 1440x900 without anyone having to "
            f"argue for it: the sidebar's {common.n_apps()} items + 5 "
            "headings ask for far more height than a laptop gives, so "
            "the navigation is a scrolling column rather than a list you "
            "can see, and the page beside it needs a vertical scrollbar "
            "of its own before the last band is on screen. Both "
            "scrollbars in this render are real; neither is a defect "
            "any more.",
    notes="Live GPU/disk/journal/queue readings are frozen to fixed values "
          "for the render; everything else is the shipped widget.")
def v01(ctx: Ctx) -> QWidget:
    _patch_startup_determinism()
    # make_home_page() rather than HomePage(...): the grouping, the
    # stages, the notes and the icon provider are four arguments that
    # have to agree, and a baseline that assembles its own HomePage is a
    # render of a page that does not ship.
    from spacr.qt.app import make_home_page
    page = Page(ctx, margins=(0, 0, 0, 0), spacing=0)
    page.add_rail(real_sidebar(ctx))
    page.body.addWidget(make_home_page())
    return page.finish(status="Ready")


# ---------------------------------------------------------------------------
# 02
# ---------------------------------------------------------------------------

@variant(
    "stages-grid", "Workflow stages, wrapping tile grid",
    changes="Categories are renamed from kinds-of-thing to stages of a "
            "run — Acquire, Segment, Measure, Analyse, Report — and each "
            "one is a wrapping grid instead of a horizontal scroller, so "
            "no app is hidden off the right edge.",
    adds="Nothing.",
    removes="The insights dashboard and the empty 'Reserved for featured "
            "content' box. The hint bar stays.",
    argument="Same five-band shape people already know, but the names "
            "answer 'where am I in my run?' instead of 'what kind of "
            f"code is this?', and all {common.n_apps()} apps are on one "
            "surface with nothing hidden off the right edge.")
def v02(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS)
    page.body.addWidget(hero(ctx, compact=True))
    # Seven columns. Six was too few — the seven-app bands each wrapped
    # onto a second row and the page asked for 905 px, which Qt resolved
    # by silently squashing something. Eight is too many: 1384 px of
    # content over eight columns is a 166 px tile, and at that width
    # thirty-four of the thirty-eight names elide however small the font
    # is set (measured; nine px still elides six of them). Seven is the
    # widest grid whose tile can hold a name.
    #
    # So this no longer fits five rows: the registry outgrew five bands
    # of seven when Illumination, Barcode QC, Layer Viewer and Graph
    # Builder arrived, and thirty-eight apps cannot go into thirty-five
    # slots. The three bands that hold eight take a second row. That is
    # the trade this variant now records — a taller page against an
    # unreadable one — and it is why the argument above no longer claims
    # vertical slack.
    for title, keys in CATS_STAGE5:
        page.body.addWidget(cat_header(ctx, title, note=f"{len(keys)} apps"))
        page.body.addWidget(htile_grid(ctx, keys, cols=7, width=190,
                                       icon_px=40, name_px=11, height=64))
    page.body.addStretch(1)
    return page.finish(footer=hint_bar(ctx))


# ---------------------------------------------------------------------------
# 03
# ---------------------------------------------------------------------------

@variant(
    "three-broad", "Three broad categories",
    changes="Five categories collapse to three — Prepare, Run, Review — "
            "which is the smallest split that still means something. "
            "Tiles are wider and the whole page is one column.",
    adds="Nothing.",
    removes="The insights dashboard, the reserved surface, the hint bar, "
            "and the hero shrinks to one line.",
    argument="Three headings is the most a person actually holds in "
            "their head while scanning. It is also the fewest headings "
            "that never needs a scroll: everything is above the fold "
            "with room to spare.")
def v03(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    page.body.addWidget(hero(ctx, compact=True))
    for title, keys in CATS_BROAD3:
        page.body.addWidget(cat_header(ctx, title, note=f"{len(keys)}"))
        page.body.addWidget(htile_grid(ctx, keys, cols=5, width=266))
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 04
# ---------------------------------------------------------------------------

@variant(
    "eight-narrow", "Eight narrow categories, as panels",
    changes="Eight tightly-drawn categories (Segment, Train models, "
            "Measure, Label, Classify, Screens & reports, Import & "
            "batch, Toxoplasma) laid out as a 3x3 board of panels, each "
            "listing its apps as compact rows with their one-line "
            "descriptions on the same row.",
    adds="Per-category counts in the headings.",
    removes="Tiles entirely — every app is a one-line row. Also the "
            "hero, the dashboard and the reserved surface.",
    argument="Narrow categories are the only ones you can name honestly: "
            "'Segment' is three apps and it is obvious which three. The "
            "cost is that two categories only hold two apps, which the "
            "current design guidance says is not worth a heading.")
def v04(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=f"Eight categories · {common.n_apps()} apps",
        actions=(("Search…", False), ("Preferences", False))))
    board = QWidget()
    board.setObjectName("Transparent")
    grid = QGridLayout(board)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(12)
    pw = (CONTENT_W - 2 * 12) // 3
    for i, (title, keys) in enumerate(CATS_NARROW8):
        frame, col = panel(ctx, margins=(14, 11, 14, 11), spacing=4)
        frame.setFixedWidth(pw)
        col.addWidget(text_label(ctx, f"{title}  ({len(keys)})", size=11,
                                 weight=600, color=ctx.P["fg_muted"],
                                 tracking="1.6px", upper=True))
        col.addWidget(dense_list(ctx, keys, width=pw - 28, name_width=136))
        col.addStretch(1)
        grid.addWidget(frame, i // 3, i % 3, Qt.AlignTop)
    page.body.addWidget(board)
    page.body.addStretch(1)
    return page.finish(footer=hint_bar(ctx))


# ---------------------------------------------------------------------------
# 05
# ---------------------------------------------------------------------------

@variant(
    "flat-search", "No categories at all — flat searchable grid",
    changes=f"There are no sections. All {common.n_apps()} apps sit in one "
            "alphabetical "
            "grid under a search field, with filter chips as the only "
            "grouping and no default filter applied.",
    adds="A search field and a row of filter chips.",
    removes="Every category heading, the dashboard, the reserved "
            "surface, the hint bar.",
    argument="Nobody agrees on the categories, and a flat grid is the "
            f"only arrangement that cannot be wrong. {common.n_apps()} items is "
            "small "
            "enough to scan, and the search field is faster than any "
            "hierarchy once you know the name.")
def v05(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    top, row = transparent(horizontal=True, spacing=14)
    row.addWidget(text_label(ctx, "spaCR", size=26, weight=300,
                             color=ctx.P["accent"], tracking="-0.6px"))
    box = search_box(ctx, f"Search {common.n_apps()} apps —  mask, barcode, "
                          "plate, κ …")
    box.setFixedWidth(560)
    row.addWidget(box)
    row.addStretch(1)
    row.addWidget(text_label(ctx, f"{common.n_apps()} apps", size=12,
                             color=ctx.P["fg_dim"]))
    page.body.addWidget(top)

    chips, crow = transparent(horizontal=True, spacing=8)
    crow.addWidget(chip(ctx, "All", on=True))
    for label in ("Segmentation", "Measurement", "Screens", "Models",
                  "Import/export", "Toxoplasma"):
        crow.addWidget(chip(ctx, label))
    crow.addStretch(1)
    page.body.addWidget(chips)

    page.body.addWidget(htile_grid(ctx, alphabetical(), cols=6, width=224,
                                   icon_px=40, name_px=13, height=66))
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 06
# ---------------------------------------------------------------------------

@variant(
    "search-only", "Search-first — the grid is what you get after you type",
    changes="The home screen is a search box and almost nothing else. "
            "The app grid does not exist until you type; before that you "
            "get eight most-used apps as a 'jump to' row.",
    adds="A large centred search field and a keyboard hint.",
    removes=f"All {common.n_apps()} tiles, all categories, the hero, the "
            "dashboard, the reserved surface, the hint bar. "
            f"{common.n_apps() - 8} of the {common.n_apps()} apps have no "
            "presence on the screen at all until you search.",
    argument="The most honest reading of 'too much on the home page' is "
            "to put nothing on it. Every app is one keystroke away and "
            "the eight that matter are already there.")
def v06(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(28, 20, 28, 16), spacing=18)
    page.body.addStretch(2)
    centre, col = transparent(spacing=16)
    pix = ctx.logo(84)
    if pix is not None:
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setFixedSize(84, 84)
        lbl.setStyleSheet("background: transparent;")
        col.addWidget(lbl, 0, Qt.AlignHCenter)
    col.addWidget(text_label(ctx, "spaCR", size=44, weight=300,
                             color=ctx.P["accent"], tracking="-1px"),
                  0, Qt.AlignHCenter)
    box = search_box(ctx, "What do you want to do?", big=True, width=720)
    col.addWidget(box, 0, Qt.AlignHCenter)
    hint, hrow = transparent(horizontal=True, spacing=8)
    hrow.addStretch(1)
    hrow.addWidget(text_label(ctx, "Press", size=11,
                              color=ctx.P["fg_dim"]))
    hrow.addWidget(kbd(ctx, "Ctrl+K"))
    hrow.addWidget(text_label(ctx, "anywhere in spaCR to get back here.",
                              size=11, color=ctx.P["fg_dim"]))
    hrow.addStretch(1)
    col.addWidget(hint)
    page.body.addWidget(centre)
    page.body.addStretch(1)

    jump, jcol = transparent(spacing=10)
    jcol.addWidget(text_label(ctx, "or jump straight to", size=11,
                              weight=600, color=ctx.P["fg_dim"],
                              tracking="2px", upper=True),
                   0, Qt.AlignHCenter)
    jcol.addWidget(big_tile_grid(ctx, by_frequency()[:8], cols=8, width=148,
                                 height=110, icon_px=40),
                   0, Qt.AlignHCenter)
    page.body.addWidget(jump)
    page.body.addStretch(1)
    return page.finish(status="Type to search · Esc to clear")


# ---------------------------------------------------------------------------
# 07
# ---------------------------------------------------------------------------

@variant(
    "rail-and-pane", "Category rail on the left, content pane on the right",
    changes="Categories move off the page and into a left rail; the pane "
            "shows one category at a time as large tiles with their "
            "one-line descriptions visible, not hidden behind a hover.",
    adds="A category rail with per-category counts; the descriptions "
            "become permanently visible.",
    removes="The app sidebar (the rail replaces it), the five stacked "
            "section headings, the dashboard, the reserved surface, the "
            "hint bar — the hint bar exists only because descriptions "
            "were hidden, and here they are not.",
    argument="It is the only arrangement where every app's description "
            "is readable without hovering, which is what the hint bar "
            "was a workaround for. One click of cost, and the page can "
            "never overflow no matter how many apps get added.")
def v07(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(24, 18, 24, 16), spacing=14)
    titles = [t for t, _ in CATS_STAGE5]
    counts = [len(k) for _, k in CATS_STAGE5]
    page.add_rail(cat_rail(ctx, titles, selected=1, header="Stages",
                           counts=counts, width=236))
    keys = CATS_STAGE5[1][1]
    page.body.addWidget(plain_header(
        ctx, "Segment", f"turn images into objects · {len(keys)} apps"))
    page.body.addWidget(big_tile_grid(ctx, keys, cols=4, width=272,
                                      height=172, icon_px=54,
                                      blurb_lines=3))
    page.body.addStretch(1)
    return page.finish(status="Segment")


# ---------------------------------------------------------------------------
# 08
# ---------------------------------------------------------------------------

@variant(
    "tabs", "Tabs, one per stage",
    changes="The five categories become a real tab bar. Only the active "
            "stage's apps are on screen, as large tiles with visible "
            "descriptions.",
    adds="A tab bar; descriptions become permanently visible.",
    removes="Four fifths of the apps at any moment, plus the dashboard, "
            "the reserved surface and the hint bar.",
    argument="Tabs put the categories on one line instead of five, which "
            "buys back about 380 px of vertical space, and a tab bar is "
            "a control everyone already knows how to use.")
def v08(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle="End-to-end microscopy → single-cell measurements",
        actions=(("Resume last run", True),)))
    tabs = QTabWidget()
    tabs.setDocumentMode(False)
    for title, keys in CATS_STAGE5:
        holder = QWidget()
        holder.setObjectName("Transparent")
        col = QVBoxLayout(holder)
        col.setContentsMargins(16, 16, 16, 16)
        col.setSpacing(12)
        col.addWidget(big_tile_grid(ctx, keys, cols=5, width=246,
                                    height=180, icon_px=52, blurb_lines=3))
        col.addStretch(1)
        tabs.addTab(holder, f"{title}  ({len(keys)})")
    tabs.setCurrentIndex(1)
    page.body.addWidget(tabs, 1)
    return page.finish(status="Segment")


# ---------------------------------------------------------------------------
# 09
# ---------------------------------------------------------------------------

@variant(
    "start-a-run", "One prominent 'start a run' path",
    changes="The top half of the screen is a single task: choose a "
            "folder, tick the stages you want, press Run. Everything "
            "else drops to a secondary compact list underneath.",
    adds="A start-a-run panel with a source field, pipeline chips and a "
            "Run button — the home screen can launch a pipeline without "
            "opening an app first.",
    removes="Tiles, the five section headings as headings (they become "
            "column captions), the dashboard, the reserved surface.",
    argument="Ninety per cent of home-screen visits end in 'run Mask "
            "then Measure on this folder'. This is the only variant "
            "where that takes zero navigation.")
def v09(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    top, row = transparent(horizontal=True, spacing=14)
    row.addWidget(start_run_panel(ctx, width=CONTENT_W - 340, height=196), 1)
    row.addWidget(recent_runs_list(ctx, count=4, width=326))
    page.body.addWidget(top)

    page.body.addWidget(cat_header(ctx, "All apps", note=str(common.n_apps())))
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(18)
    colw = (CONTENT_W - 2 * 18) // 3
    groups = [CATS_BROAD3[0], CATS_BROAD3[1], CATS_BROAD3[2]]
    for title, keys in groups:
        block, bcol = transparent(spacing=4)
        bcol.addWidget(text_label(ctx, title, size=10, weight=600,
                                  color=ctx.P["fg_dim"], tracking="1.6px",
                                  upper=True))
        bcol.addWidget(dense_list(ctx, keys, width=colw, name_width=136))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 10
# ---------------------------------------------------------------------------

@variant(
    "resume-first", "Resume the last run, then everything else",
    changes="The screen opens on what you were doing, not on what spaCR "
            "can do. A resume banner and the last three runs come "
            "first; the apps are a dense three-column list below.",
    adds="A resume-last-run banner (names the plate, the stage and what "
            "comes next) and a recent-runs strip with Resume / Settings "
            "on each card.",
    removes="Tiles, the hero, the dashboard, the reserved surface, the "
            "hint bar.",
    argument="A returning user's actual question is 'where was I?', and "
            "no version of the current screen answers it. The apps are "
            "still all there, just no longer the loudest thing.")
def v10(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=MOCK["project"],
        actions=(("Search…", False), ("New project…", False))))
    page.body.addWidget(resume_banner(ctx))
    page.body.addWidget(cat_header(ctx, "Recent"))
    page.body.addWidget(recent_runs_strip(ctx, count=3, card_width=336))
    page.body.addWidget(cat_header(ctx, "All apps", note=str(common.n_apps())))
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(18)
    colw = (CONTENT_W - 2 * 18) // 3
    for title, keys in CATS_BROAD3:
        block, bcol = transparent(spacing=4)
        bcol.addWidget(text_label(ctx, title, size=10, weight=600,
                                  color=ctx.P["fg_dim"], tracking="1.6px",
                                  upper=True))
        bcol.addWidget(dense_list(ctx, keys, width=colw, name_width=136,
                                  show_blurb=False))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish(status="Last run: Measure · plate_07 · 18 min ago")


# ---------------------------------------------------------------------------
# 11
# ---------------------------------------------------------------------------

@variant(
    "quick-start", "Guided quick-start for a first-time user",
    changes="The page is a three-step path — point at images, segment "
            "and measure, call your hits — with a real button on each "
            "step. The app list is demoted to one compact row per "
            "category underneath.",
    adds="A three-card guided quick-start with working actions "
            "(Choose folder / Run Mask → Measure / Open Annotate).",
    removes="The hero, the dashboard, the reserved surface. Tiles become "
            "one-line rows.",
    argument=f"A new user faced with {common.n_apps()} tiles has no idea "
            "which three "
            "matter. This tells them, and it is dismissible — after the "
            "first successful run the strip can collapse to a single "
            "line.")
def v11(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle="First time here? Start at step 1.",
        actions=(("Skip the tour", False),)))
    page.body.addWidget(quick_start(ctx))
    page.body.addWidget(cat_header(ctx, "Or open an app directly",
                                   note=f"{common.n_apps()} apps"))
    cols = QWidget()
    cols.setObjectName("Transparent")
    grid = QGridLayout(cols)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(18)
    grid.setVerticalSpacing(6)
    colw = (CONTENT_W - 2 * 18) // 3
    for i, (title, keys) in enumerate(CATS_BROAD3):
        block, bcol = transparent(spacing=4)
        bcol.addWidget(text_label(ctx, title, size=10, weight=600,
                                  color=ctx.P["fg_dim"], tracking="1.6px",
                                  upper=True))
        bcol.addWidget(dense_list(ctx, keys, width=colw, name_width=136,
                                  show_blurb=False))
        bcol.addStretch(1)
        grid.addWidget(block, 0, i, Qt.AlignTop)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 12
# ---------------------------------------------------------------------------

@variant(
    "pinned-first", "Pinned favourites first, then three categories",
    changes="Ordering, not grouping: the six apps this user pinned sit "
            "at the top as large tiles, and the rest follow in three "
            "broad categories as compact rows.",
    adds="A pinned row with a '+' slot, so the user curates their own "
            "top of page.",
    removes="The hero, the dashboard, the reserved surface, the hint bar.",
    argument="Whatever the categories are, everyone uses four or five "
            "apps and ignores the rest. Let the user say which, and the "
            "argument about the taxonomy stops mattering.")
def v12(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=MOCK["project"],
        actions=(("Edit pins", False), ("Search…", False))))
    page.body.addWidget(cat_header(ctx, "Pinned", note="drag to reorder"))
    page.body.addWidget(pinned_row(ctx, PINNED, tile_w=168, tile_h=116))
    page.body.addWidget(cat_header(
        ctx, "Everything else",
        note=f"{common.n_apps() - len(PINNED)} apps"))
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(18)
    colw = (CONTENT_W - 2 * 18) // 3
    for title, keys in CATS_BROAD3:
        rest = [k for k in keys if k not in PINNED]
        block, bcol = transparent(spacing=4)
        bcol.addWidget(text_label(ctx, title, size=10, weight=600,
                                  color=ctx.P["fg_dim"], tracking="1.6px",
                                  upper=True))
        bcol.addWidget(dense_list(ctx, rest, width=colw, name_width=136))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 13
# ---------------------------------------------------------------------------

@variant(
    "dense-two-column", "Dense two-column list, today's five categories",
    changes="No tiles anywhere. Today's five categories are kept "
            "verbatim, but every app is a 30 px row with its icon, its "
            "name and its description on one line, in two columns.",
    adds="Descriptions are permanently visible.",
    removes="Tiles, the hero, the dashboard, the reserved surface, the "
            "hint bar.",
    argument=f"It is the densest honest layout: all {common.n_apps()} apps "
            f"*and* all {common.n_apps()} descriptions above the fold at "
            "1440x900. Nothing is hidden, nothing needs a hover.")
def v13(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=f"{common.n_apps()} apps · everything on one screen",
        actions=(("Search…", False),)))
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(24)
    colw = (CONTENT_W - 24) // 2
    cats = cats_current()
    split = [cats[:2], cats[2:]]
    for group in split:
        block, bcol = transparent(spacing=10)
        for title, keys in group:
            bcol.addWidget(cat_header(ctx, title, note=f"{len(keys)}"))
            bcol.addWidget(dense_list(ctx, keys, width=colw, name_width=152))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 14
# ---------------------------------------------------------------------------

@variant(
    "by-frequency", "Ordered by how often you actually use it",
    changes="Ordering replaces grouping. One flat list, most-used first, "
            "with each app's run count beside it. Three tiers marked "
            "'daily', 'sometimes' and 'rarely' are the only headings.",
    adds="Per-app run counts drawn from the run journal.",
    removes="All five categories, the hero, the dashboard, the reserved "
            "surface, the hint bar.",
    argument="The taxonomy argument is unwinnable; usage is measurable. "
            "It also self-corrects — a new app that people use rises "
            "without anyone editing a table.",
    notes="Run counts are illustrative values in the generator, not real "
          "telemetry.")
def v14(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle="ordered by your run history",
        actions=(("Order: usage", False), ("Search…", False))))
    order = by_frequency()
    tiers = (("Daily", order[:8]), ("Sometimes", order[8:18]),
             ("Rarely", order[18:]))
    badges = {k: f"{USE_COUNTS[k]} runs" for k in order}
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(20)
    colw = (CONTENT_W - 2 * 20) // 3
    for title, keys in tiers:
        block, bcol = transparent(spacing=6)
        bcol.addWidget(cat_header(ctx, title, note=f"{len(keys)}"))
        bcol.addWidget(dense_list(ctx, keys, width=colw, name_width=136,
                                  show_blurb=False, badges=badges))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 15
# ---------------------------------------------------------------------------

@variant(
    "pipeline-flow", "The pipeline, drawn as a pipeline",
    changes="Five stage columns read left to right with arrows between "
            "them, so the home screen is a picture of the workflow "
            "rather than a list of categories.",
    adds="Arrows between stages, and a per-stage caption saying what "
            "goes in and what comes out.",
    removes="Tiles, the hero, the dashboard, the reserved surface, the "
            "hint bar.",
    argument="The categories in a pipeline tool *are* an order, and no "
            "vertical stack of headings shows that. A new user can read "
            "the whole method off the home screen.")
def v15(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle="images → objects → measurements → hits → a report"))
    captions = {
        "Acquire": "folders, formats, plates",
        "Segment": "images → masks",
        "Measure": "masks → a table",
        "Analyse": "table → scores",
        "Report":  "scores → a document",
    }
    flow = QWidget()
    flow.setObjectName("Transparent")
    row = QHBoxLayout(flow)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    colw = 244
    for i, (title, keys) in enumerate(CATS_STAGE5):
        block, col = transparent(spacing=8)
        col.addWidget(text_label(ctx, f"{i + 1}. {title}", size=15,
                                 weight=600, color=ctx.P["accent"]))
        col.addWidget(text_label(ctx, captions[title], size=11, weight=300,
                                 color=ctx.P["fg_dim"]))
        col.addWidget(dense_list(ctx, keys, width=colw, name_width=150,
                                 show_blurb=False, spacing=2))
        col.addStretch(1)
        block.setFixedWidth(colw)
        row.addWidget(block, 0, Qt.AlignTop)
        if i < len(CATS_STAGE5) - 1:
            arrow = text_label(ctx, "→", size=26, weight=300,
                               color=ctx.P["fg_dim"])
            arrow.setFixedWidth((CONTENT_W - 5 * colw) // 4)
            arrow.setAlignment(Qt.AlignCenter)
            row.addWidget(arrow, 0, Qt.AlignTop)
    page.body.addWidget(flow)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 16
# ---------------------------------------------------------------------------

@variant(
    "status-first", "Project status first, then the pipeline",
    changes="A project bar names the open dataset and its size at the "
            "top of every home visit; the pipeline stages follow as "
            "columns.",
    adds="A dataset/plate status strip (project, plates, images, "
            "objects, database size, switch-project), plus a queue panel "
            "showing what runs next.",
    removes="The hero, the dashboard, the reserved surface, the hint bar.",
    argument="spaCR is always pointed at *something*, and today the home "
            "screen never says what. Nearly every support question "
            "starts with 'which folder were you on?'.")
def v16(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=13)
    page.body.addWidget(project_status_strip(ctx))
    body, brow = transparent(horizontal=True, spacing=18)
    left, lcol = transparent(spacing=10)
    stage_row = QWidget()
    stage_row.setObjectName("Transparent")
    srow = QHBoxLayout(stage_row)
    srow.setContentsMargins(0, 0, 0, 0)
    srow.setSpacing(14)
    colw = 200
    for i, (title, keys) in enumerate(CATS_STAGE5):
        block, col = transparent(spacing=6)
        col.addWidget(text_label(ctx, f"{i + 1}. {title}", size=13,
                                 weight=600, color=ctx.P["accent"]))
        col.addWidget(dense_list(ctx, keys, width=colw, name_width=150,
                                 show_blurb=False, spacing=2))
        col.addStretch(1)
        block.setFixedWidth(colw)
        srow.addWidget(block, 0, Qt.AlignTop)
    srow.addStretch(1)
    lcol.addWidget(stage_row)
    lcol.addStretch(1)
    brow.addWidget(left, 1)

    aside, acol = transparent(spacing=12)
    acol.addWidget(queue_panel(ctx, width=300))
    acol.addWidget(recent_runs_list(ctx, count=4, width=300))
    acol.addStretch(1)
    brow.addWidget(aside, 0)
    page.body.addWidget(body, 1)
    return page.finish(status=f"Project: {MOCK['project']}")


# ---------------------------------------------------------------------------
# 17
# ---------------------------------------------------------------------------

@variant(
    "split-apps-aside", "Apps left, everything-about-your-machine right",
    changes="A hard vertical split. The left two thirds are apps and "
            "nothing else; the right third is state — recent runs, "
            "system, what changed.",
    adds="A persistent right-hand aside carrying recent runs, disk/GPU "
            "state and a what's-new panel.",
    removes="The horizontally-scrolling section rows, the reserved "
            "surface, the hint bar. The insights dashboard is not "
            "removed but relocated to the aside, where it stops pushing "
            "the apps down the page.",
    argument="The current dashboard's problem is not that it exists, it "
            "is that it sits *under* the apps and so nothing fits. Put "
            "it beside them and both halves work.")
def v17(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=MOCK["project"],
        actions=(("Resume last run", True),)))
    body, brow = transparent(horizontal=True, spacing=20)
    left, lcol = transparent(spacing=10)
    for title, keys in CATS_BROAD3:
        lcol.addWidget(cat_header(ctx, title, note=f"{len(keys)}"))
        lcol.addWidget(htile_grid(ctx, keys, cols=4, width=250,
                                  icon_px=36, name_px=13, height=62))
    lcol.addStretch(1)
    brow.addWidget(left, 1)
    aside, acol = transparent(spacing=12)
    acol.addWidget(recent_runs_list(ctx, count=4, width=328))
    acol.addWidget(system_panel(ctx, width=328))
    acol.addWidget(whats_new_panel(ctx, width=328, items=3))
    acol.addStretch(1)
    brow.addWidget(aside, 0)
    page.body.addWidget(body, 1)
    return page.finish()


# ---------------------------------------------------------------------------
# 18
# ---------------------------------------------------------------------------

#: Every app that is not on the core pipeline — the ones variant 18
#: puts behind its one door. Derived, because the list used to be
#: twenty names typed into the prose and it named neither Distributed
#: Jobs, Classifier Evaluation, Run History nor Replication Assay.
_BEHIND_THE_DOOR = [k for k in common.all_keys() if k not in common.core_keys()]


@variant(
    "core-nine-only",
    f"Nine apps, and a door to the other {len(_BEHIND_THE_DOOR)}",
    changes="The home screen shows only the nine Core-pipeline apps, as "
            "large illustrated tiles with their descriptions. Everything "
            "else lives behind one button.",
    adds="A 'More tools' door with a count.",
    removes=f"{len(_BEHIND_THE_DOOR)} apps: "
            + ", ".join(common.name_of(k) for k in _BEHIND_THE_DOOR)
            + ". Also the dashboard, the reserved surface and the hint "
              "bar.",
    argument="This is what 'too much on the home page' looks like taken "
            "seriously. Nine tiles, each big enough to read, each one a "
            "thing you would actually do today — and the other "
            f"{len(_BEHIND_THE_DOOR)} are one click away, not gone.")
def v18(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(28, 20, 28, 16), spacing=16)
    head, hrow = transparent(horizontal=True, spacing=14)
    hrow.addWidget(text_label(ctx, "spaCR", size=32, weight=300,
                              color=ctx.P["accent"], tracking="-0.8px"))
    hrow.addWidget(text_label(ctx, "the nine steps of a screen", size=13,
                              weight=300, color=ctx.P["fg_muted"]))
    hrow.addStretch(1)
    more = QPushButton(f"More tools  ({len(_BEHIND_THE_DOOR)})")
    more.setCursor(Qt.PointingHandCursor)
    hrow.addWidget(more)
    search = QPushButton("Search…")
    hrow.addWidget(search)
    page.body.addWidget(head)
    core = common.core_keys()
    page.body.addWidget(big_tile_grid(ctx, core, cols=3, width=440,
                                      height=210, icon_px=62,
                                      blurb_lines=2,
                                      badges={k: v for k, v in
                                              _shortcuts().items()}),
                        0, Qt.AlignHCenter)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 19
# ---------------------------------------------------------------------------

@variant(
    "by-question", "Categories named as the question you arrived with",
    changes="Four categories, each phrased as a question a biologist "
            "actually asks — 'I have images. Where are my objects?', 'I "
            "have objects. What are they like?', 'I have a screen. Which "
            "genes matter?', 'Should I believe any of this?'.",
    adds="Nothing beyond the wording.",
    removes="The five kind-of-thing headings, the hero, the dashboard, "
            "the reserved surface, the hint bar.",
    argument="Names are the cheapest thing to change and the thing "
            "people actually navigate by. 'Segmentation models' is a "
            "category of code; 'Where are my objects?' is a category of "
            "intent, and the same five apps sit under it.")
def v19(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(ctx, subtitle="pick the question you "
                                                     "came with"))
    board = QWidget()
    board.setObjectName("Transparent")
    grid = QGridLayout(board)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(14)
    pw = (CONTENT_W - 16) // 2
    for i, (question, keys) in enumerate(CATS_QUESTIONS):
        frame, col = panel(ctx, margins=(16, 14, 16, 12), spacing=8)
        frame.setFixedWidth(pw)
        col.addWidget(wrapped(ctx, question, pw - 32, 1, size=16, weight=600,
                              color=ctx.P["fg"]))
        col.addWidget(dense_list(ctx, keys, width=pw - 32, name_width=140))
        col.addStretch(1)
        grid.addWidget(frame, i // 2, i % 2, Qt.AlignTop)
    page.body.addWidget(board)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 20
# ---------------------------------------------------------------------------

@variant(
    "whats-new", "What changed in this version, above the apps",
    changes="A release panel runs along the top; the apps sit beneath it "
            "as a five-column grid with today's five categories reduced "
            "to inline captions.",
    adds="A 'New in 1.3.6' panel with links straight into the apps that "
            "changed, and an update check.",
    removes="The hero, the dashboard, the reserved surface, the hint bar.",
    argument="spaCR ships often and nobody reads the changelog. The home "
            "screen is the only page every user sees every session, and "
            "four bullets is a cheap rent to charge it.")
def v20(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=13)
    top, row = transparent(horizontal=True, spacing=16)
    frame, col = panel(ctx, margins=(18, 14, 18, 14), spacing=8)
    col.addWidget(text_label(ctx, f"New in spaCR {MOCK['version']}", size=17,
                             weight=600))
    bullets, brow = transparent(horizontal=True, spacing=22)
    for line in MOCK["whats_new"]:
        item, icol = transparent(horizontal=True, spacing=7)
        icol.setAlignment(Qt.AlignTop)
        icol.addWidget(text_label(ctx, "•", size=12, color=ctx.P["accent"]))
        icol.addWidget(wrapped(ctx, line, 220, 2, size=11))
        brow.addWidget(item)
    brow.addStretch(1)
    col.addWidget(bullets)
    row.addWidget(frame, 1)
    side, scol = transparent(spacing=8)
    up = QPushButton("Check for updates")
    up.setObjectName("PrimaryButton")
    scol.addWidget(up)
    notes = QPushButton("Full release notes")
    scol.addWidget(notes)
    scol.addStretch(1)
    row.addWidget(side, 0)
    page.body.addWidget(top)

    for title, keys in cats_current():
        page.body.addWidget(text_label(ctx, title, size=10, weight=600,
                                       color=ctx.P["fg_dim"],
                                       tracking="1.6px", upper=True))
        page.body.addWidget(htile_grid(ctx, keys, cols=6, width=222,
                                       icon_px=36, name_px=12, height=58,
                                       vspace=6))
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 21
# ---------------------------------------------------------------------------

@variant(
    "dashboard-first", "Dashboard across the top, apps beneath",
    changes="The insights dashboard is promoted to the top of the page "
            "and widened into a stat row plus three panels; the apps "
            "become a compact three-column list under it.",
    adds="Big-number stat tiles (runs, plates, objects, models), a "
            "queue panel and a system panel built on the real UsageBar "
            "widget.",
    removes="Tiles, the hero, the reserved surface, the hint bar.",
    argument="If the dashboard is worth having at all it is worth "
            "reading first — today it sits below the fold and is "
            "effectively invisible. This variant is the honest test of "
            "whether anyone wants it.")
def v21(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(stat_row(ctx, (("148", "runs"), ("12", "plates"),
                                       ("1.42 M", "objects"),
                                       ("12", "models"), ("4.1 GB", "db"))))
    row_w, row = transparent(horizontal=True, spacing=12)
    row.addWidget(system_panel(ctx, width=300))
    row.addWidget(recent_runs_list(ctx, count=4, width=380))
    row.addWidget(queue_panel(ctx, width=340))
    row.addWidget(whats_new_panel(ctx, width=CONTENT_W - 300 - 380 - 340 - 36,
                                  items=3))
    page.body.addWidget(row_w)
    page.body.addWidget(cat_header(ctx, "Apps", note=str(common.n_apps())))
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(20)
    colw = (CONTENT_W - 2 * 20) // 3
    for title, keys in CATS_BROAD3:
        block, bcol = transparent(spacing=4)
        bcol.addWidget(text_label(ctx, title, size=10, weight=600,
                                  color=ctx.P["fg_dim"], tracking="1.6px",
                                  upper=True))
        bcol.addWidget(dense_list(ctx, keys, width=colw, name_width=136,
                                  show_blurb=False))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 22
# ---------------------------------------------------------------------------

@variant(
    "a-to-z", "A-to-Z index",
    changes="No categories, no ranking: an alphabetical index with "
            "letter headers, three columns, descriptions on every row.",
    adds="Letter headers.",
    removes="All five categories, the hero, the dashboard, the reserved "
            "surface, the hint bar.",
    argument="Alphabetical is the only order that never needs "
            "maintaining and never surprises anyone. If a user knows the "
            "app's name — and after a week they all do — it is the "
            "fastest possible lookup.")
def v22(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=f"all {common.n_apps()} apps, A to Z",
        actions=(("Search…", False),)))
    keys = alphabetical()
    thirds = [keys[0:10], keys[10:20], keys[20:]]
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(20)
    colw = (CONTENT_W - 2 * 20) // 3
    for group in thirds:
        block, bcol = transparent(spacing=2)
        letter = ""
        for key in group:
            first = name_of(key)[0].upper()
            if first != letter:
                letter = first
                head = text_label(ctx, letter, size=12, weight=700,
                                  color=ctx.P["accent"], tracking="1px")
                head.setContentsMargins(6, 6, 0, 0)
                bcol.addWidget(head)
            bcol.addWidget(DenseRow(ctx, key, width=colw, name_width=142))
        bcol.addStretch(1)
        crow.addWidget(block)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 23
# ---------------------------------------------------------------------------

@variant(
    "illustrated-tiles", "Large illustrated tiles, five stage bands",
    changes="Tiles get much bigger and the icon does the work: seven "
            "per row, icon over name, grouped in five stage bands.",
    adds="Nothing.",
    removes="Descriptions from the surface (they stay in the tooltip and "
            "the hint bar), the dashboard, the reserved surface.",
    argument="This is the launcher reading of the home screen — a big, "
            "quiet, recognisable target per app. It is also the variant "
            "that most rewards the icon work happening in parallel; "
            "with weak icons it is the worst of the thirty.")
def v23(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(28, 14, 28, 10), spacing=8)
    for title, keys in CATS_STAGE5:
        page.body.addWidget(text_label(ctx, title, size=10, weight=600,
                                       color=ctx.P["fg_dim"],
                                       tracking="1.8px", upper=True))
        page.body.addWidget(big_tile_grid(ctx, keys, cols=7, width=186,
                                          height=118, icon_px=48,
                                          hspace=10, vspace=8))
    page.body.addStretch(1)
    return page.finish(footer=hint_bar(ctx))


# ---------------------------------------------------------------------------
# 24
# ---------------------------------------------------------------------------

@variant(
    "command-palette", "Keyboard-first command palette",
    changes="The home screen *is* the command palette: a query field "
            "over a two-column result list, every row carrying its "
            "keyboard shortcut, ordered by usage rather than category.",
    adds="Visible Ctrl+1..9 shortcuts on the nine core apps, and a "
            "recent-commands block at the top of the list.",
    removes="All categories, tiles, the hero, the dashboard, the "
            "reserved surface, the hint bar.",
    argument="Everyone who uses spaCR daily ends up wanting Ctrl+K. "
            "Making the home screen the palette means the beginner and "
            "the expert are using the same surface, and the shortcuts "
            "teach themselves.")
def v24(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(120, 26, 120, 16), spacing=14)
    inner_w = 1440 - 240
    box = search_box(ctx, "Type a command —  mask, resume, κ, plate heatmap …",
                     big=True)
    page.body.addWidget(box)
    hintw, hrow = transparent(horizontal=True, spacing=8)
    hrow.addWidget(text_label(ctx, "Recent", size=10, weight=600,
                              color=ctx.P["fg_dim"], tracking="1.6px",
                              upper=True))
    for key, plate, when, ok, _e in MOCK["recent"][:3]:
        hrow.addWidget(chip(ctx, f"{name_of(key)} · {plate}"))
    hrow.addStretch(1)
    hrow.addWidget(text_label(ctx, "↑↓ to move · ⏎ to open", size=11,
                              color=ctx.P["fg_dim"]))
    page.body.addWidget(hintw)

    order = by_frequency()
    sc = _shortcuts()
    cols = QWidget()
    cols.setObjectName("Transparent")
    crow = QHBoxLayout(cols)
    crow.setContentsMargins(0, 0, 0, 0)
    crow.setSpacing(18)
    colw = (inner_w - 18) // 2
    for group in (order[:15], order[15:]):
        crow.addWidget(dense_list(ctx, group, width=colw, name_width=150,
                                  shortcuts=sc, spacing=1), 0)
    page.body.addWidget(cols)
    page.body.addStretch(1)
    return page.finish(status=f"{common.n_apps()} commands")


# ---------------------------------------------------------------------------
# 25
# ---------------------------------------------------------------------------

@variant(
    "project-home", "Home is the project, navigation is the sidebar",
    changes="The home screen stops being a launcher altogether. The "
            "sidebar already lists every app; home becomes the page "
            "about your data — project, queue, recent runs, system, "
            "what changed.",
    adds="A project header with the dataset's size and database, a "
            "queue panel, a recent-runs list, a system panel, a "
            "what's-new panel.",
    removes="Every app tile and every category from the home surface — "
            f"all {common.n_apps()} apps are reachable only from the "
            "sidebar or Ctrl+K.",
    argument=f"Two navigation surfaces listing the same {common.n_apps()} "
            "apps is one "
            "too many, and the sidebar is the one that is available from "
            "every screen. Deleting the duplicate is the largest "
            "simplification available.",
    notes="Shows the real Sidebar, and therefore shows that it does not "
          "fit in 900 px — it needs a scroll area before this variant is "
          "viable.")
def v25(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(26, 20, 26, 16), spacing=14)
    page.add_rail(real_sidebar(ctx))
    page.body.addWidget(project_status_strip(ctx))
    row_w, row = transparent(horizontal=True, spacing=14)
    row.addWidget(queue_panel(ctx, width=300))
    row.addWidget(recent_runs_list(ctx, count=4, width=330))
    row.addWidget(system_panel(ctx, width=300))
    row.addStretch(1)
    page.body.addWidget(row_w)
    page.body.addWidget(cat_header(ctx, "Pick up where you left off"))
    page.body.addWidget(resume_banner(ctx))
    page.body.addWidget(whats_new_panel(ctx, width=640, items=4))
    page.body.addStretch(1)
    return page.finish(status=f"Project: {MOCK['project']}")


# ---------------------------------------------------------------------------
# 26
# ---------------------------------------------------------------------------

@variant(
    "pins-recent-accordion", "Pins, recents, and everything else collapsed",
    changes="Two strips the user cares about sit open — pinned apps and "
            f"recent runs — and the whole {common.n_apps()}-app taxonomy "
            "collapses into five closed accordion rows underneath.",
    adds="A pinned strip and a recent-runs strip; the categories become "
            "the real collapsible Section widget from the settings "
            "screens.",
    removes="Every app that is not pinned disappears from the surface "
            "until a section is opened; the hero, the dashboard, the "
            "reserved surface.",
    argument="It makes the page's default state small without deleting "
            "anything, and it reuses a widget spaCR already ships, so "
            "there is nothing new to design.")
def v26(ctx: Ctx) -> QWidget:
    from spacr.qt.widgets.section import Section
    page = Page(ctx, margins=MARGINS, spacing=12)
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=MOCK["project"],
        actions=(("Resume", True), ("Search…", False))))
    page.body.addWidget(cat_header(ctx, "Pinned"))
    page.body.addWidget(pinned_row(ctx, PINNED, tile_w=162, tile_h=110))
    page.body.addWidget(cat_header(ctx, "Recent"))
    page.body.addWidget(recent_runs_strip(ctx, count=3, card_width=336))
    page.body.addWidget(cat_header(ctx, "All apps"))
    for i, (title, keys) in enumerate(cats_current()):
        # QToolButton reads a lone "&" as a mnemonic and swallows it
        # ("Data & batch runs" renders as "Data _batch runs").
        sec = Section(f"{title.replace('&', '&&')}  ({len(keys)})")
        sec.add_widget(dense_list(ctx, keys, width=CONTENT_W - 60,
                                  name_width=150))
        if i == 0:
            sec.set_expanded(False)
        page.body.addWidget(sec)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 27
# ---------------------------------------------------------------------------

@variant(
    "accordion-eight", "Eight accordions, one open",
    changes="Nothing but headings, in eight narrow categories, using the "
            "shipped collapsible Section widget. One is open; the rest "
            "are one line each.",
    adds="Per-category counts, and the memory of which section you last "
            "had open.",
    removes="Tiles, the hero, the dashboard, the reserved surface, the "
            f"hint bar. All but the open group's apps are one click away "
            "rather than on screen.",
    argument="The whole taxonomy fits in about 300 px, so the home "
            "screen can be small *and* complete. It also scales: a "
            "ninth category costs 34 px, not a whole row.")
def v27(ctx: Ctx) -> QWidget:
    from spacr.qt.widgets.section import Section
    page = Page(ctx, margins=(160, 24, 160, 16), spacing=10)
    inner = 1440 - 320
    page.body.addWidget(parts.top_bar(
        ctx, subtitle=f"{common.n_apps()} apps in "
                      f"{len(CATS_NARROW8)} groups"))
    for i, (title, keys) in enumerate(CATS_NARROW8):
        sec = Section(f"{title.replace('&', '&&')}  ({len(keys)})")
        sec.add_widget(dense_list(ctx, keys, width=inner - 60,
                                  name_width=150))
        sec.set_expanded(i == 0)
        page.body.addWidget(sec)
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 28
# ---------------------------------------------------------------------------

@variant(
    "grid-no-chrome", "Nothing but the grid",
    changes="The page starts at the tiles. Categories survive only as "
            "four-word captions between the bands; there is no hero, no "
            "logo, no footer, no panels.",
    adds="Nothing at all.",
    removes="The hero and wordmark, the insights dashboard, the reserved "
            "surface, the hint bar, and every heading rule.",
    argument="Measured against the complaint that started this — too "
            "much on the home page — this is the answer with the least "
            f"on it that still shows all {common.n_apps()} apps. "
            "Everything on screen is "
            "clickable.")
def v28(ctx: Ctx) -> QWidget:
    page = Page(ctx, chrome=True, margins=(30, 22, 30, 18), spacing=10)
    for title, keys in cats_current():
        page.body.addWidget(text_label(ctx, title, size=10, weight=600,
                                       color=ctx.P["fg_dim"],
                                       tracking="1.8px", upper=True))
        page.body.addWidget(htile_grid(ctx, keys, cols=5, width=262,
                                       icon_px=40, vspace=6, height=68))
    page.body.addStretch(1)
    return page.finish()


# ---------------------------------------------------------------------------
# 29
# ---------------------------------------------------------------------------

@variant(
    "intent-wizard", "Four intents on the left, their apps on the right",
    changes="Four large intent buttons stack down the left; picking one "
            "fills the right pane with that intent's apps as tiles with "
            "descriptions. It is the rail-and-pane idea with four "
            "buttons instead of a list.",
    adds="Intent buttons carrying a count and a one-line explanation.",
    removes="The five kind-of-thing categories, the hero, the "
            "dashboard, the reserved surface, the hint bar.",
    argument="Four targets is the fewest a person has to choose between, "
            "and each is big enough to hit without aiming. Good for the "
            "occasional user; probably slow for a daily one.")
def v29(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=MARGINS, spacing=14)
    page.body.addWidget(parts.top_bar(ctx,
                                      subtitle="what are you doing today?"))
    body, brow = transparent(horizontal=True, spacing=22)
    left, lcol = transparent(spacing=10)
    explain = {
        "Segment images": "masks for cells, nuclei and pathogens",
        "Measure objects": "intensity, morphology, motility, invasion",
        "Analyse a screen": "classify, map barcodes, regress, embed",
        "Check & share": "QC the plate, then write the report",
    }
    for i, (title, keys) in enumerate(CATS_INTENT4):
        btn = parts.FixedButton(340, 96)
        btn.setObjectName("BigTileAccent" if i == 1 else "BigTile")
        btn.setCursor(Qt.PointingHandCursor)
        col = QVBoxLayout(btn)
        col.setContentsMargins(18, 12, 18, 12)
        col.setSpacing(3)
        col.addWidget(text_label(ctx, title, size=17, weight=600))
        col.addWidget(text_label(ctx, explain[title], size=11, weight=300,
                                 color=ctx.P["fg_muted"]))
        col.addWidget(text_label(ctx, f"{len(keys)} apps", size=10,
                                 weight=600, color=ctx.P["fg_dim"],
                                 tracking="1.2px", upper=True))
        lcol.addWidget(btn)
    lcol.addStretch(1)
    brow.addWidget(left, 0)

    right, rcol = transparent(spacing=10)
    rcol.addWidget(plain_header(ctx, CATS_INTENT4[1][0],
                                f"{len(CATS_INTENT4[1][1])} apps"))
    rcol.addWidget(big_tile_grid(ctx, CATS_INTENT4[1][1], cols=3, width=310,
                                 height=176, icon_px=52, blurb_lines=3))
    rcol.addStretch(1)
    brow.addWidget(right, 1)
    page.body.addWidget(body, 1)
    return page.finish()


# ---------------------------------------------------------------------------
# 30
# ---------------------------------------------------------------------------

@variant(
    "kitchen-sink", "Everything at once (the reference for 'too much')",
    changes="Every element proposed anywhere in this set is on one page "
            "at the same time: brand bar with resume, project status, "
            "pinned row, recent runs, five stage bands of tiles, system, "
            "queue, what's new, hint bar.",
    adds="All of it.",
    removes="Nothing.",
    argument="Not a proposal — a control at the other end from variant "
            "18. It shows exactly how far past 900 px the maximal "
            "reading of 'add elements' goes: the page needs a scrollbar "
            "before the pinned row is fully visible, which is the same "
            "failure the current screen has, only louder.",
    notes="Deliberately scrolls; the render shows the top 900 px only.")
def v30(ctx: Ctx) -> QWidget:
    page = Page(ctx, margins=(0, 0, 0, 0), spacing=0)
    inner = QWidget()
    col = QVBoxLayout(inner)
    col.setContentsMargins(26, 18, 26, 18)
    col.setSpacing(12)
    col.addWidget(parts.top_bar(
        ctx, subtitle=MOCK["project"],
        actions=(("Resume last run", True), ("Search…", False))))
    col.addWidget(project_status_strip(ctx))
    col.addWidget(cat_header(ctx, "Pinned"))
    col.addWidget(pinned_row(ctx, PINNED, tile_w=158, tile_h=108))
    col.addWidget(cat_header(ctx, "Recent"))
    col.addWidget(recent_runs_strip(ctx, count=3, card_width=330))
    for title, keys in CATS_STAGE5:
        col.addWidget(cat_header(ctx, title, note=f"{len(keys)} apps"))
        col.addWidget(htile_grid(ctx, keys, cols=6, width=214, icon_px=36,
                                 name_px=12, height=58))
    col.addWidget(cat_header(ctx, "Your machine"))
    row_w, row = transparent(horizontal=True, spacing=12)
    row.addWidget(system_panel(ctx, width=320))
    row.addWidget(queue_panel(ctx, width=320))
    row.addWidget(whats_new_panel(ctx, width=380, items=4))
    row.addStretch(1)
    col.addWidget(row_w)
    col.addStretch(1)
    page.body.addWidget(scroll_area(inner))
    return page.finish(footer=hint_bar(ctx))
