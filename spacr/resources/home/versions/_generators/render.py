#!/usr/bin/env python
"""Render the thirty Home-screen candidates to PNGs.

Standalone::

    python spacr/resources/home/versions/_generators/render.py
    python .../render.py --only 7 --only 23      # just those two
    python .../render.py --themes dark           # one theme
    python .../render.py --check                 # audit existing PNGs only

Writes, under ``spacr/resources/home/versions/``::

    vNN_<slug>/dark.png   vNN_<slug>/light.png   [vNN_<slug>/space.png]
    _sheet.png            all thirty, numbered, dark
    VARIANTS.md           one paragraph per variant

Every render is deterministic: offscreen Qt, a throwaway QSettings
path, bundled fonts, and fixed mock content (see :mod:`common`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402


# ---------------------------------------------------------------------------
# Layout audit — a variant with clipped text is a bug, not a variant
# ---------------------------------------------------------------------------

def audit(page) -> Dict[str, list]:
    """Inspect a laid-out page for the defects that invalidate a render.

    * elided labels — an ellipsis where the name should be;
    * clipped plain labels — text wider than the box drawing it;
    * visible scrollbars;
    * a layout whose minimum height exceeds the canvas, which means Qt
      squeezed something to make it fit.
    """
    from PySide6.QtWidgets import QLabel, QScrollBar
    from spacr.qt.widgets.eliding import ElidingLabel, ElidingPushButton

    out: Dict[str, list] = {"elided": [], "clipped": [], "scrollbars": [],
                            "overflow": []}
    for w in page.findChildren(ElidingLabel):
        if w.is_elided():
            out["elided"].append(w.full_text())
    for w in page.findChildren(ElidingPushButton):
        if w.is_elided():
            out["elided"].append(w.full_text())
    for w in page.findChildren(QLabel):
        if isinstance(w, ElidingLabel) or w.wordWrap() or w.pixmap():
            continue
        if not w.isVisible() or not w.text():
            continue
        need = w.sizeHint()
        if need.width() > w.width() + 1 or need.height() > w.height() + 1:
            out["clipped"].append(
                f"{w.text()[:40]!r} needs {need.width()}x{need.height()}, "
                f"has {w.width()}x{w.height()}")
    for sb in page.findChildren(QScrollBar):
        if sb.isVisible():
            out["scrollbars"].append(
                f"{sb.orientation().name} in "
                f"{sb.parent().__class__.__name__}")
    lay = page.layout()
    if lay is not None:
        need = lay.minimumSize()
        if need.height() > common.CANVAS_H + 1:
            out["overflow"].append(
                f"layout needs {need.height()} px of height, canvas is "
                f"{common.CANVAS_H}")
        if need.width() > common.CANVAS_W + 1:
            out["overflow"].append(
                f"layout needs {need.width()} px of width, canvas is "
                f"{common.CANVAS_W}")
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_one(app, spec: dict, theme: str, out_path: str) -> Dict[str, list]:
    """Build one variant in one theme, grab it, save it. Returns its audit."""
    ctx = common.Ctx(app, theme)
    ctx.apply_theme()
    page = spec["build"](ctx)
    page.resize(common.CANVAS_W, common.CANVAS_H)
    page.show()
    for _ in range(4):
        app.processEvents()
    page.setFixedSize(common.CANVAS_W, common.CANVAS_H)
    for _ in range(3):
        app.processEvents()

    report = audit(page)
    pix = page.grab()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not pix.save(out_path, "PNG"):
        raise RuntimeError(f"could not write {out_path}")
    page.hide()
    page.setParent(None)
    page.deleteLater()
    # processEvents() does NOT drain DeferredDelete, so without this the
    # thirty pages (and their few thousand child widgets) all stay alive
    # for the whole run and each successive render gets slower.
    from PySide6.QtCore import QEvent
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    return report


def variant_dir(spec: dict) -> str:
    """``vNN_<slug>`` directory for a variant, absolute."""
    return os.path.join(common.versions_dir(),
                        f"v{spec['n']:02d}_{spec['slug']}")


#: Where the per-render audit is cached, so ``--md-only`` can rewrite
#: VARIANTS.md after a prose edit without re-rendering ninety PNGs.
AUDIT_CACHE = "_audit.json"


def render_all(app, specs: Sequence[dict], themes: Sequence[str]
               ) -> Dict[Tuple[int, str], Dict[str, list]]:
    """Render every (variant, theme) pair. Returns the audit per pair."""
    reports: Dict[Tuple[int, str], Dict[str, list]] = load_audit()
    for spec in specs:
        for theme in themes:
            path = os.path.join(variant_dir(spec), f"{theme}.png")
            reports[(spec["n"], theme)] = render_one(app, spec, theme, path)
            flags = {k: v for k, v in reports[(spec["n"], theme)].items() if v}
            state = "ok" if not flags else ", ".join(
                f"{k}={len(v)}" for k, v in flags.items())
            print(f"  v{spec['n']:02d} {theme:<6} {spec['slug']:<24} {state}")
    save_audit(reports)
    return reports


def _audit_path() -> str:
    return os.path.join(common.here(), AUDIT_CACHE)


def save_audit(reports: Dict[Tuple[int, str], Dict[str, list]]) -> None:
    """Persist the audit, keyed ``"<n>|<theme>"`` so it round-trips JSON."""
    import json
    payload = {f"{n}|{theme}": rep for (n, theme), rep in reports.items()}
    with open(_audit_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)


def load_audit() -> Dict[Tuple[int, str], Dict[str, list]]:
    """The cached audit, or an empty dict when there is none."""
    import json
    try:
        with open(_audit_path(), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for key, rep in payload.items():
        n, _, theme = key.partition("|")
        out[(int(n), theme)] = rep
    return out


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def build_sheet(specs: Sequence[dict], theme: str = "dark",
                cols: int = 5, thumb_w: int = 440) -> str:
    """Compose all thirty renders into one numbered grid. Returns the path."""
    from PIL import Image, ImageDraw

    thumb_h = int(round(thumb_w * common.CANVAS_H / common.CANVAS_W))
    pad, label_h = 16, 26
    rows = (len(specs) + cols - 1) // cols
    width = cols * thumb_w + (cols + 1) * pad
    height = rows * (thumb_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (width, height), "#0b0c0e")
    draw = ImageDraw.Draw(sheet)
    font = _sheet_font(15)

    for i, spec in enumerate(specs):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + label_h + pad)
        src = os.path.join(variant_dir(spec), f"{theme}.png")
        if os.path.isfile(src):
            with Image.open(src) as im:
                sheet.paste(im.convert("RGB").resize(
                    (thumb_w, thumb_h), Image.LANCZOS), (x, y + label_h))
        draw.rectangle([x, y + label_h, x + thumb_w - 1,
                        y + label_h + thumb_h - 1], outline="#2a2d33")
        draw.text((x, y + 4), f"{spec['n']:02d}  {spec['title']}",
                  fill="#ffffff", font=font)
    out = os.path.join(common.versions_dir(), "_sheet.png")
    sheet.save(out)
    return out


def _sheet_font(size: int):
    from PIL import ImageFont
    path = os.path.join(common.repo_root(), "spacr", "qt", "resources",
                        "fonts", "OpenSans-SemiBold.ttf")
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# VARIANTS.md
# ---------------------------------------------------------------------------

_INTRO = """# Thirty Home-screen arrangements

Candidates for review. **Nothing here is installed** — no file under
`spacr/qt/` was changed to produce them.

Every screen below is built out of the **real Qt widgets** (`HTile`,
`Card`, `Section`, `Divider`, `UsageBar`, `ElidingLabel`, and the real
`Sidebar`/`HomePage` in variant 01) and the **real app registry**
(`spacr.qt.app.APPS`, all {n_apps} apps, unmodified names and blurbs), then
grabbed with `QWidget.grab()` under `QT_QPA_PLATFORM=offscreen`. Where a
variant needs something spaCR does not have yet — a recent-runs strip, a
resume banner, a guided quick-start, a project status bar, a what's-new
panel, a big illustrated tile — that widget was built for real in
`_generators/parts.py`. So whatever you pick is known-buildable.

**Canvas: 1440x900**, the realistic laptop case, including the app's
menu strip and status bar so the space a variant actually gets is what
is drawn. Variants that also show the app sidebar say so.

**Themes:** `dark.png` and `light.png` in every folder{space_note}. The
Space renders use that theme's *offline* sky — the deep-space gradient
it falls back to when no generated star image is cached — because the
generated image is per-user and would make these renders
non-reproducible.

**Numbers in the panels are fixed mock values** (`_generators/common.py`,
`MOCK`) — plate counts, run history, disk and GPU. A screen that renders
differently every run cannot be reviewed.

Re-render, or tweak one and re-render just that one:

```bash
R=spacr/resources/home/versions/_generators/render.py
python $R                        # all thirty, every theme
python $R --only 7 --themes dark # just variant 7, just dark
python $R --md-only              # rewrite this file after a prose edit
python $R --check                # audit the PNGs already on disk
```

The variants live in `_generators/variants.py`, one function each; the
widgets they are assembled from live in `_generators/parts.py`.

## The contact sheet

![all thirty](_sheet.png)

## Findings that apply to every variant

1. **The sidebar still does not fit at 1440x900 — but it scrolls now.**
   Its {n_apps} app rows plus five headings ask for roughly {sidebar_h} px
   against the {sidebar_avail} a laptop gives. The `QScrollArea` described
   at the bottom of this file **has since landed** in `spacr/qt/app.py`,
   so the rows scroll and nothing is unreachable; the vertical scrollbar
   variants 01 and 25 show is that fix working, not the old defect.
2. **The shipped home page needs a vertical scrollbar** before the
   last band is fully on screen (variant 01).
   {scroll_finding}
3. **The hint bar exists because descriptions are hidden.** Any variant
   that shows the one-line description on the row itself (04, 07, 08,
   09, 10, 13, 19, 22, 24, 29) does not need it.
4. **`HTile` cannot do more than five columns on this screen.** Its name
   is drawn at the 17 px "subtitle" size, so the longest app name needs
   about 255 px of tile — 5 x 265 px fits 1440, 6 does not, and at six
   columns the name silently elides. Variants 05, 17, 20 and 30 restyle
   that one label to 12-13 px to get six columns and 02 goes to 11 px to
   get seven; 03, 07, 08, 23 and 28 keep the shipped size and use five or
   fewer. Both are legitimate, but it is a real constraint on any
   tile-grid answer — and it tightens every time a longer app name is
   registered, which "Classifier Evaluation" duly did.

---

"""

_OUTRO = """
---

## Notes on size

Everything above is drawn at 1440x900 and nothing depends on a larger
window, with these exceptions:

* **07 rail-and-pane**, **08 tabs** and **29 intent-wizard** show one
  category at a time with four to five tiles per row. On a wider screen
  they simply fit more per row; on a narrower one the grid rewraps.
* **13 dense-two-column** and **19 by-question** are two columns of
  ~660 px. Below about 1200 px they would want to become one column.
* **24 command-palette** and **27 accordion-eight** are deliberately
  inset (a fixed centre column), so they look the same at any width
  above 1100 px.
* **30 kitchen-sink** does not fit at any realistic size — that is its
  point.

## The product change these renders argued for — and it landed

This section used to carry a proposed diff, because `spacr/qt/app.py`
belonged to another effort at the time. That effort has since made the
change: `Sidebar` now puts its rows in a `QScrollArea` with the title
pinned above it. The measurement is re-taken on every render, and it
still says the same thing about *why* the scroll area has to be there —
the {n_apps} app rows plus five headings ask for ~{sidebar_h} px against
the ~{sidebar_avail} a 1440x900 laptop gives, so without it the last
three apps ({last_three}) could not be reached at all.

What that means for these renders: the vertical scrollbar variants 01
and 25 report in their layout audit is the **fix working**. It is not
the defect this file was raised to record, and any future variant that
keeps the sidebar inherits the scroll area rather than the problem.

## One trap for whoever implements the winner

`QPushButton.setFixedSize()` does **not** survive the app stylesheet.
`theme.stylesheet()` carries `QPushButton { min-height: 22px }`, and
`QStyleSheetStyle` re-applies that rule's geometry to the widget on
polish, wiping the minimum that `setFixedSize` had set. A 116 px tile
then reports a 40 px minimum to its parent layout and gets squashed to
48 px — text and icon still painted, just cropped, with no warning
anywhere. Two of these variants hit it before it was found. The fix is
to report the size through `sizeHint()`/`minimumSizeHint()` instead;
see `FixedButton` in `_generators/parts.py`. (`HTile` is already safe
because it overrides both.)
"""


_CLEAN = ("clean — no elided or clipped text, no scrollbar, "
          "fits 1440x900.")


def _fill_outro(sidebar_h: int, sidebar_avail: int) -> str:
    """``_OUTRO`` with the registry-derived numbers substituted in.

    ``str.replace`` rather than ``str.format``: the outro quotes a QSS
    rule (``QPushButton { min-height: 22px }``) and a ``format`` call
    would read those braces as a field and raise. The placeholders are
    still filled from the live registry, because the version of this
    text that hardcoded "29 app rows ... the last three apps (Plaque
    Assay, Recruitment, Invasion Assay)" named the wrong three the
    moment Replication Assay was registered.
    """
    last_three = ", ".join(common.name_of(k) for k in common.all_keys()[-3:])
    return (_OUTRO
            .replace("{n_apps}", str(common.n_apps()))
            .replace("{sidebar_h}", str(sidebar_h))
            .replace("{sidebar_avail}", str(sidebar_avail))
            .replace("{last_three}", last_three))


def _scroll_finding(specs: Sequence[dict],
                    reports: Dict[Tuple[int, str], Dict[str, list]]) -> str:
    """Finding 2's second sentence, **counted from the audit**.

    This paragraph was rewritten to stop hardcoding the app count and
    promptly typed a different one — "Twenty-seven of the thirty
    variants below fit 1440x900 with no scrollbar at all" — into the
    same breath. The audit is the only thing that knows: a variant
    starts needing a scrollbar the moment a longer name or one more app
    pushes it over 900 px, and nobody edits prose when that happens.

    When the audit does not cover every variant it says so, instead of
    dividing by a total it never looked at. (``--only 7`` normally still
    has the full picture — :func:`render_all` seeds ``reports`` from the
    cached ``_audit.json`` and overwrites only the pairs it re-rendered
    — so that branch is really for a missing or truncated cache.)
    Quietly reporting "29 of 30 fit" off one rendered variant is exactly
    the plausible-wrong-number failure this file exists to avoid.
    """
    import textwrap

    numbers = [spec["n"] for spec in specs]
    titles = {spec["n"]: spec["title"] for spec in specs}
    measured = {n for (n, _theme) in reports}
    scrolling = sorted({n for (n, _theme), flags in reports.items()
                        if flags.get("scrollbars")})

    def _listed(ns):
        return ", ".join(f"{n:02d} ({titles.get(n, '?')})" for n in ns)

    def _wrapped(sentence: str) -> str:
        # Re-wrapped to the width the surrounding hand-written findings
        # use, continuation lines under the list item's hanging indent.
        # Substituting one very long line into a numbered list turns the
        # whole findings block into something nobody re-reads.
        return textwrap.fill(sentence, width=72,
                             subsequent_indent="   ").lstrip()

    unmeasured = [n for n in numbers if n not in measured]
    if unmeasured:
        return _wrapped(
            f"This pass only audited {len(measured)} of the "
            f"{len(numbers)} variants, so the usual "
            "how-many-fit-without-a-scrollbar count is not stated here "
            "— re-render without `--only` to restore it. Of the ones it "
            "did audit, " + (f"{_listed(scrolling)} need a scrollbar."
                             if scrolling else "none need a scrollbar."))
    if not scrolling:
        return _wrapped(
            f"All {len(numbers)} variants below fit 1440x900 with no "
            "scrollbar at all, so scrolling the Home screen is a "
            "choice, not a constraint.")
    return _wrapped(
        f"{len(numbers) - len(scrolling)} of the {len(numbers)} "
        "variants below fit 1440x900 with no scrollbar at all — the "
        f"{len(scrolling)} that do not are {_listed(scrolling)} — so "
        "scrolling the Home screen is a choice, not a constraint.")


def _audit_sentence(spec: dict, themes: Sequence[str],
                    reports: Dict[Tuple[int, str], Dict[str, list]]) -> str:
    """One line summarising a variant's audit across the rendered themes."""
    per_theme = {}
    for theme in themes:
        rep = reports.get((spec["n"], theme))
        if rep is None:
            continue
        per_theme[theme] = tuple(sorted(f"{k} ({len(v)})"
                                        for k, v in rep.items() if v))
    if not per_theme:
        return "not re-rendered in this pass."
    distinct = set(per_theme.values())
    if distinct == {()}:
        return _CLEAN
    if len(distinct) == 1:
        return "every theme — " + ", ".join(next(iter(distinct)))
    return "; ".join(f"{t}: {', '.join(v) or 'clean'}"
                     for t, v in per_theme.items())


def write_markdown(specs: Sequence[dict], themes: Sequence[str],
                   reports: Dict[Tuple[int, str], Dict[str, list]],
                   sidebar_h: int, sidebar_avail: int) -> str:
    """Write ``VARIANTS.md``. Returns its path."""
    space_note = (", plus `space.png`" if "space" in themes
                  else " (the `space` palette was not available when these "
                       "were rendered)")
    # n_apps is read, never typed: this document said "29 apps" for long
    # enough that five more were registered without anyone noticing. The
    # scrollbar tally is read too, for the same reason and out of the
    # `reports` this function was already handed.
    lines = [_INTRO.format(space_note=space_note, sidebar_h=sidebar_h,
                           sidebar_avail=sidebar_avail,
                           n_apps=common.n_apps(),
                           scroll_finding=_scroll_finding(specs, reports))]
    for spec in specs:
        folder = os.path.basename(variant_dir(spec))
        lines.append(f"### {spec['n']:02d} · {spec['title']}\n")
        lines.append(f"`{folder}/`\n")
        imgs = " ".join(
            f"[{t}]({folder}/{t}.png)" for t in themes
            if os.path.isfile(os.path.join(variant_dir(spec), f"{t}.png")))
        lines.append(f"{imgs}\n")
        lines.append(f"![{spec['title']}]({folder}/dark.png)\n")
        lines.append(f"**Changes.** {spec['changes']}\n")
        lines.append(f"**Adds.** {spec['adds']}\n")
        lines.append(f"**Removes.** {spec['removes']}\n")
        lines.append(f"**The argument for it.** {spec['argument']}\n")
        if spec["notes"]:
            lines.append(f"*Note.* {spec['notes']}\n")
        lines.append(f"*Layout audit: {_audit_sentence(spec, themes, reports)}*\n")
        lines.append("")
    lines.append(_fill_outro(sidebar_h, sidebar_avail))
    out = os.path.join(common.versions_dir(), "VARIANTS.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def self_check(specs: Sequence[dict], themes: Sequence[str]) -> dict:
    """Verify every PNG exists, is 1440x900, and is not near-uniform."""
    import numpy as np
    from PIL import Image

    rows = []
    for spec in specs:
        for theme in themes:
            path = os.path.join(variant_dir(spec), f"{theme}.png")
            if not os.path.isfile(path):
                rows.append({"path": path, "ok": False, "why": "missing"})
                continue
            with Image.open(path) as im:
                size = im.size
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            # Pack RGB into one int32 before counting: np.unique(axis=0)
            # over 1.3 M rows takes seconds per image, and there are
            # ninety images.
            packed = ((arr[..., 0].astype(np.int32) << 16)
                      | (arr[..., 1].astype(np.int32) << 8)
                      | arr[..., 2].astype(np.int32))
            uniq = int(np.unique(packed).size)
            std = float(arr.std())
            ok = (size == (common.CANVAS_W, common.CANVAS_H)
                  and uniq >= 64 and std >= 3.0)
            rows.append({"path": path, "ok": ok, "size": size,
                         "unique": uniq, "std": round(std, 2),
                         "bytes": os.path.getsize(path),
                         "why": "" if ok else "blank or wrong size"})
    sheet = os.path.join(common.versions_dir(), "_sheet.png")
    sheet_ok = os.path.isfile(sheet)
    return {"rows": rows, "sheet": sheet, "sheet_ok": sheet_ok}


def measure_sidebar(app) -> Tuple[int, int]:
    """``(height the Sidebar's rows need, height a 1440x900 window gives)``.

    Measures the *scrolled content*, not the widget. The QScrollArea
    these renders argued for has since landed in ``spacr.qt.app``, so
    ``Sidebar.layout().minimumSize()`` is now about 85 px — the height
    of a title over a collapsible viewport — and says nothing at all
    about whether the navigation fits. The number that still means
    something is how tall the rows inside the viewport are.
    """
    ctx = common.Ctx(app, "dark")
    ctx.apply_theme()
    from spacr.qt.app import Sidebar
    bar = Sidebar()
    bar.resize(bar.width(), 850)
    bar.show()
    app.processEvents()
    # Private attribute on purpose: there is no public accessor for the
    # scrolled widget, and falling back to the outer layout keeps this
    # working (with a smaller number) if the scroll area is ever removed.
    scroll = getattr(bar, "_scroll", None)
    inner = scroll.widget() if scroll is not None else None
    layout = inner.layout() if inner is not None else bar.layout()
    need = layout.minimumSize().height()
    bar.hide()
    bar.setParent(None)
    bar.deleteLater()
    app.processEvents()
    # 900 window - 26 menu strip - 24 status bar
    return int(need), common.CANVAS_H - 26 - 24


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", type=int, default=None,
                    help="render only these variant numbers (repeatable)")
    ap.add_argument("--themes", action="append", default=None,
                    help="themes to render (default: every available one)")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--no-md", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="only run the self-check over existing PNGs")
    ap.add_argument("--md-only", action="store_true",
                    help="rewrite VARIANTS.md from the cached audit "
                         "(use after editing a variant's prose)")
    args = ap.parse_args(argv)

    app = common.bootstrap()
    import variants

    specs = variants.VARIANTS
    themes = tuple(args.themes) if args.themes else common.available_themes()

    if args.check:
        _report_check(self_check(specs, themes))
        return 0

    if args.md_only:
        need, avail = measure_sidebar(app)
        print("markdown:", write_markdown(specs, themes, load_audit(),
                                          need, avail))
        return 0

    chosen = ([s for s in specs if s["n"] in set(args.only)]
              if args.only else specs)
    print(f"themes: {', '.join(themes)}")
    print(f"variants: {len(chosen)} of {len(specs)}")
    _prune_stale_dirs(specs)
    reports = render_all(app, chosen, themes)

    if not args.no_sheet:
        print("sheet:", build_sheet(specs))
    if not args.no_md:
        need, avail = measure_sidebar(app)
        # A partial run keeps the prose for every variant, but only the
        # audit lines for the ones just rendered.
        print("markdown:", write_markdown(specs, themes, reports, need, avail))
    _report_check(self_check(specs, themes))
    return 0


def _prune_stale_dirs(specs: Sequence[dict]) -> None:
    """Delete ``vNN_*`` folders that no longer match a registered variant."""
    keep = {os.path.basename(variant_dir(s)) for s in specs}
    root = common.versions_dir()
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if (os.path.isdir(path) and name.startswith("v")
                and name not in keep and not name.startswith("_")):
            shutil.rmtree(path)
            print(f"  pruned stale {name}")


def _report_check(result: dict) -> None:
    rows = result["rows"]
    bad = [r for r in rows if not r["ok"]]
    stds = [r["std"] for r in rows if r.get("std") is not None]
    uniq = [r["unique"] for r in rows if r.get("unique") is not None]
    print("\nself-check")
    print(f"  PNGs:            {len(rows)} checked, {len(rows) - len(bad)} ok")
    print(f"  all 1440x900:    "
          f"{all(r.get('size') == (common.CANVAS_W, common.CANVAS_H) for r in rows)}")
    if stds:
        print(f"  pixel std dev:   min {min(stds)}  max {max(stds)}")
        print(f"  unique colours:  min {min(uniq)}  max {max(uniq)}")
    print(f"  contact sheet:   {'written' if result['sheet_ok'] else 'MISSING'}")
    for r in bad:
        print(f"  FAIL {r['path']}: {r['why']}")


if __name__ == "__main__":
    raise SystemExit(main())
