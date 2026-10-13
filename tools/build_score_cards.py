"""Draw one SVG score card per published spaCR model.

A score card is the graphic at the top of each ``docs/models/<key>.md`` page
and in the README's model zoo grid. Every card has the same size and the
same internal grid, so a grid of cards lines up edge to edge:

* a title slot: the family's short name, and the version as a small tag;
* a headline slot: the main hold-out score as one large number, the metric
  and hold-out size beside it, and a chip with the difference from the
  baseline (stock Cellpose-SAM, or the earlier detector);
* exactly ``ROWS`` bar rows: this model in spaCR blue, the baseline in a
  cool slate under it; a row the record has no number for stays empty;
* a legend.

Nothing else is on the card face; the hold-out details, caveats and re-scores
are on the Markdown page the card links to.

Every number is read from the zoo record in ``spacr/model_zoo.py``
(``BUNDLED_REMOTE_MODELS``): the structured ``metrics`` entries first, then
the record's metric strings, then its ``versus_stock`` sentence and notes.
Nothing numeric is typed here, so a re-score published to the zoo reaches the
cards when ``python tools/build_model_cards.py`` is run again; that tool
writes the cards next to the Markdown pages.

The card is one dark surface in GitHub dark mode's own neutrals with a
coloured rim (teal by default, magenta as the alternative), the same in both
GitHub themes and on PyPI; spaCR blue marks this model's bars. Each card is
drawn for the README's phone and desktop rows at its true size, so its text
is the page's one text size. A transparent margin makes the gutters of the
README grid, and a faint shadow sets the card off a white page.
"""
from __future__ import annotations

import base64
import functools
import io
import re
from html import escape
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple)

CARD_SUBDIR = "cards"
HEADING_NAME = f"{CARD_SUBDIR}/heading.svg"
HEADING_TEXT = "Default models · held-out scores"

#: Every card is drawn twice, at its true display size in CSS pixels: half
#: the README's desktop row (an 830 px column) and half its phone row (a
#: 358 px column), two cards to a row in both. One SVG unit is one CSS
#: pixel, so every word on a card is the page's one text size, GitHub's
#: code-block text (13.6 px); the big number is the one exception. The
#: ``<key>.svg`` card is the desktop one (the model pages show it too);
#: ``<key>.phone.svg`` is the phone one.
LAYOUTS = {"desktop": 414, "phone": 178}
WIDTH = LAYOUTS["desktop"]
#: Half the gutter between two cards on each side, and the space under a
#: card, which with the line box's descent makes the gap between rows.
MARGIN_X = 4
MARGIN_Y = 1
MARGIN_BOTTOM = 4
CARD_W = WIDTH - 2 * MARGIN_X
#: Card heights per layout; every card of a layout has the same.
CARD_HEIGHTS = {"desktop": 222, "phone": 272}
CARD_H = CARD_HEIGHTS["desktop"]
HEIGHT = CARD_H + MARGIN_Y + MARGIN_BOTTOM
HEADING_H = 38
HEADING_GAP = 4
TEXT = 13.6
TITLE = 16
ROWS = 3
FONT = ("'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "'Noto Sans', Helvetica, Arial, sans-serif")
#: Open Sans Regular for the title and the version tag, Open Sans Light for
#: everything else. An SVG shown through ``<img>`` cannot fetch a web font,
#: so each card embeds the two faces, subset to the characters it draws.
TITLE_WEIGHT = 400
TEXT_WEIGHT = 300
FONT_DIR = (Path(__file__).resolve().parent.parent / "docs" / "resources"
            / "font" / "open_sans" / "static")
FONT_FILES = {400: "OpenSans-Regular.ttf", 300: "OpenSans-Light.ttf"}
#: The embedded fonts as they appear in a card, for comparisons that must
#: not depend on the fontTools version that subset them.
FONT_DATA = re.compile(r"url\(data:font/woff;base64,[A-Za-z0-9+/=]+\)")

#: The card surface, edge-free fill, track and inks are GitHub dark mode's
#: neutrals; a coloured rim sets the card off both GitHub themes and PyPI.
#: This model's bars are spaCR blue (``spacr.qt.theme`` accent), the
#: baseline a cool slate. The rim is teal (the logo's) by default; magenta
#: is the alternative the maintainer is choosing between.
PALETTE = {
    "card": "#161b22",
    "edge": "#30363d",
    "track": "#21262d",
    "ink": "#e6edf3",
    "ink2": "#c9d1d9",
    "ink3": "#8b949e",
    "model": "#4a9eff",
    "base": "#566a7e",
    "glow": "#2ec4b6",
    "blue": "#6cb6ff",
    "shadow": "#1f2328",
}
RIMS = {"teal": "#2ec4b6", "magenta": "#d946ef"}
RIM = "teal"

#: Short family names for the title slot, as the maintainer named them.
FAMILY_TITLES: Dict[str, str] = {
    "toxoplasma_pv": "Parasite vacuoles",
    "toxoplasma_plaque": "Plaques",
    "toxoplasma_well_detector": "Plaque wells",
    "live_cell": "Live cells",
    "toxoplasma_from_cellmask": "Vacuoles from cell mask",
    "nuclei_from_cellmask": "Nuclei from cell mask",
    "cell_from_hoechst": "Cells from Hoechst",
    "toxoplasma_from_hoechst": "Vacuoles from Hoechst",
}

#: What each family segments and why the trained model exists. Words only,
#: shown on the Markdown page under the card, never on the card face.
FAMILY_TEXT: Dict[str, Tuple[str, str]] = {
    "toxoplasma_pv": (
        "Toxoplasma parasitophorous vacuoles in fluorescence images",
        "Stock Cellpose-SAM misses or misdraws many vacuoles in infected "
        "host-cell monolayers."),
    "toxoplasma_plaque": (
        "Toxoplasma plaques in crystal-violet plaque-assay wells",
        "Stock Cellpose-SAM was never trained on plaques and finds few of "
        "them, fewest on Gel Doc scans."),
    "toxoplasma_well_detector": (
        "Plaque-assay wells in plate photos and published figures (boxes)",
        "No stock detector has a plaque-well class; the well found first "
        "makes plaque sizes comparable across scanners."),
    "toxoplasma_from_cellmask": (
        "Toxoplasma vacuoles from a whole-cell stain alone",
        "Frees the parasite channel: stock Cellpose-SAM cannot predict one "
        "object type from another stain."),
    "toxoplasma_from_hoechst": (
        "Toxoplasma vacuoles from a Hoechst (nuclear) stain alone",
        "Frees the parasite channel: stock Cellpose-SAM cannot predict one "
        "object type from another stain."),
    "nuclei_from_cellmask": (
        "Nuclei from a whole-cell stain alone",
        "Frees the DAPI or Hoechst channel for another marker; stock "
        "Cellpose-SAM segments the stain it is shown."),
    "cell_from_hoechst": (
        "Host-cell outlines from a Hoechst (nuclear) stain alone",
        "No cell stain needed: the outline comes from the nuclear channel, "
        "which stock Cellpose-SAM cannot do."),
    "live_cell": (
        "Unstained cells in phase contrast, brightfield and DIC",
        "Generalises to transmitted-light data stock was not trained on; "
        "stock still wins on LIVECell-style phase."),
}

STATUS_WORDS = {"default": "default", "alternative": "alternative",
                "candidate": "candidate", "older": "older version"}
TAG_WORDS = {"alternative": "alt", "candidate": "candidate",
             "older": "older"}

LABELS = {
    "f1": "F1", "precision": "Precision", "recall": "Recall", "aji": "AJI",
    "dice": "Dice", "mAP50": "mAP50", "mAP50_95": "mAP50-95",
}
#: Short row labels for an F1 measured on part of the hold-out set.
SUBSET_LABELS = {
    "all test fields": "All fields", "datasets stock never saw":
    "New datasets", "in-domain": "In-domain",
}
SEGMENTER_ORDER = ("f1", "aji", "dice", "precision", "recall")
DETECTOR_ORDER = ("mAP50", "mAP50_95", "recall", "precision")

_NUM = r"(\d*\.\d+)"
_TOKEN = re.compile(r"(?<![\w/-])(mAP50-95|mAP50|F1|AJI|Dice|P|R)\s+" + _NUM)
_TOKEN_NAMES = {"mAP50-95": "mAP50_95", "mAP50": "mAP50", "F1": "f1",
                "AJI": "aji", "Dice": "dice", "P": "precision",
                "R": "recall"}
_TRIPLE = re.compile(
    r"F1 " + _NUM + r", AJI " + _NUM + r", Dice " + _NUM
    + r" at IoU 0\.5 against stock ([\w.]+?)'s " + _NUM + "/" + _NUM + "/"
    + _NUM)
_PR_NOTE = re.compile(r"precision " + _NUM + r" against recall " + _NUM)
_MODALITY = re.compile(r"\b(brightfield|phase|DIC) " + _NUM
                       + r" \(stock " + _NUM + r"\)")
_STOCK_F1 = re.compile(
    r"\bstock ([\w.]+?)(?:'s)? " + _NUM
    + r"(?: there)?(?: with the ([\w -]+?) scorer)?\b")
_HELD_OUT = re.compile(r"([\d][\d,]*)\s+(?:[\w-]+\s+){0,3}?"
                       r"(fields|wells|images|pairs)\b")


RESCORED_FIELDS = (("f1", "object_f1"), ("precision", "precision"),
                   ("recall", "recall"), ("aji", "aji"), ("dice", "dice"))
_OLD_HOLDOUT_FIELDS = ("pixel_iou", "mAP", "ap", "true_positives",
                       "false_positives", "false_negatives", "stock_f1", "stock_aji",
                       "stock_dice", "holdout", "holdout_version")


def rescored(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of a zoo record with the re-scored numbers of its model card.

    The 3-page model card (``spacr/resources/model_cards/<repo>.json``, built
    from the ground-truth re-score of 2026-10-11) holds the pooled strict
    hold-out scores of the model and of the stock model it was fine-tuned
    from, on the same fields. Where a card exists, its Object F1 (IoU 0.5),
    precision, recall, AJI and Dice replace the record's, with the hold-out
    size and the ``versus_stock`` sentence; a record without a card is
    returned unchanged.
    """
    import sys

    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from spacr import model_cards

    out = dict(entry)
    card = model_cards._cards().get(str(entry.get("key") or ""))
    if not card:
        return out
    scores_ = card.get("scores") or {}
    by_key = {row["key"]: row for row in scores_.get("rows") or ()}
    metrics = {name: value for name, value in (entry.get("metrics") or {}).items()
               if name not in _OLD_HOLDOUT_FIELDS
               and name not in {field for field, _ in RESCORED_FIELDS}}
    for field, card_key in RESCORED_FIELDS:
        row = by_key.get(card_key)
        if row is None or row.get("this") is None:
            continue
        mine, stock = float(row["this"]), row.get("stock")
        metrics[field] = {
            "finetuned": mine,
            "vanilla": None if stock is None else float(stock),
            "delta": None if stock is None else mine - float(stock)}
    holdout = scores_.get("holdout") or {}
    counts = scores_.get("counts") or {}
    if counts.get("this"):
        for name, field in (("true_positives", "tp"),
                            ("false_positives", "fp"),
                            ("false_negatives", "fn")):
            if counts["this"].get(field) is not None:
                metrics[name] = counts["this"][field]
    metrics["n_test"] = f"{holdout.get('n_fields')} fields"
    metrics["test_objects"] = f"{int(holdout.get('n_objects') or 0):,}"
    metrics["n_fields"] = holdout.get("n_fields")
    metrics["n_objects"] = holdout.get("n_objects")
    metrics["holdout"] = "gt_rescore_20261011"
    metrics["holdout_version"] = "strict hold-out, pooled"
    out["metrics"] = metrics
    f1, aji = by_key.get("object_f1") or {}, by_key.get("aji") or {}
    stock_name = scores_.get("stock_model") or "stock"
    if f1.get("this") is not None and f1.get("stock") is not None:
        out["versus_stock"] = (
            f"F1 {f1['this']:.3f} against stock {stock_name}'s "
            f"{f1['stock']:.3f} on the {holdout.get('n_fields')} strict "
            f"hold-out fields at IoU 0.5")
        if aji.get("this") is not None and aji.get("stock") is not None:
            out["versus_stock"] += (
                f"; AJI {aji['this']:.3f} against {aji['stock']:.3f}")
    return out


def _key(entry: Mapping[str, Any]) -> str:
    return str(entry.get("key") or entry.get("name") or "")


def _stem(key: str) -> str:
    return re.sub(r"_v\d+$", "", key)


def _version(key: str) -> str:
    match = re.search(r"_v(\d+)$", key)
    return f"v{match.group(1)}" if match else "v1"


def is_detector(entry: Mapping[str, Any]) -> bool:
    return (str(entry.get("kind") or "") == "detector"
            or "yolo" in str(entry.get("architecture") or "").lower())


def _unit(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _parse_text(text: Any, default: str) -> Dict[str, float]:
    """Numbers in one metric string, keyed by metric (or a labelled F1)."""
    text = str(text or "").split(";")[0].strip()
    if not text:
        return {}
    if re.fullmatch(_NUM, text):
        value = _unit(float(text))
        return {default: value} if value is not None else {}
    found: Dict[str, float] = {}
    for label, number in _TOKEN.findall(text):
        name = _TOKEN_NAMES[label]
        value = _unit(float(number))
        if name not in found and value is not None:
            found[name] = value
    if found:
        return found
    parts = [part.strip() for part in text.split(" / ")]
    if len(parts) > 1 and all(re.match(_NUM + r"\s+\S", p) for p in parts):
        for part in parts:
            number, rest = part.split(None, 1)
            rest = re.sub(r"^on\s+", "", rest.strip())
            rest = "all test fields" if rest == "all" else rest
            value = _unit(float(number))
            if value is not None:
                found[f"{default}, {rest}"] = value
        return found
    first = re.search(_NUM, text)
    if first and re.match(r"^v\d+\b", text):
        value = _unit(float(first.group(1)))
        if value is not None:
            found[default] = value
    return found


def _architecture_base(entry: Mapping[str, Any]) -> str:
    match = re.search(r"\((cpsam[\w]*)", str(entry.get("architecture") or ""))
    return match.group(1) if match else "Cellpose-SAM"


def _row_label(name: str) -> str:
    if name.startswith("f1, "):
        rest = name[4:]
        label = SUBSET_LABELS.get(rest, rest[:1].upper() + rest[1:])
        return label if len(label) <= 13 else label[:12].rstrip() + "…"
    return LABELS.get(name, name)


def holdout_size(entry: Mapping[str, Any]) -> str:
    """``"463 held-out fields"``, or ``""`` when the record has no count."""
    metrics = entry.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    n_test = str(metrics.get("n_test") or "").strip()
    unit = "images" if is_detector(entry) else "fields"
    if re.fullmatch(r"[\d,]+", n_test):
        return f"{int(n_test.replace(',', '')):,} held-out {unit}"
    for text in (n_test, str(entry.get("versus_stock") or "")):
        found = _HELD_OUT.search(text)
        if found:
            count = int(found.group(1).replace(",", ""))
            return f"{count:,} held-out {found.group(2)}"
    if isinstance(metrics.get("n_fields"), int):
        return f"{metrics['n_fields']:,} held-out {unit}"
    return ""


def scores(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The card's numbers, all read from the zoo record.

    :returns: ``rows``, exactly ``ROWS`` tuples of (label, this model or None,
        baseline or None), a row of ``None`` being one the record has no
        number for; ``baseline``, the comparison's name; ``caveat``, a note
        on how the baseline was scored; ``holdout``, the hold-out size.
    """
    metrics = entry.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    notes = " ".join(str(note) for note in entry.get("notes") or ())
    versus = str(entry.get("versus_stock") or "")
    this: Dict[str, float] = {}
    base: Dict[str, float] = {}
    baseline = ""
    caveat = ""

    for name, value in metrics.items():
        if isinstance(value, Mapping):
            mine = _unit(value.get("finetuned"))
            if mine is None:
                continue
            this.setdefault(name, mine)
            theirs = _unit(value.get("vanilla"))
            if theirs is not None:
                base.setdefault(name, theirs)

    for field in ("f1", "aji", "dice"):
        mine = _parse_text(metrics.get(field), field)
        for name, value in mine.items():
            this.setdefault(name, value)
        default = next(iter(mine)) if len(mine) == 1 else field
        stock_text = str(metrics.get(f"stock_{field}") or "")
        theirs = _parse_text(stock_text, default)
        for name, value in theirs.items():
            base.setdefault(name, value)
        version = re.match(r"^(v\d+)\b", stock_text)
        if theirs and version and not baseline:
            baseline = f"{version.group(1)} (earlier version)"

    triple = _TRIPLE.search(notes)
    if triple:
        f1, aji, dice, stock, s_f1, s_aji, s_dice = triple.groups()
        for name, value in (("f1", f1), ("aji", aji), ("dice", dice)):
            this.setdefault(name, float(value))
        for name, value in (("f1", s_f1), ("aji", s_aji), ("dice", s_dice)):
            base.setdefault(name, float(value))
        baseline = baseline or f"Stock {stock}"
    pr_note = _PR_NOTE.search(notes)
    if pr_note:
        this.setdefault("precision", float(pr_note.group(1)))
        this.setdefault("recall", float(pr_note.group(2)))
    subsets = [name for name in this if name.startswith("f1, ")]
    if subsets:
        for kind, mine, theirs in _MODALITY.findall(notes):
            this.setdefault(f"f1, {kind}", float(mine))
            base.setdefault(f"f1, {kind}", float(theirs))
            subsets.append(f"f1, {kind}")

    if "f1" in this and "f1" not in base:
        stock = _STOCK_F1.search(versus)
        if stock:
            base["f1"] = float(stock.group(2))
            baseline = baseline or f"Stock {stock.group(1)}"
            if stock.group(3):
                caveat = (f"Stock scored with the {stock.group(3)} scorer, "
                          f"this model through spaCR.")

    if base and not baseline:
        stock = re.search(r"\bstock (cpsam[\w]*)", versus)
        baseline = f"Stock {stock.group(1) if stock else _architecture_base(entry)}"

    if is_detector(entry):
        order = list(DETECTOR_ORDER)
    elif subsets:
        order = subsets + list(SEGMENTER_ORDER)
    elif "aji" in this and "dice" in this:
        order = ["f1", "aji", "dice"] + list(SEGMENTER_ORDER)
    else:
        order = ["f1", "precision", "recall"] + list(SEGMENTER_ORDER)
    names: List[str] = []
    for name in order:
        if name in this and name not in names:
            names.append(name)
    rows: List[Tuple[str, Optional[float], Optional[float]]] = [
        (_row_label(name), this[name], base.get(name))
        for name in names[:ROWS]]
    rows += [("", None, None)] * (ROWS - len(rows))
    if not base:
        baseline = ""
    return {"rows": rows, "baseline": baseline, "caveat": caveat,
            "holdout": holdout_size(entry)}


def headline(entry: Mapping[str, Any], card: Mapping[str, Any]
             ) -> Optional[Tuple[str, float, Optional[float]]]:
    """The big number: the first row, as (metric, this model, baseline)."""
    label, mine, theirs = card["rows"][0]
    if mine is None:
        return None
    if label in LABELS.values():
        return label, mine, theirs
    return "F1", mine, theirs


def segments_and_why(entry: Mapping[str, Any]) -> Tuple[str, str]:
    stem = _stem(_key(entry))
    if stem in FAMILY_TEXT:
        return FAMILY_TEXT[stem]
    trained = str(entry.get("trained_on") or "").split(":")[0].strip()
    return trained[:90], ""


def title_of(entry: Mapping[str, Any]) -> str:
    """The card's short title: the family's name without its version."""
    stem = _stem(_key(entry))
    if stem in FAMILY_TITLES:
        return FAMILY_TITLES[stem]
    title = str(entry.get("display_name") or _key(entry))
    title = re.sub(r"\s*\(.*\)$", "", title)
    return re.sub(r"\s+v\d+$", "", title).strip()


def baseline_name(card: Mapping[str, Any]) -> str:
    """``"Stock cpsam_v2"`` -> ``"stock cpsam_v2"``; ``"v1 (earlier
    version)"`` -> ``"v1"``."""
    name = re.sub(r"^Stock ", "stock ", str(card.get("baseline") or ""))
    return re.sub(r"\s*\(earlier version\)$", "", name)


def alt_text(entry: Mapping[str, Any], status: str = "default") -> str:
    """One sentence a screen reader gets for the card image."""
    card = scores(entry)
    head = headline(entry, card)
    key = _key(entry)
    text = (f"Score card of {title_of(entry)} {_version(key)} ({key}, "
            f"{STATUS_WORDS.get(status, status)})")
    if head:
        label, mine, theirs = head
        text += f": {label} {mine:.3f}"
        if card["holdout"]:
            text += f" on {card['holdout']}"
        if theirs is not None and card["baseline"]:
            text += f" against {theirs:.3f} for {baseline_name(card)}"
    return text + "."


@functools.lru_cache(maxsize=None)
def _font(weight: int):
    """The Open Sans face of ``weight`` as a fontTools ``TTFont``."""
    from fontTools.ttLib import TTFont

    return TTFont(str(FONT_DIR / FONT_FILES[weight]))


def _width(text: str, size: float, weight: int = TEXT_WEIGHT) -> float:
    """Rendered width of ``text`` in Open Sans, from the font's advances."""
    font = _font(weight)
    cmap = font.getBestCmap()
    metrics = font["hmtx"].metrics
    units = sum(metrics[cmap.get(ord(c), ".notdef")][0] for c in text)
    return units * size / font["head"].unitsPerEm


@functools.lru_cache(maxsize=None)
def _font_face(weight: int, characters: str) -> str:
    """An ``@font-face`` rule embedding ``weight`` subset to ``characters``."""
    from fontTools import subset

    options = subset.Options()
    options.flavor = "woff"
    options.hinting = False
    options.desubroutinize = True
    options.name_IDs = [1, 2]
    options.notdef_outline = True
    options.layout_features = ["kern"]
    font = subset.load_font(str(FONT_DIR / FONT_FILES[weight]), options)
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=characters)
    subsetter.subset(font)
    buffer = io.BytesIO()
    subset.save_font(font, buffer, options)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return ("@font-face{font-family:'Open Sans';font-style:normal;"
            f"font-weight:{weight};src:url(data:font/woff;base64,{data})"
            " format('woff')}")


def without_font_data(svg: str) -> str:
    """``svg`` with its embedded font bytes blanked, for staleness checks."""
    return FONT_DATA.sub("url(data:font/woff;base64,)", svg)


def _fit(text: str, size: float, width: float,
         weight: int = TEXT_WEIGHT) -> str:
    if _width(text, size, weight) <= width:
        return text
    while text and _width(text + "…", size, weight) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """A bar anchored at ``x`` with a rounded data end."""
    w = max(w, 2.0)
    r = min(3.0, h / 2, w / 2)
    return (f'<path class="{cls}" d="M{x:.1f} {y:.1f}h{w - r:.1f}'
            f'a{r:g} {r:g} 0 0 1 {r:g} {r:g}v{h - 2 * r:.1f}'
            f'a{r:g} {r:g} 0 0 1 -{r:g} {r:g}h-{w - r:.1f}z"/>')


def _style(used: Mapping[int, Iterable[str]], rim: str) -> str:
    p = PALETTE
    faces = "".join(_font_face(weight, "".join(sorted(set(used[weight]))))
                    for weight in sorted(used) if used[weight])
    return (
        "<style>" + faces +
        f".card{{fill:{p['card']};stroke:{RIMS[rim]}}}"
        f".i1{{fill:{p['ink']}}}.i2{{fill:{p['ink2']}}}.i3{{fill:{p['ink3']}}}"
        f".m{{fill:{p['model']}}}.b{{fill:{p['base']}}}.tr{{fill:{p['track']}}}"
        f".tag{{fill:{RIMS[rim]};fill-opacity:.12;stroke:{RIMS[rim]};"
        f"stroke-opacity:.6}}.tagt{{fill:{RIMS[rim]}}}"
        f".up{{fill:{p['model']};fill-opacity:.28}}"
        f".dn{{fill:{p['base']};fill-opacity:.45}}"
        f".na{{fill:none;stroke:{p['edge']}}}"
        "text{font-family:" + FONT + "}"
        "</style>")


def _frame(uid: str, x: float, y: float, w: float, h: float) -> List[str]:
    """The dark card with its coloured rim, and a faint shadow under it for
    a white page."""
    p = PALETTE
    return [
        f'<defs><filter id="s{uid}" x="-4%" y="-4%" width="108%"'
        f' height="112%"><feDropShadow dx="0" dy="1" stdDeviation="1.2"'
        f' flood-color="{p["shadow"]}" flood-opacity=".22"/></filter></defs>',
        f'<rect class="card" x="{x + 0.75:g}" y="{y + 0.75:g}"'
        f' width="{w - 1.5:g}" height="{h - 1.5:g}" rx="12"'
        f' stroke-width="1.5" filter="url(#s{uid})"/>',
    ]


def card_size(layout: str = "desktop") -> Tuple[int, int]:
    """The outer ``(width, height)`` of a card in ``layout``."""
    return LAYOUTS[layout], CARD_HEIGHTS[layout] + MARGIN_Y + MARGIN_BOTTOM


def _short_holdout(card: Mapping[str, Any]) -> str:
    """``"209 held-out fields"`` -> ``"209 fields"``, for the phone."""
    return re.sub(r" held-out ", " ", str(card.get("holdout") or ""))


def render_svg(entry: Mapping[str, Any], status: str = "default",
               layout: str = "desktop", rim: str = RIM) -> str:
    """The score card of one zoo record as standalone SVG text."""
    key = _key(entry)
    uid = re.sub(r"\W", "", key)
    card = scores(entry)
    head = headline(entry, card)
    width, height = card_size(layout)
    card_w, card_h = width - 2 * MARGIN_X, CARD_HEIGHTS[layout]
    phone = layout == "phone"
    pad = 8 if phone else 18
    x0, y0 = MARGIN_X, MARGIN_Y
    left, right = x0 + pad, x0 + card_w - pad
    parts: List[str] = _frame(uid, x0, y0, card_w, card_h)
    used: Dict[int, Set[str]] = {TITLE_WEIGHT: set(), TEXT_WEIGHT: set()}

    def text(x, y, value, cls, size=TEXT, weight=TEXT_WEIGHT,
             anchor="start"):
        used[weight].update(value)
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}"'
            f' font-weight="{weight}" text-anchor="{anchor}">'
            f'{escape(value)}</text>')

    def tag_at(x_right, top):
        tag = _version(key)
        if status in TAG_WORDS:
            tag += f" · {TAG_WORDS[status]}"
        tag_w = _width(tag, TEXT, TITLE_WEIGHT) + 14
        parts.append(f'<rect class="tag" x="{x_right - tag_w:.1f}"'
                     f' y="{top}" width="{tag_w:.1f}" height="20" rx="10"/>')
        text(x_right - tag_w / 2, top + 14.6, tag, "tagt", TEXT,
             TITLE_WEIGHT, "middle")
        return tag_w

    def chip_at(x, top):
        label, mine, theirs = head
        if theirs is not None and card["baseline"]:
            delta = mine - theirs
            sign = "+" if delta >= 0 else "−"
            chip = f"{sign}{abs(delta):.3f} vs {baseline_name(card).split()[0]}"
            cls = "up" if delta >= 0 else "dn"
        else:
            chip = ("no stock detector" if is_detector(entry)
                    else "no baseline")
            cls = "na"
        chip_w = _width(chip, TEXT) + 18
        parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{top}"'
                     f' width="{chip_w:.1f}" height="22" rx="11"/>')
        text(x + chip_w / 2, top + 15.6, chip,
             "i1" if cls != "na" else "i3", anchor="middle")

    if phone:
        text(left, y0 + 25, _fit(title_of(entry), TEXT, right - left,
                                 TITLE_WEIGHT), "i1", TEXT, TITLE_WEIGHT)
        tag_at(right, y0 + 40)
        if head:
            label, mine, theirs = head
            text(left, y0 + 63, f"{mine:.3f}", "i1", 30)
            caption = (f"{label} · {_short_holdout(card)}"
                       if card["holdout"] else f"{label} at IoU 0.5")
            text(left, y0 + 86, _fit(caption, TEXT, right - left), "i3")
            chip_at(left, y0 + 94)
        else:
            text(left, y0 + 63, "—", "i3", 30)
            text(left, y0 + 86, "Score not recorded", "i3")
        rows_y, pitch = y0 + 126, 38
    else:
        tag_w = tag_at(right, y0 + 15)
        text(left, y0 + 30, _fit(title_of(entry), TEXT,
                                 right - left - tag_w - 12, TITLE_WEIGHT),
             "i1", TEXT, TITLE_WEIGHT)
        side = left + 132
        if head:
            label, mine, theirs = head
            text(left, y0 + 84, f"{mine:.3f}", "i1", 40)
            caption = (f"{label} · {card['holdout']}" if card["holdout"]
                       else f"{label} at IoU 0.5")
            text(side, y0 + 62, _fit(caption, TEXT, right - side), "i3")
            chip_at(side, y0 + 70)
        else:
            text(left, y0 + 84, "—", "i3", 40)
            text(side, y0 + 72, "Held-out score not recorded", "i3")
        rows_y, pitch = y0 + 106, 26

    for index, (label, mine, theirs) in enumerate(card["rows"]):
        y = rows_y + index * pitch
        if mine is None:
            text(left, y + 14, "—", "i3")
            continue
        if phone:
            text(left, y + 14, label, "i2")
            text(right, y + 14, f"{mine:.3f}", "i1", anchor="end")
            track_x, track_w, bar_y = left, right - left, y + 20
        else:
            track_x = left + 92
            track_w = right - track_x - 52
            bar_y = y + 3
            text(left, y + 14, label, "i2")
            text(right, y + 14, f"{mine:.3f}", "i1", anchor="end")
        parts.append(f'<rect class="tr" x="{track_x}" y="{bar_y}"'
                     f' width="{track_w}" height="7" rx="3"/>')
        parts.append(_bar(track_x, bar_y, track_w * mine, 7, "m"))
        if theirs is not None:
            parts.append(f'<rect class="tr" x="{track_x}" y="{bar_y + 9}"'
                         f' width="{track_w}" height="5" rx="2.5"/>')
            parts.append(_bar(track_x, bar_y + 9, track_w * theirs, 5, "b"))

    legend_y = y0 + card_h - (15 if phone else 17)
    parts.append(f'<rect class="m" x="{left}" y="{legend_y - 9}" width="10"'
                 ' height="10" rx="2"/>')
    text(left + 15, legend_y, "this model", "i3")
    if card["baseline"] and any(theirs is not None
                                for _l, _m, theirs in card["rows"]):
        lx = left + 15 + _width("this model", TEXT) + 14
        parts.append(f'<rect class="b" x="{lx:.1f}" y="{legend_y - 9}"'
                     ' width="10" height="10" rx="2"/>')
        base = (baseline_name(card).split()[0] if phone
                else baseline_name(card))
        text(lx + 15, legend_y, base, "i3")

    title = f"{title_of(entry)} {_version(key)} score card"
    desc = alt_text(entry, status)
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}"'
        f' height="{height}" viewBox="0 0 {width} {height}" role="img"'
        f' aria-labelledby="t d">\n'
        f'<title id="t">{escape(title)}</title>\n'
        f'<desc id="d">{escape(desc)}</desc>\n'
        f'{_style(used, rim)}\n{body}\n</svg>\n')


def render_heading(layout: str = "desktop", rim: str = RIM) -> str:
    """The centred heading over the README grid, two cards wide."""
    width = 2 * LAYOUTS[layout]
    height = HEADING_H + MARGIN_Y + MARGIN_BOTTOM + HEADING_GAP
    parts = _frame("heading", MARGIN_X, MARGIN_Y, width - 2 * MARGIN_X,
                   HEADING_H)
    parts.append(f'<text x="{width / 2:.1f}" y="{MARGIN_Y + 25}" class="i1"'
                 f' font-size="{TITLE}" font-weight="{TITLE_WEIGHT}"'
                 f' text-anchor="middle">{escape(HEADING_TEXT)}</text>')
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}"'
        f' height="{height}" viewBox="0 0 {width} {height}" role="img"'
        f' aria-labelledby="t">\n'
        f'<title id="t">{escape(HEADING_TEXT)}</title>\n'
        f'{_style({TITLE_WEIGHT: HEADING_TEXT}, rim)}\n{body}\n</svg>\n')


def heading_name(layout: str = "desktop") -> str:
    """The heading strip's path inside ``docs/models``."""
    return HEADING_NAME if layout == "desktop" else \
        f"{CARD_SUBDIR}/heading.{layout}.svg"


def card_name(key: str, layout: str = "desktop") -> str:
    """The card's path inside ``docs/models``: ``<key>.svg`` for the
    desktop card, ``<key>.<layout>.svg`` otherwise."""
    suffix = "" if layout == "desktop" else f".{layout}"
    return f"{CARD_SUBDIR}/{key}{suffix}.svg"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the cards; ``tools/build_model_cards.py`` does the same."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("build_model_cards.py")
    spec = importlib.util.spec_from_file_location("spacr_model_cards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
