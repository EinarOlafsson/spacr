"""Draw one SVG score card per published spaCR model.

A score card is the graphic at the top of each ``docs/models/<key>.md`` page
and in the README's model zoo grid: the model's name and version, what it
segments, its main hold-out score as a large number, bars for this model and
its baseline (stock Cellpose-SAM, or the earlier detector where no stock model
has the class) on the key metrics, the hold-out set and one line on why the
model exists.

Every number is read from the zoo record in ``spacr/model_zoo.py``
(``BUNDLED_REMOTE_MODELS``): the structured ``metrics`` entries first, then
the record's metric strings, then its ``versus_stock`` sentence and notes.
Nothing numeric is typed here, so a re-score published to the zoo reaches the
cards when ``python tools/build_model_cards.py`` is run again; that tool
writes the cards next to the Markdown pages.

The SVG paints its own background and switches palette with
``prefers-color-scheme``, so it reads on GitHub's light and dark themes and on
PyPI, which shows it on white.
"""
from __future__ import annotations

import re
from html import escape
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CARD_SUBDIR = "cards"
WIDTH = 480
PAD = 20
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', "
        "Helvetica, Arial, sans-serif")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

#: What each family segments and why the trained model exists. Words only:
#: every number on a card comes from the zoo record.
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

LABELS = {
    "f1": "F1", "precision": "Precision", "recall": "Recall", "aji": "AJI",
    "dice": "Dice", "mAP50": "mAP50", "mAP50_95": "mAP50-95",
}
SEGMENTER_ORDER = ("f1", "precision", "recall", "aji", "dice")
DETECTOR_ORDER = ("mAP50", "mAP50_95", "precision", "recall")
MAX_ROWS = 4

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
_STOCK_F1 = re.compile(
    r"\bstock ([\w.]+?)(?:'s)? " + _NUM
    + r"(?: there)?(?: with the ([\w -]+?) scorer)?\b")
_SAME_FIELDS = re.compile(r"\(([a-z][\w]*_v\d+) " + _NUM + r"\)")
_RESCORE = re.compile(
    r"Re-scored (\d{4}-\d{2}-\d{2}).*?fields this model never saw, F1 "
    + _NUM + r" at IoU 0\.5 on n = (\d+)")
_HELD_OUT = re.compile(r"on (?:the )?([\d,]+ [\w-]+(?: [\w-]+)? fields)")


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


def scores(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The card's numbers, all read from the zoo record.

    :returns: ``rows`` (label, this model, baseline or None), ``baseline``
        (the comparison's name), ``caveat``, ``holdout``, ``rescore`` and
        ``same_fields`` lines, each empty when the record has none, and
        ``context``, the record's first note when it has no baseline.
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

    order = DETECTOR_ORDER if is_detector(entry) else SEGMENTER_ORDER
    names = [name for name in order if name in this]
    names = [name for name in this if name.startswith("f1, ")] + names
    rows = [(LABELS.get(name, name.replace("f1, ", "F1, ")), this[name],
             base.get(name)) for name in names[:MAX_ROWS]]

    holdout = ""
    n_test = metrics.get("n_test")
    found = _HELD_OUT.search(versus)
    if isinstance(n_test, str) and re.fullmatch(r"[\d,]+", n_test.strip()):
        holdout = found.group(1) if found else f"{n_test.strip()} fields"
    elif isinstance(n_test, str) and n_test.strip():
        holdout = n_test.strip()
    elif isinstance(metrics.get("n_fields"), int):
        holdout = f"{metrics['n_fields']} fields"
    elif found:
        holdout = found.group(1)
    objects = str(metrics.get("test_objects") or "").strip()
    if holdout and re.fullmatch(r"[\d,]+", objects):
        holdout += f" · {int(objects.replace(',', '')):,} objects"

    rescore = ""
    found = _RESCORE.search(notes)
    if found:
        rescore = (f"Re-scored {found.group(1)} on fields it never saw: "
                   f"F1 {float(found.group(2)):.3f} (n = {found.group(3)})")
    same = _SAME_FIELDS.search(versus)
    same_fields = (f"Same fields: {same.group(1)} "
                   f"{float(same.group(2)):.3f}" if same else "")
    context = ""
    if not base:
        baseline = ""
        first = [str(note).strip() for note in entry.get("notes") or ()
                 if str(note).strip()]
        context = first[0] if first else ""
    return {"rows": rows, "baseline": baseline, "caveat": caveat,
            "holdout": holdout, "rescore": rescore,
            "same_fields": same_fields, "context": context}


def headline(entry: Mapping[str, Any], card: Mapping[str, Any]
             ) -> Optional[Tuple[str, float, Optional[float]]]:
    """The big number: the first row, labelled for the reader."""
    if not card["rows"]:
        return None
    label, mine, theirs = card["rows"][0]
    if label == "F1" or label.startswith("F1,"):
        label = label.replace("F1", "F1 at IoU 0.5", 1)
    elif label == "mAP50":
        label = "Box mAP at IoU 0.5"
    return label, mine, theirs


def segments_and_why(entry: Mapping[str, Any]) -> Tuple[str, str]:
    stem = _stem(_key(entry))
    if stem in FAMILY_TEXT:
        return FAMILY_TEXT[stem]
    trained = str(entry.get("trained_on") or "").split(":")[0].strip()
    return trained[:90], ""


def title_of(entry: Mapping[str, Any]) -> str:
    title = str(entry.get("display_name") or _key(entry))
    title = re.sub(r",\s*(default|alternative|candidate)\)", ")", title)
    title = re.sub(r"\s*\((default|alternative|candidate)\)", "", title)
    return title.strip()


def _wrap(text: str, size: float, width: float, lines: int) -> List[str]:
    limit = max(10, int(width / (size * 0.54)))
    words, out, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + (1 if line else 0) <= limit:
            line = f"{line} {word}".strip()
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1].rstrip(".,;") + " …"
    return out


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """A bar anchored at ``x`` with a rounded data end."""
    w = max(w, 1.5)
    r = min(3.0, h / 2, w / 2)
    return (f'<path class="{cls}" d="M{x:.1f} {y:.1f}h{w - r:.1f}'
            f'a{r} {r} 0 0 1 {r} {r}v{h - 2 * r:.1f}'
            f'a{r} {r} 0 0 1 -{r} {r}h-{w - r:.1f}z"/>')


STYLE = (
    "<style>"
    ".bg{fill:#fcfcfb;stroke:#d9d8d3}"
    ".t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#6b6a65}"
    ".ft{fill:#2a78d6}.va{fill:#86857f}.tr{fill:#efeeea}"
    ".ax{stroke:#d9d8d3}.pl{fill:none;stroke:#2a78d6}.pt{fill:#1c5cab}"
    ".po{fill:none;stroke:#86857f}.pu{fill:#52514e}"
    "@media (prefers-color-scheme: dark){"
    ".bg{fill:#1a1a19;stroke:#3a3936}"
    ".t1{fill:#f0efec}.t2{fill:#c3c2b7}.t3{fill:#a3a29a}"
    ".ft{fill:#3987e5}.va{fill:#7c7b75}.tr{fill:#262624}"
    ".ax{stroke:#3a3936}.pl{stroke:#3987e5}.pt{fill:#86b6ef}"
    ".po{stroke:#7c7b75}.pu{fill:#c3c2b7}}"
    "</style>")


def alt_text(entry: Mapping[str, Any], status: str = "default") -> str:
    """One sentence a screen reader gets for the card image."""
    card = scores(entry)
    head = headline(entry, card)
    text = f"Score card of {title_of(entry)} ({STATUS_WORDS.get(status, status)})"
    if head:
        label, mine, theirs = head
        text += f": {label} {mine:.3f}"
        if theirs is not None and card["baseline"]:
            baseline = re.sub(r"^Stock ", "stock ", card["baseline"])
            text += f" against {theirs:.3f} for {baseline}"
    return text + "."


def render_svg(entry: Mapping[str, Any], status: str = "default") -> str:
    """The score card of one zoo record as standalone SVG text."""
    key = _key(entry)
    card = scores(entry)
    head = headline(entry, card)
    segments, why = segments_and_why(entry)
    inner = WIDTH - 2 * PAD
    parts: List[str] = []

    def text(x, y, value, cls, size, weight=400, anchor="start",
             family=FONT, italic=False):
        style = ' font-style="italic"' if italic else ""
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}"'
            f' font-weight="{weight}" text-anchor="{anchor}"'
            f' font-family="{escape(family)}"{style}>{escape(value)}</text>')

    y = PAD + 18
    status_word = STATUS_WORDS.get(status, status)
    pill_w = 12 + len(status_word) * 6.6
    title_lines = _wrap(title_of(entry), 17, inner - pill_w - 12, 2)
    for i, line in enumerate(title_lines):
        text(PAD, y + i * 22, line, "t1", 17, 600)
    pill_cls, pill_text = (("pl", "pt") if status == "default"
                           else ("po", "pu"))
    parts.append(f'<rect class="{pill_cls}" x="{WIDTH - PAD - pill_w:.1f}"'
                 f' y="{y - 14}" width="{pill_w:.1f}" height="20" rx="10"/>')
    text(WIDTH - PAD - pill_w / 2, y, status_word, pill_text, 12, 600,
         "middle")
    y += 22 * (len(title_lines) - 1) + 20
    arch = str(entry.get("architecture") or "").strip()
    text(PAD, y, f"{key} · {_version(key)}", "t3", 12, family=MONO)
    y += 18
    for line in _wrap(segments, 13, inner, 2):
        text(PAD, y, line, "t2", 13)
        y += 18
    if arch:
        for line in _wrap(arch, 12, inner, 1):
            text(PAD, y, line, "t3", 12)
            y += 16

    y += 14
    if head:
        label, mine, theirs = head
        text(PAD, y + 30, f"{mine:.3f}", "t1", 44, 600)
        text(PAD + 128, y + 6, label, "t2", 13)
        if theirs is not None and card["baseline"]:
            delta = mine - theirs
            text(PAD + 128, y + 24, f"{card['baseline']}: {theirs:.3f}", "t2",
                 13)
            text(PAD + 128, y + 42, f"Difference {delta:+.3f}", "t1", 13, 600)
        elif is_detector(entry):
            text(PAD + 128, y + 24, "No stock detector has this class;",
                 "t3", 12)
            text(PAD + 128, y + 40, "no baseline in the record.", "t3", 12)
        else:
            text(PAD + 128, y + 24, "No stock baseline recorded", "t3", 12)
            text(PAD + 128, y + 40, "for this hold-out set.", "t3", 12)
        y += 60
    else:
        text(PAD, y + 8, "Hold-out scores are on the model card.", "t2", 13)
        y += 24

    rows = card["rows"]
    if rows:
        y += 6
        has_base = any(theirs is not None for _l, _m, theirs in rows)
        parts.append(f'<rect class="ft" x="{PAD}" y="{y - 9}" width="10"'
                     ' height="10" rx="2"/>')
        text(PAD + 16, y, "This model", "t2", 12)
        if has_base:
            parts.append(f'<rect class="va" x="{PAD + 100}" y="{y - 9}"'
                         ' width="10" height="10" rx="2"/>')
            text(PAD + 116, y, card["baseline"], "t2", 12)
        y += 14
        label_w = 128 if any(len(l) > 12 for l, _m, _t in rows) else 92
        x0 = PAD + label_w
        track = inner - label_w - 48
        top = y
        for label, mine, theirs in rows:
            row_h = 30 if theirs is not None else 18
            label_lines = _wrap(label, 12, label_w - 8, 2)
            for i, line in enumerate(label_lines):
                text(PAD, y + 11 + i * 13, line, "t2", 12)
            parts.append(f'<rect class="tr" x="{x0}" y="{y + 2}"'
                         f' width="{track}" height="10" rx="2"/>')
            parts.append(_bar(x0, y + 2, track * mine, 10, "ft"))
            text(x0 + track + 6, y + 11, f"{mine:.3f}", "t1", 12)
            if theirs is not None:
                parts.append(f'<rect class="tr" x="{x0}" y="{y + 14}"'
                             f' width="{track}" height="10" rx="2"/>')
                parts.append(_bar(x0, y + 14, track * theirs, 10, "va"))
                text(x0 + track + 6, y + 23, f"{theirs:.3f}", "t2", 12)
            y += max(row_h, 13 * len(label_lines)) + 8
        for tick in (0.0, 0.5, 1.0):
            x = x0 + track * tick
            parts.append(f'<line class="ax" x1="{x:.1f}" y1="{top - 2}"'
                         f' x2="{x:.1f}" y2="{y - 6}" stroke-width="1"'
                         ' stroke-dasharray="2 3"/>')
            text(x, y + 6, f"{tick:g}", "t3", 11, anchor="middle")
        y += 14

    y += 8
    lines: List[Tuple[str, str, bool]] = []
    if card["holdout"]:
        lines += [(line, "t2", False) for line in
                  _wrap(f"Hold-out: {card['holdout']}", 12, inner, 2)]
    for extra in (card["same_fields"], card["caveat"], card["rescore"]):
        if extra:
            lines += [(line, "t2", False) for line in _wrap(extra, 12, inner, 2)]
    if card["context"]:
        lines += [(line, "t2", False) for line in
                  _wrap(card["context"][:1].upper() + card["context"][1:],
                        12, inner, 4)]
    if why:
        lines += [(line, "t1", True) for line in _wrap(why, 12.5, inner, 2)]
    for line, cls, italic in lines:
        y += 17
        text(PAD, y, line, cls, 12.5 if italic else 12, italic=italic)
    height = int(y + PAD)

    title = f"{title_of(entry)} score card"
    desc = alt_text(entry, status)
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}"'
        f' height="{height}" viewBox="0 0 {WIDTH} {height}" role="img"'
        f' aria-labelledby="t d">\n'
        f'<title id="t">{escape(title)}</title>\n'
        f'<desc id="d">{escape(desc)}</desc>\n'
        f'{STYLE}\n'
        f'<rect class="bg" x="0.5" y="0.5" width="{WIDTH - 1}"'
        f' height="{height - 1}" rx="12" stroke-width="1"/>\n'
        f'{body}\n</svg>\n')


def card_name(key: str) -> str:
    """The card's path inside ``docs/models``."""
    return f"{CARD_SUBDIR}/{key}.svg"


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
