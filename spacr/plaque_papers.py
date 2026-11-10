"""Plaque measurements from published papers: a DOI, PMID or PDF in, rows out.

Given a paper, this module finds its figures, finds the plaque
images inside them with the zoo's well detector, reads the text printed
around each image, works out which condition each image shows, segments and
measures the plaques, and writes everything to a SQLite database together
with where every value came from.

The condition each image shows is read two independent ways, always both,
because two readings that agree support each other and a single reading
does not:

* **the label** -- the text nearest the image inside the figure: the column
  header above it, the row label beside it, anything printed beneath it. A
  figure that prints ``TATi | iGAPDH2`` over two columns and ``-ATc | +ATc``
  beside two rows gives each of its four crops two words, and those two words
  are the condition.
* **the legend** -- the panel letter nearest the image keys into the figure
  legend, which is fetched from Europe PMC, read from the PDF, or pasted by
  the user when neither has it.

When both are present they are both kept; a label whose words the legend also
uses is ``strong``, one the legend does not echo is ``medium`` -- a legend that
says "under indicated conditions" is not a disagreement. They DISAGREE when the
panel's legend passage names conditions this figure prints on its other
images and none of this image's own: that sets ``conflict``, with the words
that caused it, and the row is kept for a person to settle rather than one
reading being preferred. When neither is present the image is named by its
figure and its row and column in the panel, and marked ``weak`` so a dataset
can leave those rows out.

**Sizes.** A plaque area in pixels depends on the paper's printing, so every
plaque also carries its area relative to the median plaque in the same panel,
and an area in mm^2 only when there is a ruler: a scale bar read in or under
the image (its length from the label beside it, or from the legend's "scale
bar, 1 mm"), or a whole well of a plate format the settings or the legend
state. A stated magnification is recorded but is never a ruler, because a
printed figure has been rescaled since the picture was taken. Rows with no
ruler say their sizes are in pixels.

**Duplicates.** A figure is identified by its paper (the DOI when there is one)
and its image hash. The same image seen again -- the same bytes, or the same
pixels in another file format, under another name or another paper, such as a
preprint and its published version -- is measured once and recorded in the
``duplicates`` table against the copy that was measured. A paper already
measured from one source (Europe PMC) is not measured again from another (its
PDF). An image that only looks the same (a near-identical perceptual hash,
as a re-encoded or resized copy would have) is recorded as a possible
duplicate and still measured, because a similar hash does not prove that two
figures are the same image.

**Text.** A PDF's own text layer is used for legends and labels before any OCR;
OCR is asked only when the text layer has nothing near the plaque images,
which is what a figure pasted into the PDF as one picture looks like.

The optional pieces -- ``ultralytics`` for the detector, ``rapidocr`` for the
text in figure images, ``pdfplumber`` for a PDF's text layer -- are the
``spacr[papers]`` extra. Everything here that touches them takes the callable
as a parameter, so the logic is testable without them.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import sqlite3
import time
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "Paper",
    "Figure",
    "Word",
    "Region",
    "Annotation",
    "parse_reference",
    "resolve_paper",
    "fetch_figures",
    "figures_from_pdf",
    "legends_from_text",
    "split_legend",
    "read_words",
    "find_plaque_regions",
    "text_near",
    "TextOptions",
    "DEFAULT_TEXT_OPTIONS",
    "text_options_from_settings",
    "annotate_regions",
    "measure_region",
    "measure_plaques_from_papers",
    "measure_figure_folder",
    "fetch_paper_to_folder",
    "figures_in_folder",
    "figure_folders",
    "read_legends",
    "read_annotation_overrides",
    "write_annotation_overrides",
    "apply_overrides",
    "reread_around",
    "console_legend_prompt",
    "console_review",
    "ReaderNeedsInstall",
    "reader_problem",
]

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = {"User-Agent": "spacr-plaque-papers/1.0 "
                            "(https://github.com/EinarOlafsson/spacr)"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp", ".webp",
                  ".scn"}
FORMAT_RANK = {".scn": 0, ".tiff": 0, ".tif": 0, ".png": 1, ".webp": 2, ".jpg": 2,
               ".jpeg": 2, ".gif": 3, ".bmp": 3}

DEFAULT_DETECTOR = "toxoplasma_well_detector_v2"
DEFAULT_SEGMENTER = "toxoplasma_plaque_v9"
DEFAULT_IMGSZ: Tuple[int, ...] = (640, 1280)

INSTALL_HINT = 'pip install "spacr[papers]"'

_PANEL_WORD = re.compile(r"^\(?([A-Za-z])\)?[.:]?$")
_DOI = re.compile(r"\b(10\.\d{4,9}/\S+)", re.IGNORECASE)
_PMCID = re.compile(r"^PMC\d+$", re.IGNORECASE)
_PMID = re.compile(r"^\d{1,9}$")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_FIGURE_START = re.compile(
    r"(?m)^\s*(?:Fig(?:ure)?\.?|FIG\.?|FIGURE)\s*(\d+)[.:|\s]")


@dataclass(frozen=True)
class Paper:
    """One paper, however it was named.

    :param key: the identifier rows are filed under: the DOI when there is
        one, else the PMC id, else the PDF's sha256.
    :param source: ``'europepmc'``, ``'pdf'`` or ``'folder'``.
    :param doi_from: where the DOI was read: ``'europepmc'``, ``'pdf
        metadata'`` or ``'pdf text'``; None when there is no DOI.
    """

    key: str
    source: str
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    title: Optional[str] = None
    licence: Optional[str] = None
    pdf: Optional[str] = None
    doi_from: Optional[str] = None


@dataclass
class Figure:
    """One figure image and its legend.

    :param path: the image file on disk.
    :param label: the figure's name in the paper, e.g. ``'Fig 7'``.
    :param caption: the legend, or ``''`` when none was found.
    :param legend_source: where ``caption`` came from: ``'europepmc'``,
        ``'pdf'``, ``'pasted'`` or ``'none'``.
    :param words: text already known with coordinates, e.g. a PDF's text
        layer; OCR fills this when it is empty.
    """

    path: Path
    label: str = ""
    caption: str = ""
    legend_source: str = "none"
    sha256: str = ""
    words: List["Word"] = field(default_factory=list)


@dataclass(frozen=True)
class Word:
    """One piece of text in a figure, in image pixels.

    :param text: the text as read.
    :param x0: left edge of its box.
    :param y0: top edge of its box.
    :param x1: right edge of its box.
    :param y1: bottom edge of its box.
    :param confidence: the reader's confidence in the text, 0 to 1.
    """

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        """Horizontal centre."""
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        """Vertical centre."""
        return (self.y0 + self.y1) / 2.0

    @property
    def panel_letter(self) -> Optional[str]:
        """The letter when this word is a panel label such as ``F`` or ``(b)``."""
        match = _PANEL_WORD.match(self.text.strip())
        return match.group(1) if match else None


@dataclass(frozen=True)
class Region:
    """One plaque image the detector found inside a figure.

    :param x0: left edge of the box, in figure pixels.
    :param y0: top edge of the box.
    :param x1: right edge of the box.
    :param y1: bottom edge of the box.
    :param confidence: the detector's confidence, 0 to 1.
    :param sizes: every inference size that found it, so a run records which
        size a region depended on.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float = 1.0
    sizes: Tuple[int, ...] = ()

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return int(self.y1 - self.y0)

    @property
    def axis_ratio(self) -> float:
        """Shorter side over longer, so 1.0 is square."""
        long_side = max(self.width, self.height)
        return min(self.width, self.height) / long_side if long_side else 0.0

    def contains(self, word: Word) -> bool:
        """Whether a word's centre lies inside this box.

        :param word: a word read from the same figure.
        :returns: True when the word's centre is on or inside the box.
        """
        return self.x0 <= word.cx <= self.x1 and self.y0 <= word.cy <= self.y1


@dataclass
class Annotation:
    """What one plaque image shows, and how that was decided.

    :param region: the plaque image's box in the figure.
    :param panel: the panel letter read nearest the image, or ``None``.
    :param label_text: strategy 1 -- the text around the image.
    :param legend_text: strategy 2 -- the legend's sentence for the panel.
    :param condition: the proposed condition.
    :param source: ``'label+legend'``, ``'label'``, ``'legend'``, ``'manual'``
        or ``'position'``.
    :param strength: ``'strong'`` when both readings agree, ``'medium'`` for
        one reading, ``'weak'`` for position only, ``'manual'`` when a person
        wrote it.
    :param conflict: True when the two readings disagree -- the panel's
        legend names conditions the figure prints elsewhere and none of this
        image's (see :func:`annotate_regions`).
    :param conflict_terms: the legend's words that caused ``conflict``.
    :param approved: ``True``/``False`` once a person has reviewed it,
        ``None`` when nobody has.
    """

    region: Region
    panel: Optional[str] = None
    row: int = 0
    column: int = 0
    near: Dict[str, List[str]] = field(default_factory=dict)
    label_text: str = ""
    legend_text: str = ""
    condition: str = ""
    source: str = "position"
    strength: str = "weak"
    conflict: bool = False
    conflict_terms: List[str] = field(default_factory=list)
    approved: Optional[bool] = None
    pixels_per_um: Optional[float] = None
    formation_hours: Optional[float] = None


def _conflict_reason(a: "Annotation") -> str:
    """An annotation's conflict in words, for the database and the annotations file.

    :param a: the annotation.
    :returns: ``''`` when its two readings do not disagree.
    """
    if not a.conflict:
        return ""
    return (f"legend for panel {a.panel or '?'} names "
            f"{', '.join(a.conflict_terms)}; the label reads "
            f"{a.label_text!r}")


def _get(url: str, **kwargs: Any):
    """One GET with this module's user agent.

    :param url: the address.
    :param kwargs: passed to :func:`requests.get`.
    :returns: the response.
    """
    import requests

    kwargs.setdefault("headers", USER_AGENT)
    kwargs.setdefault("timeout", 120)
    return requests.get(url, **kwargs)


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of some bytes.

    :param data: the bytes.
    :returns: the digest.
    """
    return hashlib.sha256(data).hexdigest()


def parse_reference(ref: Any) -> Dict[str, str]:
    """Classify a reference as a DOI, PMC id, PMID or PDF path.

    :param ref: a DOI (bare, ``doi:`` or a doi.org URL), a PMC id, a PMID,
        or a path to a PDF.
    :returns: ``{'kind': 'doi'|'pmcid'|'pmid'|'pdf', 'value': ...}``.
    :raises ValueError: when it is none of these.
    """
    text = str(ref).strip()
    if text.lower().endswith(".pdf"):
        return {"kind": "pdf", "value": text}
    if _PMCID.match(text):
        return {"kind": "pmcid", "value": text.upper()}
    if _PMID.match(text):
        return {"kind": "pmid", "value": text}
    match = _DOI.search(text)
    if match:
        return {"kind": "doi", "value": match.group(1).rstrip(".,;")}
    raise ValueError(
        f"{text!r} is not a DOI, a PMC id, a PMID or a path to a PDF")


_DOI_CHARS = re.compile(r"10\.\d{4,9}/[A-Za-z0-9._;()/:-]+")
_FIGURE_DOI_SUFFIX = re.compile(r"\.(?:g|s|t|sd|e)\d{3,4}$", re.IGNORECASE)


def _doi_candidates(text: str) -> List[str]:
    """The DOIs a stretch of PDF text may name, likeliest first.

    A PDF's text layer breaks a DOI wherever the line did --
    ``https://d oi.org/10.1371/j ournal.\\nppat.1011009`` is what one PLOS
    first page gives -- so the DOI is looked for twice: as printed, and with
    the whitespace taken out, which rejoins it and also glues it to the next
    word (``...1011009Editor:``). A glued run is cut where a digit meets a
    capital letter. A figure's own DOI (``....g004``) names its paper once
    the suffix is dropped, and journals print one under every figure, so
    the DOI named most often comes first.

    :param text: the text.
    :returns: distinct candidates, most frequently named first; the longest
        of equally frequent ones first.
    """
    counts: Dict[str, int] = {}
    order: List[str] = []

    def add(doi: str) -> None:
        """Count one sighting of a cleaned candidate."""
        doi = _FIGURE_DOI_SUFFIX.sub("", doi.rstrip(".,;:)"))
        if not re.match(r"^10\.\d{4,9}/.+", doi):
            return
        if doi not in counts:
            order.append(doi)
        counts[doi] = counts.get(doi, 0) + 1
    joined = re.sub(r"\s+", "", text or "")
    for match in _DOI_CHARS.finditer(joined):
        run = match.group(0)
        cut = re.search(r"\d(?=[A-Z])", run)
        add(run[:cut.end()] if cut else run)
    for match in _DOI.finditer(text or ""):
        add(match.group(1))
    return sorted(order, key=lambda d: (-counts[d], -len(d), order.index(d)))


def _doi_in_pdf(pdf: Any) -> Tuple[Optional[str], Optional[str]]:
    """The DOI a PDF states, from its metadata or its first pages' text.

    Read with ``pypdf``, which spaCR already depends on, so this works
    without the ``spacr[papers]`` extra. See :func:`_doi_candidates` for how
    a DOI broken across lines is put back together.

    :param pdf: the PDF path.
    :returns: ``(doi, 'pdf metadata'|'pdf text')``, the likeliest candidate,
        or ``(None, None)`` when it names none or cannot be read.
    """
    found = _pdf_doi_candidates(pdf)
    return found[0] if found else (None, None)


def _pdf_doi_candidates(pdf: Any) -> List[Tuple[str, str]]:
    """Every DOI a PDF's metadata or first pages may name, likeliest first.

    :param pdf: the PDF path.
    :returns: ``[(doi, 'pdf metadata'|'pdf text')]``; empty when it names
        none or cannot be read.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf))
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    try:
        meta = reader.metadata or {}
        for key in ("/doi", "/DOI", "/prism:doi", "/Subject", "/Keywords"):
            match = _DOI.search(str(meta.get(key) or ""))
            if match:
                out.append((match.group(1).rstrip(".,;)"), "pdf metadata"))
    except Exception:
        pass
    text = []
    for page in list(reader.pages)[:2]:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            continue
    for doi in _doi_candidates("\n".join(text)):
        if doi not in [d for d, _ in out]:
            out.append((doi, "pdf text"))
    return out


def _epmc_hit(query: str, fetch: Callable) -> Optional[Dict[str, Any]]:
    """The first Europe PMC search hit, or None.

    A reply that is not a search result -- an error status, or a 200 with
    a short body, which Europe PMC sometimes sends under load -- is asked
    again, up to three times, a second apart and then two.

    :param query: a Europe PMC query.
    :param fetch: the HTTP getter.
    :returns: the hit.
    """
    for attempt in range(3):
        response = fetch(f"{EPMC}/search", params={
            "query": query, "format": "json", "resultType": "core",
            "pageSize": 1})
        try:
            body = response.json() if getattr(response, "status_code", 200) == 200 \
                else None
        except ValueError:
            body = None
        if isinstance(body, dict) and "resultList" in body:
            hits = (body.get("resultList") or {}).get("result", [])
            return hits[0] if hits else None
        time.sleep(1.0 * (attempt + 1))
    return None


def resolve_paper(ref: Any, *, get: Optional[Callable] = None) -> Paper:
    """Look a reference up in Europe PMC, or describe a local PDF.

    A PDF is filed under its DOI when it states one and Europe PMC knows
    that DOI (the three likeliest candidates are asked) -- so the same paper given once as a DOI and once as a PDF is
    one paper, not two -- and under its sha256 otherwise, with the DOI it
    names kept as unconfirmed.

    :param ref: anything :func:`parse_reference` accepts.
    :param get: ``fn(url, params=...) -> response``; defaults to requests.
    :returns: the paper. A reference Europe PMC does not know comes back with
        only the identifier filled in, which still lets a PDF be measured.
    """
    parsed = parse_reference(ref)
    kind, value = parsed["kind"], parsed["value"]
    fetch = get or _get
    if kind == "pdf":
        digest = sha256_bytes(Path(value).read_bytes())
        candidates = _pdf_doi_candidates(value)
        for doi, doi_from in candidates[:3]:
            try:
                hit = _epmc_hit(f'DOI:"{doi}"', fetch)
            except Exception:
                hit = None
            if hit and str(hit.get("doi") or "").lower() == doi.lower():
                return Paper(key=hit["doi"], source="pdf", doi=hit["doi"],
                             pmcid=hit.get("pmcid"), pmid=hit.get("pmid"),
                             title=hit.get("title"),
                             licence=(hit.get("license") or "").strip().lower() or None,
                             pdf=str(value), doi_from=doi_from)
        doi, doi_from = candidates[0] if candidates else (None, None)
        return Paper(key=f"pdf:{digest}", source="pdf", pdf=str(value), doi=doi,
                     doi_from=f"{doi_from}, unconfirmed" if doi else None)
    query = {"doi": f'DOI:"{value}"', "pmcid": f"PMCID:{value}",
             "pmid": f"EXT_ID:{value} AND SRC:MED"}[kind]
    hit = _epmc_hit(query, fetch)
    if not hit:
        return Paper(key=value, source="europepmc", **{kind: value},
                     doi_from="reference" if kind == "doi" else None)
    doi = hit.get("doi") or (value if kind == "doi" else None)
    pmcid = hit.get("pmcid") or (value if kind == "pmcid" else None)
    return Paper(key=doi or pmcid or value, source="europepmc", doi=doi,
                 pmcid=pmcid, pmid=hit.get("pmid"), title=hit.get("title"),
                 licence=(hit.get("license") or "").strip().lower() or None,
                 doi_from="europepmc" if doi else None)


def _graphic_stem(name: str) -> str:
    """A figure file's name without an IMAGE extension, and nothing more.

    ``Path.stem`` drops whatever follows the last dot, so a JATS graphic named
    without an extension -- ``ppat.1011009.g006`` -- lost its ``.g006`` and
    matched no image: every legend of such a paper went unfound.

    :param name: a file name or a graphic reference.
    :returns: the name, minus a trailing image extension if it has one.
    """
    base = Path(str(name)).name
    suffix = Path(base).suffix.lower()
    return base[:-len(suffix)] if suffix in IMAGE_SUFFIXES else base


def _captions_from_jats(xml: bytes) -> Dict[str, Dict[str, str]]:
    """Figure label and legend for each graphic named in a JATS article.

    :param xml: the article's full-text XML.
    :returns: ``{graphic file stem: {'label': ..., 'caption': ...}}``.
    """
    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for fig in root.iter("fig"):
        label = (fig.findtext("label") or "").strip()
        caption_el = fig.find("caption")
        caption = ""
        if caption_el is not None:
            caption = " ".join("".join(caption_el.itertext()).split())
        for graphic in fig.iter("graphic"):
            href = (graphic.get("{http://www.w3.org/1999/xlink}href")
                    or graphic.get("href") or "")
            if href:
                out[_graphic_stem(href)] = {"label": label, "caption": caption}
    return out


def fetch_figures(paper: Paper, dest: Any, *,
                  get: Optional[Callable] = None) -> List[Figure]:
    """Download a paper's figures from Europe PMC with their legends.

    The images come from the ``supplementaryFiles`` bundle, which carries the
    main figures as well as the supplements; one file per figure survives,
    the largest, because journals also ship thumbnails, which are too small
    for the detector to measure the figure. Legends come from the
    full-text XML.

    :param paper: a paper with a PMC id.
    :param dest: directory for the images; created if missing.
    :param get: ``fn(url, **kw) -> response``; defaults to requests.
    :returns: the figures, sorted by file name; empty when Europe PMC has no
        bundle for the paper.
    """
    if not paper.pmcid:
        return []
    fetch = get or _get
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    captions: Dict[str, Dict[str, str]] = {}
    response = fetch(f"{EPMC}/{paper.pmcid}/fullTextXML")
    if getattr(response, "status_code", 0) == 200 and response.content:
        captions = _captions_from_jats(response.content)
    response = fetch(f"{EPMC}/{paper.pmcid}/supplementaryFiles", timeout=300)
    if getattr(response, "status_code", 0) != 200 or not response.content:
        return []
    figures: List[Figure] = []
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            best: Dict[str, Tuple[int, int, str]] = {}
            for info in bundle.infolist():
                suffix = Path(info.filename).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    continue
                stem = _graphic_stem(info.filename)
                score = (-int(info.file_size), FORMAT_RANK[suffix])
                if stem not in best or score < best[stem][:2]:
                    best[stem] = (score[0], score[1], info.filename)
            for stem in sorted(best):
                name = best[stem][2]
                data = bundle.read(name)
                target = dest / Path(name).name
                target.write_bytes(data)
                meta = captions.get(stem, {})
                caption = meta.get("caption", "")
                figures.append(Figure(
                    path=target, label=meta.get("label", ""), caption=caption,
                    legend_source="europepmc" if caption else "none",
                    sha256=sha256_bytes(data)))
    except zipfile.BadZipFile:
        return []
    return figures


def legends_from_text(text: str) -> Dict[str, str]:
    """Figure legends found in a paper's running text.

    :param text: the text of a PDF, pages joined.
    :returns: ``{figure number: legend}``, the LONGEST paragraph starting
        ``Fig N`` for each number, because the legend is the paragraph and a
        short hit is a mention in the body.
    """
    out: Dict[str, str] = {}
    starts = list(_FIGURE_START.finditer(text))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        chunk = text[match.start():end]
        paragraph = re.split(r"\n\s*\n", chunk, maxsplit=1)[0]
        paragraph = " ".join(paragraph.split())
        number = match.group(1)
        if len(paragraph) > len(out.get(number, "")):
            out[number] = paragraph
    return out


#: How close two characters must sit to be one word in a PDF's text layer,
#: in points. pdfplumber's default of 3 runs a tightly set journal page
#: into one word per line ("Fig4.ThecytosolicTPI1...", PLOS Pathogens
#: 2022); 1.5 splits it into words and still keeps each word whole.
PDF_X_TOLERANCE = 1.5


def _pdf_pages_here(pdf: Any, dest: Path, dpi: int,
                    opener: Callable) -> List[Dict[str, Any]]:
    """Render a PDF's pages and read their text layer in this process.

    :param pdf: the PDF path.
    :param dest: directory for the page images.
    :param dpi: render resolution.
    :param opener: ``fn(path) -> pdfplumber-like document``.
    :returns: per page ``{'path', 'text', 'words': [[text, x0, y0, x1, y1]]}``
        with the words in the rendered image's pixels.
    """
    scale = dpi / 72.0
    pages: List[Dict[str, Any]] = []
    with opener(str(pdf)) as document:
        for number, page in enumerate(document.pages, start=1):
            target = dest / f"page_{number:03d}.png"
            page.to_image(resolution=dpi).save(str(target))
            pages.append({
                "path": str(target),
                "text": page.extract_text(x_tolerance=PDF_X_TOLERANCE) or "",
                "words": [[w["text"], w["x0"] * scale, w["top"] * scale,
                           w["x1"] * scale, w["bottom"] * scale]
                          for w in page.extract_words(x_tolerance=PDF_X_TOLERANCE)]})
    return pages


def _pdf_pages_in_reader(pdf: Any, dest: Path, dpi: int) -> List[Dict[str, Any]]:
    """:func:`_pdf_pages_here`, answered by the figure reader's environment.

    :param pdf: the PDF path.
    :param dest: directory for the page images.
    :param dpi: render resolution.
    :returns: the same per-page records.
    """
    from ._segmentation_backends import _BackendError, _worker_for

    kind, why = reader_problem(pdf=True)
    if kind:
        raise ReaderNeedsInstall(why, reinstall=kind == "reinstall")
    try:
        reply = _worker_for(READER_BACKEND, reader_environment()).request(
            "read_pdf", pdf=str(Path(pdf).resolve()), dest=str(dest.resolve()),
            dpi=int(dpi), x_tolerance=PDF_X_TOLERANCE)
    except _BackendError as exc:
        if getattr(exc, "remote_type", "") == "ModuleNotFoundError" and \
                "pdfplumber" in str(exc):
            raise ReaderNeedsInstall(str(exc), reinstall=True) from exc
        raise
    return list(reply.get("pages", []))


def figures_from_pdf(pdf: Any, dest: Any, *, dpi: int = 200,
                     opener: Optional[Callable] = None) -> List[Figure]:
    """Every page of a PDF as a figure image, with its text layer attached.

    A page is rendered whole rather than its embedded images extracted,
    because a published figure is often assembled from many embedded images
    and vector labels, and the labels are what the condition is read from.
    The text layer comes back as :class:`Word` objects in the rendered
    image's pixels, so no OCR is needed for a PDF that has one.

    Rendering needs ``pdfplumber``: in this process when it is installed
    here, else in the figure reader's own environment, which installs it.

    :param pdf: the PDF path.
    :param dest: directory for the page images.
    :param dpi: render resolution.
    :param opener: ``fn(path) -> pdfplumber-like document``; defaults to
        :func:`pdfplumber.open`.
    :returns: one figure per page, legend attached from the text when the
        page names a figure.
    :raises ImportError: when pdfplumber is available neither way.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if opener is None and not _importable("pdfplumber") and \
            reader_environment() is not None:
        pages = _pdf_pages_in_reader(pdf, dest, dpi)
    else:
        if opener is None:
            try:
                import pdfplumber
            except ImportError as exc:
                raise ReaderNeedsInstall(
                    "Reading a PDF needs 'pdfplumber'. Install the plaque "
                    "figure reader from Plaque Assay's Figure mode or the "
                    "Model Zoo, or install it here with:\n  "
                    + INSTALL_HINT) from exc
            opener = pdfplumber.open
        pages = _pdf_pages_here(pdf, dest, dpi, opener)
    legends = legends_from_text("\n\n".join(p["text"] for p in pages))
    figures: List[Figure] = []
    for number, page in enumerate(pages, start=1):
        target = Path(page["path"])
        words = [Word(str(w[0]), float(w[1]), float(w[2]), float(w[3]),
                      float(w[4])) for w in page["words"]]
        named = [m.group(1) for m in _FIGURE_START.finditer(page["text"])]
        label, caption = "", ""
        for figure_number in named:
            if figure_number in legends:
                label, caption = f"Fig {figure_number}", legends[figure_number]
                break
        figures.append(Figure(
            path=target, label=label or f"page {number}", caption=caption,
            legend_source="pdf" if caption else "none",
            sha256=sha256_bytes(target.read_bytes()), words=words))
    return figures


def _expand_letters(spec: str) -> List[str]:
    """``'B-E'`` -> B, C, D, E; ``'A and B'`` / ``'A, C'`` -> the letters named.

    :param spec: the marker's letters.
    :returns: the letters, in order.
    """
    letters: List[str] = []
    for part in re.split(r"\s*(?:,|and|&)\s*", spec):
        part = part.strip()
        if not part:
            continue
        span = re.match(r"^([A-Za-z])\s*[-–—]\s*([A-Za-z])$", part)
        if span:
            first, last = span.group(1), span.group(2)
            if ord(last) >= ord(first):
                letters.extend(chr(c) for c in range(ord(first), ord(last) + 1))
            continue
        if len(part) == 1 and part.isalpha():
            letters.append(part)
    return letters


_LEGEND_MARKER = re.compile(
    r"\(([A-Za-z](?:\s*(?:[-–—,&]|and)\s*[A-Za-z])*)\)\s*"
    r"|(?:^|(?<=[.;:)\s]))([A-Za-z](?:\s*(?:[-–—]|,|and|&)\s*[A-Za-z])*)"
    r"\s*[,.):]\s+")


def split_legend(caption: str) -> Dict[str, str]:
    """A figure legend cut into one passage per panel letter.

    Legends mark panels as ``(A)``, ``A,``, ``A.`` or ``A)``, sometimes glued
    to the title (``...growth.A, strategy``), sometimes as a range
    (``B-E, diagnostic PCRs``). A letter is accepted as a marker only when
    it is the NEXT letter after the last marker, which is what stops
    ``(B, D)`` inside panel B-E's own text, or ``P. falciparum``, from
    being read as markers.

    :param caption: the legend.
    :returns: ``{letter: passage}`` in upper case, plus ``''`` for the title
        sentence before the first marker. Empty when no marker was found.
    """
    text = " ".join(str(caption or "").split())
    accepted: List[Tuple[int, int, List[str]]] = []
    expected = "A"
    for match in _LEGEND_MARKER.finditer(text):
        letters = _expand_letters(match.group(1) or match.group(2) or "")
        if not letters or letters[0].upper() != expected:
            continue
        if any(ord(b.upper()) <= ord(a.upper()) for a, b in zip(letters, letters[1:])):
            continue
        accepted.append((match.start(), match.end(), [c.upper() for c in letters]))
        expected = chr(ord(letters[-1].upper()) + 1)
    if not accepted:
        return {}
    out: Dict[str, str] = {"": text[:accepted[0][0]].strip()}
    for index, (_start, end, letters) in enumerate(accepted):
        stop = accepted[index + 1][0] if index + 1 < len(accepted) else len(text)
        passage = text[end:stop].strip()
        for letter in letters:
            out[letter] = passage
    return out


def read_words(image: Any, *, engine: Optional[Callable] = None) -> List[Word]:
    """The text printed in a figure image, with where it is.

    :param image: an image path or an ``H x W x 3`` array.
    :param engine: ``fn(image) -> (results, elapsed)`` in RapidOCR's shape,
        each result ``[box points, text, confidence]``; defaults to RapidOCR.
    :returns: the words found, top-to-bottom then left-to-right.
    :raises ImportError: when no engine is given and RapidOCR is missing.
    """
    if engine is None:
        engine = _rapidocr()
    results, _elapsed = engine(image if not isinstance(image, Path) else str(image))
    words: List[Word] = []
    for box, text, confidence in results or ():
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        words.append(Word(str(text), min(xs), min(ys), max(xs), max(ys),
                          float(confidence)))
    words.sort(key=lambda w: (w.y0, w.x0))
    return words


_ENGINE: Dict[str, Any] = {}

#: The figure reader's own environment: ultralytics and RapidOCR
#: installed apart from spaCR, so their torch and opencv cannot change it.
READER_BACKEND = "papers"


def _importable(module: str) -> bool:
    """Whether ``module`` can be imported here, without importing it.

    :param module: a top-level module name.
    :returns: True when it is on this interpreter's path.
    """
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def reader_environment() -> Optional[str]:
    """The figure reader's own environment when it is installed, else None.

    :returns: the environment folder.
    """
    try:
        from ._segmentation_backends import _backend_state

        state = _backend_state(READER_BACKEND)
    except Exception:
        return None
    return state.env if state.ready and not state.in_process else None


class ReaderNeedsInstall(ImportError):
    """The figure reader must be installed, or installed again, first.

    An :class:`ImportError`, so every caller that already reported a missing
    reader still does; Plaque Assay catches this one to offer the install
    in place.

    :param message: what is missing and why.
    :param reinstall: True when the reader is installed but was built
        before a package it now needs was pinned.
    """

    def __init__(self, message: str, *, reinstall: bool = False) -> None:
        """Keep whether this is a first install or a reinstall."""
        super().__init__(message)
        self.reinstall = bool(reinstall)


def reader_problem(*, pdf: bool = False) -> Tuple[str, str]:
    """Whether Figure mode can read now, and what to do when it cannot.

    Read from disk only -- spaCR's own packages and the reader's install
    record (:func:`spacr._segmentation_backends._stale_requirements`) -- so
    it is cheap enough for the GUI thread.

    :param pdf: the question is about reading a PDF, which needs
        pdfplumber; otherwise the detector and OCR.
    :returns: ``('', '')`` when ready; ``('install', why)`` when the reader
        is not installed; ``('reinstall', why)`` when it is installed
        without a package spaCR now pins for it.
    """
    if pdf and _importable("pdfplumber"):
        return "", ""
    if not pdf and _importable("ultralytics") and _importable("rapidocr_onnxruntime"):
        return "", ""
    try:
        from ._segmentation_backends import _backend_state, _stale_requirements

        state = _backend_state(READER_BACKEND)
    except Exception as exc:
        return "install", str(exc)
    if state.in_process:
        return "", ""
    if not state.ready:
        return "install", state.reason
    stale = [item for item in _stale_requirements(READER_BACKEND, state.record)
             if pdf or not item.lower().startswith("pdfplumber")]
    if stale:
        return "reinstall", (
            "The plaque figure reader in {env} was installed before spaCR "
            "pinned {packages} for it; installing it again adds them.".format(
                env=state.env, packages=", ".join(stale)))
    return "", ""


def _reader_request(op: str, image: Any, **payload: Any) -> Dict[str, Any]:
    """Send one image to the figure reader's worker and return its reply.

    :param op: ``detect`` or ``read_text``.
    :param image: an ``H x W x 3`` array or an image path.
    :param payload: the request's other fields.
    :returns: the reply.
    :raises ImportError: when the reader is not installed anywhere.
    """
    import tempfile

    from ._segmentation_backends import _worker_for

    env = reader_environment()
    if env is None:
        raise ReaderNeedsInstall(
            "Figure mode needs the plaque figure reader (YOLO and RapidOCR). "
            "Install it from Plaque Assay's Figure mode or the Model Zoo; it "
            "goes into an environment of its own.")
    array = image if isinstance(image, np.ndarray) else _load_image(Path(image))
    with tempfile.TemporaryDirectory(prefix="spacr_reader_") as folder:
        path = str(Path(folder) / "image.npy")
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        return _worker_for(READER_BACKEND, env).request(op, image=path, **payload)


def reader_detect(image: np.ndarray, weights: str, *, confidence: float = 0.25,
                  imgsz: int = 640, min_axis_ratio: float = 0.0) -> List[Any]:
    """Run :func:`spacr.plaque.detect_wells` in the figure reader's environment.

    :param image: the figure, RGB.
    :param weights: the detector checkpoint.
    :param confidence: minimum score.
    :param imgsz: the inference size.
    :param min_axis_ratio: accepted for the same signature; nothing is
        dropped for its shape here, as in Figure mode's in-process call.
    :returns: boxes with ``x0, y0, x1, y1, confidence``.
    """
    from types import SimpleNamespace

    reply = _reader_request("detect", image, weights=str(weights),
                            imgsz=[int(imgsz)], confidence=float(confidence))
    return [SimpleNamespace(x0=b[0], y0=b[1], x1=b[2], y1=b[3], confidence=b[4])
            for b in reply.get("boxes", [])]


def _reader_ocr(image: Any):
    """RapidOCR's call shape, answered by the reader's environment.

    :param image: an array or a path.
    :returns: ``(results, None)`` as RapidOCR returns them.
    """
    reply = _reader_request("read_text", image)
    return [tuple(word) for word in reply.get("words", [])], None


def default_detect() -> Callable:
    """The detector call Figure mode uses: in process, or in the reader.

    :returns: :func:`spacr.plaque.detect_wells` when ultralytics imports in
        spaCR itself, else :func:`reader_detect`.
    """
    if _importable("ultralytics") or reader_environment() is None:
        from .plaque import detect_wells

        return detect_wells
    return reader_detect


def _rapidocr() -> Callable:
    """The RapidOCR engine: in process when installed, else the reader's.

    :returns: the engine.
    :raises ImportError: when RapidOCR is available neither way.
    """
    if "rapidocr" not in _ENGINE:
        if _importable("rapidocr_onnxruntime"):
            from rapidocr_onnxruntime import RapidOCR

            _ENGINE["rapidocr"] = RapidOCR()
        elif reader_environment() is not None:
            return _reader_ocr
        else:
            raise ReaderNeedsInstall(
                "Reading the text in figure images needs RapidOCR. Install "
                "the plaque figure reader from Plaque Assay's Figure mode or "
                "the Model Zoo; it goes into an environment of its own.")
    return _ENGINE["rapidocr"]


def _iou(a: Region, b: Region) -> float:
    """Intersection over union of two boxes.

    :param a: one box.
    :param b: the other.
    :returns: a value in 0..1.
    """
    ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def find_plaque_regions(image: np.ndarray, weights: Any, *,
                        imgsz: Sequence[int] = DEFAULT_IMGSZ,
                        confidence: float = 0.25, iou: float = 0.5,
                        detect: Optional[Callable] = None) -> List[Region]:
    """Plaque images in one figure, asked at every inference size given.

    One size is not right for everything a literature corpus prints: at 640
    the detector finds whole faint panels and misses dilution-spot strips, at
    1280 the reverse (``features/data/424_imgsz_sweep_2026-09-20.json``). So
    each size is asked and the boxes are merged, a box found at two sizes
    kept once with the higher score and both sizes recorded.

    :param image: the figure, ``H x W x 3`` RGB.
    :param weights: detector checkpoint path.
    :param imgsz: the inference sizes to ask at.
    :param confidence: minimum detector score.
    :param iou: boxes overlapping more than this are the same region.
    :param detect: ``fn(image, weights, confidence=, imgsz=, min_axis_ratio=)
        -> boxes with x0, y0, x1, y1, confidence``; defaults to
        :func:`spacr.plaque.detect_wells`. Nothing is dropped for not being
        square: a figure crop need not be a round well.
    :returns: the regions, top-to-bottom then left-to-right.
    """
    if detect is None:
        detect = default_detect()
    merged: List[Region] = []
    for size in imgsz:
        for box in detect(image, weights, confidence=confidence,
                          imgsz=int(size), min_axis_ratio=0.0):
            found = Region(int(box.x0), int(box.y0), int(box.x1), int(box.y1),
                           float(box.confidence), (int(size),))
            for index, kept in enumerate(merged):
                if _iou(kept, found) > iou:
                    better = found if found.confidence > kept.confidence else kept
                    merged[index] = replace(
                        better, sizes=tuple(sorted(set(kept.sizes) | {int(size)})))
                    break
            else:
                merged.append(found)
    merged.sort(key=lambda r: (r.y0, r.x0))
    return merged


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length two intervals share.

    :returns: the shared length, 0 when disjoint.
    """
    return max(0.0, min(a1, b1) - max(a0, b0))


def _nearest_line(candidates: List[Tuple[float, Word]]) -> List[Word]:
    """The words on the line nearest the region, from ``(gap, word)`` pairs.

    :param candidates: every word on one side, with its gap to the region.
    :returns: the nearest word and those on the same line with it.
    """
    if not candidates:
        return []
    candidates.sort(key=lambda pair: pair[0])
    first = candidates[0][1]
    tolerance = max(4.0, (first.y1 - first.y0) * 0.6)
    line = [w for _gap, w in candidates if abs(w.cy - first.cy) <= tolerance]
    return sorted(line, key=lambda w: w.x0)


def _nearest_column(candidates: List[Tuple[float, Word]]) -> List[Word]:
    """The words in the column nearest the region, from ``(gap, word)`` pairs.

    :param candidates: every word on one side, with its gap to the region.
    :returns: the nearest word and those stacked in its column.
    """
    if not candidates:
        return []
    candidates.sort(key=lambda pair: pair[0])
    first = candidates[0][1]
    tolerance = max(4.0, (first.x1 - first.x0) * 0.6)
    column = [w for _gap, w in candidates if abs(w.cx - first.cx) <= tolerance]
    return sorted(column, key=lambda w: w.y0)


def _blocks(regions: Sequence[Region]) -> List[int]:
    """Which grid of images each region belongs to.

    Two images are in one grid when they sit side by side or one above the
    other with a gap under half the smaller image, which is how a panel of
    plaque crops is printed.

    :param regions: every plaque image in the figure.
    :returns: a block number per region.
    """
    parent = list(range(len(regions)))

    def root(i: int) -> int:
        """The block ``i`` belongs to, compressing the path on the way."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            b = regions[j]
            small = 0.5 * min(a.width, a.height, b.width, b.height)
            gap_x = max(b.x0 - a.x1, a.x0 - b.x1)
            gap_y = max(b.y0 - a.y1, a.y0 - b.y1)
            side = gap_x <= small and _overlap(a.y0, a.y1, b.y0, b.y1) >= 0.3 * min(a.height, b.height)
            stacked = gap_y <= small and _overlap(a.x0, a.x1, b.x0, b.x1) >= 0.3 * min(a.width, b.width)
            if side or stacked:
                parent[root(j)] = root(i)
    return [root(i) for i in range(len(regions))]


@dataclass(frozen=True)
class TextOptions:
    """How the text around a plaque image is turned into its condition.

    Text near a panel is usually read correctly but not always assigned to
    the right image, so every choice the assignment makes is exposed here.
    These are the knobs the reading has; the defaults
    are the values measured on PMC9744290 Fig 6 and Fig 7 (8 of 8 read).

    :ivar reach_above: how far above the grid a column header may sit, in
        image heights.
    :ivar reach_left: how far left of the grid a row label may sit, in image
        widths.
    :ivar reach_below: how far below the grid text may sit, in image heights.
    :ivar use_above: take the column header.
    :ivar use_left: take the row label.
    :ivar use_below: take text printed under the images.
    :ivar panel_reach: how far up and left of the grid's corner a panel letter
        may sit, in image sizes.
    :ivar min_confidence: OCR words scoring below this are ignored.
    :ivar ignore: regular expressions; a word matching any is ignored (scale
        bars such as ``5 μm``, axis numbers).
    :ivar order: which labels come first in the condition, e.g.
        ``("above", "left", "below")``.
    :ivar separator: what joins the labels into one condition.
    :ivar reread: read the text around each grid again, enlarged.
    :ivar reread_scale: how much to enlarge for that second reading.
    """

    reach_above: float = 1.0
    reach_left: float = 1.0
    reach_below: float = 0.5
    use_above: bool = True
    use_left: bool = True
    use_below: bool = True
    panel_reach: float = 1.0
    min_confidence: float = 0.0
    ignore: Tuple[str, ...] = ()
    order: Tuple[str, ...] = ("above", "left", "below")
    separator: str = " / "
    reread: bool = True
    reread_scale: int = 3

    def keeps(self, word: "Word") -> bool:
        """Whether a word survives the confidence and ignore filters.

        :param word: the word.
        :returns: True to keep it.
        """
        if float(word.confidence) < float(self.min_confidence):
            return False
        text = word.text.strip()
        return not any(re.search(pattern, text) for pattern in self.ignore
                       if pattern)


DEFAULT_TEXT_OPTIONS = TextOptions()


def text_options_from_settings(settings: Mapping[str, Any]) -> TextOptions:
    """:class:`TextOptions` from a settings dict's ``text_*`` keys.

    :param settings: settings; missing keys keep the defaults.
    :returns: the options.
    """
    base = DEFAULT_TEXT_OPTIONS
    def pick(key, default, cast):
        """``settings[key]`` cast by ``cast``, or ``default`` if blank or invalid."""
        value = settings.get(key)
        if value in (None, ""):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default
    def patterns(value):
        """A tuple of patterns from a comma-separated string or a sequence."""
        if isinstance(value, str):
            return tuple(p.strip() for p in value.split(",") if p.strip())
        return tuple(str(p) for p in value)
    def sides(value):
        """The valid sides named in ``value``, or the default order if none are."""
        if isinstance(value, str):
            value = [p.strip() for p in value.split(",")]
        chosen = tuple(v for v in value if v in ("above", "left", "below"))
        return chosen or base.order
    return TextOptions(
        reach_above=pick("text_reach_above", base.reach_above, float),
        reach_left=pick("text_reach_left", base.reach_left, float),
        reach_below=pick("text_reach_below", base.reach_below, float),
        use_above=pick("text_use_above", base.use_above, bool),
        use_left=pick("text_use_left", base.use_left, bool),
        use_below=pick("text_use_below", base.use_below, bool),
        panel_reach=pick("text_panel_reach", base.panel_reach, float),
        min_confidence=pick("text_min_confidence", base.min_confidence, float),
        ignore=pick("text_ignore", base.ignore, patterns),
        order=pick("text_order", base.order, sides),
        separator=pick("text_separator", base.separator, str),
        reread=pick("text_reread", base.reread, bool),
        reread_scale=pick("text_reread_scale", base.reread_scale, int))


def text_near(region: Region, words: Sequence[Word], *,
              regions: Sequence[Region] = (), reach: float = 1.0,
              options: Optional[TextOptions] = None) -> Dict[str, Any]:
    """The text a reader would take as this image's label.

    Distances are measured from the edge of the GRID the image sits in, not
    from the image itself, because a header printed once above a column
    labels every row in it and a row label printed once at the left labels
    every column:

    * **above** -- the nearest line of text over the image's column.
    * **left** -- the nearest text level with the image's row, left of the
      grid, including a label printed rotated.
    * **below** -- the nearest line under the image's column.
    * **panel** -- a panel letter above and to the left of the grid's top-left
      corner, within one image size of it (``panel_reach``, 1.0 by
      default). A letter further away is taken
      to belong to another panel and is not used.

    Text inside any plaque image (a scale bar, an inset label) is not a
    label and is left out.

    :param region: the image.
    :param words: every word in the figure.
    :param regions: every plaque image in the figure, including ``region``.
    :param reach: how far past the grid to look, in image sizes; used when
        ``options`` is not given.
    :param options: the reading's settings (:class:`TextOptions`).
    :returns: ``{'panel': letter or None, 'above': [...], 'left': [...],
        'below': [...]}``.
    """
    if options is None:
        options = replace(DEFAULT_TEXT_OPTIONS, reach_above=reach,
                          reach_left=reach, reach_below=0.5 * reach)
    boxes = list(regions) or [region]
    if region not in boxes:
        boxes.append(region)
    block = _blocks(boxes)
    mine = block[boxes.index(region)]
    grid = [b for b, k in zip(boxes, block) if k == mine]
    column = [b for b in grid
              if _overlap(b.x0, b.x1, region.x0, region.x1) >= 0.5 * region.width]
    row = [b for b in grid
           if _overlap(b.y0, b.y1, region.y0, region.y1) >= 0.5 * region.height]
    top = min(b.y0 for b in column)
    bottom = max(b.y1 for b in column)
    left_edge = min(b.x0 for b in row)
    free = [w for w in words if not any(r.contains(w) for r in boxes)
            and options.keeps(w)]
    above, left, below = [], [], []
    for w in free:
        if w.panel_letter is not None:
            continue
        width, height = w.x1 - w.x0, w.y1 - w.y0
        if _overlap(w.x0, w.x1, region.x0, region.x1) >= 0.3 * max(width, 1):
            gap = top - w.y1
            if options.use_above and (
                    -0.5 * height <= gap <= options.reach_above * region.height):
                above.append((gap, w))
            gap = w.y0 - bottom
            if options.use_below and (
                    -0.5 * height <= gap <= options.reach_below * region.height):
                below.append((gap, w))
        if _overlap(w.y0, w.y1, region.y0, region.y1) >= 0.3 * max(height, 1):
            gap = left_edge - w.x1
            if options.use_left and (
                    -0.5 * width <= gap <= options.reach_left * region.width):
                left.append((gap, w))
    return {"panel": _panel_letter(free, grid, options.panel_reach),
            "above": [w.text for w in _nearest_line(above)],
            "left": [w.text for w in _nearest_column(left)],
            "below": [w.text for w in _nearest_line(below)]}


def _panel_window(grid: Sequence[Region], reach: float = 1.0
                  ) -> Tuple[float, float, float, float]:
    """Where a grid's panel letter can be: up and left of its corner.

    :param grid: the images of one grid.
    :param reach: how far up and left, in image sizes.
    :returns: ``(x0, y0, x1, y1)`` in figure pixels.
    """
    x0 = min(b.x0 for b in grid)
    y0 = min(b.y0 for b in grid)
    size_w = float(np.median([b.width for b in grid]))
    size_h = float(np.median([b.height for b in grid]))
    return (x0 - reach * size_w, y0 - reach * size_h,
            x0 + 0.25 * size_w, y0 + 0.25 * size_h)


def _panel_letter(words: Sequence[Word], grid: Sequence[Region],
                  reach: float = 1.0) -> Optional[str]:
    """The panel letter printed at a grid's top-left corner, if any.

    :param words: the figure's words outside every image.
    :param grid: the images of one grid.
    :param reach: how far from the corner, in image sizes.
    :returns: the letter in upper case, or None.
    """
    wx0, wy0, wx1, wy1 = _panel_window(grid, reach)
    corner = (min(b.x0 for b in grid), min(b.y0 for b in grid))
    best, letter = None, None
    for w in words:
        found = w.panel_letter
        if found is None or not (wx0 <= w.cx <= wx1 and wy0 <= w.cy <= wy1):
            continue
        distance = float(np.hypot(corner[0] - w.cx, corner[1] - w.cy))
        if best is None or distance < best:
            best, letter = distance, found.upper()
    return letter


def reread_around(image: np.ndarray, regions: Sequence[Region],
                  words: Sequence[Word], *, engine: Optional[Callable] = None,
                  scale: int = 3) -> List[Word]:
    """Read the text around each grid of plaque images again, enlarged.

    A figure's labels are small: a lone panel letter or a rotated ``+ATc``
    beside a crop is a few pixels tall, and OCR run on the whole figure
    misses some (Fig 7 of PMC9744290 loses its ``F`` and one row label).
    Each grid's surroundings -- one image size up and left, half down and
    right -- are cut out, enlarged ``scale`` times and read again, and any
    word not already found is added in figure coordinates.

    :param image: the figure, ``H x W x 3``.
    :param regions: its plaque images.
    :param words: the words already read.
    :param engine: a RapidOCR-shaped engine; defaults to RapidOCR.
    :param scale: the enlargement.
    :returns: ``words`` plus whatever the second reading found.
    """
    from PIL import Image

    if not regions:
        return list(words)
    height, width = image.shape[:2]
    out = list(words)
    block = _blocks(regions)
    for key in sorted(set(block)):
        grid = [r for r, k in zip(regions, block) if k == key]
        size_w = float(np.median([r.width for r in grid]))
        size_h = float(np.median([r.height for r in grid]))
        x0 = int(max(0, min(r.x0 for r in grid) - size_w))
        y0 = int(max(0, min(r.y0 for r in grid) - size_h))
        x1 = int(min(width, max(r.x1 for r in grid) + 0.5 * size_w))
        y1 = int(min(height, max(r.y1 for r in grid) + 0.5 * size_h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crop = Image.fromarray(np.ascontiguousarray(image[y0:y1, x0:x1]))
        big = np.asarray(crop.resize(((x1 - x0) * scale, (y1 - y0) * scale),
                                     Image.BICUBIC))
        for w in read_words(big, engine=engine):
            found = Word(w.text, x0 + w.x0 / scale, y0 + w.y0 / scale,
                         x0 + w.x1 / scale, y0 + w.y1 / scale, w.confidence)
            if any(_overlap(found.x0, found.x1, o.x0, o.x1) > 0
                   and _overlap(found.y0, found.y1, o.y0, o.y1) > 0
                   for o in out):
                continue
            out.append(found)
    return out


def _tokens(text: str) -> set:
    """Lower-case words of two or more characters.

    :param text: any text.
    :returns: the set of tokens.
    """
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) >= 2}


def _grid_positions(regions: Sequence[Region]) -> List[Tuple[int, int]]:
    """Row and column of each region among the regions given.

    Rows are regions whose vertical centres lie within half a region height
    of each other; columns the same, horizontally.

    :param regions: the regions of one panel.
    :returns: ``(row, column)`` per region, both starting at 1.
    """
    def ranks(centres: List[float], sizes: List[float]) -> List[int]:
        """A 1-based rank per centre, sharing a rank within half a size."""
        order = sorted(range(len(centres)), key=lambda i: centres[i])
        out = [0] * len(centres)
        rank, last = 0, None
        for i in order:
            if last is None or centres[i] - last > 0.5 * sizes[i]:
                rank += 1
                last = centres[i]
            out[i] = rank
        return out
    rows = ranks([(r.y0 + r.y1) / 2 for r in regions], [r.height for r in regions])
    cols = ranks([(r.x0 + r.x1) / 2 for r in regions], [r.width for r in regions])
    return list(zip(rows, cols))


def annotate_regions(regions: Sequence[Region], words: Sequence[Word], *,
                     caption: str = "", figure_label: str = "",
                     options: Optional[TextOptions] = None) -> List[Annotation]:
    """Propose a condition for every plaque image in one figure.

    Both strategies run for every image. See the module docstring for how
    they combine and what ``source``, ``strength`` and ``conflict`` mean.

    :param regions: the figure's plaque images.
    :param words: the figure's text.
    :param caption: the figure legend, ``''`` when unknown.
    :param figure_label: the figure's name, used when nothing else is known.
    :param options: how the text is read (:class:`TextOptions`); defaults
        to :data:`DEFAULT_TEXT_OPTIONS`.
    :returns: one annotation per region, in the order given.
    """
    options = options or DEFAULT_TEXT_OPTIONS
    legend = split_legend(caption)
    near = [text_near(r, words, regions=regions, options=options)
            for r in regions]
    by_panel: Dict[Optional[str], List[int]] = {}
    for index, found in enumerate(near):
        by_panel.setdefault(found["panel"], []).append(index)
    positions: Dict[int, Tuple[int, int]] = {}
    for members in by_panel.values():
        for index, spot in zip(members, _grid_positions([regions[i] for i in members])):
            positions[index] = spot
    out: List[Annotation] = []
    for index, region in enumerate(regions):
        found = near[index]
        label_parts = [" ".join(found[side]) for side in options.order
                       if found.get(side)]
        label_text = options.separator.join(label_parts)
        legend_text = legend.get(found["panel"] or "", "") if found["panel"] else ""
        row, column = positions[index]
        annotation = Annotation(
            region=region, panel=found["panel"], row=row, column=column,
            near={k: list(found[k]) for k in ("above", "left", "below")},
            label_text=label_text, legend_text=legend_text)
        if label_text and legend_text:
            annotation.condition = label_text
            annotation.source = "label+legend"
            agree = bool(_tokens(label_text) & _tokens(legend_text))
            annotation.strength = "strong" if agree else "medium"
        elif label_text:
            annotation.condition, annotation.source = label_text, "label"
            annotation.strength = "medium"
        elif legend_text:
            annotation.condition, annotation.source = legend_text, "legend"
            annotation.strength = "medium"
        else:
            where = f"panel {found['panel']}" if found["panel"] else "no panel"
            annotation.condition = (f"{figure_label or 'figure'}, {where}, "
                                    f"row {row}, column {column}")
            annotation.source, annotation.strength = "position", "weak"
        out.append(annotation)
    _flag_conflicts(out)
    return out


def _flag_conflicts(annotations: Sequence[Annotation]) -> List[Annotation]:
    """Mark the images whose label and legend passage disagree.

    The figure's own labels are the vocabulary: every word printed beside
    any of its plaque images. A legend passage that uses none of that
    vocabulary ("plaque assays under the indicated conditions") says nothing
    about which image is which, and is not a disagreement. A passage that
    names some of it -- but none of the words beside THIS image -- is
    describing other conditions than the label here, and one of the two
    readings is wrong. Which one is not decided here; both are kept and the
    row is flagged.

    :param annotations: one figure's annotations; changed in place.
    :returns: the same annotations.
    """
    vocabulary: set = set()
    for a in annotations:
        vocabulary |= _tokens(a.label_text)
    for a in annotations:
        a.conflict, a.conflict_terms = False, []
        if a.source != "label+legend":
            continue
        own = _tokens(a.label_text)
        legend = _tokens(a.legend_text)
        if own & legend:
            continue
        named = sorted((legend & vocabulary) - own)
        if named:
            a.conflict, a.conflict_terms = True, named
    return annotations


def measure_region(labels: np.ndarray, *, px_per_mm: Optional[float] = None
                   ) -> List[Dict[str, Any]]:
    """One row per plaque in a segmented plaque image.

    :param labels: the segmenter's label image, 0 = background, or
        ``(labels, per_label_metrics)`` from a diagnostic segmenter.
    :param px_per_mm: the image's scale, when it has a ruler.
    :returns: ``[{'label', 'area_px', 'area_mm2'}]``; ``area_mm2`` is None
        without a ruler.
    """
    metrics = {}
    if isinstance(labels, tuple):
        labels, metrics = labels
    from .plaque import _PLAQUE_METRIC_COLUMNS
    ids, counts = np.unique(np.asarray(labels), return_counts=True)
    rows = []
    for label, area in zip(ids.tolist(), counts.tolist()):
        if label == 0:
            continue
        mm2 = float(area) / (px_per_mm ** 2) if px_per_mm else None
        rows.append({"label": int(label), "area_px": int(area), "area_mm2": mm2,
                     **metrics.get(int(label), dict.fromkeys(_PLAQUE_METRIC_COLUMNS))})
    return rows


def calibration_number(value, *, name, allow_zero=False):
    """Parse an optional finite calibration value without treating invalid input as missing."""
    if value is None or str(value).strip() in ('', 'None'):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a number or blank.') from exc
    if not np.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f'{name} must be finite and ' + ('nonnegative.' if allow_zero else 'positive.'))
    return number


def calibration_values(annotation, scale=None, formation_hours=None):
    """Return per-well size, scale and timing metadata for preview and saved tables.

    The well diameter is the mean detected bounding-box extent, not a fitted
    circle or a segmentation-model diameter. It remains available without a
    physical ruler. Time is entered metadata, never inferred from plaque size.
    """
    region = annotation.region
    hours = calibration_number(annotation.formation_hours, name='formation_hours', allow_zero=True)
    source = 'manual annotation' if hours is not None else 'unknown'
    if hours is None:
        hours = calibration_number(formation_hours, name='formation_hours', allow_zero=True)
        if hours is not None:
            source = 'settings'
    ppm = getattr(scale, 'px_per_mm', None)
    return dict(well_diameter_px=(region.width + region.height) / 2.0,
                well_diameter_method='mean detected bounding-box extent',
                pixels_per_um=ppm / 1000.0 if ppm else None,
                formation_hours=hours, formation_time_source=source)


def _ruler(region: Region, plate_format: Optional[str],
           min_axis_ratio: float = 0.9) -> Optional[float]:
    """Pixels per mm when the image is a whole well of a known plate.

    :param region: the image.
    :param plate_format: a key of :data:`spacr.plaque.WELL_DIAMETERS_MM`.
    :param min_axis_ratio: how square a box must be to be a whole well.
    :returns: the scale, or None when there is no ruler.
    """
    if not plate_format or region.axis_ratio < min_axis_ratio:
        return None
    from .plaque import WELL_DIAMETERS_MM

    diameter_mm = WELL_DIAMETERS_MM.get(plate_format)
    if not diameter_mm:
        return None
    return ((region.width + region.height) / 2.0) / diameter_mm


_UNIT_MM = {"nm": 1e-6, "um": 1e-3, "µm": 1e-3, "μm": 1e-3, "mm": 1.0,
            "cm": 10.0}
_SCALE_TEXT = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(nm|um|µm|μm|mm|cm)$", re.IGNORECASE)
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
_UNIT = re.compile(r"^(nm|um|µm|μm|mm|cm)$", re.IGNORECASE)
_LEGEND_BAR = re.compile(
    r"scale\s*bars?\s*(?:[,:=]|represents?|indicates?|are|is|of)?\s*"
    r"(\d+(?:[.,]\d+)?)\s*(nm|um|µm|μm|mm|cm)\b", re.IGNORECASE)
_LEGEND_PLATE = re.compile(r"\b(6|12|24|48|96)\s*-?\s*well", re.IGNORECASE)
_LEGEND_MAGNIFICATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*[x×]\s*(?:magnification|objective|lens)"
    r"|magnification[^.;]{0,20}?(\d+(?:\.\d+)?)\s*[x×]", re.IGNORECASE)


@dataclass(frozen=True)
class _Scale:
    """How a plaque image's pixels become millimetres, or why they do not.

    :param px_per_mm: pixels per millimetre, or None when there is no ruler
        and sizes stay in pixels.
    :param source: ``'scale bar'`` (read in or under this image),
        ``'scale bar, same panel'`` (read on another image of the same size
        in the same grid), ``'scale bar, length from legend'``,
        ``'well: <format> (settings)'``, ``'well: <format> (legend)'``,
        ``'image metadata'`` (the pixel size the image file records) or
        ``'none'``.
    :param detail: what was measured, e.g. ``'1 mm bar = 120 px'``.
    :param magnification: a magnification the legend states, recorded and
        never used as a ruler.
    """

    px_per_mm: Optional[float] = None
    source: str = "none"
    detail: str = ""
    magnification: str = ""

    @property
    def unit(self) -> str:
        """``'mm2'`` when areas can be given in mm^2, else ``'px'``."""
        return "mm2" if self.px_per_mm else "px"


def _unit_mm(unit: str) -> float:
    """Millimetres in one ``unit``.

    :param unit: ``nm``, ``um``, ``µm``, ``mm`` or ``cm``, any case.
    :returns: the factor.
    """
    unit = unit.strip()
    return _UNIT_MM.get(unit, _UNIT_MM.get(unit.lower(), 1.0))


def _scale_labels(words: Sequence[Word]) -> List[Tuple[Word, float]]:
    """The words that state a length, such as ``1 mm`` or ``500 µm``.

    A number and its unit read as two words on one line are joined.

    :param words: a figure's words.
    :returns: ``(word, millimetres)`` pairs, the word spanning number and
        unit.
    """
    out: List[Tuple[Word, float]] = []
    used = set()
    ordered = sorted(words, key=lambda w: (w.y0, w.x0))
    for index, w in enumerate(ordered):
        match = _SCALE_TEXT.match(w.text.strip())
        if match:
            value = float(match.group(1).replace(",", "."))
            out.append((w, value * _unit_mm(match.group(2))))
            continue
        if index in used or not _NUMBER.match(w.text.strip()):
            continue
        height = max(1.0, w.y1 - w.y0)
        for j in range(index + 1, len(ordered)):
            unit = ordered[j]
            if not _UNIT.match(unit.text.strip()):
                continue
            if abs(unit.cy - w.cy) <= 0.6 * height and \
                    0 <= unit.x0 - w.x1 <= 1.5 * height:
                value = float(w.text.strip().replace(",", "."))
                joined = Word(f"{w.text.strip()} {unit.text.strip()}", w.x0,
                              min(w.y0, unit.y0), unit.x1, max(w.y1, unit.y1),
                              min(w.confidence, unit.confidence))
                out.append((joined, value * _unit_mm(unit.text.strip())))
                used.add(j)
                break
    return out


def _bar_components(gray: np.ndarray, *, max_thickness: float,
                    min_length: int) -> List[Tuple[int, int, int, int]]:
    """Solid, thin, horizontal bars in a greyscale window, either polarity.

    :param gray: the window, ``H x W`` 0..255.
    :param max_thickness: the thickest a bar may be, in pixels.
    :param min_length: the shortest a bar may be, in pixels.
    :returns: ``(y0, x0, y1, x1)`` per bar that does not touch the window's
        left or right edge -- a gutter between two images runs out of the
        window, a scale bar ends inside it.
    """
    from skimage.measure import label as label_components, regionprops

    height, width = gray.shape[:2]
    out = []
    for mask in (gray < 80, gray > 190):
        for prop in regionprops(label_components(mask, connectivity=2)):
            y0, x0, y1, x1 = prop.bbox
            thick, long = y1 - y0, x1 - x0
            if x0 == 0 or x1 == width or long < min_length:
                continue
            if thick > max_thickness or long < 3 * thick:
                continue
            if prop.area < 0.85 * thick * long:
                continue
            out.append((y0, x0, y1, x1))
    return out


def _measure_scale_bar(image: np.ndarray, word: Word) -> Optional[Tuple[int, int, int, int]]:
    """The bar a length label such as ``1 mm`` sits beside, if one is there.

    Looked for above and below the label, within three label heights, as a
    solid thin horizontal line in black or white that ends inside the search
    window.

    :param image: the figure, ``H x W x 3``.
    :param word: the length label, from :func:`_scale_labels`.
    :returns: the bar as ``(x0, y0, x1, y1)`` in figure pixels, or None.
    """
    height, width = image.shape[:2]
    h = max(2.0, word.y1 - word.y0)
    w = max(h, word.x1 - word.x0)
    x0 = int(max(0, word.x0 - 3 * w))
    x1 = int(min(width, word.x1 + 3 * w))
    y0 = int(max(0, word.y0 - 3 * h))
    y1 = int(min(height, word.y1 + 3 * h))
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    window = np.asarray(image[y0:y1, x0:x1], dtype=float)
    gray = window.mean(axis=2) if window.ndim == 3 else window
    best, best_key = None, None
    for by0, bx0, by1, bx1 in _bar_components(
            gray, max_thickness=max(2.0, 0.8 * h), min_length=max(8, int(0.5 * h))):
        fx0, fy0, fx1, fy1 = bx0 + x0, by0 + y0, bx1 + x0, by1 + y0
        if _overlap(fy0, fy1, word.y0, word.y1) > 0 and \
                _overlap(fx0, fx1, word.x0, word.x1) > 0:
            continue
        if _overlap(fx0, fx1, word.x0 - 0.5 * w, word.x1 + 0.5 * w) <= 0:
            continue
        gap = min(abs(fy0 - word.y1), abs(word.y0 - fy1))
        key = (gap, -(fx1 - fx0))
        if best_key is None or key < best_key:
            best, best_key = (fx0, fy0, fx1, fy1), key
    return best


def _unlabelled_bar(image: np.ndarray, region: Region) -> Optional[Tuple[int, int, int, int]]:
    """A scale bar printed inside an image with no length beside it.

    Only the bottom quarter of the image is searched, and only a line 5 to
    60 % of the image's width and at most 4 % of its height thick counts,
    which is how an unlabelled scale bar whose length the legend states is
    drawn.

    :param image: the figure.
    :param region: the plaque image.
    :returns: the bar as ``(x0, y0, x1, y1)`` in figure pixels, or None.
    """
    y0 = int(region.y0 + 0.75 * region.height)
    crop = np.asarray(image[y0:region.y1, region.x0:region.x1], dtype=float)
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 20:
        return None
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop
    found = [b for b in _bar_components(
        gray, max_thickness=max(2.0, 0.04 * region.height),
        min_length=max(8, int(0.05 * region.width)))
        if (b[3] - b[1]) <= 0.6 * region.width]
    if len(found) != 1:
        return None
    by0, bx0, by1, bx1 = found[0]
    return (bx0 + region.x0, by0 + y0, bx1 + region.x0, by1 + y0)


def _legend_scale_facts(text: str) -> Dict[str, Any]:
    """What a legend says about scale.

    :param text: a legend or one panel's passage.
    :returns: ``{'bar_mm': float or None, 'bar_text': str,
        'plate_format': '6-well'... or None, 'magnification': str}``.
    """
    text = str(text or "")
    out: Dict[str, Any] = {"bar_mm": None, "bar_text": "",
                           "plate_format": None, "magnification": ""}
    bar = _LEGEND_BAR.search(text)
    if bar:
        out["bar_mm"] = float(bar.group(1).replace(",", ".")) * _unit_mm(bar.group(2))
        out["bar_text"] = bar.group(0)
    plates = {f"{m.group(1)}-well" for m in _LEGEND_PLATE.finditer(text)}
    if len(plates) == 1:
        out["plate_format"] = plates.pop()
    magnification = _LEGEND_MAGNIFICATION.search(text)
    if magnification:
        out["magnification"] = magnification.group(0)
    return out


def _scales_for_regions(image: np.ndarray, regions: Sequence[Region],
                       words: Sequence[Word], *, caption: str = "",
                       annotations: Optional[Sequence[Annotation]] = None,
                       plate_format: Optional[str] = None,
                       pixels_per_um: Optional[float] = None,
                       image_pixels_per_um: Optional[float] = None) -> List[_Scale]:
    """The ruler, if any, for every plaque image in one figure.

    Manual per-annotation ``pixels_per_um`` overrides the global value,
    which overrides the scale the image file itself records
    (``image_pixels_per_um``, e.g. a Bio-Rad ``.scn``'s known field size),
    which overrides automatic rulers. Values must be finite and positive.
    In order of preference among automatic rulers:

    1. a scale bar in or directly under the image, its length read from the
       label beside it (``1 mm``);
    2. a scale bar inside the image with no label, when the panel's legend
       passage (or the legend) states its length (``_Scale bar, 500 µm``);
    3. agreeing scale bars on other images of the same grid and approximately
       the same size, discovered from either labels or legend passages;
    4. a whole well (a nearly square box) of a plate format the settings
       give, or else that the legend names (``6-well plates``) -- the same
       ruler the plate pipeline uses.

    Otherwise there is no ruler and the image's sizes stay in pixels. A
    stated magnification is recorded on every image of the panel but never
    used: a printed figure has been rescaled since the picture was taken.
    More than one plate format in a legend is not guessed between.
    Conflicting peer calibrations leave an image without its own bar in
    pixels, with a conflict note; they do not fall through to well-size
    calibration. An image's own bar takes priority over peer bars.

    :param image: the figure, ``H x W x 3``.
    :param regions: its plaque images.
    :param words: its words, OCR or text layer, including those inside the
        images.
    :param caption: the figure legend.
    :param annotations: the images' annotations, whose ``legend_text`` is
        the panel's passage.
    :param plate_format: a key of :data:`spacr.plaque.WELL_DIAMETERS_MM`
        from the settings; it wins over the legend.
    :param image_pixels_per_um: the pixel scale recorded by the instrument in
        the image file, used when no manual or settings scale is given.
    :returns: one :class:`_Scale` per region.
    """
    pixels_per_um = calibration_number(pixels_per_um, name='pixels_per_um')
    image_pixels_per_um = calibration_number(image_pixels_per_um,
                                             name='image_pixels_per_um')
    whole = _legend_scale_facts(caption)
    if annotations:
        passages = [_legend_scale_facts(a.legend_text) if a.legend_text else {}
                    for a in annotations]
    else:
        passages = [{} for _ in regions]
    own: Dict[int, _Scale] = {}
    for word, mm in _scale_labels(words):
        if mm <= 0:
            continue
        bar = _measure_scale_bar(image, word)
        if bar is None:
            continue
        length = bar[2] - bar[0]
        centre_x, centre_y = (bar[0] + bar[2]) / 2.0, (bar[1] + bar[3]) / 2.0
        for index, r in enumerate(regions):
            inside = r.x0 <= centre_x <= r.x1 and r.y0 <= centre_y <= r.y1
            under = (r.x0 <= centre_x <= r.x1
                     and 0 <= min(bar[1], word.y0) - r.y1 <= 0.25 * r.height)
            if (inside or under) and index not in own:
                own[index] = _Scale(length / mm, "scale bar",
                                   f"{word.text} bar = {length} px")
                break
    for index, r in enumerate(regions):
        if index in own:
            continue
        facts = passages[index] if index < len(passages) else {}
        bar_mm = facts.get("bar_mm") or whole["bar_mm"]
        if bar_mm:
            bar = _unlabelled_bar(image, r)
            if bar is not None:
                length = bar[2] - bar[0]
                stated = facts.get("bar_text") or whole["bar_text"]
                own[index] = _Scale(length / bar_mm, "scale bar, length from legend",
                                    f"{stated!r}: bar = {length} px")
    block = _blocks(regions) if regions else []
    out: List[_Scale] = []
    for index, r in enumerate(regions):
        facts = passages[index] if index < len(passages) else {}
        magnification = facts.get("magnification") or whole["magnification"]
        manual = calibration_number(annotations[index].pixels_per_um, name='pixels_per_um') if annotations and index < len(annotations) else None
        if manual is not None or pixels_per_um is not None:
            value = manual if manual is not None else pixels_per_um
            out.append(_Scale(value * 1000.0, 'manual annotation' if manual is not None else 'settings',
                              f'{value:g} px/µm', magnification))
            continue
        if image_pixels_per_um is not None:
            out.append(_Scale(image_pixels_per_um * 1000.0, 'image metadata',
                              f'{image_pixels_per_um:g} px/µm', magnification))
            continue
        if index in own:
            out.append(replace(own[index], magnification=magnification))
            continue
        peers = [own[j] for j in own if block[j] == block[index]
                     and abs(regions[j].width - r.width) <= 0.1 * max(r.width, 1)
                     and abs(regions[j].height - r.height) <= 0.1 * max(r.height, 1)]
        if peers:
            twin = peers[0]
            if not all(np.isclose(peer.px_per_mm, twin.px_per_mm) for peer in peers[1:]):
                out.append(_Scale(None, "none", "conflicting scale bars; sizes in pixels",
                                  magnification))
                continue
            out.append(_Scale(twin.px_per_mm, "scale bar, same panel",
                             twin.detail, magnification))
            continue
        chosen, where = plate_format, "settings"
        if not chosen:
            chosen, where = facts.get("plate_format") or whole["plate_format"], "legend"
        ppm = _ruler(r, chosen)
        if ppm:
            out.append(_Scale(ppm, f"well: {chosen} ({where})",
                             f"{(r.width + r.height) / 2.0:.0f} px across", magnification))
            continue
        out.append(_Scale(None, "none", "sizes in pixels", magnification))
    return out


def console_legend_prompt(figure: Figure, annotations: Sequence[Annotation], *,
                          ask: Callable[[str], str] = input) -> Optional[str]:
    """Prompt at the terminal for a figure legend that could not be fetched.

    :param figure: the figure whose legend is missing.
    :param annotations: what was read so far, to show which panels matter.
    :param ask: the prompt function.
    :returns: the pasted legend, or None to annotate by hand instead.
    """
    panels = sorted({a.panel for a in annotations if a.panel})
    print(f"\nNo legend found for {figure.label or figure.path.name} "
          f"(panels with plaques: {', '.join(panels) or 'none'}).")
    print("Paste the figure legend on one line, or press Enter to annotate "
          "each plaque image by hand.")
    text = ask("legend> ").strip()
    return text or None


def console_review(figure: Figure, annotations: List[Annotation], *,
                   ask: Callable[[str], str] = input) -> List[Annotation]:
    """Show each proposed condition at the terminal and take OK, an edit, or skip.

    :param figure: the figure the annotations belong to.
    :param annotations: the proposals; changed in place.
    :param ask: the prompt function.
    :returns: the same annotations, reviewed.
    """
    for index, a in enumerate(annotations, start=1):
        print(f"\n{figure.label or figure.path.name} image {index}: panel "
              f"{a.panel or '?'}, row {a.row}, column {a.column}")
        print(f"  text near it : {a.label_text or '-'}")
        print(f"  legend       : {a.legend_text or '-'}")
        flag = "  CONFLICT" if a.conflict else ""
        print(f"  proposed     : {a.condition}  [{a.source}, {a.strength}]{flag}")
        answer = ask("Enter = OK, s = skip, anything else replaces it> ").strip()
        if not answer:
            a.approved = True
        elif answer.lower() == "s":
            a.approved = False
        else:
            a.condition, a.source, a.strength = answer, "manual", "manual"
            a.approved = True
    return annotations


#: The results database, table by table: ``(column, SQL type)`` in order.
#: A database written by an older spaCR is brought up to this by adding the
#: columns it lacks (:func:`open_database`); nothing is dropped or renamed.
TABLES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "runs": (
        ("run_id", "INTEGER PRIMARY KEY AUTOINCREMENT"), ("started", "REAL"),
        ("finished", "REAL"), ("entry", "TEXT"), ("source", "TEXT"),
        ("detector", "TEXT"), ("segmenter", "TEXT"), ("imgsz", "TEXT"),
        ("confidence", "REAL"), ("plate_format", "TEXT"),
        ("confirm_each", "INTEGER"), ("text_options", "TEXT"),
        ("spacr_version", "TEXT"), ("summary", "TEXT")),
    "papers": (
        ("paper_key", "TEXT PRIMARY KEY"), ("source", "TEXT"), ("doi", "TEXT"),
        ("pmcid", "TEXT"), ("pmid", "TEXT"), ("title", "TEXT"),
        ("licence", "TEXT"), ("pdf", "TEXT"), ("added", "REAL"),
        ("doi_from", "TEXT"), ("run_id", "INTEGER")),
    "figures": (
        ("figure_sha256", "TEXT PRIMARY KEY"), ("paper_key", "TEXT"),
        ("label", "TEXT"), ("caption", "TEXT"), ("legend_source", "TEXT"),
        ("path", "TEXT"), ("width", "INTEGER"), ("height", "INTEGER"),
        ("pixel_sha256", "TEXT"), ("dhash", "TEXT"), ("words_source", "TEXT"),
        ("words", "INTEGER"), ("regions_found", "INTEGER"),
        ("run_id", "INTEGER")),
    "legend_panels": (
        ("figure_sha256", "TEXT"), ("panel", "TEXT"), ("passage", "TEXT"),
        ("legend_source", "TEXT")),
    "figure_annotations": (
        ("figure_sha256", "TEXT"), ("paper_key", "TEXT"),
        ("region_index", "INTEGER"), ("x0", "INTEGER"), ("y0", "INTEGER"),
        ("x1", "INTEGER"), ("y1", "INTEGER"), ("detector_confidence", "REAL"),
        ("found_at_sizes", "TEXT"), ("panel", "TEXT"), ("panel_row", "INTEGER"),
        ("panel_column", "INTEGER"), ("near_text", "TEXT"),
        ("label_text", "TEXT"), ("legend_text", "TEXT"), ("condition", "TEXT"),
        ("condition_source", "TEXT"), ("strength", "TEXT"),
        ("conflict", "INTEGER"), ("conflict_reason", "TEXT"),
        ("approved", "INTEGER"), ("measured", "INTEGER"),
        ("region_id", "INTEGER"), ("scale_source", "TEXT"),
        ("px_per_mm", "REAL"), ("run_id", "INTEGER")),
    "regions": (
        ("region_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("figure_sha256", "TEXT"), ("paper_key", "TEXT"), ("x0", "INTEGER"),
        ("y0", "INTEGER"), ("x1", "INTEGER"), ("y1", "INTEGER"),
        ("detector_confidence", "REAL"), ("found_at_sizes", "TEXT"),
        ("panel", "TEXT"), ("panel_row", "INTEGER"), ("panel_column", "INTEGER"),
        ("near_text", "TEXT"), ("label_text", "TEXT"), ("legend_text", "TEXT"),
        ("condition", "TEXT"), ("condition_source", "TEXT"),
        ("strength", "TEXT"), ("conflict", "INTEGER"), ("approved", "INTEGER"),
        ("crop_path", "TEXT"), ("plaque_count", "INTEGER"),
        ("has_ruler", "INTEGER"), ("px_per_mm", "REAL"), ("detector", "TEXT"),
        ("segmenter", "TEXT"), ("imgsz", "TEXT"), ("conflict_reason", "TEXT"),
        ("scale_source", "TEXT"), ("scale_detail", "TEXT"),
        ("magnification", "TEXT"), ("size_unit", "TEXT"),
        ("words_source", "TEXT"), ("region_index", "INTEGER"),
        ("run_id", "INTEGER")),
    "plaques": (
        ("region_id", "INTEGER"), ("label", "INTEGER"), ("area_px", "INTEGER"),
        ("area_mm2", "REAL"), ("area_vs_panel_median", "REAL"),
        ("cell_probability_mean", "REAL"), ("cell_probability_pixel_fraction", "REAL"),
        ("flow_error", "REAL"), ("flow_magnitude_mean", "REAL"),
        ("flow_alignment_mean", "REAL"), ("flow_pixel_fraction", "REAL")),
    "duplicates": (
        ("figure_sha256", "TEXT"), ("paper_key", "TEXT"), ("path", "TEXT"),
        ("label", "TEXT"), ("match", "TEXT"), ("distance", "INTEGER"),
        ("duplicate_of_sha256", "TEXT"), ("duplicate_of_paper_key", "TEXT"),
        ("duplicate_of_path", "TEXT"), ("measured", "INTEGER"),
        ("run_id", "INTEGER"), ("noted", "REAL")),
}


_CALIBRATION_COLUMNS = (
    ('well_diameter_px', 'REAL'), ('well_diameter_method', 'TEXT'),
    ('pixels_per_um', 'REAL'), ('formation_hours', 'REAL'), ('formation_time_source', 'TEXT'),
    ('estimated_pixels_per_um', 'REAL'), ('estimated_formation_hours', 'REAL'),
    ('estimation_source', 'TEXT'), ('growth_estimate_provenance', 'TEXT'),
)
for _table in ('regions', 'figure_annotations'):
    TABLES[_table] += _CALIBRATION_COLUMNS


def open_database(path: Any) -> sqlite3.Connection:
    """Open (and create, or bring up to date) the results database.

    Every table in :data:`TABLES` is created if missing, and a table an
    older spaCR wrote gets the columns it lacks, so rows already in it stay
    readable next to new ones.

    Waits up to 30 seconds for a lock another process holds, as
    :func:`spacr.database_concurrency.connect` does, rather than failing
    at SQLite's five-second default. Opened with ``sqlite3.connect`` and
    not with that helper because the callers commit once per figure: the
    helper's autocommit mode would make a figure's deletes and inserts
    separate transactions, and a crash between them would leave half a
    figure in the table.

    :param path: the ``.db`` file.
    :returns: an open connection.
    """
    connection = sqlite3.connect(str(path), timeout=30.0)
    for table, columns in TABLES.items():
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            + ", ".join(f"{name} {kind}" for name, kind in columns) + ")")
        have = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns:
            if name not in have:
                kind = kind.replace(" PRIMARY KEY AUTOINCREMENT", "") \
                    .replace(" PRIMARY KEY", "")
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    connection.commit()
    return connection


def _insert(connection: sqlite3.Connection, table: str, row: Mapping[str, Any],
            *, replace_row: bool = False) -> sqlite3.Cursor:
    """Insert one row by column name.

    :param connection: the open database.
    :param table: a table of :data:`TABLES`.
    :param row: ``{column: value}``; columns not named are left NULL.
    :param replace_row: ``INSERT OR REPLACE`` instead of ``INSERT``.
    :returns: the cursor, whose ``lastrowid`` is the new row.
    """
    names = list(row)
    verb = "INSERT OR REPLACE" if replace_row else "INSERT"
    return connection.execute(
        f"{verb} INTO {table} ({', '.join(names)}) VALUES "
        f"({', '.join('?' for _ in names)})", [row[n] for n in names])


def _start_run(connection: sqlite3.Connection, **fields: Any) -> int:
    """Record one run of the pipeline and return its id.

    :param connection: the open database.
    :param fields: the run's settings, columns of the ``runs`` table.
    :returns: the ``run_id`` every row this run writes carries.
    """
    try:
        from importlib.metadata import version

        spacr_version = version("spacr")
    except Exception:
        spacr_version = ""
    fields.setdefault("spacr_version", spacr_version)
    fields.setdefault("started", time.time())
    return int(_insert(connection, "runs", fields).lastrowid)


def _finish_run(connection: sqlite3.Connection, run_id: int,
                summary: Mapping[str, Any]) -> None:
    """Stamp a run finished, with its summary.

    :param connection: the open database.
    :param run_id: the run.
    :param summary: what the run measured.
    """
    connection.execute("UPDATE runs SET finished=?, summary=? WHERE run_id=?",
                       (time.time(), json.dumps(dict(summary)), run_id))
    connection.commit()


def _store_paper(connection: sqlite3.Connection, paper: Paper, run_id: int) -> None:
    """Write (or refresh) a paper's row.

    :param connection: the open database.
    :param paper: the paper.
    :param run_id: the run writing it.
    """
    _insert(connection, "papers", {
        "paper_key": paper.key, "source": paper.source, "doi": paper.doi,
        "pmcid": paper.pmcid, "pmid": paper.pmid, "title": paper.title,
        "licence": paper.licence, "pdf": paper.pdf, "added": time.time(),
        "doi_from": paper.doi_from, "run_id": run_id}, replace_row=True)


def _image_fingerprint(image: np.ndarray) -> Tuple[str, str]:
    """Two hashes of a figure's pixels, for finding it again.

    :param image: the figure, ``H x W x 3`` uint8.
    :returns: ``(pixel_sha256, dhash)``: the sha256 of the decoded pixels
        and their shape -- the same picture in another file format or with
        other metadata has the same one -- and a 64-bit difference hash of
        a 9 x 8 greyscale thumbnail, which a resized or re-encoded copy
        shares to within a few bits.
    """
    from PIL import Image

    array = np.ascontiguousarray(image)
    pixel = hashlib.sha256(repr(array.shape).encode() + array.tobytes()).hexdigest()
    small = np.asarray(Image.fromarray(array).convert("L").resize(
        (9, 8), Image.BILINEAR), dtype=float)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return pixel, f"{value:016x}"


def _hamming(a: str, b: str) -> int:
    """Bits that differ between two hex difference hashes.

    :param a: one hash.
    :param b: the other.
    :returns: the count; 64 when either is missing.
    """
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (TypeError, ValueError):
        return 64


#: Difference hashes this close are the same picture resized or re-encoded
#: -- or a different figure that looks alike, which is why such a match is
#: recorded and not acted on.
SIMILAR_BITS = 4


def _find_duplicate(connection: sqlite3.Connection, paper: Paper, figure: Figure,
                   pixel_sha256: str, dhash: str, *, width: int = 0,
                   height: int = 0) -> Optional[Dict[str, Any]]:
    """Whether this figure was already measured, and as what.

    Identity is the paper (the DOI when there is one) plus the image hash.
    The same file of the same paper is the same figure measured again, not a
    duplicate.

    :param connection: the open database.
    :param paper: the paper the figure came with.
    :param figure: the figure.
    :param pixel_sha256: from :func:`_image_fingerprint`.
    :param dhash: from :func:`_image_fingerprint`.
    :param width: the figure's width, to compare shapes for a similar match.
    :param height: its height.
    :returns: None, or ``{'match': 'bytes'|'pixels'|'similar',
        'distance', 'sha256', 'paper_key', 'path', 'skip'}`` -- ``skip`` is
        True when the figure must not be measured again.
    """
    rows = connection.execute(
        "SELECT figure_sha256, paper_key, path, pixel_sha256, dhash, width, "
        "height FROM figures").fetchall()
    name = Path(str(figure.path)).name

    def same_figure(row) -> bool:
        """Return whether the row is this file of this paper."""
        return row[1] == paper.key and Path(str(row[2])).name == name

    def found(row, match, distance=0, skip=True) -> Dict[str, Any]:
        """The duplicate record for ``row``."""
        return {"match": match, "distance": distance, "sha256": row[0],
                "paper_key": row[1], "path": row[2], "skip": skip}
    for row in rows:
        if row[0] == figure.sha256 and not same_figure(row):
            return found(row, "bytes")
    for row in rows:
        if pixel_sha256 and row[3] == pixel_sha256 and not same_figure(row):
            return found(row, "pixels")
    best = None
    for row in rows:
        if same_figure(row) or not row[4] or not width or not height:
            continue
        if not row[5] or not row[6]:
            continue
        if abs(row[5] / row[6] - width / height) > 0.03 * (width / height):
            continue
        distance = _hamming(dhash, row[4])
        if distance <= SIMILAR_BITS and (best is None or distance < best[0]):
            best = (distance, row)
    if best is not None:
        return found(best[1], "similar", best[0], skip=False)
    return None


def _record_duplicate(connection: sqlite3.Connection, paper: Paper,
                      figure: Figure, duplicate: Mapping[str, Any],
                      run_id: int) -> None:
    """Write one row of the ``duplicates`` table, replacing an older one.

    :param connection: the open database.
    :param paper: the paper the duplicate came with.
    :param figure: the duplicate figure.
    :param duplicate: from :func:`_find_duplicate`.
    :param run_id: the run.
    """
    connection.execute("DELETE FROM duplicates WHERE paper_key=? AND path=?",
                       (paper.key, str(figure.path)))
    _insert(connection, "duplicates", {
        "figure_sha256": figure.sha256, "paper_key": paper.key,
        "path": str(figure.path), "label": figure.label,
        "match": duplicate["match"], "distance": int(duplicate["distance"]),
        "duplicate_of_sha256": duplicate["sha256"],
        "duplicate_of_paper_key": duplicate["paper_key"],
        "duplicate_of_path": duplicate["path"],
        "measured": int(not duplicate["skip"]), "run_id": run_id,
        "noted": time.time()})


def _load_image(path: Path) -> np.ndarray:
    """A figure as an ``H x W x 3`` uint8 RGB array.

    A Bio-Rad Image Lab ``.scn`` is read by :func:`spacr.convert.read_scn`
    and rendered linearly to its scanner ceiling
    (:func:`spacr.convert.scn_to_rgb8`).

    :param path: the image file.
    :returns: the pixels.
    """
    if Path(path).suffix.lower() == ".scn":
        from .convert import read_scn, scn_to_rgb8

        return scn_to_rgb8(*read_scn(path))
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _image_pixels_per_um(path: Any) -> Optional[float]:
    """The pixel scale an image file records about itself, or None.

    Only a Bio-Rad Image Lab ``.scn`` with a known physical size records one.

    :param path: the image file.
    :returns: pixels per micrometre.
    """
    if Path(str(path)).suffix.lower() != ".scn":
        return None
    from .convert import read_scn

    try:
        return read_scn(path)[1].get("pixels_per_um")
    except (OSError, ValueError, IndexError):
        return None


def _zoo_path(key: str, cache: Path, *, kind=None, setting=None) -> Tuple[str, str]:
    """A zoo model's local path and a description to record with results.

    :param key: a zoo key, or a path to a checkpoint.
    :param cache: legacy destination argument; the shared verified cache is used.
    :param kind: checkpoint kind required by the actual consumer, when known.
    :param setting: actual caller input name; direct helper calls use ``key``.
    :returns: ``(path, 'name sha256')``.
    """
    if Path(str(key)).is_file():
        return str(key), f"{Path(str(key)).name}"
    from . import model_zoo as zoo

    field = setting if setting is not None else "key"
    resolved = zoo._ensure_model_file(str(key), kinds=(kind,) if kind else (), setting=field)
    if resolved is None:
        raise zoo.ModelUnreadable(zoo._model_resolution_context(
            str(key), setting=field, kinds=(kind,) if kind else ()))
    return str(resolved), f"{resolved.name} {zoo.sha256_file(resolved)}"


def _cellpose_segmenter(path: str, *, return_metrics: bool = False,
                        min_size: Optional[float] = None,
                        well_resize: Optional[Dict[str, Any]] = None,
                        settings: Optional[Dict[str, Any]] = None) -> Callable:
    """A plaque segmenter from a Cellpose checkpoint.

    A Cellpose 3 checkpoint, which Cellpose 4 refuses, segments through the
    Cellpose 3 backend when it is installed, and is refused with advice when
    it is not.

    :param path: the checkpoint.
    :param return_metrics: Include raw probability and flow diagnostics.
    :param min_size: optional unrounded minimum object area in input pixels;
        None leaves the backend's default unchanged.
    :param well_resize: a plan from :func:`spacr.plaque._well_resize_plan`;
        each crop is then resized whole to the size the model was trained at
        (:func:`spacr.plaque._segment_resized_wells`) and its plaques mapped
        back to the crop's pixels.
    :returns: ``fn(crop) -> labels`` or, with metrics, ``(labels, metrics)``.
    """
    if not Path(path).is_file():
        from .submodules import ModelZooMissing

        raise ModelZooMissing(str(path))
    from cellpose import models
    from .plaque import _segment_resized_wells

    try:
        from .accelerator import cellpose_kwargs

        kwargs = cellpose_kwargs()
    except Exception:
        kwargs = {"gpu": False}
    kwargs.pop("device", None)
    from .model_zoo import _model_precision_kwargs

    kwargs = _model_precision_kwargs(path, kwargs)
    try:
        model = models.CellposeModel(pretrained_model=path, device=None,
                                     **kwargs)
    except ValueError as exc:
        from .submodules import (Cellpose3Checkpoint,
                                 _cellpose3_plaque_backend, explain_cellpose3)

        explained = explain_cellpose3(exc, path)
        if not isinstance(explained, Cellpose3Checkpoint):
            raise
        model = _cellpose3_plaque_backend(path)
        if model is None:
            raise explained from exc

    size_options = {} if min_size is None else {"min_size": float(min_size)}

    def segment(crop: np.ndarray) -> Any:
        """The Cellpose label mask of one plaque image crop."""
        from .plaque import _plaque_segmentation_metrics, _host_array
        if settings is not None:
            from .plaque import segment_plaque_image

            selected = dict(settings)
            if min_size is not None:
                selected['minimum_object_size'] = min_size
            return segment_plaque_image(model, crop, selected, well_resize=well_resize,
                                        whole_frame=True, return_metrics=return_metrics)
        if well_resize:
            declared = dict(well_resize.get("thresholds") or {})
            least = declared.pop("min_size", min_size)
            result = _segment_resized_wells(
                model, np.asarray(crop), well_resize, eval_options=declared,
                min_size=least,
                whole_frame=True, want_metrics=return_metrics)
            return ((result["labels"], result["metrics"]) if return_metrics
                    else result["labels"])
        output = model.eval(crop, **size_options)
        masks = _host_array(output[0])
        return (masks, _plaque_segmentation_metrics(masks, output)) if return_metrics else masks
    return segment


def _model_well_resize(segmenter: Any) -> Optional[Dict[str, Any]]:
    """The well-resize plan a segmenter's own zoo record asks for, or ``None``.

    A Figure crop is already one well, so the plan carries no detector.

    :param segmenter: a zoo key or checkpoint path.
    :returns: :func:`spacr.plaque._well_resize_plan` for that model in
        ``'auto'``.
    """
    from .plaque import _well_resize_plan

    return _well_resize_plan({"plaque_model": str(segmenter)}, detect=False)


def _new_summary(database: Path, run_id: int) -> Dict[str, Any]:
    """The counters every entry point reports.

    :param database: the database path.
    :param run_id: the run.
    :returns: the summary, zeroed.
    """
    return {"papers": 0, "figures": 0, "skipped_figures": 0, "duplicates": 0,
            "possible_duplicates": 0, "regions": 0, "plaques": 0,
            "conflicts": 0, "with_ruler": 0, "text_layer_figures": 0,
            "database": str(database), "run_id": run_id}


def _text_options_json(options: Optional["TextOptions"]) -> str:
    """The text reading's settings as JSON, for the run's record.

    :param options: the options, or None for the defaults.
    :returns: the JSON.
    """
    return json.dumps(asdict(options or DEFAULT_TEXT_OPTIONS))


def measure_plaques_from_papers(
        references: Iterable[Any], dst: Any, *,
        detector: str = DEFAULT_DETECTOR, segmenter: str = DEFAULT_SEGMENTER,
        imgsz: Sequence[int] = DEFAULT_IMGSZ, confidence: float = 0.25,
        confirm_each: bool = False, plate_format: Optional[str] = None,
        ask_legend: Optional[Callable] = None, review: Optional[Callable] = None,
        read_text: Optional[Callable] = None, detect: Optional[Callable] = None,
        segment: Optional[Callable] = None, get: Optional[Callable] = None,
        pdf_opener: Optional[Callable] = None) -> Dict[str, Any]:
    """Measure the plaques in every figure of every paper given.

    A figure already measured -- the same image in another paper, or the
    same paper from another source -- is recorded in the ``duplicates``
    table and not measured again (see the module docstring).

    :param references: DOIs, PMC ids, PMIDs or PDF paths.
    :param dst: output folder: ``plaque_papers.db``, ``figures/`` and
        ``crops/`` are written under it.
    :param detector: zoo key or checkpoint for the plaque-image detector.
    :param segmenter: zoo key or checkpoint for the plaque segmenter.
    :param imgsz: detector inference sizes, all asked, results merged.
    :param confidence: minimum detector score.
    :param confirm_each: when True, every proposed condition is shown to a
        person (``review``) before it is stored as approved.
    :param plate_format: plate format for whole-well images, which gives
        them a ruler; None lets the legend name one, and otherwise keeps
        every area in pixels and ratios.
    :param ask_legend: ``fn(figure, annotations) -> legend or None``, called
        when a panel letter was read but no legend could be fetched.
        Defaults to :func:`console_legend_prompt`.
    :param review: ``fn(figure, annotations) -> annotations``, called per
        figure when ``confirm_each``. Defaults to :func:`console_review`.
    :param read_text: ``fn(image path) -> [Word]``; defaults to
        :func:`read_words`. Not called for a figure whose PDF text layer
        already has the words around its plaque images.
    :param detect: passed to :func:`find_plaque_regions`.
    :param segment: ``fn(crop) -> labels``; defaults to a Cellpose model
        loaded from ``segmenter``.
    :param get: HTTP getter for Europe PMC.
    :param pdf_opener: passed to :func:`figures_from_pdf`.
    :returns: a summary: papers, figures, figures skipped as already
        measured, duplicates (not measured) and possible duplicates
        (measured), regions, plaques, conflicts, regions with a ruler,
        figures read from a text layer, the run id and the database path.
    """
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    database = dst / "plaque_papers.db"
    connection = open_database(database)
    ask_legend = ask_legend or console_legend_prompt
    review = review or console_review
    read_text = read_text or read_words
    detector_path, detector_id = (detector, str(detector)) if detect else \
        _zoo_path(detector, dst / "models", kind="detector", setting="detector")
    segmenter_id = str(segmenter)
    if segment is None:
        segmenter_path, segmenter_id = _zoo_path(segmenter, dst / "models", kind="cellpose", setting="segmenter")
        plan = _model_well_resize(segmenter)
        segment = _cellpose_segmenter(segmenter_path, return_metrics=True,
                                      **({} if plan is None else {"well_resize": plan}))
    references = list(references)
    run_id = _start_run(
        connection, entry="papers", source=json.dumps([str(r) for r in references]),
        detector=detector_id, segmenter=segmenter_id,
        imgsz=json.dumps(list(imgsz)), confidence=float(confidence),
        plate_format=plate_format, confirm_each=int(bool(confirm_each)),
        text_options=_text_options_json(None))
    summary = _new_summary(database, run_id)
    for reference in references:
        paper = resolve_paper(reference, get=get)
        folder = re.sub(r"[^A-Za-z0-9._-]+", "_", paper.key)[:120]
        if paper.source == "pdf":
            figures = figures_from_pdf(paper.pdf, dst / "figures" / folder,
                                       opener=pdf_opener)
        else:
            figures = fetch_figures(paper, dst / "figures" / folder, get=get)
        summary["papers"] += 1
        prior = connection.execute("SELECT source FROM papers WHERE paper_key=?",
                                   (paper.key,)).fetchone()
        measured_before = connection.execute(
            "SELECT 1 FROM figures WHERE paper_key=?", (paper.key,)).fetchone()
        if prior and prior[0] != paper.source and measured_before:
            for figure in figures:
                _record_duplicate(connection, paper, figure, {
                    "match": "paper", "distance": 0, "sha256": None,
                    "paper_key": paper.key, "path": f"measured from {prior[0]}",
                    "skip": True}, run_id)
                summary["duplicates"] += 1
            connection.commit()
            continue
        _store_paper(connection, paper, run_id)
        for figure in figures:
            again = connection.execute(
                "SELECT path FROM figures WHERE figure_sha256=? AND paper_key=?",
                (figure.sha256, paper.key)).fetchone()
            if again and Path(str(again[0])).name == figure.path.name:
                summary["skipped_figures"] += 1
                continue
            _measure_figure(connection, paper, figure, dst / "crops" / folder,
                            detector_path=detector_path, detector_id=detector_id,
                            segmenter_id=segmenter_id, imgsz=imgsz,
                            confidence=confidence, confirm_each=confirm_each,
                            plate_format=plate_format, ask_legend=ask_legend,
                            review=review, read_text=read_text, detect=detect,
                            segment=segment, summary=summary, run_id=run_id)
            connection.commit()
    _finish_run(connection, run_id, summary)
    connection.close()
    return summary


def _text_lines(words: Sequence[Word]) -> List[List[Word]]:
    """Words grouped into printed lines, left to right.

    :param words: the words.
    :returns: the lines; two words share one when their centres are within
        half a letter height vertically and the gap between them is under
        1.5 letter heights. A letter height is a box's shorter side, so a
        label printed rotated, whose box is tall, does not reach across the
        figure and gather a line of other words.
    """
    lines: List[List[Word]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        height = max(1.0, min(w.y1 - w.y0, w.x1 - w.x0))
        for line in lines:
            last = line[-1]
            if abs(last.cy - w.cy) <= 0.5 * height and \
                    -0.5 * height <= w.x0 - last.x1 <= 1.5 * height:
                line.append(w)
                break
        else:
            lines.append([w])
    return lines


def _label_words(layer: Sequence[Word], *, max_words: int = 6) -> List[Word]:
    """The words of a text layer that can be a figure's labels.

    A PDF page's text layer holds the article as well as the figure: the
    legend paragraph under the figure sits right next to its bottom row of
    images, and taken as labels it names every crop "4. The cytosolic"
    (PLOS Pathogens, PMC9744290, 2026-09-21). Labels are short lines on
    their own, so a line of more than ``max_words`` words is running text,
    and so is a short line directly below one -- the last line of a
    paragraph. A short line directly ABOVE a paragraph is kept: that is
    where a figure's bottom row label sits, one line over its legend. Words are counted inside each piece of text too, as
    OCR returns a whole phrase as one piece.

    :param layer: the text layer's words.
    :param max_words: the most words a label line has.
    :returns: the words that may be labels.
    """
    def size(line: List[Word]) -> int:
        """Words in a line, counting inside each piece of text."""
        return sum(max(1, len(w.text.split())) for w in line)
    lines = _text_lines(layer)
    long = [line for line in lines if size(line) > max_words]
    out: List[Word] = []
    for line in lines:
        if size(line) > max_words:
            continue
        x0, x1 = line[0].x0, line[-1].x1
        y0, y1 = min(w.y0 for w in line), max(w.y1 for w in line)
        height = max(1.0, min(min(w.y1 - w.y0, w.x1 - w.x0) for w in line))
        near_paragraph = any(
            _overlap(x0, x1, other[0].x0, other[-1].x1) > 0
            and -0.5 * height <= y0 - max(w.y1 for w in other)
            <= 1.0 * min(height, max(1.0, max(w.y1 for w in other)
                                     - min(w.y0 for w in other)))
            for other in long)
        if not near_paragraph:
            out.extend(line)
    return out


def _layer_reaches(regions: Sequence[Region], words: Sequence[Word],
                   options: "TextOptions") -> bool:
    """Whether a text layer has anything to say about these plaque images.

    :param regions: the figure's plaque images.
    :param words: the text layer's words.
    :param options: how the text is read.
    :returns: True when some image gets a panel letter or a label from it.
    """
    for region in regions:
        near = text_near(region, words, regions=regions, options=options)
        if near["panel"] or near["above"] or near["left"] or near["below"]:
            return True
    return False


def _figure_words(image: np.ndarray, regions: Sequence[Region],
                 layer: Sequence[Word], *, path: Any = None,
                 read_text: Optional[Callable] = None,
                 options: Optional["TextOptions"] = None,
                 default_reader: bool = False) -> Tuple[List[Word], str]:
    """The words to read a figure's conditions from, and where they came from.

    A PDF's own text layer comes first: it is exact, and reading it costs
    nothing. Only its label-like words are used (:func:`_label_words`), so
    the legend paragraph printed under a figure is not read as the labels
    of the images above it. OCR is asked only when those words say nothing
    about any plaque image -- a figure pasted into the PDF as a single
    picture has a text layer for the article around it and none for its own
    labels -- and what OCR finds is added to them, not put in their place.

    :param image: the figure.
    :param regions: its plaque images.
    :param layer: the text layer's words; empty when there is none.
    :param path: the figure's file, for ``read_text``.
    :param read_text: ``fn(path) -> [Word]``; None means text is not read
        beyond the layer.
    :param options: how the text is read (:class:`TextOptions`).
    :param default_reader: ``read_text`` is RapidOCR, so the enlarged second
        reading (:func:`reread_around`) is added when the options ask.
    :returns: ``(words, source)``, source one of ``'pdf text layer'``,
        ``'pdf text layer + ocr'``, ``'ocr'`` or ``'none'``.
    """
    options = options or DEFAULT_TEXT_OPTIONS
    had_layer = bool(layer)
    layer = _label_words(layer or [])
    if layer and (not regions or _layer_reaches(regions, layer, options)):
        return layer, "pdf text layer"
    if not regions or read_text is None:
        return layer, "pdf text layer" if had_layer else "none"
    ocr = list(read_text(path))
    if default_reader and options.reread:
        ocr = reread_around(image, regions, ocr, scale=int(options.reread_scale))
    if not had_layer:
        return ocr, "ocr"
    ocr = _label_words(ocr)
    if not ocr:
        return layer, "pdf text layer"
    merged = list(layer) + [
        w for w in ocr if not any(_overlap(w.x0, w.x1, o.x0, o.x1) > 0
                                  and _overlap(w.y0, w.y1, o.y0, o.y1) > 0
                                  for o in layer)]
    return merged, "pdf text layer + ocr"


def _annotation_row(figure: Figure, paper: Paper, index: int, a: Annotation,
                    scale: _Scale, *, measured: bool, region_id: Optional[int],
                    run_id: Optional[int], formation_hours=None) -> Dict[str, Any]:
    """One row of the ``figure_annotations`` table.

    :param figure: the figure.
    :param paper: its paper.
    :param index: the image's 1-based place in reading order.
    :param a: its annotation.
    :param scale: its ruler.
    :param measured: whether its plaques were measured.
    :param region_id: its ``regions`` row when measured.
    :param run_id: the run.
    :returns: the row.
    """
    r = a.region
    return {"figure_sha256": figure.sha256, "paper_key": paper.key,
            "region_index": index, "x0": r.x0, "y0": r.y0, "x1": r.x1,
            "y1": r.y1, "detector_confidence": r.confidence,
            "found_at_sizes": json.dumps(list(r.sizes)), "panel": a.panel,
            "panel_row": a.row, "panel_column": a.column,
            "near_text": json.dumps(a.near), "label_text": a.label_text,
            "legend_text": a.legend_text, "condition": a.condition,
            "condition_source": a.source, "strength": a.strength,
            "conflict": int(a.conflict), "conflict_reason": _conflict_reason(a),
            "approved": None if a.approved is None else int(a.approved),
            "measured": int(measured), "region_id": region_id,
            "scale_source": scale.source, "px_per_mm": scale.px_per_mm,
            **calibration_values(a, scale, formation_hours), "run_id": run_id}


def _measure_figure(connection: sqlite3.Connection, paper: Paper,
                    figure: Figure, crops: Path, **kw: Any) -> None:
    """Detect, annotate, review, segment and store one figure.

    :param connection: the open database.
    :param paper: the paper the figure belongs to.
    :param figure: the figure.
    :param crops: where this paper's crops are written.
    :param kw: the options :func:`measure_plaques_from_papers` resolved.
    """
    image = _load_image(figure.path)
    summary = kw["summary"]
    run_id = kw.get("run_id")
    summary["figures"] += 1
    height, width = int(image.shape[0]), int(image.shape[1])
    pixel_sha, dhash = _image_fingerprint(image)
    duplicate = _find_duplicate(connection, paper, figure, pixel_sha, dhash,
                               width=width, height=height)
    if duplicate is not None:
        _record_duplicate(connection, paper, figure, duplicate, run_id)
        if duplicate["skip"]:
            summary["duplicates"] += 1
            return
        summary["possible_duplicates"] += 1
    else:
        connection.execute("DELETE FROM duplicates WHERE paper_key=? AND path=?",
                           (paper.key, str(figure.path)))
    connection.execute("DELETE FROM plaques WHERE region_id IN (SELECT region_id "
                       "FROM regions WHERE figure_sha256=? AND paper_key=?)",
                       (figure.sha256, paper.key))
    connection.execute("DELETE FROM regions WHERE figure_sha256=? AND paper_key=?",
                       (figure.sha256, paper.key))
    regions = find_plaque_regions(image, kw["detector_path"], imgsz=kw["imgsz"],
                                  confidence=kw["confidence"], detect=kw["detect"])
    options = kw.get("text_options") or DEFAULT_TEXT_OPTIONS
    words, words_source = _figure_words(
        image, regions, figure.words, path=figure.path,
        read_text=kw["read_text"], options=options,
        default_reader=kw["read_text"] is read_words)
    if words_source.startswith("pdf text layer"):
        summary["text_layer_figures"] += 1
    annotations = annotate_regions(regions, words, caption=figure.caption,
                                   figure_label=figure.label, options=options) \
        if regions else []
    if any(a.panel for a in annotations) and not figure.caption:
        pasted = kw["ask_legend"](figure, annotations)
        if pasted:
            figure.caption, figure.legend_source = pasted, "pasted"
            annotations = annotate_regions(regions, words, caption=pasted,
                                           figure_label=figure.label,
                                           options=options)
        elif not kw["confirm_each"]:
            annotations = kw["review"](figure, annotations)
    if annotations and kw["confirm_each"]:
        annotations = kw["review"](figure, annotations)
    _insert(connection, "figures", {
        "figure_sha256": figure.sha256, "paper_key": paper.key,
        "label": figure.label, "caption": figure.caption,
        "legend_source": figure.legend_source, "path": str(figure.path),
        "width": width, "height": height, "pixel_sha256": pixel_sha,
        "dhash": dhash, "words_source": words_source, "words": len(words),
        "regions_found": len(regions), "run_id": run_id}, replace_row=True)
    connection.execute("DELETE FROM legend_panels WHERE figure_sha256=?",
                       (figure.sha256,))
    for panel, passage in split_legend(figure.caption).items():
        _insert(connection, "legend_panels", {
            "figure_sha256": figure.sha256, "panel": panel, "passage": passage,
            "legend_source": figure.legend_source})
    connection.execute(
        "DELETE FROM figure_annotations WHERE figure_sha256=? AND paper_key=?",
        (figure.sha256, paper.key))
    if not regions:
        return
    scales = _scales_for_regions(image, regions, words, caption=figure.caption,
                                annotations=annotations,
                                plate_format=kw["plate_format"],
                                pixels_per_um=kw.get("pixels_per_um"),
                                image_pixels_per_um=_image_pixels_per_um(figure.path))
    crops.mkdir(parents=True, exist_ok=True)
    measured: List[Tuple[int, Annotation, List[Dict[str, Any]], str, _Scale]] = []
    for index, a in enumerate(annotations, start=1):
        scale = scales[index - 1]
        if a.approved is False:
            _insert(connection, "figure_annotations", _annotation_row(
                figure, paper, index, a, scale, measured=False,
                region_id=None, run_id=run_id, formation_hours=kw.get("formation_hours")))
            continue
        r = a.region
        crop = image[r.y0:r.y1, r.x0:r.x1]
        crop_path = crops / f"{figure.path.stem}_r{index:02d}.png"
        _save_png(crop, crop_path)
        rows = measure_region(kw["segment"](crop), px_per_mm=scale.px_per_mm)
        measured.append((index, a, rows, str(crop_path), scale))
    medians: Dict[Optional[str], float] = {}
    for panel in {a.panel for _i, a, *_ in measured}:
        areas = [row["area_px"] for _i, a, rows, *_ in measured if a.panel == panel
                 for row in rows]
        medians[panel] = float(np.median(areas)) if areas else 0.0
    from .plaque_growth import estimates_from_settings
    growth = estimates_from_settings([
        dict(well=i, areas_px=[r['area_px'] for r in rows],
             **calibration_values(a, scale, kw.get('formation_hours')))
        for i, a, rows, _path, scale in measured], kw.get('growth_settings') or {})
    for index, a, rows, crop_path, scale in measured:
        r = a.region
        cursor = _insert(connection, "regions", {
            "figure_sha256": figure.sha256, "paper_key": paper.key,
            "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
            "detector_confidence": r.confidence,
            "found_at_sizes": json.dumps(list(r.sizes)), "panel": a.panel,
            "panel_row": a.row, "panel_column": a.column,
            "near_text": json.dumps(a.near), "label_text": a.label_text,
            "legend_text": a.legend_text, "condition": a.condition,
            "condition_source": a.source, "strength": a.strength,
            "conflict": int(a.conflict), "conflict_reason": _conflict_reason(a),
            "approved": None if a.approved is None else int(a.approved),
            "crop_path": crop_path, "plaque_count": len(rows),
            "has_ruler": int(scale.px_per_mm is not None),
            "px_per_mm": scale.px_per_mm, "scale_source": scale.source,
            "scale_detail": scale.detail, "magnification": scale.magnification,
            "size_unit": scale.unit, "words_source": words_source,
            "region_index": index, "detector": kw["detector_id"],
            "segmenter": kw["segmenter_id"], "imgsz": json.dumps(list(kw["imgsz"])),
            **calibration_values(a, scale, kw.get("formation_hours")),
            **growth.get(index, {}), "run_id": run_id})
        region_id = int(cursor.lastrowid)
        _insert(connection, "figure_annotations", _annotation_row(
            figure, paper, index, a, scale, measured=True, region_id=region_id,
            run_id=run_id, formation_hours=kw.get("formation_hours")))
        if index in growth:
            values = growth[index]
            connection.execute(
                "UPDATE figure_annotations SET estimated_pixels_per_um=?, estimated_formation_hours=?, "
                "estimation_source=?, growth_estimate_provenance=? WHERE region_id=?",
                (values['estimated_pixels_per_um'], values['estimated_formation_hours'],
                 values['estimation_source'], values['growth_estimate_provenance'], region_id))
        median = medians.get(a.panel) or 0.0
        for row in rows:
            _insert(connection, "plaques", {
                "region_id": region_id, **row,
                "area_vs_panel_median": row["area_px"] / median if median else None})
        summary["regions"] += 1
        summary["plaques"] += len(rows)
        summary["conflicts"] += int(a.conflict)
        summary["with_ruler"] += int(scale.px_per_mm is not None)


def _save_png(array: np.ndarray, path: Path) -> None:
    """Write an image crop.

    :param array: the pixels.
    :param path: the ``.png`` path.
    """
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(array)).save(path)


LEGENDS_FILE = "legends.csv"
ANNOTATIONS_FILE = "figure_annotations.csv"
TEXT_LAYER_FILE = "text_layer.json"
PAPER_FILE = "paper.json"

#: The columns of ``figure_annotations.csv``. ``condition`` and ``approved``
#: are what a person sets and what is read back; the rest record what was
#: proposed and why, the conflict flag among them.
ANNOTATION_COLUMNS = ("file", "region", "condition", "approved", "panel",
                      "label_text", "legend_text", "source", "strength",
                      "conflict", "conflict_reason", "well_diameter_px",
                      "well_diameter_method", "pixels_per_um", "formation_hours",
                      "resolved_pixels_per_um", "resolved_formation_hours",
                      "scale_source", "formation_time_source", "estimated_pixels_per_um",
                      "estimated_formation_hours", "estimation_source", "growth_estimate_provenance")


def read_legends(path: Any) -> Dict[str, str]:
    """Figure legends supplied beside a folder of figures.

    :param path: a CSV with ``file`` and ``legend`` columns; ``file`` is the
        figure's file name or stem.
    :returns: ``{stem: legend}``; empty when the file does not exist.
    """
    import csv

    path = Path(path)
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("file") or "").strip()
            if name:
                out[Path(name).stem] = (row.get("legend") or "").strip()
    return out


def _annotation_rows(path: Any) -> Dict[Tuple[str, int], Dict[str, str]]:
    """Every row of ``figure_annotations.csv``, all columns, by figure and image.

    :param path: the CSV.
    :returns: ``{(stem, region): row}``; empty when the file does not exist.
    """
    import csv

    path = Path(path)
    if not path.is_file():
        return {}
    out: Dict[Tuple[str, int], Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (Path(row["file"]).stem, int(row["region"]))
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = {k: (v or "") for k, v in row.items() if k}
    return out


def read_annotation_overrides(path: Any) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Conditions a person edited or approved, keyed by figure and image.

    :param path: a CSV with ``file``, ``region`` (1-based, in reading order),
        ``condition`` and ``approved`` columns -- what the Figure preview
        saves. Its other columns are the record of what was proposed and are
        not read back.
    :returns: ``{(stem, region): {'condition': str, 'approved': bool}}``.
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, row in _annotation_rows(path).items():
        approved = str(row.get("approved", "")).strip().lower()
        out[key] = {"condition": (row.get("condition") or "").strip(),
                    "approved": approved in ("1", "true", "yes", "ok")}
        for name in ('pixels_per_um', 'formation_hours'):
            value = calibration_number(row.get(name), name=name, allow_zero=name == 'formation_hours')
            if value is not None:
                out[key][name] = value
    return out


def write_annotation_overrides(path: Any, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Save reviewed conditions for :func:`read_annotation_overrides`.

    Rows for figures not in ``rows`` are kept, so reviewing one figure does
    not erase another's approvals. Every column of
    :data:`ANNOTATION_COLUMNS` is written; a row that does not give one
    leaves it blank.

    :param path: the CSV to write.
    :param rows: mappings with ``file``, ``region``, ``condition``,
        ``approved`` and optionally the other annotation columns.
    :returns: the path written.
    """
    import csv

    path = Path(path)
    kept = _annotation_rows(path)
    for row in rows:
        key = (Path(str(row["file"])).stem, int(row["region"]))
        record = {k: "" if row.get(k) is None else str(row.get(k))
                  for k in ANNOTATION_COLUMNS}
        record["file"], record["region"] = key[0], str(key[1])
        record["approved"] = "true" if row.get("approved") else "false"
        record["conflict"] = "true" if row.get("conflict") in (True, "true", "1", 1) \
            else ("false" if "conflict" in row else "")
        kept[key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANNOTATION_COLUMNS)
        for (stem, region), value in sorted(kept.items()):
            cells = [stem, region] + [value.get(k, "") for k in ANNOTATION_COLUMNS[2:]]
            writer.writerow(cells)
    return path


def _annotation_file_row(file: str, region: int, a: Annotation, *,
                        condition: Optional[str] = None,
                        approved: Optional[bool] = None) -> Dict[str, Any]:
    """One ``figure_annotations.csv`` row for an annotation.

    :param file: the figure's file name.
    :param region: the image's 1-based place in reading order.
    :param a: its annotation.
    :param condition: the condition as a person left it; ``a.condition``
        when None.
    :param approved: whether a person OK'd it; ``a.approved`` when None.
    :returns: the row, for :func:`write_annotation_overrides`.
    """
    return {"file": file, "region": int(region),
            "condition": a.condition if condition is None else condition,
            "approved": bool(a.approved if approved is None else approved),
            "panel": a.panel or "", "label_text": a.label_text,
            "legend_text": a.legend_text, "source": a.source,
            "strength": a.strength, "conflict": bool(a.conflict),
            "conflict_reason": _conflict_reason(a),
            "well_diameter_px": (a.region.width + a.region.height) / 2.0,
            "well_diameter_method": 'mean detected bounding-box extent',
            "pixels_per_um": a.pixels_per_um, "formation_hours": a.formation_hours}


def apply_overrides(stem: str, annotations: List[Annotation],
                    overrides: Mapping[Tuple[str, int], Mapping[str, Any]], *,
                    confirm_each: bool = False) -> List[Annotation]:
    """Put a person's edits and approvals onto one figure's proposals.

    A conflict flag stays on an edited image: the record keeps that the two
    readings disagreed, and the edit and OK record how a person settled it.

    :param stem: the figure's file stem.
    :param annotations: the proposals, in reading order.
    :param overrides: from :func:`read_annotation_overrides`.
    :param confirm_each: when True, an image nobody approved is marked
        ``approved=False`` and is not measured.
    :returns: the same annotations.
    """
    for index, a in enumerate(annotations, start=1):
        edit = overrides.get((stem, index))
        if edit is not None:
            if edit["condition"] and edit["condition"] != a.condition:
                a.condition, a.source, a.strength = edit["condition"], "manual", "manual"
            a.approved = bool(edit["approved"])
            a.pixels_per_um = calibration_number(edit.get('pixels_per_um'), name='pixels_per_um')
            a.formation_hours = calibration_number(edit.get('formation_hours'), name='formation_hours', allow_zero=True)
        elif confirm_each:
            a.approved = False
    return annotations


def _read_text_layer(path: Any) -> Dict[str, List[Word]]:
    """The PDF text layer saved beside a folder of page images.

    :param path: ``text_layer.json``: ``{file name: [[text, x0, y0, x1,
        y1], ...]}`` in the page image's pixels.
    :returns: ``{stem: [Word]}``; empty when the file does not exist or
        cannot be read.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: Dict[str, List[Word]] = {}
    for name, words in (data or {}).items():
        try:
            out[Path(name).stem] = [Word(str(w[0]), float(w[1]), float(w[2]),
                                         float(w[3]), float(w[4])) for w in words]
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _write_text_layer(path: Any, figures: Iterable[Figure]) -> Optional[Path]:
    """Save the figures' text-layer words for :func:`_read_text_layer`.

    :param path: the JSON file.
    :param figures: figures, those with ``words`` saved.
    :returns: the path, or None when no figure had a text layer.
    """
    data = {f.path.name: [[w.text, w.x0, w.y0, w.x1, w.y1] for w in f.words]
            for f in figures if f.words}
    if not data:
        return None
    path = Path(path)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def figures_in_folder(src: Any, legends: Optional[Mapping[str, str]] = None,
                      text_layer: Optional[Mapping[str, List[Word]]] = None
                      ) -> List[Figure]:
    """Every figure image directly in ``src``, with its legend when known.

    :param src: the folder.
    :param legends: ``{stem: legend}``.
    :param text_layer: ``{stem: [Word]}``, a PDF's words for its pages;
        read from ``src/text_layer.json`` when None.
    :returns: the figures, by file name.
    """
    legends = legends or {}
    if text_layer is None:
        text_layer = _read_text_layer(Path(src) / TEXT_LAYER_FILE)
    out = []
    for path in sorted(Path(src).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        caption = legends.get(path.stem, "")
        out.append(Figure(path=path, label=path.stem, caption=caption,
                          legend_source="file" if caption else "none",
                          sha256=sha256_bytes(path.read_bytes()),
                          words=list(text_layer.get(path.stem, []))))
    return out


def _is_paper_folder(path: Path) -> bool:
    """Whether ``path`` is a folder a paper was read into.

    :param path: a folder.
    :returns: True when it holds ``paper.json``, ``legends.csv`` or
        ``text_layer.json``.
    """
    return any((path / marker).is_file()
               for marker in (PAPER_FILE, LEGENDS_FILE, TEXT_LAYER_FILE))


def figure_folders(src: Any) -> List[Path]:
    """The folders Figure mode reads for ``src``, in reading order.

    A paper folder, or a folder of figure images, is read as it is. A folder
    that holds paper folders -- several PDFs read at once, each into a
    folder of its own -- is read paper by paper: its own figure
    images first when it has any, then each paper folder by name.

    :param src: the folder Figure mode was given.
    :returns: the folders; ``[src]`` when it holds no paper folders.
    """
    path = Path(str(src or "")).expanduser()
    if not path.is_dir() or _is_paper_folder(path):
        return [path]
    try:
        children = sorted(path.iterdir())
    except OSError:
        return [path]
    papers = []
    own = False
    for child in children:
        try:
            if child.is_dir():
                if _is_paper_folder(child):
                    papers.append(child)
            elif child.suffix.lower() in IMAGE_SUFFIXES:
                own = True
        except OSError:
            continue
    if not papers:
        return [path]
    return ([path] if own else []) + papers


def _folder_paper(src: Any) -> Paper:
    """The paper a folder of figures came from.

    :param src: the folder.
    :returns: the paper :func:`fetch_paper_to_folder` recorded in
        ``paper.json``, so rows keep the DOI and licence; a paper named after
        the folder when there is none.
    """
    from dataclasses import fields

    src = Path(src)
    record = src / PAPER_FILE
    if record.is_file():
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
            known = {f.name for f in fields(Paper)}
            paper = Paper(**{k: v for k, v in data.items() if k in known})
            if paper.key:
                return paper
        except (OSError, ValueError, TypeError):
            pass
    return Paper(key=f"folder:{src.resolve()}", source="folder", title=src.name)


def measure_figure_folder(
        src: Any, dst: Any = None, *, detector: str = DEFAULT_DETECTOR,
        segmenter: str = DEFAULT_SEGMENTER, imgsz: Sequence[int] = DEFAULT_IMGSZ,
        confidence: float = 0.25, confirm_each: bool = False,
        plate_format: Optional[str] = None, pixels_per_um=None, formation_hours=None, growth_settings=None, legends: Any = None,
        annotations: Any = None, read_text: Optional[Callable] = None,
        detect: Optional[Callable] = None, segment: Optional[Callable] = None,
        text_options: Optional[TextOptions] = None,
        min_size: Optional[float] = None,
        well_resize: Any = "model") -> Dict[str, Any]:
    """Figure mode of Plaque Assay: find, read, annotate and measure a folder.

    The function does not prompt. Legends come from ``legends`` (default
    ``<src>/legends.csv``), a PDF's text layer from ``<src>/text_layer.json``,
    the paper from ``<src>/paper.json``, and a person's edits and approvals
    from ``annotations`` (default ``<src>/figure_annotations.csv``), which the
    Figure preview writes. With ``confirm_each`` on, only approved images are
    measured and the rest are counted as waiting. A figure measured again
    replaces its earlier rows; the same image under a second name is
    measured once and recorded as a duplicate -- the copy with a legend is
    the one measured, since figures with legends are taken first.

    :param src: the folder of figure images.
    :param dst: output folder; default ``<src>/plaque_figures``.
    :param detector: zoo key or checkpoint for the plaque-image detector.
    :param segmenter: zoo key or checkpoint for the plaque segmenter.
    :param imgsz: detector inference sizes.
    :param confidence: minimum detector score.
    :param confirm_each: measure only images a person approved.
    :param plate_format: plate format for whole-well images.
    :param pixels_per_um: optional positive manual pixel scale; per-well
        annotations override this value, then automatic rulers are considered.
    :param formation_hours: optional nonnegative elapsed time in hours;
        per-well manual times take precedence. Stored as metadata, not inferred.
    :param growth_settings: optional experimental growth settings consumed by
        :func:`spacr.plaque_growth.estimates_from_settings`; disabled by default.
    :param legends: CSV of legends, see :func:`read_legends`.
    :param annotations: CSV of reviews, see :func:`read_annotation_overrides`.
    :param read_text: ``fn(path) -> [Word]``.
    :param detect: passed to :func:`find_plaque_regions`.
    :param segment: ``fn(crop) -> labels``.
    :param text_options: how the text is read (:class:`TextOptions`).
    :param well_resize: well-resize plan for the segmenter (see
        :func:`_cellpose_segmenter`); ``None`` segments each crop as it is.
        Ignored for a supplied ``segment``.
    :param min_size: minimum plaque area in square pixels of the input well
        crop, without rounding; 0 disables small-object removal. None leaves
        the segmenter's default unchanged. Ignored for a supplied ``segment``.
    :returns: the summary, with ``awaiting_approval`` added.
    """
    pixels_per_um = calibration_number(pixels_per_um, name='pixels_per_um')
    formation_hours = calibration_number(formation_hours, name='formation_hours', allow_zero=True)
    src = Path(src)
    dst = Path(dst) if dst else src / "plaque_figures"
    dst.mkdir(parents=True, exist_ok=True)
    legend_map = read_legends(legends or src / LEGENDS_FILE)
    overrides = read_annotation_overrides(annotations or src / ANNOTATIONS_FILE)
    waiting = {"n": 0}

    def review(figure: Figure, found: List[Annotation]) -> List[Annotation]:
        """Apply the saved overrides to a figure and count what still awaits approval."""
        stem = figure.path.stem
        out = apply_overrides(stem, found, overrides, confirm_each=confirm_each)
        waiting["n"] += sum(1 for index, a in enumerate(out, start=1)
                            if a.approved is False and (stem, index) not in overrides)
        return out

    database = dst / "plaque_figures.db"
    connection = open_database(database)
    detector_path, detector_id = (detector, str(detector)) if detect else \
        _zoo_path(detector, dst / "models", kind="detector", setting="detector")
    segmenter_id = str(segmenter)
    if segment is None:
        segmenter_path, segmenter_id = _zoo_path(segmenter, dst / "models", kind="cellpose", setting="segmenter")
        if isinstance(well_resize, str) and well_resize == "model":
            well_resize = _model_well_resize(segmenter)
        resize_options = {} if well_resize is None else {"well_resize": well_resize}
        segment = _cellpose_segmenter(segmenter_path, return_metrics=True,
                                      min_size=min_size, **resize_options)
    run_id = _start_run(
        connection, entry="folder", source=str(src.resolve()),
        detector=detector_id, segmenter=segmenter_id,
        imgsz=json.dumps(list(imgsz)), confidence=float(confidence),
        plate_format=plate_format, confirm_each=int(bool(confirm_each)),
        text_options=_text_options_json(text_options))
    paper = _folder_paper(src)
    _store_paper(connection, paper, run_id)
    summary = _new_summary(database, run_id)
    summary["papers"] = 1
    figures = sorted(figures_in_folder(src, legend_map),
                     key=lambda f: (not f.caption, f.path.name))
    for figure in figures:
        _measure_figure(connection, paper, figure, dst / "crops",
                        detector_path=detector_path, detector_id=detector_id,
                        segmenter_id=segmenter_id, imgsz=imgsz,
                        confidence=confidence, confirm_each=True,
                        plate_format=plate_format, pixels_per_um=pixels_per_um, formation_hours=formation_hours,
                        growth_settings=growth_settings,
                        ask_legend=lambda *_a: None, review=review,
                        read_text=read_text or read_words, detect=detect,
                        segment=segment, summary=summary,
                        text_options=text_options, run_id=run_id)
        connection.commit()
    summary["awaiting_approval"] = waiting["n"]
    _finish_run(connection, run_id, summary)
    connection.close()
    return summary


def fetch_paper_to_folder(reference: Any, dest: Any, *,
                          get: Optional[Callable] = None,
                          pdf_opener: Optional[Callable] = None) -> Dict[str, Any]:
    """Put a paper's figures in a folder Plaque Assay's Figure mode can read.

    When a figure's panel letters are detected, its legend is gathered
    automatically. A DOI,
    PMID or PMC id is fetched from Europe PMC with its JATS legends; a PDF is
    rendered page by page with the legends found in its text. Either way the
    images land in ``dest`` and each figure's legend in ``dest/legends.csv``,
    which Figure mode and :func:`measure_figure_folder` read -- so a paper
    becomes an ordinary folder of figures, annotated with its own legends.
    A PDF's text layer is kept in ``dest/text_layer.json`` so its labels are
    read from the PDF itself before any OCR, and the paper's identity in
    ``dest/paper.json``.

    :param reference: a DOI, PMID, PMC id or PDF path.
    :param dest: the folder to fill; created if missing.
    :param get: HTTP getter for Europe PMC.
    :param pdf_opener: passed to :func:`figures_from_pdf`.
    :returns: ``{'folder', 'paper', 'figures', 'with_legend', 'licence',
        'text_layer'}`` -- ``text_layer`` counts the pages that have one.
    """
    import csv

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paper = resolve_paper(reference, get=get)
    if paper.source == "pdf":
        figures = figures_from_pdf(paper.pdf, dest, opener=pdf_opener)
    else:
        figures = fetch_figures(paper, dest, get=get)
    rows = [{"file": figure.path.name, "legend": figure.caption}
            for figure in figures if figure.caption]
    existing = read_legends(dest / LEGENDS_FILE)
    for row in rows:
        existing[Path(row["file"]).stem] = row["legend"]
    with open(dest / LEGENDS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "legend"])
        names = {Path(f.path.name).stem: f.path.name for f in figures}
        for stem, legend in sorted(existing.items()):
            writer.writerow([names.get(stem, stem), legend])
    _write_text_layer(dest / TEXT_LAYER_FILE, figures)
    (dest / PAPER_FILE).write_text(json.dumps(asdict(paper), indent=2),
                                   encoding="utf-8")
    return {"folder": str(dest), "paper": paper.key, "figures": len(figures),
            "with_legend": len(rows), "licence": paper.licence,
            "text_layer": sum(1 for f in figures if f.words)}


def annotation_as_dict(annotation: Annotation) -> Dict[str, Any]:
    """An annotation as plain values, for JSON and the review dialog.

    :param annotation: the annotation.
    :returns: its fields, the region flattened.
    """
    out = asdict(annotation)
    out["region"] = asdict(annotation.region)
    return out



#: Strain names that mean the parental / wild-type line of a plaque figure.
WILD_TYPE_STRAINS = frozenset({
    'wt', 'wildtype', 'parental', 'parentaltir1', 'tir', 'tir1', 'tati', 'rh', 'rhdku80',
    'dku80', 'ku80', 'pru', 'prududprt', 'pruduprt', 'rhgfp', 'wtpruduprt', 'cas9',
    'rhdku80dhxgprt'})
#: Treatment agents a figure condition is read for, with their sign.
TREATMENT_AGENTS = ('iaa', 'atc', 'auxin', 'ifng', 'ifn', 'tbooh', 'halofuginone', 'pacap',
                    'rapamycin', 'dmso', 'iron', 'glutamine', 'induced')
FIGURE_METRICS = ('area', 'number', 'cleared_fraction')


def _condition_norm(text):
    """Lower-case NFKC text with delta, gamma, minus and dash spellings unified."""
    import unicodedata

    text = unicodedata.normalize('NFKC', text or '').lower()
    for old, new in (('∆', 'd'), ('δ', 'd'), ('△', 'd'), ('ifny', 'ifng'), ('k0', 'ko'),
                     ('−', '-'), ('–', '-'), ('γ', 'g')):
        text = text.replace(old, new)
    return text


def _alnum(text):
    """The letters and digits of normalised text."""
    return re.sub(r'[^a-z0-9]', '', _condition_norm(text))


def _strain_key(text):
    """The tokens that identify a mutant strain name."""
    text = _condition_norm(text)
    drop = {'tg', '12ha', '3ha', 'ha', 'aid', 'mut', 'strain', 'line', 'the', 'and', 'of',
            'parasites', 'clone', 'ko'}
    out = []
    for token in re.findall(r'[a-z0-9]+', text):
        token = re.sub(r'^tg(?=[a-z])', '', token)
        if token in drop or len(token) < 2:
            continue
        out.append(token)
    clone = re.search(r'clone\s*(\d)', text)
    if clone:
        out.append('clone' + clone.group(1))
    return out


def is_wild_type(strain) -> bool:
    """Whether a strain name means the parental or wild-type line.

    :param strain: the strain as written in a figure or a reading.
    :returns: True for the names in :data:`WILD_TYPE_STRAINS`, and names
        starting with wt or parental.
    """
    name = re.sub(r'\(.*\)', '', _alnum(strain))
    return (name in WILD_TYPE_STRAINS or name.startswith('wt') or name.startswith('parental')
            or name in ('rhku80',))


def treatment_tokens(text) -> set:
    """The signed agents, doses and condition words of a treatment.

    :param text: a treatment as written, e.g. ``'+IAA 500 µM'``.
    :returns: tokens such as ``'+iaa'``, ``'500um'``, ``'untreated'``.
    """
    text = _condition_norm(text)
    text = text.replace('ifn-g', 'ifng').replace('ifn g', 'ifng').replace('deprivation', 'depleted')
    text = re.sub(r'\bid\b', 'iron depleted', text)
    tokens = set()
    for agent in TREATMENT_AGENTS:
        pattern = (r'([+-])\s*' + agent + r'|' + agent + r'\s*([+-])|' + agent
                   + r'\s*\((\+|-)\)')
        for found in re.finditer(pattern, text):
            sign = found.group(1) or found.group(2) or found.group(3)
            tokens.add(sign + agent.replace('ifng', 'ifn'))
    for found in re.finditer(r'(\d+(?:\.\d+)?)\s*(nm|µm|um|days?|d)\b', text):
        unit = found.group(2)
        unit = 'um' if unit in ('µm', 'um') else ('d' if unit.startswith('d') else unit)
        tokens.add(found.group(1) + unit)
    for word in ('untreated', 'depleted', 'full', 'low', 'control'):
        if re.search(r'\b' + word + r'\b', text):
            tokens.add(word)
    return tokens


def strain_matches(gold, predicted) -> bool:
    """Whether a read strain names the same strain as the reference strain.

    :param gold: the reference strain, or None (anything matches).
    :param predicted: the strain text that was read.
    :returns: for a wild-type ``gold``, whether ``predicted`` is blank or a
        wild-type name; otherwise whether every token of ``gold`` occurs in
        ``predicted``.
    """
    if gold is None:
        return True
    text = _alnum(predicted)
    if is_wild_type(gold) and not text:
        return True
    if is_wild_type(gold) and re.match(r'(rh|wt|parental|pru|tir|tati|dku80|ku80)', text):
        return True
    if is_wild_type(gold):
        return (is_wild_type(predicted) or any(text.startswith(w) for w in ('wt', 'parental'))
                or any(w in text for w in ('wt', 'parental', 'wildtype', 'rhdku80', 'dku80',
                                           'tir', 'tati', 'pru')))
    keys = _strain_key(gold)
    text = text.replace('tg', '')
    return all(k.replace('tg', '') in text for k in keys) and len(keys) > 0


def treatment_matches(gold, predicted) -> bool:
    """Whether a read treatment carries every token of the reference treatment.

    A missing negative agent is excused when the read text says untreated or
    control; 'untreated' is excused unless the read text is about iron; a
    missing ``0nm`` is excused for a control.

    :param gold: the reference treatment, or blank (anything matches).
    :param predicted: the treatment text that was read.
    :returns: True when no reference token is missing.
    """
    if not gold:
        return True
    wanted, found = treatment_tokens(gold), treatment_tokens(predicted or '')
    if not wanted:
        return True
    missing = {m for m in wanted - found
               if not (m.startswith('-') and ('untreated' in found or 'control' in found))}
    if 'depleted' not in found and 'iron' not in found:
        missing -= {'untreated'}
    if missing == {'0nm'} and ('control' in found or 'untreated' in found):
        missing = set()
    return not missing


def condition_matches(gold_strain, gold_treatment, predicted_strain, predicted_treatment) -> bool:
    """Whether a read figure condition is the reported condition.

    :param gold_strain: the reported strain.
    :param gold_treatment: the reported treatment.
    :param predicted_strain: the strain text of a measured well.
    :param predicted_treatment: the treatment text of a measured well.
    :returns: :func:`strain_matches` and :func:`treatment_matches`.
    """
    return (strain_matches(gold_strain, predicted_strain)
            and treatment_matches(gold_treatment, predicted_treatment))


def resolve_plaque_model(name=None, settings=None) -> Tuple[str, str]:
    """The local checkpoint of a plaque segmentation model.

    :param name: a model-zoo key or a checkpoint path; None takes the
        ``plaque_model`` setting (default :data:`DEFAULT_SEGMENTER`).
    :param settings: run settings.
    :returns: ``(path, description)``; a zoo model is downloaded and
        checksum-verified into the shared cache first.
    """
    name = name or (settings or {}).get('plaque_model') or DEFAULT_SEGMENTER
    return _zoo_path(str(name), None, kind='cellpose', setting='plaque_model')


def segment_wells(images, model=None, settings=None) -> List[Dict[str, Any]]:
    """Segment the plaques of well crops as Figure mode does.

    Every crop is one well. With ``plaque_well_resize='auto'`` (the default)
    it is resized whole to the size the model was trained at and its
    plaques mapped back to the crop (the model's own zoo plan); ``'off'``
    segments the crop as it is.

    :param images: well crops (arrays).
    :param model: a segmenter ``fn(crop) -> labels``; None builds the one of
        ``plaque_model`` (:func:`resolve_plaque_model`).
    :param settings: ``plaque_model`` and ``plaque_well_resize``.
    :returns: one dict per crop with ``labels``, ``n`` (plaques), ``areas``
        (pixels, by label) and ``cleared_frac`` (plaque pixels over crop
        pixels).
    """
    settings = dict(settings or {})
    if model is None:
        path, _ = resolve_plaque_model(settings=settings)
        resize = str(settings.get('plaque_well_resize') or 'auto')
        plan = _model_well_resize(settings.get('plaque_model') or DEFAULT_SEGMENTER) \
            if resize != 'off' else None
        model = _cellpose_segmenter(path, well_resize=plan)
    out = []
    for image in images:
        labels = np.asarray(model(np.asarray(image)))
        ids, areas = np.unique(labels[labels > 0], return_counts=True)
        out.append({'labels': labels, 'n': int(len(ids)), 'areas': [int(a) for a in areas],
                    'cleared_frac': float((labels > 0).mean())})
    return out


def measured_by_condition(regions, metric, labeler) -> List[Tuple[str, str, float]]:
    """The measured value of every labelled, segmented well of one figure.

    :param regions: ``{box: region}`` with ``seg`` (segmented), ``n``,
        ``mean_area`` and ``frac_area`` per region.
    :param metric: ``'number'`` (plaques per well), ``'area'`` (mean plaque
        area) or ``'cleared_fraction'``; the reference spellings
        ``'plaque_number'`` and ``'plaque_area'`` are accepted.
    :param labeler: ``fn(box) -> (strain, treatment)`` or None for a well
        that is not a plaque image.
    :returns: ``(strain, treatment, value)`` per well.
    :raises ValueError: for an unknown ``metric``.
    """
    metric = {'plaque_number': 'number', 'plaque_area': 'area'}.get(metric, metric)
    if metric not in FIGURE_METRICS:
        raise ValueError(f'figure_metric must be one of {FIGURE_METRICS}, not {metric!r}')
    column = {'number': 'n', 'area': 'mean_area', 'cleared_fraction': 'frac_area'}[metric]
    values = []
    for box, region in regions.items():
        label = labeler(box)
        if label is None or not region['seg']:
            continue
        values.append((label[0], label[1], region[column]))
    return values


def _box_iou(a, b):
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def figure_mode_regions(db_path, boxes, segmentation, min_iou=0.5) -> Dict[str, Dict[str, Any]]:
    """The Figure-mode regions of one figure matched to detector boxes and their plaques.

    Each region of ``plaque_figures.db`` is matched to the detector box it
    overlaps most; one under ``min_iou`` is dropped. The box's segmentation
    (``n``, ``areas``, ``cleared_frac``) is attached when there is one.

    :param db_path: the figure's ``plaque_figures.db``.
    :param boxes: the figure's detector boxes, ``[x0, y0, x1, y1, ...]``.
    :param segmentation: ``fn(box_index) -> dict or None`` (a
        :func:`segment_wells` result).
    :param min_iou: the least overlap that matches.
    :returns: ``{str(box_index): region}`` with the condition texts,
        ``seg``, ``n``, ``mean_area`` and ``frac_area``.
    """
    import statistics

    from .tabular import read_table

    regions = read_table(db_path, table='regions', canonicalise=False, report=None)
    out = {}
    for r in regions.to_dict('records'):
        bb = [r['x0'], r['y0'], r['x1'], r['y1']]
        if not boxes:
            continue
        j = max(range(len(boxes)), key=lambda k: _box_iou(bb, boxes[k][:4]))
        if _box_iou(bb, boxes[j][:4]) < min_iou:
            continue
        seg = segmentation(j)
        areas = seg['areas'] if seg else []

        def text(value):
            """A database text that may be missing."""
            return None if value is None or (isinstance(value, float) and np.isnan(value)) else value

        out[str(j)] = dict(text=text(r['condition']) or '', legend=text(r['legend_text']) or '',
                           condition=text(r['condition']), label=text(r['label_text']),
                           strength=text(r['strength']), conflict=text(r['conflict']),
                           seg=bool(seg), n=len(areas),
                           mean_area=statistics.mean(areas) if areas else 0.0,
                           frac_area=seg['cleared_frac'] if seg else 0)
    return out


def _spearman(a, b):
    """Spearman's rho as the Pearson correlation of argsort ranks."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_reported(figures, settings=None) -> Dict[str, Any]:
    """Score measured plaque effects against the effects the papers report.

    For every reported panel of a figure (``plaque_area``, ``plaque_number``
    or ``relative_plaque_size``), each labelling of the wells gives the
    measured value of a condition as the mean over its matching wells
    (:func:`condition_matches`); the measured effect is that over the
    reference condition's, and the reported effect is the panel's value over
    its reference value. The summary has, per labelling and readout (area,
    number, all), Spearman's rho of the log2 effects (clipped at 0.02), how
    many effects that differ from 1 by more than 15% point the same way,
    how many agree within 1.5-fold, and a permutation control: rho with the
    measured effects shuffled ``compare_permutations`` times (seed
    ``compare_seed``), its mean, its 95th percentile and a one-sided p.

    :param figures: ``{figure: {'quantification': [...], 'regions': {...},
        'labelers': {name: fn(box) -> (strain, treatment) or None}}}``;
        ``quantification`` is the reported panels (``panel``, ``metric``,
        ``values`` with ``strain``, ``treatment``, ``value`` and
        ``is_reference``) and ``regions`` :func:`figure_mode_regions`.
    :param settings: ``figure_metric`` (the area readout:
        ``'area'`` or ``'cleared_fraction'``), ``compare_permutations`` and
        ``compare_seed``.
    :returns: dict with ``rows`` (a DataFrame: figure, panel, metric,
        labels, strain, treatment, reported_ratio, measured_ratio) and
        ``summary`` (a list of dicts).
    """
    import math
    import statistics

    import pandas as pd

    settings = settings or {}
    area_metric = settings.get('figure_metric') or 'area'
    rows = []
    for figure, record in figures.items():
        if not record.get('regions'):
            continue
        for panel in record['quantification']:
            if panel['metric'] not in ('plaque_area', 'plaque_number', 'relative_plaque_size'):
                continue
            metric = 'number' if panel['metric'] == 'plaque_number' else area_metric
            for source, labeler in record['labelers'].items():
                measured = measured_by_condition(record['regions'], metric, labeler)

                def value_of(entry, measured=measured):
                    """The mean measured value of the wells of one reported condition."""
                    xs = [v for s, t, v in measured
                          if condition_matches(entry['strain'], entry['treatment'],
                                               s + ' ' + (t or ''), (t or '') + ' ' + s)]
                    return statistics.mean(xs) if xs else None

                reference = [v for v in panel['values'] if v['is_reference']][0]
                measured_reference = value_of(reference)
                for entry in panel['values']:
                    if entry['is_reference']:
                        continue
                    value = value_of(entry)
                    if value is None or not measured_reference or not reference['value']:
                        continue
                    rows.append(dict(figure=figure, panel=panel['panel'], metric=panel['metric'],
                                     labels=source, strain=entry['strain'],
                                     treatment=entry['treatment'],
                                     reported_ratio=entry['value'] / reference['value'],
                                     measured_ratio=value / measured_reference))
    permutations = int(settings.get('compare_permutations') or 0)
    rng = np.random.default_rng(int(settings.get('compare_seed') or 0))
    summary = []
    sources = list(dict.fromkeys(r['labels'] for r in rows))
    for source in sources:
        for readout in ('plaque_area', 'plaque_number', 'all'):
            chosen = [r for r in rows if r['labels'] == source and (
                readout == 'all' or (r['metric'] == readout if readout == 'plaque_number'
                                     else r['metric'] != 'plaque_number'))]
            entry = {'labels': source, 'readout': readout, 'n': len(chosen),
                     'figures': len({r['figure'] for r in chosen})}
            if len(chosen) >= 3:
                a = [math.log2(max(r['reported_ratio'], 0.02)) for r in chosen]
                b = [math.log2(max(r['measured_ratio'], 0.02)) for r in chosen]
                moved = [r for r in chosen if abs(r['reported_ratio'] - 1) > 0.15]
                entry.update(
                    spearman=_spearman(a, b), direction_agree=sum(
                        (r['measured_ratio'] - 1) * (r['reported_ratio'] - 1) > 0 for r in moved),
                    direction_n=len(moved),
                    within_1_5x=sum(abs(x - y) <= math.log2(1.5) for x, y in zip(a, b)))
                if permutations:
                    null = np.array([_spearman(a, rng.permutation(b)) for _ in range(permutations)])
                    entry.update(permuted_spearman_mean=float(np.nanmean(null)),
                                 permuted_spearman_p95=float(np.nanpercentile(null, 95)),
                                 permutation_p=float((1 + np.sum(null >= entry['spearman']))
                                                     / (1 + permutations)))
            summary.append(entry)
    columns = ['figure', 'panel', 'metric', 'labels', 'strain', 'treatment', 'reported_ratio',
               'measured_ratio']
    return {'rows': pd.DataFrame(rows, columns=columns), 'summary': summary}


def _json_file(path):
    """A JSON file's content."""
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _csv_rows(path):
    """A CSV file's rows as dicts of strings, exactly as written."""
    import csv

    csv.field_size_limit(10 ** 9)
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def score_figure_folder(settings) -> Dict[str, Any]:
    """Score a Figure-mode run against the reported effects, from settings.

    Reads the reported panels of ``figure_condition_labels`` (a JSON of
    figure -> boxes and quantification, gold or model-read), the stored
    model readings of ``figure_model_labels`` (optional, data, never re-read
    here), the detector boxes of ``figure_detections``, the Figure-mode
    databases under ``figure_mode_folder`` (``<paper>/plaque_figures.db``)
    and the per-well segmentations under ``figure_segmentation_folder``
    (``<figure>__b<box>.json`` from :func:`segment_wells`), and runs
    :func:`compare_reported`. Writes ``results_measured_vs_reported.csv``
    and ``summary.json`` to ``figure_score_folder`` (default
    ``<src>/results/figure_score``).

    :param settings: the settings above plus ``figure_metric``,
        ``compare_permutations`` and ``compare_seed``.
    :returns: the :func:`compare_reported` dict with ``folder``.
    """
    import os

    from .tabular import write_table

    gold = _json_file(settings['figure_condition_labels'])
    readings = ({d['figure']: d for d in _json_file(settings['figure_model_labels'])}
                if settings.get('figure_model_labels') else {})
    detections = {r['figure']: json.loads(r['boxes'])
                  for r in _csv_rows(settings['figure_detections'])}
    seg_dir = settings['figure_segmentation_folder']
    figures = {}
    for figure, record in gold.items():
        paper = figure.split('__')[0]
        db = os.path.join(settings['figure_mode_folder'], paper, 'plaque_figures.db')
        if not os.path.exists(db):
            continue

        def segmentation(j, figure=figure):
            """The stored segmentation of one detector box, or None."""
            path = os.path.join(seg_dir, f'{figure}__b{j}.json')
            return _json_file(path) if os.path.exists(path) else None

        regions = figure_mode_regions(db, detections.get(figure, []), segmentation)
        labelers = {'gold': lambda i, g=record: (
            None if g['boxes'].get(i) is None
            else (g['boxes'][i]['strain'], g['boxes'][i]['treatment']))}
        if figure in readings:
            boxes = {str(b['i']): b for b in readings[figure]['boxes']}
            labelers['vlm'] = lambda i, vb=boxes: (
                None if not vb.get(i, {}).get('is_plaque_image')
                else (vb[i].get('strain') or '', vb[i].get('treatment') or ''))
        figures[figure] = {'quantification': record['quantification'], 'regions': regions,
                           'labelers': labelers}
    result = compare_reported(figures, settings)
    out = settings.get('figure_score_folder') or os.path.join(
        settings.get('src') or '.', 'results', 'figure_score')
    os.makedirs(out, exist_ok=True)
    write_table(result['rows'], os.path.join(out, 'results_measured_vs_reported.csv'),
                canonicalise=False)
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(result['summary'], handle, indent=1)
    result['folder'] = out
    return result


_ZOOM = re.compile(r'zoom|inset|magnif|enlarg|close-?up|duplicate|same well|crop of', re.I)


def merge_figure_wells(root, settings=None) -> Dict[str, Any]:
    """Merge the plaque wells of a figure reading into one meta-analysis table.

    Wells are the detector boxes the full reading
    (``vlm_index_all/annotations.json``) called plaque images plus the
    missed tiles the check round kept
    (``vlm_index_all/check_snapped/annotations_new.json``). Every record is
    kept; flags mark what is not measured (``rejected_not_plaque``,
    ``zoom_duplicate``) and what needs a second look (``low_confidence`` at
    or under ``figure_low_confidence``, ``snap_*``, ``reader_second_look``
    for the tasks in ``figure_second_look``). Toxoplasma wells go to the
    main table. Reported values become within-panel effects: value over the
    panel's reference value, and log2 of it (floored at 1/1000 of the
    reference).

    :param root: the folder holding ``heldout/`` and ``vlm_index_all/``.
    :param settings: ``figure_low_confidence`` (0.7) and
        ``figure_second_look`` (task ids, space- or comma-separated).
    :returns: dict of ``wells_all``, ``wells_toxo``, ``reported_values_all``
        and ``reported_effects_toxo`` (lists of dicts) and ``counts``.
    """
    import collections
    import math
    import os

    settings = settings or {}
    low = float(settings.get('figure_low_confidence', 0.7))
    second = set(re.split(r'[\s,]+', str(settings.get('figure_second_look') or '').strip())) - {''}
    held = os.path.join(root, 'heldout')
    det = {r['figure']: json.loads(r['boxes']) for r in _csv_rows(os.path.join(held, 'detections_v3.csv'))}
    full = {o['figure']: o for o in _json_file(os.path.join(root, 'vlm_index_all', 'annotations.json'))}
    check = os.path.join(root, 'vlm_index_all', 'check_snapped')
    chk = {o['figure']: o for o in _json_file(os.path.join(check, 'annotations_new.json'))}
    snapq = {(r['figure'], int(r['box_index'])): r['flags']
             for r in _csv_rows(os.path.join(held, 'snapped_box_index_map.csv'))}
    kept_rows = _csv_rows(os.path.join(held, 'detections_missed_snapped_kept.csv'))
    kept = {r['figure']: json.loads(r['new_boxes']) for r in kept_rows}
    first = {r['figure']: int(r['n_existing']) for r in kept_rows}
    task_of = {v['figure']: k for k, v in _json_file(os.path.join(check, 'index_new.json')).items()}
    man = {r['figure']: r for r in _csv_rows(os.path.join(held, 'heldout_manifest.csv'))}
    rows = []

    def add(f, i, src, bbox, b, extra_flags, task=None):
        """Append one candidate well with its flags."""
        flags = list(extra_flags)
        if not b['is_plaque_image']:
            flags.append('rejected_not_plaque')
        if b.get('confidence') is not None and b['confidence'] <= low:
            flags.append('low_confidence')
        text = ' '.join(str(b.get(k) or '') for k in ('treatment', 'replicate', 'notes', 'strain'))
        if _ZOOM.search(text):
            flags.append('zoom_duplicate')
        org = (full.get(f) or chk.get(f) or {}).get('organism', '')
        if org != 'Toxoplasma':
            flags.append(f'organism_{org}')
        measurable = b['is_plaque_image'] and 'zoom_duplicate' not in flags
        rows.append(dict(figure=f, pmcid=man[f]['pmcid'], box_index=i, source=src, task=task or '',
                         x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], panel=b.get('panel'),
                         strain=b.get('strain'), treatment=b.get('treatment'),
                         replicate=b.get('replicate'),
                         is_reference=int(bool(b.get('is_parental_control'))),
                         confidence=b.get('confidence'), notes=b.get('notes') or '',
                         organism=org, detector_trained=man[f]['detector_trained'],
                         measurable=int(measurable), main_toxo=int(measurable and org == 'Toxoplasma'),
                         flags=';'.join(flags)))

    for f, o in full.items():
        for b in o['boxes']:
            if b['i'] < len(det.get(f, [])):
                add(f, b['i'], 'detector_v3', [int(v) for v in det[f][b['i']][:4]], b, [])
    for f, o in chk.items():
        for b in o['boxes']:
            bb = kept[f][b['i'] - first[f]][:4]
            snap = snapq.get((f, b['i']), '')
            task = task_of.get(f)
            add(f, b['i'], 'reader_tile_snapped', bb, b,
                [f'snap_{x}' for x in snap.split(';') if x]
                + (['reader_second_look'] if task in second else []), task=task)
    values, effects = [], []
    for f, o in full.items():
        for pi, q in enumerate(o['quantification']):
            ref = [v for v in q['values'] if v['is_parental_control'] and v['mean'] is not None]
            refv = ref[0]['mean'] if ref else None
            for v in q['values']:
                r = dict(figure=f, pmcid=man[f]['pmcid'], organism=o['organism'], panel=q['panel'],
                         panel_index=pi, metric=q['metric'], unit=q['unit'],
                         statistic=q['statistic'], condition=v['condition'], strain=v['strain'],
                         treatment=v['treatment'], value=v['mean'],
                         is_reference=int(v['is_parental_control']),
                         panel_confidence=q['confidence'],
                         significance=q['significance_vs_control'], n_references_in_panel=len(ref))
                values.append(r)
                if (o['organism'] == 'Toxoplasma' and refv and v['mean'] is not None
                        and not v['is_parental_control'] and refv > 0 and v['mean'] >= 0):
                    e = dict(r)
                    e['reference_condition'] = ref[0]['condition']
                    e['reference_value'] = refv
                    e['ratio'] = v['mean'] / refv
                    e['log2_ratio'] = math.log2(max(v['mean'], 1e-3 * refv) / refv)
                    e['flags'] = ';'.join(x for x in (
                        'multiple_references' if len(ref) > 1 else '',
                        'low_panel_confidence' if (q['confidence'] or 0) <= low else '') if x)
                    effects.append(e)
    toxo = [r for r in rows if r['main_toxo']]
    second_look = ('low_confidence', 'reader_second_look', 'snap_')
    count = collections.Counter
    counts = {
        'candidate_wells': len(rows),
        'detector_boxes': sum(r['source'] == 'detector_v3' for r in rows),
        'detector_plaque': sum(r['source'] == 'detector_v3' and 'rejected_not_plaque' not in r['flags']
                               for r in rows),
        'snapped_tiles_checked': sum(r['source'] == 'reader_tile_snapped' for r in rows),
        'snapped_kept': sum(r['source'] == 'reader_tile_snapped'
                            and 'rejected_not_plaque' not in r['flags'] for r in rows),
        'snapped_rejected': sum(r['source'] == 'reader_tile_snapped'
                                and 'rejected_not_plaque' in r['flags'] for r in rows),
        'zoom_duplicates_excluded': sum('zoom_duplicate' in r['flags']
                                        and 'rejected_not_plaque' not in r['flags'] for r in rows),
        'measurable_wells_all_organisms': sum(r['measurable'] for r in rows),
        'measurable_wells_toxo': len(toxo),
        'measurable_wells_toxo_by_source': dict(count(r['source'] for r in toxo)),
        'low_confidence_wells': sum('low_confidence' in r['flags'] for r in rows),
        'reader_second_look_wells': sum('reader_second_look' in r['flags'] for r in rows),
        'needs_second_look_wells': sum(any(k in r['flags'] for k in second_look) and r['measurable'] == 1
                                       for r in rows),
        'figures_with_toxo_wells': len({r['figure'] for r in toxo}),
        'articles_with_toxo_wells': len({r['pmcid'] for r in toxo}),
        'toxo_wells_in_detector_trained_articles': sum(r['detector_trained'] == '1' for r in toxo),
        'toxo_reference_wells': sum(r['is_reference'] for r in toxo),
        'toxo_figures_without_reference_well': len({r['figure'] for r in toxo}
                                                   - {r['figure'] for r in toxo if r['is_reference']}),
        'organism_wells_measurable': dict(count(r['organism'] for r in rows if r['measurable'])),
        'reported_panels_all': len({(v['figure'], v['panel_index']) for v in values}),
        'reported_values_all': len(values),
        'reported_panels_toxo': len({(v['figure'], v['panel_index']) for v in values
                                     if v['organism'] == 'Toxoplasma'}),
        'reported_values_toxo': sum(v['organism'] == 'Toxoplasma' for v in values),
        'toxo_panels_with_one_reference': len({(v['figure'], v['panel_index']) for v in values
                                               if v['organism'] == 'Toxoplasma'
                                               and v['n_references_in_panel'] == 1}),
        'toxo_panels_without_reference': len({(v['figure'], v['panel_index']) for v in values
                                              if v['organism'] == 'Toxoplasma'
                                              and v['n_references_in_panel'] == 0}),
        'toxo_effects': len(effects),
        'toxo_effects_by_metric': dict(count(e['metric'] for e in effects)),
        'toxo_articles_with_effects': len({e['pmcid'] for e in effects}),
    }
    return {'wells_all': rows, 'wells_toxo': toxo, 'reported_values_all': values,
            'reported_effects_toxo': effects, 'counts': counts}



def run_figure_scoring(settings) -> Dict[str, Any]:
    """The Plaque module's figure scoring and figure merge, from a settings file.

    ``plaque_mode='figure_score'`` runs :func:`score_figure_folder`,
    ``'figure_sizes'`` :func:`compare_figure_sizes`, and
    ``'figure_merge'`` runs :func:`merge_figure_wells` on ``src`` and writes
    wells_all.csv, wells_toxo.csv, reported_values_all.csv,
    reported_effects_toxo.csv and merge_counts.json to
    ``<src>/results/figure_merge``.

    :param settings: Plaque module settings.
    :returns: the result dict of the chosen step, with ``folder``.
    :raises ValueError: for another ``plaque_mode``.
    """
    import os

    from .tabular import write_table

    mode = str(settings.get('plaque_mode'))
    if mode == 'figure_score':
        return score_figure_folder(settings)
    if mode == 'figure_sizes':
        return compare_figure_sizes(settings)
    if mode != 'figure_merge':
        raise ValueError(f"plaque_mode must be 'figure_score', 'figure_sizes' or "
                         f"'figure_merge', not {mode!r}")
    import pandas as pd

    result = merge_figure_wells(settings['src'], settings)
    out = os.path.join(settings['src'], 'results', 'figure_merge')
    os.makedirs(out, exist_ok=True)
    for name in ('wells_all', 'wells_toxo', 'reported_values_all', 'reported_effects_toxo'):
        write_table(pd.DataFrame(result[name]), os.path.join(out, f'{name}.csv'), canonicalise=False)
    with open(os.path.join(out, 'merge_counts.json'), 'w', encoding='utf-8') as handle:
        json.dump(result['counts'], handle, indent=1)
    result['folder'] = out
    return result



FIGURE_SCALES = ('absolute', 'rel_panel_max', 'rel_well_extent', 'rel_reference')
_SIZE_READOUTS = ('plaque_area', 'plaque_number', 'relative_plaque_size')


def plaque_size_readouts(areas, well_box=None) -> Dict[str, Any]:
    """Absolute and relative plaque sizes of one well.

    :param areas: plaque areas in pixels.
    :param well_box: the well box ``(x0, y0, x1, y1)``, or None.
    :returns: ``plaque_count``, ``plaque_mean_area_px2``,
        ``plaque_median_area_px2``, ``plaque_max_area_px2`` (0 without
        plaques), ``plaque_max_over_mean`` (NaN without plaques),
        ``area_rel_well_max`` (each plaque over the largest), and with a box
        ``well_extent_px2`` (width x height), ``well_extent_px`` (mean of
        width and height) and ``plaque_area_rel_well_extent``.
    """
    areas = np.asarray(list(areas), dtype=float)
    out = {'plaque_count': int(len(areas)),
           'plaque_mean_area_px2': float(areas.mean()) if len(areas) else 0.0,
           'plaque_median_area_px2': float(np.median(areas)) if len(areas) else 0.0,
           'plaque_max_area_px2': float(areas.max()) if len(areas) else 0.0}
    out['plaque_max_over_mean'] = (out['plaque_max_area_px2'] / out['plaque_mean_area_px2']
                                   if len(areas) and out['plaque_mean_area_px2'] else float('nan'))
    out['area_rel_well_max'] = (areas / areas.max()).tolist() if len(areas) and areas.max() else []
    if well_box is not None:
        width, height = well_box[2] - well_box[0], well_box[3] - well_box[1]
        out['well_extent_px2'] = float(width * height)
        out['well_extent_px'] = float((width + height) / 2)
        out['plaque_area_rel_well_extent'] = (out['plaque_mean_area_px2'] / out['well_extent_px2']
                                              if out['well_extent_px2'] else float('nan'))
    return out


def figure_condition_sizes(gold, segmentation, settings=None) -> pd.DataFrame:
    """One row per reported condition with the plaque sizes measured in its wells.

    The reported values are the panels of each figure's quantification whose
    metric is plaque_area, plaque_number or relative_plaque_size. A
    condition's wells are the boxes whose condition label matches it
    (:func:`condition_matches`); over several wells the value is their mean
    (``figure_condition_aggregate='median'`` takes the median), and a well
    without plaques has area 0. ``figure_size_stat`` picks the per-well mean
    or median plaque area.

    :param gold: ``{figure: {'boxes': {box: {'strain', 'treatment'}},
        'quantification': [...]}}``.
    :param segmentation: ``fn(figure, box) -> {'areas', 'bbox', 'n'} or None``.
    :param settings: ``figure_condition_aggregate`` and ``figure_size_stat``.
    :returns: one row per reported condition with at least one matched
        well (a blank treatment as ``''``, ``is_ref`` 0 or 1): figure,
        pmcid, panel, metric, unit, strain, treatment, is_ref, reported,
        n_wells, spacr_n, spacr_mean_area_px2, spacr_max_area_px2,
        well_box_px2 and well_box_px.
    """
    import statistics

    import pandas as pd

    settings = settings or {}
    pool = statistics.median if settings.get('figure_condition_aggregate') == 'median' \
        else statistics.mean
    size_key = ('plaque_median_area_px2' if settings.get('figure_size_stat') == 'median'
                else 'plaque_mean_area_px2')
    rows = []
    for figure, record in gold.items():
        wells = []
        for box, label in record.get('boxes', {}).items():
            seg = segmentation(figure, box) if label else None
            if seg is None:
                continue
            readout = plaque_size_readouts(seg.get('areas', []), seg.get('bbox'))
            wells.append((label['strain'], label.get('treatment'), readout))
        for panel in record.get('quantification', []):
            if panel['metric'] not in _SIZE_READOUTS:
                continue
            for value in panel['values']:
                matched = [r for s, t, r in wells if condition_matches(
                    value['strain'], value['treatment'], s + ' ' + (t or ''), (t or '') + ' ' + s)]
                if not matched:
                    continue
                rows.append(dict(
                    figure=figure, pmcid=figure.split('__')[0], panel=panel['panel'],
                    metric=panel['metric'], unit=panel.get('unit'), strain=value['strain'],
                    treatment=value['treatment'] or '', is_ref=int(bool(value['is_reference'])),
                    reported=value['value'], n_wells=len(matched),
                    spacr_n=pool([r['plaque_count'] for r in matched]),
                    spacr_mean_area_px2=pool([r[size_key] for r in matched]),
                    spacr_max_area_px2=pool([r['plaque_max_area_px2'] for r in matched]),
                    well_box_px2=pool([r.get('well_extent_px2', np.nan) for r in matched]),
                    well_box_px=pool([r.get('well_extent_px', np.nan) for r in matched])))
    columns = ['figure', 'pmcid', 'panel', 'metric', 'unit', 'strain', 'treatment', 'is_ref',
               'reported', 'n_wells', 'spacr_n', 'spacr_mean_area_px2', 'spacr_max_area_px2',
               'well_box_px2', 'well_box_px']
    return pd.DataFrame(rows, columns=columns)


def _panel_relative(table, column):
    """``column`` over its (pmcid, panel) maximum; NaN when the maximum is 0."""
    peak = table.groupby(['pmcid', 'panel'])[column].transform('max')
    return table[column] / peak.where(peak != 0)


def figure_size_scales(conditions, reference_rows=None, settings=None) -> pd.DataFrame:
    """Measured and reported plaque sizes on the four comparison scales.

    ``absolute``: counts only, for readouts whose unit names one of
    ``figure_absolute_units``. ``rel_panel_max``: each side over the largest
    value of its figure panel. ``rel_well_extent``: area readouts; spaCR
    area over the well extent, then over the panel maximum, against the
    reported value over its panel maximum. ``rel_reference``: log2 of each
    side's ratio to the reference condition, floored at
    ``figure_ratio_floor``, taken from the gold rows of
    :func:`compare_reported`.

    :param conditions: :func:`figure_condition_sizes` rows.
    :param reference_rows: :func:`compare_reported` rows, or None.
    :param settings: ``figure_scales``, ``figure_absolute_units`` and
        ``figure_ratio_floor``.
    :returns: one row per scale and condition with ``reported_value``,
        ``spacr_value``, ``readout``, figure, pmcid, panel, strain and
        treatment.
    """
    import math

    import pandas as pd

    settings = settings or {}
    scales = settings.get('figure_scales') or FIGURE_SCALES
    scales = [s.strip() for s in scales.split(',')] if isinstance(scales, str) else list(scales)
    units = settings.get('figure_absolute_units') or ['count']
    units = [u.strip().lower() for u in (units.split(',') if isinstance(units, str) else units)]
    floor = float(settings.get('figure_ratio_floor') or 0.02)
    t = conditions[(conditions['n_wells'] > 0) & conditions['reported'].notna()].copy()
    t['spacr_value'] = np.where(t['metric'] == 'plaque_number', t['spacr_n'],
                                t['spacr_mean_area_px2'])
    area = t['metric'] != 'plaque_number'
    t['spacr_well'] = np.where(area, t['spacr_value'] / t['well_box_px2'], np.nan)
    t['rep_rel'] = _panel_relative(t, 'reported')
    t['sp_rel'] = _panel_relative(t, 'spacr_value')
    t['sp_wellrel'] = _panel_relative(t, 'spacr_well')
    keep = ['figure', 'pmcid', 'panel', 'metric', 'strain', 'treatment']
    out = []
    unit = t['unit'].fillna('').str.lower()
    counted = (t['metric'] == 'plaque_number') & unit.apply(lambda u: any(k in u for k in units))
    for scale, mask, reported, measured in (
            ('absolute', counted, 'reported', 'spacr_value'),
            ('rel_panel_max', t['metric'].notna(), 'rep_rel', 'sp_rel'),
            ('rel_well_extent', area, 'rep_rel', 'sp_wellrel')):
        if scale in scales and mask.any():
            part = t.loc[mask]
            out.append(part[keep].assign(scale=scale, reported_value=part[reported],
                                         spacr_value=part[measured]))
    if 'rel_reference' in scales and reference_rows is not None and len(reference_rows):
        r = reference_rows[reference_rows['labels'] == 'gold']
        out.append(pd.DataFrame({
            'figure': r['figure'], 'pmcid': r['figure'].str.split('__').str[0],
            'panel': r['panel'], 'metric': r['metric'], 'strain': r['strain'],
            'treatment': r['treatment'], 'scale': 'rel_reference',
            'reported_value': [math.log2(max(v, floor)) for v in r['reported_ratio']],
            'spacr_value': [math.log2(max(v, floor)) for v in r['measured_ratio']]}))
    columns = ['scale'] + keep + ['reported_value', 'spacr_value']
    table = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)
    table = table[columns].dropna(subset=['reported_value', 'spacr_value'])
    table['readout'] = table['metric'].map({'plaque_number': 'number', 'plaque_area': 'area',
                                            'relative_plaque_size': 'relative size'})
    return table.reset_index(drop=True)


def figure_size_table(conditions) -> pd.DataFrame:
    """Per condition: published and spaCR value, both relative to their panel maximum, and the spread.

    :param conditions: :func:`figure_condition_sizes` rows.
    :returns: one row per condition with figure (pmcid and panel), readout,
        reported, spacr (count for number readouts, mean plaque area
        otherwise), reported_rel, spacr_rel and largest_over_mean (largest
        over mean plaque area, area readouts with plaques only).
    """
    import pandas as pd

    t = conditions.copy()
    t['spacr'] = np.where(t['metric'] == 'plaque_number', t['spacr_n'], t['spacr_mean_area_px2'])
    readout = {'plaque_number': 'Plaque number', 'plaque_area': 'Plaque area',
               'relative_plaque_size': 'Relative plaque size'}
    spread = np.where((t['spacr_mean_area_px2'] > 0) & (t['metric'] != 'plaque_number'),
                      t['spacr_max_area_px2'] / t['spacr_mean_area_px2'].where(
                          t['spacr_mean_area_px2'] > 0), np.nan)
    return pd.DataFrame({'figure': t['pmcid'] + ' ' + t['panel'].astype(str),
                         'readout': t['metric'].map(readout), 'reported': t['reported'],
                         'spacr': t['spacr'], 'reported_rel': _panel_relative(t, 'reported'),
                         'spacr_rel': _panel_relative(t, 'spacr'),
                         'largest_over_mean': spread})


def _agreement(x, y):
    """R2, p, Spearman rho, Lin's concordance and median log2 bias of two value sets."""
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {'n': int(len(x)), 'r2': np.nan, 'p': np.nan, 'spearman': np.nan, 'lin_ccc': np.nan,
           'median_log2_bias': np.nan}
    if len(x) < 3:
        return out
    r, p = stats.pearsonr(x, y)
    out.update(r2=float(r ** 2), p=float(p), spearman=float(stats.spearmanr(x, y)[0]))
    sx, sy = x.var(), y.var()
    out['lin_ccc'] = float(2 * np.cov(x, y, bias=True)[0, 1] / (sx + sy + (x.mean() - y.mean()) ** 2))
    positive = (x > 0) & (y > 0)
    if positive.any():
        out['median_log2_bias'] = float(np.median(np.log2(y[positive] / x[positive])))
    return out


def figure_size_agreement(scales, settings=None) -> pd.DataFrame:
    """Agreement of measured with reported sizes per scale, overall and per readout.

    R2 is the squared Pearson r (NaN under 3 points), beside its p,
    Spearman rho, Lin's concordance and the median log2 bias of spaCR over
    the paper (positive pairs). Controls: for every relative scale the spaCR
    values are permuted among the conditions of each panel
    ``figure_scale_permutations`` times (seed ``figure_scale_seed``), giving
    the null median and 95th percentile of R2; and the agreement with each
    panel's maximum-defining conditions left out (``rel_panel_max`` and
    ``rel_well_extent``).

    :param scales: :func:`figure_size_scales` rows.
    :param settings: ``figure_scale_permutations`` (1000) and
        ``figure_scale_seed`` (0).
    :returns: one row per scale and readout (``'all'`` included) with
        ``r2``, ``p``, ``n``, ``figures``, ``spearman``, ``lin_ccc``,
        ``median_log2_bias``, ``null_r2_median``, ``null_r2_p95``,
        ``r2_without_panel_max`` and ``n_without_panel_max``.
    """
    import pandas as pd

    settings = settings or {}
    permutations = int(settings.get('figure_scale_permutations', 1000) or 0)
    rng = np.random.default_rng(int(settings.get('figure_scale_seed') or 0))
    rows = []
    for scale in [s for s in FIGURE_SCALES if s in set(scales['scale'])]:
        table = scales[scales['scale'] == scale]
        for readout in ['all'] + [r for r in ('number', 'area', 'relative size')
                                  if r in set(table['readout'])]:
            chosen = table if readout == 'all' else table[table['readout'] == readout]
            row = {'scale': scale, 'readout': readout,
                   **_agreement(chosen['reported_value'], chosen['spacr_value']),
                   'figures': int(chosen['figure'].nunique())}
            if scale != 'absolute' and permutations and len(chosen) >= 3:
                groups = [np.nonzero((chosen['pmcid'] + '|' + chosen['panel'].astype(str)
                                      == key).to_numpy())[0]
                          for key in pd.unique(chosen['pmcid'] + '|' + chosen['panel'].astype(str))]
                values = chosen['spacr_value'].to_numpy(float)
                null = []
                for _ in range(permutations):
                    shuffled = values.copy()
                    for idx in groups:
                        shuffled[idx] = values[rng.permutation(idx)]
                    null.append(_agreement(chosen['reported_value'], shuffled)['r2'])
                row['null_r2_median'] = float(np.nanmedian(null))
                row['null_r2_p95'] = float(np.nanpercentile(null, 95))
            if scale in ('rel_panel_max', 'rel_well_extent'):
                rest = chosen[(chosen['reported_value'] < 1) & (chosen['spacr_value'] < 1)]
                trimmed = _agreement(rest['reported_value'], rest['spacr_value'])
                row['r2_without_panel_max'] = trimmed['r2']
                row['n_without_panel_max'] = trimmed['n']
            rows.append(row)
    return pd.DataFrame(rows)


def figure_size_figure(scales, agreement):
    """Four scatter panels of measured against reported plaque size, one per scale.

    :param scales: :func:`figure_size_scales` rows.
    :param agreement: :func:`figure_size_agreement` rows.
    :returns: a :class:`matplotlib.figure.Figure`; points coloured by
        readout at 60% opacity, the identity line dashed, R2 and n in the
        corner and the permutation null R2 (95th percentile) beside them.
    """
    from matplotlib.figure import Figure

    colours = {'number': '#2A9D8F', 'area': '#3A7BD5', 'relative size': '#1F4E79'}
    titles = {'absolute': 'absolute (counts)', 'rel_panel_max': 'relative to panel maximum',
              'rel_well_extent': 'relative to well extent', 'rel_reference': 'log2 vs reference'}
    fig = Figure(figsize=(12, 3.2), dpi=100)
    axes = fig.subplots(1, 4)
    for ax, scale in zip(axes, FIGURE_SCALES):
        table = scales[scales['scale'] == scale]
        for readout, part in table.groupby('readout'):
            ax.scatter(part['reported_value'], part['spacr_value'], s=14, alpha=0.6,
                       color=colours.get(readout, '#1F4E79'), linewidths=0, label=readout)
        if len(table):
            lo = float(min(table['reported_value'].min(), table['spacr_value'].min()))
            hi = float(max(table['reported_value'].max(), table['spacr_value'].max()))
            ax.plot([lo, hi], [lo, hi], ls='--', lw=0.8, color='#8A94A6')
        row = agreement[(agreement['scale'] == scale) & (agreement['readout'] == 'all')]
        if len(row):
            row = row.iloc[0]
            text = f"R² {row['r2']:.2f}, n {int(row['n'])}"
            if np.isfinite(row.get('null_r2_p95', np.nan)):
                text += f"\nnull p95 {row['null_r2_p95']:.2f}"
            ax.text(0.04, 0.96, text, transform=ax.transAxes, va='top', fontsize='small')
        ax.set_title(titles[scale], fontsize='small')
        ax.set_xlabel('reported')
        ax.set_ylabel('spaCR')
    if len(scales):
        axes[0].legend(fontsize='x-small', frameon=False, loc='lower right')
    fig.tight_layout()
    return fig


def compare_figure_sizes(settings) -> Dict[str, Any]:
    """The plaque-size comparison of a figure set, from settings.

    Builds :func:`figure_condition_sizes` from ``figure_condition_labels``
    and the per-well segmentations of ``figure_segmentation_folder``, the
    scales (:func:`figure_size_scales`, with the gold rows of
    :func:`score_figure_folder` for ``rel_reference``) and their agreement,
    and writes conditions_measured_vs_reported.csv,
    agreement_by_scale.csv, size_scales.csv, sizes_per_condition.csv
    (:func:`figure_size_table`) and the plaque_size_agreement
    figure to ``figure_score_folder``.

    :param settings: the figure scoring settings and the size settings.
    :returns: dict with ``conditions``, ``scales``, ``agreement`` and
        ``folder``.
    """
    import os

    from .plot import save_figure
    from .tabular import write_table

    gold = _json_file(settings['figure_condition_labels'])
    seg_dir = settings['figure_segmentation_folder']

    def segmentation(figure, box):
        """The stored segmentation of one well box, or None."""
        path = os.path.join(seg_dir, f'{figure}__b{box}.json')
        return _json_file(path) if os.path.exists(path) else None

    conditions = figure_condition_sizes(gold, segmentation, settings)
    reference = score_figure_folder(dict(settings, figure_metric='area'))['rows'] \
        if settings.get('figure_detections') else None
    scales = figure_size_scales(conditions, reference, settings)
    agreement = figure_size_agreement(scales, settings)
    out = settings.get('figure_score_folder') or os.path.join(
        settings.get('src') or '.', 'results', 'figure_score')
    os.makedirs(out, exist_ok=True)
    write_table(conditions, os.path.join(out, 'conditions_measured_vs_reported.csv'),
                canonicalise=False)
    write_table(scales, os.path.join(out, 'size_scales.csv'), canonicalise=False)
    write_table(agreement, os.path.join(out, 'agreement_by_scale.csv'), canonicalise=False)
    write_table(figure_size_table(conditions), os.path.join(out, 'sizes_per_condition.csv'),
                canonicalise=False)
    save_figure(figure_size_figure(scales, agreement),
                os.path.join(out, 'plaque_size_agreement.pdf'), close=True)
    return {'conditions': conditions, 'scales': scales, 'agreement': agreement, 'folder': out}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m spacr.plaque_papers DOI|PMID|PMC|PDF ... --dst DIR``.

    :param argv: arguments; defaults to ``sys.argv[1:]``.
    :returns: the exit status.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m spacr.plaque_papers",
        description="Measure plaques in the figures of published papers.")
    parser.add_argument("references", nargs="+",
                        help="DOIs, PMIDs, PMC ids or PDF paths")
    parser.add_argument("--dst", required=True, help="output folder")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--segmenter", default=DEFAULT_SEGMENTER)
    parser.add_argument("--imgsz", default=",".join(map(str, DEFAULT_IMGSZ)),
                        help="comma-separated inference sizes")
    parser.add_argument("--plate-format", default=None,
                        help="e.g. 6-well, for whole-well images")
    parser.add_argument("--confirm-each", action="store_true",
                        help="approve every proposed condition by hand")
    args = parser.parse_args(argv)
    summary = measure_plaques_from_papers(
        args.references, args.dst, detector=args.detector,
        segmenter=args.segmenter,
        imgsz=tuple(int(s) for s in args.imgsz.split(",") if s.strip()),
        plate_format=args.plate_format, confirm_each=args.confirm_each)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
