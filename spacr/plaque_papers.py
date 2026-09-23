"""Plaque measurements from published papers: a DOI, PMID or PDF in, rows out.

Given a paper, this module finds its figures, finds the plaque
images inside them with the zoo's well detector, reads the text printed
around each image, works out which condition each image shows, segments and
measures the plaques, and writes everything to a SQLite database together
with where every value came from.

THE CONDITION IS THE HARD PART, and it is read two independent ways, always
both, because two readings that agree are evidence and one is a guess:

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

SIZES. A plaque area in pixels depends on the paper's printing, so every
plaque also carries its area relative to the median plaque in the same panel,
and an area in mm^2 only when there is a ruler: a scale bar read in or under
the image (its length from the label beside it, or from the legend's "scale
bar, 1 mm"), or a whole well of a plate format the settings or the legend
state. A stated magnification is recorded but is never a ruler, because a
printed figure has been rescaled since the picture was taken. Rows with no
ruler say their sizes are in pixels.

DUPLICATES. A figure is identified by its paper (the DOI when there is one)
and its image hash. The same image seen again -- the same bytes, or the same
pixels in another file format, under another name or another paper, such as a
preprint and its published version -- is measured once and recorded in the
``duplicates`` table against the copy that was measured. A paper already
measured from one source (Europe PMC) is not measured again from another (its
PDF). An image that merely LOOKS the same (a near-identical perceptual hash,
as a re-encoded or resized copy would have) is recorded as a possible
duplicate and still measured, because on figures that is a resemblance, not
a proof.

TEXT. A PDF's own text layer is used for legends and labels before any OCR;
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
    "read_legends",
    "read_annotation_overrides",
    "write_annotation_overrides",
    "apply_overrides",
    "reread_around",
    "console_legend_prompt",
    "console_review",
]

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = {"User-Agent": "spacr-plaque-papers/1.0 "
                            "(https://github.com/EinarOlafsson/spacr)"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp", ".webp"}
FORMAT_RANK = {".tiff": 0, ".tif": 0, ".png": 1, ".webp": 2, ".jpg": 2,
               ".jpeg": 2, ".gif": 3, ".bmp": 3}

DEFAULT_DETECTOR = "toxoplasma_well_detector_v2"
DEFAULT_SEGMENTER = "toxoplasma_plaque_v2"
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
    """Say what kind of reference a user typed.

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
    ``https://d oi.org/10.1371/j ournal.\nppat.1011009`` is what one PLOS
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
    the largest, because journals also ship thumbnails and a detector shown
    a thumbnail has not been asked the question. Legends come from the
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
    from ._segmentation_backends import _worker_for

    reply = _worker_for(READER_BACKEND, reader_environment()).request(
        "read_pdf", pdf=str(Path(pdf).resolve()), dest=str(dest.resolve()),
        dpi=int(dpi), x_tolerance=PDF_X_TOLERANCE)
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
                raise ImportError(
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
        raise ImportError(
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
    """:func:`spacr.plaque.detect_wells`, answered by the reader's environment.

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
            raise ImportError(
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
      corner, within one image size of it. Further than that is another
      panel's letter, and no letter is better than the wrong one.

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

    :param labels: the segmenter's label image, 0 = background.
    :param px_per_mm: the image's scale, when it has a ruler.
    :returns: ``[{'label', 'area_px', 'area_mm2'}]``; ``area_mm2`` is None
        without a ruler.
    """
    ids, counts = np.unique(np.asarray(labels), return_counts=True)
    rows = []
    for label, area in zip(ids.tolist(), counts.tolist()):
        if label == 0:
            continue
        mm2 = float(area) / (px_per_mm ** 2) if px_per_mm else None
        rows.append({"label": int(label), "area_px": int(area), "area_mm2": mm2})
    return rows


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
        ``'well: <format> (settings)'``, ``'well: <format> (legend)'`` or
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
                       plate_format: Optional[str] = None) -> List[_Scale]:
    """The ruler, if any, for every plaque image in one figure.

    In order of preference:

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
    :returns: one :class:`_Scale` per region.
    """
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
    """Ask at the terminal for a legend nobody could fetch.

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
        ("area_mm2", "REAL"), ("area_vs_panel_median", "REAL")),
    "duplicates": (
        ("figure_sha256", "TEXT"), ("paper_key", "TEXT"), ("path", "TEXT"),
        ("label", "TEXT"), ("match", "TEXT"), ("distance", "INTEGER"),
        ("duplicate_of_sha256", "TEXT"), ("duplicate_of_paper_key", "TEXT"),
        ("duplicate_of_path", "TEXT"), ("measured", "INTEGER"),
        ("run_id", "INTEGER"), ("noted", "REAL")),
}


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
        """The row is this very file of this very paper."""
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

    :param path: the image file.
    :returns: the pixels.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _zoo_path(key: str, cache: Path) -> Tuple[str, str]:
    """A zoo model's local path and a description to record with results.

    :param key: a zoo key, or a path to a checkpoint.
    :param cache: where a downloaded model is kept.
    :returns: ``(path, 'name sha256')``.
    """
    if Path(str(key)).exists():
        return str(key), f"{Path(str(key)).name}"
    from . import model_zoo as zoo

    entry = zoo.resolve(key, zoo.catalogue())
    path = entry.path if getattr(entry, "path", None) and Path(entry.path).exists() \
        else zoo.fetch(entry, cache)
    return str(path), f"{entry.name} {entry.sha256 or ''}".strip()


def _cellpose_segmenter(path: str) -> Callable[[np.ndarray], np.ndarray]:
    """A plaque segmenter from a Cellpose checkpoint.

    :param path: the checkpoint.
    :returns: ``fn(crop) -> labels``.
    """
    from cellpose import models

    try:
        from .accelerator import cellpose_kwargs

        kwargs = cellpose_kwargs()
    except Exception:
        kwargs = {"gpu": False}
    kwargs.pop("device", None)
    model = models.CellposeModel(pretrained_model=path, device=None, **kwargs)

    def segment(crop: np.ndarray) -> np.ndarray:
        """The Cellpose label mask of one plaque image crop."""
        masks = model.eval(crop)[0]
        return np.asarray(masks)
    return segment


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
        _zoo_path(detector, dst / "models")
    segmenter_id = str(segmenter)
    if segment is None:
        segmenter_path, segmenter_id = _zoo_path(segmenter, dst / "models")
        segment = _cellpose_segmenter(segmenter_path)
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
                    run_id: Optional[int]) -> Dict[str, Any]:
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
            "run_id": run_id}


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
                                plate_format=kw["plate_format"])
    crops.mkdir(parents=True, exist_ok=True)
    measured: List[Tuple[int, Annotation, List[Dict[str, Any]], str, _Scale]] = []
    for index, a in enumerate(annotations, start=1):
        scale = scales[index - 1]
        if a.approved is False:
            _insert(connection, "figure_annotations", _annotation_row(
                figure, paper, index, a, scale, measured=False,
                region_id=None, run_id=run_id))
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
            "run_id": run_id})
        region_id = int(cursor.lastrowid)
        _insert(connection, "figure_annotations", _annotation_row(
            figure, paper, index, a, scale, measured=True, region_id=region_id,
            run_id=run_id))
        median = medians.get(a.panel) or 0.0
        connection.executemany(
            "INSERT INTO plaques (region_id, label, area_px, area_mm2, "
            "area_vs_panel_median) VALUES (?,?,?,?,?)",
            [(region_id, row["label"], row["area_px"], row["area_mm2"],
              row["area_px"] / median if median else None) for row in rows])
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
                      "conflict", "conflict_reason")


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
            "conflict_reason": _conflict_reason(a)}


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
        plate_format: Optional[str] = None, legends: Any = None,
        annotations: Any = None, read_text: Optional[Callable] = None,
        detect: Optional[Callable] = None, segment: Optional[Callable] = None,
        text_options: Optional[TextOptions] = None) -> Dict[str, Any]:
    """Figure mode of Plaque Assay: find, read, annotate and measure a folder.

    Nothing here stops to ask. Legends come from ``legends`` (default
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
    :param legends: CSV of legends, see :func:`read_legends`.
    :param annotations: CSV of reviews, see :func:`read_annotation_overrides`.
    :param read_text: ``fn(path) -> [Word]``.
    :param detect: passed to :func:`find_plaque_regions`.
    :param segment: ``fn(crop) -> labels``.
    :param text_options: how the text is read (:class:`TextOptions`).
    :returns: the summary, with ``awaiting_approval`` added.
    """
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
        _zoo_path(detector, dst / "models")
    segmenter_id = str(segmenter)
    if segment is None:
        segmenter_path, segmenter_id = _zoo_path(segmenter, dst / "models")
        segment = _cellpose_segmenter(segmenter_path)
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
                        plate_format=plate_format,
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
