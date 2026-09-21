"""Plaque measurements from published papers: a DOI, PMID or PDF in, rows out.

Item 424. Given a paper, this module finds its figures, finds the plaque
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
says "under indicated conditions" is not a disagreement. ``conflict`` is kept
for a reviewer to set. When neither is present the image is named by its
figure and its row and column in the panel, and marked ``weak`` so a dataset
can leave those rows out.

SIZES. A plaque area in pixels depends on the paper's printing, so every
plaque also carries its area relative to the median plaque in the same panel,
and an area in mm^2 only when the image is a whole well of a stated plate
format. Rows with no ruler say so.

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
    :param source: ``'europepmc'`` or ``'pdf'``.
    """

    key: str
    source: str
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    title: Optional[str] = None
    licence: Optional[str] = None
    pdf: Optional[str] = None


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
    """One piece of text in a figure, in image pixels."""

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
        """Whether a word's centre lies inside this box."""
        return self.x0 <= word.cx <= self.x1 and self.y0 <= word.cy <= self.y1


@dataclass
class Annotation:
    """What one plaque image shows, and how that was decided.

    :param panel: the panel letter read nearest the image, or ``None``.
    :param label_text: strategy 1 -- the text around the image.
    :param legend_text: strategy 2 -- the legend's sentence for the panel.
    :param condition: the proposed condition.
    :param source: ``'label+legend'``, ``'label'``, ``'legend'``, ``'manual'``
        or ``'position'``.
    :param strength: ``'strong'`` when both readings agree, ``'medium'`` for
        one reading, ``'weak'`` for position only, ``'manual'`` when a person
        wrote it.
    :param conflict: set by a reviewer who finds the two readings disagree.
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
    approved: Optional[bool] = None


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


def resolve_paper(ref: Any, *, get: Optional[Callable] = None) -> Paper:
    """Look a reference up in Europe PMC, or describe a local PDF.

    :param ref: anything :func:`parse_reference` accepts.
    :param get: ``fn(url, params=...) -> response``; defaults to requests.
    :returns: the paper. A reference Europe PMC does not know comes back with
        only the identifier filled in, which still lets a PDF be measured.
    """
    parsed = parse_reference(ref)
    kind, value = parsed["kind"], parsed["value"]
    if kind == "pdf":
        digest = sha256_bytes(Path(value).read_bytes())
        return Paper(key=f"pdf:{digest}", source="pdf", pdf=str(value))
    query = {"doi": f'DOI:"{value}"', "pmcid": f"PMCID:{value}",
             "pmid": f"EXT_ID:{value} AND SRC:MED"}[kind]
    fetch = get or _get
    response = fetch(f"{EPMC}/search", params={
        "query": query, "format": "json", "resultType": "core", "pageSize": 1})
    hits = []
    if getattr(response, "status_code", 200) == 200:
        hits = (response.json().get("resultList", {}) or {}).get("result", [])
    if not hits:
        return Paper(key=value, source="europepmc", **{kind: value})
    hit = hits[0]
    doi = hit.get("doi") or (value if kind == "doi" else None)
    pmcid = hit.get("pmcid") or (value if kind == "pmcid" else None)
    return Paper(key=doi or pmcid or value, source="europepmc", doi=doi,
                 pmcid=pmcid, pmid=hit.get("pmid"), title=hit.get("title"),
                 licence=(hit.get("license") or "").strip().lower() or None)


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


def figures_from_pdf(pdf: Any, dest: Any, *, dpi: int = 200,
                     opener: Optional[Callable] = None) -> List[Figure]:
    """Every page of a PDF as a figure image, with its text layer attached.

    A page is rendered whole rather than its embedded images extracted,
    because a published figure is often assembled from many embedded images
    and vector labels, and the labels are what the condition is read from.
    The text layer comes back as :class:`Word` objects in the rendered
    image's pixels, so no OCR is needed for a PDF that has one.

    :param pdf: the PDF path.
    :param dest: directory for the page images.
    :param dpi: render resolution.
    :param opener: ``fn(path) -> pdfplumber-like document``; defaults to
        :func:`pdfplumber.open`.
    :returns: one figure per page, legend attached from the text when the
        page names a figure.
    :raises ImportError: when pdfplumber is not installed.
    """
    if opener is None:
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "Reading a PDF needs 'pdfplumber'. Install it with:\n  "
                + INSTALL_HINT) from exc
        opener = pdfplumber.open
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    figures: List[Figure] = []
    with opener(str(pdf)) as document:
        pages = list(document.pages)
        texts = [page.extract_text() or "" for page in pages]
        legends = legends_from_text("\n\n".join(texts))
        for number, page in enumerate(pages, start=1):
            target = dest / f"page_{number:03d}.png"
            page.to_image(resolution=dpi).save(str(target))
            words = [Word(w["text"], w["x0"] * scale, w["top"] * scale,
                          w["x1"] * scale, w["bottom"] * scale)
                     for w in page.extract_words()]
            named = [m.group(1) for m in _FIGURE_START.finditer(texts[number - 1])]
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

#: The figure reader's own environment (item 469): ultralytics and RapidOCR
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

    The maintainer, 2026-09-21: the text "seems to pick up the text correctly
    but it is not annotating correctly allways so whatever settings you can
    add there please do". These are the knobs the reading has; the defaults
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
        value = settings.get(key)
        if value in (None, ""):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default
    def patterns(value):
        if isinstance(value, str):
            return tuple(p.strip() for p in value.split(",") if p.strip())
        return tuple(str(p) for p in value)
    def sides(value):
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
    return out


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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_key TEXT PRIMARY KEY, source TEXT, doi TEXT, pmcid TEXT, pmid TEXT,
    title TEXT, licence TEXT, pdf TEXT, added REAL);
CREATE TABLE IF NOT EXISTS figures (
    figure_sha256 TEXT PRIMARY KEY, paper_key TEXT, label TEXT, caption TEXT,
    legend_source TEXT, path TEXT, width INTEGER, height INTEGER);
CREATE TABLE IF NOT EXISTS regions (
    region_id INTEGER PRIMARY KEY AUTOINCREMENT, figure_sha256 TEXT,
    paper_key TEXT, x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    detector_confidence REAL, found_at_sizes TEXT, panel TEXT, panel_row INTEGER,
    panel_column INTEGER, near_text TEXT, label_text TEXT, legend_text TEXT,
    condition TEXT, condition_source TEXT, strength TEXT, conflict INTEGER,
    approved INTEGER, crop_path TEXT, plaque_count INTEGER, has_ruler INTEGER,
    px_per_mm REAL, detector TEXT, segmenter TEXT, imgsz TEXT);
CREATE TABLE IF NOT EXISTS plaques (
    region_id INTEGER, label INTEGER, area_px INTEGER, area_mm2 REAL,
    area_vs_panel_median REAL);
"""


def open_database(path: Any) -> sqlite3.Connection:
    """Open (and create) the results database.

    :param path: the ``.db`` file.
    :returns: an open connection.
    """
    connection = sqlite3.connect(str(path))
    connection.executescript(_SCHEMA)
    return connection


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
        masks = model.eval(crop)[0]
        return np.asarray(masks)
    return segment


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
        them a ruler; None keeps every area in pixels and ratios.
    :param ask_legend: ``fn(figure, annotations) -> legend or None``, called
        when a panel letter was read but no legend could be fetched.
        Defaults to :func:`console_legend_prompt`.
    :param review: ``fn(figure, annotations) -> annotations``, called per
        figure when ``confirm_each``. Defaults to :func:`console_review`.
    :param read_text: ``fn(image path) -> [Word]``; defaults to
        :func:`read_words`.
    :param detect: passed to :func:`find_plaque_regions`.
    :param segment: ``fn(crop) -> labels``; defaults to a Cellpose model
        loaded from ``segmenter``.
    :param get: HTTP getter for Europe PMC.
    :param pdf_opener: passed to :func:`figures_from_pdf`.
    :returns: a summary: papers, figures, figures skipped as already
        measured, regions, plaques, and the database path.
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
    summary = {"papers": 0, "figures": 0, "skipped_figures": 0, "regions": 0,
               "plaques": 0, "database": str(database)}
    for reference in references:
        paper = resolve_paper(reference, get=get)
        folder = re.sub(r"[^A-Za-z0-9._-]+", "_", paper.key)[:120]
        if paper.source == "pdf":
            figures = figures_from_pdf(paper.pdf, dst / "figures" / folder,
                                       opener=pdf_opener)
        else:
            figures = fetch_figures(paper, dst / "figures" / folder, get=get)
        connection.execute(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?)",
            (paper.key, paper.source, paper.doi, paper.pmcid, paper.pmid,
             paper.title, paper.licence, paper.pdf, time.time()))
        summary["papers"] += 1
        for figure in figures:
            if connection.execute("SELECT 1 FROM figures WHERE figure_sha256=?",
                                  (figure.sha256,)).fetchone():
                summary["skipped_figures"] += 1
                continue
            _measure_figure(connection, paper, figure, dst / "crops" / folder,
                            detector_path=detector_path, detector_id=detector_id,
                            segmenter_id=segmenter_id, imgsz=imgsz,
                            confidence=confidence, confirm_each=confirm_each,
                            plate_format=plate_format, ask_legend=ask_legend,
                            review=review, read_text=read_text, detect=detect,
                            segment=segment, summary=summary)
            connection.commit()
    connection.close()
    return summary


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
    summary["figures"] += 1
    regions = find_plaque_regions(image, kw["detector_path"], imgsz=kw["imgsz"],
                                  confidence=kw["confidence"], detect=kw["detect"])
    connection.execute(
        "INSERT OR REPLACE INTO figures VALUES (?,?,?,?,?,?,?,?)",
        (figure.sha256, paper.key, figure.label, figure.caption,
         figure.legend_source, str(figure.path), int(image.shape[1]),
         int(image.shape[0])))
    if not regions:
        return
    options = kw.get("text_options") or DEFAULT_TEXT_OPTIONS
    words = figure.words or kw["read_text"](figure.path)
    if not figure.words and kw["read_text"] is read_words and options.reread:
        words = reread_around(image, regions, words,
                              scale=int(options.reread_scale))
    annotations = annotate_regions(regions, words, caption=figure.caption,
                                   figure_label=figure.label, options=options)
    if any(a.panel for a in annotations) and not figure.caption:
        pasted = kw["ask_legend"](figure, annotations)
        if pasted:
            figure.caption, figure.legend_source = pasted, "pasted"
            annotations = annotate_regions(regions, words, caption=pasted,
                                           figure_label=figure.label,
                                           options=options)
            connection.execute(
                "UPDATE figures SET caption=?, legend_source=? WHERE figure_sha256=?",
                (pasted, "pasted", figure.sha256))
        elif not kw["confirm_each"]:
            annotations = kw["review"](figure, annotations)
    if kw["confirm_each"]:
        annotations = kw["review"](figure, annotations)
    crops.mkdir(parents=True, exist_ok=True)
    measured: List[Tuple[Annotation, List[Dict[str, Any]], str, Optional[float]]] = []
    for index, a in enumerate(annotations, start=1):
        if a.approved is False:
            continue
        r = a.region
        crop = image[r.y0:r.y1, r.x0:r.x1]
        crop_path = crops / f"{figure.path.stem}_r{index:02d}.png"
        _save_png(crop, crop_path)
        scale = _ruler(r, kw["plate_format"])
        rows = measure_region(kw["segment"](crop), px_per_mm=scale)
        measured.append((a, rows, str(crop_path), scale))
    medians: Dict[Optional[str], float] = {}
    for panel in {a.panel for a, *_ in measured}:
        areas = [row["area_px"] for a, rows, *_ in measured if a.panel == panel
                 for row in rows]
        medians[panel] = float(np.median(areas)) if areas else 0.0
    for a, rows, crop_path, scale in measured:
        r = a.region
        cursor = connection.execute(
            "INSERT INTO regions (figure_sha256, paper_key, x0, y0, x1, y1, "
            "detector_confidence, found_at_sizes, panel, panel_row, panel_column, "
            "near_text, label_text, legend_text, condition, condition_source, "
            "strength, conflict, approved, crop_path, plaque_count, has_ruler, "
            "px_per_mm, detector, segmenter, imgsz) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (figure.sha256, paper.key, r.x0, r.y0, r.x1, r.y1, r.confidence,
             json.dumps(list(r.sizes)), a.panel, a.row, a.column,
             json.dumps(a.near), a.label_text, a.legend_text, a.condition,
             a.source, a.strength, int(a.conflict),
             None if a.approved is None else int(a.approved), crop_path,
             len(rows), int(scale is not None), scale, kw["detector_id"],
             kw["segmenter_id"], json.dumps(list(kw["imgsz"]))))
        median = medians.get(a.panel) or 0.0
        connection.executemany(
            "INSERT INTO plaques VALUES (?,?,?,?,?)",
            [(cursor.lastrowid, row["label"], row["area_px"], row["area_mm2"],
              row["area_px"] / median if median else None) for row in rows])
        summary["regions"] += 1
        summary["plaques"] += len(rows)


def _save_png(array: np.ndarray, path: Path) -> None:
    """Write an image crop.

    :param array: the pixels.
    :param path: the ``.png`` path.
    """
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(array)).save(path)


LEGENDS_FILE = "legends.csv"
ANNOTATIONS_FILE = "figure_annotations.csv"


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


def read_annotation_overrides(path: Any) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Conditions a person edited or approved, keyed by figure and image.

    :param path: a CSV with ``file``, ``region`` (1-based, in reading order),
        ``condition`` and ``approved`` columns -- what the Figure preview
        saves.
    :returns: ``{(stem, region): {'condition': str, 'approved': bool}}``.
    """
    import csv

    path = Path(path)
    if not path.is_file():
        return {}
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (Path(row["file"]).stem, int(row["region"]))
            except (KeyError, ValueError):
                continue
            approved = str(row.get("approved", "")).strip().lower()
            out[key] = {"condition": (row.get("condition") or "").strip(),
                        "approved": approved in ("1", "true", "yes", "ok")}
    return out


def write_annotation_overrides(path: Any, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Save reviewed conditions for :func:`read_annotation_overrides`.

    Rows for figures not in ``rows`` are kept, so reviewing one figure does
    not erase another's approvals.

    :param path: the CSV to write.
    :param rows: mappings with ``file``, ``region``, ``condition``, ``approved``.
    :returns: the path written.
    """
    import csv

    path = Path(path)
    kept = read_annotation_overrides(path)
    for row in rows:
        kept[(Path(str(row["file"])).stem, int(row["region"]))] = {
            "condition": str(row.get("condition", "")),
            "approved": bool(row.get("approved"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "region", "condition", "approved"])
        for (stem, region), value in sorted(kept.items()):
            writer.writerow([stem, region, value["condition"],
                             "true" if value["approved"] else "false"])
    return path


def apply_overrides(stem: str, annotations: List[Annotation],
                    overrides: Mapping[Tuple[str, int], Mapping[str, Any]], *,
                    confirm_each: bool = False) -> List[Annotation]:
    """Put a person's edits and approvals onto one figure's proposals.

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


def figures_in_folder(src: Any, legends: Optional[Mapping[str, str]] = None
                      ) -> List[Figure]:
    """Every figure image directly in ``src``, with its legend when known.

    :param src: the folder.
    :param legends: ``{stem: legend}``.
    :returns: the figures, by file name.
    """
    legends = legends or {}
    out = []
    for path in sorted(Path(src).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        caption = legends.get(path.stem, "")
        out.append(Figure(path=path, label=path.stem, caption=caption,
                          legend_source="file" if caption else "none",
                          sha256=sha256_bytes(path.read_bytes())))
    return out


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
    ``<src>/legends.csv``), and a person's edits and approvals from
    ``annotations`` (default ``<src>/figure_annotations.csv``), which the
    Figure preview writes. With ``confirm_each`` on, only approved images are
    measured and the rest are counted as waiting.

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
    paper = Paper(key=f"folder:{src.resolve()}", source="folder")
    connection.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?)",
                       (paper.key, "folder", None, None, None, src.name, None,
                        None, time.time()))
    summary = {"papers": 1, "figures": 0, "skipped_figures": 0, "regions": 0,
               "plaques": 0, "database": str(database)}
    for figure in figures_in_folder(src, legend_map):
        connection.execute("DELETE FROM plaques WHERE region_id IN (SELECT region_id "
                           "FROM regions WHERE figure_sha256=?)", (figure.sha256,))
        connection.execute("DELETE FROM regions WHERE figure_sha256=?", (figure.sha256,))
        _measure_figure(connection, paper, figure, dst / "crops",
                        detector_path=detector_path, detector_id=detector_id,
                        segmenter_id=segmenter_id, imgsz=imgsz,
                        confidence=confidence, confirm_each=True,
                        plate_format=plate_format,
                        ask_legend=lambda *_a: None, review=review,
                        read_text=read_text or read_words, detect=detect,
                        segment=segment, summary=summary,
                        text_options=text_options)
        connection.commit()
    connection.close()
    summary["awaiting_approval"] = waiting["n"]
    return summary


def fetch_paper_to_folder(reference: Any, dest: Any, *,
                          get: Optional[Callable] = None,
                          pdf_opener: Optional[Callable] = None) -> Dict[str, Any]:
    """Put a paper's figures in a folder Plaque Assay's Figure mode can read.

    The maintainer's 424 request: "if the figure pannel letter is detected,
    there should be a mechanism to auto gather the figure ledgend". A DOI,
    PMID or PMC id is fetched from Europe PMC with its JATS legends; a PDF is
    rendered page by page with the legends found in its text. Either way the
    images land in ``dest`` and each figure's legend in ``dest/legends.csv``,
    which Figure mode and :func:`measure_figure_folder` read -- so a paper
    becomes an ordinary folder of figures, annotated with its own legends.

    :param reference: a DOI, PMID, PMC id or PDF path.
    :param dest: the folder to fill; created if missing.
    :param get: HTTP getter for Europe PMC.
    :param pdf_opener: passed to :func:`figures_from_pdf`.
    :returns: ``{'folder', 'paper', 'figures', 'with_legend', 'licence'}``.
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
    (dest / "paper.json").write_text(json.dumps(asdict(paper), indent=2),
                                     encoding="utf-8")
    return {"folder": str(dest), "paper": paper.key, "figures": len(figures),
            "with_legend": len(rows), "licence": paper.licence}


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
