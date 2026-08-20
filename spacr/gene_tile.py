"""Resolve regression features to gene identities and screen statistics.

The :func:`gene_tile` function combines a regression feature with optional
coefficient, gRNA-reference, annotation, and localisation data. It returns a
:class:`GeneTile` that can be rendered by the GUI or used independently.

Guide identifiers are resolved with :func:`spacr.hits.gene_of`. When the same
protospacer occurs under more than one gene in the supplied gRNA reference,
all candidate genes are retained and the mapping is marked as ambiguous.
Missing references or annotations are described in the returned record rather
than represented by an empty panel.

Bundled reference tables are read locally and cached. External database URLs
are constructed for the user to open; this module does not fetch them.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .hits import gene_of, guide_of, tested_family

__all__ = [
    "BUNDLED",
    "GeneCandidate",
    "GuideRow",
    "GeneTile",
    "Reference",
    "METADATA_FIELDS",
    "TOXODB_GENE_URL",
    "gene_tile",
    "is_toxoplasma_gene_id",
    "toxodb_url",
]

#: Sentinel meaning "the file spaCR ships". Passing ``None`` for a source
#: instead means "do not consult that source at all", which is what a caller
#: working on a screen that is not Toxoplasma wants.
BUNDLED = "<bundled>"

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resources", "data")

#: The gRNA reference: ``name,sequence``, name shaped ``TGGT1_<gene>_<n>``.
BUNDLED_BARCODES = os.path.join(_DATA, "barcodes_grna.csv")
#: The curated gene annotation, keyed ``Gene ID`` = ``TGME49_<gene>``.
BUNDLED_METADATA = os.path.join(_DATA, "toxoplasma_metadata.csv")
#: TAGM/LOPIT subcellular localisation, keyed ``gene_nr`` = ``<gene>``.
BUNDLED_LOCALISATION = os.path.join(_DATA, "lopit.csv")

#: A Toxoplasma gene id as every spaCR table carries it once
#: :func:`spacr.hits.gene_of` has stripped the strain prefix: five or six
#: digits, optionally with a paralog letter (``201180A``). Checked against the
#: real screen: all 389 gene ids in ``plate1_dv`` match, and nothing else does.
#: This is what stops a tile offering a ToxoDB link for an id from a screen of
#: some other organism.
_TOXO_GENE = re.compile(r"^\d{5,6}[A-Za-z]?$")

#: A gRNA reference name, ``TGGT1_239740_3``: strain, gene, guide number.
_REFERENCE_NAME = re.compile(r"^(TG[A-Za-z0-9]{2,6})_(.+?)_(\d+)$")

#: ``TGGT1_239740_3`` or ``TGME49_239740`` written out in full.
_ACCESSION = re.compile(r"^(TG[A-Za-z0-9]{2,6})_(\d{5,6}[A-Za-z]?)(?:_(\d+))?$")

#: The gene id the shipped reference and every spaCR default use for the
#: non-targeting control block. It is not a gene and has no ToxoDB record.
CONTROL_GENE = "000000"

#: VEuPathDB gene record page. The per-site webapp path (``/toxo/`` here) is a
#: table rather than a rule across VEuPathDB, so it is written out for the one
#: site this module knows about rather than derived.
TOXODB_GENE_URL = "https://toxodb.org/toxo/app/record/gene/{accession}"

#: UniProt record and search URL templates.
#:
#: Accessions are resolved through the bundled ME49 reference-proteome table
#: (``UP000001529``), keyed by the ToxoDB gene-number suffix shared across
#: strains. Restricting records to the reference proteome avoids choosing an
#: alternate unassembled-proteome entry for the same gene. Genes absent from
#: the mapping use the search template and are labelled as searches rather
#: than being presented as verified record links.
UNIPROT_SEARCH_URL = (
    "https://www.uniprot.org/uniprotkb?query={query}")
UNIPROT_RECORD_URL = "https://www.uniprot.org/uniprotkb/{accession}/entry"

#: The bundled ToxoDB-gene-number -> UniProt-accession table.
UNIPROT_TABLE = os.path.join(_DATA, "uniprot.csv")


@lru_cache(maxsize=1)
def uniprot_accessions() -> Dict[str, str]:
    """``{gene number: accession}`` from the bundled table.

    Cached: it is read to build one line of a gene tile, which happens every
    time a point is clicked.

    Returns an empty mapping rather than raising when the file is absent --
    a screen of another organism has no reason to carry it, and a gene tile
    without a UniProt line is still a gene tile.
    """
    out: Dict[str, str] = {}
    try:
        import csv

        with open(UNIPROT_TABLE, newline="") as handle:
            for row in csv.DictReader(handle):
                gene = str(row.get("gene_nr", "")).strip()
                accession = str(row.get("uniprot", "")).strip()
                if gene and accession:
                    # Keyed on the bare number, and the file is written with
                    # leading zeros preserved (039160), so both spellings
                    # resolve.
                    out.setdefault(gene, accession)
                    out.setdefault(gene.lstrip("0") or gene, accession)
    except Exception:                                            # noqa: BLE001
        return {}
    return out


#: Annotation columns that would hold a UniProt accession if one were there.
UNIPROT_COLUMNS = ("UniProt ID", "UniProt", "UniProtKB", "uniprot_id",
                   "UniProt Accession")

#: Metadata columns worth a line of their own on the tile, in reading order,
#: as ``(column, label)``. Everything else the annotation file carries is kept
#: in :attr:`GeneCandidate.annotation` for a caller that wants it.
METADATA_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("Product Description", "product"),
    ("Gene Name or Symbol", "symbol"),
    ("Protein Length", "protein length"),
    ("# TM Domains", "TM domains"),
    ("SignalP Peptide", "signal peptide"),
    ("Curated GO Components", "GO component (curated)"),
    ("Curated GO Functions", "GO function (curated)"),
    ("Curated GO Processes", "GO process (curated)"),
    ("Computed GO Components", "GO component (computed)"),
    ("Computed GO Functions", "GO function (computed)"),
    ("Computed GO Processes", "GO process (computed)"),
    ("EC numbers", "EC number"),
)

#: Column carrying the product description, used for the tile's subtitle.
_PRODUCT = "Product Description"
#: Column carrying the gene symbol, used for the tile's title when there is one.
_SYMBOL = "Gene Name or Symbol"


# Canonical English used when a structured gene record is rendered in Qt.
# These strings live in the headless data model rather than in literal Qt
# calls, so the runtime catalog builder imports this inventory explicitly.
_GENE_TILE_UI_SOURCES = frozenset({
    "non-targeting control",
    "{feature} (model covariate)",
    "unrecognised term",
    "one protospacer, {count} genes — the effect cannot be assigned to any "
    "one of them",
    "no product description",
    "no product description · {accession}",
    "gene id",
    "localisation (TAGM/LOPIT)",
    "also known as",
    "not resolved",
    "identity",
    "identity — {name}",
    "the gene the counts were attributed to",
    "effect (coefficient)",
    "p-value",
    "q-value",
    "uncorrected p-value (q_value column)",
    "condition",
    "gene-level effect",
    "gene-level p",
    "guides fitted",
    "guides agreeing in sign",
    "this screen",
    "opposes the gene effect",
    "clicked",
    "the guides behind it",
    "protospacer",
    "sequence",
    "what could not be resolved",
    "notes",
    "external records",
    "Ambiguous mapping",
    "Click a point in the volcano, or a row in the results table, to see what "
    "that gene is.",
    "Could not build a tile for {feature}. The plot is unaffected; see the "
    "log for details.",
    "ambiguous mapping — every gene it could be is listed above",
    "no annotation row for gene {gene} in the metadata file, so there is no "
    "product description or symbol for it here.",
    "no results table was supplied, so this screen's own numbers for the "
    "gene are not on this tile.",
    "{feature!r} is not a row in the results table, so its effect and p-value "
    "are not shown.",
    "{feature!r} is a model covariate — the intercept or a plate row/column "
    "term — not a gene, so there is nothing to look up.",
    "{feature!r} is not in a form spaCR recognises as a model term, a gRNA "
    "accession or a gene id, so no gene could be resolved.",
    "guide {identifier} is a non-targeting control, not a gene. It has no "
    "gene record and no ToxoDB entry; it is in the fit so the screen has a "
    "null to measure the real guides against.",
    "the other {count} control guides are listed with it: they are the null "
    "this screen measures every real guide against.",
    "no gRNA reference was supplied, so it is not known whether this guide's "
    "protospacer is unique to one gene.",
    "guide {guide} is not in the gRNA reference, so its protospacer could not "
    "be checked for other genes carrying it. References may exclude guides "
    "whose protospacers map to multiple genes; verify the guide identifier "
    "and reference version.",
    "protospacer {protospacer} appears in the gRNA reference under {count} "
    "gene names ({names}). Reads from this guide cannot be told apart between "
    "them, so the effect below belongs to all {count} equally — the "
    "regression attributed it to {gene} because that is the name the counting "
    "pipeline saw last, which is a bookkeeping fact, not a result.",
    "gene id {gene!r} is not shaped like a Toxoplasma accession, so no ToxoDB "
    "record is offered for it. If this is a screen of another organism, spaCR "
    "has no annotation for it.",
    "no strain prefix was known for gene {gene}, so no ToxoDB link could be "
    "built. Supply the gRNA reference the screen was counted against and the "
    "accession will resolve.",
}) | frozenset(label for _column, label in METADATA_FIELDS)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class _TranslatableText(str):
    """English text retaining the template and values used to build it."""

    def __new__(cls, source: str, values: Dict[str, Any]):
        rendered = str(source).format(**values)
        instance = super().__new__(cls, rendered)
        instance.source = str(source)
        instance.values = dict(values)
        return instance


def _message(source: str, **values: Any) -> _TranslatableText:
    """Format English text while retaining its localization template."""
    return _TranslatableText(source, values)


def _translated(
    value: Any,
    translator: Optional[Callable[..., str]] = None,
) -> str:
    """Translate a structured message, leaving scientific data untouched."""
    if translator is not None and isinstance(value, _TranslatableText):
        return str(translator(value.source, **value.values))
    return str(value)


def _label(
    source: str,
    translator: Optional[Callable[..., str]] = None,
) -> str:
    """Translate a presentation label when a translator is supplied."""
    return str(translator(source)) if translator is not None else str(source)

def _clean(value: Any) -> str:
    """A metadata cell as a printable string, or ``""`` for every kind of gap.

    A curated export spells "nothing here" at least four ways — an empty cell,
    the literal ``N/A``, ``null``, and whatever pandas turned a blank into —
    and a tile that printed ``nan`` under "product" would be reporting the
    export's punctuation as a fact about the gene.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "n/a", "na", "null", "<na>", "-"):
        return ""
    return text


def _text(value: Any) -> str:
    """A cell as a string, treating ONLY a real gap as a gap.

    :func:`_clean` reads the literal ``"none"`` as an empty cell, which is
    right for a curated export writing ``None`` under "product" and wrong for
    ``multiple_testing_method``, where ``none`` is the answer: it means no
    correction was applied and the ``q_value`` column stores uncorrected
    p-values.
    Passing that through :func:`_clean` silently deleted the one warning a
    reader of those q-values needs.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "<na>") else text


def _number(value: Any) -> float:
    """A results cell as a float, ``nan`` for anything that is not one."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def is_toxoplasma_gene_id(gene: Any) -> bool:
    """Is ``gene`` shaped like a Toxoplasma gene id spaCR would recognise?

    :param gene: a bare gene id (``239740``), an accession (``TGGT1_239740``),
        or anything at all.
    :returns: ``True`` for a Toxoplasma-shaped id. The point of the check is
        the negative: an id from a screen of some other organism must not be
        handed a ToxoDB link, because a link to a record that does not exist
        is worse than no link.
    """
    text = _clean(gene)
    if not text:
        return False
    match = _ACCESSION.match(text)
    if match:
        return True
    return bool(_TOXO_GENE.match(text))


def uniprot_reference(accession: str, annotation=None):
    """A UniProt link for this gene: the record when known, else a search.

    :param accession: the gene accession, e.g. ``TGGT1_239740``.
    :param annotation: the gene's annotation row, if there is one.
    :returns: ``(label, url, is_record)``, or ``None`` when there is nothing
        useful to offer.

    THE DISTINCTION MATTERS AND IS CARRIED IN THE LABEL. A record link says
    "this is the protein"; a search link says "here is the question". Only
    one of those is a claim, and only one of them can be wrong in a way the
    reader cannot see -- a fabricated record URL that happens to resolve
    opens a real page for a different protein and looks exactly like a
    correct link.
    """
    if not accession:
        return None
    if annotation is not None:
        for column in UNIPROT_COLUMNS:
            try:
                value = annotation.get(column)
            except AttributeError:
                value = None
            text = "" if value is None else str(value).strip()
            if text and text.lower() not in ("nan", "none", ""):
                return (f"UniProt {text}",
                        UNIPROT_RECORD_URL.format(accession=text), True)
    # THE BUNDLED MAPPING. Keyed on the gene NUMBER, so a full ToxoDB
    # accession (`TGGT1_224750`) and a bare number (`224750`) both resolve --
    # the tile is handed either depending on where the click came from.
    number = str(accession).strip()
    if "_" in number:
        number = number.rsplit("_", 1)[-1]
    known = uniprot_accessions().get(number) or uniprot_accessions().get(
        number.lstrip("0") or number)
    if known:
        return (f"UniProt {known}",
                UNIPROT_RECORD_URL.format(accession=known), True)

    return (f"UniProt search: {accession}",
            UNIPROT_SEARCH_URL.format(query=accession), False)


def toxodb_url(accession: str) -> str:
    """The ToxoDB gene record page for a full accession. Never fetched."""
    return TOXODB_GENE_URL.format(accession=accession)


@lru_cache(maxsize=50000)
def _gene_of_cached(feature: str) -> Optional[str]:
    """:func:`spacr.hits.gene_of`, memoised on the term string.

    Finding a gene's family means asking which gene every row names, and a
    screen has thousands of rows. Calling the real rule and caching the
    answer keeps ONE definition of guide -> gene — a vectorised re-statement
    of the same regex here would be the second copy this module exists to
    avoid — while making the second click on a screen free. Term strings are
    a small fixed set per results file, so the cache converges immediately.
    """
    return gene_of(feature)


@lru_cache(maxsize=8)
def _read_csv(path: str, stamp: Tuple[float, int]) -> pd.DataFrame:
    """Read a reference CSV, cached on (path, mtime, size).

    Clicking a point must not re-read an 8,800-row annotation file. The stamp
    is in the key so a file edited under a running window is picked up rather
    than served stale from the cache.
    """
    del stamp
    return pd.read_csv(path)


#: Built indices, keyed ``(builder name, file token)``. Reading the annotation
#: CSV is cached by :func:`_read_csv`, but turning 8,800 rows into a lookup was
#: still being redone on every click and cost ~90 ms of it — which is a click
#: that feels broken. Only file-backed sources are cached; a frame handed in
#: directly belongs to the caller and is indexed fresh, because nothing here
#: can tell whether they mutated it.
_INDEX_CACHE: Dict[Tuple[str, str], Any] = {}
_INDEX_CACHE_MAX = 24


def _load(source: Any, default_path: str) -> Tuple[Optional[pd.DataFrame],
                                                   Optional[str]]:
    """Resolve a source argument to ``(frame, token)``.

    :param source: :data:`BUNDLED` for the shipped file, ``None`` to skip the
        source entirely, a path, or a frame the caller already has.
    :returns: the frame, or ``None`` for "do not consult"; and a cache token
        identifying the file it came from, or ``None`` when there is no file
        to key a cache on.
    """
    if source is None:
        return None, None
    if isinstance(source, pd.DataFrame):
        return source, None
    path = default_path if source is BUNDLED else os.path.abspath(
        os.path.expanduser(os.fspath(source)))
    if not os.path.isfile(path):
        return None, None
    try:
        stat = os.stat(path)
        frame = _read_csv(path, (stat.st_mtime, stat.st_size))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None, None
    return frame, f"{path}|{stat.st_mtime}|{stat.st_size}"


def _indexed(name: str, builder, frame: Optional[pd.DataFrame],
             token: Optional[str]):
    """``builder(frame)``, memoised when the frame came from a named file."""
    if token is None:
        return builder(frame)
    key = (name, token)
    if key not in _INDEX_CACHE:
        if len(_INDEX_CACHE) >= _INDEX_CACHE_MAX:
            _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = builder(frame)
    return _INDEX_CACHE[key]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reference:
    """An external record for this gene: a label and a URL to open.

    :param label: what to show, e.g. ``ToxoDB TGGT1_239740``.
    :param url: where it goes. Built by string formatting and NEVER fetched —
        see the module docstring.
    """

    label: str
    url: str


@dataclass(frozen=True)
class GeneCandidate:
    """One gene the clicked feature could be naming.

    Usually there is exactly one. There is more than one when the guide's
    protospacer sits in more than one gene, and then every one of them is here
    with whatever is known about it, because picking one would be inventing a
    result.

    :param gene: the bare gene id, as every spaCR table keys it (``411710``).
    :param accession: the library's own accession (``TGGT1_411710``), when the
        gRNA reference names a strain; ``""`` when nothing said which strain.
    :param annotation_id: the ``Gene ID`` of the annotation row that matched
        (``TGME49_239740``), or ``""`` when none did.
    :param product: the product description, or ``""``.
    :param symbol: the gene symbol, or ``""``.
    :param localisation: TAGM/LOPIT subcellular localisation, or ``""``.
    :param aliases: every other name this gene is known by here.
    :param fields: the annotation columns worth showing, as
        ``((label, value), ...)`` in reading order, gaps already dropped.
    :param annotation: the whole matched annotation row, for a caller that
        wants a column this module does not show.
    :param reported: ``True`` for the gene the regression itself attributed
        the reads to — the one a naive resolver would have picked alone.
    :param notes: what could not be resolved ABOUT THIS GENE, in sentences.
    """

    gene: str
    accession: str = ""
    annotation_id: str = ""
    product: str = ""
    symbol: str = ""
    localisation: str = ""
    aliases: Tuple[str, ...] = ()
    fields: Tuple[Tuple[str, str], ...] = ()
    annotation: Dict[str, Any] = field(default_factory=dict)
    reported: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """The most human name available: the symbol, else the accession."""
        return self.symbol or self.accession or self.gene

    @property
    def references(self) -> Tuple[Reference, ...]:
        """External records for this gene. Built, never fetched."""
        out: List[Reference] = []
        for accession in (self.accession, self.annotation_id):
            if accession and all(accession != r.label.split()[-1] for r in out):
                out.append(Reference(f"ToxoDB {accession}",
                                     toxodb_url(accession)))
        found = uniprot_reference(self.accession or self.annotation_id,
                                  self.annotation)
        if found:
            label, url, _is_record = found
            out.append(Reference(label, url))
        return tuple(out)


@dataclass(frozen=True)
class GuideRow:
    """One guide of this gene, and how it behaved in this screen.

    :param guide: the guide id as the results table carries it (``239740_3``).
    :param feature: the model term it came from.
    :param effect: the fitted coefficient.
    :param p_value: its p-value.
    :param q_value: its q-value, as the run reported one.
    :param n_obs: the ``n_grna`` count behind it.
    :param agrees: whether it pushes the same way as the gene-level effect;
        ``None`` when there is no gene-level effect to compare against, which
        is different from disagreeing and is shown differently.
    :param clicked: ``True`` for the guide the user actually clicked.
    """

    guide: str
    feature: str = ""
    effect: float = float("nan")
    p_value: float = float("nan")
    q_value: float = float("nan")
    n_obs: float = float("nan")
    agrees: Optional[bool] = None
    clicked: bool = False

    @property
    def direction(self) -> str:
        """``up``, ``down``, ``flat`` or ``""`` when there is no effect."""
        if not math.isfinite(self.effect):
            return ""
        if self.effect == 0.0:
            return "flat"
        return "up" if self.effect > 0 else "down"


@dataclass(frozen=True)
class GeneTile:
    """Everything spaCR can say about one clicked point, in reading order.

    Identity, then this screen's numbers, then what spaCR already knew, then a
    link out — because a user who clicked a point wants to know what they
    clicked before they want a bibliography.

    :param feature: the string that was clicked, verbatim.
    :param kind: what the feature turned out to be — ``guide``, ``gene``,
        ``control``, ``nuisance`` or ``unresolved``.
    :param guide: the guide id, for a guide term; ``""`` otherwise.
    :param gene: the gene id the regression attributed the reads to.
    :param candidates: every gene the feature could name, the reported one
        first. Length > 1 means :attr:`ambiguous`.
    :param ambiguous: ``True`` when the guide's protospacer sits in more than
        one gene, so the effect cannot be assigned to any one of them.
    :param ambiguity: the sentence explaining that, or ``""``.
    :param protospacer: the guide's sequence, when the reference had it.
    :param effect: the clicked term's coefficient.
    :param p_value: its p-value.
    :param q_value: Its q-value, or the uncorrected p-value stored in the
        ``q_value`` column when :attr:`correction` is ``"none"``.
    :param correction: The multiple-testing method recorded by the run. When
        this is ``"none"``, the tile labels the stored ``q_value`` explicitly
        as an uncorrected p-value.
    :param condition: ``control`` / ``pc`` / ``nc`` / ``other``, as fitted.
    :param n_obs: the observation count the results row carried.
    :param n_obs_column: which column that count came from, ``n_grna`` or
        ``n_gene``. Named rather than relabelled: the two count different
        things and only the run knows which one a row carries.
    :param gene_effect: the gene-level coefficient for this gene.
    :param gene_p_value: its p-value.
    :param gene_q_value: its q-value.
    :param guides: every guide of this gene that was fitted, and how it moved.
    :param n_agree: how many of them agree in sign with the gene effect.
    :param unresolved: what could not be worked out, in sentences. NEVER
        empty when something is missing — this is the field that stops the
        tile reading as a bug.
    :param notes: everything else worth saying that is not a failure.
    """

    feature: str
    kind: str = "unresolved"
    guide: str = ""
    gene: str = ""
    candidates: Tuple[GeneCandidate, ...] = ()
    ambiguous: bool = False
    ambiguity: str = ""
    protospacer: str = ""
    effect: float = float("nan")
    p_value: float = float("nan")
    q_value: float = float("nan")
    correction: str = ""
    condition: str = ""
    n_obs: float = float("nan")
    n_obs_column: str = ""
    gene_effect: float = float("nan")
    gene_p_value: float = float("nan")
    gene_q_value: float = float("nan")
    guides: Tuple[GuideRow, ...] = ()
    n_agree: int = 0
    unresolved: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    # -- identity ---------------------------------------------------------

    @property
    def resolved(self) -> bool:
        """Did this land on at least one gene?"""
        return bool(self.candidates)

    @property
    def title(self) -> str:
        """The tile's first line: a name a human recognises where one exists.

        For an ambiguous guide the title names all of the genes, joined by
        ``/``. That is deliberately awkward to read: the mapping IS awkward,
        and a title that showed one of three would be a lie that fits.
        """
        if self.ambiguous:
            return " / ".join(c.name for c in self.candidates)
        if self.candidates:
            return self.candidates[0].name
        if self.kind == "control":
            return _message("non-targeting control")
        if self.kind == "nuisance":
            return _message("{feature} (model covariate)",
                            feature=self.feature)
        return _clean(self.feature) or _message("unrecognised term")

    @property
    def subtitle(self) -> str:
        """The line under the title: what the thing IS, in words."""
        if self.ambiguous:
            return _message(
                "one protospacer, {count} genes — the effect cannot be "
                "assigned to any one of them",
                count=len(self.candidates),
            )
        if self.candidates:
            candidate = self.candidates[0]
            product = candidate.product
            if not product and candidate.accession:
                return _message(
                    "no product description · {accession}",
                    accession=candidate.accession,
                )
            if not product:
                return _message("no product description")
            if candidate.symbol and candidate.accession:
                return f"{product} · {candidate.accession}"
            return product
        if self.unresolved:
            return self.unresolved[0]
        return ""

    @property
    def n_guides(self) -> int:
        """How many of this gene's guides the results table fitted."""
        return len(self.guides)

    @property
    def references(self) -> Tuple[Reference, ...]:
        """Every external record, across every candidate gene."""
        out: List[Reference] = []
        seen = set()
        for candidate in self.candidates:
            for reference in candidate.references:
                if reference.url not in seen:
                    seen.add(reference.url)
                    out.append(reference)
        return tuple(out)

    # -- rendering --------------------------------------------------------

    def sections(
        self,
        translator: Optional[Callable[..., str]] = None,
    ) -> Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...]:
        """The tile as ``((heading, ((label, value), ...)), ...)``.

        The one ordering, defined once, so the Qt tile, the text form and the
        HTML form cannot drift into presenting the same record three ways.
        ``translator`` localizes presentation labels and structured messages;
        identifiers, measurements, annotations, and URLs remain unchanged.
        """
        out: List[Tuple[str, Tuple[Tuple[str, str], ...]]] = []

        for candidate in self.candidates:
            rows: List[Tuple[str, str]] = [(
                _label("gene id", translator),
                candidate.accession or candidate.gene,
            )]
            rows.extend(
                (_label(label, translator), value)
                for label, value in candidate.fields
            )
            if candidate.localisation:
                rows.append((_label("localisation (TAGM/LOPIT)", translator),
                             candidate.localisation))
            if candidate.aliases:
                rows.append((_label("also known as", translator),
                             ", ".join(candidate.aliases)))
            for note in candidate.notes:
                rows.append((_label("not resolved", translator),
                             _translated(note, translator)))
            heading = _label("identity", translator)
            if self.ambiguous:
                heading = _translated(
                    _message("identity — {name}", name=candidate.name),
                    translator,
                )
                if candidate.reported:
                    heading += (
                        " ("
                        + _label(
                            "the gene the counts were attributed to",
                            translator,
                        )
                        + ")"
                    )
            out.append((heading, tuple(rows)))

        screen: List[Tuple[str, str]] = []
        if math.isfinite(self.effect):
            screen.append((_label("effect (coefficient)", translator),
                           f"{self.effect:.4g}"))
        if math.isfinite(self.p_value):
            screen.append((_label("p-value", translator),
                           f"{self.p_value:.3g}"))
        if math.isfinite(self.q_value):
            label = "q-value"
            if self.correction and self.correction.lower() == "none":
                label = "uncorrected p-value (q_value column)"
            screen.append((_label(label, translator), f"{self.q_value:.3g}"))
        if self.condition:
            screen.append((_label("condition", translator), self.condition))
        if math.isfinite(self.n_obs) and self.n_obs_column:
            screen.append((self.n_obs_column, f"{self.n_obs:g}"))
        if math.isfinite(self.gene_effect):
            screen.append((_label("gene-level effect", translator),
                           f"{self.gene_effect:.4g}"))
        if math.isfinite(self.gene_p_value):
            screen.append((_label("gene-level p", translator),
                           f"{self.gene_p_value:.3g}"))
        if self.guides:
            screen.append((_label("guides fitted", translator),
                           str(self.n_guides)))
            if math.isfinite(self.gene_effect):
                screen.append((_label("guides agreeing in sign", translator),
                               f"{self.n_agree} of {self.n_guides}"))
        if screen:
            out.append((_label("this screen", translator), tuple(screen)))

        if self.guides:
            rows = []
            for row in self.guides:
                text = f"{row.effect:+.4g}" if math.isfinite(row.effect) else "—"
                if math.isfinite(row.p_value):
                    text += f"   p {row.p_value:.3g}"
                if row.agrees is False:
                    text += (
                        "   ("
                        + _label("opposes the gene effect", translator)
                        + ")"
                    )
                if row.clicked:
                    text += "   <- " + _label("clicked", translator)
                rows.append((row.guide, text))
            out.append((_label("the guides behind it", translator),
                        tuple(rows)))

        if self.protospacer:
            out.append((_label("protospacer", translator),
                        ((_label("sequence", translator), self.protospacer),)))

        # A note already showing as the subtitle is not repeated: the tile
        # would otherwise print its own headline twice, which reads as two
        # different problems rather than one.
        rest = tuple(n for n in self.unresolved if n != self.subtitle)
        if rest:
            out.append((_label("what could not be resolved", translator),
                        tuple(("", _translated(note, translator))
                              for note in rest)))
        if self.notes:
            out.append((_label("notes", translator),
                        tuple(("", _translated(note, translator))
                              for note in self.notes)))
        if self.references:
            out.append((_label("external records", translator),
                        tuple((r.label, r.url) for r in self.references)))
        return tuple(out)

    def to_text(
        self,
        translator: Optional[Callable[..., str]] = None,
    ) -> str:
        """The tile as plain text — what a test reads and a log records."""
        lines = [_translated(self.title, translator)]
        if self.subtitle:
            lines.append(_translated(self.subtitle, translator))
        for heading, rows in self.sections(translator):
            lines.append("")
            lines.append(heading.upper())
            for label, value in rows:
                lines.append(f"  {label}: {value}" if label else f"  {value}")
        return "\n".join(lines)

    def to_html(
        self,
        translator: Optional[Callable[..., str]] = None,
    ) -> str:
        """The tile as HTML, for a Qt rich-text view."""
        import html as _html

        title = _translated(self.title, translator)
        parts = [f"<h2 style='margin-bottom:2px'>{_html.escape(title)}</h2>"]
        if self.subtitle:
            parts.append("<p style='margin-top:0;color:#999'>"
                         f"{_html.escape(_translated(self.subtitle, translator))}</p>")
        if self.ambiguity:
            parts.append(
                "<p style='color:#c9a227'><b>"
                f"{_html.escape(_label('Ambiguous mapping', translator))}"
                "</b> — "
                f"{_html.escape(_translated(self.ambiguity, translator))}</p>"
            )
        external_heading = _label("external records", translator)
        for heading, rows in self.sections(translator):
            if heading == external_heading:
                links = " · ".join(
                    f"<a href='{_html.escape(url)}'>{_html.escape(label)}</a>"
                    for label, url in rows)
                parts.append(f"<p>{links}</p>")
                continue
            parts.append(f"<h4>{_html.escape(heading)}</h4><table>")
            for label, value in rows:
                parts.append(
                    f"<tr><td style='color:#888'>{_html.escape(label)}</td>"
                    f"<td>{_html.escape(value)}</td></tr>")
            parts.append("</table>")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _parse(feature: Any) -> Tuple[str, str, str]:
    """Split a clicked string into ``(kind, gene, guide)``.

    ``kind`` is ``gene``, ``guide``, ``nuisance`` or ``unresolved``. The gene
    comes from :func:`spacr.hits.gene_of` and the guide from
    :func:`spacr.hits.guide_of` whenever the string is a model term, so this
    module cannot disagree with the hit list about which gene a dot is. Bare
    ids typed or pasted in (``TGGT1_239740_3``, ``239740``) are also accepted,
    because the table is not the only thing that can hand this module a string.
    """
    text = _clean(feature)
    if not text:
        return "unresolved", "", ""

    if "[" in text:
        if not bool(tested_family([text])[0]):
            return "nuisance", "", ""
        gene = gene_of(text) or ""
        guide = guide_of(text) or ""
        if not gene:
            return "unresolved", "", ""
        # THE TERM SAYS WHICH IT IS, not the shape of the token inside it.
        # `guide_of` returns the WHOLE bracketed token, so on a numeric screen
        # a gene term's token has no underscore and guide == "" -- which is
        # how "guide if there is a guide" happened to work. It stops working
        # the moment the id itself contains an underscore:
        # `gene_fraction:gene[TGGT1_231640]` is a GENE term whose token is
        # `TGGT1_231640`, and it was reported as a guide of gene TGGT1.
        if "gene_fraction:gene[" in text or ":gene[" in text:
            return "gene", gene, ""
        return ("guide", gene, guide) if guide and guide != gene \
            else ("gene", gene, "")

    if not bool(tested_family([text])[0]):
        return "nuisance", "", ""

    accession = _ACCESSION.match(text)
    if accession:
        gene = accession.group(2)
        number = accession.group(3)
        return ("guide", gene, f"{gene}_{number}") if number else ("gene", gene, "")

    bare = re.match(r"^(\d{5,6}[A-Za-z]?)(?:_(\d+))?$", text)
    if bare:
        gene = bare.group(1)
        return ("guide", gene, text) if bare.group(2) else ("gene", gene, "")
    return "unresolved", "", ""


def _reference_index(barcodes: Optional[pd.DataFrame]
                     ) -> Tuple[Dict[str, Tuple[str, str]],
                                Dict[str, List[Tuple[str, str]]],
                                Dict[str, str]]:
    """Index the gRNA reference three ways.

    :returns: ``(by_guide, by_sequence, strain_of_gene)`` where ``by_guide``
        maps ``239740_3`` to ``(accession, sequence)``, ``by_sequence`` maps a
        protospacer to every ``(gene, accession)`` carrying it — the map that
        makes ambiguity visible — and ``strain_of_gene`` remembers which
        strain prefix the reference wrote a gene under, so the tile can name
        the accession the library actually used instead of guessing one.
    """
    by_guide: Dict[str, Tuple[str, str]] = {}
    by_sequence: Dict[str, List[Tuple[str, str]]] = {}
    strain: Dict[str, str] = {}
    if barcodes is None or barcodes.empty:
        return by_guide, by_sequence, strain
    if "name" not in barcodes.columns or "sequence" not in barcodes.columns:
        return by_guide, by_sequence, strain
    for name, sequence in zip(barcodes["name"], barcodes["sequence"]):
        text = _clean(name)
        seq = _clean(sequence).upper()
        if not text:
            continue
        match = _REFERENCE_NAME.match(text)
        if match:
            prefix, gene, number = match.group(1), match.group(2), match.group(3)
            guide = f"{gene}_{number}"
            accession = f"{prefix}_{gene}"
        else:
            gene = text.split("_")[0]
            guide = text
            accession = ""
        by_guide.setdefault(guide, (accession, seq))
        if accession:
            strain.setdefault(gene, accession.split("_")[0])
        if seq:
            entry = (gene, accession)
            bucket = by_sequence.setdefault(seq, [])
            if entry not in bucket:
                bucket.append(entry)
    return by_guide, by_sequence, strain


def _annotation_index(metadata: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """``{gene id: annotation row}`` from a curated export.

    The join is on the numeric suffix, which is the cross-strain convention
    VEuPathDB uses for syntenic orthologs — ``TGGT1_239740`` and
    ``TGME49_239740`` are the same gene — and it is the same key
    :func:`spacr.hits.load_gene_metadata` builds, so the tile and the hit list
    annotate a gene identically. An export listing a gene once per transcript
    would otherwise be read here as several genes; the FIRST row wins and the
    rest are dropped, exactly as the hit list does it.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if metadata is None or metadata.empty:
        return out
    key = next((c for c in ("Gene ID", "gene_id", "gene") if c in metadata.columns),
               None)
    if key is None:
        return out
    for row in metadata.to_dict("records"):
        identifier = _clean(row.get(key))
        gene = identifier.split("_")[1] if "_" in identifier else identifier
        if gene:
            out.setdefault(gene, row)
    return out


def _localisation_index(frame: Optional[pd.DataFrame]) -> Dict[str, str]:
    """``{gene id: TAGM location}`` from the bundled LOPIT table."""
    out: Dict[str, str] = {}
    if frame is None or frame.empty:
        return out
    key = "gene_nr" if "gene_nr" in frame.columns else None
    value = next((c for c in ("tagm_location", "location")
                  if c in frame.columns), None)
    if key is None or value is None:
        return out
    for gene, location in zip(frame[key], frame[value]):
        text, place = _clean(gene), _clean(location)
        if text and place:
            out.setdefault(text, place)
    return out


def _candidate(gene: str, accession: str, annotations: Dict[str, Dict[str, Any]],
               localisation: Dict[str, str], *, reported: bool,
               metadata_named: bool) -> GeneCandidate:
    """Assemble one gene's identity from every annotation source."""
    row = annotations.get(gene)
    notes: List[str] = []
    fields: List[Tuple[str, str]] = []
    annotation_id = ""
    product = symbol = ""
    if row is None:
        if metadata_named and is_toxoplasma_gene_id(gene) and gene != CONTROL_GENE:
            notes.append(
                _message(
                    "no annotation row for gene {gene} in the metadata file, "
                    "so there is no product description or symbol for it "
                    "here.",
                    gene=gene,
                ))
    else:
        key = next((c for c in ("Gene ID", "gene_id", "gene") if c in row), "")
        annotation_id = _clean(row.get(key)) if key else ""
        product = _clean(row.get(_PRODUCT))
        symbol = _clean(row.get(_SYMBOL))
        for column, label in METADATA_FIELDS:
            text = _clean(row.get(column))
            if text:
                fields.append((label, text))

    place = localisation.get(gene, "")
    aliases = [a for a in (accession, annotation_id, symbol) if a]
    if accession and accession in aliases:
        aliases.remove(accession)

    return GeneCandidate(
        gene=gene, accession=accession, annotation_id=annotation_id,
        product=product, symbol=symbol, localisation=place,
        aliases=tuple(dict.fromkeys(aliases)), fields=tuple(fields),
        annotation=dict(row) if row is not None else {},
        reported=reported, notes=tuple(notes))


def _screen_numbers(results: Optional[pd.DataFrame], feature: str, gene: str,
                    guide: str) -> Dict[str, Any]:
    """This screen's own numbers for the clicked term and for its gene."""
    out: Dict[str, Any] = {
        "effect": float("nan"), "p_value": float("nan"),
        "q_value": float("nan"), "condition": "", "correction": "",
        "n_obs": float("nan"), "n_obs_column": "", "gene_effect": float("nan"),
        "gene_p_value": float("nan"), "gene_q_value": float("nan"),
        "guides": (), "n_agree": 0, "missing": [],
    }
    if results is None or not len(results) or "feature" not in results.columns:
        out["missing"].append(
            _message(
                "no results table was supplied, so this screen's own numbers "
                "for the gene are not on this tile."
            ))
        return out

    # Deliberately not `results.copy()`: a click must not duplicate the whole
    # coefficient table to read one row out of it. The feature column is taken
    # as strings once and used as the index for every lookup below.
    frame = results
    terms = frame["feature"].astype(str)
    clicked = frame[(terms == feature).to_numpy()]
    if clicked.empty:
        out["missing"].append(
            _message(
                "{feature!r} is not a row in the results table, so its effect "
                "and p-value are not shown.",
                feature=feature,
            ))
    else:
        row = clicked.iloc[0]
        out["effect"] = _number(row.get("coefficient"))
        out["p_value"] = _number(row.get("p_value"))
        out["q_value"] = _number(row.get("q_value"))
        out["condition"] = _clean(row.get("condition"))
        out["correction"] = _text(row.get("multiple_testing_method"))
        out["n_obs"] = _number(row.get("n_grna"))
        out["n_obs_column"] = "n_grna"
        if not math.isfinite(out["n_obs"]):
            out["n_obs"] = _number(row.get("n_gene"))
            out["n_obs_column"] = "n_gene" if math.isfinite(out["n_obs"]) else ""

    if not gene:
        return out

    in_family = [_gene_of_cached(f) == gene for f in terms]
    family = frame[in_family]
    family_terms = terms[in_family]
    is_gene_term = family_terms.str.startswith("gene_fraction").to_numpy()
    gene_rows = family[is_gene_term]
    if not gene_rows.empty:
        row = gene_rows.iloc[0]
        out["gene_effect"] = _number(row.get("coefficient"))
        out["gene_p_value"] = _number(row.get("p_value"))
        out["gene_q_value"] = _number(row.get("q_value"))
        if not out["correction"]:
            out["correction"] = _text(row.get("multiple_testing_method"))

    guide_rows = family[~is_gene_term]
    direction = (math.copysign(1.0, out["gene_effect"])
                 if math.isfinite(out["gene_effect"]) and out["gene_effect"] != 0
                 else None)
    rows: List[GuideRow] = []
    for _, row in guide_rows.iterrows():
        term = str(row.get("feature"))
        identifier = _clean(row.get("grna")) or guide_of(term) or term
        effect = _number(row.get("coefficient"))
        agrees: Optional[bool] = None
        if direction is not None and math.isfinite(effect):
            agrees = effect != 0.0 and math.copysign(1.0, effect) == direction
        rows.append(GuideRow(
            guide=identifier, feature=term, effect=effect,
            p_value=_number(row.get("p_value")),
            q_value=_number(row.get("q_value")),
            n_obs=_number(row.get("n_grna")),
            agrees=agrees, clicked=term == feature))
    rows.sort(key=lambda r: r.guide)
    out["guides"] = tuple(rows)
    out["n_agree"] = sum(1 for r in rows if r.agrees)
    return out


def gene_tile(feature: Any,
              results: Optional[pd.DataFrame] = None,
              *,
              barcodes: Any = BUNDLED,
              metadata: Any = BUNDLED,
              localisation: Any = BUNDLED) -> GeneTile:
    """Build a gene record for a regression feature.

    Parameters
    ----------
    feature : Any
        Model term such as ``fraction:grna[239740_3]`` or
        ``gene_fraction:gene[239740]``. Bare gRNA accessions and gene
        identifiers are also accepted.
    results : pandas.DataFrame or None, optional
        Regression coefficient table. When ``None``, identity and annotation
        are still resolved and the missing screen statistics are recorded in
        :attr:`GeneTile.unresolved`.
    barcodes : {BUNDLED, None}, path-like, or pandas.DataFrame, optional
        gRNA reference with ``name`` and ``sequence`` columns. Use
        :data:`BUNDLED` for the packaged reference or ``None`` to skip
        protospacer lookup. Ambiguous mappings can be detected only when this
        reference contains the guide.
    metadata : {BUNDLED, None}, path-like, or pandas.DataFrame, optional
        Gene annotation keyed by ``Gene ID``. Use ``None`` to omit annotation.
    localisation : {BUNDLED, None}, path-like, or pandas.DataFrame, optional
        TAGM/LOPIT localisation table keyed by ``gene_nr``. Use ``None`` to
        omit localisation data.

    Returns
    -------
    GeneTile
        Resolved identity, screen statistics, annotations, references, and
        explicit notes about missing or ambiguous data. Unrecognised features
        return an unresolved record rather than ``None``.

    Notes
    -----
    The function reads only local data. Values in
    :attr:`GeneTile.references` are URLs for the caller to open and are not
    fetched while building the record.
    """
    text = _clean(feature)
    kind, gene, guide = _parse(feature)

    reference, reference_token = _load(barcodes, BUNDLED_BARCODES)
    annotation_frame, annotation_token = _load(metadata, BUNDLED_METADATA)
    place_frame, place_token = _load(localisation, BUNDLED_LOCALISATION)

    annotations = _indexed("annotation", _annotation_index, annotation_frame,
                           annotation_token)
    places = _indexed("localisation", _localisation_index, place_frame,
                      place_token)
    by_guide, by_sequence, strain_of = _indexed(
        "reference", _reference_index, reference, reference_token)
    metadata_named = metadata is not None

    unresolved: List[str] = []
    notes: List[str] = []

    numbers = _screen_numbers(results, text, gene, guide)
    unresolved.extend(numbers.pop("missing"))

    # --- the control block, which is not a gene and must not pretend to be --
    is_control = (numbers["condition"].lower() == "control"
                  or (gene == CONTROL_GENE and kind in ("guide", "gene")))
    if kind == "nuisance":
        unresolved.append(
            _message(
                "{feature!r} is a model covariate — the intercept or a plate "
                "row/column term — not a gene, so there is nothing to look "
                "up.",
                feature=text,
            ))
        return GeneTile(feature=text, kind="nuisance",
                        unresolved=tuple(unresolved), **numbers)
    if kind == "unresolved":
        unresolved.append(
            _message(
                "{feature!r} is not in a form spaCR recognises as a model "
                "term, a gRNA accession or a gene id, so no gene could be "
                "resolved.",
                feature=text,
            ))
        return GeneTile(feature=text, kind="unresolved", guide=guide, gene=gene,
                        unresolved=tuple(unresolved), **numbers)
    if is_control:
        unresolved.append(
            _message(
                "guide {identifier} is a non-targeting control, not a gene. "
                "It has no gene record and no ToxoDB entry; it is in the fit "
                "so the screen has a null to measure the real guides against.",
                identifier=guide or gene,
            ))
        # The control block is fitted as if it were one gene, so a "gene-level
        # effect" and a sign agreement exist for it arithmetically. Neither
        # means anything — there is no gene for the guides to agree ABOUT —
        # and printing them would dress the null distribution up as a result.
        # The sibling guides stay: they ARE that null, and seeing the other 23
        # sit near zero is how a user reads one control that did not.
        numbers["gene_effect"] = float("nan")
        numbers["gene_p_value"] = float("nan")
        numbers["gene_q_value"] = float("nan")
        numbers["n_agree"] = 0
        numbers["guides"] = tuple(
            GuideRow(guide=r.guide, feature=r.feature, effect=r.effect,
                     p_value=r.p_value, q_value=r.q_value, n_obs=r.n_obs,
                     agrees=None, clicked=r.clicked)
            for r in numbers["guides"])
        notes.append(
            _message(
                "the other {count} control guides are listed with it: they "
                "are the null this screen measures every real guide against.",
                count=max(len(numbers["guides"]) - 1, 0),
            ))
        return GeneTile(feature=text, kind="control", guide=guide, gene=gene,
                        unresolved=tuple(unresolved), notes=tuple(notes),
                        **numbers)

    # --- which genes could this be? -----------------------------------------
    protospacer = ""
    genes: List[Tuple[str, str]] = []
    reported_accession = ""
    if strain_of.get(gene):
        reported_accession = f"{strain_of[gene]}_{gene}"

    if guide:
        entry = by_guide.get(guide)
        if entry is None:
            if reference is None:
                unresolved.append(
                    _message(
                        "no gRNA reference was supplied, so it is not known "
                        "whether this guide's protospacer is unique to one "
                        "gene."
                    ))
            else:
                unresolved.append(
                    _message(
                        "guide {guide} is not in the gRNA reference, so its "
                        "protospacer could not be checked for other genes "
                        "carrying it. References may exclude guides whose "
                        "protospacers map to multiple genes; verify the guide "
                        "identifier and reference version.",
                        guide=guide,
                    ))
        else:
            reported_accession = entry[0] or reported_accession
            protospacer = entry[1]
            for other_gene, other_accession in by_sequence.get(protospacer, []):
                genes.append((other_gene, other_accession))

    if not genes:
        genes = [(gene, reported_accession)]
    else:
        # The gene the counts were attributed to leads, then the rest in a
        # stable order, so two runs of the same click read the same.
        genes.sort(key=lambda item: (item[0] != gene, item[0]))
        if gene not in [g for g, _ in genes]:
            genes.insert(0, (gene, reported_accession))

    candidates = tuple(
        _candidate(g, a or (f"{strain_of[g]}_{g}" if strain_of.get(g) else ""),
                   annotations, places, reported=g == gene,
                   metadata_named=metadata_named)
        for g, a in genes)

    # A candidate nothing is known about, for an id that is not even shaped
    # like a Toxoplasma gene, is not a gene record — it is the clicked string
    # echoed back under a heading that says "identity". Drop it, so the tile
    # says plainly that it resolved nothing instead of implying it resolved
    # something empty.
    if not is_toxoplasma_gene_id(gene) and not any(
            c.annotation or c.accession or c.localisation for c in candidates):
        candidates = ()

    ambiguous = len(candidates) > 1
    ambiguity = ""
    if ambiguous:
        named = ", ".join(c.accession or c.gene for c in candidates)
        ambiguity = _message(
            "protospacer {protospacer} appears in the gRNA reference under "
            "{count} gene names ({names}). Reads from this guide cannot be "
            "told apart between them, so the effect below belongs to all "
            "{count} equally — the regression attributed it to {gene} "
            "because that is the name the counting pipeline saw last, which "
            "is a bookkeeping fact, not a result.",
            protospacer=protospacer,
            count=len(candidates),
            names=named,
            gene=gene,
        )

    if not is_toxoplasma_gene_id(gene):
        unresolved.append(
            _message(
                "gene id {gene!r} is not shaped like a Toxoplasma accession, "
                "so no ToxoDB record is offered for it. If this is a screen "
                "of another organism, spaCR has no annotation for it.",
                gene=gene,
            ))
    elif not any(c.references for c in candidates):
        notes.append(
            _message(
                "no strain prefix was known for gene {gene}, so no ToxoDB "
                "link could be built. Supply the gRNA reference the screen "
                "was counted against and the accession will resolve.",
                gene=gene,
            ))

    if not candidates:
        kind = "unresolved"

    return GeneTile(
        feature=text, kind=kind, guide=guide, gene=gene, candidates=candidates,
        ambiguous=ambiguous, ambiguity=ambiguity, protospacer=protospacer,
        unresolved=tuple(unresolved), notes=tuple(notes), **numbers)
