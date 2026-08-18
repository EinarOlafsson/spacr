"""What spaCR already knows about one gene, ready to put on a tile.

Instruction 121 asks that clicking a gene in the interactive regression show
"all the information on that gene". :mod:`spacr.gene_tile` answers the half
that can be WRONG -- which gene a clicked guide actually names, and whether
that mapping is ambiguous. This module answers the other half: given a gene,
what does spaCR hold about it.

IT HOLDS NOTHING ITSELF. Every value here comes out of :mod:`spacr.annotation`
-- the same join that writes the annotated exports -- so the tile and the CSV
a reader opens beside it cannot disagree about GRA14's compartment. This
module reads no file, parses no gene id and owns no table; it groups
:func:`spacr.annotation.annotate`'s 23 columns into the order a human reads
them and adds the per-segment DeepTMHMM coordinates that ``annotate``
deliberately leaves out of a coefficient export.

Four rules, each of them a failure this project has already had:

1.  ONE PARSE. ``TGGT1_224750``, ``gene_fraction:gene[224750]`` and the guide
    ``224750_2`` are all gene ``224750``, and the function that says so is
    :func:`spacr.annotation.gene_number`. A second copy of that rule is how
    the volcano and the hit list start naming different genes.

2.  A GAP IS SAID OUT LOUD. A gene with no annotation row gets
    :attr:`GeneFacts.reason` -- "no row in the bundled Toxoplasma annotation"
    -- and NOT a block of empty fields, which reads as "measured, found
    nothing" rather than "not available here". :attr:`GeneFacts.known` is the
    flag a caller greys a control on.

3.  A GROUP IS DERIVED FROM THE TABLE, NOT LISTED HERE. The fitness and
    expression rows are every ``fit_``/``expr_`` column
    :func:`spacr.annotation.columns` reports, so an eighth published screen
    added to ``phenotype.csv`` appears on the tile without this file being
    touched, and a column this module has never heard of lands in "other
    annotation" instead of being silently dropped.

4.  NOTHING HERE IS FOR THE GUI THREAD TO DISCOVER. The first call reads five
    bundled CSVs (360 ms) and the first :attr:`GeneFacts.segments` reads
    DeepTMHMM's 8,140 rows. :func:`warm` does both off the GUI thread, and
    takes the screen's own terms while it is there: `annotate` re-checks all
    five right-hand keys for uniqueness on every call, so one gene costs
    20 ms and four hundred cost 21. A warmed click is a dict lookup, 0.02 ms.

Public API::

    from spacr import gene_facts

    known = gene_facts.facts("fraction:grna[239740_3]")
    known.known                 # True
    known.value("gene_name")    # 'GRA14'
    known.sections()            # (('identity', (('gene name', 'GRA14'), ...
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

from . import annotation

__all__ = [
    "GROUPS",
    "OTHER",
    "GeneFacts",
    "Segment",
    "available",
    "clear_cache",
    "facts",
    "facts_for",
    "unavailable_reason",
    "warm",
]

#: The groups whose columns are named one by one, in reading order. Identity
#: first, because a user who clicked a dot wants to know what they clicked
#: before they want a phenotype score.
#:
#: ``fit_`` and ``expr_`` are deliberately NOT here -- see rule 3 in the
#: module docstring. Everything else the join can add is, and a column that
#: matches neither a name here nor a prefix falls into :data:`OTHER` rather
#: than off the tile.
GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("identity", ("gene_name", "product_description", "uniprot")),
    ("membrane topology", ("topology", "signal_peptide",
                           "signal_peptide_length", "transmembrane",
                           "n_transmembrane")),
    ("compartment", ("hyperlopit",)),
)

#: ``(heading, prefix)`` for the groups whose columns are whatever the
#: bundled table has. See rule 3.
_PREFIXED: Tuple[Tuple[str, str], ...] = (
    ("CRISPR fitness screens", "fit_"),
    ("expression", "expr_"),
)

#: Heading for a column this module has never heard of. It exists so that a
#: source added to :data:`spacr.annotation.SOURCES` cannot go unshown.
OTHER = "other annotation"

#: Human labels for the columns whose names are not already readable. A
#: column missing from here is labelled by its own name with the prefix
#: dropped and the underscores opened out, which is why an unknown ``fit_``
#: screen still reads as "in vivo lung".
_LABELS: Dict[str, str] = {
    "gene_name": "gene name",
    "product_description": "product",
    "uniprot": "UniProt",
    "topology": "DeepTMHMM class",
    "signal_peptide": "signal peptide",
    "signal_peptide_length": "signal peptide length",
    "transmembrane": "transmembrane",
    "n_transmembrane": "transmembrane helices",
    "hyperlopit": "compartment (hyperLOPIT/TAGM)",
    # The seven screens bundled today, respelled but NOT interpreted: "PE"
    # stays "PE" because expanding an abbreviation the source file does not
    # expand is how a caption ends up asserting an experiment nobody ran.
    "fit_invitro_hff": "in vitro (HFF)",
    "fit_invivo_PE": "in vivo (PE)",
    "fit_invivo_lung": "in vivo (lung)",
    "fit_invivo_liver": "in vivo (liver)",
    "fit_invivo_spleen": "in vivo (spleen)",
    "fit_naive_bmdm": "naive BMDM",
    "fit_ifng": "IFN-γ",
    "expr_tachyzoite": "tachyzoite",
    "expr_bradyzoite": "tissue cyst (bradyzoite)",
    "expr_ees1": "enteroepithelial stage 1",
    "expr_ees2": "enteroepithelial stage 2",
    "expr_ees3": "enteroepithelial stage 3",
    "expr_ees4": "enteroepithelial stage 4",
    "expr_ees5": "enteroepithelial stage 5",
}

#: How many transmembrane segments to look for. The bundled run tops out at
#: 24; the scan stops at the first column the file does not have, and this is
#: only the bound that keeps a malformed header from running away.
_MAX_SEGMENTS = 64

#: Every way the bundled tables spell an empty cell, lowercased.
_EMPTY = ("", "nan", "none", "n/a", "na", "null", "<na>", "-")

#: Records already built, keyed by gene number. Not an optimisation for its
#: own sake: `annotate` re-verifies the uniqueness of all five right-hand
#: keys on every call because the merge is declared many_to_one, so ONE gene
#: costs 20 ms and FOUR HUNDRED cost 21. Warming a whole screen at once is
#: therefore free, and it is what turns a click from 20 ms into 0.02.
_CACHE: Dict[str, "GeneFacts"] = {}

#: Genes held before the cache is dropped wholesale. A screen is ~400 genes
#: and the whole annotation is 8,800, so this only ever bites on a session
#: that has loaded twenty different screens -- and then it costs one 21 ms
#: rejoin, not a wrong answer.
_CACHE_MAX = 20000


def _label(column: str) -> str:
    """The human label for an annotation column."""
    if column in _LABELS:
        return _LABELS[column]
    text = column
    for _heading, prefix in _PREFIXED:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.replace("_", " ")


def _plain(value: Any) -> Any:
    """A cell as a plain Python scalar.

    ``numpy.bool_`` is NOT a ``bool`` and ``numpy.float64`` is not an ``int``,
    so a value read straight out of a merged frame falls through every
    ``isinstance`` test below and prints as its repr. Normalising once here is
    why :func:`_show` can be three branches instead of nine.
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (AttributeError, ValueError):
            return value
    return value


def _is_gap(value: Any) -> bool:
    """Is this cell empty in every way a bundled table spells empty?

    ``False`` and ``0.0`` are NOT gaps: "this protein has no signal peptide"
    and "this gene scored zero in the lung" are both answers, and a tile that
    dropped them would turn a measurement into a silence.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _EMPTY
    return value != value


def _show(column: str, value: Any) -> str:
    """One annotation cell as the string the tile prints."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        # `n_transmembrane` and `signal_peptide_length` arrive as floats
        # because their column has blanks in it. "7.0 helices" is a pandas
        # detail leaking onto a figure caption.
        return f"{int(value):d}"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value).strip()


@dataclass(frozen=True)
class Segment:
    """One DeepTMHMM segment of a protein, with its residue coordinates.

    :param kind: ``signal peptide`` or ``transmembrane``.
    :param index: 1-based position among the segments of that kind; ``1`` for
        a signal peptide, of which there is at most one.
    :param start: first residue, 1-based and inclusive, as DeepTMHMM reports.
    :param end: last residue, inclusive.
    :param length: residues spanned, as the source file records it rather
        than recomputed -- a disagreement between the two is a fact about the
        run and is not this module's to hide.
    """

    kind: str
    index: int
    start: int
    end: int
    length: int

    @property
    def label(self) -> str:
        """``signal peptide span`` or ``TM 3`` -- the tile's left column.

        "span", not "signal peptide": the summary row two lines above is
        already labelled "signal peptide", and two rows with one label reads
        as the panel having printed the same field twice.
        """
        if self.kind == "signal peptide":
            return "signal peptide span"
        return f"TM {self.index}"

    @property
    def text(self) -> str:
        """``residues 62-78 (17 aa)``."""
        return f"residues {self.start}–{self.end} ({self.length} aa)"


@dataclass(frozen=True)
class GeneFacts:
    """Everything the bundled annotation holds about one gene.

    :param gene: the bare gene number, or ``""`` when the caller's value named
        no gene at all.
    :param values: the annotation columns that had something in them, keyed by
        the column name :func:`spacr.annotation.columns` uses. A column with a
        gap is ABSENT here rather than present and empty.
    :param segments: the DeepTMHMM signal peptide and transmembrane segments,
        in residue order. Empty for a soluble protein, and empty for a gene
        DeepTMHMM never saw -- :attr:`reason` is what tells those apart.
    :param reason: why there is nothing, in a sentence, or ``""`` when there
        is something. Never blank and empty at the same time: an empty panel
        reads as a bug, "no row in the bundled annotation" reads as an answer.
    """

    gene: str = ""
    values: Dict[str, Any] = field(default_factory=dict)
    segments: Tuple[Segment, ...] = ()
    reason: str = ""

    @property
    def known(self) -> bool:
        """Did the annotation hold anything at all about this gene?

        The flag a caller greys a control on -- instruction 106: a control
        that cannot do anything says why rather than sitting there inert.
        """
        return bool(self.values)

    def value(self, column: str, default: Any = None) -> Any:
        """One annotation column, or ``default`` when it had a gap."""
        return self.values.get(column, default)

    def sections(self) -> Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...]:
        """The facts as ``((heading, ((label, value), ...)), ...)``.

        THE ONE ORDERING, so the Qt tile and the text form cannot drift into
        presenting the same record two ways -- the same shape
        :meth:`spacr.gene_tile.GeneTile.sections` returns, so one renderer
        lays both halves of a tile out with one loop.

        A group with nothing in it is not emitted. That is rule 2: a heading
        over a blank space is a claim that something was looked for and not
        found, which is a different statement from "not bundled here".
        """
        out: List[Tuple[str, Tuple[Tuple[str, str], ...]]] = []
        for heading, columns in _layout():
            rows = [(_label(c), _show(c, self.values[c]))
                    for c in columns if c in self.values]
            if heading == "membrane topology" and self.segments:
                rows.extend((s.label, s.text) for s in self.segments)
            if rows:
                out.append((heading, tuple(rows)))
        return tuple(out)

    def to_text(self) -> str:
        """The facts as plain text -- what a test reads and a log records."""
        lines: List[str] = []
        for heading, rows in self.sections():
            if lines:
                lines.append("")
            lines.append(heading.upper())
            lines.extend(f"  {label}: {value}" for label, value in rows)
        if not lines:
            lines.append(self.reason or "Nothing is known about this gene.")
        return "\n".join(lines)

    def to_html(self) -> str:
        """The facts as HTML, for a Qt rich-text view.

        Matches :meth:`spacr.gene_tile.GeneTile.to_html` mark for mark, so the
        two halves of one tile do not read as two different documents.
        """
        if not self.known:
            said = self.reason or "Nothing is known about this gene."
            return f"<p style='color:#999'>{html.escape(said)}</p>"
        parts: List[str] = []
        for heading, rows in self.sections():
            parts.append(f"<h4>{html.escape(heading)}</h4><table>")
            for label, value in rows:
                parts.append(
                    f"<tr><td style='color:#888'>{html.escape(label)}</td>"
                    f"<td>{html.escape(value)}</td></tr>")
            parts.append("</table>")
        return "".join(parts)


@lru_cache(maxsize=1)
def _layout() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """The groups and their columns, for the annotation actually installed.

    Built from :func:`spacr.annotation.columns` so a column that is not
    bundled never becomes a heading, and a column this module does not name
    still gets one.
    """
    present = list(annotation.columns())
    placed = set()
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for heading, columns in GROUPS:
        chosen = tuple(c for c in columns if c in present)
        placed.update(chosen)
        out.append((heading, chosen))
    for heading, prefix in _PREFIXED:
        chosen = tuple(c for c in present if c.startswith(prefix))
        placed.update(chosen)
        out.append((heading, chosen))
    rest = tuple(c for c in present if c not in placed)
    if rest:
        out.append((OTHER, rest))
    return tuple(out)


@lru_cache(maxsize=1)
def _segment_index() -> Dict[str, Tuple[Segment, ...]]:
    """``{gene number: segments}``, built once from the DeepTMHMM table.

    A DERIVED INDEX, NOT THE TABLE. ``supplementary()`` hands back 8,140 rows
    of 82 columns -- 6.8 MB held for the sake of a handful of integers per
    click -- and it re-reads its 1.1 MB CSV on every call, because it is the
    function that WRITES the supplementary file and handing a caller a cached
    frame would let that caller mutate the cache. So the read happens once
    here and what is kept is ~8,000 short tuples.

    Column by column rather than row by row: one filtered pass per segment
    slot instead of 8,140 x 72 cell lookups.
    """
    frame = annotation.supplementary()
    if frame is None or "gene_nr" not in getattr(frame, "columns", ()):
        return {}

    found: Dict[str, List[Segment]] = {}

    def collect(kind: str, index: int, start: str, end: str, length: str
                ) -> None:
        if start not in frame.columns or end not in frame.columns:
            return
        keep = ["gene_nr", start, end]
        if length in frame.columns:
            keep.append(length)
        rows = frame.loc[frame[start].notna() & frame[end].notna(), keep]
        for cells in rows.itertuples(index=False, name=None):
            gene = annotation.gene_number(cells[0])
            if gene is None:
                continue
            first, last = int(cells[1]), int(cells[2])
            span = _plain(cells[3]) if len(cells) > 3 else None
            found.setdefault(gene, []).append(Segment(
                kind, index, first, last,
                int(span) if isinstance(span, (int, float))
                and not _is_gap(span) else last - first + 1))

    collect("signal peptide", 1, "sp_start", "sp_end", "sp_length")
    for n in range(1, _MAX_SEGMENTS + 1):
        if f"tm_{n}_start" not in frame.columns:
            break
        collect("transmembrane", n, f"tm_{n}_start", f"tm_{n}_end",
                f"tm_{n}_length")

    return {gene: tuple(sorted(segments, key=lambda s: s.start))
            for gene, segments in found.items()}


def available() -> Tuple[str, ...]:
    """Every annotation column this module can show, in reading order.

    Empty when nothing is bundled, which is what :func:`unavailable_reason`
    turns into a sentence.
    """
    return tuple(c for _heading, columns in _layout() for c in columns)


def unavailable_reason() -> str:
    """Why there is no annotation on this install, or ``""``.

    A sentence rather than a flag, because it is shown to the user: a panel
    that merely refused would be indistinguishable from one that broke.
    """
    if available():
        return ""
    return ("The bundled Toxoplasma annotation tables are not installed with "
            "this copy of spaCR, so nothing can be shown about a gene.")


def facts_for(values: Iterable[Any]) -> Dict[str, GeneFacts]:
    """The facts for several genes at once, keyed by gene number.

    :param values: anything :func:`spacr.annotation.gene_number` accepts --
        design terms, accessions, guide ids, bare numbers, in any mixture.
    :returns: one entry per DISTINCT gene named, in the order first named. A
        value naming no gene contributes nothing, so an empty result means
        nothing in ``values`` was a gene.

    ONE JOIN FOR THE WHOLE SET. The ambiguous case is three genes at once and
    each of them wants the same 23 columns; three separate merges cost three
    times as much and, worse, would be three chances for the key to be built
    differently.
    """
    import pandas as pd

    genes: List[str] = []
    for value in values:
        gene = annotation.gene_number(value)
        if gene is not None and gene not in genes:
            genes.append(gene)
    if not genes:
        return {}

    reason = unavailable_reason()
    if reason:
        return {gene: GeneFacts(gene=gene, reason=reason) for gene in genes}

    wanted = [gene for gene in genes if gene not in _CACHE]
    if wanted:
        # `gene`, NOT `gene_nr`: `annotate` merges every source with
        # right_on="gene_nr", so a left column of that name collides, pandas
        # renames both halves, and the tidy-up then drops a column that is no
        # longer there. Reported for a fix; avoided here so the tile does not
        # have to wait for one.
        joined = annotation.annotate(pd.DataFrame({"gene": wanted}),
                                     key_column="gene", quiet=True)
        columns = [c for c in available() if c in joined.columns]
        segments = _segment_index()
        if len(_CACHE) + len(wanted) > _CACHE_MAX:
            _CACHE.clear()
        for position, gene in enumerate(wanted):
            row = joined.iloc[position]
            held = {c: _plain(row[c]) for c in columns if not _is_gap(row[c])}
            _CACHE[gene] = GeneFacts(
                gene=gene, values=held, segments=segments.get(gene, ()),
                reason="" if held else
                (f"Gene {gene} has no row in the bundled Toxoplasma "
                 "annotation. That is a fact about the annotation table, not "
                 "about the screen: the coefficient above stands on its own."))
    return {gene: _CACHE[gene] for gene in genes}


def facts(value: Any) -> GeneFacts:
    """Everything the bundled annotation holds about the gene ``value`` names.

    :param value: a design term, an accession, a guide id or a bare gene
        number -- every spelling :func:`spacr.annotation.gene_number` takes.
    :returns: a :class:`GeneFacts`, ALWAYS. A term that names no gene comes
        back with :attr:`GeneFacts.known` false and a :attr:`GeneFacts.reason`
        saying so, because a caller that has to tell ``None`` apart from an
        empty record is a caller that will forget to.
    """
    gene = annotation.gene_number(value)
    if gene is None:
        return GeneFacts(reason=(
            f"{value!r} does not name a Toxoplasma gene, so there is no "
            "annotation to show for it."))
    return facts_for([gene]).get(gene, GeneFacts(gene=gene))


def warm(values: Iterable[Any] = ()) -> Tuple[str, ...]:
    """Load every table this module reads. CALL THIS OFF THE GUI THREAD.

    :param values: the terms a user might click -- a whole results table's
        ``feature`` column is the intended argument. Every gene among them is
        joined in ONE pass and cached, which is why passing four hundred
        costs the same 21 ms as passing one.
    :returns: the columns that came out available, so a caller can tell "the
        tables loaded" from "the tables are not installed" without a second
        call.

    The whole point of the function, measured: cold, the first click pays
    360 ms of CSV reading plus a 20 ms join, inside a mouse press. A plot
    that freezes for a third of a second when clicked reads as broken. Warmed
    with the screen's own terms, a click is a dict lookup -- 0.02 ms.
    """
    columns = available()
    if not columns:
        return columns
    _segment_index()
    facts_for(values)
    return columns


def clear_cache() -> None:
    """Forget the derived indices. For tests, and for a reinstall mid-session.

    :func:`spacr.annotation.clear_cache` is called too: the layout here is
    derived from that module's tables, and dropping one without the other
    leaves a layout describing columns that are no longer loaded.
    """
    _CACHE.clear()
    _layout.cache_clear()
    _segment_index.cache_clear()
    annotation.clear_cache()
