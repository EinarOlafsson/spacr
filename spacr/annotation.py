"""Join bundled *Toxoplasma* gene annotations to exported results.

The joined fields include gene names, predicted signal peptides and
transmembrane domains, published phenotype scores, stage-specific expression
and hyperLOPIT localisation.

All supported identifier forms are normalized to a bare gene number:
``TGGT1_224750``, ``TGME49_224750``, ``gene_fraction:gene[224750]`` and the
guide identifier ``224750_2`` resolve to gene ``224750`` through
:func:`gene_number`. Each source is reduced to one row per gene and merged
with ``many_to_one`` validation, preventing annotation duplicates from
multiplying result rows. A field unavailable in a source is omitted rather
than emitted as an all-missing column.

Bundled annotation sources:

    toxoplasma_metadata.csv  2.97 MB  gene name, product, expression
    deeptmhmm.csv            1.11 MB  signal peptide and transmembrane
    phenotype.csv            0.48 MB  the published CRISPR fitness screens
    lopit.csv                0.20 MB  hyperLOPIT/TAGM compartment
    uniprot.csv              0.13 MB  accession

``deeptmhmm.csv`` contains a DeepTMHMM analysis of 8,140 proteins. Summary
fields (``dtm_type``, ``n_tm`` and ``sp_length``) are joined by
:func:`annotate`; coordinates for as many as 24 transmembrane segments are
written separately by :func:`supplementary`. Per-segment sequence columns are
excluded because they are recoverable from the reference proteome.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

#: The first run of four or more digits in a name. Four, not six: ToxoDB gene
#: numbers are six digits today, but the rule has to reject the ``_2`` of a
#: guide id and the ``1`` of the ``TGGT1`` prefix, and a floor of four does
#: both without asserting a length the database never promised.
_DIGITS = re.compile(r"(\d{4,})")

#: Where the bundled tables live.
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resources", "data")


def gene_number(value) -> Optional[str]:
    """The bare gene number named by ``value``, or ``None``.

    :param value: feature, accession, guide, or design-term value to parse.

    Accepts every spelling this project has: a design term
    (``gene_fraction:gene[224750]``, ``fraction:grna[224750_2]``), a gene id
    in either strain (``TGGT1_224750``, ``TGME49_224750``), a split gene
    model (``TGME49_201180A``), a guide id (``224750_2``) or the bare number.

    :returns: the number as a string, or ``None`` for anything that names no
        gene -- ``Intercept``, ``rowID[T.r03]``, an empty cell, NaN.

    A SPLIT GENE MODEL COLLAPSES TO ITS PARENT. ``TGME49_201180A`` and
    ``TGME49_201180B`` are both ``201180``, because the screen's library
    targets the locus and has no way to tell the two models apart. It is why
    every source here is deduplicated before it is joined.
    """
    if value is None:
        return None
    text = str(value)
    if text in ("", "nan", "NaN", "None", "<NA>"):
        return None
    match = _DIGITS.search(text)
    return match.group(1) if match else None


def _read(filename: str):
    """The bundled CSV, or ``None`` with the reason printed.

    A missing or unreadable table is not fatal. The export is still an
    export; it simply carries fewer columns, and the console says which and
    why rather than leaving the reader to notice.
    """
    import pandas as pd

    path = os.path.join(_DATA, filename)
    if not os.path.isfile(path):
        print(f"Toxoplasma annotation: {filename} is not bundled with this "
              f"install, so its columns are left out of the export.")
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as error:                      # noqa: BLE001
        print(f"Toxoplasma annotation: {filename} could not be read "
              f"({error}), so its columns are left out of the export.")
        return None


def _keyed(frame, source_column: str, wanted: Sequence[Tuple[str, str]]):
    """``frame`` reduced to the key and ``wanted``, one row per gene.

    :param source_column: the column holding a gene id or number.
    :param wanted: ``(source name, exported name)`` pairs. A pair whose
        source column is absent is dropped, not filled with NaN.
    """
    if frame is None:
        return None
    pairs = [(src, out) for src, out in wanted if src in frame.columns]
    if not pairs or source_column not in frame.columns:
        return None
    out = frame[[source_column] + [src for src, _ in pairs]].copy()
    # A SCRATCH NAME, not "gene_nr" directly: three of the five bundled
    # tables already spell their key `gene_nr`, and writing the parsed key
    # over the source column meant the very next line dropped it.
    out["_key"] = out[source_column].map(gene_number)
    out = out.loc[out["_key"].notna()]
    out = out.drop(columns=[source_column])
    out = out.rename(columns=dict(pairs))
    out = out.rename(columns={"_key": "gene_nr"})
    # KEEP THE FIRST. The alternative -- letting duplicates through -- is a
    # row-multiplying join, and the alternative to that -- averaging text
    # columns -- is not defined. The tables that carry numbers are already
    # collapsed on disk; what reaches here is a handful of split models.
    return out.drop_duplicates(subset="gene_nr", keep="first")


@lru_cache(maxsize=1)
def _metadata():
    """Return bundled gene metadata keyed by ``gene_nr``, or ``None``."""
    return _keyed(_read("toxoplasma_metadata.csv"), "Gene ID", (
        ("Gene Name or Symbol", "gene_name"),
        ("Product Description", "product_description"),
        ("sense - Tachyzoites", "expr_tachyzoite"),
        ("sense - Tissue cysts", "expr_bradyzoite"),
        ("sense - EES1", "expr_ees1"),
        ("sense - EES2", "expr_ees2"),
        ("sense - EES3", "expr_ees3"),
        ("sense - EES4", "expr_ees4"),
        ("sense - EES5", "expr_ees5"),
    ))


@lru_cache(maxsize=1)
def _topology():
    """Signal peptide and transmembrane, from this project's DeepTMHMM run.

    ``dtm_type`` is one field carrying both answers -- GLOB, SP, TM, SP+TM,
    BETA -- so it is split into two booleans a reader can filter on, and the
    raw label is kept beside them for anyone who wants BETA.
    """
    frame = _keyed(_read("deeptmhmm.csv"), "identifier", (
        ("dtm_type", "topology"),
        ("n_tm", "n_transmembrane"),
        ("sp_length", "signal_peptide_length"),
    ))
    if frame is None:
        return None
    label = frame["topology"].astype("string").str.upper()
    frame["signal_peptide"] = label.isin(["SP", "SP+TM"])
    frame["transmembrane"] = label.isin(["TM", "SP+TM"])
    return frame[["gene_nr", "signal_peptide", "transmembrane",
                  "n_transmembrane", "signal_peptide_length", "topology"]]


@lru_cache(maxsize=1)
def _phenotype():
    """Return bundled ``fit_*`` phenotype scores keyed by ``gene_nr``."""
    frame = _read("phenotype.csv")
    if frame is None:
        return None
    columns = [c for c in frame.columns if c.startswith("fit_")]
    return _keyed(frame, "gene_nr", tuple((c, c) for c in columns))


@lru_cache(maxsize=1)
def _lopit():
    """Return bundled TAGM localisation as ``hyperlopit`` by ``gene_nr``."""
    return _keyed(_read("lopit.csv"), "gene_nr",
                  (("tagm_location", "hyperlopit"),))


@lru_cache(maxsize=1)
def _uniprot():
    """Return bundled UniProt accessions keyed by ``gene_nr``, or ``None``."""
    return _keyed(_read("uniprot.csv"), "gene_nr", (("uniprot", "uniprot"),))


#: The sources, in the order their columns appear in an export.
SOURCES = (
    ("gene name and expression", _metadata),
    ("signal peptide and transmembrane", _topology),
    ("hyperLOPIT compartment", _lopit),
    ("CRISPR phenotype", _phenotype),
    ("UniProt accession", _uniprot),
)


def columns() -> List[str]:
    """Every column :func:`annotate` can add, in order.

    Empty when nothing is bundled. Used to state up front what an export will
    carry, and by the tests, so a source added here cannot be forgotten.
    """
    names: List[str] = []
    for _label, source in SOURCES:
        frame = source()
        if frame is None:
            continue
        names.extend(c for c in frame.columns if c != "gene_nr")
    return names


#: The name the annotation key is merged under. Private, and deliberately not
#: "gene_nr": the caller's table may carry that name itself, and a merge whose
#: right key collides with a left column is suffixed rather than shared.
_JOIN_KEY = "_spacr_annotation_gene_nr"


def _key_column(frame) -> Optional[str]:
    """The column of ``frame`` that names a gene, or ``None``.

    Preference order, and it matters: an explicit ``gene`` column beats the
    design term, because a table that carries both has already parsed the
    term once and this must not disagree with it.
    """
    for name in ("gene_nr", "gene", "gene_id", "Gene ID", "grna", "feature",
                 "variable", "index"):
        if name in getattr(frame, "columns", ()):
            keys = frame[name].map(gene_number)
            if keys.notna().any():
                return name
    return None


def annotate(frame, *, key_column: Optional[str] = None, quiet: bool = False):
    """``frame`` with the bundled *Toxoplasma* annotation joined on.

    :param frame: any exported table -- coefficients, significant hits, the
        montage sidecar.
    :param key_column: the column naming a gene. Found automatically when not
        given, and a table naming no gene comes back UNCHANGED rather than
        gaining a block of empty columns.
    :param quiet: suppress the console line saying what was joined.
    :returns: a COPY. The caller is usually holding the run's own results,
        and annotating them in place would move what every other panel and
        export sees.

    Columns already present in ``frame`` are not overwritten -- a run that
    computed its own ``gene_name`` keeps it, and the annotation's version is
    left out rather than silently replacing it.
    """
    import pandas as pd

    if frame is None or not len(getattr(frame, "columns", ())):
        return frame
    key = key_column or _key_column(frame)
    if key is None or key not in frame.columns:
        if not quiet:
            print("Toxoplasma annotation: this table names no gene column, "
                  "so nothing was joined onto it.")
        return frame

    out = frame.copy()
    out["_gene_nr"] = out[key].map(gene_number)
    added: List[str] = []
    for label, source in SOURCES:
        right = source()
        if right is None:
            continue
        new = [c for c in right.columns
               if c != "gene_nr" and c not in out.columns]
        if not new:
            if not quiet:
                print(f"Toxoplasma annotation: {label} is already on this "
                      f"table, so it was not joined again.")
            continue
        # THE JOIN KEY IS RENAMED BEFORE THE MERGE, not dropped after it.
        # Merging on a right column called "gene_nr" while the caller's own
        # table already has one makes pandas suffix BOTH into gene_nr_x and
        # gene_nr_y, so the drop below found no "gene_nr" and raised KeyError.
        # That is not a rare table: "gene_nr" is the FIRST name _key_column
        # looks for, so it is what spaCR's own annotated output is keyed on,
        # and re-annotating a table this function had already written crashed.
        keyed = right[["gene_nr"] + new].rename(
            columns={"gene_nr": _JOIN_KEY})
        out = out.merge(keyed, how="left",
                        left_on="_gene_nr", right_on=_JOIN_KEY,
                        validate="many_to_one")
        out = out.drop(columns=[_JOIN_KEY])
        added.extend(new)

    matched = int(out["_gene_nr"].notna().sum())
    out = out.drop(columns=["_gene_nr"])
    if not quiet:
        if added:
            # ANY ADDED COLUMN, NOT THE FIRST ONE. This read
            # `out[added[0]]`, which is `gene_name` -- and a gene can be in
            # every bundled table while having no NAME: `TGME49_200130` is
            # one, and it carries a product description, an in-vivo fitness
            # score and a UniProt accession. The line said "1 matched" of
            # three rows when two had matched, which understates the join in
            # exactly the direction that makes a reader distrust it.
            hit = int(out[added].notna().any(axis=1).sum()) if len(out) else 0
            print(f"Toxoplasma annotation: {len(added)} column(s) joined onto "
                  f"{matched} of {len(frame)} row(s) by gene number; "
                  f"{hit} matched the annotation.")
        else:
            print("Toxoplasma annotation: no bundled table could be read, so "
                  "the export carries no annotation.")
    return out


#: Columns of the supplementary topology table that are not per-segment.
_TOPOLOGY_SUMMARY = ("gene_nr", "identifier", "accession", "length",
                     "dtm_type", "n_signal", "n_tm", "sp_start", "sp_end",
                     "sp_length")


def supplementary(genes=None, path=None):
    """The full DeepTMHMM table, as its own supplementary data table.

    This table contains the complete bundled DeepTMHMM results for all
    *Toxoplasma* proteins.

    :param genes: restrict to these genes -- any spelling
        :func:`gene_number` accepts. ``None`` writes all 8,140 proteins.
    :param path: write here as CSV as well as returning the frame.
    :returns: the table, or ``None`` when DeepTMHMM is not bundled.

    SEPARATE FROM :func:`annotate` ON PURPOSE. The per-segment coordinates
    are 72 columns, and a coefficient export carrying them is a table nobody
    opens twice. What belongs beside a coefficient is "does this protein have
    a signal peptide, and how many transmembrane helices"; where each helix
    starts is a different question and gets a different file.

    ONLY THE SEGMENTS THAT EXIST. A screen of soluble proteins gets a table
    ending at `n_tm`, not 72 columns of nothing -- the same rule the rest of
    this module follows.
    """
    frame = _read("deeptmhmm.csv")
    if frame is None:
        return None
    if genes is not None:
        wanted = {gene_number(one) for one in genes} - {None}
        keys = frame["gene_nr"].astype("string").map(gene_number)
        frame = frame.loc[keys.isin(wanted)]
    live = [c for c in frame.columns
            if c in _TOPOLOGY_SUMMARY or frame[c].notna().any()]
    frame = frame[live].reset_index(drop=True)
    if path is not None:
        target = os.path.abspath(os.path.expanduser(os.fspath(path)))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        frame.to_csv(target, index=False)
        print(f"Toxoplasma annotation: DeepTMHMM topology for "
              f"{len(frame)} protein(s) written to {target}")
    return frame


def clear_cache() -> None:
    """Forget the bundled tables. For tests, and for a reinstall mid-session."""
    for _label, source in SOURCES:
        source.cache_clear()


__all__ = ["SOURCES", "annotate", "clear_cache", "columns", "gene_number",
           "supplementary"]


# ---------------------------------------------------------------------------
# Any organism, not only this one
# ---------------------------------------------------------------------------

#: Columns UniProt is asked for, and what the joined table calls them.
#:
#: RENAMED, because the bundled path's columns are what every downstream
#: panel and export already reads. A UniProt annotation that called its gene
#: name "Gene Names" would be a second vocabulary for the same facts.
UNIPROT_COLUMNS = {
    "Entry": "uniprot_accession",
    "Entry Name": "uniprot_entry",
    "Protein names": "product",
    "Gene Names": "gene_name",
    "Organism": "organism",
    "Length": "protein_length",
    "Subcellular location [CC]": "localisation",
    "Function [CC]": "function",
    "Signal peptide": "signal_peptide",
    "Transmembrane": "transmembrane",
    "Gene Ontology (biological process)": "go_process",
    "Gene Ontology (cellular component)": "go_component",
    "Gene Ontology (molecular function)": "go_function",
    "AlphaFoldDB": "alphafold",
}


def _uniprot_keys(values):
    """Every spelling of a gene in a UniProt "Gene Names" cell, lower case.

    UniProt puts the synonyms in one space-separated cell -- "TP53 P53" --
    so a screen naming either one has to match. Returned as a list per row
    so the frame can be exploded onto them.
    """
    out = []
    for value in values:
        names = str(value or "").replace(",", " ").split()
        out.append([n.strip().lower() for n in names if n.strip()])
    return out


def _uniprot_key_column(frame) -> Optional[str]:
    """The column naming a gene, for a table that is NOT Toxoplasma's.

    `_key_column` parses every candidate with :func:`gene_number`, which is
    the bundled path's key space: a bare Toxoplasma gene NUMBER. Human,
    mouse and Plasmodium screens name their targets `TP53`, `Cdkn1a`,
    `PF3D7_1206100` -- none of which is a number -- so that finder rejected
    every column and the UniProt join reported "this table names no gene
    column" about tables that named nothing else.

    Same preference order, different test: a column counts when it holds
    text at all.
    """
    for name in ("gene", "gene_name", "gene_id", "Gene ID", "gene_nr",
                 "grna", "feature", "variable", "index"):
        if name not in getattr(frame, "columns", ()):
            continue
        values = frame[name].astype(str).str.strip()
        if values.replace("", None).notna().any() and (values != "").any():
            return name
    return None


def annotate_from_uniprot(frame, source, *, cache_dir=None,
                          key_column: Optional[str] = None,
                          quiet: bool = False):
    """``frame`` with UniProt's annotation for ``source`` joined onto it.

    The non-bundled half of :func:`annotate_with`. Joins on the gene NAME --
    UniProt's own, including its synonyms -- and on the accession, because a
    screen library names its targets one way or the other and neither is
    wrong.

    Never raises and never multiplies rows: the annotation is collapsed to
    one row per key before the merge, and the merge is ``many_to_one``, for
    the reason this module's header gives.

    :returns: ``(frame, note)``. The frame is unchanged when nothing could
        be joined, and the note says why.
    """
    import pandas as pd

    from .uniprot import annotation_for

    if frame is None or not len(getattr(frame, "columns", ())):
        return frame, ""
    # THE GENES THIS TABLE NAMES, so the query can ask for them rather than
    # for a whole proteome. Read before the key column is resolved because
    # the key finder is cheap and the fetch is not.
    asked = _uniprot_key_column(frame) if key_column is None else key_column
    genes = (frame[asked].astype(str).tolist()
             if asked and asked in frame.columns else None)
    table, note = annotation_for(source, cache_dir=cache_dir, genes=genes)
    if table is None or not len(table):
        return frame, note

    key = key_column or _uniprot_key_column(frame)
    if key is None or key not in frame.columns:
        return frame, ("this table names no gene column, so the UniProt "
                       "annotation was not joined onto it.")

    table = table.rename(columns=UNIPROT_COLUMNS)
    keep = [c for c in UNIPROT_COLUMNS.values() if c in table.columns]
    if not keep:
        return frame, "UniProt returned no usable columns."
    table = table[keep].copy()

    # ONE ROW PER KEY. A gene name that appears on two entries -- an isoform
    # pair, a duplicated locus -- would otherwise turn one coefficient into
    # two rows, which is the failure this module was written to prevent.
    table["_key"] = _uniprot_keys(table.get("gene_name", ""))
    exploded = table.explode("_key").dropna(subset=["_key"])
    if "uniprot_accession" in table.columns:
        by_accession = table.copy()
        by_accession["_key"] = (by_accession["uniprot_accession"]
                                .astype(str).str.strip().str.lower())
        exploded = pd.concat([exploded, by_accession], ignore_index=True)
    exploded = exploded.drop_duplicates(subset=["_key"], keep="first")

    joined = frame.copy()
    joined["_key"] = joined[key].astype(str).str.strip().str.lower()
    # Columns the caller already computed are theirs, not UniProt's.
    new = [c for c in keep if c not in frame.columns]
    if not new:
        return frame, ""
    merged = joined.merge(exploded[["_key", *new]], on="_key", how="left",
                          validate="many_to_one")
    merged = merged.drop(columns=["_key"])
    matched = int(merged[new[0]].notna().sum()) if new else 0
    if not quiet:
        print(f"UniProt annotation ({source}): {matched} of {len(merged)} "
              f"rows matched, {len(new)} column(s) joined.")
    if matched == 0:
        note = (note + " " if note else "") + (
            f"none of the {len(frame)} gene identifiers matched a UniProt "
            f"entry for {source}.")
    return merged, note


def annotate_with(frame, source, *, cache_dir=None,
                  key_column: Optional[str] = None, quiet: bool = False):
    """Annotate ``frame`` from whatever ``source`` names.

    The one call the pipeline makes. ``source`` is the ``annotation_source``
    setting:

    * empty or any spelling of Toxoplasma -- the BUNDLED tables, offline,
      exactly as before. This is the default and it does not touch
      :mod:`spacr.uniprot` at all;
    * an organism name or taxon id -- that organism from UniProt;
    * an accession -- that entry.

    :returns: ``(frame, note)``; the note is empty on a clean join.
    """
    from .uniprot import resolve

    if resolve(source).kind == "bundled":
        return annotate(frame, key_column=key_column, quiet=quiet), ""
    return annotate_from_uniprot(frame, source, cache_dir=cache_dir,
                                 key_column=key_column, quiet=quiet)
