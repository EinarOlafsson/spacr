"""Everything spaCR knows about a *Toxoplasma* gene, joined onto an export.

Asked for on 2026-08-17: "if it is on all the exported tables should be merged
with the relevant Toxoplasma information (gene name, signal peptide,
transmembrane domain, the phenotype scores from the screens we have
downloaded, tachyzoite expression, bradyzoite expression, sexual stages
expression, hyperLopit expression)".

Instruction 133. Until this module, `toxo=True` reached exactly two places --
the volcano's colours and two heatmaps -- and the CSV a reader actually opens
came out as bare gene numbers and coefficients. Somebody then joined the
annotation by hand in Excel, which is where the wrong-key mistakes live.

THREE THINGS MAKE THIS SAFE, and each is a bug this project has already had.

1.  ONE KEY SPACE: the bare gene NUMBER. `TGGT1_224750` (what the screen
    library uses), `TGME49_224750` (what every annotation table uses),
    `gene_fraction:gene[224750]` (what patsy names a term) and `224750_2` (a
    guide) are all gene ``224750``. :func:`gene_number` is the single parse.

2.  THE JOIN CANNOT MULTIPLY ROWS. Every source is collapsed to one row per
    gene before it is merged and the merge is declared ``many_to_one``, so a
    duplicated annotation key raises instead of silently turning one
    coefficient into four rows. That failure has already reached a user's
    figure once, and it looked entirely plausible while it was wrong.

3.  A COLUMN THAT CANNOT BE FILLED IS ABSENT, WITH A REASON. Never a column
    of NaN, which reads as "measured, found nothing" rather than "not
    available here".

WHAT IS BUNDLED, and why it is small enough to bundle:

    toxoplasma_metadata.csv  2.97 MB  gene name, product, expression
    deeptmhmm.csv            1.11 MB  signal peptide and transmembrane
    phenotype.csv            0.48 MB  the published CRISPR fitness screens
    lopit.csv                0.20 MB  hyperLOPIT/TAGM compartment
    uniprot.csv              0.13 MB  accession

deeptmhmm.csv is this project's own DeepTMHMM run over 8,140 proteins, at 82
columns: the summary (`dtm_type`, `n_tm`, `sp_length`) that :func:`annotate`
joins, and the coordinates of every one of the up to 24 transmembrane
segments, which :func:`supplementary` writes as its own table. The 25
per-segment SEQUENCE columns of the source file are dropped -- they are 16 of
its 17 MB and every one of them is recoverable from the proteome.
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
    frame = _read("phenotype.csv")
    if frame is None:
        return None
    columns = [c for c in frame.columns if c.startswith("fit_")]
    return _keyed(frame, "gene_nr", tuple((c, c) for c in columns))


@lru_cache(maxsize=1)
def _lopit():
    return _keyed(_read("lopit.csv"), "gene_nr",
                  (("tagm_location", "hyperlopit"),))


@lru_cache(maxsize=1)
def _uniprot():
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
        out = out.merge(right[["gene_nr"] + new], how="left",
                        left_on="_gene_nr", right_on="gene_nr",
                        validate="many_to_one")
        out = out.drop(columns=["gene_nr"])
        added.extend(new)

    matched = int(out["_gene_nr"].notna().sum())
    out = out.drop(columns=["_gene_nr"])
    if not quiet:
        if added:
            hit = int(out[added[0]].notna().sum()) if len(out) else 0
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

    Asked for on 2026-08-17: "add that as a suplimentary data table (a csv
    with all the information from DeepTMHMM on all toxo proteins)".

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
