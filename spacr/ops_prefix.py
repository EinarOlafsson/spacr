"""Decode sgRNA spacer prefixes read in situ (item 920, alpha).

CROP-seq style optical screens (Feldman et al. 2019; PERISCOPE, Ramezani et
al. 2025) sequence the guide's own spacer, not a separate barcode. The library
table (``Barcodes.csv``, column ``sgRNA``) is the whitelist, and only the first
N bases are read: CellProfiler's ``CallBarcodes`` used 10 cycles for A549 and
12 for HeLa although 12 were imaged. Spacers of different guides can share a
short prefix, so the whitelist is the set of UNIQUE prefixes, and prefixes
shared by guides of different genes are reported and never called.

What this module adds on top of spaCR's base calling:

* the library model (:func:`spacer_prefix_library`): prefixes after a cycle
  mask, collisions, the whitelist's minimum pairwise distance;
* matching (:func:`match_prefix_reads`): exact, then Hamming or Levenshtein
  distance up to a limit, an equally close pair is ambiguous;
* a shuffled-whitelist control (:func:`shuffled_library`);
* per-cell calls with a quality class (:func:`prefix_cell_calls`), by
  spaCR's read-count rules or by the pooled Cell Painting recipe's "simple"
  rules PERISCOPE used (:func:`recipe_simple_quality`);
* per-cycle or pooled crosstalk compensation (:func:`compensate_prefix`);
* the site decode (:func:`decode_prefix_site`) and the run phase
  (:func:`run_prefix_decode`) over the cropped-site layout of item 919.

A masked cycle is a base that was not read. It is removed from both the
library prefix and the read, so the key of a 10-cycle prefix with cycle 6
masked has nine letters.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Read statuses, in report order.
READ_STATUSES = ("exact", "corrected", "ambiguous", "collision", "out_of_library")

#: The per-cell quality classes and what each means. Written to the run folder.
QUALITY_CLASSES = {
    "perfect": "at least ops_quality_min_reads exact reads of one barcode and no "
               "read of any other barcode",
    "great": "not perfect; the top barcode has more reads than any other "
             "(exact reads with a minority competitor, a single exact read, "
             "or corrected reads only)",
    "ambiguous": "matched reads, but two barcodes tie for the most reads",
    "none": "no read matched a callable barcode",
}

#: The per-cell quality methods (``ops_prefix_quality_method``).
QUALITY_METHODS = ("spacr", "recipe_simple")

#: The classes of the ``'recipe_simple'`` method, the "simple" cell quality
#: of the Broad pooled Cell Painting profiling recipe that PERISCOPE used.
#: A read's score is one minus its distance over the key length, as
#: CellProfiler's ``CallBarcodes`` scores a match.
RECIPE_SIMPLE_CLASSES = {
    "perfect": "one barcode in the cell and every read of it an exact match "
               "(mean score 1)",
    "great": "one barcode in the cell and at least one corrected read "
             "(mean score below 1)",
    "imperfect": "several barcodes; one barcode has both the highest mean score "
                 "and the most reads",
    "bad": "several barcodes and no single barcode has both the highest mean "
           "score and the most reads",
    "none": "no read matched a callable barcode",
}

#: Where the ``'recipe_simple'`` rules were ported from.
RECIPE_SIMPLE_SOURCE = {
    "repository": "https://github.com/broadinstitute/pooled-cell-painting-profiling-recipe",
    "function": "simple_categorize (cell_quality_utils.py) and category_counts (spot_utils.py)",
    "commit": "49e62a30c6a624ae68fdb8fec0b9702345ffa01a",
    "same_rules_as": "1d5e071 (2020-08-20, scripts/cell_quality_utils.py)",
    "pycytominer": "pycytominer 1.7.1 has no cell-quality method",
}

_RECIPE_SIMPLE_INDEX = {1: "perfect", 2: "great", 3: "imperfect", 4: "bad"}

_METRICS = ("hamming", "levenshtein")
_COLLISIONS = ("exclude", "gene_level")
_CROSSTALK = ("none", "per_cycle", "global")


@dataclass
class PrefixLibrary:
    """A spacer-prefix whitelist.

    :param positions: the 1-based cycles the key is read from, in order.
    :param keys: every distinct prefix key, sorted.
    :param guides: ``key -> sorted guide spacers`` sharing it.
    :param genes: ``key -> sorted genes`` of those guides.
    :param callable_keys: ``key -> (guide, gene, level)`` for the keys that
        may be called; ``level`` is ``'guide'``, or ``'gene'`` for a key
        shared only by guides of one gene under ``'gene_level'``.
    :param collisions: one row per key shared by two or more guides.
    :param extended: ``key -> set of one-base-longer prefixes`` (the next
        unmasked spacer base appended), for Levenshtein matching.
    :param summary: counts and the whitelist's minimum distance.
    """

    positions: Tuple[int, ...]
    keys: List[str]
    guides: Dict[str, List[str]]
    genes: Dict[str, List[str]]
    callable_keys: Dict[str, Tuple[str, str, str]]
    collisions: pd.DataFrame
    extended: Dict[str, set] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


def prefix_positions(prefix_length: int, cycle_mask: Sequence[int] = (),
                     n_cycles: Optional[int] = None) -> Tuple[int, ...]:
    """The 1-based cycles a prefix key is read from.

    :param prefix_length: the prefix covers cycles 1 to this.
    :param cycle_mask: cycles not read; removed from the key.
    :param n_cycles: cycles imaged; the prefix may not run past them.
    :returns: the kept cycles, in order.
    :raises ValueError: for a prefix longer than the cycles imaged, a mask
        outside the prefix range, or a mask that leaves nothing.
    """
    length = int(prefix_length)
    if length < 1:
        raise ValueError(f"ops_prefix_length must be at least 1; got {length}")
    if n_cycles is not None and length > int(n_cycles):
        raise ValueError(f"ops_prefix_length {length} is longer than the "
                         f"{int(n_cycles)} cycles imaged (ops_sbs_cycles)")
    mask = sorted({int(c) for c in (cycle_mask or ())})
    bad = [c for c in mask if c < 1 or (n_cycles is not None and c > int(n_cycles))]
    if bad:
        raise ValueError(f"ops_cycle_mask names cycles outside 1..{n_cycles}: {bad}")
    kept = tuple(c for c in range(1, length + 1) if c not in mask)
    if not kept:
        raise ValueError("ops_cycle_mask removes every cycle of the prefix")
    return kept


def _spacer_bases(spacer: str, cycle_mask: set) -> str:
    """The spacer's bases in cycle order with the masked cycles removed."""
    return "".join(base for cycle, base in enumerate(spacer, start=1)
                   if cycle not in cycle_mask)


def _deletion_index(keys: Sequence[str]) -> Dict[Tuple[int, str], set]:
    """``(position, key without that position) -> keys``, for distance-one lookups."""
    index: Dict[Tuple[int, str], set] = {}
    for key in keys:
        for position in range(len(key)):
            index.setdefault((position, key[:position] + key[position + 1:]),
                             set()).add(key)
    return index


def whitelist_min_distance(keys: Sequence[str]) -> Dict[str, Any]:
    """The smallest Hamming distance between two keys, and how many pairs are at one.

    :param keys: distinct keys of one length.
    :returns: ``min_distance`` (1, 2, or 3 meaning three or more),
        ``min_distance_capped`` (True when 3 stands for three or more) and
        ``pairs_at_distance_1``.
    """
    keys = sorted(set(keys))
    if len(keys) < 2:
        return {"min_distance": None, "min_distance_capped": False,
                "pairs_at_distance_1": 0}
    pairs = 0
    for members in _deletion_index(keys).values():
        n = len(members)
        pairs += n * (n - 1) // 2
    if pairs:
        return {"min_distance": 1, "min_distance_capped": False,
                "pairs_at_distance_1": int(pairs)}
    seen: Dict[Tuple[int, int, str], str] = {}
    for key in keys:
        for i in range(len(key)):
            for j in range(i + 1, len(key)):
                bucket = (i, j, key[:i] + key[i + 1:j] + key[j + 1:])
                other = seen.setdefault(bucket, key)
                if other != key:
                    return {"min_distance": 2, "min_distance_capped": False,
                            "pairs_at_distance_1": 0}
    return {"min_distance": 3, "min_distance_capped": True, "pairs_at_distance_1": 0}


def spacer_prefix_library(library: Any, *, sequence_col: str = "sgRNA",
                          gene_col: str = "gene_symbol", prefix_length: int = 10,
                          n_cycles: Optional[int] = 12,
                          cycle_mask: Sequence[int] = (),
                          collision: str = "exclude") -> PrefixLibrary:
    """Build the spacer-prefix whitelist of a CROP-seq library.

    Each guide's key is its spacer's bases at :func:`prefix_positions`.
    Identical spacers on several rows count as one guide. A key shared by
    guides of different genes is ambiguous and never called. A key shared by
    several guides of one gene is not called either under ``'exclude'``; under
    ``'gene_level'`` it is called as that gene, with the guides joined by
    ``|``.

    :param library: a table, or the path of one (CSV, TSV, Parquet), holding
        ``sequence_col`` and ``gene_col``.
    :param sequence_col: the spacer column.
    :param gene_col: the gene column.
    :param prefix_length: the prefix covers cycles 1 to this.
    :param n_cycles: cycles imaged.
    :param cycle_mask: 1-based cycles not read.
    :param collision: ``'exclude'`` or ``'gene_level'``.
    :returns: the :class:`PrefixLibrary`.
    :raises ValueError: for a missing column, an unknown ``collision`` or a
        spacer shorter than the prefix.
    """
    if collision not in _COLLISIONS:
        raise ValueError(f"ops_prefix_collision must be one of {list(_COLLISIONS)}; "
                         f"got {collision!r}")
    if isinstance(library, (str, os.PathLike)):
        from .tabular import read_table

        library = read_table(str(library), canonicalise=False, report=None)
    table = pd.DataFrame(library)
    absent = [c for c in (sequence_col, gene_col) if c not in table.columns]
    if absent:
        raise ValueError(f"the library has no column {absent}; columns are "
                         f"{list(table.columns)[:20]}")
    positions = prefix_positions(prefix_length, cycle_mask, n_cycles)
    mask = set(int(c) for c in (cycle_mask or ()))
    spacers = table[sequence_col].astype(str).str.strip().str.upper()
    genes = table[gene_col].astype(str)
    short = spacers[spacers.str.len() < max(positions)]
    if len(short):
        raise ValueError(f"{len(short)} spacers are shorter than the "
                         f"{max(positions)}-cycle prefix, for example {short.iloc[0]!r}")
    guides: Dict[str, set] = {}
    gene_of: Dict[str, set] = {}
    extended: Dict[str, set] = {}
    by_guide = dict(zip(spacers, genes))
    for spacer, gene in sorted(by_guide.items()):
        bases = "".join(spacer[c - 1] for c in positions)
        guides.setdefault(bases, set()).add(spacer)
        gene_of.setdefault(bases, set()).add(gene)
        tail = _spacer_bases(spacer, mask)
        if len(tail) > len(bases):
            extended.setdefault(bases, set()).add(tail[:len(bases) + 1])
    keys = sorted(guides)
    callable_keys: Dict[str, Tuple[str, str, str]] = {}
    rows = []
    for key in keys:
        members, names = sorted(guides[key]), sorted(gene_of[key])
        if len(members) == 1:
            callable_keys[key] = (members[0], names[0], "guide")
            continue
        kind = "same_gene" if len(names) == 1 else "different_genes"
        kept = kind == "same_gene" and collision == "gene_level"
        if kept:
            callable_keys[key] = ("|".join(members), names[0], "gene")
        rows.append({"prefix": key, "n_guides": len(members), "n_genes": len(names),
                     "kind": kind, "called": kept, "guides": "|".join(members),
                     "genes": "|".join(names)})
    collisions = pd.DataFrame(rows, columns=["prefix", "n_guides", "n_genes", "kind",
                                             "called", "guides", "genes"])
    distance = whitelist_min_distance(list(callable_keys))
    summary = {
        "positions": list(positions), "prefix_length": int(prefix_length),
        "cycle_mask": sorted(mask), "key_length": len(positions),
        "library_rows": int(len(table)), "guides": int(len(by_guide)),
        "keys": len(keys), "callable_keys": len(callable_keys),
        "guide_level_keys": sum(1 for v in callable_keys.values() if v[2] == "guide"),
        "gene_level_keys": sum(1 for v in callable_keys.values() if v[2] == "gene"),
        "collision_keys": int(len(collisions)),
        "collision_keys_different_genes": int((collisions["kind"] == "different_genes").sum()),
        "collision_keys_same_gene": int((collisions["kind"] == "same_gene").sum()),
        "guides_in_collisions": int(collisions["n_guides"].sum()) if len(collisions) else 0,
        "collision": collision, **distance}
    return PrefixLibrary(positions=positions, keys=keys,
                         guides={k: sorted(v) for k, v in guides.items()},
                         genes={k: sorted(v) for k, v in gene_of.items()},
                         callable_keys=callable_keys, collisions=collisions,
                         extended=extended, summary=summary)


def shuffled_library(library: PrefixLibrary, seed: int = 0) -> PrefixLibrary:
    """The random-barcode control: every key's letters shuffled.

    Each key (and its one-base extension) is permuted with one generator, so
    the control has the same number of keys, the same length and the same
    base composition as the whitelist; a shuffled key that lands on a real
    key stays in the control.

    :param library: the real whitelist.
    :param seed: the generator seed (``ops_control_seed``).
    :returns: a :class:`PrefixLibrary` whose keys all call ``'control'``.
    """
    rng = np.random.default_rng(int(seed))
    length = len(library.positions)
    keys: set = set()
    extended: Dict[str, set] = {}
    for key in library.keys:
        longer = sorted(library.extended.get(key, ()))
        source = longer[0] if longer else key
        letters = np.array(list(source))
        shuffled = "".join(letters[rng.permutation(len(letters))])
        new = shuffled[:length]
        keys.add(new)
        if len(shuffled) > length:
            extended.setdefault(new, set()).add(shuffled)
    keys_sorted = sorted(keys)
    return PrefixLibrary(positions=library.positions, keys=keys_sorted,
                         guides={k: [k] for k in keys_sorted},
                         genes={k: ["control"] for k in keys_sorted},
                         callable_keys={k: (k, "control", "guide") for k in keys_sorted},
                         collisions=library.collisions.iloc[0:0], extended=extended,
                         summary={"keys": len(keys_sorted), "seed": int(seed)})


def _levenshtein_indexes(library: PrefixLibrary):
    """The deletion and insertion lookups for one-indel matches.

    A read that skipped base ``i`` equals the key's one-base extension with
    ``i`` deleted; a read with an extra base at ``i`` equals the key's first
    ``K - 1`` bases once that base is deleted.
    """
    dropped: Dict[str, set] = {}
    for key, longer in library.extended.items():
        for text in longer:
            for i in range(len(text) - 1):
                variant = text[:i] + text[i + 1:]
                if variant != key:
                    dropped.setdefault(variant, set()).add(key)
    shorter: Dict[str, set] = {}
    for key in library.keys:
        shorter.setdefault(key[:-1], set()).add(key)
    return dropped, shorter


def _compare(read: str, table: np.ndarray, entries: Sequence[str]) -> Tuple[int, List[str]]:
    """The smallest Hamming distance to ``entries`` (N matching any base) and who has it."""
    letters = np.frombuffer(read.encode("ascii"), dtype=np.uint8)
    distance = ((table != letters) & (letters != ord("N"))).sum(axis=1)
    best = int(distance.min())
    return best, [entries[i] for i in np.flatnonzero(distance == best)]


def match_prefix_reads(reads: Sequence[str], library: PrefixLibrary, *,
                       metric: str = "hamming", max_distance: int = 1,
                       keyed: bool = False) -> pd.DataFrame:
    """Match called reads to a prefix whitelist.

    A read is first cut to the key's cycles (:attr:`PrefixLibrary.positions`;
    reads given as keys with ``keyed=True`` are taken as they are). An exact
    match is ``exact``. Otherwise the keys within ``max_distance`` are found:
    one closest key is ``corrected``, two or more equally close keys are
    ``ambiguous``, none is ``out_of_library``. Under ``'hamming'`` a
    distance is the number of differing cycles; under ``'levenshtein'`` a
    single skipped or extra base also counts as one, matched against the
    spacer's next base. An ``N`` (a cycle that was not measured) matches any
    base and costs nothing; reads with one are matched by Hamming distance.
    A unique match to a key that cannot be called (a prefix collision) is
    ``collision``.

    :param reads: the called reads, one letter per cycle from cycle 1.
    :param library: from :func:`spacer_prefix_library`.
    :param metric: ``'hamming'`` or ``'levenshtein'``.
    :param max_distance: the largest distance corrected; 0 is exact only.
        Levenshtein takes 0 or 1.
    :param keyed: ``reads`` are already keys.
    :returns: one row per read: ``read``, ``key`` (the read's key),
        ``prefix`` (the matched whitelist key, or None), ``distance`` (-1
        when unmatched), ``status`` (:data:`READ_STATUSES`), ``guide``,
        ``gene`` and ``level``.
    :raises ValueError: for an unknown metric, a negative distance, or
        Levenshtein beyond one.
    """
    if metric not in _METRICS:
        raise ValueError(f"ops_prefix_metric must be one of {list(_METRICS)}; got {metric!r}")
    max_distance = int(max_distance)
    if max_distance < 0:
        raise ValueError("ops_prefix_max_distance must be 0 or more")
    if metric == "levenshtein" and max_distance > 1:
        raise ValueError("Levenshtein matching corrects at most one edit")
    index0 = [p - 1 for p in library.positions]
    known = set(library.keys)
    deletion = _deletion_index(library.keys) if max_distance >= 1 else {}
    dropped, shorter = (_levenshtein_indexes(library) if metric == "levenshtein"
                        and max_distance >= 1 else ({}, {}))
    entries = list(library.keys)
    table = (np.frombuffer("".join(entries).encode("ascii"), dtype=np.uint8)
             .reshape(len(entries), len(index0)) if entries else None)
    rows = []
    for read in reads:
        read = str(read)
        key = read if keyed else "".join(read[i] if i < len(read) else "N" for i in index0)
        best, found = -1, []
        if key in known:
            best, found = 0, [key]
        elif "N" in key:
            if table is not None:
                distance, closest = _compare(key, table, entries)
                if distance <= max_distance:
                    best, found = distance, closest
        elif max_distance >= 1:
            found_set = set()
            for position in range(len(key)):
                found_set |= deletion.get((position, key[:position] + key[position + 1:]), set())
            if metric == "levenshtein":
                found_set |= dropped.get(key, set())
                for position in range(len(key)):
                    found_set |= shorter.get(key[:position] + key[position + 1:], set())
            if found_set:
                best, found = 1, sorted(found_set)
            elif max_distance >= 2 and table is not None:
                distance, closest = _compare(key, table, entries)
                if distance <= max_distance:
                    best, found = distance, closest
        if not found:
            status, prefix = "out_of_library", None
        elif len(found) > 1:
            status, prefix = "ambiguous", None
        else:
            prefix = found[0]
            if prefix not in library.callable_keys:
                status = "collision"
            else:
                status = "exact" if best == 0 else "corrected"
        call = library.callable_keys.get(prefix) if status in ("exact", "corrected") else None
        rows.append({"read": read, "key": key, "prefix": prefix,
                     "distance": int(best) if found else -1, "status": status,
                     "guide": call[0] if call else None,
                     "gene": call[1] if call else None,
                     "level": call[2] if call else None})
    return pd.DataFrame(rows, columns=["read", "key", "prefix", "distance", "status",
                                       "guide", "gene", "level"])


def classify_cell(counts: Mapping[str, Tuple[int, int]], min_reads: int = 2) -> str:
    """One cell's quality class from its matched reads.

    :param counts: ``barcode -> (exact reads, corrected reads)`` of the
        reads that matched a callable barcode.
    :param min_reads: exact reads a ``perfect`` cell needs.
    :returns: a key of :data:`QUALITY_CLASSES`.
    """
    totals = sorted(((e + c, e, c, b) for b, (e, c) in counts.items() if e + c > 0),
                    reverse=True)
    if not totals:
        return "none"
    if len(totals) > 1 and totals[0][0] == totals[1][0]:
        return "ambiguous"
    top_exact = totals[0][1]
    if len(totals) == 1 and top_exact >= int(min_reads):
        return "perfect"
    return "great"


def recipe_simple_quality(spots: pd.DataFrame, *, cell_cols: Sequence[str] = ("cell",),
                          barcode_col: str = "prefix", gene_col: str = "gene",
                          score_col: str = "score") -> pd.DataFrame:
    """Each cell's class under the pooled Cell Painting recipe's "simple" rules.

    A port of ``simple_categorize`` from the Broad pooled Cell Painting
    profiling recipe (:data:`RECIPE_SIMPLE_SOURCE`), which assigned
    PERISCOPE's per-cell ``Perfect`` and ``Great`` classes. The spots of a
    cell are grouped by gene and barcode, each group's mean score and
    spot count are taken, and then: one barcode with a mean score of
    exactly 1 is ``perfect`` and below 1 is ``great``; with several
    barcodes, one barcode holding both the single highest mean score and
    the single highest count is ``imperfect``, anything else is ``bad``.
    Spots with a missing gene or barcode are left out, as the recipe's
    group-by leaves them out.

    :param spots: one row per spot inside a cell, with ``cell_cols``,
        ``barcode_col``, ``gene_col`` and ``score_col`` (1 for an exact
        match).
    :param cell_cols: the columns naming a cell.
    :param barcode_col: the matched barcode.
    :param gene_col: the matched barcode's gene.
    :param score_col: the match score.
    :returns: one row per cell with a scored spot: ``cell_cols``,
        ``barcodes`` (distinct barcodes), ``barcode``, ``gene``,
        ``score_mean`` and ``score_count`` of the recipe's top barcode (the
        first with the highest mean score, in gene and barcode order),
        ``quality_index`` (1 to 4) and ``quality`` (a key of
        :data:`RECIPE_SIMPLE_CLASSES`).
    """
    cell_cols = list(cell_cols)
    columns = cell_cols + ["barcodes", "barcode", "gene", "score_mean", "score_count",
                           "quality_index", "quality"]
    groups = cell_cols + [gene_col, barcode_col]
    frame = spots.dropna(subset=groups)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    table = (frame.groupby(groups, sort=True)[score_col].agg(["mean", "count"])
             .reset_index())
    per_cell = table.groupby(cell_cols, sort=False)
    max_score = per_cell["mean"].transform("max")
    max_count = per_cell["count"].transform("max")
    table = table.assign(at_score=table["mean"] == max_score,
                         at_count=table["count"] == max_count)
    per_cell = table.groupby(cell_cols, sort=False)
    top = table[table["at_score"]].groupby(cell_cols, sort=False).head(1)
    summary = pd.DataFrame({
        "barcodes": per_cell.size(),
        "n_at_score": per_cell["at_score"].sum(),
        "n_at_count": per_cell["at_count"].sum()})
    top = top.set_index(cell_cols)
    summary = summary.join(top[[gene_col, barcode_col, "mean", "count", "at_count"]])
    single = summary["barcodes"] == 1
    unique = (summary["n_at_score"] == 1) & (summary["n_at_count"] == 1)
    index = np.where(single, np.where(summary["mean"] == 1, 1, 2),
                     np.where(unique & summary["at_count"].astype(bool), 3, 4))
    out = summary.reset_index()
    out = pd.DataFrame({
        **{name: out[name] for name in cell_cols},
        "barcodes": out["barcodes"].astype(int),
        "barcode": out[barcode_col], "gene": out[gene_col],
        "score_mean": out["mean"].astype(float), "score_count": out["count"].astype(int),
        "quality_index": index.astype(int),
        "quality": [_RECIPE_SIMPLE_INDEX[int(i)] for i in index]})
    return out.sort_values(cell_cols, kind="stable").reset_index(drop=True)[columns]


def prefix_cell_calls(reads: pd.DataFrame, *, min_reads: int = 2,
                      keys: Sequence[str] = ("plate", "well", "site", "cell"),
                      method: str = "spacr") -> pd.DataFrame:
    """Each cell's barcode, its read counts and its quality class.

    :param reads: one row per read with ``keys``, ``status``, ``key``,
        ``distance``, ``prefix``, ``guide``, ``gene`` and ``level``
        (:func:`match_prefix_reads` joined to the read positions); rows
        whose ``cell`` is 0 are not in a cell.
    :param min_reads: exact reads a ``perfect`` cell needs
        (``ops_quality_min_reads``) under ``'spacr'``.
    :param keys: the columns naming a cell.
    :param method: ``ops_prefix_quality_method``: ``'spacr'``
        (:func:`classify_cell`, classes :data:`QUALITY_CLASSES`) or
        ``'recipe_simple'`` (:func:`recipe_simple_quality` on the exact and
        corrected reads, each scored one minus its distance over the key
        length; classes :data:`RECIPE_SIMPLE_CLASSES`).
    :returns: one row per cell with any read: ``keys``, ``reads``,
        ``matched`` (exact or corrected), ``exact``, ``corrected``,
        ``ambiguous``, ``collision``, ``out_of_library``, ``barcodes``
        (distinct callable barcodes), ``prefix``, ``guide``, ``gene`` and
        ``level`` of the top barcode (empty unless ``perfect`` or ``great``),
        ``top_reads``, ``top_exact``, ``top_share`` (of matched reads),
        ``second_reads``, ``has_exact_read`` and ``quality``.
    :raises ValueError: for an unknown method.
    """
    if method not in QUALITY_METHODS:
        raise ValueError(f"ops_prefix_quality_method must be one of {list(QUALITY_METHODS)}; "
                         f"got {method!r}")
    columns = list(keys) + ["reads", "matched", "exact", "corrected", "ambiguous",
                            "collision", "out_of_library", "barcodes", "prefix", "guide",
                            "gene", "level", "top_reads", "top_exact", "top_share",
                            "second_reads", "has_exact_read", "quality"]
    frame = reads[reads["cell"] > 0] if "cell" in reads.columns else reads.iloc[0:0]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    recipe = {}
    if method == "recipe_simple":
        scored = frame[frame["status"].isin(("exact", "corrected"))]
        scored = scored.assign(score=1.0 - scored["distance"].astype(float)
                               / scored["key"].str.len().astype(float))
        classes = recipe_simple_quality(scored, cell_cols=list(keys), barcode_col="prefix",
                                        gene_col="gene", score_col="score")
        recipe = dict(zip(map(tuple, classes[list(keys)].to_numpy().tolist()),
                          classes["quality"]))
    out = []
    for cell_key, group in frame.groupby(list(keys), sort=True):
        status = group["status"].value_counts()
        matched = group[group["status"].isin(("exact", "corrected"))]
        counts: Dict[str, List[int]] = {}
        for prefix, kind in zip(matched["prefix"], matched["status"]):
            slot = counts.setdefault(prefix, [0, 0])
            slot[0 if kind == "exact" else 1] += 1
        if method == "recipe_simple":
            named = cell_key if isinstance(cell_key, tuple) else (cell_key,)
            quality = recipe.get(tuple(named), "none")
        else:
            quality = classify_cell({k: tuple(v) for k, v in counts.items()}, min_reads)
        ranked = sorted(counts.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))
        top = ranked[0] if ranked else None
        call = (matched[matched["prefix"] == top[0]].iloc[0]
                if top is not None and quality in ("perfect", "great") else None)
        n_matched = int(len(matched))
        top_reads = int(sum(top[1])) if top else 0
        out.append({
            **dict(zip(keys, cell_key if isinstance(cell_key, tuple) else (cell_key,))),
            "reads": int(len(group)), "matched": n_matched,
            **{name: int(status.get(name, 0)) for name in READ_STATUSES},
            "barcodes": len(counts),
            "prefix": top[0] if call is not None else None,
            "guide": call["guide"] if call is not None else None,
            "gene": call["gene"] if call is not None else None,
            "level": call["level"] if call is not None else None,
            "top_reads": top_reads, "top_exact": int(top[1][0]) if top else 0,
            "top_share": float(top_reads / n_matched) if n_matched else 0.0,
            "second_reads": int(sum(ranked[1][1])) if len(ranked) > 1 else 0,
            "has_exact_read": bool(status.get("exact", 0) > 0),
            "quality": quality})
    return pd.DataFrame(out, columns=columns)


def compensate_prefix(values: np.ndarray, mode: str = "none") -> np.ndarray:
    """Undo dye crosstalk per cycle, pooled over cycles, or not at all.

    The unmixing is brieflow's median transform (:mod:`spacr.ops_sbs`): each
    channel's axis is the median of the spots it wins, after each channel is
    divided by its median; ``'per_cycle'`` fits one 4 x 4 matrix per cycle,
    ``'global'`` one over every cycle's spots.

    :param values: ``(reads, cycles, channels)`` intensities; a cycle with
        a NaN is left as it is.
    :param mode: ``'none'``, ``'per_cycle'`` or ``'global'``.
    :returns: float32 values, same shape.
    :raises ValueError: for an unknown mode.
    """
    from .ops_sbs import _argmax_median_unmix

    if mode not in _CROSSTALK:
        raise ValueError(f"ops_prefix_crosstalk must be one of {list(_CROSSTALK)}; got {mode!r}")
    data = np.asarray(values, dtype=np.float32).copy()
    if mode == "none" or data.size == 0:
        return data
    n_channels = data.shape[-1]
    if mode == "global":
        flat = data.reshape(-1, n_channels)
        rows = np.isfinite(flat).all(axis=1)
        if rows.sum() >= n_channels:
            flat[rows] = _argmax_median_unmix(flat[rows])
        return flat.reshape(data.shape)
    for cycle in range(data.shape[1]):
        plane = data[:, cycle, :]
        rows = np.isfinite(plane).all(axis=1)
        if rows.sum() >= n_channels:
            plane[rows] = _argmax_median_unmix(plane[rows])
        data[:, cycle, :] = plane
    return data


def detect_spots(stack: np.ndarray, *, threshold: float, margin: int = 5,
                 sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Find reads in a ``(cycles, Y, X, bases)`` stack as the tile decode does.

    Each plane is Laplacian-of-Gaussian filtered (sigma 1, negative part
    clipped); the read score is the per-channel standard deviation across
    cycles averaged over channels (:func:`spacr.ops_sbs.estimate_read_locations`,
    one channel at a time to bound memory); local maxima at least 2 px apart
    whose score rises more than ``threshold`` above its 5 x 5 minimum are
    kept, ``margin`` px from the border.

    :param stack: ``(cycles, Y, X, bases)`` intensities.
    :param threshold: the score rise a read needs (``ops_read_threshold``).
    :param margin: border, in pixels, where reads are not taken.
    :param sigma: the LoG width.
    :returns: ``(peaks, values)``: ``(N, 2)`` int64 ``(y, x)``, strongest
        first, and ``(N, cycles, bases)`` float32 filtered intensities (the
        3 x 3 maximum at each peak).
    """
    from scipy import ndimage

    from .ops_sbs import estimate_read_locations, extract_bases, find_peaks

    n_cycles, height, width, n_bases = stack.shape
    filtered = np.empty((n_cycles, n_bases, height, width), np.float32)
    for c in range(n_cycles):
        for k in range(n_bases):
            filtered[c, k] = np.clip(-ndimage.gaussian_laplace(
                stack[c, :, :, k].astype(np.float32), sigma), 0, None)
    score = np.zeros((height, width), np.float32)
    for k in range(n_bases):
        score += estimate_read_locations(filtered[:, k:k + 1])
    score /= np.float32(n_bases)
    peaks = find_peaks(score, min_distance=2, gpu=False)
    if peaks.size:
        strength = score - ndimage.minimum_filter(score, size=5)
        keep = strength[peaks[:, 0], peaks[:, 1]] > float(threshold)
        keep &= (peaks[:, 0] >= margin) & (peaks[:, 0] < height - margin)
        keep &= (peaks[:, 1] >= margin) & (peaks[:, 1] < width - margin)
        peaks = peaks[keep]
    values = extract_bases(filtered, peaks, window=1)
    return peaks, values


def decode_prefix_site(stack: np.ndarray, library: PrefixLibrary, *,
                       bases: Sequence[str] = ("G", "T", "A", "C"),
                       labels: Optional[np.ndarray] = None,
                       threshold: float = 315.0, crosstalk: str = "none",
                       metric: str = "hamming", max_distance: int = 1,
                       control: Optional[PrefixLibrary] = None) -> pd.DataFrame:
    """Read, call and match one site's sequencing spots.

    :param stack: ``(cycles, Y, X, bases)`` from cycle 1, as
        :func:`spacr.cpg_cropped.load_cpg_site` returns it.
    :param library: the whitelist.
    :param bases: the base letter of each plane, in order.
    :param labels: a ``(Y, X)`` cell label image; each read takes the label
        under it (0 outside every cell). None leaves ``cell`` at 0.
    :param threshold: the read score rise (:func:`detect_spots`).
    :param crosstalk: :func:`compensate_prefix` mode.
    :param metric: :func:`match_prefix_reads` metric.
    :param max_distance: :func:`match_prefix_reads` limit.
    :param control: a shuffled whitelist (:func:`shuffled_library`); adds
        ``control_status``.
    :returns: one row per read: ``y``, ``x``, ``cell``, ``quality`` (the
        smallest per-cycle margin), then :func:`match_prefix_reads`'s
        columns, and ``control_status`` when a control is given.
    :raises ValueError: when ``labels`` is not the stack's grid.
    """
    from .ops_sbs import called_bases

    peaks, values = detect_spots(stack, threshold=threshold)
    values = compensate_prefix(values, crosstalk)
    _data, calls, margin = called_bases(values, bases=tuple(bases), compensate=False,
                                        normalise=True, gpu=False)
    if len(calls):
        worst = np.where(np.isnan(margin), np.inf, margin).min(axis=1)
        quality = np.where(np.isfinite(worst), worst, 0.0).astype(np.float32)
    else:
        quality = np.zeros(0, np.float32)
    cell = np.zeros(len(peaks), np.int64)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != stack.shape[1:3]:
            raise ValueError(f"the cell labels are {labels.shape}, the sequencing "
                             f"stack {stack.shape[1:3]}")
        if len(peaks):
            cell = labels[peaks[:, 0], peaks[:, 1]].astype(np.int64)
    matched = match_prefix_reads(calls, library, metric=metric, max_distance=max_distance)
    frame = pd.DataFrame({"y": peaks[:, 0] if len(peaks) else np.zeros(0, np.int64),
                          "x": peaks[:, 1] if len(peaks) else np.zeros(0, np.int64),
                          "cell": cell, "quality": quality})
    frame = pd.concat([frame, matched], axis=1)
    if control is not None:
        frame["control_status"] = match_prefix_reads(
            calls, control, metric=metric, max_distance=max_distance)["status"].to_numpy()
    return frame


def read_fractions(reads: pd.DataFrame) -> Dict[str, Any]:
    """The share of reads in each status, and the control's match rate.

    :param reads: rows of :func:`decode_prefix_site`.
    :returns: ``reads``, a count and ``frac_<status>`` per status,
        ``frac_matched`` (exact or corrected), and with a control
        ``control_matched`` and ``control_frac_matched``.
    """
    n = int(len(reads))
    out: Dict[str, Any] = {"reads": n}
    counts = reads["status"].value_counts() if n else {}
    for name in READ_STATUSES:
        value = int(counts.get(name, 0)) if n else 0
        out[name] = value
        out[f"frac_{name}"] = value / n if n else 0.0
    out["frac_matched"] = (out["exact"] + out["corrected"]) / n if n else 0.0
    if "control_status" in reads.columns:
        hit = int(reads["control_status"].isin(("exact", "corrected")).sum())
        out["control_matched"] = hit
        out["control_frac_matched"] = hit / n if n else 0.0
    return out


def prefix_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The item-920 settings, checked and with defaults filled in.

    :param settings: the OPS settings: ``ops_library``,
        ``ops_library_sequence_col``, ``ops_library_gene_col``,
        ``ops_prefix_length``, ``ops_sbs_cycles``, ``ops_cycle_mask``,
        ``ops_prefix_collision``, ``ops_prefix_metric``,
        ``ops_prefix_max_distance``, ``ops_prefix_crosstalk``,
        ``ops_quality_min_reads``, ``ops_prefix_quality_method``,
        ``ops_random_barcode_control``, ``ops_control_seed``,
        ``ops_prefix_masks`` and ``ops_read_threshold``.
    :returns: the normalised values under the same keys.
    :raises ValueError: without a library, or for an unknown choice.
    """
    library = settings.get("ops_library")
    if not library:
        raise ValueError("ops_library_mode 'spacer_prefix' needs ops_library, the "
                         "library table (for example Barcodes.csv)")
    mask = settings.get("ops_cycle_mask") or []
    if isinstance(mask, str):
        mask = [int(p) for p in mask.replace(";", ",").split(",") if p.strip()]
    out = {
        "ops_library": str(library),
        "ops_library_sequence_col": str(settings.get("ops_library_sequence_col") or "sgRNA"),
        "ops_library_gene_col": str(settings.get("ops_library_gene_col") or "gene_symbol"),
        "ops_prefix_length": int(settings.get("ops_prefix_length", 10)),
        "ops_sbs_cycles": int(settings.get("ops_sbs_cycles", 12)),
        "ops_cycle_mask": sorted({int(c) for c in mask}),
        "ops_prefix_collision": str(settings.get("ops_prefix_collision") or "exclude"),
        "ops_prefix_metric": str(settings.get("ops_prefix_metric") or "hamming"),
        "ops_prefix_max_distance": int(settings.get("ops_prefix_max_distance", 1)),
        "ops_prefix_crosstalk": str(settings.get("ops_prefix_crosstalk") or "none"),
        "ops_quality_min_reads": int(settings.get("ops_quality_min_reads", 2)),
        "ops_prefix_quality_method": str(settings.get("ops_prefix_quality_method") or "spacr"),
        "ops_random_barcode_control": bool(settings.get("ops_random_barcode_control", True)),
        "ops_control_seed": int(settings.get("ops_control_seed", 0) or 0),
        "ops_prefix_masks": settings.get("ops_prefix_masks") or None,
        "ops_read_threshold": float(settings.get("ops_read_threshold", 315.0)),
    }
    if out["ops_prefix_crosstalk"] not in _CROSSTALK:
        raise ValueError(f"ops_prefix_crosstalk must be one of {list(_CROSSTALK)}")
    if out["ops_prefix_metric"] not in _METRICS:
        raise ValueError(f"ops_prefix_metric must be one of {list(_METRICS)}")
    if out["ops_prefix_quality_method"] not in QUALITY_METHODS:
        raise ValueError(f"ops_prefix_quality_method must be one of {list(QUALITY_METHODS)}")
    return out


def site_labels(template: Optional[str], *, plate: str, well: str, well_id: str,
                site: int, merged: Optional[str] = None) -> Optional[np.ndarray]:
    """The cell label image of one site, or None.

    :param template: ``ops_prefix_masks``: a path with ``{plate}``,
        ``{well}`` (``Well1``), ``{well_id}`` (``W1``), ``{well_number}`` and
        ``{site}`` fields, to a TIFF, ``.npy`` or ``.npz`` (``cells``) label
        image. None falls back to ``merged``.
    :param plate: the plate.
    :param well: the well folder name.
    :param well_id: the normalised well id.
    :param site: the site.
    :param merged: a ``merged/`` site file; its ``cells`` array is used when
        it has one.
    :returns: the labels, or None when neither source has them.
    :raises ValueError: when the template names a file that does not exist.
    """
    if template:
        path = str(template).format(plate=plate, well=well, well_id=well_id,
                                    well_number=well_id[1:], site=int(site))
        if not os.path.isfile(path):
            raise ValueError(f"no cell labels at {path} (ops_prefix_masks)")
        if path.endswith(".npz"):
            with np.load(path) as saved:
                return saved["cells"]
        if path.endswith(".npy"):
            return np.load(path)
        import tifffile

        return tifffile.imread(path)
    if merged and os.path.isfile(merged):
        with np.load(merged) as saved:
            if "cells" in saved.files:
                return saved["cells"]
    return None


def run_prefix_decode(settings: Mapping[str, Any],
                      report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The ``prefix_decode`` phase: decode every indexed site against spacer prefixes.

    Reads the cropped-site layout (item 919, ``ops_layout='cpg_cropped'``),
    builds the whitelist, decodes each complete site's sequencing stack
    (cycles 1 to the prefix's last cycle), takes each read's cell from
    ``ops_prefix_masks`` (or the site's ``merged/`` labels), calls cells and
    writes, under ``<dst_root>/results/ops_prefix/``:

    * ``library_prefix_collisions.csv`` and ``library_summary.json``;
    * ``quality_classes.json`` (the quality method, its class definitions
      and the settings);
    * ``reads_<plate>_<well_id>.parquet`` (one row per read) and
      ``cells_<plate>_<well_id>.csv`` (:func:`prefix_cell_calls`);
    * ``site_summary.csv`` (per site: :func:`read_fractions`, cells with
      reads, cells with an exact read, cells per class);
    * ``cells_per_guide.csv`` and ``cells_per_gene.csv`` (perfect and great
      cells).

    :param settings: the OPS settings: ``dst_root``, the ``cpg_*`` reader
        settings and the item-920 settings (:func:`prefix_settings`).
    :param report: called with progress lines.
    :returns: the output paths, the library summary and the run's totals.
    :raises ValueError: without ``dst_root`` or a library, or when the
        layout has fewer sequencing cycles than the prefix needs.
    """
    from . import __version__
    from .cpg_cropped import cpg_site_index, load_cpg_site, merged_path
    from .tabular import write_table

    say = report or (lambda text: None)
    destination = settings.get("dst_root")
    if not destination:
        raise ValueError("the prefix_decode phase needs dst_root")
    started = time.perf_counter()
    chosen = prefix_settings(settings)
    library = spacer_prefix_library(
        chosen["ops_library"], sequence_col=chosen["ops_library_sequence_col"],
        gene_col=chosen["ops_library_gene_col"], prefix_length=chosen["ops_prefix_length"],
        n_cycles=chosen["ops_sbs_cycles"], cycle_mask=chosen["ops_cycle_mask"],
        collision=chosen["ops_prefix_collision"])
    control = (shuffled_library(library, chosen["ops_control_seed"])
               if chosen["ops_random_barcode_control"] else None)
    built = cpg_site_index(settings, md5=False)
    pinned = built["pinned"]
    cycles = max(library.positions)
    if cycles > pinned["cpg_sbs_cycles"]:
        raise ValueError(f"the prefix needs cycles 1..{cycles}; cpg_sbs_cycles is "
                         f"{pinned['cpg_sbs_cycles']}")
    folder = os.path.join(str(destination), "results", "ops_prefix")
    os.makedirs(folder, exist_ok=True)
    write_table(library.collisions, os.path.join(folder, "library_prefix_collisions.csv"),
                canonicalise=False)
    with open(os.path.join(folder, "library_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(library.summary, handle, indent=1)
    method = chosen["ops_prefix_quality_method"]
    class_names = RECIPE_SIMPLE_CLASSES if method == "recipe_simple" else QUALITY_CLASSES
    with open(os.path.join(folder, "quality_classes.json"), "w", encoding="utf-8") as handle:
        json.dump({"method": method, "classes": class_names,
                   "source": RECIPE_SIMPLE_SOURCE if method == "recipe_simple" else None,
                   "settings": chosen,
                   "bases": pinned["cpg_sbs_bases"], "spacr_version": __version__},
                  handle, indent=1, default=str)
    index = built["index"]
    site_rows, all_cells = [], []
    for (plate, well, well_id), sites in index.groupby(["plate", "well", "well_id"], sort=True):
        reads_frames, cell_frames = [], []
        for site in sorted(sites["site"].unique()):
            loaded = load_cpg_site(index, plate, well, int(site), sbs_cycles=cycles,
                                   sbs_bases=pinned["cpg_sbs_bases"], sbs_dapi=None,
                                   parts=("sbs",))
            labels = site_labels(chosen["ops_prefix_masks"], plate=plate, well=well,
                                 well_id=well_id, site=int(site),
                                 merged=merged_path(str(destination), plate, well_id, int(site)))
            reads = decode_prefix_site(
                loaded["sbs"], library, bases=pinned["cpg_sbs_bases"], labels=labels,
                threshold=chosen["ops_read_threshold"], crosstalk=chosen["ops_prefix_crosstalk"],
                metric=chosen["ops_prefix_metric"],
                max_distance=chosen["ops_prefix_max_distance"], control=control)
            del loaded
            reads.insert(0, "site", int(site))
            reads.insert(0, "well", well_id)
            reads.insert(0, "plate", plate)
            cells = prefix_cell_calls(reads, min_reads=chosen["ops_quality_min_reads"],
                                      method=method)
            classes = cells["quality"].value_counts() if len(cells) else {}
            site_rows.append({"plate": plate, "well": well_id, "site": int(site),
                              **read_fractions(reads),
                              "labels": labels is not None,
                              "cells_with_reads": int(len(cells)),
                              "cells_with_exact_read": int(cells["has_exact_read"].sum())
                              if len(cells) else 0,
                              **{f"cells_{name}": int(classes.get(name, 0)) if len(cells) else 0
                                 for name in class_names}})
            reads_frames.append(reads)
            cell_frames.append(cells)
            say(f"prefix_decode {plate} {well_id} site {site}: {len(reads)} reads, "
                f"{len(cells)} cells")
        reads_all = pd.concat(reads_frames, ignore_index=True)
        cells_all = pd.concat(cell_frames, ignore_index=True)
        write_table(reads_all, os.path.join(folder, f"reads_{plate}_{well_id}.parquet"),
                    canonicalise=False)
        write_table(cells_all, os.path.join(folder, f"cells_{plate}_{well_id}.csv"),
                    canonicalise=False)
        all_cells.append(cells_all)
    summary = pd.DataFrame(site_rows)
    write_table(summary, os.path.join(folder, "site_summary.csv"), canonicalise=False)
    cells = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    called = cells[cells["quality"].isin(("perfect", "great"))] if len(cells) else cells
    per_guide = (called.groupby(["guide", "gene"]).size().rename("cells").reset_index()
                 if len(called) else pd.DataFrame(columns=["guide", "gene", "cells"]))
    per_gene = (called.groupby("gene").size().rename("cells").reset_index()
                if len(called) else pd.DataFrame(columns=["gene", "cells"]))
    write_table(per_guide, os.path.join(folder, "cells_per_guide.csv"), canonicalise=False)
    write_table(per_gene, os.path.join(folder, "cells_per_gene.csv"), canonicalise=False)
    totals = {key: int(summary[key].sum()) for key in
              ("reads", "exact", "corrected", "ambiguous", "collision", "out_of_library")} \
        if len(summary) else {}
    return {"folder": folder, "library": library.summary, "sites": int(len(summary)),
            "totals": totals, "cells": int(len(cells)), "called_cells": int(len(called)),
            "seconds": round(time.perf_counter() - started, 1)}
