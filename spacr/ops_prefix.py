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

import itertools
import json
import math
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

#: The per-cell quality methods (``ops_prefix_quality_method``). The third,
#: ``'weighted_majority'`` (item 988, alpha), calls a cell's top barcode by
#: its share of the summed read weights (:data:`WEIGHTED_CLASSES`).
QUALITY_METHODS = ("spacr", "recipe_simple", "weighted_majority")

#: The classes of the ``'weighted_majority'`` method (item 988). A read's
#: weight is its quality term times its distance term
#: (``ops_prefix_distance_weight``); a minority barcode that is the top
#: barcode of a touching cell has its weight multiplied by
#: ``ops_read_neighbour_discount``.
WEIGHTED_CLASSES = {
    "perfect": "one barcode in the cell, holding the whole read weight",
    "great": "several barcodes; the top barcode holds at least "
             "ops_prefix_weighted_min_share of the read weight and more than "
             "any other barcode",
    "ambiguous": "matched reads, but no barcode holds the share, or two "
                 "barcodes tie for the most weight",
    "none": "no read matched a callable barcode",
}

#: The read-weight modes of ``ops_prefix_distance_weight`` and
#: ``ops_read_distance_weight`` (item 988), the first being the default:
#: ``'off'`` keeps the quality term only, ``'decay'`` multiplies it by the
#: distance term, and ``'distance_only'`` uses the distance term alone.
READ_WEIGHT_MODES = ("off", "decay", "distance_only")

#: The tuning-grid keys of the weighted majority (item 988), in grid order
#: after the read threshold, read-quality floor and mismatch axes. A decay
#: length of 0 turns the distance term off, a discount of 1 turns the
#: neighbour discount off, and a quality term of 0 weights every read 1.
WEIGHT_GRID_KEYS = ("ops_prefix_distance_decay_px", "ops_read_neighbour_discount",
                    "ops_prefix_weighted_quality")

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


#: The gene a decoy prefix calls (item 942). A cell whose top barcode is a
#: decoy is counted as a decoy call and never written as a genotype.
DECOY_GENE = "DECOY"


def prefix_decoy_library(library: PrefixLibrary, k: int = 4,
                         seed: int = 1) -> PrefixLibrary:
    """The whitelist with decoy prefixes added, for a decoy FDR (item 942).

    The decoys are :func:`spacr.ops_sbs.decoy_library` of every key of the
    whitelist, collision keys included, so each decoy lies at least two
    edits from every prefix. They join the whitelist as callable keys of
    gene :data:`DECOY_GENE` (level ``'decoy'``), so a read is matched against
    guides and decoys together and a read as close to a decoy as to a guide
    is ambiguous. Decoys have no one-base extension, so a Levenshtein indel
    never lands on one.

    :param library: from :func:`spacer_prefix_library`.
    :param k: decoys per prefix (``ops_prefix_decoy_k``).
    :param seed: the decoys' seed (``ops_prefix_decoy_seed``).
    :returns: a :class:`PrefixLibrary` holding the keys and the decoys; its
        summary adds ``decoys``, ``decoy_k`` and ``decoy_seed``.
    """
    from .ops_sbs import decoy_library

    decoys = [key for key in decoy_library(library.keys, k=int(k), seed=int(seed))
              if key not in library.guides]
    out = _with_prefix_decoys(library, decoys)
    out.summary = {**library.summary, "decoys": len(decoys),
                   "decoy_k": int(k), "decoy_seed": int(seed)}
    return out


def _with_prefix_decoys(library: PrefixLibrary, decoys) -> PrefixLibrary:
    """Add named control sequences to the existing prefix whitelist representation.

    :param library: the real prefix whitelist, including collision metadata.
    :param decoys: control sequences assigned the existing decoy gene and level.
    :returns: a whitelist for the unchanged prefix matcher and voters.
    """
    keys = sorted(set(library.keys) | set(decoys))
    callable_keys = dict(library.callable_keys)
    guides = dict(library.guides)
    genes = dict(library.genes)
    for key in decoys:
        callable_keys[key] = (key, DECOY_GENE, "decoy")
        guides[key] = [key]
        genes[key] = [DECOY_GENE]
    return PrefixLibrary(positions=library.positions, keys=keys, guides=guides,
                         genes=genes, callable_keys=callable_keys,
                         collisions=library.collisions, extended=library.extended,
                         summary=library.summary)

#: The gene a near-miss decoy prefix calls (item 949). A cell whose top
#: barcode is a near-miss decoy is a decoy call, counted apart from the
#: far decoys of :data:`DECOY_GENE` and never written as a genotype.
NEARMISS_GENE = "DECOY_NEARMISS"


def prefix_nearmiss_decoys(library: PrefixLibrary, k: int = 4,
                           seed: int = 1) -> PrefixLibrary:
    """The whitelist with near-miss decoys added, one edit from a prefix (item 949).

    The far decoys of :func:`prefix_decoy_library` lie two or more edits
    from every prefix, so they never catch a read error that lands one edit
    from a guide. A near-miss decoy is one such neighbour: for each library
    prefix (decoys excluded), in key order, one generator
    (``numpy.random.default_rng(seed)``) draws a cycle and a different base;
    the result is kept when it is one edit from that prefix and from no other
    key (prefix, far decoy or near-miss decoy already kept), until ``k`` are
    kept or ``8 * k`` draws are spent. Near-miss decoys join the whitelist
    as callable keys of gene :data:`NEARMISS_GENE` (level ``'nearmiss'``),
    so a read that matches one, exactly or as its single closest key, is a
    near-miss decoy call. Exact-only matching loses no real read to them;
    with one mismatch a read one edit from both a prefix and a near-miss
    decoy is ambiguous. They have no one-base extension.

    :param library: from :func:`spacer_prefix_library` or
        :func:`prefix_decoy_library`.
    :param k: near-miss decoys per prefix (``ops_prefix_nearmiss_decoy_k``).
    :param seed: the generator's seed (``ops_prefix_decoy_seed``).
    :returns: a :class:`PrefixLibrary` holding the keys and the near-miss
        decoys; its summary adds ``nearmiss_decoys``, ``nearmiss_k`` and
        ``nearmiss_seed``.
    :raises ValueError: for ``k`` below 1.
    """
    if int(k) < 1:
        raise ValueError("ops_prefix_nearmiss_decoy_k must be 1 or more to add near-miss decoys")
    sources = [key for key in library.keys
               if library.callable_keys.get(key, ("", "", ""))[2] != "decoy"]
    letters = sorted({base for key in library.keys for base in key}) or list("ACGT")
    neighbour: Dict[Tuple[int, str], Optional[str]] = {}

    def occupy(key: str) -> None:
        """Index a kept key by each of its one-deletion variants."""
        for position in range(len(key)):
            slot = (position, key[:position] + key[position + 1:])
            neighbour[slot] = key if slot not in neighbour else None

    for key in library.keys:
        occupy(key)
    taken = set(library.keys)
    rng = np.random.default_rng(int(seed))
    nearmiss: List[str] = []
    for key in sources:
        kept = 0
        for _ in range(8 * int(k)):
            if kept >= int(k):
                break
            position = int(rng.integers(len(key)))
            options = [base for base in letters if base != key[position]]
            candidate = key[:position] + options[int(rng.integers(len(options)))] + \
                key[position + 1:]
            if candidate in taken:
                continue
            near = {neighbour.get((p, candidate[:p] + candidate[p + 1:]), "")
                    for p in range(len(candidate))} - {""}
            if near != {key}:
                continue
            taken.add(candidate)
            occupy(candidate)
            nearmiss.append(candidate)
            kept += 1
    callable_keys = dict(library.callable_keys)
    guides = dict(library.guides)
    genes = dict(library.genes)
    for key in nearmiss:
        callable_keys[key] = (key, NEARMISS_GENE, "nearmiss")
        guides[key] = [key]
        genes[key] = [NEARMISS_GENE]
    return PrefixLibrary(positions=library.positions, keys=sorted(taken), guides=guides,
                         genes=genes, callable_keys=callable_keys,
                         collisions=library.collisions, extended=library.extended,
                         summary={**library.summary, "nearmiss_decoys": len(nearmiss),
                                  "nearmiss_k": int(k), "nearmiss_seed": int(seed)})


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


#: The lookups of the last whitelists matched, so a run that matches many
#: sites against one whitelist builds them once.
_INDEX_CACHE: List[Tuple[Any, ...]] = []

#: How many whitelists' lookups :data:`_INDEX_CACHE` keeps: the library,
#: the control and item 949's shuffled whitelists fit at once.
_INDEX_CACHE_SIZE = 6


def _match_indexes(library: PrefixLibrary, metric: str, max_distance: int):
    """The exact, deletion, indel and table lookups of one whitelist, cached.

    :param library: the whitelist.
    :param metric: ``'hamming'`` or ``'levenshtein'``.
    :param max_distance: the largest distance corrected.
    :returns: ``(known, deletion, dropped, shorter, entries, table)``.
    """
    for held, held_metric, held_distance, indexes in _INDEX_CACHE:
        if held is library and held_metric == metric and held_distance == max_distance:
            return indexes
    known = set(library.keys)
    deletion = _deletion_index(library.keys) if max_distance >= 1 else {}
    dropped, shorter = (_levenshtein_indexes(library) if metric == "levenshtein"
                        and max_distance >= 1 else ({}, {}))
    entries = list(library.keys)
    width = len(library.positions)
    table = (np.frombuffer("".join(entries).encode("ascii"), dtype=np.uint8)
             .reshape(len(entries), width) if entries else None)
    indexes = (known, deletion, dropped, shorter, entries, table)
    _INDEX_CACHE.insert(0, (library, metric, max_distance, indexes))
    del _INDEX_CACHE[_INDEX_CACHE_SIZE:]
    return indexes


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
    known, deletion, dropped, shorter, entries, table = _match_indexes(
        library, metric, max_distance)
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


#: The barcode rankings ``ops_call_sort`` names (item 961, alpha).
_CALL_SORTS = ("count", "quality", "peak")


def _sorted_call(matched: pd.DataFrame, counts: Mapping[str, List[int]], sort: str,
                 min_reads: int) -> Tuple[List[Tuple[str, List[int]]], str]:
    """One cell's barcodes ranked by ``ops_call_sort``, and its quality class.

    :param matched: the cell's exact and corrected reads (``prefix``,
        ``quality`` and, for ``'peak'``, ``peak``).
    :param counts: ``barcode -> [exact, corrected]``.
    :param sort: a key of :data:`_CALL_SORTS`.
    :param min_reads: exact reads a ``perfect`` cell needs.
    :returns: ``(ranked, quality)``: ``(barcode, counts)`` best first, and a
        key of :data:`QUALITY_CLASSES`.
    """
    if not counts:
        return [], "none"
    if sort == "count":
        score = {b: float(sum(v)) for b, v in counts.items()}
    else:
        column = "quality" if sort == "quality" else "peak"
        grouped = matched.groupby("prefix", sort=False)[column]
        values = grouped.mean() if sort == "quality" else grouped.max()
        score = {b: float(values.get(b, 0.0)) for b in counts}
    ranked = sorted(counts.items(), key=lambda kv: (-score[kv[0]], -sum(kv[1]), kv[0]))
    if len(ranked) > 1 and score[ranked[0][0]] == score[ranked[1][0]]:
        return ranked, "ambiguous"
    if len(ranked) == 1 and ranked[0][1][0] >= int(min_reads):
        return ranked, "perfect"
    return ranked, "great"


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


def _minority_ignored(matched: pd.DataFrame, cell_cols: Sequence[str],
                      min_reads: int) -> np.ndarray:
    """Which matched reads the minority-read rule (item 967) leaves out of the cell call.

    :param matched: the exact and corrected reads, with ``cell_cols`` and
        ``prefix``.
    :param cell_cols: the columns naming a cell.
    :param min_reads: ``ops_prefix_minority_min_reads``; 0 and 1 leave out
        nothing.
    :returns: one flag per row of ``matched``: True for a read of a barcode
        with fewer than ``min_reads`` reads in its cell while another barcode
        of that cell has strictly more reads. Barcodes tied for the most
        reads are always kept, so a tie still blocks the call.
    """
    if int(min_reads) <= 1 or matched.empty:
        return np.zeros(len(matched), bool)
    cell_cols = list(cell_cols)
    per_barcode = matched.groupby(cell_cols + ["prefix"], sort=False)["prefix"].transform("size")
    most = per_barcode.groupby([matched[c] for c in cell_cols], sort=False).transform("max")
    return ((per_barcode < int(min_reads)) & (per_barcode < most)).to_numpy(bool)


def _class_counts(matched: pd.DataFrame) -> Dict[str, List[int]]:
    """``barcode -> [exact reads, corrected reads]`` of one cell's matched reads.

    :param matched: the cell's exact and corrected reads.
    :returns: the counts, in first-read order.
    """
    counts: Dict[str, List[int]] = {}
    for prefix, kind in zip(matched["prefix"], matched["status"]):
        slot = counts.setdefault(prefix, [0, 0])
        slot[0 if kind == "exact" else 1] += 1
    return counts


def _touching_cells(labels: np.ndarray) -> np.ndarray:
    """The pairs of cells whose labels touch, in both orders (item 988).

    Two cells touch when a pixel of one is the 4-connected neighbour of a
    pixel of the other.

    :param labels: a ``(Y, X)`` cell label image, 0 for background.
    :returns: ``(P, 2)`` int64 ``(cell, neighbour)`` rows, each pair in both
        orders, sorted and without repeats.
    """
    lab = np.asarray(labels)
    a = np.concatenate([lab[1:, :].ravel(), lab[:, 1:].ravel()])
    b = np.concatenate([lab[:-1, :].ravel(), lab[:, :-1].ravel()])
    keep = (a != b) & (a > 0) & (b > 0)
    if not keep.any():
        return np.zeros((0, 2), np.int64)
    pairs = np.unique(np.stack([a[keep], b[keep]], 1).astype(np.int64), axis=0)
    return np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)


def _read_nucleus_gaps(y: np.ndarray, x: np.ndarray, cell: np.ndarray,
                       cells: np.ndarray, nuclei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each read's distance to the nucleus of its own cell, in pixels (item 988).

    The distance is 0 inside the nucleus and the distance to the nearest
    nucleus pixel outside it. When the nearest nucleus belongs to another
    cell, or the cell has none, the read gets a fallback distance instead:
    its distance to the cell's centroid minus the site's median nucleus
    radius (the radius of a disc of the median nucleus area), at least 0.

    :param y: the reads' rows.
    :param x: the reads' columns.
    :param cell: each read's cell label, 0 outside every cell.
    :param cells: the site's cell labels.
    :param nuclei: the site's nucleus labels; a nucleus carries the label of
        its cell.
    :returns: ``(gap, fallback)``: float distances, NaN for reads outside
        every cell, and True where the fallback was used.
    :raises ValueError: when the two label images differ in shape.
    """
    from scipy import ndimage

    cells = np.asarray(cells)
    nuclei = np.asarray(nuclei)
    if cells.shape != nuclei.shape:
        raise ValueError(f"the nucleus labels are {nuclei.shape}, the cell labels "
                         f"{cells.shape}")
    y = np.asarray(y, np.int64)
    x = np.asarray(x, np.int64)
    cell = np.asarray(cell, np.int64)
    gap = np.full(len(cell), np.nan)
    fallback = np.zeros(len(cell), bool)
    inside = cell > 0
    if not inside.any():
        return gap, fallback
    n = int(cells.max())
    area = np.bincount(cells.ravel(), minlength=n + 1).astype(float)
    rows, cols = np.indices(cells.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        cy = ndimage.sum_labels(rows, cells, np.arange(n + 1)) / area
        cx = ndimage.sum_labels(cols, cells, np.arange(n + 1)) / area
    del rows, cols
    nuclear_area = np.bincount(nuclei.ravel())[1:]
    radius = (float(np.median(np.sqrt(nuclear_area[nuclear_area > 0] / np.pi)))
              if (nuclear_area > 0).any() else 0.0)
    distance, (iy, ix) = ndimage.distance_transform_edt(nuclei == 0, return_indices=True)
    yy, xx, cc = y[inside], x[inside], cell[inside]
    own = nuclei[iy[yy, xx], ix[yy, xx]] == cc
    centre = np.maximum(np.hypot(yy - cy[cc], xx - cx[cc]) - radius, 0)
    gap[inside] = np.where(own, distance[yy, xx], centre)
    fallback[inside] = ~own
    return gap, fallback


def _weight_terms(mode: str, quality: bool, decay_px: float) -> Tuple[bool, float]:
    """The quality term and the decay length one weight setting uses (item 988).

    :param mode: ``ops_prefix_distance_weight``, a name in
        :data:`READ_WEIGHT_MODES`.
    :param quality: ``ops_prefix_weighted_quality``.
    :param decay_px: the decay length; 0 turns the distance term off, so a
        caller passes 0 while ``mode`` is ``'off'`` and the grid does not
        try a length.
    :returns: ``(quality term on, decay length)``; ``'distance_only'``
        turns the quality term off.
    """
    return bool(quality) and mode != "distance_only", float(decay_px) if decay_px > 0 else 0.0


def _read_weights(quality: np.ndarray, gap: Optional[np.ndarray], *, quality_term: bool,
                  decay_px: float, interior_px: float = 0.0,
                  min_weight: float = 0.05) -> np.ndarray:
    """Each read's weight in the weighted majority (item 988).

    The weight is a quality term times a distance term. The quality term is
    ``min(1, quality) + 0.05``, the weight of the OPS barcode decode's
    quality vote, or 1. The distance term is 1 up to ``interior_px`` from
    the nucleus and then ``exp(-(g - interior_px) / decay_px)``, at least
    ``min_weight``, or 1 without a decay length.

    :param quality: the reads' qualities.
    :param gap: the reads' nucleus distances (:func:`_read_nucleus_gaps`);
        only read with a decay length.
    :param quality_term: use the quality term.
    :param decay_px: the decay length in pixels; 0 is no distance term.
    :param interior_px: the distance with full weight.
    :param min_weight: the smallest distance term.
    :returns: one float weight per read.
    """
    quality = np.asarray(quality, float)
    q = (np.minimum(1.0, np.maximum(quality, 0.0)) + 0.05 if quality_term
         else np.ones(len(quality)))
    if decay_px > 0 and gap is not None:
        g = np.maximum(np.asarray(gap, float) - float(interior_px), 0.0)
        return q * np.maximum(np.exp(-g / float(decay_px)), float(min_weight))
    return q


def _barcode_groups(cell: np.ndarray, barcode: np.ndarray,
                    neighbours: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The (cell, barcode) groups of matched reads, and the neighbour-bleed flags (item 988).

    :param cell: one integer cell code per read.
    :param barcode: one barcode per read.
    :param neighbours: ``(P, 2)`` ``(cell, neighbour)`` codes of touching
        cells, or None for no neighbour flags.
    :returns: ``group`` (each read's group), ``cell`` and ``barcode`` (codes)
        and ``n`` (reads) per group, sorted by cell, ``starts`` and ``sizes``
        of each cell's groups, ``barcodes`` (the barcode of each code), and
        ``bleed``: True for a group whose barcode has strictly fewer reads
        than the cell's top barcode and is a top barcode, by reads with
        ties kept, of a touching cell.
    """
    names, code = np.unique(np.asarray(barcode).astype(str), return_inverse=True)
    width = max(len(names), 1)
    pair = np.asarray(cell, np.int64) * width + code.astype(np.int64)
    keys, group = np.unique(pair, return_inverse=True)
    g_cell, g_barcode = keys // width, keys % width
    n = np.bincount(group, minlength=len(keys))
    starts = np.flatnonzero(np.r_[True, g_cell[1:] != g_cell[:-1]]) if len(keys) \
        else np.zeros(0, np.int64)
    sizes = np.diff(np.r_[starts, len(keys)])
    most = np.repeat(np.maximum.reduceat(n, starts), sizes) if len(keys) else n
    bleed = np.zeros(len(keys), bool)
    if neighbours is not None and len(neighbours) and len(keys):
        top = keys[n == most]
        tops = pd.DataFrame({"neighbour": top // width, "barcode": top % width})
        pairs = pd.DataFrame({"cell": np.asarray(neighbours)[:, 0].astype(np.int64),
                              "neighbour": np.asarray(neighbours)[:, 1].astype(np.int64)})
        hit = pairs.merge(tops, on="neighbour")
        flagged = hit["cell"].to_numpy(np.int64) * width + hit["barcode"].to_numpy(np.int64)
        bleed = np.isin(keys, flagged) & (n < most)
    return {"group": group, "cell": g_cell, "barcode": g_barcode, "n": n,
            "starts": starts, "sizes": sizes, "barcodes": names, "bleed": bleed}


def _weighted_decision(weights: np.ndarray, groups: Mapping[str, np.ndarray],
                       share: float) -> Tuple[np.ndarray, np.ndarray]:
    """Which cells the weighted majority calls, and each cell's top group (item 988).

    :param weights: the summed weight of each group of :func:`_barcode_groups`.
    :param groups: from :func:`_barcode_groups`.
    :param share: ``ops_prefix_weighted_min_share``.
    :returns: ``(called, top)`` per cell, in the groups' cell order: a cell
        is called when one group holds the most weight alone, the weight is
        above 0, and the group holds at least ``share`` of it; ``top`` is
        the index of the first group with the most weight.
    """
    starts, sizes = groups["starts"], groups["sizes"]
    if not len(starts):
        return np.zeros(0, bool), np.zeros(0, np.int64)
    top = np.maximum.reduceat(weights, starts)
    total = np.add.reduceat(weights, starts)
    is_top = weights == np.repeat(top, sizes)
    ties = np.add.reduceat(is_top.astype(np.int64), starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        called = (ties == 1) & (total > 0) & (top / total >= float(share))
    position = np.where(is_top, np.arange(len(weights)), len(weights))
    return called, np.minimum.reduceat(position, starts)


def _discounted(weights: np.ndarray, groups: Mapping[str, np.ndarray],
                discount: float) -> np.ndarray:
    """The group weights after the neighbour-bleed discount (item 988).

    :param weights: the summed weight of each group.
    :param groups: from :func:`_barcode_groups`.
    :param discount: ``ops_read_neighbour_discount``; 1 changes nothing.
    :returns: the weights, a flagged group's multiplied by ``discount``.
    """
    if float(discount) == 1.0:
        return weights
    return np.where(groups["bleed"], weights * float(discount), weights)


def _weighted_counts(reads: pd.DataFrame, *, neighbours: Optional[np.ndarray],
                     mode: str, decay: Sequence[float], discount: Sequence[float],
                     quality: Sequence[float], share: float, interior_px: float,
                     min_weight: float, opportunity_profiles=None):
    """Real and decoy cells the weighted majority calls at every weight setting (item 988).

    :param reads: one site's rows with ``cell``, ``status``, ``prefix``,
        ``gene``, ``quality`` and, for a decay length above 0, ``nuc_gap_px``.
    :param neighbours: the site's touching cells (:func:`_touching_cells`).
    :param mode: ``ops_prefix_distance_weight``.
    :param decay: the decay lengths; 0 is no distance term.
    :param discount: the neighbour discounts.
    :param quality: the quality-term switches, 1 on and 0 off.
    :param share: ``ops_prefix_weighted_min_share``.
    :param interior_px: ``ops_prefix_distance_interior_px``.
    :param min_weight: ``ops_prefix_distance_min_weight``.
    :param opportunity_profiles: optional shared guide profiles for opportunity totals.
    :returns: ``(decay, discount, quality, 2)`` int64 real and decoy calls,
        or a dict of counts and opportunity grids when profiles are supplied.
    """
    out = np.zeros((len(decay), len(discount), len(quality), 2), np.int64)
    opportunities = {f'opp{d}_{name}': np.zeros(out.shape[:-1], np.int64)
                     for name in (opportunity_profiles or {}) if name != 'guides'
                     for d in (1, 2, 3)}
    matched = reads[(reads["cell"] > 0) & reads["status"].isin(("exact", "corrected"))]
    if matched.empty:
        return {'counts': out, 'opportunities': opportunities} if opportunity_profiles else out
    use_bleed = any(float(b) != 1.0 for b in discount)
    groups = _barcode_groups(matched["cell"].to_numpy(np.int64), matched["prefix"].to_numpy(),
                             neighbours if use_bleed else None)
    gene = dict(zip(matched["prefix"].astype(str), matched["gene"]))
    labels = np.array([gene.get(name) for name in groups["barcodes"]], dtype=object)
    decoy = labels == DECOY_GENE
    near = labels == NEARMISS_GENE
    gap = matched["nuc_gap_px"].to_numpy(float) if "nuc_gap_px" in matched.columns else None
    q_values = matched["quality"].to_numpy(float)
    for a, length in enumerate(decay):
        for c, q_on in enumerate(quality):
            term, used = _weight_terms(mode, bool(q_on), float(length))
            per_read = _read_weights(q_values, gap, quality_term=term, decay_px=used,
                                     interior_px=interior_px, min_weight=min_weight)
            summed = np.bincount(groups["group"], weights=per_read,
                                 minlength=len(groups["n"]))
            for b, value in enumerate(discount):
                called, top = _weighted_decision(_discounted(summed, groups, value),
                                                 groups, share)
                top_bc = groups["barcode"][top]
                is_decoy = decoy[top_bc] & called
                real = called & ~decoy[top_bc] & ~near[top_bc]
                out[a, b, c] = (int(real.sum()), int(is_decoy.sum()))
                if opportunity_profiles:
                    values = _prefix_opportunities(groups['barcodes'][top_bc], real,
                                                    opportunity_profiles)
                    for key, value in values.items():
                        opportunities[key][a, b, c] = value
    return {'counts': out, 'opportunities': opportunities} if opportunity_profiles else out


def _guide_multiplicity(reads: pd.DataFrame, neighbours: Optional[np.ndarray],
                        min_reads: int, keys: Sequence[str]) -> Tuple[Dict[str, int], pd.DataFrame]:
    """One site's guide multiplicity counts and its multi-guide cells (item 988).

    Only library guides count as guides; decoy barcodes are counted apart.
    The counts follow two rules: every matched read, and only guides with at
    least ``min_reads`` reads. A cell with two or more guides and one clear
    top guide by reads is a bleed candidate. Its runner-up guide is
    neighbour bleed when it is a top guide of a touching cell.

    :param reads: the site's rows with ``keys``, ``status``, ``prefix``,
        ``guide``, ``gene``, ``level`` and ``quality``.
    :param neighbours: the site's touching cells (:func:`_touching_cells`),
        or None.
    :param min_reads: ``ops_multi_guide_min_reads``.
    :param keys: the columns naming a cell, ``cell`` last.
    :returns: ``(counts, multi)``. ``counts`` holds ``all_<k>`` and
        ``min_<k>`` cells with k guides (k = 1, 2, 3 and ``ge4``),
        ``all_decoy`` and ``min_decoy`` cells with a decoy barcode,
        ``bleed_tested`` and ``bleed_neighbour``. ``multi`` has one row per
        cell with two or more guides of ``min_reads`` reads, none of them
        neighbour bleed: ``keys``, ``n_guides``, ``guides``, ``genes``,
        ``reads`` and ``median_quality``, the last four joined by ``|`` in
        read order.
    """
    keys = list(keys)
    counts: Dict[str, int] = {f"{rule}_{k}": 0 for rule in ("all", "min")
                              for k in ("1", "2", "3", "ge4", "decoy")}
    counts.update({"bleed_tested": 0, "bleed_neighbour": 0})
    columns = keys + ["n_guides", "guides", "genes", "reads", "median_quality"]
    matched = reads[(reads["cell"] > 0) & reads["status"].isin(("exact", "corrected"))]
    if matched.empty:
        return counts, pd.DataFrame(columns=columns)
    is_decoy = matched["gene"].isin((DECOY_GENE, NEARMISS_GENE)).to_numpy(bool)
    for rule, floor in (("all", 1), ("min", int(min_reads))):
        per = matched.groupby(["cell", "prefix"], sort=True).size()
        per = per[per >= floor]
        dec = set(matched.loc[is_decoy, "prefix"])
        flags = np.array([p in dec for p in per.index.get_level_values(1)], bool)
        guides = pd.Series(~flags, index=per.index).groupby(level=0).sum()
        decoys = pd.Series(flags, index=per.index).groupby(level=0).any()
        for k, name in ((1, "1"), (2, "2"), (3, "3")):
            counts[f"{rule}_{name}"] = int((guides == k).sum())
        counts[f"{rule}_ge4"] = int((guides >= 4).sum())
        counts[f"{rule}_decoy"] = int(decoys.sum())
    library = matched[~is_decoy]
    if library.empty:
        return counts, pd.DataFrame(columns=columns)
    groups = _barcode_groups(library["cell"].to_numpy(np.int64), library["prefix"].to_numpy(),
                             neighbours)
    n, starts, sizes = groups["n"], groups["starts"], groups["sizes"]
    order = np.lexsort((-n, groups["cell"]))
    ranked_n = n[order]
    ranks = np.arange(len(n)) - np.repeat(starts, sizes)
    second_rows = np.flatnonzero(ranks == 1)
    if len(second_rows):
        lead = ranked_n[second_rows - 1]
        clear = lead > ranked_n[second_rows]
        runner = order[second_rows[clear]]
        counts["bleed_tested"] = int(clear.sum())
        counts["bleed_neighbour"] = int(groups["bleed"][runner].sum())
    strong = n >= int(min_reads)
    per_cell = np.add.reduceat(strong.astype(np.int64), starts)
    any_bleed = np.logical_or.reduceat(groups["bleed"] & strong, starts)
    multi_cells = groups["cell"][starts][(per_cell >= 2) & ~any_bleed]
    if not len(multi_cells):
        return counts, pd.DataFrame(columns=columns)
    rows = []
    chosen = library[library["cell"].isin(multi_cells)]
    for cell_key, group in chosen.groupby(keys, sort=True):
        tally = group.groupby("prefix", sort=False).agg(
            guide=("guide", "first"), gene=("gene", "first"), n=("prefix", "size"),
            q=("quality", "median"))
        tally = tally[tally["n"] >= int(min_reads)]
        named = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        rows.append({**dict(zip(keys, named)), "n_guides": int(len(tally)),
                     "guides": "|".join(map(str, tally["guide"])),
                     "genes": "|".join(map(str, tally["gene"])),
                     "reads": "|".join(str(int(v)) for v in tally["n"]),
                     "median_quality": "|".join(f"{float(v):.4f}" for v in tally["q"])})
    return counts, pd.DataFrame(rows, columns=columns)


#: The cell-table columns the weighted majority adds (item 988).
_WEIGHTED_COLUMNS = ("weight_top", "weight_total", "d_weight_top", "g_median_top",
                     "bleed_discounted")


def _neighbour_codes(cells: pd.DataFrame, keys: Sequence[str],
                     neighbours: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    """The touching cells as pairs of cell codes (item 988).

    :param cells: one row per cell code, with ``keys`` and ``code``.
    :param keys: the columns naming a cell, ``cell`` among them.
    :param neighbours: rows with ``keys`` and ``neighbour``, or None.
    :returns: ``(P, 2)`` code pairs; a cell without a code has no matched
        read and is left out. None without ``neighbours``.
    """
    if neighbours is None:
        return None
    keys = list(keys)
    left = neighbours.merge(cells, on=keys, how="inner")
    other = cells.rename(columns={"cell": "neighbour", "code": "neighbour_code"})
    both = left.merge(other, on=[k if k != "cell" else "neighbour" for k in keys],
                      how="inner")
    return np.stack([both["code"].to_numpy(np.int64),
                     both["neighbour_code"].to_numpy(np.int64)], 1)


def _weighted_cells(matched: pd.DataFrame, keys: Sequence[str], *, distance_weight: str,
                    decay_px: float, interior_px: float, min_weight: float,
                    weighted_quality: bool, neighbour_discount: float, share: float,
                    neighbours: Optional[pd.DataFrame]) -> Dict[tuple, Dict[str, Any]]:
    """Each cell's weighted-majority call from its matched reads (item 988).

    :param matched: the exact and corrected reads in cells.
    :param keys: the columns naming a cell.
    :param distance_weight: ``ops_prefix_distance_weight``.
    :param decay_px: the decay length, read unless ``distance_weight`` is
        ``'off'``.
    :param interior_px: the distance with full weight.
    :param min_weight: the smallest distance term.
    :param weighted_quality: use the quality term.
    :param neighbour_discount: the neighbour-bleed discount; 1 is off.
    :param share: the share of the weight the top barcode needs.
    :param neighbours: the touching cells (:func:`prefix_cell_calls`).
    :returns: cell key -> ``called``, ``prefix`` (the top barcode) and the
        columns of :data:`_WEIGHTED_COLUMNS`.
    :raises ValueError: for a distance term without ``nuc_gap_px``, or a
        discount below 1 without ``neighbours``.
    """
    keys = list(keys)
    if matched.empty:
        return {}
    length = float(decay_px) if distance_weight != "off" else 0.0
    if length > 0 and "nuc_gap_px" not in matched.columns:
        raise ValueError("ops_prefix_distance_weight needs each read's nucleus distance "
                         "(nuc_gap_px); set ops_prefix_nuclei_masks")
    discount = float(neighbour_discount)
    if discount != 1.0 and neighbours is None:
        raise ValueError("ops_read_neighbour_discount below 1 needs the touching cells")
    code = matched.groupby(keys, sort=True).ngroup().to_numpy(np.int64)
    table = matched[keys].assign(code=code).drop_duplicates("code").sort_values("code")
    pairs = _neighbour_codes(table, keys, neighbours) if discount != 1.0 else None
    groups = _barcode_groups(code, matched["prefix"].to_numpy(), pairs)
    term, used = _weight_terms(distance_weight, weighted_quality, length)
    gap = matched["nuc_gap_px"].to_numpy(float) if "nuc_gap_px" in matched.columns else None
    per_read = _read_weights(matched["quality"].to_numpy(float), gap, quality_term=term,
                             decay_px=used, interior_px=interior_px, min_weight=min_weight)
    distance = (_read_weights(np.zeros(len(matched)), gap, quality_term=False, decay_px=used,
                              interior_px=interior_px, min_weight=min_weight))
    size = len(groups["n"])
    summed = _discounted(np.bincount(groups["group"], weights=per_read, minlength=size),
                         groups, discount)
    d_summed = np.bincount(groups["group"], weights=distance, minlength=size)
    called, top = _weighted_decision(summed, groups, share)
    total = np.add.reduceat(summed, groups["starts"])
    bleed = (np.add.reduceat(groups["bleed"].astype(np.int64), groups["starts"])
             if discount != 1.0 else np.zeros(len(groups["starts"]), np.int64))
    if gap is not None:
        medians = (pd.Series(gap).groupby(groups["group"]).median()
                   .reindex(range(size)).to_numpy(float))
    else:
        medians = np.full(size, np.nan)
    out: Dict[tuple, Dict[str, Any]] = {}
    for i, row in enumerate(table[keys].itertuples(index=False, name=None)):
        g = int(top[i])
        out[tuple(row)] = {"called": bool(called[i]),
                           "prefix": str(groups["barcodes"][groups["barcode"][g]]),
                           "weight_top": float(summed[g]), "weight_total": float(total[i]),
                           "d_weight_top": float(d_summed[g]),
                           "g_median_top": float(medians[g]),
                           "bleed_discounted": int(bleed[i])}
    return out


def prefix_cell_calls(reads: pd.DataFrame, *, min_reads: int = 2,
                      keys: Sequence[str] = ("plate", "well", "site", "cell"),
                      method: str = "spacr", minority_min_reads: int = 0,
                      distance_weight: str = "off", decay_px: float = 8.0,
                      interior_px: float = 0.0, min_weight: float = 0.05,
                      weighted_quality: bool = True, neighbour_discount: float = 1.0,
                      weighted_min_share: float = 0.6,
                      neighbours: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
    :param minority_min_reads: ``ops_prefix_minority_min_reads`` (item 967,
        alpha). From 2 up, a barcode with fewer reads than this is left out
        of the cell's call when another barcode in the cell has strictly
        more reads; tied barcodes still block the call. This rescues cells
        that a single stray read of a second barcode makes ``bad`` under
        ``'recipe_simple'``; under ``'spacr'`` it changes no call and can
        only turn ``great`` into ``perfect``. 0 (the default) and 1 change
        nothing.
    :param distance_weight: ``ops_prefix_distance_weight`` (item 988,
        alpha), read only by ``'weighted_majority'``: ``'off'``,
        ``'decay'`` or ``'distance_only'`` (:data:`READ_WEIGHT_MODES`).
        The distance term needs a ``nuc_gap_px`` column, each read's
        distance in pixels to the nucleus of its cell.
    :param decay_px: the decay length in pixels
        (``ops_prefix_distance_decay_px``).
    :param interior_px: the distance with full weight
        (``ops_prefix_distance_interior_px``).
    :param min_weight: the smallest distance term
        (``ops_prefix_distance_min_weight``).
    :param weighted_quality: ``ops_prefix_weighted_quality``. True weights
        each read by ``min(1, quality) + 0.05``, and False weights it 1.
    :param neighbour_discount: ``ops_read_neighbour_discount``. A barcode
        with strictly fewer reads than the cell's top barcode has its weight
        multiplied by this when it is a top barcode of a touching cell. 1
        is off.
    :param weighted_min_share: ``ops_prefix_weighted_min_share``. The share
        of the cell's read weight the top barcode needs.
    :param neighbours: the touching cells, needed for a discount below 1.
        The table has the ``keys`` columns, naming a cell, and
        ``neighbour``, the label of a cell it touches in the same site.
    :returns: one row per cell with any read: ``keys``, ``reads``,
        ``matched`` (exact or corrected), ``exact``, ``corrected``,
        ``ambiguous``, ``collision``, ``out_of_library``, ``barcodes``
        (distinct callable barcodes), ``prefix``, ``guide``, ``gene`` and
        ``level`` of the top barcode (empty unless ``perfect`` or ``great``),
        ``top_reads``, ``top_exact``, ``top_share`` (of matched reads),
        ``second_reads``, ``has_exact_read`` and ``quality``. With
        ``minority_min_reads`` 2 or more, ``barcodes``, ``second_reads`` and
        the class count only the reads the rule keeps, and two columns are
        added: ``minority_ignored`` (reads left out) and
        ``minority_rescued`` (True when the cell is called only because of
        the rule). Under ``'weighted_majority'`` (:data:`WEIGHTED_CLASSES`)
        the top barcode is the one with the most weight, and five columns
        are added. ``weight_top`` and ``weight_total`` are the summed read
        weights of the top barcode and of the cell. ``d_weight_top`` is the
        summed distance term of the top barcode's reads, and
        ``g_median_top`` their median nucleus distance (NaN without one).
        ``bleed_discounted`` counts the barcodes the neighbour discount
        weighed down.
    :raises ValueError: for an unknown method or read-weight mode, for the
        minority-read rule combined with ``'weighted_majority'``, or for a
        discount below 1 without ``neighbours``.
    """
    return _prefix_cell_calls(reads, min_reads=min_reads, keys=keys, method=method,
                              minority_min_reads=minority_min_reads,
                              distance_weight=distance_weight, decay_px=decay_px,
                              interior_px=interior_px, min_weight=min_weight,
                              weighted_quality=weighted_quality,
                              neighbour_discount=neighbour_discount,
                              weighted_min_share=weighted_min_share, neighbours=neighbours)


def _prefix_cell_calls(reads: pd.DataFrame, *, min_reads: int = 2,
                       keys: Sequence[str] = ("plate", "well", "site", "cell"),
                       method: str = "spacr", sort: Optional[str] = None,
                       minority_min_reads: int = 0, distance_weight: str = "off",
                       decay_px: float = 8.0, interior_px: float = 0.0,
                       min_weight: float = 0.05, weighted_quality: bool = True,
                       neighbour_discount: float = 1.0, weighted_min_share: float = 0.6,
                       neighbours: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """:func:`prefix_cell_calls` with item 961's ``ops_call_sort``.

    :param reads: as for :func:`prefix_cell_calls`.
    :param min_reads: as for :func:`prefix_cell_calls`.
    :param keys: as for :func:`prefix_cell_calls`.
    :param method: as for :func:`prefix_cell_calls`.
    :param sort: None keeps :func:`prefix_cell_calls`' rule. ``'count'``,
        ``'quality'`` or ``'peak'`` rank a cell's barcodes by reads, by mean
        read quality or by the brightest read's spot strength (the reads need
        a ``peak`` column), with ``'spacr'`` only: one barcode with
        ``min_reads`` exact reads is ``perfect``, a tie on the key is
        ``ambiguous``, any other unique top is ``great``.
    :param minority_min_reads: as for :func:`prefix_cell_calls` (item 967);
        a sort ranks the reads the rule keeps.
    :param distance_weight: as for :func:`prefix_cell_calls` (item 988).
    :param decay_px: as for :func:`prefix_cell_calls` (item 988).
    :param interior_px: as for :func:`prefix_cell_calls` (item 988).
    :param min_weight: as for :func:`prefix_cell_calls` (item 988).
    :param weighted_quality: as for :func:`prefix_cell_calls` (item 988).
    :param neighbour_discount: as for :func:`prefix_cell_calls` (item 988).
    :param weighted_min_share: as for :func:`prefix_cell_calls` (item 988).
    :param neighbours: as for :func:`prefix_cell_calls` (item 988).
    :returns: :func:`prefix_cell_calls`' table.
    :raises ValueError: for an unknown method or sort, a sort with
        ``'recipe_simple'`` or ``'weighted_majority'``, ``'peak'`` without a
        ``peak`` column, or as :func:`prefix_cell_calls` raises.
    """
    if method not in QUALITY_METHODS:
        raise ValueError(f"ops_prefix_quality_method must be one of {list(QUALITY_METHODS)}; "
                         f"got {method!r}")
    if sort is not None:
        if sort not in _CALL_SORTS:
            raise ValueError(f"ops_call_sort must be one of {list(_CALL_SORTS)}; got {sort!r}")
        if method != "spacr":
            raise ValueError("ops_call_sort ranks barcodes under ops_prefix_quality_method "
                             "'spacr'; 'recipe_simple' keeps the recipe's own ranking")
        if sort == "peak" and "peak" not in reads.columns:
            raise ValueError("ops_call_sort 'peak' needs each read's spot strength (a "
                             "'peak' column)")
    minority = int(minority_min_reads or 0) > 1
    weighted = method == "weighted_majority"
    if weighted and minority:
        from .qt.i18n import tr

        raise ValueError(tr("ops_prefix_minority_min_reads and ops_prefix_quality_method "
                            "'weighted_majority' are alternatives; set "
                            "ops_prefix_minority_min_reads to 0 (its default is 3) "
                            "or choose another quality method"))
    if distance_weight not in READ_WEIGHT_MODES:
        raise ValueError(f"ops_prefix_distance_weight must be one of {list(READ_WEIGHT_MODES)}; "
                         f"got {distance_weight!r}")
    columns = list(keys) + ["reads", "matched", "exact", "corrected", "ambiguous",
                            "collision", "out_of_library", "barcodes", "prefix", "guide",
                            "gene", "level", "top_reads", "top_exact", "top_share",
                            "second_reads", "has_exact_read", "quality"]
    if minority:
        columns += ["minority_ignored", "minority_rescued"]
    if weighted:
        columns += list(_WEIGHTED_COLUMNS)
    frame = reads[reads["cell"] > 0] if "cell" in reads.columns else reads.iloc[0:0]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    decided: Dict[tuple, Dict[str, Any]] = {}
    if weighted:
        decided = _weighted_cells(
            frame[frame["status"].isin(("exact", "corrected"))], keys,
            distance_weight=distance_weight, decay_px=decay_px, interior_px=interior_px,
            min_weight=min_weight, weighted_quality=weighted_quality,
            neighbour_discount=neighbour_discount, share=weighted_min_share,
            neighbours=neighbours)
    ignored = pd.Series(False, index=frame.index)
    if minority:
        is_matched = frame["status"].isin(("exact", "corrected"))
        ignored[is_matched] = _minority_ignored(frame[is_matched], keys, minority_min_reads)
    recipe: Dict[tuple, str] = {}
    recipe_all: Dict[tuple, str] = {}
    if method == "recipe_simple":
        scored = frame[frame["status"].isin(("exact", "corrected"))]
        scored = scored.assign(score=1.0 - scored["distance"].astype(float)
                               / scored["key"].str.len().astype(float))
        sets = [(recipe, scored[~ignored[scored.index].to_numpy()] if minority else scored)]
        if minority:
            sets.append((recipe_all, scored))
        for target, rows in sets:
            classes = recipe_simple_quality(rows, cell_cols=list(keys), barcode_col="prefix",
                                            gene_col="gene", score_col="score")
            target.update(zip(map(tuple, classes[list(keys)].to_numpy().tolist()),
                              classes["quality"]))
    out = []
    for cell_key, group in frame.groupby(list(keys), sort=True):
        status = group["status"].value_counts()
        all_matched = group[group["status"].isin(("exact", "corrected"))]
        matched = (all_matched[~ignored[all_matched.index].to_numpy()] if minority
                   else all_matched)
        counts = _class_counts(matched)
        named = tuple(cell_key) if isinstance(cell_key, tuple) else (cell_key,)
        decision = decided.get(named)
        if method == "recipe_simple":
            quality = recipe.get(named, "none")
            before = recipe_all.get(named, "none") if minority else quality
        elif weighted:
            quality = ("none" if decision is None else
                       ("perfect" if len(counts) == 1 else "great") if decision["called"]
                       else "ambiguous")
            before = quality
        else:
            quality = classify_cell({k: tuple(v) for k, v in counts.items()}, min_reads)
            before = (classify_cell({k: tuple(v) for k, v in _class_counts(all_matched).items()},
                                    min_reads) if minority else quality)
        ranked = sorted(counts.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))
        if sort is not None:
            ranked, quality = _sorted_call(matched, counts, sort, min_reads)
        elif decision is not None:
            ranked = ([(decision["prefix"], counts[decision["prefix"]])]
                      + [kv for kv in ranked if kv[0] != decision["prefix"]])
        top = ranked[0] if ranked else None
        call = (matched[matched["prefix"] == top[0]].iloc[0]
                if top is not None and quality in ("perfect", "great") else None)
        n_matched = int(len(all_matched))
        top_reads = int(sum(top[1])) if top else 0
        extra = ({"minority_ignored": int(len(all_matched) - len(matched)),
                  "minority_rescued": bool(quality in ("perfect", "great")
                                           and before not in ("perfect", "great"))}
                 if minority else {})
        if weighted:
            extra = {name: (decision[name] if decision is not None else
                            (0 if name == "bleed_discounted" else float("nan")))
                     for name in _WEIGHTED_COLUMNS}
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
            "quality": quality, **extra})
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
    peaks, _strength, values = _spot_candidates(stack, floor=threshold, margin=margin,
                                                sigma=sigma)
    return peaks, values


def _spot_candidates(stack: np.ndarray, *, floor: float, margin: int = 5,
                     sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The spots :func:`detect_spots` keeps at a threshold of ``floor``, with their strength.

    :param stack: ``(cycles, Y, X, bases)`` intensities.
    :param floor: the lowest read threshold; spots at or below it are dropped.
    :param margin: border, in pixels, where reads are not taken.
    :param sigma: the LoG width.
    :returns: ``(peaks, strength, values)``, strongest first; any read
        threshold at or above ``floor`` keeps the spots whose strength is
        above it, exactly as :func:`detect_spots` at that threshold.
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
    strength = np.zeros(0, np.float32)
    if peaks.size:
        surface = score - ndimage.minimum_filter(score, size=5)
        strength = surface[peaks[:, 0], peaks[:, 1]]
        keep = strength > float(floor)
        keep &= (peaks[:, 0] >= margin) & (peaks[:, 0] < height - margin)
        keep &= (peaks[:, 1] >= margin) & (peaks[:, 1] < width - margin)
        peaks, strength = peaks[keep], strength[keep]
    values = extract_bases(filtered, peaks, window=1)
    return peaks, strength, values


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
    return _decode_site(stack, library, bases=bases, labels=labels, threshold=threshold,
                        crosstalk=crosstalk, metric=metric, max_distance=max_distance,
                        control=control)


def _prefix_calls(values: np.ndarray, *, bases: Sequence[str], crosstalk: str,
                  read_mode: str = "log", margins: bool = False):
    """Call one site's reads as the spacer-prefix decode calls them.

    :param values: ``(N, cycles, bases)`` read intensities.
    :param bases: the base letter of each plane.
    :param crosstalk: :func:`compensate_prefix` mode, used in ``log`` mode.
    :param read_mode: ``log`` (unmixed as asked, then the brightest base
        after the per-cycle median normalisation) or ``raw`` (item 952: the
        brightest base as read, no unmixing or normalisation).
    :param margins: also return the per-cycle margins (item 961's
        per-segment quality).
    :returns: ``(calls, quality)``: one string per read and its worst
        cycle's margin; with ``margins`` also the ``(N, cycles)`` margins.
    """
    from .ops_sbs import _raw_called_bases, called_bases

    if read_mode == "raw":
        _data, calls, margin = _raw_called_bases(values, bases)
    else:
        values = compensate_prefix(values, crosstalk)
        _data, calls, margin = called_bases(values, bases=tuple(bases), compensate=False,
                                            normalise=True, gpu=False)
    if len(calls):
        worst = np.where(np.isnan(margin), np.inf, margin).min(axis=1)
        quality = np.where(np.isfinite(worst), worst, 0.0).astype(np.float32)
    else:
        quality = np.zeros(0, np.float32)
    if margins:
        return calls, quality, np.asarray(margin, np.float32).reshape(len(calls), -1)
    return calls, quality


def _raw_candidates(stack: np.ndarray, candidates):
    """Spot candidates with their raw reads in place of the -LoG reads (item 952).

    :param stack: ``(cycles, Y, X, bases)`` intensities.
    :param candidates: ``(peaks, strength, values, ...)``, as
        :func:`_spot_candidates` returns them, with anything after the
        values kept.
    :returns: the same tuple with ``values`` from
        :func:`spacr.ops_sbs._read_values` in ``raw`` mode at the same peaks.
    """
    from .ops_sbs import _read_values

    peaks = candidates[0]
    raw = _read_values(np.moveaxis(np.asarray(stack), -1, 1), None, peaks, "raw")
    return (candidates[0], candidates[1], raw) + tuple(candidates[3:])


def _decode_site(stack: np.ndarray, library: PrefixLibrary, *, bases: Sequence[str],
                 labels: Optional[np.ndarray], threshold: float, crosstalk: str,
                 metric: str, max_distance: int, control: Optional[PrefixLibrary],
                 read_mode: str = "log", peak: bool = False) -> pd.DataFrame:
    """:func:`decode_prefix_site` with the read mode of item 952.

    :param stack: as for :func:`decode_prefix_site`.
    :param library: the whitelist.
    :param bases: the base letter of each plane.
    :param labels: the cell labels, or None.
    :param threshold: the read score rise.
    :param crosstalk: :func:`compensate_prefix` mode.
    :param metric: :func:`match_prefix_reads` metric.
    :param max_distance: :func:`match_prefix_reads` limit.
    :param control: a shuffled whitelist, or None.
    :param read_mode: ``log`` or ``raw`` (:func:`_prefix_calls`).
    :param peak: add a ``peak`` column, each read's spot strength (item
        961's peak-prioritised calling).
    :returns: :func:`decode_prefix_site`'s rows.
    :raises ValueError: when ``labels`` is not the stack's grid.
    """
    strength = None
    if read_mode == "raw":
        peaks, strength, values = _raw_candidates(
            stack, _spot_candidates(stack, floor=threshold))
    elif peak:
        peaks, strength, values = _spot_candidates(stack, floor=threshold)
    else:
        peaks, values = detect_spots(stack, threshold=threshold)
    calls, quality = _prefix_calls(values, bases=bases, crosstalk=crosstalk,
                                   read_mode=read_mode)
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
    if peak:
        frame["peak"] = np.asarray(strength, np.float32)
    frame = pd.concat([frame, matched], axis=1)
    if control is not None:
        frame["control_status"] = match_prefix_reads(
            calls, control, metric=metric, max_distance=max_distance)["status"].to_numpy()
    return frame


#: The grid ``ops_prefix_threshold_tuning='per_well_decoy'`` tries when
#: ``ops_prefix_tuning_grid`` is empty (item 942): read thresholds around the
#: item-920 value 80, read-quality floors spanning the deciles of the
#: PERISCOPE P1 reads, and 1-mismatch correction or exact matches only. Each
#: axis runs from the loosest to the strictest value.
PREFIX_TUNING_GRID: Dict[str, Tuple[float, ...]] = {
    "ops_read_threshold": (40.0, 60.0, 80.0, 100.0, 130.0, 160.0),
    "ops_prefix_min_read_quality": (0.0, 0.05, 0.1, 0.15, 0.25),
    "ops_prefix_max_distance": (1.0, 0.0),
}

#: The grid ``'per_well_decoy'`` tries under ``ops_prefix_quality_method``
#: ``'weighted_majority'`` when ``ops_prefix_tuning_grid`` is empty (item
#: 988, alpha). It adds read thresholds 20 and 30 below the item-942 grid
#: and tunes the decay length, the neighbour discount and the quality term
#: (:data:`WEIGHT_GRID_KEYS`) with the other axes. Each weight axis ends on
#: its off value (decay length 0, discount 1, quality term on), so a tie
#: keeps the weight off.
PREFIX_TUNING_GRID_WEIGHTED: Dict[str, Tuple[float, ...]] = {
    "ops_read_threshold": (20.0, 30.0) + PREFIX_TUNING_GRID["ops_read_threshold"],
    "ops_prefix_min_read_quality": PREFIX_TUNING_GRID["ops_prefix_min_read_quality"],
    "ops_prefix_max_distance": PREFIX_TUNING_GRID["ops_prefix_max_distance"],
    "ops_prefix_distance_decay_px": (4.0, 8.0, 16.0, 32.0, 0.0),
    "ops_read_neighbour_discount": (0.0, 0.25, 0.5, 1.0),
    "ops_prefix_weighted_quality": (0.0, 1.0),
}

#: The read-quality floor a small well's tuning keeps (item 982 small-well
#: guard): with fewer sites than ``ops_prefix_tuning_min_sites`` the per-well
#: tuning does not choose the floor, because on one or two tuning sites it
#: picked the strictest floor (0.25) where 0.1 genotyped 64.3 % instead of
#: 39.7 % of Funk plate A cells at the same decoy FDR (item 960 proof).
SMALL_WELL_FLOOR = 0.1

#: The spacer-prefix threshold modes, the first being the default.
PREFIX_TUNING_MODES = ("none", "per_well_decoy")


def _tuning_axes(chosen: Mapping[str, Any]) -> List[Tuple[str, List[float]]]:
    """The tuning grid's axes, loosest value first.

    :param chosen: from :func:`prefix_settings`.
    :returns: ``(setting, values)`` for the read threshold, the read-quality
        floor and the mismatches, and under ``'weighted_majority'`` then the
        weight axes (:func:`_weight_axes`); a key the grid leaves out keeps
        its entered value.
    :raises ValueError: for a grid that is not a mapping of those keys to
        lists of numbers of 0 or more.
    """
    grid = chosen["ops_prefix_tuning_grid"]
    if isinstance(grid, str) and grid.strip():
        try:
            grid = json.loads(grid)
        except ValueError:
            raise ValueError("ops_prefix_tuning_grid must be a mapping such as "
                             '{"ops_read_threshold": [60, 80]}; got '
                             f"{grid!r}") from None
    weighted = chosen.get("ops_prefix_quality_method") == "weighted_majority"
    allowed = PREFIX_TUNING_GRID_WEIGHTED if weighted else PREFIX_TUNING_GRID
    if grid is None or grid == "" or grid == {}:
        grid = allowed
    if not isinstance(grid, Mapping) or set(grid) - set(allowed):
        raise ValueError(f"ops_prefix_tuning_grid takes the keys "
                         f"{sorted(allowed)}; got {grid!r}")
    axes = []
    for key in PREFIX_TUNING_GRID:
        values = grid.get(key, [chosen[key]])
        try:
            values = sorted({float(v) for v in values})
        except (TypeError, ValueError):
            raise ValueError(f"ops_prefix_tuning_grid {key} must be a list of "
                             f"numbers; got {values!r}") from None
        if not values or values[0] < 0:
            raise ValueError(f"ops_prefix_tuning_grid {key} needs values of 0 "
                             f"or more; got {values!r}")
        if key == "ops_prefix_max_distance":
            values = values[::-1]
        axes.append((key, values))
    if weighted:
        axes += _weight_axes(chosen, grid)
    return axes


def _weight_axes(chosen: Mapping[str, Any], grid: Mapping[str, Any]
                 ) -> List[Tuple[str, List[float]]]:
    """The weighted majority's tuning axes (item 988), each ending on its off value.

    :param chosen: from :func:`prefix_settings`.
    :param grid: the tuning grid in use.
    :returns: ``(key, values)`` for the keys of :data:`WEIGHT_GRID_KEYS`. A
        key the grid leaves out keeps its entered value: the decay length
        when ``ops_prefix_distance_weight`` is on, else 0. A tuned decay
        length or discount always holds its off value too, so the tuner
        can keep the weight off.
    :raises ValueError: for a value out of range.
    """
    entered = {"ops_prefix_distance_decay_px": (chosen["ops_prefix_distance_decay_px"]
                                                if chosen["ops_prefix_distance_weight"] != "off"
                                                else 0.0),
               "ops_read_neighbour_discount": chosen["ops_read_neighbour_discount"],
               "ops_prefix_weighted_quality": float(bool(chosen["ops_prefix_weighted_quality"]))}
    off = {"ops_prefix_distance_decay_px": 0.0, "ops_read_neighbour_discount": 1.0,
           "ops_prefix_weighted_quality": 1.0}
    axes = []
    for key in WEIGHT_GRID_KEYS:
        if key in grid:
            try:
                values = {float(v) for v in grid[key]}
            except (TypeError, ValueError):
                raise ValueError(f"ops_prefix_tuning_grid {key} must be a list of "
                                 f"numbers; got {grid[key]!r}") from None
            if key != "ops_prefix_weighted_quality":
                values.add(off[key])
        else:
            values = {float(entered[key])}
        if min(values) < 0 or (key != "ops_prefix_distance_decay_px" and max(values) > 1):
            raise ValueError(f"ops_prefix_tuning_grid {key} is out of range; got "
                             f"{sorted(values)}")
        ordered = sorted(values - {off[key]}) + ([off[key]] if off[key] in values else [])
        axes.append((key, ordered))
    return axes


def _matched_rows(calls: Sequence[str], library: PrefixLibrary, memo: Dict[str, tuple], *,
                  metric: str, max_distance: int) -> pd.DataFrame:
    """:func:`match_prefix_reads` of the calls, each distinct read matched once per run.

    :param calls: the called reads.
    :param library: the whitelist (with decoys).
    :param memo: ``read -> row`` of the reads matched so far, filled in.
    :param metric: the match metric.
    :param max_distance: the largest distance corrected.
    :returns: the same rows :func:`match_prefix_reads` returns.
    """
    new = sorted({str(c) for c in calls} - set(memo))
    if new:
        matched = match_prefix_reads(new, library, metric=metric, max_distance=max_distance)
        for row in matched.itertuples(index=False):
            memo[row.read] = tuple(row)
    columns = ["read", "key", "prefix", "distance", "status", "guide", "gene", "level"]
    return pd.DataFrame([memo[str(c)] for c in calls], columns=columns)


def _site_tuning_reads(candidates, library: PrefixLibrary, memo: Dict[str, tuple], *,
                       bases: Sequence[str], threshold: float, crosstalk: str, metric: str,
                       max_distance: int, read_mode: str = "log",
                       extra: Optional[Mapping[str, np.ndarray]] = None,
                       peak: bool = False) -> pd.DataFrame:
    """One site's reads at one read threshold, from its spot candidates.

    The reads above the threshold are compensated, called and matched as
    :func:`decode_prefix_site` does at that threshold.

    :param candidates: ``(peaks, strength, values, cell)``: a site's
        :func:`_spot_candidates` and the cell label under each spot.
    :param library: the whitelist (with decoys).
    :param memo: the run's match memo (:func:`_matched_rows`).
    :param bases: the base letter of each plane.
    :param threshold: the read threshold.
    :param crosstalk: :func:`compensate_prefix` mode.
    :param metric: the match metric.
    :param max_distance: the largest distance matched (the grid's loosest).
    :param read_mode: ``log`` or ``raw`` (:func:`_prefix_calls`); the
        candidates' values must be that mode's reads.
    :param extra: columns with one value per candidate (item 988: the
        nucleus distance), kept for the reads above the threshold; None adds
        none.
    :param peak: add a ``peak`` column, each read's spot strength (item 961).
    :returns: :func:`decode_prefix_site`'s rows, without the control.
    """
    peaks, strength, values, cell = candidates
    keep = strength > float(threshold)
    peaks, cell = peaks[keep], cell[keep]
    kept_strength = np.asarray(strength)[keep]
    calls, quality = _prefix_calls(values[keep], bases=bases, crosstalk=crosstalk,
                                   read_mode=read_mode)
    frame = pd.DataFrame({"y": peaks[:, 0] if len(peaks) else np.zeros(0, np.int64),
                          "x": peaks[:, 1] if len(peaks) else np.zeros(0, np.int64),
                          "cell": cell, "quality": quality})
    for name, column in (extra or {}).items():
        frame[name] = np.asarray(column)[keep]
    if peak:
        frame["peak"] = kept_strength.astype(np.float32)
    matched = _matched_rows(list(calls), library, memo, metric=metric,
                            max_distance=max_distance)
    return pd.concat([frame, matched], axis=1)


def _at_setting(reads: pd.DataFrame, min_quality: float, max_distance: int) -> pd.DataFrame:
    """A site's reads under one read-quality floor and mismatch limit.

    :param reads: rows matched at the grid's loosest mismatch limit.
    :param min_quality: reads below this quality are dropped (0 keeps all).
    :param max_distance: a read matched further than this becomes
        ``out_of_library``, as matching at this limit would leave it.
    :returns: the rows kept, statuses restated.
    """
    if min_quality > 0:
        reads = reads[reads["quality"] >= min_quality]
    far = (reads["distance"] > int(max_distance)) | (reads["distance"] < 0)
    if far.any():
        reads = reads.copy()
        reads.loc[far, ["prefix", "guide", "gene", "level"]] = None
        reads.loc[far, "status"] = "out_of_library"
        reads.loc[far, "distance"] = -1
    return reads


def _called_counts(reads: pd.DataFrame, method: str, nearmiss: bool = False,
                   minority_min_reads: int = 0, *, opportunity_profiles=None):
    """Real and decoy cells called from one site's reads, as :func:`prefix_cell_calls` calls them.

    :param reads: rows with ``cell``, ``status``, ``prefix`` and ``gene``.
    :param method: ``'spacr'`` (a unique top barcode) or ``'recipe_simple'``
        (one barcode only); the perfect and great classes of each.
    :param nearmiss: also return the near-miss decoy calls (item 949).
    :param minority_min_reads: ``ops_prefix_minority_min_reads`` (item 967):
        the reads :func:`_minority_ignored` flags are left out first.
    :param opportunity_profiles: optional shared profiles reduced from these same winners.
    :returns: with profiles a dict of counts and opportunity totals; otherwise
        ``(real, decoy)`` called cells, or ``(real, decoy, nearmiss)``;
        a near-miss decoy call is never a real call.
    """
    matched = reads[(reads["cell"] > 0) & reads["status"].isin(("exact", "corrected"))]
    if int(minority_min_reads or 0) > 1 and len(matched):
        matched = matched[~_minority_ignored(matched, ["cell"], minority_min_reads)]
    if matched.empty:
        counts = (0, 0, 0) if nearmiss else (0, 0)
        return {'counts': counts, 'opportunities': _prefix_opportunities([], [], opportunity_profiles)} \
            if opportunity_profiles else counts
    tally = matched.groupby(["cell", "prefix"], sort=False).size().rename("n").reset_index()
    tally = tally.sort_values(["cell", "n"], ascending=[True, False], kind="stable")
    tally["rank"] = tally.groupby("cell", sort=False).cumcount()
    top = tally[tally["rank"] == 0].set_index("cell")
    runner = (tally[tally["rank"] == 1].set_index("cell")["n"]
              .reindex(top.index).fillna(0).to_numpy())
    if method == "recipe_simple":
        barcodes = tally.groupby("cell", sort=False).size().reindex(top.index)
        called = barcodes.to_numpy() == 1
    else:
        called = top["n"].to_numpy() > runner
    gene = dict(zip(matched["prefix"], matched["gene"]))
    decoy = np.array([gene.get(p) == DECOY_GENE for p in top["prefix"]], bool)
    near = np.array([gene.get(p) == NEARMISS_GENE for p in top["prefix"]], bool)
    selected = called & ~decoy & ~near
    real = int(selected.sum())
    if opportunity_profiles:
        counts = (real, int((called & decoy).sum()))
        if nearmiss:
            counts += (int((called & near).sum()),)
        return {'counts': counts,
                'opportunities': _prefix_opportunities(top['prefix'], selected, opportunity_profiles)}
    if nearmiss:
        return real, int((called & decoy).sum()), int((called & near).sum())
    return real, int((called & decoy).sum())


def _flag_decoy_cells(cells: pd.DataFrame, nearmiss: bool = False) -> pd.DataFrame:
    """Mark the cells called as a decoy; they keep their prefix but no genotype.

    :param cells: from :func:`prefix_cell_calls`.
    :param nearmiss: the run has near-miss decoys (item 949): a near-miss
        call is a decoy too, and a ``decoy_class`` column says ``'far'`` or
        ``'nearmiss'``.
    :returns: the cells with a ``decoy`` column; a decoy call's ``guide``,
        ``gene`` and ``level`` are emptied.
    """
    cells = cells.copy()
    decoy = (cells["gene"] == DECOY_GENE).to_numpy(bool) if len(cells) else np.zeros(0, bool)
    near = ((cells["gene"] == NEARMISS_GENE).to_numpy(bool) if nearmiss and len(cells)
            else np.zeros(len(cells), bool))
    cells["decoy"] = decoy | near
    if nearmiss:
        cells["decoy_class"] = np.where(decoy, "far", np.where(near, "nearmiss", None))
    decoy = decoy | near
    if decoy.any():
        cells.loc[decoy, ["guide", "gene", "level"]] = None
    return cells


def _labelled_cells(labels: Optional[np.ndarray], cells: pd.DataFrame) -> int:
    """The site's cells, the genotyped share's denominator.

    :param labels: the cell labels, or None.
    :param cells: the site's cell calls.
    :returns: the distinct labels above 0, or without labels the cells with
        a read.
    """
    if labels is None:
        return int(len(cells))
    values = np.unique(np.asarray(labels))
    return int((values > 0).sum())


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
        ``ops_prefix_masks``, ``ops_read_threshold`` and the item-942 decoy
        and tuning settings (``ops_prefix_decoy_k``,
        ``ops_prefix_decoy_seed``, ``ops_prefix_fdr_scale``,
        ``ops_prefix_min_read_quality``, ``ops_prefix_threshold_tuning``,
        ``ops_prefix_fdr_target_pct``, ``ops_prefix_tuning_grid``) and the
        item-949 ``ops_prefix_nearmiss_decoy_k`` and
        ``ops_prefix_fdr_calibration``, the item-967
        ``ops_prefix_minority_min_reads``, the item-982 small-well guard
        ``ops_prefix_tuning_min_sites``, and the item-988 weighted-majority
        settings: ``ops_prefix_distance_weight``,
        ``ops_prefix_distance_decay_px``, ``ops_prefix_distance_interior_px``,
        ``ops_prefix_distance_min_weight``, ``ops_prefix_weighted_quality``,
        ``ops_prefix_weighted_min_share``, ``ops_read_neighbour_discount``,
        ``ops_prefix_nuclei_masks``, ``ops_keep_multi_guide_cells`` and
        ``ops_multi_guide_min_reads``, and shared controls
        ``ops_never_barcode_k``, ``ops_never_barcode_seed`` and
        ``ops_never_barcode_sets``. Never calibration without decoys keeps
        the entered scale and reports the fallback.
    :returns: the normalised values under the same keys.
    :raises ValueError: without a library, for an unknown choice, for
        tuning or a shuffle calibration without decoys, for a negative
        decoy count, minority-read floor or small-well site count, or for a
        weight setting that is out of range or set without
        ``'weighted_majority'``.
    """
    from .ops_decode import FDR_CALIBRATIONS, _never_settings

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
        "ops_prefix_masks_frame": str(settings.get("ops_prefix_masks_frame")
                                      or MASK_FRAMES[0]).strip().lower(),
        "cpg_align_source": str(settings.get("cpg_align_source")
                                or "periscope_logged").strip().lower(),
        "cpg_align_table": settings.get("cpg_align_table") or None,
        "ops_read_threshold": float(settings.get("ops_read_threshold", 315.0)),
        "ops_prefix_decoy_k": int(settings.get("ops_prefix_decoy_k") or 0),
        "ops_prefix_decoy_seed": int(settings.get("ops_prefix_decoy_seed", 1) or 0),
        "ops_prefix_fdr_scale": float(settings.get("ops_prefix_fdr_scale") or 1.0),
        "ops_prefix_min_read_quality": float(settings.get("ops_prefix_min_read_quality")
                                             or 0.0),
        "ops_prefix_threshold_tuning": str(settings.get("ops_prefix_threshold_tuning")
                                           or PREFIX_TUNING_MODES[0]),
        "ops_prefix_fdr_target_pct": float(settings.get("ops_prefix_fdr_target_pct", 0.1)
                                           if settings.get("ops_prefix_fdr_target_pct")
                                           not in (None, "") else 0.1),
        "ops_prefix_tuning_grid": settings.get("ops_prefix_tuning_grid") or None,
        "ops_prefix_nearmiss_decoy_k": int(settings.get("ops_prefix_nearmiss_decoy_k") or 0),
        "ops_prefix_fdr_calibration": str(settings.get("ops_prefix_fdr_calibration")
                                          or FDR_CALIBRATIONS[0]).strip().lower(),
        "ops_prefix_minority_min_reads": int(settings.get("ops_prefix_minority_min_reads")
                                             or 0),
        "ops_prefix_distance_weight": str(settings.get("ops_prefix_distance_weight")
                                          or READ_WEIGHT_MODES[0]).strip().lower(),
        "ops_prefix_distance_decay_px": _setting_float(settings, "ops_prefix_distance_decay_px",
                                                       8.0),
        "ops_prefix_distance_interior_px": _setting_float(
            settings, "ops_prefix_distance_interior_px", 0.0),
        "ops_prefix_distance_min_weight": _setting_float(
            settings, "ops_prefix_distance_min_weight", 0.05),
        "ops_prefix_weighted_quality": _setting_flag(settings, "ops_prefix_weighted_quality",
                                                     True),
        "ops_prefix_weighted_min_share": _setting_float(
            settings, "ops_prefix_weighted_min_share", 0.6),
        "ops_read_neighbour_discount": _setting_float(settings, "ops_read_neighbour_discount",
                                                      1.0),
        "ops_prefix_nuclei_masks": settings.get("ops_prefix_nuclei_masks") or None,
        "ops_keep_multi_guide_cells": _setting_flag(settings, "ops_keep_multi_guide_cells",
                                                    False),
        "ops_multi_guide_min_reads": int(settings.get("ops_multi_guide_min_reads") or 2),
        "ops_prefix_tuning_min_sites": int(settings.get("ops_prefix_tuning_min_sites") or 0),
    }
    from .ops_settings import OPS_DEFAULTS

    out.update(_never_settings({
        'ops_never_barcode_k': settings.get('ops_never_barcode_k', OPS_DEFAULTS['ops_never_barcode_k']),
        'ops_never_barcode_seed': settings.get('ops_never_barcode_seed', OPS_DEFAULTS['ops_never_barcode_seed']),
        'ops_never_barcode_sets': settings.get('ops_never_barcode_sets', OPS_DEFAULTS['ops_never_barcode_sets'])}))
    if out["ops_prefix_threshold_tuning"] not in PREFIX_TUNING_MODES:
        raise ValueError(f"ops_prefix_threshold_tuning must be one of "
                         f"{list(PREFIX_TUNING_MODES)}")
    if out["ops_prefix_decoy_k"] < 0:
        raise ValueError("ops_prefix_decoy_k must be 0 or more")
    if out["ops_prefix_threshold_tuning"] != "none" and out["ops_prefix_decoy_k"] < 1:
        raise ValueError("ops_prefix_threshold_tuning 'per_well_decoy' chooses "
                         "thresholds from decoy calls; set ops_prefix_decoy_k to 1 "
                         "or more (4 recommended)")
    if out["ops_prefix_nearmiss_decoy_k"] < 0:
        raise ValueError("ops_prefix_nearmiss_decoy_k must be 0 or more")
    if out["ops_prefix_tuning_min_sites"] < 0:
        raise ValueError("ops_prefix_tuning_min_sites must be 0 or more (0 = off)")
    if out["ops_prefix_minority_min_reads"] < 0:
        raise ValueError("ops_prefix_minority_min_reads must be 0 or more (0 = off)")
    if out["ops_prefix_fdr_calibration"] not in FDR_CALIBRATIONS:
        raise ValueError(f"ops_prefix_fdr_calibration must be one of {list(FDR_CALIBRATIONS)}")
    if out["ops_prefix_fdr_calibration"] == "shuffle" and out["ops_prefix_decoy_k"] < 1:
        raise ValueError("ops_prefix_fdr_calibration 'shuffle' scales the decoy FDR; set "
                         "ops_prefix_decoy_k to 1 or more (4 recommended)")
    if not out["ops_prefix_fdr_target_pct"] > 0:
        raise ValueError("ops_prefix_fdr_target_pct must be above 0")
    if out["ops_prefix_masks_frame"] not in MASK_FRAMES:
        raise ValueError(f"ops_prefix_masks_frame must be one of {list(MASK_FRAMES)}")
    from .cpg_cropped import ALIGN_SOURCES

    if out["cpg_align_source"] not in ALIGN_SOURCES:
        raise ValueError(f"cpg_align_source must be one of {list(ALIGN_SOURCES)}")
    if out["ops_prefix_crosstalk"] not in _CROSSTALK:
        raise ValueError(f"ops_prefix_crosstalk must be one of {list(_CROSSTALK)}")
    if out["ops_prefix_metric"] not in _METRICS:
        raise ValueError(f"ops_prefix_metric must be one of {list(_METRICS)}")
    if out["ops_prefix_quality_method"] not in QUALITY_METHODS:
        raise ValueError(f"ops_prefix_quality_method must be one of {list(QUALITY_METHODS)}")
    _check_weight_settings(out)
    return out


def _setting_float(settings: Mapping[str, Any], key: str, default: float) -> float:
    """One float setting, its default when empty (item 988).

    :param settings: the settings.
    :param key: the key.
    :param default: the value of an empty or missing key.
    :returns: the value.
    """
    value = settings.get(key)
    return float(default) if value is None or value == "" else float(value)


def _setting_flag(settings: Mapping[str, Any], key: str, default: bool) -> bool:
    """One boolean setting, its default when empty (item 988).

    :param settings: the settings.
    :param key: the key.
    :param default: the value of an empty or missing key.
    :returns: the value; the strings ``false``, ``0``, ``no`` and ``off`` are
        False.
    """
    value = settings.get(key)
    if value is None or value == "":
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def _check_weight_settings(out: Mapping[str, Any]) -> None:
    """Check the item-988 settings of :func:`prefix_settings`.

    :param out: the normalised settings.
    :raises ValueError: for an unknown read-weight mode, a value out of
        range, a weight setting without ``'weighted_majority'``, or the
        weighted majority combined with the minority-read rule.
    """
    weighted = out["ops_prefix_quality_method"] == "weighted_majority"
    if out["ops_prefix_distance_weight"] not in READ_WEIGHT_MODES:
        raise ValueError(f"ops_prefix_distance_weight must be one of {list(READ_WEIGHT_MODES)}")
    if not out["ops_prefix_distance_decay_px"] > 0:
        raise ValueError("ops_prefix_distance_decay_px must be above 0")
    if out["ops_prefix_distance_interior_px"] < 0:
        raise ValueError("ops_prefix_distance_interior_px must be 0 or more")
    if not 0 <= out["ops_prefix_distance_min_weight"] <= 1:
        raise ValueError("ops_prefix_distance_min_weight must lie in [0, 1]")
    if not 0 < out["ops_prefix_weighted_min_share"] <= 1:
        raise ValueError("ops_prefix_weighted_min_share must lie in (0, 1]")
    if not 0 <= out["ops_read_neighbour_discount"] <= 1:
        raise ValueError("ops_read_neighbour_discount must lie in [0, 1] (1 = off)")
    if out["ops_multi_guide_min_reads"] < 1:
        raise ValueError("ops_multi_guide_min_reads must be 1 or more")
    if not weighted and (out["ops_prefix_distance_weight"] != "off"
                         or out["ops_read_neighbour_discount"] != 1.0):
        raise ValueError("ops_prefix_distance_weight and ops_read_neighbour_discount act in "
                         "the weighted majority; set ops_prefix_quality_method "
                         "'weighted_majority'")
    if weighted and out["ops_prefix_minority_min_reads"] > 1:
        from .qt.i18n import tr

        raise ValueError(tr("ops_prefix_minority_min_reads and ops_prefix_quality_method "
                            "'weighted_majority' are alternatives; set "
                            "ops_prefix_minority_min_reads to 0 (its default is 3) "
                            "or choose another quality method"))


#: The frame of the cell labels a prefix decode reads (item 919 frame fix).
#: ``'sbs'``: the labels are already in the sequencing frame (PERISCOPE's
#: deposited masks, or labels drawn on the sequencing planes). ``'painting'``:
#: the labels were segmented on the painting crops (CorrDNA), which PERISCOPE
#: publishes in their own frame, so each site's labels are moved onto the
#: sequencing frame by :func:`spacr.cpg_cropped.painting_to_sbs_shift` before
#: reads are assigned, and the shift is written to
#: ``label_frame_<plate>_<well_id>.csv``.
MASK_FRAMES = ("sbs", "painting")

#: The item-919 frame keys of :func:`prefix_settings`, left out of
#: ``quality_classes.json`` while ``ops_prefix_masks_frame`` is ``'sbs'``, so
#: such a run writes what it wrote before.
_ITEM_919_FRAME_KEYS = ("ops_prefix_masks_frame", "cpg_align_source", "cpg_align_table")


def check_masks_frame(chosen: Mapping[str, Any]) -> None:
    """Refuse painting-frame labels that are about to be read as sequencing-frame ones.

    A label folder written by spaCR's large-site segmentation
    (:mod:`spacr.ops_large_sites`) has a ``settings.json`` with
    ``seg_sites_src`` next to its ``labels/`` folder: those labels were drawn on
    the painting crops. Reading them with ``ops_prefix_masks_frame 'sbs'``
    would assign every read to the cell that sits PERISCOPE's 04Align shift
    away, so the run stops.

    :param chosen: from :func:`prefix_settings`.
    :raises ValueError: for spaCR painting-crop labels under ``'sbs'``.
    """
    template = chosen.get("ops_prefix_masks")
    if not template or chosen.get("ops_prefix_masks_frame") != "sbs":
        return
    folder = os.path.dirname(os.path.dirname(str(template)))
    if "{" in folder:
        return
    record = os.path.join(folder, "settings.json")
    if not os.path.isfile(record):
        return
    try:
        with open(record, encoding="utf-8") as handle:
            written = json.load(handle)
    except (OSError, ValueError):
        return
    if isinstance(written, dict) and written.get("seg_sites_src"):
        raise ValueError(
            f"ops_prefix_masks {template!r} are spaCR segmentation labels of the painting "
            f"crops ({record} has seg_sites_src). PERISCOPE's painting crops are not in "
            "the sequencing frame; set ops_prefix_masks_frame to 'painting' so each "
            "site's labels are moved by its 04Align shift before reads are assigned")


def frame_labels(labels: Optional[np.ndarray], chosen: Mapping[str, Any], *,
                 index: pd.DataFrame, plate: str, well: str, site: int, cpg_root: str,
                 record: Dict[int, Dict[str, Any]]) -> Optional[np.ndarray]:
    """One site's cell labels in the sequencing frame (item 919 frame fix).

    :param labels: from :func:`site_labels`, or None.
    :param chosen: from :func:`prefix_settings`.
    :param index: the cropped-site index (for ``cpg_align_source 'register'``).
    :param plate: the plate.
    :param well: the well folder name.
    :param site: the site.
    :param cpg_root: the reader's root.
    :param record: filled with the site's shift; reused when the site comes
        again (tuning, then calling).
    :returns: ``labels`` unchanged under ``'sbs'``; else moved by the site's
        painting-to-sequencing shift (:func:`spacr.cpg_cropped.shift_image`).
    :raises ValueError: when the site has no shift record (never a silent 0).
    """
    if labels is None or chosen["ops_prefix_masks_frame"] == "sbs":
        return labels
    from .cpg_cropped import painting_to_sbs_shift, shift_image

    if site not in record:
        record[site] = painting_to_sbs_shift(
            index, plate, well, site, cpg_root=cpg_root, source=chosen["cpg_align_source"],
            template=chosen["cpg_align_table"])
    found = record[site]
    return shift_image(labels, found["dy"], found["dx"])


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


def _site_nuclei(template: Optional[str], *, plate: str, well: str, well_id: str, site: int,
                 merged: Optional[str] = None) -> Optional[np.ndarray]:
    """The nucleus label image of one site, or None (item 988).

    :param template: ``ops_prefix_nuclei_masks``, a path template with the
        fields of ``ops_prefix_masks`` (:func:`site_labels`), to a TIFF,
        ``.npy`` or ``.npz`` (``nuclei``) label image. None falls back to
        ``merged``.
    :param plate: the plate.
    :param well: the well folder name.
    :param well_id: the normalised well id.
    :param site: the site.
    :param merged: a ``merged/`` site file; its ``nuclei`` array is used
        when it has one.
    :returns: the labels, or None when neither source has them.
    :raises ValueError: when the template names a file that does not exist.
    """
    if template:
        path = str(template).format(plate=plate, well=well, well_id=well_id,
                                    well_number=well_id[1:], site=int(site))
        if not os.path.isfile(path):
            raise ValueError(f"no nucleus labels at {path} (ops_prefix_nuclei_masks)")
        if path.endswith(".npz"):
            with np.load(path) as saved:
                return saved["nuclei"]
        if path.endswith(".npy"):
            return np.load(path)
        import tifffile

        return tifffile.imread(path)
    if merged and os.path.isfile(merged):
        with np.load(merged) as saved:
            if "nuclei" in saved.files:
                return saved["nuclei"]
    return None


def _weight_plan(chosen: Mapping[str, Any], axes: Sequence[Tuple[str, Sequence[float]]]
                 ) -> Dict[str, Any]:
    """What a run's weighted majority and multi-guide table need per site (item 988).

    :param chosen: from :func:`prefix_settings`.
    :param axes: the tuning axes, or empty without tuning.
    :returns: ``weighted``, ``axes`` (the weight axes, one value each
        without tuning), ``gaps`` (the nucleus distance is needed),
        ``neighbours`` (the touching cells are needed) and ``multi`` (the
        multi-guide table is written).
    """
    weighted = chosen["ops_prefix_quality_method"] == "weighted_majority"
    named = {key: list(values) for key, values in axes if key in WEIGHT_GRID_KEYS}
    if weighted and not named:
        named = dict(_weight_axes(chosen, {}))
    multi = bool(chosen["ops_keep_multi_guide_cells"])
    return {"weighted": weighted, "axes": named, "multi": multi,
            "gaps": weighted and any(v > 0 for v in named.get(WEIGHT_GRID_KEYS[0], [])),
            "neighbours": multi or (weighted and any(
                v != 1.0 for v in named.get(WEIGHT_GRID_KEYS[1], [])))}


def _setting_counts(frame: pd.DataFrame, method: str, minority_k: int,
                    plan: Mapping[str, Any], chosen: Mapping[str, Any],
                    neighbours: Optional[np.ndarray], *, opportunity_profiles=None, sort=None):
    """Real and decoy calls of one site at one read setting, every weight setting included.

    :param frame: the site's reads at one read threshold, quality floor and
        mismatch limit.
    :param method: ``ops_prefix_quality_method``.
    :param minority_k: ``ops_prefix_minority_min_reads``.
    :param plan: from :func:`_weight_plan`.
    :param chosen: from :func:`prefix_settings`.
    :param neighbours: the site's touching cells, or None.
    :param opportunity_profiles: optional shared guide profiles; counts and
        opportunities then share exactly the same selected winners.
    :param sort: optional existing cell-call sort; sorted calls use
        ``_prefix_cell_calls`` directly, including its tie decisions.
    :returns: :func:`_called_counts`'s ``(real, decoy)``, or under the
        weighted majority (item 988) :func:`_weighted_counts`'s array over
        the weight axes.
    """
    if sort is not None:
        cells = _prefix_cell_calls(frame, keys=['cell'], min_reads=chosen['ops_quality_min_reads'],
                                    method=method, sort=sort, minority_min_reads=minority_k)
        hit = cells['quality'].isin(('perfect', 'great'))
        real = hit & ~cells['gene'].isin((DECOY_GENE, NEARMISS_GENE))
        counts = (int(real.sum()), int((hit & cells['gene'].eq(DECOY_GENE)).sum()))
        return {'counts': counts,
                'opportunities': _prefix_opportunities(cells['prefix'], real, opportunity_profiles)} \
            if opportunity_profiles else counts
    if not plan["weighted"]:
        return _called_counts(frame, method, minority_min_reads=minority_k,
                              opportunity_profiles=opportunity_profiles)
    named = plan["axes"]
    return _weighted_counts(
        frame, neighbours=neighbours, mode=chosen["ops_prefix_distance_weight"],
        decay=named[WEIGHT_GRID_KEYS[0]], discount=named[WEIGHT_GRID_KEYS[1]],
        quality=named[WEIGHT_GRID_KEYS[2]], share=chosen["ops_prefix_weighted_min_share"],
        interior_px=chosen["ops_prefix_distance_interior_px"],
        min_weight=chosen["ops_prefix_distance_min_weight"],
        opportunity_profiles=opportunity_profiles)


def _weight_arguments(chosen: Mapping[str, Any], best: Optional[Mapping[str, Any]]
                      ) -> Dict[str, Any]:
    """The weighted-majority arguments of :func:`prefix_cell_calls` (item 988).

    :param chosen: from :func:`prefix_settings`.
    :param best: the tuner's chosen setting, or None to use the entered one.
    :returns: the keyword arguments; the tuner's decay length of 0 turns the
        distance term off.
    """
    mode = chosen["ops_prefix_distance_weight"]
    decay = chosen["ops_prefix_distance_decay_px"]
    quality = chosen["ops_prefix_weighted_quality"]
    discount = chosen["ops_read_neighbour_discount"]
    if best is not None and WEIGHT_GRID_KEYS[0] in best:
        length = float(best[WEIGHT_GRID_KEYS[0]])
        quality, _ = _weight_terms(mode, bool(best[WEIGHT_GRID_KEYS[2]]), length)
        mode = "decay" if length > 0 else "off"
        decay = length if length > 0 else decay
        discount = float(best[WEIGHT_GRID_KEYS[1]])
    return {"distance_weight": mode, "decay_px": decay,
            "interior_px": chosen["ops_prefix_distance_interior_px"],
            "min_weight": chosen["ops_prefix_distance_min_weight"],
            "weighted_quality": quality, "neighbour_discount": discount,
            "weighted_min_share": chosen["ops_prefix_weighted_min_share"]}


def _neighbour_table(pairs: Optional[np.ndarray], plate: str, well_id: str,
                     site: int) -> Optional[pd.DataFrame]:
    """A site's touching cells in the layout :func:`prefix_cell_calls` takes (item 988).

    :param pairs: from :func:`_touching_cells`, or None.
    :param plate: the plate.
    :param well_id: the well.
    :param site: the site.
    :returns: rows with ``plate``, ``well``, ``site``, ``cell`` and
        ``neighbour``, or None.
    """
    if pairs is None:
        return None
    return pd.DataFrame({"plate": plate, "well": well_id, "site": int(site),
                         "cell": pairs[:, 0], "neighbour": pairs[:, 1]})


def _multiplicity_table(counts: Mapping[str, int], cells: int, min_reads: int) -> pd.DataFrame:
    """A well's guide multiplicity QC table (item 988).

    :param counts: the summed :func:`_guide_multiplicity` counts of its
        sites.
    :param cells: the cells in the sites' labels.
    :param min_reads: ``ops_multi_guide_min_reads``.
    :returns: two rows, one per counting rule: ``counting``, ``cells`` and
        the % of cells with 0, 1, 2, 3 and 4 or more guides and with a
        decoy barcode; then ``bleed_tested`` (cells with two or more
        guides and one clear top guide), ``bleed_neighbour`` (of those, the
        cells whose runner-up guide is a touching cell's top guide) and
        ``bleed_pct``.
    """
    rows = []
    tested, hit = int(counts.get("bleed_tested", 0)), int(counts.get("bleed_neighbour", 0))
    for rule, label in (("all", "all matched reads"),
                        ("min", f"guides with at least {int(min_reads)} reads")):
        with_guide = sum(int(counts.get(f"{rule}_{k}", 0)) for k in ("1", "2", "3", "ge4"))
        pct = {f"pct_{k}": 100.0 * int(counts.get(f"{rule}_{k}", 0)) / max(cells, 1)
               for k in ("1", "2", "3", "ge4")}
        rows.append({"counting": label, "cells": int(cells),
                     "pct_0": 100.0 * (int(cells) - with_guide) / max(cells, 1), **pct,
                     "pct_with_decoy": 100.0 * int(counts.get(f"{rule}_decoy", 0)) / max(cells, 1),
                     "bleed_tested": tested, "bleed_neighbour": hit,
                     "bleed_pct": 100.0 * hit / tested if tested else float("nan")})
    return pd.DataFrame(rows)


def _site_geometry(plan: Mapping[str, Any], chosen: Mapping[str, Any],
                   labels: Optional[np.ndarray], y: np.ndarray, x: np.ndarray,
                   cell: np.ndarray, *, plate: str, well: str, well_id: str, site: int,
                   destination: str,
                   frame: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[np.ndarray],
                                              Optional[Dict[str, np.ndarray]]]:
    """A site's touching cells and its reads' nucleus distances, when the run needs them (item 988).

    :param plan: from :func:`_weight_plan`.
    :param chosen: from :func:`prefix_settings`.
    :param labels: the site's cell labels, or None.
    :param y: the reads' rows.
    :param x: the reads' columns.
    :param cell: the reads' cells.
    :param plate: the plate.
    :param well: the well folder name.
    :param well_id: the well.
    :param site: the site.
    :param destination: ``dst_root``, for the ``merged/`` fallback.
    :returns: ``(pairs, extra)``: :func:`_touching_cells` or None, and
        ``nuc_gap_px`` and ``nuc_gap_fallback`` per read or None.
    :raises ValueError: when the distance or the touching cells are needed
        and the site has no cell or nucleus labels.
    """
    from .cpg_cropped import merged_path

    pairs = extra = None
    if not (plan["neighbours"] or plan["gaps"]):
        return pairs, extra
    if labels is None:
        raise ValueError("the weighted majority's distance term and neighbour discount and "
                         "the multi-guide table need cell labels (ops_prefix_masks)")
    if plan["neighbours"]:
        pairs = _touching_cells(labels)
    if plan["gaps"]:
        nuclei = _site_nuclei(chosen["ops_prefix_nuclei_masks"], plate=plate, well=well,
                              well_id=well_id, site=site,
                              merged=merged_path(destination, plate, well_id, site))
        if nuclei is None:
            raise ValueError("ops_prefix_distance_weight needs nucleus labels; set "
                             "ops_prefix_nuclei_masks")
        if frame is not None:
            from .cpg_cropped import shift_image

            nuclei = shift_image(nuclei, frame["dy"], frame["dx"])
        gap, fallback = _read_nucleus_gaps(y, x, cell, labels, nuclei)
        extra = {"nuc_gap_px": gap, "nuc_gap_fallback": fallback}
    return pairs, extra


def _write_multiplicity(folder: str, plate: str, well_id: str, counts: Mapping[str, int],
                        cells: int, tables: Sequence[pd.DataFrame],
                        min_reads: int) -> Dict[str, Any]:
    """Write a well's guide multiplicity table and its multi-guide cells (item 988).

    :param folder: the run's ``ops_prefix`` folder.
    :param plate: the plate.
    :param well_id: the well.
    :param counts: the summed :func:`_guide_multiplicity` counts.
    :param cells: the cells in the well's labels.
    :param tables: the sites' multi-guide cell tables.
    :param min_reads: ``ops_multi_guide_min_reads``.
    :returns: the two paths and ``multi_guide_cells``, the number of
        multi-guide cells.
    """
    from .tabular import write_table

    qc = os.path.join(folder, f"guide_multiplicity_{plate}_{well_id}.csv")
    listed = os.path.join(folder, f"multi_guide_cells_{plate}_{well_id}.csv")
    write_table(_multiplicity_table(counts, cells, min_reads), qc, canonicalise=False)
    kept = [table for table in tables if len(table)]
    multi = pd.concat(kept, ignore_index=True) if kept else (
        tables[0] if tables else pd.DataFrame())
    write_table(multi, listed, canonicalise=False)
    return {"multiplicity": qc, "cells": listed, "multi_guide_cells": int(len(multi))}


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
      cells);
    * with decoys (``ops_prefix_decoy_k`` above 0, item 942): a ``decoy``
      column in the cell tables (a decoy call keeps its prefix but no
      genotype), ``cells_in_labels``, ``cells_real``, ``cells_decoy`` and
      ``fdr_pct`` in ``site_summary.csv``, and ``fdr_<plate>_<well_id>.json``
      (:func:`spacr.ops_decode.fdr_report`);
    * with ``ops_prefix_threshold_tuning`` 'per_well_decoy': per well
      ``threshold_tuning_<plate>_<well_id>.csv`` and ``.png`` (every grid
      setting's tuning numbers and the genotyped-vs-FDR curve,
      :func:`spacr.ops_decode.write_tuning_curve`) and ``.json`` (the choice,
      :func:`spacr.ops_decode.tune_thresholds`); each site's spots are found
      once, down to the lowest read threshold, and the chosen setting's reads
      and cells are written;
    * with near-miss decoys (``ops_prefix_nearmiss_decoy_k`` above 0, item
      949, :func:`prefix_nearmiss_decoys`): a ``decoy_class`` column
      (``'far'`` or ``'nearmiss'``) in the cell tables,
      ``cells_decoy_nearmiss`` and ``fdr_nearmiss_pct`` (near-miss calls per
      100 real calls over ``ops_prefix_nearmiss_decoy_k``, scale 1) in
      ``site_summary.csv``, and the near-miss count, FDR and its interval in
      ``fdr_<plate>_<well_id>.json``; ``cells_decoy`` and ``fdr_pct`` keep
      counting the far decoys only;
    * with ``ops_prefix_fdr_calibration`` 'shuffle' (item 949): every read is
      also matched against :data:`spacr.ops_decode.SHUFFLE_SEEDS` shuffled
      whitelists (:func:`shuffled_library`, seeds ``ops_control_seed`` and
      up), and each well's FDR scale is raised from that control
      (:func:`spacr.ops_decode.calibrate_fdr_scale`; without tuning at the
      entered setting); ``fdr_pct`` uses the calibrated scale, and
      ``site_summary.csv`` gains ``shuffle_called`` (mean over the shuffled
      whitelists), ``shuffle_call_rate_pct`` and ``fdr_scale``; the FDR
      report gains the entered scale, the calibration, the FDR's interval
      and the shuffle call rate with its interval
      (:func:`spacr.ops_decode.shuffle_call_rate`). With both item-949
      settings off the outputs are those of item 942;
    * with ``ops_prefix_minority_min_reads`` 2 or more (item 967, alpha): the
      cell calls and the tuning leave out minority stray reads
      (:func:`prefix_cell_calls`), the cell table gains
      ``minority_ignored`` and ``minority_rescued``, ``site_summary.csv``
      gains ``cells_minority_rescued`` (genotyped cells called only because
      of the rule), and one line per well reports the rescued cells. 0, the
      default, writes what the run wrote before;
    * with ``ops_prefix_quality_method`` 'weighted_majority' (item 988,
      alpha): cells are called by their barcodes' summed read weights
      (:func:`prefix_cell_calls`, classes :data:`WEIGHTED_CLASSES`), the
      tuning also tries the weight axes (:data:`WEIGHT_GRID_KEYS`;
      :data:`PREFIX_TUNING_GRID_WEIGHTED` when the grid is empty), and the
      cell table gains the weight columns. With a distance term the read
      table gains ``nuc_gap_px`` and ``nuc_gap_fallback``, measured to the
      nuclei of ``ops_prefix_nuclei_masks``;
    * with ``ops_keep_multi_guide_cells`` (item 988, alpha): per well
      ``guide_multiplicity_<plate>_<well_id>.csv`` (the share of cells by
      number of guides, and the share of two-guide cells whose runner-up is
      a touching cell's top guide) and ``multi_guide_cells_<plate>_<well_id>.csv``
      (the cells with two or more guides of ``ops_multi_guide_min_reads``
      reads, none of them from a neighbour). With both item-988 settings
      off the run writes what it wrote before.
    * with ``ops_prefix_tuning_min_sites`` above 0 (item 982 small-well
      guard, alpha): a tuned well with fewer sites keeps the read-quality
      floor at :data:`SMALL_WELL_FLOOR` and tunes only the other settings;
      the log says so, and the well's tuning json and ``tuning`` entry gain
      ``small_well_floor`` and ``small_well_sites``. 0, the default, writes
      what the run wrote before.

    Never-barcode calibration (item 1014) adds independent distance-one
    and distance-two control passes through the same prefix matcher and
    cell votes. Their calls never replace production calls. The shared
    estimator calibrates on tuning sites and reports held-out sites;
    fixed thresholds use a one-setting grid. The shuffle call rate is
    reported alongside never calibration and enters its scale only with
    ``'never+shuffle'``. Each well writes ``fdr_calibration_<well>.json``.

    :param settings: the OPS settings: ``dst_root``, the ``cpg_*`` reader
        settings, the item-920, item-942, item-949, item-967 and item-988
        settings
        (:func:`prefix_settings`)
        and, for tuning, the shared tuner's settings
        (``ops_decode_auto_objective``, ``ops_decode_auto_f_beta``,
        ``ops_decode_auto_unreachable``, ``ops_decode_auto_ci``,
        ``ops_decode_auto_tune_fraction`` and ``ops_decode_auto_seed``).
    :param report: called with progress lines.
    :returns: the output paths, the library summary and the run's totals;
        with any item-942 setting on also ``seconds_per_well``, ``fdr`` (per
        well) and ``tuning`` (per well: the choice and its numbers); with
        the shuffle calibration also ``calibration`` (per well); with the
        minority-read rule (item 967) also ``minority_rescued`` (genotyped
        cells rescued, per well); with ``ops_keep_multi_guide_cells`` (item
        988) also ``multi_guide`` (per well: the two table paths and the
        number of multi-guide cells); with tuning also ``well_choices``, one
        line per well (item 966): the read mode and thresholds chosen and
        the decoy FDR on the tuning and the report sites, also sent to
        ``report`` last. An empty ``ops_read_mode`` is
        ``'auto'`` with tuning and ``'log'`` without.
    :raises ValueError: without ``dst_root`` or a library, when the layout
        has fewer sequencing cycles than the prefix needs, for tuning,
        near-miss decoys or a shuffle calibration without decoys, or for a
        distance term or the touching cells without the labels they need.
    """
    from . import __version__
    from .cpg_cropped import cpg_site_index, load_cpg_site, merged_path
    from .ops_decode import fdr, fdr_report, tune_thresholds, _tuning_options, \
        write_tuning_curve, calibrate_fdr_scale, SHUFFLE_SEEDS, _write_calibration_report
    from .tabular import write_table

    say = report or (lambda text: None)
    destination = settings.get("dst_root")
    if not destination:
        raise ValueError("the prefix_decode phase needs dst_root")
    started = time.perf_counter()
    from .ops_insitu import _compartment_labels, _insitu_options, _run_combinatorial_decode

    chosen = prefix_settings(settings)
    insitu = _insitu_options(settings, chosen)
    if insitu["layout"] == "combinatorial":
        return _run_combinatorial_decode(settings, report=report)
    if insitu["min_reads"] is not None:
        chosen = {**chosen, "ops_quality_min_reads": insitu["min_reads"]}
    read_mode = _prefix_read_mode(settings, chosen)
    if read_mode == "auto" and chosen["ops_prefix_threshold_tuning"] != "per_well_decoy":
        raise ValueError("ops_read_mode 'auto' chooses the read per well from decoy "
                         "calls; set ops_prefix_threshold_tuning 'per_well_decoy' and "
                         "ops_prefix_decoy_k above 0, or choose 'log' or 'raw'")
    library = spacer_prefix_library(
        chosen["ops_library"], sequence_col=chosen["ops_library_sequence_col"],
        gene_col=chosen["ops_library_gene_col"], prefix_length=chosen["ops_prefix_length"],
        n_cycles=chosen["ops_sbs_cycles"], cycle_mask=chosen["ops_cycle_mask"],
        collision=chosen["ops_prefix_collision"])
    decoy_k = chosen["ops_prefix_decoy_k"]
    tuning = chosen["ops_prefix_threshold_tuning"] == "per_well_decoy"
    min_quality = chosen["ops_prefix_min_read_quality"]
    nearmiss_k = chosen["ops_prefix_nearmiss_decoy_k"]
    calibration_method = chosen["ops_prefix_fdr_calibration"]
    using_never = calibration_method.startswith("never") and decoy_k > 0
    calibration = calibration_method != "none" and decoy_k > 0
    report_shuffles = calibration
    minority_k = chosen["ops_prefix_minority_min_reads"]
    minority = minority_k > 1
    if nearmiss_k > 0 and decoy_k < 1:
        raise ValueError("ops_prefix_nearmiss_decoy_k reports near-miss decoys next to the "
                         "far decoys; set ops_prefix_decoy_k to 1 or more (4 recommended)")
    legacy = decoy_k == 0 and not tuning and min_quality <= 0
    item_949 = nearmiss_k > 0 or report_shuffles
    matching = (prefix_decoy_library(library, decoy_k, chosen["ops_prefix_decoy_seed"])
                if decoy_k > 0 else library)
    if nearmiss_k > 0:
        matching = prefix_nearmiss_decoys(matching, nearmiss_k, chosen["ops_prefix_decoy_seed"])
    never_context = _prefix_never_context(library, matching, chosen) if using_never else None
    axes = _tuning_axes(chosen) if tuning else []
    plan = _weight_plan(chosen, axes)
    options = _tuning_options(settings) if tuning or calibration else {}
    control = (shuffled_library(library, chosen["ops_control_seed"])
               if chosen["ops_random_barcode_control"] else None)
    shuffles = ([shuffled_library(library, chosen["ops_control_seed"] + s)
                 for s in range(SHUFFLE_SEEDS)] if report_shuffles else [])
    shuffle_memo: List[Dict[str, tuple]] = [{} for _ in shuffles]
    check_masks_frame(chosen)
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
        json.dump(matching.summary, handle, indent=1)
    method = chosen["ops_prefix_quality_method"]
    class_names = (RECIPE_SIMPLE_CLASSES if method == "recipe_simple" else
                   WEIGHTED_CLASSES if method == "weighted_majority" else QUALITY_CLASSES)
    recorded = ({key: value for key, value in chosen.items() if key not in _ITEM_942_KEYS}
                if legacy else chosen)
    if not item_949:
        recorded = {key: value for key, value in recorded.items() if key not in _ITEM_949_KEYS}
    if not minority:
        recorded = {key: value for key, value in recorded.items() if key not in _ITEM_967_KEYS}
    if not (plan["weighted"] or plan["multi"]):
        recorded = {key: value for key, value in recorded.items() if key not in _ITEM_988_KEYS}
    if not chosen["ops_prefix_tuning_min_sites"]:
        recorded = {key: value for key, value in recorded.items()
                    if key not in _SMALL_WELL_KEYS}
    if chosen["ops_prefix_masks_frame"] == "sbs":
        recorded = {key: value for key, value in recorded.items()
                    if key not in _ITEM_919_FRAME_KEYS}
    if read_mode != "log":
        recorded = {**recorded, "ops_read_mode": read_mode}
    if insitu["recorded"]:
        recorded = {**recorded, **insitu["recorded"]}
    with open(os.path.join(folder, "quality_classes.json"), "w", encoding="utf-8") as handle:
        json.dump({"method": method, "classes": class_names,
                   "source": RECIPE_SIMPLE_SOURCE if method == "recipe_simple" else None,
                   "settings": recorded,
                   "bases": pinned["cpg_sbs_bases"], "spacr_version": __version__},
                  handle, indent=1, default=str)
    index = built["index"]
    site_rows, all_cells, tuned, well_fdr, well_seconds = [], [], {}, {}, {}
    well_lines: List[str] = []
    calibrated: Dict[str, Any] = {}
    rescued_wells: Dict[str, int] = {}
    multi_paths: Dict[str, Dict[str, Any]] = {}
    memo: Dict[str, tuple] = {}
    for (plate, well, well_id), sites in index.groupby(["plate", "well", "well_id"], sort=True):
        well_started = time.perf_counter()
        site_list = [int(site) for site in sorted(sites["site"].unique())]
        site_reads: Dict[int, pd.DataFrame] = {}
        site_cells: Dict[int, pd.DataFrame] = {}
        site_cells_n: Dict[int, int] = {}
        site_labelled: Dict[int, bool] = {}
        site_shuffled: Dict[int, float] = {}
        site_pairs: Dict[int, Optional[np.ndarray]] = {}
        site_frames: Dict[int, Dict[str, Any]] = {}
        site_extra: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
        well_multi: Dict[str, int] = {}
        multi_rows: List[pd.DataFrame] = []
        multi_cells = 0
        scale = chosen["ops_prefix_fdr_scale"]
        choice: Dict[str, Any] = {}
        if tuning:
            modes = ["raw", "log"] if read_mode == "auto" else [read_mode]
            grid_axes, small_well = _small_well_axes(axes, len(site_list),
                                                     chosen["ops_prefix_tuning_min_sites"])
            if small_well:
                say(f"prefix_decode {plate} {well_id}: {len(site_list)} sites, fewer than "
                    f"ops_prefix_tuning_min_sites {chosen['ops_prefix_tuning_min_sites']}; "
                    f"the read-quality floor stays at {SMALL_WELL_FLOOR} and is not tuned")
            well_axes = ([("read_mode", [1.0, 0.0])] if read_mode == "auto" else []) + grid_axes
            shape = tuple(len(values) for _, values in well_axes)
            counts = np.zeros((len(site_list),) + shape + (2,), np.int64)
            shuffled_grid = np.zeros((len(site_list),) + shape)
            never_grid = ({key: np.zeros((len(site_list),) + shape, np.int64)
                           for key in ([f'opp{d}_{name}' for name in ('L', 'D', 'N1', 'N2')
                                        for d in (1, 2, 3)]
                                       + [key for name in never_context['controls']
                                          for key in (f'never{name[-1]}_calls',
                                                      f'real_calls_{name.lower()}')])}
                          if using_never else None)
            held = {}
            held_raw = {}
            loosest = int(max(axes[2][1]))
            for i, site in enumerate(site_list):
                loaded = load_cpg_site(index, plate, well, site, sbs_cycles=cycles,
                                       sbs_bases=pinned["cpg_sbs_bases"], sbs_dapi=None,
                                       parts=("sbs",))
                labels = site_labels(chosen["ops_prefix_masks"], plate=plate, well=well,
                                     well_id=well_id, site=site,
                                     merged=merged_path(str(destination), plate, well_id,
                                                        site))
                if insitu["compartment"] != "cell":
                    labels = _compartment_labels(labels, insitu, plate=plate, well=well,
                                                 well_id=well_id, site=site,
                                                 destination=str(destination))
                labels = frame_labels(labels, chosen, index=index, plate=plate, well=well,
                                      site=site, cpg_root=pinned["cpg_root"],
                                      record=site_frames)
                stack = loaded["sbs"]
                if labels is not None and np.asarray(labels).shape != stack.shape[1:3]:
                    raise ValueError(f"the cell labels are {np.asarray(labels).shape}, "
                                     f"the sequencing stack {stack.shape[1:3]}")
                peaks, strength, values = _spot_candidates(stack, floor=min(axes[0][1]))
                cell = (np.asarray(labels)[peaks[:, 0], peaks[:, 1]].astype(np.int64)
                        if labels is not None and len(peaks)
                        else np.zeros(len(peaks), np.int64))
                held[site] = (peaks, strength, values, cell)
                site_pairs[site], site_extra[site] = _site_geometry(
                    plan, chosen, labels, peaks[:, 0], peaks[:, 1], cell, plate=plate,
                    well=well, well_id=well_id, site=site, destination=str(destination),
                    frame=site_frames.get(site))
                if "raw" in modes:
                    held_raw[site] = _raw_candidates(stack, held[site])
                del loaded, stack
                site_cells_n[site] = _labelled_cells(labels, pd.DataFrame())
                site_labelled[site] = labels is not None
                for m, mode in enumerate(modes):
                    source = held_raw if mode == "raw" else held
                    at = (i, m) if read_mode == "auto" else (i,)
                    for a, threshold in enumerate(axes[0][1]):
                        frame = _site_tuning_reads(
                            source[site], matching, memo, bases=pinned["cpg_sbs_bases"],
                            threshold=threshold, crosstalk=chosen["ops_prefix_crosstalk"],
                            metric=chosen["ops_prefix_metric"], max_distance=loosest,
                            read_mode=mode, extra=site_extra[site])
                        for b, quality in enumerate(grid_axes[1][1]):
                            for c, distance in enumerate(axes[2][1]):
                                if using_never:
                                    result = _prefix_never_counts(
                                        frame, never_context, quality=quality, distance=int(distance),
                                        metric=chosen["ops_prefix_metric"], method=method,
                                        minority_k=minority_k, plan=plan, chosen=chosen,
                                        neighbours=site_pairs[site], sort=insitu["call_sort"])
                                    counts[at + (a, b, c)] = result['counts']
                                    for key, value in result['never'].items():
                                        never_grid[key][at + (a, b, c)] = value
                                else:
                                    counts[at + (a, b, c)] = _setting_counts(
                                        _at_setting(frame, quality, int(distance)), method,
                                        minority_k, plan, chosen, site_pairs[site])
                        for s, shuffled_lib in enumerate(shuffles):
                            shuffled_frame = _shuffle_frame(
                                frame, shuffled_lib, shuffle_memo[s],
                                metric=chosen["ops_prefix_metric"], max_distance=loosest)
                            for b, quality in enumerate(grid_axes[1][1]):
                                for c, distance in enumerate(axes[2][1]):
                                    shuffled_grid[at + (a, b, c)] += np.asarray(_setting_counts(
                                        _at_setting(shuffled_frame, quality, int(distance)),
                                        method, minority_k, plan, chosen,
                                        site_pairs[site]))[..., 0]
                if labels is None:
                    first = (i, len(modes) - 1) if read_mode == "auto" else (i,)
                    site_cells_n[site] = int(counts[first + (0,) * len(axes)].sum())
                say(f"prefix_decode {plate} {well_id} site {site}: {len(peaks)} spot "
                    f"candidates scored over {int(np.prod(shape))} settings")
            tune_args = dict(
                k=decoy_k, scale=chosen["ops_prefix_fdr_scale"],
                fdr_max=chosen["ops_prefix_fdr_target_pct"],
                tune_fraction=float(settings.get("ops_decode_auto_tune_fraction") or 0.5),
                seed=int(settings.get("ops_decode_auto_seed") or 0), **options)
            if shuffles:
                shuffled_grid /= float(len(shuffles))
            if calibration:
                from .ops_decode import _never_payload

                never = _never_payload(never_grid, never_context) if using_never else None
                choice = calibrate_fdr_scale(
                    counts, shuffled_grid, [site_cells_n[s] for s in site_list], site_list,
                    well_axes, method=calibration_method, never=never, **tune_args)
                scale = choice["fdr_scale"]
                for i, site in enumerate(site_list):
                    site_shuffled[site] = float(shuffled_grid[(i,) + tuple(choice["index"])])
            else:
                choice = tune_thresholds(
                    counts, [site_cells_n[s] for s in site_list], site_list, well_axes,
                    **tune_args)
            if shuffles:
                for i, site in enumerate(site_list):
                    site_shuffled[site] = float(shuffled_grid[(i,) + tuple(choice["index"])])
            stem = f"threshold_tuning_{plate}_{well_id}"
            paths = write_tuning_curve(choice, folder, stem,
                                       title=f"{plate} {well_id}: prefix decode tuning")
            best = choice["chosen"]
            well_mode = (("raw" if best["read_mode"] == 1.0 else "log")
                         if read_mode == "auto" else read_mode)
            mode_record = {"read_mode": well_mode} if read_mode != "log" else {}
            if small_well:
                mode_record = {**mode_record, "small_well_floor": SMALL_WELL_FLOOR,
                               "small_well_sites": len(site_list)}
            with open(os.path.join(folder, f"{stem}.json"), "w", encoding="utf-8") as handle:
                json.dump({"plate": plate, "well": well_id,
                           **{key: value for key, value in choice.items() if key != "grid"},
                           **mode_record, "paths": paths}, handle, indent=1, default=str)
            tuned[f"{plate}_{well_id}"] = {**{key: choice[key] for key in (
                "chosen", "auto_met", "fdr_lowest", "tune", "report", "all_fields")},
                **mode_record}
            if calibration:
                tuned[f"{plate}_{well_id}"].update(
                    {"fdr_scale": choice["fdr_scale"], "calibration": choice["calibration"]})
            say(f"prefix_decode {plate} {well_id}: tuned {choice['chosen']}, report sites "
                f"{choice['report']['genotyped_pct']} % genotyped, FDR "
                f"{choice['report']['fdr_per_100']} %")
            well_lines.append(_well_choice_line(plate, well_id, choice, well_mode, read_mode))
            if small_well:
                well_lines[-1] += (f"; small well ({len(site_list)} sites): read-quality floor "
                                   f"kept at {SMALL_WELL_FLOOR}")
            for site in site_list:
                frame = _site_tuning_reads(
                    (held_raw if well_mode == "raw" else held)[site], matching, memo,
                    bases=pinned["cpg_sbs_bases"],
                    threshold=best["ops_read_threshold"],
                    crosstalk=chosen["ops_prefix_crosstalk"],
                    metric=chosen["ops_prefix_metric"], max_distance=loosest,
                    read_mode=well_mode, extra=site_extra[site],
                    peak=insitu["call_sort"] == "peak")
                frame = _at_setting(frame, best["ops_prefix_min_read_quality"],
                                    int(best["ops_prefix_max_distance"])).reset_index(drop=True)
                if control is not None:
                    frame["control_status"] = match_prefix_reads(
                        frame["read"].tolist(), control, metric=chosen["ops_prefix_metric"],
                        max_distance=int(best["ops_prefix_max_distance"]))["status"].to_numpy()
                site_reads[site] = frame
            del held, held_raw
        for site in site_list:
            if tuning:
                reads = site_reads.pop(site)
                labelled = site_labelled[site]
                n_labelled = site_cells_n[site]
            else:
                loaded = load_cpg_site(index, plate, well, site, sbs_cycles=cycles,
                                       sbs_bases=pinned["cpg_sbs_bases"], sbs_dapi=None,
                                       parts=("sbs",))
                labels = site_labels(chosen["ops_prefix_masks"], plate=plate, well=well,
                                     well_id=well_id, site=site,
                                     merged=merged_path(str(destination), plate, well_id,
                                                        site))
                if insitu["compartment"] != "cell":
                    labels = _compartment_labels(labels, insitu, plate=plate, well=well,
                                                 well_id=well_id, site=site,
                                                 destination=str(destination))
                labels = frame_labels(labels, chosen, index=index, plate=plate, well=well,
                                      site=site, cpg_root=pinned["cpg_root"],
                                      record=site_frames)
                reads = _decode_site(
                    loaded["sbs"], matching, bases=pinned["cpg_sbs_bases"], labels=labels,
                    threshold=chosen["ops_read_threshold"],
                    crosstalk=chosen["ops_prefix_crosstalk"],
                    metric=chosen["ops_prefix_metric"],
                    max_distance=chosen["ops_prefix_max_distance"], control=control,
                    read_mode=read_mode, peak=insitu["call_sort"] == "peak")
                del loaded
                if min_quality > 0:
                    reads = reads[reads["quality"] >= min_quality].reset_index(drop=True)
                labelled = labels is not None
                n_labelled = None if legacy else _labelled_cells(labels, pd.DataFrame())
                site_pairs[site], extra = _site_geometry(
                    plan, chosen, labels, reads["y"].to_numpy(), reads["x"].to_numpy(),
                    reads["cell"].to_numpy(), plate=plate, well=well, well_id=well_id,
                    site=site, destination=str(destination), frame=site_frames.get(site))
                for name, column in (extra or {}).items():
                    reads[name] = column
                if plan["multi"]:
                    site_cells_n[site] = _labelled_cells(labels, pd.DataFrame())
                if shuffles:
                    site_shuffled[site] = float(np.mean([np.asarray(_setting_counts(
                        _shuffle_frame(reads, shuffled_lib, shuffle_memo[s],
                                       metric=chosen["ops_prefix_metric"],
                                       max_distance=chosen["ops_prefix_max_distance"]),
                        method, minority_k, plan, chosen, site_pairs[site])).reshape(-1, 2)[0, 0]
                        for s, shuffled_lib in enumerate(shuffles)]))
            never_row = None
            if using_never and not tuning:
                never_row = _prefix_never_counts(
                    reads, never_context, quality=0.0,
                    distance=chosen["ops_prefix_max_distance"],
                    metric=chosen["ops_prefix_metric"], method=method,
                    minority_k=minority_k, plan=plan, chosen=chosen,
                    neighbours=site_pairs[site], sort=insitu["call_sort"])['never']
            reads.insert(0, "site", int(site))
            reads.insert(0, "well", well_id)
            reads.insert(0, "plate", plate)
            weighting = ({**_weight_arguments(chosen, choice.get("chosen") if tuning else None),
                          "neighbours": _neighbour_table(site_pairs.get(site), plate, well_id,
                                                         site)}
                         if plan["weighted"] else {})
            row, cells = _site_row(
                reads, plate=plate, well_id=well_id, site=site, chosen=chosen,
                method=method, class_names=class_names, labelled=labelled,
                n_labelled=n_labelled, decoy_k=decoy_k, nearmiss_k=nearmiss_k,
                scale=scale, shuffled=site_shuffled.get(site, 0.0) if report_shuffles else None,
                weighting=weighting, sort=insitu["call_sort"])
            if never_row is not None:
                row.update({key: float(np.asarray(value).reshape(-1)[0])
                            for key, value in never_row.items()})
            if plan["multi"]:
                site_counts, site_table = _guide_multiplicity(
                    reads, site_pairs.get(site), chosen["ops_multi_guide_min_reads"],
                    ["plate", "well", "site", "cell"])
                for key, value in site_counts.items():
                    well_multi[key] = well_multi.get(key, 0) + int(value)
                multi_cells += int(site_cells_n.get(site, 0) or len(cells))
                multi_rows.append(site_table)
            site_rows.append(row)
            say(f"prefix_decode {plate} {well_id} site {site}: {len(reads)} reads, "
                f"{len(cells)} cells")
            site_reads[site] = reads
            site_cells[site] = cells
        reads_all = pd.concat([site_reads.pop(s) for s in site_list], ignore_index=True)
        cells_all = pd.concat([site_cells.pop(s) for s in site_list], ignore_index=True)
        write_table(reads_all, os.path.join(folder, f"reads_{plate}_{well_id}.parquet"),
                    canonicalise=False)
        write_table(cells_all, os.path.join(folder, f"cells_{plate}_{well_id}.csv"),
                    canonicalise=False)
        if plan["multi"]:
            multi_paths[f"{plate}_{well_id}"] = _write_multiplicity(
                folder, plate, well_id, well_multi, multi_cells, multi_rows,
                chosen["ops_multi_guide_min_reads"])
            say(f"prefix_decode {plate} {well_id}: "
                f"{multi_paths[f'{plate}_{well_id}']['multi_guide_cells']} multi-guide cells "
                f"(ops_keep_multi_guide_cells)")
        if decoy_k > 0:
            here = [r for r in site_rows if r["plate"] == plate and r["well"] == well_id]
            if calibration and not tuning:
                choice = _calibrate_fixed(here, chosen, decoy_k, settings, options,
                                          never_context=never_context)
                scale = choice["fdr_scale"]
                for r in here:
                    r.update({"fdr_pct": fdr(r["cells_real"], r["cells_decoy"], decoy_k, scale),
                              "fdr_scale": scale})
            stats = pd.DataFrame([{"nuclei": r["cells_in_labels"],
                                   "assigned_real": r["cells_real"],
                                   "assigned_decoy": r["cells_decoy"]}
                                  for r in here])
            well_fdr[f"{plate}_{well_id}"] = fdr_report(
                stats, plate=plate, well=well_id, k=decoy_k, scale=scale,
                calibration=choice["calibration"] if calibration else None)
            if item_949:
                well_fdr[f"{plate}_{well_id}"].update(_well_report_949(
                    here, decoy_k=decoy_k, scale=scale, nearmiss_k=nearmiss_k,
                    entered_scale=chosen["ops_prefix_fdr_scale"],
                    choice=choice if calibration else None))
                if calibration:
                    calibrated[f"{plate}_{well_id}"] = {
                        "fdr_scale": scale, "calibration": choice["calibration"]}
            if report_shuffles and not calibration:
                from .ops_decode import shuffle_call_rate

                well_fdr[f"{plate}_{well_id}"].update(shuffle_call_rate(
                    [r["shuffle_called"] for r in here], [r["cells_in_labels"] for r in here]))
            if using_never:
                _write_calibration_report(folder, plate, well_id, choice['calibration'],
                                           csv=False, log=say)
            with open(os.path.join(folder, f"fdr_{plate}_{well_id}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(well_fdr[f"{plate}_{well_id}"], handle, indent=1)
        if calibration_method.startswith("never") and decoy_k < 1:
            fallback = {"method": "none", "requested_method": calibration_method,
                        "status": "no_decoys", "entered_scale": scale, "scale": scale,
                        "rounds": [], "converged": False}
            _write_calibration_report(folder, plate, well_id, fallback, csv=False, log=say)
        all_cells.append(cells_all)
        if minority:
            here_rescued = sum(r["cells_minority_rescued"] for r in site_rows
                               if r["plate"] == plate and r["well"] == well_id)
            rescued_wells[f"{plate}_{well_id}"] = int(here_rescued)
            say(f"prefix_decode {plate} {well_id}: the minority-read rule "
                f"(ops_prefix_minority_min_reads {minority_k}) rescued {here_rescued} "
                f"genotyped cells")
        if site_frames:
            write_table(pd.DataFrame([{"plate": plate, "well": well_id, "site": int(site), **found}
                                      for site, found in sorted(site_frames.items())]),
                        os.path.join(folder, f"label_frame_{plate}_{well_id}.csv"),
                        canonicalise=False)
            say(f"prefix_decode {plate} {well_id}: painting-frame labels moved onto the "
                f"sequencing frame on {len(site_frames)} sites "
                f"(cpg_align_source {chosen['cpg_align_source']})")
        well_seconds[f"{plate}_{well_id}"] = round(time.perf_counter() - well_started, 1)
    summary = pd.DataFrame(site_rows)
    write_table(summary, os.path.join(folder, "site_summary.csv"), canonicalise=False)
    cells = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    called = cells[cells["quality"].isin(("perfect", "great"))] if len(cells) else cells
    if len(called) and "decoy" in called.columns:
        called = called[~called["decoy"].astype(bool)]
    per_guide = (called.groupby(["guide", "gene"]).size().rename("cells").reset_index()
                 if len(called) else pd.DataFrame(columns=["guide", "gene", "cells"]))
    per_gene = (called.groupby("gene").size().rename("cells").reset_index()
                if len(called) else pd.DataFrame(columns=["gene", "cells"]))
    write_table(per_guide, os.path.join(folder, "cells_per_guide.csv"), canonicalise=False)
    write_table(per_gene, os.path.join(folder, "cells_per_gene.csv"), canonicalise=False)
    totals = {key: int(summary[key].sum()) for key in
              ("reads", "exact", "corrected", "ambiguous", "collision", "out_of_library")} \
        if len(summary) else {}
    out = {"folder": folder, "library": library.summary, "sites": int(len(summary)),
           "totals": totals, "cells": int(len(cells)), "called_cells": int(len(called)),
           "seconds": round(time.perf_counter() - started, 1)}
    if not legacy:
        out.update({"seconds_per_well": well_seconds, "fdr": well_fdr, "tuning": tuned})
    if calibration:
        out["calibration"] = calibrated
    if minority:
        out["minority_rescued"] = rescued_wells
    if plan["multi"]:
        out["multi_guide"] = multi_paths
    if well_lines:
        out["well_choices"] = well_lines
        for line in well_lines:
            say(line)
    return out


def _site_row(reads: pd.DataFrame, *, plate: str, well_id: str, site: int,
              chosen: Mapping[str, Any], method: str, class_names: Sequence[str],
              labelled: bool, n_labelled: Optional[int], decoy_k: int, nearmiss_k: int,
              scale: float, shuffled: Optional[float],
              weighting: Optional[Mapping[str, Any]] = None,
              sort: Optional[str] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """One site's cell calls and its ``site_summary.csv`` row.

    :func:`run_prefix_decode` and the parameter grid (item 960,
    :mod:`spacr.ops_grid_search`) both call this, so a grid point counts its
    cells exactly as a run at that setting does.

    :param reads: the site's reads with ``plate``, ``well`` and ``site``.
    :param plate: the plate.
    :param well_id: the normalised well id.
    :param site: the site.
    :param chosen: from :func:`prefix_settings`; read for
        ``ops_quality_min_reads`` and ``ops_prefix_minority_min_reads``
        (item 967).
    :param method: the quality method.
    :param class_names: the method's quality classes.
    :param labelled: whether the site has cell labels.
    :param n_labelled: the labelled cells, or None.
    :param decoy_k: decoys per prefix.
    :param nearmiss_k: near-miss decoys per prefix.
    :param scale: the decoy FDR scale.
    :param shuffled: the shuffle-called cells with the shuffle calibration,
        else None.
    :param weighting: the item-988 read-weight arguments of
        :func:`prefix_cell_calls` under ``'weighted_majority'``; None or
        empty for none.
    :param sort: ``ops_call_sort`` (item 961); None keeps the decode's own rule.
    :returns: ``(row, cells)``.
    """
    from .ops_decode import fdr

    minority_k = int(chosen.get("ops_prefix_minority_min_reads") or 0)
    cells = _prefix_cell_calls(reads, min_reads=chosen["ops_quality_min_reads"],
                               method=method, sort=sort, minority_min_reads=minority_k,
                               **(weighting or {}))
    if decoy_k > 0:
        cells = _flag_decoy_cells(cells, nearmiss=nearmiss_k > 0)
    classes = cells["quality"].value_counts() if len(cells) else {}
    row = {"plate": plate, "well": well_id, "site": int(site),
           **read_fractions(reads), "labels": labelled,
           "cells_with_reads": int(len(cells)),
           "cells_with_exact_read": int(cells["has_exact_read"].sum())
           if len(cells) else 0,
           **{f"cells_{name}": int(classes.get(name, 0)) if len(cells) else 0
              for name in class_names}}
    if minority_k > 1:
        rescued = cells["minority_rescued"].astype(bool) if len(cells) else None
        if rescued is not None and "decoy" in cells.columns:
            rescued = rescued & ~cells["decoy"].astype(bool)
        row["cells_minority_rescued"] = int(rescued.sum()) if rescued is not None else 0
    if decoy_k > 0:
        if n_labelled is None or not labelled:
            n_labelled = int(len(cells))
        hit = cells[cells["quality"].isin(("perfect", "great"))] if len(cells) \
            else cells
        real = int((~hit["decoy"].astype(bool)).sum()) if len(hit) else 0
        far = (hit["decoy_class"].eq("far") if "decoy_class" in hit.columns
               else hit["decoy"].astype(bool)) if len(hit) else None
        decoy = int(far.sum()) if far is not None else 0
        row.update({"cells_in_labels": int(n_labelled), "cells_real": real,
                    "cells_decoy": decoy,
                    "fdr_pct": fdr(real, decoy, decoy_k, scale)})
        if nearmiss_k > 0:
            near = int(hit["decoy_class"].eq("nearmiss").sum()) if len(hit) else 0
            row.update({"cells_decoy_nearmiss": near,
                        "fdr_nearmiss_pct": fdr(real, near, nearmiss_k, 1.0)})
        if shuffled is not None:
            row.update({"shuffle_called": shuffled,
                        "shuffle_call_rate_pct": 100.0 * shuffled
                        / max(int(n_labelled), 1),
                        "fdr_scale": scale})
    return row, cells


def _prefix_read_mode(settings: Mapping[str, Any], chosen: Mapping[str, Any]) -> str:
    """The read mode of a spacer-prefix run (item 966).

    An empty ``ops_read_mode`` takes the spacer-prefix decode's own default:
    ``'auto'`` while ``ops_prefix_threshold_tuning`` is ``'per_well_decoy'``,
    since auto chooses the read from the tuning's decoy calls, and ``'log'``
    otherwise. The OPS barcode decode reads an empty value as ``'log'``
    (:func:`spacr.ops_sbs._read_mode_setting`), so the two decodes share the
    setting but not its default.

    :param settings: the run's settings.
    :param chosen: from :func:`prefix_settings`.
    :returns: ``'log'``, ``'raw'`` or ``'auto'``.
    :raises ValueError: for an unknown read mode.
    """
    from .ops_sbs import _read_mode_setting

    value = settings.get("ops_read_mode")
    if value is None or str(value).strip().lower() in ("", "none"):
        return "auto" if chosen["ops_prefix_threshold_tuning"] == "per_well_decoy" else "log"
    return _read_mode_setting(settings)


def _well_choice_line(plate: str, well_id: str, choice: Mapping[str, Any], well_mode: str,
                      read_mode: str) -> str:
    """One run-summary line for a tuned well (item 966).

    :param plate: the plate.
    :param well_id: the well.
    :param choice: :func:`spacr.ops_decode.tune_thresholds`'s result.
    :param well_mode: the read the well was decoded with.
    :param read_mode: the run's read mode (``'auto'`` chose ``well_mode``).
    :returns: the read mode and thresholds chosen and the decoy FDR of the
        tuning and the report half of the sites, as written to
        ``threshold_tuning_<plate>_<well_id>.json``.
    """
    best = choice["chosen"]
    tune, report = choice.get("tune") or {}, choice.get("report") or {}
    how = "chosen by auto" if read_mode == "auto" else "as set"
    tune_fdr, report_fdr = _fdr_text(tune.get("fdr_per_100")), _fdr_text(report.get("fdr_per_100"))
    return (f"prefix_decode {plate} {well_id} choice: read {well_mode} ({how}), read threshold "
            f"{best.get('ops_read_threshold')}, read quality "
            f"{best.get('ops_prefix_min_read_quality')}, mismatches "
            f"{best.get('ops_prefix_max_distance')}; decoy FDR {tune_fdr} % on the "
            f"{len(choice.get('tune_sites') or [])} tuning sites, {report_fdr} % on the "
            f"{len(choice.get('report_sites') or [])} report sites; target met "
            f"{choice.get('auto_met')}" + _weight_choice_text(best))


def _weight_choice_text(best: Mapping[str, Any]) -> str:
    """The weight settings a tuned weighted-majority well chose, for its summary line (item 988).

    :param best: the tuner's chosen setting.
    :returns: an empty string unless the setting has the weight axes.
    """
    if WEIGHT_GRID_KEYS[0] not in best:
        return ""
    length = float(best[WEIGHT_GRID_KEYS[0]])
    return (f"; weighted majority: decay length {length if length > 0 else 'off'}, "
            f"neighbour discount {best[WEIGHT_GRID_KEYS[1]]}, quality term "
            f"{'on' if best[WEIGHT_GRID_KEYS[2]] else 'off'}")


def _fdr_text(value: Any) -> str:
    """A decoy FDR in % to three decimals for a run-summary line, or 'n/a' (item 966).

    :param value: the FDR, or None.
    :returns: the text.
    """
    return "n/a" if value is None else f"{float(value):.3f}"


def _prefix_opportunities(prefixes, called, profiles):
    """Reduce selected real prefix winners to the shared distance-shell opportunities.

    :param prefixes: one production prefix per candidate cell.
    :param called: boolean mask selecting real called cells.
    :param profiles: aligned guide profiles from ``ops_decode._never_context``.
    :returns: twelve scalar opportunity totals from the selected winners.
    """
    from .ops_decode import _opportunity_index, _opportunity_totals

    index = _opportunity_index(tuple(profiles['guides']))
    real = [str(prefix) for prefix, keep in zip(prefixes, called) if keep]
    return _opportunity_totals(real, {name: values for name, values in profiles.items()
                                      if name != 'guides'}, index)


def _prefix_never_context(library, matching, chosen):
    """Wrap the shared independent never sets in the existing prefix whitelist type.

    :param library: the production prefix library with its cycle mask applied.
    :param matching: the production whitelist carrying far decoys.
    :param chosen: validated decode settings, including never-control options.
    :returns: the shared profiles and metadata, named control whitelists and
        independent match memos. Each control replaces the far decoys.
    """
    from .ops_decode import _never_context

    far = tuple(sorted(key for key, value in matching.callable_keys.items()
                       if value[1] == DECOY_GENE))
    context = _never_context(tuple(sorted(library.keys)), far,
                             chosen['ops_never_barcode_k'], chosen['ops_never_barcode_seed'],
                             chosen['ops_never_barcode_sets'])
    return {**context,
            'whitelists': {name: _with_prefix_decoys(library, data['barcodes'])
                           for name, data in context['controls'].items()},
            'memos': {}}


def _prefix_never_counts(reads, context, *, quality, distance, metric,
                         method, minority_k, plan, chosen, neighbours, sort=None):
    """Count production opportunities and control calls with the same prefix voters.

    :param reads: all reads at the selected detection threshold, before mismatch filtering.
    :param context: shared controls and prefix wrappers from ``_prefix_never_context``.
    :param quality: minimum read quality.
    :param distance: maximum correction distance.
    :param metric: existing prefix matching metric.
    :param method: existing cell vote method.
    :param minority_k: existing minority-read rule.
    :param plan: existing weighted-grid plan.
    :param chosen: validated prefix settings.
    :param neighbours: existing touching-cell pairs, or None.
    :param sort: optional production cell-call sort.
    :returns: production counts and the opportunity/control grids over any
        weight axes. Control calls never replace production calls.
    """
    base = _setting_counts(_at_setting(reads, quality, distance), method, minority_k,
                           plan, chosen, neighbours, opportunity_profiles=context['profiles'], sort=sort)
    out = dict(base['opportunities'])
    for name, library in context['whitelists'].items():
        control = _shuffle_frame(reads, library, context['memos'].setdefault((name, distance), {}),
                                  metric=metric, max_distance=distance)
        counts = np.asarray(_setting_counts(_at_setting(control, quality, distance),
                                            method, minority_k, plan, chosen, neighbours, sort=sort))
        out[f'never{name[-1]}_calls'] = counts[..., 1]
        out[f'real_calls_{name.lower()}'] = counts[..., 0]
    return {'counts': base['counts'], 'never': out}


def _shuffle_frame(reads: pd.DataFrame, shuffled_lib: PrefixLibrary, memo: Dict[str, tuple], *,
                   metric: str, max_distance: int) -> pd.DataFrame:
    """A site's reads matched against one shuffled whitelist instead of the library (item 949).

    :param reads: rows with ``y``, ``x``, ``cell``, ``quality`` and ``read``.
    :param shuffled_lib: from :func:`shuffled_library`.
    :param memo: the run's match memo for that whitelist.
    :param metric: the match metric.
    :param max_distance: the largest distance matched.
    :returns: the positions and qualities, and the nucleus distance when
        the reads carry one (item 988), with :func:`match_prefix_reads`'s
        columns against the shuffled whitelist (every key calls ``'control'``).
    """
    matched = _matched_rows(list(reads["read"]), shuffled_lib, memo, metric=metric,
                            max_distance=int(max_distance))
    kept = ["y", "x", "cell", "quality"] + [key for key in ("nuc_gap_px", "peak")
                                                    if key in reads.columns]
    return pd.concat([reads[kept].reset_index(drop=True), matched], axis=1)


def _calibrate_fixed(rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any], decoy_k: int,
                     settings: Mapping[str, Any], options: Mapping[str, Any], *,
                     never_context=None) -> Dict[str, Any]:
    """The selected FDR calibration of a well decoded at the entered thresholds (item 949).

    :param rows: the well's site summary rows (``site``, ``cells_in_labels``,
        ``cells_real``, ``cells_decoy``, ``shuffle_called``).
    :param chosen: from :func:`prefix_settings`.
    :param decoy_k: decoys per prefix.
    :param settings: read for ``ops_decode_auto_tune_fraction`` and
        ``ops_decode_auto_seed``.
    :param options: the shared tuning options (``ci`` and the objective).
    :param never_context: shared control metadata when rows carry never counts
        and opportunities; None for the existing shuffle method.
    :returns: :func:`spacr.ops_decode.calibrate_fdr_scale` on a one-setting
        grid, so the scale is set on the tuning half of the sites and the
        other half reports.
    """
    from .ops_decode import calibrate_fdr_scale, _never_payload

    counts = np.array([[r["cells_real"], r["cells_decoy"]] for r in rows],
                      np.int64).reshape(len(rows), 1, 1, 1, 2)
    shuffled = np.array([r["shuffle_called"] for r in rows], float).reshape(len(rows), 1, 1, 1)
    axes = [("ops_read_threshold", [chosen["ops_read_threshold"]]),
            ("ops_prefix_min_read_quality", [chosen["ops_prefix_min_read_quality"]]),
            ("ops_prefix_max_distance", [float(chosen["ops_prefix_max_distance"])])]
    never = None
    if never_context is not None:
        keys = ['never2_calls', 'opp2_N2', 'opp2_L', 'opp1_L', 'real_calls_n2']
        if chosen['ops_never_barcode_sets'] == 'n1_n2':
            keys += ['never1_calls', 'opp1_N1', 'real_calls_n1']
        never = {key: np.array([r[key] for r in rows], float).reshape(len(rows), 1, 1, 1)
                 for key in keys}
        never = _never_payload(never, never_context)
    return calibrate_fdr_scale(
        counts, shuffled, [r["cells_in_labels"] for r in rows], [r["site"] for r in rows],
        axes, k=decoy_k, scale=chosen["ops_prefix_fdr_scale"],
        fdr_max=chosen["ops_prefix_fdr_target_pct"],
        tune_fraction=float(settings.get("ops_decode_auto_tune_fraction") or 0.5),
        seed=int(settings.get("ops_decode_auto_seed") or 0),
        method=chosen["ops_prefix_fdr_calibration"], never=never, **options)


def _well_report_949(rows: Sequence[Mapping[str, Any]], *, decoy_k: int, scale: float,
                     nearmiss_k: int, entered_scale: float,
                     choice: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The item-949 keys of a well's FDR report: intervals, near-miss decoys, shuffle rate.

    :param rows: the well's site summary rows.
    :param decoy_k: far decoys per prefix.
    :param scale: the FDR scale used.
    :param nearmiss_k: near-miss decoys per prefix (0 for none).
    :param entered_scale: ``ops_prefix_fdr_scale``.
    :param choice: the calibration's result, or None without one.
    :returns: ``fdr_ci95`` (site bootstrap, :data:`spacr.ops_decode._AUTO_REPS`
        resamples, seed 1); with near-miss decoys ``assigned_decoy_nearmiss``,
        ``nearmiss_decoy_k``, ``fdr_nearmiss_per_100`` and its interval; with
        a calibration ``fdr_calibration``, ``fdr_scale_entered``,
        :func:`spacr.ops_decode.shuffle_call_rate`'s keys,
        ``shuffle_limit_pct`` (1.5 times the FDR), ``shuffle_le_limit`` and
        ``calibration``.
    """
    from .ops_decode import (SHUFFLE_FDR_RATIO, _AUTO_BOOT_SEED, _AUTO_REPS,
                             _bootstrap_rows, fdr, shuffle_call_rate)

    real = np.array([r["cells_real"] for r in rows], float)
    decoy = np.array([r["cells_decoy"] for r in rows], float)
    draws = _bootstrap_rows(len(rows), _AUTO_REPS, _AUTO_BOOT_SEED) if len(rows) > 1 else None
    weights = (np.stack([np.bincount(row, minlength=len(rows)) for row in draws])
               if draws is not None else None)

    def interval(numerator: np.ndarray, k: float, factor: float):
        """The site-bootstrap 95 % interval of a decoy FDR, or None for one site."""
        if weights is None:
            return None
        boot = 100.0 * max(float(factor), 1.0) / float(k) * (weights @ numerator) \
            / np.maximum(weights @ real, 1.0)
        return [round(float(np.percentile(boot, 2.5)), 4),
                round(float(np.percentile(boot, 97.5)), 4)]

    out: Dict[str, Any] = {"fdr_ci95": interval(decoy, decoy_k, scale)}
    if nearmiss_k > 0:
        near = np.array([r["cells_decoy_nearmiss"] for r in rows], float)
        out.update({"assigned_decoy_nearmiss": int(near.sum()),
                    "nearmiss_decoy_k": int(nearmiss_k),
                    "fdr_nearmiss_per_100": round(fdr(int(real.sum()), int(near.sum()),
                                                      nearmiss_k, 1.0), 4),
                    "fdr_nearmiss_ci95": interval(near, nearmiss_k, 1.0)})
    if choice is not None:
        rate = shuffle_call_rate([r["shuffle_called"] for r in rows],
                                 [r["cells_in_labels"] for r in rows])
        limit = SHUFFLE_FDR_RATIO * fdr(int(real.sum()), int(decoy.sum()), decoy_k, scale)
        out.update({"fdr_calibration": choice["calibration"]["method"], "fdr_scale_entered": float(entered_scale),
                    **rate, "shuffle_limit_pct": limit,
                    "shuffle_le_limit": bool(rate["shuffle_call_rate_pct"] <= limit),
                    "calibration": choice["calibration"]})
    return out


#: The item-942 keys of :func:`prefix_settings`, left out of
#: ``quality_classes.json`` while every one is off, so such a run writes
#: what it wrote before.
_ITEM_942_KEYS = ("ops_prefix_decoy_k", "ops_prefix_decoy_seed", "ops_prefix_fdr_scale",
                  "ops_prefix_min_read_quality", "ops_prefix_threshold_tuning",
                  "ops_prefix_fdr_target_pct", "ops_prefix_tuning_grid")

#: The item-949 keys of :func:`prefix_settings`, left out of
#: ``quality_classes.json`` while both are off, so such a run writes what it
#: wrote before.
_ITEM_949_KEYS = ("ops_prefix_nearmiss_decoy_k", "ops_prefix_fdr_calibration",
                  "ops_never_barcode_k", "ops_never_barcode_seed", "ops_never_barcode_sets")

#: The item-967 key of :func:`prefix_settings`, left out of
#: ``quality_classes.json`` while the minority-read rule is off, so such a
#: run writes what it wrote before.
_ITEM_967_KEYS = ("ops_prefix_minority_min_reads",)

#: The item-988 keys of :func:`prefix_settings`, left out of
#: ``quality_classes.json`` unless the weighted majority or the multi-guide
#: table is on, so such a run writes what it wrote before.
_ITEM_988_KEYS = ("ops_prefix_distance_weight", "ops_prefix_distance_decay_px",
                  "ops_prefix_distance_interior_px", "ops_prefix_distance_min_weight",
                  "ops_prefix_weighted_quality", "ops_prefix_weighted_min_share",
                  "ops_read_neighbour_discount", "ops_prefix_nuclei_masks",
                  "ops_keep_multi_guide_cells", "ops_multi_guide_min_reads")

#: The small-well guard's key (item 982), left out of
#: ``quality_classes.json`` while it is 0 (off).
_SMALL_WELL_KEYS = ("ops_prefix_tuning_min_sites",)


def _small_well_axes(axes: Sequence[Tuple[str, Sequence[float]]], n_sites: int,
                     min_sites: int) -> Tuple[List[Tuple[str, List[float]]], bool]:
    """The tuning axes of one well under the small-well guard (item 982).

    :param axes: from :func:`_tuning_axes` (read threshold, read-quality
        floor, mismatches and, under item 988, the weight axes).
    :param n_sites: the well's sites.
    :param min_sites: ``ops_prefix_tuning_min_sites``; 0 is off.
    :returns: ``(axes, guarded)``: with fewer than ``min_sites`` sites the
        read-quality floor axis holds :data:`SMALL_WELL_FLOOR` only and
        ``guarded`` is True; otherwise ``axes`` unchanged and False.
    """
    if min_sites <= 0 or n_sites >= min_sites:
        return [(name, list(values)) for name, values in axes], False
    return [(name, [SMALL_WELL_FLOOR] if name == "ops_prefix_min_read_quality"
             else list(values)) for name, values in axes], True


#: The cell rules of the matched-FDR sweep (item 955), from least to most
#: strict: ``'maxcount'`` keeps a cell whose top barcode has more matched
#: reads than any other, ``'purity50'`` also needs half of the cell's matched
#: reads on it, and ``'purity100'`` needs every matched read on one barcode.
FDR_SWEEP_RULES = ("maxcount", "purity50", "purity100")

_SWEEP_DEFAULTS: Dict[str, Any] = {
    "ops_fdr_sweep_targets": [0.5, 1.0, 2.0, 5.0],
    "ops_fdr_sweep_max_distance": [0, 1],
    "ops_fdr_sweep_min_reads": [1, 2, 3],
    "ops_fdr_sweep_rules": list(FDR_SWEEP_RULES),
    "ops_fdr_sweep_quality_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "ops_fdr_sweep_bootstrap": 2000,
    "ops_fdr_sweep_seed": 0,
    "ops_fdr_sweep_shuffle_seeds": [0, 1, 2],
}

_SWEEP_KEY = ["method", "max_d", "min_reads", "rule", "qmin"]


def _sweep_list(value: Any, kind: Callable[[Any], Any]) -> List[Any]:
    """A list setting of the sweep, from a list or a comma-separated string.

    :param value: the setting's value.
    :param kind: the type of each element.
    :returns: the elements, in the order given.
    """
    if isinstance(value, str):
        value = [part for part in value.replace(";", ",").split(",") if part.strip()]
    return [kind(part.strip()) if isinstance(part, str) else kind(part) for part in value]


def _sweep_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The item-955 sweep settings, checked and with defaults filled in.

    :param settings: the OPS settings.
    :returns: the normalised ``ops_fdr_sweep_*`` values and the decoy and
        match settings the sweep uses.
    :raises ValueError: for an empty or invalid grid, an unknown rule, a
        quantile outside (0, 1), fewer than one bootstrap replicate or no
        decoys.
    """
    value = {key: (settings.get(key) if settings.get(key) not in (None, "", [])
                   else default) for key, default in _SWEEP_DEFAULTS.items()}
    out = {
        "ops_fdr_sweep_targets": _sweep_list(value["ops_fdr_sweep_targets"], float),
        "ops_fdr_sweep_max_distance": _sweep_list(value["ops_fdr_sweep_max_distance"], int),
        "ops_fdr_sweep_min_reads": _sweep_list(value["ops_fdr_sweep_min_reads"], int),
        "ops_fdr_sweep_rules": [str(rule).strip().lower() for rule in
                                _sweep_list(value["ops_fdr_sweep_rules"], str)],
        "ops_fdr_sweep_quality_quantiles": _sweep_list(
            value["ops_fdr_sweep_quality_quantiles"], float),
        "ops_fdr_sweep_bootstrap": int(value["ops_fdr_sweep_bootstrap"]),
        "ops_fdr_sweep_seed": int(value["ops_fdr_sweep_seed"]),
        "ops_fdr_sweep_shuffle_seeds": _sweep_list(value["ops_fdr_sweep_shuffle_seeds"], int),
        "ops_prefix_decoy_k": int(settings.get("ops_prefix_decoy_k") or 0),
        "ops_prefix_decoy_seed": int(settings.get("ops_prefix_decoy_seed", 1) or 0),
        "ops_prefix_metric": str(settings.get("ops_prefix_metric") or "hamming"),
    }
    if not all(out[key] for key in ("ops_fdr_sweep_targets", "ops_fdr_sweep_max_distance",
                                    "ops_fdr_sweep_min_reads", "ops_fdr_sweep_rules")):
        raise ValueError("the FDR sweep needs at least one target, max distance, "
                         "minimum read count and rule")
    if any(not target > 0 for target in out["ops_fdr_sweep_targets"]):
        raise ValueError("ops_fdr_sweep_targets must be above 0 (in %)")
    if any(d < 0 for d in out["ops_fdr_sweep_max_distance"]):
        raise ValueError("ops_fdr_sweep_max_distance must be 0 or more")
    if any(n < 1 for n in out["ops_fdr_sweep_min_reads"]):
        raise ValueError("ops_fdr_sweep_min_reads must be 1 or more")
    unknown = [rule for rule in out["ops_fdr_sweep_rules"] if rule not in FDR_SWEEP_RULES]
    if unknown:
        raise ValueError(f"unknown ops_fdr_sweep_rules {unknown}; choose from "
                         f"{list(FDR_SWEEP_RULES)}")
    if any(not 0 < q < 1 for q in out["ops_fdr_sweep_quality_quantiles"]):
        raise ValueError("ops_fdr_sweep_quality_quantiles must lie between 0 and 1")
    if out["ops_fdr_sweep_bootstrap"] < 1:
        raise ValueError("ops_fdr_sweep_bootstrap must be 1 or more")
    if out["ops_prefix_decoy_k"] < 1:
        raise ValueError("the FDR sweep counts decoy calls; set ops_prefix_decoy_k to 1 or "
                         "more (4 recommended)")
    return out


def _sweep_matched(frame: pd.DataFrame, library: PrefixLibrary, *, metric: str,
                   max_distance: int, decoy: bool) -> pd.DataFrame:
    """One method's reads matched against one whitelist, matched reads only.

    Each distinct read is matched once.

    :param frame: ``site``, ``cell``, ``read`` and ``quality`` per read.
    :param library: the whitelist (with decoys, or a shuffled one).
    :param metric: the match metric.
    :param max_distance: the largest distance matched.
    :param decoy: mark reads whose key is a decoy (gene :data:`DECOY_GENE`);
        False marks none.
    :returns: ``site``, ``cell``, ``prefix``, ``distance``, ``quality`` and
        ``decoy`` of the exact and corrected reads.
    """
    reads = frame["read"].astype(str).to_numpy()
    unique, inverse = np.unique(reads, return_inverse=True)
    matched = match_prefix_reads(list(unique), library, metric=metric,
                                 max_distance=int(max_distance))
    status = matched["status"].to_numpy()[inverse]
    keep = (status == "exact") | (status == "corrected")
    out = pd.DataFrame({
        "site": frame["site"].to_numpy()[keep],
        "cell": frame["cell"].to_numpy()[keep],
        "prefix": matched["prefix"].to_numpy()[inverse][keep],
        "distance": matched["distance"].to_numpy()[inverse][keep],
        "quality": frame["quality"].to_numpy(float)[keep],
        "decoy": (matched["gene"].to_numpy()[inverse][keep] == DECOY_GENE) if decoy
        else np.zeros(int(keep.sum()), bool)})
    return out


def _sweep_tops(matched: pd.DataFrame, max_d: int, qmin: float) -> pd.DataFrame:
    """Each cell's top barcode at one read filter.

    :param matched: from :func:`_sweep_matched`.
    :param max_d: the largest read distance kept.
    :param qmin: the lowest read quality kept; minus infinity keeps all.
    :returns: per cell ``site``, ``n`` (reads of the top barcode, ties
        broken by the alphabetically first barcode), ``second`` (reads of the
        runner-up, 0 without), ``barcodes``, ``total`` (matched reads) and
        ``decoy`` (the top barcode is a decoy).
    """
    keep = matched["distance"].to_numpy() <= max_d
    if qmin > -np.inf:
        keep &= matched["quality"].to_numpy() >= qmin
    reads = matched[keep]
    if reads.empty:
        return pd.DataFrame(columns=["site", "cell", "n", "second", "barcodes", "total", "decoy"])
    counts = (reads.groupby(["site", "cell", "prefix"], sort=False)
              .agg(n=("decoy", "size"), decoy=("decoy", "first")).reset_index())
    counts = counts.sort_values(["site", "cell", "n", "prefix"],
                                ascending=[True, True, False, True], kind="mergesort")
    groups = counts.groupby(["site", "cell"], sort=False)
    counts["barcodes"] = groups["n"].transform("size")
    counts["total"] = groups["n"].transform("sum")
    counts["second"] = groups["n"].shift(-1).fillna(0)
    return groups.head(1)[["site", "cell", "n", "second", "barcodes", "total",
                           "decoy"]].reset_index(drop=True)


def _sweep_called(tops: pd.DataFrame, min_reads: int, rule: str) -> pd.Series:
    """Which cells one configuration calls.

    :param tops: from :func:`_sweep_tops`.
    :param min_reads: the top barcode's fewest reads.
    :param rule: one of :data:`FDR_SWEEP_RULES`.
    :returns: a boolean per row of ``tops``.
    """
    keep = tops["n"] >= min_reads
    if rule == "purity100":
        return keep & (tops["barcodes"] == 1)
    keep &= tops["n"] > tops["second"]
    if rule == "purity50":
        keep &= 2 * tops["n"] >= tops["total"]
    return keep


def _sweep_site_counts(tops: pd.DataFrame, called: pd.Series,
                       sites: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Real and decoy calls per site.

    :param tops: from :func:`_sweep_tops`.
    :param called: from :func:`_sweep_called`.
    :param sites: the sites, in output order.
    :returns: ``(real, decoy)`` arrays over ``sites``.
    """
    hit = tops[called.to_numpy(bool)]
    decoy = hit["decoy"].to_numpy(bool)
    real = pd.Series(~decoy).groupby(hit["site"].to_numpy()).sum()
    fake = pd.Series(decoy).groupby(hit["site"].to_numpy()).sum()
    return (real.reindex(list(sites), fill_value=0).to_numpy(np.int64),
            fake.reindex(list(sites), fill_value=0).to_numpy(np.int64))


def _sweep_strictness(row: Mapping[str, Any]) -> Tuple[float, ...]:
    """A configuration's sort key among equals: stricter first.

    :param row: a pooled grid row.
    :returns: fewer mismatches, more reads, a stricter rule, a higher
        quality cut, in that order, as ascending sort keys.
    """
    return (float(row["max_d"]), -float(row["min_reads"]),
            -float(FDR_SWEEP_RULES.index(row["rule"])), -float(row["qmin"]))


def _sweep_frames(reads: Any) -> Dict[str, pd.DataFrame]:
    """The reads per method, with the columns the sweep reads.

    :param reads: ``{method: frame}`` or one frame with a ``method`` column;
        each frame has ``site``, ``cell`` and ``read`` and may have
        ``quality``.
    :returns: ``{method: frame}`` in the order given, cells above 0 only,
        ``quality`` in its stored precision (its quantiles are taken in it),
        filled with infinity where a method has none.
    :raises ValueError: without reads or when a column is missing.
    """
    if isinstance(reads, pd.DataFrame):
        if "method" in reads.columns:
            reads = {str(name): group for name, group in reads.groupby("method", sort=False)}
        else:
            reads = {"spacr": reads}
    if not reads:
        raise ValueError("the FDR sweep needs reads of at least one method")
    out = {}
    for name, frame in reads.items():
        missing = [col for col in ("site", "cell", "read") if col not in frame.columns]
        if missing:
            raise ValueError(f"the reads of method {name!r} lack columns {missing}")
        quality = (pd.to_numeric(frame["quality"], errors="coerce").to_numpy()
                   if "quality" in frame.columns else np.full(len(frame), np.inf))
        table = pd.DataFrame({"site": pd.to_numeric(frame["site"]).to_numpy(np.int64),
                              "cell": pd.to_numeric(frame["cell"]).to_numpy(np.int64),
                              "read": frame["read"].astype(str).to_numpy(),
                              "quality": quality})
        out[str(name)] = table[table["cell"] > 0].reset_index(drop=True)
    return out


def _sweep_cells(cells: Any) -> Dict[int, int]:
    """Cells per site, the genotyped share's denominator.

    :param cells: ``{site: cells}``, a :class:`pandas.Series` indexed by site,
        or a frame with ``site`` and ``cells``.
    :returns: ``{site: cells}``.
    """
    if isinstance(cells, pd.DataFrame):
        cells = dict(zip(cells["site"], cells["cells"]))
    elif isinstance(cells, pd.Series):
        cells = cells.to_dict()
    return {int(site): int(count) for site, count in dict(cells).items()}


class _SweepDraws:
    """The sweep's site-bootstrap resamples, drawn in blocks on first use.

    :param seed: the generator seed (``ops_fdr_sweep_seed``).
    :param reps: resamples per block (``ops_fdr_sweep_bootstrap``).
    :param n: report sites.
    """

    def __init__(self, seed: int, reps: int, n: int) -> None:
        """Start the generator; no block is drawn yet."""
        self.rng = np.random.default_rng(int(seed))
        self.reps = int(reps)
        self.n = int(n)
        self.blocks: List[np.ndarray] = []

    def block(self, i: int) -> np.ndarray:
        """The i-th block, drawing the blocks before it first.

        :param i: the block, from 0.
        :returns: ``(reps, n)`` site indices.
        """
        while len(self.blocks) <= i:
            self.blocks.append(self.rng.integers(0, self.n, (self.reps, self.n)))
        return self.blocks[i]


def _sweep_interval(values: np.ndarray) -> List[float]:
    """The 2.5th and 97.5th percentiles of bootstrap values.

    :param values: one value per resample.
    :returns: ``[low, high]``.
    """
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def fdr_sweep(reads: Any, cells: Any, library: PrefixLibrary,
              settings: Mapping[str, Any], *,
              selection_sites: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Genotyped share at matched decoy FDR, method by method (item 955).

    Every method's reads (spaCR's own, or another pipeline's per-spot reads)
    are matched by one matcher (:func:`match_prefix_reads`, ``hamming``
    unless ``ops_prefix_metric`` says otherwise, up to the largest
    ``ops_fdr_sweep_max_distance``) against the whitelist with decoys
    (:func:`prefix_decoy_library`, ``ops_prefix_decoy_k`` and
    ``ops_prefix_decoy_seed``) and, for the control, against shuffled
    whitelists (:func:`shuffled_library`, one per
    ``ops_fdr_sweep_shuffle_seeds``). A configuration is a largest read
    distance, a fewest top-barcode reads, a cell rule
    (:data:`FDR_SWEEP_RULES`) and, for a method whose reads carry a quality,
    a lowest read quality: no cut or the ``ops_fdr_sweep_quality_quantiles``
    of that method's reads in cells. Each site's calls are counted at every
    configuration; pooled over the selection sites, the decoy FDR is
    ``100 / k * decoys / real calls`` and the genotyped share
    ``100 * real calls / cells``.

    For each ``ops_fdr_sweep_targets`` value, and for the lowest FDR every
    method reaches (rounded up to 0.0001 %), each method's configuration is
    the one with the most real calls at a pooled FDR no higher than the
    target; among equals the lower FDR, then the stricter configuration
    (fewer mismatches, more reads, stricter rule, higher quality cut) is
    kept. Each chosen configuration is then scored on the report sites: its
    genotyped share and decoy FDR with site-bootstrap 95 % intervals, and its
    barcode-shuffle call rate (shuffle-called cells per 100 cells, the mean
    over the shuffled whitelists, :func:`spacr.ops_decode.shuffle_call_rate`).
    Between methods the paired difference in genotyped share at the same
    target gets a site-bootstrap 95 % interval. The bootstrap draws site
    indices from ``numpy.random.default_rng(ops_fdr_sweep_seed)``,
    ``ops_fdr_sweep_bootstrap`` resamples at a time: the first draw serves
    every share and FDR interval and the first target's differences, each
    later target with two or more chosen methods takes the next draw.

    :param reads: ``{method: frame}``, or one frame with a ``method``
        column; each frame has ``site``, ``cell`` (0 or less is outside
        every cell and dropped) and ``read`` (the called bases from cycle 1),
        and may have ``quality`` (a higher value is a better read).
    :param cells: cells per site, as ``{site: cells}``, a series indexed by
        site, or a frame with ``site`` and ``cells``; the same cells serve
        every method.
    :param library: the whitelist, from :func:`spacer_prefix_library`,
        without decoys.
    :param settings: the ``ops_fdr_sweep_*`` settings, ``ops_prefix_decoy_k``
        (1 or more), ``ops_prefix_decoy_seed`` and ``ops_prefix_metric``.
    :param selection_sites: the sites the configurations are chosen on; the
        others are the report sites. None chooses and reports on every site.
    :returns: ``settings`` (the values used), ``sites``,
        ``selection_sites``, ``report_sites``, ``decoys``,
        ``lowest_fdr_pct`` and ``quality_cuts`` per method, ``targets`` (per
        target ``label``, ``target_fdr_pct``, per method the chosen
        configuration with its selection and report numbers or None, and
        ``diffs``, the paired differences ``'<a> - <b>'``), ``table`` (one
        row per target and method), ``pooled`` (every configuration on the
        selection sites) and ``grid`` (every configuration per site) as
        frames.
    :raises ValueError: for invalid settings, missing columns, a site
        without a cell count, or selection sites that leave no report site.
    """
    from .ops_decode import shuffle_call_rate

    chosen = _sweep_settings(settings)
    frames = _sweep_frames(reads)
    counts = _sweep_cells(cells)
    sites = sorted({int(s) for frame in frames.values() for s in frame["site"].unique()}
                   | set(counts))
    lacking = [s for s in sites if s not in counts]
    if lacking:
        raise ValueError(f"no cell count for sites {lacking[:10]}")
    if selection_sites is None:
        select, report = list(sites), list(sites)
    else:
        select = sorted({int(s) for s in selection_sites} & set(sites))
        report = [s for s in sites if s not in set(select)]
        if not select or not report:
            raise ValueError("selection_sites must hold some sites and leave others to report")
    k = chosen["ops_prefix_decoy_k"]
    metric = chosen["ops_prefix_metric"]
    reach = max(chosen["ops_fdr_sweep_max_distance"])
    joint = prefix_decoy_library(library, k, chosen["ops_prefix_decoy_seed"])
    cell_array = np.array([counts[s] for s in sites], np.int64)
    position = {s: i for i, s in enumerate(sites)}
    sel_idx = np.array([position[s] for s in select], np.int64)
    rep_idx = np.array([position[s] for s in report], np.int64)
    matched, cuts, rows = {}, {}, []
    per_site: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
    for name, frame in frames.items():
        matched[name] = _sweep_matched(frame, joint, metric=metric, max_distance=reach, decoy=True)
        quality = frame.loc[frame["site"].isin(select), "quality"].to_numpy()
        finite = quality[np.isfinite(quality)]
        cuts[name] = sorted({-np.inf} | {float(np.quantile(finite, q)) for q in
                                         chosen["ops_fdr_sweep_quality_quantiles"]}
                            ) if finite.size else [-np.inf]
        for max_d in chosen["ops_fdr_sweep_max_distance"]:
            for qi, qmin in enumerate(cuts[name]):
                tops = _sweep_tops(matched[name], max_d, qmin)
                for min_reads in chosen["ops_fdr_sweep_min_reads"]:
                    for rule in chosen["ops_fdr_sweep_rules"]:
                        real, decoy = _sweep_site_counts(
                            tops, _sweep_called(tops, min_reads, rule), sites)
                        per_site[(name, max_d, min_reads, rule, qmin)] = (real, decoy)
                        for i, site in enumerate(sites):
                            rows.append({"method": name, "max_d": max_d,
                                         "min_reads": min_reads, "rule": rule,
                                         "qmin": qmin, "quality_index": qi, "site": site,
                                         "real": int(real[i]), "decoy": int(decoy[i]),
                                         "cells": int(cell_array[i])})
    grid = pd.DataFrame(rows)
    pooled_rows = []
    for config, (real, decoy) in per_site.items():
        r, d, c = int(real[sel_idx].sum()), int(decoy[sel_idx].sum()), int(cell_array[sel_idx].sum())
        pooled_rows.append({**dict(zip(_SWEEP_KEY, config)),
                            "quality_index": cuts[config[0]].index(config[4]),
                            "real": r, "decoy": d, "cells": c,
                            "share": 100 * r / c if c else 0.0,
                            "fdr": 100 / k * d / max(r, 1)})
    pooled = pd.DataFrame(pooled_rows)
    lowest = {name: float(group.loc[group["real"] > 0, "fdr"].min())
              for name, group in pooled.groupby("method", sort=False)
              if (group["real"] > 0).any()}
    targets = [(f"<= {t:g} %", float(t)) for t in chosen["ops_fdr_sweep_targets"]]
    if lowest:
        common = math.ceil(max(lowest.values()) * 1e4 - 1e-9) / 1e4
        targets.append(("lowest common", common))
    draws_of = _SweepDraws(chosen["ops_fdr_sweep_seed"], chosen["ops_fdr_sweep_bootstrap"],
                           len(rep_idx))
    shuffles: Dict[Tuple[str, int], pd.DataFrame] = {}
    shuffle_libs = {seed: shuffled_library(library, seed)
                    for seed in chosen["ops_fdr_sweep_shuffle_seeds"]}
    rep_cells = cell_array[rep_idx]
    entries, table, compared = [], [], 0
    for label, target in targets:
        entry: Dict[str, Any] = {"label": label, "target_fdr_pct": target, "methods": {},
                                 "diffs": {}}
        picks = {}
        for name in frames:
            ok = pooled[(pooled["method"] == name) & (pooled["fdr"] <= target)]
            if ok.empty:
                entry["methods"][name] = None
                continue
            best = min(ok.to_dict("records"),
                       key=lambda row: (-row["real"] / max(row["cells"], 1), row["fdr"],
                                        _sweep_strictness(row)))
            config = tuple(best[key] for key in _SWEEP_KEY)
            real, decoy = per_site[config]
            r_rep, d_rep = real[rep_idx], decoy[rep_idx]
            draws = draws_of.block(0)
            boot_r, boot_d = r_rep[draws].sum(1), d_rep[draws].sum(1)
            boot_c = rep_cells[draws].sum(1)
            shuffled = []
            for seed, shuffled_lib in shuffle_libs.items():
                if (name, seed) not in shuffles:
                    shuffles[(name, seed)] = _sweep_matched(frames[name], shuffled_lib,
                                                            metric=metric, max_distance=reach,
                                                            decoy=False)
                tops = _sweep_tops(shuffles[(name, seed)], int(best["max_d"]), best["qmin"])
                called, _ = _sweep_site_counts(
                    tops, _sweep_called(tops, int(best["min_reads"]), best["rule"]), sites)
                shuffled.append(called[rep_idx].astype(float))
            rate = (shuffle_call_rate(np.mean(shuffled, axis=0), rep_cells)
                    if shuffled else None)
            per_seed = [100.0 * float(s.sum()) / max(float(rep_cells.sum()), 1.0)
                        for s in shuffled]
            pick = {
                "max_d": int(best["max_d"]), "min_reads": int(best["min_reads"]),
                "rule": best["rule"], "qmin": float(best["qmin"]),
                "quality_index": int(best["quality_index"]),
                "selection": {"real": int(best["real"]), "decoy": int(best["decoy"]),
                              "cells": int(best["cells"]), "share": float(best["share"]),
                              "fdr": float(best["fdr"])},
                "real": int(r_rep.sum()), "decoy": int(d_rep.sum()),
                "cells": int(rep_cells.sum()),
                "share": 100.0 * float(r_rep.sum()) / max(float(rep_cells.sum()), 1.0),
                "fdr": 100.0 / k * float(d_rep.sum()) / max(float(r_rep.sum()), 1.0),
                "share_ci95": _sweep_interval(100.0 * boot_r / np.maximum(boot_c, 1)),
                "fdr_ci95": _sweep_interval(100.0 / k * boot_d / np.maximum(boot_r, 1)),
                "shuffle_call_rate_pct": None if rate is None else rate["shuffle_call_rate_pct"],
                "shuffle_ci95": None if rate is None else rate["shuffle_ci95"],
                "shuffle_per_seed_pct": dict(zip(shuffle_libs, per_seed)),
            }
            entry["methods"][name] = pick
            picks[name] = (r_rep, pick)
        if len(picks) > 1:
            draws = draws_of.block(compared)
            compared += 1
            names = list(picks)
            share = {name: 100 * picks[name][0][draws].sum(1) / rep_cells[draws].sum(1)
                     for name in names}
            for a, b in itertools.combinations(names, 2):
                entry["diffs"][f"{a} - {b}"] = {
                    "diff_share_pts": picks[a][1]["share"] - picks[b][1]["share"],
                    "diff_ci95": _sweep_interval(share[a] - share[b])}
        entries.append(entry)
        first = next(iter(picks), None)
        for name, pick in entry["methods"].items():
            if pick is None:
                continue
            diff = entry["diffs"].get(f"{first} - {name}") if name != first else None
            table.append({
                "target": label, "target_fdr_pct": target, "method": name,
                "max_d": pick["max_d"], "min_reads": pick["min_reads"], "rule": pick["rule"],
                "qmin": pick["qmin"], "quality_index": pick["quality_index"],
                "genotyped_pct": pick["share"], "genotyped_ci95_lo": pick["share_ci95"][0],
                "genotyped_ci95_hi": pick["share_ci95"][1], "decoy_fdr_pct": pick["fdr"],
                "decoy_fdr_ci95_lo": pick["fdr_ci95"][0], "decoy_fdr_ci95_hi": pick["fdr_ci95"][1],
                "real": pick["real"], "decoy": pick["decoy"], "cells": pick["cells"],
                "shuffle_call_rate_pct": pick["shuffle_call_rate_pct"],
                "shuffle_ci95_lo": None if pick["shuffle_ci95"] is None else pick["shuffle_ci95"][0],
                "shuffle_ci95_hi": None if pick["shuffle_ci95"] is None else pick["shuffle_ci95"][1],
                **{f"shuffle_seed{seed}_pct": value
                   for seed, value in pick["shuffle_per_seed_pct"].items()},
                "selection_genotyped_pct": pick["selection"]["share"],
                "selection_decoy_fdr_pct": pick["selection"]["fdr"],
                "paired_diff_vs": None if diff is None else first,
                "paired_diff_pts": None if diff is None else -diff["diff_share_pts"],
                "paired_diff_ci95_lo": None if diff is None else -diff["diff_ci95"][1],
                "paired_diff_ci95_hi": None if diff is None else -diff["diff_ci95"][0]})
    return {"settings": chosen, "sites": sites, "selection_sites": select,
            "report_sites": report, "decoys": int(joint.summary.get("decoys", 0)),
            "lowest_fdr_pct": lowest, "quality_cuts": cuts, "targets": entries,
            "table": pd.DataFrame(table), "pooled": pooled, "grid": grid}


_SWEEP_COLUMNS = {"site": ("site",), "cell": ("cell", "Parent_Cells"),
                  "read": ("read", "Barcode_BarcodeCalled"), "quality": ("quality",)}


def _sweep_inputs(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """``ops_fdr_sweep_inputs`` as a dictionary.

    :param settings: the OPS settings.
    :returns: the inputs: empty, a dictionary as given, a JSON string read,
        or a ``.json`` file's contents.
    :raises ValueError: when the value is none of those.
    """
    value = settings.get("ops_fdr_sweep_inputs")
    if value in (None, "", {}):
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("{") and os.path.isfile(os.path.expanduser(text)):
            with open(os.path.expanduser(text), encoding="utf-8") as handle:
                text = handle.read()
        try:
            value = json.loads(text)
        except ValueError as error:
            raise ValueError("ops_fdr_sweep_inputs must be a dictionary, JSON text or the "
                             "path of a .json file") from error
    if not isinstance(value, Mapping):
        raise ValueError("ops_fdr_sweep_inputs must be a dictionary")
    return dict(value)


def _sweep_read_files(spec: Any, sites: Optional[Sequence[int]]) -> pd.DataFrame:
    """One method's reads, from the files its input names.

    :param spec: a path, a glob, or a path with ``{site}``; or a dictionary
        with ``reads`` (that path) and optional ``site_col``, ``cell_col``,
        ``read_col`` and ``quality_col``. Without a column name the usual
        names are tried (``site``; ``cell`` or ``Parent_Cells``; ``read`` or
        ``Barcode_BarcodeCalled``; ``quality``, optional).
    :param sites: the sites to keep, and the sites a ``{site}`` path is
        filled with; None keeps every site.
    :returns: ``site``, ``cell``, ``read`` and, when present, ``quality``.
    :raises ValueError: when no file matches, a ``{site}`` path has no
        sites, or a column is missing.
    """
    import glob

    from .tabular import read_table

    spec = {"reads": spec} if not isinstance(spec, Mapping) else dict(spec)
    pattern = os.path.expanduser(str(spec.get("reads") or ""))
    if "{site}" in pattern:
        if not sites:
            raise ValueError(f"the read path {pattern!r} names {{site}}; give the sites")
        paths = [(pattern.format(site=s), int(s)) for s in sites]
        paths = [(p, s) for p, s in paths if os.path.isfile(p)]
    else:
        paths = [(p, None) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise ValueError(f"no read file matches {pattern!r}")
    frames = []
    for path, site in paths:
        frame = read_table(path, canonicalise=False, report=None)
        out = {}
        for name, options in _SWEEP_COLUMNS.items():
            given = spec.get(f"{name}_col")
            found = [given] if given else [col for col in options if col in frame.columns]
            if found and found[0] in frame.columns:
                out[name] = frame[found[0]].to_numpy()
            elif name == "site" and site is not None:
                out[name] = np.full(len(frame), site, np.int64)
            elif name != "quality":
                raise ValueError(f"{path} has no {name} column (tried {found or list(options)})")
        frames.append(pd.DataFrame(out))
    reads = pd.concat(frames, ignore_index=True)
    if sites:
        reads = reads[reads["site"].isin([int(s) for s in sites])]
    return reads.reset_index(drop=True)


def _run_fdr_sweep_phase(settings: Mapping[str, Any],
                         report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The ``prefix_fdr_sweep`` phase: :func:`fdr_sweep` over a run's reads (item 955).

    ``ops_fdr_sweep_inputs`` names the reads: ``methods`` maps each method
    to its read files (see :func:`_sweep_read_files`); without it the
    method ``'spacr'`` reads ``<dst_root>/results/ops_prefix/reads_*.parquet``
    (the ``prefix_decode`` phase's reads). ``cells`` is a table with
    ``site`` and a cell count (``cells``, ``Count_Cells`` or ``count_col``);
    without it a site's cells are the cells with any read of any method.
    ``sites`` keeps some sites; ``selection_sites`` chooses the
    configurations on those and reports on the rest. Writes, under
    ``<dst_root>/results/ops_prefix/``, ``fdr_sweep_result.json`` (the
    result without its frames), ``fdr_sweep_table.csv`` (one row per target
    and method), ``fdr_sweep_pooled.csv`` and ``fdr_sweep_grid.csv``.

    :param settings: the OPS settings: ``dst_root``, the item-920 library
        settings (:func:`prefix_settings`), the ``ops_fdr_sweep_*`` settings
        and ``ops_prefix_decoy_k`` and ``ops_prefix_decoy_seed``.
    :param report: called with progress lines.
    :returns: the output paths, the cells' source and the result summary.
    :raises ValueError: without ``dst_root``, a library or reads.
    """
    from .tabular import read_table, write_table

    say = report or (lambda text: None)
    destination = settings.get("dst_root")
    if not destination:
        raise ValueError("the prefix_fdr_sweep phase needs dst_root")
    started = time.perf_counter()
    chosen = prefix_settings(settings)
    _sweep_settings(settings)
    library = spacer_prefix_library(
        chosen["ops_library"], sequence_col=chosen["ops_library_sequence_col"],
        gene_col=chosen["ops_library_gene_col"], prefix_length=chosen["ops_prefix_length"],
        n_cycles=chosen["ops_sbs_cycles"], cycle_mask=chosen["ops_cycle_mask"],
        collision=chosen["ops_prefix_collision"])
    inputs = _sweep_inputs(settings)
    folder = os.path.join(str(destination), "results", "ops_prefix")
    sites = [int(s) for s in inputs.get("sites") or []] or None
    methods = inputs.get("methods") or {
        "spacr": os.path.join(folder, "reads_*.parquet")}
    reads = {}
    for name, spec in methods.items():
        reads[str(name)] = _sweep_read_files(spec, sites)
        say(f"prefix_fdr_sweep: {name} {len(reads[str(name)])} reads")
    cell_spec = inputs.get("cells")
    if cell_spec:
        cell_spec = {"path": cell_spec} if not isinstance(cell_spec, Mapping) else dict(cell_spec)
        table = read_table(os.path.expanduser(str(cell_spec["path"])), canonicalise=False,
                           report=None)
        count_col = cell_spec.get("count_col") or next(
            col for col in ("cells", "Count_Cells") if col in table.columns)
        site_col = cell_spec.get("site_col") or "site"
        cells = {int(s): int(n) for s, n in zip(table[site_col], table[count_col])}
        source = f"{cell_spec['path']}:{count_col}"
    else:
        union = pd.concat([frame.loc[pd.to_numeric(frame["cell"]) > 0, ["site", "cell"]]
                           for frame in reads.values()]).drop_duplicates()
        cells = union.groupby("site").size().to_dict()
        source = "cells with any read"
    if sites:
        cells = {s: n for s, n in cells.items() if s in set(sites)}
    present = {int(s) for frame in reads.values() for s in frame["site"].unique()}
    cells = {s: n for s, n in cells.items() if s in present or sites}
    result = fdr_sweep(reads, cells, library, settings,
                       selection_sites=inputs.get("selection_sites"))
    os.makedirs(folder, exist_ok=True)
    paths = {name: os.path.join(folder, f"fdr_sweep_{name}.csv")
             for name in ("table", "pooled", "grid")}
    for name, path in paths.items():
        write_table(result[name], path, canonicalise=False)
    summary = {key: value for key, value in result.items() if key not in paths}
    summary.update({"cells_source": source, "methods": list(reads),
                    "seconds": round(time.perf_counter() - started, 1)})
    paths["result"] = os.path.join(folder, "fdr_sweep_result.json")
    with open(paths["result"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=float)
    say(f"prefix_fdr_sweep: {len(result['targets'])} targets, {len(result['sites'])} sites "
        f"-> {paths['result']}")
    return {"paths": paths, **summary}
