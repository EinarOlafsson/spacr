"""QC for a barcode-mapping run, and a target-driven abundance-threshold sweep.

This module is the analysis that happens **after**
:func:`spacr.sequencing.generate_barecode_mapping` has written its
``unique_combinations.csv`` / ``qc.csv``. It answers the two questions a
pooled-screen experimenter actually asks of a mapping run:

1. *Did the run work?* — reads per well and which wells are starved,
   how many reads never mapped, whether the barcode references are
   distinguishable at all, whether a plate row or column is systematically
   under-read, and how evenly the library is covered.
2. *Where do I cut?* — a gRNA is kept in a well when its share of that
   well's reads reaches an abundance threshold. Too low and a well
   collects bleed-through guides it never contained, so no phenotype can
   be attributed to any one of them; too high and wells lose the guides
   they did contain, and with them the statistical power of the screen.

The second question used to be answered with a hand-picked number (2% of
a well's reads, read off a histogram once and then copied forward). Here
the user states the **biological** quantity instead — how many gRNAs per
well the design intends — and :func:`derive_threshold` solves for the
abundance cutoff that delivers it. :func:`threshold_sweep` then walks a
range around that cutoff so the trade-off is visible rather than
asserted, and :func:`recommend_threshold` writes the answer out in words,
with the derived number stated explicitly. The user picks from the curve;
nothing here silently picks for them.

**Why a module of its own rather than more of** :mod:`spacr.sequencing`:
that module is the read path — FASTQ in, count table out — and it is
imported into every ``multiprocessing`` worker of a mapping run. The QC
here is a separate, later job over the table that path produced; it pulls
in plotting and statistics that the read workers must not pay for, and it
is the piece a user re-runs a dozen times while choosing a threshold,
long after the reads are mapped.

Example:
    .. code-block:: python

        from spacr.sequencing_qc import barcode_qc
        result = barcode_qc({
            'count_data': '/data/screen/sample1_paired/unique_combinations.csv',
            'qc_data':    '/data/screen/sample1_paired/qc.csv',
            'target_grnas_per_well': 4,
        })
        print(result['recommendation'])

See Also:
    :func:`spacr.sequencing.generate_barecode_mapping` — produces the
    inputs.
    :func:`spacr.ml.process_reads` — applies the chosen threshold as
    ``fraction_threshold`` when the screen is regressed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema

#: The app key this module registers its settings under.
APP_KEY = "barcode_qc"

#: Columns a count table must end up with. ``grna`` is spelled
#: ``grna_name`` in ``unique_combinations.csv`` and ``grna`` in the tables
#: :mod:`spacr.ml` passes around; both are accepted on the way in.
COUNT_COLUMNS: Tuple[str, ...] = ("plateID", "rowID", "columnID", "grna", "count")

#: Column-name spellings accepted for each canonical column. A count table
#: reaches this module from three places (the mapping pipeline, a
#: hand-assembled CSV, and :mod:`spacr.ml`) and they do not agree, so the
#: normalisation is explicit rather than a guess per caller.
_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "plateID": ("plateID", "plate_id", "plate", "plateid"),
    "rowID": ("rowID", "row_id", "row", "rowid", "row_name"),
    "columnID": ("columnID", "column_id", "column", "columnid", "col",
                 "column_name"),
    "grna": ("grna", "grna_name", "gRNA", "sgrna", "sgRNA", "guide"),
    "count": ("count", "counts", "reads", "n_reads"),
}

#: Default share of the median well's read total below which a well is
#: called starved. A well with a tenth of the typical depth cannot support
#: a per-well abundance fraction at all: one stray read is already 10% of
#: it, so it manufactures gRNA calls out of noise.
DEFAULT_STARVED_READ_FRACTION = 0.1

#: Default ratio a plate row's or column's median read depth may sit away
#: from its plate's median before it is flagged as a position effect.
#: Two-fold is the point at which a whole edge row is visibly a different
#: experiment from the plate it sits in.
DEFAULT_POSITION_RATIO = 2.0

#: How far either side of the derived threshold the sweep runs, as a
#: multiplicative factor. Four-fold each way spans the range any real
#: choice sits in (a 3% cutoff sweeps 0.75%–12%) and keeps the log axis
#: readable.
DEFAULT_SWEEP_SPAN = 4.0

#: Points on the sweep, log-spaced across the span. Enough that the knee
#: in the collision curve is a curve and not a corner.
DEFAULT_SWEEP_POINTS = 25


# ---------------------------------------------------------------------------
# Loading and normalising the count table
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the column of ``df`` that plays the ``canonical`` role."""
    lowered = {str(c).lower(): c for c in df.columns}
    for alias in _COLUMN_ALIASES[canonical]:
        if alias in df.columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def load_count_table(count_data, plate: Optional[str] = None) -> pd.DataFrame:
    """Read one or more per-well gRNA count tables into one normalised frame.

    Accepts what a barcode-mapping run writes
    (``unique_combinations.csv``: ``rowID``, ``columnID``, ``grna_name``,
    ``count``) as well as already-loaded DataFrames, and a list mixing
    both. Each source that carries no ``plateID`` is given one — ``plate``
    when supplied, otherwise ``plate1``, ``plate2``, ... in the order the
    sources are listed — so several plates can be QC'd together without
    their wells colliding.

    :param count_data: path, DataFrame, or list of either.
    :param plate: plate name for the first source that does not carry
        one. Any further nameless source is called ``plate<N>`` for its
        1-based position in ``count_data``, so the name traces back to
        the file it came from — and two plates can never share a name,
        which would merge their wells into one.
    :returns: DataFrame with :data:`COUNT_COLUMNS` plus ``prc`` (the
        ``plate_row_column`` well key), ``well_reads`` (the well's read
        total) and ``fraction`` (this gRNA's share of it).
    :raises ValueError: when a source is missing a required column, or
        when no source holds any usable row.

    Example:
        .. code-block:: python

            from spacr.sequencing_qc import load_count_table
            counts = load_count_table(
                ['/data/p1/unique_combinations.csv',
                 '/data/p2/unique_combinations.csv'])
    """
    if isinstance(count_data, (str, os.PathLike)) or isinstance(count_data, pd.DataFrame):
        sources: List[Any] = [count_data]
    else:
        sources = list(count_data)
    if not sources:
        raise ValueError("count_data holds no sources to read.")

    frames = []
    unnamed = 0
    for index, source in enumerate(sources):
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            label = f"count_data[{index}]"
        else:
            df = pd.read_csv(source)
            label = str(source)

        renames = {}
        missing = []
        for canonical in COUNT_COLUMNS:
            found = _resolve_column(df, canonical)
            if found is None:
                if canonical == "plateID":
                    continue
                missing.append(canonical)
            elif found != canonical:
                renames[found] = canonical
        if missing:
            raise ValueError(
                f"{label} is missing required column(s): "
                f"{', '.join(missing)}. A count table needs "
                f"{', '.join(COUNT_COLUMNS)} (gRNA may be spelled "
                "'grna_name').")
        if renames:
            df = df.rename(columns=renames)

        if "plateID" not in df.columns:
            if plate is not None and unnamed == 0:
                df["plateID"] = str(plate)
            else:
                df["plateID"] = f"plate{index + 1}"
            unnamed += 1

        df = df.loc[:, list(COUNT_COLUMNS)].copy()
        for key in ("plateID", "rowID", "columnID", "grna"):
            df[key] = df[key].astype("string")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df = df.dropna(subset=list(COUNT_COLUMNS))
        # A zero or negative count is not a call — it is an artefact of a
        # table that was merged or hand-edited. Keeping it would put a
        # gRNA in a well at fraction 0.0 and inflate every "gRNAs per
        # well" count below every threshold.
        df = df[df["count"] > 0]
        frames.append(df)

    counts = pd.concat(frames, axis=0, ignore_index=True)
    if counts.empty:
        raise ValueError(
            "No usable rows in count_data: every row was missing a key or "
            "carried a non-positive count.")

    sep = schema.KEY_SEPARATOR
    counts["prc"] = (counts["plateID"].astype(str) + sep
                     + counts["rowID"].astype(str) + sep
                     + counts["columnID"].astype(str))
    # Sum first: two sources may legitimately hold the same well (a
    # resequenced lane), and their reads belong to the same well total.
    counts = (counts.groupby(["prc", "plateID", "rowID", "columnID", "grna"],
                             as_index=False, observed=True)["count"].sum())
    # No zero-total well can reach this line: every surviving row carries a
    # positive count, so every well's sum is positive and the division
    # below is always defined. That is why there is no guard here — one
    # would be unreachable, and unreachable guards get believed.
    counts["well_reads"] = counts.groupby("prc")["count"].transform("sum")
    counts["fraction"] = counts["count"] / counts["well_reads"]
    return counts.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-well read depth and starvation
# ---------------------------------------------------------------------------

def reads_per_well(counts: pd.DataFrame) -> pd.DataFrame:
    """Return one row per well: its read total and how many gRNAs it saw.

    :param counts: normalised table from :func:`load_count_table`.
    :returns: DataFrame ``[prc, plateID, rowID, columnID, reads,
        n_grnas]`` sorted by ``reads`` ascending, so the starved end of
        the plate reads off the top.
    """
    grouped = counts.groupby(["prc", "plateID", "rowID", "columnID"],
                             as_index=False, observed=True).agg(
        reads=("count", "sum"), n_grnas=("grna", "nunique"))
    return grouped.sort_values("reads").reset_index(drop=True)


def starvation_cutoff(per_well: pd.DataFrame, min_reads: int = 0,
                      starved_read_fraction: float =
                      DEFAULT_STARVED_READ_FRACTION) -> float:
    """Return the read count below which a well counts as starved.

    ``min_reads`` above zero is used verbatim — an absolute floor the
    experimenter knows from the library prep. Otherwise the cut is
    ``starved_read_fraction`` of the median well's depth, which is the
    only rule that transfers between runs of different total depth.

    :param per_well: output of :func:`reads_per_well`.
    :param min_reads: absolute floor; ``0`` (the default) means derive
        one.
    :param starved_read_fraction: share of the median well's reads used
        when deriving.
    :returns: the cutoff as a float. A well is starved when its read
        total is **strictly below** it.
    :raises ValueError: on a negative ``min_reads`` or a
        ``starved_read_fraction`` outside ``(0, 1]`` — both would mark
        either no well or every well and say nothing.
    """
    if min_reads < 0:
        raise ValueError(f"min_reads must not be negative; got {min_reads!r}.")
    if not 0 < starved_read_fraction <= 1:
        raise ValueError(
            "starved_read_fraction must be in (0, 1]; got "
            f"{starved_read_fraction!r}.")
    if min_reads > 0:
        return float(min_reads)
    if per_well.empty:
        return 0.0
    return float(starved_read_fraction * per_well["reads"].median())


def starved_wells(counts: pd.DataFrame, min_reads: int = 0,
                  starved_read_fraction: float =
                  DEFAULT_STARVED_READ_FRACTION) -> pd.DataFrame:
    """Return the wells whose read depth is below the starvation cutoff.

    :param counts: normalised table from :func:`load_count_table`, or an
        already-computed :func:`reads_per_well` frame.
    :param min_reads: absolute floor; ``0`` derives one.
    :param starved_read_fraction: share of the median used when deriving.
    :returns: the starved subset of :func:`reads_per_well`, with the
        cutoff recorded in ``.attrs['cutoff']``.
    """
    per_well = (counts if "reads" in counts.columns
                else reads_per_well(counts))
    cutoff = starvation_cutoff(per_well, min_reads, starved_read_fraction)
    out = per_well[per_well["reads"] < cutoff].copy()
    out.attrs["cutoff"] = cutoff
    return out


def position_effects(counts: pd.DataFrame,
                     ratio: float = DEFAULT_POSITION_RATIO) -> pd.DataFrame:
    """Flag plate rows and columns whose read depth departs from their plate.

    A pooled screen is pipetted, and pipetting has geometry: an edge row
    that dried, a column the multichannel missed. Both show up as a whole
    row or column of wells sitting at a different depth from the rest of
    the plate, and both bias every per-well fraction computed inside them.

    :param counts: normalised table from :func:`load_count_table`.
    :param ratio: fold-change from the plate median at which a row or
        column is flagged. Must be greater than 1.
    :returns: DataFrame ``[plateID, axis, label, n_wells, median_reads,
        plate_median, ratio_to_plate, flagged]``, one row per plate row
        and per plate column, sorted worst-first.
    :raises ValueError: when ``ratio`` is not above 1 — at 1 every row is
        flagged and the report says nothing.
    """
    if ratio <= 1:
        raise ValueError(
            f"ratio must be greater than 1; got {ratio!r}. At 1 every row "
            "and column differs from the plate median and is flagged.")
    per_well = reads_per_well(counts)
    rows = []
    for plate, plate_wells in per_well.groupby("plateID", observed=True):
        plate_median = float(plate_wells["reads"].median())
        for axis, key in (("row", "rowID"), ("column", "columnID")):
            for label, group in plate_wells.groupby(key, observed=True):
                median = float(group["reads"].median())
                # A plate median of zero cannot happen (load_count_table
                # rejects empty wells), but a defensive guard here keeps
                # the ratio finite for a caller that built the frame by
                # hand.
                fold = median / plate_median if plate_median else np.inf
                rows.append({
                    "plateID": plate, "axis": axis, "label": label,
                    "n_wells": int(len(group)), "median_reads": median,
                    "plate_median": plate_median, "ratio_to_plate": fold,
                    "flagged": bool(fold >= ratio or fold <= 1.0 / ratio),
                })
    out = pd.DataFrame(rows, columns=["plateID", "axis", "label", "n_wells",
                                      "median_reads", "plate_median",
                                      "ratio_to_plate", "flagged"])
    if out.empty:
        return out
    # Worst first: distance from parity on a log scale, so a half-depth
    # row and a double-depth row rank equally badly.
    order = np.abs(np.log2(out["ratio_to_plate"].replace(0, np.nan)))
    return (out.assign(_order=order.fillna(np.inf))
               .sort_values("_order", ascending=False)
               .drop(columns="_order").reset_index(drop=True))


# ---------------------------------------------------------------------------
# Library read depth
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = even, 1 = one winner)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if (values < 0).any():
        raise ValueError("Gini is undefined for negative read counts.")
    total = values.sum()
    if total <= 0:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * ordered).sum()) / (n * total) - (n + 1.0) / n)


def library_depth(counts: pd.DataFrame,
                  expected_grnas: Optional[Iterable[str]] = None
                  ) -> Dict[str, Any]:
    """Summarise how evenly the gRNA library is covered by the run.

    :param counts: normalised table from :func:`load_count_table`.
    :param expected_grnas: the library as designed — names from the gRNA
        reference. Supplying it is what turns "we saw 4,900 guides" into
        "2% of the library was never seen", which is the number that says
        whether the screen has the coverage it was powered for.
    :returns: dict with ``n_grnas_observed``, ``n_grnas_expected``,
        ``dropout_fraction``, ``dropped_grnas`` (sorted names),
        ``gini``, ``skew_ratio`` (90th/10th percentile of per-gRNA read
        totals — the standard pooled-library evenness number),
        ``top_decile_share`` and ``reads_per_grna`` (a Series, descending).
    """
    per_grna = (counts.groupby("grna", observed=True)["count"].sum()
                .sort_values(ascending=False))
    values = per_grna.to_numpy(dtype=float)
    observed = set(per_grna.index.astype(str))

    expected: Optional[List[str]] = None
    dropped: List[str] = []
    if expected_grnas is not None:
        expected = sorted({str(g) for g in expected_grnas})
        dropped = sorted(set(expected) - observed)

    if values.size:
        p10, p90 = np.percentile(values, [10, 90])
        # p10 of a library where a tenth of the guides are absent is 0.
        # Reporting inf is honest — the skew is unbounded — and is what a
        # user needs to see rather than a silently clipped ratio.
        skew = float(p90 / p10) if p10 > 0 else float("inf")
        top_n = max(1, int(np.ceil(0.1 * values.size)))
        top_share = float(np.sort(values)[::-1][:top_n].sum() / values.sum())
    else:
        skew, top_share = float("nan"), float("nan")

    return {
        "n_grnas_observed": int(per_grna.size),
        "n_grnas_expected": (len(expected) if expected is not None else None),
        "dropout_fraction": (len(dropped) / len(expected)
                             if expected else None),
        "dropped_grnas": dropped,
        "gini": _gini(values),
        "skew_ratio": skew,
        "top_decile_share": top_share,
        "reads_per_grna": per_grna,
    }


# ---------------------------------------------------------------------------
# Unmapped reads
# ---------------------------------------------------------------------------

def unmapped_read_fractions(qc_data, counts: Optional[pd.DataFrame] = None
                            ) -> Dict[str, Any]:
    """Read the mapping run's ``qc.csv`` and report what failed to map.

    ``qc.csv`` accumulates, per barcode field, the number of reads whose
    sequence matched no entry in that field's reference CSV, alongside
    ``total_reads``.

    **What the denominator is.** ``total_reads`` counts reads that
    matched the barcode regex — a read that never found the anchor
    sequence is not in the file at all. So these fractions are "of the
    reads that reached barcode lookup", not "of the FASTQ". That is the
    number that diagnoses a wrong or reverse-complemented barcode
    reference, which is what this panel is for; a run where the regex
    itself misses is already loud in the mapping log.

    :param qc_data: path, DataFrame, or list of either.
    :param counts: optional normalised count table for the same run. When
        given, ``unmapped_fraction`` is exact — the count table holds only
        reads whose three barcodes *all* resolved, so the shortfall
        against ``total_reads`` is the true joint unmapped share.
        Without it only the per-field fractions and bounds are reported.
    :returns: dict with ``total_reads``, ``per_field`` (field -> unmapped
        fraction), ``mapped_reads``/``unmapped_fraction`` (only with
        ``counts``), and ``unmapped_fraction_lower``/``_upper`` — the
        bounds implied by the per-field numbers alone.
    :raises ValueError: when no ``total_reads`` column is present, or it
        sums to zero.
    """
    if isinstance(qc_data, (str, os.PathLike)) or isinstance(qc_data, pd.DataFrame):
        sources: List[Any] = [qc_data]
    else:
        sources = list(qc_data)

    frames = [src if isinstance(src, pd.DataFrame) else pd.read_csv(src)
              for src in sources]
    qc = pd.concat(frames, axis=0, ignore_index=True)
    if "total_reads" not in qc.columns:
        raise ValueError(
            "qc_data has no 'total_reads' column; it is not a qc.csv from a "
            f"barcode-mapping run (columns: {list(qc.columns)}).")
    total = float(pd.to_numeric(qc["total_reads"], errors="coerce").sum())
    if total <= 0:
        raise ValueError(
            "qc_data reports zero total reads, so no fraction of it is "
            "defined. The mapping run matched nothing.")

    fields = [f for f in ("columnID", "rowID", "grna_name") if f in qc.columns]
    per_field = {
        field: float(pd.to_numeric(qc[field], errors="coerce").sum() / total)
        for field in fields
    }
    out: Dict[str, Any] = {
        "total_reads": total,
        "per_field": per_field,
        # A read is lost if ANY field failed. At best the failures all
        # coincide on the same reads (lower bound = the worst field); at
        # worst they are disjoint (upper bound = their sum).
        "unmapped_fraction_lower": max(per_field.values()) if per_field else 0.0,
        "unmapped_fraction_upper": min(1.0, sum(per_field.values())),
    }
    if counts is not None:
        mapped = float(counts["count"].sum())
        out["mapped_reads"] = mapped
        out["unmapped_fraction"] = max(0.0, 1.0 - mapped / total)
    return out


# ---------------------------------------------------------------------------
# Barcode reference collisions
# ---------------------------------------------------------------------------

def _read_reference(reference) -> Dict[str, str]:
    """Return ``{name: sequence}`` from a barcode CSV, FASTA, or mapping."""
    if isinstance(reference, Mapping):
        return {str(k): str(v).upper() for k, v in reference.items()}
    path = str(reference)
    if path.lower().endswith((".fa", ".fasta", ".fna")):
        table: Dict[str, str] = {}
        name = None
        chunks: List[str] = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        table[name] = "".join(chunks).upper()
                    name, chunks = line[1:].split()[0], []
                else:
                    chunks.append(line)
        if name is not None:
            table[name] = "".join(chunks).upper()
        return table
    df = pd.read_csv(path)
    missing = {"name", "sequence"}.difference(df.columns)
    if missing:
        raise ValueError(
            f"Barcode reference {path!r} is missing column(s): "
            f"{', '.join(sorted(missing))}. It needs 'name' and 'sequence'.")
    return {str(n): str(s).upper()
            for n, s in zip(df["name"], df["sequence"])}


def barcode_collisions(references: Mapping[str, Any], max_distance: int = 1
                       ) -> pd.DataFrame:
    """Find barcode pairs a sequencing error could turn into each other.

    Two barcodes of the same length within ``max_distance`` substitutions
    are a **collision**: one miscalled base moves a read from one well (or
    one guide) to another, and nothing downstream can tell that it
    happened. Exact duplicates are reported too, at distance 0 — those are
    fatal rather than risky, and :func:`spacr.sequencing.map_sequences_to_names`
    refuses to run on them.

    Only substitutions are considered, and only within a reference set.
    Indels would change the barcode's length and so shift every field
    after it in the read, which the regex rejects outright rather than
    mis-assigning; and a row barcode cannot be confused with a gRNA
    barcode because they are read out of different positions.

    :param references: ``{label: source}``, where each source is a
        ``name,sequence`` CSV path, a FASTA path, or a ``{name: sequence}``
        mapping. The label ("row", "column", "grna") names the set in the
        output.
    :param max_distance: maximum number of substitutions. ``1`` is the
        default because a single miscalled base is the common event; ``0``
        reports only exact duplicates.
    :returns: DataFrame ``[reference, name_a, name_b, distance,
        sequence_a, sequence_b]``, one row per colliding pair, sorted by
        reference then distance.
    :raises ValueError: on a negative ``max_distance``.

    Example:
        .. code-block:: python

            from spacr.sequencing_qc import barcode_collisions
            pairs = barcode_collisions({'row': '/data/barcodes/row.csv'})
    """
    if max_distance < 0:
        raise ValueError(
            f"max_distance must not be negative; got {max_distance!r}.")

    rows = []
    for label, source in references.items():
        table = _read_reference(source)
        names = list(table)
        seqs = [table[n] for n in names]
        seen: set = set()

        # Exact duplicates, by sequence.
        by_sequence: Dict[str, List[str]] = {}
        for name, seq in zip(names, seqs):
            by_sequence.setdefault(seq, []).append(name)
        for seq, group in by_sequence.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair = tuple(sorted((group[i], group[j])))
                    seen.add(pair)
                    rows.append({"reference": label, "name_a": pair[0],
                                 "name_b": pair[1], "distance": 0,
                                 "sequence_a": seq, "sequence_b": seq})

        if max_distance >= 1:
            # Hamming-1 neighbours without the N^2 comparison: two equal-
            # length sequences differ in at most one position exactly when
            # they agree after masking that position. A pooled gRNA
            # library is 10^4-10^5 barcodes, where N^2 is not affordable
            # and this is linear in N.
            buckets: Dict[Tuple[int, str], List[int]] = {}
            for idx, seq in enumerate(seqs):
                for pos in range(len(seq)):
                    key = (pos, seq[:pos] + "\0" + seq[pos + 1:])
                    buckets.setdefault(key, []).append(idx)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = members[i], members[j]
                        pair = tuple(sorted((names[a], names[b])))
                        if pair in seen or seqs[a] == seqs[b]:
                            continue
                        seen.add(pair)
                        rows.append({
                            "reference": label,
                            "name_a": pair[0], "name_b": pair[1],
                            "distance": 1,
                            "sequence_a": table[pair[0]],
                            "sequence_b": table[pair[1]]})

        if max_distance >= 2:
            # Beyond one substitution the masking trick no longer applies
            # and the honest implementation is the pairwise one. It is
            # only reachable when the caller asks for it.
            arrays = [np.frombuffer(s.encode(), dtype=np.uint8) for s in seqs]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if arrays[i].size != arrays[j].size:
                        continue
                    pair = tuple(sorted((names[i], names[j])))
                    if pair in seen:
                        continue
                    distance = int((arrays[i] != arrays[j]).sum())
                    if 2 <= distance <= max_distance:
                        seen.add(pair)
                        rows.append({
                            "reference": label,
                            "name_a": pair[0], "name_b": pair[1],
                            "distance": distance,
                            "sequence_a": table[pair[0]],
                            "sequence_b": table[pair[1]]})

    out = pd.DataFrame(rows, columns=["reference", "name_a", "name_b",
                                      "distance", "sequence_a", "sequence_b"])
    if out.empty:
        return out
    return out.sort_values(["reference", "distance", "name_a", "name_b"]
                           ).reset_index(drop=True)


def collision_summary(references: Mapping[str, Any],
                      collisions: pd.DataFrame,
                      counts: Optional[pd.DataFrame] = None
                      ) -> pd.DataFrame:
    """Per-reference collision rate, and the share of reads it touches.

    :param references: the same mapping passed to
        :func:`barcode_collisions`.
    :param collisions: its output.
    :param counts: optional normalised count table. When given, the gRNA
        reference also reports ``reads_at_risk`` — the share of mapped
        reads carrying a barcode that has a near neighbour, which is what
        turns a list of risky pairs into an amount of data at risk.
    :returns: DataFrame ``[reference, n_barcodes, n_colliding_pairs,
        n_barcodes_at_risk, collision_rate, reads_at_risk]``.
    """
    rows = []
    for label, source in references.items():
        table = _read_reference(source)
        subset = collisions[collisions["reference"] == label] if not collisions.empty \
            else collisions
        at_risk = set()
        if not subset.empty:
            at_risk = set(subset["name_a"]).union(subset["name_b"])
        n = len(table)
        reads_at_risk = None
        if counts is not None and n:
            total = float(counts["count"].sum())
            if total > 0:
                hit = counts[counts["grna"].astype(str).isin(at_risk)]
                reads_at_risk = float(hit["count"].sum() / total)
        rows.append({
            "reference": label,
            "n_barcodes": n,
            "n_colliding_pairs": int(len(subset)),
            "n_barcodes_at_risk": len(at_risk),
            "collision_rate": (len(at_risk) / n) if n else float("nan"),
            "reads_at_risk": reads_at_risk,
        })
    return pd.DataFrame(rows, columns=["reference", "n_barcodes",
                                       "n_colliding_pairs",
                                       "n_barcodes_at_risk",
                                       "collision_rate", "reads_at_risk"])


# ---------------------------------------------------------------------------
# The threshold: derive it from the target, then sweep around it
# ---------------------------------------------------------------------------

class WellFractions:
    """Per-well gRNA abundance fractions, prepared for repeated thresholding.

    Sorting each well's fractions once turns "how many gRNAs survive
    threshold t" into a binary search, which is what makes an exact
    derivation over every observed fraction affordable — the sweep and the
    bisection between them evaluate the same population dozens of times.

    :param counts: normalised table from :func:`load_count_table`.
    :param wells: optional restriction of the well population, as ``prc``
        keys. Excluding starved wells here is what keeps them from
        dragging the derived threshold: a well with nine reads reports one
        gRNA at any cutoff and pulls the median down.
    :raises ValueError: when the restriction leaves no wells.
    """

    def __init__(self, counts: pd.DataFrame,
                 wells: Optional[Iterable[str]] = None):
        if wells is not None:
            keep = set(str(w) for w in wells)
            counts = counts[counts["prc"].astype(str).isin(keep)]
        if counts.empty:
            raise ValueError(
                "No wells left to threshold: the count table is empty, or "
                "every well was excluded.")
        self.wells: List[str] = []
        self._sorted: List[np.ndarray] = []
        for prc, group in counts.groupby("prc", observed=True):
            self.wells.append(str(prc))
            self._sorted.append(np.sort(group["fraction"].to_numpy(float)))
        self.n_wells = len(self.wells)
        self.all_fractions = np.sort(counts["fraction"].to_numpy(float))
        self.total_reads = float(counts["count"].sum())
        self._reads_sorted_by_fraction = counts.sort_values("fraction")

    def counts_at(self, thresholds) -> np.ndarray:
        """gRNAs surviving each threshold, as ``[n_wells, n_thresholds]``.

        :param thresholds: scalar or array of abundance cutoffs. A gRNA
            is kept when its fraction is greater than or equal to the
            cutoff, matching :func:`spacr.ml.process_reads`.
        :returns: integer array of per-well surviving gRNA counts.
        """
        thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
        out = np.empty((self.n_wells, thresholds.size), dtype=np.int64)
        for index, fractions in enumerate(self._sorted):
            out[index] = fractions.size - np.searchsorted(
                fractions, thresholds, side="left")
        return out

    def reads_retained_at(self, thresholds) -> np.ndarray:
        """Share of all mapped reads surviving each threshold."""
        thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
        frame = self._reads_sorted_by_fraction
        fractions = frame["fraction"].to_numpy(float)
        cumulative = np.concatenate([[0.0],
                                     np.cumsum(frame["count"].to_numpy(float))])
        start = np.searchsorted(fractions, thresholds, side="left")
        kept = cumulative[-1] - cumulative[start]
        return kept / self.total_reads if self.total_reads else kept * 0.0

    def statistic_at(self, thresholds, statistic: str = "median") -> np.ndarray:
        """Median or mean gRNAs per well, over the **fixed** well population.

        The population is fixed — a well that loses its last gRNA stays in
        the denominator as a zero — and that is what makes the result a
        non-increasing function of the threshold: every well's count only
        ever falls, so every order statistic of them only ever falls.
        Dropping emptied wells instead would let the median jump *up* when
        a sparse well is removed, and a target could then be met at two
        thresholds far apart with no way to say which was meant.

        :param thresholds: scalar or array of cutoffs.
        :param statistic: ``'median'`` or ``'mean'``.
        :returns: array of the statistic, aligned with ``thresholds``.
        :raises ValueError: on an unknown statistic.
        """
        table = self.counts_at(thresholds)
        if statistic == "median":
            return np.median(table, axis=0)
        if statistic == "mean":
            return table.mean(axis=0)
        raise ValueError(
            f"statistic must be 'median' or 'mean'; got {statistic!r}.")


@dataclass(frozen=True)
class ThresholdChoice:
    """The abundance threshold a stated gRNAs-per-well target implies.

    :param threshold: the derived cutoff — a gRNA's minimum share of its
        well's reads.
    :param achieved: the gRNAs-per-well statistic actually obtained at it.
    :param target: what the user asked for.
    :param statistic: ``'median'`` or ``'mean'``.
    :param n_wells: size of the well population it was derived over.
    :param attainable: False when even keeping every observed gRNA falls
        short of the target — the library, not the threshold, is the
        limit, and ``threshold`` is then the most permissive cutoff there is.
    :param n_candidates: how many distinct observed fractions were
        searched. The derivation is exact over this set: the statistic can
        only change at a fraction that is actually in the data.
    :param interval_low: exclusive lower end of the plateau of thresholds
        that all yield ``achieved``. ``threshold`` is its geometric
        middle.
    :param interval_high: inclusive upper end of the same plateau — one
        step further and the statistic drops below ``achieved``.
    """

    threshold: float
    achieved: float
    target: float
    statistic: str
    n_wells: int
    attainable: bool
    n_candidates: int
    interval_low: float
    interval_high: float

    def as_dict(self) -> Dict[str, Any]:
        """Return the choice as a plain dict, for CSV/JSON output."""
        return asdict(self)


def derive_threshold(counts: pd.DataFrame, target_grnas_per_well: float,
                     statistic: str = "median",
                     wells: Optional[Iterable[str]] = None) -> ThresholdChoice:
    """Solve for the abundance threshold that delivers a stated gRNAs-per-well target.

    This replaces choosing a cutoff by eye. The experimenter states the
    biological quantity — how many gRNAs a well is meant to carry, which
    is a design decision about power versus attributability — and the
    cutoff is whatever number delivers it *in this run's data*. Two runs
    at different depth get different numbers for the same target, which is
    the point.

    The search is exact rather than gridded: the gRNAs-per-well statistic
    is a step function that can only change at a fraction actually present
    in the table, so every distinct observed fraction is a candidate and
    the monotonicity of the statistic (see
    :meth:`WellFractions.statistic_at`) lets a bisection find the answer
    in ``log2(n)`` evaluations. Where no cutoff hits the target exactly —
    the statistic is a median of integers and jumps — the candidate
    landing closest is returned, preferring the one that still *meets* the
    target over the one that falls short.

    **The number returned sits in the middle of its plateau, not on its
    edge.** A whole range of thresholds gives the same gRNAs-per-well
    answer — on a clean run that range is the empty space between the
    guides a well really carried and the bleed-through tail below them,
    which is exactly what a histogram is being read for when a cutoff is
    picked by eye. Returning either edge of it would put the cutoff where
    a re-sequenced run or a rounded count flips guides across it. The
    geometric middle is the same answer and holds up; both edges are
    reported as ``interval_low`` / ``interval_high`` so the width of the
    plateau — how much slack the choice has — is visible too.

    :param counts: normalised table from :func:`load_count_table`.
    :param target_grnas_per_well: the target. Must be positive.
    :param statistic: ``'median'`` (default) or ``'mean'``.
    :param wells: optional ``prc`` restriction of the well population —
        pass the non-starved wells to keep unusable wells out of the fit.
    :returns: a :class:`ThresholdChoice`.
    :raises ValueError: on a non-positive target or an unknown statistic.

    Example:
        .. code-block:: python

            from spacr.sequencing_qc import load_count_table, derive_threshold
            counts = load_count_table('unique_combinations.csv')
            choice = derive_threshold(counts, target_grnas_per_well=4)
            print(choice.threshold, choice.achieved)
    """
    if target_grnas_per_well <= 0:
        raise ValueError(
            "target_grnas_per_well must be positive; got "
            f"{target_grnas_per_well!r}. A well with no gRNA carries no "
            "screen.")
    if statistic not in ("median", "mean"):
        raise ValueError(
            f"statistic must be 'median' or 'mean'; got {statistic!r}.")

    fractions = WellFractions(counts, wells=wells)
    candidates = np.unique(fractions.all_fractions)
    target = float(target_grnas_per_well)

    def stat(value: float) -> float:
        return float(fractions.statistic_at(value, statistic)[0])

    def choice_at(index: int, achieved: float, attainable: bool
                  ) -> ThresholdChoice:
        """Build the result for candidate ``index``, centred in its plateau.

        ``index`` is the strictest candidate yielding ``achieved``. The
        plateau reaches down to just above the last candidate that
        yielded MORE than ``achieved``, found by a second bisection on
        the same monotone statistic.
        """
        high = float(candidates[index])
        # Last candidate whose statistic is still strictly above the one
        # we settled on; everything after it is on the plateau.
        low_index = 0
        if index > 0 and stat(float(candidates[0])) > achieved:
            lo, hi = 0, index - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if stat(float(candidates[mid])) > achieved:
                    lo = mid
                else:
                    hi = mid - 1
            low_index = lo + 1
        # Thresholds below the smallest observed fraction all behave
        # identically, so that fraction — not zero — is the meaningful
        # bottom of an open-ended plateau.
        low = float(candidates[low_index - 1] if low_index > 0
                    else candidates[0])
        # Geometric middle: abundances are ratios spanning orders of
        # magnitude, so halfway between 0.004 and 0.22 is 0.03, not 0.11.
        middle = float(np.sqrt(low * high)) if high > low else high
        return ThresholdChoice(
            threshold=middle, achieved=achieved, target=target,
            statistic=statistic, n_wells=fractions.n_wells,
            attainable=attainable, n_candidates=int(candidates.size),
            interval_low=low, interval_high=high)

    lowest = stat(candidates[0])
    if lowest < target:
        # Even keeping every observed gRNA does not reach the target. No
        # threshold can; say so instead of returning the smallest number
        # in the table as though it were a choice.
        return choice_at(0, lowest, attainable=False)

    # Largest candidate whose statistic still meets the target. The
    # statistic is non-increasing, so the predicate is monotone and a
    # bisection is exact.
    low, high = 0, int(candidates.size) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if stat(float(candidates[mid])) >= target:
            low = mid
        else:
            high = mid - 1
    best_index, achieved = low, stat(float(candidates[low]))

    # The next candidate up is the first that falls short. When it lands
    # closer to the target than the one that meets it, it is the better
    # answer; on a tie the one that meets the target wins, because a well
    # short of its guides has lost power that no later step recovers.
    if low + 1 < candidates.size:
        alternative_achieved = stat(float(candidates[low + 1]))
        if abs(alternative_achieved - target) < abs(achieved - target):
            best_index, achieved = low + 1, alternative_achieved

    return choice_at(best_index, achieved, attainable=True)


def sweep_grid(threshold: float, span: float = DEFAULT_SWEEP_SPAN,
               points: int = DEFAULT_SWEEP_POINTS, *,
               low: Optional[float] = None,
               high: Optional[float] = None) -> np.ndarray:
    """Log-spaced thresholds spanning ``span``-fold either side of ``threshold``.

    :param threshold: the centre — the derived cutoff.
    :param span: multiplicative half-width. ``4.0`` sweeps a quarter to
        four times the derived value.
    :param points: how many points, before the centre is inserted.
    :param low: absolute lower end, overriding ``threshold / span`` when
        it is lower. :func:`barcode_qc` uses it to make sure the sweep
        reaches down into the bleed-through tail, where the collision
        rate turns — a curve that stops above the junk cannot show the
        user the cost of relaxing into it.
    :param high: absolute upper end, overriding ``threshold * span``.
    :returns: sorted unique array of cutoffs in ``(0, 1]``, always
        containing ``threshold`` itself so the derived point is on the
        curve and not merely near it.
    :raises ValueError: on a non-positive threshold, a span at or below 1,
        fewer than 3 points, or a ``low`` that is not below ``high``.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive; got {threshold!r}.")
    if span <= 1:
        raise ValueError(
            f"span must be greater than 1; got {span!r}. At 1 the sweep is a "
            "single point and shows no trade-off.")
    if points < 3:
        raise ValueError(
            f"points must be at least 3; got {points!r}. Two points cannot "
            "show a knee.")
    bottom = min(threshold / span, low) if low is not None else threshold / span
    top = min(1.0, max(threshold * span, high) if high is not None
              else threshold * span)
    if bottom <= 0 or bottom >= top:
        raise ValueError(
            f"the sweep range ({bottom!r}, {top!r}] is empty or non-positive; "
            "check low/high against the derived threshold.")
    grid = np.geomspace(bottom, top, int(points))
    # Drop grid points that merely round to the centre before inserting
    # it. np.unique compares bit patterns, so geomspace's 0.21999999999997
    # would survive next to an inserted 0.22 as a second, near-identical
    # row of the sweep — two lines the user cannot tell apart reporting
    # different numbers.
    grid = grid[~np.isclose(grid, threshold, rtol=1e-9, atol=0.0)]
    grid = np.concatenate([grid, [threshold]])
    grid = grid[(grid > 0) & (grid <= 1.0)]
    return np.unique(grid)


def threshold_sweep(counts: pd.DataFrame, thresholds,
                    target_grnas_per_well: float,
                    statistic: str = "median",
                    wells: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Evaluate the whole trade-off at each threshold.

    :param counts: normalised table from :func:`load_count_table`.
    :param thresholds: array of cutoffs — usually :func:`sweep_grid`
        around a :func:`derive_threshold` result.
    :param target_grnas_per_well: the attribution budget. A well holding
        more gRNAs than this is counted as a **collision**: its phenotype
        is a mixture of more guides than the design set out to
        disentangle, so it cannot be attributed to any one of them.
    :param statistic: ``'median'`` or ``'mean'``, for the headline
        gRNAs-per-well column.
    :param wells: optional ``prc`` restriction of the well population.
    :returns: DataFrame, one row per threshold, with

        - ``grnas_per_well`` — the requested statistic over all wells;
          non-increasing in the threshold.
        - ``grnas_per_well_retained`` — the same statistic over wells that
          still hold at least one gRNA. Easier to read, and *not*
          monotone: it rises when a one-gRNA well drops out.
        - ``wells_retained`` / ``well_retention`` — non-increasing.
        - ``collision_rate`` — share of **all** wells over the budget;
          non-increasing, because each well's gRNA count only falls.
        - ``collision_rate_retained`` — the same numerator over retained
          wells only. Not monotone, for the same reason as above.
        - ``n_calls`` — surviving (well, gRNA) pairs; non-increasing.
        - ``reads_retained`` — share of mapped reads kept; non-increasing.
    """
    fractions = WellFractions(counts, wells=wells)
    grid = np.atleast_1d(np.asarray(thresholds, dtype=float))
    table = fractions.counts_at(grid)
    retained_mask = table >= 1
    wells_retained = retained_mask.sum(axis=0)
    over_budget = (table > float(target_grnas_per_well)).sum(axis=0)

    if statistic == "median":
        headline = np.median(table, axis=0)
    elif statistic == "mean":
        headline = table.mean(axis=0)
    else:
        raise ValueError(
            f"statistic must be 'median' or 'mean'; got {statistic!r}.")

    retained_headline = np.full(grid.size, np.nan)
    for index in range(grid.size):
        column = table[:, index]
        kept = column[column >= 1]
        if kept.size:
            retained_headline[index] = (np.median(kept) if statistic == "median"
                                        else kept.mean())

    n_wells = fractions.n_wells
    with np.errstate(invalid="ignore", divide="ignore"):
        collision_retained = np.where(wells_retained > 0,
                                      over_budget / np.maximum(wells_retained, 1),
                                      np.nan)
    return pd.DataFrame({
        "threshold": grid,
        "grnas_per_well": headline,
        "grnas_per_well_retained": retained_headline,
        "wells_retained": wells_retained,
        "well_retention": wells_retained / n_wells,
        "wells_over_budget": over_budget,
        "collision_rate": over_budget / n_wells,
        "collision_rate_retained": collision_retained,
        "n_calls": table.sum(axis=0),
        "reads_retained": fractions.reads_retained_at(grid),
    })


def _row_at(sweep: pd.DataFrame, threshold: float) -> pd.Series:
    """Return the sweep row closest to ``threshold``."""
    index = (sweep["threshold"] - threshold).abs().idxmin()
    return sweep.loc[index]


def recommend_threshold(sweep: pd.DataFrame, choice: ThresholdChoice) -> str:
    """Write the threshold recommendation out in words.

    A curve tells a reader where the knee is only if they already know
    what they are looking for. This states the derived number, what it
    buys, what relaxing and tightening it cost, and where the collision
    rate turns — in sentences, so the choice can be quoted in a methods
    section.

    :param sweep: output of :func:`threshold_sweep`.
    :param choice: output of :func:`derive_threshold`.
    :returns: a multi-line string.
    """
    lines: List[str] = []
    t = choice.threshold
    at = _row_at(sweep, t)
    statistic = choice.statistic

    lines.append(
        f"Target: {choice.target:g} gRNAs per well ({statistic}). "
        f"Derived abundance threshold: {t:.4f} "
        f"({100 * t:.2f}% of a well's reads).")

    if not choice.attainable:
        lines.append(
            f"WARNING: the target is out of reach. Keeping every gRNA "
            f"observed gives a {statistic} of only {choice.achieved:.1f} per "
            f"well, so no threshold reaches {choice.target:g}. The limit is "
            f"the library or the read depth, not the cutoff — check the "
            f"starved wells and the library-dropout figures before lowering "
            f"anything further.")
    else:
        lines.append(
            f"At {t:.4f} the {statistic} well carries "
            f"{at['grnas_per_well']:.1f} gRNAs and "
            f"{100 * at['well_retention']:.0f}% of wells are retained "
            f"({int(at['wells_retained'])} of {choice.n_wells}); "
            f"{100 * at['collision_rate']:.0f}% of wells hold more than "
            f"{choice.target:g} gRNAs and so cannot have a phenotype "
            f"attributed to a single guide. "
            f"{100 * at['reads_retained']:.0f}% of mapped reads survive.")
        if abs(choice.achieved - choice.target) > 1e-9:
            lines.append(
                f"Note: no cutoff hits {choice.target:g} exactly — the "
                f"{statistic} of a per-well count moves in steps — so "
                f"{t:.4f} is the closest, at {choice.achieved:.1f}.")
        if choice.interval_high > choice.interval_low:
            lines.append(
                f"Every cutoff from {choice.interval_low:.4f} to "
                f"{choice.interval_high:.4f} gives the same {statistic} of "
                f"{choice.achieved:.1f}; {t:.4f} is the geometric middle of "
                f"that range and so the point least sensitive to a re-run. "
                f"The collision rate does still vary across the range — see "
                f"the sweep — which is why the middle is the number to quote "
                f"and not either edge.")

    # Quote the thresholds where the answer actually CHANGES, not the ends
    # of the sweep: on a wide plateau the ends both report the same
    # gRNAs-per-well and the sentence says nothing.
    here = float(at["grnas_per_well"])
    looser = sweep[(sweep["threshold"] < t) & (sweep["grnas_per_well"] > here)]
    tighter = sweep[(sweep["threshold"] > t) & (sweep["grnas_per_well"] < here)]
    if looser.empty:
        looser = sweep[sweep["threshold"] < t]
    if tighter.empty:
        tighter = sweep[sweep["threshold"] > t]
    if not looser.empty:
        row = looser.iloc[-1]
        lines.append(
            f"Relaxing to {row['threshold']:.4f} takes the {statistic} to "
            f"{row['grnas_per_well']:.1f} gRNAs per well and the collision "
            f"rate to {100 * row['collision_rate']:.0f}%, for "
            f"{100 * (row['well_retention'] - at['well_retention']):+.0f} "
            f"percentage points of well retention.")
    if not tighter.empty:
        row = tighter.iloc[0]
        if (row["grnas_per_well"] < here
                or row["well_retention"] < at["well_retention"]):
            lines.append(
                f"Tightening to {row['threshold']:.4f} drops the {statistic} "
                f"to {row['grnas_per_well']:.1f} gRNAs per well and retention "
                f"to {100 * row['well_retention']:.0f}%.")
        else:
            lines.append(
                f"Tightening anywhere up to {sweep['threshold'].max():.4f} "
                f"changes neither the {statistic} nor well retention — the "
                f"guides this keeps are well clear of the cutoff.")

    # The knee: the adjacent pair of thresholds below the derived one
    # across which the collision rate climbs fastest per octave. That is
    # the sentence a methods section wants — "below X, collisions rise
    # sharply" — so it is quoted as the two measured values, not as a
    # slope the reader has to integrate.
    below = sweep[sweep["threshold"] <= t].sort_values("threshold")
    if len(below) >= 2:
        thresholds = below["threshold"].to_numpy(float)
        rate = below["collision_rate"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.diff(rate) / np.diff(np.log2(thresholds))
        if slope.size and np.isfinite(slope).any() and np.nanmin(slope) < 0:
            steepest = int(np.nanargmin(slope))
            lines.append(
                f"Below {thresholds[steepest + 1]:.4f} the collision rate "
                f"rises sharply: {100 * rate[steepest + 1]:.0f}% at "
                f"{thresholds[steepest + 1]:.4f} becomes "
                f"{100 * rate[steepest]:.0f}% at {thresholds[steepest]:.4f}. "
                f"That is the floor worth defending.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figure(fig, dst: Optional[str], name: str) -> Optional[str]:
    """Write ``fig`` to ``dst/name.pdf`` and return the path (or None)."""
    if dst is None:
        return None
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, f"{name}.pdf")
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    return path


def plot_threshold_sweep(sweep: pd.DataFrame, choice: ThresholdChoice,
                         dst: Optional[str] = None):
    """Plot the sweep, with the derived threshold marked and labelled.

    Two stacked panels share one log-scaled threshold axis: gRNAs per
    well above, well retention and collision rate below. Two panels
    rather than a twin y-axis because the quantities have nothing in
    common — a count and two percentages — and overlaying them puts the
    flat 100% retention line on top of the frame, where it cannot be
    read. The derived threshold is a labelled vertical line carrying its
    own numeric value in both panels: the user must be able to see what
    their target translated to without reading it off the axis.

    The gRNAs-per-well axis is symlog around 1. A relaxed threshold puts
    tens of guides in a well while the interesting region is a handful,
    and on a linear axis the answer is a flat line at the bottom of the
    plot.

    :param sweep: output of :func:`threshold_sweep`.
    :param choice: output of :func:`derive_threshold`.
    :param dst: folder to write ``threshold_sweep.pdf`` into; ``None``
        returns the figure without saving.
    :returns: the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    teal = (0 / 255, 155 / 255, 155 / 255)
    amber = (200 / 255, 130 / 255, 0 / 255)
    red = (180 / 255, 40 / 255, 60 / 255)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})

    top.plot(sweep["threshold"], sweep["grnas_per_well"], color=teal, lw=2,
             label=f"gRNAs per well ({choice.statistic}, all wells)")
    top.plot(sweep["threshold"], sweep["grnas_per_well_retained"],
             color=teal, lw=1.2, ls=":",
             label="gRNAs per well (retained wells only)")
    top.axhline(choice.target, color="black", ls="--", lw=1,
                label=f"target ({choice.target:g})")
    # linscale keeps the 0-1 linear band from eating a third of the panel;
    # "no guides left" needs to be visible, not prominent.
    top.set_yscale("symlog", linthresh=1, linscale=0.35)
    top.set_ylim(bottom=0)
    top.set_ylabel("gRNAs per well")
    top.legend(loc="upper right", fontsize=8)
    top.set_title(
        f"Threshold sweep around the target of {choice.target:g} gRNAs/well")

    bottom.plot(sweep["threshold"], 100 * sweep["well_retention"],
                color=amber, lw=1.8, label="wells retained (%)")
    bottom.plot(sweep["threshold"], 100 * sweep["collision_rate"],
                color=red, lw=1.8,
                label=f"wells over {choice.target:g} gRNAs (%)")
    bottom.set_ylim(-2, 102)
    bottom.set_ylabel("% of wells")
    bottom.set_xscale("log")
    bottom.set_xlabel("abundance threshold (gRNA share of a well's reads)")
    bottom.legend(loc="center left", fontsize=8)

    for axis in (top, bottom):
        axis.axvline(choice.threshold, color="black", lw=1.2)
    # Anchored in axes coordinates on the y and data coordinates on the x,
    # so the label rides the line at a fixed height whatever the symlog
    # axis does with its limits.
    top.annotate(f"derived: {choice.threshold:.4f}",
                 xy=(choice.threshold, 0.02),
                 xycoords=top.get_xaxis_transform(),
                 xytext=(5, 0), textcoords="offset points",
                 rotation=90, va="bottom", ha="left", fontsize=9,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                       "ec": "none", "alpha": 0.75})
    _save_figure(fig, dst, "threshold_sweep")
    return fig


def plot_barcode_qc(counts: pd.DataFrame, *, per_well: pd.DataFrame,
                    starved: pd.DataFrame, positions: pd.DataFrame,
                    depth: Mapping[str, Any],
                    unmapped: Optional[Mapping[str, Any]] = None,
                    dst: Optional[str] = None):
    """Draw the four QC panels of a barcode-mapping run.

    ``reads per well`` (with the starvation cut marked), ``position
    effects`` (every plate row and column against its plate median),
    ``library coverage`` (the Lorenz curve of per-gRNA read totals, with
    its Gini) and ``read fate`` (what fraction of reads mapped).

    :param counts: normalised table from :func:`load_count_table`.
    :param per_well: output of :func:`reads_per_well`.
    :param starved: output of :func:`starved_wells`.
    :param positions: output of :func:`position_effects`.
    :param depth: output of :func:`library_depth`.
    :param unmapped: optional output of :func:`unmapped_read_fractions`.
    :param dst: folder to write ``barcode_qc.pdf`` into; ``None`` returns
        the figure without saving.
    :returns: the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1 — reads per well.
    ax = axes[0][0]
    reads = per_well["reads"].to_numpy(float)
    bins = min(40, max(5, int(np.sqrt(max(reads.size, 1)))))
    ax.hist(reads, bins=bins, color=(0 / 255, 155 / 255, 155 / 255),
            alpha=0.85)
    cutoff = starved.attrs.get("cutoff")
    if cutoff:
        ax.axvline(cutoff, color="black", ls="--", lw=1.2,
                   label=f"starved below {cutoff:,.0f} reads "
                         f"({len(starved)} well(s))")
        ax.legend(fontsize=8)
    ax.set_xlabel("reads per well")
    ax.set_ylabel("wells")
    ax.set_title(f"Read depth across {len(per_well)} wells "
                 f"({counts['count'].sum():,.0f} mapped reads)")

    # 2 — position effects.
    ax = axes[0][1]
    if positions.empty:
        ax.set_axis_off()
    else:
        # Natural order, so c2 sits between c1 and c10 rather than after
        # c12 — a position-effect panel whose columns are out of plate
        # order cannot be read against the plate.
        def _natural(value):
            text = str(value)
            digits = "".join(ch for ch in text if ch.isdigit())
            return ("".join(ch for ch in text if not ch.isdigit()),
                    int(digits) if digits else 0)

        ordered = positions.assign(
            _sort=[(a, *_natural(b))
                   for a, b in zip(positions["axis"], positions["label"])]
        ).sort_values("_sort").drop(columns="_sort")
        colors = [(180 / 255, 40 / 255, 60 / 255) if flag
                  else (120 / 255, 120 / 255, 120 / 255)
                  for flag in ordered["flagged"]]
        # The axis initial is only worth a prefix when the label does not
        # already carry it.
        labels = [str(b) if str(b).lower().startswith(a[0])
                  else f"{a[0]}:{b}"
                  for a, b in zip(ordered["axis"], ordered["label"])]
        ax.bar(range(len(ordered)), ordered["ratio_to_plate"], color=colors)
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("median reads / plate median")
        ax.set_title(
            f"Row and column position effects "
            f"({int(ordered['flagged'].sum())} flagged)")

    # 3 — library coverage, as a Lorenz curve.
    ax = axes[1][0]
    values = np.sort(depth["reads_per_grna"].to_numpy(float))
    if values.size and values.sum() > 0:
        share = np.concatenate([[0.0], np.cumsum(values) / values.sum()])
        x = np.linspace(0, 1, share.size)
        ax.plot(x, share, color=(0 / 255, 155 / 255, 155 / 255), lw=2)
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
        ax.set_xlabel("gRNAs, least abundant first")
        ax.set_ylabel("cumulative share of reads")
        dropout = depth.get("dropout_fraction")
        title = (f"Library coverage — Gini {depth['gini']:.2f}, "
                 f"skew {depth['skew_ratio']:.1f}x")
        if dropout is not None:
            title += f", {100 * dropout:.1f}% never seen"
        ax.set_title(title, fontsize=10)

    # 4 — read fate.
    ax = axes[1][1]
    if unmapped:
        names = list(unmapped["per_field"])
        values = [100 * unmapped["per_field"][n] for n in names]
        if "unmapped_fraction" in unmapped:
            names = names + ["any field"]
            values = values + [100 * unmapped["unmapped_fraction"]]
        ax.bar(names, values, color=(180 / 255, 40 / 255, 60 / 255))
        ax.set_ylabel("% of regex-matched reads unmapped")
        ax.set_title(
            f"Unmapped reads (of {unmapped['total_reads']:,.0f} matched)",
            fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    _save_figure(fig, dst, "barcode_qc")
    return fig


# ---------------------------------------------------------------------------
# Settings and entry point
# ---------------------------------------------------------------------------

def barcode_qc_defaults(settings=None) -> Dict[str, Any]:
    """Return the default settings for :func:`barcode_qc`.

    :param settings: optional dict to fill in place; a new one is made
        when omitted.
    :returns: the settings dict with defaults applied.
    """
    settings = dict(settings or {})
    settings.setdefault("count_data", "path to unique_combinations.csv")
    settings.setdefault("qc_data", "")
    settings.setdefault("grna_csv", "")
    settings.setdefault("row_csv", "")
    settings.setdefault("column_csv", "")
    settings.setdefault("target_grnas_per_well", 5)
    settings.setdefault("target_statistic", "median")
    settings.setdefault("min_reads_per_well", 0)
    settings.setdefault("starved_read_fraction", DEFAULT_STARVED_READ_FRACTION)
    settings.setdefault("exclude_starved_wells", True)
    settings.setdefault("position_effect_ratio", DEFAULT_POSITION_RATIO)
    settings.setdefault("collision_max_distance", 1)
    settings.setdefault("sweep_span", DEFAULT_SWEEP_SPAN)
    settings.setdefault("sweep_points", DEFAULT_SWEEP_POINTS)
    settings.setdefault("dst", "")
    settings.setdefault("plot", True)
    settings.setdefault("save", True)
    settings.setdefault("verbose", True)
    return settings


#: Tooltips for the keys this module introduces. Keys it shares with other
#: modules (``count_data``, ``grna_csv``, ``row_csv``, ``column_csv``,
#: ``plot``, ``save``, ``verbose``) are deliberately absent: the registry
#: refuses to let one module rewrite another's help text, and those
#: entries already say the right thing.
_TOOLTIPS: Dict[str, str] = {
    "qc_data": (
        "(str or list) - Path(s) to the qc.csv a barcode-mapping run wrote "
        "beside its count table. Supplies the unmapped-read panel: how many "
        "reads reached barcode lookup and how many of them matched no entry "
        "in each reference. Leave empty to skip that panel; every other panel "
        "works from count_data alone. Default ''."),
    "target_grnas_per_well": (
        "(int) - How many gRNAs a well is meant to carry. This is the "
        "biological target that replaces picking an abundance cutoff by eye: "
        "spaCR solves for the read-fraction threshold that delivers it in "
        "THIS run's data and prints the number it derived, then sweeps around "
        "it so the trade-off is visible. Raise it for statistical power "
        "(more guides per well, more wells kept), lower it for "
        "attributability (a phenotype traceable to fewer guides). A well "
        "holding more than this is counted as a collision. Default 5."),
    "target_statistic": (
        "(str) - Whether target_grnas_per_well is a 'median' or a 'mean' over "
        "wells. Median is the default because a handful of wells that soaked "
        "up the whole library drag a mean far off the typical well. Default "
        "'median'."),
    "min_reads_per_well": (
        "(int) - Absolute read floor below which a well is called starved and "
        "left out of the threshold fit. 0 derives one from the run instead, "
        "as starved_read_fraction of the median well's depth. Starved wells "
        "are always reported either way. Default 0."),
    "starved_read_fraction": (
        "(float) - Share of the median well's read total used as the "
        "starvation cut when min_reads_per_well is 0. A well at a tenth of "
        "typical depth turns single stray reads into 10% abundances, which is "
        "why 0.1 is the default. Ignored when min_reads_per_well is set."),
    "exclude_starved_wells": (
        "(bool) - Leave starved wells out of the population the threshold is "
        "derived and swept over. They report one gRNA at any cutoff and pull "
        "the target down onto a threshold the healthy wells never needed. "
        "They stay in the QC panels regardless. Default True."),
    "position_effect_ratio": (
        "(float) - Fold-change from its plate's median read depth at which a "
        "plate row or column is flagged as a position effect. 2.0 flags a row "
        "at half or double the plate. Must be above 1. Default 2.0."),
    "collision_max_distance": (
        "(int) - How many substituted bases still count as a barcode "
        "collision. 1 catches the pairs a single miscalled base can turn into "
        "each other, which is the common event; 0 reports only exact "
        "duplicates. Above 1 the search is pairwise and slow on a full gRNA "
        "library. Default 1."),
    "sweep_span": (
        "(float) - How far either side of the derived threshold the sweep "
        "runs, as a multiplicative factor. 4.0 sweeps a quarter to four times "
        "the derived value. Must be above 1. Default 4.0."),
    "sweep_points": (
        "(int) - Log-spaced points on the sweep, before the derived threshold "
        "is added to them. Must be at least 3. Default 25."),
    "dst": (
        "(str) - Folder for the QC figures, tables and written "
        "recommendation. Empty writes a 'barcode_qc' folder beside the first "
        "count_data file. Default ''."),
}

_EXPECTED_TYPES: Dict[str, Any] = {
    "qc_data": (str, list),
    "target_grnas_per_well": int,
    "target_statistic": str,
    "min_reads_per_well": int,
    "starved_read_fraction": float,
    "exclude_starved_wells": bool,
    "position_effect_ratio": float,
    "collision_max_distance": int,
    "sweep_span": float,
    "sweep_points": int,
    "dst": str,
}

_DESCRIPTION = (
    "QC a barcode-mapping run — reads per well, starved wells, unmapped "
    "reads, barcode collisions, row/column position effects and library "
    "coverage — then state how many gRNAs per well the design intends and "
    "let spaCR derive the abundance threshold that delivers it, sweeping "
    "around it so the power/attributability trade-off is visible before the "
    "number is chosen."
)


def _register() -> None:
    """Register this module's settings through the defaults seam.

    Guarded so an ``importlib.reload`` of this module in a test session is
    not a duplicate registration; a genuine second claimant on the key
    would be a different module, and that still raises.
    """
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, barcode_qc_defaults,
                      expected_types=_EXPECTED_TYPES,
                      tooltips=_TOOLTIPS,
                      description=_DESCRIPTION)


_register()


def _reference_map(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``{label: path}`` barcode references present in settings."""
    references = {}
    for label, key in (("row", "row_csv"), ("column", "column_csv"),
                       ("grna", "grna_csv")):
        path = settings.get(key)
        if path and os.path.isfile(str(path)):
            references[label] = str(path)
    return references


def barcode_qc(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """QC a barcode-mapping run and derive its abundance threshold from a target.

    Runs every panel this module provides over one run's outputs, derives
    the threshold that delivers ``target_grnas_per_well``, sweeps around
    it, writes the figures and tables, and returns everything it computed
    along with a recommendation in words.

    :param settings: dict; see :func:`barcode_qc_defaults` for every key
        and :data:`_TOOLTIPS` for what each one does. The two that matter
        are ``count_data`` (the run's ``unique_combinations.csv``) and
        ``target_grnas_per_well``.
    :returns: dict with ``choice`` (:class:`ThresholdChoice`),
        ``threshold`` (the derived number), ``sweep``, ``recommendation``,
        ``per_well``, ``starved``, ``positions``, ``depth``,
        ``collisions``, ``collision_summary``, ``unmapped`` and ``dst``.
    :raises ValueError: from :func:`load_count_table` or
        :func:`derive_threshold` on unusable inputs.

    Example:
        .. code-block:: python

            from spacr.sequencing_qc import barcode_qc
            out = barcode_qc({'count_data': 'unique_combinations.csv',
                              'target_grnas_per_well': 4})
            print(out['threshold'], out['recommendation'], sep='\\n')
    """
    settings = barcode_qc_defaults(settings)
    counts = load_count_table(settings["count_data"])

    dst = str(settings.get("dst") or "")
    if not dst:
        first = settings["count_data"]
        if isinstance(first, (list, tuple)):
            first = first[0]
        base = (os.path.dirname(str(first))
                if not isinstance(first, pd.DataFrame) else os.getcwd())
        dst = os.path.join(base or os.getcwd(), "barcode_qc")

    per_well = reads_per_well(counts)
    starved = starved_wells(per_well, int(settings["min_reads_per_well"]),
                            float(settings["starved_read_fraction"]))
    positions = position_effects(counts,
                                 float(settings["position_effect_ratio"]))

    references = _reference_map(settings)
    expected = None
    if "grna" in references:
        expected = list(_read_reference(references["grna"]))
    depth = library_depth(counts, expected)

    collisions = (barcode_collisions(references,
                                     int(settings["collision_max_distance"]))
                  if references else pd.DataFrame(
                      columns=["reference", "name_a", "name_b", "distance",
                               "sequence_a", "sequence_b"]))
    summary = (collision_summary(references, collisions, counts)
               if references else pd.DataFrame())

    unmapped = None
    if settings.get("qc_data"):
        unmapped = unmapped_read_fractions(settings["qc_data"], counts)

    population = None
    if settings.get("exclude_starved_wells", True) and not starved.empty:
        keep = set(per_well["prc"]) - set(starved["prc"])
        # Never hand an empty population to the derivation: when every
        # well is below the cut the run is starved as a whole, and the
        # honest answer is to fit on what there is and say so in the QC.
        population = keep or None

    choice = derive_threshold(counts, float(settings["target_grnas_per_well"]),
                              str(settings["target_statistic"]),
                              wells=population)
    # Reach down into the bleed-through tail even when the derived
    # threshold sits far above it, so the collision knee is on the curve.
    # Floored at a thousandth of the derived value: a run with one
    # freakishly small fraction should not stretch the plot over six
    # decades of empty space.
    tail = float(np.quantile(counts["fraction"].to_numpy(float), 0.01))
    grid = sweep_grid(choice.threshold, float(settings["sweep_span"]),
                      int(settings["sweep_points"]),
                      low=max(tail, choice.threshold / 1e3),
                      # ...and up past the top of the plateau, so the cost
                      # of tightening is on the curve too rather than
                      # sitting just off the right-hand edge of it.
                      high=choice.interval_high * 1.5)
    sweep = threshold_sweep(counts, grid,
                            float(settings["target_grnas_per_well"]),
                            str(settings["target_statistic"]),
                            wells=population)
    recommendation = recommend_threshold(sweep, choice)

    if settings.get("verbose", True):
        print(recommendation)
        if not starved.empty:
            print(f"Starved wells (< {starved.attrs['cutoff']:,.0f} reads): "
                  f"{len(starved)} of {len(per_well)} — "
                  f"{', '.join(starved['prc'].astype(str).head(10))}")
        flagged = positions[positions["flagged"]] if not positions.empty \
            else positions
        if not flagged.empty:
            print(f"Position effects flagged: "
                  + ", ".join(f"{r.plateID} {r.axis} {r.label} "
                              f"({r.ratio_to_plate:.2f}x)"
                              for r in flagged.itertuples()))
        if not collisions.empty:
            print(f"Barcode collisions: {len(collisions)} pair(s) within "
                  f"{settings['collision_max_distance']} substitution(s).")
        if unmapped and "unmapped_fraction" in unmapped:
            print(f"Unmapped reads: "
                  f"{100 * unmapped['unmapped_fraction']:.2f}% of "
                  f"{unmapped['total_reads']:,.0f} regex-matched reads.")

    figure_dst = dst if settings.get("save", True) else None
    if settings.get("save", True):
        os.makedirs(dst, exist_ok=True)
        sweep.to_csv(os.path.join(dst, "threshold_sweep.csv"), index=False)
        per_well.to_csv(os.path.join(dst, "reads_per_well.csv"), index=False)
        starved.to_csv(os.path.join(dst, "starved_wells.csv"), index=False)
        positions.to_csv(os.path.join(dst, "position_effects.csv"),
                         index=False)
        if not collisions.empty:
            collisions.to_csv(os.path.join(dst, "barcode_collisions.csv"),
                              index=False)
        if not summary.empty:
            summary.to_csv(os.path.join(dst, "collision_summary.csv"),
                           index=False)
        with open(os.path.join(dst, "threshold_recommendation.txt"), "w") as f:
            f.write(recommendation + "\n")

    if settings.get("plot", True):
        import matplotlib.pyplot as plt

        figure = plot_threshold_sweep(sweep, choice, figure_dst)
        panels = plot_barcode_qc(counts, per_well=per_well, starved=starved,
                                 positions=positions, depth=depth,
                                 unmapped=unmapped, dst=figure_dst)
        # Closed rather than shown: this runs inside a mapping pipeline and
        # inside the Qt worker thread, neither of which owns a GUI event
        # loop, and an accumulating figure stack is a memory leak over a
        # plate's worth of samples.
        plt.close(figure)
        plt.close(panels)

    return {
        "choice": choice,
        "threshold": choice.threshold,
        "sweep": sweep,
        "recommendation": recommendation,
        "counts": counts,
        "per_well": per_well,
        "starved": starved,
        "positions": positions,
        "depth": depth,
        "collisions": collisions,
        "collision_summary": summary,
        "unmapped": unmapped,
        "dst": dst,
    }
