"""Every other channel, sampled at the coordinates the numbering assigned.

Sampling every other channel at the object coordinates the numbering step
assigned -- the part that can be built and checked without a real cycle stack.
The numbering step produced a plate-level object id and a well-frame centroid
for every nucleus; this is what reads the remaining channels at those
coordinates and keys the result to those ids.

    C1  SEQUENCING CHANNELS, PER CYCLE ... NEVER averaged across cycles --
        that is the one operation that destroys a barcode while leaving it
        decodable.

This constraint is enforced in code. A mean across cycles of a four-channel
readout still looks like a four-channel readout: ``call_reads`` accepts it,
returns a barcode of the right length, and every downstream table fills in. The
run completes, the numbers are plausible, and the barcodes are noise. There is
no later check that can catch it, because nothing downstream knows what the
per-cycle values were. So :func:`decode_input` refuses an array whose cycle
axis has been collapsed, and this module offers no way to average one.

    The phenotype path is the opposite case, which is why the two are separate
    functions rather than one with a flag. A phenotype channel measured in
    several cycles is several samples of one quantity and averaging it is the
    right thing, and what it emits is the schema ``measure`` already emits.
    Same plate, same objects, opposite correct answer; a shared flag would make
    that a caller's choice, and it is not one.

The samplers are injected, as they are in :mod:`spacr.ops_compose` and
:mod:`spacr.ops_objects`: nothing here opens a file or knows a transform's
provenance, so all of it is testable on planted truth.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.ops_sample")

__all__ = [
    "SampleError",
    "sample_objects", "decode_input", "average_over_cycles",
    "reads_rows", "barcode_rows",
    "pair_nuclei_with_cells", "measure_phenotype_field",
    "funk_object_features",
]


class SampleError(ValueError):
    """Raised for an invalid sampling; the message states the remedy."""


def sample_objects(objects: Sequence, image: np.ndarray, *,
                   transform: Optional[Callable[[float, float],
                                                Tuple[float, float]]] = None,
                   radius: int = 1) -> np.ndarray:
    """Read ``image`` at each object's centroid, through ``transform``.

    :param objects: :class:`spacr.ops_objects.PlateObject` values, or anything
        with ``centroid_y`` and ``centroid_x`` in the well frame.
    :param image: one channel of one cycle, in that cycle's own frame.
    :param transform: maps a well-frame ``(y, x)`` into this image's frame.
        ``None`` means the image is already in the well frame.
    :param radius: half-width of the square averaged around each point. The
        default 1 gives a 3x3, which is what a diffraction-limited spot
        occupies; 0 reads the single pixel.
    :returns: one float per object, NaN where the point falls outside.

    NaN rather than a clamped edge pixel: an object whose coordinates land off
    this cycle's field was not measured in it, and the brightness of the
    nearest border pixel would be averaged into a barcode as if it were a
    measurement.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise SampleError(
            f"a channel is one 2-D image; got shape {image.shape}. Pass one "
            f"cycle's one channel, not a stack -- the axes are what keeps "
            f"cycles separate.")
    height, width = image.shape
    radius = max(0, int(radius))

    out = np.full(len(objects), np.nan, dtype=float)
    for index, one in enumerate(objects):
        y, x = float(one.centroid_y), float(one.centroid_x)
        if transform is not None:
            y, x = transform(y, x)
        row, column = int(round(y)), int(round(x))
        if not (0 <= row < height and 0 <= column < width):
            continue
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        left, right = max(0, column - radius), min(width, column + radius + 1)
        patch = image[top:bottom, left:right]
        if patch.size:
            out[index] = float(patch.mean())
    return out


def decode_input(per_cycle: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Stack per-cycle, per-channel samples into ``call_reads``' input.

    :param per_cycle: ``per_cycle[c][k]`` is :func:`sample_objects`' output
        for cycle ``c``, channel ``k``. Every entry must be the same length.
    :returns: ``(objects, cycles, channels)`` -- the shape
        :func:`spacr.ops_sbs.call_reads` expects.
    :raises SampleError: when a cycle is missing channels, when the lengths
        disagree, or -- the one that matters -- when only one cycle is offered,
        which is what a caller who has already averaged them hands over.

    One cycle is refused. A barcode is a sequence across cycles; a single
    "cycle" reaching here is either a run that lost its cycle axis or a caller
    who collapsed it. Both produce a decodable barcode of noise, and neither is
    visible downstream. If a single-cycle experiment ever needs decoding it
    should say so explicitly rather than arrive looking like this mistake.
    """
    cycles = [list(one) for one in per_cycle]
    if not cycles:
        raise SampleError(
            "no cycles were sampled; a barcode needs one letter per cycle")
    if len(cycles) == 1:
        raise SampleError(
            "only one cycle reached the decoder. A barcode is a sequence "
            "ACROSS cycles, so this is either a lost cycle axis or an "
            "average taken over them -- and 372 names that as 'the one "
            "operation that destroys a barcode while leaving it decodable'. "
            "The call would succeed and the barcodes would be noise.")

    widths = {len(one) for one in cycles}
    if len(widths) != 1:
        raise SampleError(
            f"the cycles offer different channel counts ({sorted(widths)}); "
            f"a base call compares the same channels in every cycle")

    lengths = {len(channel) for one in cycles for channel in one}
    if len(lengths) != 1:
        raise SampleError(
            f"the samples have different lengths ({sorted(lengths)}); every "
            f"cycle and channel must describe the same objects in the same "
            f"order, because the object id is the join key")

    return np.stack([np.stack(one, axis=-1) for one in cycles], axis=1)


def average_over_cycles(per_cycle: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of one phenotype channel across cycles.

    Phenotype channels are sampled "at the same object ids", and a phenotype
    measured in several cycles is several samples of one quantity -- so
    averaging raises its precision, exactly as averaging the Hoechst across
    tiles does when they overlap.

    Separate from :func:`decode_input`: the same operation is right for a
    phenotype channel and destroys a sequencing one, so it is two functions and
    not one with a flag: a flag would make that a caller's decision, and it is
    a property of the channel.

    NaN-aware, because :func:`sample_objects` returns NaN for an object this
    cycle did not cover, and an object measured in nine cycles of eleven
    should get the mean of the nine rather than NaN.

    :param per_cycle: one array of per-object values per cycle, every array
        the same length and in the same object order. The order is the
        caller's to keep: nothing here can detect two cycles sampled against
        different object tables.
    """
    stack = np.stack([np.asarray(one, dtype=float) for one in per_cycle])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def reads_rows(objects: Sequence, values: np.ndarray, *,
               channels: Sequence[str],
               bases: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """``ops_reads`` rows: one per object per cycle, per the storage contract's columns.

    :param objects: the numbered objects the values were sampled at, in the
        same order as ``values``' first axis.
    :param values: ``(objects, cycles, channels)`` from :func:`decode_input`.
    :param channels: a name per channel, for the column names.
    :param bases: the letter each channel votes for; when given, each row
        carries the base this cycle called and its margin.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise SampleError(
            f"ops_reads needs (objects, cycles, channels); got {values.shape}")
    if values.shape[0] != len(objects):
        raise SampleError(
            f"{values.shape[0]} sampled rows against {len(objects)} objects; "
            f"the object id is the join key and cannot be inferred")
    if values.shape[2] != len(channels):
        raise SampleError(
            f"{values.shape[2]} channels sampled but {len(channels)} names")

    rows: List[Dict[str, object]] = []
    for index, one in enumerate(objects):
        for cycle in range(values.shape[1]):
            intensities = values[index, cycle]
            row: Dict[str, object] = {
                "object_id": int(one.object_id),
                "cycle": int(cycle + 1),
            }
            for name, value in zip(channels, intensities):
                row[str(name)] = float(value)
            if bases is not None and np.isfinite(intensities).all():
                order = np.argsort(intensities)
                best, second = intensities[order[-1]], intensities[order[-2]]
                total = best + second
                row["base"] = str(bases[int(order[-1])])
                row["quality"] = float((best - second) / total) if total > 0 else 0.0
            rows.append(row)
    return rows


def barcode_rows(objects: Sequence, values: np.ndarray, *,
                 bases: Optional[Sequence[str]] = None,
                 library: Optional[Sequence[str]] = None,
                 ) -> List[Dict[str, object]]:
    """``ops_barcodes`` rows: one per object, assembled across cycles.

    Delegates the call itself to :func:`spacr.ops_sbs.call_reads`, which
    already normalises each cycle's channels and reports quality as the minimum
    over cycles -- "a barcode is only as trustworthy as its worst base".
    Repeating that here would be a second decoder to keep in step.

    :param objects: the numbered objects, in the same order as ``values``'
        first axis.
    :param values: ``(objects, cycles, channels)`` intensities, never averaged
        over cycles -- see :func:`decode_input`, which refuses a single cycle
        for the same reason.
    :param library: when given, each barcode is corrected to the nearest
        library member through :func:`spacr.ops_sbs.correct_to_library`, and
        the row carries what it mapped to.
    """
    from .ops_sbs import call_reads

    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise SampleError(
            f"ops_barcodes needs (objects, cycles, channels); got {values.shape}")

    kwargs = {"bases": bases} if bases is not None else {}
    barcodes, quality = call_reads(values, **kwargs)

    mapped: Optional[Sequence] = None
    if library is not None:
        from .ops_sbs import correct_to_library

        mapped = correct_to_library(barcodes, library)

    rows: List[Dict[str, object]] = []
    for index, one in enumerate(objects):
        row: Dict[str, object] = {
            "object_id": int(one.object_id),
            "barcode": str(barcodes[index]),
            "quality": float(quality[index]),
            "n_cycles": int(values.shape[1]),
        }
        if mapped is not None:
            entry = mapped[index]
            row["mapped_guide"] = "" if entry is None else str(entry)
        rows.append(row)
    return rows


def pair_nuclei_with_cells(nuclei: np.ndarray, cells: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """One nucleus per cell, both carrying the same label.

    :param nuclei: a 2-D nucleus label mask.
    :param cells: a 2-D cell label mask of the same field.
    :returns: ``(nuclei, cells, counts)``: both relabelled ``1..K`` so that
        nucleus ``k`` lies inside cell ``k``, and how many objects were
        dropped and why.

    Two independent segmentations of one field disagree at the edges: a
    nucleus straddles two cells, a cell holds two nuclei, a nucleus has no
    cell at all. spaCR's measurement joins a nucleus to one parent cell and
    refuses a field where that is ambiguous, so the ambiguity is resolved
    here: each nucleus goes to the cell it overlaps most and is clipped to
    that cell; a cell keeps its largest such nucleus; cells without a
    nucleus and nuclei without a cell are dropped and counted.
    """
    return _pair_nuclei_with_cells(nuclei, cells)[:3]


def _pair_nuclei_with_cells(nuclei, cells, pairing=None, dna=None,
                            spanning="merge"):
    """:func:`pair_nuclei_with_cells` with the item-1024 alpha options and
    the spanning-nucleus rule chosen.

    :param nuclei: a 2-D nucleus label mask.
    :param cells: a 2-D cell label mask of the same field.
    :param pairing: :func:`spacr.measure._nucleus_pairing_options` under the
        Measure names, or None for the pairing of
        :func:`pair_nuclei_with_cells`. The nuclei go to cells in the one
        shared :func:`spacr.measure._assign_nuclei_to_cells`, so OPS and
        Measure pair the same way.
    :param dna: the field's nucleus stain for the dip of the fragment
        test, or None.
    :param spanning: ``ops_ph_spanning_merge``, passed to
        :func:`spacr.measure._assign_nuclei_to_cells`. ``'merge'`` (the
        default, and what :func:`pair_nuclei_with_cells` uses) first applies
        Measure's ownership rule: a nucleus split under 90 % between cells
        merges them, and at 90 % or more its pixels in the other cells are
        removed; ``counts['merged_cells']`` counts the merges.
        ``'best_parent'`` skips that rule, so every nucleus goes to the cell
        it overlaps most and no cell is merged: masks and counts are those
        of the OPS pairing before settings schema 959 (commit 26bc722c15),
        with ``merged_cells`` 0. A cell touched only by a piece of a nucleus
        that belongs to another cell still counts as touched there, as it
        did then.
    :returns: ``(nuclei, cells, counts, record)``. ``record`` is None
        without an active option; otherwise ``fragments``, one
        ``(label, kept_label, merged_label)`` per merged piece, ``label``
        the output key and the other two the input labels (the merged
        nucleus keeps the lowest piece label into the pairing),
        ``second_nuclei``, one ``(label, cell_label)`` per further nucleus
        kept in a binucleate cell, and ``nucleus_only``, the output keys of
        nuclei kept without a cell.

    The keys: the ``K`` paired nuclei keep ``1..K``, nucleus ``k`` in cell
    ``k``, exactly as without the options (a cell's paired nucleus is its
    largest). With ``nucleus_fragment_merge`` ``'rule'``, the further
    nuclei of a binucleate cell (pieces that failed the fragment test) are
    kept too, with the keys ``K + 1`` onwards in the order of their cell's
    key; no cell carries such a key, so the row of that nucleus has empty
    cell and cytoplasm columns and names its cell in ``cell_label``
    (:func:`spacr.ops_engine._ph_pairing_columns`). With
    ``cellless_nuclei`` ``'nucleus_only'``, a nucleus without a cell keeps
    a key after those, in the same nucleus key space, with empty cell and
    cytoplasm columns. ``counts`` then also holds ``merged_fragments``,
    ``binucleate_cells``, ``binucleate_kept``, ``attached_cellless`` and
    ``nucleus_only``.
    """
    from .measure import (_assign_nuclei_to_cells, _best_parent,
                          _label_overlaps, _nucleus_pairing_active,
                          _spanning_mode)

    spanning = _spanning_mode(spanning)
    nuclei = np.asarray(nuclei)
    cells = np.asarray(cells)
    if nuclei.shape != cells.shape:
        raise SampleError(
            f"nucleus and cell masks differ in shape: {nuclei.shape} and "
            f"{cells.shape}")
    active = _nucleus_pairing_active(pairing)
    record = {} if active else None
    before = int(np.unique(cells[cells > 0]).size)
    nuclei, cells, _shared_n, shared_c, merged = _assign_nuclei_to_cells(
        nuclei.copy(), cells.copy(), options=pairing if active else None,
        dna=dna if active else None, record=record,
        spanning=spanning)
    n_total = int(np.unique(nuclei[nuclei > 0]).size)
    c_total = int(np.unique(cells[cells > 0]).size)
    pair_n, pair_c, overlap = _label_overlaps(nuclei, cells)
    touched = np.unique(pair_c)
    if spanning == "best_parent" and shared_c:
        touched = np.union1d(touched, np.fromiter(shared_c, np.int64))
    keep_only = (record is not None
                 and pairing.get("cellless_nuclei") == "nucleus_only")
    alone = np.asarray(record["cellless"] if keep_only else [], np.int64)
    if not pair_n.size and not alone.size:
        counts = {"nuclei": n_total, "cells": before, "paired": 0,
                  "nuclei_without_cell": n_total,
                  "cells_without_nucleus": c_total,
                  "extra_nuclei_in_cell": 0, "merged_cells": merged}
        if record is not None:
            counts.update(_pairing_counts(record, 0, 0))
            record = {"fragments": [], "second_nuclei": [], "nucleus_only": []}
        return (np.zeros(nuclei.shape, np.int32),
                np.zeros(cells.shape, np.int32), counts, record)
    best_n, best_c, best_o = _best_parent(pair_n, pair_c, overlap)
    keep_c, keep_n, _ = _best_parent(best_c, best_n, best_o)
    labels = np.arange(1, keep_n.size + 1, dtype=np.int32)
    n_lut = np.zeros(int(nuclei.max()) + 1, np.int32)
    c_lut = np.zeros(int(cells.max()) + 1, np.int32)
    n_lut[keep_n] = labels
    c_lut[keep_c] = labels
    paired = int(keep_n.size)
    second_n = np.zeros(0, np.int64)
    homes = np.zeros(0, np.int32)
    if record is not None and pairing.get("nucleus_fragment_merge") == "rule":
        extra = ~np.isin(best_n, keep_n)
        homes = c_lut[best_c[extra]]
        order = np.lexsort((best_n[extra], homes))
        second_n, homes = best_n[extra][order], homes[order]
        n_lut[second_n] = np.arange(paired + 1, paired + second_n.size + 1,
                                    dtype=np.int32)
    kept = paired + int(second_n.size)
    if alone.size:
        n_lut[alone] = np.arange(kept + 1, kept + alone.size + 1,
                                 dtype=np.int32)
    new_cells = c_lut[cells.astype(np.int64)]
    new_nuclei = n_lut[nuclei.astype(np.int64)]
    clip = new_nuclei != new_cells
    if second_n.size:
        home_of = np.zeros(kept + 1, np.int32)
        home_of[paired + 1:] = homes
        further = (new_nuclei > paired) & (new_nuclei <= kept)
        clip[further] = new_cells[further] != home_of[new_nuclei[further]]
    if alone.size:
        clip &= new_nuclei <= kept
    new_nuclei[clip] = 0
    counts = {
        "nuclei": n_total, "cells": before, "paired": int(keep_n.size),
        "nuclei_without_cell": int(n_total - np.unique(pair_n).size),
        "cells_without_nucleus": int(c_total - touched.size),
        "extra_nuclei_in_cell": int(best_n.size - keep_n.size),
        "merged_cells": merged,
    }
    if record is not None:
        counts.update(_pairing_counts(record, int(alone.size),
                                      int(second_n.size)))
        record = {
            "fragments": [(int(n_lut[label]), int(label), int(piece))
                          for label, piece in record["fragments"]
                          if n_lut[label]],
            "second_nuclei": [(int(n_lut[label]), int(home))
                              for label, home in zip(second_n, homes)],
            "nucleus_only": [int(v) for v in n_lut[alone]]}
    return new_nuclei, new_cells, counts, record


def _pairing_counts(record, nucleus_only, second_nuclei):
    """The item-1024 counts of one field from the record of the shared pairing."""
    return {"merged_fragments": len(record["fragments"]),
            "binucleate_cells": len(record["binucleate"]),
            "binucleate_kept": int(second_nuclei),
            "attached_cellless": len(record["attached"]),
            "nucleus_only": int(nucleus_only)}


def _measurement_settings(channels: int, texture: bool,
                          gpu: bool = False,
                          profile_features=None,
                          granularity_subsample: float = 1.0,
                          funk_features: bool = False,
                          outside_rule: str = "object",
                          cytoskeleton_channels=None,
                          channel_names=None,
                          zernike: bool = False,
                          summarize: bool = False
                          ) -> Dict[str, object]:
    """The measure settings one phenotype field is measured with.

    :param channels: how many channels the image holds.
    :param texture: compute the GLCM homogeneity texture block.
    :param gpu: run the per-object intensity and texture tables on the CUDA
        card (``measure_gpu``), which computes the same columns within float
        tolerance; morphology stays on the CPU.
    :param profile_features: the measure setting ``profile_features``
        (:func:`spacr.measure.profile_feature_families`); with
        ``'colocalisation'`` among them the per-object Pearson and Manders
        block (``calculate_correlation``) is on too, as before. The blocks of
        :data:`spacr.measure.MEASURE_BLOCK_FAMILIES` it names are switched on
        by :func:`spacr.measure._intensity_measurements` itself, as in
        Measure: ``coloc_block`` alone gives that Pearson and Manders block
        without the slow ``colocalisation`` family.
    :param granularity_subsample: ``profile_granularity_subsample``, the
        ``subsample`` of :func:`spacr.measure.measure_granularity`.
    :param outside_rule: ``profile_outside_rule``, the ring of the
        ``outside`` family (:func:`spacr.measure._outside_excludes_foreground`).
    :param funk_features: ``funk_object_features``: Funk et al. 2022's
        object features from :func:`spacr.measure._add_funk_frames`, for
        every compartment, as Measure writes them.
    :param cytoskeleton_channels: channel indices or names for the cell-only
        cytoskeleton block; None measures every channel when selected.
    :param channel_names: image channel names for resolving that selection
        and for naming the columns (``cell_dapi_mean_intensity``).
    :param zernike: ``zernike``, the Zernike moments of the cell and nucleus.
    :param summarize: ``summarize_organelles_by='cell'``, the per-cell
        summary of further objects (item 970).
    :returns: the settings :func:`spacr.measure._measure_field_objects`
        reads.
    """
    from .measure import profile_feature_families
    from .ops_settings import OPS_TO_MEASURE_KEYS

    families = profile_feature_families(profile_features)
    ops = {"ops_ph_texture": bool(texture), "ops_ph_measure_gpu": bool(gpu),
           "ops_ph_funk_features": bool(funk_features),
           "ops_ph_zernike": bool(zernike),
           "ops_ph_profile_features": list(families) or None,
           "ops_ph_granularity_subsample": float(granularity_subsample),
           "ops_ph_profile_outside_rule": str(outside_rule or "object")}
    return {
        "cell_mask_dim": channels, "nucleus_mask_dim": channels + 1,
        "pathogen_mask_dim": None, "cytoplasm": True,
        "radial_dist": False,
        "cytoskeleton_channels": cytoskeleton_channels,
        "channel_names": channel_names,
        "calculate_correlation": "colocalisation" in families,
        "homogeneity_distances": [2, 4, 8],
        "distance_gaussian_sigma": 10, "spatial_measurements": False,
        "channels": list(range(channels)), "verbose": False,
        "summarize_organelles_by": "cell" if summarize else None,
        **{OPS_TO_MEASURE_KEYS[key]: value for key, value in ops.items()},
    }


def funk_object_features(labels: np.ndarray, image: np.ndarray,
                         channel_names: Sequence[str], prefix: str):
    """Funk et al. 2022's mass displacement and channel correlations.

    :param labels: one compartment's label mask.
    :param image: ``(H, W, C)`` intensities over it.
    :param channel_names: one short name per channel.
    :param prefix: the compartment, ``nucleus`` or ``cell``.
    :returns: one row per label: ``label``,
        ``<prefix>_<ch>_mass_displacement`` per channel and
        ``<prefix>_correlation_<a>_<b>`` per channel pair, in the order of
        :func:`itertools.combinations`.

    These are the definitions of Funk et al.'s OpticalPooledScreens
    ``ops/cp_emulator.py``, which their phenotype step
    (``Snake._extract_phenotype_cp_multichannel``) applied to every nucleus
    and cell:

    - ``mass_displacement`` (``intensity_features_multichannel``): the
      distance in pixels from the object's centroid to its
      intensity-weighted centroid, ``sqrt(sum((local_centroid -
      weighted_local_centroid) ** 2))``. A displaced DNA or tubulin mass
      marks an asymmetric cell.
    - ``correlation`` (``correlation_features_multichannel``, through
      ``ops.features.correlate_channels_all_multichannel``): Pearson's r of
      two channels over the object's pixels, ``np.corrcoef``.

    Both are computed from per-label sums, centred on each label's mean, so
    a field is one pass rather than one regionprops object per cell.
    """
    import pandas as pd

    from .measure import _funk_object_columns

    labels = np.asarray(labels).astype(np.int64)
    image = np.asarray(image, np.float64)
    if not (labels > 0).any():
        return pd.DataFrame({"label": np.zeros(0, np.int64)})
    names = [str(n).lower() for n in channel_names]
    present, columns = _funk_object_columns(labels, image, names)
    out = {"label": present.astype(np.int64)}
    for index, name in enumerate(names):
        out[f"{prefix}_{name}_mass_displacement"] = columns[
            f"channel_{index}_mass_displacement"]
    for column, values in columns.items():
        if column.startswith("correlation_"):
            out[f"{prefix}_{column}"] = values
    return pd.DataFrame(out)


def measure_phenotype_field(nuclei: np.ndarray, cells: np.ndarray,
                            image: np.ndarray, channel_names: Sequence[str],
                            *, texture: bool = True,
                            funk_features: bool = False,
                            zernike: bool = False,
                            gpu: bool = False,
                            profile_features=None,
                            granularity_subsample: float = 1.0,
                            cytoskeleton_channels=None):
    """Per-cell features of one phenotype field, with spaCR's own measurement.

    :param nuclei: nucleus labels, paired with ``cells`` by
        :func:`pair_nuclei_with_cells`, so nucleus ``k`` is in cell ``k``.
    :param cells: cell labels.
    :param image: ``(H, W, C)`` intensities.
    :param channel_names: one short name per channel, used in the column
        names in place of spaCR's ``channel_<i>``.
    :param texture: include the GLCM homogeneity block.
    :param funk_features: add Funk et al. 2022's features: the mass
        displacement of every channel, the correlation of every channel
        pair and the form factor, ``4 * pi * area / perimeter ** 2``
        (``shape_features['form_factor']`` of their ``ops/cp_emulator.py``),
        in the nucleus, the cell and the cytoplasm. They come from
        :func:`spacr.measure._add_funk_frames`, the code Measure's
        ``funk_object_features`` runs, with the names
        :func:`funk_object_features` gives (``<compartment>_<ch>_mass_displacement``,
        ``<compartment>_correlation_<a>_<b>``, ``<compartment>_form_factor``).
    :param zernike: add the Zernike moments of degree 8
        (``zernike_0`` to ``zernike_24``) of the cell and the nucleus, as
        :func:`spacr.measure._calculate_zernike` computes them with mahotas;
        the cytoplasm gets none. Needs ``spacr[zernike]``. Default False.
    :param gpu: measure intensity and texture on the CUDA card
        (:func:`_measurement_settings`).
    :param profile_features: the profiling families spaCR's measure adds
        (:func:`spacr.measure.measure_profile_features`): ``'profiling'``
        for the default set, ``'all'``, or family names. ``None``, the
        default, measures exactly as before.
    :param granularity_subsample: the ``subsample`` of the granularity
        family (:func:`spacr.measure.measure_granularity`); 1.0 sieves at
        full resolution, 0.25 is CellProfiler's default.
    :param cytoskeleton_channels: image channel indices or names for the
        cell-only ``cytoskeleton`` block in ``profile_features``. None
        selects every channel. The block is absent unless selected.
    :returns: one row per cell, keyed by ``label``, with ``nucleus_``,
        ``cell_`` and ``cytoplasm_`` columns and the nucleus centroid as
        ``centroid_row``/``centroid_col`` in field pixels.

    The features are the ones spaCR's measure writes -- morphology from
    :func:`spacr.measure._morphological_measurements`, intensity and texture
    from :func:`spacr.measure._intensity_measurements` -- so an OPS cell and
    a measured well's cell carry the same columns and nothing is
    re-derived. The cytoplasm is the cell minus its nucleus, under the
    cell's label. The object labels the colocalisation block carries
    (``*_label_correlation``) are left out, as spaCR's database write
    leaves them out, so no column repeats the label.
    """
    return _measure_phenotype_field(
        nuclei, cells, image, channel_names, texture=texture,
        funk_features=funk_features, zernike=zernike, gpu=gpu,
        profile_features=profile_features,
        granularity_subsample=granularity_subsample,
        cytoskeleton_channels=cytoskeleton_channels)


def _measure_phenotype_field(nuclei, cells, image, channel_names, *,
                             texture=True, funk_features=False,
                             zernike=False, gpu=False, profile_features=None,
                             granularity_subsample=1.0,
                             cytoskeleton_channels=None, objects=None,
                             outside_rule="object"):
    """:func:`measure_phenotype_field` with the field's further objects.

    :param nuclei: nucleus labels, nucleus ``k`` in cell ``k``.
    :param cells: cell labels.
    :param image: ``(H, W, C)`` intensities.
    :param channel_names: one short name per channel.
    :param texture: as :func:`measure_phenotype_field`.
    :param funk_features: as :func:`measure_phenotype_field`.
    :param zernike: as :func:`measure_phenotype_field`.
    :param gpu: as :func:`measure_phenotype_field`.
    :param profile_features: as :func:`measure_phenotype_field`.
    :param granularity_subsample: as :func:`measure_phenotype_field`.
    :param cytoskeleton_channels: as :func:`measure_phenotype_field`.
    :param outside_rule: ``ops_ph_profile_outside_rule``, the ring of the
        ``outside`` family (:func:`spacr.measure._outside_excludes_foreground`);
        ``'object'``, the default of :func:`measure_phenotype_field`,
        measures the nucleus ring as Measure does.
    :param objects: ``{class: labels}`` of the field's further objects
        (``pathogen``, ``organelle``), or None. Their pixels are not
        cytoplasm: the cytoplasm is Measure's
        (:func:`spacr.measure._derive_cytoplasm`), the cell minus its
        nucleus and every further object, under the cell's label.
    :returns: as :func:`measure_phenotype_field`.
    """
    import pandas as pd

    from .measure import _derive_cytoplasm, _field_role_tables, _label_centroids

    image = np.asarray(image, np.float32)
    nuclei = np.asarray(nuclei, np.int32)
    cells = np.asarray(cells, np.int32)
    if image.ndim != 3 or image.shape[:2] != cells.shape:
        raise SampleError(
            f"the image must be (H, W, C) over the masks' {cells.shape}; "
            f"got {image.shape}")
    if len(channel_names) != image.shape[2]:
        raise SampleError(
            f"{len(channel_names)} channel names for {image.shape[2]} channels")
    cytoplasm = _derive_cytoplasm(cells, [nuclei, *(objects or {}).values()])
    masks = {"cell": cells, "nucleus": nuclei, "cytoplasm": cytoplasm}
    if not nuclei.any():
        return pd.DataFrame(columns=["label", "centroid_row", "centroid_col"])
    settings = _measurement_settings(image.shape[2], texture, gpu,
                                     profile_features, granularity_subsample,
                                     funk_features=funk_features,
                                     cytoskeleton_channels=cytoskeleton_channels,
                                     channel_names=channel_names,
                                     zernike=zernike,
                                     outside_rule=outside_rule)
    measured = _measure_core(masks, image, settings)
    tables = _field_role_tables(measured["morphology"], measured["intensity"],
                               ("cell", "nucleus", "cytoplasm"))
    centres = _label_centroids(nuclei)
    out = pd.DataFrame({"label": centres["label"],
                        "centroid_row": centres["y"],
                        "centroid_col": centres["x"]})
    for prefix in ("cell", "nucleus", "cytoplasm"):
        frame = tables.get(prefix)
        if frame is None:
            continue
        frame = frame.rename(columns={
            column: column if str(column).startswith(f"{prefix}_")
            else f"{prefix}_{column}"
            for column in frame.columns if column != "label"})
        frame = frame.loc[:, [c for c in frame.columns
                              if "pathogen" not in str(c)
                              and not str(c).endswith("_cell_id")
                              and not str(c).endswith("label_correlation")]]
        frame = frame.loc[:, [c for c in frame.columns
                              if c == "label" or c not in out.columns]]
        frame = frame.assign(label=frame["label"].astype(np.int64))
        frame = frame.drop_duplicates(subset=["label"])
        out = out.merge(frame, on="label", how="left")
    keep = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    return out.loc[:, keep].reset_index(drop=True)


def _measure_core(masks, image, settings, *, objects=None, tables=True):
    """One call of :func:`spacr.measure._measure_field_objects` for the OPS field.

    :param masks: the ``cell``, ``nucleus`` and ``cytoplasm`` labels.
    :param image: the ``(H, W, C)`` intensities.
    :param settings: :func:`_measurement_settings`.
    :param objects: ``{class: labels}`` summarised per cell, or None.
    :param tables: False measures only the per-cell object summaries.
    """
    from .measure import _measure_field_objects

    return _measure_field_objects(
        masks, image, settings, periphery=False, outside=False,
        summarize=dict(objects) if objects else {}, tables=tables)


#: The per-cell statistics :func:`spacr.measure._summarize_organelles_per_parent`
#: writes for one object class, in the order :func:`_cell_object_features`
#: gives them, before the two intensity columns of each channel.
_CELL_OBJECT_STATS = (
    "count", "total_area", "fraction", "mean_area", "std_area",
    "mean_eccentricity", "std_eccentricity", "mean_solidity", "std_solidity",
    "mean_major_axis", "mean_minor_axis")


def _cell_object_features(cells: np.ndarray, objects, image: np.ndarray,
                          channel_names: Sequence[str]):
    """Per-cell summaries of the pathogens and organelles in one field (item 970).

    :param cells: cell labels.
    :param objects: ``{class: labels}``, such as ``{'pathogen': ...,
        'organelle': ...}``, each the shape of ``cells``.
    :param image: ``(H, W, C)`` intensities, as the field is measured.
    :param channel_names: one short name per channel.
    :returns: one row per cell label, ``label`` and, for each class,
        ``cell_<class>_`` + count, total area, area fraction of the cell,
        mean and SD area, mean and SD eccentricity and solidity, mean major
        and minor axis length, and ``<channel>_mean_intensity`` and
        ``<channel>_std_intensity``: the mean and SD over the cell's objects
        of each object's mean intensity. A cell without objects has 0
        everywhere.

    Measure's own summary, computed by
    :func:`spacr.measure._measure_field_objects` with
    ``summarize_organelles_by='cell'``
    (:func:`spacr.measure._summarize_organelles_per_parent`): an object
    belongs to the cell that holds most of its pixels
    (:func:`spacr.measure._map_child_to_parent`) and an object outside every
    cell belongs to none. The ``cell_`` prefix makes the columns per-cell
    features of the OPS tables, so the aggregate phase carries them into
    the per-gene profiles as it does the cell's own.
    """
    cells = np.asarray(cells, np.int32)
    image = np.asarray(image, np.float32)
    summary = None
    if (cells > 0).any():
        settings = {**_measurement_settings(image.shape[2], False,
                                            summarize=True),
                    "channel_names": None}
        summary = _measure_core({"cell": cells}, image, settings,
                                objects=objects, tables=False)["summaries"].get("cell")
    return _cell_object_table(cells, objects, summary, channel_names)


def _cell_object_table(cells, objects, summary, channel_names):
    """Measure's per-cell object summary under the OPS column names.

    :param cells: cell labels.
    :param objects: ``{class: labels}`` that were summarised.
    :param summary: the ``cell`` table of
        :func:`spacr.measure._measure_field_objects`, with its
        ``organelle_summary_<class>_`` columns, or None.
    :param channel_names: one short name per channel.
    :returns: one row per cell label, as :func:`_cell_object_features`
        documents.
    """
    import pandas as pd

    names = [str(name).lower() for name in channel_names]
    labels = np.unique(cells[cells > 0]).astype(np.int64)
    out = pd.DataFrame({"label": labels})
    for role in objects:
        columns = [f"cell_{role}_{stat}" for stat in _CELL_OBJECT_STATS]
        for name in names:
            columns += [f"cell_{role}_{name}_mean_intensity",
                        f"cell_{role}_{name}_std_intensity"]
        if not len(labels):
            out = out.reindex(columns=["label", *columns])
            continue
        renamed = {f"organelle_summary_{role}_{stat}": f"cell_{role}_{stat}"
                   for stat in _CELL_OBJECT_STATS}
        for index, name in enumerate(names):
            for stat in ("mean", "std"):
                renamed[f"organelle_summary_{role}_channel_{index}_{stat}_intensity_per_cell"] = (
                    f"cell_{role}_{name}_{stat}_intensity")
        frame = (summary if summary is not None
                 else pd.DataFrame({"label": np.zeros(0, np.int64)}))
        frame = frame.loc[:, ["label", *[c for c in frame.columns if c in renamed]]]
        frame = frame.rename(columns=renamed)
        frame = frame.assign(label=frame["label"].astype(np.int64))
        frame = frame.reindex(columns=["label", *columns])
        frame[columns] = frame[columns].astype(np.float64).fillna(0.0)
        out = out.merge(frame, on="label", how="left")
    return out
