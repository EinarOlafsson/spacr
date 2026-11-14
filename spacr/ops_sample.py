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
    nuclei = np.asarray(nuclei)
    cells = np.asarray(cells)
    if nuclei.shape != cells.shape:
        raise SampleError(
            f"nucleus and cell masks differ in shape: {nuclei.shape} and "
            f"{cells.shape}")
    n_flat = nuclei.ravel().astype(np.int64)
    c_flat = cells.ravel().astype(np.int64)
    both = (n_flat > 0) & (c_flat > 0)
    n_total = int(np.unique(n_flat[n_flat > 0]).size)
    c_total = int(np.unique(c_flat[c_flat > 0]).size)
    if not both.any():
        return (np.zeros(nuclei.shape, np.int32),
                np.zeros(cells.shape, np.int32),
                {"nuclei": n_total, "cells": c_total, "paired": 0,
                 "nuclei_without_cell": n_total,
                 "cells_without_nucleus": c_total,
                 "extra_nuclei_in_cell": 0})
    width = int(c_flat.max()) + 1
    pair, overlap = np.unique(n_flat[both] * width + c_flat[both],
                              return_counts=True)
    pair_n, pair_c = pair // width, pair % width
    order = np.lexsort((-overlap, pair_n))
    pair_n, pair_c, overlap = pair_n[order], pair_c[order], overlap[order]
    first = np.r_[True, pair_n[1:] != pair_n[:-1]]
    best_n, best_c, best_o = pair_n[first], pair_c[first], overlap[first]
    order = np.lexsort((-best_o, best_c))
    best_n, best_c = best_n[order], best_c[order]
    first = np.r_[True, best_c[1:] != best_c[:-1]]
    keep_n, keep_c = best_n[first], best_c[first]
    labels = np.arange(1, keep_n.size + 1, dtype=np.int32)
    n_lut = np.zeros(int(n_flat.max()) + 1, np.int32)
    c_lut = np.zeros(width, np.int32)
    n_lut[keep_n] = labels
    c_lut[keep_c] = labels
    new_cells = c_lut[cells.astype(np.int64)]
    new_nuclei = n_lut[nuclei.astype(np.int64)]
    new_nuclei[new_nuclei != new_cells] = 0
    counts = {
        "nuclei": n_total, "cells": c_total, "paired": int(keep_n.size),
        "nuclei_without_cell": int(n_total - np.unique(pair_n).size),
        "cells_without_nucleus": int(c_total - np.unique(pair_c).size),
        "extra_nuclei_in_cell": int(best_n.size - keep_n.size),
    }
    return new_nuclei, new_cells, counts


def _measurement_settings(channels: int, texture: bool,
                          gpu: bool = False,
                          profile_features=None,
                          granularity_subsample: float = 1.0
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
        block (``calculate_correlation``) is on too.
    :param granularity_subsample: ``profile_granularity_subsample``, the
        ``subsample`` of :func:`spacr.measure.measure_granularity`.
    :returns: the dict :func:`spacr.measure._morphological_measurements` and
        :func:`spacr.measure._intensity_measurements` read.
    """
    from .measure import profile_feature_families
    from .ops_settings import OPS_TO_MEASURE_KEYS

    families = profile_feature_families(profile_features)
    ops = {"ops_ph_texture": bool(texture), "ops_ph_measure_gpu": bool(gpu),
           "ops_ph_profile_features": list(families) or None,
           "ops_ph_granularity_subsample": float(granularity_subsample)}
    return {
        "cell_mask_dim": channels, "nucleus_mask_dim": channels + 1,
        "pathogen_mask_dim": None, "cytoplasm": True,
        "radial_dist": False,
        "calculate_correlation": "colocalisation" in families,
        "homogeneity_distances": [2, 4, 8],
        "distance_gaussian_sigma": 10, "spatial_measurements": False,
        "channels": list(range(channels)), "verbose": False,
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
                            granularity_subsample: float = 1.0):
    """Per-cell features of one phenotype field, with spaCR's own measurement.

    :param nuclei: nucleus labels, paired with ``cells`` by
        :func:`pair_nuclei_with_cells`, so nucleus ``k`` is in cell ``k``.
    :param cells: cell labels.
    :param image: ``(H, W, C)`` intensities.
    :param channel_names: one short name per channel, used in the column
        names in place of spaCR's ``channel_<i>``.
    :param texture: include the GLCM homogeneity block.
    :param funk_features: add Funk et al. 2022's features: the mass
        displacement of every channel and the correlation of every channel
        pair in the nucleus and the cell (:func:`funk_object_features`),
        and ``nucleus_form_factor`` and ``cell_form_factor``,
        ``4 * pi * area / perimeter ** 2`` (``shape_features['form_factor']``
        of their ``ops/cp_emulator.py``) from spaCR's own area and
        perimeter.
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
    import pandas as pd
    from scipy import ndimage

    from .measure import (_channel_named, _form_factor,
                          _intensity_measurements,
                          _morphological_measurements)

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
    if not nuclei.any():
        return pd.DataFrame(columns=["label", "centroid_row", "centroid_col"])
    cytoplasm = np.where(nuclei > 0, 0, cells).astype(np.int32)
    blank = np.zeros_like(cells)
    settings = _measurement_settings(image.shape[2], texture, gpu,
                                     profile_features, granularity_subsample)
    morphology = _morphological_measurements(
        cells, nuclei, blank, blank, cytoplasm, settings,
        zernike=bool(zernike))
    intensity = _intensity_measurements(
        cells, nuclei, blank, blank, cytoplasm, image, settings,
        sizes=[1, 2, 3, 4, 5], periphery=False, outside=False)
    names = [str(name).lower() for name in channel_names]
    labels = np.unique(nuclei[nuclei > 0])
    centres = np.asarray(ndimage.center_of_mass(
        np.ones_like(nuclei), nuclei, labels), float).reshape(-1, 2)
    out = pd.DataFrame({"label": labels.astype(np.int64),
                        "centroid_row": centres[:, 0],
                        "centroid_col": centres[:, 1]})
    for slot, prefix in ((0, "cell"), (1, "nucleus"), (4, "cytoplasm")):
        for frame in (morphology[slot], intensity[slot]):
            if frame is None or not len(frame) or "label" not in frame:
                continue
            frame = frame.loc[:, ~frame.columns.duplicated()]
            renamed = {}
            for column in frame.columns:
                if column == "label":
                    continue
                text = _channel_named(str(column), names)
                if not text.startswith(f"{prefix}_"):
                    text = f"{prefix}_{text}"
                renamed[column] = text
            frame = frame.rename(columns=renamed)
            frame = frame.loc[:, [c for c in frame.columns
                                  if "pathogen" not in str(c)
                                  and not str(c).endswith("_cell_id")
                                  and not str(c).endswith("label_correlation")]]
            frame = frame.loc[:, [c for c in frame.columns
                                  if c == "label" or c not in out.columns]]
            frame = frame.assign(label=frame["label"].astype(np.int64))
            frame = frame.drop_duplicates(subset=["label"])
            out = out.merge(frame, on="label", how="left")
    if funk_features:
        for prefix, mask in (("nucleus", nuclei), ("cell", cells)):
            funk = funk_object_features(mask, image, channel_names, prefix)
            funk = funk.loc[:, [c for c in funk.columns
                                if c == "label" or c not in out.columns]]
            out = out.merge(funk, on="label", how="left")
            area, perimeter = f"{prefix}_area", f"{prefix}_perimeter"
            if (area in out and perimeter in out
                    and f"{prefix}_form_factor" not in out):
                out[f"{prefix}_form_factor"] = _form_factor(
                    out[area].to_numpy(np.float64),
                    out[perimeter].to_numpy(np.float64))
    keep = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    return out.loc[:, keep].reset_index(drop=True)
