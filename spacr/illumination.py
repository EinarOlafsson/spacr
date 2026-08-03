"""Illumination / flat-field correction for the measurement path.

The problem
-----------

No microscope lights a field of view evenly. A lamp profile, vignetting in
the objective, a tilted condenser and dirt on the optics together make the
same cell measure brighter at the centre of the field than at its edge --
routinely 10-40 % between the middle and a corner on a widefield screen.
Every intensity feature spaCR writes (``*_mean_intensity``,
``*_percentile_*``, the radial distribution, the texture channels that are
computed on intensity) therefore carries a **position-dependent bias**, and
because objects are not distributed identically over every well, that bias
does not average out: it survives into the per-well aggregate, and from there
into classification and regression as an effect that looks entirely real and
is entirely an artefact of the optics.

It is also the root cause of the plate-scale edge effects
:mod:`spacr.plate_qc` already detects and reports. Detecting them is useful;
removing them is what changes the answer.

The model
---------

Per channel, per plate::

    observed(y, x) = dark + flat(y, x) * true(y, x)

``flat`` is the multiplicative illumination field, normalised so its mean is
1 -- the plate's overall intensity level is preserved, so a corrected number
stays on the same scale as an uncorrected one and only its *position
dependence* is removed. ``dark`` is the additive camera offset. The
correction the hook applies is::

    corrected = (observed - dark) / flat

What is estimated, and why
--------------------------

**Retrospective, from the data itself**: a per-pixel median across many
fields of the plate, followed by a fit of a smooth low-order surface.

*Why the median across fields.* A pixel is covered by a cell in only a
minority of the fields on a plate, so the across-field median at that pixel
sees background almost every time and the objects drop out. Each field is
first divided by its own median, so a densely-seeded field does not pull the
estimate up simply because it has more cells in it -- what is being averaged
is the *relative* profile, not the brightness.

*Why the surface fit on top.* Illumination is a physically smooth,
low-frequency function of position: a lamp profile plus a vignette. Fitting a
low-order 2-D polynomial (default degree 4, 15 terms) to the per-pixel median
imposes exactly that prior, so residual object structure and photon noise
cannot leak into the gain map and be baked into every measurement on the
plate. The fit is trimmed twice against a MAD threshold, so a persistent
bright artefact -- a fluorescent speck in the same place on every field --
is rejected rather than smeared into the surface. For illumination that is
genuinely not polynomial (a dust shadow, a sharply structured lamp) pass
``estimator='smooth'``: the same per-pixel median, Gaussian-smoothed at
1/16 of the short side and interpolated back to full resolution.

*Why not BaSiC.* BaSiC's low-rank + sparse decomposition is the better
estimator when you have hundreds of fields and a genuine dark-field to
recover, but it is an iterative optimisation with its own convergence
failure modes, and its dark-field term is only identifiable because of those
extra assumptions. From a single acquisition, ``dark`` and ``flat`` are not
separately identifiable at all: multiplying ``flat`` by a constant and
absorbing it into the per-field brightness leaves every observation
unchanged. Estimating a dark-field anyway -- for instance from the per-pixel
minimum across fields, which is the usual shortcut -- returns *dark plus the
dimmest background*, and subtracting it removes real signal. So spaCR does
not guess: ``dark`` is 0 unless you supply the camera offset you measured
from a dark frame, via ``illumination_dark``.

Reaching the worker processes
-----------------------------

:func:`spacr.measure.measure_crop` measures fields in a
:class:`multiprocessing.Pool`. Under ``spawn`` / ``forkserver`` each worker is
a fresh interpreter with an empty hook registry, so a correction registered
only in the parent applies to **nothing** while the run looks perfectly
normal -- the single worst outcome for this feature, because the user then
believes their numbers are corrected. :func:`enable_illumination_correction`
therefore does not merely register the hook: it writes the model path to
:data:`MODEL_ENV_VAR` and appends ``spacr.illumination:install`` to
``SPACR_MEASURE_HOOKS``, which every start method inherits, so each worker
installs the correction for itself. :func:`worker_delivery_status` reports
whether that is actually in place, and :func:`enable_illumination_correction`
prints it.

Using it
--------

Off by default. From a settings dict (the keys are registered through the
:func:`spacr.settings.register_defaults` seam, see
:func:`illumination_settings`)::

    settings = get_measure_crop_settings(settings={...})
    settings['illumination_correction'] = True
    prepare_illumination_correction(settings)   # estimate, save, enable, QC
    measure_crop(settings)

or explicitly::

    model = estimate_illumination(src, channels=[0, 1, 2])
    model.save('/data/plate1/illumination/illumination_model.npz')
    illumination_qc(model, src, save_dir='/data/plate1/illumination')
    enable_illumination_correction('/data/plate1/illumination/illumination_model.npz')

Nothing in this module runs unless one of those calls is made, and
:func:`disable_illumination_correction` returns the process to a state where
``measure_crop`` measures exactly what it measured before.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field as _dataclass_field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigurationError
from .measure_hooks import (
    HOOKS_ENV_VAR,
    register_preprocessing_hook,
    preprocessing_hooks,
    unregister_preprocessing_hook,
)

__all__ = [
    'APP_KEY',
    'HOOK_NAME',
    'HOOK_PRIORITY',
    'INSTALLER_ENTRY',
    'MODEL_ENV_VAR',
    'ON_MISSING_ENV_VAR',
    'IlluminationError',
    'IlluminationField',
    'IlluminationModel',
    'IlluminationCorrector',
    'estimate_illumination',
    'load_illumination_model',
    'plate_of_field',
    'position_intensity_slope',
    'illumination_qc',
    'enable_illumination_correction',
    'disable_illumination_correction',
    'worker_delivery_status',
    'install',
    'prepare_illumination_correction',
    'illumination_settings',
    'register_illumination_settings',
]

#: Settings key namespace and :func:`spacr.settings.register_defaults` key.
APP_KEY = 'illumination'

#: Registry key the correction is registered under. Fixed, and always passed
#: explicitly: the parent process and a spawned worker both register the same
#: name, and :mod:`spacr.measure_hooks` replaces rather than appends, so a
#: field cannot be corrected twice however many routes installed the hook.
HOOK_NAME = 'spacr.illumination.correct'

#: Illumination correction runs before any other preprocessing hook. It is a
#: correction of the *sensor*, so anything else a user chains on top -- a
#: background subtraction, a ratio -- should see corrected pixels.
HOOK_PRIORITY = -100

#: What goes in ``SPACR_MEASURE_HOOKS`` so worker processes install it too.
INSTALLER_ENTRY = 'spacr.illumination:install'

#: Path to the saved model. Read by :func:`install` in each worker.
MODEL_ENV_VAR = 'SPACR_ILLUMINATION_MODEL'

#: ``'error'`` (default) or ``'skip'`` -- what a worker does with a field
#: whose plate has no estimated illumination field.
ON_MISSING_ENV_VAR = 'SPACR_ILLUMINATION_ON_MISSING'

#: Key used when the model is estimated across all plates at once.
ALL_PLATES = '*'

#: Below this many fields the across-field median is not a reliable object
#: rejector, so the estimate is still produced but loudly qualified.
MIN_FIELDS_FOR_A_TRUSTWORTHY_ESTIMATE = 10

#: A fitted surface is floored at this fraction of its own median before it is
#: inverted. Dividing by a gain that a polynomial fit dragged to ~0 in a corner
#: would turn a handful of pixels into astronomic intensities.
FLAT_FLOOR_FRACTION = 0.05

#: Clipping warnings printed per process before they are summarised instead.
_MAX_CLIP_WARNINGS = 5


class IlluminationError(ConfigurationError):
    """Illumination correction was asked for and could not be delivered.

    A :class:`spacr.errors.ConfigurationError`, not a per-field data error:
    every failure this class reports (no fields to estimate from, a model that
    does not cover the plate being measured, a channel the model was never
    estimated for) is wrong for the whole run, and the alternative -- measuring
    on quietly uncorrected pixels -- is the outcome this module exists to
    prevent.
    """


# ---------------------------------------------------------------------------
# The estimated field
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IlluminationField:
    """The illumination estimate for one plate, for one or more channels.

    :ivar plate: the plate key, or :data:`ALL_PLATES` when the model was
        estimated across every plate at once.
    :ivar channels: source channel indices, in the order they index
        :attr:`flatfield`'s first axis. These are indices into the *merged
        stack*, i.e. exactly the values in ``settings['channels']``.
    :ivar flatfield: ``(C, Y, X)`` float32 multiplicative field, normalised so
        each channel's mean is 1.0. ``corrected = (observed - dark) /
        flatfield``.
    :ivar dark: per-channel additive offset subtracted before dividing. Zero
        unless the user supplied a measured camera offset -- see the module
        docstring for why it is not estimated.
    :ivar n_fields: how many fields the estimate was made from.
    :ivar estimator: ``'polynomial'`` or ``'smooth'``.
    :ivar degree: polynomial degree, or 0 for the smooth estimator.
    :ivar bin_size: the binning factor the per-pixel median was computed at.
        Illumination is low-frequency, so binning costs nothing and buys both
        the memory to hold many fields at once and a quieter statistic.
    :ivar floored: pixels the fitted surface had to be floored at (see
        :data:`FLAT_FLOOR_FRACTION`). Non-zero means the fit went negative
        somewhere and the estimate should be looked at before it is trusted.
    """

    plate: str
    channels: Tuple[int, ...]
    flatfield: np.ndarray
    dark: np.ndarray
    n_fields: int
    estimator: str
    degree: int
    bin_size: int
    floored: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        """``(Y, X)`` pixel shape of the estimated field."""
        return (int(self.flatfield.shape[1]), int(self.flatfield.shape[2]))

    def index_of(self, channel: int) -> int:
        """Position of source ``channel`` along :attr:`flatfield`'s first axis.

        :param channel: a merged-stack channel index.
        :raises IlluminationError: if the model was never estimated for it.
            Correcting the channels that happen to be present and leaving the
            rest alone would put corrected and uncorrected numbers in the same
            table.
        """
        try:
            return self.channels.index(int(channel))
        except ValueError:
            raise IlluminationError(
                f"the illumination model for plate {self.plate!r} covers "
                f"channels {list(self.channels)}, but the run measures "
                f"channel {channel}. Re-estimate with "
                f"channels={sorted(set(self.channels) | {int(channel)})}."
            ) from None

    def gain_stack(self, channels: Sequence[int]) -> np.ndarray:
        """``(Y, X, C)`` multiplicative gains ``1 / flatfield`` for ``channels``.

        Shaped for the array a preprocessing hook is handed, so applying the
        correction is one broadcast multiply.

        :param channels: source channel indices, in the order they appear
            along the last axis of the array being corrected.
        """
        gains = [1.0 / self.flatfield[self.index_of(c)] for c in channels]
        return np.stack(gains, axis=-1).astype(np.float32, copy=False)

    def dark_stack(self, channels: Sequence[int]) -> np.ndarray:
        """``(C,)`` additive offsets for ``channels``, ready to broadcast."""
        return np.asarray([self.dark[self.index_of(c)] for c in channels],
                          dtype=np.float32)

    def nonuniformity(self) -> Dict[int, float]:
        """Per channel, ``(p98 - p2) / mean`` of the field, as a fraction.

        The headline "how uneven is this microscope" number: 0.30 means the
        bright and dim ends of the field of view differ by 30 % of the mean,
        and therefore so does the same cell measured in those two places.
        """
        out = {}
        for position, channel in enumerate(self.channels):
            plane = self.flatfield[position]
            low, high = np.percentile(plane, [2, 98])
            out[int(channel)] = float((high - low) / max(plane.mean(), 1e-12))
        return out

    def describe(self) -> str:
        """One line per channel: range, non-uniformity and how it was made."""
        lines = []
        nonuniform = self.nonuniformity()
        for position, channel in enumerate(self.channels):
            plane = self.flatfield[position]
            lines.append(
                f"plate {self.plate}, channel {channel}: gain field "
                f"{plane.min():.3f}-{plane.max():.3f} (mean 1.000), "
                f"non-uniformity {100 * nonuniform[int(channel)]:.1f}%, "
                f"{self.estimator}"
                f"{f' degree {self.degree}' if self.degree else ''} over "
                f"{self.n_fields} field(s), dark={self.dark[position]:g}")
        return '\n'.join(lines)


@dataclass
class IlluminationModel:
    """Estimated illumination fields for every plate in a source folder.

    :ivar fields: plate key -> :class:`IlluminationField`. A model estimated
        with ``per_plate=False`` holds the single key :data:`ALL_PLATES`,
        which matches every plate.
    :ivar meta: provenance -- source folders, channels, when it was estimated,
        the settings it was estimated with. Written into the ``.npz`` and read
        back, so a model on disk can always say what produced it.
    """

    fields: Dict[str, IlluminationField]
    meta: Dict[str, Any] = _dataclass_field(default_factory=dict)

    @property
    def per_plate(self) -> bool:
        """Whether the model holds one field per plate rather than one field."""
        return ALL_PLATES not in self.fields

    def field_for(self, plate: str) -> IlluminationField:
        """The :class:`IlluminationField` that applies to ``plate``.

        :raises IlluminationError: when nothing in the model covers it. This
            is deliberately not a fall back to "some other plate's field":
            illumination differs between acquisition sessions, which is the
            whole reason the default is one field per plate.
        """
        if ALL_PLATES in self.fields:
            return self.fields[ALL_PLATES]
        try:
            return self.fields[plate]
        except KeyError:
            raise IlluminationError(
                f"no illumination field was estimated for plate {plate!r}; "
                f"the model covers {sorted(self.fields)}. Re-estimate over a "
                f"source folder that contains this plate, or estimate one "
                f"field for everything with illumination_per_plate=False."
            ) from None

    def describe(self) -> str:
        """Every field's :meth:`IlluminationField.describe`, one per line."""
        return '\n'.join(self.fields[key].describe()
                         for key in sorted(self.fields))

    def save(self, path: str) -> str:
        """Write the model to ``path`` as a compressed ``.npz``.

        The path is what :func:`enable_illumination_correction` puts in the
        environment, and what each worker process loads: the model has to be
        on disk for a ``spawn`` worker to be able to see it at all.

        :param path: destination file. Parent folders are created.
        :returns: the absolute path written.
        """
        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        arrays = {}
        index = {}
        for key, item in self.fields.items():
            slot = f'field{len(index)}'
            index[slot] = {
                'plate': item.plate,
                'channels': [int(c) for c in item.channels],
                'dark': [float(d) for d in np.asarray(item.dark).ravel()],
                'n_fields': int(item.n_fields),
                'estimator': item.estimator,
                'degree': int(item.degree),
                'bin_size': int(item.bin_size),
                'floored': int(item.floored),
                'key': key,
            }
            arrays[slot] = np.asarray(item.flatfield, dtype=np.float32)
        payload = {'index': index, 'meta': self.meta, 'format': 1}
        np.savez_compressed(path, manifest=np.asarray(json.dumps(payload)),
                            **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> 'IlluminationModel':
        """Read a model written by :meth:`save`.

        :param path: the ``.npz`` file.
        :raises IlluminationError: when the file is missing or is not a
            spaCR illumination model. A worker that cannot load the model must
            say so rather than measure uncorrected pixels.
        """
        if not os.path.isfile(path):
            raise IlluminationError(
                f"illumination model {path!r} does not exist. "
                f"{MODEL_ENV_VAR} must point at a file "
                f"IlluminationModel.save() wrote.")
        try:
            with np.load(path, allow_pickle=False) as handle:
                payload = json.loads(str(handle['manifest']))
                fields = {}
                for slot, entry in payload['index'].items():
                    flat = np.asarray(handle[slot], dtype=np.float32)
                    fields[entry['key']] = IlluminationField(
                        plate=entry['plate'],
                        channels=tuple(int(c) for c in entry['channels']),
                        flatfield=flat,
                        dark=np.asarray(entry['dark'], dtype=np.float32),
                        n_fields=int(entry['n_fields']),
                        estimator=str(entry['estimator']),
                        degree=int(entry['degree']),
                        bin_size=int(entry['bin_size']),
                        floored=int(entry.get('floored', 0)))
        except IlluminationError:
            raise
        except Exception as exc:
            raise IlluminationError(
                f"illumination model {path!r} could not be read: "
                f"{type(exc).__name__}: {exc}") from exc
        return cls(fields=fields, meta=dict(payload.get('meta', {})))


def load_illumination_model(path: str) -> IlluminationModel:
    """Read a saved model. Thin alias for :meth:`IlluminationModel.load`.

    :param path: the ``.npz`` written by :meth:`IlluminationModel.save`.
    """
    return IlluminationModel.load(path)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def plate_of_field(file_name: str) -> str:
    """The plate a merged field belongs to, from its file name.

    spaCR names merged fields ``<plateID>_<wellID>_<fieldID>.npy`` (see
    :mod:`spacr.io`), so the plate is the first underscore-separated token.
    A name with no underscore is its own plate, which keeps a hand-assembled
    folder working instead of silently pooling it.

    :param file_name: file name or stem, with or without directories.
    """
    stem = os.path.splitext(os.path.basename(str(file_name)))[0]
    head, sep, _rest = stem.partition('_')
    return head if sep else stem


def _source_folders(src) -> Tuple[str, ...]:
    """Normalise ``settings['src']`` -- a folder or a list of them -- to a tuple."""
    if isinstance(src, (str, os.PathLike)):
        return (os.path.abspath(str(src)),)
    return tuple(os.path.abspath(str(item)) for item in src)


def _merged_files(src) -> Dict[str, list]:
    """Map plate key -> sorted list of merged ``.npy`` paths under ``src``."""
    grouped: Dict[str, list] = {}
    for folder in _source_folders(src):
        if not os.path.isdir(folder):
            raise IlluminationError(
                f"illumination correction was asked to estimate from "
                f"{folder!r}, which is not a folder. Point it at the merged "
                f"field folder measure_crop reads (settings['src']).")
        for name in sorted(os.listdir(folder)):
            if name.endswith('.npy'):
                grouped.setdefault(plate_of_field(name), []).append(
                    os.path.join(folder, name))
    return grouped


def _sample(paths: Sequence[str], limit: int) -> list:
    """Take at most ``limit`` paths, evenly spaced across the sorted list.

    Evenly spaced rather than random: the files are sorted by well and field,
    so this walks the whole plate instead of over-weighting whichever corner a
    random draw happened to hit, and it is deterministic -- two runs of the
    estimator on the same folder produce the same field, which matters when
    the output is a correction applied to published numbers.
    """
    paths = list(paths)
    if limit <= 0 or len(paths) <= limit:
        return paths
    step = len(paths) / float(limit)
    return [paths[min(len(paths) - 1, int(i * step))] for i in range(limit)]


def _bin_factor(shape: Tuple[int, int], grid: int) -> int:
    """Binning factor that puts the long side of ``shape`` at or below ``grid``."""
    height, width = shape
    factor = int(math.ceil(max(height, width) / float(max(1, grid))))
    return max(1, min(factor, height, width))


def _bin_image(plane: np.ndarray, factor: int) -> np.ndarray:
    """Block-average ``plane`` by ``factor``, dropping any partial edge blocks."""
    if factor == 1:
        return np.asarray(plane, dtype=np.float32)
    height = (plane.shape[0] // factor) * factor
    width = (plane.shape[1] // factor) * factor
    block = np.asarray(plane[:height, :width], dtype=np.float64)
    block = block.reshape(height // factor, factor, width // factor, factor)
    return block.mean(axis=(1, 3)).astype(np.float32)


def _bin_centres(length: int, factor: int) -> np.ndarray:
    """Full-resolution pixel coordinates of each block centre."""
    count = length // factor
    return np.arange(count, dtype=np.float64) * factor + (factor - 1) / 2.0


def _normalised(coords: np.ndarray, length: int) -> np.ndarray:
    """Map pixel coordinates onto ``[-1, 1]`` across an axis of ``length``."""
    half = max((length - 1) / 2.0, 1e-9)
    return (np.asarray(coords, dtype=np.float64) - half) / half


def _polynomial_terms(degree: int) -> Tuple[Tuple[int, int], ...]:
    """Exponent pairs ``(row_power, col_power)`` of a 2-D polynomial."""
    return tuple((i, j) for i in range(degree + 1)
                 for j in range(degree + 1 - i))


def _fit_polynomial_surface(stat: np.ndarray, shape: Tuple[int, int],
                            factor: int, degree: int) -> np.ndarray:
    """Fit a trimmed 2-D polynomial to ``stat`` and evaluate it at ``shape``.

    ``stat`` is the binned per-pixel median; ``shape`` is the full-resolution
    field. Two rounds of MAD trimming reject bins the surface cannot explain
    -- a fluorescent speck that sits in the same place on every field, an
    always-confluent corner -- rather than bending the surface towards them.
    """
    height, width = shape
    rows = _normalised(_bin_centres(height, factor), height)
    cols = _normalised(_bin_centres(width, factor), width)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    terms = _polynomial_terms(degree)
    design = np.stack([(grid_rows ** i) * (grid_cols ** j)
                       for i, j in terms], axis=-1)
    design = design.reshape(-1, len(terms))
    target = np.asarray(stat, dtype=np.float64).reshape(-1)

    keep = np.isfinite(target) & (target > 0)
    if keep.sum() < 4 * len(terms):
        keep = np.isfinite(target)
    coefficients = None
    for _round in range(3):
        if keep.sum() < len(terms) + 1:
            break
        coefficients, *_ = np.linalg.lstsq(design[keep], target[keep],
                                           rcond=None)
        residual = target - design @ coefficients
        scale = 1.4826 * np.median(np.abs(residual[keep] -
                                          np.median(residual[keep])))
        if not np.isfinite(scale) or scale <= 0:
            break
        tighter = keep & (np.abs(residual) <= 3.0 * scale)
        if tighter.sum() < max(4 * len(terms), len(terms) + 1):
            break
        if tighter.sum() == keep.sum():
            break
        keep = tighter
    if coefficients is None:
        raise IlluminationError(
            'the illumination surface could not be fitted: every binned '
            'pixel was rejected as non-finite or non-positive. Check that '
            'the channel being estimated actually holds image data.')

    full_rows = _normalised(np.arange(height), height)
    full_cols = _normalised(np.arange(width), width)
    surface = np.zeros((height, width), dtype=np.float64)
    for coefficient, (i, j) in zip(coefficients, terms):
        surface += coefficient * np.outer(full_rows ** i, full_cols ** j)
    return surface


def _smooth_surface(stat: np.ndarray, shape: Tuple[int, int],
                    factor: int) -> np.ndarray:
    """Gaussian-smooth ``stat`` and interpolate it up to ``shape``.

    The alternative estimator, for illumination a polynomial cannot express.
    Sigma is 1/16 of the binned short side: large enough that cell-scale
    structure (a few binned pixels at most) cannot survive it, small enough to
    keep a real lamp profile or a dust shadow.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    sigma = max(1.0, min(stat.shape) / 16.0)
    smoothed = gaussian_filter(np.asarray(stat, dtype=np.float64), sigma,
                               mode='nearest')
    height, width = shape
    rows = (np.arange(height, dtype=np.float64) - (factor - 1) / 2.0) / factor
    cols = (np.arange(width, dtype=np.float64) - (factor - 1) / 2.0) / factor
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    return map_coordinates(smoothed, [grid_rows, grid_cols], order=1,
                           mode='nearest')


def _read_binned_field(path: str, channels: Sequence[int],
                       factor: Optional[int], grid: int
                       ) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """Load one merged field and return its binned intensity channels.

    :returns: ``(binned (C, y, x), factor, full (Y, X) shape)``.
    """
    data = np.load(path, mmap_mode='r')
    if data.ndim not in (3, 4):
        raise IlluminationError(
            f"{path!r} is a {data.ndim}-D array; a merged field is (Y, X, C) "
            f"or (Z, Y, X, C).")
    full_shape = (int(data.shape[-3]), int(data.shape[-2])) if data.ndim == 4 \
        else (int(data.shape[0]), int(data.shape[1]))
    if factor is None:
        factor = _bin_factor(full_shape, grid)
    planes = []
    for channel in channels:
        if channel >= data.shape[-1]:
            raise IlluminationError(
                f"{path!r} has {data.shape[-1]} channels, so channel "
                f"{channel} does not exist. settings['channels'] names "
                f"indices into the merged stack.")
        plane = np.asarray(data[..., int(channel)])
        if plane.ndim == 3:
            # A z-stack is corrected with one 2-D field: the illumination is a
            # property of the optics in x/y. The median over z is what the
            # estimate sees, so an out-of-focus slice does not dominate it.
            plane = np.median(plane, axis=0)
        planes.append(_bin_image(plane, factor))
    return np.stack(planes, axis=0), factor, full_shape


def _relative_profile(stack: np.ndarray) -> np.ndarray:
    """Per-pixel median of fields, each normalised by its own median.

    ``stack`` is ``(K, y, x)``. Dividing each field by its own median before
    the pixel-wise median is what makes this an estimate of the *profile*: a
    field packed with bright cells then contributes its shape, not its
    brightness.
    """
    levels = np.median(stack.reshape(stack.shape[0], -1), axis=1)
    usable = np.isfinite(levels) & (levels > 0)
    if not usable.any():
        raise IlluminationError(
            'every field used for the illumination estimate had a median of '
            'zero or worse; there is no signal to estimate a profile from.')
    scaled = stack[usable] / levels[usable][:, None, None]
    return np.median(scaled, axis=0)


def estimate_illumination(src, channels: Sequence[int], *,
                          per_plate: bool = True,
                          estimator: str = 'polynomial',
                          degree: int = 4,
                          max_fields: int = 50,
                          grid: int = 256,
                          dark: float = 0.0,
                          verbose: bool = True) -> IlluminationModel:
    """Estimate the illumination field from the merged fields in ``src``.

    Retrospective: the data corrects itself. See the module docstring for what
    is estimated and why.

    :param src: the merged field folder ``measure_crop`` reads, or a list of
        them (``settings['src']`` accepts both).
    :param channels: merged-stack channel indices to estimate, i.e.
        ``settings['channels']``. One field is estimated per channel: two
        fluorophores go through different filters and vignette differently.
    :param per_plate: one field per plate (default) or one for everything.
        Per plate is the default because illumination differs between
        acquisition sessions -- lamp age, a re-seated filter cube, a different
        objective -- and pooling two sessions estimates neither.
    :param estimator: ``'polynomial'`` (default) or ``'smooth'``.
    :param degree: polynomial degree. 4 gives 15 terms: enough for a lamp
        profile plus a vignette and a tilt, far too few to fit a cell.
    :param max_fields: fields per plate to estimate from, sampled evenly
        across the sorted file list. 50 is well past the point where the
        across-field median stops moving, and bounds the memory.
    :param grid: the per-pixel median is computed on a binned grid whose long
        side is at most this. Illumination is low-frequency, so binning loses
        nothing, quiets the photon noise and is what makes 50 fields fit in
        memory. The fitted surface is returned at full resolution.
    :param dark: additive camera offset subtracted before dividing, in raw
        counts. Not estimated -- see the module docstring.
    :param verbose: print one line per estimated field.
    :returns: an :class:`IlluminationModel`.
    :raises IlluminationError: when ``src`` holds no merged fields, or a field
        is unreadable in a way that would make the estimate meaningless.
    """
    channels = [int(c) for c in channels]
    if not channels:
        raise IlluminationError(
            'illumination correction needs at least one channel to estimate; '
            "settings['channels'] was empty.")
    if estimator not in ('polynomial', 'smooth'):
        raise IlluminationError(
            f"unknown illumination estimator {estimator!r}; use 'polynomial' "
            f"(a trimmed low-order surface) or 'smooth' (a Gaussian-smoothed "
            f"median).")
    grouped = _merged_files(src)
    if not grouped:
        raise IlluminationError(
            f"no .npy fields found under {list(_source_folders(src))}; there "
            f"is nothing to estimate an illumination field from.")
    if not per_plate:
        pooled = [path for paths in grouped.values() for path in paths]
        grouped = {ALL_PLATES: sorted(pooled)}

    fields = {}
    for plate in sorted(grouped):
        paths = _sample(grouped[plate], max_fields)
        stack = []
        factor = None
        full_shape = None
        skipped = 0
        for path in paths:
            binned, factor, shape = _read_binned_field(path, channels, factor,
                                                       grid)
            if full_shape is None:
                full_shape = shape
            elif shape != full_shape:
                # Mixed field sizes in one folder: correcting a 512x512 field
                # with a 1024x1024 gain map is not a thing that can be made to
                # mean anything, so those fields sit the estimate out and the
                # corrector refuses them later by shape.
                skipped += 1
                continue
            stack.append(binned)
        if not stack:
            raise IlluminationError(
                f"plate {plate!r} contributed no usable field to the "
                f"illumination estimate ({skipped} had a different shape to "
                f"the first).")
        stack = np.stack(stack, axis=0)  # (K, C, y, x)
        planes = []
        floored_total = 0
        for position in range(len(channels)):
            profile = _relative_profile(stack[:, position])
            if estimator == 'polynomial':
                surface = _fit_polynomial_surface(profile, full_shape, factor,
                                                  degree)
            else:
                surface = _smooth_surface(profile, full_shape, factor)
            floor = FLAT_FLOOR_FRACTION * float(np.median(surface))
            floored = int(np.count_nonzero(surface < floor))
            floored_total += floored
            if floored:
                surface = np.maximum(surface, floor)
            mean = float(surface.mean())
            if not np.isfinite(mean) or mean <= 0:
                raise IlluminationError(
                    f"the illumination surface for plate {plate!r} channel "
                    f"{channels[position]} has mean {mean!r}; it cannot be "
                    f"normalised or inverted.")
            planes.append((surface / mean).astype(np.float32))
        item = IlluminationField(
            plate=str(plate),
            channels=tuple(channels),
            flatfield=np.stack(planes, axis=0),
            dark=np.full(len(channels), float(dark), dtype=np.float32),
            n_fields=int(stack.shape[0]),
            estimator=estimator,
            degree=int(degree) if estimator == 'polynomial' else 0,
            bin_size=int(factor),
            floored=floored_total)
        fields[str(plate)] = item
        if verbose:
            print(item.describe())
            if item.n_fields < MIN_FIELDS_FOR_A_TRUSTWORTHY_ESTIMATE:
                print(f"WARNING: plate {plate!r} was estimated from only "
                      f"{item.n_fields} field(s). The across-field median "
                      f"rejects objects because a pixel is covered by a cell "
                      f"in a minority of fields; with this few, cells can "
                      f"survive into the gain map. Check the QC image.")
            if item.floored:
                print(f"WARNING: {item.floored} pixel(s) of the fitted "
                      f"surface for plate {plate!r} fell below "
                      f"{FLAT_FLOOR_FRACTION:g} of its median and were "
                      f"floored. Look at the QC image before trusting this "
                      f"field, or use estimator='smooth'.")
            if skipped:
                print(f"NOTE: {skipped} field(s) of plate {plate!r} were a "
                      f"different pixel shape and sat the estimate out.")

    meta = {
        'src': list(_source_folders(src)),
        'channels': channels,
        'per_plate': bool(per_plate),
        'estimator': estimator,
        'degree': int(degree),
        'max_fields': int(max_fields),
        'grid': int(grid),
        'dark': float(dark),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return IlluminationModel(fields=fields, meta=meta)


# ---------------------------------------------------------------------------
# Application: the preprocessing hook
# ---------------------------------------------------------------------------

class IlluminationCorrector:
    """The preprocessing hook that applies an :class:`IlluminationModel`.

    Registered through
    :func:`spacr.measure_hooks.register_preprocessing_hook`, so it is handed
    exactly the array the intensity measurements see -- the channels named by
    ``settings['channels']``, selected out of the merged stack, before a
    single feature is computed.

    **The dtype round trip is this class's decision, and it is made here
    rather than in the hook machinery on purpose** (see
    :func:`spacr.measure_hooks.apply_preprocessing_hooks`). Integer input is
    corrected in float32 and returned by *rounding to nearest* and then
    clipping to the dtype's range:

    * **round, not truncate.** Truncation would shave a mean of 0.5 counts off
      every corrected pixel. Averaged over a 500-pixel object that does not
      wash out -- it is a systematic, one-directional shift of exactly the
      kind this feature exists to remove. Rounding is unbiased.
    * **clip, and count.** A gain above 1 at the edge of the field can push a
      near-full-scale pixel past the top of a uint16. Clipping is the only
      option that keeps the dtype the hook contract requires, but silently
      clipping real signal is a lie about the data, so every pixel that was
      *below* full scale before the correction and lands *at* full scale after
      it is counted and reported. Pixels that were already saturated are not
      counted: they were destroyed by the microscope, not by this class.

    Float input is returned in its own float dtype with no rounding and no
    clipping at all -- there is nothing to round to and no range to leave.

    :param model: the estimated :class:`IlluminationModel`.
    :param on_missing: ``'error'`` (default) or ``'skip'`` for a field whose
        plate the model does not cover. The default is to fail the field:
        a half-corrected table is worse than a failed one.
    :param verbose: print the clipping reports.
    """

    def __init__(self, model: IlluminationModel, *, on_missing: str = 'error',
                 verbose: bool = True) -> None:
        if on_missing not in ('error', 'skip'):
            raise IlluminationError(
                f"on_missing={on_missing!r}; use 'error' (fail the field) or "
                f"'skip' (measure it uncorrected).")
        self.model = model
        self.on_missing = on_missing
        self.verbose = bool(verbose)
        #: fields corrected, fields skipped, pixels clipped, fields that clipped.
        self.stats = {'corrected': 0, 'skipped': 0,
                      'clipped_pixels': 0, 'clipped_fields': 0}
        self._warned = 0
        self._cache: Dict[Tuple[str, Tuple[int, ...]], Tuple] = {}

    # -- the hook ---------------------------------------------------------
    def __call__(self, channel_arrays: np.ndarray, context) -> np.ndarray:
        """Return ``channel_arrays`` corrected, in the same shape and dtype.

        :param channel_arrays: ``(Y, X, C)`` or ``(Z, Y, X, C)`` intensities.
        :param context: the :class:`spacr.measure_hooks.PreprocessingContext`.
        """
        array = np.asarray(channel_arrays)
        plate = plate_of_field(context.file_name)
        try:
            item = self.model.field_for(plate)
        except IlluminationError:
            if self.on_missing == 'error':
                raise
            self.stats['skipped'] += 1
            if self.verbose and self.stats['skipped'] <= _MAX_CLIP_WARNINGS:
                print(f"WARNING: no illumination field for plate {plate!r}; "
                      f"{context.file_name} is measured UNCORRECTED "
                      f"(illumination_on_missing='skip').")
            return channel_arrays

        gains, darks = self._gains_for(item, context.channels)
        spatial = tuple(array.shape[-3:-1])
        if spatial != item.shape:
            raise IlluminationError(
                f"the illumination field for plate {plate!r} is "
                f"{item.shape[0]}x{item.shape[1]} but {context.file_name} is "
                f"{spatial[0]}x{spatial[1]}. A gain map cannot be stretched "
                f"onto a different sensor geometry; re-estimate over these "
                f"fields.")

        corrected = (array.astype(np.float32) - darks) * gains
        return self._to_input_dtype(corrected, array, context)

    def _gains_for(self, item: IlluminationField, channels: Sequence[int]):
        """Cache the ``(Y, X, C)`` gain stack per (plate, channel order)."""
        key = (item.plate, tuple(int(c) for c in channels))
        cached = self._cache.get(key)
        if cached is None:
            cached = (item.gain_stack(channels), item.dark_stack(channels))
            self._cache[key] = cached
        return cached

    def _to_input_dtype(self, corrected: np.ndarray, original: np.ndarray,
                        context) -> np.ndarray:
        """Cast back to ``original.dtype``; see the class docstring."""
        dtype = original.dtype
        if not np.issubdtype(dtype, np.integer):
            self.stats['corrected'] += 1
            return corrected.astype(dtype, copy=False)
        info = np.iinfo(dtype)
        rounded = np.rint(corrected)
        result = np.clip(rounded, info.min, info.max).astype(dtype)
        # Only signal this correction pushed out of range counts: a pixel the
        # microscope had already saturated was never recoverable.
        lost = int(np.count_nonzero((rounded > info.max) &
                                    (original < info.max)))
        lost += int(np.count_nonzero((rounded < info.min) &
                                     (original > info.min)))
        self.stats['corrected'] += 1
        if lost:
            self.stats['clipped_pixels'] += lost
            self.stats['clipped_fields'] += 1
            self._report_clipping(lost, corrected.size, context)
        return result

    def _report_clipping(self, lost: int, total: int, context) -> None:
        """Say that real signal was clipped, without printing it 384 times."""
        if not self.verbose:
            return
        self._warned += 1
        if self._warned > _MAX_CLIP_WARNINGS:
            return
        tail = ('' if self._warned < _MAX_CLIP_WARNINGS else
                ' Further clipping warnings are suppressed; the running total '
                'is on the corrector\'s .stats.')
        print(f"WARNING: illumination correction clipped {lost} pixel(s) "
              f"({100.0 * lost / max(total, 1):.4f}% of "
              f"{context.file_name}) at the top of its dtype. Those pixels "
              f"were not saturated before the correction, so real signal was "
              f"lost. Re-acquire with more headroom, or measure on a wider "
              f"dtype.{tail}")

    def report(self) -> str:
        """One line summarising what this corrector has done so far."""
        return (f"illumination correction: {self.stats['corrected']} field(s) "
                f"corrected, {self.stats['skipped']} skipped, "
                f"{self.stats['clipped_pixels']} pixel(s) clipped across "
                f"{self.stats['clipped_fields']} field(s)")


# ---------------------------------------------------------------------------
# Enabling it -- including in worker processes
# ---------------------------------------------------------------------------

def _env_entries(value: str) -> list:
    """Split a ``SPACR_MEASURE_HOOKS`` value into its non-empty entries."""
    return [item.strip() for item in str(value or '').split(',') if item.strip()]


def install() -> str:
    """Install the correction in **this** process from the environment.

    This is the zero-argument installer ``SPACR_MEASURE_HOOKS`` names, and the
    only route that survives a ``spawn`` / ``forkserver`` worker: the worker is
    a fresh interpreter, so it imports this module and calls this function for
    itself, reading the model from :data:`MODEL_ENV_VAR`.

    :returns: the registry key the hook was registered under.
    :raises IlluminationError: if the environment does not name a readable
        model. Refusing loudly is the point -- a worker that cannot load the
        model must not go on to measure uncorrected pixels.
    """
    path = os.environ.get(MODEL_ENV_VAR, '').strip()
    if not path:
        raise IlluminationError(
            f"{INSTALLER_ENTRY} is in {HOOKS_ENV_VAR} but {MODEL_ENV_VAR} is "
            f"not set, so there is no illumination model to install. Call "
            f"spacr.illumination.enable_illumination_correction(path), which "
            f"sets both.")
    model = IlluminationModel.load(path)
    on_missing = os.environ.get(ON_MISSING_ENV_VAR, 'error').strip() or 'error'
    corrector = IlluminationCorrector(model, on_missing=on_missing)
    return register_preprocessing_hook(corrector, name=HOOK_NAME,
                                       priority=HOOK_PRIORITY)


def enable_illumination_correction(model, *, path: Optional[str] = None,
                                   on_missing: str = 'error',
                                   verbose: bool = True) -> str:
    """Turn the correction on, here and in every worker process.

    Three things happen, and the third is the one that matters:

    1. the model is saved to disk if it is not already there -- a ``spawn``
       worker can only reach it through the file system;
    2. :data:`MODEL_ENV_VAR` and :data:`ON_MISSING_ENV_VAR` are set;
    3. ``spacr.illumination:install`` is appended to ``SPACR_MEASURE_HOOKS``
       (appended, not assigned -- another extension may already be in there),
       and the registry is then consulted so that this process installs the
       hook through that same environment route.

    :param model: an :class:`IlluminationModel`, or the path to a saved one.
    :param path: where to save ``model`` when it is not already a path.
        Defaults to ``<first src>/../illumination/illumination_model.npz``.
    :param on_missing: ``'error'`` or ``'skip'``; see
        :class:`IlluminationCorrector`.
    :param verbose: print what was enabled and whether workers will see it.
    :returns: the registry key the hook is registered under.
    :raises IlluminationError: when the model cannot be saved or loaded.
    """
    if isinstance(model, (str, os.PathLike)):
        model_path = os.path.abspath(str(model))
        IlluminationModel.load(model_path)  # fail here, not in a worker
    else:
        if path is None:
            sources = model.meta.get('src') or []
            base = (os.path.join(os.path.dirname(str(sources[0])),
                                 'illumination')
                    if sources else os.path.join(os.getcwd(), 'illumination'))
            path = os.path.join(base, 'illumination_model.npz')
        model_path = model.save(path)

    os.environ[MODEL_ENV_VAR] = model_path
    os.environ[ON_MISSING_ENV_VAR] = on_missing
    entries = _env_entries(os.environ.get(HOOKS_ENV_VAR, ''))
    if INSTALLER_ENTRY not in entries:
        entries.append(INSTALLER_ENTRY)
    os.environ[HOOKS_ENV_VAR] = ','.join(entries)

    # Consulting the registry runs the environment installers, which is how
    # this process ends up with a hook tagged 'env' -- the same tag a worker
    # gets, and the one measure_crop's start-method warning knows not to shout
    # about. If the variable was already read in this process (it is read once)
    # that does nothing, so fall back to installing directly.
    registered = [entry.name for entry in preprocessing_hooks()]
    if HOOK_NAME not in registered:
        install()
    if verbose:
        ok, message = worker_delivery_status()
        print(f"illumination correction ENABLED from {model_path}")
        print(('  ' if ok else '  WARNING: ') + message)
    return HOOK_NAME


def disable_illumination_correction() -> bool:
    """Turn the correction off, here and for any worker started afterwards.

    Unregisters the hook and removes this module's entry from
    ``SPACR_MEASURE_HOOKS`` -- leaving any other extension's entries alone --
    plus the two model variables.

    :returns: True if a correction was registered and has been removed.
    """
    removed = unregister_preprocessing_hook(HOOK_NAME)
    entries = [item for item in _env_entries(os.environ.get(HOOKS_ENV_VAR, ''))
               if item != INSTALLER_ENTRY]
    if entries:
        os.environ[HOOKS_ENV_VAR] = ','.join(entries)
    else:
        os.environ.pop(HOOKS_ENV_VAR, None)
    os.environ.pop(MODEL_ENV_VAR, None)
    os.environ.pop(ON_MISSING_ENV_VAR, None)
    return removed


def worker_delivery_status(start_method: Optional[str] = None
                           ) -> Tuple[bool, str]:
    """Whether the correction will actually reach ``measure_crop``'s workers.

    The failure this answers is silent by construction: a hook registered only
    in the parent process is a no-op in every ``spawn`` worker, the run
    completes, and the numbers are uncorrected while the user believes they
    are not.

    :param start_method: the pool start method to judge against. Defaults to
        whatever ``SPACR_START_METHOD`` selects, falling back to the platform
        default -- i.e. what :func:`spacr.measure.measure_crop` will use.
    :returns: ``(ok, message)``. ``ok`` is False whenever a field could be
        measured uncorrected without anything saying so.
    """
    if start_method is None:
        import multiprocessing as mp
        start_method = (os.environ.get('SPACR_START_METHOD', '').strip().lower()
                        or mp.get_start_method())
    registered = {entry.name: entry for entry in preprocessing_hooks()}
    entry = registered.get(HOOK_NAME)
    if entry is None:
        return False, ('illumination correction is not registered in this '
                       'process; measure_crop will measure uncorrected '
                       'pixels.')
    in_env = INSTALLER_ENTRY in _env_entries(os.environ.get(HOOKS_ENV_VAR, ''))
    model = os.environ.get(MODEL_ENV_VAR, '').strip()
    if in_env and model and os.path.isfile(model):
        return True, (f"workers install it themselves from {HOOKS_ENV_VAR}="
                      f"'{INSTALLER_ENTRY}' and {MODEL_ENV_VAR}='{model}', so "
                      f"a '{start_method}' pool is covered.")
    if start_method == 'fork':
        return True, (f"a 'fork' pool inherits this process's registry, so the "
                      f"correction reaches the workers -- but {HOOKS_ENV_VAR} "
                      f"does not name {INSTALLER_ENTRY}, so it would NOT "
                      f"survive SPACR_START_METHOD=spawn.")
    return False, (f"the correction is registered in this process only and the "
                   f"pool starts workers with '{start_method}', which does not "
                   f"inherit it: every field would be measured uncorrected. "
                   f"Call enable_illumination_correction(), which sets "
                   f"{HOOKS_ENV_VAR} and {MODEL_ENV_VAR}.")


# ---------------------------------------------------------------------------
# QC: show that it worked
# ---------------------------------------------------------------------------

def position_intensity_slope(intensities: Sequence[float],
                             coordinates: np.ndarray,
                             shape: Tuple[int, int]) -> float:
    """Least-squares slope of intensity against distance from the field centre.

    This is the number illumination correction exists to drive to zero, and
    the same statistic is used on pixels (the QC images) and on objects (the
    scientific claim: the same cell must not measure brighter in the middle of
    the field).

    Intensities are divided by their own mean and the radius is normalised so
    that 0 is the centre of the field and 1 is a corner, so the slope reads as
    *the fraction of the mean intensity gained or lost between the centre of
    the field and its corner*. -0.30 means a corner object measures 30 % of
    the mean below a central one.

    :param intensities: one value per position.
    :param coordinates: ``(N, 2)`` array of ``(row, col)`` positions in pixels.
    :param shape: ``(Y, X)`` shape of the field the positions live in.
    :returns: the slope, or 0.0 when there is nothing to fit.
    """
    values = np.asarray(intensities, dtype=np.float64).ravel()
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    if values.size < 2 or values.size != len(coordinates):
        return 0.0
    mean = values.mean()
    if not np.isfinite(mean) or mean == 0:
        return 0.0
    height, width = shape
    rows = (coordinates[:, 0] - (height - 1) / 2.0) / max((height - 1) / 2.0, 1e-9)
    cols = (coordinates[:, 1] - (width - 1) / 2.0) / max((width - 1) / 2.0, 1e-9)
    radius = np.sqrt(rows ** 2 + cols ** 2) / math.sqrt(2.0)
    design = np.stack([np.ones_like(radius), radius], axis=1)
    solution, *_ = np.linalg.lstsq(design, values / mean, rcond=None)
    return float(solution[1])


def _image_slope(image: np.ndarray, factor: int,
                 shape: Tuple[int, int]) -> float:
    """:func:`position_intensity_slope` over every binned pixel of ``image``."""
    rows = _bin_centres(shape[0], factor)[:image.shape[0]]
    cols = _bin_centres(shape[1], factor)[:image.shape[1]]
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    coordinates = np.stack([grid_rows.ravel(), grid_cols.ravel()], axis=1)
    return position_intensity_slope(image.ravel(), coordinates, shape)


def illumination_qc(model: IlluminationModel, src, *,
                    channels: Optional[Sequence[int]] = None,
                    save_dir: Optional[str] = None,
                    max_fields: int = 25,
                    verbose: bool = True) -> Dict[str, Any]:
    """Show that the correction worked, and say by how much.

    Three things, per plate and per channel:

    * **the estimated field as an image**, so a lamp profile that is really a
      dirty objective is visible rather than inferred;
    * **the position-versus-intensity trend before and after**, measured on
      the same fields the estimate came from -- the curve that should be flat
      after correction and is not before it;
    * **a number**: the residual slope of intensity against distance from the
      centre of the field, before and after, and the percentage of that bias
      the correction removed.

    :param model: the estimated model.
    :param src: the merged field folder(s) to measure the trend on.
    :param channels: channels to report; defaults to the model's own.
    :param save_dir: where the figure goes. Defaults to
        ``<first src>/../illumination``. Pass ``''`` to skip the figure and
        compute only the numbers.
    :param max_fields: fields per plate to measure the trend over.
    :param verbose: print the per-channel summary.
    :returns: ``{plate: {channel: {...metrics...}}}`` with, per channel,
        ``slope_before``, ``slope_after``, ``bias_removed_pct``,
        ``nonuniformity_pct``, ``gain_min``, ``gain_max`` and ``n_fields``;
        plus, when a figure was written, one extra key ``'_figures'`` mapping
        each plate to the path of its PNG.
    """
    grouped = _merged_files(src)
    if model.per_plate:
        plates = {plate: paths for plate, paths in grouped.items()
                  if plate in model.fields}
    else:
        plates = {ALL_PLATES: [path for paths in grouped.values()
                               for path in paths]}
    if save_dir is None:
        sources = _source_folders(src)
        save_dir = os.path.join(os.path.dirname(sources[0]), 'illumination')

    report: Dict[str, Any] = {}
    for plate in sorted(plates):
        item = model.field_for(plate)
        wanted = [int(c) for c in (channels if channels is not None
                                   else item.channels)]
        paths = _sample(sorted(plates[plate]), max_fields)
        stack = []
        factor = item.bin_size
        for path in paths:
            binned, _factor, shape = _read_binned_field(path, wanted, factor,
                                                        256)
            if shape == item.shape:
                stack.append(binned)
        if not stack:
            continue
        stack = np.stack(stack, axis=0)

        metrics = {}
        panels = {}
        for position, channel in enumerate(wanted):
            observed = _relative_profile(stack[:, position])
            gain_full = 1.0 / item.flatfield[item.index_of(channel)]
            gain_binned = _bin_image(gain_full, factor)
            corrected = observed * gain_binned[:observed.shape[0],
                                               :observed.shape[1]]
            before = _image_slope(observed, factor, item.shape)
            after = _image_slope(corrected, factor, item.shape)
            removed = (100.0 * (1.0 - abs(after) / abs(before))
                       if abs(before) > 1e-12 else 0.0)
            plane = item.flatfield[item.index_of(channel)]
            low, high = np.percentile(plane, [2, 98])
            metrics[int(channel)] = {
                'slope_before': before,
                'slope_after': after,
                'bias_removed_pct': removed,
                'nonuniformity_pct': float(100 * (high - low) /
                                           max(plane.mean(), 1e-12)),
                'gain_min': float(plane.min()),
                'gain_max': float(plane.max()),
                'n_fields': int(stack.shape[0]),
            }
            panels[int(channel)] = (plane, observed, corrected)
            if verbose:
                print(f"plate {plate}, channel {channel}: "
                      f"position-intensity slope {before:+.4f} -> "
                      f"{after:+.4f} per corner-radius "
                      f"({removed:.1f}% of the bias removed), field "
                      f"non-uniformity "
                      f"{metrics[int(channel)]['nonuniformity_pct']:.1f}%")
        report[plate] = metrics
        if save_dir:
            report.setdefault('_figures', {})[plate] = _write_qc_figure(
                plate, item, wanted, panels, metrics, save_dir, factor)
    return report


def _write_qc_figure(plate, item, channels, panels, metrics, save_dir,
                     factor) -> str:
    """Render one figure per plate: field, trend before/after, residual.

    Uses the object-oriented matplotlib API rather than pyplot: this can be
    called from a worker or a headless run, and pyplot's global figure
    registry is a leak in both.
    """
    from matplotlib.figure import Figure

    os.makedirs(save_dir, exist_ok=True)
    rows = max(len(channels), 1)
    figure = Figure(figsize=(13, 3.4 * rows), dpi=120)
    for index, channel in enumerate(channels):
        plane, observed, corrected = panels[int(channel)]
        stats = metrics[int(channel)]

        axis = figure.add_subplot(rows, 3, index * 3 + 1)
        image = axis.imshow(plane, cmap='viridis')
        axis.set_title(f'plate {plate} ch{channel}: estimated field\n'
                       f'gain {stats["gain_min"]:.2f}-{stats["gain_max"]:.2f}, '
                       f'non-uniformity {stats["nonuniformity_pct"]:.1f}%',
                       fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)

        axis = figure.add_subplot(rows, 3, index * 3 + 2)
        radius, before = _radial_profile(observed, factor, item.shape)
        _, after = _radial_profile(corrected, factor, item.shape)
        axis.plot(radius, before, 'o-', ms=3, label='before')
        axis.plot(radius, after, 's-', ms=3, label='after')
        axis.axhline(1.0, color='0.6', lw=0.8, ls='--')
        axis.set_xlabel('distance from field centre (0 = centre, 1 = corner)',
                        fontsize=8)
        axis.set_ylabel('mean intensity / field mean', fontsize=8)
        axis.set_title(f'position-intensity trend\nslope '
                       f'{stats["slope_before"]:+.3f} -> '
                       f'{stats["slope_after"]:+.3f} '
                       f'({stats["bias_removed_pct"]:.1f}% removed)',
                       fontsize=9)
        axis.legend(fontsize=8)

        axis = figure.add_subplot(rows, 3, index * 3 + 3)
        span = float(np.max(np.abs(observed - 1.0))) or 0.1
        image = axis.imshow(corrected, cmap='coolwarm', vmin=1 - span,
                            vmax=1 + span)
        axis.set_title('after correction (same scale as the\nobserved '
                       'deviation before it)', fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    path = os.path.join(save_dir, f'illumination_qc_{plate}.png')
    figure.savefig(path)
    return path


def _radial_profile(image: np.ndarray, factor: int,
                    shape: Tuple[int, int], bins: int = 20):
    """Mean of ``image`` in ``bins`` rings of normalised corner-radius."""
    rows = _bin_centres(shape[0], factor)[:image.shape[0]]
    cols = _bin_centres(shape[1], factor)[:image.shape[1]]
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
    normalised_rows = ((grid_rows - (shape[0] - 1) / 2.0) /
                       max((shape[0] - 1) / 2.0, 1e-9))
    normalised_cols = ((grid_cols - (shape[1] - 1) / 2.0) /
                       max((shape[1] - 1) / 2.0, 1e-9))
    radius = np.sqrt(normalised_rows ** 2 + normalised_cols ** 2) / math.sqrt(2)
    edges = np.linspace(0, radius.max() + 1e-9, bins + 1)
    which = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    values = image.ravel() / max(float(image.mean()), 1e-12)
    centres, means = [], []
    for index in range(bins):
        selected = values[which == index]
        if selected.size:
            centres.append(0.5 * (edges[index] + edges[index + 1]))
            means.append(float(selected.mean()))
    return np.asarray(centres), np.asarray(means)


# ---------------------------------------------------------------------------
# The settings-driven entry point
# ---------------------------------------------------------------------------

def prepare_illumination_correction(settings: Mapping[str, Any], *,
                                    verbose: Optional[bool] = None):
    """Estimate, save, enable and QC the correction from a settings dict.

    The one call a pipeline makes before ``measure_crop``. It does nothing at
    all -- and returns None -- unless ``settings['illumination_correction']``
    is True, which is the shipped default.

    :param settings: a ``measure_crop`` settings dict. Reads
        ``illumination_correction``, ``illumination_model``,
        ``illumination_estimator``, ``illumination_degree``,
        ``illumination_per_plate``, ``illumination_max_fields``,
        ``illumination_dark``, ``illumination_on_missing``,
        ``illumination_qc``, plus ``src`` and ``channels``.
    :param verbose: overrides ``settings['verbose']``.
    :returns: the :class:`IlluminationModel` that was enabled, or None.
    """
    if not settings.get('illumination_correction', False):
        return None
    talk = settings.get('verbose', True) if verbose is None else verbose
    src = settings.get('src')
    if not src:
        raise IlluminationError(
            "illumination_correction is on but settings['src'] is empty; "
            "there is nothing to estimate the illumination field from.")
    existing = str(settings.get('illumination_model', '') or '').strip()
    if existing:
        model = IlluminationModel.load(existing)
        model_path: Optional[str] = existing
    else:
        model = estimate_illumination(
            src,
            channels=settings.get('channels') or [],
            per_plate=bool(settings.get('illumination_per_plate', True)),
            estimator=str(settings.get('illumination_estimator', 'polynomial')),
            degree=int(settings.get('illumination_degree', 4)),
            max_fields=int(settings.get('illumination_max_fields', 50)),
            dark=float(settings.get('illumination_dark', 0.0)),
            verbose=talk)
        model_path = None
    folder = os.path.join(os.path.dirname(_source_folders(src)[0]),
                          'illumination')
    if settings.get('illumination_qc', True):
        illumination_qc(model, src, save_dir=folder, verbose=talk)
    enable_illumination_correction(
        model_path if model_path else model,
        path=os.path.join(folder, 'illumination_model.npz'),
        on_missing=str(settings.get('illumination_on_missing', 'error')),
        verbose=talk)
    return model


def illumination_settings(settings=None):
    """Defaults for illumination correction. Registered through the seam.

    :param settings: values to seed, exactly like every ``set_default_*`` in
        :mod:`spacr.settings`.
    :returns: the settings dict with the illumination defaults applied.
    """
    settings = dict(settings or {})
    settings.setdefault('src', '')
    settings.setdefault('channels', [0, 1, 2])
    settings.setdefault('illumination_correction', False)
    settings.setdefault('illumination_model', '')
    settings.setdefault('illumination_estimator', 'polynomial')
    settings.setdefault('illumination_degree', 4)
    settings.setdefault('illumination_per_plate', True)
    settings.setdefault('illumination_max_fields', 50)
    settings.setdefault('illumination_dark', 0.0)
    settings.setdefault('illumination_on_missing', 'error')
    settings.setdefault('illumination_qc', True)
    return settings


_TOOLTIPS = {
    'illumination_correction': (
        '(bool) - Estimate the uneven illumination of the microscope from the '
        'fields themselves and divide it out before any intensity feature is '
        'measured. Off by default. On, the same cell measures the same '
        'wherever it sits in the field of view, which is what removes the '
        'position-dependent bias behind plate edge effects. Default False.'),
    'illumination_model': (
        '(str) - Path to an illumination model saved earlier. Empty means '
        'estimate a fresh one from the fields in src, which is what you want '
        'unless you are re-measuring a plate and must reproduce the exact '
        'correction an earlier run applied. Default empty.'),
    'illumination_estimator': (
        "(str) - How the smooth field is fitted to the across-field median: "
        "'polynomial' fits a low-order surface, which cannot bend around a "
        "cell and is the right choice for a lamp profile plus a vignette; "
        "'smooth' Gaussian-blurs the median instead and can follow a dust "
        "shadow the polynomial would miss. Default polynomial."),
    'illumination_degree': (
        '(int) - Order of the fitted illumination surface. 4 gives fifteen '
        'terms, enough for a lamp profile, a vignette and a tilt. Raising it '
        'lets the surface follow finer structure and, past about 6, start '
        'absorbing the cells you are trying to measure. Default 4.'),
    'illumination_per_plate': (
        '(bool) - Estimate one illumination field per plate rather than one '
        'for every plate together. Lamp age, a re-seated filter cube or a '
        'different objective change the field between acquisition sessions, '
        'so pooling two sessions estimates neither of them well. Default True.'),
    'illumination_max_fields': (
        '(int) - How many fields per plate the estimate reads, sampled evenly '
        'across the plate. More fields make the across-field median a better '
        'object rejector and cost linear time; below about ten, cells start '
        'surviving into the gain map. Default 50.'),
    'illumination_dark': (
        '(float) - Camera dark offset in raw counts, subtracted before the '
        'gain is applied. Leave at zero unless you measured it from a dark '
        'frame: it is not identifiable from the images themselves, and '
        'guessing it subtracts real background signal. Default 0.0.'),
    'illumination_on_missing': (
        "(str) - What to do with a field whose plate the model does not "
        "cover: 'error' fails that field and stamps the run incomplete, "
        "'skip' measures it uncorrected. Default error, because corrected and "
        "uncorrected rows sharing one table is worse than a failed field."),
    'illumination_qc': (
        '(bool) - Write the QC figure beside the model: the estimated field '
        'as an image, the intensity-versus-position trend before and after, '
        'and the percentage of the position bias the correction removed. '
        'Cheap, and the only way to see that it worked. Default True.'),
}

_TYPES = {
    'illumination_correction': bool,
    'illumination_model': str,
    'illumination_estimator': str,
    'illumination_degree': int,
    'illumination_per_plate': bool,
    'illumination_max_fields': int,
    'illumination_dark': float,
    'illumination_on_missing': str,
    'illumination_qc': bool,
}

_DESCRIPTION = (
    'Illumination / flat-field correction. Estimates the microscope\'s '
    'uneven illumination from the fields of a plate and divides it out '
    'before any intensity feature is measured, so the same cell measures '
    'the same wherever it sat in the field of view.'
)


def register_illumination_settings(replace: bool = False) -> bool:
    """Register the illumination settings through the defaults seam.

    Called once at import. Uses
    :func:`spacr.settings.register_defaults` rather than appending to
    ``spacr/settings.py``, so this module owns its own knobs.

    Types, tooltips and the module description are contributed; **categories
    deliberately are not**. ``spacr.settings.categories`` is one shared,
    ordered map that every settings panel walks, and its growth is guarded by
    an exact-equality test against a hand-kept list. A key contributed at
    *import* time is in that map only in a session that imported this module,
    so contributing categories would make that test's result depend on which
    files pytest was pointed at. No shipped panel offers these keys yet
    either; the commit that adds an Illumination screen is the one that
    should file them under a heading, and can then declare the growth
    honestly.

    :param replace: re-register over an existing registration.
    :returns: True if it registered, False if it was already registered.
    """
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY) and not replace:
        return False
    register_defaults(
        APP_KEY, illumination_settings, replace=replace,
        expected_types=_TYPES, tooltips=_TOOLTIPS,
        description=_DESCRIPTION)
    return True


register_illumination_settings()
