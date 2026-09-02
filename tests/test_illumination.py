"""Illumination / flat-field correction, held to the claim it makes.

The claim is that the same cell measures the same wherever it sits in the
field of view. So every test here is built the same way: identical, uniform
objects are scattered over a synthetic plate, a **known** illumination field
is multiplied into the pixels, and the tests ask two questions of the module.

1. Does it recover the field it was given? (``max |estimated - true| / true``,
   with a stated tolerance per estimator.)
2. Does correcting flatten the thing that actually matters -- the trend of
   *measured object intensity* against *position in the field*? That trend is
   a single number, the slope of mean-normalised intensity against normalised
   distance from the centre of the field (see
   :func:`spacr.illumination.position_intensity_slope`), and the tests assert
   on it before and after. A correction that estimated a beautiful field and
   applied it to nothing would pass question 1 and fail question 2.

The third thing tested is the one that is silent when it breaks: a hook
registered in the parent process is a no-op in every ``spawn`` worker, the run
completes, and the user believes numbers are corrected that are not. There is
a positive test (the correction reaches a real spawned worker and the numbers
come back corrected) and its negative control (a parent-only registration
reaches nothing, and the control proves the positive test can fail).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sqlite3

import numpy as np
import pytest

from spacr import illumination as ill
from spacr import measure_hooks as mh
from spacr.measure_hooks import PreprocessingContext


# ---------------------------------------------------------------------------
# Fixtures and synthetic data with a known illumination field
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _pristine(monkeypatch):
    """Empty hook registries and no illumination environment, before and after.

    Both are process-global, and a hook or an environment variable leaking out
    of one test would silently change the next test's measurements -- which is
    the exact failure mode this module exists to prevent.
    """
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR,
                 'SPACR_START_METHOD'):
        monkeypatch.delenv(name, raising=False)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        os.environ.pop(name, None)


#: Background and in-object signal of the synthetic fields, in raw counts.
BACKGROUND = 120.0
SIGNAL = 1200.0


def quadratic_vignette(shape, strength=0.45):
    """A radially quadratic field, normalised to mean 1.

    Exactly representable by the polynomial estimator, so it is the case where
    a tight tolerance is a fair thing to ask for.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    radius = ((((yy - (height - 1) / 2) / ((height - 1) / 2)) ** 2 +
               ((xx - (width - 1) / 2) / ((width - 1) / 2)) ** 2) / 2.0)
    flat = 1.0 - strength * radius
    return flat / flat.mean()


def lamp_vignette(shape):
    """cos^4 falloff plus an off-centre lamp hot spot, normalised to mean 1.

    Deliberately *not* a polynomial: a degree-4 surface can only approximate
    it, so this is what the honest tolerance is set against. The range is
    about 0.67-1.28, i.e. a 50 % non-uniformity, which is a bad but entirely
    ordinary widefield microscope.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    u = (xx - (width - 1) / 2) / ((width - 1) / 2)
    v = (yy - (height - 1) / 2) / ((height - 1) / 2)
    flat = np.cos(np.arctan(np.sqrt(u ** 2 + v ** 2) * 0.42)) ** 4
    flat = flat * (1.0 + 0.10 * np.exp(-(((u - 0.35) ** 2 +
                                          (v + 0.3) ** 2) / 0.15)))
    return flat / flat.mean()


def write_plate(folder, flat, *, plate='plate1', n_fields=20, seed=0,
                n_objects=8, radius=8, shape=None, noise=False,
                channel_flats=None):
    """Write merged fields of identical objects under a known illumination field.

    Every object is the same size and the same true intensity, so any spread
    the tests see in the measured intensities is the illumination and nothing
    else. Layout of the merged stack: channel 0 and 1 are intensity, 2 is the
    cell mask, 3 the nucleus mask, 4 the pathogen mask -- what
    ``measure_crop`` expects.

    :returns: the folder, as a string.
    """
    flats = channel_flats if channel_flats is not None else [flat, flat]
    shape = shape or flats[0].shape
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for index in range(n_fields):
        cell = np.zeros(shape, np.uint16)
        nucleus = np.zeros(shape, np.uint16)
        pathogen = np.zeros(shape, np.uint16)
        for label in range(1, n_objects + 1):
            cy = int(rng.integers(radius + 4, shape[0] - radius - 4))
            cx = int(rng.integers(radius + 4, shape[1] - radius - 4))
            disc = (yy - cy) ** 2 + (xx - cx) ** 2
            cell[disc <= radius ** 2] = label
            nucleus[disc <= (radius // 3) ** 2] = label
            pathogen[disc <= 2 ** 2] = label
        planes = []
        for channel_flat in flats:
            true = np.full(shape, BACKGROUND)
            true[cell > 0] = BACKGROUND + SIGNAL
            observed = true * channel_flat
            if noise:
                observed = rng.poisson(observed).astype(float)
            planes.append(np.rint(observed).astype(np.uint16))
        data = np.stack(planes + [cell, nucleus, pathogen], axis=-1)
        np.save(os.path.join(folder, f'{plate}_A01_F{index:03d}.npy'),
                data.astype(np.uint16))
    return str(folder)


def object_intensities(folder, corrector=None, channels=(0, 1), channel=0):
    """Mean intensity and centroid of every object, corrected or not.

    This is what ``measure_crop`` computes, done here directly so the
    estimation tests do not have to pay for the whole measure pipeline.
    """
    channels = list(channels)
    values, coordinates = [], []
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.npy'):
            continue
        data = np.load(os.path.join(folder, name))
        array = data[..., channels]
        if corrector is not None:
            array = corrector(array, PreprocessingContext(
                file_name=name, channels=channels, settings={}))
        array = np.asarray(array)
        mask = data[..., 2]
        for label in np.unique(mask):
            if label == 0:
                continue
            selected = mask == label
            values.append(float(array[..., channel][selected].mean()))
            rows, cols = np.nonzero(selected)
            coordinates.append((rows.mean(), cols.mean()))
    return np.asarray(values), np.asarray(coordinates)


def slope_of(folder, corrector, shape, channels=(0, 1), channel=0):
    """The position-versus-intensity slope over every object in ``folder``."""
    values, coordinates = object_intensities(folder, corrector, channels,
                                             channel)
    return ill.position_intensity_slope(values, coordinates, shape), values


def hand_model(flats, *, plate='plate1', channels=(0,), dark=0.0):
    """An :class:`IlluminationModel` built from given fields, no estimation."""
    field = ill.IlluminationField(
        plate=plate,
        channels=tuple(int(c) for c in channels),
        flatfield=np.stack([np.asarray(f, dtype=np.float32) for f in flats]),
        dark=np.full(len(channels), float(dark), dtype=np.float32),
        n_fields=1, estimator='polynomial', degree=4, bin_size=1)
    return ill.IlluminationModel(fields={plate: field}, meta={})


def context(name='plate1_A01_F000', channels=(0,), **kwargs):
    """A :class:`PreprocessingContext` for a hand-made array."""
    return PreprocessingContext(file_name=name, channels=list(channels),
                                settings={}, **kwargs)


def segmentation_prepared(tmp_path, *, legacy=False):
    """A tiny saved model with the segmentation application contract."""
    model = hand_model([
        np.asarray([[0.5, 1.0], [1.0, 1.0]], dtype=np.float32)
    ])
    if not legacy:
        model.meta.update({
            'application_contract_version': 1,
            'channel_index_space': 'persisted-intensity-axis',
            'estimated_from_intensity_state': 'raw',
        })
    path = model.save(str(tmp_path / 'illumination' / 'model.npz'))
    digest = hashlib.sha256((tmp_path / 'illumination' / 'model.npz').read_bytes())
    return ill.PreparedIllumination(
        model=model,
        corrector=ill.IlluminationCorrector(model, verbose=False),
        model_path=path,
        model_sha256=digest.hexdigest(),
    )


# ---------------------------------------------------------------------------
# 1. Does it recover the field it was given?
# ---------------------------------------------------------------------------

def test_a_known_quadratic_vignette_is_recovered_to_within_one_percent(tmp_path):
    """The estimator's own arithmetic, on the case it can represent exactly.

    A radially quadratic field is inside the degree-4 polynomial's span, so
    what is being measured here is whether the across-field median rejects the
    objects and whether the binning/coordinate arithmetic lines up -- not how
    well a surface approximates a curve. Anything worse than 1 % is a bug, not
    an approximation.
    """
    shape = (256, 256)
    truth = quadratic_vignette(shape)
    merged = write_plate(tmp_path / 'merged', truth, n_fields=20)

    model = ill.estimate_illumination(merged, channels=[0, 1], verbose=False)
    field = model.field_for('plate1')
    estimate = field.flatfield[0]

    error = np.abs(estimate - truth) / truth
    assert error.max() < 0.01, f'worst pixel off by {error.max():.3%}'
    assert error.mean() < 0.002
    # Normalised to mean 1, so a corrected number stays on the plate's own
    # intensity scale instead of being silently rescaled.
    assert estimate.mean() == pytest.approx(1.0, abs=1e-4)
    assert estimate.shape == shape
    assert field.n_fields == 20 and field.estimator == 'polynomial'


def test_a_lamp_shaped_vignette_is_recovered_within_the_stated_tolerance(tmp_path):
    """The honest case: a cos^4 falloff and a hot spot, with photon noise.

    Not representable by a degree-4 polynomial, so the surface approximates
    it. The tolerances below are what that approximation is actually worth --
    8 % on the single worst pixel, 1.5 % on average -- and the smooth
    estimator, which follows structure the polynomial cannot, does better on
    the worst pixel. Both numbers are asserted so that a regression in either
    estimator shows up as a failure rather than as a slightly worse figure
    nobody reads.
    """
    shape = (450, 600)
    truth = lamp_vignette(shape)
    merged = write_plate(tmp_path / 'merged', truth, plate='plate2',
                         n_fields=25, n_objects=12, radius=10, noise=True)

    fitted = ill.estimate_illumination(merged, channels=[0], verbose=False)
    smoothed = ill.estimate_illumination(merged, channels=[0],
                                         estimator='smooth', verbose=False)

    for model, worst, average in ((fitted, 0.08, 0.015),
                                  (smoothed, 0.07, 0.015)):
        estimate = model.field_for('plate2').flatfield[0]
        error = np.abs(estimate - truth) / truth
        assert error.max() < worst, f'worst pixel off by {error.max():.3%}'
        assert error.mean() < average
        assert estimate.shape == shape

    # The statistic was computed on a binned grid (600 / 256 -> factor 3) and
    # the surface returned at full resolution; both matter, because a
    # coordinate mapping that ignored the binning would land the field a
    # pixel and a half off and nothing above would notice.
    assert fitted.field_for('plate2').bin_size == 3


def test_each_channel_gets_its_own_field(tmp_path):
    """Two fluorophores go through different filters and vignette differently.

    A single field applied to both would leave one channel over-corrected and
    the other under-corrected, and the plate-level artefact would survive in
    whichever channel lost.
    """
    shape = (256, 256)
    first = quadratic_vignette(shape, strength=0.5)
    second = lamp_vignette(shape)
    merged = write_plate(tmp_path / 'merged', first, n_fields=20,
                         channel_flats=[first, second])

    field = ill.estimate_illumination(merged, channels=[0, 1],
                                      verbose=False).field_for('plate1')

    assert field.channels == (0, 1)
    assert np.abs(field.flatfield[0] - first).max() < 0.02
    assert np.abs(field.flatfield[1] - second).max() < 0.08
    # And they really are different fields, not one field stored twice.
    assert np.abs(field.flatfield[0] - field.flatfield[1]).max() > 0.05


def test_each_plate_gets_its_own_field_by_default(tmp_path):
    """Illumination differs between acquisition sessions, so plates do not share.

    Pooling two plates estimates the average of two different microscopes and
    corrects neither -- which is why per-plate is the default and pooling is
    the thing you have to ask for.
    """
    shape = (256, 256)
    gentle = quadratic_vignette(shape, strength=0.2)
    harsh = quadratic_vignette(shape, strength=0.6)
    merged = tmp_path / 'merged'
    write_plate(merged, gentle, plate='plateA', n_fields=15, seed=1)
    write_plate(merged, harsh, plate='plateB', n_fields=15, seed=2)

    model = ill.estimate_illumination(merged, channels=[0], verbose=False)
    assert sorted(model.fields) == ['plateA', 'plateB']
    assert model.per_plate is True
    assert np.abs(model.field_for('plateA').flatfield[0] - gentle).max() < 0.01
    assert np.abs(model.field_for('plateB').flatfield[0] - harsh).max() < 0.01

    pooled = ill.estimate_illumination(merged, channels=[0], per_plate=False,
                                       verbose=False)
    assert list(pooled.fields) == [ill.ALL_PLATES]
    assert pooled.per_plate is False
    # One field for both plates matches neither: it is the average, and it is
    # visibly wrong for each -- which is the cost of pooling, made explicit.
    averaged = pooled.field_for('plateA').flatfield[0]
    assert np.abs(averaged - gentle).max() > 0.05
    assert np.abs(averaged - harsh).max() > 0.05
    # Any plate name resolves against a pooled model, including one that was
    # never seen at estimation time.
    assert pooled.field_for('plateZ') is pooled.field_for('plateA')


def test_a_z_stack_is_estimated_and_corrected_with_one_two_dimensional_field(
        tmp_path):
    """Illumination is a property of the optics in x and y, not of z."""
    shape = (48, 48)
    truth = quadratic_vignette(shape, strength=0.4)
    merged = tmp_path / 'merged'
    os.makedirs(merged)
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for index in range(16):
        volume = []
        mask = np.zeros(shape, np.uint16)
        for label in range(1, 4):
            cy = int(rng.integers(8, shape[0] - 8))
            cx = int(rng.integers(8, shape[1] - 8))
            mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4 ** 2] = label
        for _z in range(3):
            plane = np.full(shape, BACKGROUND)
            plane[mask > 0] = BACKGROUND + SIGNAL
            volume.append(np.rint(plane * truth).astype(np.uint16))
        stack = np.stack(volume, axis=0)                     # (Z, Y, X)
        data = np.stack([stack, np.broadcast_to(mask, stack.shape)], axis=-1)
        np.save(merged / f'plate1_A01_F{index:03d}.npy', data.astype(np.uint16))

    model = ill.estimate_illumination(merged, channels=[0], verbose=False)
    field = model.field_for('plate1')
    assert field.shape == shape
    assert np.abs(field.flatfield[0] - truth).max() < 0.02

    volume = np.load(merged / 'plate1_A01_F000.npy')[..., [0]]
    corrected = ill.IlluminationCorrector(model)(
        volume, context(name='plate1_A01_F000', channels=[0], volumetric=True))
    assert corrected.shape == volume.shape and corrected.dtype == volume.dtype
    # Every slice got the same 2-D gain, so a flat volume stays flat in z.
    background = corrected[..., 0][:, :4, :4].astype(float)
    assert background.std(axis=0).max() < 1.0


# ---------------------------------------------------------------------------
# 2. Does correcting flatten the position-versus-intensity trend?
# ---------------------------------------------------------------------------

def test_the_correction_flattens_the_position_intensity_slope(tmp_path, capsys):
    """The headline claim, measured as a number before and after.

    Identical objects are scattered over the field. Uncorrected, an object in
    a corner measures well below one in the middle: that is the whole bias, and
    it is what leaks into every intensity feature and out into classification.
    The slope of mean-normalised intensity against normalised corner-distance
    is that bias in one number; correcting must drive it to a twentieth of
    what it was.
    """
    shape = (256, 256)
    truth = quadratic_vignette(shape, strength=0.45)
    merged = write_plate(tmp_path / 'merged', truth, n_fields=20)
    model = ill.estimate_illumination(merged, channels=[0, 1], verbose=False)

    before, raw = slope_of(merged, None, shape)
    after, corrected = slope_of(merged, ill.IlluminationCorrector(model), shape)

    # The uncorrected data really does carry the artefact this removes; a test
    # whose "before" was already flat would prove nothing.
    assert before < -0.30, f'the synthetic bias is too weak to test: {before}'
    assert abs(after) < abs(before) / 20.0
    assert abs(after) < 0.02

    # And the spread across positions collapses: same object, same number.
    spread_before = raw.std() / raw.mean()
    spread_after = corrected.std() / corrected.mean()
    assert spread_before > 0.05
    assert spread_after < spread_before / 10.0
    print(f'slope {before:+.4f} -> {after:+.4f} per corner-radius; '
          f'object-intensity CV {spread_before:.4f} -> {spread_after:.4f}')
    assert 'slope' in capsys.readouterr().out


def test_the_slope_is_flattened_for_a_lamp_field_with_noise(tmp_path):
    """The same claim where the estimator cannot be exactly right.

    A cos^4 field plus a hot spot plus Poisson noise: the surface is an
    approximation, so the residual slope is not zero. It is still an order of
    magnitude smaller than the bias it replaced, which is the honest promise.
    """
    shape = (450, 600)
    truth = lamp_vignette(shape)
    merged = write_plate(tmp_path / 'merged', truth, plate='plate2',
                         n_fields=25, n_objects=12, radius=10, noise=True)
    model = ill.estimate_illumination(merged, channels=[0], verbose=False)

    before, raw = slope_of(merged, None, shape, channels=[0])
    after, corrected = slope_of(merged, ill.IlluminationCorrector(model),
                                shape, channels=[0])

    assert before < -0.40
    assert abs(after) < abs(before) / 10.0
    assert corrected.std() / corrected.mean() < 0.25 * (raw.std() / raw.mean())


def test_the_uncorrected_path_is_untouched(tmp_path):
    """With nothing registered, the measurement path sees the same object.

    The guard on "off by default": importing this module must not change a
    single number anywhere.
    """
    assert mh.preprocessing_hooks() == ()
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    assert mh.apply_preprocessing_hooks(array, context()) is array


# ---------------------------------------------------------------------------
# 3. The dtype round trip -- the hook contract's explicit decision
# ---------------------------------------------------------------------------

def test_the_correction_returns_the_input_dtype_and_rounds_to_nearest():
    """Round, never truncate: truncation biases every pixel down by ~0.5.

    Averaged over a 500-pixel object that does not wash out; it is a
    systematic shift in one direction, which is precisely the class of
    artefact this feature exists to remove.
    """
    model = hand_model([np.full((1, 3), 0.8, np.float32)])
    corrector = ill.IlluminationCorrector(model)
    array = np.array([[3, 100, 101]], dtype=np.uint16)[..., None]

    result = corrector(array, context(channels=[0]))

    assert result.dtype == np.uint16 and result.shape == array.shape
    # 3 * 1.25 = 3.75 -> 4 (truncation would give 3), 100 -> 125,
    # 101 * 1.25 = 126.25 -> 126.
    assert result[..., 0].tolist() == [[4, 125, 126]]
    assert corrector.stats['clipped_pixels'] == 0


def test_clipping_real_signal_is_counted_and_reported(capsys):
    """Silently clipping is a lie about the data, so it is counted and said.

    A gain above 1 at the edge of the field can push a bright-but-unsaturated
    pixel past the top of a uint16. Clipping is the only way to honour the
    hook contract's "same dtype", so the pixels it costs are reported rather
    than swallowed.
    """
    model = hand_model([np.full((1, 3), 0.5, np.float32)])   # gain 2.0
    corrector = ill.IlluminationCorrector(model)
    array = np.array([[40000, 65535, 100]], dtype=np.uint16)[..., None]

    result = corrector(array, context(channels=[0]))

    assert result[..., 0].tolist() == [[65535, 65535, 200]]
    # 40000 was real signal and is now clipped: counted. 65535 was already
    # saturated when the microscope wrote it -- that pixel was destroyed by
    # the acquisition, not by this correction, so it is not counted.
    assert corrector.stats['clipped_pixels'] == 1
    assert corrector.stats['clipped_fields'] == 1
    printed = capsys.readouterr().out
    assert 'clipped 1 pixel(s)' in printed
    assert 'real signal was lost' in printed
    assert 'clipped' in corrector.report()


def test_clipping_warnings_are_rate_limited_but_the_total_is_not(capsys):
    """One line per field over 384 wells is noise; a wrong total is a lie."""
    model = hand_model([np.full((1, 1), 0.5, np.float32)])
    corrector = ill.IlluminationCorrector(model)
    array = np.array([[40000]], dtype=np.uint16)[..., None]
    for index in range(12):
        corrector(array, context(name=f'plate1_A01_F{index:03d}'))

    assert corrector.stats['clipped_pixels'] == 12
    assert corrector.stats['clipped_fields'] == 12
    printed = capsys.readouterr().out
    assert printed.count('WARNING') == ill._MAX_CLIP_WARNINGS
    assert 'suppressed' in printed


def test_a_float_stack_keeps_its_float_dtype_and_is_never_clipped():
    """Nothing to round to and no range to leave, so neither happens."""
    model = hand_model([np.full((2, 2), 0.5, np.float32)])
    corrector = ill.IlluminationCorrector(model)
    array = np.full((2, 2, 1), 1e9, dtype=np.float32)

    result = corrector(array, context(channels=[0]))

    assert result.dtype == np.float32
    assert float(result.max()) == pytest.approx(2e9)
    assert corrector.stats['clipped_pixels'] == 0


def test_a_dark_offset_is_subtracted_before_the_gain_is_applied():
    """(observed - dark) / flat, in that order -- the model in the docstring."""
    model = hand_model([np.full((1, 2), 0.5, np.float32)], dark=100.0)
    corrector = ill.IlluminationCorrector(model)
    array = np.array([[300, 1100]], dtype=np.uint16)[..., None]

    result = corrector(array, context(channels=[0]))

    assert result[..., 0].tolist() == [[400, 2000]]


def test_the_hook_contract_is_satisfied_end_to_end(tmp_path):
    """apply_preprocessing_hooks checks shape and dtype; it must not complain.

    The hook machinery refuses a hook that returns the wrong dtype rather than
    coercing it, so this is the test that the corrector's own cast is
    acceptable to the pipeline it is plugged into.
    """
    shape = (64, 64)
    truth = quadratic_vignette(shape)
    merged = write_plate(tmp_path / 'merged', truth, n_fields=12, radius=5,
                         n_objects=4)
    model = ill.estimate_illumination(merged, channels=[0, 1], verbose=False)
    mh.register_preprocessing_hook(ill.IlluminationCorrector(model),
                                   name=ill.HOOK_NAME,
                                   priority=ill.HOOK_PRIORITY)

    array = np.load(merged + '/plate1_A01_F000.npy')[..., [0, 1]]
    result = mh.apply_preprocessing_hooks(
        array, context(name='plate1_A01_F000', channels=[0, 1]))

    assert result.dtype == array.dtype and result.shape == array.shape
    assert not np.array_equal(result, array)


# ---------------------------------------------------------------------------
# 4. Refusing to be silently wrong
# ---------------------------------------------------------------------------

def test_a_plate_the_model_does_not_cover_is_an_error(tmp_path):
    """Better a failed field than a table of half-corrected rows."""
    model = hand_model([np.full((4, 4), 0.8, np.float32)], plate='plateA')
    corrector = ill.IlluminationCorrector(model)
    array = np.full((4, 4, 1), 100, np.uint16)

    with pytest.raises(ill.IlluminationError, match='no illumination field'):
        corrector(array, context(name='plateB_A01_F001'))

    # ...and inside the pipeline it arrives as the hook error that fails the
    # field, names the hook and stamps the run incomplete.
    mh.register_preprocessing_hook(corrector, name=ill.HOOK_NAME)
    with pytest.raises(mh.MeasurementHookError, match=ill.HOOK_NAME):
        mh.apply_preprocessing_hooks(array, context(name='plateB_A01_F001'))


def test_skip_measures_the_field_uncorrected_and_says_so(capsys):
    """The opt-out exists, and it is not allowed to be quiet about itself."""
    model = hand_model([np.full((4, 4), 0.8, np.float32)], plate='plateA')
    corrector = ill.IlluminationCorrector(model, on_missing='skip')
    array = np.full((4, 4, 1), 100, np.uint16)

    result = corrector(array, context(name='plateB_A01_F001'))

    assert result is array
    assert corrector.stats['skipped'] == 1
    printed = capsys.readouterr().out
    assert 'UNCORRECTED' in printed and 'plateB' in printed


def test_a_channel_the_model_never_saw_is_an_error():
    """Correcting some channels and not others puts both in one table."""
    model = hand_model([np.full((4, 4), 0.8, np.float32)], channels=[0])
    corrector = ill.IlluminationCorrector(model)
    array = np.full((4, 4, 2), 100, np.uint16)

    with pytest.raises(ill.IlluminationError, match='channel 3'):
        corrector(array, context(channels=[0, 3]))


def test_a_field_of_the_wrong_shape_is_an_error():
    """A gain map cannot be stretched onto a different sensor geometry."""
    model = hand_model([np.full((8, 8), 0.8, np.float32)])
    corrector = ill.IlluminationCorrector(model)

    with pytest.raises(ill.IlluminationError, match='8x8'):
        corrector(np.full((4, 4, 1), 100, np.uint16), context())


@pytest.mark.parametrize('kwargs,fragment', [
    ({'channels': []}, 'at least one channel'),
    ({'channels': [0], 'estimator': 'wavelet'}, 'unknown illumination estimator'),
])
def test_estimation_refuses_what_it_cannot_do(tmp_path, kwargs, fragment):
    os.makedirs(tmp_path / 'merged')
    np.save(tmp_path / 'merged' / 'plate1_A01_F001.npy',
            np.zeros((8, 8, 2), np.uint16))
    with pytest.raises(ill.IlluminationError, match=fragment):
        ill.estimate_illumination(tmp_path / 'merged', **kwargs)


def test_estimation_refuses_an_empty_or_missing_source(tmp_path):
    os.makedirs(tmp_path / 'merged')
    with pytest.raises(ill.IlluminationError, match='no .npy fields'):
        ill.estimate_illumination(tmp_path / 'merged', channels=[0])
    with pytest.raises(ill.IlluminationError, match='not a folder'):
        ill.estimate_illumination(tmp_path / 'nope', channels=[0])


def test_a_channel_outside_the_merged_stack_is_named(tmp_path):
    os.makedirs(tmp_path / 'merged')
    np.save(tmp_path / 'merged' / 'plate1_A01_F001.npy',
            np.zeros((8, 8, 2), np.uint16))
    with pytest.raises(ill.IlluminationError, match='channel 5 does not exist'):
        ill.estimate_illumination(tmp_path / 'merged', channels=[5],
                                  verbose=False)


def test_a_model_round_trips_through_disk(tmp_path):
    """A spawn worker can only reach the model through the file system."""
    shape = (64, 64)
    merged = write_plate(tmp_path / 'merged', quadratic_vignette(shape),
                         n_fields=12, radius=5, n_objects=4)
    model = ill.estimate_illumination(merged, channels=[0, 1], verbose=False)

    path = model.save(str(tmp_path / 'illumination' / 'model.npz'))
    restored = ill.load_illumination_model(path)

    assert sorted(restored.fields) == sorted(model.fields)
    original = model.field_for('plate1')
    copy = restored.field_for('plate1')
    np.testing.assert_array_equal(copy.flatfield, original.flatfield)
    assert copy.channels == original.channels
    assert copy.n_fields == original.n_fields
    assert copy.bin_size == original.bin_size
    assert restored.meta['channels'] == [0, 1]
    assert restored.describe() == model.describe()


def test_an_unreadable_model_is_refused_rather_than_ignored(tmp_path):
    """A worker that cannot load the model must not measure uncorrected."""
    with pytest.raises(ill.IlluminationError, match='does not exist'):
        ill.load_illumination_model(str(tmp_path / 'nope.npz'))
    junk = tmp_path / 'junk.npz'
    junk.write_bytes(b'not an npz')
    with pytest.raises(ill.IlluminationError, match='could not be read'):
        ill.load_illumination_model(str(junk))


@pytest.mark.parametrize('name,plate', [
    ('plate1_A01_F001.npy', 'plate1'),
    ('/data/exp/merged/plate12_B03_F007.npy', 'plate12'),
    ('single.npy', 'single'),
])
def test_the_plate_is_read_from_the_field_name(name, plate):
    assert ill.plate_of_field(name) == plate


# ---------------------------------------------------------------------------
# 5. Reaching the workers -- the failure that is silent by construction
# ---------------------------------------------------------------------------

def _apply_in_this_process(payload):
    """Run in a worker: apply whatever hooks *this* process has installed.

    Defined at module scope because a ``spawn`` worker gets it by importing
    this module and looking it up by name; a closure would not survive.

    :returns: ``(hook names this process sees, the resulting array)``.
    """
    from spacr.measure_hooks import (PreprocessingContext,
                                     apply_preprocessing_hooks,
                                     preprocessing_hooks)
    array, file_name, channels = payload
    names = [entry.name for entry in preprocessing_hooks()]
    result = apply_preprocessing_hooks(
        array, PreprocessingContext(file_name=file_name, channels=channels,
                                    settings={}))
    return names, np.asarray(result)


def test_the_correction_reaches_a_real_spawn_worker(tmp_path):
    """The deliverable: a cold interpreter corrects the pixels for itself.

    A ``spawn`` worker inherits no imports, no globals and no hook registry.
    It inherits the environment -- which is why enabling the correction writes
    ``SPACR_MEASURE_HOOKS`` and the model path there, and why the worker below
    ends up with the hook installed without the parent having sent it
    anything.
    """
    shape = (64, 64)
    merged = write_plate(tmp_path / 'merged', quadratic_vignette(shape),
                         n_fields=12, radius=5, n_objects=4)
    model = ill.estimate_illumination(merged, channels=[0], verbose=False)
    ill.enable_illumination_correction(
        model, path=str(tmp_path / 'illumination' / 'model.npz'), verbose=False)

    array = np.load(merged + '/plate1_A01_F000.npy')[..., [0]]
    expected = ill.IlluminationCorrector(model)(
        array, context(name='plate1_A01_F000', channels=[0]))

    with mp.get_context('spawn').Pool(1) as pool:
        names, corrected = pool.apply(
            _apply_in_this_process, ((array, 'plate1_A01_F000', [0]),))

    assert names == [ill.HOOK_NAME], (
        'the spawned worker did not install the illumination correction, so '
        'every field it measured would be uncorrected')
    np.testing.assert_array_equal(corrected, expected)
    assert corrected.dtype == array.dtype
    assert not np.array_equal(corrected, array)


def test_a_parent_only_registration_reaches_no_spawn_worker(tmp_path):
    """The negative control, which is what makes the test above mean anything.

    Registering the hook through the Python API and nothing else is the
    natural thing to write and a silent no-op in every ``spawn`` worker. If
    this test ever starts passing by accident -- because something else set
    the environment -- the positive test above would be proving nothing.
    """
    model = hand_model([np.full((8, 8), 0.5, np.float32)])
    mh.register_preprocessing_hook(ill.IlluminationCorrector(model),
                                   name=ill.HOOK_NAME)
    array = np.full((8, 8, 1), 100, np.uint16)

    with mp.get_context('spawn').Pool(1) as pool:
        names, corrected = pool.apply(
            _apply_in_this_process, ((array, 'plate1_A01_F000', [0]),))

    assert names == []
    np.testing.assert_array_equal(corrected, array)
    # ...and the module says so, in advance, rather than letting it happen.
    ok, message = ill.worker_delivery_status('spawn')
    assert ok is False
    assert 'measured uncorrected' in message


def test_enable_writes_the_environment_every_start_method_inherits(tmp_path,
                                                                   monkeypatch):
    """Three things, and the environment is the one that reaches a worker."""
    import sys
    import types

    other = types.ModuleType('spacr_other_extension')
    other.install = lambda: mh.register_preprocessing_hook(
        lambda array, ctx: array, name='other-extension')
    monkeypatch.setitem(sys.modules, 'spacr_other_extension', other)

    model = hand_model([np.full((8, 8), 0.5, np.float32)])
    os.environ[mh.HOOKS_ENV_VAR] = 'spacr_other_extension:install'

    name = ill.enable_illumination_correction(
        model, path=str(tmp_path / 'model.npz'), verbose=False)

    assert name == ill.HOOK_NAME
    assert os.path.isfile(os.environ[ill.MODEL_ENV_VAR])
    # Appended, not assigned: another extension may already be in there.
    entries = os.environ[mh.HOOKS_ENV_VAR].split(',')
    assert entries == ['spacr_other_extension:install', ill.INSTALLER_ENTRY]
    # Installed here through that same route, so it is tagged 'env' and
    # measure_crop's start-method warning knows not to shout about it.
    registered = {entry.name: entry for entry in mh.preprocessing_hooks()}
    assert registered[ill.HOOK_NAME].source == 'env'
    assert registered[ill.HOOK_NAME].priority == ill.HOOK_PRIORITY
    assert mh.warn_if_hooks_will_not_reach_workers('spawn') is False

    ok, message = ill.worker_delivery_status('spawn')
    assert ok is True and 'workers install it themselves' in message

    assert ill.disable_illumination_correction() is True
    # The other extension's entry and its hook are left exactly alone.
    assert os.environ[mh.HOOKS_ENV_VAR] == 'spacr_other_extension:install'
    assert ill.MODEL_ENV_VAR not in os.environ
    assert [entry.name for entry in mh.preprocessing_hooks()] == \
        ['other-extension']
    assert ill.disable_illumination_correction() is False


def test_enabling_twice_corrects_once(tmp_path):
    """A GUI action re-run must not apply the correction twice.

    The registry replaces by name rather than appending, and both the direct
    registration and the environment installer use the same name, so however
    many routes install it there is exactly one entry.
    """
    model = hand_model([np.full((4, 4), 0.5, np.float32)])
    path = str(tmp_path / 'model.npz')
    ill.enable_illumination_correction(model, path=path, verbose=False)
    ill.enable_illumination_correction(path, verbose=False)
    ill.install()

    hooks = mh.preprocessing_hooks()
    assert [entry.name for entry in hooks] == [ill.HOOK_NAME]
    assert os.environ[mh.HOOKS_ENV_VAR].split(',') == [ill.INSTALLER_ENTRY]

    array = np.full((4, 4, 1), 100, np.uint16)
    result = mh.apply_preprocessing_hooks(array, context(channels=[0]))
    assert result[0, 0, 0] == 200        # x2 once, not x4


def test_the_installer_refuses_an_environment_with_no_model(monkeypatch):
    """Named in SPACR_MEASURE_HOOKS with no model is a configuration error.

    A worker that shrugged here would measure uncorrected fields on a run the
    user had explicitly asked to correct.
    """
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, ill.INSTALLER_ENTRY)
    with pytest.raises(ill.IlluminationError, match=ill.MODEL_ENV_VAR):
        ill.install()
    # ...and through the registry it arrives as the hook machinery's error.
    with pytest.raises(mh.MeasurementHookError, match='illumination'):
        mh.preprocessing_hooks()


def test_worker_delivery_status_describes_every_case(tmp_path, monkeypatch):
    """The question "will this actually be applied?" has an answer to print."""
    ok, message = ill.worker_delivery_status('spawn')
    assert ok is False and 'not registered' in message

    model = hand_model([np.full((4, 4), 0.5, np.float32)])
    mh.register_preprocessing_hook(ill.IlluminationCorrector(model),
                                   name=ill.HOOK_NAME)
    ok, message = ill.worker_delivery_status('fork')
    assert ok is True and 'would NOT survive' in message

    # Defaults to the start method measure_crop will actually use.
    monkeypatch.setenv('SPACR_START_METHOD', 'spawn')
    ok, _message = ill.worker_delivery_status()
    assert ok is False


def test_measure_crop_writes_corrected_intensities_under_a_spawn_pool(
        tmp_path, monkeypatch):
    """End to end, through the real pool, on the real database.

    The fields measured here put identical cells at the centre and in the
    corners of the frame. Uncorrected, the corner cells measure well below the
    central one and the spread across positions is the artefact. Every field
    is measured by a cold ``spawn`` interpreter, so the numbers in
    measurements.db are corrected only if the correction genuinely reached
    those interpreters.
    """
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    shape = (128, 128)
    truth = quadratic_vignette(shape, strength=0.5)
    # A separate, larger folder to estimate from: the estimate wants many
    # fields, the measure run wants few, and they are different questions.
    estimate_from = write_plate(tmp_path / 'estimate', truth, n_fields=16,
                                radius=6, n_objects=6)

    run = tmp_path / 'run'
    merged = run / 'merged'
    os.makedirs(merged)
    (run / 'measurements').mkdir(parents=True)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cell = np.zeros(shape, np.uint16)
    nucleus = np.zeros(shape, np.uint16)
    pathogen = np.zeros(shape, np.uint16)
    for label, (cy, cx) in enumerate([(64, 64), (18, 18), (18, 110),
                                      (110, 18), (110, 110)], start=1):
        disc = (yy - cy) ** 2 + (xx - cx) ** 2
        cell[disc <= 9 ** 2] = label
        nucleus[disc <= 3 ** 2] = label
        pathogen[disc <= 2 ** 2] = label
    plane = np.full(shape, BACKGROUND)
    plane[cell > 0] = BACKGROUND + SIGNAL
    observed = np.rint(plane * truth).astype(np.uint16)
    data = np.stack([observed, observed, cell, nucleus, pathogen], axis=-1)
    for field in (1, 2):
        np.save(merged / f'plate1_A01_F{field:03d}.npy', data.astype(np.uint16))

    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(merged), 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4,
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 2, 'test_mode': False,
        'cytoplasm': False, 'homogeneity': False, 'radial_dist': False,
        'calculate_correlation': False,
    })

    model = ill.estimate_illumination(estimate_from, channels=[0, 1],
                                      verbose=False)
    ill.enable_illumination_correction(
        model, path=str(tmp_path / 'illumination' / 'model.npz'),
        verbose=False)
    monkeypatch.setenv('SPACR_START_METHOD', 'spawn')

    measure_crop(settings)

    connection = sqlite3.connect(str(run / 'measurements' / 'measurements.db'))
    try:
        measured = [row[0] for row in connection.execute(
            'SELECT cell_channel_0_mean_intensity FROM cell')]
        status = connection.execute(
            'SELECT status, n_succeeded, n_failed FROM run_status').fetchone()
    finally:
        connection.close()

    assert status == ('complete', 2, 0)
    assert len(measured) == 10          # five cells, two fields
    measured = np.asarray(measured, dtype=float)

    # What the same rows look like with no correction at all, for the
    # comparison this whole feature is about.
    uncorrected = []
    for label in range(1, 6):
        uncorrected.append(observed[cell == label].mean())
    uncorrected = np.asarray(uncorrected, dtype=float)

    truth_value = BACKGROUND + SIGNAL
    assert uncorrected.std() / uncorrected.mean() > 0.05
    assert measured.std() / measured.mean() < 0.01
    assert measured.mean() == pytest.approx(truth_value, rel=0.02)
    # The corner cells were the ones the microscope under-reported, and they
    # are the ones the correction moved.
    assert measured.min() > uncorrected.min() * 1.05


# ---------------------------------------------------------------------------
# 6. QC: the user has to be able to see that it worked
# ---------------------------------------------------------------------------

def test_qc_writes_the_field_image_and_reports_the_bias_it_removed(tmp_path):
    """An image of the field, the trend before and after, and one number."""
    shape = (256, 256)
    merged = write_plate(tmp_path / 'merged',
                         quadratic_vignette(shape, strength=0.45), n_fields=20)
    model = ill.estimate_illumination(merged, channels=[0, 1], verbose=False)

    report = ill.illumination_qc(model, merged,
                                 save_dir=str(tmp_path / 'qc'), verbose=False)

    metrics = report['plate1'][0]
    assert metrics['slope_before'] < -0.30
    assert abs(metrics['slope_after']) < 0.02
    assert metrics['bias_removed_pct'] > 95.0
    assert metrics['nonuniformity_pct'] > 20.0
    assert metrics['gain_min'] < 1.0 < metrics['gain_max']
    assert metrics['n_fields'] == 20

    figure = report['_figures']['plate1']
    assert os.path.isfile(figure)
    assert os.path.getsize(figure) > 10_000
    assert figure.endswith('illumination_qc_plate1.png')


def test_qc_prints_the_before_and_after_slope(tmp_path, capsys):
    """The number has to reach the terminal, not only the return value."""
    shape = (64, 64)
    merged = write_plate(tmp_path / 'merged', quadratic_vignette(shape),
                         n_fields=12, radius=5, n_objects=4)
    model = ill.estimate_illumination(merged, channels=[0], verbose=False)

    ill.illumination_qc(model, merged, save_dir='', verbose=True)

    printed = capsys.readouterr().out
    assert 'position-intensity slope' in printed
    assert 'of the bias removed' in printed


def test_the_slope_metric_is_zero_on_flat_data_and_signed_on_a_gradient():
    """The metric itself, so the assertions above rest on something checked."""
    shape = (100, 100)
    coordinates = np.array([[50, 50], [0, 0], [99, 99], [0, 99], [99, 0]],
                           dtype=float)
    assert ill.position_intensity_slope([5, 5, 5, 5, 5], coordinates,
                                        shape) == pytest.approx(0.0, abs=1e-9)
    # Dimmer away from the centre -> negative, and scaled so that the value is
    # the fraction of the mean lost between the centre and a corner.
    dimmer = ill.position_intensity_slope([1.0, 0.5, 0.5, 0.5, 0.5],
                                          coordinates, shape)
    assert dimmer < -0.3
    assert ill.position_intensity_slope([1.0], [[0, 0]], shape) == 0.0


def test_the_estimator_says_when_it_had_too_little_to_work_with(tmp_path,
                                                                capsys):
    """Below ten fields the across-field median stops rejecting objects."""
    merged = write_plate(tmp_path / 'merged', quadratic_vignette((64, 64)),
                         n_fields=4, radius=5, n_objects=4)
    ill.estimate_illumination(merged, channels=[0], verbose=True)

    printed = capsys.readouterr().out
    assert 'estimated from only 4 field(s)' in printed
    assert 'non-uniformity' in printed


def test_fields_of_a_different_shape_sit_the_estimate_out(tmp_path, capsys):
    """One gain map cannot serve two sensor geometries; say so and go on."""
    merged = write_plate(tmp_path / 'merged', quadratic_vignette((64, 64)),
                         n_fields=12, radius=5, n_objects=4)
    np.save(os.path.join(merged, 'plate1_A01_F900.npy'),
            np.zeros((32, 32, 5), np.uint16))

    model = ill.estimate_illumination(merged, channels=[0], verbose=True)

    assert model.field_for('plate1').shape == (64, 64)
    assert model.field_for('plate1').n_fields == 12
    assert 'different pixel shape' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7. Settings, registered through the seam and off by default
# ---------------------------------------------------------------------------


def _categories_in_the_settings_literal():
    """``{category: [key, ...]}`` read out of the settings.py source.

    Parsed rather than imported so it shows what EVERY process sees. The
    live ``spacr.settings.categories`` also holds whatever the modules this
    interpreter happened to import contributed through ``register_defaults``,
    and a heading that is only there for some processes is a heading the
    panel sometimes loses.
    """
    import ast
    import pathlib

    import spacr.settings as S

    tree = ast.parse(pathlib.Path(S.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(getattr(t, 'id', None) == 'categories' for t in node.targets):
            continue
        return {
            key.value: [k.value for k in value.elts]
            for key, value in zip(node.value.keys, node.value.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.List)
        }
    raise AssertionError('no `categories = {...}` literal in settings.py')


def test_the_settings_are_registered_off_by_default_and_typed():
    """Registered through register_defaults, not appended to settings.py.

    The category half of this test was inverted when the illumination
    settings were folded into Measure. It used to require that no
    illumination key appear in ``spacr.settings.categories`` at all, on the
    grounds that no shipped panel offered them; Measure now does, and an
    offered key with no category is dumped into the panel's trailing
    bucket. What made the old rule worth having -- that the map must not
    depend on which modules a process happened to import -- is kept, and
    checked directly: the keys are in the ``categories`` literal in
    settings.py, which every process sees, not contributed by this module
    at import time.
    """
    import spacr.settings as S

    assert S.has_registered_defaults(ill.APP_KEY)
    defaults = S.defaults_for(ill.APP_KEY)
    assert defaults['illumination_correction'] is False
    assert defaults['illumination_estimator'] == 'polynomial'
    assert defaults['illumination_per_plate'] is True

    for key in defaults:
        if not key.startswith('illumination_'):
            continue
        assert key in S.expected_types, f'{key} is untyped'
        assert key in S.tooltips, f'{key} has no tooltip'
        assert S.tooltips[key].startswith('('), f'{key} tooltip has no type'
        homes = [name for name, keys in S.categories.items() if key in keys]
        assert homes == ['Illumination Correction'], \
            f'{key} is filed under {homes}, not one illumination heading'
    # Contributed by the settings.py literal, not by this module:
    # `register_illumination_settings` passes no categories, so the heading
    # is there for a process that never imported spacr.illumination.
    assert set(_categories_in_the_settings_literal()['Illumination Correction']) \
        == {k for k in defaults if k.startswith('illumination_')}
    assert S.descriptions[ill.APP_KEY].startswith('Illumination')
    # Registering again is a no-op rather than a duplicate declaration.
    assert ill.register_illumination_settings() is False


def test_the_registered_types_survive_the_shipped_validator():
    """check_settings must coerce these keys rather than drop them."""
    import spacr.settings as S

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    variables = {key: (None, None, _Var(value), None) for key, value in {
        'illumination_correction': True,
        'illumination_degree': '4',
        'illumination_max_fields': '50',
        'illumination_dark': '0.0',
        'illumination_estimator': 'polynomial',
    }.items()}
    settings, errors = S.check_settings(variables, S.expected_types)
    assert not errors
    assert settings['illumination_degree'] == 4
    assert settings['illumination_max_fields'] == 50
    assert settings['illumination_correction'] is True


def test_prepare_does_nothing_at_all_unless_it_is_asked_to(tmp_path):
    """Off by default means no estimate, no files, no hook."""
    merged = write_plate(tmp_path / 'merged', quadratic_vignette((64, 64)),
                         n_fields=6, radius=5, n_objects=4)
    settings = ill.illumination_settings({'src': merged, 'channels': [0]})

    assert ill.prepare_illumination_correction(settings) is None
    assert ill.prepare_segmentation_illumination(
        settings, pipeline_style='v1') is None
    assert mh.preprocessing_hooks() == ()
    assert not os.path.isdir(tmp_path / 'illumination')


def test_prepare_estimates_saves_enables_and_qcs_in_one_call(tmp_path):
    """The one call a pipeline makes before measure_crop."""
    shape = (64, 64)
    merged = write_plate(tmp_path / 'merged', quadratic_vignette(shape),
                         n_fields=12, radius=5, n_objects=4)
    settings = ill.illumination_settings({
        'src': merged, 'channels': [0, 1],
        'illumination_correction': True, 'verbose': False})

    model = ill.prepare_illumination_correction(settings)

    assert isinstance(model, ill.IlluminationModel)
    folder = tmp_path / 'illumination'
    assert os.path.isfile(folder / 'illumination_model.npz')
    assert os.path.isfile(folder / 'illumination_qc_plate1.png')
    assert [entry.name for entry in mh.preprocessing_hooks()] == [ill.HOOK_NAME]
    ok, _message = ill.worker_delivery_status('spawn')
    assert ok is True

    # A second run with illumination_model set reuses the saved field rather
    # than estimating a new one -- what re-measuring a plate needs.
    mh.clear_measurement_hooks()
    settings['illumination_model'] = str(folder / 'illumination_model.npz')
    reused = ill.prepare_illumination_correction(settings)
    np.testing.assert_array_equal(reused.field_for('plate1').flatfield,
                                  model.field_for('plate1').flatfield)


def test_prepare_refuses_to_correct_without_a_source(tmp_path):
    with pytest.raises(ill.IlluminationError, match='src'):
        ill.prepare_illumination_correction(
            {'illumination_correction': True, 'src': '', 'channels': [0]})


def test_segmentation_preparation_labels_qc_and_preserves_raw_fields(tmp_path):
    """Preparation fits beside the run but never rewrites microscope pixels."""
    merged = write_plate(
        tmp_path / 'merged', quadratic_vignette((64, 64)),
        n_fields=12, radius=5, n_objects=4)
    raw_before = {
        name: (tmp_path / 'merged' / name).read_bytes()
        for name in os.listdir(merged)
    }
    settings = ill.illumination_settings({
        'src': merged,
        'channels': [0],
        'illumination_correction': True,
        'verbose': False,
    })

    session = ill.prepare_segmentation_illumination(
        settings, pipeline_style='v1')

    assert isinstance(session, ill.SegmentationIlluminationSession)
    assert mh.preprocessing_hooks() == ()
    assert session.prepared.model.meta['application_contract_version'] == 1
    assert session.prepared.model.meta['channel_index_space'] == \
        'persisted-intensity-axis'
    assert session.prepared.model.meta['estimated_from_intensity_state'] == 'raw'
    folder = tmp_path / 'illumination'
    assert (folder / 'illumination_model.npz').is_file()
    assert (folder / 'illumination_qc_segmentation_input_plate1.png').is_file()
    record = json.loads(
        (folder / 'segmentation_application.json').read_text(encoding='utf-8'))
    assert record['application_state'] == 'prepared'
    assert record['completed_fields'] == []
    assert record['target_scope'] == 'segmentation-input-only'
    assert record['raw_persisted_intensities_modified'] is False
    assert {
        name: (tmp_path / 'merged' / name).read_bytes()
        for name in os.listdir(merged)
    } == raw_before


def test_segmentation_session_corrects_a_copy_and_commits_only_after_mark(
        tmp_path, monkeypatch):
    """The application record distinguishes prepared, durable and complete."""
    from spacr import run_journal

    events = []

    class FakeRun:
        def _record_stage(self, stage_id, **payload):
            events.append((stage_id, payload))

    monkeypatch.setattr(run_journal, 'current_run', lambda: FakeRun())
    base = segmentation_prepared(tmp_path)

    class MutatingCorrector:
        """A legal future corrector that edits the array it is handed."""
        def __init__(self):
            self.stats = {
                'corrected': 0, 'skipped': 0,
                'clipped_pixels': 0, 'clipped_fields': 0,
            }

        def __call__(self, array, _context):
            array[0, 0, 0] *= 2
            self.stats['corrected'] += 1
            return array

    prepared = ill.PreparedIllumination(
        model=base.model,
        corrector=MutatingCorrector(),
        model_path=base.model_path,
        model_sha256=base.model_sha256,
    )
    record_path = tmp_path / 'illumination' / 'application.json'
    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')

    initial = json.loads(record_path.read_text(encoding='utf-8'))
    assert initial['application_state'] == 'prepared'
    assert initial['completed_fields'] == []
    assert events[-1][1]['metrics']['applied'] is False
    raw = np.full((2, 2, 1), 10, dtype=np.uint16)
    untouched = raw.copy()

    corrected = session.correct('field-1', raw, context(channels=[0]))

    np.testing.assert_array_equal(raw, untouched)
    assert corrected is not raw
    assert corrected[0, 0, 0] == 20
    # In-memory correction is not durable output, so it cannot advance the
    # application record on its own.
    assert json.loads(record_path.read_text(encoding='utf-8')) == initial
    assert events[-1][1]['metrics']['applied'] is True
    assert session.mark_completed('field-1') is True
    durable = json.loads(record_path.read_text(encoding='utf-8'))
    assert durable['application_state'] == 'running'
    assert durable['completed_fields'] == ['field-1']
    assert session.mark_completed('field-1') is False
    with pytest.raises(ill.IlluminationError, match='twice'):
        session.correct('field-1', raw, context(channels=[0]))
    with pytest.raises(ill.IlluminationError, match='missing'):
        session.finish(['field-1', 'field-2'])
    assert json.loads(record_path.read_text(encoding='utf-8')) == durable

    session.finish(['field-1'])

    complete = json.loads(record_path.read_text(encoding='utf-8'))
    assert complete['application_state'] == 'complete'
    assert complete['completed_fields'] == ['field-1']
    assert events[-1][1]['state'] == 'done'

    fresh = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')
    reset = json.loads(record_path.read_text(encoding='utf-8'))
    assert fresh.completed_fields == ()
    assert reset['application_state'] == 'prepared'
    assert reset['completed_fields'] == []
    assert events[-1][1]['metrics']['applied'] is False


def test_a_skipped_segmentation_field_cannot_be_claimed_as_corrected(tmp_path):
    """Measure may skip a missing plate; segmentation provenance may not."""
    base = segmentation_prepared(tmp_path)
    prepared = ill.PreparedIllumination(
        model=base.model,
        corrector=ill.IlluminationCorrector(
            base.model, on_missing='skip', verbose=False),
        model_path=base.model_path,
        model_sha256=base.model_sha256,
    )
    record_path = tmp_path / 'illumination' / 'application.json'
    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v2')

    with pytest.raises(ill.IlluminationError, match='uncorrected field'):
        session.correct(
            'field-2', np.full((2, 2, 1), 10, dtype=np.uint16),
            context(name='plate2_A01_F000', channels=[0]))

    assert session.applied_fields == ()
    assert session.completed_fields == ()
    with pytest.raises(ill.IlluminationError, match='before'):
        session.mark_completed('field-2')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['application_state'] == 'prepared'
    assert record['completed_fields'] == []


def test_segmentation_completion_is_atomic_when_the_replace_fails(
        tmp_path, monkeypatch):
    prepared = segmentation_prepared(tmp_path)
    record_path = tmp_path / 'illumination' / 'application.json'
    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')
    session.correct(
        'field-1', np.full((2, 2, 1), 10, dtype=np.uint16),
        context(channels=[0]))
    before = record_path.read_bytes()

    monkeypatch.setattr(ill.os, 'replace', lambda *_args: (_ for _ in ()).throw(
        OSError('disk refused replace')))
    with pytest.raises(OSError, match='disk refused replace'):
        session.mark_completed('field-1')

    assert session.completed_fields == ()
    assert record_path.read_bytes() == before
    assert list(record_path.parent.glob('.segmentation_application_*.json')) == []


def test_a_secondary_temp_cleanup_failure_keeps_the_original_record(
        tmp_path, monkeypatch):
    prepared = segmentation_prepared(tmp_path)
    record_path = tmp_path / 'illumination' / 'application.json'
    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')
    session.correct(
        'field-1', np.full((2, 2, 1), 10, dtype=np.uint16),
        context(channels=[0]))
    before = record_path.read_bytes()

    def replace_fails(*_args):
        raise OSError('original replace failure')

    def cleanup_fails(*_args):
        raise OSError('secondary cleanup failure')

    monkeypatch.setattr(ill.os, 'replace', replace_fails)
    monkeypatch.setattr(ill.os, 'remove', cleanup_fails)
    with pytest.raises(OSError, match='original replace failure'):
        session.mark_completed('field-1')

    assert session.completed_fields == ()
    assert record_path.read_bytes() == before
    assert len(list(record_path.parent.glob(
        '.segmentation_application_*.json'))) == 1


def test_preprocess_false_resume_is_read_only_and_requires_a_complete_record(
        tmp_path, monkeypatch):
    prepared = segmentation_prepared(tmp_path)
    record_path = tmp_path / 'illumination' / 'application.json'
    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')
    session.correct(
        'field-1', np.full((2, 2, 1), 10, dtype=np.uint16),
        context(channels=[0]))
    session.mark_completed('field-1')
    partial_bytes = record_path.read_bytes()
    partial_mtime = record_path.stat().st_mtime_ns

    with pytest.raises(ill.IlluminationError, match='only.*running'):
        ill.load_segmentation_illumination_resume(
            {'illumination_correction': True, 'illumination_model': ''},
            provenance_path=str(record_path), pipeline_style='v1',
            expected_fields=['field-1'], verbose=False)
    assert record_path.read_bytes() == partial_bytes
    assert record_path.stat().st_mtime_ns == partial_mtime

    session.finish(['field-1'])
    complete_bytes = record_path.read_bytes()
    complete_mtime = record_path.stat().st_mtime_ns
    partial_resume = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1',
        resume=True)
    assert partial_resume.completed_fields == ('field-1',)
    assert record_path.read_bytes() == complete_bytes
    assert record_path.stat().st_mtime_ns == complete_mtime
    paths_before = sorted(path.name for path in record_path.parent.iterdir())
    record_before = record_path.read_bytes()
    record_mtime = record_path.stat().st_mtime_ns
    model_before = (tmp_path / 'illumination' / 'model.npz').read_bytes()
    model_mtime = (tmp_path / 'illumination' / 'model.npz').stat().st_mtime_ns

    def forbidden(*_args, **_kwargs):
        raise AssertionError('a read-only resume attempted preparation work')

    monkeypatch.setattr(ill, 'estimate_illumination', forbidden)
    monkeypatch.setattr(ill, 'illumination_qc', forbidden)
    monkeypatch.setattr(ill, 'enable_illumination_correction', forbidden)
    resumed = ill.load_segmentation_illumination_resume(
        {'illumination_correction': True, 'illumination_model': ''},
        provenance_path=str(record_path), pipeline_style='v1',
        expected_fields=['field-1'], verbose=False)

    assert resumed.model_sha256 == prepared.model_sha256
    assert mh.preprocessing_hooks() == ()
    assert sorted(path.name for path in record_path.parent.iterdir()) == paths_before
    assert record_path.read_bytes() == record_before
    assert record_path.stat().st_mtime_ns == record_mtime
    assert (tmp_path / 'illumination' / 'model.npz').read_bytes() == model_before
    assert (tmp_path / 'illumination' / 'model.npz').stat().st_mtime_ns == \
        model_mtime
    with pytest.raises(ill.IlluminationError, match='does not cover exactly'):
        ill.load_segmentation_illumination_resume(
            {'illumination_correction': True, 'illumination_model': ''},
            provenance_path=str(record_path), pipeline_style='v1',
            expected_fields=['field-1', 'field-2'], verbose=False)
    assert record_path.read_bytes() == record_before
    assert record_path.stat().st_mtime_ns == record_mtime


def test_segmentation_rejects_legacy_tampered_and_unknown_pipeline_models(
        tmp_path):
    legacy = segmentation_prepared(tmp_path / 'legacy', legacy=True)
    with pytest.raises(ill.IlluminationError, match='legacy or incompatible'):
        ill.SegmentationIlluminationSession(
            legacy,
            provenance_path=str(tmp_path / 'legacy' / 'application.json'),
            pipeline_style='v1')

    tampered = segmentation_prepared(tmp_path / 'tampered')
    with open(tampered.model_path, 'ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ill.IlluminationError, match='SHA-256'):
        ill.SegmentationIlluminationSession(
            tampered,
            provenance_path=str(tmp_path / 'tampered' / 'application.json'),
            pipeline_style='v1')

    valid = segmentation_prepared(tmp_path / 'valid')
    absent_resume_path = tmp_path / 'valid' / 'absent.json'
    with pytest.raises(ill.IlluminationError, match='no record exists'):
        ill.SegmentationIlluminationSession(
            valid,
            provenance_path=str(absent_resume_path),
            pipeline_style='v1', resume=True)
    assert not absent_resume_path.exists()
    with pytest.raises(ill.IlluminationError, match='v1.*v2'):
        ill.SegmentationIlluminationSession(
            valid,
            provenance_path=str(tmp_path / 'valid' / 'application.json'),
            pipeline_style='legacy')


def test_malformed_segmentation_application_records_fail_closed(tmp_path):
    prepared = segmentation_prepared(tmp_path)
    record_path = tmp_path / 'illumination' / 'application.json'
    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')
    session.correct(
        'field-1', np.full((2, 2, 1), 10, dtype=np.uint16),
        context(channels=[0]))
    session.mark_completed('field-1')
    session.finish(['field-1'])
    original = json.loads(record_path.read_text(encoding='utf-8'))

    def refused(payload, fragment, *, settings=None):
        if isinstance(payload, str):
            record_path.write_text(payload, encoding='utf-8')
        else:
            record_path.write_text(
                json.dumps(payload), encoding='utf-8')
        before = record_path.read_bytes()
        with pytest.raises(ill.IlluminationError, match=fragment):
            ill.load_segmentation_illumination_resume(
                settings or {
                    'illumination_correction': True,
                    'illumination_model': '',
                },
                provenance_path=str(record_path), pipeline_style='v1',
                expected_fields=['field-1'], verbose=False)
        assert record_path.read_bytes() == before

    changed = dict(original)
    changed['pipeline_style'] = 'v2'
    refused(changed, 'different provenance')
    changed = dict(original)
    changed['completed_fields'] = 'field-1'
    refused(changed, 'must be a list')
    changed = dict(original)
    changed['application_state'] = 'claimed'
    refused(changed, 'application_state')
    changed = dict(original)
    changed['model_path'] = ''
    refused(changed, 'no usable model_path')
    changed = dict(original)
    changed['model_path'] = 'missing-model.npz'
    refused(changed, 'cannot be read')
    changed = dict(original)
    changed['model_sha256'] = '0' * 64
    refused(changed, 'hash does not match')
    changed = dict(original)
    changed['qc_artifacts'] = 'not-a-list'
    refused(changed, 'qc_artifacts must be a list')
    changed = dict(original)
    changed['model_path'] = prepared.model_path
    refused(changed, 'different provenance')
    refused(['not', 'an', 'object'], 'JSON object')
    refused('{broken JSON', 'unreadable')

    record_path.write_text(json.dumps(original), encoding='utf-8')
    refused(
        original,
        "does not name the model recorded",
        settings={
            'illumination_correction': True,
            'illumination_model': str(tmp_path / 'other-model.npz'),
        },
    )
    with pytest.raises(ill.IlluminationError, match='requires.*True'):
        ill.load_segmentation_illumination_resume(
            {'illumination_correction': False},
            provenance_path=str(record_path), pipeline_style='v1',
            expected_fields=['field-1'], verbose=False)


def test_segmentation_validation_names_a_missing_saved_model(tmp_path):
    base = segmentation_prepared(tmp_path)
    missing = ill.PreparedIllumination(
        model=base.model,
        corrector=base.corrector,
        model_path=str(tmp_path / 'illumination' / 'gone.npz'),
        model_sha256=base.model_sha256,
    )
    with pytest.raises(ill.IlluminationError, match='cannot verify'):
        ill.SegmentationIlluminationSession(
            missing,
            provenance_path=str(tmp_path / 'illumination' / 'application.json'),
            pipeline_style='v1')


def test_a_broken_run_journal_cannot_replace_the_scientific_result(
        tmp_path, monkeypatch):
    from spacr import run_journal

    class BrokenRun:
        def _record_stage(self, *_args, **_kwargs):
            raise RuntimeError('journal storage unavailable')

    monkeypatch.setattr(run_journal, 'current_run', lambda: BrokenRun())
    prepared = segmentation_prepared(tmp_path)
    record_path = tmp_path / 'illumination' / 'application.json'

    session = ill.SegmentationIlluminationSession(
        prepared, provenance_path=str(record_path), pipeline_style='v1')
    corrected = session.correct(
        'field-1', np.full((2, 2, 1), 10, dtype=np.uint16),
        context(channels=[0]))

    assert corrected[0, 0, 0] == 20
    assert record_path.is_file()


def test_an_external_model_keeps_application_provenance_in_the_current_run(
        tmp_path):
    shared = segmentation_prepared(tmp_path / 'shared')
    stack = tmp_path / 'current_run' / 'stack'
    stack.mkdir(parents=True)
    settings = ill.illumination_settings({
        'src': str(tmp_path / 'wrong_run' / 'stack'),
        'channels': [0],
        'illumination_correction': True,
        'illumination_model': shared.model_path,
        'illumination_qc': False,
        'verbose': False,
    })

    session = ill.prepare_segmentation_illumination(
        settings, src=str(stack), channels=[0], pipeline_style='v2')

    expected = tmp_path / 'current_run' / 'illumination' / \
        'segmentation_application.json'
    assert session.provenance_path == str(expected)
    assert expected.is_file()
    assert not (tmp_path / 'shared' / 'illumination' /
                'segmentation_application.json').exists()


def test_a_measure_qc_failure_does_not_leave_a_new_model_behind(
        tmp_path, monkeypatch):
    merged = write_plate(
        tmp_path / 'merged', quadratic_vignette((64, 64)),
        n_fields=12, radius=5, n_objects=4)
    settings = ill.illumination_settings({
        'src': merged,
        'channels': [0],
        'illumination_correction': True,
        'verbose': False,
    })
    monkeypatch.setattr(
        ill, 'illumination_qc',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError('QC could not be rendered')))

    with pytest.raises(RuntimeError, match='QC could not be rendered'):
        ill.prepare_illumination_correction(settings)

    assert not (tmp_path / 'illumination' / 'illumination_model.npz').exists()
    assert mh.preprocessing_hooks() == ()


def test_describe_says_what_the_field_is_and_where_it_came_from(tmp_path):
    """A model on disk has to be able to say what produced it."""
    merged = write_plate(tmp_path / 'merged', quadratic_vignette((64, 64)),
                         n_fields=12, radius=5, n_objects=4)
    model = ill.estimate_illumination(merged, channels=[0], verbose=False)

    text = model.describe()
    assert 'plate plate1, channel 0' in text
    assert 'non-uniformity' in text
    assert 'polynomial degree 4 over 12 field(s)' in text
    assert model.meta['src'] == [os.path.abspath(merged)]
    assert model.field_for('plate1').nonuniformity()[0] > 0.1


# ---------------------------------------------------------------------------
# 8. Set on the run it affects: the settings belong to Measure
# ---------------------------------------------------------------------------
# The call in measure_crop was there, and the settings it reads were not:
# `get_measure_crop_settings` returned 61 keys and none of them was
# `illumination_correction`, so `prepare_illumination_correction` returned
# None on every run started from the Measure panel. The switch that decides
# whether every intensity feature carries a position-dependent bias could
# only be thrown on the separate Illumination screen -- which enables the
# correction by setting process environment variables, so whether a table
# was corrected depended on what had been run in the same process before it.


def test_measure_offers_every_illumination_setting():
    """Measure's defaults are a superset of the Illumination module's."""
    from spacr.settings import get_measure_crop_settings

    measure = get_measure_crop_settings(settings={})
    for key, value in ill.illumination_settings({}).items():
        assert key in measure, (
            f'{key} is not in the measure defaults, so a Measure run cannot '
            f'set it and prepare_illumination_correction never sees it')
        if key.startswith('illumination_'):
            assert measure[key] == value, (
                f'{key} defaults to {measure[key]!r} under Measure and '
                f'{value!r} under Illumination; the same run would be '
                f'corrected differently depending on which screen started it')

    # Measure's own `src` and `channels` win. The estimate reads the fields
    # the run measures, and Illumination's three-channel default would
    # silently drop the fourth channel of a four-channel measure run.
    assert measure['channels'] == [0, 1, 2, 3]
    assert measure['src'] == 'path'


def test_the_measure_panel_shows_them_under_one_heading():
    """Folded into one section, not the four the Illumination screen uses.

    Illumination splits its knobs across Correction Model, Field Sampling and
    QC & Failure Handling because estimating a field is that screen's whole
    job. Inside Measure it is one decision -- correct these fields or do not
    -- so it is one group, and every key is in it: a key the layout does not
    place lands in the trailing "Additional Settings" bucket, which is not a
    heading anyone chose.
    """
    pytest.importorskip('PySide6')
    from spacr.qt.screens.settings_model import (
        categories_for_app, resolve_default_settings,
    )
    import spacr.settings as S

    sections = categories_for_app('measure', S.categories)
    defaults = resolve_default_settings('measure')
    illumination_keys = {k for k in defaults if k.startswith('illumination_')}
    assert illumination_keys, 'the Measure panel offers no illumination keys'
    homes = {name for name, keys in sections.items()
             if illumination_keys & set(keys)}
    assert homes == {'Illumination Correction'}
    assert illumination_keys <= set(sections['Illumination Correction'])
    # The trailing bucket is built from the whole shared map, so it always
    # exists; what matters is that nothing Measure OFFERS ends up in it,
    # because those are the keys that would render under that non-heading.
    assert not [key for key in sections.get('Additional Settings', ())
                if key in defaults]


def test_a_measure_run_hands_the_panel_settings_to_the_correction(
        tmp_path, monkeypatch):
    """The whole point: the values set on Measure reach the correction.

    Driven from `resolve_default_settings('measure')` -- what the Qt panel
    builds its widgets from -- rather than from a hand-written dict, so the
    test fails if the keys stop being offered there. Every illumination key
    is checked by VALUE and not merely by presence: the defect was a dict
    that lacked them, and a dict that carries them at somebody else's
    defaults corrects the plate with settings the user never chose.
    """
    pytest.importorskip('PySide6')
    from spacr import measure as measure_mod
    from spacr.qt.screens.settings_model import resolve_default_settings

    seen = {}
    monkeypatch.setattr(ill, 'prepare_illumination_correction',
                        lambda settings, **kw: seen.update(settings) or None)
    monkeypatch.setattr(measure_mod, '_start_manager',
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError('stop before the pool')))

    plate = tmp_path / 'plate'
    merged = plate / 'merged'
    os.makedirs(merged)
    write_plate(merged, quadratic_vignette((64, 64)), n_fields=2,
                n_objects=2, radius=5)

    settings = resolve_default_settings('measure')
    settings.update({
        'src': str(plate), 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4,
        'save_measurements': False, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 1, 'test_mode': False,
    })
    # What a user changes on the panel, in the panel's own dict.
    chosen = {
        'illumination_correction': True,
        'illumination_estimator': 'smooth',
        'illumination_degree': 3,
        'illumination_per_plate': False,
        'illumination_max_fields': 12,
        'illumination_dark': 2.5,
        'illumination_qc': False,
        'illumination_on_missing': 'skip',
        'illumination_model': '',
    }
    # ASSIGNED, not injected. `settings.update(chosen)` would put the keys in
    # the dict whether or not the panel offers them, and measure_crop's
    # `get_measure_crop_settings` only ever setdefaults -- so the test would
    # pass against exactly the defect it exists to catch. Writing into keys
    # that must already be there is what makes it a control.
    for key, value in chosen.items():
        assert key in settings, (
            f'{key} is not among the settings the Measure panel builds, so '
            f'there is no control for it and no value for the correction '
            f'to read')
        settings[key] = value

    try:
        measure_mod.measure_crop(settings)
    except Exception:
        # Stopped at the worker pool on purpose; the call under test is
        # everything that happened before it.
        pass

    assert seen, 'measure_crop never called prepare_illumination_correction'
    for key, value in chosen.items():
        assert seen.get(key) == value, (
            f'the correction was handed {key}={seen.get(key)!r} instead of '
            f'the {value!r} the Measure panel was set to')


def test_a_saved_model_is_reused_and_nothing_is_re_estimated(tmp_path,
                                                             monkeypatch):
    """`illumination_model` is a path, and filling it skips the estimate.

    That is what makes the estimate a once-per-plate cost rather than a
    once-per-run one: the first Measure run over a plate fits the field and
    saves it, and every later run over the same plate points at the file. A
    reuse that quietly re-fitted would still produce corrected numbers, so
    the estimator is replaced by one that raises -- the only way to tell
    "reused" from "re-derived the same answer".
    """
    from spacr.settings import get_measure_crop_settings

    shape = (64, 64)
    merged = write_plate(tmp_path / 'merged', quadratic_vignette(shape),
                         n_fields=12, radius=5, n_objects=4)
    first = get_measure_crop_settings(settings={})
    first.update({'src': merged, 'channels': [0],
                  'illumination_correction': True, 'verbose': False})
    model = ill.prepare_illumination_correction(first)
    saved = os.path.join(str(tmp_path), 'illumination',
                         'illumination_model.npz')
    assert os.path.isfile(saved)

    mh.clear_measurement_hooks()

    def no_estimate(*args, **kwargs):
        raise AssertionError('re-estimated a field that was already saved')

    monkeypatch.setattr(ill, 'estimate_illumination', no_estimate)
    second = get_measure_crop_settings(settings={})
    second.update({'src': merged, 'channels': [0],
                   'illumination_correction': True, 'verbose': False,
                   'illumination_model': saved})

    reused = ill.prepare_illumination_correction(second)

    np.testing.assert_array_equal(reused.field_for('plate1').flatfield,
                                  model.field_for('plate1').flatfield)
    assert [entry.name for entry in mh.preprocessing_hooks()] == [ill.HOOK_NAME]


# ---------------------------------------------------------------------------
# The pipeline call: measure_crop installs the correction itself (item 4.1b)
# ---------------------------------------------------------------------------
# Everything above proves the correction works once something has called
# `prepare_illumination_correction`. Nothing did. `illumination_correction`
# was a setting the GUI showed, the CLI accepted and the docs described, and
# the only effect of turning it on was that a run took the same length of
# time and produced the same biased numbers.

def test_measure_crop_prepares_the_correction_before_it_measures(
        tmp_path, monkeypatch):
    """The call, at the point in ``measure_crop`` where it has to be.

    Two things have to be true and only one of them is "it is called":
    the settings it is handed must already have ``src`` pointed at the
    *merged* folder (the estimate reads the same fields the run
    measures, and the raw plate folder holds no ``.npy`` at all), and it
    must happen before the worker pool is built, because installing the
    hook is what writes the env vars a spawned worker reads.
    """
    from spacr import measure as measure_mod
    from spacr.settings import get_measure_crop_settings

    seen = {}

    def spy(settings, **kwargs):
        seen['src'] = settings['src']
        seen['before_pool'] = 'pool' not in seen
        return None

    monkeypatch.setattr(ill, 'prepare_illumination_correction', spy)

    def no_pool(*args, **kwargs):
        seen['pool'] = True
        raise RuntimeError('stop here')

    # `_start_manager` is the last thing measure_crop does before it opens
    # the worker pool, so failing it puts the boundary exactly where the
    # ordering claim is.
    monkeypatch.setattr(measure_mod, '_start_manager', no_pool)

    plate = tmp_path / 'plate'
    merged = plate / 'merged'
    os.makedirs(merged)
    truth = quadratic_vignette((64, 64))
    write_plate(merged, truth, n_fields=2, n_objects=2, radius=5)

    settings = get_measure_crop_settings(settings={})
    settings.update({
        # The plate folder, NOT the merged one: measure_crop appends
        # /merged itself, and the call has to sit after that or the
        # estimate looks for fields in a folder that has none.
        'src': str(plate), 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4,
        'illumination_correction': True,
        'save_measurements': False, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 1, 'test_mode': False,
    })
    try:
        measure_mod.measure_crop(settings)
    except Exception:
        # The run is stopped at the pool on purpose; what is under test
        # is everything that happened before it.
        pass

    assert seen.get('src'), (
        "measure_crop never called prepare_illumination_correction, so "
        "illumination_correction=True does nothing at all")
    assert os.path.basename(seen['src']) == 'merged', (
        f"the correction was pointed at {seen['src']!r}, which holds no "
        f"merged fields to estimate from")
    assert seen.get('before_pool') is True, (
        "the correction was installed after the worker pool was built, so "
        "no spawned worker inherits it")


def test_measure_crop_leaves_an_uncorrected_run_alone(tmp_path, monkeypatch):
    """Off by default: the call is made, and it does nothing.

    The guard lives in `prepare_illumination_correction` rather than at
    the call site, so this asserts the real function -- not a spy -- is
    reached and returns None without touching the hook registry.
    """
    from spacr import measure as measure_mod
    from spacr.settings import get_measure_crop_settings

    before = dict(mh.preprocessing_hooks())
    calls = []
    real = ill.prepare_illumination_correction

    def watched(settings, **kwargs):
        calls.append(settings.get('illumination_correction'))
        return real(settings, **kwargs)

    monkeypatch.setattr(ill, 'prepare_illumination_correction', watched)
    monkeypatch.setattr(measure_mod, '_start_manager',
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError('stop here')))

    plate = tmp_path / 'plate'
    merged = plate / 'merged'
    os.makedirs(merged)
    write_plate(merged, quadratic_vignette((64, 64)), n_fields=2,
                n_objects=2, radius=5)

    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(plate), 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': 4,
        'save_measurements': False, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 1, 'test_mode': False,
    })
    try:
        measure_mod.measure_crop(settings)
    except Exception:
        pass

    assert calls and not calls[0], "the default is no longer 'off'"
    assert dict(mh.preprocessing_hooks()) == before, (
        "a run with illumination_correction off installed a hook anyway")
    assert not os.path.isdir(plate / 'illumination')
