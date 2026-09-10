"""Illumination correction where the data does not cooperate.

The estimator turns a folder of merged fields into a gain map that multiplies
every published intensity, so the ways it can quietly produce nonsense matter
more than the happy path: a surface fitted to bins that were all rejected, a
gain map with a corner at zero, a plate the sampler handed back empty, a model
file that fails while it is being read. Each of those has to end in a named
``IlluminationError`` or a reported number, never in a silent gain of nan.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import illumination as ill
from spacr import measure_hooks as mh


@pytest.fixture(autouse=True)
def _pristine(monkeypatch):
    """No illumination hook or environment before or after each test.

    Both the hook registry and the environment variables are process-global;
    one leaking out of a test here would change what a later test measures.
    """
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR,
                 'SPACR_START_METHOD'):
        monkeypatch.delenv(name, raising=False)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        os.environ.pop(name, None)


def _model(flat, *, plate='plate1', channels=(0,), key=None):
    """A model holding one hand-built field, with no estimation involved."""
    field = ill.IlluminationField(
        plate=plate,
        channels=tuple(int(c) for c in channels),
        flatfield=np.stack([np.asarray(flat, np.float32)
                            for _ in channels]),
        dark=np.zeros(len(channels), np.float32),
        n_fields=1, estimator='polynomial', degree=4, bin_size=1)
    return ill.IlluminationModel(fields={key or plate: field}, meta={})


def _write_fields(folder, flat, *, plate='plate1', count=3, channels=2):
    """Merged ``(Y, X, C)`` fields of flat signal times a known gain map."""
    os.makedirs(folder, exist_ok=True)
    flat = np.asarray(flat, float)
    for index in range(count):
        planes = [np.rint(1000.0 * flat).astype(np.uint16)
                  for _ in range(channels)]
        np.save(os.path.join(folder, f'{plate}_A01_F{index:03d}.npy'),
                np.stack(planes, axis=-1))
    return str(folder)


# ---------------------------------------------------------------------------
# Reading a saved model
# ---------------------------------------------------------------------------

def test_a_failure_inside_the_model_file_keeps_its_own_message(tmp_path,
                                                               monkeypatch):
    """An IlluminationError raised while decoding is not re-wrapped.

    The reader wraps arbitrary exceptions into "could not be read"; an error
    that already says exactly what is wrong with the model must reach the user
    with that sentence intact instead of being flattened into the generic one.
    """
    path = str(tmp_path / 'model.npz')
    _model(np.full((4, 4), 0.5, np.float32)).save(path)

    def refuse(**kwargs):
        raise ill.IlluminationError('this field was estimated for 3 channels')

    monkeypatch.setattr(ill, 'IlluminationField', refuse)
    with pytest.raises(ill.IlluminationError,
                       match='estimated for 3 channels'):
        ill.IlluminationModel.load(path)


# ---------------------------------------------------------------------------
# Source folders and sampling
# ---------------------------------------------------------------------------

def test_several_source_folders_all_become_absolute_paths(tmp_path):
    """``settings['src']`` may be a list, and every entry is resolved."""
    one = tmp_path / 'one'
    two = tmp_path / 'two'
    one.mkdir()
    two.mkdir()

    resolved = ill._source_folders([str(one), two])

    assert resolved == (os.path.abspath(str(one)), os.path.abspath(str(two)))


def test_sampling_walks_the_whole_plate_and_repeats_itself():
    """The sample is evenly spaced and deterministic.

    The files are sorted by well and field, so an evenly spaced sample covers
    the plate; a random draw would over-weight whichever corner it landed in,
    and two runs of the estimator on the same folder would disagree.
    """
    paths = [f'f{index:03d}.npy' for index in range(20)]

    picked = ill._sample(paths, 5)

    assert picked == ['f000.npy', 'f004.npy', 'f008.npy', 'f012.npy',
                      'f016.npy']
    assert ill._sample(paths, 5) == picked
    assert ill._sample(paths, 50) == paths


# ---------------------------------------------------------------------------
# Fitting the surface
# ---------------------------------------------------------------------------

def _plane_stat():
    """A 4x4 binned statistic lying exactly on a plane."""
    rows, cols = np.mgrid[:4, :4]
    return 1.0 + 0.1 * rows + 0.05 * cols


def test_a_binned_statistic_with_no_positive_value_is_still_fitted():
    """Non-positive bins are fitted rather than thrown away wholesale.

    The first trim keeps only finite positive bins; when that leaves too few
    to constrain the surface the fit falls back to every finite bin, because
    refusing outright would turn a background-subtracted channel into an
    error instead of a surface.
    """
    surface = ill._fit_polynomial_surface(-np.ones((4, 4)), (8, 8), 2, 1)

    assert surface.shape == (8, 8)
    assert np.allclose(surface, -1.0)


def test_a_statistic_with_nothing_finite_is_refused_by_name():
    """Every bin rejected means there is no surface, and it says so."""
    with pytest.raises(ill.IlluminationError, match='could not be fitted'):
        ill._fit_polynomial_surface(np.full((4, 4), np.nan), (8, 8), 2, 1)


def test_trimming_stops_before_it_runs_out_of_bins():
    """Rejection stops rather than shrinking the fit below what it needs.

    One wild bin among twelve would leave eleven, fewer than the surface is
    allowed to be fitted from, so the round is abandoned and the previous
    coefficients stand instead of the trim being applied anyway.
    """
    stat = _plane_stat().ravel().copy()
    stat[:4] = np.nan      # leaves exactly the minimum usable bin count
    stat[5] = 50.0         # and one bin the plane cannot explain
    surface = ill._fit_polynomial_surface(stat.reshape(4, 4), (8, 8), 2, 1)

    assert surface.shape == (8, 8)
    assert np.all(np.isfinite(surface))


# ---------------------------------------------------------------------------
# Reading fields
# ---------------------------------------------------------------------------

def test_a_merged_field_that_is_not_a_stack_is_refused(tmp_path):
    """A 2-D ``.npy`` has no channel axis, so its dimensionality is named."""
    path = str(tmp_path / 'plate1_A01_F000.npy')
    np.save(path, np.zeros((8, 8), np.uint16))

    with pytest.raises(ill.IlluminationError, match='2-D array'):
        ill._read_binned_field(path, [0], 1, 256)


def test_fields_with_no_signal_at_all_cannot_give_a_profile():
    """A stack whose every field has a median of zero has nothing to scale by."""
    with pytest.raises(ill.IlluminationError, match='no signal'):
        ill._relative_profile(np.zeros((3, 4, 4)))


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def test_a_plate_that_contributes_no_field_is_named(tmp_path, monkeypatch):
    """An empty sample must not reach np.stack as an empty list.

    ``np.stack([])`` raises a bare ValueError with no plate in it; the plate
    the estimate failed on is the one thing the user needs to know.
    """
    src = _write_fields(tmp_path / 'merged', np.ones((8, 8)))
    monkeypatch.setattr(ill, '_sample', lambda paths, limit: [])

    with pytest.raises(ill.IlluminationError, match="plate 'plate1'"):
        ill.estimate_illumination(src, [0], verbose=False)


def test_a_dark_corner_is_floored_and_the_flooring_is_reported(tmp_path,
                                                               capsys):
    """A gain map is never allowed to divide by an almost-zero surface.

    Half a field at a thousandth of the other half fits a surface that goes
    below five percent of its own median, and dividing by that would multiply
    those pixels by hundreds. The surface is floored, and the count of floored
    pixels is printed, because a floored field is one to look at before
    trusting.
    """
    flat = np.ones((32, 32))
    flat[:, :16] = 0.001
    src = _write_fields(tmp_path / 'merged', flat)

    model = ill.estimate_illumination(src, [0], estimator='smooth',
                                      verbose=True)

    assert model.fields['plate1'].floored > 0
    printed = capsys.readouterr().out
    assert 'fell below' in printed
    assert 'floored' in printed


def test_a_surface_that_cannot_be_normalised_is_refused(tmp_path, monkeypatch):
    """A surface with a mean of zero cannot be inverted into a gain map."""
    src = _write_fields(tmp_path / 'merged', np.ones((8, 8)))
    monkeypatch.setattr(ill, '_fit_polynomial_surface',
                        lambda stat, shape, factor, degree: np.zeros(shape))

    with pytest.raises(ill.IlluminationError, match='cannot be'):
        ill.estimate_illumination(src, [0], verbose=False)


# ---------------------------------------------------------------------------
# The corrector
# ---------------------------------------------------------------------------

def test_an_unknown_on_missing_policy_is_refused_at_construction():
    """A misspelt policy must fail before a run, not silently mean 'error'."""
    model = _model(np.full((4, 4), 0.5, np.float32))

    with pytest.raises(ill.IlluminationError, match='on_missing='):
        ill.IlluminationCorrector(model, on_missing='ignore')


def test_a_quiet_corrector_reports_no_clipping(capsys):
    """``verbose=False`` silences the clipping report but still counts it."""
    model = _model(np.full((4, 4), 0.5, np.float32))
    corrector = ill.IlluminationCorrector(model, verbose=False)

    corrector._report_clipping(17, 16, None)

    assert capsys.readouterr().out == ''
    assert corrector._warned == 0


# ---------------------------------------------------------------------------
# Turning it on and off
# ---------------------------------------------------------------------------

def test_enabling_a_model_object_saves_it_beside_its_source(tmp_path,
                                                            capsys):
    """With no path given, the model is written next to the folder it came from.

    A worker started with ``spawn`` can only reach the model through the file
    system, so enabling a model that exists only in memory has to put it on
    disk somewhere predictable.
    """
    source = tmp_path / 'plate1' / 'merged'
    source.mkdir(parents=True)
    model = _model(np.full((4, 4), 0.5, np.float32))
    model.meta['src'] = [str(source)]
    # The environment route is read once per process; pretend it already has
    # been, which is the case where enabling must install the hook directly.
    mh._ENV_LOADED = True

    name = ill.enable_illumination_correction(model, verbose=True)

    assert name == ill.HOOK_NAME
    expected = str(tmp_path / 'plate1' / 'illumination' /
                   'illumination_model.npz')
    assert os.environ[ill.MODEL_ENV_VAR] == expected
    assert os.path.isfile(expected)
    assert [entry.name for entry in mh.preprocessing_hooks()] == \
        [ill.HOOK_NAME]
    printed = capsys.readouterr().out
    assert 'ENABLED' in printed and expected in printed

    assert ill.disable_illumination_correction() is True
    assert mh.HOOKS_ENV_VAR not in os.environ
    assert ill.MODEL_ENV_VAR not in os.environ


def test_a_model_with_no_source_is_saved_under_the_working_directory(
        tmp_path, monkeypatch):
    """A hand-built model still gets a definite path rather than None."""
    monkeypatch.chdir(tmp_path)
    model = _model(np.full((4, 4), 0.5, np.float32))

    ill.enable_illumination_correction(model, verbose=False)

    assert os.environ[ill.MODEL_ENV_VAR] == str(
        tmp_path / 'illumination' / 'illumination_model.npz')
    ill.disable_illumination_correction()


# ---------------------------------------------------------------------------
# The slope statistic and the QC report
# ---------------------------------------------------------------------------

def test_intensities_that_average_to_zero_have_no_slope():
    """A slope normalised by the mean is undefined when the mean is zero."""
    coordinates = np.array([[0.0, 0.0], [7.0, 7.0], [3.0, 4.0]])

    assert ill.position_intensity_slope([1.0, -1.0, 0.0], coordinates,
                                        (8, 8)) == 0.0
    assert ill.position_intensity_slope([5.0], [[0.0, 0.0]], (8, 8)) == 0.0


def test_qc_over_a_pooled_model_skips_fields_of_the_wrong_shape(tmp_path):
    """A pooled model covers every plate, and only matching fields are read.

    Correcting a 32x32 field with an 8x8 gain map cannot be made to mean
    anything, so those fields sit the report out; with none left the plate
    contributes no row rather than a row of nan.
    """
    src = _write_fields(tmp_path / 'merged', np.ones((32, 32)), channels=1)
    model = _model(np.full((8, 8), 1.0, np.float32), key=ill.ALL_PLATES)
    assert model.per_plate is False

    report = ill.illumination_qc(model, src, verbose=False)

    assert report == {}
    assert not os.path.isdir(str(tmp_path / 'illumination'))
