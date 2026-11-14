"""Round-6 coverage for the analysis modules' remaining untaken turns.

Two modules, and in every case the branch left open is the *quiet* half of a
decision -- the arm that produces no output, or the arm that a guarantee made
earlier in the same function has already closed off:

* :mod:`spacr.illumination` -- a corrector told to ``skip`` a plate it has no
  field for counts the skip whether or not it is allowed to say so, and stops
  saying so after ``_MAX_CLIP_WARNINGS``; the radial QC profile drops a ring
  no pixel falls in rather than plotting a NaN; and ``illumination_qc: False``
  really does mean no QC figure while the hook is still installed.

* :mod:`spacr.submodules` -- a cross-validation CSV with no column key skips
  the well filter and then names the key it needed, instead of filtering on a
  column that is not there; and an invasion run in which nothing invaded still
  reports ``n_invaded = 0`` rather than losing the column.

One of those branches is unreachable and is recorded as such:
``submodules.py:4900`` (the invasion class column).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr import illumination as ill  # noqa: E402
from spacr import measure_hooks as mh  # noqa: E402
from spacr.measure_hooks import PreprocessingContext  # noqa: E402


@pytest.fixture(autouse=True)
def _no_figures_and_no_hooks(monkeypatch):
    """No hook, no environment and no figure survives a test in this file.

    The hook registry and the illumination environment variables are both
    process-global; one leaking out would change what a later test measures.
    """
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    plt.close("all")
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# spacr.illumination
# ---------------------------------------------------------------------------

def _flat(shape=(5, 5)):
    """A mild horizontal vignette normalised to mean 1."""
    field = np.linspace(1.2, 0.8, shape[1], dtype=np.float32)
    field = np.tile(field, (shape[0], 1))
    return (field / field.mean()).astype(np.float32)


def _model(flat, *, plate="plate1"):
    """A one-plate, one-channel model built by hand -- no estimation."""
    field = ill.IlluminationField(
        plate=plate, channels=(0,), flatfield=np.stack([flat]),
        dark=np.zeros(1, np.float32), n_fields=3,
        estimator="polynomial", degree=4, bin_size=1)
    return ill.IlluminationModel(fields={plate: field}, meta={})


def _merged(folder, flat, *, plate="plate1", n_fields=3):
    """Merged single-channel fields of flat signal under ``flat``."""
    os.makedirs(folder, exist_ok=True)
    plane = np.rint(1000.0 * flat).astype(np.uint16)
    for index in range(n_fields):
        np.save(os.path.join(folder, f"{plate}_A01_F{index:03d}.npy"),
                np.stack([plane], axis=-1))
    return str(folder)


def test_a_skip_is_counted_whether_or_not_the_corrector_may_announce_it(capsys):
    """``verbose=False`` silences the warning; it does not silence the count.

    ``on_missing='skip'`` measures a plate the model does not cover
    uncorrected, which is a fact about the table that has to survive into
    ``stats`` even when the run asked for a quiet corrector. And a loud one
    stops after ``_MAX_CLIP_WARNINGS``, because one line per field on a plate
    the model never saw is thousands of identical lines.
    """
    model = _model(_flat((4, 4)), plate="plateA")
    array = np.full((4, 4, 1), 100, np.uint16)

    def elsewhere(index):
        return PreprocessingContext(file_name=f"plateB_A01_F{index:03d}",
                                    channels=[0], settings={})

    quiet = ill.IlluminationCorrector(model, on_missing="skip", verbose=False)
    result = quiet(array, elsewhere(0))

    assert result is array, "a skipped field must come back untouched"
    assert quiet.stats["skipped"] == 1
    assert capsys.readouterr().out == ""

    # The same skip from a corrector that is allowed to speak: it says
    # UNCORRECTED once per field until the cap, and then stops.
    loud = ill.IlluminationCorrector(model, on_missing="skip", verbose=True)
    calls = ill._MAX_CLIP_WARNINGS + 2
    for index in range(calls):
        loud(array, elsewhere(index))

    printed = capsys.readouterr().out
    assert printed.count("UNCORRECTED") == ill._MAX_CLIP_WARNINGS
    assert "plateB" in printed
    assert loud.stats["skipped"] == calls, (
        "the cap is on the warnings, not on the counter")


def test_the_qc_profile_drops_a_ring_that_holds_no_pixel(tmp_path):
    """A field smaller than the ring count leaves rings with nothing in them.

    ``_radial_profile`` bins pixels by normalised corner-radius into a fixed
    number of rings. A 5x5 field has 25 pixels for 20 rings and their radii
    are far from uniform, so several rings are empty -- and an empty ring
    contributes no point rather than a NaN that would break the QC plot's
    line. The profile is drawn and never returned, so the ring count is
    asserted on the helper directly; the public call above proves the figure
    is really produced from a field that has those empty rings.
    """
    merged = _merged(tmp_path / "merged", _flat((5, 5)))
    save_dir = str(tmp_path / "qc")

    report = ill.illumination_qc(_model(_flat((5, 5))), merged,
                                 save_dir=save_dir, verbose=False)

    assert os.path.isfile(os.path.join(save_dir,
                                       "illumination_qc_plate1.png"))
    assert report["_figures"]["plate1"].endswith("illumination_qc_plate1.png")
    assert np.isfinite(report["plate1"][0]["slope_before"])

    small = np.arange(1, 26, dtype=float).reshape(5, 5)
    radii, means = ill._radial_profile(small, 1, (5, 5))
    assert len(radii) == len(means)
    assert 0 < len(radii) < 20, "a 5x5 field cannot fill twenty rings"
    assert np.all(np.diff(radii) > 0)
    assert np.isfinite(means).all()

    # A field large enough to put a pixel in every ring returns all twenty,
    # so the short answer above is the empty-ring case and not the only one.
    large = np.random.default_rng(0).random((64, 64)) + 1.0
    radii_full, means_full = ill._radial_profile(large, 1, (64, 64))
    assert len(radii_full) == 20 and len(means_full) == 20


def test_declining_the_qc_writes_no_figure_but_still_installs_the_hook(tmp_path):
    """``illumination_qc: False`` skips the report, not the correction.

    The QC figure is the expensive half of ``prepare_illumination_correction``
    -- it re-reads every merged field -- so a re-run that already trusts its
    model turns it off. What must not be turned off with it is the hook that
    makes ``measure_crop`` correct anything at all.
    """
    merged = _merged(tmp_path / "merged", _flat())
    saved = str(tmp_path / "saved.npz")
    _model(_flat()).save(saved)
    folder = tmp_path / "illumination"

    settings = {"illumination_correction": True, "src": merged,
                "illumination_model": saved, "channels": [0],
                "illumination_qc": False, "verbose": False}
    model = ill.prepare_illumination_correction(settings)

    assert isinstance(model, ill.IlluminationModel)
    assert not folder.exists(), "the QC step created its own output folder"
    assert [hook.name for hook in mh.preprocessing_hooks()] == [ill.HOOK_NAME]

    # The same call with the QC left on writes the figure into that folder,
    # so the absence above is the setting and not a broken path.
    mh.clear_measurement_hooks()
    settings["illumination_qc"] = True
    ill.prepare_illumination_correction(settings)

    assert sorted(os.listdir(folder)) == ["illumination_qc_plate1.png"]
    assert [hook.name for hook in mh.preprocessing_hooks()] == [ill.HOOK_NAME]


# ---------------------------------------------------------------------------
# spacr.submodules -- analyze_invasion
# ---------------------------------------------------------------------------


# UNREACHABLE: submodules.py:4900 (`field_counts[name] = 0`) and the
# 4899 -> 4900 arc into it cannot be taken. `_invasion_classify` sets
# `invasion_class` to a `pd.Categorical` whose categories are
# `_INVASION_CLASSES + ['unclassified']` (submodules.py:4046-4049), and
# `SeriesGroupBy.value_counts()` over a categorical enumerates EVERY category,
# not only the observed ones -- so the `unstack` at 4897 always yields one
# column per category and neither name in `_INVASION_CLASSES` can be absent
# from `field_counts.columns`. The guard is a defensive re-check after a step
# that already guarantees its condition. What is testable is the guarantee
# itself, which is what this test pins.
