"""Optional measurement families are added or skipped, never half-added.

Three families in :mod:`spacr.measure` are opt-in or depend on an optional
package: Zernike morphology, the object-distance block, and the organelle
spatial columns. Each one has to reach the frame complete when it is
available and leave the frame untouched when it is not -- a column that is
present but empty deletes itself out of every model matrix downstream, which
is a silent loss rather than a visible skip.
"""
from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pandas as pd
import pytest

import spacr.measure as measure


def _block_mahotas(monkeypatch):
    """Exercise the optional-dependency branch even in the full CI profile."""
    original = builtins.__import__
    monkeypatch.setattr(measure, "_ZERNIKE_AVAILABLE", None)

    def guarded(name, *args, **kwargs):
        if name == "mahotas" or name.startswith("mahotas."):
            raise ModuleNotFoundError("blocked Mahotas for test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


# ---------------------------------------------------------------------------
# Zernike, with a stand-in for the optional package
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub_mahotas(monkeypatch):
    """Stand in for the optional Mahotas descriptor.

    The moments themselves are Mahotas' arithmetic; what spaCR owns is the
    plumbing -- the per-region radius, the column names, and the join back
    onto the props frame. That plumbing is what this exercises, and it can
    only run when the package resolves.
    """
    calls = []

    def zernike_moments(image, radius, degree=8):
        calls.append((image.shape, float(radius), int(degree)))
        return np.arange(9, dtype=float) + float(radius)

    features = types.ModuleType("mahotas.features")
    features.zernike_moments = zernike_moments
    package = types.ModuleType("mahotas")
    package.features = features
    monkeypatch.setitem(sys.modules, "mahotas", package)
    monkeypatch.setitem(sys.modules, "mahotas.features", features)
    zernike_moments.calls = calls
    return zernike_moments


def test_without_the_optional_package_zernike_is_named_not_guessed(
        monkeypatch):
    """The message has to say what to install, not just that it failed."""
    _block_mahotas(monkeypatch)
    with pytest.raises(ImportError, match=r"spacr\[zernike\]"):
        measure._load_zernike_moments()


def test_with_the_package_present_the_descriptor_is_returned(stub_mahotas):
    """The loader hands back the callable rather than importing at module load."""
    assert measure._load_zernike_moments() is stub_mahotas


def test_the_moments_are_appended_as_one_column_per_coefficient(stub_mahotas):
    """Every region gets the same columns, joined onto the props in row order.

    The radius is scaled per object so the coefficients are comparable
    across object sizes; a fixed radius described a 2000 px cell on an 8 px
    disk.
    """
    mask = np.zeros((20, 20), np.int32)
    mask[2:6, 2:6] = 1
    mask[10:18, 10:18] = 2
    frame = pd.DataFrame({"label": [1, 2], "area": [16.0, 64.0]})

    out = measure._calculate_zernike(mask, frame)

    assert [c for c in out.columns if c.startswith("zernike_")] == \
        [f"zernike_{i}" for i in range(9)]
    assert len(out) == 2
    radii = [call[1] for call in stub_mahotas.calls]
    assert radii[1] > radii[0], "the disk must grow with the object"


def test_moment_vectors_of_different_lengths_are_refused(monkeypatch,
                                                         stub_mahotas):
    """Ragged coefficients cannot become columns, and must not be padded.

    Padding would put one object's coefficient 5 in another's coefficient 5
    column, which is a wrong number rather than a missing one.
    """
    lengths = iter([9, 7])

    def ragged(image, radius, degree=8):
        return np.zeros(next(lengths), dtype=float)

    monkeypatch.setattr(sys.modules["mahotas.features"], "zernike_moments",
                        ragged)
    mask = np.zeros((20, 20), np.int32)
    mask[2:6, 2:6] = 1
    mask[10:18, 10:18] = 2

    with pytest.raises(ValueError, match="same length"):
        measure._calculate_zernike(mask, pd.DataFrame({"label": [1, 2]}))


def _masks(size=40):
    cell = np.zeros((size, size), np.int32)
    cell[4:16, 4:16] = 1
    cell[22:34, 22:34] = 2
    nucleus = np.zeros_like(cell)
    nucleus[6:12, 6:12] = 1
    nucleus[24:30, 24:30] = 2
    pathogen = np.zeros_like(cell)
    pathogen[13:15, 13:15] = 1
    pathogen[31:33, 31:33] = 2
    return cell, nucleus, pathogen


def _settings(**over):
    base = {"cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
            "organelle_mask_dim": None, "cytoplasm": False,
            "channels": [0, 1], "spatial_measurements": False,
            "object_distances": False}
    base.update(over)
    return base


def test_zernike_columns_reach_every_object_frame(stub_mahotas):
    """Cell, nucleus and pathogen frames all gain the descriptor.

    Adding it to only one object type would make two objects in the same
    table describable and the third not, with nothing saying why.
    """
    cell, nucleus, pathogen = _masks()

    cell_df, nucleus_df, pathogen_df, _organelle, _cyto = \
        measure._morphological_measurements(
            cell, nucleus, pathogen, None, None, _settings(), zernike=None)

    for prefix, frame in (("cell", cell_df), ("nucleus", nucleus_df),
                          ("pathogen", pathogen_df)):
        assert f"{prefix}_zernike_0" in frame.columns


def test_without_the_package_no_object_frame_gains_the_columns(monkeypatch):
    """The control: absent everywhere rather than absent in some frames."""
    _block_mahotas(monkeypatch)
    cell, nucleus, pathogen = _masks()

    frames = measure._morphological_measurements(
        cell, nucleus, pathogen, None, None, _settings(), zernike=None)

    for frame in frames[:3]:
        assert not [c for c in frame.columns if "zernike" in c]


def test_the_organelle_frame_gains_the_columns_too(stub_mahotas):
    """An organelle is measured like any other object once it has a mask."""
    cell, nucleus, pathogen = _masks()
    organelle = np.zeros_like(cell)
    organelle[7:11, 7:11] = 1
    organelle[25:29, 25:29] = 2

    frames = measure._morphological_measurements(
        cell, nucleus, pathogen, organelle, None,
        _settings(organelle_mask_dim=7), zernike=None)

    assert "organelle_zernike_0" in frames[3].columns


# ---------------------------------------------------------------------------
# The object-distance block
# ---------------------------------------------------------------------------

def test_object_distances_widen_every_object_frame_when_asked_for():
    """Opt-in and off by default, because it is real time on a 3-D field."""
    cell, nucleus, pathogen = _masks()
    images = np.stack([np.full(cell.shape, 100, np.uint16),
                       np.full(cell.shape, 200, np.uint16)], axis=-1)

    cell_df, nucleus_df, pathogen_df, _o, _c = \
        measure._morphological_measurements(
            cell, nucleus, pathogen, None, None,
            _settings(object_distances=True), zernike=False,
            channel_arrays=images)

    added = [c for c in cell_df.columns if "distance" in c or "nearest" in c]
    assert added, "the block added no columns at all"
    assert list(cell_df.columns)[0] == "label", \
        "label has to keep column position 0"
    assert len(nucleus_df) == 2 and len(pathogen_df) == 2


def test_a_distance_family_that_fails_leaves_the_rest_of_the_frame_correct(
        monkeypatch, capsys):
    """A measurement family that fails is not a failed run.

    Every other measurement in the frame is still correct, so the run keeps
    them and says on the console which family was lost.
    """
    from spacr import object_distances as od

    def refuse(*_args, **_kwargs):
        raise RuntimeError("the distance transform ran out of memory")

    monkeypatch.setattr(od, "object_distances", refuse)
    cell, nucleus, pathogen = _masks()

    cell_df, _n, _p, _o, _c = measure._morphological_measurements(
        cell, nucleus, pathogen, None, None,
        _settings(object_distances=True), zernike=False)

    assert "cell_area" in cell_df.columns
    assert not [c for c in cell_df.columns if "nearest" in c]
    assert "object distances for cell were not measured" in capsys.readouterr().out


def test_a_distance_block_with_nothing_but_labels_is_not_merged(monkeypatch):
    """A block of one column carries no measurement, so it is left off.

    Merging it would add nothing and cost a join on every field.
    """
    from spacr import object_distances as od

    monkeypatch.setattr(od, "object_distances",
                        lambda *a, **k: pd.DataFrame({"label": [1, 2]}))
    cell, nucleus, pathogen = _masks()

    cell_df, _n, _p, _o, _c = measure._morphological_measurements(
        cell, nucleus, pathogen, None, None,
        _settings(object_distances=True), zernike=False)

    assert list(cell_df.columns).count("label") == 1


# ---------------------------------------------------------------------------
# Every organelle type keeps requested spatial columns
# ---------------------------------------------------------------------------

def test_the_compatibility_predicate_never_gates_a_measurement():
    """Type contributes caveats, never a silent family-level switch."""
    assert measure._spatial_organelle_eligible({}) is True
    assert measure._spatial_organelle_eligible(
        {"organelle_type": "reticular",
         "organelle_morphology": "network"}) is True


def test_a_network_organelle_still_writes_every_requested_spatial_column():
    """The output frame proves the production path no longer consults Type."""
    cell, nucleus, pathogen = _masks()
    organelle = np.zeros_like(cell)
    organelle[7:10, 7:10] = 1
    organelle[25:28, 25:28] = 2
    settings = _settings(
        organelle_mask_dim=7,
        spatial_measurements=True,
        organelle_type="reticular",
        organelle_morphology="network",
    )

    frames = measure._morphological_measurements(
        cell, nucleus, pathogen, organelle, None, settings, zernike=False)

    organelle_frame = frames[3]
    expected = {
        f"organelle_{name}" for name in measure.spatial_column_names(50)
    }
    assert expected <= set(organelle_frame.columns)
    assert len(organelle_frame) == 2


def test_no_organelle_type_named_defers_to_the_morphology_setting():
    """A settings file from before the type presets carries only morphology."""
    assert measure._morphology_of_organelle_type({}) is None
    assert measure._morphology_of_organelle_type({"organelle_type": ""}) is None


def test_an_organelle_type_that_cannot_be_resolved_defers_too(monkeypatch):
    """An unrecognised preset falls back rather than guessing a morphology."""
    from spacr import organelle_types

    def refuse(_name):
        raise ValueError("no such organelle preset")

    monkeypatch.setattr(organelle_types, "resolve_type", refuse)

    assert measure._morphology_of_organelle_type(
        {"organelle_type": "not-a-preset"}) is None


# ---------------------------------------------------------------------------
# An upper size bound, and saying what it removed
# ---------------------------------------------------------------------------

def _merged_stack(size=96):
    """One small and one huge object of each type, in a merged (H, W, C) stack."""
    cell = np.zeros((size, size), np.uint16)
    cell[4:16, 4:16] = 1            # 144 px
    cell[30:90, 30:90] = 2          # 3600 px
    nucleus = np.zeros_like(cell)
    nucleus[6:12, 6:12] = 1         # 36 px
    nucleus[40:80, 40:80] = 2       # 1600 px
    pathogen = np.zeros_like(cell)
    pathogen[13:15, 13:15] = 1      # 4 px
    pathogen[50:70, 50:70] = 2      # 400 px
    rng = np.random.default_rng(0)
    channels = []
    for _ in range(4):
        plane = rng.integers(50, 200, size=(size, size)).astype(np.uint16)
        plane[cell > 0] += 3000
        channels.append(plane)
    return np.stack(channels + [cell, nucleus, pathogen], axis=-1)


def _crop_settings(merged_dir, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings(settings={})
    settings.update({
        "src": str(merged_dir), "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [32, 32],
        "save_measurements": True, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True,
    })
    settings.update(over)
    return settings


@pytest.mark.parametrize("kind,limit", [
    ("cell", 1000), ("nucleus", 500), ("pathogen", 100),
])
def test_an_upper_size_bound_says_how_many_objects_it_removed(
        tmp_path, capsys, kind, limit):
    """A maximum removes a segmentation blow-up, and must report doing so.

    A blow-up passes every minimum and carries its area into every ratio
    downstream, so the run has to say the bound fired rather than silently
    producing a plate with fewer objects than the masks hold.
    """
    from spacr.measure import _measure_crop_core

    merged = tmp_path / "merged"
    merged.mkdir(parents=True)
    (tmp_path / "measurements").mkdir(parents=True)
    name = "plate1_A01_F001.npy"
    np.save(merged / name, _merged_stack())

    settings = _crop_settings(merged, **{f"{kind}_max_size": limit})
    _measure_crop_core(0, [], name, settings)

    printed = capsys.readouterr().out
    assert f"{kind}: 1 object(s) outside" in printed
    assert f"{limit}] px" in printed
