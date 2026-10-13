"""The scripted entry points expand, refuse, and dispatch a real run.

`MaskConfig` and `MeasureConfig` are what a script writes instead of a
settings dict; each expands through the same defaults the GUI uses, refuses
a configuration that cannot mean anything, and hands the result to the
pipeline. A ``dry_run`` configuration takes the whole path through
:func:`spacr.core.preprocess_generate_masks` without writing anything.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.api import MaskConfig, MeasureConfig, run_mask, run_measure


def test_several_plate_folders_survive_expansion(tmp_path):
    """A sequence of sources stays a list of strings, one per plate."""
    plates = [tmp_path / "plate01", tmp_path / "plate02"]
    for plate in plates:
        plate.mkdir()

    settings = MaskConfig(plates, cell_channel=0).to_settings()

    assert settings["src"] == [str(p) for p in plates]


def test_a_pathlib_source_survives_expansion(tmp_path):
    """A single ``Path`` is one source, not an iterable of characters."""
    settings = MaskConfig(Path(tmp_path), nucleus_channel=1).to_settings()

    assert settings["src"] == str(tmp_path)


def test_a_mask_run_with_no_channel_is_refused(tmp_path):
    """Segmenting nothing is a configuration error, not an empty run."""
    with pytest.raises(ValueError, match="at least one segmentation channel"):
        MaskConfig(tmp_path).to_settings()


def test_an_unknown_pipeline_style_is_refused(tmp_path):
    """Only the two shipped pipelines exist; a typo must not run either."""
    with pytest.raises(ValueError, match="v1.*v2"):
        MaskConfig(tmp_path, cell_channel=0,
                   pipeline_style="fast").to_settings()


def test_a_measure_run_with_no_mask_plane_is_refused(tmp_path):
    """Measuring no object type would write an empty database."""
    with pytest.raises(ValueError, match="at least one mask plane"):
        MeasureConfig(tmp_path, cell_mask_dim=None, nucleus_mask_dim=None,
                      pathogen_mask_dim=None).to_settings()


def test_extra_may_not_repeat_a_typed_setting(tmp_path):
    """Two answers for one setting is a script bug, and it is named."""
    with pytest.raises(ValueError, match="cell_channel"):
        MaskConfig(tmp_path, cell_channel=0,
                   extra={"cell_channel": 2}).to_settings()


def test_a_dry_mask_run_reaches_the_pipeline_and_writes_nothing(
        yokogawa_cellvoyager_dir):
    """`run_mask` dispatches into core and a dry run leaves the folder alone."""
    src = Path(yokogawa_cellvoyager_dir["src"])
    before = sorted(p.name for p in src.iterdir())

    problems = run_mask(MaskConfig(src, cell_channel=0, nucleus_channel=1,
                                   cell_diameter=30, nucleus_diameter=15,
                                   dry_run=True))

    assert problems is not None
    assert sorted(p.name for p in src.iterdir()) == before


def test_a_dry_measure_run_reaches_the_pipeline_and_writes_nothing(tmp_path):
    """`run_measure` dispatches into measure and a dry run writes nothing."""
    merged = tmp_path / "merged"
    merged.mkdir()

    problems = run_measure(MeasureConfig(merged, save_png=False,
                                         dry_run=True))

    assert problems is not None
    assert list(merged.iterdir()) == []


def test_a_settings_mapping_is_accepted_without_a_config(monkeypatch,
                                                         tmp_path):
    """An existing settings dict is passed through unchanged, and copied."""
    import spacr.core as core

    seen = {}

    def _record(settings):
        seen.update(settings)
        return "ran"

    monkeypatch.setattr(core, "preprocess_generate_masks", _record)
    original = {"src": str(tmp_path), "cell_channel": 0}

    assert run_mask(original) == "ran"
    assert seen["cell_channel"] == 0
    seen["cell_channel"] = 9
    assert original["cell_channel"] == 0


def test_a_measure_settings_mapping_is_accepted_without_a_config(monkeypatch,
                                                                 tmp_path):
    """The same passthrough on the measure side, copied the same way."""
    import spacr.measure as measure

    seen = {}

    def _record(settings):
        seen.update(settings)
        return "measured"

    monkeypatch.setattr(measure, "measure_crop", _record)
    original = {"src": str(tmp_path), "cell_mask_dim": 4}

    assert run_measure(original) == "measured"
    assert seen["cell_mask_dim"] == 4
    seen["cell_mask_dim"] = 9
    assert original["cell_mask_dim"] == 4
