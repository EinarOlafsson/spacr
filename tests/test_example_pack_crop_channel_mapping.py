"""317: the shipped Measure pack's legacy RGB mapping must reach the run."""
from __future__ import annotations

import csv

import pytest

from spacr.crops import resolve_png_channel_mapping
from spacr.qt.settings_pack import settings_from_pack
from spacr.qt.screens.settings_model import resolve_default_settings
from spacr.settings import get_measure_crop_settings


@pytest.mark.parametrize("pipeline", [False, True], ids=["form", "pipeline"])
@pytest.mark.parametrize("current", [False, True], ids=["legacy", "current-wins"])
@pytest.mark.parametrize("legacy_last", [False, True])
def test_legacy_crop_channels_survive_without_overriding_current_mapping(
        tmp_path, pipeline, current, legacy_last):
    defaults = (get_measure_crop_settings(settings={}) if pipeline else
                resolve_default_settings("measure"))
    assert "png_channel_mapping" in defaults
    assert "png_dims" not in defaults
    assert defaults["png_channel_mapping"] != {"b": 1, "g": 3, "r": 0}
    rows = [("png_dims", [1, 3, 0])]
    expected = {"b": 1, "g": 3, "r": 0}
    if current:
        expected = {"b": 2, "g": 0, "r": 1}
        rows.append(("png_channel_mapping", expected))
    if legacy_last:
        rows.reverse()
    with (tmp_path / "crop_measure_settings.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Key", "Value"))
        writer.writerows(rows)

    settings, report = settings_from_pack("measure", str(tmp_path), defaults=defaults)

    assert settings["png_channel_mapping"] == expected
    assert resolve_png_channel_mapping(settings) == expected
    assert report.renamed == [("png_dims", "png_channel_mapping")]
    assert report.dropped == []
    assert len(report.applied) + len(report.renamed) == len(rows)


def test_legacy_only_caller_keeps_its_own_supported_schema(tmp_path):
    (tmp_path / "measure_settings.csv").write_text(
        'Key,Value\npng_dims,"[1, 3, 0]"\n', encoding="utf-8")

    settings, report = settings_from_pack(
        "measure", str(tmp_path), defaults={"png_dims": [0, 1, 2]})

    assert settings == {"png_dims": [1, 3, 0]}
    assert report.applied == ["png_dims"]
    assert report.renamed == report.dropped == []
