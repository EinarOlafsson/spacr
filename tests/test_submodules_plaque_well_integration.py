"""The plaque analysis' well detector, model resolution, and physical ruler."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import submodules as sm
from spacr.plaque import Well


def _geometry() -> dict[str, int]:
    return {"x0": 0, "y0": 0, "x1": 348, "y1": 348}


def test_a_detected_well_supplies_the_physical_scale():
    settings = {
        "_well_geometry": {"field_well01.tif": _geometry()},
        "plate_format": "6-well",
    }

    scale = sm._plaque_scale_for("field_well01.tif", settings)

    assert scale is not None
    assert scale.px_per_mm == pytest.approx(10.0)
    assert scale.area_mm2(5_000) == pytest.approx(50.0)


def test_an_unknown_plate_format_leaves_pixels_unscaled(caplog):
    settings = {
        "_well_geometry": {"field_well01.tif": _geometry()},
        "plate_format": "5-well",
    }

    with caplog.at_level("WARNING"):
        scale = sm._plaque_scale_for("field_well01.tif", settings)

    assert scale is None
    assert "not a known format" in caplog.text


def test_a_disabled_well_detector_leaves_the_source_untouched(tmp_path):
    settings = {"src": str(tmp_path), "well_detection": False}

    assert sm._resolve_well_detector(settings) is None


def test_a_local_well_detector_checkpoint_wins(tmp_path):
    checkpoint = tmp_path / "wells.pt"
    checkpoint.write_bytes(b"weights")

    assert sm._resolve_well_detector({"well_detection": checkpoint}) == str(
        checkpoint
    )


def test_true_fetches_the_default_well_detector(tmp_path, monkeypatch):
    from spacr import model_zoo

    entry = SimpleNamespace(key="toxoplasma_well_detector_v1")
    fetched = tmp_path / "downloaded.pt"
    calls = []
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])

    def fetch(chosen, destination):
        calls.append((chosen, Path(destination)))
        return fetched

    monkeypatch.setattr(model_zoo, "fetch", fetch)

    assert sm._resolve_well_detector({"well_detection": True}) == str(fetched)
    assert calls == [(entry, tmp_path / ".spacr" / "models")]


def test_an_unknown_well_detector_key_is_refused(monkeypatch):
    from spacr import model_zoo

    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [])

    with pytest.raises(ValueError, match="neither a file nor a model_zoo key"):
        sm._resolve_well_detector({"well_detection": "not-a-detector"})


def test_split_wells_is_inert_when_detection_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_resolve_well_detector", lambda _settings: None)
    settings = {"src": str(tmp_path)}

    assert sm.split_wells(settings) == str(tmp_path)
    assert "_well_geometry" not in settings


def test_no_detected_well_passes_the_original_images_through(
    tmp_path, monkeypatch, caplog
):
    from spacr import plaque

    (tmp_path / "field.tif").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "folder.png").mkdir()
    monkeypatch.setattr(sm, "_resolve_well_detector", lambda _settings: "wells.pt")
    monkeypatch.setattr(sm.cellpose.io, "imread", lambda _path: np.zeros((8, 8)))
    monkeypatch.setattr(plaque, "detect_wells", lambda *_args, **_kwargs: [])
    settings = {"src": str(tmp_path)}

    with caplog.at_level("WARNING"):
        result = sm.split_wells(settings)

    assert result == str(tmp_path)
    assert "_well_geometry" not in settings
    assert "no wells detected in field.tif" in caplog.text


def test_detected_wells_are_cropped_named_and_recorded(tmp_path, monkeypatch):
    from spacr import plaque

    (tmp_path / "plate.tif").write_bytes(b"image")
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    wells = [Well(0, 0, 4, 4, 0.9), Well(4, 4, 9, 9, 0.8)]
    saved = {}
    crop_calls = []
    monkeypatch.setattr(sm, "_resolve_well_detector", lambda _settings: "wells.pt")
    monkeypatch.setattr(sm.cellpose.io, "imread", lambda _path: image)
    monkeypatch.setattr(
        plaque, "detect_wells", lambda _image, _weights, confidence: wells
    )

    def crop(source, well, *, pad=0):
        crop_calls.append((source, well, pad))
        return np.full((2, 2), well.confidence)

    def save(path, pixels):
        saved[Path(path).name] = np.asarray(pixels)
        Path(path).write_bytes(b"crop")

    monkeypatch.setattr(plaque, "crop_well", crop)
    monkeypatch.setattr(sm.cellpose.io, "imsave", save)
    settings = {
        "src": str(tmp_path),
        "well_confidence": 0.7,
        "well_pad": 3,
    }

    result = sm.split_wells(settings)

    assert result == str(tmp_path / "wells")
    assert set(saved) == {"plate_well01.tif", "plate_well02.tif"}
    assert [call[2] for call in crop_calls] == [3, 3]
    assert settings["_well_geometry"] == {
        "plate_well01.tif": wells[0].as_dict(),
        "plate_well02.tif": wells[1].as_dict(),
    }


def test_the_plate_reaches_the_detector_and_the_crop_in_reader_order(
    tmp_path, monkeypatch
):
    """The split pass hands on what the reader gave it, and converts nothing.

    ``cellpose.io.imread`` is spaCR's house reader and returns RGB. Since
    instruction 445 the BGR conversion ultralytics needs lives inside
    :func:`spacr.plaque.detect_wells`, at its own door, so EVERY caller is
    right rather than each caller remembering. A conversion here as well
    would swap twice and restore the bug, and -- the reason this is asserted
    of ``crop_well`` too -- the pixels that get SAVED must stay RGB, or every
    well crop the user opens has red and blue exchanged.
    """
    from spacr import plaque

    (tmp_path / "plate.tif").write_bytes(b"image")
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    seen = {}
    monkeypatch.setattr(sm, "_resolve_well_detector", lambda _settings: "wells.pt")
    monkeypatch.setattr(sm.cellpose.io, "imread", lambda _path: image)
    monkeypatch.setattr(sm.cellpose.io, "imsave", lambda *_a, **_k: None)

    def detect(source, _weights, confidence):
        seen["detect"] = source
        return [Well(0, 0, 4, 4, 0.9)]

    def crop(source, well, *, pad=0):
        seen["crop"] = source
        return source[well.y0:well.y1, well.x0:well.x1]

    monkeypatch.setattr(plaque, "detect_wells", detect)
    monkeypatch.setattr(plaque, "crop_well", crop)

    sm.split_wells({"src": str(tmp_path)})

    assert seen["detect"] is image, "detect_wells is given the reader's RGB"
    assert seen["crop"] is image, "and so is the crop that gets written out"


def test_an_image_with_no_detected_well_still_reaches_the_analysis(
    tmp_path, monkeypatch, caplog
):
    """The silent loss the warning always denied.

    The dangerous case is PARTIAL detection. An image nothing is found in
    used to be warned about and then skipped, so when some images in a folder
    split and others did not, the others produced no crop, no row, and no
    further mention -- while the warning said they had been "passed through
    whole". They are now copied into the split folder whole, with no geometry,
    so they are analysed and simply carry no physical scale.
    """
    from spacr import plaque

    (tmp_path / "plate.tif").write_bytes(b"image")
    (tmp_path / "blank.tif").write_bytes(b"image")
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    saved = {}
    monkeypatch.setattr(sm, "_resolve_well_detector", lambda _settings: "wells.pt")
    monkeypatch.setattr(sm.cellpose.io, "imread", lambda _path: image)
    monkeypatch.setattr(
        plaque, "detect_wells",
        lambda _image, _weights, confidence: (
            [Well(0, 0, 4, 4, 0.9)] if not saved else []),
    )
    monkeypatch.setattr(plaque, "crop_well",
                        lambda source, well, *, pad=0: source[:2, :2])

    def save(path, pixels):
        saved[Path(path).name] = np.asarray(pixels)

    monkeypatch.setattr(sm.cellpose.io, "imsave", save)
    settings = {"src": str(tmp_path)}

    with caplog.at_level("WARNING"):
        result = sm.split_wells(settings)

    assert result == str(tmp_path / "wells")
    assert set(saved) == {"blank_well01.tif", "plate.tif"}
    assert np.array_equal(saved["plate.tif"], image), "whole, not cropped"
    assert "plate.tif" not in settings["_well_geometry"], "and with no ruler"
    assert sm._plaque_scale_for("plate.tif", settings) is None
    assert "no wells detected in plate.tif" in caplog.text


def test_a_local_plaque_checkpoint_wins(tmp_path):
    checkpoint = tmp_path / "plaque.CP_model"
    checkpoint.write_bytes(b"model")

    assert sm._resolve_plaque_model({"plaque_model": checkpoint}) == str(
        checkpoint
    )


def test_a_missing_bundled_plaque_model_is_named(tmp_path, monkeypatch):
    from spacr import utils

    monkeypatch.setattr(utils, "download_models", lambda: tmp_path / "models")
    monkeypatch.setattr(sm, "__file__", str(tmp_path / "package" / "submodules.py"))

    with pytest.raises(sm.ModelZooMissing, match="bundled plaque model"):
        sm._resolve_plaque_model({"plaque_model": "bundled"})


def test_an_unknown_plaque_model_key_lists_the_known_keys(monkeypatch):
    from spacr import model_zoo

    entry = SimpleNamespace(key="known-plaque")
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])

    with pytest.raises(ValueError, match="known-plaque"):
        sm._resolve_plaque_model({"plaque_model": "unknown-plaque"})


def test_a_plaque_model_zoo_key_is_fetched(tmp_path, monkeypatch):
    from spacr import model_zoo

    entry = SimpleNamespace(key="chosen-plaque")
    fetched = tmp_path / "chosen.CP_model"
    calls = []
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])

    def fetch(chosen, destination):
        calls.append((chosen, Path(destination)))
        return fetched

    monkeypatch.setattr(model_zoo, "fetch", fetch)

    assert sm._resolve_plaque_model({"plaque_model": entry.key}) == str(fetched)
    assert calls == [(entry, tmp_path / ".spacr" / "models")]


def test_analyze_plaques_hands_detected_well_crops_to_the_analysis(
    tmp_path, monkeypatch
):
    import tifffile

    from spacr import utils

    source = tmp_path / "source"
    split = tmp_path / "split"
    masks = split / "masks"
    source.mkdir()
    masks.mkdir(parents=True)
    model = tmp_path / "plaque.CP_model"
    model.write_bytes(b"model")
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[2:6, 2:6] = 1
    tifffile.imwrite(masks / "plate_well01.tif", labels)
    seen = []

    def split_first(settings):
        seen.append(settings["src"])
        return str(split)

    monkeypatch.setattr(sm, "split_wells", split_first)
    monkeypatch.setattr(utils, "save_settings", lambda *_args, **_kwargs: None)
    settings = {
        "src": str(source),
        "masks": False,
        "well_detection": "detector.pt",
        "plaque_model": str(model),
    }

    assert sm.analyze_plaques(settings) is None
    assert seen == [str(source)]
    assert settings["src"] == str(split)
    assert (masks / "plaques_analysis.db").is_file()
