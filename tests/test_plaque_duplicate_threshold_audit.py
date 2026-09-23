"""The measurement rejects altered sources and preserves all input artifacts."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from tools import evaluate_plaque_duplicate_threshold as audit


@pytest.fixture
def corpus(tmp_path):
    images = tmp_path / "images"
    figures = []
    for index in range(2):
        path = images / f"PMC{index}" / "figure.png"
        path.parent.mkdir(parents=True)
        image = np.random.default_rng(index).integers(0, 256, (97, 131, 3), dtype=np.uint8)
        Image.fromarray(image).save(path)
        figures.append({"key": f"PMC{index}__figure.png", "pmcid": f"PMC{index}",
                        "doi": f"10.test/{index}", "licence": "cc by",
                        "pixels": [131, 97],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"arms": {"test": {"figures": figures}}}))
    return manifest, images, tmp_path / "output.json"


def invoke(monkeypatch, corpus):
    manifest, images, output = corpus
    monkeypatch.setattr(sys, "argv", ["audit", "--manifest", str(manifest),
                                      "--images", str(images), "--output", str(output)])
    audit.main()


def test_real_encodings_are_measured_without_changing_the_sources(corpus, monkeypatch):
    manifest, images, output = corpus
    before = {path: path.read_bytes() for path in [manifest, *images.rglob("*.png")]}
    invoke(monkeypatch, corpus)
    result = json.loads(output.read_text())
    assert len(result["references"]) == 2
    assert len(result["controls"]) == 16
    assert result["distinct_source_ordered_pairs"] == 2
    assert result["aspect_eligible_ordered_pairs"] == 2
    assert result["by_control"]["png"]["skipped_exact"] == 2
    assert result["by_control"]["bmp"]["skipped_exact"] == 2
    lossy = [row for row in result["controls"] if row["control"].startswith("jpeg")]
    assert all(not row["pixel_identical"] for row in lossy)
    assert all(row["match"] is None or not row["match"]["skip"] for row in lossy)
    for row in result["controls"]:
        if row["control"] == "half":
            assert row["pixels"] == [66, 48]
        if row["control"] == "double":
            assert row["pixels"] == [262, 194]
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("target", ["manifest", "image", "symlink", "hardlink"])
def test_output_cannot_overwrite_an_input(corpus, monkeypatch, target):
    manifest, images, output = corpus
    source = next(images.rglob("*.png"))
    if target == "manifest":
        output = manifest
    elif target == "image":
        output = source
    elif target == "symlink":
        output.symlink_to(source)
    else:
        os.link(source, output)
    before = {manifest: manifest.read_bytes(), source: source.read_bytes()}
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, (manifest, images, output))
    assert exc.value.code == 2
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("damage", ["bytes", "shape", "missing", "conflicting_hash"])
def test_altered_or_missing_sources_leave_existing_evidence_intact(corpus, monkeypatch, damage):
    manifest, images, output = corpus
    source = next(images.rglob("*.png"))
    data = json.loads(manifest.read_text())
    if damage == "bytes":
        Image.new("RGB", (131, 97), "black").save(source)
    elif damage == "shape":
        data["arms"]["test"]["figures"][0]["pixels"] = [130, 97]
    elif damage == "missing":
        source.unlink()
    else:
        data["arms"]["other"] = {"figures": [dict(
            data["arms"]["test"]["figures"][0], sha256="different")]}
    manifest.write_text(json.dumps(data))
    output.write_text("prior measurement\n")
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, corpus)
    assert exc.value.code == 2
    assert output.read_text() == "prior measurement\n"


def test_symlink_cannot_import_an_image_outside_the_source_root(corpus, monkeypatch):
    manifest, images, output = corpus
    source = next(images.rglob("*.png"))
    outside = images.parent / "outside.png"
    source.rename(outside)
    source.symlink_to(outside)
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, corpus)
    assert exc.value.code == 2
    assert not output.exists()
