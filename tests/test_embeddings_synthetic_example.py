"""The bundled functional example loads as ordered crops with aligned labels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tarfile

import numpy as np

from spacr.crop_loader import CROP_SOURCE_FOLDER, CropQuery, load_crops, plan_crops
from spacr.example_archives import _prepare_embeddings_example
from spacr.tabular import read_table


ROOT = Path(__file__).resolve().parents[1]


def test_the_bundled_example_loads_all_crops_in_label_order(tmp_path):
    paths = _prepare_embeddings_example(tmp_path / "example")
    labels = read_table(paths["labels"], report=None)
    plan = plan_crops(CropQuery(source=CROP_SOURCE_FOLDER,
                               path=str(paths["crops"])))
    record = {}
    pixels = load_crops(plan, record=record)

    assert [Path(row).name for row in plan.rows] == labels["crop_filename"].tolist()
    assert pixels.shape == (24, 64, 64, 3)
    assert np.isfinite(pixels).all()
    assert np.unique(pixels.reshape(24, -1), axis=0).shape[0] == 24
    assert record["loaded"] == 24 and record["dropped"] == 0
    assert labels["label"].value_counts().to_dict() == {
        "circle": 8, "square": 8, "triangle": 8}
    help_text = (paths["folder"] / "README.txt").read_text(encoding="utf-8")
    assert "SYNTHETIC" in help_text and "not a scientific benchmark" in help_text


def test_the_builder_reproduces_the_bundled_payload(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_embeddings_example_dataset",
        ROOT / "tools" / "build_embeddings_example_dataset.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    first = tool.build(tmp_path / "first.tar.gz")
    second = tool.build(tmp_path / "second.tar.gz")
    assert first.read_bytes() == second.read_bytes()

    bundled = ROOT / "spacr/resources/data/embeddings_synthetic_example.tar.gz"
    with tarfile.open(first) as rebuilt, tarfile.open(bundled) as shipped:
        assert rebuilt.getnames() == shipped.getnames()
        for member in rebuilt.getmembers():
            assert member.isfile()
            assert rebuilt.extractfile(member).read() == shipped.extractfile(member.name).read()
