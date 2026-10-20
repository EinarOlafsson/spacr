"""The Colab mask generator notebooks are valid, current and call only real helpers."""
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tools.web_mask_generator import notebook, run
from tools.web_mask_generator.examples import ExampleStore
from tools.web_mask_generator.registry import model_families

FOLDER = Path(notebook.__file__).parent / "colab"
NOTEBOOKS = sorted(FOLDER.rglob("*.ipynb"))


def _code(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def _python(source):
    return "\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%")) else line
                     for line in source.splitlines())


def test_one_notebook_per_class_plus_the_generic_one():
    assert set(notebook.FAMILIES) == {family["family"] for family in model_families()}
    assert {path.relative_to(FOLDER).as_posix() for path in NOTEBOOKS} == set(notebook.rendered())
    assert notebook.main([]) == 0


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_is_valid_json_with_parsable_code(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4 and document["metadata"]["accelerator"] == "GPU"
    assert {cell["cell_type"] for cell in document["cells"]} == {"markdown", "code"}
    for source in _code(path):
        ast.parse(_python(source))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_code_cells_use_only_existing_spacr_functions(path):
    names = set()
    for source in _code(path):
        tree = ast.parse(_python(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module.startswith(("tools", "spacr")):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name) or importlib.util.find_spec(f"{node.module}.{alias.name}")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "spacr_web":
                names.add(node.attr)
    assert names >= {"fetch_model", "generate", "random_example", "offer_download"}
    assert all(callable(getattr(notebook, name)) for name in names)


@pytest.mark.parametrize("family", notebook.FAMILIES)
def test_class_notebook_preselects_its_class(family):
    source = "\n".join(_code(FOLDER / "by_class" / f"{family}.ipynb"))
    assert f'MODEL_CLASS = "{family}"  #@param' in source
    assert notebook.colab_url(family) == (
        "https://colab.research.google.com/github/EinarOlafsson/spacr/blob/nightly/"
        f"tools/web_mask_generator/colab/by_class/{family}.ipynb")


def test_examples_are_the_reviewed_test_images_for_every_class():
    records = notebook.example_records()
    ExampleStore(records)
    assert {record["family"] for record in records} == set(notebook.FAMILIES)


def _argv(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "main", lambda argv: calls.append(argv) or 0)
    return calls


def test_generate_passes_the_settings_to_the_operator_command(tmp_path, monkeypatch):
    image = tmp_path / "rgb.tif"
    tifffile.imwrite(image, np.zeros((3, 8, 8), np.uint8), metadata={"axes": "CYX"})
    model = {"key": "k_v1", "version": "1", "sha256": "a" * 64, "path": "/m"}
    calls = _argv(monkeypatch)
    out = notebook.generate(image, "toxoplasma_plaque", model, output_dir=tmp_path, diameter="auto", cellprob_threshold="")
    argv = calls[-1]
    assert out == tmp_path / "rgb-spacr-masks.zip" and "--native-scale" in argv
    assert argv.count("--channel") == 3 and "--cellprob-threshold" not in argv and "--magnification" not in argv
    notebook.generate(image, "toxoplasma_pv", model, output_dir=tmp_path, diameter="12", cellprob_threshold="-1", channels="2")
    argv = calls[-1]
    assert argv[argv.index("--diameter") + 1] == "12.0" and argv[argv.index("--channel") + 1] == "2"
    assert argv[argv.index("--cellprob-threshold") + 1] == "-1.0"
    out = notebook.generate(image, "toxoplasma_well_detector", model, output_dir=tmp_path)
    assert out.name == "rgb-spacr-wells.zip" and "--confidence" in calls[-1] and "--diameter" not in calls[-1]
    with pytest.raises(ValueError):
        notebook.generate(image, "toxoplasma_plaque", model, output_dir=tmp_path, diameter="")
