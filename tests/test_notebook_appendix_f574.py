"""Optional archive cells stay after the maintained measurement workflow."""
import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _layout_helpers():
    """Load only the generator's real layout helpers without pipeline imports."""
    tree = ast.parse((ROOT / "tools/build_notebook_settings.py").read_text())
    names = {"_cell_kind", "_insertion_point", "_place_generated_cells", "_place_run_cell"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
    namespace = {"RUN_CELL_KIND": "function-run"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<notebook-layout-helpers>", "exec"), namespace)
    return namespace


def test_regeneration_layout_keeps_archive_after_measurement():
    helpers = _layout_helpers()
    notebook = json.loads((ROOT / "Notebooks/02_measure_and_crop.ipynb").read_text())
    cells = copy.deepcopy(notebook["cells"])
    kind = helpers["_cell_kind"]
    help_cell = next(cell for cell in cells if kind(cell) == "settings-help")
    settings = [cell for cell in cells if kind(cell) in ("settings", "settings-organelle")]
    run = next(cell for cell in cells if kind(cell) == "function-run")
    helpers["_place_generated_cells"](cells, help_cell, settings)
    helpers["_place_run_cell"](cells, settings[-1], run)
    assert cells == notebook["cells"]
    assert cells[-1]["metadata"]["spacr"]["feature"] == 574
    assert cells[-2]["metadata"]["spacr"]["feature"] == 574


def test_unmarked_notebooks_retain_heading_and_last_code_fallback():
    insertion = _layout_helpers()["_insertion_point"]
    cells = [{"cell_type": "code", "source": ["first()"]},
             {"cell_type": "markdown", "source": ["notes"]},
             {"cell_type": "code", "source": ["last()"]}]
    assert insertion(cells) == 2
    cells[1]["source"] = ["## 4. Run it"]
    assert insertion(cells) == 1
    assert insertion([]) == 0


def test_archive_notebook_cell_ids_and_python_are_valid():
    notebook = json.loads((ROOT / "Notebooks/02_measure_and_crop.ipynb").read_text())
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(identifiers) == len(set(identifiers))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
