"""Structural guards for the desktop PyInstaller bundle."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "spacr.spec"


def _tree() -> ast.Module:
    return ast.parse(SPEC.read_text(encoding="utf-8"), filename=str(SPEC))


def test_repo_precedes_hidden_import_discovery() -> None:
    """The isolated collector must inspect this tree, not an old install."""
    source = SPEC.read_text(encoding="utf-8")

    pin = source.index("sys.path.insert(0, str(ROOT))")
    collect = source.index('collect_submodules(\n    "spacr"')

    assert pin < collect


def test_only_runtime_packages_are_recursively_collected() -> None:
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collect_submodules"
    ]

    assert [ast.literal_eval(call.args[0]) for call in calls] == [
        "spacr",
        "cellpose",
    ]
    assert all(any(keyword.arg == "filter" for keyword in call.keywords)
               for call in calls)
    assert all(any(keyword.arg == "on_error"
                   and ast.literal_eval(keyword.value) == "raise"
                   for keyword in call.keywords)
               for call in calls)

    source = SPEC.read_text(encoding="utf-8")
    assert 'not name.startswith(("cellpose.contrib", "cellpose.gui"))' in source


def test_dynamic_desktop_backends_are_explicit() -> None:
    source = SPEC.read_text(encoding="utf-8")

    for module in (
        "spacr.qt.prerun",
        "spacr.qt.maturity",
        "vispy.app.backends._pyside6",
        "vispy.gloo.gl.gl2",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
    ):
        if module.startswith("spacr."):
            # spaCR modules come from the repo-pinned runtime collection.
            assert (ROOT / module.replace(".", "/")).with_suffix(".py").is_file()
        else:
            assert f'"{module}"' in source

    analysis = next(
        node.value
        for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "a"
                for target in node.targets)
    )
    hooksconfig = next(
        keyword.value
        for keyword in analysis.keywords
        if keyword.arg == "hooksconfig"
    )
    assert ast.literal_eval(hooksconfig) == {
        "matplotlib": {"backends": ["QtAgg", "Agg"]},
    }


def test_non_core_packages_cannot_leak_from_the_build_environment() -> None:
    assignment = next(
        node
        for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "_NON_CORE_IMPORTS"
                for target in node.targets)
    )
    excluded = set(ast.literal_eval(assignment.value))

    assert {
        # Declared spaCR extras.
        "anndata", "btrack", "catboost", "cuml", "cupy", "jax",
        "lightgbm", "mahotas", "napari", "numcodecs", "numpyro", "omero",
        "piper", "pylibCZIrw", "pymc", "torchcam", "trackastra", "ultrack",
        "zarr",
        # The largest accidental imports observed in a polluted builder.
        "bokeh", "dask", "onnxruntime", "panel", "pyarrow", "spacy",
        "transformers", "xarray",
    } <= excluded

    analysis = next(
        node.value
        for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "a"
                for target in node.targets)
    )
    excludes = next(
        keyword.value
        for keyword in analysis.keywords
        if keyword.arg == "excludes"
    )
    assert any(isinstance(element, ast.Starred)
               and isinstance(element.value, ast.Name)
               and element.value.id == "_NON_CORE_IMPORTS"
               for element in excludes.elts)
    assert "tensorboard" not in {
        ast.literal_eval(element)
        for element in excludes.elts
        if isinstance(element, ast.Constant)
    }
