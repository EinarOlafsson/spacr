"""Structural guards for the desktop PyInstaller bundle."""

import ast
import fnmatch
from pathlib import Path

import pytest

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


def test_import_time_distribution_metadata_reaches_bundle_analysis() -> None:
    """ImageIO's version lookup failed in both genuine native Measure runs."""
    tree = _tree()
    collected = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "a"
                        for target in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Analysis"):
            data_arg = next(k.value for k in node.value.keywords if k.arg == "datas")
            assert isinstance(data_arg, ast.Name) and data_arg.id == "datas"
            break
        if (isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name) and node.target.id == "datas"
                and isinstance(node.op, ast.Add)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "copy_metadata"):
            collected.add(ast.literal_eval(node.value.args[0]))
    else:
        raise AssertionError("No bundle Analysis consumes the collected data")
    assert {"spacr", "imageio"} <= collected


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


@pytest.mark.parametrize("library", [
    "_C.so", "_C.abi3.so", "_C.pyd", "_C.dylib",
    "_C_stable.so", "_C_stable.abi3.so", "_C_stable.pyd", "_C_stable.dylib",
])
def test_torchvision_operator_library_is_passed_to_binary_analysis(library):
    """Directly loaded ops must reach binary analysis on every platform."""
    tree = _tree()
    collector = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_torchvision_binaries")
    expected = [(f"/wheel/torchvision/{library}", "torchvision")]

    def collect(package, *, search_patterns):
        """Use a wheel-shaped file listing without importing Torch or PyInstaller."""
        assert package == "torchvision"
        assert any(fnmatch.fnmatch(library, pattern) for pattern in search_patterns)
        return expected

    namespace = {"Path": Path, "collect_dynamic_libs": collect}
    exec(compile(ast.Module(body=[collector], type_ignores=[]), str(SPEC), "exec"), namespace)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "binaries"
                              for t in n.targets))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(SPEC), "exec"), namespace)
    analysis = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "a"
                            for target in n.targets)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "Analysis")
    argument = next(k.value for k in analysis.keywords if k.arg == "binaries")
    actual = eval(compile(ast.Expression(argument), str(SPEC), "eval"), namespace)
    assert actual is expected


@pytest.mark.parametrize("libraries", [
    [], [("/wheel/torchvision/image.so", "torchvision")],
    [("/wheel/torchvision/_C_unrelated.so", "torchvision")],
])
def test_missing_torchvision_ops_abort_the_build(libraries):
    """A bundle without _C must fail during collection, before a native run."""
    collector = next(n for n in _tree().body if isinstance(n, ast.FunctionDef)
                     and n.name == "_torchvision_binaries")
    namespace = {"Path": Path, "collect_dynamic_libs": lambda *a, **k: libraries}
    exec(compile(ast.Module(body=[collector], type_ignores=[]), str(SPEC), "exec"), namespace)
    with pytest.raises(RuntimeError, match="_C operator library was not collected"):
        namespace["_torchvision_binaries"]()
