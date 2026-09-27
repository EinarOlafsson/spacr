"""
Guard tests: spacr must not (re-)introduce a TensorFlow dependency.

Stardist was the only remaining TF-backed component; if any future commit
re-adds a `stardist` code path or an `import tensorflow` / `import keras`,
one of these tests should fail loudly.

2026-09-26 (item 43, after item 551). StarDist came back, at the
maintainer's request, as an OUT-OF-PROCESS backend: it is installed into an
environment of its own under ``~/.spacr/backends/stardist`` and runs there as
``<env python> -I _segmentation_backends.py --serve stardist``. That file is
both spaCR's client and the worker, so its StarDist adapter and the
worker's device probe import stardist, csbdeep and tensorflow -- inside
function bodies that only the worker executes. The rule these tests keep is
the one the original rule was for: no TensorFlow in spaCR's own process or
dependencies. So an import of a TF-backed root is allowed only at the
worker-only sites named in :data:`WORKER_ONLY_TF_IMPORTS`, a stardist
reference only where :data:`STARDIST_MAY_BE_NAMED` says why, and
``test_the_backend_module_loads_no_tensorflow_in_spacrs_process`` asserts
that spaCR's process, importing and using that module, still loads none.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent / "spacr"

#: The roots whose import brings TensorFlow with it.
TF_ROOTS = ("tensorflow", "keras", "tf_keras", "stardist", "csbdeep")

#: ``(file, qualified function)`` -> why that import only runs in a worker.
#: Each is reached only from ``_worker_main`` -> ``_serve`` inside the
#: backend's own environment: the adapter is built by ``_worker_adapter``
#: and the device probe is ``_worker_device``'s fallback when the
#: environment has no PyTorch, which spaCR's always has.
WORKER_ONLY_TF_IMPORTS = {
    ("_segmentation_backends.py", "_StarDistAdapter.__init__"):
        "loads the StarDist model inside the stardist backend worker",
    ("_segmentation_backends.py", "_StarDistAdapter._segment"):
        "csbdeep's percentile normalisation, inside the worker",
    ("_segmentation_backends.py", "_tensorflow_device"):
        "the worker's GPU probe in an environment without PyTorch",
}

#: Files that may NAME StarDist, and why (item 551, 2026-09-26).
STARDIST_MAY_BE_NAMED = {
    "_segmentation_backends.py":
        "the StarDist backend's registry entry and its worker adapter",
    "model_zoo.py": "the zoo lists StarDist's models as a prefixed backend",
    "object.py": "the mask path dispatches stardist:<model> to the backend",
    "settings.py": "ALPHA_FEATURES[551] names the StarDist zoo rows",
}


def _all_py_files():
    return sorted(PKG_ROOT.glob("*.py"))


def tf_rooted_imports(path):
    """Every import of a TF-backed root in ``path``, with where it sits.

    :param path: a Python source file.
    :returns: ``[(lineno, module, qualname)]``; ``qualname`` is the
        enclosing ``Class.function`` path, or ``''`` at module scope.
    """
    tree = ast.parse(Path(path).read_text())
    found = []

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                visit(child, scope + (child.name,))
                continue
            names = []
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and not child.level:
                names = [child.module or ""]
            for name in names:
                if name.split(".")[0] in TF_ROOTS:
                    found.append((child.lineno, name, ".".join(scope)))
            visit(child, scope)

    visit(tree, ())
    return found


def unexplained_tf_imports(path):
    """The TF-rooted imports in ``path`` not named as worker-only."""
    return [(line, name, scope) for line, name, scope in tf_rooted_imports(path)
            if (Path(path).name, scope) not in WORKER_ONLY_TF_IMPORTS]


def _alpha_features_lines():
    """The line range of the ``ALPHA_FEATURES`` literal in settings.py."""
    tree = ast.parse((PKG_ROOT / "settings.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "ALPHA_FEATURES"
                for t in node.targets):
            return range(node.lineno, node.end_lineno + 1)
    return range(0)


@pytest.mark.parametrize("path", _all_py_files(), ids=lambda p: p.name)
def test_no_stardist_references(path):
    """No source file may reference stardist unless it is named as allowed.

    2026-09-26: the four files in :data:`STARDIST_MAY_BE_NAMED` name the
    out-of-process backend item 551 added. In settings.py only the
    ``ALPHA_FEATURES`` registry may, so a StarDist SETTING cannot slip in
    under the file's allowance.
    """
    src = path.read_text()
    if "stardist" not in src.lower():
        return
    allowed = range(0)
    if path.name == "settings.py":
        allowed = _alpha_features_lines()
    elif path.name in STARDIST_MAY_BE_NAMED:
        return
    for i, line in enumerate(src.splitlines(), 1):
        if "stardist" in line.lower() and i not in allowed:
            pytest.fail(f"{path.name}:{i}: {line.strip()!r} still references stardist")


@pytest.mark.parametrize("path", _all_py_files(), ids=lambda p: p.name)
def test_no_tensorflow_or_keras_imports(path):
    """Fail if any module imports a TF-backed root outside a worker-only site."""
    try:
        offenders = unexplained_tf_imports(path)
    except SyntaxError:
        pytest.skip(f"{path.name} did not parse")
    assert not offenders, "\n".join(
        f"{path.name}:{line} imports {name} "
        f"{'at module scope' if not scope else 'in ' + scope}"
        for line, name, scope in offenders)


def test_every_worker_only_site_still_exists():
    """A stale allowance is a hole the next import could walk through."""
    present = {(path.name, scope) for path in _all_py_files()
               for _line, _name, scope in tf_rooted_imports(path)}
    stale = sorted(set(WORKER_ONLY_TF_IMPORTS) - present)
    assert not stale, f"no longer import a TF-backed root: {stale}"


def test_no_worker_only_import_is_at_module_scope():
    """Importing the backend module in spaCR's process must import none."""
    assert all(scope for _file, scope in WORKER_ONLY_TF_IMPORTS)


def test_the_backend_module_loads_no_tensorflow_in_spacrs_process(tmp_path):
    """spaCR's own process never imports TensorFlow, StarDist or csbdeep.

    Imports the modules that name the backend and runs the client-side
    calls that touch it -- the registry, the backend state, the device a
    worker would pick -- in a fresh interpreter, then reads sys.modules.
    """
    import os
    import subprocess
    import sys

    body = f"""
import sys
import spacr._segmentation_backends as sb
import spacr.model_zoo, spacr.object, spacr.settings
sb._spec("stardist")
sb._backend_state("stardist", root={str(tmp_path)!r})
sb._worker_device()
loaded = sorted(r for r in {TF_ROOTS!r} if r in sys.modules)
print("LOADED " + ",".join(loaded) if loaded else "CLEAN")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run([sys.executable, "-c", body], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "CLEAN", proc.stdout


def test_object_module_has_no_segment_stardist():
    import spacr.object as m
    assert not hasattr(m, "_segment_stardist")
    assert not hasattr(m, "_load_stardist_model")


def test_settings_module_has_no_stardist_keys():
    import spacr.settings as s
    # expected_types dict should not carry stardist knobs anymore.
    et = getattr(s, "expected_types", {})
    stardist_keys = [k for k in et if "stardist" in k.lower()]
    assert not stardist_keys, f"stardist keys still in expected_types: {stardist_keys}"
    # Same check on descriptions dict.
    d = getattr(s, "descriptions", {})
    star_desc = [k for k in d if "stardist" in k.lower()]
    assert not star_desc, f"stardist keys still in descriptions: {star_desc}"
