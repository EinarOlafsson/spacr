"""Regression tests for third-party warnings emitted during Qt startup."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]


def _core_dependency_names():
    """Return normalized distribution names declared by ``setup.py``."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dependencies"
            for target in node.targets
        ):
            dependencies = ast.literal_eval(node.value)
            return {
                Requirement(value).name.lower().replace("_", "-")
                for value in dependencies
            }
    raise AssertionError("setup.py has no module-level dependencies list")


def test_spacr_declares_the_maintained_nvml_distribution():
    names = _core_dependency_names()
    assert "nvidia-ml-py" in names
    assert "pynvml" not in names


def test_startup_suppresses_only_the_known_third_party_future_notices():
    """A fresh process mirrors the warning order of the installed CLI."""
    code = r'''
import warnings
import spacr

# The deprecated pynvml compatibility distribution installs a .pth hook, so
# existing environments can still emit this even after spaCR's dependency is
# corrected. The filter keeps an upgrade quiet until that wrapper is removed.
warnings.warn(
    "The pynvml package is deprecated. Please install nvidia-ml-py instead.",
    FutureWarning,
)
warnings.warn(
    "You are using a Python version (3.10.19) which Google will stop "
    "supporting in new releases of google.api_core.",
    FutureWarning,
)

# Exercise the real heavy-import paths too. They are optional in packaging
# metadata tests, hence the guarded imports.
try:
    import torch
except ImportError:
    pass
try:
    import spacr.ml
except ImportError:
    pass
print("startup-imports-complete")
'''
    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "startup-imports-complete" in proc.stdout
    assert "pynvml package is deprecated" not in proc.stderr
    assert "logit link alias is deprecated" not in proc.stderr
    assert "Google will stop supporting" not in proc.stderr


def test_unrelated_future_warnings_are_not_hidden():
    code = r'''
import warnings
import spacr
warnings.warn("spaCR test sentinel", FutureWarning)
'''
    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert "spaCR test sentinel" in proc.stderr


# ---------------------------------------------------------------------------
# The cellpose sparse-tensor notice.
#
# Torch reports that invariant checking is off every time cellpose builds the
# sparse COO tensor it makes masks out of, so `spacr` printed it on every
# start. The filter is exercised the only way that proves anything: by
# raising the warning from a compiled unit whose filename really is
# cellpose's `dynamics.py`, because the filter is scoped by the *path* of the
# raising file and a warning raised from the test's own file would not be
# scoped the same way whatever the message said.
# ---------------------------------------------------------------------------

_RAISE_FROM_CELLPOSE = r'''
import sys
import warnings
{setup}
import cellpose.dynamics as _dyn

_src = (
    "import warnings\n"
    "def emit(text):\n"
    "    warnings.warn(text, UserWarning)\n"
)
_ns = {{"__name__": "cellpose.dynamics", "__file__": _dyn.__file__}}
exec(compile(_src, _dyn.__file__, "exec"), _ns)
{body}
print("emitted")
'''


def _run_emitting(setup: str, body: str):
    code = _RAISE_FROM_CELLPOSE.format(setup=setup, body=body)
    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "emitted" in proc.stdout
    return proc


SPARSE_NOTICE = "Sparse invariant checks are implicitly disabled"


def test_the_cellpose_sparse_notice_is_silent_after_importing_spacr():
    """The message as reported, and the same message with a prefix.

    The prefixed case is the one that matters: the filter this replaced
    anchored at the start of the sentence, because
    ``warnings.filterwarnings`` matches ``message`` with ``re.match`` and
    not ``re.search``. Any build of torch that says
    ``torch.sparse_coo_tensor: <sentence>`` walked straight past it, which
    is a filter that looks right in a diff and does nothing on the machine
    that has the problem.
    """
    proc = _run_emitting(
        setup="import spacr",
        body=('_ns["emit"]("%s. To enable them, use ...")\n'
              '_ns["emit"]("torch.sparse_coo_tensor: %s")'
              % (SPARSE_NOTICE, SPARSE_NOTICE)),
    )
    assert SPARSE_NOTICE not in proc.stderr


def test_the_qt_launcher_restores_the_filter_and_does_not_stack_it():
    """``spacr.qt._quiet_library_warnings`` is what ``run()`` calls first.

    Asserted against a wiped filter list, because that is the only state in
    which it does anything: on a clean launch ``import spacr`` has already
    installed the rule and this is a no-op. Wiping first is also how the
    idempotence claim is made honestly — the count below would be satisfied
    by the import's own filter otherwise, whatever the function did.

    Importing ``spacr.qt`` must not need PySide6 for any of it: the quieters
    live in the package ``__init__`` precisely so they run before anything
    heavy is imported.
    """
    proc = _run_emitting(
        setup=("import spacr.qt\n"
               "warnings.resetwarnings()\n"
               "warnings.simplefilter('default')\n"
               "spacr.qt._quiet_library_warnings()\n"
               "spacr.qt._quiet_library_warnings()  # idempotent\n"
               "assert sum(1 for f in warnings.filters\n"
               "           if getattr(f[3], 'pattern', '') "
               "and 'cellpose' in f[3].pattern) == 1"),
        body='_ns["emit"]("%s")' % SPARSE_NOTICE,
    )
    assert SPARSE_NOTICE not in proc.stderr


def test_the_sparse_filter_does_not_swallow_the_rest_of_cellpose():
    """Precision, both ways round.

    A different warning from cellpose still reaches the user, and the
    silenced sentence still reaches the user when something that is not
    cellpose says it — otherwise "ignore this text" would hide a real
    spaCR bug that happened to word itself the same way.
    """
    proc = _run_emitting(
        setup="import spacr",
        body='_ns["emit"]("cellpose has something real to say")',
    )
    assert "cellpose has something real to say" in proc.stderr

    code = (
        "import warnings\n"
        "import spacr\n"
        'warnings.warn("%s", UserWarning)\n'
        'print("emitted")\n' % SPARSE_NOTICE
    )
    other = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        cwd=REPO_ROOT, text=True, capture_output=True, timeout=180,
        check=False,
    )
    assert other.returncode == 0, other.stderr
    assert SPARSE_NOTICE in other.stderr
