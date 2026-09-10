"""The headless listing must not reach off the standard library path.

Preserved from tests/test_installed_import_path.py, which was deleted while the behaviour it pins is
still live. Sixteen of that file's nineteen tests genuinely stopped
holding and were rightly dropped; this one still passes against the
current tree, so deleting it would have retired a real contract rather
than a stale one. Kept verbatim -- only the tests that no longer hold
were removed.
"""

from __future__ import annotations

import json

import os

import re

import subprocess

import sys

import tempfile

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCIENTIFIC_ROOTS = (
    "IPython",
    "PySide6",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
)

def _isolated_probe(body: str) -> dict:
    """Run ``body`` without site packages and return its final JSON line."""
    temporary_home = tempfile.gettempdir()
    environment = os.environ.copy()
    environment.update({
        "HOME": temporary_home,
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(ROOT),
        "USERPROFILE": temporary_home,
    })
    environment.pop("PYTHONHOME", None)
    process = subprocess.run(
        [sys.executable, "-S", "-c", body],
        cwd=temporary_home,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.splitlines()[-1])

def test_headless_list_stays_on_the_standard_library_path():
    """Resolving ``spacr-run --list`` must not import a pipeline backend."""
    result = _isolated_probe(
        "import contextlib, io, json, sys\n"
        "import spacr.cli as cli\n"
        "output = io.StringIO()\n"
        "with contextlib.redirect_stdout(output):\n"
        "    result = cli.main(['--list'])\n"
        "roots = %r\n"
        "print(json.dumps({\n"
        "    'result': result,\n"
        "    'listed_measure': 'measure' in output.getvalue(),\n"
        "    'scientific': [name for name in roots if name in sys.modules],\n"
        "}))\n" % (SCIENTIFIC_ROOTS,)
    )

    assert result == {
        "result": 0,
        "listed_measure": True,
        "scientific": [],
    }

def _source_version() -> str:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(
        r'^VERSION\s*=\s*["\']([^"\']+)["\']',
        setup_text,
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)
