"""Structural guards for the dependency-light installed package and CLI path."""

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


def test_package_facade_avoids_metadata_and_scientific_imports():
    """The wheel's known version must not require a distribution scan."""
    result = _isolated_probe(
        "import json, sys\n"
        "import spacr\n"
        "roots = %r\n"
        "print(json.dumps({\n"
        "    'version': spacr.__version__,\n"
        "    'metadata': 'importlib.metadata' in sys.modules,\n"
        "    'version_module': 'spacr.version' in sys.modules,\n"
        "    'typing': 'typing' in sys.modules,\n"
        "    'scientific': [name for name in roots if name in sys.modules],\n"
        "}))\n" % (SCIENTIFIC_ROOTS,)
    )

    assert result == {
        "version": _source_version(),
        "metadata": False,
        "version_module": False,
        "typing": False,
        "scientific": [],
    }


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


def test_cli_plugin_opt_out_avoids_distribution_metadata():
    """Disabling extensions must skip both plugin imports and their scan."""
    result = _isolated_probe(
        "import json, os, sys\n"
        "os.environ['SPACR_DISABLE_PLUGINS'] = '1'\n"
        "import spacr.cli\n"
        "roots = %r\n"
        "print(json.dumps({\n"
        "    'metadata': 'importlib.metadata' in sys.modules,\n"
        "    'scientific': [name for name in roots if name in sys.modules],\n"
        "}))\n" % (SCIENTIFIC_ROOTS,)
    )

    assert result == {"metadata": False, "scientific": []}


def test_cli_version_uses_the_wheel_literal_without_metadata():
    """The extension-free version command must not rescan distributions."""
    result = _isolated_probe(
        "import contextlib, io, json, os, sys\n"
        "os.environ['SPACR_DISABLE_PLUGINS'] = '1'\n"
        "import spacr.cli as cli\n"
        "output = io.StringIO()\n"
        "with contextlib.redirect_stdout(output):\n"
        "    status = cli.main(['--version'])\n"
        "print(json.dumps({\n"
        "    'metadata': 'importlib.metadata' in sys.modules,\n"
        "    'status': status,\n"
        "    'version': output.getvalue().strip(),\n"
        "}))\n"
    )

    assert result == {
        "metadata": False,
        "status": 0,
        "version": _source_version(),
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


def test_lightweight_version_matches_the_distribution_source():
    from spacr._version import __version__

    assert __version__ == _source_version()
