"""Small, bounded probes for resource-dependent integration tests.

Resource tests should run because their dependency is available, not because
an operator remembered an environment-variable opt-in.  These helpers keep
that detection deterministic and, importantly for autofs NAS mounts, bounded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import find_spec
from typing import Callable, Iterable, Tuple
from urllib.request import Request, urlopen

PathRequirement = Tuple[str, str]


def cuda_available() -> bool:
    """Return whether PyTorch can actually use at least one CUDA device."""
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def package_available(name: str,
                      finder: Callable[[str], object] = find_spec) -> bool:
    """Return whether an optional integration-test dependency is importable."""
    try:
        return finder(name) is not None
    except Exception:
        return False


def endpoint_available(
    url: str = "https://huggingface.co",
    timeout: float = 5.0,
    opener: Callable = urlopen,
) -> bool:
    """Probe a network endpoint without downloading its response body."""
    request = Request(url, method="HEAD")
    try:
        response = opener(request, timeout=timeout)
        try:
            status = getattr(response, "status", 200)
            return int(status) < 500
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except Exception:
        return False


def paths_available(
    requirements: Iterable[PathRequirement],
    timeout: float = 5.0,
    runner: Callable = subprocess.run,
) -> bool:
    """Check required directories/files in a time-bounded child process.

    Accessing a disconnected autofs mount can block the calling process for
    minutes. A child probe lets collection continue and cleanly skip the NAS
    suite when the mount cannot answer within ``timeout``.

    ``requirements`` contains ``(path, kind)`` pairs where kind is ``"dir"``
    or ``"file"``.
    """
    items = tuple((str(path), str(kind)) for path, kind in requirements)
    if not items or any(kind not in {"dir", "file"} for _, kind in items):
        return False
    script = (
        "import json, os, sys\n"
        "items = json.loads(sys.argv[1])\n"
        "ok = all((os.path.isdir(p) if k == 'dir' else os.path.isfile(p)) "
        "for p, k in items)\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    try:
        completed = runner(
            [sys.executable, "-c", script, json.dumps(items)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0
