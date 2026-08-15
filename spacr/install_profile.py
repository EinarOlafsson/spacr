"""Read and write the hardware choice made by the desktop installer.

The online installers live outside the Python environment they create, so
their durable hand-off is a tiny JSON file beside the private ``venv``.  This
module owns the schema and probes the *installed* torch build before writing
it.  Keeping that probe here means ``spacr-doctor`` reports facts rather than
trying to infer an install-time choice from whatever hardware is visible
later.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROFILE_NAME = "install-profile.json"
PROFILE_SCHEMA = 1
_BACKEND_RE = re.compile(r"^[a-z0-9]{2,32}$")
VALID_DETECTED_ACCELERATORS = frozenset(
    {"nvidia", "apple-silicon", "none", "unknown"}
)


def default_profile_path() -> Path:
    """Return the installer profile path for this Python environment."""
    override = os.environ.get("SPACR_INSTALL_PROFILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(sys.prefix).resolve().parent / PROFILE_NAME


def _torch_facts() -> Dict[str, Any]:
    """Describe the installed torch build without assuming optional APIs."""
    import torch

    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda and cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        mps_available = bool(mps_backend and mps_backend.is_available())
    except Exception:
        mps_available = False
    if cuda_available:
        active = "cuda"
    elif mps_available:
        active = "mps"
    else:
        active = "cpu"
    return {
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "active_backend": active,
    }


def build_profile(requested_backend: str, detected_accelerator: str) -> Dict[str, Any]:
    """Build a validated profile using the torch installation now on disk."""
    requested = str(requested_backend).strip().lower()
    detected = str(detected_accelerator).strip().lower()
    if not _BACKEND_RE.fullmatch(requested):
        raise ValueError(f"unsupported requested backend: {requested_backend!r}")
    if detected not in VALID_DETECTED_ACCELERATORS:
        raise ValueError(f"unsupported detected accelerator: {detected_accelerator!r}")
    return {
        "schema": PROFILE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_backend": requested,
        "detected_accelerator": detected,
        "platform": platform.system().lower(),
        "machine": platform.machine().lower(),
        **_torch_facts(),
    }


def write_profile(
    path: Path,
    requested_backend: str,
    detected_accelerator: str,
    *,
    consent_collected: bool = False,
    share_diagnostics: bool = False,
    report_issues: bool = False,
    sign_in_now: bool = False,
) -> Dict[str, Any]:
    """Atomically write and return an installer profile."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_profile(requested_backend, detected_accelerator)
    payload["consent"] = {
        "collected": bool(consent_collected),
        "share_diagnostics": bool(share_diagnostics),
        "report_issues": bool(report_issues),
        "sign_in_now": bool(sign_in_now),
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return payload


def read_profile(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return a valid installer profile, or ``None`` when absent/invalid."""
    target = default_profile_path() if path is None else Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA:
        return None
    if not _BACKEND_RE.fullmatch(str(payload.get("requested_backend", ""))):
        return None
    if payload.get("active_backend") not in {"cpu", "cuda", "mps"}:
        return None
    return payload


def build_parser() -> argparse.ArgumentParser:
    """The command line the installers call this module with.

    Separate from :func:`main` so the installer scripts and the tests can
    inspect the accepted options without running an install -- an argument
    that silently stopped being accepted would otherwise only show up as a
    failed install on somebody's machine.

    The four consent flags take ``"0"`` / ``"1"`` rather than being
    store_true switches: the installer passes a value for every one of them
    on every run, so an unchecked box is recorded as a deliberate NO rather
    than as an absent answer.

    :returns: the parser, with ``--path``, ``--requested``, ``--detected``
        and the consent flags.
    """
    parser = argparse.ArgumentParser(description="Record the spaCR installer profile")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--requested", required=True)
    parser.add_argument(
        "--detected", choices=sorted(VALID_DETECTED_ACCELERATORS), required=True
    )
    for option in (
        "consent-collected",
        "share-diagnostics",
        "report-issues",
        "sign-in-now",
    ):
        parser.add_argument(f"--{option}", choices=("0", "1"), default="0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the installer profile and print it, for the installer to read.

    The entry point behind ``python -m spacr.install_profile``. The profile
    is echoed to stdout as sorted JSON so the installer can log exactly what
    was recorded, and so a support request can quote it.

    :param argv: command line to parse; ``None`` reads ``sys.argv``.
    :returns: a process exit code -- ``0``, since a profile that cannot be
        written raises rather than returning a code nobody checks.
    """
    args = build_parser().parse_args(argv)
    profile = write_profile(
        args.path,
        args.requested,
        args.detected,
        consent_collected=args.consent_collected == "1",
        share_diagnostics=args.share_diagnostics == "1",
        report_issues=args.report_issues == "1",
        sign_in_now=args.sign_in_now == "1",
    )
    print(json.dumps(profile, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module
    raise SystemExit(main())
