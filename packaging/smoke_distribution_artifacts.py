#!/usr/bin/env python3
"""Install and smoke the exact wheel and sdist produced by CI.

The compatibility matrix runs this file from its checkout, but every package
probe runs in isolated mode from a temporary directory.  That distinction is
important: a green import from the checkout says nothing about the archive a
user downloads.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

# This is a functional hang guard, not a startup microbenchmark.  Timings are
# retained in the report so regressions can be compared across like runners,
# while a deliberately generous ceiling avoids making slower ARM and Windows
# hosts fail over a few seconds of ordinary variance.
SMOKE_TIMEOUT_SECONDS = 180
CLI_TIMEOUT_SECONDS = 60
CORE_MODULES = (
    "spacr.io",
    "spacr.measure",
    "spacr.utils",
    "spacr.timelapse",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_origin(distribution: importlib.metadata.Distribution) -> str:
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("installed distribution has no PEP 610 origin")
    url = json.loads(raw).get("url", "")
    return Path(unquote(urlparse(url).path)).name


def _probe(expected_version: str, expected_artifact: str,
           source_root: Path) -> int:
    """Exercise the installed package and print one machine-readable line."""
    started = time.perf_counter()

    import_started = time.perf_counter()
    import spacr

    distribution = importlib.metadata.distribution("spacr")
    installed = Path(spacr.__file__).resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    platlib = Path(sysconfig.get_path("platlib")).resolve()
    assert _inside(installed, purelib) or _inside(installed, platlib), (
        f"spacr did not resolve from site-packages: {installed}"
    )
    assert not _inside(installed, source_root.resolve()), (
        f"checkout masked the installed artifact: {installed}"
    )
    assert all(
        not _inside(Path(entry).resolve(), source_root.resolve())
        for entry in sys.path if entry
    ), f"checkout is present on isolated sys.path: {sys.path!r}"
    assert spacr.__version__ == expected_version
    assert distribution.version == expected_version
    origin = _distribution_origin(distribution)
    assert origin == expected_artifact, (
        f"installed {origin!r}, expected the downloaded {expected_artifact!r}"
    )
    for module_name in CORE_MODULES:
        importlib.import_module(module_name)
    import_seconds = time.perf_counter() - import_started

    qt_started = time.perf_counter()
    from PySide6.QtWidgets import QApplication
    from spacr.qt.screens.app_screen import AppScreen

    app = QApplication.instance() or QApplication([])
    screen = AppScreen("measure")
    assert screen.app_key == "measure"
    screen.close()
    screen.deleteLater()
    app.processEvents()
    qt_seconds = time.perf_counter() - qt_started

    result = {
        "artifact": expected_artifact,
        "core_modules": list(CORE_MODULES),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "import_seconds": round(import_seconds, 3),
        "installed_from": str(installed),
        "origin_verified": True,
        "qt_screen": "measure",
        "qt_seconds": round(qt_seconds, 3),
        "version": spacr.__version__,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _artifact_pair(dist_dir: Path) -> tuple[Path, Path]:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            "expected exactly one wheel and one sdist in "
            f"{dist_dir}, found wheels={wheels!r}, sdists={sdists!r}"
        )
    return wheels[0].resolve(), sdists[0].resolve()


def _wheel_version(wheel: Path) -> str:
    """Read the expected version without importing packaging or the project."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(
                f"wheel has {len(metadata_names)} METADATA files: "
                f"{metadata_names!r}"
            )
        metadata = email.parser.BytesParser().parsebytes(
            archive.read(metadata_names[0])
        )
    version = metadata.get("Version", "").strip()
    if not version:
        raise RuntimeError("wheel METADATA has no Version")
    return version


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def _pip(*arguments: str) -> None:
    _run([
        sys.executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        *arguments,
    ])


def _parse_json_line(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.lstrip().startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"installed smoke emitted no JSON result:\n{output}")


def _clean_environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "HOME": str(home),
        "MPLBACKEND": "Agg",
        "QT_QPA_PLATFORM": "offscreen",
        "SPACR_DISABLE_PLUGINS": "1",
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
    })
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _entrypoint_smoke(environment: dict[str, str], cwd: Path,
                      expected_version: str) -> dict:
    # Do not trust inherited PATH: a developer machine can have another
    # spacr-run in a base Conda environment even while this interpreter has
    # the artifact under test.  sysconfig names this interpreter's Scripts
    # directory on Windows and bin directory on POSIX; shutil.which adds the
    # platform-specific executable suffix.
    scripts = Path(sysconfig.get_path("scripts")).resolve()
    entrypoint = shutil.which("spacr-run", path=str(scripts))
    if entrypoint is None:
        raise RuntimeError("the installed distribution created no spacr-run")
    version = _run(
        [entrypoint, "--version"],
        cwd=cwd,
        env=environment,
        timeout=CLI_TIMEOUT_SECONDS,
        capture_output=True,
    ).stdout.strip()
    if version != expected_version:
        raise RuntimeError(
            f"spacr-run --version returned {version!r}, "
            f"expected {expected_version!r}"
        )
    listing = _run(
        [entrypoint, "--list"],
        cwd=cwd,
        env=environment,
        timeout=CLI_TIMEOUT_SECONDS,
        capture_output=True,
    ).stdout
    if not any(line.startswith("  measure ") for line in listing.splitlines()):
        raise RuntimeError("spacr-run --list omitted the measure command")
    return {"entrypoint": str(entrypoint), "list_has_measure": True}


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_summary(report: dict) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path or report.get("status") != "passed":
        return
    lines = [
        "## Installed distribution smoke",
        "",
        "| format | sha256 | install (s) | smoke (s) | Qt (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for item in report["artifacts"]:
        probe = item["probe"]
        lines.append(
            f"| {item['format']} | `{item['sha256'][:16]}...` | "
            f"{item['install_seconds']:.3f} | "
            f"{probe['elapsed_seconds']:.3f} | {probe['qt_seconds']:.3f} |"
        )
    with Path(summary_path).open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def _exercise_artifacts(dist_dir: Path, extras: str,
                        report_path: Path) -> int:
    wheel, sdist = _artifact_pair(dist_dir)
    expected_version = _wheel_version(wheel)
    source_root = Path(__file__).resolve().parents[1]
    report = {
        "artifacts": [],
        "functional_timeout_seconds": SMOKE_TIMEOUT_SECONDS,
        "python": sys.version,
        "status": "running",
    }
    _write_report(report_path, report)

    try:
        dependency_started = time.perf_counter()
        requirement = str(wheel) + (f"[{extras}]" if extras else "")
        _pip("install", requirement)
        report["dependency_prepare_seconds"] = round(
            time.perf_counter() - dependency_started, 3
        )
        # Dependencies remain.  Removing the project ensures each archive
        # below becomes the final installed spaCR, never an editable checkout.
        _pip("uninstall", "-y", "spacr")

        for format_name, artifact in (("wheel", wheel), ("sdist", sdist)):
            item = {
                "artifact": artifact.name,
                "format": format_name,
                "sha256": _sha256(artifact),
                "status": "running",
            }
            report["artifacts"].append(item)
            _write_report(report_path, report)

            _pip("uninstall", "-y", "spacr")
            install_started = time.perf_counter()
            _pip(
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--force-reinstall",
                str(artifact),
            )
            item["install_seconds"] = round(
                time.perf_counter() - install_started, 3
            )

            with tempfile.TemporaryDirectory(
                prefix=f"spacr-{format_name}-smoke-"
            ) as temporary:
                home = Path(temporary).resolve()
                environment = _clean_environment(home)
                process = _run(
                    [
                        sys.executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        "--probe",
                        "--expected-version",
                        expected_version,
                        "--expected-artifact",
                        artifact.name,
                        "--source-root",
                        str(source_root),
                    ],
                    cwd=home,
                    env=environment,
                    timeout=SMOKE_TIMEOUT_SECONDS,
                    capture_output=True,
                )
                item["probe"] = _parse_json_line(process.stdout)
                item["cli"] = _entrypoint_smoke(
                    environment, home, expected_version
                )
            item["status"] = "passed"
            _write_report(report_path, report)

        report["status"] = "passed"
        _write_report(report_path, report)
        _append_summary(report)
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["status"] = "failed"
        _write_report(report_path, report)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--extras", default="")
    parser.add_argument("--expected-version", help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-artifact", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.probe:
        if (arguments.expected_version is None
                or arguments.expected_artifact is None
                or arguments.source_root is None):
            raise SystemExit(
                "--probe needs --expected-version, --expected-artifact and "
                "--source-root"
            )
        return _probe(
            arguments.expected_version,
            arguments.expected_artifact,
            arguments.source_root,
        )
    if arguments.dist_dir is None or arguments.report is None:
        raise SystemExit("artifact mode needs --dist-dir and --report")
    return _exercise_artifacts(
        arguments.dist_dir,
        arguments.extras,
        arguments.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
