"""Install and launch the separate Starplast desktop application.

Starplast uses PyQt6 and never imports into spaCR's PySide6 process. Its
environment lives under ``~/.spacr/apps`` (or ``SPACR_APPS_DIR``). Installation
reuses the segmentation installers' interpreter discovery, isolated command
environment and cancellable subprocess runner, without registering a detector.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from . import _segmentation_backends as environments


REPOSITORY = "git+https://github.com/EinarOlafsson/starplast.git@main"
"""Optional explicit source for development installations."""
PYPI_PACKAGE = "starplast"
"""Install the latest compatible stable release from the official PyPI index."""
_SPEC = SimpleNamespace(name="starplast", label="Starplast", python=((3, 10), (3, 13)), size_gb=12)
_OWNER = ".spacr-starplast.json"
_SELFTEST = (
    "import json, importlib.metadata; from PyQt6 import QtWidgets; "
    "from starplast import paths; from starplast.app import main; "
    "ok, message = paths.check(); assert ok, message; "
    "print(json.dumps({'ok': True, 'version': importlib.metadata.version('starplast')}))"
)


def apps_root(root=None) -> Path:
    """Return the external-application folder without creating it.

    :param root: explicit folder, otherwise ``SPACR_APPS_DIR`` or ``~/.spacr/apps``.
    :returns: absolute application root.
    """
    return Path(root or os.environ.get("SPACR_APPS_DIR") or "~/.spacr/apps").expanduser().absolute()


def default_source() -> str:
    """Use PyPI unless a developer explicitly overrides the installation source.

    :returns: ``SPACR_STARPLAST_SOURCE`` or the unpinned PyPI package name.
    """
    override = os.environ.get("SPACR_STARPLAST_SOURCE")
    if override:
        return override
    return PYPI_PACKAGE


def _record(env: Path) -> dict:
    """Read only a valid spaCR-owned Starplast environment record."""
    try:
        value = json.loads((env / _OWNER).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and value.get("app") == "starplast" else {}
    except (OSError, ValueError):
        return {}


def is_installed(root=None) -> bool:
    """Whether an owned environment has passed the import and bundled-data checks.

    :param root: external-application folder.
    :returns: true only with a completed record and environment interpreter.
    """
    env = apps_root(root) / "starplast"
    return (not env.is_symlink() and _record(env).get("ready") is True
            and Path(environments._env_python(str(env))).is_file())


def process_environment(env: Path) -> dict:
    """Isolate Python, pip and Qt plugin lookup from the running spaCR process.

    :param env: Starplast's virtual environment.
    :returns: subprocess variables retaining desktop display and user settings.
    """
    values = environments._clean_env(str(env))
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH",
                "QML_IMPORT_PATH", "PIP_CONFIG_FILE", "PIP_EXTRA_INDEX_URL"):
        values.pop(key, None)
    values["PIP_CONFIG_FILE"] = os.devnull
    values["PYQTGRAPH_QT_LIB"] = "PyQt6"
    values["QT_API"] = "pyqt6"
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        if key + "_ORIG" in values:
            values[key] = values.pop(key + "_ORIG")
        elif getattr(sys, "frozen", False):
            values.pop(key, None)
    return values


def _claim(root: Path) -> None:
    """Claim the install, refusing concurrent work and retaining live locks."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "starplast.lock"
    if path.exists():
        if environments._read_lock(str(root), "starplast"):
            raise environments._InstallFailed("Starplast is already being installed.")
        path.unlink()
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
    except FileExistsError as exc:
        raise environments._InstallFailed("Starplast is already being installed.") from exc


def install_starplast(source=None, *, root=None, progress=None, cancel=None,
                     runner=None, preflight=None) -> Path:
    """Build an isolated environment, install Starplast and check its bundled data.

    Local Git sources are archived from committed HEAD into temporary storage;
    pip never writes build files into the original checkout. A failed install
    removes only the environment created here, and retains ``starplast-install.log``.

    :param source: starplast on PyPI, or an explicit local Git checkout or
        repository URL; default_source otherwise. PyPI resolves the latest
        compatible stable release at installation time, without a version pin.
    :param root: external-application folder.
    :param progress: callback ``(step, steps, text)``; called off the GUI thread.
    :param cancel: threading.Event; cancels commands and their descendants.
    :param runner: optional subprocess-runner substitute for focused tests.
    :param preflight: optional interpreter-preflight substitute for focused tests.
    :returns: completed environment path.
    :raises RuntimeError: installation failed, was cancelled, or folder is unowned.
    :raises ValueError: the selected source is neither a suitable local Git
        checkout, the PyPI package, nor an allowed official repository URL.
    """
    root = apps_root(root)
    env = root / "starplast"
    if is_installed(root):
        return env
    source = str(source or default_source()).strip()
    local = Path(source).expanduser()
    is_local = source != PYPI_PACKAGE and local.exists()
    if is_local:
        if not (local / ".git").exists() or not (local / "starplast/app.py").is_file():
            raise ValueError("Choose a Starplast Git checkout containing starplast/app.py.")
        source = str(local.resolve())
    elif source not in (PYPI_PACKAGE, REPOSITORY) and not source.startswith("git+https://github.com/EinarOlafsson/starplast.git@"):
        raise ValueError("Choose starplast on PyPI, a local Starplast Git checkout or its official GitHub repository.")
    report = progress or (lambda *_args: None)
    run = runner or environments._run_step
    _claim(root)
    built = False
    try:
        if is_installed(root):
            return env
        if cancel is not None and cancel.is_set():
            raise environments._InstallCancelled("Starplast installation cancelled.")
        if env.is_symlink() or (env.exists() and not _record(env)):
            raise environments._InstallFailed(f"The folder {env} is not a spaCR-owned Starplast environment.")
        report(0, 5, "Checking Python, network and free disk space")
        interpreter = (preflight or environments._preflight)(_SPEC, str(root))
        if cancel is not None and cancel.is_set():
            raise environments._InstallCancelled("Starplast installation cancelled.")
        if env.exists():
            environments._remove_tree(str(env), str(root))
        env.mkdir()
        built = True
        (env / _OWNER).write_text(json.dumps({"app": "starplast", "ready": False}), encoding="utf-8")
        python = environments._env_python(str(env))
        variables = process_environment(env)
        with tempfile.TemporaryDirectory(prefix="starplast-source-", dir=root) as temporary:
            requirement = source
            steps = []
            if is_local:
                requirement = str(Path(temporary) / "starplast.tar")
                steps.append(("Snapshot committed Starplast source", ["git", "-C", source, "archive", "--format=tar", "--output", requirement, "HEAD"]))
            steps.extend([
                ("Create Starplast environment", list(interpreter) + ["-m", "venv", str(env)]),
                ("Install pip", [python, "-I", "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "--upgrade", "pip"]),
                ("Install Starplast and dependencies", [python, "-I", "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "--upgrade", "--no-cache-dir", requirement]),
                ("Check Starplast and bundled data", [python, "-I", "-c", _SELFTEST]),
            ])
            with (root / "starplast-install.log").open("w", encoding="utf-8") as log:
                for number, (label, argv) in enumerate(steps):
                    report(number, len(steps), label)
                    log.write("$ " + environments._quote(argv) + "\n")
                    log.flush()

                    def line(text, number=number, label=label):
                        """Persist all subprocess output and send the latest line to Qt."""
                        log.write(text + "\n")
                        log.flush()
                        report(number, len(steps), label + ": " + text)

                    code, tail = run(argv, env=variables, cwd=str(root), on_line=line, cancel=cancel)
                    if code:
                        raise environments._InstallFailed(label + " failed:\n" + "\n".join(tail[-40:]) + f"\nLog: {root / 'starplast-install.log'}")
            try:
                result = json.loads(tail[-1])
                if result.get("ok") is not True:
                    raise ValueError("self-test did not succeed")
            except (IndexError, ValueError, AttributeError) as exc:
                raise environments._InstallFailed("Starplast did not confirm that its imports and bundled data are ready.") from exc
        record = {"app": "starplast", "ready": True, "source": source, "version": result.get("version", "")}
        temporary = env / (_OWNER + ".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(env / _OWNER)
        report(len(steps), len(steps), "Starplast is installed")
        return env
    except BaseException:
        if built:
            environments._remove_tree(str(env), str(root))
        raise
    finally:
        environments._release_lock(str(root), "starplast")


def launch_starplast(*, root=None, popen=None):
    """Start Starplast as a separate process, logging output beside its environment.

    :param root: external-application folder.
    :param popen: optional subprocess.Popen substitute for focused tests.
    :returns: child process, which the UI monitors for launch errors.
    :raises RuntimeError: no completed environment exists.
    """
    root = apps_root(root)
    if not is_installed(root):
        raise RuntimeError("Starplast must be installed before it can open.")
    env = root / "starplast"
    with (root / "starplast-launch.log").open("ab") as log:
        return (popen or subprocess.Popen)(
            [environments._env_python(str(env)), "-I", "-m", "starplast"],
            cwd=str(root), env=process_environment(env), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, **environments._detached())
