#!/usr/bin/env python3
"""Build an offline (air-gapped) spaCR installer bundle.

The online installers download uv, a private CPython and every wheel at
install time. A locked-down microscope PC cannot. This builder runs on a
machine WITH network access and packs everything that install needs into
one folder (and optionally one ``.tar``):

    install.sh / install.ps1   the online installer, run with --offline-bundle
    uv/                        the pinned uv binary for the target platform
    python/<tag>/<archive>     the CPython build uv would have downloaded,
                               laid out as a uv ``UV_PYTHON_INSTALL_MIRROR``
    wheels/                    every wheel of the locked environment
    requirements.txt           the lock, resolved for the target platform
    models/cellpose/           Cellpose weights (default: cpsam)
    test_data/plate1/          Mask test images and gen_mask_settings.csv
    offline_mask_check.py      runs Mask on test_data after install
    bundle.json, SHA256SUMS    what is inside and the checksum of every file

The target platform is chosen at build time, and so is the PyTorch backend
(``cpu`` or a CUDA line such as ``cu126``): an offline machine cannot let the
installer pick one. Wheels are fetched for the target platform with
``pip download --platform``, so a Windows bundle can be built on Linux; the
installer for a Windows bundle is ``install_spacr_windows.ps1``.

Usage::

    python packaging/offline/build_offline_bundle.py --platform linux-x86_64 \\
        --torch-backend cpu --from-source . --out dist/offline \\
        --test-data ~/.cache/spacr/example_data/plate1 --archive

Nothing is published. The online installers remain the default download.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform as host_platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
PACKAGING = HERE.parent
REPO = PACKAGING.parent


def _load_release():
    """``packaging/release.py``, loaded by path: ``packaging/`` is a folder of
    scripts, not a package, and must not shadow the PyPI ``packaging``."""
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper", PACKAGING / "release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _load_release()

ONLINE = PACKAGING / "online"
UNIX_INSTALLER = ONLINE / "install_spacr_unix.sh"
WINDOWS_INSTALLER = ONLINE / "install_spacr_windows.ps1"
MASK_CHECK = HERE / "offline_mask_check.py"
PYTHON_VERSION = "3.12"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/download"
TORCH_INDEX = "https://download.pytorch.org/whl/{backend}"
DEFAULT_CELLPOSE_MODELS = ("cpsam",)
DEFAULT_EXTRAS = "qt"
FIELD_PATTERN = re.compile(r"^(?P<field>.+?_[A-Z]+\d+_T\d+F\d+)")


@dataclass(frozen=True)
class Target:
    """One platform an offline bundle can be built for."""

    key: str
    uv_triple: str
    uv_archive: str
    python_platform: str
    pip_platforms: tuple[str, ...]
    python_os: str
    python_arch: str
    python_libc: str
    installer: Path


#: pip does not widen an explicit ``--platform``, so every tag a compatible
#: wheel may carry is listed: manylinux 2.17 to 2.28 (glibc 2.28 and newer
#: systems such as RHEL 8 and Ubuntu 20.04), and macOS 11 to 15.
_MANYLINUX = tuple(f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)) + (
    "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64")
_MACOS_ARM = tuple(f"macosx_{major}_0_arm64" for major in range(15, 10, -1)) + tuple(
    f"macosx_{major}_0_universal2" for major in range(15, 10, -1)) + tuple(
    f"macosx_10_{minor}_universal2" for minor in range(16, 8, -1))

TARGETS = {
    "linux-x86_64": Target(
        "linux-x86_64", "x86_64-unknown-linux-gnu", "tar.gz",
        "x86_64-manylinux_2_28", _MANYLINUX,
        "linux", "x86_64", "gnu", UNIX_INSTALLER),
    "windows-x86_64": Target(
        "windows-x86_64", "x86_64-pc-windows-msvc", "zip",
        "x86_64-pc-windows-msvc", ("win_amd64",),
        "windows", "x86_64", "none", WINDOWS_INSTALLER),
    "macos-arm64": Target(
        "macos-arm64", "aarch64-apple-darwin", "tar.gz",
        "aarch64-apple-darwin", _MACOS_ARM,
        "macos", "aarch64", "none", UNIX_INSTALLER),
}


def uv_version() -> str:
    """The uv release the online installers pin, read from the Unix one."""
    match = re.search(r'^UV_VERSION="([^"]+)"', UNIX_INSTALLER.read_text(
        encoding="utf-8"), re.MULTILINE)
    if not match:
        raise SystemExit(f"UV_VERSION not found in {UNIX_INSTALLER}")
    return match.group(1)


def resolver_guards() -> list[str]:
    """The default resolver guards of the online installer, so both agree."""
    match = re.search(r'^RESOLVER_GUARDS=\(([^)]*)\)', UNIX_INSTALLER.read_text(
        encoding="utf-8"), re.MULTILINE)
    return re.findall(r'"([^"]+)"', match.group(1)) if match else []


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of one file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` over HTTPS unless it is already there."""
    if not url.startswith("https://"):
        raise SystemExit(f"refusing a non-HTTPS download: {url}")
    if dest.is_file() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "spacr-offline-bundle"})
    with urllib.request.urlopen(request, timeout=60) as response, \
            open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 20)
    partial.replace(dest)
    return dest


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a build step, echoing it, and stop the build on failure."""
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def fetch_uv(target: Target, version: str, dest: Path) -> Path:
    """Download and unpack the pinned uv for ``target``; verify its sha256."""
    name = f"uv-{target.uv_triple}.{target.uv_archive}"
    archive = download(f"{UV_RELEASES}/{version}/{name}", dest / "download" / name)
    sidecar = download(f"{UV_RELEASES}/{version}/{name}.sha256",
                       dest / "download" / f"{name}.sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(archive) != expected:
        raise SystemExit(f"{name}: checksum does not match the uv release")
    exe = "uv.exe" if target.uv_archive == "zip" else "uv"
    if target.uv_archive == "zip":
        with zipfile.ZipFile(archive) as bundle:
            member = next(n for n in bundle.namelist() if n.endswith("/" + exe) or n == exe)
            data = bundle.read(member)
    else:
        with tarfile.open(archive) as bundle:
            member = next(m for m in bundle.getmembers() if m.name.endswith("/" + exe))
            data = bundle.extractfile(member).read()
    out = dest / exe
    out.write_bytes(data)
    out.chmod(0o755)
    shutil.rmtree(dest / "download")
    return out


def host_target() -> Target:
    """The target matching the machine the builder runs on."""
    system, machine = sys.platform, host_platform.machine().lower()
    if system.startswith("linux") and machine in ("x86_64", "amd64"):
        return TARGETS["linux-x86_64"]
    if system == "win32":
        return TARGETS["windows-x86_64"]
    if system == "darwin" and machine == "arm64":
        return TARGETS["macos-arm64"]
    raise SystemExit(f"no uv build is pinned for this host ({system}/{machine})")


def choose_python_download(entries: list[dict], target: Target) -> dict:
    """The CPython build uv would install for ``target`` from its listing.

    ``uv python list --all-platforms --only-downloads`` lists newest first,
    which is also the build ``uv python install 3.12`` picks, so the first
    default-variant CPython for the target's OS, architecture and libc is the
    one the offline install will ask its mirror for.
    """
    for entry in entries:
        if (entry.get("implementation") == "cpython"
                and entry.get("variant") == "default"
                and entry.get("os") == target.python_os
                and entry.get("arch") == target.python_arch
                and entry.get("libc") == target.python_libc
                and entry.get("url")):
            return entry
    raise SystemExit(f"uv lists no CPython {PYTHON_VERSION} for {target.key}")


def mirror_relative_path(url: str) -> Path:
    """Where a python-build-standalone URL lives under a uv install mirror.

    uv swaps its download prefix for ``UV_PYTHON_INSTALL_MIRROR`` and keeps
    the last two components, ``<release tag>/<archive name>``.
    """
    parts = urlsplit(url).path.split("/")
    return Path(unquote(parts[-2])) / unquote(parts[-1])


def fetch_python(host_uv: Path, target: Target, dest: Path) -> dict:
    """Place the target's CPython archive in a uv install mirror layout."""
    listing = run([host_uv, "python", "list", PYTHON_VERSION, "--all-platforms",
                   "--only-downloads", "--output-format", "json"],
                  capture_output=True, text=True).stdout
    entry = choose_python_download(json.loads(listing), target)
    relative = mirror_relative_path(entry["url"])
    download(entry["url"], dest / relative)
    return {"version": entry["version"], "key": entry["key"],
            "archive": str(Path("python") / relative).replace(os.sep, "/")}


def build_spacr_wheel(source: Path, wheels: Path, host_uv: Path) -> Path:
    """Build spaCR's own wheel from a checkout into ``wheels``."""
    run([host_uv, "build", "--wheel", "--out-dir", wheels, source])
    built = sorted(wheels.glob("spacr-*.whl"), key=lambda p: p.stat().st_mtime)
    if not built:
        raise SystemExit("building the spaCR wheel produced no spacr-*.whl")
    return built[-1]


def lock_requirements(host_uv: Path, target: Target, backend: str,
                      requirements_in: Path, wheels: Path, out: Path) -> None:
    """Resolve the environment for the target platform into ``out``."""
    run([host_uv, "pip", "compile", requirements_in,
         "--python-version", PYTHON_VERSION,
         "--python-platform", target.python_platform,
         "--torch-backend", backend,
         "--find-links", wheels,
         "--no-header", "--no-annotate", "--quiet",
         "--output-file", out])


def locked_pins(lock: Path) -> list[str]:
    """``name==version`` lines of a compiled lock, without options or comments."""
    pins = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-") and "==" in line:
            pins.append(line.split(";", 1)[0].strip())
    return pins


def _normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _have_wheel(pin: str, wheels: Path) -> bool:
    name, version = pin.split("==", 1)
    stem = _normalised(name.split("[", 1)[0]).replace("-", "_")
    return any(_normalised(p.name.split("-", 1)[0]).replace("-", "_") == stem
               and p.name.split("-")[1] == version for p in wheels.glob("*.whl"))


def download_wheels(pins: list[str], target: Target, backend: str,
                    wheels: Path) -> list[str]:
    """``pip download`` every pin as a wheel for the target platform.

    Returns the pins no binary wheel exists for. A pure-Python sdist is then
    built into a ``py3-none-any`` wheel on this machine, which installs on
    every platform; anything else is reported and the build stops.
    """
    wanted = [pin for pin in pins if not _have_wheel(pin, wheels)]
    base = [sys.executable, "-m", "pip", "download", "--no-deps",
            "--only-binary=:all:", "--dest", wheels,
            "--python-version", PYTHON_VERSION, "--implementation", "cp",
            "--abi", "cp312", "--abi", "abi3", "--abi", "none",
            "--extra-index-url", TORCH_INDEX.format(backend=backend)]
    for tag in target.pip_platforms:
        base += ["--platform", tag]
    if wanted and subprocess.run([str(p) for p in base + wanted]).returncode == 0:
        return []
    missing = []
    for pin in wanted:
        if _have_wheel(pin, wheels):
            continue
        if subprocess.run([str(p) for p in base + [pin]]).returncode != 0:
            missing.append(pin)
    unbuilt = []
    for pin in missing:
        with tempfile.TemporaryDirectory() as scratch:
            built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                                    "--wheel-dir", scratch, pin])
            made = list(Path(scratch).glob("*-py3-none-any.whl")) + list(
                Path(scratch).glob("*-py2.py3-none-any.whl"))
            if built.returncode == 0 and made:
                shutil.move(str(made[0]), wheels / made[0].name)
            else:
                unbuilt.append(pin)
    return unbuilt


def copy_cellpose_models(names, source: Path, dest: Path) -> list[str]:
    """Copy named Cellpose weights (and their ``size_*`` files) into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        weights = source / name
        if not weights.is_file():
            raise SystemExit(
                f"Cellpose model {name!r} is not in {source}. Run Cellpose once "
                "with network access so it downloads, or pass --models-from.")
        for path in [weights, *source.glob(f"size_{name}.npy")]:
            shutil.copy2(path, dest / path.name)
            copied.append(path.name)
    return copied


def select_test_fields(images: list[Path], limit: int) -> list[Path]:
    """Every channel file of the first ``limit`` fields; all files at 0.

    spaCR names each acquisition ``<plate>_<well>_T<t>F<field>...C<ch>.tif``,
    and Mask needs every channel of a field, so fields are chosen whole.
    """
    images = sorted(images)
    if limit <= 0:
        return images
    fields: list[str] = []
    for path in images:
        match = FIELD_PATTERN.match(path.name)
        key = match.group("field") if match else path.stem
        if key not in fields:
            fields.append(key)
    keep = set(fields[:limit])
    return [p for p in images if (m := FIELD_PATTERN.match(p.name))
            and m.group("field") in keep or p.stem in keep]


def copy_test_data(source: Path, dest: Path, fields: int) -> int:
    """Copy Mask test images and their Mask settings into ``dest``."""
    images = select_test_fields(list(source.glob("*.tif")), fields)
    settings = source / "settings" / "gen_mask_settings.csv"
    if not images or not settings.is_file():
        raise SystemExit(f"{source} holds no *.tif and settings/gen_mask_settings.csv; "
                         "point --test-data at the Mask example plate folder")
    (dest / "settings").mkdir(parents=True, exist_ok=True)
    for path in images:
        shutil.copy2(path, dest / path.name)
    shutil.copy2(settings, dest / "settings" / settings.name)
    return len(images)


def write_checksums(root: Path) -> Path:
    """``SHA256SUMS`` over every file of the bundle, in ``sha256sum`` format."""
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    out = root / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def render_installer(target: Target, version: str, bundle: Path) -> Path:
    """Place the target's online installer, messages embedded, in the bundle."""
    if target.installer == UNIX_INSTALLER:
        out = bundle / "install.sh"
        run([sys.executable, PACKAGING / "i18n" / "render.py",
             "--embed-unix", UNIX_INSTALLER, "--output", out, "--version", version])
        out.chmod(0o755)
        return out
    out = bundle / "install.ps1"
    shutil.copy2(WINDOWS_INSTALLER, out)
    shutil.copy2(REPO / "spacr" / "install_cleanup.py", bundle / "install_cleanup.py")
    run([sys.executable, PACKAGING / "i18n" / "render.py"])
    generated = bundle / "generated"
    generated.mkdir(exist_ok=True)
    catalog = (ONLINE / "generated" / "installer_messages.ps1").read_text(encoding="utf-8")
    (generated / "installer_messages.ps1").write_text(catalog, encoding="utf-8-sig")
    return out


def build(args: argparse.Namespace) -> Path:
    """Assemble the bundle folder and, with ``--archive``, its ``.tar``."""
    target = TARGETS[args.platform]
    version = release.read_version(REPO / "setup.py")
    name = release.offline_bundle_name(version, target.key, args.torch_backend)
    bundle = Path(args.out) / name
    if bundle.exists():
        shutil.rmtree(bundle)
    wheels = bundle / "wheels"
    wheels.mkdir(parents=True)
    uv_pin = uv_version()
    cache = Path(args.out) / ".host-uv"
    host = host_target()
    host_uv = cache / ("uv.exe" if host.uv_archive == "zip" else "uv")
    if not host_uv.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        fetch_uv(host, uv_pin, cache)
    target_uv = fetch_uv(target, uv_pin, bundle / "uv")
    python = fetch_python(host_uv, target, bundle / "python")

    spec = args.package_spec or f"spacr[{args.extras}]=={version}"
    if args.from_source:
        build_spacr_wheel(Path(args.from_source), wheels, host_uv)
        spec = f"spacr[{args.extras}]=={version}"
    requirements_in = bundle / "requirements.in"
    requirements_in.write_text("\n".join([spec, *resolver_guards()]) + "\n",
                               encoding="utf-8")
    lock = bundle / "requirements.txt"
    lock_requirements(host_uv, target, args.torch_backend, requirements_in,
                      wheels, lock)
    unbuilt = download_wheels(locked_pins(lock), target, args.torch_backend, wheels)
    if unbuilt:
        raise SystemExit("no wheel for the target platform: " + ", ".join(unbuilt))

    models = copy_cellpose_models(args.cellpose_model or DEFAULT_CELLPOSE_MODELS,
                                  Path(args.models_from).expanduser(),
                                  bundle / "models" / "cellpose")
    test_files = 0
    if args.test_data:
        test_files = copy_test_data(Path(args.test_data).expanduser(),
                                    bundle / "test_data" / "plate1", args.test_fields)
    shutil.copy2(MASK_CHECK, bundle / MASK_CHECK.name)
    installer = render_installer(target, version, bundle)

    manifest = {
        "format": 1, "spacr_version": version, "platform": target.key,
        "torch_backend": args.torch_backend, "package_spec": spec,
        "uv": {"version": uv_pin, "path": target_uv.relative_to(bundle).as_posix()},
        "python": python, "installer": installer.name,
        "wheels": len(list(wheels.glob("*.whl"))),
        "cellpose_models": models, "test_data_files": test_files,
    }
    (bundle / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    write_checksums(bundle)
    print(f"Built {bundle}")
    if args.archive:
        archive = bundle.with_name(bundle.name + ".tar")
        with tarfile.open(archive, "w") as tar:
            tar.add(bundle, arcname=bundle.name)
        print(f"Built {archive}")
        return archive
    return bundle


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of the builder."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--platform", choices=sorted(TARGETS), default="linux-x86_64")
    parser.add_argument("--torch-backend", default="cpu",
                        help="PyTorch wheel line: cpu, or a CUDA line such as cu126.")
    parser.add_argument("--extras", default=DEFAULT_EXTRAS,
                        help="spaCR extras to install, comma separated. Default qt.")
    parser.add_argument("--package-spec", default="",
                        help="Requirement for spaCR itself; default the setup.py version.")
    parser.add_argument("--from-source", default="",
                        help="Build spaCR's wheel from this checkout instead of PyPI.")
    parser.add_argument("--cellpose-model", action="append",
                        help="Cellpose weights to include; repeatable. Default cpsam.")
    parser.add_argument("--models-from", default=str(Path.home() / ".cellpose" / "models"))
    parser.add_argument("--test-data", default="",
                        help="Mask example plate folder (*.tif + settings/).")
    parser.add_argument("--test-fields", type=int, default=0,
                        help="Fields of test data to include; 0 means all.")
    parser.add_argument("--out", default="dist/offline")
    parser.add_argument("--archive", action="store_true",
                        help="Also write the bundle as one uncompressed .tar.")
    return parser


def main(argv=None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[a-z0-9]+", args.torch_backend) or args.torch_backend == "auto":
        raise SystemExit("--torch-backend must name one wheel line, e.g. cpu or cu126")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
