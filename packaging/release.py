#!/usr/bin/env python3
"""Small release helper used locally and by GitHub Actions.

The native builders remain platform-specific. This helper owns the two pieces
that must be identical everywhere: changing the single package version and
collecting native online installers into the repository while updating the
README download links.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

from packaging.version import InvalidVersion, Version


VERSION_PATTERN = re.compile(
    r'^(VERSION\s*=\s*)(["\'])([^"\']+)(["\'])$', re.MULTILINE)
README_BEGIN = ".. spacr-installer-links-begin"
README_END = ".. spacr-installer-links-end"
PLATFORMS = (
    ("Windows 10/11", "Windows-Online-Setup.exe"),
    ("macOS 11+ (Intel and Apple silicon)", "macOS-Universal-Online.pkg"),
    ("64-bit Linux", "Linux-x86_64-Online.run"),
)
RELEASE_DOWNLOAD_ROOT = "https://github.com/EinarOlafsson/spacr/releases/download"


def read_version(setup_path: Path) -> str:
    match = VERSION_PATTERN.search(setup_path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Could not find VERSION in {setup_path}")
    return match.group(3)


def bump_version(
    setup_path: Path,
    requested: str,
    *,
    allow_current: bool = False,
) -> str:
    try:
        new = Version(requested)
    except InvalidVersion as exc:
        raise ValueError(f"{requested!r} is not a valid Python package version") from exc
    current_text = setup_path.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(current_text)
    if not match:
        raise ValueError(f"Could not find VERSION in {setup_path}")
    current = Version(match.group(3))
    if new == current and allow_current:
        return str(current)
    if new <= current:
        raise ValueError(
            f"New version {new} must be greater than current version {current}")
    replacement = f'{match.group(1)}"{new}"'
    setup_path.write_text(
        current_text[:match.start()] + replacement + current_text[match.end():],
        encoding="utf-8",
    )
    return str(new)


def _installer_paths(source: Path, version: str) -> list[tuple[str, Path]]:
    found = []
    for label, suffix in PLATFORMS:
        name = f"SpaCR-{version}-{suffix}"
        matches = list(source.rglob(name))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {name} below {source}, found {len(matches)}")
        found.append((label, matches[0]))
    return found


def _readme_links(version: str, branch: str) -> str:
    # ``branch`` remains part of the public helper API for compatibility with
    # older local release commands. Published README links intentionally use
    # immutable GitHub release assets rather than mutable branch contents.
    del branch
    lines = [README_BEGIN, ""]
    for label, suffix in PLATFORMS:
        name = f"SpaCR-{version}-{suffix}"
        url = f"{RELEASE_DOWNLOAD_ROOT}/v{version}/{name}"
        lines.append(f"* `{label}: download SpaCR {version} <{url}>`_")
    lines.extend(["", README_END])
    return "\n".join(lines)


def _replace_readme_block(readme: Path, replacement: str) -> None:
    text = readme.read_text(encoding="utf-8")
    start = text.find(README_BEGIN)
    end = text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(
            f"{readme} must contain {README_BEGIN!r} and {README_END!r}")
    end += len(README_END)
    readme.write_text(text[:start] + replacement + text[end:],
                      encoding="utf-8")


def collect_installers(
    source: Path,
    destination: Path,
    readme: Path,
    setup_path: Path,
    branch: str,
) -> list[Path]:
    version = read_version(setup_path)
    installers = _installer_paths(source, version)
    destination.mkdir(parents=True, exist_ok=True)

    # This folder intentionally contains only the current lightweight set.
    expected_names = {path.name for _, path in installers}
    for old in destination.glob("SpaCR-*-Online*"):
        if old.name not in expected_names and old.is_file():
            old.unlink()

    copied = []
    manifest_rows = []
    for label, path in installers:
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest_rows.append((label, target.name, target.stat().st_size, digest))

    application_readme = [
        "SpaCR lightweight installers",
        "============================",
        "",
        f"Current version: ``{version}``",
        "",
        "These small online installers download a private Python runtime and",
        "dependencies during installation. SHA-256 hashes:",
        "",
    ]
    for label, name, size, digest in manifest_rows:
        application_readme.extend([
            label,
            "-" * len(label),
            "",
            f"* File: ``{name}``",
            f"* Size: ``{size}`` bytes",
            f"* SHA-256: ``{digest}``",
            "",
        ])
    (destination / "README.rst").write_text(
        "\n".join(application_readme), encoding="utf-8")
    _replace_readme_block(readme, _readme_links(version, branch))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version")
    version_parser.add_argument("--setup", type=Path, default=Path("setup.py"))

    bump_parser = subparsers.add_parser("bump")
    bump_parser.add_argument("new_version")
    bump_parser.add_argument("--setup", type=Path, default=Path("setup.py"))
    bump_parser.add_argument(
        "--allow-current",
        action="store_true",
        help="treat an already-current requested version as an idempotent rerun",
    )

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--source", type=Path, default=Path("dist/online"))
    collect_parser.add_argument(
        "--destination", type=Path, default=Path("spacr/application"))
    collect_parser.add_argument("--readme", type=Path, default=Path("README.rst"))
    collect_parser.add_argument("--setup", type=Path, default=Path("setup.py"))
    collect_parser.add_argument("--branch", default="spacr-nightly")

    args = parser.parse_args()
    if args.command == "version":
        print(read_version(args.setup))
    elif args.command == "bump":
        print(bump_version(
            args.setup,
            args.new_version,
            allow_current=args.allow_current,
        ))
    else:
        for path in collect_installers(
                args.source, args.destination, args.readme, args.setup,
                args.branch):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
