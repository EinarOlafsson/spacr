#!/usr/bin/env python3
"""Small release helper used locally and by GitHub Actions.

The native builders remain platform-specific. This helper owns the two pieces
that must be identical everywhere: changing the single package version and
collecting native online installers into the repository while updating the
README download links in every supported documentation language.
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
LOCALIZED_README_CODES = (
    "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
)
#: The README shows one drawn platform icon per download instead of a text
#: link. The artwork is committed and served from this repository -- a README
#: icon hotlinked from a CDN dies the day that host moves and leaks a request
#: to a third party for every reader of the page.
README_ICON_ROOT = (
    "https://raw.githubusercontent.com/EinarOlafsson/spacr/main"
    "/spacr/resources/icons/platforms"
)
README_ICON_WIDTH = 64
#: ``(substitution key, artwork stem, alt-text platform)``, in ``PLATFORMS`` order
README_ICONS = (
    ("Windows", "windows", "Windows 10/11"),
    ("MacOS", "macos", "macOS 11+ (Intel and Apple silicon)"),
    ("Linux", "linux", "64-bit Linux"),
)

#: Where the generated download archive is written. It is a page of the
#: published Sphinx site (``docs/source`` -> ``docs/_build/html`` ->
#: einarolafsson.github.io/spacr), NOT of the built copy checked in beside it:
#: writing into ``docs/`` directly is a change the next docs build deletes.
INSTALLER_INDEX_PATH = Path("docs/source/installers.rst")
#: Column order on that page, as requested: version, then Linux, macOS,
#: Windows. The icons are the same artwork the README's download row uses --
#: one glyph set, one place it is drawn.
INSTALLER_INDEX_COLUMNS = ("Linux", "MacOS", "Windows")
INSTALLER_INDEX_ICON_WIDTH = 32
#: GitHub's release list for this project. Public, so no token is required,
#: but one is used when the environment offers it because the anonymous rate
#: limit is 60 requests an hour per address and CI shares its address.
GITHUB_RELEASES_API = (
    "https://api.github.com/repos/EinarOlafsson/spacr/releases?per_page=100")


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
        name = f"spaCR-{version}-{suffix}"
        matches = list(source.rglob(name))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {name} below {source}, found {len(matches)}")
        found.append((label, matches[0]))
    return found


def _installer_url_pattern(suffix: str) -> re.Pattern[str]:
    """Match one release-asset URL wherever it sits in the README block.

    The download link used to be the target of an inline ``\\`text <url>\\`_``
    reference and is now the ``:target:`` of an image directive, so the URL is
    no longer wrapped in angle brackets and no longer shares a line with its
    label. Matching the URL itself rather than the syntax around it keeps a
    translated README that has not been converted yet working unchanged.

    The stem is matched case-insensitively. The builders name their output
    ``spaCR-<version>-...`` but the assets published under the current tag
    carry a capital S, and GitHub's release download endpoint serves either
    spelling; a case-sensitive match here silently failed every ``collect``.
    """
    return re.compile(
        rf"{re.escape(RELEASE_DOWNLOAD_ROOT)}/v(?P<tag_version>[^/\s<>]+)/"
        rf"spaCR-(?P<file_version>[^/\s<>]+)-{re.escape(suffix)}\b",
        re.IGNORECASE,
    )


def _readme_links(version: str, branch: str) -> str:
    # ``branch`` remains part of the public helper API for compatibility with
    # older local release commands. Published README links intentionally use
    # immutable GitHub release assets rather than mutable branch contents.
    del branch
    references = " ".join(f"|Installer{key}|" for key, _, _ in README_ICONS)
    lines = [README_BEGIN, "", references, ""]
    for (key, platform, alt), (label, suffix) in zip(README_ICONS, PLATFORMS):
        name = f"spaCR-{version}-{suffix}"
        url = f"{RELEASE_DOWNLOAD_ROOT}/v{version}/{name}"
        lines.extend([
            f".. |Installer{key}| image:: {README_ICON_ROOT}/{platform}.png",
            f"   :width: {README_ICON_WIDTH}",
            f"   :alt: Download spaCR {version} for {alt or label}",
            f"   :target: {url}",
        ])
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


def _localized_readmes(readme: Path, directory: Path | None) -> tuple[Path, ...]:
    """Return the exact translated README set belonging to ``readme``.

    A release is only coherent when every language advertises the same
    immutable assets.  Missing translations therefore fail collection instead
    of quietly leaving stale download links behind.
    """
    locale_dir = directory or readme.parent / "docs" / "i18n" / "readme"
    paths = tuple(
        locale_dir / f"README.{code}.rst" for code in LOCALIZED_README_CODES
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing localized README files: " + ", ".join(missing)
        )
    return paths


def _updated_readme_text(readme: Path, version: str) -> str:
    """Update installer versions and URLs without replacing translated prose."""
    text = readme.read_text(encoding="utf-8")
    start = text.find(README_BEGIN)
    end = text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(
            f"{readme} must contain {README_BEGIN!r} and {README_END!r}"
        )
    block_end = end + len(README_END)
    block = text[start:block_end]

    advertised = set()
    for _label, suffix in PLATFORMS:
        matches = list(_installer_url_pattern(suffix).finditer(block))
        if len(matches) != 1:
            raise ValueError(
                f"{readme} must contain exactly one installer link for {suffix}; "
                f"found {len(matches)}"
            )
        match = matches[0]
        if match.group("tag_version") != match.group("file_version"):
            raise ValueError(
                f"{readme} has a malformed installer link for {suffix}"
            )
        advertised.add(match.group("tag_version"))
    if len(advertised) != 1:
        raise ValueError(
            f"{readme} advertises more than one installer version: "
            + ", ".join(sorted(advertised))
        )
    old_version = advertised.pop()

    # The version appears in the URL tag, in the asset file name and in the
    # human-readable label or ``:alt:`` text beside it. Rewriting the whole
    # block leaves every one of them consistent whether the block is still a
    # list of text links or the icon row, and whatever language it is in.
    block = block.replace(old_version, version)

    for _label, suffix in PLATFORMS:
        match = _installer_url_pattern(suffix).search(block)
        if match is None or match.group("tag_version") != version or (
                match.group("file_version") != version):
            raise ValueError(
                f"{readme} could not be updated to installer version {version}"
            )

    return text[:start] + block + text[block_end:]


def installer_index_rows(releases) -> list[tuple[str, dict[str, str]]]:
    """Return ``[(version, {suffix: url})]``, newest release first.

    ``releases`` is GitHub's release list as the API returns it: mappings
    carrying ``tag_name`` and ``assets``.

    A release that shipped no installer at all is left out rather than given a
    row of empty cells -- ``v1.4.9.8``, ``v1.3.6`` and ``v1.3.5`` are real
    releases with real wheels and no native installers, and listing them on a
    page whose whole subject is installers only invites the click that finds
    nothing. A release that shipped SOME of the three keeps its row and gets
    an empty cell for the platform it missed, because that is a fact about
    that release rather than a reason to hide it.

    :param releases: mappings with ``tag_name`` and ``assets``.
    :returns: version + ``{asset suffix: download URL}``, newest first.
    :raises ValueError: when a release carrying installers has a tag that is
        not a version -- silently dropping it would be a missing row on a page
        nobody re-reads.
    """
    rows = []
    for release in releases:
        tag = str(release.get("tag_name") or "")
        found = {}
        for asset in release.get("assets") or ():
            name = str(asset.get("name") or "")
            for _label, suffix in PLATFORMS:
                if name.lower().endswith(f"-{suffix}".lower()):
                    found[suffix] = str(
                        asset.get("browser_download_url")
                        or f"{RELEASE_DOWNLOAD_ROOT}/{tag}/{name}")
        if not found:
            continue
        try:
            version = Version(tag.lstrip("vV"))
        except InvalidVersion as exc:
            raise ValueError(
                f"release {tag!r} ships installers but its tag is not a "
                f"version, so it cannot be placed in the archive table"
            ) from exc
        rows.append((version, found))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [(str(version), assets) for version, assets in rows]


def render_installer_index(releases, current_version: str) -> str:
    """Render the download archive page.

    Every released installer stays on GitHub -- a new tag cannot touch an old
    tag's assets -- but ``collect`` rewrites the README's three links to the
    version being released, so the moment a new version ships there is no path
    anywhere in the project to the previous one. This page is that path, and
    it is generated rather than maintained: a table of download links typed by
    hand is a table that is wrong one release later.

    :param releases: GitHub's release list; see :func:`installer_index_rows`.
    :param current_version: the version the README currently advertises.
    :returns: the reStructuredText page.
    """
    rows = installer_index_rows(releases)
    suffix_by_key = dict(zip((key for key, _, _ in README_ICONS),
                             (suffix for _label, suffix in PLATFORMS)))
    alt_by_key = {key: alt for key, _stem, alt in README_ICONS}
    stem_by_key = {key: stem for key, stem, _alt in README_ICONS}

    lines = [
        ".. _installer-archive:",
        "",
        "Installer archive",
        "=================",
        "",
        "Every spaCR release that shipped desktop installers, newest first.",
        "The README always links to the latest; this page is where to find",
        "any other version.",
        "",
        "These are *online* installers, and each one pins the version it was",
        "built for -- ``spacr[...]==<version>`` -- so an installer from this",
        "table installs that release and not the current one.",
        "",
    ]
    for key in INSTALLER_INDEX_COLUMNS:
        lines.extend([
            f".. |{key}| image:: {README_ICON_ROOT}/{stem_by_key[key]}.png",
            f"   :width: {INSTALLER_INDEX_ICON_WIDTH}",
            f"   :alt: {alt_by_key[key]}",
            "",
        ])
    lines.extend([
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 25 25 25 25",
        "",
        "   * - Version",
    ])
    lines.extend(f"     - |{key}|" for key in INSTALLER_INDEX_COLUMNS)
    for version, assets in rows:
        label = version + (" (current)" if version == current_version else "")
        lines.append(f"   * - {label}")
        for key in INSTALLER_INDEX_COLUMNS:
            url = assets.get(suffix_by_key[key])
            if url is None:
                # No installer for that platform in that release. An empty
                # cell, never a link that 404s.
                lines.append("     - ")
            else:
                extension = Path(suffix_by_key[key]).suffix
                # An ANONYMOUS reference (two underscores). Every row reuses
                # the same three link labels, and a named target repeated with
                # a different URL is a duplicate-target error in docutils.
                lines.append(f"     - `{extension} <{url}>`__")
    lines.append("")
    return "\n".join(lines)


def fetch_releases(url: str = GITHUB_RELEASES_API):
    """Return GitHub's release list for this project.

    Kept apart from the rendering so the page can be generated from a
    recorded list in a test without reaching the network.

    :param url: the releases endpoint.
    :returns: the decoded JSON list.
    """
    import json
    import os
    import urllib.request

    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "spacr-release-helper"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def write_installer_index(
    destination: Path,
    setup_path: Path,
    releases=None,
) -> Path:
    """Write the archive page and return where it went.

    :param destination: the page to write.
    :param setup_path: ``setup.py``, read for the current version.
    :param releases: release list; fetched from GitHub when omitted.
    :returns: the written path.
    """
    if releases is None:
        releases = fetch_releases()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_installer_index(releases, read_version(setup_path)),
        encoding="utf-8")
    return destination


def collect_installers(
    source: Path,
    destination: Path,
    readme: Path,
    setup_path: Path,
    branch: str,
    localized_readme_dir: Path | None = None,
) -> list[Path]:
    version = read_version(setup_path)
    installers = _installer_paths(source, version)
    readmes = (readme, *_localized_readmes(readme, localized_readme_dir))
    # Validate every translated block before copying or deleting anything.
    readme_updates = {
        path: _updated_readme_text(path, version) for path in readmes
    }
    destination.mkdir(parents=True, exist_ok=True)

    # This folder intentionally contains only the current lightweight set.
    expected_names = {path.name for _, path in installers}
    for old in destination.glob("spaCR-*-Online*"):
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
        "spaCR lightweight installers",
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
    # ``branch`` remains accepted for compatibility with old release commands;
    # every README now points to the immutable release tag.
    del branch
    for path, updated in readme_updates.items():
        path.write_text(updated, encoding="utf-8")
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
    collect_parser.add_argument(
        "--localized-readme-dir",
        type=Path,
        help=(
            "directory containing all nine README.<language>.rst files; "
            "defaults to docs/i18n/readme beside the root README"
        ),
    )
    collect_parser.add_argument("--setup", type=Path, default=Path("setup.py"))
    collect_parser.add_argument("--branch", default="main")

    index_parser = subparsers.add_parser(
        "index", help="regenerate the installer archive page from GitHub")
    index_parser.add_argument(
        "--output", type=Path, default=INSTALLER_INDEX_PATH)
    index_parser.add_argument("--setup", type=Path, default=Path("setup.py"))
    index_parser.add_argument(
        "--releases", type=Path,
        help="a recorded GitHub release list, instead of fetching one")

    args = parser.parse_args()
    if args.command == "version":
        print(read_version(args.setup))
    elif args.command == "bump":
        print(bump_version(
            args.setup,
            args.new_version,
            allow_current=args.allow_current,
        ))
    elif args.command == "index":
        import json
        releases = (
            json.loads(args.releases.read_text(encoding="utf-8"))
            if args.releases else None)
        print(write_installer_index(args.output, args.setup, releases))
    else:
        for path in collect_installers(
                args.source, args.destination, args.readme, args.setup,
                args.branch, args.localized_readme_dir):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
