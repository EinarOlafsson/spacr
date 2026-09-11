"""Two files whose names differ only in case make an uninstallable wheel.

MEASURED, 2026-09-10, on a published release. spaCR 1.5.0.5 shipped both
``cell_Signal_to_noise.gif`` and ``cell_signal_to_noise.gif`` -- and the same
pair for nucleus and pathogen. On Linux those are three separate files and
nothing complains. On macOS and Windows the filesystem is case-insensitive,
so the installer is asked to write two different payloads to one path, and
uv refuses the whole archive:

    Failed to extract archive: spacr-1.5.0.5-py3-none-any.whl
    ZIP file contains multiple entries with different contents for:
    spacr/resources/setting_animations/gifs/cell_signal_to_noise.gif

The release was therefore installable on Linux and NOT INSTALLABLE ON MAC OR
WINDOWS, and every check in the release pipeline passed, because every one of
them runs on Linux. The packaging tests, the sdist build, the wheel build,
`twine check` and the Linux installer's own smoke test are all blind to it by
construction.

HOW IT HAPPENED, because the shape recurs: the animation generator was
re-run on 2026-09-02 and emits lowercase slugs. The July files with a capital
S were left behind, orphaned -- the manifest names the lowercase set nine
times and the capital set not at all. On a case-sensitive checkout an orphan
like that is invisible; it is not a stale duplicate, it is simply another
file.

This test is cheap and it is the only thing standing between that and the
next release.
"""
from __future__ import annotations

import collections
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    """Every tracked path, which is what the sdist and wheel are built from."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def test_no_two_tracked_paths_differ_only_in_case():
    """The whole repository, not just the package.

    Checked over everything git tracks rather than over ``spacr/`` alone: an
    sdist carries tests, docs and packaging too, and a collision anywhere in
    it fails the same extraction on the same filesystems.
    """
    by_lower: dict[str, list[str]] = collections.defaultdict(list)
    for path in _tracked_files():
        by_lower[path.lower()].append(path)
    collisions = {low: names for low, names in by_lower.items()
                  if len(names) > 1}
    assert not collisions, (
        "these tracked paths differ only in case, which makes a wheel that "
        "cannot be extracted on macOS or Windows:\n"
        + "\n".join(f"  {low}\n" + "".join(f"    {n}\n" for n in names)
                    for low, names in sorted(collisions.items())))


def test_the_setting_animation_gifs_are_the_ones_the_manifest_names():
    """The orphan that caused it would have been caught here too.

    The manifest is the index the application reads; a GIF beside it that the
    manifest never names is not a spare, it is weight in every wheel and a
    case collision waiting for a filesystem that folds case.
    """
    import json

    root = REPO_ROOT / "spacr" / "resources" / "setting_animations"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    named = set()
    for entry in manifest.get("animations", []):
        if not isinstance(entry, dict):
            continue
        # `file` rather than `slug`: the two agree today, and `file` is the
        # field the application actually resolves.
        name = entry.get("file") or (f"{entry['slug']}.gif"
                                     if entry.get("slug") else None)
        if name:
            named.add(pathlib.Path(name).name)
    if not named:                      # schema moved; the other test still holds
        return
    on_disk = {p.name for p in (root / "gifs").glob("*.gif")}
    orphans = sorted(on_disk - named)
    assert not orphans, (
        "these GIFs are shipped but the manifest never names them: "
        + ", ".join(orphans))
