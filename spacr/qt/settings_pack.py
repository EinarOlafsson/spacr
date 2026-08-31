"""Read a downloaded settings pack into settings this spaCR can run.

The demo dataset on HuggingFace ships ``<app>_settings.csv`` files beside
the images. They were written by an older spaCR, so some of what they name
no longer exists -- keys renamed, keys removed with the feature they
configured, keys that were never settings at all.

MERGING THEM BLIND IS THE BUG THIS MODULE EXISTS TO FIX. The previous
loader read every row of the CSV straight over the defaults, so a key the
current build has never heard of arrived in the settings dict and travelled
into the pipeline, where it was either ignored silently or produced an
error naming a setting the user did not type and cannot find in the form.
Either way the user is told nothing at the point where it could be
explained.

So this MIGRATES rather than merges, and REPORTS rather than guesses:

* a key the app still has is applied;
* a key that has been renamed is applied under its new name and counted;
* a key that no longer exists anywhere is dropped and named.

The report is the deliverable as much as the settings are. "Loaded 34
settings, renamed 2, dropped 3 this version no longer has" is a sentence a
user can act on. Thirty-nine settings applied silently, three of which do
nothing, is not.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("spacr.qt.settings_pack")

#: Keys that MOVED, per app: ``{app_key: {old name: new name}}``.
#:
#: Deliberately small and deliberately explicit. A rename belongs here only
#: when the old key and the new one mean the SAME THING -- if the semantics
#: changed with the name, carrying the old value across is worse than
#: dropping it, because the user gets a setting that looks deliberate and
#: is wrong.
#:
#: Empty for an app means "nothing has been renamed that we know of", which
#: is not the same as "nothing has ever been renamed". A key that turns out
#: to have moved is added here with a note, and the note is what stops the
#: next person re-deciding it.
PACK_RENAMES: Dict[str, Dict[str, str]] = {
    "mask": {},
    "measure": {},
}


@dataclass
class PackReport:
    """What became of a settings pack, in terms a user can be told."""

    #: Keys applied under their own name.
    applied: List[str] = field(default_factory=list)
    #: ``(old, new)`` for keys applied under a different name.
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    #: Keys this build has no setting for, dropped.
    dropped: List[str] = field(default_factory=list)
    #: Rows that were not ``key,value`` at all.
    malformed: int = 0

    def summary(self) -> str:
        """One sentence for a status bar, naming what was lost.

        Names the dropped keys rather than counting them: "dropped 3" tells
        a user something went missing without telling them what, which is
        the worst of both.
        """
        parts = [f"{len(self.applied) + len(self.renamed)} settings loaded"]
        if self.renamed:
            parts.append("renamed " + ", ".join(
                f"{old} to {new}" for old, new in self.renamed))
        if self.dropped:
            parts.append("dropped " + ", ".join(sorted(self.dropped))
                         + " (this version has no such setting)")
        if self.malformed:
            parts.append(f"{self.malformed} unreadable row(s)")
        return "; ".join(parts) + "."


def _coerce(text: str) -> Any:
    """Turn one CSV field into the type the settings form expects.

    Order matters: ``"true"`` must not become a float, and ``"1"`` must
    become an int rather than staying a string, because a form that is
    handed ``"1"`` for a spin box shows nothing.
    """
    value = text.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_pack(app_key: str, pack_dir: str) -> Tuple[Dict[str, Any], int]:
    """``({key: value}, malformed row count)`` from ``<app>_settings.csv``.

    Missing file is not an error: a pack legitimately carries settings for
    some apps and not others, and the caller gets an empty dict.
    """
    path = os.path.join(str(pack_dir), f"{app_key}_settings.csv")
    values: Dict[str, Any] = {}
    malformed = 0
    if not os.path.isfile(path):
        return values, malformed
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) < 2:
                    malformed += 1
                    continue
                values[row[0].strip()] = _coerce(row[1])
    except Exception:                                       # noqa: BLE001
        LOG.exception("Could not read the settings pack at %s", path)
    return values, malformed


def settings_from_pack(app_key: str, pack_dir: str, *,
                       src: Optional[str] = None,
                       defaults: Optional[Dict[str, Any]] = None,
                       ) -> Tuple[Dict[str, Any], PackReport]:
    """The app's defaults, with the pack migrated over them.

    :param app_key: the module the pack is for.
    :param pack_dir: folder holding ``<app_key>_settings.csv``.
    :param src: dataset folder; overrides whatever ``src`` the pack names,
        because a pack written on somebody else's machine names a path
        that does not exist on this one.
    :param defaults: the app's defaults; resolved from the registry when
        omitted.
    :returns: the settings to apply, and what became of the pack.
    """
    if defaults is None:
        from .screens.settings_model import resolve_default_settings
        defaults = dict(resolve_default_settings(app_key))
    settings = dict(defaults)
    renames = PACK_RENAMES.get(app_key, {})
    raw, malformed = read_pack(app_key, pack_dir)
    report = PackReport(malformed=malformed)

    for key, value in raw.items():
        if key in settings:
            settings[key] = value
            report.applied.append(key)
            continue
        moved = renames.get(key)
        if moved and moved in settings:
            settings[moved] = value
            report.renamed.append((key, moved))
            continue
        report.dropped.append(key)

    if src is not None:
        # LAST, and unconditionally. The pack's own `src` is a path on the
        # machine that produced it; applying it would point the run at a
        # folder that is not there, which fails much later and blames the
        # dataset rather than the pack.
        settings["src"] = str(src)
        if "src" in report.dropped:
            report.dropped.remove("src")
    if report.dropped:
        LOG.info("settings pack for %s dropped %d key(s): %s",
                 app_key, len(report.dropped), ", ".join(sorted(report.dropped)))
    return settings, report
