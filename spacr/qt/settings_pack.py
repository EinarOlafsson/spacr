"""Read a downloaded settings pack into settings this spaCR can run.

The demo dataset on HuggingFace ships a settings CSV per stage beside the
images. They were written by an older spaCR, so some of what they name no
longer exists -- keys renamed, keys removed with the feature they
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
* a key that no longer exists anywhere is dropped and named;
* a key that is a setting in this build but not on THIS app's form is
  named separately, because "this version has no such setting" is a false
  sentence about it.

The report is the deliverable as much as the settings are. "Loaded 34
settings, renamed 2, dropped 3 this version no longer has" is a sentence a
user can act on. Thirty-nine settings applied silently, three of which do
nothing, is not.

WHY IT READ NOTHING AT ALL UNTIL 2026-09-14, which is the correction item
317 needed. This module was written as the safe successor to a reader that
merged blind, and then could not replace it, because three things it did
not do are three things the published pack needs:

* IT LOOKED FOR THE WRONG FILE. It opened ``<app>_settings.csv`` only, and
  ``einarolafsson/spacr_settings`` has never shipped a file by that name --
  it ships ``gen_masks_settings.csv`` and ``crop_measure_settings.csv``.
  Measured on the real pack at ``~/datasets/settings`` on 2026-09-14:
  applied 0, renamed 0, dropped 0, malformed 0, on a 58-setting file. A
  miss rather than an error, so the caller filled the form with plain
  defaults and said nothing. `tests/test_hf_e2e_integration.py` found the
  same hole from the other side and worked around it with a local reader
  and a hard assertion on the names, which is where `PACK_FILES` comes
  from.
* IT READ THE HEADER ROW AS A SETTING. The shipped files start ``Key,Value``
  and that arrived as a key named ``Key``, reported to the user as a
  dropped setting. It is not a setting; it is a column heading.
* IT HANDED LISTS OVER AS STRINGS. ``channels,"[0, 1, 2, 3]"`` became the
  eleven-character string, counted as APPLIED, replacing a real list
  default -- which is worse than a drop, because the report says it worked.

Measured again with those three closed, same file, same defaults: 32
applied, 9 renamed, 17 dropped, of which 10 are live settings this form does
not carry. 32 + 9 + 17 = 58, which is every row the reader saw, and the file
has 59 non-blank lines -- the fifty-ninth is the header it now skips. Nothing
is excluded from that count.
"""
from __future__ import annotations

import ast
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

#: What a published pack actually CALLS each stage's file, beyond the
#: ``<app_key>_settings.csv`` spelling that is always tried first.
#:
#: SEVERAL NAMES PER STAGE because the published sets were written by
#: different runs at different times: a mask run saves
#: ``gen_mask_settings.csv``, the older pack shipped ``gen_masks_settings.csv``
#: (plural), and the measure settings have been called both
#: ``measure_crop_settings`` and ``crop_measure_settings``.
#:
#: THIS TABLE IS A COPY AND THE COPY IS ON PURPOSE. The list it mirrors is
#: `spacr.qt.screens.app_screen.AppScreen._EXAMPLE_SETTINGS_FILES`, and
#: importing that here would put a Qt screen -- and everything it imports --
#: on the import path of a module whose whole job is reading a CSV. The two
#: are pinned to each other by
#: `tests/test_a_settings_pack_finds_the_file_the_pack_ships.py` instead, so
#: a name added to one and not the other fails rather than drifting. That is
#: the lesson of 364, one level out: a second table nobody checks is a table
#: that disagrees.
PACK_FILES: Dict[str, Tuple[str, ...]] = {
    "mask": ("gen_mask_settings.csv", "gen_masks_settings.csv"),
    "measure": ("measure_crop_settings.csv", "crop_measure_settings.csv"),
}

#: Header rows a settings CSV may open with, lowercased.
#:
#: Mirrors the column pairs `AppScreen._CSV_COLUMNS` tries, and is pinned to
#: it by the same test. Skipped only in FIRST position: a heading is a
#: heading because of where it is, and a row that says ``key,value`` further
#: down is a setting called ``key`` as far as anything here can tell.
_HEADER_ROWS = frozenset({
    ("key", "value"),
    ("setting_key", "setting_value"),
    ("setting", "value"),
    ("name", "value"),
})


@dataclass
class PackReport:
    """What became of a settings pack, in terms a user can be told."""

    #: Keys applied under their own name.
    applied: List[str] = field(default_factory=list)
    #: ``(old, new)`` for keys applied under a different name.
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    #: Every key that did not land, whatever the reason.
    dropped: List[str] = field(default_factory=list)
    #: The subset of :attr:`dropped` that IS a setting in this build, just
    #: not one this app's form carries. Always a subset, never a fourth
    #: bucket, so a caller that asks "did anything not land?" still gets one
    #: list to ask it of.
    elsewhere: List[str] = field(default_factory=list)
    #: Rows that were not ``key,value`` at all.
    malformed: int = 0
    #: The file the pack was read from, bare name, or ``""`` when the pack
    #: held nothing for this app. That distinction is the one the caller
    #: could not make before: a pack that was never found and a pack that
    #: applied cleanly both produced an empty report.
    source: str = ""

    def summary(self) -> str:
        """One sentence for a status bar, naming what was lost.

        Names the dropped keys rather than counting them: "dropped 3" tells
        a user something went missing without telling them what, which is
        the worst of both.

        TWO SENTENCES FOR TWO DIFFERENT LOSSES. "this version has no such
        setting" was said about `timelapse` and nine others on the real
        mask pack, and it is untrue of every one of them -- they are live
        settings that this particular form does not show. A user told the
        version dropped them goes looking for a version that has them.
        """
        head = f"{len(self.applied) + len(self.renamed)} settings loaded"
        if self.source:
            head += f" from {self.source}"
        parts = [head]
        if self.renamed:
            parts.append("renamed " + ", ".join(
                f"{old} to {new}" for old, new in self.renamed))
        moved = set(self.elsewhere)
        gone = sorted(key for key in self.dropped if key not in moved)
        if gone:
            parts.append("dropped " + ", ".join(gone)
                         + " (this version has no such setting)")
        if self.elsewhere:
            parts.append("ignored " + ", ".join(sorted(self.elsewhere))
                         + " (settings this build has, but not on this form)")
        if self.malformed:
            parts.append(f"{self.malformed} unreadable row(s)")
        return "; ".join(parts) + "."


def _coerce(text: str) -> Any:
    """Turn one CSV field into the type the settings form expects.

    Order matters: ``"true"`` must not become a float, and ``"1"`` must
    become an int rather than staying a string, because a form that is
    handed ``"1"`` for a spin box shows nothing.

    A BRACKETED CELL IS A CONTAINER, and it is the case that used to be got
    wrong quietly. The pack writes ``channels,"[0, 1, 2, 3]"`` and
    ``timelapse_objects,['cell']``; without this the first of those became
    an eleven-character STRING that replaced a real list default and was
    counted as applied -- a wrong value reported as a right one, which is
    the failure this module exists to stop. ``expected_types["channels"]``
    is ``list``.

    SAME RULE AS THE IMPORT PATH, deliberately: ``utils.load_settings``'s
    ``parse_value`` tests ``value.startswith(('(', '[', '{'))`` and tries
    ``ast.literal_eval``, and "Import settings..." goes through it. A pack
    and a hand-imported copy of the same file must not disagree about what
    a cell means.
    """
    value = text.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null", ""):
        return None
    if value[:1] in ("[", "(", "{"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _pack_candidates(app_key: str) -> Tuple[str, ...]:
    """Every file name a pack might hold this app's settings under, best first.

    ``<app_key>_settings.csv`` leads because it is the name this project
    would choose today; the published spellings follow from `PACK_FILES`.
    """
    generic = f"{app_key}_settings.csv"
    return (generic,) + tuple(name for name in PACK_FILES.get(app_key, ())
                              if name != generic)


def _pack_path(app_key: str, pack_dir: str) -> Optional[str]:
    """The file in ``pack_dir`` holding this app's settings, or ``None``."""
    for name in _pack_candidates(app_key):
        path = os.path.join(str(pack_dir), name)
        if os.path.isfile(path):
            return path
    return None


def read_pack(app_key: str, pack_dir: str) -> Tuple[Dict[str, Any], int]:
    """``({key: value}, malformed row count)`` from this app's pack file.

    Missing file is not an error: a pack legitimately carries settings for
    some apps and not others, and the caller gets an empty dict. Use
    :func:`_pack_path` to tell that case apart from a file that was read.
    """
    values: Dict[str, Any] = {}
    malformed = 0
    path = _pack_path(app_key, pack_dir)
    if path is None:
        return values, malformed
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            seen_a_row = False
            for row in csv.reader(handle):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) < 2:
                    malformed += 1
                    seen_a_row = True
                    continue
                if not seen_a_row and (row[0].strip().lower(),
                                       row[1].strip().lower()) in _HEADER_ROWS:
                    seen_a_row = True
                    continue
                seen_a_row = True
                values[row[0].strip()] = _coerce(row[1])
    except Exception:                                       # noqa: BLE001
        LOG.exception("Could not read the settings pack at %s", path)
    return values, malformed


def _package_renames(key: str) -> Tuple[str, ...]:
    """What ``spacr.settings`` says this key became, or nothing.

    A SECOND OPINION, NOT A REPLACEMENT for `PACK_RENAMES`. That table is
    curated per app and can say "this one changed meaning, drop it"; this
    only knows the mechanical renames the package records, which is exactly
    the set a pack written against an older spaCR trips over.

    Returns a single-element tuple for an ordinary rename and an empty one
    for anything it cannot resolve -- including a SPLIT, where one old key
    became several. A split has no safe automatic answer: the value belongs
    to one of the new keys or to none, and guessing puts a number on a
    control that looks deliberate and is wrong.

    THE IMPORT IS INSIDE THE FUNCTION ON PURPOSE. `spacr.settings` is large
    and NO module under `spacr/qt/` imports it at module scope -- pulling it
    in here would put it on the import path of every Qt screen, which is the
    cost 282 and 284 exist to remove. This runs once per unmatched pack key,
    a few dozen times per load at most, not per event. Do not hoist it
    without measuring what it adds to screen construction first.
    """
    try:
        from ..settings import surviving_setting_name
    except Exception:                                       # noqa: BLE001
        return ()
    try:
        names = surviving_setting_name(key)
    except Exception:                                       # noqa: BLE001
        return ()
    if not names or len(names) != 1:
        return ()
    return tuple(names)


def _is_a_setting_somewhere(key: str) -> bool:
    """Whether ``key`` is a live spaCR setting on some other form.

    Answers the difference between "retired" and "not on this screen", which
    is the difference between the two sentences :meth:`PackReport.summary`
    says. Same deferred import, for the same reason, as `_package_renames`.
    """
    try:
        from ..settings import expected_types
    except Exception:                                       # noqa: BLE001
        return False
    try:
        return key in expected_types
    except Exception:                                       # noqa: BLE001
        return False


def settings_from_pack(app_key: str, pack_dir: str, *,
                       src: Optional[str] = None,
                       defaults: Optional[Dict[str, Any]] = None,
                       ) -> Tuple[Dict[str, Any], PackReport]:
    """The app's defaults, with the pack migrated over them.

    :param app_key: the module the pack is for.
    :param pack_dir: folder holding this app's settings CSV, under any of
        the names :func:`_pack_candidates` lists.
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
    found = _pack_path(app_key, pack_dir)
    raw, malformed = read_pack(app_key, pack_dir)
    report = PackReport(malformed=malformed,
                        source=os.path.basename(found) if found else "")

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
        for survivor in _package_renames(key):
            if survivor in settings:
                settings[survivor] = value
                report.renamed.append((key, survivor))
                break
        else:
            report.dropped.append(key)

    if src is not None:
        settings["src"] = str(src)
        if "src" in report.dropped:
            report.dropped.remove("src")
    report.elsewhere = [key for key in report.dropped
                        if _is_a_setting_somewhere(key)]
    if report.dropped:
        LOG.info("settings pack for %s dropped %d key(s): %s",
                 app_key, len(report.dropped), ", ".join(sorted(report.dropped)))
    return settings, report
