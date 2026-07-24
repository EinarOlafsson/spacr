"""
Filename-metadata regex helpers for drag-and-drop and manual configuration.

Central place for four related concerns:

* :func:`apply_regex` — run a compiled regex over a list of filenames,
  return a list of :class:`MetadataRecord` dicts (one per file that
  matched).
* :func:`validate_records` — check whether the parsed records supply
  the fields the rest of spaCR needs (``wellID``/``fieldID`` +
  ``chanID`` for multi-channel; ``fieldID`` for single-channel).
  Returns human-friendly warning strings for anything missing.
* :func:`auto_detect_regex` — heuristic detector that tries each
  built-in regex, and if none fit, synthesises a fresh one from the
  common shape of the sampled filenames.
* :func:`tabulate_records` — render a small aligned text table of
  records for the Console.

Public constants:

* :data:`BUILTIN_REGEXES` — ordered ``{label: pattern}`` map tried by
  :func:`auto_detect_regex` before falling back to synthesis.
* :data:`REQUIRED_MULTICHANNEL` / :data:`REQUIRED_SINGLECHANNEL` —
  which group names must be present for spaCR to be happy.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Built-in regex patterns
# ---------------------------------------------------------------------------

CELLVOYAGER = (
    r"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
    r"L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*)"
    r"\.(?:tif|tiff|png|jpg|jpeg)$"
)

YOKOGAWA = (
    r"(?P<plateID>.*)_(?P<wellID>[A-Z]\d{2})_"
    r"T(?P<timeID>\d{4})F(?P<fieldID>\d{3})"
    r"L(?P<laserID>\d{2})A(?P<AID>\d{2})Z(?P<sliceID>\d{2})C(?P<chanID>\d{2})"
    r"\.(?:tif|tiff)$"
)

# Yokogawa CQ1 naming: W<well>F<field>T<time>Z<slice>C<chan> (well is numeric).
# Matches spacr.utils._get_regex('cq1', ...).
CQ1 = (
    r"W(?P<wellID>.*)F(?P<fieldID>.*)T(?P<timeID>.*)"
    r"Z(?P<sliceID>.*)C(?P<chanID>.*)\.(?:tif|tiff|png|jpg|jpeg)$"
)

# Bare-bones canonical form spaCR generates when auto-normalising
CANONICAL = (
    r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d{2})_"
    r"F(?P<fieldID>\d+)_C(?P<chanID>\d+)\.(?:tif|tiff|png)$"
)
CANONICAL_WITH_TIME = (
    r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d{2})_"
    r"F(?P<fieldID>\d+)_T(?P<timeID>\d+)_C(?P<chanID>\d+)\.(?:tif|tiff|png)$"
)

BUILTIN_REGEXES: Dict[str, str] = {
    "cellvoyager":         CELLVOYAGER,
    "cq1":                 CQ1,
    "yokogawa":            YOKOGAWA,
    "canonical":           CANONICAL,
    "canonical_timelapse": CANONICAL_WITH_TIME,
}


# ---------------------------------------------------------------------------
# Required fields per dataset shape
# ---------------------------------------------------------------------------

#: For a multi-channel plate spaCR needs at least a channel field
#: PLUS either a well or a field id (usually both).
REQUIRED_MULTICHANNEL: Set[str] = {"chanID"}

#: For a single-channel dataset a field id alone is enough.
REQUIRED_SINGLECHANNEL: Set[str] = {"fieldID"}

#: The "location" fields — wellID and fieldID. At least one required
#: for multi-channel data.
LOCATION_FIELDS: Set[str] = {"wellID", "fieldID"}

#: Every field name spaCR downstream code understands.
KNOWN_FIELDS: Tuple[str, ...] = (
    "plateID", "wellID", "fieldID", "chanID",
    "timeID", "sliceID", "laserID", "AID",
)


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------

@dataclass
class MetadataRecord:
    """One parsed filename.

    :ivar filename: bare filename (no path).
    :ivar groups: mapping of regex group name → matched text.
    """
    filename: str
    groups:   Dict[str, str]

    def get(self, name: str, default: str = "") -> str:
        return self.groups.get(name, default)


# ---------------------------------------------------------------------------
# apply_regex
# ---------------------------------------------------------------------------

def apply_regex(
    filenames: Sequence[str],
    pattern: str,
) -> Tuple[List[MetadataRecord], List[str]]:
    """Run ``pattern`` over each filename and return matched records.

    :param filenames: bare filenames (no directory).
    :param pattern: regex string; anchored with ``re.match`` semantics.
    :returns: ``(records, non_matching_filenames)``.
    """
    try:
        rx = re.compile(pattern)
    except re.error:
        return [], list(filenames)
    records: List[MetadataRecord] = []
    missed:  List[str] = []
    for name in filenames:
        m = rx.match(name)
        if m is None:
            missed.append(name)
            continue
        records.append(MetadataRecord(name, dict(m.groupdict())))
    return records, missed


# ---------------------------------------------------------------------------
# validate_records
# ---------------------------------------------------------------------------

def validate_records(
    records: Sequence[MetadataRecord],
    multichannel: bool = True,
) -> List[str]:
    """Check parsed records against spaCR's downstream requirements.

    :param records: output of :func:`apply_regex`.
    :param multichannel: True if the dataset has more than one channel;
        False for single-channel data (relaxes the requirement set).
    :returns: list of warning strings — empty when everything is fine.
    """
    if not records:
        return ["No filenames matched the regex."]

    warnings: List[str] = []
    all_group_names: Set[str] = set()
    for r in records:
        for k, v in r.groups.items():
            if v:
                all_group_names.add(k)

    if multichannel:
        # Need channel id
        if "chanID" not in all_group_names:
            warnings.append(
                "Missing required field: chanID (multi-channel data "
                "needs the regex to capture the channel number)."
            )
        # Need at least one location field
        if not (all_group_names & LOCATION_FIELDS):
            warnings.append(
                "Missing location field: at least one of wellID or "
                "fieldID must be captured to map each image to a "
                "well / field."
            )
    else:
        if "fieldID" not in all_group_names:
            warnings.append(
                "Missing required field: fieldID (single-channel data "
                "still needs a field id to distinguish images)."
            )
    # plateID is optional but VERY handy — soft warn
    if "plateID" not in all_group_names:
        warnings.append(
            "Optional: no plateID captured. spaCR will name the "
            "generated stack `plate1` by default; edit the regex to "
            "capture a plate id if you have one."
        )
    return warnings


# ---------------------------------------------------------------------------
# auto_detect_regex
# ---------------------------------------------------------------------------

def auto_detect_regex(
    filenames: Sequence[str],
) -> Tuple[Optional[str], str, int]:
    """Return the best-fitting regex for a set of filenames.

    Strategy:
    1. Try every :data:`BUILTIN_REGEXES` pattern; if one matches every
       file it wins immediately.
    2. Otherwise pick the built-in that matches the MOST files (>=50 %).
    3. If nothing crosses the 50 % bar, synthesise a fresh regex from
       the common shape of the sample (see :func:`_synthesise_regex`).

    :param filenames: sample filenames to fit against.
    :returns: ``(pattern_or_None, label, n_matches)``.
        ``pattern_or_None`` is None only when synthesis also fails.
    """
    n = len(filenames)
    if n == 0:
        return None, "empty", 0

    # 1 + 2: try built-ins, remember the best
    best_label = "none"
    best_pattern: Optional[str] = None
    best_hits = -1
    for label, pattern in BUILTIN_REGEXES.items():
        try:
            rx = re.compile(pattern)
        except re.error:
            continue
        hits = sum(1 for f in filenames if rx.match(f))
        if hits == n:
            return pattern, label, n
        if hits > best_hits:
            best_hits = hits
            best_label = label
            best_pattern = pattern

    if best_hits >= n / 2:
        return best_pattern, best_label, best_hits

    # 3: synthesise
    synth = _synthesise_regex(filenames)
    if synth is None:
        return best_pattern, best_label, best_hits
    return synth, "synthesised", n


def _synthesise_regex(filenames: Sequence[str]) -> Optional[str]:
    """Best-effort: build a regex from the common shape of filenames.

    Recognises Illumina/Yokogawa-style tokens:
        <letter><digits>   → single-letter prefix + digits
                              (F00013, C02, Z01, T0001, etc.)
        <letter>\d{2}      → likely wellID (A01, B12, ...).
        [A-Z]\d{3}         → also wellID plus a big serial.
        [_-]               → literal separators.

    Strategy: pick ONE filename as a template, walk char-by-char, and
    replace every digit run with ``\d+``, every letter-prefix + digits
    combo with a named group when the prefix maps to a known tag.
    """
    if not filenames:
        return None
    # Take the "shortest, alphanumeric-only" as template — least
    # likely to have noise like an underscore-suffixed acquisition tag.
    template = min(filenames, key=lambda s: (len(s), s))

    # Map single-char prefixes to standard field names
    prefix_map = {
        "F": "fieldID",
        "T": "timeID",
        "C": "chanID",
        "Z": "sliceID",
        "L": "laserID",
        "A": "AID",
    }
    parts: List[str] = []
    i = 0
    used_groups: Set[str] = set()
    stem, dot, suffix = template.rpartition(".")
    if not dot:
        return None
    tokens = re.split(r"([_-])", stem)
    for tok in tokens:
        if tok in ("_", "-"):
            parts.append(re.escape(tok))
            continue
        # Well id shape: single letter + 2-3 digits, or two-letter + digits
        wm = re.fullmatch(r"([A-Za-z])(\d{2,3})", tok)
        if wm and "wellID" not in used_groups:
            parts.append(r"(?P<wellID>[A-Z]\d{2,3})")
            used_groups.add("wellID")
            continue
        # Single-letter prefix + digits combos (F001, T0001, ...)
        m = re.fullmatch(r"([A-Za-z])(\d+)", tok)
        if m and m.group(1).upper() in prefix_map:
            gname = prefix_map[m.group(1).upper()]
            if gname not in used_groups:
                parts.append(f"{re.escape(m.group(1))}(?P<{gname}>\\d+)")
                used_groups.add(gname)
                continue
        # Multi-prefix runs like "T0001F001L01A01Z01C01"
        multi = re.findall(r"([A-Za-z])(\d+)", tok)
        if multi and all(mp[0].upper() in prefix_map for mp in multi):
            for letter, digits in multi:
                gname = prefix_map[letter.upper()]
                if gname in used_groups:
                    parts.append(f"{re.escape(letter)}\\d+")
                else:
                    parts.append(f"{re.escape(letter)}(?P<{gname}>\\d+)")
                    used_groups.add(gname)
            continue
        # Single letter + digits with an unrecognised prefix (e.g.
        # `W1`, `M12`) — keep the letter literal, let the digits vary
        # so plates that use W2, W3, … still match. Otherwise we'd
        # bake the exact digits from the template into the regex and
        # only match one file.
        if m:
            parts.append(f"{re.escape(m.group(1))}\\d+")
            continue
        # Pure identifier → plateID (only first free identifier)
        if re.fullmatch(r"[A-Za-z0-9]+", tok) and "plateID" not in used_groups:
            parts.append(r"(?P<plateID>[A-Za-z0-9]+)")
            used_groups.add("plateID")
            continue
        # Fallback: literal escaped shape
        parts.append(re.escape(tok))
    # Suffix: allow any of the common image extensions
    exts = r"(?:tif|tiff|png|jpg|jpeg)$"
    return "".join(parts) + r"\." + exts


# ---------------------------------------------------------------------------
# tabulate_records — plain-text table for the Console
# ---------------------------------------------------------------------------

def tabulate_records(
    records: Sequence[MetadataRecord],
    columns: Optional[Sequence[str]] = None,
    max_rows: int = 10,
    random_sample: bool = True,
    seed: int = 42,
) -> str:
    """Render a small aligned-column table of records for the Console.

    :param records: list of parsed records.
    :param columns: which group names to include; auto-inferred from
        the first record when None.
    :param max_rows: cap on rows; sampled at random if exceeded.
    :param random_sample: True → pick ``max_rows`` at random when the
        list is longer; False → take the first ``max_rows``.
    :param seed: RNG seed so the sample is reproducible between runs.
    :returns: multi-line string ready to feed to
        :py:meth:`ConsolePanel.append_stdout`.
    """
    if not records:
        return "(no records — regex did not match any files)"
    if columns is None:
        # Deterministic order — use KNOWN_FIELDS then any extras
        found: Set[str] = set()
        for r in records:
            found.update(r.groups.keys())
        columns = [f for f in KNOWN_FIELDS if f in found]
        columns += sorted(found - set(KNOWN_FIELDS))
    columns = list(columns) + ["filename"]

    if len(records) > max_rows:
        if random_sample:
            rng = random.Random(seed)
            sample = rng.sample(list(records), max_rows)
        else:
            sample = list(records[:max_rows])
    else:
        sample = list(records)

    # Compute per-column widths
    widths = {c: max(len(c), max(
        (len(_render_cell(r, c)) for r in sample), default=len(c)
    )) for c in columns}
    def _row(vals: Sequence[str]) -> str:
        return "  " + "  ".join(v.ljust(widths[c])
                                  for c, v in zip(columns, vals))

    header = _row(columns)
    rule = "  " + "  ".join("-" * widths[c] for c in columns)
    body = [_row([_render_cell(r, c) for c in columns]) for r in sample]
    return "\n".join([header, rule, *body])


def _render_cell(r: MetadataRecord, col: str) -> str:
    if col == "filename":
        return r.filename
    return r.get(col, "—")
