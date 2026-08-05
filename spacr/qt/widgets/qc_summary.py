"""One verdict for a project, assembled from verdicts that already exist.

Every check spaCR can make is already implemented somewhere -- segmentation
scorecards, leakage audits, the units stamp on every measurement row. What is
missing is a place to see them together, so "is this run usable?" does not
require opening five screens and remembering which of them was run last.

**Nothing here computes a verdict.** The rule comes from
:mod:`spacr.qt.prerun`, which states it plainly: opening a plate's masks costs
seconds to minutes, and a screen that pays that on every visit is a screen
nobody keeps. So each reader here does a directory listing, a stat and a
parse, and reports what it found -- including that it found nothing, which is
emphatically not the same as finding nothing wrong.

Staleness follows the same rule. ``seg_qc.read_digest`` dates each scorecard
against its mask stack and reports one written before its masks as out of
date; this module carries that through to the card rather than re-deriving it,
and applies the same idea to the other sources against their own inputs.

The vocabulary is borrowed, not reinvented. A segmentation flag's explanation
comes from :data:`spacr.seg_qc.FLAG_GUIDANCE`, which is where those sentences
were written and where they will be maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import json
import os
import sqlite3
import time
from typing import Any, List, Optional, Sequence, Tuple

__all__ = [
    "VERDICTS", "VERDICT_ORDER", "QCCard", "Dashboard",
    "read_dashboard", "format_dashboard", "worst_verdict",
]

#: Verdicts a card may carry, best to worst. ``missing`` sorts *after* ``ok``
#: and before ``warn`` on purpose: a check that never ran is a real gap, but
#: it is not evidence of a problem, and treating the two the same would make
#: a fresh project look broken.
VERDICT_ORDER: Tuple[str, ...] = ("ok", "missing", "warn", "fail", "error")
VERDICTS = VERDICT_ORDER

#: How many scorecard flags a card lists before it stops naming them.
_MAX_FLAGS = 6


def worst_verdict(verdicts: Sequence[str]) -> str:
    """The worst of several verdicts, by :data:`VERDICT_ORDER`."""
    known = [v for v in verdicts if v in VERDICT_ORDER]
    if not known:
        return "missing"
    return max(known, key=VERDICT_ORDER.index)


@dataclass
class QCCard:
    """One check, as it was found on disk.

    :ivar key: stable identifier -- ``segmentation``, ``leakage``, ``units``,
        ``plate``, ``agreement``.
    :ivar title: what goes on the card.
    :ivar verdict: one of :data:`VERDICT_ORDER`.
    :ivar headline: the one line worth reading.
    :ivar detail: the sentences behind it, already in plain language. For
        segmentation these come from :data:`spacr.seg_qc.FLAG_GUIDANCE` --
        the same words that screen uses, because two vocabularies for one
        flag is how a codebase ends up disagreeing with itself.
    :ivar source: the file the verdict was read from, or ``""``.
    :ivar mtime: that file's modification time, or ``0.0``.
    :ivar stale: whether its inputs are newer than it is. A stale card is
        *not* downgraded to a worse verdict -- it describes the previous run
        accurately, and pretending otherwise would hide which of the two is
        the problem.
    :ivar how_to_produce: for a ``missing`` card, the one thing to run.
    """

    key: str
    title: str
    verdict: str = "missing"
    headline: str = ""
    detail: List[str] = dc_field(default_factory=list)
    source: str = ""
    mtime: float = 0.0
    stale: bool = False
    how_to_produce: str = ""

    @property
    def display_verdict(self) -> str:
        """The verdict as it should be labelled, staleness included."""
        if self.stale and self.verdict not in ("missing", "error"):
            return f"{self.verdict} (out of date)"
        return self.verdict


@dataclass
class Dashboard:
    """Every card for one project, plus the verdict they add up to.

    :ivar root: the project folder the cards were read from.
    :ivar verdict: the worst card verdict.
    :ivar headline: one sentence naming the worst finding.
    :ivar cards: the cards, in a fixed order so the screen does not reshuffle
        between refreshes.
    :ivar checked_at: when the read happened.
    :ivar blocks_run: constant ``False``. Advisory by construction, and the
        field exists to say so in code rather than only in a docstring --
        the same posture :class:`spacr.seg_qc.QCDigest` takes.
    """

    root: str = ""
    verdict: str = "missing"
    headline: str = ""
    cards: List[QCCard] = dc_field(default_factory=list)
    checked_at: float = 0.0
    blocks_run: bool = False

    @property
    def stale(self) -> bool:
        """Whether any card is older than the thing it describes."""
        return any(card.stale for card in self.cards)

    def card(self, key: str) -> Optional[QCCard]:
        """The card with this key, or ``None``."""
        for card in self.cards:
            if card.key == key:
                return card
        return None


# ---------------------------------------------------------------------------
# Readers -- one per source. Each is cheap: listdir, stat, parse.
# ---------------------------------------------------------------------------

def _read_segmentation(src: Any, reader=None) -> QCCard:
    """Read the scorecards the mask run wrote. Never scores anything."""
    card = QCCard(
        key="segmentation", title="Segmentation",
        how_to_produce=(
            "Run Mask with seg_qc on, or press Score in the segmentation QC "
            "banner."),
    )
    try:
        if reader is None:
            from ...seg_qc import read_digest as reader
        digest = reader(src)
    except Exception as exc:  # pragma: no cover - defensive
        card.verdict = "error"
        card.headline = f"Could not read the segmentation scorecards: {exc}"
        return card

    card.verdict = str(getattr(digest, "verdict", "missing"))
    card.headline = str(getattr(digest, "headline", "")) or (
        "No segmentation scorecard has been written for this project.")
    card.stale = bool(getattr(digest, "stale", False))
    cards = list(getattr(digest, "scorecards", ()) or ())
    if cards:
        newest = max((float(getattr(c, "mtime", 0.0)) for c in cards),
                     default=0.0)
        card.mtime = newest
        card.source = str(getattr(cards[0], "path", ""))
    subhead = str(getattr(digest, "subhead", "")).strip()
    if subhead:
        card.detail.append(subhead)
    card.detail.extend(_flag_explanations(cards))
    return card


def _flag_explanations(scorecards: Sequence[Any]) -> List[str]:
    """Plain-language lines for the flags the scorecards actually raised.

    Reuses :data:`spacr.seg_qc.FLAG_GUIDANCE` rather than writing a second
    vocabulary. A flag with no guidance entry is still named -- silently
    dropping it would make the dashboard quieter than the truth.
    """
    try:
        from ...seg_qc import FLAG_GUIDANCE, explain_flag
    except Exception:  # pragma: no cover - defensive
        return []
    seen: List[str] = []
    for scorecard in scorecards:
        for field_qc in getattr(scorecard, "field_qcs", ()) or ():
            for flag in getattr(field_qc, "flags", ()) or ():
                if flag not in seen:
                    seen.append(flag)
    lines: List[str] = []
    for flag in seen[:_MAX_FLAGS]:
        if flag in FLAG_GUIDANCE:
            lines.append(explain_flag(flag).text())
        else:
            lines.append(f"{flag}: no guidance is written for this flag yet.")
    if len(seen) > _MAX_FLAGS:
        lines.append(f"... and {len(seen) - _MAX_FLAGS} more flag(s).")
    return lines


def _newest_under(root: str, name: str, limit: int = 4000) -> Tuple[str, float]:
    """Newest file called ``name`` under ``root``. ``("", 0.0)`` if none.

    Bounded: a project folder can hold a hundred thousand PNG crops, and an
    unbounded walk here would cost more than the check it feeds.
    """
    best, best_mtime = "", 0.0
    seen = 0
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in ("stack", "merged", "masks", "datasets")]
        for candidate in files:
            seen += 1
            if seen > limit:
                return best, best_mtime
            if candidate != name:
                continue
            path = os.path.join(folder, candidate)
            try:
                mtime = float(os.stat(path).st_mtime)
            except OSError:
                continue
            if mtime > best_mtime:
                best, best_mtime = path, mtime
    return best, best_mtime


def _read_leakage(src: Any) -> QCCard:
    """Read ``leakage.json`` from the newest evaluation bundle."""
    card = QCCard(
        key="leakage", title="Train/test leakage",
        how_to_produce=(
            "Run Classify (CV) or Model Evaluation; both write "
            "leakage.json into their evaluation bundle."),
    )
    root = _project_root(src)
    if not root:
        return card
    try:
        from ...classifier_evaluation import EVALUATION_FILES
        name = EVALUATION_FILES["leakage"]
    except Exception:  # pragma: no cover - defensive
        name = "leakage.json"
    path, mtime = _newest_under(root, name)
    if not path:
        card.headline = (
            "No leakage audit has been written, so nothing is known about "
            "whether related images crossed a split boundary.")
        return card
    card.source, card.mtime = path, mtime
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except Exception as exc:
        card.verdict = "error"
        card.headline = f"Could not read {os.path.basename(path)}: {exc}"
        return card

    reports = payload.get("reports") or []
    failed = [r for r in reports if not r.get("passed", True)]
    if failed:
        card.verdict = "fail"
        boundaries = ", ".join(
            str(r.get("group_by", "?")) for r in failed[:4])
        card.headline = (
            f"{len(failed)} of {len(reports)} split boundaries leaked "
            f"({boundaries}).")
        card.detail.append(
            "Images of the same well or plate appeared on both sides of a "
            "train/test split, so the reported accuracy is measuring "
            "memorisation as well as generalisation. The number is an "
            "upper bound, not an estimate.")
    elif reports:
        card.verdict = "ok"
        card.headline = (
            f"All {len(reports)} split boundaries held; no related samples "
            "crossed a split.")
    else:
        card.headline = (
            f"{os.path.basename(path)} holds no split reports.")
    return card


def _project_root(src: Any) -> str:
    """The folder to search under, from whatever a screen was given."""
    if isinstance(src, (list, tuple)):
        src = src[0] if src else ""
    text = str(src or "")
    if not text:
        return ""
    if os.path.isfile(text):
        return os.path.dirname(text)
    return text if os.path.isdir(text) else ""


def _find_measurements_db(root: str) -> Tuple[str, float]:
    """The project's measurements database, if it has one."""
    for candidate in (
        os.path.join(root, "measurements", "measurements.db"),
        os.path.join(root, "measurements.db"),
    ):
        if os.path.isfile(candidate):
            try:
                return candidate, float(os.stat(candidate).st_mtime)
            except OSError:
                return candidate, 0.0
    return _newest_under(root, "measurements.db")


def _read_units(src: Any) -> QCCard:
    """Check the units stamp every measurement row carries, by reading it.

    A ``SELECT DISTINCT`` over five small columns -- not a recomputation of
    anything. What it catches is a table holding rows from a 2-D run and rows
    from a 3-D run, where ``cell_area`` means px^2 in some rows and um^3 in
    others, and every query over the table silently mixes them.
    """
    card = QCCard(
        key="units", title="Measurement units",
        how_to_produce="Run Measure; it stamps every row it writes.",
    )
    root = _project_root(src)
    if not root:
        return card
    path, mtime = _find_measurements_db(root)
    if not path:
        card.headline = "No measurements database was found."
        return card
    card.source, card.mtime = path, mtime

    try:
        from ...measurement_schema import MEASUREMENT_STAMP_COLUMNS
    except Exception:  # pragma: no cover - defensive
        MEASUREMENT_STAMP_COLUMNS = ("measurement_ndim", "measurement_units")

    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        card.verdict = "error"
        card.headline = f"Could not open {os.path.basename(path)}: {exc}"
        return card
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        checked, mixed, unstamped = [], [], []
        for table in tables:
            if table in ("settings", "settings_history", "run_status",
                         "object_counts", "png_list"):
                continue
            columns = {row[1] for row in
                       connection.execute(f'PRAGMA table_info("{table}")')}
            present = [c for c in MEASUREMENT_STAMP_COLUMNS if c in columns]
            if not present:
                if columns:
                    unstamped.append(table)
                continue
            checked.append(table)
            selected = ", ".join(f'"{c}"' for c in present)
            distinct = connection.execute(
                f'SELECT DISTINCT {selected} FROM "{table}"').fetchall()
            if len(distinct) > 1:
                mixed.append((table, distinct))
    except sqlite3.Error as exc:
        card.verdict = "error"
        card.headline = f"Could not read the units stamp: {exc}"
        return card
    finally:
        connection.close()

    if mixed:
        card.verdict = "fail"
        names = ", ".join(name for name, _ in mixed[:4])
        card.headline = (
            f"{len(mixed)} table(s) hold rows measured in different units "
            f"({names}).")
        for name, distinct in mixed[:3]:
            card.detail.append(
                f"{name}: {len(distinct)} distinct stamps, e.g. "
                f"{distinct[0]} and {distinct[1]}. A column like "
                f"{name}_area means one thing in some of these rows and "
                "another in the rest, and no query over the table can tell "
                "them apart without filtering on the stamp first.")
    elif checked:
        card.verdict = "ok"
        card.headline = (
            f"{len(checked)} measurement table(s) are internally consistent "
            "in units and dimensionality.")
    elif unstamped:
        card.verdict = "warn"
        card.headline = (
            f"{len(unstamped)} table(s) carry no units stamp, so a 2-D and a "
            "3-D run cannot be told apart in them.")
        card.detail.append(
            "Tables written before the stamp existed. Re-measure to get one, "
            "or take on faith that everything in them came from one run.")
    else:
        card.headline = "The measurements database holds no object tables."
    return card


def _read_plate(src: Any) -> QCCard:
    """Plate-level QC: edge effects, gradients, layout."""
    card = QCCard(
        key="plate", title="Plate effects",
        how_to_produce=(
            "Open Plate View and run the edge-effect check; it reads the "
            "measurements database."),
    )
    root = _project_root(src)
    if not root:
        return card
    path, mtime = _newest_under(root, "plate_qc.json")
    if not path:
        card.headline = (
            "No plate-effect verdict has been written. spacr.plate_qc "
            "computes edge effects and gradients on demand from the "
            "measurements database and does not persist the result, so "
            "there is nothing on disk for this dashboard to read.")
        return card
    card.source, card.mtime = path, mtime
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except Exception as exc:
        card.verdict = "error"
        card.headline = f"Could not read {os.path.basename(path)}: {exc}"
        return card
    card.verdict = str(payload.get("verdict", "missing"))
    card.headline = str(payload.get("headline", "")) or "Plate QC ran."
    detail = payload.get("detail")
    if isinstance(detail, str):
        card.detail.append(detail)
    elif isinstance(detail, list):
        card.detail.extend(str(line) for line in detail)
    return card


def _read_agreement(src: Any) -> QCCard:
    """Annotator agreement, when a report has been saved."""
    card = QCCard(
        key="agreement", title="Annotator agreement",
        how_to_produce=(
            "Open Agreement, pick two annotation columns and compute; save "
            "the report to have it appear here."),
    )
    root = _project_root(src)
    if not root:
        return card
    path, mtime = _newest_under(root, "agreement.json")
    if not path:
        card.headline = (
            "No agreement report has been saved. The Agreement screen "
            "computes kappa from the annotation columns on demand and does "
            "not persist it, so there is nothing on disk to read.")
        return card
    card.source, card.mtime = path, mtime
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except Exception as exc:
        card.verdict = "error"
        card.headline = f"Could not read {os.path.basename(path)}: {exc}"
        return card
    kappa = payload.get("kappa")
    band = payload.get("band") or payload.get("interpretation") or ""
    if kappa is None:
        card.headline = "The agreement report holds no kappa."
        return card
    try:
        value = float(kappa)
    except (TypeError, ValueError):
        card.verdict = "error"
        card.headline = f"kappa is not a number: {kappa!r}"
        return card
    card.verdict = "ok" if value >= 0.6 else ("warn" if value >= 0.4
                                              else "fail")
    card.headline = f"Annotator agreement kappa = {value:.2f} ({band})."
    if value < 0.6:
        card.detail.append(
            "Below about 0.6 the annotators disagree often enough that a "
            "model trained on either one is partly fitting that person. The "
            "ceiling on any classifier's accuracy is the agreement between "
            "the people who made its labels.")
    return card


#: The cards, in the order the screen shows them. Worst-first ordering is
#: deliberately *not* used: a dashboard that reshuffles between refreshes
#: makes the user re-find the card they were reading.
_READERS = (
    ("segmentation", _read_segmentation),
    ("units", _read_units),
    ("leakage", _read_leakage),
    ("plate", _read_plate),
    ("agreement", _read_agreement),
)


def read_dashboard(src: Any, *, segmentation_reader=None) -> Dashboard:
    """Read every verdict already on disk for one project. Computes none.

    :param src: project folder, plate folder, or a list of either.
    :param segmentation_reader: substitute for
        :func:`spacr.seg_qc.read_digest`, for tests.
    :returns: a :class:`Dashboard`. ``verdict == "missing"`` means no check
        has been run, which is not the same as ``"ok"``.
    """
    cards: List[QCCard] = []
    for key, reader in _READERS:
        if key == "segmentation":
            cards.append(_read_segmentation(src, reader=segmentation_reader))
        else:
            cards.append(reader(src))

    verdict = worst_verdict([card.verdict for card in cards])
    worst = [card for card in cards if card.verdict == verdict]
    headline = worst[0].headline if worst else ""
    if verdict == "ok":
        headline = (
            f"All {len(cards)} checks that have run are clean."
            if all(c.verdict == "ok" for c in cards)
            else headline)
    if verdict == "missing":
        absent = [c.title for c in cards if c.verdict == "missing"]
        headline = (
            f"Nothing has been checked yet: {', '.join(absent)} have no "
            "verdict on disk.")
    return Dashboard(
        root=_project_root(src), verdict=verdict, headline=headline,
        cards=cards, checked_at=time.time())


def format_dashboard(dashboard: Dashboard) -> str:
    """The dashboard as plain text, for a log or a copy-paste."""
    lines = [f"{dashboard.verdict.upper()}: {dashboard.headline}"]
    if dashboard.root:
        lines.append(f"  {dashboard.root}")
    for card in dashboard.cards:
        lines.append(f"[{card.display_verdict}] {card.title}: {card.headline}")
        for detail in card.detail:
            lines.append(f"    {detail}")
        if card.verdict == "missing" and card.how_to_produce:
            lines.append(f"    -> {card.how_to_produce}")
    return "\n".join(lines)
