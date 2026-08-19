"""What a failed regression says, instead of "failed for an unknown reason".

Instruction 161, filed 2026-08-18 after a run that merged four databases, fitted,
and stopped with nothing a user could act on.

THE MESSAGE IS ITS OWN BUG, whatever the underlying failure was. A run that got
far enough to fail KNOWS four things -- the stage it reached, the shape of the
design it had built by then, the exception, and often what to change about it --
and reporting "failed" discards all four. This module turns them into a report,
writes it beside the run, and returns it so the console shows the same text the
folder keeps.

It is the same correction already made three times elsewhere: 153's "no summary"
named the wrong cause, 154 E's "nothing to scan" named none, and 155 A's "not
read from a run folder" named a requirement that did not exist. Each was fixed
by saying the true thing rather than by saying more.

NOTHING HERE RAISES. A reporting path that can fail takes the real exception
with it, and the user is then worse off than with the bare message this replaces.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Mapping, Optional

__all__ = [
    "FAILURE_FILENAME",
    "REMEDIES",
    "describe_failure",
    "write_failure_report",
]

#: What the report is called in the run folder. Plain text, and named for what
#: it is: a run that failed is a run somebody asks about tomorrow, and a console
#: has scrolled by then.
FAILURE_FILENAME = "regression_failure.txt"

#: Exception signatures whose remedy is known, mapped to what to change. Matched
#: on the message rather than the type, because the informative part of a numpy
#: or statsmodels error is almost always its text -- and a wrong remedy is worse
#: than none, so anything not recognised gets no advice rather than a guess.
REMEDIES = (
    ("singular matrix", (
        "The design is singular: two or more terms carry the same information, "
        "so no unique solution exists. Drop a redundant term, or fit at gene "
        "level where the guides of one gene collapse into one column.")),
    ("exog contains inf", (
        "A predictor holds inf or NaN. A fraction computed from a well with no "
        "reads is the usual source; raise min_cell_count, or filter the wells "
        "with no counts before fitting.")),
    ("0-size array", (
        "The design came out empty, so the join produced no rows. The scores "
        "and the counts did not meet -- check that both name the same plates "
        "(a plate stored as 'pplate1' does not meet one written 'plate1').")),
    ("out of memory", (
        "The fit asked for more memory than there is. Fit at well level rather "
        "than cell level, or use regression_type='ols', which does not build "
        "the dense random-effects design.")),
    ("cuda", (
        "The GPU backend failed. regression_backend='statsmodels (CPU)' fits "
        "the same model without a device.")),
    ("no such file", (
        "An input named in the settings is not on disk. The path is in the "
        "settings block below.")),
    ("keyerror", (
        "A column the fit expected is not in the frame. The column set it "
        "actually had is in the design block below.")),
)


def _remedy(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    for needle, advice in REMEDIES:
        if needle in text:
            return advice
    return ""


def _design_lines(frame, settings: Optional[Mapping[str, Any]]) -> list:
    """What had been built when it failed, as far as it can be known.

    EVERY LINE IS OPTIONAL AND NONE IS INVENTED. A stage that failed before the
    frame existed has no rows to report, and saying "0 rows" there would be a
    number that reads as a finding.
    """
    lines = []
    try:
        if frame is not None and hasattr(frame, "__len__"):
            lines.append(f"  rows in the design      {len(frame):,}")
        columns = getattr(frame, "columns", None)
        if columns is not None:
            lines.append(f"  columns                 {len(list(columns))}")
        for name, label in (("prc", "wells"), ("grna", "guides"),
                            ("gene", "genes")):
            if columns is not None and name in list(columns):
                lines.append(
                    f"  distinct {label:<14} {frame[name].nunique():,}")
    except Exception:                                            # noqa: BLE001
        lines.append("  (the design could not be described)")
    if settings:
        for key in ("regression_type", "inference", "analysis_mode",
                    "regression_backend", "level", "fdr_alpha",
                    "min_cell_count", "fraction_threshold"):
            if key in settings:
                lines.append(f"  {key:<23} {settings[key]!r}")
    return lines


def describe_failure(error: BaseException, *, stage: str = "",
                     settings: Optional[Mapping[str, Any]] = None,
                     frame: Any = None,
                     include_traceback: bool = True) -> str:
    """The report, as text. Never raises."""
    try:
        parts = ["THE REGRESSION FAILED."]
        if stage:
            parts.append(f"\nSTAGE REACHED\n  {stage}")
        design = _design_lines(frame, settings)
        if design:
            parts.append("\nTHE DESIGN IT HAD BUILT\n" + "\n".join(design))
        parts.append(f"\nWHAT WENT WRONG\n  {type(error).__name__}: {error}")
        advice = _remedy(error)
        if advice:
            parts.append(f"\nWHAT TO CHANGE\n  {advice}")
        else:
            parts.append(
                "\nWHAT TO CHANGE\n  This failure has no recorded remedy. The "
                "traceback below is the whole of what is known; a guess here "
                "would be worse than none.")
        # WHAT IT COST ON THE WAY (instruction 160). A failure that ran out of
        # memory looks identical to one that did not, unless the readings taken
        # per stage are beside it.
        try:
            from .fit_resources import describe_resources

            costs = describe_resources(settings)
        except Exception:                                        # noqa: BLE001
            costs = ""
        if costs:
            parts.append("\nWHAT IT COST, PER STAGE\n" + costs)
        if include_traceback:
            tb = "".join(traceback.format_exception(
                type(error), error, error.__traceback__))
            parts.append("\nTRACEBACK\n" + tb.rstrip())
        return "\n".join(parts) + "\n"
    except Exception:                                            # noqa: BLE001
        # The reporter must never replace the failure it is reporting.
        return f"THE REGRESSION FAILED: {type(error).__name__}: {error}\n"


def write_failure_report(res_folder, error: BaseException, *,
                         stage: str = "",
                         settings: Optional[Mapping[str, Any]] = None,
                         frame: Any = None) -> Optional[str]:
    """Write the report beside the run and return its path, or ``None``.

    ``None`` when there is nowhere to write it -- a failure early enough to
    have no destination folder is still reported to the console by the caller.
    """
    text = describe_failure(error, stage=stage, settings=settings, frame=frame)
    try:
        if not res_folder:
            return None
        folder = os.path.abspath(os.fspath(res_folder))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, FAILURE_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path
    except Exception:                                            # noqa: BLE001
        return None
