"""Saving a graph writes a FOLDER, not a file (instruction 223).

    <graph name>/
        <graph name>.pdf      vector, for the manuscript
        <graph name>.png      raster, for slides and Slack
        data.csv              the rows the graph was drawn from
        statistics.csv        the test, its assumptions, and its result
        settings.json         what produced it

A PDF ON ITS OWN CANNOT BE CHECKED. Six months later the question is always
the same -- what were the numbers, and was that difference tested -- and a
figure file answers neither.

THE STATISTICS COME FROM `spacr.figures.stats.compare`, never a second
implementation. A figure whose saved statistics disagree with the same
comparison drawn on screen is worse than one with no statistics at all.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.figures.bundle")

#: The files a bundle always has. ALWAYS, including the ones with nothing to
#: say -- an absent file reads as a bug, and "there was nothing to compare"
#: is a result.
FILES = ("data.csv", "statistics.csv", "settings.json")

#: What `statistics.csv` says when a graph has no groups. A scatter or a
#: one-column histogram has no test to run, which is not the same as a test
#: that failed.
NOTHING_TO_COMPARE = (
    "This graph has no groups to compare, so no test was run. A scatter or "
    "a histogram of one column is a description rather than a comparison; "
    "the data it was drawn from is in data.csv.")


def _unique(folder: str) -> str:
    """A folder that does not exist yet, by suffixing.

    OVERWRITING A FOLDER IS A BIGGER ACT THAN OVERWRITING A FILE. A file
    replaced loses one thing the user can see; a folder replaced can take
    the data and the statistics of an earlier save with it, silently. So a
    second save of the same graph sits beside the first rather than on it.
    """
    if not os.path.exists(folder):
        return folder
    for index in range(2, 1000):
        candidate = f"{folder}-{index}"
        if not os.path.exists(candidate):
            return candidate
    raise FileExistsError(f"{folder} and 998 suffixes of it all exist")


def statistics_rows(comparison) -> list:
    """One :class:`~spacr.figures.stats.Comparison` as csv rows.

    A P-VALUE ALONE IS NOT A RESULT, so every row the instruction lists is
    here: the groups and their n, the unit, both assumption checks with the
    test used for each and why, the chosen test and the reason it was chosen,
    the statistic, the p-value, and an effect size with its interval.
    """
    rows = [("unit", comparison.unit,
             "what ONE observation is; a test across cells when the "
             "replicate is the well is pseudoreplication")]
    for label, count in zip(comparison.groups, comparison.n):
        rows.append((f"n [{label}]", int(count), "usable observations"))
    for assumption in comparison.assumptions:
        # READ THE CHECK'S OWN VERDICT (`passed`), never re-derive it from
        # the p-value. The normality check compares the worst of k groups
        # against a BONFERRONI threshold, and a caller re-deriving
        # `p_value >= 0.05` silently discards the correction -- which in
        # this codebase sent 18% of four-group comparisons on normal data to
        # a rank test instead of 5%.
        state = "holds" if assumption.passed else "does not hold"
        if not assumption.informative:
            # A CHECK THAT COULD NOT SEE IS NOT A CHECK THAT PASSED, and a
            # file recording it as "holds" would be the more misleading of
            # the two.
            state = "could not tell"
        rows.append((assumption.name, state, assumption.verdict))
        rows.append((f"{assumption.name} p", float(assumption.p_value),
                     "the number behind the verdict above"))
    rows.append(("test", comparison.test, comparison.reason))
    rows.append(("statistic", float(comparison.statistic), ""))
    rows.append(("p_value", float(comparison.p_value), ""))
    if np.isfinite(comparison.p_adjusted):
        rows.append(("p_adjusted", float(comparison.p_adjusted),
                     comparison.correction))
    if np.isfinite(comparison.effect_size):
        rows.append(("effect_size", float(comparison.effect_size),
                     comparison.effect_name))
    if comparison.ci is not None:
        rows.append(("effect_ci_low", float(comparison.ci[0]), "95%"))
        rows.append(("effect_ci_high", float(comparison.ci[1]), "95%"))
    rows.append(("sentence", comparison.sentence(),
                 "the legend line, as it would be reported"))
    return rows


def statistics_frame(groups: Optional[Mapping[str, Sequence]] = None, *,
                     unit: str = "observation",
                     paired: bool = False) -> pd.DataFrame:
    """The statistics table for these groups. Never raises.

    A COMPARISON THAT COULD NOT BE MADE IS NOT A COMPARISON WITH AN UNKNOWN
    ANSWER. `compare` raises rather than returning NaN, and the file says the
    comparison was refused and WHY rather than leaving an empty cell.
    """
    columns = ["item", "value", "note"]
    if not groups or len(groups) < 2:
        return pd.DataFrame(
            [("comparison", "none", NOTHING_TO_COMPARE)], columns=columns)
    try:
        from .stats import compare

        comparison = compare(groups, unit=unit, paired=paired)
    except Exception as error:                                   # noqa: BLE001
        return pd.DataFrame(
            [("comparison", "refused",
              f"{type(error).__name__}: {error}")], columns=columns)
    return pd.DataFrame(statistics_rows(comparison), columns=columns)


def save(folder: str, name: str, *,
         render: Callable[[str], Any],
         data: Optional[pd.DataFrame] = None,
         groups: Optional[Mapping[str, Sequence]] = None,
         unit: str = "observation",
         paired: bool = False,
         settings: Optional[Mapping[str, Any]] = None) -> str:
    """Write the whole bundle. Returns the folder actually written.

    :param folder: where the bundle folder goes.
    :param name: the graph's name; the folder and both figures take it.
    :param render: called with a path per format. RENDERED ONCE PER FORMAT
        BY THE CALLER, because only the caller knows how to draw itself --
        but from the same state both times, so the pdf and the png are the
        same figure rather than two draws that could differ.
    :param data: the rows the graph was drawn from, AFTER filtering, which
        is what the graph shows.
    :param groups: ``{label: values}`` for the statistics, or None for a
        graph with nothing to compare.
    :param settings: what produced it. Without the filters recorded beside
        the data the numbers cannot be reproduced.
    :returns: the folder path.
    """
    safe = "".join(c if c.isalnum() or c in "-_. " else "_"
                   for c in str(name or "graph")).strip() or "graph"
    out = _unique(os.path.join(str(folder), safe))
    os.makedirs(out, exist_ok=True)

    written = []
    for extension in ("pdf", "png"):
        path = os.path.join(out, f"{safe}.{extension}")
        try:
            render(path)
            written.append(path)
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not write %s", path, exc_info=True)

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame()
    frame.to_csv(os.path.join(out, "data.csv"), index=False)
    statistics_frame(groups, unit=unit, paired=paired).to_csv(
        os.path.join(out, "statistics.csv"), index=False)

    payload = {str(k): _plain(v) for k, v in dict(settings or {}).items()}
    payload.setdefault("graph", safe)
    payload.setdefault("rows", int(len(frame)))
    with open(os.path.join(out, "settings.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return out


def _plain(value):
    """Whatever JSON can hold; a string for everything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)
