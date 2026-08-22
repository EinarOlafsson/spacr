"""Export a graph with its data, statistics, and generating settings.

Each export creates a uniquely named directory containing PDF and PNG
renderings, the plotted data, a statistical summary, and a JSON settings
record. Statistical summaries use :func:`spacr.figures.stats.compare`, the
same comparison engine used by interactive figures.
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
    """Convert a statistical comparison into labeled CSV rows.

    Parameters
    ----------
    comparison : spacr.figures.stats.Comparison
        Comparison containing group sizes, assumption checks, test results,
        and effect estimates.

    Returns
    -------
    list of tuple
        ``(item, value, note)`` rows suitable for ``statistics.csv``.
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
    """Build the statistical table stored with an exported graph.

    Parameters
    ----------
    groups : mapping of str to sequence, optional
        Values grouped by comparison label. Fewer than two groups produce an
        explanatory row rather than a statistical test.
    unit : str, default="observation"
        Experimental unit represented by one value.
    paired : bool, default=False
        Whether observations are paired across groups.

    Returns
    -------
    pandas.DataFrame
        Columns ``item``, ``value``, and ``note``. Comparison errors are
        recorded in the table instead of being raised.
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
    """Write a reproducible figure-export bundle.

    Parameters
    ----------
    folder : str
        Parent directory for the export.
    name : str
        Graph name used for the directory and image files.
    render : callable
        Function called once with the PDF path and once with the PNG path.
    data : pandas.DataFrame, optional
        Filtered rows represented by the graph.
    groups : mapping of str to sequence, optional
        Values used to generate ``statistics.csv``.
    unit : str, default="observation"
        Experimental unit represented by one value.
    paired : bool, default=False
        Whether group observations are paired.
    settings : mapping, optional
        Settings and filters used to generate the graph.

    Returns
    -------
    str
        Path to the newly created bundle directory. A numeric suffix is added
        when the requested directory already exists.
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

    from ..tabular import write_table

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame()
    write_table(frame, os.path.join(out, "data.csv"))
    write_table(
        statistics_frame(groups, unit=unit, paired=paired),
        os.path.join(out, "statistics.csv"))

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
