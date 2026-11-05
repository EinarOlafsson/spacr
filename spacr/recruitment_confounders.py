"""Field quality control and confounder adjustment for image-based readouts.

A recruitment ratio (or a vacuole size) measured per object is not only a
function of the condition in the well. It also moves with things the
experimenter did not set on purpose: how crowded the field was, how many
parasites landed in it, how large the host cell is, how bright the parasite
marker is (spectral spill-over into the readout channel), where on the plate
the well sits. When such a variable differs between the conditions being
compared *and* moves the readout inside a condition, it confounds the
comparison.

This module does three things, in the order an analysis needs them:

``field_density`` / ``flag_crowded_fields``
    Count host cells and parasites per imaged field straight from
    ``measurements.db`` and flag fields that are crowded beyond a robust
    (median + k·MAD) limit, have too few host cells to be informative, or carry
    too many parasites per host cell. Over-infected fields are where vacuoles
    fuse, masks merge and host cells lyse, so they are excluded, not adjusted.

``confounder_report``
    For each candidate covariate: its within-condition slope on the outcome
    (condition fixed effects, so a difference *between* conditions is not
    mistaken for a slope), and how unevenly it is distributed across
    conditions (eta squared). A covariate is a likely confounder when both are
    non-trivial.

``adjust_for_covariates``
    ANCOVA-style adjustment with a pooled within-condition slope:
    ``y = alpha_condition + sum_k beta_k * z_k``, where ``z_k`` are the covariates
    standardised over all rows. The adjusted value is ``y - sum_k beta_k * z_k``,
    i.e. every object is moved to the grand mean of every covariate. Because
    the slopes are estimated within conditions, the adjustment cannot absorb a
    true condition effect into a covariate; it only removes the part of the
    difference explained by covariate imbalance.

    Only adjust for variables the condition does not itself cause. A variable
    on the causal path (a mediator) - for example vacuole size when the
    question is whether a mutant grows slower - is removed together with the
    effect it carries. ``confounder_report`` cannot tell the two apart; that
    decision belongs to the experimenter.

All functions are pure pandas/numpy and do not modify their input.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import numpy as np
import pandas as pd

__all__ = [
    "FIELD_KEYS",
    "field_density",
    "well_density",
    "apply_well_density",
    "flag_crowded_fields",
    "apply_field_qc",
    "confounder_report",
    "adjust_for_covariates",
]

#: Columns that identify one imaged field.
FIELD_KEYS: Tuple[str, ...] = ("plateID", "rowID", "columnID", "fieldID")

#: Scales a median absolute deviation to a standard deviation for normal data.
_MAD_SCALE = 1.4826


def _field_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the field-key columns with the plate id URL-decoded.

    Measurement tables have been written with both ``AssayPlate%5FPerkinElmer``
    and ``AssayPlate_PerkinElmer`` spellings of the same plate; decoding both
    sides makes them join.
    """
    keys = frame[list(FIELD_KEYS)].astype(str).copy()
    keys["plateID"] = keys["plateID"].map(lambda v: unquote(unquote(v)))
    return keys


def field_density(db_path: str, cell_table: str = "cell",
                  pathogen_table: Optional[str] = "pathogen") -> pd.DataFrame:
    """Count host cells and parasites in every imaged field of a plate.

    :param db_path: path to a spacr ``measurements.db``.
    :param cell_table: table holding one row per host cell (or per object to
        count, e.g. the vacuole table of a replication plate).
    :param pathogen_table: table holding one row per parasite / vacuole, or
        ``None`` to count only ``cell_table``.
    :returns: one row per field with ``n_cells_field``, ``n_pathogens_field``,
        ``n_pathogens_in_cells_field`` (pathogens assigned to a host cell) and
        ``pathogens_per_cell_field``.
    """
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        tables = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")}
        keys = ", ".join(FIELD_KEYS)
        cells = pd.read_sql(
            f'select {keys}, count(*) as n_cells_field from "{cell_table}" group by {keys}', con)
        if pathogen_table and pathogen_table in tables:
            cols = {r[1] for r in con.execute(f'pragma table_info("{pathogen_table}")')}
            in_cell = ("sum(case when cell_id > 0 then 1 else 0 end)"
                       if "cell_id" in cols else "count(*)")
            path = pd.read_sql(
                f'select {keys}, count(*) as n_pathogens_field, {in_cell} as n_pathogens_in_cells_field '
                f'from "{pathogen_table}" group by {keys}', con)
        else:
            path = None
    cells[list(FIELD_KEYS)] = _field_key_frame(cells)
    if path is not None:
        path[list(FIELD_KEYS)] = _field_key_frame(path)
        fields = cells.merge(path, on=list(FIELD_KEYS), how="outer")
        fields[["n_cells_field", "n_pathogens_field", "n_pathogens_in_cells_field"]] = (
            fields[["n_cells_field", "n_pathogens_field", "n_pathogens_in_cells_field"]].fillna(0))
        fields["pathogens_per_cell_field"] = (
            fields["n_pathogens_field"] / fields["n_cells_field"].replace(0, np.nan))
    else:
        fields = cells
    return fields


#: Columns that identify one well.
WELL_KEYS: Tuple[str, ...] = ("plateID", "rowID", "columnID")


def well_density(db_path: str, cell_table: str = "cell",
                 pathogen_table: Optional[str] = "pathogen") -> pd.DataFrame:
    """Count host cells and parasites in every well of a plate.

    The well-level twin of :func:`field_density`, summed over the fields of
    each well.

    :returns: one row per well (:data:`WELL_KEYS`) with ``n_cells_well``,
        ``n_pathogens_well`` (when ``pathogen_table`` exists),
        ``n_pathogens_in_cells_well``, ``pathogens_per_cell_well`` and
        ``n_fields_well``.
    """
    fields = field_density(db_path, cell_table=cell_table, pathogen_table=pathogen_table)
    sums = {c: c.replace("_field", "_well") for c in
            ("n_cells_field", "n_pathogens_field", "n_pathogens_in_cells_field")
            if c in fields.columns}
    wells = fields.groupby(list(WELL_KEYS))[list(sums)].sum().rename(columns=sums)
    wells["n_fields_well"] = fields.groupby(list(WELL_KEYS)).size()
    wells = wells.reset_index()
    if "n_pathogens_well" in wells.columns:
        wells["pathogens_per_cell_well"] = (
            wells["n_pathogens_well"] / wells["n_cells_well"].replace(0, np.nan))
    return wells


def apply_well_density(df: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    """Attach the :func:`well_density` columns to object rows (a new frame).

    Plate ids are URL-decoded on both sides, as for :func:`apply_field_qc`.
    Rows from wells absent from ``wells`` get ``NaN``.
    """
    decode = lambda frame: frame[list(WELL_KEYS)].astype(str).assign(
        plateID=lambda k: k["plateID"].map(lambda v: unquote(unquote(v))))
    cols = [c for c in wells.columns if c not in WELL_KEYS]
    table = wells.copy()
    table[list(WELL_KEYS)] = decode(table)
    table = table.drop_duplicates(list(WELL_KEYS))
    merged = decode(df).reset_index(drop=True).merge(table, on=list(WELL_KEYS), how="left")
    out = df.drop(columns=[c for c in cols if c in df.columns]).copy()
    for column in cols:
        out[column] = merged[column].to_numpy()
    return out


def _robust_upper(values: pd.Series, k: float) -> float:
    values = values.dropna()
    if values.empty:
        return np.inf
    med = values.median()
    mad = (values - med).abs().median() * _MAD_SCALE
    return med + k * mad if mad > 0 else np.inf


def flag_crowded_fields(fields: pd.DataFrame, mad_threshold: Optional[float] = 3.5,
                        columns: Sequence[str] = ("n_pathogens_field", "pathogens_per_cell_field"),
                        min_cells: Optional[int] = 10,
                        max_pathogens_per_cell: Optional[float] = None,
                        by: Optional[str] = "plateID") -> pd.DataFrame:
    """Flag imaged fields that are over-crowded or too sparse to use.

    A field fails when any of ``columns`` exceeds ``median + mad_threshold * MAD``
    (computed per ``by`` group, i.e. per plate), when it has fewer than
    ``min_cells`` host cells, or when its parasites-per-cell exceeds
    ``max_pathogens_per_cell``. The robust limit adapts to each plate's own
    infection level, so a plate infected at a higher MOI is not emptied.

    :param fields: output of :func:`field_density`, one row per field.
    :param mad_threshold: robust standard deviations above the median at which
        a field fails; ``None`` skips the robust limit.
    :param columns: field columns tested against the robust limit.
    :param min_cells: fewest host cells a field may hold; ``None`` skips it.
    :param max_pathogens_per_cell: absolute ceiling on parasites per host
        cell; ``None`` sets no ceiling.
    :param by: column whose groups get their own limit, normally the plate;
        ``None`` uses one limit for all fields.
    :returns: ``fields`` plus ``field_qc_pass`` (bool), ``field_qc_reason``
        (``''`` when passing, ``;``-joined reasons otherwise) and one
        ``limit_<column>`` column per robust limit applied.
    """
    out = fields.copy()
    reasons = pd.Series([[] for _ in range(len(out))], index=out.index)
    groups = out.groupby(by) if by and by in out.columns else [(None, out)]
    if mad_threshold is not None:
        for column in columns:
            if column not in out.columns:
                continue
            limit_col = f"limit_{column}"
            out[limit_col] = np.nan
            for _, group in groups:
                limit = _robust_upper(group[column], mad_threshold)
                out.loc[group.index, limit_col] = limit
                bad = group.index[group[column] > limit]
                for idx in bad:
                    reasons.at[idx] = reasons.at[idx] + [f"{column}>{limit:.3g}"]
    if min_cells is not None and "n_cells_field" in out.columns:
        for idx in out.index[out["n_cells_field"] < min_cells]:
            reasons.at[idx] = reasons.at[idx] + [f"n_cells_field<{min_cells}"]
    if max_pathogens_per_cell is not None and "pathogens_per_cell_field" in out.columns:
        for idx in out.index[out["pathogens_per_cell_field"] > max_pathogens_per_cell]:
            reasons.at[idx] = reasons.at[idx] + [f"pathogens_per_cell_field>{max_pathogens_per_cell}"]
    out["field_qc_reason"] = reasons.map(";".join)
    out["field_qc_pass"] = out["field_qc_reason"] == ""
    return out


def apply_field_qc(df: pd.DataFrame, fields: pd.DataFrame,
                   keep_columns: Iterable[str] = ("n_cells_field", "n_pathogens_field",
                                                  "pathogens_per_cell_field"),
                   drop_failed: bool = True) -> pd.DataFrame:
    """Attach per-field density columns to object rows and drop failed fields.

    :param df: object-level frame with the :data:`FIELD_KEYS` columns.
    :param fields: output of :func:`flag_crowded_fields`.
    :param keep_columns: field columns to copy onto every object row.
    :param drop_failed: remove rows from fields where ``field_qc_pass`` is False.
    :returns: a new frame; rows from fields absent from ``fields`` are kept and
        marked as passing (nothing is known against them).
    """
    keys = _field_key_frame(df)
    cols = [c for c in keep_columns if c in fields.columns]
    extra = [c for c in ("field_qc_pass", "field_qc_reason") if c in fields.columns]
    table = fields[list(FIELD_KEYS) + cols + extra].copy()
    table[list(FIELD_KEYS)] = _field_key_frame(table)
    table = table.drop_duplicates(list(FIELD_KEYS))
    merged = keys.reset_index(drop=True).merge(table, on=list(FIELD_KEYS), how="left")
    out = df.drop(columns=[c for c in cols + extra if c in df.columns]).copy()
    for column in cols + extra:
        out[column] = merged[column].to_numpy()
    if "field_qc_pass" in out.columns:
        out["field_qc_pass"] = [True if pd.isna(v) else bool(v) for v in out["field_qc_pass"]]
        if drop_failed:
            out = out[out["field_qc_pass"]]
    return out


def _design(df: pd.DataFrame, outcome: str, group: str, covariates: Sequence[str],
            log_outcome: bool, log_covariates) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Tuple[float, float]]]:
    """Clean rows and build the standardised covariate matrix."""
    if isinstance(log_covariates, bool):
        log_cov = set(covariates) if log_covariates else set()
    else:
        log_cov = set(log_covariates or ())
    data = df[[outcome, group, *covariates]].copy()
    y = pd.to_numeric(data[outcome], errors="coerce")
    if log_outcome:
        y = np.log(y.where(y > 0))
    data["_y"] = y
    scales = {}
    for c in covariates:
        x = pd.to_numeric(data[c], errors="coerce").astype(float)
        if c in log_cov:
            x = np.log1p(x.where(x >= 0))
        data[f"_z_{c}"] = x
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["_y", group] + [f"_z_{c}" for c in covariates])
    for c in covariates:
        x = data[f"_z_{c}"]
        mu, sd = float(x.mean()), float(x.std(ddof=0))
        scales[c] = (mu, sd)
        data[f"_z_{c}"] = (x - mu) / sd if sd > 0 else 0.0
    return data, data["_y"].to_numpy(), scales


def _within_ols(data: pd.DataFrame, y: np.ndarray, group: str, zcols: List[str]):
    """OLS of y on group fixed effects + z columns via within-group demeaning."""
    demean = lambda s: s - s.groupby(data[group]).transform("mean")
    yw = demean(pd.Series(y, index=data.index)).to_numpy()
    Xw = np.column_stack([demean(data[z]).to_numpy() for z in zcols]) if zcols else np.empty((len(y), 0))
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    n_groups = data[group].nunique()
    dof = max(len(y) - n_groups - len(zcols), 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(Xw.T @ Xw)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(zcols), np.nan)
    return beta, se, dof


def confounder_report(df: pd.DataFrame, outcome: str, group: str, covariates: Sequence[str],
                      log_outcome: bool = True, log_covariates=True,
                      min_abs_beta: float = 0.02, min_eta2: float = 0.01) -> pd.DataFrame:
    """Screen candidate covariates for confounding of ``outcome`` across ``group``.

    Each covariate is fitted on its own (``outcome ~ group + z``), which answers
    "does this variable move the readout inside a condition?" without the
    answer depending on which other covariates were offered.

    :param df: object-level frame.
    :param outcome: readout column, e.g. ``'recruitment'``.
    :param group: condition column; slopes are estimated within it.
    :param covariates: candidate confounder columns.
    :param log_outcome: model ``log(outcome)`` (ratios and areas are
        multiplicative).
    :param log_covariates: ``True`` (all), ``False`` (none) or a list of
        covariates to ``log1p`` before standardising.
    :param min_abs_beta: smallest within-condition effect per SD (on the
        modelled scale) counted as relevant. 0.02 on a log scale is ~2 %.
    :param min_eta2: smallest between-condition imbalance counted as relevant.
    :returns: one row per covariate - ``beta_per_sd``, ``se``, ``t``,
        ``imbalance_eta2`` (share of the covariate's variance explained by
        ``group``), ``bias_proxy`` (``|beta| * sqrt(eta2)``, the shift the
        covariate can induce between conditions, in outcome units per SD) and
        ``likely_confounder``. Sorted by ``bias_proxy``.
    """
    rows = []
    for c in covariates:
        if c not in df.columns:
            continue
        data, y, _ = _design(df, outcome, group, [c], log_outcome, log_covariates)
        if len(data) < 10 or data[f"_z_{c}"].std() == 0:
            continue
        beta, se, dof = _within_ols(data, y, group, [f"_z_{c}"])
        z = data[f"_z_{c}"]
        means = z.groupby(data[group]).transform("mean")
        ss_between = float(((means - z.mean()) ** 2).sum())
        ss_total = float(((z - z.mean()) ** 2).sum())
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        b = float(beta[0])
        rows.append(dict(covariate=c, n=len(data), beta_per_sd=b, se=float(se[0]),
                         t=b / se[0] if se[0] > 0 else np.nan,
                         imbalance_eta2=eta2, bias_proxy=abs(b) * np.sqrt(eta2),
                         likely_confounder=bool(abs(b) >= min_abs_beta and eta2 >= min_eta2)))
    report = pd.DataFrame(rows)
    if len(report):
        report = report.sort_values("bias_proxy", ascending=False).reset_index(drop=True)
    return report


def adjust_for_covariates(df: pd.DataFrame, outcome: str, group: str, covariates: Sequence[str],
                          log_outcome: bool = True, log_covariates=True,
                          out_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the covariate-explained part of ``outcome`` (pooled within-group ANCOVA).

    Fits ``outcome ~ group + z_1 + ... + z_k`` jointly, with every covariate
    standardised over all rows, and returns
    ``adjusted = outcome - sum_k beta_k * z_k`` (on the log scale when
    ``log_outcome``, back-transformed afterwards). Each row is thereby moved to
    the grand mean of every covariate; the condition means of the adjusted
    column are the ANCOVA adjusted means.

    Rows with a missing outcome or covariate keep ``NaN`` in the adjusted
    column.

    :param df: object-level frame.
    :param outcome: readout column to adjust, e.g. ``'recruitment'``.
    :param group: condition column; the slopes are pooled within it.
    :param covariates: columns to regress out; columns absent from ``df``
        are skipped.
    :param log_outcome: adjust ``log(outcome)`` and back-transform.
    :param log_covariates: ``True`` (all), ``False`` (none) or a list of
        covariates to ``log1p`` before standardising.
    :param out_column: name of the adjusted column; ``None`` names it
        ``f'{outcome}_adjusted'``.
    :returns: ``(frame, model)`` - a copy of ``df`` with ``out_column``
        (default ``f'{outcome}_adjusted'``) added, and a table of the fitted
        slopes (``covariate``, ``beta_per_sd``, ``se``, ``t``, ``mean``, ``sd``,
        ``n``).
    """
    out_column = out_column or f"{outcome}_adjusted"
    covariates = [c for c in covariates if c in df.columns]
    out = df.copy()
    out[out_column] = np.nan
    if not covariates:
        out[out_column] = pd.to_numeric(out[outcome], errors="coerce")
        return out, pd.DataFrame(columns=["covariate", "beta_per_sd", "se", "t", "mean", "sd", "n"])
    data, y, scales = _design(df, outcome, group, covariates, log_outcome, log_covariates)
    zcols = [f"_z_{c}" for c in covariates]
    beta, se, _ = _within_ols(data, y, group, zcols)
    shift = data[zcols].to_numpy() @ beta
    adjusted = y - shift
    out.loc[data.index, out_column] = np.exp(adjusted) if log_outcome else adjusted
    model = pd.DataFrame(dict(covariate=covariates, beta_per_sd=beta, se=se,
                              t=np.where(se > 0, beta / se, np.nan),
                              mean=[scales[c][0] for c in covariates],
                              sd=[scales[c][1] for c in covariates],
                              n=len(data)))
    return out, model
