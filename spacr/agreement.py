"""
Multi-annotator agreement for spaCR annotation columns.

Two people label the same 3 000 crops. How much do they actually agree,
and *where* do they disagree? This module answers both, reading nothing
but the ``png_list`` table of a ``measurements.db``.

Annotation columns are the ones the Annotate app writes: an ``INTEGER``
column added to ``png_list`` with ``ALTER TABLE … ADD COLUMN`` (see
:func:`spacr.qt.annotate_engine.ensure_annotation_column`), holding the
class the annotator pressed — ``1`` for a left click, ``2`` for a right
click — and ``NULL`` for every crop they have not looked at yet. Two
annotators means two such columns over the same ``png_path`` rows.

Public API
----------
``cohens_kappa(a, b)``
    κ for one pair of label vectors.
``fleiss_kappa(matrix)``
    κ for three or more annotators, from a subjects × categories count
    matrix.
``agreement_report(db_path, columns)``
    the whole picture for a database: per-pair κ, overall κ, per-class κ,
    confusion matrices, and the counts that make them interpretable.
``disagreements(db_path, columns)``
    exactly the rows the annotators labelled differently, for review.
``format_agreement(report)``
    the report as text.

Statistics that this module refuses to get wrong
------------------------------------------------
**An unlabelled cell is an abstention, not a disagreement.** Every pair
is scored on the rows *both* annotators labelled (pairwise complete
cases). Rows where only one of them committed are counted and reported
separately as ``n_abstained``. Treating them as disagreements would
silently deflate κ in proportion to how far behind the slower annotator
is — which is a property of the calendar, not of the annotation.

**κ has no value when there is no variance.** κ = (pₒ − pₑ)/(1 − pₑ).
If both annotators put every compared row in the same single class,
pₑ = 1 and the denominator vanishes: the answer is *undefined*, not 1.0.
If just one annotator used a single class, pₑ collapses onto the other's
marginal and κ is identically 0 no matter how well they agree. Both
cases return ``nan`` with an explanation attached, because on a screen
where 98 % of cells are negative they are the *normal* case, and a
returned 0.0 or 1.0 there is a lie with a number on it.

**Raw agreement is reported next to κ, always.** 95 % agreement with
κ ≈ 0 is the prevalence paradox (Feinstein & Cicchetti, 1990), not a
broken annotator: when one class dominates, chance agreement is already
almost as high as the observed agreement, so κ has almost no room left.
Hiding either number hides half the story.

**The interpretation bands are a convention.** Landis & Koch (1977)
"slight/fair/moderate/substantial/almost perfect" is a rule of thumb
with no distributional basis. It is reported as a label, and labelled as
a convention.

Nothing here imports torch, cellpose or any GPU stack — pandas, numpy
and the standard library only — so the Qt screen can compute agreement
without waking a 4-second import chain.
"""
from __future__ import annotations

import itertools
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote as _urlquote

import numpy as np
import pandas as pd

__all__ = [
    "AgreementReport",
    "CONVENTION",
    "LANDIS_KOCH",
    "PairAgreement",
    "PNG_TABLE",
    "PNG_KEY",
    "agreement_report",
    "annotation_columns",
    "cohens_kappa",
    "confusion_matrix",
    "disagreements",
    "fleiss_kappa",
    "format_agreement",
    "interpret_kappa",
    "kappa_detail",
    "load_annotations",
    "table_columns",
]


#: Table the Annotate app writes its per-crop labels into.
PNG_TABLE = "png_list"
#: Row key of that table — one row per object crop.
PNG_KEY = "png_path"

#: Columns ``png_list`` gets from :func:`spacr.utils.filepaths_to_database`
#: rather than from an annotator. Excluded when guessing which columns
#: hold annotations.
_METADATA_COLUMNS = frozenset({
    "png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
    # Both spellings of the timepoint. 'timeID' is what filepaths_to_database
    # writes now, and what spacr.utils.rename_columns_in_db migrates an old
    # database to on first read; 'time_id' is what a database written before
    # that still carries until then. Either one is metadata, never an
    # annotation -- and a timelapse database whose time column counted as a
    # candidate annotation column would have been scored for agreement.
    "timeID", "time_id", "prcft",
    "prcfo", "prc", "cell_id", "nucleus_id", "pathogen_id",
    "cytoplasm_id", "object_label", "plate", "row", "column", "field",
    "well", "id", "index", "level_0",
    # spacr.crops.CROP_FORMAT_DB_COLUMN, the channel-order version marker
    # stamp_crop_format_in_db adds to png_list. One or two distinct small
    # integers over every row -- the exact shape of an annotation pass, and
    # not one.
    "crop_format",
})

#: Columns of ``png_list`` written by a **model**, not by a person.
#:
#: This is the same bug as ``timeID`` and one worse. ``png_list`` is the only
#: table in spaCR that both the Annotate app and every classifier write into,
#: and the classifier's columns look exactly like an annotation pass: an
#: ``INTEGER`` class column with two distinct values and no NULLs. So
#: :func:`annotation_columns` offered them, and the resulting κ was not
#: inter-annotator agreement at all -- it silently added the model as a third
#: annotator.
#:
#: Measured on a two-well database built by the real writers, two human
#: annotators and one CV run merged in by :func:`spacr.predictions.merge_cv_predictions`:
#: candidates came back as ``['annotator_ann', 'annotator_bob', 'pred',
#: 'cv_predictions']``, and the overall κ over all four was **-0.004**
#: ("poor (no better than chance)") where the two humans alone agree at
#: **0.471** ("moderate"). The number a user would have quoted was the
#: classifier's disagreement with the people, reported as the people's
#: disagreement with each other.
#:
#: Names are kept in step with their writers by
#: ``tests/test_agreement_excludes_the_model.py``, which imports the constants
#: from :mod:`spacr.predictions` and :mod:`spacr.active_learning` and asserts
#: every one of them is listed here. They are duplicated rather than imported
#: so that this module keeps its promise of importing nothing but pandas,
#: numpy and the standard library.
_MODEL_COLUMNS = frozenset({
    # spacr.predictions: the convolutional classifier
    "pred", "cv_predictions",
    # spacr.predictions: the classical-ML classifier
    "ml_pred", "predictions",
    # spacr.active_learning.PRED_COLUMN_CANDIDATES, the two not already above
    "prediction", "score",
    # spacr.gui_elements: the Annotate app's built-in XGBoost pass. The name
    # says "annotation" and it is not one -- it is a model's call, derived
    # from a score in the very next column.
    "XGboost_annotation", "XGboost_score",
})

#: Per-class probability columns ``spacr.ml.ml_analysis`` produces
#: (``prediction_probability_class_0``, ``..._1``, ...). A prefix rather than
#: a name because the count follows the number of classes.
_MODEL_COLUMN_PREFIXES = ("prediction_probability_class_",)

#: Suffix of the sampled-negatives column ``spacr.io.generate_training_dataset``
#: writes next to an annotation column (``<col>_random``): 1 for the rows it
#: drew as controls, NULL everywhere else. Excluded only when ``<col>`` is
#: itself a column of the same table, so a genuine annotator who happens to be
#: called ``blind_random`` is still offered.
_SAMPLED_COLUMN_SUFFIX = "_random"


def _is_model_column(name: str, table_columns: Sequence[str] = ()) -> bool:
    """True when ``name`` is written by a model rather than by an annotator.

    :param name: candidate column name.
    :param table_columns: the table's other columns, used to recognise a
        ``<col>_random`` sampling column by the column it was derived from.
    :returns: True to exclude it from the annotation-column guess.
    """
    if name in _MODEL_COLUMNS:
        return True
    if any(name.startswith(prefix) for prefix in _MODEL_COLUMN_PREFIXES):
        return True
    if name.endswith(_SAMPLED_COLUMN_SUFFIX):
        base = name[: -len(_SAMPLED_COLUMN_SUFFIX)]
        if base and base in set(table_columns):
            return True
    return False


#: Landis & Koch (1977) bands, as ``(upper_bound, label)`` pairs. A
#: **convention** — see :data:`CONVENTION`.
LANDIS_KOCH: Tuple[Tuple[float, str], ...] = (
    (0.00, "poor (no better than chance)"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
)

CONVENTION = (
    "Bands follow the Landis & Koch (1977) convention, not a law: the "
    "cut-points are a rule of thumb with no distributional basis. What "
    "counts as acceptable agreement depends on the assay, the number of "
    "classes and how lopsided the classes are."
)

_TOL = 1e-12


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def _scalar_label(value: Any) -> Optional[Any]:
    """Normalise one stored cell into a class label, or ``None``.

    ``None`` means *abstention* — this annotator did not label this row.
    SQLite hands back ``None`` for NULL, but a column read through pandas
    can arrive as ``float('nan')``, and a hand-edited database can hold
    ``"1"`` where its neighbour holds ``1``. All three normalise here so
    that "1" from one annotator and 1 from another count as agreement.

    :param value: raw cell value.
    :returns: ``int``/``float``/``str`` label, or ``None`` for a missing one.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return None
        return int(v) if v.is_integer() else v
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
            except ValueError:
                return s
            return int(f) if f.is_integer() else f
    if isinstance(value, bytes):
        return _scalar_label(value.decode("utf-8", "replace"))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalise(values: Iterable[Any],
               missing_values: Sequence[Any] = ()) -> List[Optional[Any]]:
    """Normalise a whole column, honouring extra ``missing_values``.

    :param values: raw cells.
    :param missing_values: additional labels to treat as abstentions. The
        Annotate app clears a label by writing NULL, so the default is
        empty — but older Tk databases used ``0`` for "not looked at yet"
        (see ``find_last_annotated_offset``), and those need
        ``missing_values=(0,)`` or every untouched row becomes a real
        class and drags κ around.
    """
    extra = {_scalar_label(m) for m in missing_values}
    extra.discard(None)
    out: List[Optional[Any]] = []
    for v in values:
        label = _scalar_label(v)
        out.append(None if label in extra else label)
    return out


def _sorted_labels(labels: Iterable[Any]) -> List[Any]:
    """Sort mixed label types without tripping over int-vs-str comparison."""
    return sorted(labels, key=lambda x: (isinstance(x, str), x))


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_kappa(kappa: float) -> str:
    """Return the Landis & Koch band for ``kappa`` (a convention).

    :param kappa: a κ value, possibly ``nan``.
    :returns: band name, or ``"undefined"`` for ``nan``.
    """
    if kappa is None:
        return "undefined"
    try:
        k = float(kappa)
    except (TypeError, ValueError):
        return "undefined"
    if math.isnan(k):
        return "undefined"
    for upper, name in LANDIS_KOCH:
        if k <= upper:
            return name
    return LANDIS_KOCH[-1][1]


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------

@dataclass
class PairAgreement:
    """Cohen's κ for one pair of annotators, with everything needed to read it.

    :ivar kappa: Cohen's κ, or ``nan`` when it is undefined/degenerate —
        check :attr:`defined` before quoting it.
    :ivar percent_agreement: raw pₒ, the fraction of compared rows where
        the two labels are identical. Always meaningful, even when κ is not.
    :ivar expected_agreement: pₑ, agreement expected from the marginals alone.
    :ivar n_compared: rows *both* annotators labelled — κ's denominator.
    :ivar n_abstained: rows exactly one of them labelled. Excluded from κ
        (an abstention is not a disagreement) and reported here instead.
    :ivar n_neither: rows neither of them has reached yet.
    :ivar confusion: ``a`` labels down the rows, ``b`` across the columns.
    :ivar note: why κ is ``nan``, or what to watch out for when it is not.
    """

    column_a: str
    column_b: str
    kappa: float
    percent_agreement: float
    expected_agreement: float
    n_compared: int
    n_agree: int
    n_disagree: int
    n_abstained: int
    n_neither: int
    labels: List[Any]
    confusion: pd.DataFrame
    note: str = ""
    interpretation: str = "undefined"

    @property
    def defined(self) -> bool:
        """False when κ is ``nan`` — i.e. the data cannot support a κ."""
        return not (self.kappa is None or math.isnan(float(self.kappa)))

    def __str__(self) -> str:
        k = "undefined" if not self.defined else f"{self.kappa:+.3f}"
        return (f"{self.column_a} vs {self.column_b}: κ={k} "
                f"({self.interpretation}), raw agreement "
                f"{self.percent_agreement:.1%} on {self.n_compared} rows")


def confusion_matrix(a: Sequence[Any], b: Sequence[Any],
                     labels: Optional[Sequence[Any]] = None,
                     missing_values: Sequence[Any] = ()) -> pd.DataFrame:
    """Return the ``a`` × ``b`` contingency table over rows both labelled.

    :param a: first annotator's labels.
    :param b: second annotator's labels, row-aligned with ``a``.
    :param labels: label universe; inferred from the data when omitted.
    :param missing_values: extra values that count as abstentions.
    :returns: integer DataFrame, ``a`` labels on the index.
    """
    detail = kappa_detail(a, b, labels=labels, missing_values=missing_values)
    return detail.confusion


def kappa_detail(a: Sequence[Any], b: Sequence[Any],
                 labels: Optional[Sequence[Any]] = None,
                 missing_values: Sequence[Any] = (),
                 name_a: str = "a", name_b: str = "b") -> PairAgreement:
    """Compute Cohen's κ for ``a`` vs ``b`` and everything around it.

    Only rows both annotators labelled enter the calculation. Rows where
    exactly one of them abstained are counted into
    :attr:`PairAgreement.n_abstained`; they are *not* disagreements.

    :param a: first annotator's labels (``None``/NaN/"" = abstention).
    :param b: second annotator's labels, row-aligned with ``a``.
    :param labels: label universe; inferred from the compared rows when
        omitted. Passing it keeps confusion matrices comparable across pairs.
    :param missing_values: extra values that count as abstentions, e.g.
        ``(0,)`` for legacy databases where 0 meant "not looked at".
    :param name_a: name to record for the first column.
    :param name_b: name to record for the second column.
    :returns: a :class:`PairAgreement`.
    :raises ValueError: when the two sequences differ in length.
    """
    ya_all = _normalise(a, missing_values)
    yb_all = _normalise(b, missing_values)
    if len(ya_all) != len(yb_all):
        raise ValueError(
            f"annotation columns must be row-aligned: got {len(ya_all)} "
            f"and {len(yb_all)} values")

    ya: List[Any] = []
    yb: List[Any] = []
    n_abstained = 0
    n_neither = 0
    for va, vb in zip(ya_all, yb_all):
        if va is None and vb is None:
            n_neither += 1
        elif va is None or vb is None:
            n_abstained += 1
        else:
            ya.append(va)
            yb.append(vb)

    n = len(ya)
    if labels is None:
        universe = _sorted_labels(set(ya) | set(yb))
    else:
        universe = _sorted_labels({_scalar_label(l) for l in labels} - {None})
    if not universe:
        universe = []

    idx = {lab: i for i, lab in enumerate(universe)}
    counts = np.zeros((len(universe), len(universe)), dtype=np.int64)
    for va, vb in zip(ya, yb):
        if va not in idx or vb not in idx:
            raise ValueError(
                f"label {(va if va not in idx else vb)!r} is not in the "
                f"labels you passed ({', '.join(map(repr, universe))}).")
        counts[idx[va], idx[vb]] += 1
    confusion = pd.DataFrame(counts, index=list(universe),
                             columns=list(universe), dtype=np.int64)
    confusion.index.name = name_a
    confusion.columns.name = name_b

    n_agree = int(np.trace(counts)) if counts.size else 0
    n_disagree = n - n_agree

    def _pack(kappa: float, p_o: float, p_e: float, note: str) -> PairAgreement:
        return PairAgreement(
            column_a=name_a, column_b=name_b, kappa=kappa,
            percent_agreement=p_o, expected_agreement=p_e,
            n_compared=n, n_agree=n_agree, n_disagree=n_disagree,
            n_abstained=n_abstained, n_neither=n_neither,
            labels=list(universe), confusion=confusion, note=note,
            interpretation=interpret_kappa(kappa))

    if n == 0:
        return _pack(float("nan"), float("nan"), float("nan"),
                     "No rows where both annotators committed to a label — "
                     f"{n_abstained} row(s) have exactly one label and "
                     f"{n_neither} have none. κ needs overlap.")

    p_o = n_agree / n
    row_marg = counts.sum(axis=1) / n
    col_marg = counts.sum(axis=0) / n
    p_e = float(np.dot(row_marg, col_marg))

    a_classes = int(np.count_nonzero(counts.sum(axis=1)))
    b_classes = int(np.count_nonzero(counts.sum(axis=0)))

    # -- the no-variance traps ------------------------------------------
    if a_classes <= 1 and b_classes <= 1:
        if p_o >= 1.0 - _TOL:
            only = universe[int(np.argmax(counts.sum(axis=1)))]
            return _pack(
                float("nan"), p_o, p_e,
                f"κ is undefined here: both annotators put every one of the "
                f"{n} compared rows in class {only!r}. With no variance, "
                f"chance agreement is 100 % (pₑ=1) and κ's denominator "
                f"(1−pₑ) is zero — the answer is not 1.0, it is 'no "
                f"information'. Raw agreement is {p_o:.1%}.")
        return _pack(
            float("nan"), p_o, p_e,
            f"κ is undefined here: each annotator used exactly one class, "
            f"and a different one, over all {n} compared rows. The marginals "
            f"carry no variance, so the chance correction is degenerate. "
            f"Raw agreement is {p_o:.1%}.")
    if a_classes <= 1 or b_classes <= 1:
        flat = name_a if a_classes <= 1 else name_b
        only = universe[int(np.argmax(
            counts.sum(axis=1) if a_classes <= 1 else counts.sum(axis=0)))]
        return _pack(
            float("nan"), p_o, p_e,
            f"κ is degenerate here: {flat!r} assigned class {only!r} to all "
            f"{n} compared rows. When one annotator never varies, pₑ "
            f"collapses onto the other's marginal and κ is identically 0 "
            f"however well they agree — so 0 would be misleading. Raw "
            f"agreement is {p_o:.1%}.")
    # Past this point both annotators used >= 2 classes, so their marginals
    # are not unit vectors and pₑ < 1 strictly: the denominator is safe.
    kappa = (p_o - p_e) / (1.0 - p_e)
    note = ""
    if p_o >= 0.90 and abs(kappa) < 0.20:
        note = (f"Prevalence paradox: raw agreement is {p_o:.1%} but κ is "
                f"only {kappa:+.3f}, because one class dominates "
                f"({max(max(row_marg), max(col_marg)):.0%} of labels) so "
                f"chance agreement is already {p_e:.1%}. Both numbers are "
                f"real; quote them together.")
    return _pack(kappa, p_o, p_e, note)


def cohens_kappa(a: Sequence[Any], b: Sequence[Any],
                 labels: Optional[Sequence[Any]] = None,
                 missing_values: Sequence[Any] = ()) -> float:
    """Cohen's κ between two annotators' labels.

    Rows either annotator left unlabelled are dropped (abstention, not
    disagreement). Returns ``nan`` — never a flattering 0.0 or 1.0 —
    when the compared rows carry no variance; :func:`kappa_detail` gives
    the same number plus the reason.

    :param a: first annotator's labels.
    :param b: second annotator's labels, row-aligned with ``a``.
    :param labels: optional label universe.
    :param missing_values: extra values that count as abstentions.
    :returns: κ in ``[-1, 1]``, or ``nan`` when undefined.
    :raises ValueError: when the sequences differ in length.
    """
    return kappa_detail(a, b, labels=labels,
                        missing_values=missing_values).kappa


# ---------------------------------------------------------------------------
# Fleiss' kappa (3+ annotators)
# ---------------------------------------------------------------------------

def fleiss_kappa(matrix: Any) -> float:
    """Fleiss' κ from a subjects × categories count matrix.

    ``matrix[i][j]`` is how many annotators put subject ``i`` in category
    ``j``. Every row must sum to the same number of annotators ``n ≥ 2``.

    Fleiss' κ generalises *Scott's π* rather than Cohen's κ: it pools the
    annotators into one marginal distribution instead of giving each its
    own, so on two annotators it agrees with Scott's π and will usually
    differ slightly from :func:`cohens_kappa`.

    :param matrix: 2-D array-like of non-negative counts.
    :returns: κ, or ``nan`` when every rating fell in one category (no
        variance — pₑ=1, so κ has no denominator).
    :raises ValueError: for a ragged/empty matrix, negative counts, rows
        that disagree on the number of annotators, or fewer than 2
        annotators per subject.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            "fleiss_kappa needs a non-empty 2-D subjects × categories "
            f"count matrix; got shape {arr.shape}.")
    if np.any(arr < 0):
        raise ValueError("fleiss_kappa counts must be non-negative.")
    if not np.allclose(arr, np.round(arr)):
        raise ValueError("fleiss_kappa counts must be whole numbers.")
    per_subject = arr.sum(axis=1)
    n = per_subject[0]
    if not np.allclose(per_subject, n):
        raise ValueError(
            "every subject must be rated by the same number of annotators; "
            f"row sums range {per_subject.min():.0f}–{per_subject.max():.0f}. "
            "Restrict to rows every annotator labelled first.")
    if n < 2:
        raise ValueError(
            f"Fleiss' κ needs at least 2 annotators per subject, got {n:.0f}.")

    big_n = arr.shape[0]
    p_j = arr.sum(axis=0) / (big_n * n)
    p_i = (np.square(arr).sum(axis=1) - n) / (n * (n - 1))
    p_bar = float(p_i.mean())
    p_e = float(np.dot(p_j, p_j))
    if abs(1.0 - p_e) <= _TOL:
        return float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def _fleiss_per_category(arr: np.ndarray, j: int) -> float:
    """Fleiss' per-category κ for column ``j`` of a count matrix.

    The standard one-vs-rest decomposition of Fleiss' κ. Returns ``nan``
    when the category is never used or used for every rating (no variance).
    """
    big_n, _ = arr.shape
    n = arr.sum(axis=1)[0]
    p_j = arr[:, j].sum() / (big_n * n)
    q_j = 1.0 - p_j
    if p_j <= _TOL or q_j <= _TOL:
        return float("nan")
    disagreement = float((arr[:, j] * (n - arr[:, j])).sum())
    return 1.0 - disagreement / (big_n * n * (n - 1) * p_j * q_j)


# ---------------------------------------------------------------------------
# Database access — read-only, always
# ---------------------------------------------------------------------------

def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier (already schema-validated)."""
    return '"' + str(name).replace('"', '""') + '"'


def _read_only_uri(path: str) -> str:
    """``file:…?mode=ro`` URI — SQLite itself then refuses every write."""
    return "file:" + _urlquote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"


def _connect(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` read-only. Never writes, never journals.

    :raises FileNotFoundError: when the file is not there — sqlite's own
        "unable to open database file" says nothing about which file.
    """
    if not db_path or not str(db_path).strip():
        raise ValueError("No database path given.")
    path = os.path.abspath(os.path.expanduser(str(db_path).strip()))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such database: {path}")
    con = sqlite3.connect(_read_only_uri(path), uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def table_columns(db_path: str, table: str = PNG_TABLE) -> List[str]:
    """Return the column names of ``table``, in declaration order.

    :raises ValueError: when the database has no such table.
    """
    con = _connect(db_path)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()]
        if table not in names:
            raise ValueError(
                f"{os.path.basename(str(db_path))} has no {table!r} table "
                f"(found: {', '.join(names) or 'nothing'}). Annotations live "
                f"in {PNG_TABLE!r}; run Measure with save_png on first.")
        rows = con.execute(
            f"PRAGMA table_info({_quote_ident(table)})").fetchall()
    finally:
        con.close()
    return [r[1] for r in rows]


def annotation_columns(db_path: str, table: str = PNG_TABLE,
                       key: str = PNG_KEY, max_classes: int = 20,
                       min_labelled: int = 1,
                       include_model_columns: bool = False) -> List[str]:
    """Guess which columns of ``table`` hold **human** annotations.

    The Annotate app adds a plain ``INTEGER`` column per annotation pass,
    so an annotation column is one that is *not* part of the crop
    metadata, *not* written by a model, holds few distinct values, and has
    at least one non-NULL.

    The model exclusion is the point. A classifier writes into this same
    table — ``pred``/``cv_predictions`` from the CV stage,
    ``predictions``/``ml_pred`` from the ML one — and its class column is
    indistinguishable *by shape* from an annotation pass. Offering it made
    ``agreement_report`` score the classifier as a third annotator, which
    is a different question with the same units: on a real database, four
    "annotators" gave κ = -0.004 where the two humans agree at 0.471. See
    :data:`_MODEL_COLUMNS`.

    :param db_path: path to ``measurements.db``.
    :param table: table to inspect (default ``png_list``).
    :param key: row key, always excluded.
    :param max_classes: reject columns with more distinct values than
        this — a continuous measurement is not an annotation.
    :param min_labelled: reject columns with fewer labelled rows.
    :param include_model_columns: offer the model's own columns too. For
        the deliberate question "how well does the classifier agree with
        the annotators?", which is model validation, not inter-annotator
        agreement. Off by default because it is never the question
        somebody means when they ask for agreement between annotators.
    :returns: candidate column names, in table order.
    """
    columns = table_columns(db_path, table)
    con = _connect(db_path)
    out: List[str] = []
    try:
        for col in columns:
            if col == key or col in _METADATA_COLUMNS:
                continue
            if not include_model_columns and _is_model_column(col, columns):
                continue
            q = _quote_ident(col)
            n_labelled, n_distinct = con.execute(
                f"SELECT COUNT({q}), COUNT(DISTINCT {q}) "
                f"FROM {_quote_ident(table)}").fetchone()
            if not n_labelled or n_labelled < min_labelled:
                continue
            if n_distinct > max_classes:
                continue
            out.append(col)
    finally:
        con.close()
    return out


def load_annotations(db_path: str, columns: Sequence[str],
                     table: str = PNG_TABLE, key: str = PNG_KEY,
                     missing_values: Sequence[Any] = ()) -> pd.DataFrame:
    """Read ``key`` + ``columns`` from ``table`` into a normalised frame.

    Labels come back normalised: integers where the value is integral,
    ``None`` for every abstention. Row order is the table's own.

    :param db_path: path to ``measurements.db`` (opened read-only).
    :param columns: annotation columns to read.
    :param table: table holding them (default ``png_list``).
    :param key: row-identifying column (default ``png_path``).
    :param missing_values: extra values to treat as abstentions.
    :returns: DataFrame with ``key`` first, then one column per annotator.
    :raises ValueError: for an unknown table or column.
    """
    cols = list(dict.fromkeys(str(c) for c in columns))
    real = table_columns(db_path, table)
    if key not in real:
        raise ValueError(
            f"{table!r} has no {key!r} column — cannot line the annotators "
            f"up row by row.")
    unknown = [c for c in cols if c not in real]
    if unknown:
        raise ValueError(
            f"{table!r} has no column(s) {', '.join(map(repr, unknown))}. "
            f"Available: {', '.join(c for c in real if c != key)}")

    select = ", ".join(_quote_ident(c) for c in [key] + cols)
    con = _connect(db_path)
    try:
        rows = con.execute(
            f"SELECT {select} FROM {_quote_ident(table)}").fetchall()
    finally:
        con.close()

    # dtype=object throughout: a column of ints with NULLs would otherwise
    # come back as float64 with NaN, and "did this annotator abstain?"
    # would stop being an ``is None`` question.
    data: Dict[str, pd.Series] = {
        key: pd.Series([r[0] for r in rows], dtype=object)}
    for i, col in enumerate(cols, start=1):
        data[col] = pd.Series(_normalise((r[i] for r in rows), missing_values),
                              dtype=object)
    return pd.DataFrame(data, columns=[key] + cols)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class AgreementReport:
    """Everything :func:`agreement_report` worked out, in one object.

    :ivar pairs: one :class:`PairAgreement` per unordered column pair.
    :ivar overall_kappa: Cohen's κ for two annotators, Fleiss' κ for
        three or more (computed on rows *every* annotator labelled).
    :ivar per_class: one row per class — its one-vs-rest κ, how often the
        annotators were unanimous on it, and its prevalence. This is where
        "we agree on the negatives, we argue about the positives" shows up.
    :ivar n_complete: rows every annotator labelled.
    :ivar n_partial: rows some but not all labelled — abstentions, not
        disagreements.
    :ivar n_disagreements: rows where two annotators who both committed
        chose differently. This is the review queue's length.
    """

    db_path: str
    table: str
    key: str
    columns: List[str]
    pairs: List[PairAgreement]
    overall_kappa: float
    overall_method: str
    overall_note: str
    interpretation: str
    labels: List[Any]
    per_class: pd.DataFrame
    n_rows: int
    n_complete: int
    n_partial: int
    n_unlabelled: int
    n_disagreements: int
    percent_agreement: float
    convention: str = CONVENTION
    warnings: List[str] = field(default_factory=list)

    @property
    def n_annotators(self) -> int:
        return len(self.columns)

    @property
    def defined(self) -> bool:
        """False when the overall κ is ``nan``."""
        k = self.overall_kappa
        return not (k is None or math.isnan(float(k)))

    def pair(self, a: str, b: str) -> Optional[PairAgreement]:
        """Return the :class:`PairAgreement` for two columns, either order."""
        for p in self.pairs:
            if {p.column_a, p.column_b} == {a, b}:
                return p
        return None

    def kappa_table(self) -> pd.DataFrame:
        """The per-pair numbers as a DataFrame, ready to render."""
        return pd.DataFrame([{
            "annotator_a": p.column_a,
            "annotator_b": p.column_b,
            "n_compared": p.n_compared,
            "n_abstained": p.n_abstained,
            "percent_agreement": p.percent_agreement,
            "kappa": p.kappa,
            "interpretation": p.interpretation,
            "note": p.note,
        } for p in self.pairs])


def _label_counts(arr: np.ndarray, labels: Sequence[Any]) -> np.ndarray:
    """Subjects × categories count matrix for Fleiss.

    :param arr: object array of shape ``(subjects, annotators)``.
    :param labels: the category universe, in order.
    """
    out = np.zeros((arr.shape[0], len(labels)), dtype=float)
    for j, lab in enumerate(labels):
        out[:, j] = (arr == lab).sum(axis=1)
    return out


def _per_class_frame(arr: np.ndarray, n_annotators: int,
                     labels: Sequence[Any]) -> pd.DataFrame:
    """Per-class one-vs-rest agreement over the complete cases.

    Two annotators get a genuine one-vs-rest Cohen's κ (which, for a
    two-class problem, equals the overall κ — a useful sanity check).
    Three or more get Fleiss' per-category κ, the standard one-vs-rest
    decomposition of the overall Fleiss' κ.

    :param arr: object array of shape ``(complete rows, annotators)``.
    :param n_annotators: number of annotator columns.
    :param labels: the class universe.
    """
    rows = []
    n_rows = arr.shape[0]
    counts = _label_counts(arr, labels) if n_rows else None
    for j, lab in enumerate(labels):
        if n_rows == 0:
            kappa = float("nan")
            unanimous = any_used = 0
            prevalence = float("nan")
        else:
            hit = (arr == lab)
            unanimous = int((hit.sum(axis=1) == n_annotators).sum())
            any_used = int((hit.sum(axis=1) > 0).sum())
            prevalence = float(hit.sum()) / (n_rows * n_annotators)
            if n_annotators == 2:
                kappa = cohens_kappa(hit[:, 0].astype(int),
                                     hit[:, 1].astype(int), labels=(0, 1))
            else:
                kappa = _fleiss_per_category(counts, j)
        rows.append({
            "label": lab,
            "kappa": kappa,
            "interpretation": interpret_kappa(kappa),
            "n_unanimous": unanimous,
            "n_any": any_used,
            "prevalence": prevalence,
        })
    return pd.DataFrame(rows, columns=["label", "kappa", "interpretation",
                                       "n_unanimous", "n_any", "prevalence"])


def agreement_report(db_path: str, columns: Sequence[str],
                     table: str = PNG_TABLE, key: str = PNG_KEY,
                     missing_values: Sequence[Any] = (),
                     labels: Optional[Sequence[Any]] = None
                     ) -> AgreementReport:
    """Score how well two or more annotation columns agree.

    Each *pair* is scored on the rows both of its annotators labelled.
    The *overall* κ for three or more annotators is Fleiss' κ over the
    rows every annotator labelled, which is stricter — the report carries
    ``n_complete`` and ``n_partial`` so the difference is visible rather
    than silent.

    :param db_path: path to ``measurements.db``; opened read-only.
    :param columns: two or more annotation columns of ``table``.
    :param table: table holding them (default ``png_list``).
    :param key: row key (default ``png_path``).
    :param missing_values: extra values to treat as abstentions, e.g.
        ``(0,)`` for legacy databases where 0 meant "not looked at".
    :param labels: optional fixed label universe.
    :returns: an :class:`AgreementReport`.
    :raises ValueError: for fewer than two distinct columns, or an
        unknown table/column.
    """
    cols = list(dict.fromkeys(str(c) for c in columns))
    if len(cols) < 2:
        raise ValueError(
            "Agreement needs at least two annotation columns — one "
            "annotator cannot disagree with anybody. Add a second "
            f"annotation pass in the Annotate app (got: "
            f"{', '.join(cols) or 'nothing'}).")

    df = load_annotations(db_path, cols, table=table, key=key,
                          missing_values=missing_values)
    n_rows = len(df)

    observed = set()
    for col in cols:
        observed |= {v for v in df[col] if v is not None}
    universe = (_sorted_labels(observed) if labels is None
                else _sorted_labels({_scalar_label(l) for l in labels} - {None}))

    pairs = [kappa_detail(df[a], df[b], labels=universe,
                          name_a=a, name_b=b)
             for a, b in itertools.combinations(cols, 2)]

    # One pass over the table classifies every row as complete / partial /
    # untouched and counts the real disagreements. Partial rows are
    # abstentions: they never count towards n_disagreements unless the
    # annotators who *did* commit chose differently.
    values = df[cols].to_numpy(dtype=object)
    n_complete = n_partial = n_unlabelled = n_disagreements = unanimous = 0
    complete_rows: List[Any] = []
    for row in values:
        present = [v for v in row if v is not None]
        if not present:
            n_unlabelled += 1
            continue
        distinct = len(set(present))
        if len(present) == len(cols):
            n_complete += 1
            complete_rows.append(row)
            if distinct == 1:
                unanimous += 1
        else:
            n_partial += 1
        if distinct > 1:
            n_disagreements += 1
    complete_arr = (np.array(complete_rows, dtype=object)
                    if complete_rows
                    else np.empty((0, len(cols)), dtype=object))
    percent_agreement = unanimous / n_complete if n_complete else float("nan")

    warnings: List[str] = []
    if len(cols) == 2:
        method = "Cohen's κ"
        overall = pairs[0].kappa
        overall_note = pairs[0].note
    else:
        method = "Fleiss' κ"
        if n_complete == 0:
            overall = float("nan")
            overall_note = (
                "No row was labelled by all "
                f"{len(cols)} annotators, so Fleiss' κ has nothing to score. "
                "The pairwise κ values above still apply — each of those "
                "uses only the rows its own two annotators share.")
        else:
            arr = _label_counts(complete_arr, universe)
            overall = fleiss_kappa(arr)
            if math.isnan(overall):
                # n_complete > 0 means every one of those rows carries a
                # label from every annotator, so the universe is non-empty.
                only = universe[int(np.argmax(arr.sum(axis=0)))]
                overall_note = (
                    f"Fleiss' κ is undefined here: every one of the "
                    f"{n_complete} fully-labelled rows was put in class "
                    f"{only!r} by all {len(cols)} annotators. With no "
                    f"variance, chance agreement is 100 % and κ has no "
                    f"denominator — that is not the same as perfect "
                    f"agreement being meaningful.")
            else:
                overall_note = (
                    f"Fleiss' κ uses the {n_complete} row(s) all "
                    f"{len(cols)} annotators labelled; the pairwise κ values "
                    f"each use their own two annotators' shared rows.")
            if n_partial:
                warnings.append(
                    f"{n_partial} row(s) were labelled by some but not all "
                    f"annotators and are excluded from the overall Fleiss' κ "
                    f"(they are abstentions, not disagreements).")

    per_class = _per_class_frame(complete_arr, len(cols), universe)

    if not math.isnan(percent_agreement) and percent_agreement >= 0.90 and \
            (math.isnan(float(overall)) or abs(float(overall)) < 0.20):
        biggest = per_class["prevalence"].max() if len(per_class) else float("nan")
        warnings.append(
            f"Raw agreement is {percent_agreement:.1%} but κ is "
            f"{'undefined' if math.isnan(float(overall)) else f'{overall:+.3f}'}"
            f" — the prevalence paradox. One class takes "
            f"{biggest:.0%} of all labels, so agreeing by chance is already "
            f"easy and κ has little room above it. Report both numbers.")
    if n_complete and n_complete < 30:
        warnings.append(
            f"Only {n_complete} row(s) are labelled by every annotator; κ is "
            f"very noisy at that size. Treat the value as indicative.")

    # A caller can always name columns explicitly, and the Qt screen lets one
    # be ticked. Saying so is the difference between a deliberate model
    # validation and a κ quoted as inter-annotator agreement that is not one.
    model_cols = [c for c in cols if _is_model_column(c, table_columns(db_path, table))]
    if model_cols:
        warnings.append(
            f"{', '.join(model_cols)} {'is' if len(model_cols) == 1 else 'are'} "
            f"written by a model, not by an annotator, so this κ measures how "
            f"far the classifier is from the people — not how far the people "
            f"are from each other. Drop "
            f"{'it' if len(model_cols) == 1 else 'them'} for inter-annotator "
            f"agreement.")

    return AgreementReport(
        db_path=str(db_path), table=table, key=key, columns=cols,
        pairs=pairs, overall_kappa=float(overall), overall_method=method,
        overall_note=overall_note, interpretation=interpret_kappa(overall),
        labels=list(universe), per_class=per_class, n_rows=n_rows,
        n_complete=n_complete, n_partial=n_partial,
        n_unlabelled=n_unlabelled, n_disagreements=n_disagreements,
        percent_agreement=percent_agreement, warnings=warnings)


# ---------------------------------------------------------------------------
# Disagreement review
# ---------------------------------------------------------------------------

def disagreements(db_path: str, columns: Sequence[str],
                  table: str = PNG_TABLE, key: str = PNG_KEY,
                  missing_values: Sequence[Any] = (),
                  complete_only: bool = False,
                  limit: Optional[int] = None) -> pd.DataFrame:
    """Return the rows the annotators labelled *differently*.

    A row is a disagreement when at least two annotators committed to a
    label and those labels are not all the same. A row where one
    annotator abstained is **not** a disagreement — by default it is
    simply scored on whoever did label it, and dropped entirely if that
    leaves fewer than two labels.

    :param db_path: path to ``measurements.db``; opened read-only.
    :param columns: two or more annotation columns.
    :param table: table holding them (default ``png_list``).
    :param key: row key (default ``png_path``) — the crop to look at.
    :param missing_values: extra values to treat as abstentions.
    :param complete_only: when True, only consider rows every annotator
        labelled.
    :param limit: cap the number of returned rows (the review list can be
        long); ``None`` returns all of them.
    :returns: DataFrame with ``key``, one column per annotator, plus
        ``n_labelled`` and ``n_classes``, in table order.
    :raises ValueError: for fewer than two distinct columns, or an
        unknown table/column.
    """
    cols = list(dict.fromkeys(str(c) for c in columns))
    if len(cols) < 2:
        raise ValueError(
            "Reviewing disagreements needs at least two annotation columns "
            f"(got: {', '.join(cols) or 'nothing'}).")
    df = load_annotations(db_path, cols, table=table, key=key,
                          missing_values=missing_values)

    keep: List[int] = []
    n_labelled: List[int] = []
    n_classes: List[int] = []
    matrix = df[cols].to_numpy(dtype=object)
    for i, row in enumerate(matrix):
        present = [v for v in row if v is not None]
        if complete_only and len(present) != len(cols):
            continue
        if len(present) < 2:
            continue
        distinct = len(set(present))
        if distinct < 2:
            continue
        keep.append(i)
        n_labelled.append(len(present))
        n_classes.append(distinct)
        if limit is not None and len(keep) >= int(limit):
            break

    out = df.iloc[keep][[key] + cols].reset_index(drop=True)
    out["n_labelled"] = n_labelled
    out["n_classes"] = n_classes
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_kappa(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "undefined"
    return "undefined" if math.isnan(v) else f"{v:+.3f}"


def _fmt_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.1%}"


def format_agreement(report: AgreementReport) -> str:
    """Render an :class:`AgreementReport` as a plain-text block.

    Every κ is printed next to its raw percent agreement and the number
    of rows behind it, so an undefined or paradoxical κ reads as
    information rather than as a missing value.

    :param report: the report to render.
    :returns: multi-line text, no trailing newline.
    """
    lines: List[str] = []
    lines.append(f"Annotator agreement — {os.path.basename(report.db_path)} "
                 f"[{report.table}]")
    lines.append(f"Annotators ({report.n_annotators}): "
                 f"{', '.join(report.columns)}")
    lines.append(
        f"Rows: {report.n_rows} total · {report.n_complete} labelled by all · "
        f"{report.n_partial} partially labelled (abstentions) · "
        f"{report.n_unlabelled} untouched")
    lines.append(
        f"Overall {report.overall_method}: {_fmt_kappa(report.overall_kappa)} "
        f"({report.interpretation})   raw agreement "
        f"{_fmt_pct(report.percent_agreement)}   "
        f"disagreements: {report.n_disagreements}")
    if report.overall_note:
        lines.append(f"  ! {report.overall_note}")
    lines.append("")

    lines.append("Per pair (each on the rows both annotators labelled):")
    header = (f"  {'annotator a':<20}{'annotator b':<20}{'n':>7}"
              f"{'abstain':>9}{'raw':>9}{'kappa':>10}  interpretation")
    lines.append(header)
    for p in report.pairs:
        lines.append(
            f"  {p.column_a:<20}{p.column_b:<20}{p.n_compared:>7}"
            f"{p.n_abstained:>9}{_fmt_pct(p.percent_agreement):>9}"
            f"{_fmt_kappa(p.kappa):>10}  {p.interpretation}")
        if p.note:
            lines.append(f"      ! {p.note}")
    lines.append("")

    if len(report.pairs) == 1:
        p = report.pairs[0]
        lines.append(f"Confusion matrix (rows = {p.column_a}, "
                     f"columns = {p.column_b}):")
        for line in p.confusion.to_string().splitlines():
            lines.append("  " + line)
        lines.append("")

    if len(report.per_class):
        lines.append("Per class (one-vs-rest, on rows all annotators labelled):")
        lines.append(f"  {'class':<12}{'kappa':>10}{'unanimous':>11}"
                     f"{'any':>7}{'prevalence':>12}  interpretation")
        for _, row in report.per_class.iterrows():
            lines.append(
                f"  {str(row['label']):<12}{_fmt_kappa(row['kappa']):>10}"
                f"{int(row['n_unanimous']):>11}{int(row['n_any']):>7}"
                f"{_fmt_pct(row['prevalence']):>12}  {row['interpretation']}")
        lines.append("")

    for w in report.warnings:
        lines.append(f"! {w}")
    if report.warnings:
        lines.append("")
    lines.append(report.convention)
    return "\n".join(lines)
