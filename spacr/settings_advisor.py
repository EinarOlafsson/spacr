"""Inspect screen tables and propose regression settings.

The advisor separates measured choices, settings that remain undecided, and
questions the data cannot answer. Each proposed value includes its reason,
and this module never writes settings itself.

Count-table summaries use every available well. Object-level score tables
may be large, so response diagnostics use a capped sample and mark the
resulting :class:`Reading` accordingly. Family, control, and fraction choices
reuse the same analysis helpers as the regression pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: How many object rows the response is read from before the sample is
#: declared capped. Large enough that a proportion's range, boundedness and
#: shape are settled; small enough that pressing the button is not a minute.
ROW_CAP = 400_000

#: Column names the dependent variable is looked for under, when the caller
#: names none. `pred` is what `generate_ml_scores` writes.
DEFAULT_RESPONSES: Tuple[str, ...] = ("pred", "prediction", "predictions",
                                      "score", "recruitment")


@dataclass(frozen=True)
class Choice:
    """One setting the data decided, and what decided it.

    :param key: the setting name, exactly as the panel spells it.
    :param value: what to put in it.
    :param why: the reason, naming the measurement. Not "recommended" -- a
        reason a user cannot check against their own data is a slogan.
    """

    key: str
    value: Any
    why: str


@dataclass(frozen=True)
class Undecided:
    """A setting this module will not guess, and why not.

    :ivar key: setting name left unresolved.
    :ivar why: evidence-based reason the setting cannot be inferred.
    """

    key: str
    why: str


@dataclass(frozen=True)
class Question:
    """Something the data cannot answer.

    :param key: an identifier for the answer, not a setting name -- one
        answer can move several settings.
    :param prompt: the question shown to the user.
    :param kind: ``'number'`` or ``'choice'``.
    :param options: for ``'choice'``, ``((value, label), ...)``.
    :param default: the starting value, which is a position and is defended
        in ``why_it_matters`` rather than left as an unexplained number.
    :param why_it_matters: what changes in the settings depending on the
        answer. Shown beside the question, because a user who cannot see
        what a question buys cannot answer it well.
    """

    key: str
    prompt: str
    kind: str = "choice"
    options: Tuple[Tuple[Any, str], ...] = ()
    default: Any = None
    why_it_matters: str = ""


@dataclass(frozen=True)
class Reading:
    """What was measured, before any advice is derived from it.

    Separate from :class:`Advice` on purpose: the reading is checkable
    against the user's own data, and the advice is an argument FROM it. A
    reader who disagrees with a choice can see which number produced it.
    """

    plates: int = 0
    wells: int = 0
    guides: int = 0
    genes: int = 0
    rows: int = 0
    columns: int = 0
    response: str = ""
    n_response: int = 0
    low: Optional[float] = None
    high: Optional[float] = None
    inside_unit: bool = False
    on_unit: bool = False
    binary: bool = False
    integral: bool = False
    normal_p: Optional[float] = None
    skew: Optional[float] = None
    wells_per_guide: Optional[float] = None
    guides_per_gene: Optional[float] = None
    objects_per_well: Optional[float] = None
    fraction_median: Optional[float] = None
    fraction_q90: Optional[float] = None
    guides_per_well: Optional[float] = None
    kept_at_two_percent: Optional[float] = None
    capped: bool = False
    trouble: Tuple[str, ...] = ()

    # ---- what only a finished fit knows (instruction 226) ----------------
    #: The run folder these came from, or "". NAMED, because advice derived
    #: from a fit whose settings have since changed is advice about a
    #: different screen, and the reader has to be able to disbelieve it.
    run_folder: str = ""
    #: Why a run that WAS found is not being used. A stale summary is worse
    #: than none, because it looks like measurement.
    run_note: str = ""
    #: The residuals' own shape, which is what the normality assumption is
    #: actually about -- a response can be skewed while the residuals are
    #: fine, and normal while they are not.
    residual_normal_p: Optional[float] = None
    residual_kurtosis: Optional[float] = None
    durbin_watson: Optional[float] = None
    max_cooks_distance: Optional[float] = None
    max_vif: Optional[float] = None

    @property
    def read_a_run(self) -> bool:
        """Whether a previous run's diagnostics were read."""
        return bool(self.run_folder) and not self.run_note

    @property
    def read_the_counts(self) -> bool:
        """Whether count-table wells and guides were measured."""
        return self.wells > 0 and self.guides > 0

    @property
    def read_the_response(self) -> bool:
        """Whether the response count and numeric range were measured."""
        return (self.n_response > 0 and self.low is not None
                and self.high is not None)

    def sample_note(self) -> str:
        """How the response numbers should be qualified, or ``''``."""
        if not self.capped:
            return ""
        return (f"from the first {self.n_response:,} object row(s); the score "
                f"table is larger than the {ROW_CAP:,}-row sample this reads")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _columns_of(path: str, *, canonical: bool = False) -> Tuple[str, ...]:
    """The header of a table, without reading a row of it.

    THROUGH `spacr.tabular` (145) either way. `canonical=False` is the
    DEFAULT and it is what `usecols` needs: pandas selects columns by the
    name in the FILE, before any renaming, so asking a file that spells its
    column `col` for `columnID` selects nothing and raises. `canonical=True`
    answers "what will the frame's columns be called once it is read", which
    is the question every line after the read is asking.

    Silent -- `report=None` -- because a header read is not the moment to
    print a column-collision note the caller has not asked for yet.
    """
    from .tabular import read_table

    try:
        head = read_table(path, nrows=0, canonicalise=canonical, report=None)
    except Exception:                                        # noqa: BLE001
        return ()
    return tuple(str(c) for c in head.columns)


def _well_key(frame: "pd.DataFrame") -> Optional["pd.Series"]:
    """One well name per row, in whichever spelling the table uses."""
    columns = {str(c) for c in frame.columns}
    for single in ("prc", "prcf"):
        if single in columns:
            return frame[single].astype(str).str.rsplit("_f", n=1).str[0]
    trio = [("plateID", "rowID", "columnID"), ("plate", "row", "col"),
            ("plate", "row_name", "column_name")]
    for keys in trio:
        if all(k in columns for k in keys):
            parts = [frame[k].astype(str) for k in keys]
            joined = parts[0]
            for part in parts[1:]:
                joined = joined + "_" + part
            return joined
    return None


def _plate_of(wells: "pd.Series") -> "pd.Series":
    """The plate half of a `plate_row_column` well name."""
    return wells.astype(str).str.split("_").str[0]


_NO_USABLE_FRACTIONS = (
    "the count tables contain no positive guide fractions, so no well has "
    "usable reads for a threshold"
)


def read_the_counts(paths: Sequence[str]) -> Dict[str, Any]:
    """Measure plates, wells, guides, genes, and replication from count data.

    All count rows are used. Per-well fractions are calculated with
    :func:`spacr.cell_montage.fractions_from_counts`, matching the values used
    by the regression pipeline.

    :param paths: count-table paths to read together as one screen design.
    """
    from .cell_montage import fractions_from_counts
    from .control_names import common_prefix
    from .gene_measurement_sweep import gene_of_guide

    out: Dict[str, Any] = {"trouble": []}
    live = [str(p) for p in (paths or ()) if p and os.path.isfile(str(p))]
    if not live:
        out["trouble"].append("no count table is attached, so the design "
                              "could not be measured")
        return out
    try:
        frame = fractions_from_counts(live)
    except Exception as exc:                                 # noqa: BLE001
        out["trouble"].append(f"the count tables could not be read: {exc}")
        return out
    if not len(frame):
        out["trouble"].append("the count tables are empty")
        return out

    wells = frame["prc"].astype(str)
    guides = frame["grna"].astype(str)
    out["plates"] = int(_plate_of(wells).nunique())
    out["wells"] = int(wells.nunique())
    out["guides"] = int(guides.nunique())
    # r1/c1 out of `plate1_r1_c1`, so a one-row or one-column screen can be
    # recognised -- `model_plate_position` has nothing to model on one of
    # either.
    parts = wells.str.split("_")
    out["rows"] = int(parts.str[1].nunique()) if parts.str.len().min() > 1 else 0
    out["columns"] = int(parts.str[2].nunique()) if parts.str.len().min() > 2 else 0
    # THE FRACTION DISTRIBUTION ITSELF, not only its shape. It was computed
    # here and thrown away, and it is the only thing that can say whether a
    # `fraction_threshold` keeps a library or deletes it.
    try:
        share = pd.to_numeric(frame["fraction"], errors="coerce")
        share = share[share.notna() & (share > 0)]
        if len(share):
            out["fraction_median"] = float(share.median())
            out["fraction_q90"] = float(share.quantile(0.90))
            out["guides_per_well"] = float(
                frame.groupby("prc")["grna"].nunique().median())
            # What the usual default would cost THIS screen, which is the
            # number a user can act on.
            out["kept_at_two_percent"] = float((share >= 0.02).mean())
        else:
            out["trouble"].append(_NO_USABLE_FRACTIONS)
    except Exception:                                        # noqa: BLE001
        pass

    per_guide = frame.groupby("grna")["prc"].nunique()
    out["wells_per_guide"] = float(per_guide.median()) if len(per_guide) else None

    # THE GENE OF A GUIDE THROUGH THE ONE READER (145/184). `gene_of_guide`
    # measures the organism prefix itself rather than assuming `TGGT1_`, so a
    # Plasmodium or a human library is not pooled into one gene; the prefix
    # is measured here too, for the note this reading carries.
    prefix = common_prefix([str(g) for g in guides.unique()])
    genes = pd.Series([gene_of_guide(g, prefix=prefix) or "" for g in guides],
                      index=frame.index).astype(str)
    named = genes[genes.str.len() > 0]
    if len(named):
        out["genes"] = int(named.nunique())
        by_gene = (pd.DataFrame({"gene": named, "grna": guides.loc[named.index]})
                   .groupby("gene")["grna"].nunique())
        out["guides_per_gene"] = float(by_gene.median()) if len(by_gene) else None
    return out


def read_the_response(paths: Sequence[str], dependent_variable: str = "",
                      *, row_cap: int = ROW_CAP) -> Dict[str, Any]:
    """Measure response range, shape, and per-well support.

    At most ``row_cap`` object rows are read across the supplied score tables.
    The returned mapping records whether that cap was reached so callers can
    qualify sample-derived recommendations.

    Parameters
    ----------
    paths : sequence of str
        Score tables to inspect.
    dependent_variable : str, optional
        Response column. Common generated-score names are tried when omitted.
    row_cap : int, optional
        Maximum number of object rows to inspect.

    Returns
    -------
    dict
        Measured response properties and any non-fatal problems in
        ``"trouble"``.
    """
    from .tabular import read_table

    out: Dict[str, Any] = {"trouble": []}
    live = [str(p) for p in (paths or ()) if p and os.path.isfile(str(p))]
    if not live:
        out["trouble"].append("no score table is attached, so the response "
                              "could not be measured")
        return out

    columns = _columns_of(live[0])
    wanted = str(dependent_variable or "")
    if wanted and wanted not in columns:
        out["trouble"].append(
            f"{wanted!r} is not a column of {os.path.basename(live[0])}; "
            f"the response was not measured")
        return out
    if not wanted:
        wanted = next((c for c in DEFAULT_RESPONSES if c in columns), "")
    if not wanted:
        out["trouble"].append(
            "no dependent variable is named and none of "
            f"{', '.join(DEFAULT_RESPONSES)} is a column")
        return out
    out["response"] = wanted

    #: The well-naming columns, in every spelling a score table uses. BOTH
    #: the canonical names and the raw ones, because `usecols` is matched
    #: against the file's own header and one plate of the reference screen
    #: writes `col` where another writes nothing at all.
    naming = ("prc", "prcf", "plateID", "rowID", "columnID",
              "plate", "row", "col", "Plate", "Row", "Column", "Well",
              "row_name", "column_name")
    frames, taken, seen = [], 0, 0
    for path in live:
        if taken >= row_cap:
            out["capped"] = True
            break
        # EACH FILE'S OWN HEADER. Taking the columns off the FIRST file and
        # asking every other file for them is how plates 2, 3 and 4 of the
        # reference screen were dropped: plate 1 carries `col` and the others
        # do not, so `usecols` raised on each of them and the response was
        # measured from one plate while the reading said four.
        here = _columns_of(path)
        if wanted not in here:
            out["trouble"].append(
                f"{os.path.basename(path)} has no {wanted!r} column")
            continue
        keys = [c for c in naming if c in here]
        try:
            # THE ONE READER (145), with `usecols` and `nrows` passed
            # through: `read_table` forwards its kwargs to pandas, so the
            # cap this module needs costs nothing to keep.
            piece = read_table(path, usecols=[wanted] + keys,
                               nrows=row_cap - taken, report=None)
        except Exception as exc:                             # noqa: BLE001
            out["trouble"].append(f"{os.path.basename(path)}: {exc}")
            continue
        taken += len(piece)
        seen += 1
        frames.append(piece)
        if len(piece) >= row_cap:
            out["capped"] = True
    out["score_files_read"] = seen
    if not frames:
        return out

    # THE COLUMNS DIFFER BETWEEN PLATES, so the concatenation is on the
    # union and a key missing from one file is NaN there rather than an
    # error. `_well_key` picks whichever spelling is complete.
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 \
        else frames[0]
    values = pd.to_numeric(frame[wanted], errors="coerce").dropna()
    if not len(values):
        out["trouble"].append(f"{wanted!r} holds no number in the sample")
        return out

    # THE WELL IS THE UNIT THE FIT SEES. A per-object response is aggregated
    # to wells before the model touches it, so the family question is about
    # the WELL means -- and the object-level spread, which is much wider, is
    # not the distribution being modelled.
    where = _well_key(frame)
    if where is not None:
        per_well = values.groupby(where.loc[values.index]).mean()
        out["objects_per_well"] = float(
            values.groupby(where.loc[values.index]).size().median())
    else:
        per_well = values
        out["trouble"].append(
            "the score table names no well, so the response was read at the "
            "object level rather than the well level the fit uses")

    array = np.asarray(per_well, dtype=float)
    out["n_response"] = int(len(values))
    out["low"] = float(np.min(array))
    out["high"] = float(np.max(array))
    out["binary"] = bool(np.all((array == 0) | (array == 1)))
    out["inside_unit"] = bool((array > 0).all() and (array < 1).all())
    out["on_unit"] = bool((array >= 0).all() and (array <= 1).all())
    out["integral"] = bool((array >= 0).all()
                           and np.all(array.astype(np.int64) == array))
    try:
        from scipy.stats import normaltest, skew

        if len(array) >= 8:
            out["normal_p"] = float(normaltest(array).pvalue)
        out["skew"] = float(skew(array))
    except Exception:                                        # noqa: BLE001
        pass
    return out


def read_the_screen(counts: Sequence[str] = (), scores: Sequence[str] = (),
                    dependent_variable: str = "",
                    *, row_cap: int = ROW_CAP) -> Reading:
    """Measure both halves of the input and return one :class:`Reading`."""
    got: Dict[str, Any] = {}
    trouble: List[str] = []
    for part in (read_the_counts(counts),
                 read_the_response(scores, dependent_variable,
                                   row_cap=row_cap)):
        trouble.extend(part.pop("trouble", []) or [])
        got.update({k: v for k, v in part.items() if v is not None})
    got["trouble"] = tuple(trouble)
    fields = {f for f in Reading.__dataclass_fields__}
    return Reading(**{k: v for k, v in got.items() if k in fields})


# ---------------------------------------------------------------------------
# The questions the data cannot answer
# ---------------------------------------------------------------------------

#: Questions whose answers cannot be inferred reliably from the input tables.
#: The first captures the expected hit rate, a prior supplied by the user
#: rather than estimated from the observed screen.
QUESTIONS: Tuple[Question, ...] = (
    Question(
        key="hits_per_thousand",
        prompt="Out of 1,000 perturbed genes, how many do you expect to be "
               "hits?",
        kind="number",
        default=20,
        why_it_matters=(
            "This prior informs the multiple-testing threshold. A screen "
            "expected to contain 5 hits per 1,000 tests requires a different "
            "threshold from one expected to contain 200; the input tables do "
            "not provide this experimental assumption."),
    ),
    Question(
        key="direction",
        prompt="What counts as a hit?",
        kind="choice",
        options=(("either", "any change, up or down"),
                 ("up", "an increase only"),
                 ("down", "a decrease only")),
        default="either",
        why_it_matters=(
            "A directional hypothesis supports a one-sided test. A two-sided "
            "test allocates significance probability to both directions and "
            "therefore has less power for the prespecified direction."),
    ),
    Question(
        key="controls",
        prompt="What are the non-cutting controls called? Leave blank if "
               "there are none.",
        kind="text",
        default="",
        why_it_matters=(
            "Non-cutting guides define the screen's empirical null and are "
            "distinct from negative-control gene knockouts. A gene name "
            "selects all associated guides; a guide name selects only that "
            "guide. Specifying these controls estimates effects relative to "
            "the empirical null rather than zero."),
    ),
    Question(
        key="cost",
        prompt="How should false positives and false negatives be weighted?",
        kind="choice",
        options=(("precision", "false positives are more costly"),
                 ("balanced", "similar cost"),
                 ("recall", "false negatives are more costly")),
        default="balanced",
        why_it_matters=(
            "This choice adjusts the significance threshold and, when recall "
            "is prioritized, the multiple-testing correction. The appropriate "
            "balance depends on the screen objective and the cost of "
            "downstream validation."),
    ),
)


def questions_for(reading: Reading) -> Tuple[Question, ...]:
    """Return questions that cannot be answered from this screen's data.

    The direction question is omitted for a binary response, where only an
    increase can represent the positive outcome.

    :param reading: measured screen properties used to omit answered questions.
    """
    out = []
    for question in QUESTIONS:
        if question.key == "direction" and reading.binary:
            continue
        out.append(question)
    return tuple(out)


# ---------------------------------------------------------------------------
# The advice
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Advice:
    """What the data decided, what it would not, and what it was read from."""

    chosen: Tuple[Choice, ...] = ()
    undecided: Tuple[Undecided, ...] = ()
    reading: Optional[Reading] = None

    def as_settings(self) -> Dict[str, Any]:
        """``{key: value}`` -- what would be written if this is accepted."""
        return {c.key: c.value for c in self.chosen}

    def why(self, key: str) -> str:
        """The reason for one key, or ``''``.

        :param key: chosen or unresolved setting name to look up.
        """
        for choice in self.chosen:
            if choice.key == key:
                return choice.why
        for skipped in self.undecided:
            if skipped.key == key:
                return skipped.why
        return ""


def _family_and_transform(reading: Reading, chosen: List[Choice],
                          undecided: List[Undecided]) -> None:
    """regression_type, transform and glm_transform_conflict, TOGETHER.

    182's whole point: a link-carrying family and a link-like transform are
    two ways of doing one job, and applying both fits logit(log(y)). So they
    are decided in one place and the reason says which of the two is doing
    the transforming -- rather than being chosen by two rules that can
    disagree.
    """
    if not reading.read_the_response:
        undecided.append(Undecided(
            "regression_type",
            "the response was not read, so its distribution decided nothing. "
            "Attach the score table and press this again."))
        return

    note = reading.sample_note()
    where = f" ({note})" if note else ""
    span = f"[{reading.low:.4g}, {reading.high:.4g}]"

    if reading.binary:
        chosen.append(Choice(
            "regression_type", "logit",
            f"the well response {reading.response!r} is 0 or 1 and nothing "
            f"else{where}, which is a binomial mean"))
    elif reading.inside_unit:
        # `check_distribution`'s own answer, and the reason names the number
        # it turns on rather than repeating the recommendation.
        chosen.append(Choice(
            "regression_type", "beta",
            f"the well response is strictly inside (0, 1) — {span} — so it "
            f"is a bounded proportion, and beta regression models the "
            f"variance of one directly{where}"))
    elif reading.on_unit:
        chosen.append(Choice(
            "regression_type", "quasi_binomial",
            f"the well response reaches a boundary of [0, 1] — {span} — "
            f"where beta's density is undefined, so the quasi-binomial "
            f"is the bounded model that admits it{where}"))
    elif reading.integral:
        chosen.append(Choice(
            "regression_type", "glm",
            f"the well response is a non-negative whole number — {span} — "
            f"which is a count, and a Poisson GLM with a log link is the "
            f"model for one{where}"))
    elif reading.normal_p is not None and reading.normal_p < 0.05:
        chosen.append(Choice(
            "regression_type", "rlm",
            f"the well response is unbounded and NOT normal "
            f"(D'Agostino p = {reading.normal_p:.2g}, skew "
            f"{reading.skew:+.2f}), so a robust fit is the one that is not "
            f"led by the tail{where}"))
    else:
        chosen.append(Choice(
            "regression_type", "ols",
            f"the well response is unbounded and consistent with normal "
            f"(D'Agostino p = {reading.normal_p:.2g}){where}"
            if reading.normal_p is not None else
            f"the well response is unbounded — {span}{where}"))

    # THE TRANSFORM IS PART OF THE SAME DECISION. Every bounded family above
    # carries its own link, so a transform on top of it is the double one 182
    # exists to prevent.
    family = next(c.value for c in chosen if c.key == "regression_type")
    if family in ("logit", "beta", "quasi_binomial", "glm"):
        chosen.append(Choice(
            "transform", None,
            f"none: regression_type={family!r} carries its own link, and a "
            f"log or logit transform on top of it would fit the response "
            f"twice — logit(log(y))"))
        chosen.append(Choice(
            "glm_transform_conflict", "untransformed",
            "the family does the transforming, so the response is fitted as "
            "measured — which is what makes the printed pseudo-R-squared "
            "describe the response you have"))
    elif reading.skew is not None and abs(reading.skew) > 1.0 \
            and reading.low is not None and reading.low > 0:
        chosen.append(Choice(
            "transform", "log",
            f"the response is strictly positive and skewed "
            f"{reading.skew:+.2f}, and {family!r} carries no link of its own, "
            f"so the transform is where the skew is handled"))
    else:
        chosen.append(Choice(
            "transform", None,
            f"none: {family!r} fits the response on its own scale and the "
            f"skew ({reading.skew:+.2f}) does not call for one"
            if reading.skew is not None else
            f"none: {family!r} fits the response on its own scale"))


def _level(reading: Reading, chosen: List[Choice],
           undecided: List[Undecided]) -> None:
    """guide, gene, or both -- from what the library can actually identify."""
    if not reading.read_the_counts:
        undecided.append(Undecided(
            "level", "the count tables were not read, so the number of "
                     "guides per gene is unknown and neither level can be "
                     "argued for"))
        return
    per_gene = reading.guides_per_gene
    if per_gene is None:
        chosen.append(Choice(
            "level", "grna",
            f"{reading.guides:,} guide(s) and no gene could be read off "
            f"their names, so only the guide level is identifiable"))
        return
    if per_gene >= 2:
        chosen.append(Choice(
            "level", "both",
            f"a gene carries {per_gene:.0f} guides (median) across "
            f"{reading.genes:,} genes, so the gene level has replicates to "
            f"pool AND the guide level is worth seeing beside it — the two "
            f"are fitted separately because one design holding both is "
            f"rank-deficient"))
    else:
        chosen.append(Choice(
            "level", "grna",
            f"a gene carries {per_gene:.0f} guide (median), so a gene "
            f"coefficient would be its single guide's under another name"))


def _plate(reading: Reading, chosen: List[Choice],
           undecided: List[Undecided]) -> None:
    """Batch correction and plate position, both gated on what is there."""
    if not reading.read_the_counts:
        undecided.append(Undecided(
            "batch_correction",
            "the count tables were not read, so the number of plates is "
            "unknown"))
        return
    if reading.plates <= 1:
        chosen.append(Choice(
            "batch_correction", "none",
            f"there is {reading.plates} plate, and a batch correction needs "
            f"at least two batches to estimate anything"))
    else:
        # PER-PLATE CENTRING, which needs nothing but the plate. It removes
        # the plate's own mean and estimates NOTHING from the residuals, so
        # there is no design for it to mistake signal for noise against.
        chosen.append(Choice(
            "batch_correction", "center",
            f"{reading.plates} plates, so plate is a batch the response "
            f"carries; per-plate centring removes it without estimating "
            f"anything from the residuals"))
        # `control_center` IS THE BETTER ONE AND IS NOT PROPOSED, because it
        # centres each plate on its CONTROL WELLS -- values in a column --
        # and which wells those are is not in the count or score tables.
        # Naming it here is the difference between a default and a ceiling.
        undecided.append(Undecided(
            "batch_control_values",
            "left alone. `batch_correction='control_center'` is the stronger "
            "correction -- each plate centred on its own controls rather "
            "than on all its wells -- but it needs to know WHICH WELLS hold "
            "them, and that is not in these tables. Set batch_control_column "
            "and batch_control_values and it becomes available."))

    # NOT ComBat, AND THAT IS A DECISION (196). ComBat estimates the plate
    # effect from whatever the design does not explain, and it REFUSES to run
    # until the caller says which biology to protect from that -- correctly,
    # because in a pooled screen the biology is the per-well GUIDE
    # COMPOSITION, which is continuous and is not a categorical covariate
    # column. There is nothing honest to pass, so proposing ComBat means
    # proposing a run that either refuses or removes the effects being
    # looked for.
    #
    # This module used to propose it anyway, with no covariate. The proposal
    # was accepted, the run was pressed, and it failed on the refusal --
    # which is the whole reason 196 exists.
    if reading.plates > 1:
        undecided.append(Undecided(
            "batch_covariate_column",
            "not needed: the proposed correction estimates nothing from the "
            "residuals, so there is no biology for a covariate to protect. "
            "It is required only by ComBat, which is not proposed here for "
            "exactly that reason -- in a pooled screen the signal is the "
            "per-well guide composition, and that is not a column."))
    if reading.rows > 1 and reading.columns > 1:
        chosen.append(Choice(
            "model_plate_position", True,
            f"the screen spans {reading.rows} rows and {reading.columns} "
            f"columns, so edge and gradient effects have somewhere to live "
            f"and leaving them out puts them in the residual"))
    else:
        chosen.append(Choice(
            "model_plate_position", False,
            f"the screen spans {reading.rows} row(s) and "
            f"{reading.columns} column(s); a position term needs more than "
            f"one of each to estimate"))


def _inference(reading: Reading, chosen: List[Choice],
               undecided: List[Undecided]) -> None:
    """Parametric or not, argued from normality and from the well count."""
    if reading.normal_p is None:
        undecided.append(Undecided(
            "inference",
            "the response's shape was not measured, and the choice between "
            "a parametric and a permutation null turns on it"))
        return
    if reading.normal_p < 0.05 and (reading.wells or 0) >= 96:
        chosen.append(Choice(
            "inference", "nonparametric",
            f"the well response fails a normality test "
            f"(D'Agostino p = {reading.normal_p:.2g}) and there are "
            f"{reading.wells:,} wells to permute, so an empirical null costs "
            f"time rather than assumptions"))
    elif reading.normal_p < 0.05:
        chosen.append(Choice(
            "inference", "parametric",
            f"the response is not normal (p = {reading.normal_p:.2g}), but "
            f"{reading.wells:,} wells is too few to build a permutation null "
            f"with resolution — a robust parametric fit is the better trade"))
    else:
        chosen.append(Choice(
            "inference", "parametric",
            f"the well response is consistent with normal "
            f"(D'Agostino p = {reading.normal_p:.2g}), which is the "
            f"assumption a parametric p-value needs"))


def _significance(reading: Reading, answers: Dict[str, Any],
                  chosen: List[Choice], undecided: List[Undecided]) -> None:
    """The multiple-testing posture, from the prior and the cost."""
    rate = answers.get("hits_per_thousand")
    cost = str(answers.get("cost") or "balanced")
    if rate is None:
        undecided.append(Undecided(
            "fdr_alpha",
            "the expected hit rate was not given, and it is the prior the "
            "correction's strictness is argued from"))
    else:
        share = max(float(rate), 0.0) / 1000.0
        # A LOW PRIOR MAKES EVERY DISCOVERY MORE LIKELY TO BE FALSE at the
        # same alpha, which is the whole argument for moving it: at 5 real
        # hits in 1,000, a 0.1 FDR is a list that is mostly noise.
        alpha = 0.05 if share >= 0.10 else (0.01 if share <= 0.01 else 0.05)
        if cost == "precision":
            alpha = min(alpha, 0.01)
        elif cost == "recall":
            alpha = max(alpha, 0.10)
        chosen.append(Choice(
            "fdr_alpha", alpha,
            f"you expect about {float(rate):g} hit(s) in 1,000 "
            f"({share:.1%}), and a false positive is "
            f"{'expensive' if cost == 'precision' else 'cheap' if cost == 'recall' else 'no worse than a false negative'} "
            f"— at that prior this alpha is the one whose discovery list is "
            f"mostly real"))
        chosen.append(Choice(
            "multiple_testing_method",
            "fdr_by" if cost == "precision" else "fdr_bh",
            "Benjamini–Yekutieli makes no independence assumption and is the "
            "conservative choice when a false positive is expensive"
            if cost == "precision" else
            "Benjamini–Hochberg controls the false discovery rate, which is "
            "the quantity a screen's hit list is about"))

    direction = str(answers.get("direction") or "either")
    if reading.binary or direction == "either":
        chosen.append(Choice(
            "p_threshold_kind", "adjusted",
            "the hit line is drawn on the corrected p-value, so the picture "
            "and the exported hit list mean the same thing by 'significant'"))
    else:
        chosen.append(Choice(
            "p_threshold_kind", "adjusted",
            f"a hit is {'an increase' if direction == 'up' else 'a decrease'} "
            f"only, and the line is still drawn on the corrected p-value; "
            f"the direction is read off the coefficient's sign"))


def _controls(reading: Reading, answers: Dict[str, Any],
              chosen: List[Choice], undecided: List[Undecided]) -> None:
    """Resolve non-cutting controls through the measured screen inventory.

    ``controls`` contains non-cutting guides used as the empirical null.
    ``negative_control`` identifies a perturbation expected to lack a
    phenotype; it is a distinct setting and is not modified here.
    """
    typed = [t.strip() for t in
             str(answers.get("controls") or "").split(",") if t.strip()]
    if not typed:
        undecided.append(Undecided(
            "controls",
            "no non-cutting control was named, so effects stay measured from "
            "zero -- 'no dose-response' -- rather than from the guides that "
            "cut nothing. That is a defensible baseline and it is not the "
            "one a reader of a screen figure assumes."))
        return
    chosen.append(Choice(
        "controls", typed,
        f"you named {', '.join(repr(t) for t in typed)} as the non-cutting "
        f"control; spaCR resolves a gene to every one of its guides and a "
        f"guide to itself, in any of the four spellings a library writes"))


def _thresholds(reading: Reading, chosen: List[Choice],
                undecided: List[Undecided]) -> None:
    """`fraction_threshold`, the cell-count floor and `min_n`.

    THE FRACTION THRESHOLD IS THE ONE THAT QUIETLY DELETES A SCREEN. It is a
    constant by default, and a constant cannot know how crowded a well is:
    with four guides per well a 2% cut removes nothing, and with four hundred
    it removes almost everything, because the typical guide is then a quarter
    of one per cent.

    So the advice is measured against THIS screen's own distribution, and
    where the usual default would discard most of the library it says so with
    the number rather than proposing a value and hoping.
    """
    kept = reading.kept_at_two_percent
    median = reading.fraction_median
    if median is None:
        if _NO_USABLE_FRACTIONS in reading.trouble:
            reason = (
                "the count tables contain no positive guide fractions; "
                "without usable reads in any well, nothing here can say "
                "what a threshold would cost"
            )
        else:
            reason = (
                "the count tables did not yield a fraction column, so "
                "nothing here can say what a threshold would cost"
            )
        undecided.append(Undecided(
            "fraction_threshold",
            reason))
    elif kept is not None and kept < 0.5:
        # A tenth of the typical share: low enough to keep the library,
        # which is the failure that matters, and still above nothing.
        proposal = float(f"{median / 10.0:.1g}")
        chosen.append(Choice(
            "fraction_threshold", proposal,
            f"the usual 0.02 would keep only {kept:.1%} of the "
            f"guide-in-well pairs here, because the typical guide is "
            f"{median:.3%} of its well. THE THRESHOLD IS A NOISE FLOOR AND "
            f"THIS SCREEN'S NOISE HAS NOT BEEN MEASURED: control wells of "
            f"known composition give it directly, and `spacr.read_background` "
            f"computes it from them. Until then this is a value chosen to "
            f"keep the library rather than to remove noise."))
    else:
        chosen.append(Choice(
            "fraction_threshold", 0.02,
            f"the wells are not crowded enough for the cut to bite: 0.02 "
            f"keeps {kept:.1%} of the guide-in-well pairs"
            if kept is not None else
            "the usual default, with nothing here arguing against it"))

    # MIN N -- how many observations a hit must rest on.
    per_guide = reading.wells_per_guide
    if per_guide is None:
        undecided.append(Undecided(
            "min_n", "the replication per guide could not be measured"))
    elif per_guide >= 4:
        chosen.append(Choice(
            "min_n", 1,
            f"a guide is in {per_guide:.0f} wells here, so requiring more "
            f"than one observation costs almost nothing and drops hits "
            f"resting on a single well"))
    else:
        chosen.append(Choice(
            "min_n", 0,
            f"a guide is in only {per_guide:.0f} well(s), so any floor above "
            f"zero would delete real hits along with the fragile ones"))

    objects = reading.objects_per_well
    if objects is not None and objects > 0:
        floor = max(int(objects * 0.1), 1)
        chosen.append(Choice(
            "min_cell_count", floor,
            f"a well holds {objects:.0f} objects here; a well with fewer "
            f"than {floor} has a fraction too noisy to model, and dropping "
            f"it is cheaper than letting it set a coefficient"))


def _aggregation(reading: Reading, chosen: List[Choice],
                 undecided: List[Undecided]) -> None:
    """`agg_type` -- how per-cell values become one number per well.

    THE MEDIAN WHEN THE RESPONSE IS SKEWED, and the skew is measured rather
    than assumed. A mean over cells is dragged by the tail, and the tail of
    a classification score is exactly where the interesting cells are -- so
    on a skewed response the mean reports the outliers and the median
    reports the well.
    """
    skew = reading.skew
    if skew is None:
        undecided.append(Undecided(
            "agg_type",
            "the response was not read, so nothing here knows whether its "
            "tail would drag a mean"))
        return
    if abs(float(skew)) >= 1.0:
        chosen.append(Choice(
            "agg_type", "median",
            f"the response is skewed ({skew:+.2f}), and a mean over cells "
            f"is dragged by the tail -- which on a classification score is "
            f"where the interesting cells are"))
    else:
        chosen.append(Choice(
            "agg_type", "mean",
            f"the response is near-symmetric ({skew:+.2f}), so the mean "
            f"uses every cell rather than only the middle one"))


# ---------------------------------------------------------------------------
# 226: what only a finished run knows
# ---------------------------------------------------------------------------

#: The keys the QC numbers file may use for each thing the advisor reads.
#: SEVERAL SPELLINGS PER ROW, because the panels name their own statistics
#: and two of them measure normality. Reading a list rather than one key is
#: what keeps this a READ instead of a rename negotiation.
_RUN_NUMBERS: Dict[str, Tuple[str, ...]] = {
    "residual_normal_p": ("normality_p", "dagostino_p", "shapiro_p",
                          "jarque_bera_p"),
    "residual_kurtosis": ("excess_kurtosis", "kurtosis"),
    "durbin_watson": ("durbin_watson",),
    "max_cooks_distance": ("max_cooks_distance", "cooks_max",
                           "max_cooks"),
    "max_vif": ("max_vif", "vif_max"),
}


def read_the_last_run(folder: str,
                      settings: Optional[Mapping[str, Any]] = None
                      ) -> Dict[str, Any]:
    """Read a finished run's own diagnostics off disk.

    A READ OF WHAT ALREADY EXISTS, never a second diagnostic pass. The
    numbers are the ones `regression_qc` measured and printed onto its own
    report; recomputing them here would let the advisor and the QC panel
    disagree about the same fit, and the user would have no way to tell
    which was right.

    :param folder: the run folder, or the ``regression_qc`` folder inside it.
    :param settings: the settings now in the panel, so a run fitted under
        different ones is reported as STALE rather than used silently.
    :returns: the fields to merge into a :class:`Reading`. Always includes
        ``run_folder`` when a run was found, and ``run_note`` when one was
        found and is not being used.
    """
    import json
    import os

    from .regression_qc import QC_NUMBERS_FILE

    if not folder:
        return {}
    candidates = [os.path.join(str(folder), QC_NUMBERS_FILE),
                  os.path.join(str(folder), "regression_qc", QC_NUMBERS_FILE)]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        return {}
    out: Dict[str, Any] = {"run_folder": os.path.dirname(path)}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as error:                                   # noqa: BLE001
        out["run_note"] = (f"the run's diagnostics could not be read "
                           f"({type(error).__name__}), so nothing was taken "
                           f"from it")
        return out

    stale = _stale_against(payload, settings)
    if stale:
        out["run_note"] = stale
        return out

    numbers = payload.get("numbers")
    if not isinstance(numbers, Mapping):
        out["run_note"] = ("the run's diagnostics file holds no numbers, so "
                           "nothing was taken from it")
        return out
    for field, spellings in _RUN_NUMBERS.items():
        for name in spellings:
            value = numbers.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[field] = float(value)
                break
    return out


def _stale_against(payload: Mapping[str, Any],
                   settings: Optional[Mapping[str, Any]]) -> str:
    """Why this run must not be read, or ``""``.

    ONE COMPARISON, AND IT IS THE FAMILY. The diagnostics are about the
    residuals of a particular fit, so a panel now set to a different
    regression type is asking about a model this run never was -- and its
    leverage, its Durbin-Watson and its VIF would all be answers to the
    wrong question. Everything else the user can change (a threshold, a
    correction) leaves the residuals recognisable.
    """
    if not settings:
        return ""
    now = str(settings.get("regression_type") or "").strip().lower()
    was = str(payload.get("regression_type") or "").strip().lower()
    if now and was and now != was:
        return (f"the last run fitted {was!r} and the panel now says "
                f"{now!r}, so its residual diagnostics describe a different "
                f"model and were not used")
    return ""


def _from_the_run(reading: Reading, chosen: List[Choice],
                  undecided: List[Undecided]) -> None:
    """Amend the choices with what the finished fit measured.

    RUNS LAST, and only ever OVERRIDES -- everything before it argued from
    the INPUT, which is all there is before a fit, and this argues from the
    residuals, which is what the assumptions are actually about. A response
    can be skewed while the residuals are fine, and normal while they are
    not; where the two disagree the residuals win, because they are the
    thing the p-value depends on.
    """
    if not reading.read_a_run:
        return
    where = os.path.basename(reading.run_folder.rstrip(os.sep)) or \
        reading.run_folder

    def replace(key: str, value: str, why: str) -> None:
        """Replace or append one recommendation in the captured choices.

        :param key: setting-choice key to update.
        :param value: value recommended from the completed run.
        :param why: user-facing evidence for the recommendation.
        :returns: None. The first same-key choice is replaced at its existing
            position; a key not already present is appended once.
        """
        for index, one in enumerate(chosen):
            if one.key == key:
                chosen[index] = Choice(key, value, why)
                return
        chosen.append(Choice(key, value, why))

    p_value = reading.residual_normal_p
    if p_value is not None and p_value < 0.05:
        replace("inference", "nonparametric",
                f"the last run's RESIDUALS fail a normality test "
                f"(p = {p_value:.2g}, measured in {where}), which is the "
                f"assumption a parametric p-value actually rests on — the "
                f"response's own shape is not the same question")
        kurtosis = reading.residual_kurtosis
        if kurtosis is not None and kurtosis > 3.0:
            replace("grna_statistic", "rank",
                    f"the residuals' excess kurtosis is {kurtosis:+.2f} in "
                    f"{where}, so the tails are heavy enough to move a "
                    f"correlation; a rank responds to order rather than "
                    f"magnitude")
    elif p_value is not None:
        replace("inference", "parametric",
                f"the last run's residuals are consistent with normal "
                f"(p = {p_value:.2g}, measured in {where}), which is the "
                f"assumption a parametric p-value needs")

    dw = reading.durbin_watson
    if dw is not None and abs(dw - 2.0) > 0.2:
        replace("guide_nuisance_columns", "['rowID', 'columnID']",
                f"Durbin-Watson is {dw:.2f} against 2 in {where}, so "
                f"neighbouring wells are not independent and a shuffle that "
                f"ignored the position would treat that structure as noise")

    cooks = reading.max_cooks_distance
    if cooks is not None and cooks >= 1.0:
        replace("regression_type", "rlm",
                f"one observation has Cook's distance {cooks:.2f} in {where}, "
                f"above the 1.0 rule — it moves the fit on its own, and "
                f"least squares is the estimator most sensitive to that")

    vif = reading.max_vif
    if vif is not None and vif > 10.0:
        replace("regression_type", "ridge",
                f"the largest VIF in {where} is {vif:.1f}, so the predictors "
                f"carry overlapping information and their individual "
                f"coefficients are not separable")


def advise(reading: Reading,
           answers: Optional[Dict[str, Any]] = None) -> Advice:
    """Build a regression-setting proposal from measurements and answers.

    This function has no settings side effects. The caller decides whether
    and how to display or apply the returned :class:`Advice`.

    :param reading: measured screen properties from which settings are derived.
    """
    answers = dict(answers or {})
    chosen: List[Choice] = []
    undecided: List[Undecided] = []
    _family_and_transform(reading, chosen, undecided)
    _level(reading, chosen, undecided)
    _plate(reading, chosen, undecided)
    _inference(reading, chosen, undecided)
    _significance(reading, answers, chosen, undecided)
    _controls(reading, answers, chosen, undecided)
    # Add evidence thresholds and the aggregation unit after the model family,
    # transform, and batch method have been selected.
    _thresholds(reading, chosen, undecided)
    _aggregation(reading, chosen, undecided)
    # Completed-run diagnostics are applied last because they can supersede
    # recommendations inferred from input structure alone.
    _from_the_run(reading, chosen, undecided)
    if reading.run_note:
        undecided.append(Undecided("last run", reading.run_note))
    return Advice(tuple(chosen), tuple(undecided), reading)


def advise_the_screen(counts: Sequence[str] = (), scores: Sequence[str] = (),
                      dependent_variable: str = "",
                      answers: Optional[Dict[str, Any]] = None,
                      *, row_cap: int = ROW_CAP,
                      run_folder: str = "",
                      settings: Optional[Mapping[str, Any]] = None) -> Advice:
    """Read screen inputs and return a regression-settings recommendation.

    :param run_folder: optional completed run whose recorded diagnostics can
        refine the input-based recommendation. An empty value requires no
        prior fit.
    :param settings: current panel settings used to reject diagnostics from a
        run fitted with an incompatible model family.
    :returns: recommended and unresolved settings with the evidence used to
        derive them.
    """
    reading = read_the_screen(counts, scores, dependent_variable,
                              row_cap=row_cap)
    if run_folder:
        from dataclasses import replace as _replace

        extra = read_the_last_run(run_folder, settings)
        known = set(Reading.__dataclass_fields__)
        extra = {k: v for k, v in extra.items() if k in known}
        if extra:
            reading = _replace(reading, **extra)
    return advise(reading, answers)


# ---------------------------------------------------------------------------
# 196 B: a proposal that the run would refuse is not a proposal
# ---------------------------------------------------------------------------

def refusals(settings: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return reasons a regression settings mapping cannot be run.

    This check covers incompatible setting combinations that default filling
    and type coercion cannot resolve. Messages follow the runtime validation
    wording so an application can explain a refusal before starting a fit.

    Parameters
    ----------
    settings : mapping
        Proposed regression settings.

    Returns
    -------
    tuple of str
        One actionable message per refusal, or an empty tuple when the
        settings pass these preflight checks.
    """
    said: List[str] = []
    got = dict(settings or {})

    # 1. ComBat without a covariate. The one that was actually hit.
    if str(got.get("batch_correction") or "").lower() == "combat":
        from .batch_correction import NO_COVARIATE

        covariate = got.get("batch_covariate_column")
        if covariate is None or (isinstance(covariate, str)
                                 and not covariate.strip()):
            said.append(
                "batch_correction='combat' needs to know which biology to "
                "keep, and no batch_covariate_column is set. Name the "
                f"condition/treatment column, or set it to {NO_COVARIATE!r} "
                "to state that there is nothing to preserve.")

    # 2. `control_center` with nothing to centre on.
    if str(got.get("batch_correction") or "").lower() == "control_center":
        if not got.get("batch_control_column"):
            said.append(
                "batch_correction='control_center' requires "
                "batch_control_column and at least one batch_control_value.")

    # 3. A setting the chosen estimator cannot read. `perform_regression`
    #    REFUSES these rather than ignoring them, so a number left on the
    #    panel from another model stops the run.
    kind = str(got.get("regression_type") or "").lower()
    try:
        from .regression_spec import REGRESSION_SETTINGS_USED

        if kind and kind not in REGRESSION_SETTINGS_USED:
            said.append(
                f"regression_type={kind!r} is not one of "
                f"{', '.join(sorted(REGRESSION_SETTINGS_USED))}.")
    except Exception:                                        # noqa: BLE001
        pass

    # 4. THE PERMUTATION TEST CANNOT SEE OBJECTS. Hit live on 2026-08-21: a
    #    run reached "permuting the guides" thirty-one seconds in -- after
    #    the filters, the plots and two saved CSVs -- and only then raised.
    #    The incompatibility is knowable from the settings alone and had no
    #    business waiting for the data.
    mode = str(got.get("analysis_mode") or "").lower()
    unit = str(got.get("analysis_unit") or "").lower()
    if mode == "guide_permutation" and unit == "cell":
        said.append(
            "analysis_mode='guide_permutation' tests each guide across "
            "WELLS, so it needs one row per well -- but analysis_unit='cell' "
            "gives one row per object, and a well's phenotype then has many "
            "values. Set analysis_unit='well' (with an agg_type such as "
            "'mean'), or choose analysis_mode='regression', which can model "
            "objects.")

    # 5. THE SAME COMBINATION REACHED THROUGH `inference`, which is the door
    #    a user actually walks through: 'nonparametric' SELECTS
    #    guide_permutation, so the refusal has to recognise it under both
    #    names or it fires for the setting nobody typed.
    if (str(got.get("inference") or "").lower() in ("nonparametric",
                                                    "permutation")
            and unit == "cell" and mode != "regression"):
        if not any("guide_permutation" in message for message in said):
            said.append(
                "inference='nonparametric' runs the guide permutation test, "
                "which needs one row per well, and analysis_unit='cell' "
                "gives one row per object. Set analysis_unit='well' with an "
                "agg_type, or inference='parametric' to fit a model that can "
                "read objects.")

    # 6. AN AGGREGATION THAT WILL NOT BE READ. `analysis_unit='cell'` keeps
    #    every object, so an `agg_type` set beside it is a control the user
    #    changed and the run ignored -- which is how somebody concludes the
    #    setting does nothing.
    if unit == "cell" and got.get("agg_type"):
        said.append(
            f"analysis_unit='cell' keeps one row per object, so "
            f"agg_type={got.get('agg_type')!r} is never read. Clear it, or "
            f"set analysis_unit='well' if the aggregation was the intent.")

    return tuple(said)


#: Settings required by each ``analysis_unit``. User interfaces apply these
#: values when the unit changes, display them, and disable the corresponding
#: controls so an incompatible design cannot reach the run stage.
#:
#: `None` as a value means "must be empty", which is a requirement like any
#: other and not an absence of one.
UNIT_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "cell": {
        # The permutation test works well by well; only the model can read
        # objects.
        "analysis_mode": "regression",
        "inference": "parametric",
        # One row per object already: there is nothing to aggregate.
        "agg_type": None,
    },
    "well": {},
}


def requirements_for_unit(unit: str) -> Dict[str, Any]:
    """Return settings required by an analysis unit.

    :param unit: ``'cell'`` or ``'well'``.
    :returns: ``{setting: required value}``, empty when the unit constrains
        nothing.

    User interfaces can apply these values immediately and disable the
    corresponding controls, preventing incompatible combinations from
    reaching runtime validation.
    """
    return dict(UNIT_REQUIREMENTS.get(str(unit).lower(), {}))


def advise_that_runs(reading: Reading,
                     answers: Optional[Dict[str, Any]] = None) -> Advice:
    """Build advice and withdraw choices that fail runtime preflight.

    A choice named by :func:`refusals` moves from ``chosen`` to ``undecided``
    with the validation message. The function does not silently substitute a
    different value.

    Parameters
    ----------
    reading : Reading
        Measurements and metadata describing the screen.
    answers : dict, optional
        User answers to questions the screen data cannot settle.

    Returns
    -------
    Advice
        A proposal containing only choices that pass the preflight checks.
    """
    advice = advise(reading, answers)
    from .settings import get_perform_regression_default_settings

    try:
        whole = get_perform_regression_default_settings(
            dict(advice.as_settings()))
    except Exception:                                        # noqa: BLE001
        whole = dict(advice.as_settings())
    said = refusals(whole)
    if not said:
        return advice

    # WHICH SETTING TO WITHDRAW. Named from the sentence rather than guessed:
    # every refusal above quotes the key it is about.
    chosen, withdrawn = [], list(advice.undecided)
    for choice in advice.chosen:
        blamed = [s for s in said if choice.key in s]
        if blamed:
            withdrawn.append(Undecided(
                choice.key,
                f"withdrawn: the run would refuse it. {blamed[0]}"))
        else:
            chosen.append(choice)
    return Advice(tuple(chosen), tuple(withdrawn), advice.reading)
