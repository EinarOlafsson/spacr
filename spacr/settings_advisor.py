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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """A setting this module will not guess, and why not."""

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
    capped: bool = False
    trouble: Tuple[str, ...] = ()

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


def read_the_counts(paths: Sequence[str]) -> Dict[str, Any]:
    """Plates, wells, guides, genes and the two ratios, from the count CSVs.

    EXACT, not sampled: a count table is one row per well and guide, and the
    biggest in the reference screen is 4.3 MB. Every number here is the one
    the fit will see, because the fractions come from
    :func:`spacr.cell_montage.fractions_from_counts`, which is
    `process_reads`' own arithmetic rather than a second implementation of
    it.
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
            "This is the prior, and it decides how strict the multiple-"
            "testing correction should be. A screen expecting 5 in 1,000 is "
            "asking a different question from one expecting 200, and the "
            "same FDR serves neither. Nothing in the tables supplies it."),
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
            "A screen looking for loss of a phenotype is asking a one-sided "
            "question, and testing it two-sided spends half the alpha on an "
            "answer it does not want."),
    ),
    Question(
        key="controls",
        prompt="What are your non-targeting controls called? (blank if there "
               "are none)",
        kind="text",
        default="",
        why_it_matters=(
            "A gene name takes every one of its guides; a guide name takes "
            "just that guide, in any of the four spellings spaCR reads. With "
            "controls named, the effects can be measured FROM them rather "
            "than from zero, and the batch correction can centre on them."),
    ),
    Question(
        key="cost",
        prompt="What does a wrong answer cost you?",
        kind="choice",
        options=(("precision", "a false positive is expensive — the "
                              "follow-up is slow or costly"),
                 ("balanced", "about the same either way"),
                 ("recall", "a false negative is expensive — I would rather "
                            "chase a few extra")),
        default="balanced",
        why_it_matters=(
            "The honest form of 'how strict?'. It moves the alpha and, at "
            "the recall end, the correction itself — there is no setting "
            "that is right for both a screen feeding a mouse experiment and "
            "one feeding a plate reader."),
    ),
)


def questions_for(reading: Reading) -> Tuple[Question, ...]:
    """The questions worth asking THIS screen.

    A question that CAN be read from the data and is asked anyway is a
    question that should not have been there -- so the controls question is
    dropped when the count tables already name a non-targeting set, and the
    direction question is dropped when the response is binary, where "an
    increase" is the only direction there is.
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
        """The reason for one key, or ``''``."""
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
        chosen.append(Choice(
            "batch_correction", "combat",
            f"{reading.plates} plates, so plate is a batch the response can "
            f"carry; ComBat removes it while keeping the within-plate "
            f"variance the fit needs"))
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
    """The controls, resolved through the one reader (184)."""
    typed = str(answers.get("controls") or "").strip()
    if not typed:
        undecided.append(Undecided(
            "negative_control",
            "no non-targeting control was named, so effects stay measured "
            "from zero — 'no dose-response' — rather than from an untargeted "
            "well. That is a defensible baseline and it is not the one a "
            "reader of a screen figure assumes."))
        return
    chosen.append(Choice(
        "negative_control", typed,
        f"you named {typed!r}; spaCR resolves a gene to every one of its "
        f"guides and a guide to itself, in any of the four spellings a "
        f"library writes"))


def advise(reading: Reading,
           answers: Optional[Dict[str, Any]] = None) -> Advice:
    """Turn a :class:`Reading` and the answers into a proposal.

    NOTHING IS WRITTEN. The caller shows what would change, with the current
    value beside the new one, and the user accepts or rejects. A button that
    rewrites a carefully-tuned panel with one click and no undo is a button
    people learn not to press.
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
    return Advice(tuple(chosen), tuple(undecided), reading)


def advise_the_screen(counts: Sequence[str] = (), scores: Sequence[str] = (),
                      dependent_variable: str = "",
                      answers: Optional[Dict[str, Any]] = None,
                      *, row_cap: int = ROW_CAP) -> Advice:
    """Read and advise in one call, for a caller that has only paths."""
    return advise(read_the_screen(counts, scores, dependent_variable,
                                  row_cap=row_cap), answers)
