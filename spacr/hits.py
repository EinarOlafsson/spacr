"""The hit list: the ranked, annotated, filterable deliverable of a screen.

A regression run leaves a folder of plots and four CSVs. None of them is the
thing the experiment was for. ``results.csv`` is one row per model term
including the intercept; ``results_significant.csv`` is that filtered at
``p <= 0.05`` with no multiple-testing correction, no gene annotation and no
indication of whether a gene's own guides agree with each other. What a user
wants at the end of a screen is a single table they can sort, filter, act on
and send to a collaborator, where each row is a GENE and carries:

* **the effect size** — the fitted coefficient, with its standard error and a
  95% interval when the backend reports one, because "significant" without a
  magnitude is not a result;
* **the significance** — the p-value AND a Benjamini-Hochberg q-value across
  the genes actually tested, because a screen tests thousands of hypotheses
  and an uncorrected 0.05 on 2000 genes is 100 expected false hits;
* **gRNA agreement** — how many of the gene's own guides push the same way.
  A gene called by one guide out of six is the single most common way a
  pooled screen produces a confident artefact, and it is invisible in every
  table spaCR wrote before this one;
* **the metadata join** — gene name, product, location, whatever the curated
  annotation file carries.

**One row per gene. Enforced, not assumed.** The bundled
``toxoplasma_metadata.csv`` lists a gene once per transcript: 30 Gene IDs
repeat between 2 and 32 times. Joined as-is, those genes came back two to
thirty-two times and every consumer counted each copy as an independent hit.
:func:`load_gene_metadata` collapses the annotation to one row per gene
*before* the join and says how many rows it dropped, and :func:`join_metadata`
asks pandas to ``validate="many_to_one"`` so a future annotation file that
breaks the assumption fails loudly instead of silently multiplying the hits.
:func:`build_hit_list` asserts the same invariant on its own output.

The module is headless and imports neither Qt nor :mod:`spacr.ml` (which
pulls torch). It reads the CSVs a regression run already wrote.

Public API::

    from spacr.hits import build_hit_list, load_results

    hits = build_hit_list("/data/plate7/results/pred/ols/list")
    strong = hits.filter(max_q=0.05, min_agreement=0.66, min_guides=2)
    strong.write_csv("/tmp/hits.csv")
    print(strong.to_markdown(limit=20))
"""
from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_ALPHA",
    "Hit",
    "HitList",
    "NO_P_VALUE_TYPES",
    "RESULT_FILES",
    "benjamini_hochberg",
    "build_hit_list",
    "gene_of",
    "grna_agreement",
    "join_metadata",
    "load_gene_metadata",
    "load_results",
]

#: The files :func:`spacr.ml.perform_regression` writes into its results
#: folder, by the role this module needs them for.
RESULT_FILES: Dict[str, str] = {
    "all": "results.csv",
    "gene": "results_gene.csv",
    "grna": "results_grna.csv",
    "significant": "results_significant.csv",
}

#: Backends that report a coefficient but no frequentist p-value, so a
#: q-value would be a correction applied to a number that is not a p-value.
#: Mirrors :data:`spacr.ml.NO_P_VALUE_TYPES`; kept as a literal so importing
#: this module does not drag torch in, and asserted equal in the test suite.
NO_P_VALUE_TYPES: Tuple[str, ...] = ("lasso", "elasticnet")

#: The FDR a hit list defaults to calling a hit at.
DEFAULT_ALPHA = 0.05

#: Flags a row can carry. Each one is a reason to look twice, not a reason to
#: drop the row — dropping it would hide the very thing the flag is for.
FLAG_CONTROL = "control"
FLAG_SINGLE_GUIDE = "single-guide"
FLAG_GUIDES_DISAGREE = "guides-disagree"
FLAG_NO_GUIDES = "no-guide-rows"
FLAG_NO_METADATA = "unannotated"

#: What each flag means, for a legend under the table.
FLAG_MEANING: Dict[str, str] = {
    FLAG_CONTROL: "a control gRNA or gene, not a screen candidate",
    FLAG_SINGLE_GUIDE: "called by one guide, so nothing corroborates it",
    FLAG_GUIDES_DISAGREE: "fewer than half of this gene's guides agree in sign",
    FLAG_NO_GUIDES: "no per-gRNA rows for this gene, so agreement is unknown",
    FLAG_NO_METADATA: "no row in any metadata file matched this gene",
}

_BRACKET = re.compile(r"\[(.*?)\]")


# ---------------------------------------------------------------------------
# Parsing and statistics
# ---------------------------------------------------------------------------

def gene_of(feature: Any) -> Optional[str]:
    """Return the gene id a model term names, or ``None``.

    The rule is the one :func:`spacr.utils.merge_regression_res_with_metadata`
    applies, deliberately: the bracketed token, ``T.`` stripped, truncated at
    the first underscore. It maps BOTH sides of the pair to the same key —
    ``gene_fraction:gene[233460]`` and ``fraction:grna[233460_1]`` are both
    gene ``233460`` — which is what makes per-guide agreement computable at
    all, and it is the same key the metadata join uses so the two cannot
    disagree.

    :param feature: a design-matrix term name.
    :returns: the gene id, or ``None`` for a term that names no gene
        (``Intercept``, a row or column nuisance term).
    """
    if feature is None or (isinstance(feature, float) and math.isnan(feature)):
        return None
    match = _BRACKET.search(str(feature))
    if not match:
        return None
    token = re.sub(r"^T\.", "", match.group(1))
    gene = token.split("_")[0]
    return gene or None


def benjamini_hochberg(p_values: Sequence[Any]) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values for a vector of p-values.

    The step-up procedure, with the monotonicity enforced by the running
    minimum from the largest p-value down — without it a q-value can come out
    smaller than one belonging to a smaller p-value, which reads as a hit
    ranking that disagrees with itself.

    ``NaN`` p-values (a term the backend could not test) stay ``NaN`` and are
    excluded from ``m``: correcting for a test that was not run inflates every
    other q-value.

    :param p_values: p-values; anything non-finite is treated as untested.
    :returns: q-values aligned with the input.
    """
    values = np.asarray(p_values, dtype=float)
    q = np.full(values.shape, np.nan, dtype=float)
    testable = np.isfinite(values)
    m = int(testable.sum())
    if m == 0:
        return q
    order = np.argsort(values[testable], kind="mergesort")
    ranked = values[testable][order]
    scaled = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    q[testable] = out
    return q


def grna_agreement(gene_effects: Mapping[str, float],
                   grna_frame: Optional[pd.DataFrame]
                   ) -> Dict[str, Tuple[int, int, List[str]]]:
    """How many of each gene's guides push the same way as the gene.

    :param gene_effects: ``{gene id: gene-level coefficient}``.
    :param grna_frame: the per-gRNA coefficient table (``feature`` and
        ``coefficient`` columns). ``None`` or empty means "no guide-level
        evidence", which is reported as such rather than as agreement.
    :returns: ``{gene id: (n_agree, n_guides, [guide ids that agree])}``. A
        guide whose coefficient is exactly zero counts as a guide but agrees
        with nothing — a penalised backend shrinks non-contributing guides to
        zero, and counting those as agreement would turn a lasso's sparsity
        into corroboration.
    """
    result: Dict[str, Tuple[int, int, List[str]]] = {
        gene: (0, 0, []) for gene in gene_effects}
    if grna_frame is None or grna_frame.empty:
        return result
    if "feature" not in grna_frame.columns:
        return result

    per_gene: Dict[str, List[Tuple[str, float]]] = {}
    for _, row in grna_frame.iterrows():
        gene = gene_of(row.get("feature"))
        if gene is None:
            continue
        guide = row.get("grna")
        if guide is None or (isinstance(guide, float) and math.isnan(guide)):
            match = _BRACKET.search(str(row.get("feature", "")))
            guide = re.sub(r"^T\.", "", match.group(1)) if match else ""
        try:
            coefficient = float(row.get("coefficient"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(coefficient):
            continue
        per_gene.setdefault(gene, []).append((str(guide), coefficient))

    for gene, guides in per_gene.items():
        target = gene_effects.get(gene)
        if target is None or not math.isfinite(float(target)):
            result[gene] = (0, len(guides), [])
            continue
        wanted = math.copysign(1.0, float(target))
        agree = [name for name, value in guides
                 if value != 0.0 and math.copysign(1.0, value) == wanted]
        result[gene] = (len(agree), len(guides), sorted(agree))
    return result


# ---------------------------------------------------------------------------
# Reading what a regression run wrote
# ---------------------------------------------------------------------------

def load_results(folder: Union[str, os.PathLike]) -> Dict[str, pd.DataFrame]:
    """Read the coefficient tables a regression results folder holds.

    :param folder: the ``results/<score>/<type>[/list]`` folder
        :func:`spacr.ml.perform_regression` writes into.
    :returns: ``{role: DataFrame}`` for whichever of :data:`RESULT_FILES`
        exist. A folder with none of them yields an empty dict rather than an
        exception — "that is not a results folder" is something the caller
        must be able to say to a user.
    :raises FileNotFoundError: when ``folder`` does not exist at all.
    """
    root = os.path.abspath(os.path.expanduser(os.fspath(folder)))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no results folder at {root}")
    found: Dict[str, pd.DataFrame] = {}
    for role, name in RESULT_FILES.items():
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                found[role] = pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
    return found


def load_gene_metadata(path: Union[str, os.PathLike], *,
                       key: str = "Gene ID"
                       ) -> Tuple[pd.DataFrame, List[str]]:
    """Read one annotation CSV as EXACTLY one row per gene.

    A curated export lists a gene once per transcript. The bundled
    ``toxoplasma_metadata.csv`` repeats 30 Gene IDs between 2 and 32 times,
    each copy carrying a different protein length and GO-term set. Joined
    as-is, every one of those genes multiplies in the results — which is a
    hit list that counts the same gene as up to 32 independent findings.

    So the collapse happens HERE, before anything is joined, and it is
    reported: the returned notes name how many rows were dropped and for how
    many genes, and the annotations of the dropped rows are not carried over.

    :param path: the metadata CSV.
    :param key: the column holding the gene identifier; ``Gene ID`` in
        spaCR's own files, where the value is ``TGME49_233460`` and the gene
        is the part after the underscore.
    :returns: ``(frame, notes)``. The frame carries a ``gene`` column and at
        most one row per value in it.
    :raises FileNotFoundError: when the file is not there.
    :raises KeyError: when the key column is absent — a metadata file with no
        gene identifier cannot be joined, and guessing which column meant to
        be one is how the wrong annotation gets attached to a hit.
    """
    target = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(target):
        raise FileNotFoundError(f"no metadata file at {target}")
    frame = pd.read_csv(target)
    if key not in frame.columns:
        raise KeyError(
            f"{os.path.basename(target)} has no {key!r} column, so its rows "
            f"cannot be attached to a gene. Columns: {list(frame.columns)}")

    notes: List[str] = []
    frame = frame.copy()
    frame["gene"] = frame[key].map(
        lambda value: str(value).split("_")[1] if "_" in str(value) else None)
    # A NaN key must never act as a join key: pandas treats NaN as equal to
    # NaN, so every unparsable metadata row would fan out against every
    # unparsable result row.
    unparsed = int(frame["gene"].isna().sum())
    frame = frame.dropna(subset=["gene"])
    if unparsed:
        notes.append(
            f"{os.path.basename(target)}: {unparsed} row(s) had no parsable "
            f"gene in {key!r} and were dropped.")

    duplicated = frame["gene"].duplicated(keep=False)
    if bool(duplicated.any()):
        genes = sorted(frame.loc[duplicated, "gene"].unique())
        notes.append(
            f"{os.path.basename(target)}: {int(duplicated.sum())} rows share "
            f"{len(genes)} gene id(s), e.g. {genes[:5]} — usually one row per "
            f"transcript. The first row of each is kept so the join cannot "
            f"duplicate a hit; the annotations of the dropped rows are not "
            f"carried over.")
        frame = frame.drop_duplicates(subset=["gene"], keep="first")
    return frame.reset_index(drop=True), notes


def join_metadata(frame: pd.DataFrame,
                  metadata_files: Sequence[Union[str, os.PathLike]] = (),
                  *, key: str = "Gene ID"
                  ) -> Tuple[pd.DataFrame, List[str]]:
    """Attach every metadata file to ``frame`` on its ``gene`` column.

    ``validate="many_to_one"`` on every join: many result rows may name one
    gene (one per guide), but each gene gets one annotation row. pandas
    raises rather than fanning out if a file ever breaks that, which is the
    guard that keeps the collapse in :func:`load_gene_metadata` honest.

    :param frame: a table with a ``gene`` column.
    :param metadata_files: annotation CSVs, applied in order. A later file's
        columns are suffixed rather than overwriting an earlier one's.
    :param key: the gene identifier column in the metadata files.
    :returns: ``(joined, notes)``.
    """
    notes: List[str] = []
    if "gene" not in frame.columns:
        raise KeyError("the frame has no 'gene' column to join on")
    joined = frame
    for index, path in enumerate(metadata_files or ()):
        annotation, file_notes = load_gene_metadata(path, key=key)
        notes.extend(file_notes)
        before = len(joined)
        joined = joined.merge(
            annotation, on="gene", how="left", validate="many_to_one",
            suffixes=("", f"_meta{index + 1}"))
        if len(joined) != before:  # pragma: no cover - validate already raises
            raise ValueError(
                f"joining {os.path.basename(str(path))} changed the row count "
                f"from {before} to {len(joined)}; the annotation is not one "
                f"row per gene.")
    return joined, notes


# ---------------------------------------------------------------------------
# The deliverable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """One gene, everything known about it, and why to believe it.

    :param gene: the gene id parsed from the model term.
    :param feature: the model term itself, so a row can be traced back.
    :param effect: the fitted coefficient — the effect size.
    :param std_err: its standard error, when the backend reported one.
    :param ci_low: lower bound of the 95% interval; ``nan`` without an error.
    :param ci_high: upper bound of the same.
    :param p_value: the reported p-value; ``nan`` for a backend that has none.
    :param q_value: Benjamini-Hochberg FDR over the genes tested here.
    :param selection_frequency: bootstrap selection frequency, for the
        penalised backends that rank by it instead of by a p-value.
    :param n_guides: how many of this gene's guides the per-gRNA table holds.
    :param n_agree: how many of them push the same way as the gene effect.
    :param agreement: ``n_agree / n_guides``; ``nan`` with no guide rows.
    :param agreeing_guides: which guides those are.
    :param n_obs: observations behind the gene term (well x guide rows).
    :param condition: ``nc`` / ``pc`` / ``control`` / ``other`` as the
        regression labelled it.
    :param direction: ``up`` or ``down``, from the sign of the effect.
    :param rank: 1-based position in the list as ranked.
    :param flags: reasons to look twice; see :data:`FLAG_MEANING`.
    :param annotation: the metadata columns that joined onto this gene.
    """

    gene: str
    feature: str = ""
    effect: float = float("nan")
    std_err: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    q_value: float = float("nan")
    selection_frequency: float = float("nan")
    n_guides: int = 0
    n_agree: int = 0
    agreement: float = float("nan")
    agreeing_guides: Tuple[str, ...] = ()
    n_obs: int = 0
    condition: str = ""
    direction: str = ""
    rank: int = 0
    flags: Tuple[str, ...] = ()
    annotation: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The most human name available: an annotated one, else the id."""
        for column in ("Gene Name", "gene_name", "Name", "Product Description",
                       "product", "description"):
            value = self.annotation.get(column)
            if value is not None and str(value).strip() and str(value) != "nan":
                return str(value).strip()
        return self.gene

    def to_dict(self) -> Dict[str, Any]:
        """A flat, JSON-serializable row: the fields, then the annotation."""
        row: Dict[str, Any] = {
            "rank": self.rank, "gene": self.gene, "name": self.name,
            "effect": self.effect, "std_err": self.std_err,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_value": self.p_value, "q_value": self.q_value,
            "selection_frequency": self.selection_frequency,
            "n_guides": self.n_guides, "n_agree": self.n_agree,
            "agreement": self.agreement,
            "agreeing_guides": ";".join(self.agreeing_guides),
            "n_obs": self.n_obs, "condition": self.condition,
            "direction": self.direction, "flags": ";".join(self.flags),
            "feature": self.feature,
        }
        for column, value in self.annotation.items():
            row.setdefault(column, value)
        return row


@dataclass(frozen=True)
class HitList:
    """A ranked list of hits plus everything needed to interpret it.

    :param hits: the rows, already ranked.
    :param source: the results folder or a description of where it came from.
    :param regression_type: the backend, when it could be determined.
    :param ranking: how the list is ordered — ``"q-value"`` or
        ``"selection-frequency"``.
    :param alpha: the FDR the ``significant`` count is taken at.
    :param n_terms: how many model terms the source table held.
    :param n_genes: how many distinct genes were tested.
    :param filters: the filters applied to reach this list, as data.
    :param notes: metadata collapses, missing files, backend caveats.
    """

    hits: Tuple[Hit, ...] = ()
    source: str = ""
    regression_type: str = ""
    ranking: str = "q-value"
    alpha: float = DEFAULT_ALPHA
    n_terms: int = 0
    n_genes: int = 0
    filters: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def __len__(self) -> int:
        """How many rows the list holds."""
        return len(self.hits)

    def __iter__(self):
        """Iterate the rows in rank order."""
        return iter(self.hits)

    def __getitem__(self, index):
        """Index or slice the rows; a slice returns a :class:`HitList`."""
        if isinstance(index, slice):
            return self._with(self.hits[index], dict(self.filters))
        return self.hits[index]

    def gene(self, gene: str) -> Optional[Hit]:
        """The row for one gene id, or ``None``."""
        for hit in self.hits:
            if hit.gene == gene:
                return hit
        return None

    def significant(self, alpha: Optional[float] = None) -> "HitList":
        """The rows that clear the FDR (or the selection threshold)."""
        cut = self.alpha if alpha is None else float(alpha)
        if self.ranking == "selection-frequency":
            return self.filter(min_selection=cut)
        return self.filter(max_q=cut)

    def top(self, n: int) -> "HitList":
        """The first ``n`` rows, still ranked."""
        return self._with(self.hits[:max(0, int(n))],
                          dict(self.filters, top=int(n)))

    def filter(self, *,
               max_q: Optional[float] = None,
               max_p: Optional[float] = None,
               min_effect: Optional[float] = None,
               min_agreement: Optional[float] = None,
               min_guides: Optional[int] = None,
               min_selection: Optional[float] = None,
               direction: Optional[str] = None,
               conditions: Optional[Iterable[str]] = None,
               exclude_controls: bool = False,
               genes: Optional[Iterable[str]] = None,
               query: str = "",
               predicate: Optional[Callable[[Hit], bool]] = None,
               ) -> "HitList":
        """Return a narrowed list. Every argument is optional and ANDed.

        A row whose value for a criterion is missing FAILS that criterion
        rather than passing it: a gene with no q-value has not been shown to
        clear an FDR, and letting missing data through a filter is how an
        untested term ends up in a hit list.

        :param max_q: keep rows with ``q_value <= max_q``.
        :param max_p: keep rows with ``p_value <= max_p``.
        :param min_effect: keep rows with ``abs(effect) >= min_effect``.
        :param min_agreement: keep rows whose guide agreement is at least this.
        :param min_guides: keep rows with at least this many guides.
        :param min_selection: keep rows with at least this bootstrap
            selection frequency.
        :param direction: ``"up"`` or ``"down"``.
        :param conditions: keep only these ``condition`` values.
        :param exclude_controls: drop ``nc`` / ``pc`` / ``control`` rows.
        :param genes: keep only these gene ids.
        :param query: case-insensitive substring, matched against the gene
            id, the name and every annotation value.
        :param predicate: an arbitrary extra test.
        :returns: a new :class:`HitList`; the receiver is unchanged.
        """
        wanted = set(conditions) if conditions is not None else None
        keep_genes = set(genes) if genes is not None else None
        needle = query.strip().casefold()

        def _ok(hit: Hit) -> bool:
            if max_q is not None and not _at_most(hit.q_value, max_q):
                return False
            if max_p is not None and not _at_most(hit.p_value, max_p):
                return False
            if min_effect is not None and not _at_least(abs(hit.effect),
                                                        min_effect):
                return False
            if min_agreement is not None and not _at_least(hit.agreement,
                                                           min_agreement):
                return False
            if min_guides is not None and hit.n_guides < int(min_guides):
                return False
            if min_selection is not None and not _at_least(
                    hit.selection_frequency, min_selection):
                return False
            if direction and hit.direction != direction:
                return False
            if wanted is not None and hit.condition not in wanted:
                return False
            if exclude_controls and hit.condition in ("nc", "pc", "control"):
                return False
            if keep_genes is not None and hit.gene not in keep_genes:
                return False
            if needle and needle not in _searchable(hit):
                return False
            if predicate is not None and not predicate(hit):
                return False
            return True

        applied = {
            "max_q": max_q, "max_p": max_p, "min_effect": min_effect,
            "min_agreement": min_agreement, "min_guides": min_guides,
            "min_selection": min_selection, "direction": direction,
            "conditions": sorted(wanted) if wanted is not None else None,
            "exclude_controls": exclude_controls or None,
            "genes": sorted(keep_genes) if keep_genes is not None else None,
            "query": query or None,
        }
        merged = dict(self.filters)
        merged.update({k: v for k, v in applied.items() if v is not None})
        return self._with(tuple(hit for hit in self.hits if _ok(hit)), merged)

    def _with(self, hits: Sequence[Hit], filters: Dict[str, Any]) -> "HitList":
        """A copy carrying different rows, ranks renumbered from 1."""
        renumbered = tuple(
            Hit(**{**hit.__dict__, "rank": index + 1})
            for index, hit in enumerate(hits))
        return HitList(
            hits=renumbered, source=self.source,
            regression_type=self.regression_type, ranking=self.ranking,
            alpha=self.alpha, n_terms=self.n_terms, n_genes=self.n_genes,
            filters=filters, notes=self.notes)

    # -- output -----------------------------------------------------------

    def columns(self) -> List[str]:
        """Column order for the table forms, annotation columns last."""
        base = ["rank", "gene", "name", "effect", "std_err", "ci_low",
                "ci_high", "p_value", "q_value", "selection_frequency",
                "n_guides", "n_agree", "agreement", "n_obs", "condition",
                "direction", "flags", "agreeing_guides", "feature"]
        extra: List[str] = []
        for hit in self.hits:
            for column in hit.annotation:
                if column not in base and column not in extra:
                    extra.append(column)
        return base + extra

    def to_frame(self) -> pd.DataFrame:
        """The list as a DataFrame, one row per gene, in rank order."""
        rows = [hit.to_dict() for hit in self.hits]
        frame = pd.DataFrame(rows, columns=self.columns())
        return frame

    def write_csv(self, path: Union[str, os.PathLike]) -> str:
        """Write the table as CSV and return the path written."""
        target = os.path.abspath(os.path.expanduser(os.fspath(path)))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        self.to_frame().to_csv(target, index=False)
        return target

    def summary(self) -> Dict[str, Any]:
        """Counts and settings, for a header line or a run digest.

        This is the block :mod:`spacr.methods_export` puts in front of the
        model: every number a results paragraph would quote, computed here
        rather than by whatever writes the prose.
        """
        significant = self.significant()
        up = sum(1 for hit in significant if hit.direction == "up")
        down = sum(1 for hit in significant if hit.direction == "down")
        corroborated = sum(1 for hit in significant
                           if hit.n_guides >= 2 and _at_least(hit.agreement,
                                                              0.5))
        effects = [abs(hit.effect) for hit in significant
                   if math.isfinite(hit.effect)]
        return {
            "source": self.source,
            "regression_type": self.regression_type,
            "ranking": self.ranking,
            "alpha": self.alpha,
            "n_terms": self.n_terms,
            "n_genes_tested": self.n_genes,
            "n_listed": len(self.hits),
            "n_significant": len(significant),
            "n_up": up,
            "n_down": down,
            "n_corroborated": corroborated,
            "max_abs_effect": max(effects) if effects else float("nan"),
            "median_abs_effect": (float(np.median(effects)) if effects
                                  else float("nan")),
            "top_genes": [hit.gene for hit in significant[:10]],
            "filters": dict(self.filters),
            "flag_counts": self.flag_counts(),
            "notes": list(self.notes),
        }

    def flag_counts(self) -> Dict[str, int]:
        """``{flag: how many rows carry it}``."""
        counts: Dict[str, int] = {}
        for hit in self.hits:
            for flag in hit.flags:
                counts[flag] = counts.get(flag, 0) + 1
        return dict(sorted(counts.items()))

    def to_markdown(self, limit: int = 50) -> str:
        """A Markdown table of the top ``limit`` rows, with its legend.

        The form the list travels in: pasted into an email, a lab notebook or
        an issue. No trailing newline.
        """
        header = [f"# Hit list — {self.source or 'regression'}"]
        summary = self.summary()
        header.append("")
        header.append(
            f"{summary['n_significant']} of {summary['n_genes_tested']} genes "
            f"tested clear {'a selection frequency of' if self.ranking == 'selection-frequency' else 'FDR'}"
            f" {self.alpha:g} "
            f"({summary['n_up']} up, {summary['n_down']} down; "
            f"{summary['n_corroborated']} corroborated by at least two "
            f"guides).")
        if self.regression_type:
            header.append(f"Model: {self.regression_type}.")
        for note in self.notes:
            header.append(f"Note: {note}")
        header.append("")

        shown = self.hits[:max(0, int(limit))]
        columns = ["rank", "gene", "name", "effect", "p_value", "q_value",
                   "guides", "agreement", "flags"]
        header.append("| " + " | ".join(columns) + " |")
        header.append("|" + "|".join(["---"] * len(columns)) + "|")
        for hit in shown:
            header.append("| " + " | ".join([
                str(hit.rank), hit.gene, hit.name,
                _fmt(hit.effect), _fmt(hit.p_value), _fmt(hit.q_value),
                f"{hit.n_agree}/{hit.n_guides}",
                _fmt(hit.agreement), ", ".join(hit.flags) or "-",
            ]) + " |")
        if len(self.hits) > len(shown):
            header.append("")
            header.append(f"…and {len(self.hits) - len(shown)} more rows.")
        used = self.flag_counts()
        if used:
            header.append("")
            header.append("Flags:")
            header.extend(f"* **{flag}** — {FLAG_MEANING.get(flag, flag)} "
                          f"({count} row(s))"
                          for flag, count in used.items())
        return "\n".join(header)

    def to_html(self, limit: int = 500) -> str:
        """A standalone HTML table — the form handed to a collaborator.

        Self-contained: no stylesheet, no script, no network. It opens in a
        browser on a machine that has never heard of spaCR, which is the
        whole requirement.
        """
        summary = self.summary()
        rows = []
        for hit in self.hits[:max(0, int(limit))]:
            cells = [str(hit.rank), hit.gene, hit.name, _fmt(hit.effect),
                     _fmt(hit.p_value), _fmt(hit.q_value),
                     f"{hit.n_agree}/{hit.n_guides}", _fmt(hit.agreement),
                     ", ".join(hit.flags)]
            rows.append("<tr>" + "".join(
                f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
        head = ("rank", "gene", "name", "effect", "p", "q", "guides agreeing",
                "agreement", "flags")
        return (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>spaCR hit list — {html.escape(self.source or '')}</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:.25rem .5rem;font-size:14px}th{background:#eee}</style>"
            f"<h1>Hit list</h1><p>{html.escape(self.source or '')}</p>"
            f"<p>{summary['n_significant']} of {summary['n_genes_tested']} "
            f"genes clear {self.alpha:g}; {summary['n_up']} up, "
            f"{summary['n_down']} down.</p>"
            "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
            + "".join(rows) + "</table>")

    def write_html(self, path: Union[str, os.PathLike]) -> str:
        """Write :meth:`to_html` to a file and return the path."""
        target = os.path.abspath(os.path.expanduser(os.fspath(path)))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(self.to_html())
        return target


def _at_most(value: Any, limit: float) -> bool:
    """True when ``value`` is a real number no greater than ``limit``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number <= float(limit)


def _at_least(value: Any, limit: float) -> bool:
    """True when ``value`` is a real number no smaller than ``limit``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= float(limit)


def _fmt(value: Any) -> str:
    """Format a number for a table cell; an em dash for a missing one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if number and (abs(number) < 1e-3 or abs(number) >= 1e5):
        return f"{number:.3g}"
    return f"{number:.4g}"


def _searchable(hit: Hit) -> str:
    """Everything about a row that a text query should match, folded."""
    parts = [hit.gene, hit.name, hit.feature, hit.condition]
    parts.extend(str(value) for value in hit.annotation.values())
    return " ".join(parts).casefold()


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------

def build_hit_list(source: Union[str, os.PathLike, Mapping[str, pd.DataFrame]],
                   *,
                   metadata_files: Sequence[Union[str, os.PathLike]] = (),
                   metadata_key: str = "Gene ID",
                   regression_type: str = "",
                   alpha: float = DEFAULT_ALPHA,
                   include_controls: bool = True,
                   ) -> HitList:
    """Build the ranked, annotated hit list of one regression run.

    :param source: a results folder, or a ``{role: DataFrame}`` mapping as
        :func:`load_results` returns — the second form is what a caller with
        the frames already in hand (or a test) uses.
    :param metadata_files: annotation CSVs to join, each collapsed to one row
        per gene first. See :func:`load_gene_metadata`.
    :param metadata_key: the gene identifier column in those files.
    :param regression_type: the backend, if known. Only affects how the list
        is ranked: the penalised backends have no p-value, so they rank by
        bootstrap selection frequency and carry no q-value.
    :param alpha: the FDR (or the selection frequency) a hit is called at.
    :param include_controls: keep the control rows in the list. They belong
        there by default — a screen whose positive control is not near the
        top has a problem, and that is visible only if it is listed.
    :returns: a :class:`HitList`, ranked and with exactly one row per gene.
    :raises FileNotFoundError: when ``source`` is a path that is not a folder.
    :raises ValueError: when no gene-level coefficients could be found.
    """
    notes: List[str] = []
    if isinstance(source, Mapping):
        frames = dict(source)
        where = str(frames.pop("__source__", "")) or "(frames)"
    else:
        where = os.path.abspath(os.path.expanduser(os.fspath(source)))
        frames = load_results(where)

    gene_frame = _gene_level(frames)
    if gene_frame is None or gene_frame.empty:
        raise ValueError(
            f"no gene-level coefficients in {where}: expected "
            f"{RESULT_FILES['gene']} or a {RESULT_FILES['all']} carrying "
            f"gene terms.")
    n_terms = int(len(frames.get("all", gene_frame)))

    table = gene_frame.copy()
    table["gene"] = table["feature"].map(gene_of)
    table = table.dropna(subset=["gene"])
    duplicated = int(table["gene"].duplicated().sum())
    if duplicated:
        notes.append(
            f"{duplicated} duplicate gene term(s) in the coefficient table; "
            f"the first of each is kept.")
        table = table.drop_duplicates(subset=["gene"], keep="first")
    if not include_controls and "condition" in table.columns:
        table = table[~table["condition"].isin(["nc", "pc", "control"])]
    table = table.reset_index(drop=True)

    ranking = ("selection-frequency"
               if str(regression_type).strip().lower() in NO_P_VALUE_TYPES
               else "q-value")
    if ranking == "q-value":
        table["q_value"] = benjamini_hochberg(
            table.get("p_value", pd.Series([np.nan] * len(table))))
    else:
        table["q_value"] = np.nan
        notes.append(
            f"{regression_type} reports no frequentist p-value, so this list "
            f"is ranked by bootstrap selection frequency and carries no "
            f"q-value. Treat it as a selection method, not a hypothesis test.")

    effects = {str(row["gene"]): float(row["coefficient"])
               for _, row in table.iterrows()
               if _finite(row.get("coefficient"))}
    agreement = grna_agreement(effects, frames.get("grna"))
    if frames.get("grna") is None or frames["grna"].empty:
        notes.append(
            "No per-gRNA coefficient table was found, so guide agreement "
            "could not be computed for any gene.")

    joined, join_notes = join_metadata(table, metadata_files,
                                       key=metadata_key)
    notes.extend(join_notes)
    if len(joined) != len(table):  # pragma: no cover - validate already raises
        raise ValueError(
            f"the metadata join changed the row count from {len(table)} to "
            f"{len(joined)}; the annotation is not one row per gene.")
    annotated_columns = [c for c in joined.columns if c not in table.columns]

    hits = [
        _hit(row, agreement, annotated_columns, ranking)
        for _, row in joined.iterrows()
    ]
    hits = _rank(hits, ranking)

    result = HitList(
        hits=tuple(hits), source=where, regression_type=regression_type,
        ranking=ranking, alpha=float(alpha), n_terms=n_terms,
        n_genes=len(hits), notes=tuple(notes))
    _assert_one_row_per_gene(result)
    return result


def _gene_level(frames: Mapping[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """The gene-level coefficient table, however this run spelled it.

    ``results_gene.csv`` when it exists; otherwise the gene terms filtered out
    of ``results.csv``, which is what a run that predates the split wrote.
    """
    gene = frames.get("gene")
    if gene is not None and not gene.empty and "feature" in gene.columns:
        return gene
    everything = frames.get("all")
    if everything is None or everything.empty:
        return None
    if "feature" not in everything.columns:
        return None
    mask = everything["feature"].astype(str).str.contains(r"gene\[",
                                                          regex=True)
    return everything[mask]


def _finite(value: Any) -> bool:
    """True when ``value`` converts to a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _hit(row: Mapping[str, Any],
         agreement: Mapping[str, Tuple[int, int, List[str]]],
         annotation_columns: Sequence[str], ranking: str) -> Hit:
    """Turn one joined coefficient row into a :class:`Hit`."""
    gene = str(row["gene"])
    effect = float(row["coefficient"]) if _finite(row.get("coefficient")) \
        else float("nan")
    std_err = float(row["std_err"]) if _finite(row.get("std_err")) \
        else float("nan")
    if math.isfinite(std_err) and math.isfinite(effect):
        ci_low, ci_high = effect - 1.96 * std_err, effect + 1.96 * std_err
    else:
        ci_low = ci_high = float("nan")

    n_agree, n_guides, agreeing = agreement.get(gene, (0, 0, []))
    ratio = (n_agree / n_guides) if n_guides else float("nan")

    flags: List[str] = []
    condition = str(row.get("condition", "") or "")
    if condition in ("nc", "pc", "control"):
        flags.append(FLAG_CONTROL)
    if n_guides == 0:
        flags.append(FLAG_NO_GUIDES)
    elif n_guides == 1:
        flags.append(FLAG_SINGLE_GUIDE)
    elif math.isfinite(ratio) and ratio < 0.5:
        flags.append(FLAG_GUIDES_DISAGREE)

    annotation = {column: row.get(column) for column in annotation_columns}
    if annotation_columns and all(
            value is None or (isinstance(value, float) and math.isnan(value))
            for value in annotation.values()):
        flags.append(FLAG_NO_METADATA)

    return Hit(
        gene=gene, feature=str(row.get("feature", "")), effect=effect,
        std_err=std_err, ci_low=ci_low, ci_high=ci_high,
        p_value=float(row["p_value"]) if _finite(row.get("p_value"))
        else float("nan"),
        q_value=float(row["q_value"]) if _finite(row.get("q_value"))
        else float("nan"),
        selection_frequency=float(row["selection_frequency"])
        if _finite(row.get("selection_frequency")) else float("nan"),
        n_guides=int(n_guides), n_agree=int(n_agree), agreement=ratio,
        agreeing_guides=tuple(agreeing),
        n_obs=int(row["n_gene"]) if _finite(row.get("n_gene")) else 0,
        condition=condition,
        direction=("up" if math.isfinite(effect) and effect > 0
                   else "down" if math.isfinite(effect) and effect < 0
                   else ""),
        flags=tuple(flags), annotation=annotation)


def _rank(hits: Sequence[Hit], ranking: str) -> List[Hit]:
    """Order the rows and stamp a 1-based rank on each.

    Significance first, magnitude second: a screen's question is "which genes
    changed the phenotype", and two genes at the same q-value are separated
    by how much they moved it. Untestable rows sort last rather than being
    dropped — a coefficient with no p-value is still a number somebody may
    need to see.
    """
    if ranking == "selection-frequency":
        def key(hit: Hit):
            selection = (hit.selection_frequency
                         if math.isfinite(hit.selection_frequency) else -1.0)
            magnitude = abs(hit.effect) if math.isfinite(hit.effect) else -1.0
            return (-selection, -magnitude, hit.gene)
    else:
        def key(hit: Hit):
            q = hit.q_value if math.isfinite(hit.q_value) else float("inf")
            magnitude = abs(hit.effect) if math.isfinite(hit.effect) else -1.0
            return (q, -magnitude, hit.gene)

    ordered = sorted(hits, key=key)
    return [Hit(**{**hit.__dict__, "rank": index + 1})
            for index, hit in enumerate(ordered)]


def _assert_one_row_per_gene(hit_list: HitList) -> None:
    """Refuse to hand back a list that names a gene twice.

    The invariant this module exists to keep. It is checked on the OUTPUT
    rather than trusted from the inputs because there are three places a
    duplicate can enter — a coefficient table with two terms for one gene, a
    metadata file with one row per transcript, and a second metadata file
    that repeats the first — and a hit list that counts one gene as two
    findings is worse than no hit list.
    """
    seen = set()
    for hit in hit_list.hits:
        if hit.gene in seen:
            raise ValueError(
                f"gene {hit.gene!r} appears more than once in the hit list; "
                f"a metadata file with one row per transcript is the usual "
                f"cause. Collapse it to one row per gene first.")
        seen.add(hit.gene)
