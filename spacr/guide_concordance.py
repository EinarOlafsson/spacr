"""Does a gene's signal come from its guides agreeing, or from one guide?

WHY THIS EXISTS

Ranking a screen by gene-level p-value silently mixes two different things:

* a gene whose guides all move the same way, none of them individually
  striking -- the signature of a real, moderate effect; and
* a gene with ONE surviving guide, whose "gene-level" term is arithmetically
  identical to that guide's own term and carries no independent evidence.

On the TSG101 screen the second kind sits at the top of the list. Gene 244480
has three guides in the library, exactly one survives the read-fraction
filter, and its gene p-value (1.6e-12) IS that guide's p-value -- ranked above
EAF1 and GRA14. EAF1 is the opposite case: three guides fitted, not one of them
significant on its own (p = 0.51, 0.14, 0.27), and a gene-level p of 4.6e-08.

A hit list that does not separate these is not wrong so much as unreadable,
and the distinction is invisible in the volcano because both are one dot.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: ``fraction:grna[TGGT1_225160_2]`` / ``gene_fraction:gene[T.225160]`` and the
#: bracket-less variants statsmodels produces for some families.
_GENE_IN_FEATURE = re.compile(r"\[(?:T\.)?(?:TGGT1_)?([0-9A-Za-z]+?)(?:_[0-9]+)?\]")
_GUIDE_IN_FEATURE = re.compile(r"\[(?:T\.)?((?:TGGT1_)?[0-9A-Za-z]+_[0-9]+)\]")


def _gene_of(feature: str):
    match = _GENE_IN_FEATURE.search(str(feature))
    return match.group(1) if match else None


def _is_guide_term(feature: str) -> bool:
    text = str(feature)
    return text.startswith("fraction:grna") or bool(_GUIDE_IN_FEATURE.search(text))


def _is_gene_term(feature: str) -> bool:
    return str(feature).startswith("gene_fraction:gene")


def guide_support(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One row per gene: how many guides back it, and how well they agree.

    :param results: a regression coefficient table with ``feature``,
        ``coefficient`` and ``p_value``.
    :param alpha: what counts as an individually significant guide.
    :returns: a frame indexed by gene with

        ``n_guides``
            guide terms actually fitted, i.e. guides that survived filtration.
        ``n_guides_significant``
            how many reached ``alpha`` on their own.
        ``n_same_direction``
            how many share the sign of the gene's mean effect. Guides that
            disagree in DIRECTION are the strongest argument that a hit is
            noise, and no p-value threshold reveals that.
        ``concordance``
            ``n_same_direction / n_guides``. 1.0 means every guide points the
            same way.
        ``single_guide``
            True when one guide carries the whole gene. Such a gene's
            gene-level p is that guide's p and is not independent evidence.
        ``gene_p``
            the gene-level term's p-value, when there is one.
    """
    if results is None or not len(results) or "feature" not in results.columns:
        return pd.DataFrame(columns=[
            "gene", "n_guides", "n_guides_significant", "n_same_direction",
            "concordance", "single_guide", "gene_p", "gene_coefficient",
            "best_guide_p"]).set_index("gene")

    frame = results.copy()
    frame["feature"] = frame["feature"].astype(str)
    frame["_gene"] = frame["feature"].map(_gene_of)
    frame["_p"] = pd.to_numeric(frame.get("p_value"), errors="coerce")
    effect = next((c for c in ("coefficient", "coef", "estimate")
                   if c in frame.columns), None)
    frame["_effect"] = pd.to_numeric(frame[effect], errors="coerce") \
        if effect else np.nan

    guides = frame[frame["feature"].map(_is_guide_term) & frame["_gene"].notna()]
    genes = frame[frame["feature"].map(_is_gene_term) & frame["_gene"].notna()]

    rows = []
    for gene, block in guides.groupby("_gene"):
        effects = block["_effect"].to_numpy(dtype="float64")
        p_values = block["_p"].to_numpy(dtype="float64")
        finite = effects[np.isfinite(effects)]
        if len(finite):
            # Sign of the MEAN, not of the strongest guide: asking whether the
            # guides agree with each other is the question, and letting the
            # largest one define "correct" would make disagreement invisible.
            direction = np.sign(np.mean(finite)) or 1.0
            same = int(np.sum(np.sign(finite) == direction))
        else:
            same = 0
        gene_row = genes[genes["_gene"] == gene]
        rows.append({
            "gene": gene,
            "n_guides": int(len(block)),
            "n_guides_significant": int(np.nansum(p_values <= alpha)),
            "n_same_direction": same,
            "concordance": (same / len(block)) if len(block) else np.nan,
            "single_guide": len(block) <= 1,
            "gene_p": float(gene_row["_p"].min()) if len(gene_row) else np.nan,
            "gene_coefficient": (float(gene_row["_effect"].iloc[0])
                                 if len(gene_row) else np.nan),
            "best_guide_p": float(np.nanmin(p_values)) if len(p_values) else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "gene", "n_guides", "n_guides_significant", "n_same_direction",
            "concordance", "single_guide", "gene_p", "gene_coefficient",
            "best_guide_p"]).set_index("gene")
    return pd.DataFrame(rows).set_index("gene").sort_values("gene_p")


def annotate_results(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """``results`` with the guide-support columns joined on.

    Returned as a copy: a diagnostic must not quietly rewrite the table the
    caller is about to save.
    """
    support = guide_support(results, alpha=alpha)
    if not len(support):
        return results.copy() if results is not None else results
    out = results.copy()
    out["_gene"] = out["feature"].astype(str).map(_gene_of)
    joined = out.join(support[["n_guides", "n_guides_significant",
                               "concordance", "single_guide"]],
                      on="_gene")
    return joined.drop(columns=["_gene"])


def flag_single_guide_hits(results: pd.DataFrame, alpha: float = 0.05,
                           p_column: str = "p_value") -> pd.DataFrame:
    """Hits whose evidence is one guide. Ranked as the table ranks them.

    These are not necessarily false -- a single guide can be the only one that
    cut -- but they are a different claim from a gene whose guides agree, and
    a hit list that presents them identically invites the reader to treat them
    the same.
    """
    support = guide_support(results, alpha=alpha)
    if not len(support):
        return support
    hits = support[support["gene_p"] <= alpha] if "gene_p" in support else support
    return hits[hits["single_guide"]].sort_values("gene_p")


def concordance_report(results: pd.DataFrame, alpha: float = 0.05,
                       top: int = 15, controls: dict | None = None) -> str:
    """A few lines a human can read, for the console after a run.

    :param controls: ``{gene_id: role}``, e.g.
        ``{"239740": "positive", "233460": "negative"}``. A negative control
        appearing in the hit list is the most useful line in the report and
        is easy to miss when it is just another six-digit number.
    """
    controls = {str(k): v for k, v in (controls or {}).items()}
    support = guide_support(results, alpha=alpha)
    if not len(support):
        return "No guide-level terms were fitted, so guide support is unknown."

    hits = support[support["gene_p"] <= alpha].head(top)
    if not len(hits):
        return "No gene reached the significance threshold."

    lines = [f"Guide support for the top {len(hits)} gene(s):",
             f"  {'gene':<12}{'guides':>7}{'sig':>5}{'agree':>8}"
             f"{'gene p':>12}   note"]
    for gene, row in hits.iterrows():
        note = ""
        if str(gene) in controls:
            note = f"{controls[str(gene)].upper()} CONTROL -- "
        if row["single_guide"]:
            note += "SINGLE GUIDE, gene p IS that guide's p"
        elif row["concordance"] < 0.6:
            note += "guides disagree in direction"
        elif row["n_guides_significant"] == 0:
            note += "no guide significant alone; the agreement is the evidence"
        agree = f"{int(row['n_same_direction'])}/{int(row['n_guides'])}"
        lines.append(
            f"  {str(gene):<12}{int(row['n_guides']):>7}"
            f"{int(row['n_guides_significant']):>5}{agree:>8}"
            f"{row['gene_p']:>12.2e}   {note}")
    return "\n".join(lines)
