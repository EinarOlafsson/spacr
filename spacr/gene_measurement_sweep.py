"""Every gene against every measurement, corrected, in one matrix operation.

The question is "which genes move which measurements", asked of the whole
screen at once rather than one gene at a time.

IT IS ONE MATMUL. A representative screen with 1,376 guides x 785
measurements over 1,366 wells, computed in 0.1 seconds. A loop over a million
regressions answers the same question in a morning; anything here that looks
like one is a bug.

THE THREE CORRECTIONS ARE THE POINT OF THE MODULE. Each was found by hand
while sweeping ONE gene, in about ten minutes, and each would be made again by
anyone doing this themselves:

  * IDENTIFIERS ARE NOT MEASUREMENTS. `pathogen_object_label` came out at
    p=2.5e-07 for EAF1 and it is a LABEL -- the cells picked out sit lower in
    the segmentation's numbering, which is position in a list.
    `pathogen_pathogen` is the same column under another name (spearman
    0.9979, identical in 141,626 of 226,467 rows). Both were nearly reported.
  * 785 TESTS IS NOT ONE TEST. `pathogen_solidity` at p=1.8e-03 looked
    interesting and does not survive Benjamini-Hochberg.
  * CIRCULARITY IS A COLUMN. The classification score is a function of the
    image, so a measurement it already tracks cannot corroborate anything
    derived from it. On this screen
    spearman(pred, pathogen_channel_1_mean_intensity) = -0.389, and the
    "strongest result" for GRA14 was exactly that measurement, in exactly the
    direction the correlation predicts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "IDENTIFIER_PATTERNS",
    "SweepResult",
    "is_measurement",
    "measurement_columns",
    "sweep",
    "gene_fractions",
    "gene_of_guide",
]

#: WHOLE TOKENS, not substrings. `object` and `label` catch the
#: segmentation's own numbering, which is what produced a p of 2.5e-07 and
#: means nothing.
#:
#: MATCHED AS TOKENS BECAUSE SUBSTRINGS ARE WRONG HERE: "path" is in
#: "png_path" and it is also in "pathogen", so a substring rule silently
#: dropped EVERY pathogen measurement -- a third of the screen -- and the
#: sweep reported the remainder as though that were all there was. Caught by
#: the test that asks whether `pathogen_area` is a measurement.
IDENTIFIER_PATTERNS: Tuple[str, ...] = (
    "label", "id", "object", "screen", "source", "prc", "prcf", "prcfo",
    "plate", "row", "col", "column", "field", "well", "path", "file",
    "name", "png", "index",
)


def _tokens(name: str) -> Tuple[str, ...]:
    """``plateID`` and ``object_label`` alike, as lower-case words."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name).strip())
    return tuple(t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t)


def is_measurement(name: str) -> bool:
    """Whether ``name`` measures the object rather than naming it."""
    parts = _tokens(name)
    if not parts:
        return False
    return not any(token in IDENTIFIER_PATTERNS for token in parts)


def measurement_columns(frame: pd.DataFrame) -> List[str]:
    """Every numeric column of ``frame`` that is a measurement.

    A COLUMN THAT DUPLICATES AN IDENTIFIER IS ALSO OUT, whatever it is called.
    `pathogen_pathogen` passes the name test and is the object label to four
    decimal places; the only way to catch it is to look.
    """
    numeric = [c for c in frame.columns
               if pd.api.types.is_numeric_dtype(frame[c])]
    named = [c for c in numeric if is_measurement(c)]
    identifiers = [c for c in numeric if not is_measurement(c)]
    if not identifiers:
        return named
    # RANK ONCE, THEN ONE MATMUL -- not a correlation per pair. The pairwise
    # version was 715 x 70 = 50,000 spearman calls and did not finish in two
    # minutes, which is the very thing this module's docstring warns about.
    usable = [c for c in named
              if pd.to_numeric(frame[c], errors="coerce").nunique() >= 3]
    if not usable:
        return []
    left = frame[usable].apply(pd.to_numeric, errors="coerce").rank()
    right = frame[identifiers].apply(pd.to_numeric, errors="coerce").rank()
    left = left.fillna(left.mean()).to_numpy(dtype=float)
    right = right.fillna(right.mean()).to_numpy(dtype=float)
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    left /= (np.linalg.norm(left, axis=0) + 1e-12)
    right /= (np.linalg.norm(right, axis=0) + 1e-12)
    twin = np.abs(left.T @ right).max(axis=1) > 0.99
    return [c for c, is_twin in zip(usable, twin) if not is_twin]


@dataclass(frozen=True)
class SweepResult:
    """The grid, and the tidy table a reader actually looks at."""

    table: pd.DataFrame
    effects: pd.DataFrame
    n_wells: int
    n_blocks: int
    dropped: Tuple[str, ...] = ()
    #: Whether the score actually joined to the wells. False means the
    #: circularity column is NaN and MUST NOT be read as "not circular".
    circularity_known: bool = False

    def survivors(self, alpha: float = 0.05,
                  max_circularity: float = 1.0) -> pd.DataFrame:
        """Rows past the correction, optionally past a circularity bar too.

        :raises ValueError: a circularity bar was asked for and the score
            never joined to the wells. Filtering on a column of NaN returns
            nothing and looks like a result.
        """
        out = self.table[self.table["q"] < float(alpha)]
        if max_circularity < 1.0:
            if not self.circularity_known:
                raise ValueError(
                    "circularity was never computed -- the score did not join "
                    "to any well, so filtering on it would return nothing and "
                    "look like an answer. Check the plate names match "
                    "(a score CSV may say 'pplate1' where the database says "
                    "'plate1').")
            out = out[out["circularity"] < float(max_circularity)]
        return out.sort_values("q")

    def describe(self) -> str:
        survived = int((self.table["q"] < 0.05).sum())
        clean = (int(((self.table["q"] < 0.05)
                      & (self.table["circularity"] < 0.15)).sum())
                 if self.circularity_known else None)
        head = (f"{len(self.effects.index):,} gene/guide(s) x "
                f"{len(self.effects.columns):,} measurement(s) over "
                f"{self.n_wells:,} wells in {self.n_blocks} block(s). "
                f"{survived:,} pass Benjamini-Hochberg at 0.05")
        middle = (f", of which {clean:,} are not already tracked by the score. "
                  if clean is not None else
                  " (circularity NOT computed -- the score joined to no "
                  "well, so it must not be read as 'not circular'). ")
        return (head + middle
                + f"{len(self.dropped):,} identifier column(s) were left out.")


def _residualise(matrix: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Subtract each block's mean -- the plate cannot become a gene effect."""
    out = matrix.astype(float, copy=True)
    for value in np.unique(blocks):
        mask = blocks == value
        if mask.sum():
            out[mask] -= np.nanmean(out[mask], axis=0)
    return np.nan_to_num(out)




#: Organism prefixes a guide name may carry. The counts of a real screen name
#: guides `TGGT1_225160_2`; the regression's design names them `225160_2`.
GUIDE_PREFIXES: Tuple[str, ...] = ("TGGT1_", "TGME49_", "TGVEG_", "TGRH88_")


def gene_of_guide(guide: Any) -> Optional[str]:
    """The gene a GUIDE NAME belongs to, for both spellings in use.

    `spacr.hits.gene_of` reads a DESIGN TERM -- `fraction:grna[225160_1]` --
    and truncates at the first underscore. Handed the bare `TGGT1_225160_2`
    that a count table actually carries, that rule returns `TGGT1`: the
    organism, for every guide in the screen, which pools the entire library
    into one "gene". So a bracketed term still goes to `hits.gene_of`, and a
    bare name has its organism prefix removed first.
    """
    text = str(guide or "").strip()
    if not text:
        return None
    if "[" in text:
        from .hits import gene_of as _design_gene_of

        return _design_gene_of(text)
    for prefix in GUIDE_PREFIXES:
        if text.upper().startswith(prefix):
            text = text[len(prefix):]
            break
    head = text.split("_", 1)[0].strip()
    return head or None


def gene_fractions(fractions: pd.DataFrame,
                   gene_of: Optional[Any] = None) -> pd.DataFrame:
    """A gene's fraction in each well: the SUM of its guides' fractions.

    The same rule the regression applies -- `wells_for_coefficient` with
    `guide_aggregation='sum'` -- because "does this GENE move this
    measurement" must not be a different arithmetic from the fit that found
    the gene in the first place.

    :param gene_of: guide name -> gene id. Defaults to
        :func:`spacr.hits.gene_of`, which is the key the metadata join uses,
        so the two cannot disagree about which gene a guide belongs to.
    :returns: one column per gene. Guides that name no gene are left out
        rather than pooled into an "unknown" gene that no experiment ran.
    """
    if gene_of is None:
        gene_of = gene_of_guide

    mapping: Dict[str, List[str]] = {}
    for guide in fractions.columns:
        gene = None
        try:
            gene = gene_of(str(guide))
        except Exception:                                # noqa: BLE001
            gene = None
        if gene:
            mapping.setdefault(str(gene), []).append(guide)
    if not mapping:
        return pd.DataFrame(index=fractions.index)
    return pd.DataFrame(
        {gene: fractions[guides].sum(axis=1) for gene, guides in mapping.items()},
        index=fractions.index)

def sweep(wells: pd.DataFrame, fractions: pd.DataFrame, *,
          blocks: Optional[Sequence] = None,
          scores: Optional[Sequence[float]] = None,
          alpha: float = 0.05,
          controls: Optional[Sequence[str]] = None,
          min_wells: int = 5,
          measurements: Optional[Iterable[str]] = None,
          level: str = "guide") -> SweepResult:
    """Associate every guide with every measurement, blocked and corrected.

    :param wells: one row per well, the measurements in its columns.
    :param fractions: one row per well, one column per guide, holding that
        guide's fraction. Aligned to ``wells`` by index.
    :param blocks: the plate of each well. Absent means one block, which is
        honest but weaker: a plate difference can then look like a gene.
    :param scores: the per-well classification score, used ONLY to compute
        each measurement's circularity. Absent leaves that column at 0 and
        the caller must not read it as "not circular".
    :param controls: guide or gene names to MARK as controls. They are not
        removed: the regression drops the control COLUMNS of the plate because
        they are not part of the contrast it fits, but this asks a different
        question -- whether a gene moves a measurement -- and a control is
        exactly the thing whose answer you want to see. Marked so a reader can
        find them, never filtered away behind their back.
    :param min_wells: guides present in fewer wells than this are dropped --
        a correlation over three wells is not an effect.
    :param level: ``'guide'``, ``'gene'``, or ``'both'``. A gene's fraction in
        a well is the SUM of its guides' -- the same rule the regression
        applies, because "does this GENE move this measurement" must not be a
        different arithmetic from the fit that found the gene.
    :returns: a :class:`SweepResult`. With ``'both'`` the table carries a
        ``level`` column and the guide rows stay reachable beside the gene
        ones.
    """
    wanted = str(level or "guide").strip().lower()
    if wanted not in ("guide", "gene", "both"):
        raise ValueError(
            f"level must be 'guide', 'gene' or 'both'; got {level!r}")
    if wanted != "guide":
        genes = gene_fractions(fractions)
        if wanted == "gene":
            fractions = genes
        elif len(genes.columns):
            # Suffixed so a gene and one of its guides cannot collide in the
            # index -- `233460` the gene and `233460_1` the guide are
            # different rows and a reader must be able to tell which is which.
            fractions = pd.concat(
                [fractions, genes.rename(columns=lambda g: f"{g} (gene)")],
                axis=1)
    common = wells.index.intersection(fractions.index)
    wells = wells.loc[common]
    fractions = fractions.loc[common]
    n = int(len(common))
    if n < 3:
        empty = pd.DataFrame(columns=["guide", "measurement", "effect", "p",
                                      "q", "circularity", "n_wells"])
        return SweepResult(table=empty, effects=pd.DataFrame(), n_wells=n,
                           n_blocks=0)

    chosen = list(measurements) if measurements is not None \
        else measurement_columns(wells)
    dropped = tuple(c for c in wells.columns
                    if pd.api.types.is_numeric_dtype(wells[c])
                    and c not in chosen)

    present = (fractions > 0).sum(axis=0)
    guides = [g for g in fractions.columns if int(present.get(g, 0)) >= min_wells]
    if not chosen or not guides:
        empty = pd.DataFrame(columns=["guide", "measurement", "effect", "p",
                                      "q", "circularity", "n_wells"])
        return SweepResult(table=empty, effects=pd.DataFrame(), n_wells=n,
                           n_blocks=0, dropped=dropped)

    block = np.asarray(list(blocks) if blocks is not None else ["all"] * n)
    n_blocks = int(len(np.unique(block)))

    M = _residualise(wells[chosen].to_numpy(dtype=float), block)
    F = _residualise(fractions[guides].to_numpy(dtype=float), block)
    M /= (M.std(axis=0, keepdims=True) + 1e-12)
    F /= (F.std(axis=0, keepdims=True) + 1e-12)

    # ONE MATMUL. Every guide against every measurement, as a correlation
    # after the block means are gone.
    R = (F.T @ M) / max(n - n_blocks, 1)
    R = np.clip(np.nan_to_num(R), -0.999999, 0.999999)

    # THE DEGREES OF FREEDOM ARE THE GUIDE'S, NOT THE SCREEN'S.
    #
    # A guide present in 7 of 1,366 wells has a fraction vector that is zero
    # almost everywhere, and its correlation is carried by those 7 points.
    # Using n - blocks - 1 for it reported p = 0.0 from seven wells, which is
    # the most confident possible statement of almost nothing, and it sat at
    # the top of the table.
    #
    # The participation ratio (sum x^2)^2 / sum x^4 is the effective number of
    # wells actually carrying a predictor: it equals n for a dense one and
    # collapses to the count of non-zero wells for a sparse one. Cheap -- two
    # column sums -- and conservative in the direction that matters.
    sq = F * F
    n_eff = (sq.sum(axis=0) ** 2) / np.maximum((sq * sq).sum(axis=0), 1e-300)
    n_eff = np.clip(n_eff, 3.0, float(n))
    df_guide = np.maximum(n_eff - n_blocks - 1.0, 1.0)
    df = df_guide[:, None]
    # Representation, reported rather than corrected for: the right response
    # to a gene that is everywhere is to SEE that it is, not to have its
    # numbers quietly adjusted.
    presence = (fractions[guides] > 0)
    share_of = np.round(
        fractions[guides].where(presence).median(axis=0).fillna(0.0).to_numpy(), 4)
    ubiquity = (presence.sum(axis=0).to_numpy() >= 0.9 * n)
    marked = {str(c) for c in (controls or ())}
    t = R * np.sqrt(df / (1.0 - R * R))
    from scipy.stats import t as _t
    p = 2.0 * _t.sf(np.abs(t), df)

    # NaN, NOT ZERO, when it was never computed. A column of 0.00 reads as
    # "nothing here is circular", which is the most confident possible way to
    # say nothing -- and it is what the panel displayed before this line.
    circular = np.full(len(chosen), np.nan)
    circularity_known = False
    if scores is not None:
        # Ranked once and correlated as a matrix, for the same reason.
        s = pd.Series(np.asarray(list(scores), dtype=float), index=common).rank()
        block_m = wells[chosen].apply(pd.to_numeric, errors="coerce").rank()
        sv = s.fillna(s.mean()).to_numpy(dtype=float)
        mv = block_m.fillna(block_m.mean()).to_numpy(dtype=float)
        sv = sv - sv.mean()
        mv = mv - mv.mean(axis=0)
        denom = (np.linalg.norm(sv) * np.linalg.norm(mv, axis=0)) + 1e-12
        circular = np.abs((sv @ mv) / denom)
        # A SCORE THAT JOINED TO NOTHING MUST NOT READ AS "NOT CIRCULAR".
        # The score CSVs of a real screen carry `pplate1` where the
        # measurement databases carry `plate1`, so an un-canonicalised join
        # matches no well at all -- and the resulting all-NaN column was
        # reported as "0 of 5,959 hits are circular", which is the most
        # confident possible way to say nothing.
        overlap = int(pd.Series(np.asarray(list(scores), dtype=float),
                                index=common).notna().sum())
        circularity_known = overlap >= 3 and bool(np.isfinite(circular).any())
        if not circularity_known:
            circular = np.full(len(chosen), np.nan)

    # THE SUFFIX GOES FROM BOTH, or the table and the grid disagree about what
    # a row is called and `plot_sweep` looks a gene up under a name only the
    # table uses. It exists only to keep a gene and its guides apart while
    # they are concatenated, and a gene id never equals a guide id anyway.
    effects = pd.DataFrame(
        R, index=[str(g).replace(" (gene)", "") for g in guides],
        columns=chosen)
    table = pd.DataFrame({
        "guide": np.repeat(guides, len(chosen)),
        "measurement": np.tile(chosen, len(guides)),
        "effect": R.ravel(),
        "p": p.ravel(),
        "circularity": np.tile(circular, len(guides)),
        "level": np.repeat(["gene" if str(g).endswith(" (gene)")
                            or wanted == "gene" else "guide"
                            for g in guides], len(chosen)),
        # HOW MUCH OF THE SCREEN THIS GENE IS. Measured on the maintainer's
        # own: 220950 sits in ALL 1,536 wells at a median fraction of 0.176 --
        # 17.6% of every well -- while the median gene is in 73. With that
        # many wells a partial correlation of 0.396 is overwhelming, and a
        # 73-well gene needs a far larger effect to clear the same bar. So
        # ranking by q ranks by REPRESENTATION as much as by biology, and a
        # reader cannot see that unless it is on the row.
        "share": np.repeat(share_of, len(chosen)),
        "ubiquitous": np.repeat(ubiquity, len(chosen)),
        "control": np.repeat(
            [str(g).replace(" (gene)", "") in marked for g in guides],
            len(chosen)),
        "n_wells": np.repeat([int(present.get(g, 0)) for g in guides],
                             len(chosen)),
        # What the P VALUE was actually computed on, which is not the same
        # number and is the one a reader needs to judge it.
        "effective_wells": np.repeat(np.round(n_eff, 1), len(chosen)),
    })
    from .multiple_testing import adjust_p_values
    q, _rejected = adjust_p_values(table["p"].to_numpy(), method="fdr_bh",
                                   alpha=float(alpha))
    table["q"] = q
    table["guide"] = table["guide"].astype(str).str.replace(
        " (gene)", "", regex=False)
    table = table[["level", "guide", "measurement", "effect", "p", "q",
                   "circularity", "n_wells", "effective_wells", "share",
                   "ubiquitous", "control"]]
    return SweepResult(table=table, effects=effects, n_wells=n,
                       n_blocks=n_blocks, dropped=dropped,
                       circularity_known=circularity_known)


# --------------------------------------------------------------------------- #
#  The picture
# --------------------------------------------------------------------------- #

def _readable(figure, *axes) -> str:
    """Put spaCR's house ink on ``axes`` so the labels can be read.

    THE FIGURES WERE UNREADABLE WITHOUT THIS. Reported 2026-08-19: "i cannot
    see any of the x or y axes, i do see some elements but not all so it is
    difficult to interperet. any of them". These plots were built with bare
    matplotlib, whose default ink is near-black -- which on spaCR's dark
    theme is invisible against the panel it is drawn onto. Every other figure
    in the application goes through `spacr.figures.style`; these ten did not.

    SET ON THE AXIS BY HAND, not left to an rcParams context, for the reason
    `regression_diagnostics` records: rcParams only reach an artist when it
    is CREATED, so a style applied around a finished figure changes nothing
    that is already drawn.

    :returns: the ink colour, for a caller that has its own text to place.
    """
    from .figures.style import ROLES, TYPE_SCALE, WEIGHTS

    ink = ROLES["reference"]
    try:
        from .figures.style import resolve_ink, theme_target

        ink = resolve_ink(theme_target())
    except Exception:                        # pragma: no cover - style absent
        pass
    # TRANSPARENT, not a colour of our own: the page the figure lands on is
    # the application's, and painting white behind it is what makes a dark
    # theme look broken.
    try:
        figure.patch.set_alpha(0.0)
    except Exception:                        # pragma: no cover - defensive
        pass
    for axis in axes:
        if axis is None:
            continue
        try:
            axis.patch.set_alpha(0.0)
            axis.title.set_color(ink)
            axis.title.set_fontsize(TYPE_SCALE.get("label", 9))
            axis.xaxis.label.set_color(ink)
            axis.yaxis.label.set_color(ink)
            axis.tick_params(color=ink, labelcolor=ink, which="both")
            for spine in axis.spines.values():
                spine.set_edgecolor(ink)
                spine.set_linewidth(WEIGHTS.get("spine", 0.8))
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(False, which="both")
            legend = axis.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(ink)
        except Exception:                    # pragma: no cover - defensive
            continue
    return ink


def plot_sweep(result: "SweepResult", path: Optional[str] = None, *,
               alpha: float = 0.05, max_circularity: float = 1.0,
               top: int = 40, title: str = "", level: Optional[str] = None):
    """A heatmap of what SURVIVED, clustered so related things sit together.

    THE WHOLE GRID IS NOT A PICTURE. 1,240 guides x 767 measurements is
    951,080 cells; drawn, it is a texture, and every one of them is coloured
    whether or not it means anything. So the default view is the survivors --
    the guides and measurements with at least one entry past the correction --
    and everything else is a filter away.

    :param result: what :func:`sweep` returned.
    :param top: the most guides and measurements to draw. A screen with
        hundreds of survivors is a table, not a picture, and saying so beats
        drawing something illegible.
    :returns: the matplotlib Figure, or ``None`` when nothing survived.
    """
    import matplotlib.pyplot as plt

    keep = result.survivors(alpha=alpha, max_circularity=max_circularity)
    if not len(keep):
        return None

    # ONE LEVEL PER PICTURE. At `level='both'` the table holds a gene row and
    # a row for each of its guides, and drawn together they are the same
    # effect counted several times -- a block of near-identical rows that
    # reads as agreement between independent things. Genes by default,
    # because that is the question the sweep is usually asked.
    drawn = str(level or "").strip().lower()
    if "level" in keep.columns and keep["level"].nunique() > 1:
        drawn = drawn or "gene"
    if drawn and "level" in keep.columns:
        keep = keep[keep["level"] == drawn]
        if not len(keep):
            return None

    guides = (keep.groupby("guide")["q"].min().sort_values().head(top).index)
    measures = (keep.groupby("measurement")["q"].min()
                .sort_values().head(top).index)
    grid = result.effects.loc[
        [g for g in guides if g in result.effects.index],
        [m for m in measures if m in result.effects.columns]]
    if grid.empty:
        return None

    # ORDERED SO NEIGHBOURS ARE ALIKE. A heatmap whose rows are in the order
    # they happened to arrive hides every block structure in it; the
    # measurements of one compartment belong together and a reader looking for
    # "what kind of thing does this gene move" is looking for exactly that.
    grid = _order_like_neighbours(grid)

    height = max(3.0, 0.28 * len(grid.index) + 1.6)
    width = max(5.0, 0.34 * len(grid.columns) + 3.2)
    figure, axes = plt.subplots(figsize=(width, height))
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0
    image = axes.imshow(grid.to_numpy(), cmap="RdBu_r", vmin=-limit,
                        vmax=limit, aspect="auto")
    axes.set_xticks(range(len(grid.columns)))
    axes.set_xticklabels([c[:34] for c in grid.columns], rotation=90,
                         fontsize=7)
    axes.set_yticks(range(len(grid.index)))
    axes.set_yticklabels(grid.index, fontsize=7)
    axes.set_title(title or
                   f"{len(keep):,} association(s) past BH at {alpha:g}"
                   + (f", circularity < {max_circularity:g}"
                      if max_circularity < 1.0 else "")
                   + (f" — {drawn}s" if drawn else ""),
                   fontsize=9)
    bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.01)
    bar.set_label("effect (partial correlation, plate-blocked)", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def _order_like_neighbours(grid: pd.DataFrame) -> pd.DataFrame:
    """Rows and columns ordered so similar ones are adjacent.

    Hierarchical clustering when scipy is there, and a correlation-to-the-mean
    ordering when it is not -- a picture that fell back to arrival order would
    quietly lose the block structure the picture exists to show.
    """
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        def order(matrix):
            if matrix.shape[0] < 3:
                return list(range(matrix.shape[0]))
            distance = pdist(np.nan_to_num(matrix), metric="correlation")
            distance = np.nan_to_num(distance, nan=1.0)
            return list(leaves_list(linkage(distance, method="average")))

        rows = order(grid.to_numpy())
        columns = order(grid.to_numpy().T)
        return grid.iloc[rows, columns]
    except Exception:                                            # noqa: BLE001
        centre = grid.mean(axis=1).sort_values().index
        columns = grid.mean(axis=0).sort_values().index
        return grid.loc[centre, columns]


#: The measurement families, and the token that puts a column in one.
#:
#: Ordered: the first family whose token appears wins, so `pathogen_area`
#: is a pathogen measurement rather than a shape one. Deliberately COARSE --
#: six families a reader can hold in their head, not the 767 columns they
#: are drawn from. A column matching none is "other", which is a real answer
#: and not a failure: a screen may measure something none of these names.
MEASUREMENT_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("pathogen", ("pathogen",)),
    ("nucleus", ("nucleus", "nucleolus")),
    ("cytoplasm", ("cytoplasm", "cyto")),
    ("cell", ("cell",)),
    ("intensity", ("intensity", "quartile", "percentile", "mean", "median",
                   "std", "skew", "kurtosis")),
    ("shape", ("area", "perimeter", "eccentricity", "solidity", "extent",
               "diameter", "axis", "zernike", "moment")),
)


def measurement_family(name: Any) -> str:
    """Which family ``name`` belongs to. See :data:`MEASUREMENT_FAMILIES`."""
    tokens = set(_tokens(str(name)))
    for family, marks in MEASUREMENT_FAMILIES:
        if tokens & set(marks):
            return family
    return "other"


def plot_effect_against_representation(
        result: "SweepResult", path: Optional[str] = None, *,
        alpha: float = 0.05, title: str = "",
        level: Optional[str] = None):
    """Plot effect counts against each gene's effective representation.

    The plot exposes representation as a possible confound without modifying
    the underlying statistic. It shows whether a gene's number of significant
    measurements follows the overall representation trend or departs from it.

    x is the gene's EFFECTIVE WELL COUNT -- the participation ratio, which is
    literally the sample size each p-value was computed on -- and y is how
    many measurements it moved past the correction. A gene high on the trend
    line is doing what its statistical weight predicts; a gene ABOVE the line
    is the interesting one, and a gene at the far right with a huge count is
    exactly the one to be suspicious of.

    EFFECTIVE WELLS AND NOT `share`, WHICH MEASURES SOMETHING ELSE. `share`
    is the median fraction a gene takes of the wells it is IN -- how
    concentrated it is when present -- and a rare gene can score high on it
    precisely by being rare. Measured on this module's own fixture: a gene in
    18 of 120 wells has share 0.44 and one in all 120 has 0.20, which is the
    opposite of the ordering the reader is asking about. What drives the
    ranking is POWER, and the participation ratio is the number the test
    actually used.

    Controls are drawn with a distinct marker because their relationship to
    the trend provides a useful assay calibration.

    :returns: the matplotlib Figure, or ``None`` when nothing survived.
    """
    import matplotlib.pyplot as plt

    keep = result.table
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    if not len(keep):
        return None

    passed = keep[keep["q"] < float(alpha)]
    if not len(passed):
        # NOTHING, rather than every gene on a flat line at zero. That
        # picture is not false -- no gene passed -- but it reads as a
        # measured absence of effect when it is an absence of evidence, and
        # the two are the thing this whole module tries to keep apart.
        return None
    per_gene = keep.groupby("guide").agg(
        weight=("effective_wells", "first"),
        share=("share", "first"),
        wells=("n_wells", "first"),
        control=("control", "first"))
    per_gene["hits"] = passed.groupby("guide").size().reindex(
        per_gene.index, fill_value=0)
    per_gene = per_gene[per_gene["weight"].notna()]
    if not len(per_gene):
        return None

    figure, axes = plt.subplots(figsize=(7.2, 5.0))
    controls = per_gene[per_gene["control"].astype(bool)]
    rest = per_gene[~per_gene["control"].astype(bool)]
    axes.scatter(rest["weight"], rest["hits"], s=22, alpha=0.55,
                 edgecolor="none", label=f"{len(rest):,} gene(s)")
    if len(controls):
        axes.scatter(controls["weight"], controls["hits"], s=54, marker="D",
                     facecolor="none", edgecolor="#c1121f", linewidth=1.2,
                     label=f"{len(controls):,} control(s)")

    # THE TREND, and only when there is something to fit. Two points make a
    # line through themselves and say nothing; drawing it anyway would put a
    # confident diagonal on a plot that has no evidence for one.
    if len(per_gene) >= 8 and per_gene["weight"].nunique() > 2:
        x = per_gene["weight"].to_numpy(dtype=float)
        y = per_gene["hits"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        span = np.linspace(x.min(), x.max(), 50)
        axes.plot(span, slope * span + intercept, color="#6c757d",
                  linewidth=1.0, linestyle="--",
                  label="what statistical weight alone predicts")
        # Named on the plot, because the number is the answer to the question
        # and a reader should not have to eyeball the slope.
        rho = float(np.corrcoef(x, y)[0, 1]) if len(set(x)) > 1 else np.nan
        axes.set_title(
            title or (f"hits vs representation — rho = {rho:.2f} "
                      f"({'weight explains much of the ranking' if abs(rho) >= 0.5 else 'weight does not explain the ranking'})"),
            fontsize=9)
    else:
        axes.set_title(title or "hits vs representation", fontsize=9)

    axes.set_xlabel("effective wells — the sample size each p was computed "
                "on", fontsize=8)
    axes.set_ylabel(f"measurements moved past BH at {alpha:g}", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.legend(fontsize=7, frameon=False)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_measurement_families(result: "SweepResult",
                              path: Optional[str] = None, *,
                              alpha: float = 0.05, top: int = 14,
                              title: str = "", level: Optional[str] = None):
    """What KIND of thing each gene moves, as a stacked bar per gene.

    767 measurements is not a list a reader can hold, but six families is.
    "this gene moves pathogen intensity and nothing else" is a sentence about
    biology; "this gene has 41 significant measurements" is not.

    The families are coarse on purpose -- see :data:`MEASUREMENT_FAMILIES`.

    :param top: how many genes to draw, most-hits first.
    :returns: the matplotlib Figure, or ``None`` when nothing survived.
    """
    import matplotlib.pyplot as plt

    keep = result.survivors(alpha=alpha)
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    if not len(keep):
        return None

    keep = keep.assign(family=[measurement_family(m)
                              for m in keep["measurement"]])
    counts = keep.pivot_table(index="guide", columns="family",
                              values="q", aggfunc="size").fillna(0)
    order = counts.sum(axis=1).sort_values(ascending=False).head(top).index
    counts = counts.loc[order]
    if counts.empty:
        return None

    families = [f for f, _ in MEASUREMENT_FAMILIES] + ["other"]
    families = [f for f in families if f in counts.columns]

    figure, axes = plt.subplots(
        figsize=(7.6, max(3.0, 0.34 * len(counts.index) + 1.4)))
    left = np.zeros(len(counts.index))
    for family in families:
        values = counts[family].to_numpy(dtype=float)
        axes.barh(range(len(counts.index)), values, left=left, height=0.72,
                  label=family)
        left = left + values
    axes.set_yticks(range(len(counts.index)))
    axes.set_yticklabels(counts.index, fontsize=7)
    axes.invert_yaxis()
    axes.set_xlabel(f"measurements moved past BH at {alpha:g}", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.set_title(title or "what kind of measurement each gene moves",
                   fontsize=9)
    axes.legend(fontsize=7, frameon=False, ncol=min(4, len(families)))
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_guide_concordance(result: "SweepResult", path: Optional[str] = None,
                           *, alpha: float = 0.05, top: int = 20,
                           title: str = ""):
    """Do a gene's own guides agree with each other?

    THE ONLY INTERNAL CONTROL THIS DESIGN HAS. Two guides against the same
    gene are two independent perturbations of it, so a gene whose guides
    agree on the sign of an effect is saying something the gene's own
    biology explains, and a gene whose guides disagree is saying something
    about the guides.

    Needs a table with GUIDE rows: `sweep(..., level='guide')` or `'both'`.
    A gene-level table has nothing to compare, which is a reason to say so
    rather than to draw an empty axis.

    :returns: the matplotlib Figure, or ``None`` when there is nothing to
        compare.
    """
    import matplotlib.pyplot as plt

    table = result.table
    if "level" in table.columns:
        table = table[table["level"] == "guide"]
    if not len(table):
        return None

    genes = [gene_of_guide(g) for g in table["guide"]]
    table = table.assign(gene=genes)
    table = table[table["gene"].notna()]
    if not len(table):
        return None

    # A gene needs TWO guides to agree or disagree about anything.
    per_gene_guides = table.groupby("gene")["guide"].nunique()
    table = table[table["gene"].isin(
        per_gene_guides[per_gene_guides >= 2].index)]
    if not len(table):
        return None

    passed = table[table["q"] < float(alpha)]
    if not len(passed):
        return None

    rows = []
    for (gene, measurement), block in passed.groupby(["gene", "measurement"]):
        signs = np.sign(block["effect"].to_numpy(dtype=float))
        signs = signs[signs != 0]
        if len(signs) < 2:
            continue
        rows.append({"gene": gene,
                     "agree": float(np.abs(signs.sum()) / len(signs)),
                     "guides": int(len(signs))})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    summary = frame.groupby("gene").agg(agreement=("agree", "mean"),
                                        pairs=("agree", "size"))
    summary = summary.sort_values("agreement", ascending=False).head(top)

    figure, axes = plt.subplots(
        figsize=(6.6, max(3.0, 0.32 * len(summary.index) + 1.4)))
    colours = ["#2a9d8f" if v >= 0.99 else "#e9c46a" if v >= 0.6 else "#e76f51"
               for v in summary["agreement"]]
    axes.barh(range(len(summary.index)), summary["agreement"],
              color=colours, height=0.7)
    axes.set_yticks(range(len(summary.index)))
    axes.set_yticklabels(
        [f"{g}  ({int(n)} measurement(s))"
         for g, n in zip(summary.index, summary["pairs"])], fontsize=7)
    axes.invert_yaxis()
    axes.set_xlim(0.0, 1.02)
    axes.axvline(1.0, color="#6c757d", linewidth=0.8, linestyle=":")
    axes.set_xlabel("share of a gene's guides agreeing on the sign",
                    fontsize=8)
    axes.tick_params(labelsize=7)
    axes.set_title(title or "do a gene's own guides agree?", fontsize=9)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def _one_level(table: pd.DataFrame, level: Optional[str]) -> str:
    """Which level to draw, given what was asked and what the table holds.

    Shared by every picture here for the reason `plot_sweep` states: a gene
    row and its own guide rows drawn together are the same effect counted
    several times, which reads as agreement between independent things.
    """
    drawn = str(level or "").strip().lower()
    if "level" not in table.columns:
        return ""
    if table["level"].nunique() > 1:
        return drawn or "gene"
    return drawn if drawn else ""


# --------------------------------------------------------------------------- #
# The other six views                                                          #
# --------------------------------------------------------------------------- #
#
# Ten ways of looking at one grid, each answering a question the others
# cannot. The list is kept HERE rather than in a message, because the first
# four were built from a conversation and the other six nearly were not.


def plot_grid_volcano(result: "SweepResult", path: Optional[str] = None, *,
                      alpha: float = 0.05, title: str = "",
                      level: Optional[str] = None):
    """#5 -- every gene x measurement pair at once: effect against evidence.

    THE SHAPE OF THE WHOLE GRID, which the heatmap cannot show because it
    draws only survivors. A screen where everything is significant looks
    different here from one with a handful of real effects, and that
    difference is the first thing to check before reading any single row.

    Colour is CIRCULARITY where it is known -- a hit the classifier already
    tracks is a restatement, not a corroboration -- and grey where it is not.
    Grey is not "clean": the sweep says so in the legend rather than letting
    an uncomputed number read as zero.
    """
    import matplotlib.pyplot as plt

    keep = result.table
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    # COERCED, not assumed numeric: an empty frame built from a column list
    # carries object dtype, and `np.isfinite` on that raises a TypeError
    # rather than returning an empty mask.
    effect_values = pd.to_numeric(keep["effect"], errors="coerce")
    p_values = pd.to_numeric(keep["p"], errors="coerce")
    keep = keep[np.isfinite(effect_values) & np.isfinite(p_values)]
    if not len(keep):
        return None

    effect = keep["effect"].to_numpy(dtype=float)
    evidence = -np.log10(np.clip(keep["p"].to_numpy(dtype=float), 1e-300, 1.0))

    figure, axes = plt.subplots(figsize=(7.4, 5.2))
    if result.circularity_known:
        colours = keep["circularity"].to_numpy(dtype=float)
        dots = axes.scatter(effect, evidence, c=colours, cmap="viridis",
                            s=12, alpha=0.7, edgecolor="none", vmin=0.0,
                            vmax=max(0.5, float(np.nanmax(colours) or 0.5)))
        bar = figure.colorbar(dots, ax=axes, fraction=0.03, pad=0.01)
        bar.set_label("|rho(score, measurement)| — higher is circular",
                      fontsize=7)
        bar.ax.tick_params(labelsize=6)
    else:
        axes.scatter(effect, evidence, s=12, alpha=0.6, color="#6c757d",
                     edgecolor="none",
                     label="circularity NOT computed — not the same as zero")
        axes.legend(fontsize=7, frameon=False, loc="upper left")

    passed = keep[keep["q"] < float(alpha)]
    if len(passed):
        # The BH line is where the correction actually fell, not a nominal
        # 0.05: drawing the nominal one puts the threshold in the wrong place
        # on every corrected screen.
        cut = float(passed["p"].max())
        axes.axhline(-np.log10(max(cut, 1e-300)), color="#c1121f",
                     linewidth=0.9, linestyle="--")
        axes.text(axes.get_xlim()[0], -np.log10(max(cut, 1e-300)),
                  f" BH q<{alpha:g} falls at p={cut:.2g}", fontsize=7,
                  va="bottom", color="#c1121f")

    axes.set_xlabel("effect (partial correlation, plate-blocked)", fontsize=8)
    axes.set_ylabel("-log10(raw p)", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.set_title(title or f"{len(keep):,} gene x measurement pair(s)",
                   fontsize=9)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_gene_profile(result: "SweepResult", gene: Any,
                      path: Optional[str] = None, *, alpha: float = 0.05,
                      top: int = 24, title: str = ""):
    """#6 -- ONE gene's fingerprint: every measurement it moves, in order.

    The per-gene readout the grid cannot give. "GRA14 moves pathogen
    intensity up and pathogen area down, and nothing else" is a sentence
    somebody can take to a bench; a row of a heatmap is not.

    Bars are coloured by measurement family, so a profile that is all one
    family reads as one finding rather than as twenty.
    """
    import matplotlib.pyplot as plt

    name = str(gene)
    table = result.table
    mine = table[table["guide"].astype(str) == name]
    if not len(mine):
        return None
    passed = mine[mine["q"] < float(alpha)]
    # Fall back to the strongest effects when nothing cleared the
    # correction: "this gene has no significant measurement" is worth
    # SEEING, and an empty axis does not say it.
    shown = passed if len(passed) else mine
    shown = shown.reindex(
        shown["effect"].abs().sort_values(ascending=False).index).head(top)
    if not len(shown):
        return None

    families = [measurement_family(m) for m in shown["measurement"]]
    order = [f for f, _ in MEASUREMENT_FAMILIES] + ["other"]
    palette = {f: plt.get_cmap("tab10")(i % 10) for i, f in enumerate(order)}

    figure, axes = plt.subplots(
        figsize=(7.0, max(3.0, 0.30 * len(shown) + 1.4)))
    positions = range(len(shown))
    axes.barh(list(positions), shown["effect"].to_numpy(dtype=float),
              color=[palette[f] for f in families], height=0.7)
    axes.axvline(0.0, color="#343a40", linewidth=0.8)
    axes.set_yticks(list(positions))
    axes.set_yticklabels([m[:44] for m in shown["measurement"]], fontsize=7)
    axes.invert_yaxis()
    axes.set_xlabel("effect (partial correlation, plate-blocked)", fontsize=8)
    axes.tick_params(labelsize=7)
    seen = []
    for family in order:
        if family in families and family not in seen:
            seen.append(family)
    axes.legend(handles=[plt.Line2D([], [], color=palette[f], linewidth=6,
                                    label=f) for f in seen],
                fontsize=7, frameon=False, ncol=min(4, len(seen)))
    axes.set_title(
        title or (f"{name} — {len(passed):,} measurement(s) past BH at "
                  f"{alpha:g}" if len(passed) else
                  f"{name} — NOTHING past BH at {alpha:g}; the strongest "
                  f"effects are shown"),
        fontsize=9)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_gene_similarity(result: "SweepResult", path: Optional[str] = None, *,
                         alpha: float = 0.05, top: int = 30,
                         title: str = "", level: Optional[str] = None):
    """#7 -- which genes behave ALIKE, by correlating their whole profiles.

    Two genes in one pathway should move the same measurements the same way,
    and this is the only view here that can say so: every other one reads a
    gene on its own. It is also the honest way to ask "is my hit list one
    finding or twelve".

    Correlated across the WHOLE effect row, not just the significant part --
    a shared sub-threshold pattern is exactly the evidence that two genes
    belong together, and thresholding first would throw it away.
    """
    import matplotlib.pyplot as plt

    keep = result.survivors(alpha=alpha)
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    if not len(keep):
        return None

    ranked = (keep.groupby("guide")["q"].min().sort_values().head(top).index)
    genes = [g for g in ranked if g in result.effects.index]
    if len(genes) < 2:
        return None

    profiles = result.effects.loc[genes].to_numpy(dtype=float)
    profiles = np.nan_to_num(profiles, nan=0.0)
    spread = profiles.std(axis=1, keepdims=True)
    spread[spread <= 0] = 1.0
    centred = (profiles - profiles.mean(axis=1, keepdims=True)) / spread
    similarity = (centred @ centred.T) / max(profiles.shape[1], 1)
    frame = pd.DataFrame(similarity, index=genes, columns=genes)
    frame = _order_like_neighbours(frame)

    size = max(3.6, 0.26 * len(frame.index) + 2.0)
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(frame.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes.set_xticks(range(len(frame.columns)))
    axes.set_xticklabels(frame.columns, rotation=90, fontsize=7)
    axes.set_yticks(range(len(frame.index)))
    axes.set_yticklabels(frame.index, fontsize=7)
    axes.set_title(title or f"do these {len(frame.index)} genes behave alike?",
                   fontsize=9)
    bar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("correlation of effect profiles", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_measurement_hits(result: "SweepResult", path: Optional[str] = None,
                          *, alpha: float = 0.05, top: int = 26,
                          title: str = "", level: Optional[str] = None):
    """#8 -- which MEASUREMENTS are informative, and which everything moves.

    The grid read down its other axis. A measurement moved by half the
    library is not a discriminating readout: it is a plate effect, a focus
    drift or a confluence artefact wearing a measurement's name, and it will
    put a hit on every gene in the screen. Ranking measurements by how many
    genes move them is how you find those before trusting any of them.
    """
    import matplotlib.pyplot as plt

    keep = result.survivors(alpha=alpha)
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    if not len(keep):
        return None

    counts = keep.groupby("measurement")["guide"].nunique().sort_values(
        ascending=False).head(top)
    if not len(counts):
        return None
    total = int(keep["guide"].nunique())

    figure, axes = plt.subplots(
        figsize=(7.4, max(3.0, 0.30 * len(counts) + 1.5)))
    share = counts.to_numpy(dtype=float) / max(total, 1)
    colours = ["#c1121f" if s >= 0.5 else "#457b9d" for s in share]
    axes.barh(range(len(counts)), counts.to_numpy(), color=colours,
              height=0.72)
    axes.set_yticks(range(len(counts)))
    axes.set_yticklabels([m[:46] for m in counts.index], fontsize=7)
    axes.invert_yaxis()
    axes.set_xlabel(f"genes moving it, of {total:,} tested", fontsize=8)
    axes.tick_params(labelsize=7)
    promiscuous = int((share >= 0.5).sum())
    axes.set_title(
        title or (f"which measurements discriminate"
                  + (f" — {promiscuous} moved by half the library or more, "
                     f"in red" if promiscuous else "")),
        fontsize=9)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_circularity(result: "SweepResult", path: Optional[str] = None, *,
                     alpha: float = 0.05, title: str = "",
                     level: Optional[str] = None):
    """Plot whether each surviving measurement adds independent evidence.

    A measurement the classifier already tracks cannot corroborate a result
    derived from that classifier. The plot exposes that dependence rather
    than presenting a correlated measurement as separate confirmation.

    Every point is drawn because the correlation cutoff is a judgment. This
    lets users see where their hits fall and choose a threshold appropriate
    to their screen.
    """
    import matplotlib.pyplot as plt

    if not result.circularity_known:
        # NOT AN EMPTY AXIS. The column is NaN, and a scatter of NaN is a
        # blank panel that reads as "nothing is circular" -- which is the
        # exact misreading this whole column exists to prevent.
        return None

    keep = result.survivors(alpha=alpha)
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    keep = keep[np.isfinite(keep["circularity"]) & np.isfinite(keep["effect"])]
    if not len(keep):
        return None

    figure, axes = plt.subplots(figsize=(7.0, 5.0))
    circular = keep["circularity"].to_numpy(dtype=float)
    axes.scatter(np.abs(keep["effect"].to_numpy(dtype=float)), circular,
                 s=16, alpha=0.6, edgecolor="none", color="#457b9d")
    axes.axhline(0.15, color="#c1121f", linewidth=0.9, linestyle="--")
    axes.text(axes.get_xlim()[1], 0.15, "0.15 — a working bar, not a law ",
              fontsize=7, ha="right", va="bottom", color="#c1121f")
    axes.set_xlabel("|effect| of the gene on the measurement", fontsize=8)
    axes.set_ylabel("|rho(classification score, measurement)|", fontsize=8)
    axes.tick_params(labelsize=7)
    above = int((circular >= 0.15).sum())
    axes.set_title(
        title or (f"{above:,} of {len(keep):,} surviving pair(s) sit on a "
                  f"measurement the score already tracks"), fontsize=9)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_calibration(result: "SweepResult", path: Optional[str] = None, *,
                     title: str = "", level: Optional[str] = None):
    """#10 -- is this screen calibrated, or is everything significant?

    The observed P values against the uniform they would follow if nothing
    were real. A grid that hugs the diagonal has no signal; one that lifts
    off it at the left has some; one that lifts off everywhere has a
    systematic effect -- a plate term that did not get blocked out, or an
    aggregation that correlated every well with itself.

    THE FIRST PLOT TO LOOK AT and the last one anybody builds. It says
    whether the other nine are worth reading at all.
    """
    import matplotlib.pyplot as plt

    keep = result.table
    drawn = _one_level(keep, level)
    if drawn:
        keep = keep[keep["level"] == drawn]
    values = keep["p"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None

    observed = np.sort(np.clip(values, 1e-300, 1.0))
    expected = (np.arange(1, observed.size + 1) - 0.5) / observed.size

    figure, axes = plt.subplots(figsize=(5.6, 5.4))
    axes.plot(-np.log10(expected), -np.log10(observed), ".", markersize=3,
              color="#457b9d")
    edge = float(max(-np.log10(expected).max(), -np.log10(observed).max()))
    axes.plot([0, edge], [0, edge], color="#6c757d", linewidth=0.9,
              linestyle="--", label="no effect anywhere")

    # THE INFLATION FACTOR, named. A number beats an eyeballed slope, and
    # this one has a standard meaning: lambda near 1 is calibrated, and
    # well above it means something systematic is inflating every test.
    from scipy.stats import chi2

    median = float(np.median(observed))
    lam = (chi2.isf(median, 1) / chi2.isf(0.5, 1)) if median > 0 else np.nan
    axes.set_xlabel("expected -log10(p)", fontsize=8)
    axes.set_ylabel("observed -log10(p)", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.legend(fontsize=7, frameon=False)
    axes.set_title(
        title or (f"calibration — lambda = {lam:.2f} "
                  f"({'calibrated' if 0.9 <= lam <= 1.15 else 'inflated' if lam > 1.15 else 'conservative'})"),
        fontsize=9)
    _readable(figure, axes)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure
