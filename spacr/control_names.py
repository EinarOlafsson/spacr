"""Resolve user-entered controls as gene or guide identifiers.

Controls may be bare genes (``000000``), bare guides (``000000_1``),
prefixed genes (``TGGT1_000000``), or prefixed guides
(``TGGT1_000000_1``). A leading token is treated as an organism or strain
prefix only when it occurs in the configured share of distinct library
identifiers. Matching uses complete identifiers rather than substrings.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

SEPARATOR = "_"

#: Minimum share of distinct identifiers that must carry a leading token for
#: it to be treated as an organism or strain prefix.
COMMON_PREFIX_SHARE = 0.9

GENE = "gene"
GUIDE = "grna"


@dataclass(frozen=True)
class ControlSpec:
    """Resolved interpretation of a user-entered control."""

    typed: str
    level: str
    value: str
    prefix: str = ""

    @property
    def is_gene(self) -> bool:
        return self.level == GENE

    def note(self, matched_guides: int = -1, matched_wells: int = -1) -> str:
        """Return a console summary of the resolution and optional matches."""
        what = "gene" if self.is_gene else "guide"
        head = f"control {self.typed!r} resolved to {what} {self.value!r}"
        if self.prefix and self.typed.startswith(self.prefix + SEPARATOR):
            head += f" (dropping the common prefix {self.prefix!r})"
        if matched_guides < 0:
            return head
        wells = "" if matched_wells < 0 else f", {matched_wells} well(s)"
        return f"{head}: {matched_guides} guide(s){wells}"


def common_prefix(names: Iterable[str],
                  share: float = COMMON_PREFIX_SHARE) -> str:
    """Return a leading token shared by enough distinct identifiers.

    Parameters
    ----------
    names : iterable of str
        Guide identifiers in the loaded data.
    share : float, default=COMMON_PREFIX_SHARE
        Required fraction of distinct identifiers carrying the token.

    Returns
    -------
    str
        Token without its separator, or an empty string when none reaches
        ``share``.
    """
    distinct = {str(name) for name in names if str(name)}
    if not distinct:
        return ""
    leading = Counter()
    for name in distinct:
        head, sep, rest = name.partition(SEPARATOR)
        # A token is only a candidate PREFIX if something follows it.
        if sep and rest:
            leading[head] += 1
    if not leading:
        return ""
    token, count = leading.most_common(1)[0]
    return token if count >= share * len(distinct) else ""


def resolve_control(typed, names: Optional[Iterable[str]] = None,
                    prefix: Optional[str] = None) -> Optional[ControlSpec]:
    """Read one typed control as a gene or as a guide.

    Parameters
    ----------
    typed : Any
        Entered identifier. Blank values return ``None``.
    names : iterable of str, optional
        Library guide identifiers used to infer a common prefix.
    prefix : str, optional
        Previously inferred prefix. When supplied, ``names`` is ignored.

    Returns
    -------
    ControlSpec or None
        Resolved gene or guide identifier, or ``None`` for no control.
    """
    text = "" if typed is None else str(typed).strip()
    if not text:
        return None
    if prefix is None:
        # NOT `names or ()`. A pandas Series is the natural thing to pass
        # here -- guide names live in a column -- and its truthiness raises
        # "The truth value of a Series is ambiguous" rather than falling
        # back. Caught the first time this was pointed at a real library.
        prefix = common_prefix(() if names is None else names)
    prefix = str(prefix or "")

    parts = text.split(SEPARATOR)
    if len(parts) >= 3:
        # Everything after the FIRST underscore is the guide, which is
        # exactly how process_reads stores a three-component name.
        return ControlSpec(text, GUIDE, SEPARATOR.join(parts[1:]), prefix)
    if len(parts) == 2:
        head, tail = parts
        if prefix and head == prefix:
            return ControlSpec(text, GENE, tail, prefix)
        return ControlSpec(text, GUIDE, text, prefix)
    return ControlSpec(text, GENE, text, prefix)


def resolve_controls(typed: Optional[Sequence],
                     names: Optional[Iterable[str]] = None,
                     prefix: Optional[str] = None
                     ) -> Tuple[ControlSpec, ...]:
    """Resolve a sequence containing any mixture of genes and guides.

    The common prefix is measured once and applied independently to each
    nonblank entry.
    """
    if typed is None or len(typed) == 0:
        return ()
    if prefix is None:
        prefix = common_prefix(() if names is None else names)
    out = []
    for one in typed:
        spec = resolve_control(one, prefix=prefix)
        if spec is not None:
            out.append(spec)
    return tuple(out)


def matches(spec: Optional[ControlSpec], guides, genes=None):
    """Return a Boolean mask for rows covered by a resolved control.

    Parameters
    ----------
    spec : ControlSpec or None
        Control to match. ``None`` matches no rows.
    guides : array-like
        Guide identifier for each row.
    genes : array-like, optional
        Gene identifier for each row. If omitted, gene membership is inferred
        from complete guide prefixes such as ``000000_1``.

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``guides``.
    """
    import pandas as pd

    series = pd.Series(guides).astype(str)
    if spec is None:
        return pd.Series(False, index=series.index)
    if not spec.is_gene:
        return series == spec.value
    if genes is not None:
        gene_series = pd.Series(genes).astype(str)
        gene_series.index = series.index
        return gene_series == spec.value
    # No gene column: a guide belongs to the gene its name starts with.
    return series.str.startswith(spec.value + SEPARATOR) | (series == spec.value)
