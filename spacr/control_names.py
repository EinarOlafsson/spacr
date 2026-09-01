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
from typing import cast, Iterable, Optional, Sequence, Tuple

SEPARATOR = "_"

#: Minimum share of distinct identifiers that must carry a leading token for
#: it to be treated as an organism or strain prefix.
COMMON_PREFIX_SHARE = 0.9

GENE = "gene"
GUIDE = "grna"


@dataclass(frozen=True)
class ControlSpec:
    """Resolved interpretation of a user-entered control.

    :ivar typed: original nonblank identifier entered by the user.
    :ivar level: resolved level, :data:`GENE` or :data:`GUIDE`.
    :ivar value: normalized gene or guide identifier to match.
    """

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
        # THREE COMPONENTS OR IT IS NOT A SPECIES TAG. Caught by pointing
        # this at a one-guide library: "000000_1" made "000000" look like a
        # prefix carried by 100% of the names, so the GENE was read as the
        # organism and the control then matched nothing.
        #
        # spaCR's own convention is <org>_<gene>_<guide>, and `process_reads`
        # splits on exactly that -- three parts, no fewer. So the gene is
        # always the SECOND-TO-LAST component, and a two-part name's head is
        # a gene, never a species. Only three-part names can contribute a
        # candidate prefix.
        if len(name.split(SEPARATOR)) >= 3:
            leading[name.split(SEPARATOR)[0]] += 1
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

    :param typed: user-entered control identifiers; ``None`` means no controls.

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
        # THE PREFIX AGAIN. `resolve_control` strips the measured organism
        # prefix off what the user typed, so `TGGT1_000000_11` becomes the
        # guide `000000_11` -- and the count table spells that guide
        # `TGGT1_000000_11`. Comparing the stripped value against the
        # unstripped column matched nothing, for either spelling.
        head = f"{spec.prefix}{SEPARATOR}" if spec.prefix else ""
        return (series == spec.value) | (series == head + spec.value)
    if genes is not None:
        gene_series = pd.Series(genes).astype(str)
        gene_series.index = series.index
        # THE PREFIX HERE TOO, AND THIS BRANCH IS THE ONE THAT WAS MISSED.
        # The two branches on either side of it carry long comments about
        # `resolve_control` stripping the measured organism prefix off what
        # the user typed; both were fixed and this one, between them, was
        # not. A gene column that keeps its prefix -- `TGGT1_220950` against
        # a spec whose value is `220950` -- matched NOTHING.
        #
        # It is the branch a screen WITH a gene column takes, so this is not
        # an edge: `controls`, `positive_control` and `negative_control` all
        # arrive through `rows_for`, and a gene-level control on such a
        # screen selected zero rows in silence. Found on 2026-08-21 while
        # excluding a contaminant by gene name, which is the same code path.
        head = f"{spec.prefix}{SEPARATOR}" if spec.prefix else ""
        return (gene_series == spec.value)\
            | (gene_series == head + spec.value)
    # No gene column: a guide belongs to the gene its name CONTAINS as its
    # middle component.
    #
    # THE PREFIX IS PART OF THE NAME AND THIS IGNORED IT. `startswith` alone
    # asks whether the guide begins `000000_`, and the guides of a real count
    # table begin `TGGT1_000000_` -- the organism prefix that `resolve_control`
    # has already MEASURED and is carrying on the spec. So a gene control
    # matched nothing on every library whose names keep their prefix, which is
    # every count table `process_reads` writes.
    #
    # Measured on the example screen: `rows_for('000000', guides)` reported
    # "resolved to gene '000000': 0 guide(s)" against 30 guides that are
    # exactly that gene's. 184 recorded "all four spellings reach the same 28
    # guides" and that was measured WITH a gene column beside the guides; this
    # path -- the one a count table actually takes -- was never exercised on
    # prefixed names.
    #
    # AND ZERO IS NOT LOUD. A control that selects nothing leaves the
    # thresholds to fall back and the baseline at zero, and the run finishes.
    head = f"{spec.prefix}{SEPARATOR}" if spec.prefix else ""
    wanted = f"{spec.value}{SEPARATOR}"
    return (series.str.startswith(wanted)
            | series.str.startswith(head + wanted)
            | (series == spec.value)
            | (series == head + spec.value))


class ControlNotFound(ValueError):
    """Raised when a named control does not match any screen row.

    An empty control selection would invalidate normalization, reference
    baselines, and volcano annotations. :func:`rows_for` raises this exception
    when ``strict=True`` so callers can stop before computing those results.
    """


def rows_for(typed, guides, genes=None, *, names=None, prefix=None,
             strict: bool = False, label: str = "control"):
    """Resolve a typed control and select the matching screen rows.

    Guide controls use exact matches. Gene controls select every guide assigned
    to the gene. When the data omits an organism prefix that is present in the
    typed control, the prefix is removed before retrying the same whole-value
    match.

    Parameters
    ----------
    typed : object
        Control name or value accepted by :func:`resolve_control`.
    guides : array-like
        Guide names for the screen rows.
    genes : array-like, optional
        Gene names aligned with ``guides``. Guide prefixes are used when this
        column is unavailable.
    names : iterable of str, optional
        Reference names used to distinguish organism prefixes from gene names.
    prefix : str, optional
        Explicit organism prefix.
    strict : bool, default=False
        Raise :class:`ControlNotFound` when the control matches no rows.
    label : str, default="control"
        Name used in an error message when ``strict=True``.

    Returns
    -------
    tuple of pandas.Series and str
        Boolean row mask and a concise description of the resolved control.

    Raises
    ------
    ControlNotFound
        If ``strict=True`` and no row matches the resolved control.
    """
    import pandas as pd

    spec = resolve_control(typed, names=names, prefix=prefix)
    series = pd.Series(guides).astype(str)
    if spec is None:
        return pd.Series(False, index=series.index), ""
    mask = matches(spec, guides, genes)
    found = int(mask.sum())

    # THE DATA HAS ALREADY DROPPED THE PREFIX, and the user has not.
    # `process_reads` stores `TGGT1_000000_11` as guide `000000_11`, so the
    # names in hand carry no organism token at all and `common_prefix`
    # measures "" from them -- correctly. A control pasted from the LIBRARY
    # file still says `TGGT1_000000`, whose head is then not a known prefix
    # and which reads as a two-part guide that matches nothing.
    #
    # Measured on the maintainer's screen: 'TGGT1_000000' found 0 where
    # '000000' found 28, which is the same control written the way the
    # library writes it.
    #
    # So: when nothing matched and the first token is absent from every name
    # in hand, try again without it. Still WHOLE values -- this drops a
    # leading token, it does not go back to substring matching.
    if not found and SEPARATOR in spec.typed:
        head, _, tail = spec.typed.partition(SEPARATOR)
        library = {str(n) for n in (names or series)}
        unused = not any(str(n).startswith(head + SEPARATOR) for n in library)
        if tail and unused:
            shorter = cast(ControlSpec, resolve_control(tail, prefix=prefix))
            retry = matches(shorter, guides, genes)
            if int(retry.sum()):
                spec = ControlSpec(spec.typed, shorter.level,
                                   shorter.value, head)
                mask, found = retry, int(retry.sum())
    if not found and strict:
        raise ControlNotFound(
            f"{label} {spec.typed!r} was read as {'gene' if spec.is_gene else 'guide'} "
            f"{spec.value!r} and matches nothing in this screen. Every "
            f"normalisation and every baseline is computed against the "
            f"control rows, so leaving this would compute them against "
            f"nothing. Check the spelling against the count table"
            + (f", or drop the {spec.prefix!r} prefix" if spec.prefix else "")
            + ".")
    return mask, spec.note(found)
