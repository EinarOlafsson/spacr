"""Read a control the way the user wrote it: as a gene, or as a guide.

Instruction 184, asked for on 2026-08-20 -- "when adding controlls the user
can add them as grna names (000000_1, 000000_2, ...), but they should also be
able to add the grnas as a 'gene', which would be 000000 instead".

WHY IT NEEDS A MODULE. A control that does not match is not an error anyone
sees. It is silently zero controls, and every normalisation, every volcano
baseline and every nc/pc reference is then computed against nothing. The
defaults spaCR ships are bare gene ids (``nc='233460'``), and the names a
user pastes out of their library file are full guide names
(``TGGT1_233460_4``). Both have to work, and so does everything between.

THE RULE, in the maintainer's words and implemented exactly:

    TWO OR MORE underscores   everything after the FIRST underscore is the
                              GUIDE.        TGGT1_000000_1 -> guide 000000_1
    ONE underscore, and the
    part before it is COMMON  the part after it is the GENE.
                              TGGT1_000000   -> gene 000000
    ONE underscore, not
    common                    the whole string is the GUIDE.
                              000000_1       -> guide 000000_1
    NO underscore             the whole string is the GENE.
                              000000         -> gene 000000

IT IS THE SAME RULE SPACR ALREADY APPLIES TO THE DATA. `process_reads` splits
a three-component guide name into org/gene/guide and stores the last two, so
`TGGT1_000000_11` is held as gene ``000000`` and guide ``000000_11``. This
module makes the TYPED control agree with that, which it previously did only
by the accident of substring matching -- and substring matching is why
``nc='23346'`` would have quietly claimed ``233460``.

"COMMON" IS MEASURED, NEVER ASSUMED. It means a leading token shared by
(nearly) every identifier in THIS screen: a species or strain tag that
carries no information. On the maintainer's screen it is ``TGGT1``. The next
screen will use a different organism, so a hard-coded list would be wrong the
first time it was reused.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

SEPARATOR = "_"

#: How much of the library a leading token must cover before it counts as a
#: species tag rather than a gene. Proposed in the instruction and kept:
#: a prefix on 100% of guides is plainly common, and 90% leaves room for a
#: handful of oddly-named controls without admitting a second organism.
COMMON_PREFIX_SHARE = 0.9

GENE = "gene"
GUIDE = "grna"


@dataclass(frozen=True)
class ControlSpec:
    """What a typed control turned out to mean."""

    typed: str
    level: str
    value: str
    prefix: str = ""

    @property
    def is_gene(self) -> bool:
        return self.level == GENE

    def note(self, matched_guides: int = -1, matched_wells: int = -1) -> str:
        """One line saying what it matched, for the console."""
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
    """The leading token nearly every name carries, or ``''``.

    :param names: the guide identifiers present in the loaded data.
    :param share: the fraction of DISTINCT names that must carry the token.
    :returns: the token without its separator, or ``''`` when no token
        reaches ``share`` -- which is the honest answer for a library whose
        names are already bare, and for one holding two organisms.

    Counted over DISTINCT names, not rows. A screen where one guide has a
    million reads and the rest have ten would otherwise let that guide's
    prefix speak for the library.
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

    :param typed: what the user wrote. ``None`` or blank returns ``None`` --
        "no control" is a legal answer and must not become a control named
        ``''`` that matches every row.
    :param names: the guide identifiers present, used to measure the common
        prefix. Ignored when ``prefix`` is given.
    :param prefix: the common prefix, when it has already been measured.
    :returns: a :class:`ControlSpec`, or ``None`` for no control.
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
    """Resolve a LIST of controls, each on its own.

    "MIXED FORMS IN ONE LIST are allowed" -- a user may give a gene and two
    guides together, so the prefix is measured once and each entry is read
    against it separately.
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
    """A boolean mask over ``guides`` for the rows this control covers.

    :param guides: the guide identifier per row.
    :param genes: the gene identifier per row. When absent, a gene-level
        control falls back to the guide's own prefix -- ``000000_1`` belongs
        to gene ``000000`` -- so a frame that never got a gene column still
        resolves rather than matching nothing.

    WHOLE VALUES, NEVER SUBSTRINGS. The old path tested ``nc in feature``,
    which makes ``23346`` claim ``233460`` and ``2334600`` alike. A control
    that silently over-matches is worse than one that misses, because the
    rows it steals are reported as controls.
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
