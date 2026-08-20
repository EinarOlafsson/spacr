"""Protein-localization metadata for compartment-aware figure colouring.

Interactive volcano plots can highlight genes from one selected LOPIT
compartment while leaving the remaining genes grey.

Figures highlight one selected compartment against grey to keep the comparison
legible. Coefficient feature names are resolved to bare gene identifiers with
:func:`spacr.hits.gene_of` before joining the bundled TAGM/LOPIT table.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

#: Values excluded from the compartment selector because they do not identify
#: a biological compartment. Numeric values are filtered separately.
NOT_A_COMPARTMENT = ("unknown", "unassigned", "nan", "")

#: Minimum number of screen genes required before a compartment is offered for
#: highlighting. This avoids presenting very small groups as robust patterns.
MIN_GENES = 5


#: Sentinel for coloring every compartment at once instead of highlighting one
#: compartment against grey. The non-printing prefix prevents collisions with
#: compartment names in the bundled TAGM/LOPIT table.
ALL = "\x00all-localisations"


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=1)
def table() -> Dict[str, str]:
    """``{gene number: compartment}`` from the bundled TAGM/LOPIT table.

    Cached: the file is 3,832 rows and is read to build a right-click menu,
    which happens on the GUI thread every time the user opens one.

    :returns: an empty dict when the table is missing, never raising -- a
        volcano is still a volcano without compartment colouring, and a
        screen of a different organism has no reason to carry this file.
    """
    try:
        import pandas as pd

        from .gene_tile import BUNDLED_LOCALISATION

        frame = pd.read_csv(BUNDLED_LOCALISATION)
    except Exception:                                          # noqa: BLE001
        return {}

    key = "gene_nr" if "gene_nr" in frame.columns else None
    value = next((c for c in ("tagm_location", "location")
                  if c in frame.columns), None)
    if key is None or value is None:
        return {}

    out: Dict[str, str] = {}
    for gene, place in zip(frame[key], frame[value]):
        gene_text = str(gene).strip()
        place_text = str(place).strip()
        if not gene_text or gene_text.lower() == "nan":
            continue
        if place_text.lower() in NOT_A_COMPARTMENT or _looks_numeric(place_text):
            continue
        # `gene_nr` reads as a float when the column has a blank in it, so
        # 244480 arrives as "244480.0" and joins to nothing at all.
        if gene_text.endswith(".0"):
            gene_text = gene_text[:-2]
        out.setdefault(gene_text, place_text)
    return out


def of(frame, *, key_column: str = "feature"):
    """The compartment of each row of ``frame``, as a Series aligned to it.

    :returns: a pandas Series of compartment names, empty string where the
        gene is not in the table.

    Joined through :func:`spacr.hits.gene_of`, which is the one place that
    knows how a design term names a gene. Both the per-gene and the per-guide
    rows resolve, because a guide's gene is where its protein lives too.
    """
    import pandas as pd

    if frame is None or key_column not in getattr(frame, "columns", ()):
        return pd.Series([], dtype="object")

    from .hits import gene_of

    lookup = table()
    genes = frame[key_column].map(lambda value: gene_of(str(value)) or "")
    return genes.map(lambda gene: lookup.get(str(gene).strip(), ""))


def present(frame, *, key_column: str = "feature",
            minimum: int = MIN_GENES) -> List[str]:
    """The compartments this screen actually has, commonest first.

    NOT ALL 27. A menu listing every compartment in the reference table
    offers the user 22 choices that would colour nothing, and a choice that
    colours nothing is indistinguishable from a broken one.
    """
    places = of(frame, key_column=key_column)
    if not len(places):
        return []
    counts = places[places != ""].value_counts()
    return [name for name, count in counts.items() if count >= minimum]


def mask(frame, compartment: Optional[str], *, key_column: str = "feature"):
    """A boolean Series: which rows are in ``compartment``.

    :param compartment: ``None`` or ``""`` selects nothing, which is the
        right answer for "no compartment chosen" and lets a caller pass the
        menu's state straight through without a branch.
    """
    import pandas as pd

    if not compartment:
        return pd.Series(False, index=getattr(frame, "index", None))
    places = of(frame, key_column=key_column)
    if not len(places):
        return pd.Series(False, index=getattr(frame, "index", None))
    return places == compartment


__all__ = ["MIN_GENES", "NOT_A_COMPARTMENT", "mask", "of", "present", "table"]
