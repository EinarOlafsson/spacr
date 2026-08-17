"""Where each gene's protein lives, for colouring one compartment at a time.

Asked for on 2026-08-16: "make a new version of your volcanoplot where the
data are colord by localization (LOPIT or at least make it an option when
right clicking on the graph)".

ONE COMPARTMENT AT A TIME, AGAINST GREY. The bundled TAGM/LOPIT table names
27 real compartments, and a 27-colour volcano is precisely what the house
style says not to draw -- "everything is grey except what the sentence is
about". It is not only a style rule: a 27-entry legend cost 40 ms of a 49 ms
redraw, so the version that breaks the rule is also the slow one.

A screen showing "rhoptries" against grey answers a question. A screen
showing all 27 at once answers none of them, because no reader can hold 27
hues apart and the two that matter are not adjacent in the legend.

THE JOIN IS ON THE GENE, PARSED FROM THE DESIGN TERM. A coefficient's
`feature` is `gene_fraction:gene[244480]` or `fraction:grna[411710_2]`, and
the LOPIT table is keyed on the bare gene number. :func:`spacr.hits.gene_of`
is what already knows how to get from one to the other, and using it means
this file cannot invent a second key space -- which is how a screen ends up
colouring the wrong dots and looking entirely plausible while it does.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

#: Compartment names in the bundled table that are not compartments. The CSV
#: has a handful of malformed rows whose location cell holds a number, and
#: "unknown" is the table's own way of saying it could not place the protein
#: -- offering either as something to colour by would be offering to colour
#: by "we do not know", which is not a claim anybody wants on a figure.
NOT_A_COMPARTMENT = ("unknown", "unassigned", "nan", "")

#: Below this many genes IN THE SCREEN, a compartment is not offered. Not a
#: style preference: colouring three dots out of twelve hundred produces a
#: figure whose sentence rests on three points, and the eye reads the
#: highlight as a finding rather than as three points.
MIN_GENES = 5


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
