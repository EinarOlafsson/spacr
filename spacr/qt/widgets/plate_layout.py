"""Lay out a plate before it is acquired, and say what is wrong with it.

The design decisions this module exists for are the ones that cannot be
undone after acquisition. Where the controls sit, whether a condition is
confounded with a row, whether the edge of the plate is used at all -- none of
those can be repaired by a better analysis, and all of them are cheap to fix
the day before. The current alternative is a spreadsheet somebody types twice:
once to tell the plate handler where things go, and again into
``treatment_plate_metadata`` when the measurements come back.

So this produces one artifact that both halves read. The IDs it writes are
:mod:`spacr.schema`'s own -- ``r3``, ``c7``, ``C07`` -- which are the ids
``schema.parse_field_stem`` recovers from an image file name, so the exported
table joins to a measurements table on ``(rowID, columnID)`` with no
translation step and nothing to get wrong.

Edge wells
----------
The warning this module was asked for. Evaporation and thermal gradients make
the outer ring of a plate behave differently from its interior, which is
exactly why spaCR grew illumination correction. A control that lives only on
the edge is not measuring what the interior wells are doing; it is measuring
the edge. :func:`check_design` says so before the plate is poured, when the
answer is to move four wells rather than to discount a whole run.

What cannot be exported
-----------------------
``spacr.utils.annotate_conditions`` maps conditions onto wells through a
vocabulary of whole rows and whole columns (``['r1']``, ``['c2', 'c3']``). A
randomised layout -- which is the statistically correct one, since it is the
only one that cannot be confounded with a position gradient -- has no
expression in that vocabulary. :func:`to_settings_fragment` says so rather
than emitting an approximation, and the long-form well table remains the
artifact that carries a randomised design.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ...schema import column_id, letters_from_row_index, row_id, well_id

__all__ = [
    "PLATE_FORMATS", "LAYOUTS", "ROLES",
    "ROLE_TREATMENT", "ROLE_POSITIVE", "ROLE_NEGATIVE", "ROLE_BLANK",
    "EDGE_USE", "EDGE_LEAVE_EMPTY",
    "Condition", "PlateDesign", "DesignFinding",
    "plate_shape", "is_edge", "assign_wells", "check_design",
    "to_settings_fragment", "write_design", "format_findings",
]


#: Plate formats, as ``{well count: (rows, columns)}``. Only formats whose
#: geometry is unambiguous; a "384-well plate" is always 16x24.
PLATE_FORMATS: Dict[int, Tuple[int, int]] = {
    6: (2, 3), 12: (3, 4), 24: (4, 6), 48: (6, 8),
    96: (8, 12), 384: (16, 24), 1536: (32, 48),
}

ROLE_TREATMENT = "treatment"
ROLE_POSITIVE = "positive_control"
ROLE_NEGATIVE = "negative_control"
ROLE_BLANK = "blank"

#: Every role a condition may take. ``blank`` is a deliberately empty well --
#: a media-only or no-cell control -- and is laid out like any other so it
#: gets the same protection from position effects.
ROLES: Tuple[str, ...] = (
    ROLE_TREATMENT, ROLE_POSITIVE, ROLE_NEGATIVE, ROLE_BLANK,
)

#: Fill order. ``random`` is the only one that cannot be confounded with a
#: position gradient, and is the right default for anything being compared.
#: ``block`` keeps a condition's replicates together, which is easier to
#: pipette and is the one layout guaranteed to confound condition with
#: position -- it is here because people use it, and because a design tool
#: that refuses to express the common case gets replaced by a spreadsheet.
LAYOUTS: Tuple[str, ...] = ("random", "row", "column", "block")

#: Use the outer ring like any other well.
EDGE_USE = "use"
#: Leave the outer ring empty. Costs 36 of 96 wells and 76 of 384, which is
#: why it is not the default, but it removes the edge effect rather than
#: correcting for it.
EDGE_LEAVE_EMPTY = "leave_empty"


@dataclass(frozen=True)
class Condition:
    """One thing being put on the plate.

    :ivar name: label written into the exported table. Becomes the value of
        ``treatment`` (or ``host_cells``/``pathogen``) downstream.
    :ivar replicates: how many wells it gets.
    :ivar role: one of :data:`ROLES`.
    """

    name: str
    replicates: int = 3
    role: str = ROLE_TREATMENT

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("a condition needs a name")
        if int(self.replicates) < 1:
            raise ValueError(
                f"{self.name!r} asks for {self.replicates} replicate(s); a "
                "condition that gets no well is not in the experiment.")
        if self.role not in ROLES:
            raise ValueError(
                f"{self.name!r} has role {self.role!r}; choose one of {ROLES}.")


@dataclass(frozen=True)
class PlateDesign:
    """A plate map before the plate exists.

    :ivar plate_id: the plate's name. Written into ``plateID`` and must match
        the one the image file names will carry, or the exported table will
        not join to the measurements.
    :ivar plate_format: well count; a key of :data:`PLATE_FORMATS`.
    :ivar conditions: what goes on it.
    :ivar layout: one of :data:`LAYOUTS`.
    :ivar edge_policy: :data:`EDGE_USE` or :data:`EDGE_LEAVE_EMPTY`.
    :ivar seed: makes ``random`` reproducible. A layout nobody can regenerate
        is a layout nobody can check, and the plate map is the one record
        that has to survive the person who made it.
    """

    plate_id: str = "plate1"
    plate_format: int = 96
    conditions: Tuple[Condition, ...] = ()
    layout: str = "random"
    edge_policy: str = EDGE_USE
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.plate_format) not in PLATE_FORMATS:
            raise ValueError(
                f"plate_format={self.plate_format!r} is not a known plate; "
                f"choose one of {sorted(PLATE_FORMATS)}.")
        if self.layout not in LAYOUTS:
            raise ValueError(
                f"layout={self.layout!r}; choose one of {LAYOUTS}.")
        if self.edge_policy not in (EDGE_USE, EDGE_LEAVE_EMPTY):
            raise ValueError(
                f"edge_policy={self.edge_policy!r}; choose "
                f"{EDGE_USE!r} or {EDGE_LEAVE_EMPTY!r}.")

    @property
    def shape(self) -> Tuple[int, int]:
        """``(n_rows, n_columns)``."""
        return PLATE_FORMATS[int(self.plate_format)]

    @property
    def wells_requested(self) -> int:
        """Total wells the conditions ask for."""
        return sum(int(c.replicates) for c in self.conditions)

    @property
    def wells_available(self) -> int:
        """Wells the edge policy leaves usable."""
        rows, columns = self.shape
        if self.edge_policy == EDGE_LEAVE_EMPTY:
            return max(0, (rows - 2)) * max(0, (columns - 2))
        return rows * columns


@dataclass(frozen=True)
class DesignFinding:
    """One thing worth knowing before the plate is poured.

    :ivar key: stable identifier, for tests and for the exported record.
    :ivar severity: ``"error"`` (the design cannot be laid out), ``"warn"``
        (it can, and it will cost you), or ``"note"``.
    :ivar message: one sentence, naming the wells involved where that helps.
    """

    key: str
    severity: str
    message: str


def plate_shape(plate_format: int) -> Tuple[int, int]:
    """``(n_rows, n_columns)`` for a well count."""
    try:
        return PLATE_FORMATS[int(plate_format)]
    except KeyError:
        raise ValueError(
            f"{plate_format!r} is not a known plate format; choose one of "
            f"{sorted(PLATE_FORMATS)}.") from None


def is_edge(row: int, column: int, n_rows: int, n_columns: int) -> bool:
    """Whether a 1-based ``(row, column)`` is in the plate's outer ring."""
    return (row == 1 or row == n_rows
            or column == 1 or column == n_columns)


def _well_order(design: PlateDesign) -> List[Tuple[int, int]]:
    """Every usable well, in the order the layout fills them."""
    rows, columns = design.shape
    wells = [(r, c) for r in range(1, rows + 1)
             for c in range(1, columns + 1)]
    if design.edge_policy == EDGE_LEAVE_EMPTY:
        wells = [w for w in wells if not is_edge(w[0], w[1], rows, columns)]
    if design.layout == "column":
        wells.sort(key=lambda w: (w[1], w[0]))
    elif design.layout == "random":
        # A dedicated Random rather than numpy: this is a shuffle of at most
        # 1536 tuples, and seeding a local instance cannot disturb any other
        # stream in the process.
        random.Random(int(design.seed)).shuffle(wells)
    return wells


def assign_wells(design: PlateDesign) -> pd.DataFrame:
    """Place every condition's replicates on the plate.

    :param design: the design.
    :returns: one row per **assigned** well, with columns ``plateID``,
        ``well``, ``rowID``, ``columnID``, ``row_index``, ``column_index``,
        ``condition``, ``role``, ``replicate`` and ``is_edge``. The ids are
        :mod:`spacr.schema`'s, so the frame joins to a measurements table on
        ``(plateID, rowID, columnID)``.
    :raises ValueError: when the conditions need more wells than the plate
        and the edge policy leave usable. Refused rather than truncated: a
        silently dropped replicate is a plate that does not match its own map.
    """
    if not design.conditions:
        return pd.DataFrame(columns=[
            "plateID", "well", "rowID", "columnID", "row_index",
            "column_index", "condition", "role", "replicate", "is_edge"])
    order = _well_order(design)
    if design.wells_requested > len(order):
        raise ValueError(
            f"{design.wells_requested} well(s) requested but only "
            f"{len(order)} usable on a {design.plate_format}-well plate"
            + (" with the edge left empty"
               if design.edge_policy == EDGE_LEAVE_EMPTY else "")
            + ". Reduce replicates, use a bigger plate, or split across "
              "plates.")

    rows, columns = design.shape
    records: List[Dict[str, Any]] = []
    cursor = 0
    for condition in design.conditions:
        for replicate in range(1, int(condition.replicates) + 1):
            row, column = order[cursor]
            cursor += 1
            records.append({
                "plateID": str(design.plate_id),
                "well": well_id(row, column),
                "rowID": row_id(row),
                "columnID": column_id(column),
                "row_index": row,
                "column_index": column,
                "condition": condition.name,
                "role": condition.role,
                "replicate": replicate,
                "is_edge": is_edge(row, column, rows, columns),
            })
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(["row_index", "column_index"]).reset_index(
        drop=True)


def _rows_used(block: pd.DataFrame) -> set:
    return set(block["row_index"].tolist())


def _columns_used(block: pd.DataFrame) -> set:
    return set(block["column_index"].tolist())


def check_design(design: PlateDesign,
                 table: Optional[pd.DataFrame] = None) -> List[DesignFinding]:
    """Everything worth saying about a design before it is acquired.

    Ordered worst first. An empty list means nothing was found, which is not
    the same as the design being good -- there is no check here for whether
    the biology makes sense.

    :param design: the design.
    :param table: its assignment, from :func:`assign_wells`. Computed if
        omitted; pass it in when it has already been built.
    :returns: findings, ``"error"`` before ``"warn"`` before ``"note"``.
    """
    findings: List[DesignFinding] = []
    if not design.conditions:
        return [DesignFinding("no_conditions", "error",
                              "The plate has no conditions on it yet.")]
    if table is None:
        try:
            table = assign_wells(design)
        except ValueError as exc:
            return [DesignFinding("does_not_fit", "error", str(exc))]

    rows, columns = design.shape
    controls = table.loc[table["role"].isin((ROLE_POSITIVE, ROLE_NEGATIVE))]
    roles_present = set(table["role"])

    # -- the named requirement ------------------------------------------
    if len(controls) and bool(controls["is_edge"].all()):
        wells = ", ".join(sorted(controls["well"])[:8])
        findings.append(DesignFinding(
            "controls_all_on_edge", "warn",
            f"Every control well sits on the plate edge ({wells}"
            + (", ..." if len(controls) > 8 else "")
            + "). Edge wells evaporate faster and run at a different "
              "temperature than the interior -- that is why spaCR has "
              "illumination correction -- so these controls describe the edge "
              "rather than the wells they are meant to normalise. Move at "
              "least some of them inward."))
    elif len(controls) and design.edge_policy == EDGE_USE:
        edge_fraction = float(controls["is_edge"].mean())
        interior_share = 1.0 - (2.0 * (rows + columns) - 4.0) / (rows * columns)
        if edge_fraction > 0.5 and interior_share > 0.25:
            findings.append(DesignFinding(
                "controls_mostly_on_edge", "warn",
                f"{edge_fraction:.0%} of the control wells are on the plate "
                "edge, where evaporation and temperature differ from the "
                "interior. Spread them inward."))

    # -- confounding with position --------------------------------------
    for role, label in ((ROLE_POSITIVE, "positive control"),
                        (ROLE_NEGATIVE, "negative control")):
        block = table.loc[table["role"] == role]
        if len(block) < 2:
            continue
        if len(_rows_used(block)) == 1 and rows > 1:
            findings.append(DesignFinding(
                f"{role}_in_one_row", "warn",
                f"Every {label} well is in row "
                f"{letters_from_row_index(int(block['row_index'].iloc[0]))}. "
                "Any row-wise gradient -- a thermal one, a dispense-order one "
                "-- is now indistinguishable from the control itself."))
        if len(_columns_used(block)) == 1 and columns > 1:
            findings.append(DesignFinding(
                f"{role}_in_one_column", "warn",
                f"Every {label} well is in column "
                f"{int(block['column_index'].iloc[0])}. Any column-wise "
                "gradient is now indistinguishable from the control."))

    if design.layout == "block" and len(design.conditions) > 1:
        findings.append(DesignFinding(
            "block_layout_confounds_position", "warn",
            "A block layout puts each condition's replicates next to each "
            "other, so condition and plate position vary together and no "
            "analysis can separate them. Use the random layout unless the "
            "pipetting cost is genuinely prohibitive."))

    # -- the design itself ----------------------------------------------
    if ROLE_NEGATIVE not in roles_present:
        findings.append(DesignFinding(
            "no_negative_control", "warn",
            "There is no negative control. Batch correction, hit calling and "
            "the QC baseline all need one, and it cannot be added later."))
    if ROLE_POSITIVE not in roles_present:
        findings.append(DesignFinding(
            "no_positive_control", "note",
            "There is no positive control, so a run that produces nothing "
            "cannot be told apart from an assay that did not work."))

    thin = [c.name for c in design.conditions if int(c.replicates) < 2]
    if thin:
        findings.append(DesignFinding(
            "single_replicate", "warn",
            f"{', '.join(thin[:6])}"
            + (", ..." if len(thin) > 6 else "")
            + " has one well. A single well has no within-condition variance, "
              "so nothing downstream can estimate its uncertainty."))

    for condition in design.conditions:
        block = table.loc[table["condition"] == condition.name]
        if (len(block) > 1 and bool(block["is_edge"].all())
                and condition.role == ROLE_TREATMENT):
            findings.append(DesignFinding(
                "condition_all_on_edge", "note",
                f"Every well of {condition.name!r} is on the plate edge, so "
                "its measurements carry the edge effect in full."))

    spare = design.wells_available - design.wells_requested
    if spare > 0:
        findings.append(DesignFinding(
            "spare_wells", "note",
            f"{spare} usable well(s) are unassigned. Replicates are the "
            "cheapest power there is -- spending them is almost always "
            "better than leaving them empty."))

    if design.edge_policy == EDGE_LEAVE_EMPTY:
        findings.append(DesignFinding(
            "edge_left_empty", "note",
            f"The outer ring is unused: {rows * columns - design.wells_available}"
            f" of {rows * columns} wells. That removes the edge effect rather "
            "than correcting for it, at the cost of the wells."))

    order = {"error": 0, "warn": 1, "note": 2}
    return sorted(findings, key=lambda f: order[f.severity])


def to_settings_fragment(design: PlateDesign,
                         table: Optional[pd.DataFrame] = None,
                         *, key: str = "treatment") -> Dict[str, Any]:
    """The design as the settings keys the analysis modules already read.

    ``spacr.utils.annotate_conditions`` maps a condition onto wells through
    row and column ids, so this can only be produced when each condition
    occupies whole rows or whole columns. A random layout cannot be written
    that way, and this returns ``expressible=False`` with the reason rather
    than an approximation that would mislabel wells.

    :param design: the design.
    :param table: its assignment; computed if omitted.
    :param key: which settings family to fill -- ``"treatment"``,
        ``"cell"`` or ``"pathogen"``.
    :returns: ``{"expressible": bool, "reason": str, "settings": {...}}``.
        ``settings`` holds ``<key>s`` and ``<key>_plate_metadata`` plus
        ``positive_control``/``negative_control`` when those roles are used.
    """
    if table is None:
        table = assign_wells(design)
    plural = {"treatment": "treatments", "cell": "cell_types",
              "pathogen": "pathogen_types"}.get(key, f"{key}s")
    meta_key = f"{key}_plate_metadata"
    out: Dict[str, Any] = {"expressible": True, "reason": "",
                           "settings": {}}
    if table.empty:
        out["expressible"] = False
        out["reason"] = "The design has no wells assigned."
        return out

    names: List[str] = []
    locations: List[List[str]] = []
    for condition in design.conditions:
        block = table.loc[table["condition"] == condition.name]
        rows_used = sorted(_rows_used(block))
        columns_used = sorted(_columns_used(block))
        whole_rows = all(
            len(table.loc[table["row_index"] == r]["condition"].unique()) == 1
            for r in rows_used)
        whole_columns = all(
            len(table.loc[table["column_index"] == c]["condition"].unique()) == 1
            for c in columns_used)
        if whole_rows:
            locations.append([row_id(r) for r in rows_used])
        elif whole_columns:
            locations.append([column_id(c) for c in columns_used])
        else:
            out["expressible"] = False
            out["reason"] = (
                f"{condition.name!r} does not occupy whole rows or whole "
                "columns, and annotate_conditions can only address wells by "
                "row or column id. Use the exported well table "
                "(plate_map.csv) and join on (plateID, rowID, columnID) "
                "instead -- it carries the layout exactly.")
            return out
        names.append(condition.name)

    out["settings"][plural] = names
    out["settings"][meta_key] = locations
    for role, setting in ((ROLE_POSITIVE, "positive_control"),
                          (ROLE_NEGATIVE, "negative_control")):
        matching = [c.name for c in design.conditions if c.role == role]
        if matching:
            out["settings"][setting] = matching[0]
    return out


def format_findings(findings: Sequence[DesignFinding]) -> str:
    """Findings as plain text, worst first."""
    if not findings:
        return "No problems found in the layout."
    marks = {"error": "STOP", "warn": "!", "note": "-"}
    return "\n".join(
        f"{marks[f.severity]} {f.message}" for f in findings)


def write_design(design: PlateDesign, folder: Any, *,
                 table: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
    """Write the design where the pipeline and the plate handler can read it.

    Three files, because they have three readers:

    * ``plate_map.csv`` -- one row per well, keyed by ``(plateID, rowID,
      columnID)``. The artifact: it joins straight onto a measurements table
      and it survives a randomised layout, which the settings fragment does
      not.
    * ``plate_map.json`` -- the design itself plus its findings, so the
      layout can be regenerated and so the warnings that were shown are on
      the record rather than only on somebody's screen.
    * ``plate_map_settings.json`` -- the ``treatments`` /
      ``treatment_plate_metadata`` pair to paste into an analysis settings
      file, when the layout can be expressed that way.

    :param design: the design.
    :param folder: destination directory; created if absent.
    :param table: its assignment; computed if omitted.
    :returns: ``{name: path}`` for every file written.
    """
    if table is None:
        table = assign_wells(design)
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)

    findings = check_design(design, table)
    fragment = to_settings_fragment(design, table)

    paths: Dict[str, Path] = {}
    paths["plate_map"] = destination / "plate_map.csv"
    table.to_csv(paths["plate_map"], index=False)

    record = {
        "plate_id": design.plate_id,
        "plate_format": int(design.plate_format),
        "layout": design.layout,
        "edge_policy": design.edge_policy,
        "seed": int(design.seed),
        "conditions": [
            {"name": c.name, "replicates": int(c.replicates), "role": c.role}
            for c in design.conditions
        ],
        "wells_assigned": int(len(table)),
        "wells_available": int(design.wells_available),
        "findings": [
            {"key": f.key, "severity": f.severity, "message": f.message}
            for f in findings
        ],
    }
    paths["design"] = destination / "plate_map.json"
    paths["design"].write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    paths["settings"] = destination / "plate_map_settings.json"
    paths["settings"].write_text(
        json.dumps(fragment, indent=2, sort_keys=True), encoding="utf-8")
    return paths
