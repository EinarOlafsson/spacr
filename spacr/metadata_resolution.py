"""Recover missing spaCR metadata columns without silent identity guesses.

The resolver is UI-independent.  A Qt dialog can be supplied as ``prompt``;
headless callers receive one actionable exception instead of blocking on a
window no one is watching.  Known aliases are canonicalised before any human
choice is requested.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import schema

IDENTITY_COLUMNS: Tuple[str, ...] = (
    schema.PLATE_KEY,
    schema.ROW_KEY,
    schema.COLUMN_KEY,
    schema.FIELD_KEY,
    "wellID",
    schema.OBJECT_LABEL_KEY,
)


class MetadataResolutionRequired(ValueError):
    """A headless caller must provide an explicit metadata mapping.

    :param missing: canonical metadata columns that could not be resolved.
    :param available: source columns that were available for explicit mapping.
    """

    def __init__(self, missing: Sequence[str], available: Sequence[str]):
        """Retain the unresolved and available columns for programmatic repair."""
        self.missing = tuple(missing)
        self.available = tuple(available)
        super().__init__(
            "Missing metadata column(s) "
            f"{list(self.missing)}. Columns found: {list(self.available)}. "
            "Supply metadata_column_map={canonical_name: source_column}; "
            "use metadata_well_column to derive rowID/columnID from wells; "
            "or set metadata_pseudo_source with allow_pseudo_metadata=True "
            "to create distinct audited pseudo wells. The headless path "
            "never opens a dialog.")


@dataclass(frozen=True)
class MetadataRequest:
    """All unresolved targets and evidence shown to one prompt.

    :param missing: canonical columns that still require a decision.
    :param available: source columns available for mapping.
    :param examples: representative string values for each source column.
    :param guesses: best-effort source-column suggestion for each target.
    """

    missing: Tuple[str, ...]
    available: Tuple[str, ...]
    examples: Mapping[str, Tuple[str, ...]]
    guesses: Mapping[str, str]


@dataclass(frozen=True)
class MetadataDecision:
    """One all-at-once answer returned by a UI or a saved setting.

    :param column_map: mapping from each canonical target name to an existing
        source column, applied collision-safely before identity derivation.
    :param well_column: optional source column whose values are parsed strictly
        as well labels to supply missing ``rowID`` and ``columnID`` values.
    :param pseudo_source: optional source column whose distinct typed values
        receive pseudo row and column coordinates when real well metadata is
        unavailable.
    :param allow_pseudo: explicit permission to synthesize coordinates from
        ``pseudo_source``; the source alone never changes the frame.
    :param save_path: optional JSON audit destination written after successful
        resolution with the decision and generated pseudo-well assignments.
    :param remember: whether a prompted decision is cached under a supplied
        ``cache_key`` for the remainder of the current process.
    """

    column_map: Mapping[str, str] = field(default_factory=dict)
    well_column: Optional[str] = None
    pseudo_source: Optional[str] = None
    allow_pseudo: bool = False
    save_path: Optional[str] = None
    remember: bool = True


@dataclass(frozen=True)
class ResolutionResult:
    """Resolved frame plus the auditable decisions that changed it.

    :param frame: normalized frame containing every required metadata column.
    :param column_map: explicit canonical-target to source-column mappings used.
    :param derived_from_well: well column used to derive row and column IDs, if
        derivation was necessary.
    :param pseudo_map: audited source identities and their generated pseudo
        well coordinates.
    """

    frame: pd.DataFrame
    column_map: Mapping[str, str]
    derived_from_well: Optional[str]
    pseudo_map: Tuple[Mapping[str, Any], ...]


_RUN_DECISIONS: Dict[str, MetadataDecision] = {}


def clear_run_metadata_decisions() -> None:
    """Clear remembered prompt answers (primarily for tests/new runs)."""
    _RUN_DECISIONS.clear()


def _examples(frame: pd.DataFrame, limit: int = 3) -> Dict[str, Tuple[str, ...]]:
    """Return up to ``limit`` distinct non-null string examples per column."""
    out: Dict[str, Tuple[str, ...]] = {}
    for column in frame.columns:
        values = frame[column].dropna().drop_duplicates().head(limit)
        out[str(column)] = tuple(str(value) for value in values)
    return out


def build_metadata_request(frame: pd.DataFrame,
                           required: Iterable[str]) -> MetadataRequest:
    """Build the single request a GUI displays for every missing column.

    :param frame: metadata frame whose current columns and examples are shown.
    :param required: canonical columns that the calling workflow requires.
    """
    missing = tuple(column for column in required if column not in frame.columns)
    available = tuple(str(column) for column in frame.columns)

    def comparable(value: str) -> str:
        """Reduce a metadata name to case-insensitive alphanumeric text.

        :param value: canonical or source column name to compare.
        :returns: case-folded name with punctuation and whitespace removed.
        """
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    guesses: Dict[str, str] = {}
    for target in missing:
        target_key = comparable(target)
        target_root = re.sub(r"(?:id|label)$", "", target_key)

        def guess_score(source: str, target_key: str = target_key,
                        target_root: str = target_root) -> float:
            """Score one available column against the current missing target.

            :param source: available source-column name to rank.
            :param target_key: normalized current target, bound when the scorer
                is created so the surrounding loop cannot change it.
            :param target_root: target without an ``id`` or ``label`` suffix,
                likewise bound for this scorer.
            :returns: sequence similarity plus a root-name bonus when the
                target without an ``id`` or ``label`` suffix occurs in the
                normalized source name.
            """
            source_key = comparable(source)
            similarity = SequenceMatcher(None, target_key, source_key).ratio()
            if target_root and target_root in source_key:
                similarity += 0.5
            return similarity

        ranked = sorted(
            ((guess_score(source), source)
             for source in available),
            reverse=True,
        )
        if ranked and ranked[0][0] >= 0.45:
            guesses[target] = ranked[0][1]
    return MetadataRequest(
        missing=missing,
        available=available,
        examples=_examples(frame),
        guesses=guesses,
    )


def _apply_column_map(frame: pd.DataFrame,
                      column_map: Mapping[str, str]) -> pd.DataFrame:
    """Rename arbitrary user columns through schema's collision-safe plan."""
    requested: Dict[str, str] = {}
    for target, source in column_map.items():
        target = str(target)
        source = str(source)
        if source not in frame.columns:
            raise MetadataResolutionRequired([target], frame.columns)
        if target in frame.columns and target != source:
            raise ValueError(
                f"Cannot map {source!r} to {target!r}: the canonical target "
                "already exists, which would create a column collision. "
                "Keep both columns and choose which is "
                "authoritative explicitly.")
        requested[source] = target
    plan = schema.canonical_rename_plan(frame.columns, requested=requested)
    unapplied = {
        source: target for source, target in requested.items()
        if source != target and plan.get(source) != target
    }
    if unapplied:
        raise ValueError(
            "Metadata mapping would create a case-insensitive column "
            f"collision: {unapplied}.")
    return frame.rename(columns=plan) if plan else frame.copy()


def _derive_well_columns(frame: pd.DataFrame, well_column: str,
                         needed: Sequence[str]) -> Tuple[pd.DataFrame, bool]:
    """Strictly derive needed row and column ids from one complete well column."""
    if well_column not in frame.columns:
        return frame, False
    parsed = []
    for value in frame[well_column]:
        try:
            parsed.append(schema.parse_well(value, strict=True))
        except schema.WellParseError:
            return frame, False
    out = frame.copy()
    rows = pd.Series((item[0] for item in parsed), index=out.index)
    columns = pd.Series((item[1] for item in parsed), index=out.index)
    for target, values in ((schema.ROW_KEY, rows),
                           (schema.COLUMN_KEY, columns)):
        if target not in needed:
            continue
        if target in out.columns:
            inconsistent = out[target].astype(str) != values.astype(str)
            if inconsistent.any():
                raise ValueError(
                    f"Existing {target} disagrees with {well_column} at row "
                    f"{out.index[inconsistent][0]!r}; refusing a silent "
                    "plate remap.")
        else:
            out[target] = values
    return out, True


def _resolved_source_column(
        frame: pd.DataFrame,
        requested: str,
        column_map: Mapping[str, str]) -> str:
    """Return the post-normalisation name of a selected source column."""
    candidates = [str(requested), schema.canonical_column_name(requested)]
    candidates.extend(
        str(target)
        for target, source in column_map.items()
        if str(source) == str(requested)
    )
    return next((name for name in candidates if name in frame.columns),
                str(requested))


def _typed_value(value: Any) -> Tuple[str, str]:
    """A JSON-safe identity that keeps ``1`` distinct from ``'1'``."""
    return type(value).__name__, repr(value)


def _pseudo_wells(frame: pd.DataFrame, source: str,
                  needed: Sequence[str]) -> Tuple[pd.DataFrame, Tuple[Mapping[str, Any], ...]]:
    """Assign first-seen source identities to pseudo wells and return an audit."""
    if source not in frame.columns:
        raise MetadataResolutionRequired(needed, frame.columns)
    identities = [_typed_value(value) for value in frame[source]]
    unique = list(dict.fromkeys(identities))
    assignments = {
        identity: (f"r{index // 48 + 1}", f"c{index % 48 + 1}")
        for index, identity in enumerate(unique)
    }
    if len(set(assignments.values())) != len(assignments):
        raise RuntimeError("pseudo-well mapping is not injective")
    out = frame.copy()
    rows = [assignments[identity][0] for identity in identities]
    columns = [assignments[identity][1] for identity in identities]
    if schema.ROW_KEY in needed and schema.ROW_KEY not in out.columns:
        out[schema.ROW_KEY] = rows
    if schema.COLUMN_KEY in needed and schema.COLUMN_KEY not in out.columns:
        out[schema.COLUMN_KEY] = columns
    audit = tuple({
        "source_column": source,
        "source_type": identity[0],
        "source_value": identity[1],
        schema.ROW_KEY: assignments[identity][0],
        schema.COLUMN_KEY: assignments[identity][1],
    } for identity in unique)
    return out, audit


def _save_audit(path: str, decision: MetadataDecision,
                pseudo_map: Sequence[Mapping[str, Any]]) -> None:
    """Write the applied decision and pseudo-well assignments as stable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "column_map": dict(decision.column_map),
        "well_column": decision.well_column,
        "pseudo_source": decision.pseudo_source,
        "allow_pseudo": bool(decision.allow_pseudo),
        "remember": bool(decision.remember),
        "pseudo_map": list(pseudo_map),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


def resolve_metadata_columns(
        frame: pd.DataFrame,
        required: Iterable[str], *,
        column_map: Optional[Mapping[str, str]] = None,
        well_column: Optional[str] = None,
        pseudo_source: Optional[str] = None,
        allow_pseudo: bool = False,
        prompt: Optional[Callable[[MetadataRequest], MetadataDecision]] = None,
        cache_key: Optional[str] = None,
        save_path: Optional[str] = None) -> ResolutionResult:
    """Resolve every required metadata column in one deterministic pass.

    :param frame: source metadata frame to normalize and resolve.
    :param required: canonical columns that must be present in the result.

    ``column_map`` is ``{canonical_target: actual_source}``.  A prompt, when
    supplied, is called at most once and receives all unresolved columns.
    Without one, missing columns raise :class:`MetadataResolutionRequired`
    with the non-interactive settings needed to proceed.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"metadata resolver needs a DataFrame, got {type(frame).__name__}")
    required = tuple(dict.fromkeys(str(column) for column in required))
    out = schema.canonicalise_columns(frame)

    decision = _RUN_DECISIONS.get(str(cache_key)) if cache_key is not None else None
    if decision is None:
        decision = MetadataDecision(
            column_map=dict(column_map or {}),
            well_column=well_column,
            pseudo_source=pseudo_source,
            allow_pseudo=allow_pseudo,
            save_path=save_path,
        )

    out = _apply_column_map(out, decision.column_map)
    derived_from = None
    pseudo_map: Tuple[Mapping[str, Any], ...] = ()

    missing = [column for column in required if column not in out.columns]
    identity_missing = [column for column in missing
                        if column in (schema.ROW_KEY, schema.COLUMN_KEY)]
    if identity_missing and decision.well_column:
        resolved_well_column = _resolved_source_column(
            out, decision.well_column, decision.column_map)
        out, derived = _derive_well_columns(
            out, resolved_well_column, identity_missing)
        if derived:
            derived_from = decision.well_column

    missing = [column for column in required if column not in out.columns]
    identity_missing = [column for column in missing
                        if column in (schema.ROW_KEY, schema.COLUMN_KEY)]
    if identity_missing and decision.allow_pseudo and decision.pseudo_source:
        resolved_pseudo_source = _resolved_source_column(
            out, decision.pseudo_source, decision.column_map)
        out, pseudo_map = _pseudo_wells(
            out, resolved_pseudo_source, identity_missing)

    missing = [column for column in required if column not in out.columns]
    if missing and prompt is not None and cache_key not in _RUN_DECISIONS:
        prompted = prompt(build_metadata_request(out, required))
        if not isinstance(prompted, MetadataDecision):
            raise TypeError("metadata prompt must return MetadataDecision")
        if cache_key is not None and prompted.remember:
            _RUN_DECISIONS[str(cache_key)] = prompted
        return resolve_metadata_columns(
            out, required,
            column_map=prompted.column_map,
            well_column=prompted.well_column,
            pseudo_source=prompted.pseudo_source,
            allow_pseudo=prompted.allow_pseudo,
            cache_key=cache_key,
            save_path=prompted.save_path,
        )
    if missing:
        raise MetadataResolutionRequired(missing, out.columns)

    effective_path = decision.save_path or save_path
    if effective_path:
        _save_audit(effective_path, decision, pseudo_map)
    return ResolutionResult(
        frame=out,
        column_map=dict(decision.column_map),
        derived_from_well=derived_from,
        pseudo_map=pseudo_map,
    )
