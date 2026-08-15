"""Classifier evaluation, calibration, and split-leakage diagnostics.

The module is model-agnostic: it consumes labels, probabilities, fold ids,
and sample paths. Deep-learning CV and future classical-ML pipelines can
therefore write the same evaluation bundle and the Qt workbench can display
one stable artifact format.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


EVALUATION_FILES = {
    "summary": "summary.json",
    "predictions": "oof_predictions.csv",
    "confusion_counts": "confusion_counts.csv",
    "confusion_normalized": "confusion_normalized.csv",
    "per_plate": "per_plate_metrics.csv",
    "calibration": "calibration.csv",
    "leakage": "leakage.json",
    "manifest": "evaluation_manifest.json",
    "confusion_figure": "confusion_matrix.png",
    "calibration_figure": "calibration.png",
}
"""Stable file names produced by :func:`write_evaluation_bundle`."""


# Train/test grouping ladder, from least to most independent.
SPLIT_LEVELS: Tuple[str, ...] = ("cell", "field", "well", "plate")

_SPLIT_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "cell": (),
    "field": ("plateID", "rowID", "columnID", "fieldID"),
    "well": ("plateID", "rowID", "columnID"),
    "plate": ("plateID",),
}


@dataclass(frozen=True)
class SplitReport:
    """Provenance and realised sizes for one train/test split."""

    group_by: str
    requested_fraction: float
    cell_fraction: float
    group_fraction: float
    train_cells: int
    test_cells: int
    train_groups: int
    test_groups: int
    total_groups: int
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe split provenance for model cards/settings."""
        return asdict(self)

    def summary(self) -> str:
        """Describe both whole-group and object-level holdout costs."""
        unit = self.group_by if self.group_by != "cell" else "object"
        return (
            f"split_by={self.group_by}: held out {self.test_groups} of "
            f"{self.total_groups} {unit} group(s) ({self.group_fraction:.1%}) "
            f"and {self.test_cells} of {self.train_cells + self.test_cells} "
            f"cells ({self.cell_fraction:.1%}); requested "
            f"{self.requested_fraction:.1%}. {self.rule}"
        )


def normalize_split_level(group_by: Any) -> str:
    """Return a canonical split level; legacy ``none`` means ``cell``."""
    if group_by in (None, False, "none", "off"):
        return "cell"
    level = str(group_by)
    if level not in SPLIT_LEVELS:
        choices = ", ".join(SPLIT_LEVELS)
        raise ValueError(
            f"Unknown group_by/cv_group_by split level {group_by!r}; use one of "
            f"{choices}. Names "
            "are exact and lower-case. 'none' remains an alias for 'cell'."
        )
    return level


def split_columns_for(group_by: Any, columns: Sequence[str],
                      table: str = "data") -> Tuple[str, List[str]]:
    """Resolve a split level to complete metadata columns or refuse it.

    ``prcfo`` and crop filenames are handled by :func:`split_group_values`;
    this helper describes the direct-column route used by measurement tables.
    Partial keys are never accepted because, for example, ``columnID='c1'``
    is not a well identity across rows and plates.
    """
    level = normalize_split_level(group_by)
    wanted = list(_SPLIT_COLUMNS[level])
    if not wanted:
        return level, []
    missing = [column for column in wanted if column not in columns]
    if missing:
        raise ValueError(
            f"Cannot split {table} by {level}: the complete identity needs "
            f"{wanted}, but {missing} {'is' if len(missing) == 1 else 'are'} "
            "missing. Use a frame with acquisition metadata, a valid prcfo, "
            "or explicitly choose a finer split level such as 'cell'."
        )
    return level, wanted


def _groups_from_prcfo(values: Sequence[Any], level: str,
                       table: str) -> np.ndarray:
    """Parse strict measurement identities from canonical object keys."""
    from . import schema

    groups: List[str] = []
    for position, value in enumerate(values):
        try:
            identity = schema.parse_prcfo(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot split {table} by {level}: prcfo at row {position} "
                f"is not a valid object identity ({value!r}): {exc}"
            ) from exc
        parts = identity.to_dict()
        wanted = _SPLIT_COLUMNS[level]
        groups.append("\x1f".join(str(parts[column]) for column in wanted))
    return np.asarray(groups, dtype=object)


def split_group_values(*, group_by: Any = "well",
                       frame: Optional[pd.DataFrame] = None,
                       paths: Optional[Sequence[Any]] = None,
                       table: str = "data") -> Tuple[str, np.ndarray]:
    """Build one strict group id per row from metadata, ``prcfo``, or paths.

    For grouped levels every identity must be verifiable. Inventing singleton
    ids for unparseable rows would make a leaking random split look grouped.
    """
    level = normalize_split_level(group_by)
    if frame is None and paths is None:
        raise ValueError("split_group_values needs a frame or paths")
    n = len(frame) if frame is not None else len(paths or ())
    if level == "cell":
        if frame is not None and "prcfo" in frame.columns:
            values = frame["prcfo"].astype(str)
            if frame["prcfo"].isna().any() or values.str.strip().eq("").any():
                raise ValueError(
                    f"Cannot split {table} by cell: at least one row has no "
                    "object identity.")
            return level, values.to_numpy(dtype=object)
        if frame is not None and frame.index.name == "prcfo":
            values = pd.Series(frame.index.astype(str))
            if values.str.strip().eq("").any():
                raise ValueError(
                    f"Cannot split {table} by cell: at least one row has no "
                    "object identity.")
            return level, values.to_numpy(dtype=object)
        if paths is not None:
            return level, np.asarray(
                [augmentation_family(path) for path in paths], dtype=object)
        return level, np.arange(n, dtype=np.int64).astype(object)

    wanted = list(_SPLIT_COLUMNS[level])
    if frame is not None and all(column in frame.columns for column in wanted):
        metadata = frame[wanted]
        missing = metadata.isna() | metadata.astype(str).apply(
            lambda column: column.str.strip().eq(""))
        if bool(missing.to_numpy().any()):
            row, column = np.argwhere(missing.to_numpy())[0]
            raise ValueError(
                f"Cannot split {table} by {level}: row {int(row)} has no "
                f"{wanted[int(column)]} identity. A random fallback would "
                "report an optimistic score as grouped."
            )
        return level, metadata.astype(str).agg("\x1f".join, axis=1).to_numpy()

    if frame is not None:
        if "prcfo" in frame.columns:
            return level, _groups_from_prcfo(frame["prcfo"].tolist(), level,
                                             table)
        if frame.index.name == "prcfo":
            return level, _groups_from_prcfo(frame.index.tolist(), level,
                                             table)
        if paths is None and "png_path" in frame.columns:
            paths = frame["png_path"].tolist()

    if paths is not None:
        groups = []
        for position, path in enumerate(paths):
            identity = sample_identity(path)
            # A crop needs plate, well, field, and object tokens. Accepting a
            # two-token arbitrary filename as ``plate_well`` invents a well
            # identity and turns a random split into a grouped-looking one.
            encoded = augmentation_family(path).split("_")
            value = identity.get(level, "")
            if len(encoded) < 4 or not value:
                raise ValueError(
                    f"Cannot split {table} by {level}: path at row "
                    f"{position} ({path!r}) does not encode a {level}. "
                    "spaCR crop names must encode plate_well_field before "
                    "the object; choose 'cell' explicitly only if leakage "
                    "between sibling crops is intended."
                )
            groups.append(value)
        return level, np.asarray(groups, dtype=object)

    missing = [column for column in wanted if frame is None or
               column not in frame.columns]
    raise ValueError(
        f"Cannot split {table} by {level}: missing identity columns {missing}, "
        "and no valid prcfo or crop path is available. A grouped design is "
        "therefore unverifiable and will not be replaced by a random split."
    )


def grouped_split(groups: Sequence[Any], labels: Sequence[Any], holdout: float,
                  seed: int = 0, *, group_by: Any = "well"
                  ) -> Tuple[np.ndarray, np.ndarray, SplitReport]:
    """Return a stratified holdout while keeping named groups intact.

    A grouped design is refused when either side cannot contain every class.
    This is intentionally stricter than silently scoring a model on siblings
    of its training rows or on a holdout that contains only one class.
    """
    from sklearn.model_selection import (GroupShuffleSplit,
                                         StratifiedGroupKFold,
                                         train_test_split)

    level = normalize_split_level(group_by)
    y = np.asarray(labels)
    group_values = np.asarray(groups, dtype=object)
    fraction = float(holdout)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("holdout must be a finite fraction strictly between 0 and 1")
    if len(y) != len(group_values):
        raise ValueError("group-aware splitting requires one group per label")
    if len(y) < 2:
        raise ValueError("a train/test split needs at least two labelled cells")
    classes = np.unique(y)

    indices = np.arange(len(y))
    distinct = np.unique(group_values.astype(str))
    protects_groups = level != "cell" or len(distinct) < len(y)
    if not protects_groups:
        counts = pd.Series(y).value_counts()
        if len(counts) > 1 and int(counts.min()) < 2:
            raise ValueError(
                "A cell split cannot put every class in both train and test: "
                f"class counts are {counts.to_dict()}. Add another labelled "
                "cell in the rare class."
            )
        stratify = y if int(counts.min()) >= 2 else None
        train_idx, test_idx = train_test_split(
            indices, test_size=fraction, random_state=int(seed),
            stratify=stratify,
        )
        rule = "stratified random split of objects; sibling cells may cross"
    else:
        if any(value is None or str(value).strip() == "" for value in group_values):
            raise ValueError(
                f"Cannot split by {level}: at least one cell has no group "
                "identity, so independence cannot be verified."
            )
        if len(distinct) < 2:
            where = distinct[0] if len(distinct) else "unknown"
            unit = "object" if level == "cell" else level
            raise ValueError(
                f"A {unit}-grouped held-out split is impossible: every "
                f"labelled cell comes from one {unit} ({where}). A random "
                "cell split would only measure how well the model memorised "
                f"this {unit}, not whether it transfers."
            )

        candidates: List[Tuple[np.ndarray, np.ndarray, str]] = []
        requested_splits = max(
            2, min(int(round(1.0 / fraction)), len(distinct), 20))
        # The nearest whole-group fraction may not contain every class. Try a
        # small ladder down to a half holdout before declaring the design
        # impossible; 25% over four class-confounded wells, for example, has
        # no two-class one-well test set but does have an honest two-well one.
        split_counts = sorted({
            requested_splits,
            *range(2, min(5, len(distinct)) + 1),
        })
        for n_splits in split_counts:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=int(seed))
                for train, test in splitter.split(indices, y, group_values):
                    candidates.append((train, test,
                        f"StratifiedGroupKFold({n_splits}) over "
                        f"{len(distinct)} {level} groups"))
            except ValueError:
                continue
        # More candidates make uneven group sizes land closer to test_size.
        splitter = GroupShuffleSplit(
            n_splits=min(256, max(32, len(distinct) * 4)),
            test_size=fraction,
            random_state=int(seed))
        try:
            for train, test in splitter.split(indices, y, group_values):
                candidates.append((train, test,
                    f"GroupShuffleSplit over {len(distinct)} {level} groups"))
        except ValueError:
            pass

        complete = [candidate for candidate in candidates
                    if set(np.unique(y[candidate[0]])) == set(classes)
                    and set(np.unique(y[candidate[1]])) == set(classes)]
        if not complete:
            per_class = {
                str(label): int(len(np.unique(group_values[y == label])))
                for label in classes
            }
            raise ValueError(
                f"A leakage-safe {level}-grouped split cannot put every "
                "class in both train and test. Independent groups per class: "
                f"{per_class}. Add independent {level}s, choose a finer "
                "level, or collect another class-bearing group; a random "
                "fallback would report memorisation as transfer."
            )

        target = fraction * len(y)
        train_idx, test_idx, rule = min(
            complete,
            key=lambda candidate: (
                abs(len(np.unique(group_values[candidate[1]])) /
                    len(distinct) - fraction),
                abs(len(candidate[1]) - target),
            ),
        )
        train_groups = set(group_values[train_idx].astype(str))
        test_groups = set(group_values[test_idx].astype(str))
        if train_groups & test_groups:  # defensive: sklearn promises this
            raise RuntimeError(f"{level} groups crossed the train/test boundary")

    train_idx = np.sort(np.asarray(train_idx, dtype=int))
    test_idx = np.sort(np.asarray(test_idx, dtype=int))
    total_groups = len(np.unique(group_values.astype(str)))
    test_groups_n = len(np.unique(group_values[test_idx].astype(str)))
    if level == "cell" and protects_groups:
        rule += (
            "; repeated rows of one object stay together; sibling cells may "
            "cross")
    report = SplitReport(
        group_by=level,
        requested_fraction=fraction,
        cell_fraction=len(test_idx) / len(y),
        group_fraction=test_groups_n / total_groups,
        train_cells=len(train_idx),
        test_cells=len(test_idx),
        train_groups=total_groups - test_groups_n,
        test_groups=test_groups_n,
        total_groups=total_groups,
        rule=rule + ("; no group appears on both sides"
                     if protects_groups else ""),
    )
    return train_idx, test_idx, report


class LeakageError(ValueError):
    """Raised when related samples cross a protected split boundary."""


@dataclass
class LeakageReport:
    """Overlap counts and examples for one train/validation boundary.

    :ivar group_by: protected split level (``none``, ``field``, ``well``,
        or ``plate``).
    :ivar train_samples: number of training paths.
    :ivar validation_samples: number of validation paths.
    :ivar overlap_counts: overlap count at each identity level.
    :ivar examples: up to ten shared identities per level.
    :ivar critical_levels: levels that invalidate the requested split.
    :ivar warnings: non-fatal caveats.
    """

    group_by: str
    train_samples: int
    validation_samples: int
    overlap_counts: Dict[str, int]
    examples: Dict[str, List[str]]
    split_name: str = ""
    critical_levels: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unverifiable_counts: Dict[str, int] = field(default_factory=dict)
    hash_errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True when no protected identity crosses the boundary."""
        return not self.critical_levels

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""
        data = asdict(self)
        data["passed"] = self.passed
        return data


def _stem(path: Any) -> str:
    """Return a normalized basename without its final extension."""
    return os.path.splitext(os.path.basename(str(path)))[0]


_AUGMENT_SUFFIX = re.compile(
    r"(?i)(?:[_-](?:aug(?:ment)?\d*|rot(?:ate)?(?:90|180|270)|"
    r"flip(?:ped)?[_-]?[hv]|horizontal|vertical))+$"
)


def augmentation_family(path: Any) -> str:
    """Return the original-sample family after removing augmentation suffixes.

    In-memory spaCR augmentations retain the exact source filename, while
    older exported datasets use suffixes such as ``_aug3``, ``_rot90`` or
    ``_flip_h``. Both forms collapse to one family.
    """
    stem = _stem(path)
    previous = None
    while previous != stem:
        previous = stem
        stem = _AUGMENT_SUFFIX.sub("", stem)
    return stem


def sample_identity(path: Any) -> Dict[str, str]:
    """Parse plate/well/field/object identities from a crop filename.

    Unknown levels are returned as empty strings rather than guessed. The
    object identity is the augmentation-normalized full stem.
    """
    # Exported crop names use either ``plate_well_field_object`` (for example
    # ``plate1_A01_f2_o7``) or the canonical PRCFO form with separate row and
    # column tokens (``plate1_r1_c1_f2_o7``).  Keep fields within wells and
    # objects within fields in both forms; including ``f2`` in the well
    # identity would split one biological well into multiple leakage groups.
    family = augmentation_family(path)
    parts = family.split("_")
    plate = parts[0] if len(parts) >= 1 and parts[0] else ""
    split_row_column = (
        len(parts) >= 3
        and re.fullmatch(r"(?i)r\d+", parts[1]) is not None
        and re.fullmatch(r"(?i)c\d+", parts[2]) is not None
    )
    well_end = 3 if split_row_column else 2
    field_end = well_end + 1
    well = "_".join(parts[:well_end]) if len(parts) >= well_end else ""
    field_id = (
        "_".join(parts[:field_end]) if len(parts) >= field_end else ""
    )
    return {
        "sample": str(path),
        "basename": os.path.basename(str(path)),
        "augmentation_family": family,
        "object": family,
        "plate": plate,
        "well": well,
        "field": field_id,
    }


def _identity_sets(paths: Iterable[Any]) -> Dict[str, set]:
    """Return non-empty identity values for a path collection."""
    result = {
        "exact": set(),
        "augmentation_family": set(),
        "object": set(),
        "field": set(),
        "well": set(),
        "plate": set(),
    }
    for path in paths:
        identity = sample_identity(path)
        result["exact"].add(os.path.abspath(str(path)))
        for level in result:
            if level == "exact":
                continue
            value = identity[level]
            if value:
                result[level].add(value)
    return result


def _content_sha256(path: Any) -> Tuple[str, str]:
    """Return ``(sha256, error)`` for one file without loading it into memory."""
    try:
        candidate = Path(str(path))
        if not candidate.is_file():
            return "", f"{candidate}: file does not exist"
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), ""
    except OSError as exc:
        return "", f"{path}: {type(exc).__name__}: {exc}"


def _identity_sets_with_hashes(
    paths: Iterable[Any],
    *,
    hash_content: bool,
) -> Tuple[Dict[str, set], List[str]]:
    """Return identity sets, optionally including byte-content hashes."""
    result = _identity_sets(paths)
    result["content_sha256"] = set()
    errors: List[str] = []
    if hash_content:
        for path in paths:
            digest, error = _content_sha256(path)
            if digest:
                result["content_sha256"].add(digest)
            elif error:
                errors.append(error)
    return result, errors


def audit_split_leakage(
    train_paths: Sequence[Any],
    validation_paths: Sequence[Any],
    *,
    group_by: str = "well",
    raise_on_leakage: bool = False,
    split_name: str = "",
    hash_content: bool = False,
    require_identity: bool = False,
) -> LeakageReport:
    """Detect related images crossing a train/validation boundary.

    Exact/object/augmentation-family overlap is always critical. The requested
    ``group_by`` level is also critical: a well-grouped split permits the same
    plate on both sides but never the same well.

    :param train_paths: source paths used to fit the model.
    :param validation_paths: paths used only for evaluation.
    :param group_by: ``cell``, ``field``, ``well``, or ``plate``. Legacy
        ``none`` aliases ``cell``.
    :param raise_on_leakage: raise :class:`LeakageError` on a critical overlap.
    :param split_name: optional fold/split label stored in the report.
    :param hash_content: also compare file CONTENT, so a byte-identical copy
        under a different name is caught. Costs one read per file.
    :param require_identity: treat UNVERIFIABLE as critical. A filename that
        does not encode the requested identity, or a file that cannot be
        hashed, is otherwise only a warning -- so leaving this False means a
        clean report can still hide leakage nobody could check for.
    :returns: :class:`LeakageReport`.
    """
    group_by = normalize_split_level(group_by)
    train, train_hash_errors = _identity_sets_with_hashes(
        train_paths, hash_content=hash_content,
    )
    validation, validation_hash_errors = _identity_sets_with_hashes(
        validation_paths, hash_content=hash_content,
    )
    overlap = {
        level: sorted(train[level] & validation[level])
        for level in train
    }
    critical_candidates = [
        "exact", "content_sha256", "augmentation_family", "object",
    ]
    if group_by != "cell":
        critical_candidates.append(group_by)
    critical = [
        level for level in dict.fromkeys(critical_candidates)
        if overlap[level]
    ]
    warnings = []
    if group_by == "cell":
        warnings.append(
            "The split is not grouped; shared well/field acquisition context "
            "can inflate performance even when exact objects do not overlap."
        )
    missing_train = sum(
        not sample_identity(path)[group_by]
        for path in train_paths
    ) if group_by != "cell" else 0
    missing_val = sum(
        not sample_identity(path)[group_by]
        for path in validation_paths
    ) if group_by != "cell" else 0
    unverifiable = {}
    if missing_train or missing_val:
        unverifiable[group_by] = int(missing_train + missing_val)
        warnings.append(
            f"{missing_train} training and {missing_val} validation filename(s) "
            f"do not encode the requested {group_by} identity."
        )
        if require_identity:
            critical.append(f"unverifiable_{group_by}")
    hash_errors = train_hash_errors + validation_hash_errors
    if hash_content and hash_errors:
        warnings.append(
            f"{len(hash_errors)} file(s) could not be content-hashed, so renamed "
            "byte-identical copies cannot be excluded for those samples."
        )
        if require_identity:
            critical.append("unverifiable_content")
    report = LeakageReport(
        group_by=group_by,
        train_samples=len(train_paths),
        validation_samples=len(validation_paths),
        overlap_counts={level: len(values) for level, values in overlap.items()},
        examples={level: values[:10] for level, values in overlap.items()},
        split_name=str(split_name),
        critical_levels=critical,
        warnings=warnings,
        unverifiable_counts=unverifiable,
        hash_errors=hash_errors[:20],
    )
    if raise_on_leakage and not report.passed:
        details = ", ".join(
            f"{level}={report.overlap_counts.get(level, report.unverifiable_counts.get(group_by, 1))}"
            for level in report.critical_levels
        )
        raise LeakageError(
            f"Train/validation leakage detected ({details}). Rebuild folds "
            f"with cv_group_by={group_by!r}, split before augmentation, and "
            "preserve spaCR crop identities in filenames."
        )
    return report


@dataclass
class FoldLeakageAudit:
    """Whole-CV proof that each related sample family belongs to one fold."""

    group_by: str
    n_samples: int
    n_folds: int
    validation_membership_missing: List[int] = field(default_factory=list)
    validation_membership_duplicate: List[int] = field(default_factory=list)
    overlap_counts: Dict[str, int] = field(default_factory=dict)
    examples: Dict[str, List[str]] = field(default_factory=dict)
    critical_levels: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unverifiable_counts: Dict[str, int] = field(default_factory=dict)
    hash_errors: List[str] = field(default_factory=list)
    split_name: str = "all_cv_folds"

    @property
    def passed(self) -> bool:
        """Return True only when the fold partition is complete and isolated."""
        return not self.critical_levels

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable audit record."""
        result = asdict(self)
        result["passed"] = self.passed
        return result


def audit_cv_folds(
    paths: Sequence[Any],
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    *,
    labels: Optional[Sequence[Any]] = None,
    group_by: str = "well",
    hash_content: bool = False,
    require_identity: bool = True,
    raise_on_leakage: bool = False,
) -> FoldLeakageAudit:
    """Verify partition coverage and identity isolation across every CV fold.

    This checks the fold assignment as one object, rather than trusting a
    sample of pairwise boundaries. Each index must be validation exactly once;
    exact paths, byte-identical content, source/augmentation families and the
    requested plate/well/field group must map to one held-out fold only.

    :param paths: one source path per sample, indexed by the fold indices.
    :param folds: ``(train_indices, validation_indices)`` per fold. Indices
        outside ``range(len(paths))`` are reported rather than ignored.
    :param labels: optional class per path, one value per path. Only used to
        describe the folds; it does not affect leakage detection.
    :param group_by: identity level that may not cross a fold boundary --
        ``none``, ``field``, ``well`` or ``plate``. A well-grouped split
        permits the same plate on both sides but never the same well.
    :param hash_content: also compare file CONTENT, so a byte-identical copy
        under a different name is caught. Costs one read per file.
    :param require_identity: treat UNVERIFIABLE as critical. A filename that
        does not encode the requested identity, or a file that cannot be
        hashed, is otherwise only a warning -- so leaving this False means a
        clean report can still hide leakage nobody could check for.
    :param raise_on_leakage: raise :class:`LeakageError` instead of returning
        a report whose ``passed`` is False.
    :returns: a :class:`FoldLeakageAudit` carrying the per-level overlap
        counts, examples, and which levels were critical.
    :raises ValueError: for an unsupported ``group_by``, or ``labels`` whose
        length does not match ``paths``.
    """
    group_by = normalize_split_level(group_by)
    n_samples = len(paths)
    membership: List[List[int]] = [[] for _ in range(n_samples)]
    warnings: List[str] = []
    for fold_index, (train_indices, validation_indices) in enumerate(folds, 1):
        train_set = {int(index) for index in train_indices}
        validation_set = {int(index) for index in validation_indices}
        invalid = sorted(
            index for index in train_set | validation_set
            if index < 0 or index >= n_samples
        )
        if invalid:
            raise ValueError(
                f"fold {fold_index} contains out-of-range indexes {invalid[:10]}"
            )
        if train_set & validation_set:
            raise LeakageError(
                f"fold {fold_index} puts indexes in both train and validation: "
                f"{sorted(train_set & validation_set)[:10]}"
            )
        for index in validation_set:
            membership[index].append(fold_index)

    missing = [index for index, owners in enumerate(membership) if not owners]
    duplicate = [index for index, owners in enumerate(membership) if len(owners) > 1]
    levels = (
        "exact", "content_sha256", "augmentation_family", "object",
        "field", "well", "plate",
    )
    owners_by_identity: Dict[str, Dict[str, set]] = {
        level: {} for level in levels
    }
    missing_identity = 0
    hash_errors: List[str] = []
    label_by_identity: Dict[str, Dict[str, set]] = {
        level: {} for level in ("content_sha256", "augmentation_family", "object")
    }
    label_values = list(labels) if labels is not None else None
    if label_values is not None and len(label_values) != n_samples:
        raise ValueError("labels must have one value per path")

    for index, path in enumerate(paths):
        identity = sample_identity(path)
        values = {
            "exact": os.path.abspath(str(path)),
            **{level: identity[level] for level in (
                "augmentation_family", "object", "field", "well", "plate",
            )},
        }
        digest = ""
        if hash_content:
            digest, error = _content_sha256(path)
            if error:
                hash_errors.append(error)
        values["content_sha256"] = digest
        if group_by != "cell" and not values[group_by]:
            missing_identity += 1
        for level, value in values.items():
            if not value:
                continue
            owners_by_identity[level].setdefault(value, set()).update(
                membership[index]
            )
            if label_values is not None and level in label_by_identity:
                label_by_identity[level].setdefault(value, set()).add(
                    str(label_values[index])
                )

    overlaps = {
        level: {
            value: sorted(owners)
            for value, owners in owners_by_identity[level].items()
            if len(owners) > 1
        }
        for level in levels
    }
    critical = []
    if missing:
        critical.append("validation_membership_missing")
    if duplicate:
        critical.append("validation_membership_duplicate")
    protected = ["exact", "content_sha256", "augmentation_family", "object"]
    if group_by != "cell":
        protected.append(group_by)
    critical.extend(level for level in protected if overlaps[level])

    conflicts = {
        level: sorted(
            value for value, assigned in assignments.items()
            if len(assigned) > 1
        )
        for level, assignments in label_by_identity.items()
    }
    if any(conflicts.values()):
        critical.append("conflicting_labels")
        warnings.append(
            "Related crops carry different class labels; fix annotations before "
            "training even when all copies happen to be in one fold."
        )
    unverifiable = {}
    if missing_identity:
        unverifiable[group_by] = missing_identity
        warnings.append(
            f"{missing_identity} sample(s) do not encode {group_by} identity."
        )
        if require_identity:
            critical.append(f"unverifiable_{group_by}")
    if hash_content and hash_errors:
        unverifiable["content_sha256"] = len(hash_errors)
        warnings.append(f"{len(hash_errors)} sample(s) could not be hashed.")
        if require_identity:
            critical.append("unverifiable_content")

    examples = {
        level: [
            f"{value} -> folds {','.join(map(str, owners))}"
            for value, owners in list(overlaps[level].items())[:10]
        ]
        for level in levels
    }
    examples["conflicting_labels"] = [
        f"{level}:{value}"
        for level, values in conflicts.items()
        for value in values[:10]
    ][:10]
    audit = FoldLeakageAudit(
        group_by=group_by,
        n_samples=n_samples,
        n_folds=len(folds),
        validation_membership_missing=missing[:20],
        validation_membership_duplicate=duplicate[:20],
        overlap_counts={
            level: len(values) for level, values in overlaps.items()
        },
        examples=examples,
        critical_levels=list(dict.fromkeys(critical)),
        warnings=warnings,
        unverifiable_counts=unverifiable,
        hash_errors=hash_errors[:20],
    )
    if raise_on_leakage and not audit.passed:
        raise LeakageError(
            "Cross-validation leakage audit failed: "
            + ", ".join(audit.critical_levels)
        )
    return audit


def dataset_split_paths(root: Any, split: str) -> List[str]:
    """Return sorted image paths under ``root/<split>/<class>/``."""
    folder = Path(str(root)).expanduser() / str(split)
    if not folder.is_dir():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    return sorted(
        str(path) for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def audit_dataset_splits(
    root: Any,
    *,
    group_by: str = "well",
    hash_content: bool = True,
    require_identity: bool = True,
    raise_on_leakage: bool = False,
) -> LeakageReport:
    """Audit the permanent ``train/`` versus ``test/`` dataset boundary.

    :param root: dataset directory holding ``train/`` and ``test/``. Both are
        searched recursively, and only ``.png``, ``.jpg``, ``.jpeg``, ``.tif``,
        ``.tiff``, ``.bmp`` and ``.npy`` files are collected, so a folder of
        any other format audits as empty.
    :param group_by: identity level that may not appear on both sides --
        ``none``, ``field``, ``well`` or ``plate``. The default ``well`` still
        permits the same plate in train and test. It is validated only after
        the images are found, so a bad value over an empty tree reports the
        missing images instead.
    :param hash_content: defaults to True here, unlike
        :func:`audit_split_leakage`, so a byte-identical copy saved under a
        different name fails the audit -- at the cost of one read per file.
    :param require_identity: defaults to True here: filenames that do not
        encode the ``group_by`` level, and files that cannot be hashed, become
        critical instead of a warning, so an unverifiable split cannot report
        as clean.
    :param raise_on_leakage: raise :class:`LeakageError` instead of returning
        a report whose ``passed`` is False.
    :raises FileNotFoundError: when either side collects no image -- a missing
        or unreadable folder is never treated as a passing split.
    :returns: a :class:`LeakageReport` whose ``split_name`` is always
        ``train_vs_test``; the ``test/`` side is counted as
        ``validation_samples``.
    """
    train_paths = dataset_split_paths(root, "train")
    test_paths = dataset_split_paths(root, "test")
    if not train_paths or not test_paths:
        missing = [
            name for name, values in (("train", train_paths), ("test", test_paths))
            if not values
        ]
        raise FileNotFoundError(
            f"Cannot audit dataset leakage: no images found in {', '.join(missing)} "
            f"under {Path(str(root)).expanduser()}."
        )
    return audit_split_leakage(
        train_paths,
        test_paths,
        group_by=group_by,
        raise_on_leakage=raise_on_leakage,
        split_name="train_vs_test",
        hash_content=hash_content,
        require_identity=require_identity,
    )


def write_leakage_audit(path: Any, audit: Any) -> Path:
    """Atomically write a leakage report/audit as JSON and return its path."""
    destination = Path(str(path)).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = audit.to_dict() if hasattr(audit, "to_dict") else dict(audit)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, destination)
    return destination


def normalize_probabilities(
    probabilities: Any,
    *,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """Return a finite, row-normalized ``(n_samples, n_classes)`` matrix."""
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = np.column_stack([1.0 - matrix, matrix])
    elif matrix.ndim == 2 and matrix.shape[1] == 1:
        values = matrix[:, 0]
        matrix = np.column_stack([1.0 - values, values])
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(
            "probabilities must be a one-dimensional positive-class vector "
            "or a two-dimensional matrix with at least two classes."
        )
    if n_classes is not None and matrix.shape[1] != int(n_classes):
        raise ValueError(
            f"Probability matrix has {matrix.shape[1]} columns but "
            f"{n_classes} classes were declared."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("probabilities must lie between 0 and 1.")
    totals = matrix.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("At least one probability row sums to zero.")
    return matrix / totals


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Any,
    *,
    n_bins: int = 10,
) -> float:
    """Return top-label expected calibration error (ECE)."""
    y = np.asarray(y_true, dtype=int)
    probs = normalize_probabilities(probabilities)
    if len(y) != len(probs):
        raise ValueError("y_true and probabilities must have equal length.")
    if len(y) == 0:
        return float("nan")
    n_bins = max(2, int(n_bins))
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y))
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return float(ece)


def _temperature_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to an already normalized probability matrix."""
    temperature = max(float(temperature), 1e-6)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(
    y_true: Sequence[int],
    probabilities: Any,
) -> float:
    """Fit one scalar temperature by minimizing multiclass log loss."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(y_true, dtype=int)
    probs = normalize_probabilities(probabilities)
    if len(y) < 2 or np.unique(y).size < 2:
        raise ValueError(
            "Temperature calibration needs at least two classes and two samples."
        )

    def objective(log_temperature: float) -> float:
        calibrated = _temperature_probabilities(
            probs, math.exp(float(log_temperature)),
        )
        return float(log_loss(y, calibrated, labels=np.arange(probs.shape[1])))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(math.exp(float(result.x)))


def cross_calibrate_probabilities(
    y_true: Sequence[int],
    probabilities: Any,
    fold_ids: Sequence[Any],
    *,
    method: str = "temperature",
    warnings_out: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calibrate each held-out fold using every *other* out-of-fold prediction.

    This cross-fitting prevents a sample from fitting the calibrator that is
    evaluated on that same sample.

    :param y_true: class INDEX per sample, read only to fit the temperatures.
        A value outside the probability columns is REFUSED -- it used to make
        every fold's fit fail, fall back to temperature 1.0, and return
        uncalibrated probabilities while reporting that calibration ran.
    :param probabilities: predicted probabilities; a 1-D positive-class vector
        is expanded to two columns and every row is renormalized, so the
        result is always a new normalized matrix -- never the object passed
        in, even when no calibration is applied.
    :param fold_ids: held-out block per sample; any hashable value works, and
        each distinct value is calibrated from all the others, so at least two
        distinct values are required. The returned map is keyed by
        ``str(fold_id)``.
    :param method: ``temperature``, or ``none``/``off``/``false`` (also
        ``None`` and the empty string) to return the normalized probabilities
        unchanged with an empty temperature map. Case and surrounding
        whitespace are ignored; anything else is refused rather than skipped.
    :param warnings_out: list appended in place when a fold cannot be
        calibrated from the others. The message is printed either way, so
        leaving this None only discards the machine-readable copy.
    :raises ValueError: on mismatched lengths, an unrecognized ``method``, or
        fewer than two distinct ``fold_ids`` when calibrating.
    :returns: ``(calibrated_probabilities, temperature_by_held_out_fold)``.
    """
    normalized_method = str(method or "none").strip().lower()
    probs = normalize_probabilities(probabilities)
    y = np.asarray(y_true, dtype=int)
    folds = np.asarray(fold_ids)
    # REFUSE a label the head has no column for. Without this every fold's
    # fit raised inside its own try/except, each fell back to temperature
    # 1.0, and the function returned UNCALIBRATED probabilities while
    # reporting that calibration ran -- visible only as a printed warning a
    # caller has no reason to be watching.
    if len(y) and probs.shape[1]:
        stray = sorted({int(v) for v in y if v < 0 or v >= probs.shape[1]})
        if stray:
            raise ValueError(
                f"y_true contains class indices {stray} outside the "
                f"{probs.shape[1]} probability columns; calibration would "
                "silently do nothing.")
    if len(y) != len(probs) or len(folds) != len(y):
        raise ValueError(
            "y_true, probabilities, and fold_ids must have equal length."
        )
    if normalized_method in {"none", "off", "false"}:
        return probs.copy(), {}
    if normalized_method != "temperature":
        raise ValueError(
            "calibration_method must be 'none' or 'temperature', not "
            f"{method!r}."
        )
    unique_folds = list(pd.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError(
            "Cross-fitted calibration requires predictions from at least two folds."
        )
    calibrated = np.empty_like(probs)
    temperatures: Dict[str, float] = {}
    for fold in unique_folds:
        target = folds == fold
        fit = ~target
        try:
            temperature = fit_temperature(y[fit], probs[fit])
        except (ValueError, RuntimeError) as exc:
            temperature = 1.0
            warning = (
                f"Held-out fold {fold!r} could not be temperature-calibrated "
                f"from the other folds ({type(exc).__name__}: {exc}); its raw "
                "probabilities were retained."
            )
            print(f"Warning: classifier calibration: {warning}")
            if warnings_out is not None:
                warnings_out.append(warning)
        calibrated[target] = _temperature_probabilities(
            probs[target], temperature,
        )
        temperatures[str(fold)] = temperature
    return calibrated, temperatures


#: Columns of the frame :func:`calibration_table` returns, named here so an
#: empty result still has them. Without this a table with no rows is shape
#: (0, 0) and every downstream column access raises KeyError.
CALIBRATION_COLUMNS = (
    "class_index", "class_name", "bin", "bin_lower", "bin_upper",
    "n", "mean_confidence", "observed_frequency", "calibration_gap",
)


def calibration_table(
    y_true: Sequence[int],
    probabilities: Any,
    *,
    classes: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return per-class reliability bins for calibration plots.

    :param y_true: class INDEX per sample, compared column by column. A label
        outside the probability columns is REFUSED: it matches no class, so it
        used to read ``observed_frequency`` 0.0 everywhere and render as a
        catastrophically miscalibrated curve rather than an error.
    :param probabilities: predicted probabilities; a 1-D positive-class vector
        is expanded to two columns and every row is renormalized.
    :param classes: display names in column order. An EMPTY sequence falls
        back to ``class_0 ... class_n``; a non-empty one whose length
        disagrees with the columns is refused.
    :param n_bins: equal-width confidence bins over ``[0, 1]``, truncated to
        an int and floored at 2, so 0, 1 and any negative value all give two
        bins. The top bin is closed on the right, so confidence 1.0 lands in
        it rather than falling out of the table.
    :raises ValueError: when ``y_true`` and ``probabilities`` differ in
        length, ``classes`` has the wrong length, or a class index falls
        outside the probability columns.
    :returns: one row per class and NON-EMPTY bin, so the frame is shorter
        than ``n_classes * n_bins`` rows. An empty result still carries
        :data:`CALIBRATION_COLUMNS`, so a column can be indexed on it.
    """
    y = np.asarray(y_true, dtype=int)
    probs = normalize_probabilities(probabilities)
    if len(y) != len(probs):
        raise ValueError("y_true and probabilities must have equal length.")
    n_classes = probs.shape[1]
    # A label with no column matches nothing, so every observed_frequency
    # reads 0.0 and the reliability curve looks catastrophically
    # miscalibrated rather than wrong. Refuse it.
    if len(y) and n_classes:
        stray = sorted({int(v) for v in y if v < 0 or v >= n_classes})
        if stray:
            raise ValueError(
                f"y_true contains class indices {stray} outside the "
                f"{n_classes} probability columns.")
    names = list(classes or [f"class_{i}" for i in range(n_classes)])
    if len(names) != n_classes:
        raise ValueError("classes and probability columns must have equal length.")
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    rows = []
    for class_index, class_name in enumerate(names):
        observed = y == class_index
        confidence = probs[:, class_index]
        for bin_index in range(len(edges) - 1):
            if bin_index == len(edges) - 2:
                mask = (
                    (confidence >= edges[bin_index])
                    & (confidence <= edges[bin_index + 1])
                )
            else:
                mask = (
                    (confidence >= edges[bin_index])
                    & (confidence < edges[bin_index + 1])
                )
            if not mask.any():
                continue
            rows.append({
                "class_index": class_index,
                "class_name": class_name,
                "bin": bin_index + 1,
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "observed_frequency": float(observed[mask].mean()),
                "calibration_gap": float(
                    observed[mask].mean() - confidence[mask].mean()
                ),
            })
    # Name the columns even with no rows. `pd.DataFrame([])` is shape (0, 0),
    # so a caller indexing 'class_name' or 'bin' on an empty result got a
    # KeyError rather than an empty column -- and an empty table is a
    # legitimate state (every bin can be empty), not a broken frame.
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def _metric_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int,
) -> Dict[str, Any]:
    """Compute scalar classifier metrics for one sample group."""
    predicted = probabilities.argmax(axis=1)
    class_indices = np.arange(probabilities.shape[1])
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    per_class_recall = recall_score(
        y_true,
        predicted,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    supported = np.bincount(
        y_true, minlength=probabilities.shape[1],
    ) > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(per_class_recall[supported].mean()),
        "f1_macro": float(f1_score(
            y_true, predicted, average="macro", zero_division=0,
        )),
        "f1_weighted": float(f1_score(
            y_true, predicted, average="weighted", zero_division=0,
        )),
        "precision_macro": float(precision_score(
            y_true, predicted, average="macro", zero_division=0,
        )),
        "recall_macro": float(recall_score(
            y_true, predicted, average="macro", zero_division=0,
        )),
        "log_loss": float(log_loss(
            y_true, probabilities, labels=class_indices,
        )),
        "brier_multiclass": float(np.mean(np.sum(
            (probabilities - one_hot) ** 2, axis=1,
        ))),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, n_bins=n_bins,
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def evaluate_predictions(
    y_true: Sequence[int],
    probabilities: Any,
    sample_paths: Sequence[Any],
    *,
    classes: Optional[Sequence[str]] = None,
    fold_ids: Optional[Sequence[Any]] = None,
    calibration_method: str = "none",
    calibration_bins: int = 10,
) -> Dict[str, Any]:
    """Build overall, confusion, calibration, and per-plate evaluation tables.

    :param y_true: true class INDEX per sample, in ``range(n_classes)``. A
        value outside that range is refused rather than clipped.
    :param probabilities: ``(n_samples, n_classes)`` predicted probabilities.
        Its column count defines ``n_classes``.
    :param sample_paths: one source path per sample, used to derive the
        per-plate tables. Must be the same length as ``y_true``.
    :param classes: display names for the columns, in column order. Defaults
        to ``class_0 ... class_n``; a length that disagrees with the
        probability columns is refused.
    :param fold_ids: which held-out fold each sample came from. REQUIRED for
        temperature calibration and unused otherwise -- the temperature is
        fitted per held-out fold so no sample is calibrated on itself, which
        needs at least two distinct folds.
    :param calibration_method: ``'none'`` (default) or ``'temperature'``.
        Anything else is refused rather than silently ignored.
    :param calibration_bins: bin count for the expected-calibration-error
        estimate. More bins resolve the reliability curve better and make
        each bin noisier.
    :raises ValueError: on mismatched lengths, a class index outside the
        probability columns, an unknown ``calibration_method``, or temperature
        calibration with fewer than two distinct folds.
    :returns: the evaluation bundle, a dict with six keys:

        * ``summary`` — scalar metrics (``n``, accuracy, balanced accuracy,
          macro/weighted F1, macro precision/recall, log loss, multiclass
          Brier, expected calibration error, mean confidence) plus
          ``classes``, ``n_classes``, ``calibration_method``,
          ``raw_expected_calibration_error``,
          ``temperatures_by_held_out_fold``, ``calibration_warnings`` and
          ``probability_column_names``;
        * ``predictions`` — one row per sample with ``fold``, the
          :func:`sample_identity` columns, ``true_label`` / ``true_class``,
          ``predicted_label`` / ``predicted_class``, ``correct``,
          ``confidence`` (the calibrated probability of the chosen class) and
          a ``raw_prob_<name>`` / ``prob_<name>`` pair per class, where
          ``<name>`` is the sanitized, de-duplicated class name listed in
          ``probability_column_names`` rather than the class name itself;
        * ``confusion_counts`` — counts indexed and columned by class name;
        * ``confusion_normalized`` — the same matrix divided by its true-class
          row totals, with all-zero rows left at zero;
        * ``per_plate`` — the same scalar metrics per ``plate`` group, with
          ``plate`` as the first column;
        * ``calibration`` — the :func:`calibration_table` reliability bins for
          the calibrated probabilities.
    """
    y = np.asarray(y_true, dtype=int)
    raw = normalize_probabilities(probabilities)
    n_classes = raw.shape[1]
    names = list(classes or [f"class_{i}" for i in range(n_classes)])
    calibration_bins = int(calibration_bins)
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")
    if len(y) != len(raw) or len(sample_paths) != len(y):
        raise ValueError(
            "y_true, probabilities, and sample_paths must have equal length."
        )
    if len(names) != n_classes:
        raise ValueError("classes and probability columns must have equal length.")
    if len(set(names)) != len(names):
        raise ValueError("class names must be unique.")
    if len(y) == 0:
        raise ValueError("At least one prediction is required.")
    if (y < 0).any() or (y >= n_classes).any():
        raise ValueError("y_true contains a label outside the class schema.")

    folds = (
        np.asarray(fold_ids)
        if fold_ids is not None
        else np.zeros(len(y), dtype=int)
    )
    if len(folds) != len(y):
        raise ValueError(
            "fold_ids must have the same length as y_true and probabilities."
        )
    calibration_warnings: List[str] = []
    calibrated, temperatures = cross_calibrate_probabilities(
        y,
        raw,
        folds,
        method=calibration_method,
        warnings_out=calibration_warnings,
    ) if str(calibration_method).lower() not in {"none", "off", "false"} else (
        raw.copy(), {}
    )
    identities = pd.DataFrame([sample_identity(path) for path in sample_paths])
    predicted = calibrated.argmax(axis=1)
    frame = identities.copy()
    frame.insert(0, "fold", folds)
    frame["true_label"] = y
    frame["true_class"] = [names[index] for index in y]
    frame["predicted_label"] = predicted
    frame["predicted_class"] = [names[index] for index in predicted]
    frame["correct"] = predicted == y
    frame["confidence"] = calibrated.max(axis=1)
    safe_names = []
    used_safe_names = set()
    for index, name in enumerate(names):
        base = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
            or f"class_{index}"
        )
        safe_name = base
        suffix = 2
        while safe_name in used_safe_names:
            safe_name = f"{base}_{suffix}"
            suffix += 1
        used_safe_names.add(safe_name)
        safe_names.append(safe_name)
        frame[f"raw_prob_{safe_name}"] = raw[:, index]
        frame[f"prob_{safe_name}"] = calibrated[:, index]

    summary = _metric_summary(y, calibrated, n_bins=calibration_bins)
    summary.update({
        "classes": names,
        "n_classes": n_classes,
        "calibration_method": str(calibration_method or "none").lower(),
        "raw_expected_calibration_error": expected_calibration_error(
            y, raw, n_bins=calibration_bins,
        ),
        "temperatures_by_held_out_fold": temperatures,
        "calibration_warnings": calibration_warnings,
        "probability_column_names": safe_names,
    })
    counts = confusion_matrix(y, predicted, labels=np.arange(n_classes))
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )
    confusion_counts = pd.DataFrame(counts, index=names, columns=names)
    confusion_normalized = pd.DataFrame(
        normalized, index=names, columns=names,
    )

    per_plate_rows = []
    for plate, group in frame.groupby("plate", dropna=False):
        indices = group.index.to_numpy(dtype=int)
        metrics = _metric_summary(
            y[indices], calibrated[indices], n_bins=calibration_bins,
        )
        metrics["plate"] = plate or "unknown"
        per_plate_rows.append(metrics)
    per_plate = pd.DataFrame(per_plate_rows)
    if not per_plate.empty:
        columns = ["plate", *[c for c in per_plate if c != "plate"]]
        per_plate = per_plate[columns]

    return {
        "summary": summary,
        "predictions": frame,
        "confusion_counts": confusion_counts,
        "confusion_normalized": confusion_normalized,
        "per_plate": per_plate,
        "calibration": calibration_table(
            y, calibrated, classes=names, n_bins=calibration_bins,
        ),
    }


def nested_group_folds(
    labels: Sequence[int],
    *,
    outer_splits: int,
    inner_splits: int,
    groups: Optional[Sequence[Any]] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Build nested stratified/grouped outer and inner index partitions.

    Inner indexes are returned in the original/global coordinate system.

    :param labels: class label per sample, coerced to int. Only its length and
        class balance matter -- the folds carry indices, not data.
    :param outer_splits: outer fold count, coerced with ``int`` (``2.9`` gives
        2); below 2 is refused.
    :param inner_splits: inner fold count built inside each outer TRAINING
        set, so it is bounded by that subset, not by the dataset: three inner
        folds over four samples fails on the outer training half even though
        the outer split itself succeeded.
    :param groups: optional group key per sample (well or plate id) kept whole
        within a fold at BOTH levels. None gives a plain stratified split that
        will scatter crops of the same well across folds.
    :param seed: outer folds use ``seed``; the inner folds of outer fold ``k``
        use ``seed + k``, so inner partitions differ between outer folds
        instead of repeating one layout. Must be an int -- a float reaches
        ``numpy.random.default_rng`` and raises ``TypeError``.
    :raises ValueError: when either split count is below 2, or when the sample
        count (or the number of distinct groups) cannot supply that many
        folds.
    :returns: one dict per outer fold, with ``outer_fold`` (1-based),
        ``train``, ``validation`` and ``inner`` -- a list of
        ``(train, validation)`` index pairs expressed in GLOBAL indices, not
        as positions inside ``train``.
    """
    from .io import make_cv_folds

    outer_splits = int(outer_splits)
    inner_splits = int(inner_splits)
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2.")
    y = np.asarray(labels, dtype=int)
    group_values = None if groups is None else np.asarray(groups)
    outer = make_cv_folds(
        y, outer_splits, groups=group_values, seed=seed,
    )
    result = []
    for outer_index, (outer_train, outer_validation) in enumerate(
        outer, start=1,
    ):
        inner_groups = (
            None if group_values is None else group_values[outer_train]
        )
        relative_inner = make_cv_folds(
            y[outer_train],
            inner_splits,
            groups=inner_groups,
            seed=seed + outer_index,
        )
        inner = [
            (outer_train[train_relative], outer_train[val_relative])
            for train_relative, val_relative in relative_inner
        ]
        result.append({
            "outer_fold": outer_index,
            "train": outer_train,
            "validation": outer_validation,
            "inner": inner,
        })
    return result


def write_evaluation_bundle(
    output_dir: Any,
    evaluation: Mapping[str, Any],
    *,
    leakage_reports: Optional[Sequence[LeakageReport]] = None,
) -> Path:
    """Atomically write a complete evaluation bundle and diagnostic figures."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    def write_json(name: str, payload: Any) -> None:
        path = destination / name
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def write_csv(name: str, frame: pd.DataFrame, *, index: bool = False) -> None:
        path = destination / name
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_csv(temporary, index=index)
        temporary.replace(path)

    write_json(EVALUATION_FILES["summary"], evaluation["summary"])
    write_csv(EVALUATION_FILES["predictions"], evaluation["predictions"])
    write_csv(
        EVALUATION_FILES["confusion_counts"],
        evaluation["confusion_counts"],
        index=True,
    )
    write_csv(
        EVALUATION_FILES["confusion_normalized"],
        evaluation["confusion_normalized"],
        index=True,
    )
    write_csv(EVALUATION_FILES["per_plate"], evaluation["per_plate"])
    write_csv(EVALUATION_FILES["calibration"], evaluation["calibration"])
    reports = [report.to_dict() for report in (leakage_reports or [])]
    write_json(EVALUATION_FILES["leakage"], {
        "passed": all(report["passed"] for report in reports),
        "folds": reports,
    })

    figure_warnings = []
    try:
        _write_confusion_figure(
            evaluation["confusion_normalized"],
            destination / EVALUATION_FILES["confusion_figure"],
        )
    except Exception as exc:
        figure_warnings.append(
            f"Confusion figure failed ({type(exc).__name__}: {exc})."
        )
    try:
        _write_calibration_figure(
            evaluation["calibration"],
            destination / EVALUATION_FILES["calibration_figure"],
        )
    except Exception as exc:
        figure_warnings.append(
            f"Calibration figure failed ({type(exc).__name__}: {exc})."
        )
    manifest = {
        "schema_version": 1,
        "files": EVALUATION_FILES,
        "summary": evaluation["summary"],
        "leakage_passed": all(report["passed"] for report in reports),
        "warnings": figure_warnings,
    }
    write_json(EVALUATION_FILES["manifest"], manifest)
    for warning in figure_warnings:
        print(f"Warning: classifier evaluation: {warning}")
    return destination / EVALUATION_FILES["manifest"]


def _write_confusion_figure(frame: pd.DataFrame, path: Path) -> None:
    """Render a normalized confusion heatmap."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(frame.to_numpy(dtype=float), vmin=0, vmax=1,
                        cmap="Blues")
    axis.set_xticks(np.arange(len(frame.columns)), labels=frame.columns,
                    rotation=45, ha="right")
    axis.set_yticks(np.arange(len(frame.index)), labels=frame.index)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Out-of-fold confusion matrix")
    for row in range(len(frame.index)):
        for column in range(len(frame.columns)):
            value = float(frame.iloc[row, column])
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      color="white" if value > 0.5 else "black")
    fig.colorbar(image, ax=axis, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_calibration_figure(frame: pd.DataFrame, path: Path) -> None:
    """Render one reliability curve per class."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777",
              label="Perfect calibration")
    for class_name, group in frame.groupby("class_name"):
        axis.plot(
            group["mean_confidence"],
            group["observed_frequency"],
            marker="o",
            label=str(class_name),
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Out-of-fold calibration")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def find_evaluation_bundles(root: Any) -> List[Path]:
    """Return evaluation manifests below ``root``, newest first."""
    source = Path(root).expanduser()
    if source.is_file():
        if source.name == EVALUATION_FILES["manifest"]:
            return [source]
        source = source.parent
    if not source.exists():
        raise FileNotFoundError(f"Evaluation source does not exist: {source}")
    manifests = list(source.rglob(EVALUATION_FILES["manifest"]))
    return sorted(
        manifests,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_evaluation_bundle(path: Any) -> Dict[str, Any]:
    """Load one evaluation bundle for the Qt workbench."""
    source = Path(path).expanduser()
    manifest_path = (
        source if source.is_file()
        else source / EVALUATION_FILES["manifest"]
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No {EVALUATION_FILES['manifest']} found at {source}."
        )
    folder = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def read_csv(key: str, **kwargs) -> pd.DataFrame:
        file_name = manifest.get("files", EVALUATION_FILES).get(
            key, EVALUATION_FILES[key],
        )
        path_value = folder / file_name
        return pd.read_csv(path_value, **kwargs) if path_value.is_file() else pd.DataFrame()

    leakage_path = folder / manifest.get("files", EVALUATION_FILES).get(
        "leakage", EVALUATION_FILES["leakage"],
    )
    leakage = (
        json.loads(leakage_path.read_text(encoding="utf-8"))
        if leakage_path.is_file() else {}
    )
    return {
        "path": manifest_path,
        "manifest": manifest,
        "summary": manifest.get("summary", {}),
        "predictions": read_csv("predictions"),
        "confusion_counts": read_csv("confusion_counts", index_col=0),
        "confusion_normalized": read_csv(
            "confusion_normalized", index_col=0,
        ),
        "per_plate": read_csv("per_plate"),
        "calibration": read_csv("calibration"),
        "leakage": leakage,
    }
