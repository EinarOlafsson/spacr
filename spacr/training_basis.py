"""One vocabulary for "what defines a training class", shared by Classify (CV)
and Classify (ML).

The two modules did the same job in different words. Of 78 CV settings and 37
ML ones, six shared a name — and two of those six disagreed on their default
(``annotation_column`` was ``'test'`` in one and ``None`` in the other;
``n_jobs`` was 28 and -1). Three more pairs were the same setting under
different names. A settings CSV was therefore not portable between them, and
neither was a user's understanding.

**The training basis.** Classify (CV) already had all three:
``dataset_mode`` is ``'metadata'``, ``'annotation'`` or ``'measurement'``, and
:mod:`spacr.io` builds the dataset from ``class_metadata``/``metadata_rules``,
``annotation_columns``/``annotation_values`` or ``measurement_rules``
accordingly. Classify (ML) had two, and chose between them **implicitly**:
``ml.py`` asked whether ``annotation_column`` was ``None``. Nothing said so in
the settings panel, so a user who filled in an annotation column silently
stopped training on their plate controls.

So there is no new concept to invent here. ``dataset_mode`` becomes the shared
name, ML gains the basis it lacked, and the choice becomes something the user
makes rather than something they trigger.

**Backward compatibility is the whole difficulty.** A settings CSV written
before this exists in every user's project folder, and INVARIANTS §6 is the
trap: a key *absent* from the dict means the pipeline falls back to its own
default, which can differ from the GUI's, and nothing says so. Every rename
here is therefore an alias, not a replacement — :func:`normalize_settings`
translates the old name and the old *implicit* basis into the new explicit
one, and a run from an old CSV does exactly what it did before.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

#: The three ways a training class can be defined, in the order the settings
#: panel offers them.
TRAINING_BASES: Tuple[str, ...] = ("metadata", "annotation", "measurement")

#: Retired name -> shared name. The old key keeps working; it is translated
#: once, here, so no consumer has to know both.
#:
#: ``model_type_ml`` is the clearest case: it named the same thing as
#: ``model_type`` and differed only in which module you were standing in.
SETTING_ALIASES: Dict[str, str] = {
    "model_type_ml": "model_type",
    "test_size": "test_split",
    "cross_validation": "cross_validation_enabled",
}

#: Which settings each basis actually uses. The GUI greys out the rest, and
#: this is the single source for that -- a list that lived in the GUI would
#: drift from what the pipeline reads, and the symptom would be a control the
#: user can edit that changes nothing.
BASIS_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "metadata": (
        "class_metadata", "metadata_type_by", "metadata_rules",
        "location_column", "positive_control", "negative_control",
    ),
    "annotation": (
        "annotation_column", "annotation_columns", "annotation_values",
    ),
    "measurement": (
        "measurement_rules", "measurement_columns", "custom_measurement",
    ),
}


class TrainingBasisError(ValueError):
    """A basis that spaCR does not have, or one that cannot run as configured."""


def resolve_basis(settings: Mapping[str, Any]) -> str:
    """Return the training basis a settings dict asks for.

    Precedence, and the reason:

    1. ``dataset_mode``, when set. It is the explicit answer.
    2. Otherwise, the ML module's historical *implicit* rule: an
       ``annotation_column`` that is set meant "train on annotations". This
       is what makes an old settings CSV behave exactly as it used to.
    3. Otherwise ``'metadata'``, which is what both modules defaulted to.

    :param settings: the run settings.
    :returns: one of :data:`TRAINING_BASES`.
    :raises TrainingBasisError: an unrecognised ``dataset_mode``. Silently
        falling back would train on the wrong labels and report success.
    """
    declared = settings.get("dataset_mode")
    if declared:
        basis = str(declared).strip().lower()
        if basis not in TRAINING_BASES:
            raise TrainingBasisError(
                f"dataset_mode={declared!r} is not one of {list(TRAINING_BASES)}. "
                f"A run cannot guess which labels were meant.")
        return basis
    if settings.get("annotation_column"):
        return "annotation"
    return "metadata"


def normalize_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``settings`` in the shared vocabulary. Never modifies the input.

    Applies :data:`SETTING_ALIASES` and pins ``dataset_mode`` to whatever
    :func:`resolve_basis` worked out, so every consumer downstream reads one
    name and one explicit basis.

    The **new** name wins when both are present. Someone who has set the
    current key has said what they mean; a stale alias left in the same CSV
    must not override it.

    :param settings: the run settings.
    :returns: a new dict.
    """
    out = dict(settings)
    for old, new in SETTING_ALIASES.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    out["dataset_mode"] = resolve_basis(out)
    return out


def settings_for_basis(basis: str) -> Tuple[str, ...]:
    """The settings that apply to ``basis``.

    :param basis: one of :data:`TRAINING_BASES`.
    :returns: the setting keys that basis reads.
    :raises TrainingBasisError: unknown basis.
    """
    key = str(basis).strip().lower()
    if key not in BASIS_SETTINGS:
        raise TrainingBasisError(
            f"{basis!r} is not one of {list(TRAINING_BASES)}")
    return BASIS_SETTINGS[key]


def inapplicable_settings(basis: str) -> Tuple[str, ...]:
    """Settings belonging to the OTHER bases -- what the GUI greys out.

    Greyed, not removed. INVARIANTS §6: a key absent from the dict makes the
    pipeline fall back to its own default, which can differ from the value
    the module needs. A greyed control keeps its value and stops being
    editable; a deleted one changes the run.

    :param basis: the chosen basis.
    :returns: setting keys that do not apply to it.
    """
    mine = set(settings_for_basis(basis))
    other: list = []
    for name, keys in BASIS_SETTINGS.items():
        if name == basis:
            continue
        other.extend(k for k in keys if k not in mine)
    return tuple(dict.fromkeys(other))


def describe_basis(basis: str) -> str:
    """One line for the settings panel, naming what the user must fill in."""
    return {
        "metadata": ("Classes come from plate metadata — the wells named by "
                     "positive/negative control, or by class_metadata."),
        "annotation": ("Classes come from an annotation column of png_list, "
                       "as written by the Annotate module."),
        "measurement": ("Classes come from thresholds on measured features. "
                        "Use more than one measurement: a single threshold "
                        "is a gate, not a class definition."),
    }[str(basis).strip().lower()]
