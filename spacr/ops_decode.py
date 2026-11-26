"""Decode rules, decoy FDR and decode QC for optical pooled screens (alpha).

Purpose and scope
=================
:func:`spacr.ops_engine.run_ops` decodes every field of a well. This module
holds the parts of that decode that the BMC decode-yield study added
(items 873, 876, 877 and 878), so that a settings run reproduces the study's
final rule rather than a script over a field cache:

* the field rule "rule (a) + S2" (:func:`field_calls`): library-calibrated
  base calls, territory attribution, quality-weighted votes, the runner-up
  one-mismatch rule and the pixel fallback, matched against the library and
  a decoy library;
* the decoy FDR report of every decode (:func:`fdr_report`,
  :func:`plate_fdr_summary`);
* the per-field evidence a field's calls were made from, in the cache layout
  the study used (:func:`field_evidence`), which the decode QC reads.

Every rule here is off unless its setting turns it on, so a decode that
asks for none of them is the decode it was before.
"""
from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from .ops_settings import ops_setting

import numpy as np

from .checkpoint import _atomic_json

#: The settings that reproduce "rule (a) + S2", the final decode rule of the
#: decode-yield study (2026-11-07): 74.43 % of plate A's nuclei assigned at a
#: decoy FDR of 0.065 per 100. ``ops_decode_preset: rule_a_s2`` applies them
#: to every key the user did not set.
RULE_A_S2: Dict[str, Any] = {
    "ops_channel_registration": "spots_refine",
    "ops_registration_refine_px": 3,
    "ops_spot_min_peak": 20.0,
    "ops_spot_overrule_share": 0.5,
    "ops_read_threshold": 315.0,
    "ops_crosstalk": "argmax_median",
    "ops_decode_library_calibration": True,
    "ops_calibration_passes": 2,
    "ops_calibration_min_labels": 30,
    "ops_cell_rule": "mapped",
    "ops_read_mismatches": 0,
    "ops_decode_runner_up": True,
    "ops_runner_up_margin": 0.1,
    "ops_decode_impute_missing_cycles": True,
    "ops_assign_by_cell_territory": True,
    "ops_footprint": 3.0,
    "ops_tie": 1.0,
    "ops_decode_quality_weighted": True,
    "ops_vote_weight_floor": 0.05,
    "ops_min_fraction": 0.6,
    "ops_min_top_share": 0.9,
    "ops_single_read_quality": 0.06,
    "ops_decode_pixel_fallback": True,
    "ops_pixel_read_gate": True,
    "ops_pixel_percentile": 97.0,
    "ops_pixel_radius": 10.0,
    "ops_pixel_min_pixels": 2,
    "ops_pixel_min_fraction": 0.6,
    "ops_pixel_quality": 0.1,
    "ops_decoy_library_k": 4,
    "ops_fdr_scale": 1.268,
    "ops_decode_skip_unplaced_sites": True,
}

#: The presets ``ops_decode_preset`` may name.
DECODE_PRESETS: Dict[str, Dict[str, Any]] = {"rule_a_s2": RULE_A_S2}

#: The defaults of the decode keys this module reads, which are the
#: behaviour before item 873: every new rule off.
_DEFAULTS: Dict[str, Any] = {
    "ops_registration_refine_px": 3,
    "ops_spot_min_peak": 20.0,
    "ops_spot_overrule_share": 0.5,
    "ops_decode_library_calibration": False,
    "ops_calibration_passes": 2,
    "ops_calibration_min_labels": 30,
    "ops_decode_runner_up": False,
    "ops_runner_up_margin": 0.1,
    "ops_decode_impute_missing_cycles": True,
    "ops_assign_by_cell_territory": False,
    "ops_tie": 1.0,
    "ops_decode_quality_weighted": False,
    "ops_vote_weight_floor": 0.05,
    "ops_min_top_share": 0.9,
    "ops_decode_pixel_fallback": False,
    "ops_pixel_read_gate": True,
    "ops_pixel_percentile": 97.0,
    "ops_pixel_radius": 10.0,
    "ops_pixel_min_pixels": 2,
    "ops_pixel_min_fraction": 0.6,
    "ops_pixel_quality": 0.1,
    "ops_pixel_ownership": "read_rule",
    "ops_pixel_footprint": 5.0,
    "ops_decoy_library_k": 0,
    "ops_fdr_scale": 1.268,
    "ops_decode_skip_unplaced_sites": True,
    "ops_store_field_cache": False,
    "ops_decode_qc": False,
    "ops_read_distance_decay_px": 4.0,
    "ops_read_distance_interior_px": 0.0,
    "ops_read_distance_min_weight": 0.05,
    "ops_read_neighbour_discount": 0.25,
    "ops_read_neighbour_reach": 10.0,
    "ops_read_cache_floor": 100.0,
}

#: How a fallback pixel finds the nucleus it votes for
#: (``ops_pixel_ownership``, item 1044), the first being the default:
#: ``read_rule`` gives it by the read assignment (:func:`_object_owners`:
#: the footprint, the tie and territory rule, the cell label under a cell
#: mask) with the read neighbour discount on the pixel tallies and the
#: pixel bleed gate (:func:`_bleed_refusals`); ``read_reach`` is the read
#: assignment and the discount without the gate; ``nearest_nucleus`` is the
#: rule before item 1044, the nearest nucleus within ``ops_pixel_radius``.
PIXEL_OWNERSHIP = ("read_rule", "read_reach", "nearest_nucleus")

#: How far from a nucleus boundary a fallback pixel is kept at all, in
#: pixels. The pixel calls are normalised over the pixels kept, so this
#: reach is part of the rule; the study's field cache kept 15 px.
_PIXEL_REACH = 15.0

#: The lowest spot strength the field cache keeps by default, so the read
#: threshold can be swept afterwards; the study's cache kept 100. The run's
#: value is ``ops_read_cache_floor`` (item N7), carried as the rule's
#: ``cache_floor``.
_CACHE_FLOOR = _DEFAULTS["ops_read_cache_floor"]

#: The decode settings whose non-default value routes a well through
#: :func:`field_calls`.
_RULE_SWITCHES = ("ops_decode_library_calibration", "ops_decode_runner_up",
                  "ops_assign_by_cell_territory",
                  "ops_decode_quality_weighted", "ops_decode_pixel_fallback",
                  "ops_store_field_cache", "ops_decode_qc")

#: Matchers and library indices built once per process, keyed by the
#: library and decoy sets they hold.
_MATCHERS: Dict[Tuple[Any, int], Any] = {}


def resolve_settings(settings: Mapping[str, Any],
                     defaults: Optional[Mapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """The settings with ``ops_decode_preset`` applied.

    :param settings: the run's settings.
    :param defaults: the panel defaults, used only when the settings carry no
        record of the keys the user set; then a key at its default takes the
        preset's value. None reads :data:`spacr.ops_settings.OPS_DEFAULTS`.
        With the record (item 1051) a key the user set keeps its value, even
        at the default.
    :returns: a copy with the preset and a shared private notice context
        when OPS settings enable the neighbour discount. Explicit nondefault
        values keep their entered values; applied-path flags are journal metadata.
    :raises ValueError: for an unknown preset.
    """
    out = dict(settings or {})
    if (any(str(key).startswith("ops_") for key in out)
            and _number(out, "ops_read_neighbour_discount") != 1.0):
        out["_meta_neighbour_discount"] = _neighbour_discount_context(settings)
    name = str(out.get("ops_decode_preset") or "none").strip().lower()
    if name in ("", "none"):
        return out
    if name not in DECODE_PRESETS:
        raise ValueError(f"ops_decode_preset must be one of "
                         f"{['none'] + sorted(DECODE_PRESETS)}; got {name!r}")
    if defaults is None:
        from .ops_settings import OPS_DEFAULTS as defaults
    from .settings import fill_preset

    return fill_preset(out, DECODE_PRESETS[name],
                       {**_DEFAULTS, **dict(defaults)})


def _number(settings: Mapping[str, Any], key: str, cast=float):
    """One numeric setting, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :param cast: ``float`` or ``int``.
    :returns: the value.
    :raises ValueError: when it is not a number.
    """
    value = settings.get(key)
    if value is None or value == "":
        value = _DEFAULTS[key]
    try:
        return cast(float(value)) if cast is int else cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number; got {value!r}") from None


def _flag(settings: Mapping[str, Any], key: str) -> bool:
    """One boolean setting, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :returns: the value; the strings ``false``, ``0``, ``no`` and ``off`` are
        False.
    """
    value = settings.get(key)
    if value is None or value == "":
        return bool(_DEFAULTS[key])
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def _neighbour_discount_context(settings):
    """Share transient notice state across shallow settings copies in one run.

    The context remains private; only the applied flags are published. An
    open journal owns a fresh context, so replay metadata never suppresses
    the first notice of a new run. Plain API callers share their mutable
    settings context across wells without a global process cache.
    """
    from .run_journal import current_run

    run = current_run()
    if run is not None:
        context = getattr(run, '_neighbour_discount_context', None)
        if context is None:
            context = {'paths': {}, 'notices': []}
            run._neighbour_discount_context = context
    else:
        context = settings.get('_meta_neighbour_discount') if settings is not None else None
        if not isinstance(context, dict) or not isinstance(context.get('paths'), dict):
            context = {'paths': {}, 'notices': []}
    if isinstance(settings, dict):
        settings['_meta_neighbour_discount'] = context
    return context


def _neighbour_discount_state(settings, path, applied, *, discount, method=None,
                              report=None, state=None):
    """Record actual discount availability and emit one translated no-op notice per path.

    False remains false when one path includes fields without the required
    positions. The current journal receives only JSON application flags;
    transient notice bookkeeping is never published. A supplied rule-state
    mapping preserves notice deduplication during direct cache replays.
    """
    from .qt.i18n import tr
    from .run_journal import current_run, _publish_resolved_settings

    context = _neighbour_discount_context(settings)
    paths = context['paths']
    previous = state.get(path) if isinstance(state, dict) else paths.get(path)
    if isinstance(state, dict):
        for key, value in state.items():
            paths[key] = bool(paths.get(key, True) and value)
    paths[path] = bool(paths.get(path, True) and applied)
    if isinstance(state, dict):
        state.update(paths)
    if isinstance(settings, dict):
        settings['ops_read_neighbour_discount_applied'] = paths
    run = current_run()
    seen = path in context['notices'] or (run is None and settings is None and previous is False)
    if (method is not None and not applied and not seen
            and float(discount) == _DEFAULTS["ops_read_neighbour_discount"]):
        labels = {"rules_off": tr("rules off"),
                  "cache_without_positions": tr("cache replay without nucleus positions")}
        message = tr("ops_read_neighbour_discount ({discount}) does not act under {method}; reads are not discounted",
                     discount=f"{float(discount):g}", method=labels.get(method, method))
        (report or print)(message)
        context['notices'].append(path)
    if run is not None and run._settings_owner_pid == os.getpid():
        recorded = run.settings.get('ops_read_neighbour_discount_applied')
        recorded = dict(recorded) if isinstance(recorded, dict) else {}
        for key, value in paths.items():
            recorded[key] = bool(recorded.get(key, True) and value)
        _publish_resolved_settings({'ops_read_neighbour_discount_applied': recorded})
    return paths


def _qc_panel_flag(settings: Mapping[str, Any], key: str) -> bool:
    """One opt-in decode QC panel setting, False when absent or empty.

    :param settings: the settings.
    :param key: ``ops_decode_qc_bleed_panel`` or ``ops_decode_qc_depth_curve``.
    :returns: the value; the strings ``false``, ``0``, ``no`` and ``off`` are False.
    """
    value = settings.get(key)
    if value is None or value == "":
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def decode_rule(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The field rule's parameters, read from resolved settings.

    :param settings: settings after :func:`resolve_settings`.
    :returns: the parameters :func:`field_calls` takes, plus ``active`` --
        True when any rule of this module is on, which routes the well
        through :func:`field_calls` -- and ``decoy_k``. With the item-988
        alpha distance weight or the neighbour discount on, ``read_weight``
        holds their settings. The default discount is skipped with a
        notice when rules are off; applied flags are recorded per path.
        ``cache_floor`` is ``ops_read_cache_floor`` (item N7), the weakest
        spot strength the field cache keeps. Under ``ops_read_assignment``
        'cell_mask' (item 1025) it carries ``read_assignment`` 'cell_mask';
        under 'nucleus_first' (item 1052) ``read_assignment``
        'nucleus_first', ``bleed_gate`` and, for a stored curve,
        ``bleed_model`` (:func:`_nucleus_first_rule`); the default
        'nucleus_reach' adds no key, so its rule is the rule before item 1025.
    :raises ValueError: for a negative count or a fraction outside (0, 1],
        for a negative ``ops_read_cache_floor``, or for explicit nondefault
        item-988 weighting without decode rules.
    """
    rule = {
        "calibration": _flag(settings, "ops_decode_library_calibration"),
        "calibration_passes": _number(settings, "ops_calibration_passes", int),
        "calibration_min_labels": _number(settings,
                                          "ops_calibration_min_labels", int),
        "runner_up": _flag(settings, "ops_decode_runner_up"),
        "runner_up_margin": _number(settings, "ops_runner_up_margin"),
        "impute": _flag(settings, "ops_decode_impute_missing_cycles"),
        "territory": _flag(settings, "ops_assign_by_cell_territory"),
        "tie": _number(settings, "ops_tie"),
        "quality_weighted": _flag(settings, "ops_decode_quality_weighted"),
        "weight_floor": _number(settings, "ops_vote_weight_floor"),
        "min_top_share": _number(settings, "ops_min_top_share"),
        "neighbour_reach": _neighbour_reach_setting(settings),
        "pixels": _flag(settings, "ops_decode_pixel_fallback"),
        "pixel_read_gate": _flag(settings, "ops_pixel_read_gate"),
        "pixel_percentile": _number(settings, "ops_pixel_percentile"),
        "pixel_radius": _number(settings, "ops_pixel_radius"),
        "pixel_min_pixels": _number(settings, "ops_pixel_min_pixels", int),
        "pixel_min_fraction": _number(settings, "ops_pixel_min_fraction"),
        "pixel_quality": _number(settings, "ops_pixel_quality"),
        "pixel_ownership": _pixel_ownership(settings),
        "pixel_footprint": _pixel_footprint_setting(settings),
        "decoy_k": _number(settings, "ops_decoy_library_k", int),
        "fdr_scale": _number(settings, "ops_fdr_scale"),
        "store_cache": (_flag(settings, "ops_store_field_cache")
                        or _flag(settings, "ops_decode_qc")
                        or _qc_panel_flag(settings, "ops_decode_qc_bleed_panel")
                        or _qc_panel_flag(settings, "ops_decode_qc_depth_curve")),
        "cache_floor": _number(settings, "ops_read_cache_floor"),
        "features": _flag(settings, "ops_decode_qc"),
    }
    if not rule["cache_floor"] >= 0:
        from .qt.i18n import tr

        raise ValueError(tr("ops_read_cache_floor must be 0 or more; got {value}",
                            value=rule["cache_floor"]))
    if settings.get("ops_report_read_stage") not in (None, "") \
            and _flag(settings, "ops_report_read_stage"):
        rule["read_stage"] = True
    if rule["decoy_k"] < 0 or rule["pixel_min_pixels"] < 1 \
            or rule["calibration_passes"] < 1:
        raise ValueError("ops_decoy_library_k must be 0 or more, and "
                         "ops_pixel_min_pixels and ops_calibration_passes "
                         "1 or more")
    if not 0 < rule["pixel_min_fraction"] <= 1:
        raise ValueError("ops_pixel_min_fraction must lie in (0, 1]; got "
                         f"{rule['pixel_min_fraction']}")
    if not 0 <= rule["min_top_share"] <= 1:
        from .qt.i18n import tr

        raise ValueError(tr("ops_min_top_share must lie in [0, 1]; got {value}",
                            value=rule["min_top_share"]))
    rule["active"] = bool(rule["decoy_k"] > 0 or any(
        _flag(settings, key) for key in _RULE_SWITCHES))
    assignment = read_assignment_setting(settings)
    if assignment == "cell_mask":
        rule["read_assignment"] = "cell_mask"
    elif assignment == "nucleus_first":
        rule.update(_nucleus_first_rule(settings, bool(rule["active"])))
    weighting = _read_weighting(settings)
    unavailable = False
    if weighting is not None:
        if not rule["active"]:
            unavailable = (weighting["distance_weight"] == "off"
                           and weighting["neighbour_discount"] == _DEFAULTS["ops_read_neighbour_discount"])
            if not unavailable:
                raise ValueError("ops_read_distance_weight and ops_read_neighbour_discount act in "
                                 "the decode rules' read vote; turn the rules on, for example "
                                 "with ops_decode_preset 'rule_a_s2'")
        else:
            rule["read_weight"] = weighting
    discount = _number(settings, "ops_read_neighbour_discount")
    state = _neighbour_discount_state(settings, 'full_barcode',
        bool(rule["active"] and discount < 1.0), discount=discount,
        method='rules_off' if unavailable else None)
    if discount != 1.0:
        rule['ops_read_neighbour_discount_applied'] = state
    return rule


#: How a read finds its object (``ops_read_assignment``, item 1025), the
#: first being the default: ``nucleus_reach`` is the nearest nucleus disc
#: within ``ops_footprint`` with the tie rule; ``cell_mask`` is the SBS cell
#: (``ops_sbs_cell_method``) the read lies in, with no footprint or tie;
#: ``nucleus_first`` (item 1052) keeps every nucleus-reach call and gives
#: only the objects the nucleus reach leaves uncalled their cell-mask
#: reads, through the bleed gate (:func:`_nucleus_first_calls`).
READ_ASSIGNMENTS = ("nucleus_reach", "cell_mask", "nucleus_first")

#: The defaults of the ``ops_read_assignment`` 'nucleus_first' gate (item
#: 1052): the bleed curve ('fit' fits it per run from the field caches,
#: 'arm2' is the curve the paper measured, or a path to a stored curve),
#: the largest share of the agreeing reads of a cell-mask call the curve
#: may explain as bleed from a neighbour, and the least agreeing reads the
#: curve leaves. They are read here and not kept in :data:`_DEFAULTS`, so
#: the resolved settings of the other assignments are unchanged.
NUCLEUS_FIRST_DEFAULTS = {"ops_nucleus_first_bleed_curve": "fit",
                          "ops_nucleus_first_max_bleed": 0.5,
                          "ops_nucleus_first_min_reads": 1.0}

#: Why the 'nucleus_first' gate refused a cell-mask read call (items 1052,
#: 1052b): the bleed curve explains more than ``ops_nucleus_first_max_bleed``
#: of its agreeing reads (``bleed_model``), or fewer than
#: ``ops_nucleus_first_min_reads`` reads remain once the bleed is taken off
#: (``min_support``). The pixel calls keep the 1044b gate of 'nucleus_reach'.
NUCLEUS_FIRST_REASONS = ("bleed_model", "min_support")


def _nucleus_first_rule(settings: Mapping[str, Any], active: bool) -> Dict[str, Any]:
    """The rule keys of ``ops_read_assignment`` 'nucleus_first' (item 1052).

    :param settings: the settings, read for the ``ops_nucleus_first_*`` keys
        (:data:`NUCLEUS_FIRST_DEFAULTS` when missing or empty).
    :param active: whether the decode rules are on (:func:`decode_rule`).
    :returns: ``read_assignment`` 'nucleus_first', ``bleed_gate``
        (``curve``, ``max_share``, ``min_reads``) and, unless the curve is
        'fit', ``bleed_model`` (:func:`bleed_model_setting`).
    :raises ValueError: when the decode rules are off, for a share outside
        [0, 1] or a negative minimum, and from :func:`bleed_model_setting`.
    """
    if not active:
        raise ValueError("ops_read_assignment 'nucleus_first' acts in the decode rules' "
                         "calls; turn them on, for example with ops_decode_preset "
                         "'rule_a_s2'")

    def value(key):
        got = settings.get(key)
        return NUCLEUS_FIRST_DEFAULTS[key] if got in (None, "") else got

    try:
        share = float(value("ops_nucleus_first_max_bleed"))
        least = float(value("ops_nucleus_first_min_reads"))
    except (TypeError, ValueError):
        raise ValueError("ops_nucleus_first_max_bleed and ops_nucleus_first_min_reads "
                         "must be numbers") from None
    if not 0.0 <= share <= 1.0 or least < 0:
        raise ValueError("ops_nucleus_first_max_bleed must lie in [0, 1] and "
                         "ops_nucleus_first_min_reads must be 0 or more; got "
                         f"{share} and {least}")
    curve = str(value("ops_nucleus_first_bleed_curve")).strip()
    out: Dict[str, Any] = {"read_assignment": "nucleus_first",
                           "bleed_gate": {"curve": curve, "max_share": share,
                                          "min_reads": least}}
    model = bleed_model_setting(curve)
    if model is not None:
        out["bleed_model"] = model
    return out


def bleed_model_setting(curve: str) -> Optional[Dict[str, Any]]:
    """The bleed model ``ops_nucleus_first_bleed_curve`` names (item 1052).

    :param curve: 'fit', 'arm2' or a path.
    :returns: None for 'fit' (the replay fits it per run,
        :func:`decode_cached_well`); the measured arm-2 model
        (:data:`spacr.ops_decode_qc_panels.ARM2_BLEED_MODEL`) for 'arm2';
        otherwise the stored curve
        (:func:`spacr.ops_decode_qc_panels.load_bleed_model`).
    :raises FileNotFoundError: for a path that does not exist.
    """
    from . import ops_decode_qc_panels as panels

    name = str(curve).strip()
    if name.lower() == "fit":
        return None
    if name.lower() == "arm2":
        return json.loads(json.dumps(panels.ARM2_BLEED_MODEL))
    return panels.load_bleed_model(name)


def read_assignment_setting(settings: Mapping[str, Any]) -> str:
    """``ops_read_assignment`` (item 1025), checked.

    :param settings: the settings.
    :returns: one of :data:`READ_ASSIGNMENTS`; missing or empty gives
        ``nucleus_reach``.
    :raises ValueError: for another value.
    """
    value = settings.get("ops_read_assignment")
    value = str(READ_ASSIGNMENTS[0] if value in (None, "") else value).strip().lower()
    if value not in READ_ASSIGNMENTS:
        raise ValueError(f"ops_read_assignment must be one of {list(READ_ASSIGNMENTS)}; "
                         f"got {value!r}")
    return value


def _pixel_ownership(settings: Mapping[str, Any]) -> str:
    """``ops_pixel_ownership`` (item 1044), checked.

    :param settings: the settings.
    :returns: one of :data:`PIXEL_OWNERSHIP`; empty gives the default.
    :raises ValueError: for another value.
    """
    value = settings.get("ops_pixel_ownership")
    value = str(_DEFAULTS["ops_pixel_ownership"] if value in (None, "") else value).strip().lower()
    if value not in PIXEL_OWNERSHIP:
        raise ValueError("ops_pixel_ownership must be one of "
                         f"{', '.join(PIXEL_OWNERSHIP)}; got {value!r}")
    return value


def _pixel_footprint_setting(settings: Mapping[str, Any]) -> Optional[float]:
    """``ops_pixel_footprint`` (item 1044b), or None to follow ``ops_footprint``.

    :param settings: settings after :func:`resolve_settings`.
    :returns: how far beyond a nucleus boundary a fallback pixel may lie
        under ``ops_pixel_ownership`` 'read_rule' or 'read_reach'; its
        default when the key is missing, and None when it is present but
        empty, the pixels then taking the read footprint, as settings files
        saved before schema 981 do.
    :raises ValueError: for a negative or non-numeric footprint.
    """
    if "ops_pixel_footprint" not in settings:
        return float(_DEFAULTS["ops_pixel_footprint"])
    value = settings["ops_pixel_footprint"]
    if value is None or value == "":
        return None
    reach = _number(settings, "ops_pixel_footprint")
    if reach < 0:
        raise ValueError(f"ops_pixel_footprint must be 0 or more; got {reach}")
    return reach


def _pixel_footprint(rule: Mapping[str, Any], footprint: float) -> float:
    """The footprint the fallback pixels are assigned at (item 1044b).

    :param rule: from :func:`decode_rule`, read for ``pixel_footprint``.
    :param footprint: the read footprint, used when the rule has none.
    :returns: ``pixel_footprint``, or ``footprint`` when it is None or
        missing.
    """
    value = rule.get("pixel_footprint")
    return float(footprint) if value is None else float(value)


def _neighbour_reach_setting(settings: Mapping[str, Any]) -> Optional[float]:
    """``ops_read_neighbour_reach``, or None to follow ``ops_footprint``.

    :param settings: settings after :func:`resolve_settings`.
    :returns: the reach in pixels; its default when the key is missing, and
        None when it is present but empty, the reach then being the
        footprint (:func:`_neighbour_reach`).
    :raises ValueError: for a negative or non-numeric reach.
    """
    if "ops_read_neighbour_reach" not in settings:
        return float(_DEFAULTS["ops_read_neighbour_reach"])
    value = settings["ops_read_neighbour_reach"]
    if value is None or value == "":
        return None
    reach = _number(settings, "ops_read_neighbour_reach")
    if reach < 0:
        from .qt.i18n import tr

        raise ValueError(tr("ops_read_neighbour_reach must be 0 or more; got {value}",
                            value=reach))
    return reach


def _neighbour_reach(rule: Mapping[str, Any], footprint: float) -> float:
    """How far apart two nucleus discs may be and still be neighbours for the discount.

    :param rule: from :func:`decode_rule`, read for ``neighbour_reach`` and
        ``footprint``.
    :param footprint: the footprint of the vote, used when the rule has neither.
    :returns: ``neighbour_reach``, or the footprint of the rule when it is None
        or missing, as for rules recorded before the reach had a setting.
    """
    reach = rule.get("neighbour_reach")
    return float(reach) if reach is not None else float(rule.get("footprint", footprint))


def _read_weighting(settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The read-vote weighting of item 988, or None while it is off.

    :param settings: settings after :func:`resolve_settings`.
    :returns: ``distance_weight``, ``decay_px``, ``interior_px``,
        ``min_weight`` and ``neighbour_discount``, read from the
        ``ops_read_distance_*`` settings and ``ops_read_neighbour_discount``;
        None when the distance weight is ``'off'`` and the discount is 1.
    :raises ValueError: for an unknown mode or a value out of range.
    """
    mode = str(settings.get("ops_read_distance_weight") or "off").strip().lower()
    discount = _number(settings, "ops_read_neighbour_discount")
    if mode not in ("off", "decay", "distance_only"):
        raise ValueError("ops_read_distance_weight must be 'off', 'decay' or "
                         f"'distance_only'; got {mode!r}")
    if not 0 <= discount <= 1:
        raise ValueError("ops_read_neighbour_discount must lie in [0, 1] (1 = off)")
    if mode == "off" and discount == 1.0:
        return None
    out = {"distance_weight": mode,
           "decay_px": _number(settings, "ops_read_distance_decay_px"),
           "interior_px": _number(settings, "ops_read_distance_interior_px"),
           "min_weight": _number(settings, "ops_read_distance_min_weight"),
           "neighbour_discount": discount}
    if not out["decay_px"] > 0 or out["interior_px"] < 0 or not 0 <= out["min_weight"] <= 1:
        raise ValueError("ops_read_distance_decay_px must be above 0, "
                         "ops_read_distance_interior_px 0 or more and "
                         "ops_read_distance_min_weight in [0, 1]")
    return out


def nuclei_with_read(evidence: Mapping[str, np.ndarray], threshold: float) -> int:
    """How many of a field's owned nuclei have at least one read (item 915).

    :param evidence: one field's decode evidence (:func:`field_evidence` or
        its cache): ``strength``, ``id0``, ``owned0`` and ``owned_ids``.
    :param threshold: the read threshold; a spot is a read when its
        strength is above it.
    :returns: the owned nuclei whose territory holds a read: spots above the
        threshold attributed to a nucleus (``id0`` at least 0) that the
        field owns.
    """
    strength = np.asarray(evidence["strength"])
    first = np.asarray(evidence["id0"])
    read = (strength > threshold) & (first >= 0) & np.asarray(evidence["owned0"], bool)
    return int(len(np.intersect1d(np.unique(first[read]), np.asarray(evidence["owned_ids"]))))


def matcher_for(library: Sequence[str], decoys: Sequence[str]):
    """The matcher over guides and decoys, built once per process.

    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :returns: ``(matcher, decoy_set)``.
    """
    from .ops_sbs import LibraryMatcher

    library = frozenset(library)
    decoys = frozenset(decoys)
    key = (hash(library), hash(decoys))
    found = _MATCHERS.get(key)
    if found is None:
        found = (LibraryMatcher(list(library) + list(decoys)), decoys)
        for stale in [k for k in _MATCHERS if k[0] != "masked"]:
            del _MATCHERS[stale]
        _MATCHERS[key] = found
    return found


def _masked_index(library: Sequence[str]) -> dict:
    """The library's masked index, built once per process.

    :param library: the guide barcodes.
    :returns: ``(position, barcode without it) -> barcodes``.
    """
    from .ops_sbs import _library_index

    key = ("masked", hash(frozenset(library)))
    found = _MATCHERS.get(key)
    if found is None:
        found = _library_index(library)[1]
        _MATCHERS[key] = found
    return found


def _call(values: np.ndarray, rule: Mapping[str, Any], library: Sequence[str],
          crosstalk: str):
    """Base calls for reads or pixels under the rule.

    :param values: ``(N, cycles, 4)`` intensities.
    :param rule: from :func:`decode_rule`; ``read_mode`` 'raw' (item 952)
        calls the brightest channel as read, with no library calibration,
        unmixing or normalisation.
    :param library: the guide barcodes.
    :param crosstalk: the unmixing method, ``none`` for none.
    :returns: ``(data, barcodes, margin, quality)``.
    """
    from .ops_sbs import _raw_called_bases, called_bases, library_calibrated_bases

    if len(values) == 0:
        cycles = values.shape[1] if values.ndim == 3 else 0
        return (np.zeros((0, cycles, 4), np.float32), [],
                np.zeros((0, cycles), np.float32), np.zeros(0, np.float32))
    if rule.get("read_mode") == "raw":
        data, codes, margin = _raw_called_bases(values)
    elif rule.get("calibration") and library:
        data, codes, margin = library_calibrated_bases(
            values, library, passes=int(rule["calibration_passes"]),
            min_labels=int(rule["calibration_min_labels"]),
            masked=_masked_index(library))
    elif crosstalk and crosstalk != "none":
        data, codes, margin = called_bases(values, compensate=True,
                                           method=crosstalk, gpu=False)
    else:
        data, codes, margin = called_bases(values, gpu=False)
    worst = np.where(np.isnan(margin), np.inf, margin).min(1)
    quality = np.where(np.isfinite(worst), worst, 0.0)
    return data, codes, margin, quality


def _local_minimum_at(stack: np.ndarray, py: np.ndarray, px: np.ndarray,
                      size: int, ndimage) -> np.ndarray:
    """The ``size`` x ``size`` minimum of every cycle and channel image at the peaks.

    Item 1041: the same numbers as ``ndimage.minimum_filter(image, size)[py, px]``
    for each image, without filtering the whole image: a window of each peak
    is read and its minimum taken. A minimum is exact in any order, so the
    values are identical; an image with NaN, or a peak closer to the border
    than the window reaches, takes the whole-image filter.

    :param stack: ``(cycles, channels, H, W)``.
    :param py: ``(N,)`` peak rows.
    :param px: ``(N,)`` peak columns.
    :param size: the odd window size.
    :param ndimage: ``scipy.ndimage``.
    :returns: ``(N, cycles, channels)`` in the stack's dtype.
    """
    half = size // 2
    height, width = stack.shape[-2:]
    py, px = np.asarray(py), np.asarray(px)
    inside = (not len(py)) or (py.min() >= half and px.min() >= half
                               and py.max() < height - half and px.max() < width - half)
    floating = np.issubdtype(stack.dtype, np.floating)
    if not inside or (floating and not np.isfinite(stack).all()) or size % 2 == 0:
        return np.stack([np.stack([
            ndimage.minimum_filter(stack[i, k], size)[py, px]
            for k in range(stack.shape[1])], 1) for i in range(stack.shape[0])], 1)
    from numpy.lib.stride_tricks import sliding_window_view

    out = np.empty((len(py), stack.shape[0], stack.shape[1]), stack.dtype)
    for i in range(stack.shape[0]):
        for k in range(stack.shape[1]):
            windows = sliding_window_view(stack[i, k], (size, size))
            out[:, i, k] = windows[py - half, px - half].min(axis=(1, 2))
    return out


def field_evidence(stack: np.ndarray, filtered: np.ndarray,
                   score: np.ndarray, peaks: np.ndarray,
                   strength: np.ndarray, kept: Sequence[int],
                   task: Mapping[str, Any], *, edge: int,
                   floor: float, pixel_percentile: Optional[float] = None,
                   features: bool = False) -> Dict[str, np.ndarray]:
    """One field's decode evidence, in the study's field-cache layout.

    :param stack: ``(kept cycles, channels, H, W)``, the aligned stack.
    :param filtered: the stack after the -LoG(1 px) filter, clipped at 0.
    :param score: the cross-cycle read score.
    :param peaks: ``(N, 2)`` every local maximum of the score.
    :param strength: ``(N,)`` each peak's strength, its score over the local
        minimum.
    :param kept: the cycles the stack holds, in order.
    :param task: the field's task -- ``cycles``, ``centroids``, ``areas``,
        ``ids`` and ``owned``. With ``cell_labels`` (``ops_read_assignment``
        'cell_mask', item 1025: the field's SBS cells, label ``i + 1`` for
        object ``i``) the reads and the fallback pixels are given to the
        cell they lie in (:func:`spacr.ops_sbs._cell_mask_boundaries`), the
        pixels outside every cell are dropped, and the evidence carries
        :data:`CELL_MASK_KEYS`: ``read_assignment`` and ``pix_assignment``
        'cell_mask' and ``cell_labels`` with the object ids as values.
    :param edge: how many pixels from the field edge a peak must lie.
    :param floor: the weakest strength kept.
    :param pixel_percentile: keep the pixels above this percentile of the
        score for the pixel fallback; None keeps none.
    :param features: add the per-spot features the false-call model reads
        (item 876): raw intensity and local background per cycle and
        channel, score sharpness, local focus and spot density.
    :returns: ``peaks``, ``strength``, ``values`` (``(N, cycles, 4)``, NaN for
        a cycle not kept), ``gap``, ``id0``, ``id1``, ``owned0``,
        ``owned_ids``, ``owned_areas`` and ``cycles``, plus ``pix_*`` and
        ``f_*`` when asked.
    """
    from scipy import ndimage

    from .ops_sbs import _cell_mask_boundaries, extract_bases, nearest_boundaries

    cycles = list(task["cycles"])
    height, width = filtered.shape[-2:]
    peaks = np.asarray(peaks).reshape(-1, 2)
    strength = np.asarray(strength).reshape(-1)
    keep = (strength > floor) & (peaks[:, 0] >= edge) \
        & (peaks[:, 0] < height - edge) & (peaks[:, 1] >= edge) \
        & (peaks[:, 1] < width - edge)
    peaks, strength = peaks[keep], strength[keep]
    measured = extract_bases(filtered, peaks, window=1)
    n_channels = filtered.shape[1]
    values = np.full((len(peaks), len(cycles), n_channels), np.nan, np.float32)
    for i, cycle in enumerate(kept):
        values[:, cycles.index(cycle)] = measured[:, i]
    centres = np.asarray(task["centroids"], float).reshape(-1, 2)
    areas = np.asarray(task["areas"], float).reshape(-1)
    ids = np.asarray(task["ids"], np.int64).reshape(-1)
    owned = np.asarray(task["owned"], bool).reshape(-1)
    cells = task.get("cell_labels")

    def boundaries(points):
        """The gaps and object indices of points under the field's read assignment."""
        if cells is None:
            return nearest_boundaries(points, centres, areas)
        return _cell_mask_boundaries(points, cells)

    gap64, index = boundaries(peaks)
    gap = gap64.astype(np.float32)
    out: Dict[str, np.ndarray] = {
        "peaks": peaks.astype(np.int32), "strength": strength.astype(np.float32),
        "values": values, "gap": gap,
        "id0": np.where(index[:, 0] >= 0, ids[np.maximum(index[:, 0], 0)]
                        if len(ids) else -1, -1),
        "id1": np.where(index[:, 1] >= 0, ids[np.maximum(index[:, 1], 0)]
                        if len(ids) else -1, -1),
        "owned0": np.where(index[:, 0] >= 0, owned[np.maximum(index[:, 0], 0)]
                           if len(owned) else False, False),
        "owned_ids": ids[owned], "owned_areas": areas[owned],
        "cycles": np.asarray(cycles),
    }
    if features and len(peaks):
        py, px = peaks[:, 0], peaks[:, 1]
        raw = np.stack([stack[i][:, py, px].T for i in range(stack.shape[0])], 1)
        background = _local_minimum_at(stack, py, px, 9, ndimage)
        ring = ndimage.uniform_filter(score, 5)[py, px]
        from .ops_cycles import sbs_mean_target

        target = sbs_mean_target(list(stack[0]))
        lap = ndimage.laplace(target.astype(np.float32)) ** 2
        focus = ndimage.uniform_filter(lap, 64)[py, px]
        from scipy.spatial import cKDTree

        density = np.array([len(x) - 1 for x in cKDTree(peaks).query_ball_point(
            peaks.astype(float), 10.0)])
        full_raw = np.full((len(peaks), len(cycles), raw.shape[-1]), np.nan,
                           np.float32)
        full_bg = np.full_like(full_raw, np.nan)
        for i, cycle in enumerate(kept):
            full_raw[:, cycles.index(cycle)] = raw[:, i]
            full_bg[:, cycles.index(cycle)] = background[:, i]
        out.update({"f_raw": full_raw, "f_bg": full_bg,
                    "f_sharp": (score[py, px] / np.maximum(ring, 1e-6)).astype(np.float32),
                    "f_focus": focus.astype(np.float32),
                    "f_density": density.astype(np.int16)})
    if pixel_percentile is not None and len(centres):
        ys, xs = np.nonzero(score > np.percentile(score, float(pixel_percentile)))
        inside = (ys >= edge) & (ys < height - edge) & (xs >= edge) & (xs < width - edge)
        ys, xs = ys[inside], xs[inside]
        pixel_values = np.full((len(ys), len(cycles), n_channels), np.nan,
                               np.float32)
        for i, cycle in enumerate(kept):
            pixel_values[:, cycles.index(cycle)] = filtered[i][:, ys, xs].T
        pixel_gap, pixel_index = boundaries(np.column_stack([ys, xs]))
        near = pixel_gap[:, 0] <= _PIXEL_REACH
        pixel_index = pixel_index[near]
        out.update({
            "pix_yx": np.column_stack([ys, xs]).astype(np.int16)[near],
            "pix_values": pixel_values[near],
            "pix_score": score[ys, xs].astype(np.float16)[near],
            "pix_gap": pixel_gap[near].astype(np.float16),
            "pix_id0": ids[pixel_index[:, 0]],
            "pix_id1": np.where(pixel_index[:, 1] >= 0,
                                ids[np.maximum(pixel_index[:, 1], 0)], -1),
            "pix_owned0": owned[pixel_index[:, 0]],
        })
    if cells is not None:
        from .image_store import _compact_labels

        lut = np.concatenate([np.zeros(1, np.int64), ids])
        out.update({"read_assignment": np.asarray("cell_mask"),
                    "pix_assignment": np.asarray("cell_mask"),
                    "cell_labels": _compact_labels(lut[np.asarray(cells, np.int64)])})
    return out


def _rule_pixel_ownership(rule: Mapping[str, Any]) -> str:
    """The rule's ``pixel_ownership`` (item 1044).

    :param rule: from :func:`decode_rule`, or a rule built by hand.
    :returns: one of :data:`PIXEL_OWNERSHIP`; a rule without the key, as
        rules recorded before item 1044, keeps ``nearest_nucleus``.
    """
    return str(rule.get("pixel_ownership") or "nearest_nucleus")


#: The read assignment a field's evidence was made under, when it is not
#: the nucleus reach: ``read_assignment`` says how the reads were assigned
#: and ``pix_assignment`` how the fallback pixels were; ``cell_labels`` is
#: the field's cell label image (values the object ids), from which the
#: pixels can be assigned again (item 1044).
CELL_MASK_KEYS = ("read_assignment", "pix_assignment", "cell_labels")


def _cell_mask_pixels(evidence: Mapping[str, Any], cells: np.ndarray) -> Dict[str, Any]:
    """The fallback pixels of a field given to the cell they lie in (item 1044).

    The contract of the cell-mask read assignment: inside a cell the gap
    is -1 and the object is the cell's, with no runner-up; outside a cell
    the gap is 1e30 and there is no object. The label lookup is
    :func:`spacr.ops_sbs._cell_mask_boundaries`, the one the reads take
    (item 1025).

    :param evidence: one field's evidence or field cache with ``pix_yx``
        and ``owned_ids``.
    :param cells: the field's cell labels, values the object ids.
    :returns: a copy with ``pix_gap``, ``pix_id0``, ``pix_id1`` and
        ``pix_owned0`` from the cells and ``pix_assignment`` 'cell_mask'.
    """
    gap, ids = _cell_mask_ids(evidence["pix_yx"], cells)
    out = dict(evidence)
    out.update({"pix_gap": gap, "pix_id0": ids[:, 0], "pix_id1": ids[:, 1],
                "pix_owned0": (ids[:, 0] >= 0) & np.isin(
                    ids[:, 0], np.asarray(evidence["owned_ids"], np.int64)),
                "pix_assignment": "cell_mask"})
    return out


def _cell_mask_ids(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The gaps and object ids of points under stored cell labels (item 1025).

    :param points: ``(N, 2)`` positions ``(y, x)``.
    :param cells: a label image whose values are the object ids.
    :returns: ``(gap, ids)``, both ``(N, 2)``, from
        :func:`spacr.ops_sbs._cell_mask_boundaries`; an id is -1 where the
        point has no object.
    """
    from .ops_sbs import _cell_mask_boundaries

    gap, index = _cell_mask_boundaries(np.asarray(points, float).reshape(-1, 2),
                                       np.asarray(cells))
    return gap, np.where(index >= 0, index + 1, -1).astype(np.int64)


def _cell_mask_evidence(evidence: Mapping[str, Any], cells: np.ndarray) -> Dict[str, Any]:
    """A field's evidence given to its stored SBS cells (item 1025).

    The replay of ``ops_read_assignment`` 'cell_mask' from a field cache
    written under the nucleus reach: the reads (``gap``, ``id0``, ``id1``,
    ``owned0``) and the fallback pixels (:func:`_cell_mask_pixels`) go to
    the cell they lie in by :func:`spacr.ops_sbs._cell_mask_boundaries`,
    the rule of the live decode, and the result carries
    :data:`CELL_MASK_KEYS`, so :func:`field_calls` and the pixel bleed gate
    take it as cell-mask evidence. A cache written under the nucleus reach
    keeps only the fallback pixels within 15 px of a nucleus disc, so a
    replay can vote fewer pixels than a live cell-mask decode; the reads
    above the cache floor are complete.

    :param evidence: one field cache or :func:`field_evidence` result.
    :param cells: the field's ``sbs_cell`` labels, values the object ids,
        in the frame of the cache's peaks.
    :returns: a copy under the cell mask; evidence already under it is
        returned as it is.
    :raises ValueError: when ``cells`` is not a label image the size of the
        field the peaks lie in.
    """
    if _evidence_text(evidence, "read_assignment") == "cell_mask":
        return dict(evidence)
    cells = np.asarray(cells)
    peaks = np.asarray(evidence["peaks"]).reshape(-1, 2)
    if cells.ndim != 2 or (len(peaks) and (peaks.max(0) >= np.asarray(cells.shape)).any()):
        raise ValueError(f"the SBS cell labels {tuple(cells.shape)} do not cover this "
                         "field's reads; give the sbs_cell labels of the same site")
    gap, ids = _cell_mask_ids(peaks, cells)
    out = dict(evidence)
    out.update({"gap": gap.astype(np.float32), "id0": ids[:, 0], "id1": ids[:, 1],
                "owned0": (ids[:, 0] >= 0) & np.isin(
                    ids[:, 0], np.asarray(evidence["owned_ids"], np.int64))})
    if "pix_yx" in evidence:
        out = _cell_mask_pixels(out, cells)
    from .image_store import _compact_labels

    out.update({"read_assignment": np.asarray("cell_mask"),
                "pix_assignment": np.asarray("cell_mask"),
                "cell_labels": _compact_labels(cells)})
    return out


def _evidence_text(evidence: Mapping[str, Any], key: str) -> str:
    """A string entry of the evidence, as a dict or an ``.npz`` holds it; '' when absent."""
    value = evidence.get(key) if hasattr(evidence, "get") else None
    return "" if value is None else str(np.asarray(value))


def _read_assignment_of(rule: Mapping[str, Any]) -> str:
    """The rule's read assignment (``ops_read_assignment``, item 1025).

    :param rule: from :func:`decode_rule`.
    :returns: ``cell_mask`` or ``nucleus_reach``; a rule without the key,
        as every nucleus-reach rule, is ``nucleus_reach``.
    """
    return str(rule.get("read_assignment") or "nucleus_reach")


def _check_read_assignment(evidence: Mapping[str, Any], rule: Mapping[str, Any]) -> None:
    """Refuse evidence made under another read assignment than the rule's (item 1025).

    :param evidence: one field's evidence or field cache.
    :param rule: from :func:`decode_rule`.
    :raises ValueError: naming what is missing: under
        ``ops_read_assignment`` 'cell_mask' evidence whose reads follow the
        nucleus reach (no ``read_assignment`` 'cell_mask'), which must be
        given to the stored SBS cells first (:func:`_cell_mask_evidence`,
        ``decode_cached_well(..., cell_store=...)``); under the nucleus
        reach evidence whose reads follow a cell mask. A rule under
        'nucleus_first' (item 1052) is refused here: only
        :func:`field_calls` takes it, through :func:`_nucleus_first_calls`.
    """
    made = _evidence_text(evidence, "read_assignment") or "nucleus_reach"
    wanted = _read_assignment_of(rule)
    if made == wanted:
        return
    if wanted == "nucleus_first":
        raise ValueError(
            "ops_read_assignment 'nucleus_first' calls a field through field_calls at fixed "
            "thresholds (manual threshold mode, or decode_cached_well on the stored field "
            "caches); the auto and dataset threshold candidates and their controls do not "
            "take it")
    if wanted == "cell_mask":
        raise ValueError(
            "ops_read_assignment 'cell_mask' needs evidence whose reads are given to the "
            "SBS cells, but this field's gap, id0 and owned0 follow the nucleus reach "
            "(no read_assignment 'cell_mask', no cell_labels); replay the cache with the "
            "stored sbs_cell labels (decode_cached_well cell_store=<run>/sbs/fov.ome.zarr) "
            "or decode the well again with ops_sbs_cell_method on")
    raise ValueError(
        f"this field's reads were given to the SBS cells (read_assignment '{made}'); "
        "set ops_read_assignment 'cell_mask' to decode it, or decode the well again "
        "under 'nucleus_reach'")


def _pixel_assignment(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """The evidence with its fallback pixels assigned as its reads are (item 1044).

    :param evidence: one field's evidence or field cache.
    :returns: ``evidence`` itself when its reads follow the nucleus reach
        or its pixels already follow the cell mask; under a cell-mask read
        assignment (``read_assignment`` 'cell_mask') with nucleus-disc
        pixels, the pixels given to their cell from ``cell_labels``
        (:func:`_cell_mask_pixels`).
    :raises ValueError: under a cell-mask read assignment whose pixels
        carry nucleus-disc owners and no ``cell_labels`` to assign them by.
    """
    if (_evidence_text(evidence, "read_assignment") != "cell_mask"
            or _evidence_text(evidence, "pix_assignment") == "cell_mask"):
        return evidence
    if "cell_labels" in evidence:
        return _cell_mask_pixels(evidence, evidence["cell_labels"])
    raise ValueError(
        "ops_pixel_ownership gives each fallback pixel the cell it lies in when the reads "
        "follow the cell mask, but this field's pix_gap, pix_id0 and pix_owned0 are "
        "nucleus-disc owners and it holds no cell_labels; assign the pixels to the stored "
        "cell labels first, or set ops_pixel_ownership 'nearest_nucleus'")


def _pixel_owners(evidence: Mapping[str, Any], rule: Mapping[str, Any], footprint: float,
                  ownership: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Which fallback pixels vote, for which object (``ops_pixel_ownership``, item 1044).

    :param evidence: one field's evidence or field cache with the ``pix_*``
        arrays.
    :param rule: from :func:`decode_rule` with the field's ``neighbours``
        when the read neighbour discount is on.
    :param footprint: the read footprint; the pixels take
        ``ops_pixel_footprint`` in its place (:func:`_pixel_footprint`).
    :param ownership: from :func:`_rule_pixel_ownership`.
    :returns: ``(object per pixel, the voting pixels, the keyword arguments
        of the neighbour discount for :func:`spacr.ops_sbs.vote_pixels`)``.
        ``nearest_nucleus`` is the rule before item 1044: the nearest
        nucleus, owned by the field, within ``ops_pixel_radius``. The other
        modes take the read assignment (:func:`_object_owners`) at the
        read footprint, capped at ``ops_pixel_radius``, and the read
        neighbour discount. They vote the pixels of every object, owned by
        this field or not, so an object's pixel vote and the bleed gate do
        not depend on which field owns its neighbours; the calls keep the
        owned objects only (:func:`_assemble_calls`).
    """
    if ownership == "nearest_nucleus":
        pixel_own = np.asarray(evidence["pix_owned0"]) & (
            np.asarray(evidence["pix_gap"])[:, 0].astype(float)
            <= rule["pixel_radius"])
        return np.asarray(evidence["pix_id0"]), np.flatnonzero(pixel_own), {}
    evidence = _pixel_assignment(evidence)
    gap = np.asarray(evidence["pix_gap"]).astype(float).reshape(-1, 2)
    ids = np.asarray(evidence["pix_id0"])
    _, _, own = _object_owners(gap, ids >= 0, rule, _pixel_footprint(rule, footprint))
    own &= gap[:, 0] <= rule["pixel_radius"]
    discount = float((rule.get("read_weight") or {}).get("neighbour_discount", 1.0))
    weighting = ({"neighbours": rule.get("neighbours") or {}, "neighbour_discount": discount}
                 if discount != 1.0 else {})
    return ids, np.flatnonzero(own), weighting


def _read_tops(tally: Optional[Mapping[str, Any]], first: np.ndarray, codes: Sequence[str],
               foreign: np.ndarray, matcher) -> Dict[int, frozenset]:
    """Each object's top read guides, for the pixel bleed gate (item 1044).

    :param tally: the vote evidence the read vote filled, for the objects
        the field owns.
    :param first: the nearest object of every read above the threshold.
    :param codes: the called barcode of every such read.
    :param foreign: the reads the read assignment gives to an object
        another field owns; their exact library matches name its guides.
    :param matcher: from :func:`matcher_for`.
    :returns: object -> the guides with its largest read vote, ties kept,
        for every object whose reads named a guide.
    """
    from collections import Counter

    tops: Dict[int, frozenset] = {}
    for owner, votes in (tally or {}).get("votes", {}).items():
        if votes:
            best = max(votes.values())
            tops[int(owner)] = frozenset(g for g, v in votes.items() if v == best)
    counted: Dict[int, Counter] = {}
    for i in np.flatnonzero(foreign).tolist():
        guide, _ = matcher.match(codes[i], 0)
        if guide is not None:
            counted.setdefault(int(first[i]), Counter())[guide] += 1
    for owner, votes in counted.items():
        if owner not in tops:
            best = max(votes.values())
            tops[owner] = frozenset(g for g, v in votes.items() if v == best)
    return tops


#: Why the pixel bleed gate refused a pixel vote (item 1044): the guide
#: is a top read guide of a neighbour (``bleed_read``), a neighbour with no
#: read guide holds as many pixels of it or more (``bleed_pixel``), or the
#: object's pixels within ``ops_pixel_radius`` -- the vote before item
#: 1044 -- do not call the same guide (``radius_disagrees``).
BLEED_REASONS = ("bleed_read", "bleed_pixel", "radius_disagrees")


def _bleed_refusals(pixel: Mapping[int, dict], votes: Mapping[str, Any],
                    called=()) -> Dict[int, str]:
    """The pixel votes one amplicon cluster would give to a second object (item 1044).

    A pixel vote of an object its reads did not call is refused when its
    guide is a top read guide (:func:`_read_tops`) of a neighbour within
    the read neighbour reach, or when such a neighbour, whose reads named
    no guide, holds a pixel vote of the same guide with as many pixels or
    more (a tie refuses both). Sisters (``rule['sisters']``, object -> its
    sisters by geometry and mitotic labels) are exempt; a shared guide
    alone never makes two objects sisters, and the decode holds no
    mitotic labels, so no pair is exempt unless the rule supplies it. A
    vote is also refused when the object's pixels within
    ``ops_pixel_radius`` (the vote of ``nearest_nucleus``, carried as the
    hit's ``radius_guide``) call no guide or another one: the narrower
    footprint can leave a few misread pixels in the majority, and the gate
    never makes a pixel call the rule before item 1044 would not make. The
    gate reads no single-read floor or threshold choice, so the auto
    candidates and the decode refuse the same votes.

    :param pixel: object -> pixel vote, after the read gate.
    :param votes: from :func:`_field_votes`, for ``pixel_neighbours``,
        ``read_tops`` and ``sisters``.
    :param called: the objects the reads called; their pixel votes are the
        support of a read call and stay.
    :returns: object -> reason (:data:`BLEED_REASONS`).
    """
    neighbours = votes.get("pixel_neighbours") or {}
    tops = votes.get("read_tops") or {}
    sisters = votes.get("sisters") or {}
    refused: Dict[int, str] = {}
    for owner, hit in pixel.items():
        if owner in called:
            continue
        guide = hit["guide"]
        if "radius_guide" in hit and hit["radius_guide"] != guide:
            refused[owner] = "radius_disagrees"
            continue
        for other in sorted(neighbours.get(owner, ())):
            if other in sisters.get(owner, ()):
                continue
            if guide in tops.get(other, ()):
                refused[owner] = "bleed_read"
                break
            rival = pixel.get(other)
            if (rival is not None and not tops.get(other) and rival["guide"] == guide
                    and rival["n_pixels"] >= hit["n_pixels"]):
                refused[owner] = "bleed_pixel"
                break
    return refused


def _rule_with_pixel_neighbours(rule: Mapping[str, Any], evidence: Mapping[str, Any],
                                footprint: float) -> Mapping[str, Any]:
    """The rule with the field's neighbours the pixel bleed gate reads (item 1044).

    :param rule: from :func:`decode_rule`; the engine adds ``neighbours``.
    :param evidence: the field's evidence or its field cache.
    :param footprint: the read footprint (:func:`_neighbour_reach`).
    :returns: ``rule`` itself unless the gate is on (``pixels``, or
        ``pixels_given`` for a pass that reuses another pass's pixel votes,
        and ``pixel_ownership`` 'read_rule') and it has no ``neighbours``; then
        a copy with :func:`cached_neighbours` at the read neighbour reach.
    :raises ValueError: when the gate needs the neighbours and the field
        cache holds no nucleus positions.
    """
    if (not (rule.get("pixels") or rule.get("pixels_given"))
            or _rule_pixel_ownership(rule) != "read_rule"
            or rule.get("neighbours") is not None):
        return rule
    missing = [key for key in _neighbour_keys(evidence) if key not in evidence]
    if missing:
        raise ValueError("ops_pixel_ownership 'read_rule' needs the field's nucleus "
                         "positions for the pixel bleed gate; this field cache has no "
                         f"{', '.join(missing)}; decode the field again, or replay it "
                         "with ops_pixel_ownership 'read_reach'")
    return {**rule, "neighbours": cached_neighbours(evidence, _neighbour_reach(rule, footprint))}


def _object_owners(gap: np.ndarray, owned_first: np.ndarray, rule: Mapping[str, Any],
                   footprint: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which rows -- reads or fallback pixels -- the read assignment gives to their object.

    The one assignment rule of the decode (item 1044): a row is near its
    nearest object when its gap to that object's boundary is at most
    ``footprint``; it is ambiguous when the runner-up object's boundary is
    less than ``rule['tie']`` farther; and it is owned when it is near, the
    field owns the object, and it is not ambiguous unless the territory rule
    (``ops_assign_by_cell_territory``) gives ties to the nearest. Under a
    cell mask the gaps are the cell's (-1 inside, 1e30 outside, no
    runner-up), so the row takes its cell label. The reads always pass
    through here; the fallback pixels do under ``ops_pixel_ownership``
    ``read_rule`` and ``read_reach``.

    :param gap: ``(N, 2)`` gaps to the nearest and the runner-up boundary.
    :param owned_first: ``(N,)`` whether the field owns the nearest object.
    :param rule: from :func:`decode_rule`, read for ``tie`` and ``territory``.
    :param footprint: how far beyond a boundary a row may lie, in pixels.
    :returns: ``(near, ambiguous, own)``, three ``(N,)`` boolean arrays.
    """
    near = gap[:, 0] <= footprint
    ambiguous = near & ((gap[:, 1] - gap[:, 0]) < rule["tie"])
    own = near & owned_first & (bool(rule["territory"]) | ~ambiguous)
    return near, ambiguous, own


def field_calls(evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any],
                library: Sequence[str], matcher, decoys, *,
                threshold: float, footprint: float, max_distance: int,
                min_fraction: float, single_read_quality: float,
                crosstalk: str = "argmax_median",
                evidence_rows: bool = False) -> Dict[str, Any]:
    """Every call of one field under the rule, decoys included.

    :param evidence: from :func:`field_evidence`, or a field-cache ``.npz``
        of the decode-yield study opened with :func:`numpy.load`.
    :param rule: from :func:`decode_rule`.
    :param library: the guide barcodes.
    :param matcher: from :func:`matcher_for`.
    :param decoys: the decoy set from :func:`matcher_for`.
    :param threshold: the read threshold on the spot strength.
    :param footprint: how far beyond a boundary a read may lie, in pixels.
    :param max_distance: the mismatches a read may carry; the runner-up
        rule raises 0 to 1.
    :param min_fraction: the share of the votes the guide must hold.
    :param single_read_quality: the quality a one-read guide needs.
    :param crosstalk: the unmixing method.
    :param evidence_rows: also return ``evidence``, one row per call --
        decoys included -- with the read and pixel support behind it
        (:data:`EVIDENCE_COLUMNS`), which the genotype confidence reads.
    :returns: ``calls`` -- ``{object: {...}}`` with ``guide``, ``decoy``,
        ``source`` (``primary`` or ``pixel_fallback``), ``n_reads``,
        ``n_agreeing``, ``fraction``, ``quality``, ``barcode`` and
        ``n_pixels`` -- for owned objects only; ``nuclei``, the objects this
        field owns; ``assigned_real`` and ``assigned_decoy``, and of
        them the pixel-fallback calls, ``assigned_pixel_real`` and
        ``assigned_pixel_decoy``; ``split``, the reads in this field of nuclei
        other fields own (:func:`split_read_concordance`); and read counts
        for the well report. Under ``ops_read_assignment`` 'nucleus_first'
        (item 1052) the calls are :func:`_nucleus_first_calls`.
    """
    if _read_assignment_of(rule) == "nucleus_first":
        return _nucleus_first_calls(
            evidence, rule, library, matcher, decoys, threshold=threshold,
            footprint=footprint, max_distance=max_distance, min_fraction=min_fraction,
            single_read_quality=single_read_quality, crosstalk=crosstalk,
            evidence_rows=evidence_rows)
    votes = _field_votes(evidence, rule, library, matcher, threshold=threshold,
                         footprint=footprint, max_distance=max_distance,
                         min_fraction=min_fraction,
                         single_read_quality=single_read_quality,
                         pixel_qualities=(rule["pixel_quality"],),
                         crosstalk=crosstalk, evidence_rows=evidence_rows)
    primary = votes["primary"]
    refused: Dict[int, str] = {}
    pixel = (_fallback_pixels(votes["pixel"][rule["pixel_quality"]], votes, primary,
                              refused) if rule["pixels"] else {})
    out = _assemble_calls(primary, pixel, votes, decoys)
    if votes.get("bleed_gate"):
        out["pixel_refused"] = _refusal_counts(refused, votes["owned"])
    if rule.get("read_stage"):
        out["nuclei_with_read"] = nuclei_with_read(evidence, threshold)
    if evidence_rows:
        out["evidence"] = _evidence_rows(primary, pixel, votes["tallies"],
                                         votes["owned"], decoys,
                                         votes["objects"], votes["ambiguous"])
    return out


def _nucleus_first_cells(evidence: Mapping[str, Any]) -> np.ndarray:
    """The SBS cell labels a 'nucleus_first' field falls back on (item 1052).

    :param evidence: the evidence or field cache of one field under the nucleus
        reach, with ``cell_labels`` (values the object ids): the live decode
        adds them, a replay reads them from the stored ``sbs_cell`` labels
        (:func:`_nucleus_first_evidence`).
    :returns: the label image.
    :raises ValueError: for evidence whose reads were given to the cells
        (``read_assignment`` 'cell_mask'), or evidence without ``cell_labels``.
    """
    if _evidence_text(evidence, "read_assignment") == "cell_mask":
        raise ValueError(
            "ops_read_assignment 'nucleus_first' keeps the nucleus-reach calls, but this "
            "reads of this field were given to the SBS cells (read_assignment 'cell_mask'); replay "
            "a field cache written under 'nucleus_reach' with the stored sbs_cell labels")
    if "cell_labels" not in evidence:
        raise ValueError(
            "ops_read_assignment 'nucleus_first' falls back on the SBS cells, but this "
            "evidence of this field holds no cell_labels; replay the cache with the stored sbs_cell "
            "labels (decode_cached_well cell_store=<run>/sbs/fov.ome.zarr) or decode the well "
            "with ops_sbs_cell_method on")
    return np.asarray(evidence["cell_labels"])


def _own_gaps(evidence: Mapping[str, Any], points: np.ndarray, gap: np.ndarray,
              ids: np.ndarray, owner: int) -> np.ndarray:
    """The gap from each point to the boundary of its own nucleus disc (item 1052).

    The spot position of the bleed curve: the nearest or
    runner-up gap when the owner is one of its two nearest nuclei, as the
    nucleus reach measured it; otherwise the disc gap from the
    nucleus positions (:data:`CACHE_NUCLEUS_KEYS`).

    :param evidence: the evidence of the field, read for the nucleus positions
        only when a point lies nearer two other nuclei.
    :param points: ``(N, 2)`` positions.
    :param gap: ``(N, 2)`` nucleus-reach gaps.
    :param ids: ``(N, 2)`` the nucleus-reach objects.
    :param owner: the object.
    :returns: ``(N,)`` gaps; ``inf`` when the owner has no nucleus position.
    :raises ValueError: naming the missing keys when they are needed.
    """
    out = np.where(ids[:, 0] == owner, gap[:, 0],
                   np.where(ids[:, 1] == owner, gap[:, 1], np.nan)).astype(float)
    missing = np.isnan(out)
    if missing.any():
        absent = [key for key in CACHE_NUCLEUS_KEYS if key not in evidence]
        if absent:
            raise ValueError("the 'nucleus_first' bleed gate needs the nucleus "
                             f"positions; this field holds no {', '.join(absent)}; decode "
                             "the field again with the field cache on")
        where = np.flatnonzero(np.asarray(evidence["nuc_ids"]) == owner)
        if not len(where):
            out[missing] = np.inf
        else:
            centre = np.asarray(evidence["nuc_centroids"], float)[where[0]]
            radius = np.sqrt(max(float(np.asarray(evidence["nuc_areas"], float)[where[0]]), 0.0)
                             / np.pi)
            out[missing] = np.hypot(*(np.asarray(points, float)[missing] - centre).T) - radius
    return out


def _bleed_probabilities(evidence: Mapping[str, Any], points: np.ndarray, gap: np.ndarray,
                         ids: np.ndarray, owner: int, guide: str,
                         carriers: Mapping[int, frozenset], sisters: Mapping[int, Any],
                         model: Mapping[str, Any]) -> np.ndarray:
    """The share of each supporting read the bleed curve gives a neighbour (item 1052).

    The bleed model of the 'nucleus_first' cell-mask read fallback; the
    pixel calls keep the 1044b gate (item 1052b). A point of ``owner`` that names ``guide`` is bleed with the
    excess rate of the curve (:func:`spacr.ops_decode_qc_panels.bleed_rates`) at
    its gap to the boundary of a neighbour carrying that guide -- one of its
    two nearest nuclei other than the owner, not a sister, whose top read
    guides (or, without reads, its pixel vote) hold the guide -- and at its
    own spot position (:func:`_own_gaps`); the larger of the two neighbours
    when both carry it, and 0 when neither does.

    :param evidence: the nucleus-reach evidence of the field.
    :param points: ``(N, 2)`` the supporting points.
    :param gap: ``(N, 2)`` their nucleus-reach gaps.
    :param ids: ``(N, 2)`` their two nearest nuclei.
    :param owner: the object called.
    :param guide: the guide called.
    :param carriers: object -> the guides it carries.
    :param sisters: object -> its sisters, which are never a source of bleed.
    :param model: the bleed model.
    :returns: ``(N,)`` probabilities.
    """
    from .ops_decode_qc_panels import bleed_rates

    gap = np.asarray(gap, float).reshape(-1, 2)
    ids = np.asarray(ids, np.int64).reshape(-1, 2)
    out = np.zeros(len(gap))
    if not len(gap):
        return out
    exempt = set(sisters.get(owner, ()))
    slots = []
    for slot in (0, 1):
        carries = np.array([int(n) >= 0 and int(n) != owner and int(n) not in exempt
                            and guide in carriers.get(int(n), ()) for n in ids[:, slot]], bool)
        if carries.any():
            slots.append((slot, carries))
    if not slots:
        return out
    own = _own_gaps(evidence, points, gap, ids, owner)
    for slot, carries in slots:
        out = np.maximum(out, np.where(carries, bleed_rates(model, gap[:, slot], own), 0.0))
    return out


def _bleed_gate(probabilities: np.ndarray, max_share: float, least: float) -> Optional[str]:
    """The 'nucleus_first' bleed gate on the supporting reads of a cell-mask call (item 1052).

    :param probabilities: from :func:`_bleed_probabilities`.
    :param max_share: ``ops_nucleus_first_max_bleed``.
    :param least: the least support left once the bleed is taken off,
        ``ops_nucleus_first_min_reads``.
    :returns: None to keep the call, or the reason
        (:data:`NUCLEUS_FIRST_REASONS`).
    """
    n = len(probabilities)
    bleed = float(np.sum(probabilities))
    if n == 0 or bleed / n > max_share + _SHARE_TOLERANCE:
        return "bleed_model"
    if n - bleed < least - _SHARE_TOLERANCE:
        return "min_support"
    return None


def _nucleus_first_carriers(votes: Mapping[str, Any], pixel: Mapping[int, dict]
                            ) -> Dict[int, frozenset]:
    """The guides each object carries, as the bleed gate reads them (item 1052).

    :param votes: the nucleus-reach votes (:func:`_field_votes`), read for
        ``read_tops``.
    :param pixel: the nucleus-reach pixel votes before any gate.
    :returns: object -> its top read guides; an object without read guides
        carries the guide of its pixel vote, as the 1044b pixel bleed rule
        reads a neighbour.
    """
    carriers = {int(k): frozenset(v) for k, v in (votes.get("read_tops") or {}).items()}
    for owner, hit in pixel.items():
        if not carriers.get(int(owner)):
            carriers[int(owner)] = frozenset([hit["guide"]])
    return carriers


def _nucleus_first_calls(evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any],
                         library: Sequence[str], matcher, decoys, *, threshold: float,
                         footprint: float, max_distance: int, min_fraction: float,
                         single_read_quality: float, crosstalk: str,
                         evidence_rows: bool) -> Dict[str, Any]:
    """Every call of one field under ``ops_read_assignment`` 'nucleus_first' (item 1052).

    1. The nucleus reach votes the field exactly as under 'nucleus_reach';
       every read call it makes stands, and no cytoplasm read vetoes or
       dilutes it.
    2. The pixel fallback is the one of 'nucleus_reach', untouched: the
       1044b path with its own gate (``bleed_read``, ``bleed_pixel``,
       ``radius_disagrees``), so every pixel call is the pixel call of
       'nucleus_reach' (item 1052b).
    3. Only an owned object left with neither a read call nor a pixel call
       is voted again on the reads of its SBS cell
       (:func:`_cell_mask_evidence`, the item-1025 cell-mask vote,
       neighbours the cells that touch). That call is kept when the bleed
       gate passes it (:func:`_bleed_gate`): the bleed curve
       (``rule['bleed_model']``) explains at most ``max_share`` of its
       agreeing reads as bleed from a neighbour, and at least ``min_reads``
       reads are left once the bleed is taken off.

    :param evidence: nucleus-reach evidence with ``cell_labels``
        (:func:`_nucleus_first_cells`).
    :param rule: from :func:`decode_rule` under 'nucleus_first', with
        ``bleed_model``.
    :returns: as :func:`field_calls`, the calls of 'nucleus_reach' plus the
        cell-mask calls, which alone carry ``assignment`` 'cell_mask'; the
        result adds ``assigned_cell_mask_real``, ``assigned_cell_mask_decoy``
        and ``nucleus_first`` (the cell-mask candidates, calls and refusals
        by :data:`NUCLEUS_FIRST_REASONS`); ``pixel_refused`` is the 1044b
        count of 'nucleus_reach'.
    :raises ValueError: without ``bleed_model`` (``ops_nucleus_first_bleed_curve``
        'fit' is fitted by :func:`decode_cached_well`), and from
        :func:`_nucleus_first_cells`.
    """
    model = rule.get("bleed_model")
    if not model:
        raise ValueError(
            "ops_read_assignment 'nucleus_first' needs a bleed curve: "
            "ops_nucleus_first_bleed_curve 'fit' is fitted per run when the well is "
            "re-decoded from its field caches (decode_cached_well); a live decode needs "
            "'arm2' or the path of a stored curve")
    cells = _nucleus_first_cells(evidence)
    gate = rule["bleed_gate"]
    base = {key: value for key, value in rule.items()
            if key not in ("read_assignment", "bleed_gate", "bleed_model")}
    kwargs = dict(threshold=threshold, footprint=footprint, max_distance=max_distance,
                  min_fraction=min_fraction, single_read_quality=single_read_quality,
                  crosstalk=crosstalk, evidence_rows=evidence_rows)
    quality = rule["pixel_quality"]
    votes = _field_votes(evidence, {**base, "nucleus_first_stage": True}, library, matcher,
                         pixel_qualities=(quality,), **kwargs)
    primary, owned = votes["primary"], votes["owned"]
    reduced = {key: value for key, value in evidence.items() if not key.startswith("pix_")}
    cell_rule = {key: value for key, value in base.items() if key != "neighbours"}
    cell_rule.update({"read_assignment": "cell_mask", "pixels": False, "pixels_given": False,
                      "nucleus_first_stage": True})
    cell_votes = _field_votes(_cell_mask_evidence(reduced, cells), cell_rule, library,
                              matcher, pixel_qualities=(quality,), **kwargs)
    raw_pixel = votes["pixel"][quality] if rule["pixels"] else {}
    pixel_refused: Dict[int, str] = {}
    pixel = (_fallback_pixels(raw_pixel, votes, primary, pixel_refused)
             if rule["pixels"] else {})
    carriers = _nucleus_first_carriers(votes, raw_pixel)
    sisters = votes.get("sisters") or {}
    gaps = np.asarray(evidence["gap"], float).reshape(-1, 2)
    nearest = np.column_stack([np.asarray(evidence["id0"]), np.asarray(evidence["id1"])])
    peaks = np.asarray(evidence["peaks"], float).reshape(-1, 2)

    read_refused: Dict[int, str] = {}
    cell_calls: Dict[int, dict] = {}
    candidates = [o for o in cell_votes["primary"]
                  if o not in primary and o not in pixel and o in owned]
    if candidates:
        rows = cell_votes["rows"]
        where = cell_votes["selected"][rows]
        objects = np.asarray(cell_votes["objects"])
        codes = cell_votes["codes"]
        for owner in candidates:
            hit = cell_votes["primary"][owner]
            mine = np.flatnonzero(objects == owner)
            agree = np.array([matcher.match(codes[rows[i]], cell_votes["distance"])[0]
                              == hit["guide"] for i in mine], bool)
            points = where[mine[agree]] if len(mine) else np.zeros(0, np.int64)
            reason = _bleed_gate(_bleed_probabilities(
                evidence, peaks[points], gaps[points], nearest[points], int(owner),
                hit["guide"], carriers, sisters, model), gate["max_share"], gate["min_reads"])
            if reason is None:
                cell_calls[owner] = hit
            else:
                read_refused[owner] = reason

    reads_per = votes["reads_per"].copy()
    for owner in cell_calls:
        reads_per[owner] = cell_votes["reads_per"].get(owner, 0)
    out = _assemble_calls({**primary, **cell_calls}, pixel,
                          {**votes, "reads_per": reads_per}, decoys)
    cell_real = cell_decoy = 0
    for owner, call in out["calls"].items():
        if owner in cell_calls:
            call["assignment"] = "cell_mask"
            cell_real += int(not call["decoy"])
            cell_decoy += int(call["decoy"])
    out["assigned_cell_mask_real"] = cell_real
    out["assigned_cell_mask_decoy"] = cell_decoy
    counted = [r for o, r in read_refused.items() if o in owned]
    out["nucleus_first"] = {
        "cell_mask_candidates": int(len(candidates)),
        "cell_mask_called": int(sum(1 for o in cell_calls if o in owned)),
        "read_refused": {reason: counted.count(reason) for reason in NUCLEUS_FIRST_REASONS},
        "reads_cell_mask_owned": int(cell_votes["reads_owned"])}
    if votes.get("bleed_gate"):
        out["pixel_refused"] = _refusal_counts(pixel_refused, votes["owned"])
    if rule.get("read_stage"):
        out["nuclei_with_read"] = nuclei_with_read(evidence, threshold)
    if evidence_rows:
        cell_rows = _evidence_rows(cell_calls, {}, cell_votes["tallies"], owned, decoys,
                                   cell_votes["objects"], cell_votes["ambiguous"])
        for row in cell_rows:
            row["assignment"] = "cell_mask"
        out["evidence"] = _evidence_rows(primary, pixel, votes["tallies"], owned, decoys,
                                         votes["objects"], votes["ambiguous"]) + cell_rows
    return out


def _field_votes(evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any],
                 library: Sequence[str], matcher, *, threshold: float,
                 footprint: float, max_distance: int, min_fraction: float,
                 single_read_quality: float, pixel_qualities: Sequence[float],
                 crosstalk: str, evidence_rows: bool) -> Dict[str, Any]:
    """The read vote and the pixel votes of one field, before any call is kept.

    :param evidence: as for :func:`field_calls`.
    :param rule: from :func:`decode_rule`.
    :param library: the guide barcodes.
    :param matcher: from :func:`matcher_for`.
    :param threshold: the read threshold on the spot strength.
    :param footprint: how far beyond a boundary a read may lie, in pixels.
    :param max_distance: the mismatches a read may carry.
    :param min_fraction: the share of the votes the guide must hold.
    :param single_read_quality: the quality a one-read guide needs.
    :param pixel_qualities: the pixel qualities to vote the fallback at;
        ignored when the rule has no pixel fallback.
    :param crosstalk: the unmixing method.
    :param evidence_rows: keep the vote tallies for the evidence rows.
    :returns: ``primary`` (:func:`spacr.ops_sbs.vote_mapped_reads`),
        ``pixel`` (pixel quality -> :func:`spacr.ops_sbs.vote_pixels`),
        ``tallies``, ``owned``, ``objects`` and ``ambiguous`` (of the owned
        reads), ``reads_per``, ``named`` (object -> the guides its reads
        voted for, top first; empty with the gate off), ``pixel_gate`` (the
        rule's ``pixel_read_gate``, kept on when another pass votes the
        pixels), ``short_share`` (the objects whose read call the rule's
        ``min_top_share`` refused, :func:`_top_share_kept`), ``split``
        (:func:`_split_reads`) and the read counts of the well report.
        A rule with ``nucleus_first_stage`` (item 1052,
        :func:`_nucleus_first_calls`) also gets ``read_tops``, ``sisters``
        and, per read above the threshold, ``codes``, ``selected`` (their
        rows in the evidence), ``rows`` (the owned ones among them) and
        ``distance``; with pixels ``pixel_codes``, ``pixel_quality``,
        ``pixel_ids`` and ``pixel_chosen``.
    """
    from collections import Counter

    from .ops_sbs import vote_mapped_reads, vote_pixels

    _check_read_assignment(evidence, rule)
    rule = _rule_with_pixel_neighbours(
        _rule_with_neighbours(rule, evidence, footprint), evidence, footprint)
    strength = np.asarray(evidence["strength"])
    selected = strength > threshold
    values = np.asarray(evidence["values"])[selected]
    data, codes, margin, quality = _call(values, rule, library, crosstalk)
    gap = np.asarray(evidence["gap"])[selected]
    first = np.asarray(evidence["id0"])[selected]
    owned_first = np.asarray(evidence["owned0"])[selected]
    near, ambiguous, own = _object_owners(gap, owned_first, rule, footprint)
    rows = np.flatnonzero(own)
    objects = first[rows]
    distance = max(int(max_distance), 1) if rule["runner_up"] else int(max_distance)
    tallies: Optional[dict] = {} if evidence_rows else None
    gate = bool(rule.get("pixel_read_gate"))
    share = float(rule.get("min_top_share") or 0.0)
    ownership = _rule_pixel_ownership(rule)
    bleed = (bool(rule["pixels"]) or bool(rule.get("pixels_given"))) \
        and ownership == "read_rule"
    stage = bool(rule.get("nucleus_first_stage"))
    tally = tallies if tallies is not None else (
        {} if gate or share > 0 or bleed or stage else None)
    weighting = _vote_weighting(rule, gap[rows, 0])
    primary = vote_mapped_reads(
        objects, [codes[i] for i in rows], quality[rows], data[rows],
        margin[rows], matcher, max_distance=distance,
        runner_up_margin=(rule["runner_up_margin"] if rule["runner_up"] else None),
        quality_weighted=bool(rule["quality_weighted"]),
        weight_floor=float(rule["weight_floor"]), min_fraction=min_fraction,
        single_read_quality=single_read_quality,
        impute_missing=bool(rule["impute"]),
        strength=strength[selected][rows], evidence=tally, **weighting)
    primary, short_share = _top_share_kept(primary, tally, share)
    named = _read_guides(tally) if gate else {}
    pixel: Dict[float, Dict[int, dict]] = {q: {} for q in pixel_qualities}
    kept: Dict[str, Any] = {}
    if rule["pixels"] and "pix_values" in evidence and len(evidence["pix_values"]):
        _, pixel_codes, _, pixel_quality = _call(
            np.asarray(evidence["pix_values"], np.float32),
            {"calibration": False}, library, crosstalk)
        placed = evidence if ownership == "nearest_nucleus" else _pixel_assignment(evidence)
        pixel_ids, chosen, discount = _pixel_owners(placed, rule, footprint, ownership)
        if stage:
            kept.update({"pixel_codes": pixel_codes, "pixel_quality": pixel_quality,
                         "pixel_ids": pixel_ids, "pixel_chosen": chosen})
        for q in pixel_qualities:
            pixel[q] = vote_pixels(
                pixel_ids[chosen],
                [pixel_codes[i] for i in chosen], pixel_quality[chosen], matcher,
                min_quality=q, min_pixels=int(rule["pixel_min_pixels"]),
                min_fraction=rule["pixel_min_fraction"], **discount)
        if bleed:
            wide_ids, wide, _ = _pixel_owners(placed, rule, footprint, "nearest_nucleus")
            for q in pixel_qualities:
                radius = vote_pixels(
                    wide_ids[wide], [pixel_codes[i] for i in wide], pixel_quality[wide],
                    matcher, min_quality=q, min_pixels=int(rule["pixel_min_pixels"]),
                    min_fraction=rule["pixel_min_fraction"])
                for owner, hit in pixel[q].items():
                    hit["radius_guide"] = radius[owner]["guide"] if owner in radius else None
    inside = np.flatnonzero(~owned_first & (gap[:, 0] < 0) & (first >= 0))
    extra = {}
    if bleed:
        extra = {"bleed_gate": True, "pixel_neighbours": rule["neighbours"],
                 "read_tops": _read_tops(tally, first, codes, near & ~owned_first
                                         & (bool(rule["territory"]) | ~ambiguous)
                                         & (first >= 0), matcher),
                 "sisters": rule.get("sisters") or {}}
    if stage:
        extra.update(kept)
        extra.update({"codes": codes, "selected": np.flatnonzero(selected), "rows": rows,
                      "distance": distance, "sisters": rule.get("sisters") or {}})
        if "read_tops" not in extra:
            extra["read_tops"] = _read_tops(tally, first, codes, near & ~owned_first
                                            & (bool(rule["territory"]) | ~ambiguous)
                                            & (first >= 0), matcher)
    return {**extra, "primary": primary, "pixel": pixel, "tallies": tallies,
            "named": named, "pixel_gate": gate, "short_share": short_share,
            "split": _split_reads(first[inside], [codes[i] for i in inside], matcher),
            "owned": set(np.asarray(evidence["owned_ids"]).tolist()),
            "objects": objects, "ambiguous": ambiguous[rows],
            "reads_per": Counter(objects.tolist()),
            "reads": int(selected.sum()), "reads_owned": int(len(rows)),
            "reads_near": int(near.sum()),
            "reads_ambiguous": int(ambiguous.sum())}


def _refusal_counts(refused: Mapping[int, str], owned) -> Dict[str, int]:
    """How many pixel votes of owned objects the bleed gate refused, by reason (item 1044).

    :param refused: object -> reason, from :func:`_fallback_pixels`.
    :param owned: the objects the field owns.
    :returns: reason -> count, every reason of :data:`BLEED_REASONS`.
    """
    values = [reason for owner, reason in refused.items() if owner in owned]
    return {reason: values.count(reason) for reason in BLEED_REASONS}


def _read_guides(tally: Mapping[str, Any]) -> Dict[int, tuple]:
    """Each object's read-named guides, top vote first.

    :param tally: the ``evidence`` dict
        :func:`spacr.ops_sbs.vote_mapped_reads` filled.
    :returns: object -> the guides its matched reads voted for, for objects
        assigned or not.
    """
    return {owner: tuple(guide for guide, _ in sorted(votes.items(), key=lambda kv: -kv[1]))
            for owner, votes in tally.get("votes", {}).items()}


#: How far below ``ops_min_top_share`` a top share may fall from float
#: rounding and still pass, so a share of exactly the cutoff is kept.
_SHARE_TOLERANCE = 1e-12


def _top_share_kept(primary: Mapping[int, dict], tally: Optional[Mapping[str, Any]],
                    share: float) -> Tuple[Mapping[int, dict], frozenset]:
    """The read calls whose top guide holds at least ``share`` of the votes.

    The share is the vote of the top guide over the vote total of the
    nucleus, after the read weighting (``fraction`` of the read call), so a nucleus whose
    reads named one guide always passes. A nucleus below it stays uncalled:
    the pixel fallback does not call it either (:func:`_fallback_pixels`),
    whatever the single-read quality floor did to its read call, so the
    decode and the auto candidates refuse the same nuclei.

    :param primary: object -> read call (:func:`spacr.ops_sbs.vote_mapped_reads`).
    :param tally: the vote evidence the read vote filled, read for
        ``votes``; None only when ``share`` is 0.
    :param share: ``ops_min_top_share``; 0 keeps every call.
    :returns: ``(kept calls, objects below the share)``.
    """
    if share <= 0:
        return primary, frozenset()
    floor = share - _SHARE_TOLERANCE
    short = frozenset(owner for owner, votes in tally.get("votes", {}).items()
                      if max(votes.values()) / sum(votes.values()) < floor)
    return {owner: hit for owner, hit in primary.items() if owner not in short}, short


def _split_reads(objects: np.ndarray, codes: Sequence[str], matcher) -> Dict[int, tuple]:
    """The independent reads a field holds of nuclei another field owns.

    Where two sequencing fields overlap, a nucleus is read in both; the
    field that does not own it holds a second, independent read of its
    guide. Counted here are the reads strictly inside such a nucleus (gap
    below 0) whose called barcode is a library or decoy barcode exactly, with
    no decoder weighting.

    :param objects: the nucleus of each such read.
    :param codes: the called barcode of each read.
    :param matcher: from :func:`matcher_for`; its barcodes are the exact set.
    :returns: object -> ``(reads, guides named, first guide)``.
    """
    known = matcher.known
    found: Dict[int, list] = {}
    for owner, code in zip(np.asarray(objects).tolist(), codes):
        if code in known:
            found.setdefault(int(owner), []).append(code)
    return {owner: (len(reads), len(set(reads)), reads[0])
            for owner, reads in found.items()}


def _reads_allow(named: Sequence[str], guide: str) -> bool:
    """Whether an object's reads leave room for a pixel-fallback guide.

    :param named: the guides the object's reads voted for, top first.
    :param guide: the pixel vote's guide.
    :returns: True when the reads named no guide, or named this guide
        alone; False when they named another guide or two or more guides.
    """
    return not named or (len(named) == 1 and named[0] == guide)


def _fallback_pixels(pixel: Mapping[int, dict], votes: Mapping[str, Any],
                     called=(), refused: Optional[Dict[int, str]] = None
                     ) -> Mapping[int, dict]:
    """The pixel votes the fallback may turn into calls.

    With ``ops_pixel_read_gate`` on, a pixel vote is kept for an object its
    reads did not call only when :func:`_reads_allow` passes, so the
    fallback calls objects no read named and confirms a lone read guide; it
    never overrides a read tie or a read refusal. Off keeps every vote.

    :param pixel: object -> pixel vote (:func:`spacr.ops_sbs.vote_pixels`).
    :param votes: from :func:`_field_votes`, for ``named`` and
        ``pixel_gate``.
    :param called: the objects the reads called; their pixel votes stay,
        as the pixel support of a read call.
    :param refused: filled with object -> reason for the votes the pixel
        bleed gate refused (:func:`_bleed_refusals`), when given.
    :returns: the kept votes; an object whose read call
        ``ops_min_top_share`` refused (``short_share``) is never kept. Under
        ``ops_pixel_ownership`` 'read_rule' (``bleed_gate``) the pixel bleed
        gate then refuses the votes a neighbour's amplicons made.
    """
    short = votes.get("short_share") or ()
    if short:
        pixel = {owner: hit for owner, hit in pixel.items() if owner not in short}
    if votes.get("pixel_gate"):
        named = votes.get("named") or {}
        pixel = {owner: hit for owner, hit in pixel.items()
                 if owner in called or _reads_allow(named.get(owner, ()), hit["guide"])}
    if votes.get("bleed_gate"):
        dropped = _bleed_refusals(pixel, votes, called)
        if refused is not None:
            refused.update(dropped)
        pixel = {owner: hit for owner, hit in pixel.items() if owner not in dropped}
    return pixel


def _vote_weighting(rule: Mapping[str, Any], gaps: np.ndarray) -> Dict[str, Any]:
    """The item-988 keyword arguments of :func:`spacr.ops_sbs.vote_mapped_reads`.

    :param rule: from :func:`decode_rule`; the field's ``neighbours``
        (:func:`spacr.ops_sbs._disc_neighbours`) added by the engine.
    :param gaps: the owned reads' gaps to their nucleus disc.
    :returns: no arguments while the weighting is off.
    :raises ValueError: for a neighbour discount without the field's
        neighbours; :func:`_field_votes` first reads them from a field cache
        that holds the nucleus positions (:func:`cached_neighbours`).
    """
    weighting = rule.get("read_weight")
    if not weighting:
        return {}
    if weighting["neighbour_discount"] != 1.0 and rule.get("neighbours") is None:
        if weighting["neighbour_discount"] == _DEFAULTS["ops_read_neighbour_discount"]:
            rule = _rule_with_neighbours(rule, {}, float(rule.get("footprint", 0.0)))
            weighting = rule["read_weight"]
        else:
            from .qt.i18n import tr

            raise ValueError(tr("ops_read_neighbour_discount needs the field's nucleus positions; "
                                "a stored field cache does not hold them"))
    return {"gaps": np.maximum(np.asarray(gaps, float), 0.0),
            "distance_weight": weighting["distance_weight"],
            "decay_px": weighting["decay_px"], "interior_px": weighting["interior_px"],
            "min_weight": weighting["min_weight"], "neighbours": rule.get("neighbours"),
            "neighbour_discount": weighting["neighbour_discount"]}


def _assemble_calls(primary: Mapping[int, dict], pixel: Mapping[int, dict],
                    votes: Mapping[str, Any], decoys) -> Dict[str, Any]:
    """A field's owned calls from its read and pixel votes, primary first.

    :param primary: the kept read votes.
    :param pixel: the kept pixel votes.
    :param votes: from :func:`_field_votes`, for ``owned``, ``reads_per``
        and the read counts.
    :param decoys: the decoy set.
    :returns: the dict :func:`field_calls` returns, without ``evidence``;
        ``assigned_pixel_real`` and ``assigned_pixel_decoy`` count the
        pixel-fallback calls within ``assigned_real`` and ``assigned_decoy``;
        ``split`` holds the reads in this field of nuclei other fields own
        (:func:`_split_reads`).
    """
    owned = votes["owned"]
    reads_per = votes["reads_per"]
    calls: Dict[int, dict] = {}
    for owner, hit in pixel.items():
        calls[owner] = {"guide": hit["guide"], "source": "pixel_fallback",
                        "n_agreeing": hit["n_pixels"], "fraction": hit["fraction"],
                        "quality": float("nan"), "barcode": hit["barcode"],
                        "n_pixels": hit["n_pixels"]}
    for owner, hit in primary.items():
        calls[owner] = {"guide": hit["guide"], "source": "primary",
                        "n_agreeing": hit["n_support"], "fraction": hit["fraction"],
                        "quality": hit["quality"], "barcode": hit["barcode"],
                        "n_pixels": pixel[owner]["n_pixels"] if owner in pixel else 0,
                        "n_exact": hit["n_exact"]}
    real = decoy = pixel_real = pixel_decoy = 0
    kept_calls: Dict[int, dict] = {}
    for owner, hit in calls.items():
        if owner not in owned:
            continue
        hit["decoy"] = hit["guide"] in decoys
        hit["n_reads"] = int(reads_per.get(owner, 0))
        decoy += int(hit["decoy"])
        real += int(not hit["decoy"])
        if hit["source"] == "pixel_fallback":
            pixel_decoy += int(hit["decoy"])
            pixel_real += int(not hit["decoy"])
        kept_calls[owner] = hit
    return {"calls": kept_calls, "nuclei": len(owned), "assigned_real": real,
            "assigned_decoy": decoy, "assigned_pixel_real": pixel_real,
            "assigned_pixel_decoy": pixel_decoy, "reads": votes["reads"],
            "reads_owned": votes["reads_owned"], "reads_near": votes["reads_near"],
            "reads_ambiguous": votes["reads_ambiguous"],
            "split": votes.get("split", {})}


#: The thresholds auto mode chooses between when ``ops_decode_auto_grid`` is
#: empty: the single-read floor by the pixel-fallback quality grid of the BMC
#: study's S2-it1 iteration (``s2it_analyse.py``, arm iii, 2026-11-12).
AUTO_GRID: Dict[str, Tuple[float, ...]] = {
    "ops_single_read_quality": (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,
                                0.30, 0.40),
    "ops_pixel_quality": (0.10, 0.15, 0.20, 0.30, 0.40),
}

#: The threshold modes ``ops_decode_threshold_mode`` may name, the first
#: being the default. 'dataset' (item 1033, alpha) decodes every well as
#: 'manual' does and then tunes one threshold set for the whole dataset
#: from the stored field caches (:func:`tune_dataset_thresholds`).
THRESHOLD_MODES = ("manual", "auto", "dataset")

#: Field-bootstrap replicates and seed of the FDR interval auto mode judges
#: a setting by; the study's ``s2it_rule.fdr_ci`` used 1,000 and seed 1.
_AUTO_REPS = 1000
_AUTO_BOOT_SEED = 1

#: The decoy calls a tuning half with none is judged as holding: the 95 %
#: upper bound of a zero count (the rule of three), so a half too small to
#: show a false call cannot pass on a falsely confident zero.
_AUTO_ZERO_DECOYS = 3


def threshold_settings(settings: Mapping[str, Any],
                       rule: Mapping[str, Any]) -> Dict[str, Any]:
    """The decode threshold mode and, in auto mode, its grid and target.

    Manual never-barcode calibration uses a single setting, preserving the
    entered thresholds. Without decoys it requests the reported fallback.

    :param settings: the run's settings, read for
        ``ops_decode_threshold_mode``, ``ops_decode_fdr_max``,
        ``ops_decode_auto_grid``, ``ops_decode_auto_spot_grid``,
        ``ops_decode_auto_tune_fraction``, ``ops_decode_auto_seed`` and the
        shared tuner's ``ops_decode_auto_ci``, ``ops_decode_auto_objective``,
        ``ops_decode_auto_f_beta`` and ``ops_decode_auto_unreachable``, and
        ``ops_decode_fdr_calibration`` and the shared controls
        ``ops_never_barcode_k``, ``ops_never_barcode_seed`` and
        ``ops_never_barcode_sets``.
    :param rule: from :func:`decode_rule`; a grid without the pixel quality,
        or a rule without the pixel fallback, keeps ``pixel_quality``.
    :returns: ``mode``; in auto mode or manual never calibration also ``floors`` and ``pixel_qualities``
        (sorted), ``fdr_max`` (per 100 calls), ``ci``, ``objective``,
        ``f_beta``, ``unreachable``, ``tune_fraction``, ``seed`` and
        ``spot_thresholds`` (the read thresholds tried jointly, sorted; empty
        when the read threshold stays as entered); with
        a calibration method also ``fdr_calibration``.
    :raises ValueError: for an unknown mode or FDR calibration, an FDR target that is not
        above 0, a tuning fraction outside (0, 1), a grid that is not a
        mapping of the two thresholds to lists of numbers 0 or above, or
        auto mode without decoys (``ops_decoy_library_k`` 0).
    """
    from .ops_settings import OPS_DEFAULTS

    _never_settings({
        'ops_never_barcode_k': settings.get('ops_never_barcode_k', OPS_DEFAULTS['ops_never_barcode_k']),
        'ops_never_barcode_seed': settings.get('ops_never_barcode_seed', OPS_DEFAULTS['ops_never_barcode_seed']),
        'ops_never_barcode_sets': settings.get('ops_never_barcode_sets', OPS_DEFAULTS['ops_never_barcode_sets'])})
    mode = str(settings.get("ops_decode_threshold_mode") or THRESHOLD_MODES[0]
               ).strip().lower()
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"ops_decode_threshold_mode must be one of "
                         f"{list(THRESHOLD_MODES)}; got {mode!r}")
    if mode == "dataset":
        dataset_threshold_settings(settings, rule)
        return {**threshold_settings({**settings, "ops_decode_threshold_mode": "manual"},
                                     rule), "dataset": True}
    calibration = str(settings.get("ops_decode_fdr_calibration")
                      or FDR_CALIBRATIONS[0]).strip().lower()
    manual_calibration = mode == "manual" and calibration in ("never", "never+shuffle")
    if mode == "manual" and not manual_calibration:
        return {"mode": mode}
    if manual_calibration and int(rule.get("decoy_k") or 0) < 1:
        return {"mode": mode, "fdr_calibration": calibration, "no_decoys": True}
    if int(rule.get("decoy_k") or 0) < 1:
        raise ValueError("ops_decode_threshold_mode 'auto' chooses thresholds "
                         "from decoy calls; set ops_decoy_library_k to 1 or "
                         "more (4 in the BMC study)")
    fdr_max = settings.get("ops_decode_fdr_max")
    fdr_max = 0.1 if fdr_max is None or fdr_max == "" else fdr_max
    tune = settings.get("ops_decode_auto_tune_fraction")
    tune = 0.5 if tune is None or tune == "" else tune
    seed = settings.get("ops_decode_auto_seed")
    seed = 0 if seed is None or seed == "" else seed
    try:
        fdr_max, tune, seed = float(fdr_max), float(tune), int(seed)
    except (TypeError, ValueError):
        raise ValueError("ops_decode_fdr_max, ops_decode_auto_tune_fraction "
                         "and ops_decode_auto_seed must be numbers") from None
    if not fdr_max > 0:
        raise ValueError(f"ops_decode_fdr_max must be above 0; got {fdr_max}")
    if not 0 < tune < 1:
        raise ValueError("ops_decode_auto_tune_fraction must lie in (0, 1); "
                         f"got {tune}")
    grid = ({"ops_single_read_quality": [float(settings.get("ops_single_read_quality")
                                              if settings.get("ops_single_read_quality") is not None
                                              else OPS_DEFAULTS["ops_single_read_quality"])],
             "ops_pixel_quality": [rule["pixel_quality"]]}
            if manual_calibration else settings.get("ops_decode_auto_grid"))
    if isinstance(grid, str) and grid.strip():
        try:
            grid = json.loads(grid)
        except ValueError:
            raise ValueError("ops_decode_auto_grid must be a mapping such as "
                             '{"ops_single_read_quality": [0.06, 0.1]}; got '
                             f"{grid!r}") from None
    if grid is None or grid == "" or grid == {}:
        grid = AUTO_GRID
    if not isinstance(grid, Mapping) or set(grid) - set(AUTO_GRID):
        raise ValueError(f"ops_decode_auto_grid takes the keys {sorted(AUTO_GRID)}; "
                         f"got {grid!r}")
    axes = {}
    for key, fixed in (("ops_single_read_quality", None),
                       ("ops_pixel_quality", rule["pixel_quality"])):
        values = grid.get(key)
        if values is None or (key == "ops_pixel_quality" and not rule["pixels"]):
            if fixed is None:
                raise ValueError("ops_decode_auto_grid needs "
                                 "ops_single_read_quality values")
            values = [fixed]
        try:
            values = sorted({float(v) for v in values})
        except (TypeError, ValueError):
            raise ValueError(f"ops_decode_auto_grid {key} must be a list of "
                             f"numbers; got {values!r}") from None
        if not values or values[0] < 0:
            raise ValueError(f"ops_decode_auto_grid {key} needs values of 0 "
                             f"or more; got {values!r}")
        axes[key] = values
    spots = ([] if manual_calibration else _spot_grid(settings.get("ops_decode_auto_spot_grid"),
                                        "ops_decode_auto_spot_grid"))
    out = {"mode": mode, "floors": axes["ops_single_read_quality"],
           "pixel_qualities": axes["ops_pixel_quality"], "fdr_max": fdr_max,
           **_tuning_options(settings), "tune_fraction": tune, "seed": seed,
           "spot_thresholds": spots}
    calibration = str(settings.get("ops_decode_fdr_calibration")
                      or FDR_CALIBRATIONS[0]).strip().lower()
    if calibration not in FDR_CALIBRATIONS:
        raise ValueError(f"ops_decode_fdr_calibration must be one of "
                         f"{list(FDR_CALIBRATIONS)}; got {calibration!r}")
    if calibration != FDR_CALIBRATIONS[0]:
        out["fdr_calibration"] = calibration
    return out


#: The joint read-threshold grid of item 942 (``ops_read_threshold``, the
#: spot strength a read needs), which auto mode tries with the call
#: thresholds when ``ops_decode_auto_spot_grid`` names it. Since item 946
#: the setting defaults to empty (calling thresholds only): the 942
#: fast-search card did not pass. The field cache keeps spots from
#: strength 100.
AUTO_SPOT_GRID: Tuple[float, ...] = (150.0, 200.0, 250.0, 315.0, 400.0)


def _spot_grid(value: Any, key: str) -> List[float]:
    """The read thresholds a tuner tries, sorted, or none.

    :param value: a list of numbers, a comma-separated string, or None or
        an empty list for none (the read threshold stays as entered).
    :param key: the setting's name, for the error message.
    :returns: the distinct values in ascending order (the loosest first).
    :raises ValueError: for a value that is not a number of 0 or more.
    """
    if value is None or value == "" or (isinstance(value, (list, tuple)) and not value):
        return []
    if isinstance(value, str):
        value = [part for part in value.replace(";", ",").split(",") if part.strip()]
    try:
        values = sorted({float(v) for v in value})
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a list of numbers; got {value!r}") from None
    if values and values[0] < 0:
        raise ValueError(f"{key} needs values of 0 or more; got {values!r}")
    return values


def _tuning_options(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The shared tuner's objective settings (item 942), checked.

    :param settings: read for ``ops_decode_auto_objective``,
        ``ops_decode_auto_f_beta``, ``ops_decode_auto_unreachable`` and
        ``ops_decode_auto_ci``; the OPS decode and the spacer-prefix decode
        both use them.
    :returns: ``ci``, ``objective``, ``f_beta`` and ``unreachable`` for
        :func:`tune_thresholds`.
    :raises ValueError: for an unknown objective or unreachable rule, or an
        F-score beta that is not above 0.
    """
    ci = settings.get("ops_decode_auto_ci")
    if isinstance(ci, str):
        ci = ci.strip().lower() not in ("false", "0", "no", "off")
    objective = str(settings.get("ops_decode_auto_objective")
                    or TUNING_OBJECTIVES[0]).strip().lower()
    if objective not in TUNING_OBJECTIVES:
        raise ValueError(f"ops_decode_auto_objective must be one of "
                         f"{list(TUNING_OBJECTIVES)}; got {objective!r}")
    unreachable = str(settings.get("ops_decode_auto_unreachable")
                      or TUNING_UNREACHABLE[0]).strip().lower()
    if unreachable not in TUNING_UNREACHABLE:
        raise ValueError(f"ops_decode_auto_unreachable must be one of "
                         f"{list(TUNING_UNREACHABLE)}; got {unreachable!r}")
    beta = settings.get("ops_decode_auto_f_beta")
    try:
        beta = 1.0 if beta is None or beta == "" else float(beta)
    except (TypeError, ValueError):
        raise ValueError("ops_decode_auto_f_beta must be a number") from None
    if not beta > 0:
        raise ValueError(f"ops_decode_auto_f_beta must be above 0; got {beta}")
    return {"ci": True if ci is None or ci == "" else bool(ci),
            "objective": objective, "f_beta": beta, "unreachable": unreachable}


def _prune_tallies(tallies: Mapping[str, Any], keys) -> Dict[str, Any]:
    """The vote tallies the evidence rows of the given calls read, and no more.

    :param tallies: from :func:`spacr.ops_sbs.vote_mapped_reads`.
    :param keys: the ``(object, guide)`` pairs a row may be written for.
    :returns: the same layout as plain dicts, for those objects and guides;
        every object's full ``votes`` stays, since a row counts its guides.
    """
    owners = {owner for owner, _ in keys}
    out: Dict[str, Any] = {"votes": {o: dict(tallies["votes"][o])
                                     for o in owners if o in tallies["votes"]},
                           "matched": {o: tallies["matched"][o]
                                       for o in owners if o in tallies["matched"]}}
    for name in ("exact", "corrected", "best", "qualities", "strength"):
        table = tallies[name]
        kept: Dict[int, dict] = {o: {} for o in owners if o in table}
        for owner, guide in keys:
            if owner in table and guide in table[owner]:
                kept[owner][guide] = table[owner][guide]
        out[name] = kept
    return out


@lru_cache(maxsize=2)
def _opportunity_index(guides):
    """Index a sorted guide tuple once for both decode opportunity counters.

    :param guides: the ordered real guide keys used by the distance profiles.
    :returns: guide key to aligned profile row.
    """
    return {guide: index for index, guide in enumerate(guides)}


def _opportunity_totals(guides, profiles, guide_index):
    """Reduce real-cell guide winners to shared distance-shell opportunity totals.

    :param guides: the real guide selected for each called cell.
    :param profiles: named L/D/N1/N2 matrices with columns for distances one to three.
    :param guide_index: the shared mapping from guide key to aligned matrix row.
    :returns: twelve integer sums from one guide histogram and the same dot products.
    """
    bins = np.bincount([guide_index[guide] for guide in guides], minlength=len(guide_index))
    return {f'opp{d}_{name}': int(total)
            for name, matrix in profiles.items() for d, total in enumerate(bins @ matrix, 1)}


def _field_candidates(votes: Dict[str, Any], floors: Sequence[float],
                      pixel_qualities: Sequence[float], decoys,
                      pixels: bool, *, opportunity_profiles=None) -> Dict[str, Any]:
    """Count calls at every grid setting in one field and retain their reconstruction data.

    :param votes: from :func:`_field_votes` with no single-read floor and
        the pixel votes at every grid pixel quality.
    :param floors: the single-read floors of the grid.
    :param pixel_qualities: the pixel qualities of the grid.
    :param decoys: the decoy set.
    :param pixels: whether the rule has the pixel fallback.
    :param opportunity_profiles: optional sorted ``guides`` and L/D/N1/N2
        distance-profile arrays, each shaped ``(guides, 3)``. The same
        selected primary and pixel calls supply the guide bincount.
    :returns: ``counts`` -- ``(floors, pixel qualities, 2)`` real and decoy
        calls -- ``nuclei``, and ``votes`` with the tallies pruned to the
        rows any setting can write, for :func:`_candidate_calls`. With
        opportunity profiles, also ``opportunities``: twelve named
        distance-shell grids, without retaining a dense guide-count grid.
    """
    owned = votes["owned"]
    primary = votes["primary"]
    pixel = votes["pixel"] if pixels else {q: {} for q in pixel_qualities}
    allowed = {q: _fallback_pixels(pixel[q], votes) for q in pixel_qualities}
    mine = [(o, h) for o, h in primary.items() if o in owned]
    support = np.array([h["n_support"] for _, h in mine], np.int64)
    best = np.array([h["quality"] for _, h in mine], float)
    owners = [o for o, _ in mine]
    is_decoy = {o: h["guide"] in decoys for o, h in mine}
    counts = np.zeros((len(floors), len(pixel_qualities), 2), np.int64)
    profiles = dict(opportunity_profiles or {})
    guide_index = _opportunity_index(tuple(profiles.pop('guides', ())))
    profiles = {name: np.asarray(values, np.int64) for name, values in profiles.items()}
    if any(values.shape != (len(guide_index), 3) for values in profiles.values()):
        from .qt.i18n import tr

        raise ValueError(tr('Opportunity profiles must align with their sorted guide library.'))
    opportunities = {f'opp{d}_{name}': np.zeros(counts.shape[:-1], np.int64)
                     for name in profiles for d in (1, 2, 3)}
    for i, floor in enumerate(floors):
        keep = ~((support == 1) & (best < floor))
        kept = {o for o, k in zip(owners, keep) if k}
        decoy = sum(is_decoy[o] for o in kept)
        for j, q in enumerate(pixel_qualities):
            extra = [h for o, h in allowed[q].items() if o in owned and o not in kept]
            extra_decoys = sum(h['guide'] in decoys for h in extra)
            counts[i, j] = (len(kept) - decoy + len(extra) - extra_decoys,
                            decoy + extra_decoys)
            if profiles:
                selected = [primary[o]['guide'] for o in kept] + [h['guide'] for h in extra]
                real_guides = [guide for guide in selected if guide not in decoys]
                if any(guide not in guide_index for guide in real_guides):
                    from .qt.i18n import tr

                    raise ValueError(tr('Opportunity profiles do not contain every called guide.'))
                for name, total in _opportunity_totals(real_guides, profiles, guide_index).items():
                    opportunities[name][i, j] = total
    if votes["tallies"] is not None:
        keys = {(o, h["guide"]) for o, h in primary.items()}
        keys |= {(o, h["guide"]) for table in pixel.values()
                 for o, h in table.items()}
        votes = {**votes, "tallies": _prune_tallies(votes["tallies"], keys)}
    out = {"counts": counts, "nuclei": len(owned), "votes": {**votes, "pixel": pixel}}
    if profiles:
        out['opportunities'] = opportunities
    return out


def _joint_field_candidates(evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any],
                            library: Sequence[str], matcher, decoys, *,
                            spot_thresholds: Sequence[float], floors: Sequence[float],
                            pixel_qualities: Sequence[float], footprint: float,
                            max_distance: int, min_fraction: float,
                            crosstalk: str, pixel_votes=None,
                            opportunity_profiles=None) -> Dict[str, Any]:
    """Calls in one field at every read threshold and call-threshold setting (item 942).

    Spot detection and calling are tuned together: for each read threshold
    the reads at or below it are dropped, the reads left are called and
    voted again (the library calibration refits on them), and
    :func:`_field_candidates` counts the calls at every call-threshold
    setting. The pixel fallback reads no spot, so its votes are made once.

    :param evidence: from :func:`field_evidence`, kept down to the lowest
        read threshold.
    :param rule: from :func:`decode_rule`.
    :param library: the guide barcodes.
    :param matcher: from :func:`matcher_for`.
    :param decoys: the decoy set.
    :param spot_thresholds: the read thresholds, ascending.
    :param floors: the single-read floors of the grid.
    :param pixel_qualities: the pixel qualities of the grid.
    :param footprint: how far beyond a boundary a read may lie, in pixels.
    :param max_distance: the mismatches a read may carry.
    :param min_fraction: the share of the votes the guide must hold.
    :param crosstalk: the unmixing method.
    :param pixel_votes: the pixel votes of another pass over the same field
        (item 952: the log pass, reused by the raw pass); None votes them here.
    :param opportunity_profiles: the optional guide profiles passed to
        ``_field_candidates``; returned opportunities are stacked across
        the read-threshold axis alongside counts.
    :returns: ``counts`` -- ``(read thresholds, floors, pixel qualities, 2)``
        real and decoy calls -- ``nuclei``, and ``by_spot``, one
        :func:`_field_candidates` result per read threshold with its read
        counts.
    """
    by_spot: List[Dict[str, Any]] = []
    pixel = pixel_votes
    rule = _rule_with_pixel_neighbours(
        _rule_with_neighbours(rule, evidence, footprint), evidence, footprint)
    for threshold in spot_thresholds:
        votes = _field_votes(
            evidence, rule if pixel is None else {**rule, "pixels": False, "pixels_given": True},
            library,
            matcher, threshold=float(threshold), footprint=footprint,
            max_distance=max_distance, min_fraction=min_fraction,
            single_read_quality=float("-inf"), pixel_qualities=pixel_qualities,
            crosstalk=crosstalk, evidence_rows=True)
        if pixel is None:
            pixel = votes["pixel"]
        votes["pixel"] = pixel
        calls = _field_candidates(votes, floors, pixel_qualities, decoys,
                                  bool(rule["pixels"]),
                                  opportunity_profiles=opportunity_profiles)
        calls.update({key: votes[key] for key in (
            "reads", "reads_owned", "reads_near", "reads_ambiguous")})
        by_spot.append(calls)
    out = {"counts": np.stack([c["counts"] for c in by_spot]),
           "nuclei": by_spot[0]["nuclei"] if by_spot else 0, "by_spot": by_spot}
    if by_spot and 'opportunities' in by_spot[0]:
        out['opportunities'] = {key: np.stack([row['opportunities'][key] for row in by_spot])
                                for key in by_spot[0]['opportunities']}
    return out


#: Shuffled-barcode matchers built once per process (item 949), keyed by
#: the library and the seeds.
_SHUFFLE_MATCHERS: Dict[Tuple[int, Tuple[int, ...]], List[Any]] = {}


def _shuffled_barcodes(library: Sequence[str], seed: int) -> List[str]:
    """The barcode-shuffle control: every barcode's letters shuffled (item 949).

    :param library: the guide barcodes.
    :param seed: the generator's seed.
    :returns: one shuffled barcode per guide barcode, in sorted guide order,
        with the same lengths and base composition.
    """
    rng = np.random.default_rng(int(seed))
    out = []
    for barcode in sorted(set(library)):
        letters = np.array(list(barcode))
        out.append("".join(letters[rng.permutation(len(letters))]))
    return out


def _shuffle_matchers(library: Sequence[str], seeds: Sequence[int]) -> List[Any]:
    """One matcher per shuffled library, built once per process.

    :param library: the guide barcodes.
    :param seeds: the shuffle seeds.
    :returns: a :class:`spacr.ops_sbs.LibraryMatcher` per seed.
    """
    from .ops_sbs import LibraryMatcher

    key = (hash(frozenset(library)), tuple(int(s) for s in seeds))
    if key not in _SHUFFLE_MATCHERS:
        _SHUFFLE_MATCHERS.clear()
        _SHUFFLE_MATCHERS[key] = [LibraryMatcher(_shuffled_barcodes(library, s))
                                  for s in key[1]]
    return _SHUFFLE_MATCHERS[key]


def _field_shuffle_counts(evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any],
                          library: Sequence[str], *, seeds: Sequence[int],
                          spot_thresholds: Sequence[float], floors: Sequence[float],
                          pixel_qualities: Sequence[float], footprint: float,
                          max_distance: int, min_fraction: float,
                          crosstalk: str) -> np.ndarray:
    """One field's shuffle-called cells at every grid setting (item 949).

    The reads are called as for the real library (the library calibration
    fits the real guides) and voted again against each shuffled library
    (:func:`_shuffled_barcodes`); a cell called on a shuffled barcode is a
    shuffle call. The pixel votes are made once per shuffled library.

    :param evidence: from :func:`field_evidence`.
    :param rule: from :func:`decode_rule`.
    :param library: the guide barcodes.
    :param seeds: the shuffle seeds.
    :param spot_thresholds: the read thresholds of the grid.
    :param floors: the single-read floors of the grid.
    :param pixel_qualities: the pixel qualities of the grid.
    :param footprint: how far beyond a boundary a read may lie, in pixels.
    :param max_distance: the mismatches a read may carry.
    :param min_fraction: the share of the votes the guide must hold.
    :param crosstalk: the unmixing method.
    :returns: ``(read thresholds, floors, pixel qualities)`` shuffle calls,
        the mean over the shuffled libraries.
    """
    out = np.zeros((len(spot_thresholds), len(floors), len(pixel_qualities)))
    rule = _rule_with_pixel_neighbours(
        _rule_with_neighbours(rule, evidence, footprint), evidence, footprint)
    for matcher in _shuffle_matchers(library, seeds):
        pixel = None
        for a, threshold in enumerate(spot_thresholds):
            votes = _field_votes(
                evidence, rule if pixel is None else {**rule, "pixels": False, "pixels_given": True},
            library,
                matcher, threshold=float(threshold), footprint=footprint,
                max_distance=max_distance, min_fraction=min_fraction,
                single_read_quality=float("-inf"), pixel_qualities=pixel_qualities,
                crosstalk=crosstalk, evidence_rows=False)
            if pixel is None:
                pixel = votes["pixel"]
            votes["pixel"] = pixel
            out[a] += _field_candidates(votes, floors, pixel_qualities, frozenset(),
                                        bool(rule["pixels"]))["counts"][..., 0]
    return out / max(len(seeds), 1)


def _never_settings(settings):
    """Validate the shared substitution-control settings for either decode.

    :param settings: run settings, with absent controls taking their defaults.
    :returns: the three normalized setting keys.
    :raises ValueError: for negative or fractional counts/seeds or an unknown set choice.
    """
    from .qt.i18n import tr

    out = {'ops_never_barcode_k': settings.get('ops_never_barcode_k', 4),
           'ops_never_barcode_seed': settings.get('ops_never_barcode_seed', 1)}
    for key, value in out.items():
        try:
            number = int(value)
            valid = not isinstance(value, bool) and float(value) == number and number >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(tr('{key} must be a nonnegative integer.').format(key=key))
        out[key] = number
    sets = str(settings.get('ops_never_barcode_sets', 'n1_n2')).strip().lower()
    if sets not in ('n1_n2', 'n2'):
        raise ValueError(tr('ops_never_barcode_sets must be n1_n2 or n2.'))
    out['ops_never_barcode_sets'] = sets
    return out


@lru_cache(maxsize=2)
def _never_context(library: tuple, decoys: tuple, k: int, seed: int, sets: str):
    """Build independent controls and aligned opportunity profiles once per library.

    :param library: sorted production guide sequences, after any cycle mask.
    :param decoys: sorted production decoy sequences.
    :param k: attempted controls per guide in each requested set.
    :param seed: deterministic substitution seed.
    :param sets: ``n1_n2`` or ``n2``; the latter skips the report-only pass.
    :returns: controls, guide profiles and serializable run metadata shared
        by both decode paths. No production calls are replaced.
    """
    from .ops_sbs import never_barcodes, _barcode_distance_profiles

    controls = {name: never_barcodes(library, distance=distance, k=k, seed=seed)
                for name, distance in ((('N1', 1), ('N2', 2)) if sets == 'n1_n2'
                                       else (('N2', 2),))}
    profiles = {'guides': library,
                'L': controls['N2']['library_m_d'],
                'D': _barcode_distance_profiles(library, decoys),
                **{name: data['m_d'] for name, data in controls.items()}}
    profiles.setdefault('N1', np.zeros((len(library), 3), np.int64))
    metadata = {'k': int(k), 'seed': int(seed), 'sets': sets,
                'sizes': {'L': len(library), 'D': len(decoys),
                          **{name: len(data['barcodes']) for name, data in controls.items()}},
                'mean_m_d': {name: np.asarray(values).mean(axis=0).tolist()
                             if len(library) else [0.0, 0.0, 0.0]
                             for name, values in profiles.items() if name != 'guides'}}
    return {'controls': controls, 'profiles': profiles, 'never_sets': metadata,
            'library_size': len(library)}


@lru_cache(maxsize=2)
def _never_matchers(library: tuple, controls: tuple):
    """Cache existing matcher objects for each independent control whitelist.

    :param library: sorted production guide sequences.
    :param controls: named tuples of control sequences from ``_never_context``.
    :returns: named matcher/decoy-set pairs from ``matcher_for``.
    """
    return {name: matcher_for(library, barcodes) for name, barcodes in controls}


def _never_payload(grids, metadata):
    """Bind measured field grids and shared control metadata for the existing calibrator.

    :param grids: aligned control and opportunity arrays from either decode.
    :param metadata: library size and ``never_sets`` from the shared builder.
    :returns: a calibrator payload. If N1 voting was skipped, its zero
        exposure marks its rate unavailable; a library with distance-one
        pairs then retains the entered scale rather than omitting that term.
    """
    out = {key: grids[key] for key in ('never2_calls', 'opp2_N2', 'opp2_L',
                                       'opp1_L', 'real_calls_n2')}
    if 'never1_calls' in grids:
        out.update({key: grids[key] for key in ('never1_calls', 'opp1_N1', 'real_calls_n1')})
    else:
        out.update(never1_calls=np.zeros_like(out['never2_calls']),
                   opp1_N1=np.zeros_like(out['never2_calls']))
    return {**out, 'library_size': metadata['library_size'], 'never_sets': metadata['never_sets']}


def _field_never_counts(evidence, rule, library, context, *, spot_thresholds,
                        floors, pixel_qualities, footprint, max_distance,
                        min_fraction, crosstalk, pixel_votes=None):
    """Vote each never whitelist through the production field candidate path.

    :param evidence: the existing field evidence for one read mode.
    :param rule: the production decode rule.
    :param library: sorted production guides.
    :param context: cached controls and profiles from ``_never_context``.
    :param spot_thresholds: the read-threshold axis, including fixed mode.
    :param floors: the single-read quality axis.
    :param pixel_qualities: the pixel-quality axis.
    :param footprint: production territory margin.
    :param max_distance: production correction limit.
    :param min_fraction: production vote share.
    :param crosstalk: production unmixing method.
    :param pixel_votes: optional votes by control name from the same field
        and whitelist in another read mode; never production-whitelist votes.
    :returns: control count grids and per-whitelist pixel votes for reuse.
    """
    controls = tuple((name, tuple(data['barcodes']))
                     for name, data in context['controls'].items())
    matchers = _never_matchers(tuple(library), controls)
    out, pixels = {}, {}
    for name, (matcher, decoys) in matchers.items():
        called = _joint_field_candidates(
            evidence, rule, library, matcher, decoys,
            spot_thresholds=spot_thresholds, floors=floors,
            pixel_qualities=pixel_qualities, footprint=footprint,
            max_distance=max_distance, min_fraction=min_fraction,
            crosstalk=crosstalk, pixel_votes=(pixel_votes or {}).get(name))
        suffix = name.lower()
        out[f'never{name[-1]}_calls'] = called['counts'][..., 1]
        out[f'real_calls_{suffix}'] = called['counts'][..., 0]
        pixels[name] = called['by_spot'][0]['votes']['pixel']
    return out, pixels


def _candidate_calls(candidates: Mapping[str, Any], floor: float,
                     pixel_quality: float, decoys) -> Dict[str, Any]:
    """The calls :func:`field_calls` makes at one grid setting, rebuilt.

    :param candidates: from :func:`_field_candidates`.
    :param floor: the single-read floor chosen.
    :param pixel_quality: the pixel quality chosen.
    :param decoys: the decoy set.
    :returns: the dict :func:`field_calls` returns at
        ``single_read_quality=floor`` and ``ops_pixel_quality=pixel_quality``,
        with ``evidence`` when the tallies were kept.
    """
    votes = candidates["votes"]
    primary = {o: h for o, h in votes["primary"].items()
               if not (h["n_support"] == 1 and h["quality"] < floor)}
    refused: Dict[int, str] = {}
    pixel = _fallback_pixels(votes["pixel"][pixel_quality], votes, primary, refused)
    out = _assemble_calls(primary, pixel, votes, decoys)
    if votes.get("bleed_gate"):
        out["pixel_refused"] = _refusal_counts(refused, votes["owned"])
    if votes["tallies"] is not None:
        out["evidence"] = _evidence_rows(primary, pixel, votes["tallies"],
                                         votes["owned"], decoys,
                                         votes["objects"], votes["ambiguous"])
    return out


def _bootstrap_rows(n: int, reps: int, seed: int) -> np.ndarray:
    """The field resamples of the FDR interval, drawn as the study drew them.

    :param n: fields in the half.
    :param reps: replicates.
    :param seed: the seed.
    :returns: ``(reps, n)`` row indices, one draw of ``n`` per replicate in
        turn, so the stream equals ``s2it_rule.fdr_ci``'s.
    """
    rng = np.random.default_rng(seed)
    return np.array([rng.integers(0, n, n) for _ in range(int(reps))],
                    np.int64).reshape(int(reps), n)


def _half_stats(counts: np.ndarray, nuclei: np.ndarray, draws: np.ndarray,
                k: int, scale: float) -> Dict[str, np.ndarray]:
    """Calls, genotyped share and decoy FDR with its interval over one half.

    :param counts: ``(fields, floors, pixel qualities, 2)``.
    :param nuclei: owned nuclei per field.
    :param draws: from :func:`_bootstrap_rows`.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :returns: per grid setting: ``real``, ``decoy``, ``genotyped_pct``,
        ``fdr``, ``fdr_lo`` and ``fdr_hi``, and ``nuclei``.
    """
    real = counts[..., 0].sum(0)
    decoy = counts[..., 1].sum(0)
    c = 100.0 * max(float(scale), 1.0) / float(k)
    weights = np.stack([np.bincount(row, minlength=len(counts)) for row in draws])
    sums = (weights @ counts.reshape(len(counts), -1)).reshape(
        (len(draws),) + counts.shape[1:])
    boot = c * sums[..., 1] / np.maximum(sums[..., 0], 1)
    total = int(nuclei.sum())
    return {"real": real, "decoy": decoy, "nuclei": total,
            "genotyped_pct": 100.0 * real / max(total, 1),
            "fdr": c * decoy / np.maximum(real, 1),
            "fdr_lo": np.percentile(boot, 2.5, axis=0),
            "fdr_hi": np.percentile(boot, 97.5, axis=0)}


def _chosen_summary(counts: np.ndarray, nuclei: np.ndarray, fields,
                    chosen: Tuple[int, int], k: int, scale: float
                    ) -> Dict[str, Any]:
    """The chosen setting's numbers over some of a well's fields.

    :param counts: ``(fields, floors, pixel qualities, 2)``.
    :param nuclei: owned nuclei per field.
    :param fields: field indices in site order.
    :param chosen: the grid position chosen.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :returns: fields, nuclei, real and decoy calls, genotyped %, FDR and its
        interval (None for fewer than two fields).
    """
    if not len(fields):
        return {"fields": 0, "nuclei": 0, "real": 0, "decoy": 0,
                "genotyped_pct": None, "fdr_per_100": None, "fdr_ci95": None}
    idx = np.asarray(fields)
    part = counts[idx][:, chosen[0], chosen[1]][:, None, None, :]
    stats = _half_stats(part, nuclei[idx],
                        _bootstrap_rows(len(idx), _AUTO_REPS, _AUTO_BOOT_SEED),
                        k, scale)
    return {"fields": int(len(idx)), "nuclei": int(stats["nuclei"]),
            "real": int(stats["real"][0, 0]), "decoy": int(stats["decoy"][0, 0]),
            "genotyped_pct": float(stats["genotyped_pct"][0, 0]),
            "fdr_per_100": float(stats["fdr"][0, 0]),
            "fdr_ci95": ([float(stats["fdr_lo"][0, 0]), float(stats["fdr_hi"][0, 0])]
                         if len(idx) > 1 else None)}


def choose_well_thresholds(counts, nuclei, sites, floors: Sequence[float],
                           pixel_qualities: Sequence[float], *, k: int,
                           scale: float, fdr_max: float = 0.1,
                           ci: bool = True, tune_fraction: float = 0.5,
                           seed: int = 0) -> Dict[str, Any]:
    """Choose one well's single-read floor and pixel quality from its decoys.

    This is arm (iii) of the BMC study's S2-it1 iteration
    (``s2it_analyse.py``): the well's fields with nuclei, in site order, are
    split at random (``numpy.random.default_rng(seed).permutation``) into a
    tuning share and a report share. On the tuning share every grid setting
    is scored; a setting passes when its decoy FDR -- the upper end of the
    95 % field-bootstrap interval (1,000 resamples, seed 1) with ``ci``,
    else the estimate -- is at most ``fdr_max``. A tuning share with no
    decoy call is judged as holding three, the 95 % bound of a zero count.
    The passing setting with the most real calls wins; ties go to the higher
    floor, then the higher pixel quality. With none passing, the strictest
    setting (highest floor and pixel quality) is used and ``auto_met`` is
    False. Only decoy calls are read, never a reference call set.

    :param counts: ``(fields, floors, pixel qualities, 2)`` real and decoy
        calls per field and setting.
    :param nuclei: owned nuclei per field.
    :param sites: the site of each field.
    :param floors: the single-read floors, in the order of ``counts``.
    :param pixel_qualities: the pixel qualities, in the order of ``counts``.
    :param k: decoys per guide.
    :param scale: the FDR calibration (:func:`fdr`).
    :param fdr_max: the most false calls per 100 real calls accepted.
    :param ci: judge the interval's upper end rather than the estimate.
    :param tune_fraction: the share of the fields that choose; the rest
        only report.
    :param seed: the split's seed.
    :returns: ``single_read_quality``, ``pixel_quality``, ``auto_met``,
        ``fdr_max``, ``ci``, ``tune_fraction``, ``seed``, ``tune_sites``
        and ``report_sites``; ``tune``, ``report`` and ``all_fields`` --
        nuclei, real and decoy calls, genotyped %, FDR and its interval at
        the chosen setting -- and ``grid``, the tuning score of every
        setting.
    :raises ValueError: when the shapes disagree or ``k`` is below 1.
    """
    floors = [float(v) for v in floors]
    pixel_qualities = [float(v) for v in pixel_qualities]
    counts = np.asarray(counts, np.int64)
    if counts.shape[1:] != (len(floors), len(pixel_qualities), 2) \
            or len(nuclei) != len(counts) or len(sites) != len(counts):
        raise ValueError("counts must be (fields, floors, pixel qualities, 2) "
                         "with one nuclei count and site per field")
    choice = tune_thresholds(
        counts, nuclei, sites, [("single_read_quality", floors),
                                ("pixel_quality", pixel_qualities)],
        k=k, scale=scale, fdr_max=fdr_max, ci=ci, tune_fraction=tune_fraction,
        seed=seed, unreachable="strictest")
    return {"single_read_quality": choice["chosen"]["single_read_quality"],
            "pixel_quality": choice["chosen"]["pixel_quality"],
            "auto_met": choice["auto_met"], "fdr_max": choice["fdr_max"],
            "ci": choice["ci"], "tune_fraction": choice["tune_fraction"],
            "seed": choice["seed"], "tune_sites": choice["tune_sites"],
            "report_sites": choice["report_sites"], "tune": choice["tune"],
            "report": choice["report"], "all_fields": choice["all_fields"],
            "grid": [{key: value for key, value in row.items()
                      if key in _GRID_ROW_939} for row in choice["grid"]]}


#: The keys of a grid row of :func:`choose_well_thresholds` (item 939).
_GRID_ROW_939 = ("single_read_quality", "pixel_quality", "real", "decoy",
                 "genotyped_pct", "fdr_per_100", "fdr_ci95", "passes")

#: The objectives :func:`tune_thresholds` may maximise, the first being the
#: default: the most genotyped cells under the FDR target, or an F-score of
#: precision (one minus the FDR) and recall (the genotyped share).
TUNING_OBJECTIVES = ("recall_at_fdr", "f_score")

#: What :func:`tune_thresholds` does when no setting meets the FDR target:
#: use the setting with the lowest FDR bound (the default), or the strictest
#: setting (item 939's rule).
TUNING_UNREACHABLE = ("lowest_fdr", "strictest")


def _grid_score(counts: np.ndarray, nuclei: np.ndarray, idx: Sequence[int],
                k: int, scale: float, ci: bool) -> Dict[str, np.ndarray]:
    """The tuning share's score of every grid setting.

    :param counts: ``(fields, *grid, 2)`` real and decoy calls.
    :param nuclei: cells per field.
    :param idx: the tuning fields.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :param ci: judge the interval's upper end rather than the estimate.
    :returns: :func:`_half_stats`'s numbers over the grid, and ``bound``,
        the FDR each setting is judged by: a share with no decoy call is
        judged as holding :data:`_AUTO_ZERO_DECOYS`.
    """
    idx = np.asarray(idx)
    score = _half_stats(counts[idx], nuclei[idx],
                        _bootstrap_rows(len(idx), _AUTO_REPS, _AUTO_BOOT_SEED),
                        k, scale)
    bound = score["fdr_hi"] if ci else score["fdr"]
    floor_bound = 100.0 * max(float(scale), 1.0) / float(k) \
        * _AUTO_ZERO_DECOYS / np.maximum(score["real"], 1)
    score["bound"] = np.where(score["decoy"] == 0, np.maximum(bound, floor_bound),
                              bound)
    return score


def _f_score(fdr_per_100: np.ndarray, genotyped_pct: np.ndarray,
             beta: float) -> np.ndarray:
    """The F-score of precision (one minus the FDR) and recall (genotyped share).

    :param fdr_per_100: the FDR estimate, per 100 calls.
    :param genotyped_pct: the share of cells genotyped, in %.
    :param beta: recall counts ``beta`` times as much as precision.
    :returns: ``(1 + b²) P R / (b² P + R)``, 0 where both are 0.
    """
    precision = np.clip(1.0 - np.asarray(fdr_per_100, float) / 100.0, 0.0, 1.0)
    recall = np.clip(np.asarray(genotyped_pct, float) / 100.0, 0.0, 1.0)
    b2 = float(beta) ** 2
    denominator = b2 * precision + recall
    return np.where(denominator > 0, (1 + b2) * precision * recall
                    / np.where(denominator > 0, denominator, 1.0), 0.0)


def _pareto(fdr_values: Sequence[float], real: Sequence[int]) -> List[bool]:
    """Which settings no other setting beats on both FDR and real calls.

    :param fdr_values: each setting's FDR.
    :param real: each setting's real calls.
    :returns: one flag per setting: True when no setting has an FDR at most
        as high and more real calls, or a lower FDR and as many.
    """
    order = sorted(range(len(real)), key=lambda i: (fdr_values[i], -real[i]))
    best, out = -1, [False] * len(real)
    for i in order:
        if real[i] > best:
            out[i], best = True, real[i]
    return out


def _tune_report_split(order: Sequence[int], tune_fraction: float, seed: int
                       ) -> Tuple[List[int], List[int]]:
    """The tuning and report fields of the threshold tuners, drawn at random.

    :param order: the fields with cells, in their fixed order (site order
        within a well; well, then site, over a dataset).
    :param tune_fraction: the share of the fields that choose.
    :param seed: the split's seed (``numpy.random.default_rng``).
    :returns: ``(tune, report)``, each in the order of ``order``.
    """
    n = len(order)
    perm = np.random.default_rng(int(seed)).permutation(n)
    n_tune = int(n * float(tune_fraction))
    return ([order[i] for i in sorted(perm[:n_tune])],
            [order[i] for i in sorted(perm[n_tune:])])


def tune_thresholds(counts, nuclei, sites, axes: Sequence[Tuple[str, Sequence[float]]],
                    *, k: int, scale: float, objective: str = "recall_at_fdr",
                    fdr_max: float = 0.1, f_beta: float = 1.0,
                    unreachable: str = "lowest_fdr", ci: bool = True,
                    tune_fraction: float = 0.5, seed: int = 0) -> Dict[str, Any]:
    """Choose one well's thresholds jointly from its decoy calls (item 942).

    The shared tuner of the OPS decode (``ops_decode_threshold_mode``
    'auto') and the spacer-prefix decode (``ops_prefix_threshold_tuning``
    'per_well_decoy'). Each axis is one threshold, spot detection or
    calling, with its values ordered from the loosest to the strictest;
    every combination is one grid setting. The well's fields with cells, in
    site order, are split at random (``numpy.random.default_rng(seed)``)
    into a tuning share and a report share, as :func:`choose_well_thresholds`
    splits them. Only the tuning share chooses, and only decoy calls are
    read, never a reference call set.

    With ``objective`` 'recall_at_fdr' a setting passes when its decoy FDR
    -- the upper end of the 95 % field-bootstrap interval with ``ci``, else
    the estimate; a share with no decoy call is judged as holding three --
    is at most ``fdr_max``, and the passing setting with the most real calls
    wins. With none passing, ``unreachable`` 'lowest_fdr' takes the setting
    with the lowest such FDR (then the most real calls) and 'strictest' the
    last value of every axis; ``auto_met`` is then False. With 'f_score' the
    setting with the highest F-score of precision (one minus the FDR
    estimate) and recall (the genotyped share) wins. Ties go to the stricter
    setting, the later axes deciding last.

    :param counts: ``(fields, *axis lengths, 2)`` real and decoy calls per
        field and setting.
    :param nuclei: cells per field, the genotyped share's denominator.
    :param sites: the site of each field.
    :param axes: ``(name, values)`` per axis, in the order of ``counts``.
    :param k: decoys per guide.
    :param scale: the FDR calibration (:func:`fdr`).
    :param objective: a name in :data:`TUNING_OBJECTIVES`.
    :param fdr_max: the most false calls per 100 real calls accepted.
    :param f_beta: how much more recall counts than precision in the F-score.
    :param unreachable: a name in :data:`TUNING_UNREACHABLE`.
    :param ci: judge the interval's upper end rather than the estimate.
    :param tune_fraction: the share of the fields that choose.
    :param seed: the split's seed.
    :returns: ``chosen`` (axis name -> value), ``auto_met``, ``objective``,
        ``fdr_max``, ``f_beta``, ``unreachable``, ``ci``, ``tune_fraction``,
        ``seed``, ``fdr_lowest`` (the lowest FDR bound any setting reached
        on the tuning share), ``tune_sites`` and ``report_sites``; ``tune``,
        ``report`` and ``all_fields`` -- cells, real and decoy calls,
        genotyped %, FDR and its interval at the chosen setting -- and
        ``grid``: one row per setting with its axis values, the tuning
        share's numbers, ``fdr_bound``, ``f_score``, ``passes``, ``pareto``
        (on the genotyped-vs-FDR front) and ``chosen``.
    :raises ValueError: when the shapes disagree, ``k`` is below 1, or the
        objective or the unreachable rule is unknown.
    """
    if objective not in TUNING_OBJECTIVES:
        raise ValueError(f"the tuning objective must be one of "
                         f"{list(TUNING_OBJECTIVES)}; got {objective!r}")
    if unreachable not in TUNING_UNREACHABLE:
        raise ValueError(f"the unreachable rule must be one of "
                         f"{list(TUNING_UNREACHABLE)}; got {unreachable!r}")
    names = [str(name) for name, _ in axes]
    values = [[float(v) for v in vals] for _, vals in axes]
    counts = np.asarray(counts, np.int64)
    nuclei = np.asarray(nuclei, np.int64)
    sites = np.asarray(sites, np.int64)
    shape = tuple(len(v) for v in values)
    if counts.ndim != len(shape) + 2 or counts.shape[1:] != shape + (2,) \
            or len(nuclei) != len(counts) or len(sites) != len(counts):
        raise ValueError("counts must be (fields, *axis lengths, 2) with one "
                         "cell count and site per field")
    if int(k) < 1:
        raise ValueError("choosing thresholds needs decoys (k of 1 or more)")
    order = [i for i in np.argsort(sites, kind="stable") if nuclei[i] > 0]
    tune, report = _tune_report_split(order, tune_fraction, seed)
    chosen = tuple(len(v) - 1 for v in values)
    met, lowest = False, None
    rows: List[Dict[str, Any]] = []
    if len(tune) >= 2:
        score = _grid_score(counts, nuclei, tune, k, scale, ci)
        passing = score["bound"] <= float(fdr_max)
        f_scores = _f_score(score["fdr"], score["genotyped_pct"], f_beta)
        lowest = float(score["bound"].min())
        best = fallback = None
        for position in np.ndindex(*shape):
            real = int(score["real"][position])
            rows.append({**{name: values[a][position[a]] for a, name in enumerate(names)},
                         "real": real, "decoy": int(score["decoy"][position]),
                         "genotyped_pct": float(score["genotyped_pct"][position]),
                         "fdr_per_100": float(score["fdr"][position]),
                         "fdr_ci95": [float(score["fdr_lo"][position]),
                                      float(score["fdr_hi"][position])],
                         "fdr_bound": float(score["bound"][position]),
                         "f_score": float(f_scores[position]),
                         "passes": bool(passing[position])})
            if objective == "f_score":
                key = (float(f_scores[position]),) + position
            elif passing[position]:
                key = (real,) + position
            else:
                key = None
            if key is not None and (best is None or key > best):
                best = key
            low = (-float(score["bound"][position]), real) + position
            if fallback is None or low > fallback:
                fallback = low
        if best is not None:
            chosen, met = tuple(best[1:]), True
        elif unreachable == "lowest_fdr":
            chosen = tuple(fallback[2:])
        front = _pareto([r["fdr_bound"] for r in rows], [r["real"] for r in rows])
        flat_chosen = int(np.ravel_multi_index(chosen, shape))
        for i, row in enumerate(rows):
            row["pareto"] = bool(front[i])
            row["chosen"] = i == flat_chosen

    flat_counts = counts.reshape(len(counts), 1, -1, 2)
    flat_position = (0, int(np.ravel_multi_index(chosen, shape)))
    parts = {part: _chosen_summary(flat_counts, nuclei, fields, flat_position, k, scale)
             for part, fields in (("tune", tune), ("report", report),
                                  ("all_fields", order))}
    return {"chosen": {name: values[a][chosen[a]] for a, name in enumerate(names)},
            "index": [int(i) for i in chosen], "auto_met": bool(met),
            "objective": objective, "fdr_max": float(fdr_max),
            "f_beta": float(f_beta), "unreachable": unreachable, "ci": bool(ci),
            "tune_fraction": float(tune_fraction), "seed": int(seed),
            "fdr_lowest": lowest, "axes": {name: values[a] for a, name in enumerate(names)},
            "tune_sites": [int(sites[i]) for i in tune],
            "report_sites": [int(sites[i]) for i in report],
            **parts, "grid": rows}


#: The FDR calibrations ``ops_prefix_fdr_calibration`` and
#: ``ops_decode_fdr_calibration`` may name. Defaults live in the settings;
#: none keeps the entered scale, shuffle only raises it, and never measures
#: it from substitution controls (items 949 and 1014).
FDR_CALIBRATIONS = ("none", "shuffle", "never", "never+shuffle")

#: The S2 (c) check item 949 calibrates to: the barcode-shuffle call rate
#: (shuffle-called cells per 100 cells) may be at most this many times the
#: decoy FDR (false calls per 100 real calls).
SHUFFLE_FDR_RATIO = 1.5

#: How many shuffled whitelists the calibration matches (seeds ``s`` to
#: ``s + 2`` from the control seed); their mean call count is used.
SHUFFLE_SEEDS = 3


def shuffle_call_rate(shuffled, cells, *, reps: int = _AUTO_REPS,
                      seed: int = _AUTO_BOOT_SEED) -> Dict[str, Any]:
    """The barcode-shuffle call rate over some fields, with its interval (item 949).

    The shuffle control matches every read against whitelists whose keys
    had their letters shuffled; a cell called on one is a call a random
    library of the same size and base composition would make.

    :param shuffled: shuffle-called cells per field (a mean over shuffled
        whitelists may be fractional).
    :param cells: cells per field, the rate's denominator.
    :param reps: field-bootstrap replicates of the 95 % interval.
    :param seed: the bootstrap seed.
    :returns: ``shuffle_called`` (the sum), ``cells``,
        ``shuffle_call_rate_pct`` (per 100 cells) and ``shuffle_ci95`` (the
        2.5th and 97.5th percentiles of the field bootstrap; None for fewer
        than two fields).
    """
    shuffled = np.asarray(shuffled, float).reshape(-1)
    cells = np.asarray(cells, float).reshape(-1)
    total = float(cells.sum())
    out = {"shuffle_called": float(shuffled.sum()), "cells": int(total),
           "shuffle_call_rate_pct": 100.0 * float(shuffled.sum()) / max(total, 1.0),
           "shuffle_ci95": None}
    if len(cells) > 1:
        draws = _bootstrap_rows(len(cells), reps, seed)
        weights = np.stack([np.bincount(row, minlength=len(cells)) for row in draws])
        boot = 100.0 * (weights @ shuffled) / np.maximum(weights @ cells, 1.0)
        out["shuffle_ci95"] = [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))]
    return out


def _needed_scale(real: np.ndarray, decoy: np.ndarray, shuffled: np.ndarray,
                  cells: np.ndarray, k: int, ci: bool) -> float:
    """The FDR scale that brings the decoy FDR up to the shuffle call rate over 1.5.

    :param real: real calls per field at one setting.
    :param decoy: decoy calls per field at that setting.
    :param shuffled: shuffle-called cells per field at that setting.
    :param cells: cells per field.
    :param k: decoys per guide.
    :param ci: take the upper end of the 95 % field-bootstrap interval
        (:data:`_AUTO_REPS` resamples, seed :data:`_AUTO_BOOT_SEED`) rather
        than the estimate.
    :returns: ``rate / (1.5 * FDR at scale 1)``; a sum with no decoy call
        is taken as :data:`_AUTO_ZERO_DECOYS`, as the tuner judges it.
    """
    stack = np.stack([np.asarray(v, float).reshape(-1)
                      for v in (real, decoy, shuffled, cells)])

    def ratio(sums: np.ndarray) -> np.ndarray:
        """The needed scale of summed (real, decoy, shuffled, cells) rows."""
        r, d, s, c = sums
        d = np.where(d > 0, d, float(_AUTO_ZERO_DECOYS))
        fdr1 = 100.0 / float(k) * d / np.maximum(r, 1.0)
        return 100.0 * s / np.maximum(c, 1.0) / (SHUFFLE_FDR_RATIO * fdr1)

    point = float(ratio(stack.sum(axis=1)))
    if not ci or stack.shape[1] < 2:
        return point
    draws = _bootstrap_rows(stack.shape[1], _AUTO_REPS, _AUTO_BOOT_SEED)
    weights = np.stack([np.bincount(row, minlength=stack.shape[1]) for row in draws])
    return float(np.percentile(ratio(stack @ weights.T), 97.5))


def never_scale(real, decoy, never2_calls, opp2_N2, opp2_L, *, k,
                ci=True, reps=_AUTO_REPS, seed=_AUTO_BOOT_SEED,
                never1_calls=None, opp1_N1=None, opp1_L=None):
    """Estimate the decoy FDR scale from calls on substitution-built controls.

    The N2 call rate per distance-two opportunity predicts wrong library
    calls at that distance. Libraries containing distance-one guide pairs
    also contribute the N1 rate times their distance-one opportunities.
    The expected wrong calls times k, divided by the observed decoy calls,
    give the scale. A zero decoy sum uses ``_AUTO_ZERO_DECOYS``, as the
    existing threshold tuner does. With ci, the chosen scale is the upper
    end of the 95 percent field bootstrap, using the existing row sampler.
    Clipping the chosen scale to one belongs to ``calibrate_fdr_scale``.

    :param real: real production calls per field at one grid setting.
    :param decoy: production decoy calls in the same fields.
    :param never2_calls: calls on the distance-two control set.
    :param opp2_N2: production guide counts times their N2 distance-two neighbours.
    :param opp2_L: the corresponding other-library distance-two opportunities.
    :param k: positive matched-decoy multiplicity.
    :param ci: choose the upper bootstrap scale rather than its point estimate.
    :param reps: bootstrap draws, default the existing ``_AUTO_REPS``.
    :param seed: field-bootstrap seed, default ``_AUTO_BOOT_SEED``.
    :param never1_calls: optional distance-one control calls per field.
    :param opp1_N1: N1 distance-one opportunities; supplied with never1_calls.
    :param opp1_L: other-library distance-one opportunities, otherwise zero.
    :returns: dict with ``scale``, ``scale_point``, ``scale_ci95``, ``rho1``,
        ``rho2``, ``rho1_over_rho2``, ``fdr_cal_pct``, their ``*_ci95``
        intervals, ``expected_wrong`` and summed real/decoy/control counts.
        Intervals are None for fewer than two fields. Rates with no
        control opportunities and estimates depending on them are None;
        a ratio with zero rho2 is also unavailable.
    :raises ValueError: for misaligned, negative or nonfinite counts, a
        nonpositive multiplicity, or incomplete N1 inputs.
    """
    from .qt.i18n import tr

    first = np.asarray(real, float).reshape(-1)
    if (never1_calls is None) != (opp1_N1 is None):
        raise ValueError(tr('Never-barcode calibration needs both N1 calls and N1 opportunities.'))
    has_n1 = never1_calls is not None
    values = (real, decoy, never2_calls, opp2_N2, opp2_L,
              np.zeros_like(first) if never1_calls is None else never1_calls,
              np.zeros_like(first) if opp1_N1 is None else opp1_N1,
              np.zeros_like(first) if opp1_L is None else opp1_L)
    arrays = [np.asarray(value, float).reshape(-1) for value in values]
    if (any(value.shape != first.shape for value in arrays)
            or any(not np.all(np.isfinite(value) & (value >= 0)) for value in arrays)):
        raise ValueError(tr('Never-barcode calibration needs aligned, finite, nonnegative field counts.'))
    if not np.isfinite(float(k)) or float(k) <= 0:
        raise ValueError(tr('Never-barcode calibration needs a positive decoy multiplicity.'))
    if not has_n1 and np.any(arrays[7]):
        raise ValueError(tr('Distance-one library neighbours require N1 calls and opportunities.'))
    stack = np.stack(arrays)

    def estimates(sums):
        """Compute scale, both rates, their ratio and wrong-call FDR from field sums."""
        r, d, n2, o2, library2, n1, o1, library1 = sums
        rho2 = np.divide(n2, o2, out=np.full_like(n2, np.nan), where=o2 > 0)
        rho1 = np.divide(n1, o1, out=np.full_like(n1, np.nan), where=o1 > 0)
        wrong = (np.where(library2 > 0, rho2 * library2, 0.0)
                 + np.where(library1 > 0, rho1 * library1, 0.0))
        scale = float(k) * wrong / np.where(d > 0, d, float(_AUTO_ZERO_DECOYS))
        ratio = np.divide(rho1, rho2, out=np.full_like(rho2, np.nan), where=rho2 > 0)
        return np.stack((scale, rho1, rho2, ratio, 100 * wrong / np.maximum(r, 1), wrong))

    point = estimates(stack.sum(axis=1))
    names = ('scale', 'rho1', 'rho2', 'rho1_over_rho2', 'fdr_cal_pct', 'expected_wrong')
    out = {name: float(value) if np.isfinite(value) else None for name, value in zip(names, point)}
    out.update({name + '_ci95': None for name in names[:-1]})
    out['scale_point'] = out['scale']
    if ci and first.size > 1:
        draws = _bootstrap_rows(first.size, reps, seed)
        weights = np.stack([np.bincount(row, minlength=first.size) for row in draws])
        boot = estimates(stack @ weights.T)
        for index, name in enumerate(names[:-1]):
            finite = boot[index][np.isfinite(boot[index])]
            if finite.size:
                low, high = np.percentile(finite, (2.5, 97.5))
                if out[name] is not None:
                    low, high = min(float(low), out[name]), max(float(high), out[name])
                out[name + '_ci95'] = [float(low), float(high)]
        out['scale'] = out['scale_ci95'][1] if out['scale_ci95'] else out['scale_point']
    if not has_n1:
        for name in ('rho1', 'rho1_ci95', 'rho1_over_rho2', 'rho1_over_rho2_ci95'):
            out[name] = None
    out.update(real=float(stack[0].sum()), decoy=float(stack[1].sum()),
               never2_calls=float(stack[2].sum()), never1_calls=float(stack[5].sum()) if has_n1 else None)
    return out


def calibrate_fdr_scale(counts, shuffled, nuclei, sites,
                        axes: Sequence[Tuple[str, Sequence[float]]], *, k: int,
                        scale: float, ci: bool = True, max_rounds: int = 20,
                        method: str = 'shuffle', never=None, **options: Any) -> Dict[str, Any]:
    """Tune one well with a scale measured by shuffle or never-barcode controls.

    The existing shuffle method only raises the entered scale until its
    call-rate check passes. Never calibration measures wrong library calls
    from substitution-control opportunities and may lower or raise the
    scale, with a floor of one. Both methods use this loop and the same
    threshold tuner. A repeated never setting ends the loop; a cycle keeps
    the larger tested cycling scale and its matching threshold result.
    The report half is held out from never calibration.

    :param counts: ``(fields, *axis lengths, 2)`` real and decoy calls.
    :param shuffled: matching shuffle-call grid, or None without shuffle.
    :param nuclei: cells per field.
    :param sites: each field site.
    :param axes: axis names and values passed to ``tune_thresholds``.
    :param k: positive matched-decoy multiplicity.
    :param scale: entered scale; shuffle cannot lower it, never can.
    :param ci: choose the upper field-bootstrap scale and tuner intervals.
    :param max_rounds: maximum tuner passes, default 20.
    :param method: none, shuffle (the compatibility default), never or
        never+shuffle, which takes the larger of the two measured scales.
    :param never: mapping of ``never2_calls``, ``opp2_N2``, ``opp2_L`` and
        optional ``never1_calls``, ``opp1_N1``, ``opp1_L`` field/grid arrays.
        Entry points also supply ``library_size`` for the 50-guide guard,
        ``never_sets`` metadata and optional N1/N2 real-call grids.
        Fewer than 30 tuning-half N2 calls keep the entered scale.
    :param options: other ``tune_thresholds`` keywords.
    :returns: the tuner result with ``fdr_scale`` and ``calibration``.
        Shuffle keeps its existing ratio, per-part call-rate checks and
        round structure. Never adds point/interval estimates for each
        half, entered-scale FDR, status, control metadata and perturbation
        changes. Production calls are never replaced by control calls.
    :raises ValueError: for an unknown method, invalid grid shape, missing
        never arrays or fewer than one permitted tuner pass.
    """
    from .qt.i18n import tr

    method = str(method).lower()
    if method not in FDR_CALIBRATIONS:
        raise ValueError(tr('Unknown FDR calibration method: {method}.').format(method=method))
    if int(max_rounds) < 1:
        raise ValueError(tr('FDR calibration needs at least one tuning round.'))
    using_never = method in ('never', 'never+shuffle')
    counts = np.asarray(counts, np.int64)
    has_shuffle = shuffled is not None
    shuffled = np.asarray(shuffled, float) if has_shuffle else np.zeros(counts.shape[:-1])
    nuclei = np.asarray(nuclei, np.int64)
    if shuffled.shape != counts.shape[:-1] or (method in ('shuffle', 'never+shuffle') and not has_shuffle):
        raise ValueError('shuffled must be (fields, *axis lengths), the grid of counts')
    payload = dict(never or {})
    array_keys = ('never2_calls', 'opp2_N2', 'opp2_L', 'never1_calls', 'opp1_N1', 'opp1_L',
                  'real_calls_n1', 'real_calls_n2')
    arrays = {key: np.asarray(payload[key], float) for key in array_keys if key in payload}
    if using_never and (not {'never2_calls', 'opp2_N2', 'opp2_L'} <= arrays.keys()
                        or any(value.shape != counts.shape[:-1] for value in arrays.values())):
        raise ValueError(tr('Never-barcode calibration needs complete control arrays matching the field grid.'))

    def never_at(at, selected_ci):
        """Estimate exactly one selected grid point over the requested field half."""
        keywords = {key: value[at] for key, value in arrays.items()
                    if key in ('never1_calls', 'opp1_N1', 'opp1_L')}
        return never_scale(counts[at][:, 0], counts[at][:, 1], arrays['never2_calls'][at],
                           arrays['opp2_N2'][at], arrays['opp2_L'][at], k=k,
                           ci=selected_ci, **keywords)

    position_of = {int(site): i for i, site in enumerate(np.asarray(sites, np.int64))}
    entered = max(float(scale), 1.0)
    current, rounds, converged = entered, [], False
    choice: Dict[str, Any] = {}
    history, seen = [], {}
    status, cycle, reason = 'ok', False, None
    for iteration in range(int(max_rounds)):
        choice = tune_thresholds(counts, nuclei, sites, axes, k=k, scale=current,
                                 ci=ci, **options)
        fields = [position_of[s] for s in choice['tune_sites']]
        if not using_never and not fields:
            fields = [position_of[s] for s in choice['report_sites']]
        at = (np.asarray(fields, np.int64),) + tuple(choice['index'])
        if using_never:
            measured = never_at(at, ci)
            if 'library_size' in payload and int(payload['library_size']) < 50:
                status, reason = 'insufficient_never_calls', 'insufficient_library_guides'
            elif measured['never2_calls'] < 30:
                status = 'insufficient_never_calls'
            elif measured['scale'] is None:
                status = 'insufficient_never_opportunities'
            needed = entered if status != 'ok' else max(float(measured['scale']), 1.0)
            if method == 'never+shuffle' and status == 'ok':
                needed = max(needed, _needed_scale(counts[at][:, 0], counts[at][:, 1],
                                                   shuffled[at], nuclei[fields], k, ci))
        elif method == 'shuffle':
            needed = (_needed_scale(counts[at][:, 0], counts[at][:, 1], shuffled[at],
                                    nuclei[fields], k, ci) if fields else 1.0)
        else:
            needed = current
        rounds.append({'scale': current, 'chosen': choice['chosen'], 'needed_scale': needed})
        if using_never:
            signature = tuple(choice['index'])
            if signature in seen:
                previous = seen[signature]
                if previous == len(history) - 1:
                    converged = True
                else:
                    cycle = True
                    current, choice = max(history[previous:] + [(current, choice)], key=lambda row: row[0])
                break
            seen[signature] = len(history)
            history.append((current, choice))
            if status != 'ok' or abs(needed - current) <= 1e-9 * max(current, 1.0):
                converged = True
                break
            if iteration + 1 == int(max_rounds):
                break
        elif needed <= current * (1.0 + 1e-9):
            converged = True
            break
        current = float(needed)
    parts = {}
    for part, part_sites in (('tune', choice['tune_sites']),
                             ('report', choice['report_sites']),
                             ('all_fields', sorted(choice['tune_sites'] + choice['report_sites']))):
        idx = np.asarray([position_of[s] for s in part_sites], np.int64)
        at = (idx,) + tuple(choice['index'])
        fdr_part = choice[part]['fdr_per_100']
        row = {'fdr_per_100': fdr_part}
        if has_shuffle:
            rate = shuffle_call_rate(shuffled[at], nuclei[idx])
            limit = None if fdr_part is None else SHUFFLE_FDR_RATIO * float(fdr_part)
            row = {**rate, 'fdr_per_100': fdr_part, 'shuffle_limit_pct': limit,
                   'passes': None if limit is None else bool(rate['shuffle_call_rate_pct'] <= limit * (1.0 + 1e-9))}
        if using_never:
            row.update(never_at(at, True))
            real = float(counts[at][:, 0].sum())
            decoy = float(counts[at][:, 1].sum())
            row['fdr_at_entered_scale_pct'] = fdr(int(real), int(decoy), k, entered) if idx.size else None
            row['real_call_change_pct'] = {
                name: 100.0 * (float(arrays[key][at].sum()) - real) / max(real, 1.0)
                for name, key in (('n1', 'real_calls_n1'), ('n2', 'real_calls_n2')) if key in arrays}
        parts[part] = row
    calibration = {'method': method}
    if method == 'shuffle':
        calibration['ratio'] = SHUFFLE_FDR_RATIO
    calibration.update(entered_scale=entered, scale=current, rounds=rounds, converged=converged, **parts)
    if using_never:
        tune = parts['tune']
        calibration.update(status=status, reason=reason, cycle=cycle, never_sets=payload.get('never_sets', {}),
                           scale_ci95=tune['scale_ci95'] if status == 'ok' else None,
                           scale_report_half=parts['report']['scale_point'],
                           rho1=tune['rho1'], rho1_ci95=tune['rho1_ci95'],
                           rho2=tune['rho2'], rho2_ci95=tune['rho2_ci95'],
                           rho1_over_rho2=tune['rho1_over_rho2'],
                           rho1_over_rho2_ci95=tune['rho1_over_rho2_ci95'],
                           fdr_cal_pct=tune['fdr_cal_pct'], fdr_cal_pct_ci95=tune['fdr_cal_pct_ci95'],
                           fdr_at_entered_scale_pct=tune['fdr_at_entered_scale_pct'],
                           real_call_change_pct=parts['all_fields']['real_call_change_pct'],
                           fields={'tune': choice['tune_sites'], 'report': choice['report_sites']})
        if has_shuffle:
            calibration['shuffle_call_rate'] = parts['all_fields']['shuffle_call_rate_pct']
    return {**choice, 'fdr_scale': current, 'calibration': calibration}

def write_tuning_curve(choice: Mapping[str, Any], folder: str, name: str,
                       title: str = "") -> Dict[str, str]:
    """Write one well's genotyped-vs-FDR curve and the chosen point.

    :param choice: from :func:`tune_thresholds`.
    :param folder: the folder written into, created.
    :param name: the file stem, for example ``pr_curve_A1``.
    :param title: the figure title.
    :returns: the paths of ``<name>.csv`` (every grid setting's tuning
        numbers, with ``pareto`` and ``chosen``) and ``<name>.png`` (genotyped
        % against the decoy FDR bound, the front drawn as a line, the chosen
        setting marked); no figure without grid rows.
    """
    import pandas as pd

    from .tabular import write_table

    os.makedirs(folder, exist_ok=True)
    rows = [{key: (value if key != "fdr_ci95" else None) for key, value in row.items()}
            for row in choice["grid"]]
    for row, source in zip(rows, choice["grid"]):
        row.pop("fdr_ci95")
        row["fdr_ci_low"], row["fdr_ci_high"] = source["fdr_ci95"]
    paths = {"csv": os.path.join(folder, f"{name}.csv")}
    write_table(pd.DataFrame(rows), paths["csv"], canonicalise=False)
    if not rows:
        return paths
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES, figure_style, resolve_ink
    from .plot import save_figure

    frame = pd.DataFrame(rows)
    front = frame[frame["pareto"]].sort_values("fdr_bound")
    with figure_style("print"):
        figure = Figure(figsize=(5.0, 3.8), dpi=150)
        axis = figure.subplots()
        axis.scatter(frame["fdr_bound"], frame["genotyped_pct"], s=12, color="#7f9cb8",
                     alpha=0.6, label="grid settings")
        axis.plot(front["fdr_bound"], front["genotyped_pct"], color="#1f4e79", lw=1.4,
                  label="front")
        picked = frame[frame["chosen"]]
        axis.scatter(picked["fdr_bound"], picked["genotyped_pct"], s=60, color="#0e8a8a",
                     edgecolor=resolve_ink("print"), zorder=3, label="chosen")
        if choice.get("objective") == "recall_at_fdr":
            axis.axvline(float(choice["fdr_max"]), color=ROLES["reference"], ls="--", lw=0.8,
                         label="FDR target")
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlim(left=0)
        axis.set_xlabel("decoy FDR per 100 calls (tuning fields)")
        axis.set_ylabel("cells genotyped (%)")
        axis.set_title(title or name, fontsize=9)
        axis.legend(fontsize=7, frameon=False)
        _register_figure_data(figure, frame, x="fdr_bound", y="genotyped_pct",
                              hue="chosen", kind="scatter")
        figure.tight_layout()
        paths["png"] = save_figure(figure, os.path.join(folder, f"{name}.png"),
                                   fmt="png", dpi=150, close=True)
    return paths


def _write_calibration_report(folder, plate, well, calibration, *, csv=True, log=print):
    """Write and describe one never-control calibration for either decode path.

    :param folder: existing decode or prefix results folder.
    :param plate: plate identifier.
    :param well: well identifier.
    :param calibration: the shared calibrator report, including guard status.
    :param csv: also write a one-row CSV, as the full-barcode path requires.
    :param log: console logger, or None to suppress the summary.
    :returns: paths keyed by ``fdr_calibration`` and optionally
        ``fdr_calibration_csv``. Nested CSV values are JSON strings.
        Only N2 real-call changes above 0.5 percent emit a warning.
    """
    import warnings
    import pandas as pd

    from .qt.i18n import tr
    from .tabular import write_table

    os.makedirs(folder, exist_ok=True)
    payload = {'plate': plate, 'well': well, **calibration}
    path = os.path.join(folder, f'fdr_calibration_{well}.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=1, allow_nan=False)
    paths = {'fdr_calibration': path}
    if csv:
        row = {key: json.dumps(value, sort_keys=True, allow_nan=False)
               if isinstance(value, (dict, list, tuple)) else value for key, value in payload.items()}
        csv_path = os.path.join(folder, f'fdr_calibration_{well}.csv')
        write_table(pd.DataFrame([row]), csv_path, canonicalise=False)
        paths['fdr_calibration_csv'] = csv_path
    if log is not None:
        scale = format(float(calibration['scale']), '.4g')
        status = calibration.get('status', 'ok')
        if calibration.get('reason') == 'insufficient_library_guides':
            message = tr('{well}: FDR scale kept at {scale}; the library has fewer than 50 guides.').format(well=well, scale=scale)
        elif status == 'insufficient_never_calls':
            message = tr('{well}: FDR scale kept at {scale}; fewer than 30 tuning-half control calls.').format(well=well, scale=scale)
        elif status == 'insufficient_never_opportunities':
            message = tr('{well}: FDR scale kept at {scale}; control opportunities are unavailable at the chosen setting.').format(well=well, scale=scale)
        elif status == 'no_decoys':
            message = tr('{well}: FDR scale kept at {scale}; never-barcode calibration needs a positive decoy multiplicity.').format(well=well, scale=scale)
        else:
            interval = calibration.get('scale_ci95')
            text = ', '.join(format(float(value), '.4g') for value in interval) if interval else tr('not estimated')
            message = tr(
                '{well}: FDR scale calibrated from never-should-appear barcodes: '
                '{entered} entered -> {scale} [{interval}]; FDR at the chosen setting '
                '{fdr} % (entered scale {entered_fdr} %).').format(
                    well=well, entered=format(float(calibration['entered_scale']), '.4g'),
                    scale=scale, interval=text, fdr=format(float(calibration['fdr_cal_pct']), '.4g'),
                    entered_fdr=format(float(calibration['fdr_at_entered_scale_pct']), '.4g'))
            if calibration.get('cycle'):
                message += ' ' + tr('Conservative scale retained after a threshold cycle.')
        log(message)
    change = calibration.get('real_call_change_pct', {}).get('n2')
    if change is not None and abs(float(change)) > 0.5:
        warnings.warn(tr(
            '{well}: N2 control voting changed real calls by {change} %, above the 0.5 % perturbation guard; '
            'production calls were kept.').format(well=well, change=format(float(change), '.4g')),
            RuntimeWarning, stacklevel=2)
    return paths


def _write_threshold_report(db: str, plate: str, well: str,
                            choice: Mapping[str, Any]) -> Dict[str, str]:
    """Write the chosen thresholds for one well and update the plate table.

    :param db: the measurements database.
    :param plate: the plate.
    :param well: the well.
    :param choice: from :func:`choose_well_thresholds`, or the shared
        tuner choice; with calibration the table row adds the applied and entered
        scales, method and whichever control rates were measured.
    :returns: the paths of ``decode_thresholds_<well>.json`` and
        ``decode_thresholds_<plate>.csv`` under ``results/ops_decode``; the
        table holds one row per well, this well's replaced.
    """
    import pandas as pd

    from .tabular import read_table, write_table

    choice = {**choice.get('chosen', {}), **choice}
    folder = results_folder(db)
    path = os.path.join(folder, f"decode_thresholds_{well}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"plate": plate, "well": well, **choice}, handle, indent=1)
    row = {"plate": plate, "well": well,
           "single_read_quality": choice["single_read_quality"],
           "pixel_quality": choice["pixel_quality"],
           "auto_met": choice["auto_met"], "fdr_max": choice["fdr_max"]}
    if "read_threshold" in choice:
        row.update({"read_threshold": choice["read_threshold"],
                    "objective": choice["objective"]})
    if "calibration" in choice:
        calibration = choice['calibration']
        row.update(fdr_scale=choice['fdr_scale'])
        whole = calibration.get('all_fields', {})
        if 'shuffle_call_rate_pct' in whole:
            row['shuffle_call_rate_pct'] = whole['shuffle_call_rate_pct']
        if str(calibration['method']).startswith('never'):
            row.update(fdr_calibration=calibration['method'],
                       fdr_scale_entered=calibration['entered_scale'])
            row.update({key: calibration[key] for key in ('rho1', 'rho2', 'fdr_cal_pct') if key in calibration})
    if "read_mode" in choice:
        row["read_mode"] = choice["read_mode"]
    for part in ("tune", "report", "all_fields"):
        numbers = choice[part]
        interval = numbers["fdr_ci95"] or [None, None]
        row.update({f"{part}_fields": numbers["fields"],
                    f"{part}_nuclei": numbers["nuclei"],
                    f"{part}_genotyped_pct": numbers["genotyped_pct"],
                    f"{part}_fdr_per_100": numbers["fdr_per_100"],
                    f"{part}_fdr_ci_low": interval[0],
                    f"{part}_fdr_ci_high": interval[1]})
    table_path = os.path.join(folder, f"decode_thresholds_{plate}.csv")
    rows = [row]
    if os.path.isfile(table_path):
        old = read_table(table_path, canonicalise=False, report=None)
        rows = [r for r in old.to_dict("records") if str(r["well"]) != str(well)] + rows
    table = pd.DataFrame(rows).sort_values("well", kind="stable")
    write_table(table, table_path, canonicalise=False)
    return {"thresholds": path, "plate_table": table_path}


#: The columns of :func:`field_calls`'s evidence rows, after ``well`` and
#: ``site``: the per-call read support of the decode-yield study's
#: ``decode_diag.py``.
EVIDENCE_COLUMNS = ("object_id", "mapped_guide", "is_decoy", "source",
                    "n_reads", "n_matched", "n_guides_seen", "n_exact",
                    "n_corr", "vote_top", "vote_total", "vote_frac",
                    "runner_margin", "q_best", "q_mean", "strength_max",
                    "n_amb", "pix_n", "pix_total", "pix_frac",
                    "field_reads_per_nucleus", "field_nuclei")


def _evidence_rows(primary, pixel, tallies, owned, decoys, objects,
                   ambiguous) -> List[Dict[str, Any]]:
    """One evidence row per owned call, primary first where both exist.

    :param primary: from :func:`spacr.ops_sbs.vote_mapped_reads`.
    :param pixel: from :func:`spacr.ops_sbs.vote_pixels`.
    :param tallies: the vote evidence the primary vote filled.
    :param owned: the objects this field owns.
    :param decoys: the decoy set.
    :param objects: the object of every owned read.
    :param ambiguous: whether each owned read lay near two nuclei.
    :returns: the rows, in the study's order (the set of primary and pixel
        objects).
    """
    from collections import Counter

    reads = Counter(objects.tolist())
    near_two = Counter(objects[ambiguous].tolist())
    density = len(objects) / max(1, len(owned))
    votes, exact, corrected = tallies["votes"], tallies["exact"], tallies["corrected"]
    best, qualities, strength = tallies["best"], tallies["qualities"], tallies["strength"]
    rows = []
    for owner in set(primary) | set(pixel):
        if owner not in owned:
            continue
        if owner in primary:
            hit = primary[owner]
            guide, top, total = hit["guide"], hit["vote_top"], hit["vote_total"]
            margin, seen, source = hit["runner_margin"], hit["n_guides"], "primary"
        else:
            hit = pixel[owner]
            guide, top, total, margin = hit["guide"], 0.0, 0.0, hit["runner_margin"]
            seen = len(votes.get(owner, {}))
            source = "pixel_fallback"
        fallback = pixel.get(owner)
        has_guide = owner in best and guide in best[owner]
        rows.append({
            "object_id": int(owner), "mapped_guide": guide,
            "is_decoy": guide in decoys, "source": source,
            "n_reads": int(reads.get(owner, 0)),
            "n_matched": int(tallies["matched"].get(owner, 0)),
            "n_guides_seen": int(seen),
            "n_exact": int(exact[owner].get(guide, 0)) if owner in exact else 0,
            "n_corr": int(corrected[owner].get(guide, 0)) if owner in corrected else 0,
            "vote_top": float(top), "vote_total": float(total),
            "vote_frac": float(top / total) if total else np.nan,
            "runner_margin": float(margin),
            "q_best": float(best[owner][guide]) if has_guide else np.nan,
            "q_mean": (float(np.mean(qualities[owner][guide]))
                       if owner in qualities and guide in qualities[owner] else np.nan),
            "strength_max": (float(strength[owner][guide])
                             if owner in strength and guide in strength[owner] else np.nan),
            "n_amb": int(near_two.get(owner, 0)),
            "pix_n": int(fallback["n_pixels"]) if fallback and fallback["guide"] == guide else 0,
            "pix_total": int(fallback["n_matched"]) if fallback else 0,
            "pix_frac": (float(fallback["n_pixels"] / fallback["n_matched"])
                         if fallback and fallback["guide"] == guide else 0.0),
            "field_reads_per_nucleus": float(density), "field_nuclei": len(owned),
        })
    return rows


def fdr(real: int, decoy: int, k: float, scale: float) -> float:
    """The decoy-estimated false calls per 100 real calls.

    :param real: real calls.
    :param decoy: decoy calls.
    :param k: decoys per guide.
    :param scale: the calibration of a decoy match to a wrong real match
        (1.268 for the Funk et al. 2022 library); never below 1, so a decoy
        shortfall is never credited.
    :returns: ``100 * max(scale, 1) / k * decoy / max(real, 1)``; NaN without
        decoys.
    """
    if not k:
        return float("nan")
    return 100.0 * max(float(scale), 1.0) / float(k) * decoy / max(int(real), 1)


def _fdr_by_source(field_stats, real: int, decoy: int, k: int,
                   scale: float) -> Dict[str, Any]:
    """The decoy FDR report of the read calls and of the pixel-fallback calls apart.

    The pixel fallback calls nuclei the reads did not, at a decoy rate of
    its own, so the well report counts it apart from the read calls.

    :param field_stats: the field statistics of :func:`fdr_report`.
    :param real: their real calls.
    :param decoy: their decoy calls.
    :param k: decoys per guide.
    :param scale: the FDR calibration (:func:`fdr`).
    :returns: ``assigned_primary``, ``assigned_primary_decoy`` and
        ``fdr_per_100_primary`` for the read calls, and
        ``assigned_pixel_fallback``, ``assigned_pixel_fallback_decoy`` and
        ``fdr_per_100_pixel_fallback`` for the pixel-fallback calls; empty
        when the statistics have no ``assigned_pixel_real`` and
        ``assigned_pixel_decoy`` columns, as for the prefix decode.
    """
    if not {"assigned_pixel_real", "assigned_pixel_decoy"} <= set(field_stats.columns):
        return {}
    pixel_real = int(field_stats["assigned_pixel_real"].sum()) if len(field_stats) else 0
    pixel_decoy = int(field_stats["assigned_pixel_decoy"].sum()) if len(field_stats) else 0
    out: Dict[str, Any] = {}
    for name, part_real, part_decoy in (("primary", real - pixel_real, decoy - pixel_decoy),
                                        ("pixel_fallback", pixel_real, pixel_decoy)):
        out[f"assigned_{name}"] = part_real
        out[f"assigned_{name}_decoy"] = part_decoy
        out[f"fdr_per_100_{name}"] = round(fdr(part_real, part_decoy, k, scale), 3) if k else None
    return out


def fdr_report(field_stats, *, plate: str, well: str, k: int, scale: float,
               skipped_sites: Sequence[int] = (), calibration=None) -> Dict[str, Any]:
    """The decoy FDR report for one well.

    :param field_stats: one row per decoded field with ``nuclei``,
        ``assigned_real`` and ``assigned_decoy``.
    :param plate: the plate.
    :param well: the well.
    :param k: decoys per guide.
    :param scale: the FDR calibration (:func:`fdr`).
    :param skipped_sites: the sites not decoded because the stitch did not
        place them.
    :param calibration: optional shared calibration report; supplied only
        after controls were measured. Available shuffle fields retain
        their existing names. Never calibration adds its full report,
        point estimates and confidence intervals.
    :returns: the report: nuclei, assigned, assigned %, decoy calls, FDR per
        100, and the skipped sites.
    """
    nuclei = int(field_stats["nuclei"].sum()) if len(field_stats) else 0
    real = int(field_stats["assigned_real"].sum()) if len(field_stats) else 0
    decoy = int(field_stats["assigned_decoy"].sum()) if len(field_stats) else 0
    report = {"plate": plate, "well": well, "fields": int(len(field_stats)),
            "nuclei": nuclei, "assigned": real, "assigned_decoy": decoy,
            "pct": round(100 * real / max(nuclei, 1), 2),
            "fdr_per_100": round(fdr(real, decoy, k, scale), 3) if k else None,
            "decoy_library_k": int(k), "fdr_scale": float(scale),
            "skipped_sites": sorted(int(s) for s in skipped_sites)}
    report.update(_fdr_by_source(field_stats, real, decoy, k, scale))

    if calibration:
        whole = calibration.get('all_fields', {})
        report.update(fdr_calibration=calibration['method'], fdr_scale_entered=calibration['entered_scale'])
        report.update({key: whole[key] for key in ('shuffle_called', 'shuffle_call_rate_pct',
                                                  'shuffle_ci95', 'shuffle_limit_pct') if key in whole})
        if 'passes' in whole:
            report['shuffle_le_limit'] = whole['passes']
        if (str(calibration['method']).startswith('never')
                or str(calibration.get('requested_method', '')).startswith('never')):
            report['calibration'] = dict(calibration)
            report.update({key: calibration[key] for key in ('scale_ci95', 'scale_report_half',
                           'rho1', 'rho1_ci95', 'rho2', 'rho2_ci95', 'rho1_over_rho2',
                           'rho1_over_rho2_ci95', 'fdr_cal_pct', 'fdr_cal_pct_ci95',
                           'fdr_at_entered_scale_pct') if key in calibration})
    return report


def plate_fdr_summary(folder: str, *, plate: str, k: int, scale: float,
                      reps: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """Assigned percentages and FDR across all field statistics for the plate.

    Per-well never reports provide their applied scales; the plate FDR
    and its existing field bootstrap use those same scales. Fixed and
    shuffle-only plates retain their previous summary calculation.

    :param folder: the ``results/ops_decode`` folder holding
        ``field_stats_<well>.csv`` files.
    :param plate: the plate.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :param reps: field-bootstrap replicates for the 95 % FDR interval.
    :param seed: the bootstrap seed.
    :returns: the summary, with per-well rows; empty wells when no stats
        were written.
    """
    import pandas as pd

    from .tabular import read_table

    names = sorted(name for name in os.listdir(folder)
                   if name.startswith("field_stats_") and name.endswith(".csv"))
    if not names:
        return {"plate": plate, "wells": {}}
    stats = pd.concat([read_table(os.path.join(folder, name), canonicalise=False,
                                  report=None) for name in names], ignore_index=True)
    real = stats["assigned_real"].to_numpy()
    decoy = stats["assigned_decoy"].to_numpy()
    records = {}
    for well in stats['well'].astype(str).unique():
        report_path = os.path.join(folder, f'decode_fdr_{well}.json')
        if os.path.isfile(report_path):
            with open(report_path, encoding='utf-8') as handle:
                records[well] = json.load(handle)
    calibrated = any(record.get('calibration') for record in records.values())
    scales = np.array([max(float(records.get(str(well), {}).get('fdr_scale', scale)), 1.0)
                       for well in stats['well']]) if calibrated else None
    effective_decoy = decoy * scales if calibrated else decoy
    effective_scale = 1.0 if calibrated else scale
    rng = np.random.default_rng(seed)
    boot = [fdr(real[i].sum(), effective_decoy[i].sum(), k, effective_scale)
            for i in (rng.integers(0, len(stats), len(stats))
                      for _ in range(int(reps)))] if k else []
    summary = {
        "plate": plate, "nuclei_in_decoded_fields": int(stats["nuclei"].sum()),
        "assigned": int(real.sum()), "assigned_decoy": int(decoy.sum()),
        "pct_of_decoded_field_nuclei": round(100 * real.sum() / max(stats["nuclei"].sum(), 1), 2),
        "fdr_per_100": round(fdr(real.sum(), effective_decoy.sum(), k, effective_scale), 3) if k else None,
        "fdr_ci95": ([round(float(np.percentile(boot, 2.5)), 3),
                      round(float(np.percentile(boot, 97.5)), 3)] if boot else None),
        "per_well": {},
    }
    for well, group in stats.groupby("well"):
        record = records.get(str(well), {})
        well_scale = float(record.get('fdr_scale', scale)) if calibrated else scale
        summary["per_well"][str(well)] = {
            "pct": round(100 * group["assigned_real"].sum() / max(group["nuclei"].sum(), 1), 2),
            "fdr_per_100": (round(fdr(group["assigned_real"].sum(),
                                      group["assigned_decoy"].sum(), k, well_scale), 3)
                            if k else None)}
        if record.get('calibration'):
            summary['per_well'][str(well)].update(fdr_report(
                group, plate=plate, well=str(well), k=k, scale=well_scale,
                calibration=record['calibration']))
    return summary


def results_folder(db: str, name: str = "ops_decode") -> str:
    """The folder a decode writes its reports into, created.

    :param db: the measurements database; its folder is ``dst_root``.
    :param name: the subfolder of ``results``.
    :returns: ``<dst_root>/results/<name>``.
    """
    folder = os.path.join(os.path.dirname(os.path.abspath(db)), "results", name)
    os.makedirs(folder, exist_ok=True)
    return folder


#: Tile-bootstrap replicates and seed of the split-read concordance.
SPLIT_READ_REPS = 2000
SPLIT_READ_SEED = 20261125


def _split_reference(split, min_reads: int = 1):
    """The nuclei whose other-tile reads name one guide (:func:`split_read_concordance`).

    :param split: the other-tile reads, one row per nucleus and tile, with
        ``object_id``, ``n``, ``nguides``, ``guide`` and ``other_site``.
    :param min_reads: the least other-tile reads a reference needs.
    :returns: a frame of ``object_id``, ``n``, ``nguides``, ``guide`` and
        ``tiles``: the references, read in one other tile only, with one
        barcode named and at least ``min_reads`` reads.
    """
    import pandas as pd

    split = pd.DataFrame(split, columns=["object_id", "n", "nguides", "guide", "other_site"])
    reference = split.groupby("object_id").agg(
        n=("n", "sum"), nguides=("nguides", "max"), guide=("guide", "first"),
        tiles=("other_site", "nunique")).reset_index()
    return reference[(reference["n"] >= int(min_reads)) & (reference["nguides"] == 1)
                     & (reference["tiles"] == 1)]


def split_read_concordance(calls, split, sites: Sequence[int], *,
                           reps: int = SPLIT_READ_REPS, seed: int = SPLIT_READ_SEED,
                           min_reads: Sequence[int] = (1, 2)) -> Dict[str, Any]:
    """How often the call of a nucleus agrees with an independent read of it in another tile.

    Neighbouring sequencing tiles overlap, so a nucleus near a tile edge is
    imaged twice: the tile that owns it calls it, and the other holds reads
    of it that played no part in the call (:func:`_split_reads`). A nucleus
    whose other-tile reads come from one tile and name one barcode is a
    reference; its call agrees when it is that guide. The decoy FDR cannot
    see a wrong call that names a real guide; this can.

    :param calls: one row per real call of the tiles scored, with
        ``object_id``, ``site`` (the owning tile) and ``mapped_guide``.
    :param split: the other-tile reads, one row per nucleus and tile, with
        ``object_id``, ``n`` (reads), ``nguides`` (barcodes named), ``guide``
        and ``other_site``.
    :param sites: the tiles scored; the bootstrap resamples them.
    :param reps: bootstrap replicates.
    :param seed: the bootstrap seed.
    :param min_reads: the least other-tile reads a reference needs, one
        result per value.
    :returns: ``tiles``, ``reps``, ``seed`` and, per value ``k`` of
        ``min_reads``, ``min_reads_<k>``: ``nuclei`` (references called),
        ``agree``, ``pct`` and ``ci95``, the 2.5 and 97.5 percentiles over
        tiles resampled with replacement (:func:`_bootstrap_rows`, the draws
        of the FDR interval); ``pct`` is None without references.
    """
    import pandas as pd

    sites = sorted(int(s) for s in sites)
    index = {s: i for i, s in enumerate(sites)}
    draws = _bootstrap_rows(len(sites), int(reps), int(seed))
    out: Dict[str, Any] = {"tiles": len(sites), "reps": int(reps), "seed": int(seed)}
    calls = pd.DataFrame(calls, columns=["object_id", "site", "mapped_guide"])
    calls = calls[calls["site"].isin(index)]
    for k in min_reads:
        scored = _split_reference(split, k).merge(calls, on="object_id")
        position = scored["site"].map(index).to_numpy(int)
        agree = np.bincount(position, (scored["mapped_guide"] == scored["guide"]).to_numpy(float),
                            minlength=len(sites))
        total = np.bincount(position, minlength=len(sites)).astype(float)
        if len(sites) and total.sum():
            pct = 100.0 * agree[draws].sum(1) / np.maximum(total[draws].sum(1), 1e-12)
            ci = [float(np.nanquantile(pct, 0.025)), float(np.nanquantile(pct, 0.975))]
            point = float(100.0 * agree.sum() / total.sum())
        else:
            ci, point = [None, None], None
        out[f"min_reads_{int(k)}"] = {"nuclei": int(total.sum()), "agree": int(agree.sum()),
                                      "pct": point, "ci95": ci}
    return out


def _well_split_concordance(decoded: Sequence[Mapping[str, Any]],
                            rule: Mapping[str, Any]) -> Dict[str, Any]:
    """The split-read concordance of one decoded well (:func:`split_read_concordance`).

    :param decoded: the decoded fields, each with ``site`` and ``rule_a``.
    :param rule: the rule of the well; with auto thresholds the tiles scored are
        the report fields, outside the tuning fields its ``threshold_choice``
        names, so the thresholds were not chosen on them.
    :returns: the concordance, with ``tile_set`` ``'report'`` or ``'all'``.
    """
    tune = set((rule.get("threshold_choice") or {}).get("tune_sites") or ())
    sites = [int(r["site"]) for r in decoded
             if int(r["rule_a"].get("nuclei") or 0) > 0 and int(r["site"]) not in tune]
    calls = [(int(owner), int(r["site"]), hit["guide"]) for r in decoded
             for owner, hit in (r["rule_a"].get("calls") or {}).items() if not hit["decoy"]]
    split = [(int(owner), *row, int(r["site"])) for r in decoded
             for owner, row in (r["rule_a"].get("split") or {}).items()]
    out = split_read_concordance(calls, split, sites)
    out["tile_set"] = "report" if tune else "all"
    return out


def _cached_field(path: str, rule: Mapping[str, Any], library: Sequence[str],
                  decoys: Sequence[str], crosstalk: str,
                  cells: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """One stored field cache decoded under a rule (:func:`decode_cached_well`).

    :param path: the ``site<NNNN>.npz`` field cache.
    :param rule: the rule with its thresholds.
    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :param crosstalk: the unmixing method.
    :param cells: ``(store, well)`` of the stored SBS cells, for a rule under
        ``ops_read_assignment`` 'cell_mask' replaying a cache written under
        the nucleus reach (:func:`_stored_cell_evidence`); None otherwise.
    :returns: ``site`` and ``rule_a``, as the decoded fields of the engine carry.
    """
    matcher, decoy_set = matcher_for(library, decoys)
    evidence = _cached_evidence(path, rule, cells)
    calls = field_calls(evidence, rule, library, matcher, decoy_set,
                        threshold=rule["threshold"], footprint=rule["footprint"],
                        max_distance=rule["max_distance"], min_fraction=rule["min_fraction"],
                        single_read_quality=rule["single_read_quality"], crosstalk=crosstalk)
    return {"site": int(os.path.basename(path)[4:8]), "rule_a": calls}


def _cached_evidence(path: str, rule: Mapping[str, Any],
                     cells: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """One stored field cache, opened for the rule it is replayed under.

    :param path: the ``site<NNNN>.npz`` field cache.
    :param rule: the rule; under ``ops_read_assignment`` 'cell_mask' the
        reads go to the stored cells (:func:`_stored_cell_evidence`).
    :param cells: ``(store, well)`` of the stored SBS cells, or None.
    :returns: the evidence, every array read into memory.
    """
    with np.load(path) as stored:
        evidence = {key: stored[key] for key in stored.files}
    if _read_assignment_of(rule) == "cell_mask":
        evidence = _stored_cell_evidence(evidence, int(os.path.basename(path)[4:8]), cells)
    elif _read_assignment_of(rule) == "nucleus_first":
        evidence = _nucleus_first_evidence(evidence, int(os.path.basename(path)[4:8]), cells)
    return evidence


def _nucleus_first_evidence(evidence: Mapping[str, Any], site: int,
                            cells: Optional[Tuple[str, str]]) -> Dict[str, Any]:
    """A nucleus-reach field cache with the SBS cells 'nucleus_first' falls back on (item 1052).

    :param evidence: the opened field cache.
    :param site: its site.
    :param cells: ``(store, well)`` of the stored ``sbs_cell`` labels, or
        None when the cache holds ``cell_labels`` (a live 'nucleus_first'
        decode stores them).
    :returns: the evidence, unchanged but for ``cell_labels``: the stored
        labels when given, else those of the cache.
    :raises ValueError: for a cache written under the cell mask.
    :raises FileNotFoundError: naming the store, well, site and labels when
        no cells are given or found.
    """
    if _evidence_text(evidence, "read_assignment") == "cell_mask":
        raise ValueError(
            f"site {site}: ops_read_assignment 'nucleus_first' replays a field cache written "
            "under 'nucleus_reach', but this one was written under 'cell_mask'")
    if cells is None:
        if "cell_labels" in evidence:
            return dict(evidence)
        raise FileNotFoundError(
            f"ops_read_assignment 'nucleus_first' replays site {site} from a field cache "
            "that holds no cell_labels; pass cell_store, the sbs/fov.ome.zarr of the run with the "
            "sbs_cell labels of ops_sbs_cell_method, and the well")
    from .image_store import _compact_labels

    return {**evidence, "cell_labels": _compact_labels(_stored_cells(site, cells))}


def _stored_cells(site: int, cells: Tuple[str, str]) -> np.ndarray:
    """The stored ``sbs_cell`` labels of one field (items 1025, 1052).

    :param site: the site.
    :param cells: ``(store, well)``.
    :returns: the label image, values the object ids.
    :raises FileNotFoundError: naming the store, well and site when the
        labels are not stored.
    """
    from .image_store import field_key, read_labels

    store, well = cells
    key = field_key(well, site)
    if not os.path.isfile(os.path.join(str(store), key, "labels", "sbs_cell", "zarr.json")):
        raise FileNotFoundError(
            f"{store} holds no sbs_cell labels for well {well} site {site}; decode the "
            "well with ops_sbs_cell_method on so its SBS cells are stored")
    return np.asarray(read_labels(store, "sbs_cell", key=key))


def _stored_cell_evidence(evidence: Mapping[str, Any], site: int,
                          cells: Optional[Tuple[str, str]]) -> Mapping[str, Any]:
    """A field cache under ``ops_read_assignment`` 'cell_mask' (item 1025).

    :param evidence: the opened field cache.
    :param site: its site.
    :param cells: ``(store, well)``: ``<run>/sbs/fov.ome.zarr``, where the
        decode stored the ``sbs_cell`` labels (``ops_sbs_cell_method``), and
        the well; None when the cache must already hold cell-mask evidence.
    :returns: the evidence itself when it was written under the cell mask
        (it carries :data:`CELL_MASK_KEYS`); otherwise
        :func:`_cell_mask_evidence` with the field's stored cells.
    :raises FileNotFoundError: naming the store, well, site and the
        ``sbs_cell`` labels when the cache follows the nucleus reach and no
        stored cells are given or found.
    """
    if _evidence_text(evidence, "read_assignment") == "cell_mask":
        if "cell_labels" not in evidence and _evidence_text(evidence, "pix_assignment") != "cell_mask":
            raise FileNotFoundError(
                f"site {site}: this cell-mask field cache holds neither cell_labels nor "
                "cell-labelled pixels (pix_assignment 'cell_mask'); decode the well again")
        return evidence
    if cells is None:
        raise FileNotFoundError(
            f"ops_read_assignment 'cell_mask' replays site {site} from a field cache made "
            "under the nucleus reach, which holds no cell_labels; pass cell_store, the "
            "run's sbs/fov.ome.zarr with the sbs_cell labels of ops_sbs_cell_method, and "
            "the well")
    return _cell_mask_evidence(evidence, _stored_cells(site, cells))


def _cached_rule(resolved: Mapping[str, Any], library: Sequence[str], *,
                 read_threshold: float, single_read_quality: float,
                 pixel_quality: float):
    """The rule a stored field cache is decoded under at fixed thresholds.

    :param resolved: settings after :func:`resolve_settings`.
    :param library: the guide barcodes.
    :param read_threshold: the read threshold on the spot strength.
    :param single_read_quality: the quality a one-read guide needs.
    :param pixel_quality: the pixel quality of the pixel fallback.
    :returns: ``(rule, library, decoys, crosstalk)``: the rule of
        :func:`decode_rule` with its thresholds, the sorted library, the
        decoy barcodes and the unmixing method.
    """
    from .ops_engine import _cell_rule
    from .ops_sbs import decoy_library

    rule = decode_rule(resolved)
    _, mismatches, _, min_fraction = _cell_rule(resolved, library)
    rule.update({"threshold": float(read_threshold),
                 "footprint": float(resolved["ops_footprint"]),
                 "max_distance": mismatches, "min_fraction": min_fraction,
                 "single_read_quality": float(single_read_quality),
                 "pixel_quality": float(pixel_quality)})
    library = sorted(set(library))
    decoys = decoy_library(library, k=int(rule["decoy_k"])) if rule["decoy_k"] > 0 else []
    crosstalk = str(resolved.get("ops_crosstalk") or "none")
    return rule, library, decoys, crosstalk


def _replay_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings a stored run is replayed under (:func:`decode_cached_well`).

    :param settings: the settings of the run; a ``spacr_settings_version``
        stamp gives the keys the mapping leaves out the defaults of that
        version (``spacr.settings.DEFAULT_CHANGES``).
    :returns: the settings after :func:`resolve_settings`.
    """
    from .ops_settings import OPS_DEFAULTS, ops_defaults
    from .settings import SETTINGS_VERSION_KEY, _fold_older_defaults, _settings_file_version

    given = dict(settings)
    if SETTINGS_VERSION_KEY in given:
        _fold_older_defaults(given, _settings_file_version(given), known=set(OPS_DEFAULTS))
    return resolve_settings(ops_defaults(given))


def _cache_paths(cache_folder: str, sites: Optional[Sequence[int]] = None) -> List[str]:
    """The ``site<NNNN>.npz`` field caches of a folder, in site order.

    :param cache_folder: the folder (``ops_store_field_cache``).
    :param sites: only these fields; None for all.
    :returns: the paths.
    """
    wanted = None if sites is None else {int(s) for s in sites}
    return sorted(os.path.join(cache_folder, name) for name in os.listdir(cache_folder)
                  if name.startswith("site") and name.endswith(".npz")
                  and (wanted is None or int(name[4:8]) in wanted))


def decode_cached_well(cache_folder: str, settings: Mapping[str, Any],
                       library: Sequence[str], *, read_threshold: float,
                       single_read_quality: float, pixel_quality: float,
                       tune_sites: Sequence[int] = (), sites: Optional[Sequence[int]] = None,
                       workers: int = 1, cell_store: Optional[str] = None,
                       well: Optional[str] = None) -> Dict[str, Any]:
    """Decode a well again from its stored field caches, without images or objects.

    The fields are decoded by :func:`field_calls` under the rule the
    settings give (:func:`decode_rule`, with the read assignment of
    :func:`spacr.ops_engine._cell_rule`) at fixed thresholds, as a run in
    manual threshold mode would, so a change of the decode rules can be
    replayed on a stored run: for an auto run, pass the thresholds its
    ``decode_thresholds_<well>.json`` chose and its ``tune_sites``.

    :param cache_folder: the folder of ``site<NNNN>.npz`` field caches
        (``ops_store_field_cache``).
    :param settings: the settings of the run. A ``spacr_settings_version`` stamp
        gives keys the mapping leaves out the defaults of that version
        (``spacr.settings.DEFAULT_CHANGES``), as a settings file is read;
        without one the mapping is taken as today's.
    :param library: the guide barcodes.
    :param read_threshold: the read threshold on the spot strength.
    :param single_read_quality: the quality a one-read guide needs.
    :param pixel_quality: the pixel quality of the pixel fallback.
    :param tune_sites: the fields auto thresholds were tuned on; the
        split-read concordance scores the others.
    :param sites: only these fields; None decodes every cache.
    :param workers: processes decoding fields at once.
    :param cell_store: under ``ops_read_assignment`` 'cell_mask' (item
        1025), the run's ``sbs/fov.ome.zarr`` whose ``sbs_cell`` labels give
        the reads and pixels of caches written under the nucleus reach to
        their cells (:func:`_stored_cell_evidence`); caches written under
        the cell mask need none. Under 'nucleus_first' (item 1052) the same
        labels are the cells the objects the nucleus reach leaves uncalled
        fall back on (:func:`_nucleus_first_evidence`); a cache a live
        'nucleus_first' decode wrote holds them. Not read under
        'nucleus_reach'.
    :param well: the well of ``cell_store``; the cache folder's parent
        folder name when None.
    :returns: ``rule`` (the rule decoded under; under 'nucleus_first' with
        ``ops_nucleus_first_bleed_curve`` 'fit' its ``bleed_model`` is the
        curve fitted on these caches, :func:`_fit_bleed_model`), ``fields``
        (one entry per field with ``site`` and ``rule_a``, as
        :func:`field_calls` returns),
        ``nuclei``, ``assigned_real``, ``assigned_decoy`` and
        ``split_read_concordance``.
    """
    from multiprocessing import get_context

    resolved = _replay_settings(settings)
    rule, library, decoys, crosstalk = _cached_rule(
        resolved, library, read_threshold=read_threshold,
        single_read_quality=single_read_quality, pixel_quality=pixel_quality)
    paths = _cache_paths(cache_folder, sites)
    if _read_assignment_of(rule) == "nucleus_first" and not rule.get("bleed_model"):
        rule["bleed_model"] = _fit_bleed_model(paths, rule, library, decoys, crosstalk,
                                               resolved)
    jobs = [(path, rule, library, decoys, crosstalk) for path in paths]
    if _read_assignment_of(rule) in ("cell_mask", "nucleus_first") and cell_store is not None:
        name = well or os.path.basename(os.path.dirname(os.path.abspath(cache_folder)))
        jobs = [(*job, (str(cell_store), name)) for job in jobs]
    workers = max(1, min(int(workers), len(jobs)))
    if workers > 1:
        from .resource_log import _array_file_nbytes, _guard_workers

        workers = max(1, int(_guard_workers(
            "ops_decode", workers, _array_file_nbytes(paths[0]),
            settings=resolved)))
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            fields = pool.starmap(_cached_field, jobs, chunksize=1)
    else:
        fields = [_cached_field(*job) for job in jobs]
    fields.sort(key=lambda r: r["site"])
    concordance = _well_split_concordance(
        fields, {"threshold_choice": {"tune_sites": list(tune_sites)}})
    return {"rule": rule, "fields": fields,
            "nuclei": sum(int(r["rule_a"]["nuclei"]) for r in fields),
            "assigned_real": sum(int(r["rule_a"]["assigned_real"]) for r in fields),
            "assigned_decoy": sum(int(r["rule_a"]["assigned_decoy"]) for r in fields),
            "split_read_concordance": concordance}


def _fit_bleed_model(paths: Sequence[str], rule: Mapping[str, Any], library: Sequence[str],
                     decoys: Sequence[str], crosstalk: str,
                     settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The bleed curve of a run, fitted on its nucleus-reach field caches (item 1052).

    :param paths: the ``site<NNNN>.npz`` field caches.
    :param rule: the rule with its read threshold.
    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :param crosstalk: the unmixing method.
    :param settings: read for ``ops_decode_qc_seed``, the seed of the null.
    :returns: :func:`spacr.ops_decode_qc_panels.fit_bleed_model_from_caches`.
    :raises ValueError: naming the keys a cache lacks.
    """
    from .ops_decode_qc_panels import fit_bleed_model_from_caches

    def fields():
        """Each field cache in turn, without its pixels, so one field is held at a time."""
        for path in paths:
            with np.load(path) as stored:
                z = {key: stored[key] for key in stored.files
                     if not key.startswith(("pix_", "f_"))}
            absent = [key for key in CACHE_NUCLEUS_KEYS if key not in z]
            if absent:
                raise ValueError(f"{path}: ops_nucleus_first_bleed_curve 'fit' needs the "
                                 "nucleus positions of the field cache, which holds no "
                                 f"{', '.join(absent)}; give a stored curve instead")
            yield {"site": int(os.path.basename(path)[4:8]), "z": z}

    nucleus_rule = {key: value for key, value in rule.items()
                    if key not in ("read_assignment", "bleed_gate", "bleed_model")}
    return fit_bleed_model_from_caches(fields(), nucleus_rule, library, decoys, crosstalk,
                                       seed=_qc_number(settings, "ops_decode_qc_seed", 0, int))


#: The error terms of the dataset objective (item 1033) and their default
#: weights: the split-read disagreement over its floor counts in full, and
#: the decoy FDR enters as the target, not as a weight (the design card
#: 20261125, FINAL OBJECTIVE: e_total = max(Sd - Sd_ref, 0)).
DATASET_WEIGHTS: Dict[str, float] = {"split": 1.0, "decoy": 0.0}

#: The most the split-read disagreement of a dataset setting may rise over
#: the manual thresholds, in points: the upper end of the paired 95 %
#: tile-bootstrap interval (the design card margin of 0.1 points).
DATASET_SPLIT_GUARD = 0.1

#: The target of the dataset tuner when ``ops_decode_target_fdr`` is empty and no
#: Brieflow reference is given: false calls per 100 real calls (item 1033b).
DATASET_DEFAULT_TARGET_FDR = 0.06

#: The values of ``ops_decode_target_fdr_reference``.
DATASET_TARGET_REFERENCES = ("brieflow",)

#: The least supporting reads, one guide named, of the cleanest calls whose
#: split-read disagreement is the floor of the reference itself (Sd_ref).
_CLEAN_READS = 3


def dataset_threshold_settings(settings: Mapping[str, Any],
                               rule: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings of the dataset threshold tuner (item 1033, alpha), checked.

    The grid is the one auto thresholds try (``ops_decode_auto_grid`` and
    ``ops_decode_auto_spot_grid``, read by :func:`threshold_settings`), and
    the entered thresholds -- ``ops_read_threshold``,
    ``ops_single_read_quality`` and the rule's pixel quality -- are always
    one of its settings, the manual baseline.

    :param settings: read for the auto grid keys, ``ops_decode_auto_tune_fraction``,
        ``ops_decode_auto_seed``, ``ops_decode_auto_ci``, ``ops_read_threshold``,
        ``ops_single_read_quality``, ``ops_decode_dataset_rule`` ('beat_reference',
        'knee' as before schema 988, or 'target_gain' as before schema 987),
        ``ops_brieflow_yield`` and ``ops_brieflow_split_read``, ``ops_decode_target_fdr`` (under
        'knee' an optional cap, empty for none; under 'target_gain' empty
        takes ``ops_brieflow_fair_fdr`` under ``ops_decode_target_fdr_reference``
        'brieflow', else :data:`DATASET_DEFAULT_TARGET_FDR`),
        ``ops_decode_split_guard`` and
        ``ops_decode_dataset_weights``.
    :param rule: from :func:`decode_rule`.
    :returns: ``spot_thresholds``, ``floors`` and ``pixel_qualities``
        (sorted, the manual values included), ``manual`` (the three entered
        thresholds), ``target_fdr`` (per 100 calls, None for no cap), ``target_source``
        ('entered', 'brieflow', 'default' or 'none'), ``brieflow_fair_fdr``
        (None when not given), ``rule``, ``split_guard`` (points),
        ``weights``, ``ci``, ``tune_fraction`` and ``seed``.
    :raises ValueError: as :func:`threshold_settings` for the grid and the
        decoys; for a target or guard that is not a number above 0 (guard:
        0 or above), or weights that are not a mapping of term names to
        numbers of 0 or more.
    """
    from .ops_settings import OPS_DEFAULTS

    grid = threshold_settings({**settings, "ops_decode_threshold_mode": "auto",
                               "ops_decode_fdr_calibration": "none"}, rule)
    entered = {key: settings.get(key) for key in ("ops_read_threshold",
                                                  "ops_single_read_quality")}
    try:
        manual = {key: float(OPS_DEFAULTS[key] if value is None or value == "" else value)
                  for key, value in entered.items()}
    except (TypeError, ValueError):
        raise ValueError("ops_read_threshold and ops_single_read_quality must be "
                         "numbers") from None
    manual["ops_pixel_quality"] = float(rule["pixel_quality"])
    dataset_rule = settings.get("ops_decode_dataset_rule")
    dataset_rule = "beat_reference" if dataset_rule is None or str(dataset_rule).strip() == "" \
        else str(dataset_rule).strip().lower()
    if dataset_rule not in DATASET_RULES:
        raise ValueError(f"ops_decode_dataset_rule must be one of {list(DATASET_RULES)}; "
                         f"got {dataset_rule!r}")
    target = settings.get("ops_decode_target_fdr")
    reference = settings.get("ops_decode_target_fdr_reference")
    reference = None if reference is None or str(reference).strip() == "" \
        else str(reference).strip().lower()
    if reference is not None and reference not in DATASET_TARGET_REFERENCES:
        raise ValueError(f"ops_decode_target_fdr_reference must be empty or one of "
                         f"{list(DATASET_TARGET_REFERENCES)}; got {reference!r}")
    fair = settings.get("ops_brieflow_fair_fdr")
    if reference == "brieflow" or (fair is not None and fair != ""):
        try:
            fair = float(fair)
        except (TypeError, ValueError):
            raise ValueError("ops_decode_target_fdr_reference 'brieflow' needs "
                             "ops_brieflow_fair_fdr, the fair decoy FDR of "
                             "the Brieflow calls in false calls per 100") from None
    else:
        fair = None
    if target is None or target == "":
        if dataset_rule != "target_gain":
            target, target_source = None, "none"
        elif reference == "brieflow":
            target, target_source = fair, "brieflow"
        else:
            target, target_source = DATASET_DEFAULT_TARGET_FDR, "default"
    else:
        target_source = "entered"
    guard = settings.get("ops_decode_split_guard")
    guard = DATASET_SPLIT_GUARD if guard is None or guard == "" else guard
    try:
        target = None if target is None else float(target)
        guard = float(guard)
    except (TypeError, ValueError):
        raise ValueError("ops_decode_target_fdr and ops_decode_split_guard must be "
                         "numbers") from None
    if target is not None and not target > 0:
        raise ValueError(f"ops_decode_target_fdr must be above 0; got {target}")
    if not guard >= 0:
        raise ValueError(f"ops_decode_split_guard must be 0 or more; got {guard}")
    reference_numbers = {}
    for key in ("ops_brieflow_yield", "ops_brieflow_split_read"):
        value = settings.get(key)
        if value is None or value == "":
            reference_numbers[key] = None
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number; got {value!r}") from None
        inside = 0.0 < value <= 1.0 if key == "ops_brieflow_yield" else 0.0 <= value < 1.0
        if not inside:
            raise ValueError(f"{key} is a fraction between 0 and 1; got {value}")
        reference_numbers[key] = value
    weights = settings.get("ops_decode_dataset_weights")
    if isinstance(weights, str) and weights.strip():
        try:
            weights = json.loads(weights)
        except ValueError:
            raise ValueError("ops_decode_dataset_weights must be a mapping such as "
                             '{"split": 1.0, "decoy": 0.5}; got ' f"{weights!r}") from None
    if weights is None or weights == "" or weights == {}:
        weights = DATASET_WEIGHTS if dataset_rule == "target_gain" else DATASET_KNEE_WEIGHTS
    if not isinstance(weights, Mapping):
        raise ValueError(f"ops_decode_dataset_weights must be a mapping; got {weights!r}")
    try:
        weights = {str(name): float(value) for name, value in weights.items()}
    except (TypeError, ValueError):
        raise ValueError("ops_decode_dataset_weights must map term names to "
                         f"numbers; got {weights!r}") from None
    if any(not value >= 0 for value in weights.values()):
        raise ValueError(f"ops_decode_dataset_weights must be 0 or more; got {weights!r}")
    weights = {**{name: 0.0 for name in DATASET_WEIGHTS}, **weights}
    spots = grid["spot_thresholds"] or [manual["ops_read_threshold"]]
    return {"spot_thresholds": sorted(set(spots) | {manual["ops_read_threshold"]}),
            "floors": sorted(set(grid["floors"]) | {manual["ops_single_read_quality"]}),
            "pixel_qualities": sorted(set(grid["pixel_qualities"])
                                      | {manual["ops_pixel_quality"]}),
            "manual": manual, "target_fdr": target, "target_source": target_source,
            "brieflow_fair_fdr": fair, "brieflow_yield": reference_numbers["ops_brieflow_yield"],
            "brieflow_split_read": reference_numbers["ops_brieflow_split_read"],
            "rule": dataset_rule, "split_guard": guard,
            "weights": weights, "ci": grid["ci"], "tune_fraction": grid["tune_fraction"],
            "seed": grid["seed"]}


def _watched_objects(path: str) -> np.ndarray:
    """The nuclei a field cache holds reads of strictly inside, owned by another field.

    :param path: the ``site<NNNN>.npz`` field cache.
    :returns: their ids, sorted and unique: every nucleus some other field
        may hold a split read of (:func:`_split_reads`), before the read
        threshold and the barcode match.
    """
    with np.load(path) as stored:
        first = np.asarray(stored["id0"])
        mask = ~np.asarray(stored["owned0"], bool) & (np.asarray(stored["gap"])[:, 0] < 0) \
            & (first >= 0)
    return np.unique(first[mask])


def _dataset_field(path: str, rule: Mapping[str, Any], library: Sequence[str],
                   decoys: Sequence[str], crosstalk: str, config: Mapping[str, Any],
                   watched: frozenset, universe: Mapping[str, int],
                   cells: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """One stored field, called at every setting of the dataset grid (item 1033).

    The field is voted once per read threshold (:func:`_joint_field_candidates`)
    and its calls are rebuilt at every call-threshold setting
    (:func:`_candidate_calls`), so each setting calls what
    :func:`field_calls` would.

    :param path: the field cache.
    :param rule: the rule at the manual thresholds (:func:`_cached_rule`).
    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :param crosstalk: the unmixing method.
    :param config: from :func:`dataset_threshold_settings`.
    :param watched: the nuclei some field holds split reads of
        (:func:`_watched_objects`), the only calls kept.
    :param universe: barcode -> index, library and decoys.
    :param cells: ``(store, well)`` of the stored SBS cells, or None.
    :returns: ``site``, ``nuclei``, ``counts`` (``(read thresholds, floors,
        pixel qualities, 2)`` real and decoy calls), ``objects`` and
        ``guides`` (the watched calls: one row per nucleus, the barcode index
        per setting, -1 where uncalled), ``split`` (the split reads at the
        manual read threshold, :func:`_split_reads`) and ``clean`` (nucleus
        -> guide of the read calls at the manual thresholds with
        :data:`_CLEAN_READS` or more supporting reads and one guide named).
    """
    matcher, decoy_set = matcher_for(library, decoys)
    evidence = _cached_evidence(path, rule, cells)
    spots, floors, qualities = (config["spot_thresholds"], config["floors"],
                                config["pixel_qualities"])
    joint = _joint_field_candidates(
        evidence, rule, library, matcher, decoy_set, spot_thresholds=spots,
        floors=floors, pixel_qualities=qualities, footprint=rule["footprint"],
        max_distance=rule["max_distance"], min_fraction=rule["min_fraction"],
        crosstalk=crosstalk)
    owned = set(np.asarray(evidence["owned_ids"]).tolist())
    objects = sorted(watched & owned)
    row = {o: i for i, o in enumerate(objects)}
    guides = np.full((len(objects), len(spots), len(floors), len(qualities)), -1, np.int32)
    manual = config["manual"]
    m_spot = spots.index(manual["ops_read_threshold"])
    split: Dict[int, tuple] = {}
    clean: Dict[int, str] = {}
    for s, candidates in enumerate(joint["by_spot"]):
        light = {**candidates, "votes": {**candidates["votes"], "tallies": None}}
        for i, floor in enumerate(floors):
            for j, quality in enumerate(qualities):
                calls = _candidate_calls(light, floor, quality, decoy_set)["calls"]
                for owner in row.keys() & calls.keys():
                    guides[row[owner], s, i, j] = universe[calls[owner]["guide"]]
        if s == m_spot:
            votes = candidates["votes"]
            split = dict(votes.get("split") or {})
            named = (votes.get("tallies") or {}).get("votes", {})
            for owner, hit in votes["primary"].items():
                if (owner in owned and hit["n_support"] >= _CLEAN_READS
                        and len(named.get(owner, ())) == 1 and hit["guide"] not in decoy_set):
                    clean[int(owner)] = hit["guide"]
    return {"site": int(os.path.basename(path)[4:8]), "nuclei": int(joint["nuclei"]),
            "counts": np.asarray(joint["counts"], np.int64),
            "objects": np.asarray(objects, np.int64), "guides": guides,
            "split": split, "clean": clean}


def _dataset_tallies(fields: Sequence[Mapping[str, Any]], wells: Sequence[str],
                     universe: Mapping[str, int], decoy_index: np.ndarray,
                     shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    """Per field and setting, the calls scored against the fixed split reference.

    The reference is built once, from the split reads at the manual read
    threshold (:func:`_split_reference`, one read or more), and is the same
    for every setting, so differences between settings are differences of
    their calls.

    :param fields: from :func:`_dataset_field`, in dataset order.
    :param wells: the well of each field.
    :param universe: barcode -> index.
    :param decoy_index: per barcode index, whether it is a decoy.
    :param shape: the grid shape.
    :returns: ``agree`` and ``scored`` (``(fields, *shape)``: real calls on
        reference nuclei, and those naming the reference guide), and
        ``ref_agree`` and ``ref_scored`` (``(fields,)``: the same for the
        cleanest calls).
    """
    import pandas as pd

    rows = [(f"{well}\t{owner}", *hit, i) for i, (well, field) in enumerate(zip(wells, fields))
            for owner, hit in field["split"].items()]
    reference = _split_reference(pd.DataFrame(rows, columns=[
        "object_id", "n", "nguides", "guide", "other_site"]), 1)
    ref = {key: universe[guide] for key, guide in
           zip(reference["object_id"], reference["guide"]) if guide in universe}
    n = len(fields)
    agree = np.zeros((n,) + shape, np.int64)
    scored = np.zeros((n,) + shape, np.int64)
    ref_agree = np.zeros(n, np.int64)
    ref_scored = np.zeros(n, np.int64)
    for i, (well, field) in enumerate(zip(wells, fields)):
        keys = [f"{well}\t{int(o)}" for o in field["objects"]]
        has = np.array([key in ref for key in keys], bool)
        if has.any():
            target = np.array([ref[key] for key, h in zip(keys, has) if h], np.int64)
            called = field["guides"][has]
            real = (called >= 0) & ~decoy_index[np.maximum(called, 0)]
            target = target.reshape((-1,) + (1,) * len(shape))
            scored[i] = real.sum(0)
            agree[i] = (real & (called == target)).sum(0)
        for owner, guide in field["clean"].items():
            key = f"{well}\t{owner}"
            if key in ref:
                ref_scored[i] += 1
                ref_agree[i] += int(universe[guide] == ref[key])
    return {"agree": agree, "scored": scored, "ref_agree": ref_agree,
            "ref_scored": ref_scored}


def _dataset_terms(sums: Mapping[str, np.ndarray], weights: Mapping[str, float],
                   k: int, scale: float) -> Dict[str, np.ndarray]:
    """The objective of the dataset tuner from summed counts (item 1033).

    ``f = Gn (1 - e_total)^2``, ``Gn`` the real calls and ``e_total`` the
    weighted sum of the error shares, clipped to [0, 1]: ``split`` =
    ``max(Sd - Sd_ref, 0)``, the split-read disagreement over that of the
    cleanest calls; ``decoy`` = the decoy FDR as a share; any further term
    the share of its errors among its scored calls.

    :param sums: ``real``, ``decoy``, ``agree``, ``scored``, ``ref_agree``,
        ``ref_scored`` and, per further term, ``<name>_errors`` and
        ``<name>_scored``, summed over some fields; leading axes broadcast.
    :param weights: term name -> weight.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :returns: ``fdr`` (per 100), ``split_pct`` (disagreement, %), ``split_ref_pct``,
        ``e_total``, ``f`` and one share per term.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        real = sums["real"].astype(float)
        fdr = 100.0 * max(float(scale), 1.0) / float(k) * sums["decoy"] / np.maximum(real, 1)
        disagree = np.where(sums["scored"] > 0, 1.0 - sums["agree"] / np.maximum(
            sums["scored"], 1), np.nan)
        floor = np.where(sums["ref_scored"] > 0, 1.0 - sums["ref_agree"] / np.maximum(
            sums["ref_scored"], 1), 0.0)
        floor = np.reshape(floor, np.shape(floor) + (1,) * (disagree.ndim - np.ndim(floor)))
        terms = {"split": np.maximum(np.nan_to_num(disagree, nan=0.0) - floor, 0.0),
                 "decoy": fdr / 100.0}
        for name in weights:
            if name not in terms:
                errors, total = sums[f"{name}_errors"], sums[f"{name}_scored"]
                terms[name] = np.where(total > 0, errors / np.maximum(total, 1), 0.0)
        e_total = np.clip(sum(float(weights[name]) * terms[name] for name in weights), 0.0, 1.0)
    return {"fdr": fdr, "split_pct": 100.0 * disagree,
            "split_ref_pct": 100.0 * np.broadcast_to(floor, disagree.shape),
            "e_total": e_total, "f": real * (1.0 - e_total) ** 2,
            **{f"term_{name}": value for name, value in terms.items()}}


def _ci(values: np.ndarray) -> List[Optional[float]]:
    """The 2.5 and 97.5 percentiles of bootstrap values, None when all are missing.

    :param values: one value per replicate.
    :returns: ``[low, high]``.
    """
    values = np.asarray(values, float)
    if not np.isfinite(values).any():
        return [None, None]
    return [float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))]


def calibrate_error_weights(terms, truth, names: Sequence[str]) -> Dict[str, float]:
    """Weights of the dataset error terms fitted to a known error (item 1033).

    Non-negative least squares without an intercept, as the design card
    fitted them (source A): each row is one call set with a known wrong-call
    share, such as calls with errors injected at a known rate; terms and
    truth are best given as differences from the same call set without
    injection, so the set's own unknown error cancels. The weights are then
    entered as ``ops_decode_dataset_weights``.

    :param terms: ``(rows, len(names))`` measured error shares.
    :param truth: ``(rows,)`` the known wrong-call shares.
    :param names: the term of each column (``split``, ``decoy`` or a further term).
    :returns: term name -> weight, each 0 or more.
    :raises ValueError: when the shapes disagree.
    """
    from scipy.optimize import nnls

    terms = np.asarray(terms, float)
    truth = np.asarray(truth, float)
    if terms.ndim != 2 or terms.shape != (len(truth), len(names)):
        raise ValueError("terms must be (rows, terms) with one truth value per row")
    weights, _ = nnls(terms, truth)
    return {str(name): float(w) for name, w in zip(names, weights)}


def _split_rise_guard(split_boot: np.ndarray, m_pos: Tuple[int, ...], guard: float,
                      shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """The split-read guard of the dataset tuner against the manual thresholds.

    :param split_boot: ``(draws, *grid)`` split-read disagreement in % over
        paired bootstrap draws of the fields.
    :param m_pos: the grid position of the manual thresholds.
    :param guard: the largest rise in points the upper bound may show
        (``ops_decode_split_guard``).
    :param shape: the grid shape.
    :returns: ``rise_hi`` (the 97.5th percentile of the paired rise over the
        manual thresholds, per setting) and ``ok`` (True where it is at most
        ``guard``, and where it is undefined).
    """
    rise = split_boot - split_boot[(slice(None),) + tuple(m_pos)].reshape(
        (-1,) + (1,) * len(shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rise_hi = np.nanquantile(rise, 0.975, axis=0) if len(rise) else np.full(shape, np.nan)
    return rise_hi, ~(rise_hi > guard)


def _dataset_summed(arrays: Mapping[str, np.ndarray], idx: Sequence[int],
                    draws: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-field counts summed over some fields, or over bootstrap draws of them.

    :param arrays: per field and setting counts (:func:`_dataset_choice`).
    :param idx: the fields.
    :param draws: from :func:`_bootstrap_rows`, over the positions of ``idx``;
        None sums the fields once.
    :returns: name -> the sum, with one leading axis per draw when ``draws`` is given.
    """
    idx = np.asarray(idx, np.int64)
    if draws is None:
        return {name: value[idx].sum(0) for name, value in arrays.items()}
    weights = np.stack([np.bincount(row, minlength=len(idx)) for row in draws]) \
        if len(draws) else np.zeros((0, len(idx)))
    return {name: np.tensordot(weights, value[idx], axes=(1, 0))
            for name, value in arrays.items()}


def _dataset_choice_target_gain(arrays: Mapping[str, np.ndarray], nuclei: np.ndarray,
                                tune: Sequence[int], report: Sequence[int],
                                axes: Sequence[Tuple[str, Sequence[float]]], m_pos: Tuple[int, ...],
                                config: Mapping[str, Any], *, k: int, scale: float,
                                reps: int = SPLIT_READ_REPS, seed: int = SPLIT_READ_SEED) -> Dict[str, Any]:
    """The 1033b choice: a target and a held-out gain over the manual thresholds.

    Kept as ``ops_decode_dataset_rule`` 'target_gain' for settings files
    stamped before schema 987 (:func:`_dataset_choice` is the knee).

    :param arrays: per field and setting (``(fields, *grid)``): ``real``,
        ``decoy``, ``agree`` and ``scored``, and per field ``ref_agree`` and
        ``ref_scored`` (:func:`_dataset_tallies`); per further weighted term
        ``<name>_errors`` and ``<name>_scored``.
    :param nuclei: owned nuclei per field.
    :param tune: the fields that choose.
    :param report: the held-out fields.
    :param axes: ``(name, values)`` per grid axis.
    :param m_pos: the grid position of the manual thresholds.
    :param config: from :func:`dataset_threshold_settings`, for
        ``target_fdr``, ``split_guard``, ``weights`` and ``ci``.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :param reps: tile-bootstrap replicates.
    :param seed: their seed.
    :returns: ``adopted``, ``auto_met`` (a setting passed on the tuning
        half), ``reason``, ``thresholds``, ``tune``, ``report`` and ``grid``,
        as :func:`tune_dataset_thresholds` returns them.
    """
    weights = config["weights"]
    shape = tuple(len(values) for _, values in axes)
    m_pos = tuple(int(i) for i in m_pos)
    nuclei = np.asarray(nuclei, np.int64)

    def summed(idx, draws=None):
        return _dataset_summed(arrays, idx, draws)

    def half(idx):
        point = _dataset_terms(summed(idx), weights, k, scale)
        draws = _bootstrap_rows(len(idx), int(reps), int(seed))
        boot = _dataset_terms(summed(idx, draws), weights, k, scale)
        return point, boot

    def numbers(idx, point, boot, pos, fdr_part):
        sums = summed(idx)
        total = int(nuclei[np.asarray(idx, np.int64)].sum())
        real = int(sums["real"][pos])
        return {"fields": len(idx), "nuclei": total, "real": real,
                "genotyped_pct": 100.0 * real / max(total, 1),
                "fdr_per_100": fdr_part["fdr_per_100"], "fdr_ci95": fdr_part["fdr_ci95"],
                "split_disagreement_pct": _float(point["split_pct"][pos]),
                "split_disagreement_ci95": _ci(boot["split_pct"][(slice(None),) + pos]),
                "split_reference_nuclei": int(sums["scored"][pos]),
                "split_floor_pct": _float(point["split_ref_pct"][pos]),
                "e_total": _float(point["e_total"][pos]),
                "f": _float(point["f"][pos]),
                "f_ci95": _ci(boot["f"][(slice(None),) + pos])}

    counts = np.stack([arrays["real"], arrays["decoy"]], -1)
    flat = counts.reshape(len(counts), 1, -1, 2)
    halves = {part: half(idx) for part, idx in (("tune", tune), ("report", report))
              if len(idx)}
    chosen, met, rows = m_pos, False, []
    reason = None
    if len(tune) < 2 or len(report) < 2:
        reason = (f"too few fields with nuclei to tune and report on ({len(tune)} tune, "
                  f"{len(report)} report; two or more each)")
    else:
        score = _grid_score(counts, nuclei, tune, k, scale, config["ci"])
        held_score = _grid_score(counts, nuclei, report, k, scale, True)
        point, boot = halves["tune"]
        rise_hi, guard_ok = _split_rise_guard(boot["split_pct"], m_pos, config["split_guard"], shape)
        fdr_ok = score["bound"] <= config["target_fdr"]
        best = None
        for pos in np.ndindex(*shape):
            passes = bool(fdr_ok[pos] and guard_ok[pos])
            rows.append({**{name: values[pos[a]] for a, (name, values) in enumerate(axes)},
                         "real": int(score["real"][pos]), "decoy": int(score["decoy"][pos]),
                         "genotyped_pct": float(score["genotyped_pct"][pos]),
                         "fdr_per_100": float(score["fdr"][pos]),
                         "fdr_bound": float(score["bound"][pos]),
                         "held_out_fdr": float(held_score["fdr"][pos]),
                         "held_out_fdr_hi": float(held_score["fdr_hi"][pos]),
                         "split_disagreement_pct": _float(point["split_pct"][pos]),
                         "split_rise_hi": _float(rise_hi[pos]),
                         "e_total": float(point["e_total"][pos]),
                         "f": float(point["f"][pos]), "fdr_passes": bool(fdr_ok[pos]),
                         "split_passes": bool(guard_ok[pos]), "passes": passes,
                         "manual": pos == m_pos})
            if passes:
                key = (float(point["f"][pos]),) + pos
                if best is None or key > best:
                    best = key
        if best is None:
            reason = ("no setting met the FDR target and the split-read guard on the "
                      "tuning half")
        else:
            chosen, met = tuple(best[1:]), True
        for row, pos in zip(rows, np.ndindex(*shape)):
            row["chosen"] = pos == chosen

    out_parts: Dict[str, Any] = {}
    gain = {"f": None, "ci95": [None, None], "pct": None}
    for part, idx in (("tune", tune), ("report", report)):
        if len(idx) < 1:
            out_parts[part] = {}
            continue
        point, boot = halves[part]
        entry = {}
        for name, pos in (("chosen", chosen), ("manual", m_pos)):
            fdr_part = _chosen_summary(flat, nuclei, idx,
                                       (0, int(np.ravel_multi_index(pos, shape))), k, scale)
            entry[name] = numbers(idx, point, boot, pos, fdr_part)
        if part == "report":
            diff = boot["f"][(slice(None),) + chosen] - boot["f"][(slice(None),) + m_pos]
            base = float(point["f"][m_pos])
            gain = {"f": float(point["f"][chosen] - point["f"][m_pos]), "ci95": _ci(diff),
                    "pct": (100.0 * float(point["f"][chosen] - point["f"][m_pos]) / base
                            if base > 0 else None)}
            entry["gain"] = gain
        out_parts[part] = entry
    adopted = False
    held = (out_parts.get("report") or {}).get("chosen") or {}
    held_fdr = held.get("fdr_per_100")
    held_fdr = float("nan") if held_fdr is None else float(held_fdr)
    if reason is None:
        if chosen == m_pos:
            reason = "the tuning half chose the manual thresholds"
        elif not held_fdr <= config["target_fdr"]:
            reason = (f"the held-out FDR estimate, {held_fdr:.3f} per 100, is above the "
                      f"target of {config['target_fdr']:.3f}")
        elif gain["ci95"][0] is None or not gain["ci95"][0] > 0:
            reason = (f"the held-out gain in f, {gain['f']:.1f} (95 % CI {gain['ci95'][0]:.1f} "
                      f"to {gain['ci95'][1]:.1f}), does not lie above 0"
                      if gain["ci95"][0] is not None else "the held-out gain has no interval")
        else:
            adopted = True
            reason = (f"the held-out gain in f, {gain['f']:.1f} (95 % CI {gain['ci95'][0]:.1f} "
                      f"to {gain['ci95'][1]:.1f}), lies above 0")
    def as_values(pos):
        return {name: float(values[pos[a]]) for a, (name, values) in enumerate(axes)}

    return {"adopted": adopted, "auto_met": met, "reason": reason,
            "held_out_fdr": {"estimate": _float(held_fdr),
                             "ci95": held.get("fdr_ci95"),
                             "target": config["target_fdr"],
                             "at_or_below_target": bool(held_fdr <= config["target_fdr"])},
            "thresholds": {"chosen": as_values(chosen), "manual": as_values(m_pos),
                           "used": as_values(chosen if adopted else m_pos)},
            **out_parts, "grid": rows}


#: The errors drawn at least on the frontier figure, so a log axis is defined.
_FIGURE_ERROR_FLOOR = 1.0

#: The weights of the knee rule when ``ops_decode_dataset_weights`` is empty:
#: both error kinds count once.
DATASET_KNEE_WEIGHTS: Dict[str, float] = {"split": 1.0, "decoy": 1.0}

#: The values of ``ops_decode_dataset_rule``.
DATASET_RULES = ("beat_reference", "knee", "target_gain")


def _marginal_steps(n: np.ndarray, e: np.ndarray, n_boot: np.ndarray, e_boot: np.ndarray
                    ) -> List[Dict[str, Any]]:
    """The marginal against the average error rate of each step up a frontier.

    :param n: calls of the frontier points, in increasing order.
    :param e: their estimated errors.
    :param n_boot: ``(draws, points)`` calls of the same points over paired
        bootstrap draws of the fields.
    :param e_boot: their estimated errors over the same draws.
    :returns: per step from point ``j`` to ``j + 1``: ``marginal`` (the added
        errors over the added calls, ``m = dE / dN``), ``average`` (``a = E / N``
        at point ``j``), ``diff`` (``m - a``), ``ci95`` (the 2.5th and 97.5th
        percentiles of the paired ``m - a``) and ``significant`` (the lower
        bound above 0: the added calls raise the errors faster than the cells).
    """
    n, e = np.asarray(n, float), np.asarray(e, float)
    steps = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(len(n) - 1):
            rise = n[j + 1] - n[j]
            marginal = (e[j + 1] - e[j]) / rise if rise > 0 else float("nan")
            average = e[j] / n[j] if n[j] > 0 else float("nan")
            rise_b = n_boot[:, j + 1] - n_boot[:, j]
            diff_b = (e_boot[:, j + 1] - e_boot[:, j]) / np.where(rise_b > 0, rise_b, np.nan) \
                - e_boot[:, j] / np.where(n_boot[:, j] > 0, n_boot[:, j], np.nan)
            ci = _ci(diff_b)
            steps.append({"marginal": _float(marginal), "average": _float(average),
                          "diff": _float(marginal - average), "ci95": ci,
                          "significant": bool(ci[0] is not None and ci[0] > 0)})
    return steps


def _dataset_choice(arrays: Mapping[str, np.ndarray], nuclei: np.ndarray,
                    tune: Sequence[int], report: Sequence[int],
                    axes: Sequence[Tuple[str, Sequence[float]]], m_pos: Tuple[int, ...],
                    config: Mapping[str, Any], *, k: int, scale: float,
                    reps: int = SPLIT_READ_REPS, seed: int = SPLIT_READ_SEED
                    ) -> Dict[str, Any]:
    """The stop on the yield and error frontier of a dataset (item 1033c).

    The operational form of the rule "minimise errors and maximise
    genotyped cells; when errors start rising faster than the genotyped
    cells, stop; and the other way round, keep going". On the tuning half
    every grid setting has ``N`` (real calls) and ``E`` (estimated false
    calls: ``N`` times the weighted sum of the decoy FDR share and the
    split-read disagreement over its floor, :func:`_dataset_terms`). The
    Pareto frontier (more ``N``, fewer ``E``, :func:`_pareto`) is walked from
    the strictest point toward more calls. At each step the marginal error
    rate of the added calls, ``m = dE / dN``, is set against the average
    error rate of the point the step starts from, ``a = E / N``, with a
    paired tile bootstrap of the fields (``reps`` draws, ``seed``). The walk
    keeps going while ``m`` is not significantly above ``a`` (the lower 95 %
    bound of ``m - a`` is at most 0) and stops at the last point before the
    first step where it is (:func:`_marginal_steps`). So with a flat error
    rate the walk reaches the most calls, and if tightening lowers the
    errors faster than the calls, the stop is on the strict side where
    that no longer holds. The optional FDR cap leaves a setting whose FDR
    bound on the tuning half is above it off the frontier; the manual
    thresholds are only a reference.

    Under ``config["rule"]`` 'beat_reference' with a reference (yield and fair
    FDR) the frontier point with the most cells that beats it on yield and
    decoy FDR (and the split-read rate of the reference when one is given) is
    taken instead of the stop, provided its split-read disagreement does not
    rise over the manual thresholds beyond ``ops_decode_split_guard``: the
    upper bound of the paired bootstrap rise on the tuning half
    (:func:`_split_rise_guard`, the guard of the 'target_gain' rule) and the
    point estimate of the rise on the held-out half. The held-out half must
    confirm the point on all of these; with no point that qualifies the
    manual thresholds are kept (Einar, 2026-11-26) and the reason says so.
    Without a reference the stop of the walk is used.

    The report half judges the stop: ``N``, ``E``, the decoy FDR and the
    split-read disagreement at the stop and at the manual thresholds, with
    bootstrap intervals, and ``m - a`` of the step into the stop and of the
    step out of it with their intervals. The stop is adopted unless the
    step into it significantly raises the error rate on the held-out fields
    (lower bound of ``m - a`` above 0), or a cap is set and the held-out FDR
    estimate is above it.

    :param arrays: as :func:`_dataset_choice_target_gain`.
    :param nuclei: owned nuclei per field.
    :param tune: the fields that choose.
    :param report: the held-out fields.
    :param axes: ``(name, values)`` per grid axis.
    :param m_pos: the grid position of the manual thresholds.
    :param config: from :func:`dataset_threshold_settings`: ``weights``,
        ``ci`` and ``target_fdr`` (None for no cap).
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :param reps: bootstrap replicates.
    :param seed: their seed.
    :returns: ``adopted``, ``auto_met`` (a frontier point was found),
        ``reason``, ``held_out_fdr``, ``thresholds`` (``chosen``, ``manual``,
        ``used``), ``tune`` and ``report`` (``chosen`` and ``manual`` numbers
        per half), ``knee`` (``position`` of the stop on the frontier, ``points``,
        ``steps`` on the tuning half, ``into`` and ``out_of``: ``m - a`` of the
        step into and out of the stop on the tuning and report halves) and
        ``grid`` (every setting on the tuning half).
    """
    weights = config["weights"]
    cap = config.get("target_fdr")
    shape = tuple(len(values) for _, values in axes)
    m_pos = tuple(int(i) for i in m_pos)
    nuclei = np.asarray(nuclei, np.int64)

    def counted(idx, draws=None):
        sums = _dataset_summed(arrays, idx, draws)
        point = _dataset_terms(sums, weights, k, scale)
        n = sums["real"].astype(float)
        out = {"n": n, "e": n * point["e_total"], "decoy_errors": n * point["fdr"] / 100.0}
        for name in weights:
            out[f"{name}_errors"] = n * point[f"term_{name}"]
        return out, point

    def row_numbers(idx, pos, draws_):
        tally, point = counted(idx)
        boot, boot_point = counted(idx, draws_)
        total = int(nuclei[np.asarray(idx, np.int64)].sum())
        flat = np.stack([arrays["real"], arrays["decoy"]], -1).reshape(len(arrays["real"]), 1, -1, 2)
        fdr_part = _chosen_summary(flat, nuclei, idx,
                                   (0, int(np.ravel_multi_index(pos, shape))), k, scale)
        return {"fields": len(idx), "nuclei": total, "n": int(tally["n"][pos]),
                "n_ci95": _ci(boot["n"][(slice(None),) + pos]),
                "genotyped_pct": 100.0 * float(tally["n"][pos]) / max(total, 1),
                "errors": _float(tally["e"][pos]),
                "errors_ci95": _ci(boot["e"][(slice(None),) + pos]),
                "decoy_errors": _float(tally["decoy_errors"][pos]),
                "split_errors": _float(tally["split_errors"][pos]),
                "fdr_per_100": fdr_part["fdr_per_100"], "fdr_ci95": fdr_part["fdr_ci95"],
                "split_disagreement_pct": _float(point["split_pct"][pos]),
                "split_disagreement_ci95": _ci(boot_point["split_pct"][(slice(None),) + pos]),
                "split_reference_nuclei": int(_dataset_summed(arrays, idx)["scored"][pos]),
                "split_floor_pct": _float(point["split_ref_pct"][pos])}

    def front_steps(idx, front_pos, draws_):
        point, _ = counted(idx)
        boot, _ = counted(idx, draws_)
        pick = lambda tally: np.array([tally[pos] for pos in front_pos], float)
        stack = lambda tally: np.stack([tally[(slice(None),) + pos] for pos in front_pos], 1)
        return _marginal_steps(pick(point["n"]), pick(point["e"]), stack(boot["n"]),
                               stack(boot["e"]))

    def as_values(pos):
        return {name: float(values[pos[a]]) for a, (name, values) in enumerate(axes)}

    positions = list(np.ndindex(*shape))
    chosen, reason, adopted, met = m_pos, None, False, False
    rule_used, fallback, reference_report = "knee", None, None
    reference: Dict[str, Any] = {"used": False, "yield": None, "fair_fdr": None,
                                 "split_read": None, "tune_nuclei": None, "n_ref": None,
                                 "errors_ref": None, "beating_points": 0, "report": None,
                                 "fallback": None}
    rows: List[Dict[str, Any]] = []
    knee: Dict[str, Any] = {"position": None, "points": [], "steps": [],
                            "tune": {"into": None, "out_of": None},
                            "report": {"into": None, "out_of": None}}
    held = {"estimate": None, "ci95": [None, None], "target": cap, "at_or_below_target": None}
    parts: Dict[str, Any] = {"tune": {}, "report": {}}
    if len(tune) < 2 or len(report) < 2:
        reason = (f"too few fields with nuclei to tune and report on ({len(tune)} tune, "
                  f"{len(report)} report; two or more each)")
    else:
        counts = np.stack([arrays["real"], arrays["decoy"]], -1)
        score = _grid_score(counts, nuclei, tune, k, scale, config["ci"])
        held_score = _grid_score(counts, nuclei, report, k, scale, True)
        tune_counts, tune_point = counted(tune)
        report_counts, report_point = counted(report)
        allowed = np.ones(shape, bool) if cap is None else score["bound"] <= cap
        eligible = [pos for pos in positions if allowed[pos]]
        ref_yield, ref_fdr = config.get("brieflow_yield"), config.get("brieflow_fair_fdr")
        ref_split = config.get("brieflow_split_read")
        use_reference = (config.get("rule") == "beat_reference" and ref_yield is not None
                         and ref_fdr is not None)
        total_tune = max(int(nuclei[np.asarray(tune, np.int64)].sum()), 1)
        total_report = max(int(nuclei[np.asarray(report, np.int64)].sum()), 1)
        _, guard_point = counted(tune, _bootstrap_rows(len(tune), int(reps), int(seed)))
        rise_hi, guard_ok = _split_rise_guard(guard_point["split_pct"], m_pos,
                                              config["split_guard"], shape)
        if use_reference:
            floor_share = float(tune_point["split_ref_pct"][m_pos]) / 100.0
            n_ref = ref_yield * total_tune
            e_ref = None if ref_split is None else n_ref * (
                float(weights.get("decoy", 0.0)) * ref_fdr / 100.0
                + float(weights.get("split", 0.0)) * max(ref_split - floor_share, 0.0))
            reference.update({"used": True, "yield": ref_yield, "fair_fdr": ref_fdr,
                              "split_read": ref_split, "tune_nuclei": total_tune,
                              "n_ref": n_ref, "errors_ref": e_ref})

        def beats(n_part, fdr_part, split_pct, total):
            yield_fraction = n_part / total
            split_ok = True if ref_split is None else bool(
                np.isfinite(split_pct) and split_pct / 100.0 <= ref_split)
            return bool(yield_fraction >= ref_yield and fdr_part <= ref_fdr and split_ok)

        for pos in positions:
            rows.append({**{name: values[pos[a]] for a, (name, values) in enumerate(axes)},
                         "n": int(tune_counts["n"][pos]), "errors": float(tune_counts["e"][pos]),
                         "decoy_errors": float(tune_counts["decoy_errors"][pos]),
                         **{f"{name}_errors": float(tune_counts[f"{name}_errors"][pos])
                            for name in weights if name != "decoy"},
                         "genotyped_pct": float(score["genotyped_pct"][pos]),
                         "fdr_per_100": float(score["fdr"][pos]),
                         "fdr_bound": float(score["bound"][pos]),
                         "split_disagreement_pct": _float(tune_point["split_pct"][pos]),
                         "held_out_n": int(report_counts["n"][pos]),
                         "held_out_errors": float(report_counts["e"][pos]),
                         "held_out_fdr": float(held_score["fdr"][pos]),
                         "held_out_fdr_hi": float(held_score["fdr_hi"][pos]),
                         "yield_fraction": float(tune_counts["n"][pos]) / total_tune,
                         "split_rise_hi": _float(rise_hi[pos]),
                         "split_passes": bool(guard_ok[pos]),
                         "beats_reference": bool(use_reference and guard_ok[pos] and beats(
                             float(tune_counts["n"][pos]), float(score["fdr"][pos]),
                             float(tune_point["split_pct"][pos]), total_tune)),
                         "within_cap": bool(allowed[pos]), "frontier": False, "walked": False,
                         "knee": False, "manual": pos == m_pos})
        if not eligible:
            reason = "no setting is at or below the FDR cap on the tuning half"
        else:
            e_eligible = [float(tune_counts["e"][pos]) for pos in eligible]
            n_eligible = [int(tune_counts["n"][pos]) for pos in eligible]
            on = _pareto(e_eligible, n_eligible)
            front = sorted((i for i, flag in enumerate(on) if flag),
                           key=lambda i: (n_eligible[i], e_eligible[i]))
            front_pos = [eligible[i] for i in front]
            draws_t = _bootstrap_rows(len(tune), int(reps), int(seed))
            draws_r = _bootstrap_rows(len(report), int(reps), int(seed))
            steps = front_steps(tune, front_pos, draws_t)
            first = next((j for j, step in enumerate(steps) if step["significant"]), None)
            position = len(front_pos) - 1 if first is None else first
            if use_reference:
                winners = [i for i, pos in enumerate(front_pos) if guard_ok[pos] and beats(
                    float(tune_counts["n"][pos]), float(score["fdr"][pos]),
                    float(tune_point["split_pct"][pos]), total_tune)]
                if winners:
                    position = max(winners, key=lambda i: (tune_counts["n"][front_pos[i]], -i))
                    rule_used = "beat_reference"
                else:
                    fallback = ("no frontier point beats the reference on both yield and error on "
                                "the tuning half; the manual thresholds are kept")
            chosen, met = front_pos[position], True
            if fallback is not None:
                chosen = m_pos
            report_steps = front_steps(report, front_pos, draws_r)
            index = {pos: i for i, pos in enumerate(positions)}
            for i, pos in enumerate(front_pos):
                rows[index[pos]]["frontier"] = True
                rows[index[pos]]["walked"] = i <= position
            rows[index[chosen]]["knee"] = True

            def side(listing):
                return {"into": listing[position - 1] if position > 0 else None,
                        "out_of": listing[position] if position < len(listing) else None}

            knee.update({"position": position, "steps": steps, "report_steps": report_steps,
                         "points": [{**as_values(pos), "n": float(tune_counts["n"][pos]),
                                     "errors": float(tune_counts["e"][pos]),
                                     "held_out_n": float(report_counts["n"][pos]),
                                     "held_out_errors": float(report_counts["e"][pos])}
                                    for pos in front_pos],
                         "tune": side(steps), "report": side(report_steps)})
    if reason is None or met:
        draws_t = _bootstrap_rows(len(tune), int(reps), int(seed))
        draws_r = _bootstrap_rows(len(report), int(reps), int(seed))
        for part, idx, draws_ in (("tune", tune, draws_t), ("report", report, draws_r)):
            if len(idx):
                parts[part] = {name: row_numbers(idx, pos, draws_)
                               for name, pos in (("chosen", chosen), ("manual", m_pos))}
        got = parts["report"].get("chosen") or {}
        estimate = got.get("fdr_per_100")
        held = {"estimate": estimate, "ci95": got.get("fdr_ci95", [None, None]), "target": cap,
                "at_or_below_target": None if cap is None or estimate is None
                else bool(estimate <= cap)}
    if reason is None:
        into = knee["report"]["into"]

        def shown(step):
            return "no step into it" if step is None else (
                f"m - a {step['diff']:.5f} (95 % CI {step['ci95'][0]:.5f} to {step['ci95'][1]:.5f})")

        if rule_used == "beat_reference":
            got = parts["report"]["chosen"]
            tallies = {"yield": got["n"] / total_report, "fdr": got["fdr_per_100"],
                       "split": got["split_disagreement_pct"]}
            confirmed = beats(got["n"], tallies["fdr"], tallies["split"]
                              if tallies["split"] is not None else float("nan"), total_report)
            rise = float(report_point["split_pct"][chosen] - report_point["split_pct"][m_pos])
            guard_held = not rise > config["split_guard"]
            confirmed = confirmed and guard_held
            reference_report = {**tallies, "beats_both": confirmed, "split_rise": _float(rise),
                                "split_guard_ok": guard_held}
            if chosen == m_pos:
                reason = "the most-cells point that beats the reference is the manual thresholds"
            elif not confirmed:
                reason = ("on the held-out fields the chosen point no longer beats the reference "
                          f"on both or raises the split-read disagreement over the guard "
                          f"({'n/a' if reference_report['split_rise'] is None else format(reference_report['split_rise'], '+.3f')} points, guard "
                          f"{config['split_guard']:g}): yield {tallies['yield']:.4f} (reference {ref_yield:g}), "
                          f"decoy FDR {tallies['fdr']:.4f} (reference {ref_fdr:g})"
                          + ("" if ref_split is None or tallies["split"] is None else
                             f", split-read {tallies['split'] / 100.0:.4f} (reference {ref_split:g})"))
            elif held["at_or_below_target"] is False:
                reason = (f"the held-out FDR estimate, {held['estimate']:.3f} per 100, is above the "
                          f"cap of {cap:.3f}")
            else:
                adopted = True
                reason = ("the point with the most cells that beats the reference on yield and "
                          "error on the tuning half and again on the held-out half")
        elif chosen == m_pos:
            reason = "the stop of the walk is the manual thresholds"
        elif into is not None and into["significant"]:
            reason = ("on the held-out fields the step into the stop significantly raises the "
                      f"error rate: {shown(into)}")
        elif held["at_or_below_target"] is False:
            reason = (f"the held-out FDR estimate, {held['estimate']:.3f} per 100, is above the "
                      f"cap of {cap:.3f}")
        else:
            adopted = True
            reason = ("the stop of the walk up the frontier; held-out step into it: "
                      f"{shown(into)}")
        if fallback is not None:
            adopted, reason = False, fallback
    reference.update({"rule_used": rule_used, "fallback": fallback, "report": reference_report,
                      "beating_points": sum(bool(r["beats_reference"]) for r in rows)})
    return {"adopted": adopted, "auto_met": met, "reason": reason, "held_out_fdr": held,
            "thresholds": {"chosen": as_values(chosen), "manual": as_values(m_pos),
                           "used": as_values(chosen if adopted else m_pos)},
            "tune": parts["tune"], "report": parts["report"], "knee": knee, "grid": rows,
            "reference": reference}


def tune_dataset_thresholds(caches: Mapping[str, str], settings: Mapping[str, Any],
                            library: Sequence[str], *, workers: int = 1,
                            tune_sites: Optional[Mapping[str, Sequence[int]]] = None,
                            extra_terms: Optional[Mapping[str, Mapping[Tuple[str, int], Any]]] = None,
                            cell_stores: Optional[Mapping[str, str]] = None,
                            reps: int = SPLIT_READ_REPS, seed: int = SPLIT_READ_SEED
                            ) -> Dict[str, Any]:
    """Choose one decode threshold set for a whole dataset (items 1033 and 1033c, alpha).

    Under ``ops_decode_dataset_rule`` 'beat_reference' (the default, item
    1033d) the choice is the frontier point with the most cells among those
    that beat the reference (Brieflow) on yield and error, confirmed on the
    held-out half; without a reference, or when no point beats it, the
    'knee' rule decides (:func:`_dataset_choice`); the paragraphs below
    describe the
    'target_gain' rule of schema 986 and earlier
    (:func:`_dataset_choice_target_gain`), which shares the replay of the
    caches, the grid and the split.

    The dataset auto tuner, an extension of the per-well tuner
    (:func:`tune_thresholds`) that replays the stored field caches of every
    well of a plate or run, with no image. Every field is called at every
    setting of the grid (:func:`dataset_threshold_settings`: read threshold
    by single-read quality by pixel quality, the entered values included)
    as :func:`decode_cached_well` would call it. The fields with nuclei, in
    well then site order, are split into a tuning half and a report half
    (:func:`_tune_report_split`, the split of the per-well tuner), or as
    ``tune_sites`` names them.

    The objective is ``f = Gn (1 - e_total)^2`` (:func:`_dataset_terms`):
    ``Gn`` the genotyped real calls, ``e_total`` the weighted error share.
    On the tuning half a setting passes when its decoy FDR -- judged as the
    per-well tuner judges it (:func:`_grid_score`) -- is at most the target
    and the upper end of the paired 95 % tile-bootstrap interval of its
    split-read disagreement minus that of the manual thresholds is at most
    the guard. The split-read reference is fixed: the split reads at the
    manual read threshold, the same for every setting. The passing setting
    with the highest ``f`` wins; ties go to the stricter setting. On the
    report half the paired bootstrap of ``f`` chosen minus ``f`` manual
    gives the gain and its interval; the choice is adopted only when the
    interval lies above 0 and the held-out FDR estimate (not its upper
    bound) is at or below the target, otherwise the manual thresholds stay and
    ``reason`` says why.

    :param caches: well -> its folder of field caches.
    :param settings: the settings of the run (:func:`decode_cached_well`).
    :param library: the guide barcodes.
    :param workers: processes calling fields at once.
    :param tune_sites: well -> the fields to tune on, such as the
        ``tune_sites`` of an auto run; None draws the split.
    :param extra_terms: further error terms, such as an in-silico spike-in
        or phenotype penetrance: name -> ``{(well, site): (errors, scored)}``,
        two arrays of the grid shape per field; a term counts only with a
        weight in ``ops_decode_dataset_weights``.
    :param cell_stores: well -> ``sbs/fov.ome.zarr``, under
        ``ops_read_assignment`` 'cell_mask' (:func:`decode_cached_well`).
    :param reps: tile-bootstrap replicates of the guard and the gain.
    :param seed: their seed.
    :returns: ``adopted``, ``reason``, ``thresholds`` (``chosen``, ``manual``
        and ``used``, each ``ops_read_threshold``, ``ops_single_read_quality``
        and ``ops_pixel_quality``), ``objective``, ``weights``,
        ``target_fdr``, ``target_source``, ``held_out_fdr`` (the estimate and
        95 % interval of the chosen setting on the report half, with the
        target), ``split_guard``, ``tune_sites`` and ``report_sites``
        (``[well, site]`` pairs), ``tune`` and ``report`` (the chosen and the
        manual setting's numbers on each half, and on the report half the
        ``gain`` with its interval), ``grid`` (every setting on the tuning
        half) and ``fields``.
    :raises ValueError: without decoys, for unknown weight names, or when
        a term's arrays do not have the grid shape.
    """
    from multiprocessing import get_context

    resolved = _replay_settings(settings)
    config = dataset_threshold_settings(resolved, decode_rule(resolved))
    manual = config["manual"]
    rule, library, decoys, crosstalk = _cached_rule(
        resolved, library, read_threshold=manual["ops_read_threshold"],
        single_read_quality=manual["ops_single_read_quality"],
        pixel_quality=manual["ops_pixel_quality"])
    k, scale = int(rule["decoy_k"]), float(rule["fdr_scale"])
    if k < 1:
        raise ValueError("ops_decode_threshold_mode 'dataset' chooses thresholds from "
                         "decoy calls; set ops_decoy_library_k to 1 or more")
    weights = config["weights"]
    extra_terms = dict(extra_terms or {})
    unknown = sorted(set(weights) - set(DATASET_WEIGHTS) - set(extra_terms))
    if unknown:
        raise ValueError(f"ops_decode_dataset_weights names terms {unknown} that no "
                         "extra_terms supplies")
    universe = {code: i for i, code in enumerate(list(library) + list(decoys))}
    decoy_index = np.zeros(len(universe), bool)
    decoy_index[len(library):] = True
    wells = sorted(caches)
    paths = [(well, path) for well in wells for path in _cache_paths(caches[well])]
    watched: Dict[str, set] = {}
    for well, path in paths:
        watched.setdefault(well, set()).update(_watched_objects(path).tolist())
    jobs = [(path, rule, library, decoys, crosstalk, config, frozenset(watched[well]),
             universe, (str(cell_stores[well]), well) if cell_stores and well in cell_stores
             else None) for well, path in paths]
    workers = max(1, min(int(workers), len(jobs)))
    if workers > 1:
        from .resource_log import _array_file_nbytes, _guard_workers

        workers = max(1, int(_guard_workers("ops_decode", workers,
                                            _array_file_nbytes(paths[0][1]),
                                            settings=resolved)))
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            fields = pool.starmap(_dataset_field, jobs, chunksize=1)
    else:
        fields = [_dataset_field(*job) for job in jobs]
    field_wells = [well for well, _ in paths]
    axes = [("ops_read_threshold", config["spot_thresholds"]),
            ("ops_single_read_quality", config["floors"]),
            ("ops_pixel_quality", config["pixel_qualities"])]
    shape = tuple(len(values) for _, values in axes)
    position = {name: values for name, values in axes}
    m_pos = tuple(position[name].index(manual[name]) for name, _ in axes)
    counts = np.stack([f["counts"] for f in fields]) if fields \
        else np.zeros((0,) + shape + (2,), np.int64)
    nuclei = np.array([f["nuclei"] for f in fields], np.int64)
    arrays = {"real": counts[..., 0], "decoy": counts[..., 1],
              **_dataset_tallies(fields, field_wells, universe, decoy_index, shape)}
    for name, table in extra_terms.items():
        errors = np.zeros((len(fields),) + shape, float)
        scored = np.zeros((len(fields),) + shape, float)
        for i, (well, field) in enumerate(zip(field_wells, fields)):
            if (well, field["site"]) in table:
                e, t = (np.asarray(a, float) for a in table[(well, field["site"])])
                if e.shape != shape or t.shape != shape:
                    raise ValueError(f"extra term {name!r} needs arrays of the grid shape "
                                     f"{shape} per field; got {e.shape} and {t.shape}")
                errors[i], scored[i] = e, t
        arrays[f"{name}_errors"], arrays[f"{name}_scored"] = errors, scored

    keys = [(well, f["site"]) for well, f in zip(field_wells, fields)]
    order = [i for i in range(len(fields)) if nuclei[i] > 0]
    if tune_sites is None:
        tune, report = _tune_report_split(order, config["tune_fraction"], config["seed"])
    else:
        named = {(str(w), int(s)) for w, sites in tune_sites.items() for s in sites}
        tune = [i for i in order if keys[i] in named]
        report = [i for i in order if keys[i] not in named]

    legacy = config["rule"] == "target_gain"
    choose = _dataset_choice_target_gain if legacy else _dataset_choice
    choice = choose(arrays, nuclei, tune, report, axes, m_pos, config,
                    k=k, scale=scale, reps=reps, seed=seed)
    objective = ("f = Gn (1 - e_total)^2; e_total = sum of weight x term, clipped "
                 "to [0, 1]; split = max(Sd - Sd_ref, 0), decoy = FDR share" if legacy else
                 "beat_reference: the frontier point with the most calls whose yield, decoy "
                 "FDR and (if given) split-read disagreement all beat the reference, "
                 "confirmed on the held-out half; if none does, or no reference is given: "
                 "walk up the frontier of calls N against estimated errors E = N x e_total "
                 "(e_total as weighted split and decoy shares) and stop before the first step "
                 "whose marginal error rate dE / dN is significantly above the average E / N")
    return {"mode": "dataset", **{key: choice[key] for key in (
                "adopted", "auto_met", "reason", "held_out_fdr", "thresholds")},
            **({} if legacy else {"knee": choice["knee"], "reference": choice["reference"]}),
            "rule": config["rule"], "objective": objective,
            "brieflow_fair_fdr": config["brieflow_fair_fdr"],
            "brieflow_yield": config["brieflow_yield"],
            "brieflow_split_read": config["brieflow_split_read"],
            "weights": weights, "target_fdr": config["target_fdr"],
            "target_source": config["target_source"],
            "split_guard": config["split_guard"], "ci": config["ci"],
            "tune_fraction": config["tune_fraction"], "seed": config["seed"],
            "reps": int(reps), "bootstrap_seed": int(seed),
            "axes": {name: [float(v) for v in values] for name, values in axes},
            "wells": wells, "tune_sites": [list(keys[i]) for i in tune],
            "report_sites": [list(keys[i]) for i in report],
            "fields": len(fields), "nuclei": int(nuclei.sum()),
            "tune": choice["tune"], "report": choice["report"], "grid": choice["grid"]}


def _float(value) -> Optional[float]:
    """A number for a report, None when it is missing.

    :param value: a number or NaN.
    :returns: the float, or None.
    """
    value = float(value)
    return value if np.isfinite(value) else None


def plot_dataset_frontier(result: Mapping[str, Any], path: str) -> str:
    """Draw the yield and error frontier of the dataset tuner (items 1033c and 1033d).

    Every grid setting on the tuning half is a grey dot (calls N against
    estimated errors E, both on log axes), the walk up the Pareto frontier
    a solid line, the frontier beyond the stop a dotted one, the stop a
    filled dot, the manual thresholds a grey diamond and, when
    ``brieflow_fair_fdr`` is given, the decoy errors of the Brieflow fair
    FDR as a dotted line E = N x FDR / 100. With a reference
    (``ops_brieflow_yield`` and the fair FDR) the settings that beat it on
    both are green and the region that beats it is shaded: the cells at or
    above its yield and, when its split-read rate is given, the errors at or
    below its own (the reference point), otherwise only the yield; the chosen
    point is the filled dot.

    :param result: from :func:`tune_dataset_thresholds` with the knee rule.
    :param path: the image to write (``.png`` or ``.svg``); an ``.svg`` is
        written next to a ``.png``.
    :returns: ``path``.
    :raises ValueError: when the result is not a knee result.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    grid = result.get("grid") or []
    if not (result.get("knee") and grid):
        raise ValueError("plot_dataset_frontier needs the result of the knee rule")
    n = np.array([r["n"] for r in grid], float)
    e = np.maximum(np.array([r["errors"] for r in grid], float), _FIGURE_ERROR_FLOOR)
    flag = {key: np.array([bool(r[key]) for r in grid]) for key in ("frontier", "knee", "manual")}
    with plt.rc_context({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.scatter(n, e, s=6, color="#bdbdbd", linewidths=0, label="grid settings")
        ax.set_xscale("log")
        ax.set_yscale("log")
        walked = np.array([bool(r["walked"]) for r in grid])
        walk = np.flatnonzero(flag["frontier"] & walked)
        beyond = np.flatnonzero(flag["frontier"] & (~walked | flag["knee"]))
        for chain, width, style, label in ((walk, 1.2, "-", "walk up the frontier"),
                                           (beyond, 0.8, ":", "frontier beyond the stop")):
            chain = chain[np.argsort(n[chain])]
            if len(chain) > 1:
                ax.plot(n[chain], e[chain], color="#4d4d4d", lw=width, ls=style, label=label)
        fair = result.get("brieflow_fair_fdr")
        if fair:
            grid_n = np.linspace(n.min(), n.max(), 50)
            ax.plot(grid_n, np.maximum(grid_n * float(fair) / 100.0, _FIGURE_ERROR_FLOOR),
                    color="#8c6bb1", lw=0.8, ls=":", label=f"Brieflow fair FDR {float(fair):g}")
        reference = result.get("reference") or {}
        if reference.get("used"):
            beating = np.array([bool(r.get("beats_reference")) for r in grid])
            n_ref, e_ref = float(reference["n_ref"]), reference.get("errors_ref")
            if e_ref is not None:
                ax.scatter([n_ref], [max(float(e_ref), _FIGURE_ERROR_FLOOR)], s=22, marker="s",
                           color="#7a5195", zorder=3, label="Brieflow reference")
            ax.autoscale_view()
            low, high = ax.get_xlim()[0], max(ax.get_xlim()[1], n_ref)
            floor = ax.get_ylim()[0]
            top = ax.get_ylim()[1] if e_ref is None else max(float(e_ref), _FIGURE_ERROR_FLOOR)
            ax.fill_between([max(n_ref, low), high], [floor, floor], [top, top],
                            color="#1b9e77", alpha=0.15, linewidth=0, zorder=0,
                            label="beats Brieflow on both" if e_ref is not None
                            else "beats the Brieflow yield")
            if beating.any():
                ax.scatter(n[beating], e[beating], s=8, color="#1b9e77", linewidths=0, zorder=2)
        ax.scatter(n[flag["manual"]], e[flag["manual"]], s=28, marker="D", color="#969696",
                   edgecolors="black", linewidths=0.5, zorder=3, label="manual")
        ax.scatter(n[flag["knee"]], e[flag["knee"]], s=34, color="#d95f02", edgecolors="black",
                   linewidths=0.5, zorder=4, label="chosen")
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0), numticks=6))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("genotyped cells (tuning half)")
        ax.set_ylabel("estimated errors (tuning half)")
        ax.legend(frameon=False, fontsize=7, loc="center right")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        if path.lower().endswith(".png"):
            fig.savefig(path[:-4] + ".svg")
        plt.close(fig)
    return path


def write_dataset_thresholds(db: str, plate: str, result: Mapping[str, Any]) -> Dict[str, str]:
    """Write the choice of the dataset tuner (items 1033, 1033c) under results/ops_decode.

    :param db: the measurements database.
    :param plate: the plate name.
    :param result: from :func:`tune_dataset_thresholds`.
    :returns: ``json`` and ``grid``, the paths of
        ``decode_thresholds_dataset_<plate>.json`` and its grid as
        ``decode_thresholds_dataset_<plate>.csv``; under the knee rule also
        ``figure``, the frontier as ``decode_thresholds_dataset_<plate>.png``
        (and ``.svg``), when the grid is not empty.
    """
    import pandas as pd

    folder = results_folder(db)
    paths = {"json": os.path.join(folder, f"decode_thresholds_dataset_{plate}.json"),
             "grid": os.path.join(folder, f"decode_thresholds_dataset_{plate}.csv")}
    _atomic_json(paths["json"], {"plate": plate, **result}, indent=1, default=str)
    pd.DataFrame(result.get("grid") or []).to_csv(paths["grid"], index=False)
    if result.get("knee") and result.get("grid"):
        paths["figure"] = plot_dataset_frontier(
            result, os.path.join(folder, f"decode_thresholds_dataset_{plate}.png"))
    return paths


def write_well_reports(db: str, plate: str, well: str, decoded: Sequence[Mapping[str, Any]],
                       rule: Mapping[str, Any], skipped_sites: Sequence[int]
                       ) -> Dict[str, Any]:
    """Write field statistics, FDR report and plate summary for one well.

    Never calibration writes its JSON and CSV through the same report
    writer used by the prefix path, with point and interval fields folded
    into this report and the per-well plate summary.

    :param db: the measurements database.
    :param plate: the plate.
    :param well: the well.
    :param decoded: the decoded fields, each with ``site`` and ``rule_a``
        from :func:`field_calls`.
    :param rule: from :func:`decode_rule`; after auto thresholds with a
        shuffle calibration (item 949) its ``fdr_scale`` is the calibrated
        scale and its ``threshold_choice`` holds the calibration.
    :param skipped_sites: the unplaced sites.
    :returns: the FDR report for one well, with the paths written under ``paths``;
        with a shuffle calibration it adds ``fdr_calibration``,
        ``fdr_scale_entered``, ``shuffle_called`` for the whole well,
        ``shuffle_call_rate_pct``, ``shuffle_ci95``, ``shuffle_limit_pct``
        (1.5 times the calibrated FDR) and ``shuffle_le_limit``.
    """
    import pandas as pd

    from .tabular import write_table

    folder = results_folder(db)
    first = _read_assignment_of(rule) == "nucleus_first"
    reasons = BLEED_REASONS
    stats = pd.DataFrame([{"plate": plate, "well": well, "site": int(r["site"]),
                           "nuclei": int(r["rule_a"]["nuclei"]),
                           "assigned_real": int(r["rule_a"]["assigned_real"]),
                           "assigned_decoy": int(r["rule_a"]["assigned_decoy"]),
                           "assigned_pixel_real": int(r["rule_a"].get("assigned_pixel_real", 0)),
                           "assigned_pixel_decoy": int(r["rule_a"].get("assigned_pixel_decoy", 0)),
                           "nuclei_with_read": int(r["rule_a"].get("nuclei_with_read", 0)),
                           **{f"pixel_refused_{reason}": int(
                               (r["rule_a"].get("pixel_refused") or {}).get(reason, 0))
                              for reason in reasons}}
                          for r in sorted(decoded, key=lambda r: r["site"])],
                         columns=["plate", "well", "site", "nuclei",
                                  "assigned_real", "assigned_decoy",
                                  "assigned_pixel_real", "assigned_pixel_decoy",
                                  "nuclei_with_read"]
                         + [f"pixel_refused_{reason}" for reason in reasons])
    if first:
        for name in ("assigned_cell_mask_real", "assigned_cell_mask_decoy"):
            stats[name] = [int(r["rule_a"].get(name, 0))
                           for r in sorted(decoded, key=lambda r: r["site"])]
    if not rule.get("read_stage"):
        stats = stats.drop(columns=["nuclei_with_read"])
    if first:
        for reason in NUCLEUS_FIRST_REASONS:
            stats[f"cell_mask_refused_{reason}"] = [
                int(((r["rule_a"].get("nucleus_first") or {}).get("read_refused") or {})
                    .get(reason, 0)) for r in sorted(decoded, key=lambda r: r["site"])]
    if not (rule.get("pixels") and _rule_pixel_ownership(rule) == "read_rule"):
        stats = stats.drop(columns=[f"pixel_refused_{reason}" for reason in reasons])
    stats_path = os.path.join(folder, f"field_stats_{well}.csv")
    write_table(stats, stats_path, canonicalise=False)
    calibration = (rule.get('threshold_choice') or {}).get('calibration')
    report = fdr_report(stats, plate=plate, well=well, k=int(rule['decoy_k']),
                        scale=float(rule['fdr_scale']), skipped_sites=skipped_sites,
                        calibration=calibration)
    report["split_read_concordance"] = _well_split_concordance(decoded, rule)
    calibration_paths = {}
    if calibration and (str(calibration['method']).startswith('never')
                        or str(calibration.get('requested_method', '')).startswith('never')):
        calibration_paths = _write_calibration_report(folder, plate, well, calibration)
    report_path = os.path.join(folder, f"decode_fdr_{well}.json")
    _atomic_json(report_path, report, indent=1, sort_keys=False,
                 normalize=False, ensure_ascii=True, newline=False)
    summary = plate_fdr_summary(folder, plate=plate, k=int(rule["decoy_k"]),
                                scale=float(rule["fdr_scale"]))
    summary_path = os.path.join(folder, f"decode_fdr_{plate}.json")
    _atomic_json(summary_path, summary, indent=1, sort_keys=False,
                 normalize=False, ensure_ascii=True, newline=False)
    report["paths"] = {"field_stats": stats_path, "fdr": report_path,
                       "plate_summary": summary_path, **calibration_paths}
    return report


#: ``call_source`` of an ``ops_barcodes`` row, by the call's ``source``
#: (:func:`_assemble_calls`): ``read`` for a call the reads made, ``pixel``
#: for a pixel-fallback call, so a strict analysis can drop the latter.
CALL_SOURCES = {"primary": "read", "pixel_fallback": "pixel"}


def barcode_rows(plate: str, well: str, decoded: Sequence[Mapping[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """The ``ops_barcodes`` rows of a well decoded by :func:`field_calls`.

    Decoy calls are counted for the FDR and never stored. ``call_source``
    says whether the reads or the pixel fallback made the call
    (:data:`CALL_SOURCES`).

    :param plate: the plate.
    :param well: the well.
    :param decoded: the decoded fields, each with ``rule_a`` and
        ``n_cycles``.
    :returns: one row per real call, in object order.
    """
    rows = []
    for result in decoded:
        for object_id, hit in result["rule_a"]["calls"].items():
            if hit["decoy"]:
                continue
            rows.append({"plate": plate, "well": well, "object_id": int(object_id),
                         "barcode": hit["barcode"], "quality": hit["quality"],
                         "n_reads": hit["n_reads"], "n_agreeing": hit["n_agreeing"],
                         "fraction": hit["fraction"],
                         "n_cycles": int(result["n_cycles"]),
                         "mapped_guide": hit["guide"],
                         "call_source": CALL_SOURCES[hit["source"]]})
    return sorted(rows, key=lambda row: row["object_id"])


#: The field-cache arrays that hold the field's nucleus positions:
#: every object the field's task held, as the engine's neighbour-bleed
#: discount reads them (:func:`spacr.ops_sbs._disc_neighbours`).
CACHE_NUCLEUS_KEYS = ("nuc_ids", "nuc_centroids", "nuc_areas")


def _neighbour_keys(evidence: Mapping[str, Any]) -> Tuple[str, ...]:
    """The evidence keys a field's neighbours are built from (items 1013, 1025).

    :param evidence: one field's evidence or field cache.
    :returns: ``('cell_labels',)`` under the cell-mask read assignment,
        whose neighbours are the cells that touch; otherwise
        :data:`CACHE_NUCLEUS_KEYS`.
    """
    if _evidence_text(evidence, "read_assignment") == "cell_mask":
        return ("cell_labels",)
    return CACHE_NUCLEUS_KEYS


def save_field_cache(folder: str, site: int, evidence: Mapping[str, np.ndarray],
                     objects: Optional[Mapping[str, Any]] = None) -> str:
    """Write one field's evidence as ``site<NNNN>.npz``.

    :param folder: the well's cache folder, created.
    :param site: the site.
    :param evidence: from :func:`field_evidence`.
    :param objects: the field task's ``ids``, ``centroids`` and ``areas``,
        stored unrounded as :data:`CACHE_NUCLEUS_KEYS` so a replay of the
        cache can apply ``ops_read_neighbour_discount``; None
        stores the evidence alone, as caches did before.
    :returns: the path written.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"site{int(site):04d}.npz")
    stored = dict(evidence)
    if objects is not None:
        stored.update({
            "nuc_ids": np.asarray(objects["ids"], np.int64).reshape(-1),
            "nuc_centroids": np.asarray(objects["centroids"], np.float64).reshape(-1, 2),
            "nuc_areas": np.asarray(objects["areas"], np.float64).reshape(-1)})
    np.savez_compressed(path, **stored)
    return path


def cached_neighbours(evidence: Mapping[str, Any], reach: float) -> Dict[int, set]:
    """A field's touching nuclei, rebuilt from its field cache.

    :param evidence: a field cache opened with :func:`numpy.load` (or a dict
        of it) written with the nucleus positions (:func:`save_field_cache`).
    :param reach: the largest gap between two disc edges, ``ops_footprint``.
    :returns: object id -> its neighbours, as the engine builds them before
        the decode (:func:`spacr.ops_sbs._disc_neighbours`). Evidence under
        ``ops_read_assignment`` 'cell_mask' (item 1025) gives the cells that
        touch in its ``cell_labels`` (:func:`spacr.ops_sbs._cell_neighbours`),
        whatever the reach.
    :raises ValueError: for a cache written before the nucleus positions
        were stored, or cell-mask evidence without ``cell_labels``.
    """
    from .ops_sbs import _cell_neighbours, _disc_neighbours

    cell_mask = _evidence_text(evidence, "read_assignment") == "cell_mask"
    missing = [key for key in _neighbour_keys(evidence) if key not in evidence]
    if missing and cell_mask:
        raise ValueError("the cell-mask read assignment takes a field's neighbours from "
                         "the cells that touch, but this field's evidence holds no "
                         f"{', '.join(missing)}; replay it with the stored sbs_cell labels")
    if missing:
        from .qt.i18n import tr

        raise ValueError(tr("ops_read_neighbour_discount needs the field's nucleus positions; "
                            "this field cache was written before they were stored "
                            "(no {missing}); decode the field again to replay it "
                            "with the discount", missing=", ".join(missing)))
    if cell_mask:
        return _cell_neighbours(evidence["cell_labels"])
    return _disc_neighbours(evidence["nuc_ids"], evidence["nuc_centroids"],
                            evidence["nuc_areas"], float(reach))


def _rule_with_neighbours(rule: Mapping[str, Any], evidence: Mapping[str, Any],
                          footprint: float) -> Mapping[str, Any]:
    """The rule with the field's neighbours, read from a field cache when missing.

    :param rule: from :func:`decode_rule`; the engine adds ``neighbours``.
    :param evidence: the field's evidence or its field cache.
    :param footprint: the vote's footprint, the reach used when the rule
        carries neither ``neighbour_reach`` nor ``footprint``
        (:func:`_neighbour_reach`; the engine builds the neighbours at the
        same reach).
    :returns: ``rule`` itself unless its neighbour discount is on and it has
        no ``neighbours``; then a copy with :func:`cached_neighbours`.
        A cache without positions skips only the default discount, records
        applied False and emits one notice; explicit other discounts still raise.
    """
    weighting = rule.get("read_weight")
    if (not weighting or weighting["neighbour_discount"] == 1.0
            or rule.get("neighbours") is not None):
        return rule
    reach = _neighbour_reach(rule, footprint)
    if (weighting["neighbour_discount"] == _DEFAULTS["ops_read_neighbour_discount"]
            and _evidence_text(evidence, "read_assignment") != "cell_mask"
            and any(key not in evidence for key in CACHE_NUCLEUS_KEYS)):
        state = _neighbour_discount_state(None, 'cache_replay', False,
            discount=weighting["neighbour_discount"], method='cache_without_positions',
            state=rule.get('ops_read_neighbour_discount_applied'))
        if isinstance(rule, dict):
            rule['ops_read_neighbour_discount_applied'] = state
        return {**rule, "read_weight": {**weighting, "neighbour_discount": 1.0},
                'ops_read_neighbour_discount_applied': state}
    return {**rule, "neighbours": cached_neighbours(evidence, reach)}


def unplaced_sites(cycle_files: Mapping[int, Mapping[int, Any]],
                   placements: Mapping[int, Any]) -> List[int]:
    """The sites with tiles that the stitch did not place.

    :param cycle_files: ``cycle -> site -> channel -> path``.
    :param placements: ``site -> (y, x)``.
    :returns: the unplaced sites, sorted.
    """
    sites = {int(site) for files in cycle_files.values() for site in files}
    return sorted(site for site in sites if site not in placements)



#: The features the genotype confidence reads off each call's evidence
#: (item 877), in the order the reference model saw them.
CONFIDENCE_FEATURES = ("log_reads", "n_matched", "n_exact", "n_corr",
                       "vote_frac", "runner_margin", "q_best", "q_mean",
                       "log_strength", "n_amb", "pixel", "pix_n", "pix_frac",
                       "n_guides_seen", "field_reads_per_nucleus",
                       "field_nuclei", "q_best_missing", "vote_frac_missing")

#: The models :func:`genotype_confidence` fits; the setting names them.
CONFIDENCE_MODELS = {"logistic": "LR", "isotonic": "ISO", "hist_gbm": "HGB",
                     "strata": "STRATA"}

#: The seed of the reference confidence run (2026-11-07).
CONFIDENCE_SEED = 20261107


def _confidence_table(calls, folds: int = 5):
    """The evidence table with the confidence model's derived columns.

    :param calls: rows with :data:`EVIDENCE_COLUMNS` plus ``well`` and
        ``site``.
    :param folds: how many folds the field key is split into.
    :returns: a copy with ``fold``, ``pixel``, ``unanimous``, the ``_missing``
        flags, the logs, and the NaN fields filled with 0.
    """
    import hashlib

    d = calls.copy()
    d["fold"] = [int(hashlib.md5(f"{w}|{s}".encode()).hexdigest()[:8], 16) % int(folds)
                 for w, s in zip(d["well"], d["site"])]
    d["pixel"] = (d["source"] == "pixel_fallback").astype(float)
    d["unanimous"] = (d["runner_margin"] >= 0.999).astype(float)
    for column in ("q_best", "q_mean", "strength_max", "vote_frac"):
        d[f"{column}_missing"] = d[column].isna().astype(float)
        d[column] = d[column].fillna(0.0)
    d["log_reads"] = np.log1p(d["n_reads"])
    d["log_strength"] = np.log1p(d["strength_max"])
    return d


class _Logistic:
    """Logistic regression on standardised features, the reference's LR."""

    def __init__(self):
        """Build the scaler and the model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=2000)

    def fit(self, x, y, sample_weight=None):
        """Fit on ``x``.

        :param x: the features.
        :param y: the decoy labels.
        :param sample_weight: per-row weights.
        :returns: self.
        """
        self.scaler.fit(x)
        self.model.fit(self.scaler.transform(x), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        """Class probabilities.

        :param x: the features.
        :returns: ``(N, 2)``.
        """
        return self.model.predict_proba(self.scaler.transform(x))


def genotype_confidence(calls, *, model: str = "hist_gbm",
                        calibration: str = "target_decoy", bins: int = 10,
                        seed: int = CONFIDENCE_SEED, fdr_scale: float = 1.268,
                        decoy_k: int = 4, folds: int = 5,
                        max_real: int = 1_000_000) -> Dict[str, Any]:
    """Each call's calibrated probability of being correct (item 877).

    A call to a decoy is a known false call. Out of fold, by sequencing
    field (five folds keyed on the field's name, so a field is never scored
    by a model that saw it), four models estimate P(decoy | evidence) from
    each call's read and pixel support (:data:`CONFIDENCE_FEATURES`):
    logistic regression on the features and their squares, an isotonic
    recalibration of that score, histogram gradient boosting, and the decoy
    rate of matched strata. A real call's error is then
    ``c * P(decoy) / P(real)``, ``c = fdr_scale / decoy_k``, and
    ``genotype_confidence`` is one minus it. A control fits the logistic
    model on decoy labels shuffled within fields; its AUC should be 0.5.

    :param calls: one row per call, decoys included, with ``well``, ``site``
        and :data:`EVIDENCE_COLUMNS`, in decode order.
    :param model: which model's error becomes the confidence: ``hist_gbm``,
        ``logistic``, ``isotonic``, ``strata``, or ``auto`` for the lowest
        held-out log-loss.
    :param calibration: ``target_decoy`` converts P(decoy) to the error of a
        real call as above; ``none`` reports ``1 - P(decoy)``.
    :param bins: reliability bins, each expecting an equal number of decoys.
    :param seed: the subsampling, shuffling and boosting seed.
    :param fdr_scale: the decoy calibration (1.268 for the Funk library).
    :param decoy_k: decoys per guide.
    :param folds: cross-fitting folds, keyed on the md5 of ``well|site``.
    :param max_real: the most real calls each fit sees, weighted back.
    :returns: ``table`` (the calls with ``e_LR``, ``e_ISO``, ``e_HGB``,
        ``e_STRATA``, ``e`` and ``genotype_confidence``), ``metrics`` (per
        model AUC, log-loss, Brier and error counts, the chosen model),
        ``reliability`` (bin, predicted, observed, n per model) and
        ``strata``.
    :raises ValueError: for an unknown model or calibration, or calls with
        no decoy.
    """
    import pandas as pd
    from scipy.stats import chi2
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    if model != "auto" and model not in CONFIDENCE_MODELS:
        raise ValueError(f"ops_confidence_model must be one of "
                         f"{['auto'] + sorted(CONFIDENCE_MODELS)}; got {model!r}")
    if calibration not in ("target_decoy", "none"):
        raise ValueError("ops_confidence_calibration must be 'target_decoy' "
                         f"or 'none'; got {calibration!r}")
    d = _confidence_table(calls, folds)
    y = d["is_decoy"].to_numpy().astype(int)
    if not y.any() or y.all():
        raise ValueError("the genotype confidence needs decoy and real calls; "
                         "decode with ops_decoy_library_k above 0")
    c = float(fdr_scale) / float(decoy_k)
    features = list(CONFIDENCE_FEATURES)
    base = d[features].to_numpy(float)
    squared = np.c_[base, np.clip(base[:, :10], -50, 50) ** 2]
    fold = d["fold"].to_numpy()
    rng = np.random.default_rng(seed)

    def crossfit(make, x, labels, sub_real=max_real):
        """Out-of-fold P(decoy), each fit on its folds' decoys and real calls.

        :param make: builds an unfitted model.
        :param x: the features.
        :param labels: the decoy labels.
        :param sub_real: the most real calls a fit sees.
        :returns: one probability per call.
        """
        p = np.zeros(len(labels))
        for k in range(int(folds)):
            train = fold != k
            real_rows = np.flatnonzero(train & (labels == 0))
            decoy_rows = np.flatnonzero(train & (labels == 1))
            take = rng.choice(real_rows, min(sub_real, len(real_rows)), replace=False)
            rows = np.r_[take, decoy_rows]
            weight = np.r_[np.full(len(take), len(real_rows) / len(take)),
                           np.ones(len(decoy_rows))]
            fitted = make()
            fitted.fit(x[rows], labels[rows], sample_weight=weight)
            p[~train] = fitted.predict_proba(x[~train])[:, 1]
        return p

    p_lr = crossfit(_Logistic, squared, y)
    p_hgb = crossfit(lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0, random_state=seed), base, y)
    p_iso = np.zeros(len(y))
    score = np.log(np.clip(p_lr, 1e-9, 1))
    for k in range(int(folds)):
        train = fold != k
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            score[train], y[train])
        p_iso[~train] = iso.predict(score[~train])
    strata = (np.clip(d["n_reads"], 0, 5).astype(str) + "|" + d["source"] + "|"
              + d["unanimous"].astype(int).astype(str)).to_numpy()
    p_str = np.zeros(len(y))
    for k in range(int(folds)):
        train = fold != k
        rate = pd.Series(y[train]).groupby(strata[train]).mean()
        p_str[~train] = pd.Series(strata[~train]).map(rate).fillna(
            y[train].mean()).to_numpy()
    shuffled = y.copy()
    groups = (d["well"].astype(str) + "|" + d["site"].astype(str)).to_numpy()
    for _, index in pd.Series(np.arange(len(y))).groupby(groups).indices.items():
        shuffled[index] = y[index][rng.permutation(len(index))]
    p_sh = crossfit(_Logistic, squared, shuffled, sub_real=300_000)

    def error(p):
        """A real call's error from P(decoy).

        :param p: P(decoy).
        :returns: the error, 0 to 1.
        """
        if calibration == "none":
            return np.clip(p, 0, 1)
        return np.clip(c * p / np.clip(1 - p, 1e-12, None), 0, 1)

    def logloss(p, labels):
        """Mean log-loss.

        :param p: P(decoy).
        :param labels: the labels.
        :returns: the loss.
        """
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

    def reliability(p, name):
        """Reliability rows: bins of equal expected decoys.

        :param p: P(decoy).
        :param name: the model.
        :returns: one row per bin.
        """
        order = np.argsort(p)
        cumulative = np.cumsum(p[order])
        edges = np.searchsorted(cumulative, np.linspace(0, cumulative[-1], bins + 1)[1:-1])
        which = np.zeros(len(p), int)
        which[order] = np.searchsorted(edges, np.arange(len(p)), side="right")
        rows = []
        for b in range(int(bins)):
            m = which == b
            nd, nr = int(y[m].sum()), int((1 - y[m]).sum())
            lo = chi2.ppf(0.025, 2 * nd) / 2 if nd > 0 else 0.0
            hi = chi2.ppf(0.975, 2 * nd + 2) / 2
            rows.append({"model": name, "bin": b, "calls": int(m.sum()),
                         "real_calls": nr, "decoy_calls": nd,
                         "predicted_error": float(error(p[m])[y[m] == 0].mean()),
                         "observed_error": c * nd / max(nr, 1),
                         "observed_lo": c * lo / max(nr, 1),
                         "observed_hi": c * hi / max(nr, 1)})
        return rows

    probabilities = {"LR": p_lr, "ISO": p_iso, "HGB": p_hgb, "STRATA": p_str,
                     "control: shuffled decoy labels": p_sh}
    metrics: Dict[str, Any] = {}
    reliability_rows: List[Dict[str, Any]] = []
    for name, p in probabilities.items():
        labels = shuffled if name.startswith("control") else y
        e = error(p)
        metrics[name] = {
            "AUC_decoy": float(roc_auc_score(labels, p)), "logloss": logloss(p, labels),
            "brier": float(((p - labels) ** 2).mean()),
            "predicted_false_real_calls": float(e[labels == 0].sum()),
            "decoy_estimated_false_real_calls": float(c * labels.sum()),
            "mean_error_real": float(e[labels == 0].mean()),
            "real_calls_error_gt_1pct": int((e[labels == 0] > 0.01).sum()),
            "real_calls_error_gt_10pct": int((e[labels == 0] > 0.1).sum())}
        if not name.startswith("control"):
            reliability_rows += reliability(p, name)
    best = min(("LR", "ISO", "HGB", "STRATA"), key=lambda k: metrics[k]["logloss"])
    chosen = best if model == "auto" else CONFIDENCE_MODELS[model]
    metrics.update({"best_by_logloss": best, "chosen": chosen,
                    "rule": ("lowest held-out log-loss on the decoy label"
                             if model == "auto" else f"ops_confidence_model {model}"),
                    "fdr_scale_c": c, "calibration": calibration, "seed": int(seed)})
    table = calls.copy()
    for name in ("LR", "ISO", "HGB", "STRATA"):
        table[f"e_{name}"] = error(probabilities[name])
    table["e"] = table[f"e_{chosen}"]
    table["genotype_confidence"] = 1 - table["e"]
    stratum = pd.DataFrame({"stratum": strata, "is_decoy": d["is_decoy"].to_numpy(),
                            "e": table["e"].to_numpy()})
    strata_table = stratum.groupby("stratum").agg(
        calls=("is_decoy", "size"), decoys=("is_decoy", "sum"),
        model_error=("e", "mean")).reset_index()
    strata_table["observed_error"] = c * strata_table["decoys"] / (
        strata_table["calls"] - strata_table["decoys"]).clip(lower=1)
    return {"table": table, "metrics": metrics,
            "reliability": pd.DataFrame(reliability_rows), "strata": strata_table}


#: The sequence-only control features of the false-call model (item 876):
#: what a call's guide sequence alone says about whether it is a decoy.
SEQUENCE_FEATURES = ("_gc", "_homopolymer", "_frac_G", "_frac_T", "_frac_A",
                     "_frac_C", "nearest_other_guide")

#: The columns of the false-call cell table that are labels, keys or splits,
#: never features.
_QC_DROP = ("field", "obj", "label", "guide", "tp2", "test", "plate",
            "decoy_half")

#: The permissive decode the false-call model is trained on: a low read
#: threshold, matches at up to two mismatches, territory attribution and a
#: plain majority vote, so the decoy calls the strict rule refuses are there
#: to learn from.
QC_PERMISSIVE = {"threshold": 150.0, "footprint": 10.0, "max_distance": 2,
                 "tie": 1.0, "min_fraction": 0.6}


def _field_exact_rate(codes, real_matcher) -> float:
    """The share of a field's reads that match a guide exactly.

    :param codes: the field's called reads.
    :param real_matcher: a matcher over the real guides only.
    :returns: the share, 0 for a field without reads.
    """
    return float(np.mean([real_matcher.match(b, 0)[0] is not None for b in codes])
                 if codes else 0)


def false_call_tables(fields: Sequence[Mapping[str, Any]],
                      library: Sequence[str], decoys: Sequence[str], *,
                      seed: int = 7, tile_centre: float = 740.0,
                      permissive: Optional[Mapping[str, Any]] = None,
                      keep_reads: bool = False):
    """The read and cell tables the false-call model learns from (item 876).

    Each field is decoded permissively (:data:`QC_PERMISSIVE`) against the
    library and the decoys. Every read owned by a field's nucleus becomes a
    row with its match, per-cycle margins and mismatches, spot features and
    position; every owned nucleus becomes a row labelled ``FP`` (its reads
    vote for a decoy), ``TP`` (a guide) or ``TN`` (no call), with the mean
    evidence of its supporting reads and the sequence features of its guide.
    Half the fields of each plate are held out (``test``), and the decoys are
    split by the md5 of their sequence into a training half and an
    evaluation half (``decoy_half``).

    :param fields: ``{"plate", "well", "site", "z"}`` per field, ``z`` the
        field's evidence (:func:`field_evidence` with ``features=True``).
    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :param seed: the held-out field draw's seed (7 in the study).
    :param tile_centre: the tile centre, for each read's radius in the tile.
    :param permissive: overrides of :data:`QC_PERMISSIVE`.
    :param keep_reads: also return the read table, which is about ten times
        the size of the cell table; False returns an empty one.
    :returns: ``(reads, cells)`` DataFrames.
    """
    import hashlib
    from collections import Counter, defaultdict

    import pandas as pd

    from .ops_sbs import LibraryMatcher, called_bases

    rule = {**QC_PERMISSIVE, **(permissive or {})}
    library = sorted(set(library))
    decoy_set = set(decoys)
    real = LibraryMatcher(library)
    combined = LibraryMatcher(library + list(decoys))
    table = np.frombuffer("".join(library).encode(), np.uint8).reshape(len(library), -1)
    nearest: Dict[str, int] = {}

    def nearest_other(guide):
        """The fewest mismatches from a guide to another guide.

        :param guide: a library guide.
        :returns: the distance.
        """
        if guide not in nearest:
            distance = (table != np.frombuffer(guide.encode(), np.uint8)).sum(1)
            nearest[guide] = int(distance[distance > 0].min())
        return nearest[guide]

    rng = np.random.default_rng(seed)
    evaluation = {d for d in decoy_set if int(hashlib.md5(d.encode()).hexdigest(), 16) % 2}
    training = decoy_set - evaluation
    plates = np.array([str(f["plate"]) for f in fields])
    test_field = np.zeros(len(fields), bool)
    for plate in sorted(set(plates)):
        index = np.flatnonzero(plates == plate)
        test_field[rng.choice(index, len(index) // 2, replace=False)] = True
    all_reads, cell_frames = [], []
    cycle_keys = None
    for fi, fd in enumerate(fields):
        z = fd["z"]
        selected = np.asarray(z["strength"]) > rule["threshold"]
        first, _, _ = called_bases(np.asarray(z["values"])[selected], normalise=True, gpu=False)
        data, codes, margin = called_bases(first, compensate=True, method="argmax_median",
                                           normalise=False, gpu=False) if len(first) else \
            (first, [], np.zeros((0, np.asarray(z["values"]).shape[1])))
        gap = np.asarray(z["gap"])[selected]
        first_id = np.asarray(z["id0"])[selected]
        near = (gap[:, 0] <= rule["footprint"]) & np.asarray(z["owned0"])[selected]
        ambiguous = (gap[:, 1] - gap[:, 0]) < rule["tie"]
        peaks = np.asarray(z["peaks"])[selected]
        strength = np.asarray(z["strength"])[selected]
        spot = {k: np.asarray(z[k])[selected] for k in
                ("f_raw", "f_bg", "f_sharp", "f_focus", "f_density") if k in z}
        exact_rate = _field_exact_rate(codes, real)
        areas = dict(zip(np.asarray(z["owned_ids"]).tolist(),
                         np.asarray(z["owned_areas"]).tolist()))
        n_cycles = np.asarray(z["values"]).shape[1]
        cells = defaultdict(list)
        idx = np.flatnonzero(near)
        margin_rows = np.asarray(margin)[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            finite_margin = np.isfinite(margin_rows).any(1)
            q_rows = np.nanmin(margin_rows, axis=1)
            q_mean_rows = np.nanmean(margin_rows, axis=1)
            margin_out = np.nan_to_num(margin_rows, nan=-1)
            if spot:
                raw_rows = np.asarray(spot["f_raw"])[idx]
                bg_rows = np.asarray(spot["f_bg"])[idx]
                raw_ok = np.isfinite(raw_rows).any(2)
                bg_ok = np.isfinite(bg_rows).any(2)
                raw_max_rows = np.nanmax(raw_rows, axis=2)
                bg_mean_rows = np.nanmean(bg_rows, axis=2)
        q_list = np.where(finite_margin, q_rows, 0.0).tolist()
        q_mean_list = np.where(finite_margin, q_mean_rows, 0.0).tolist()
        margin_list = margin_out.tolist()
        if spot:
            raw_list = np.where(raw_ok, raw_max_rows, -1).tolist()
            bg_list = np.where(bg_ok, bg_mean_rows, -1).tolist()
            sharp_list = np.asarray(spot["f_sharp"])[idx].tolist()
            focus_list = np.asarray(spot["f_focus"])[idx].tolist()
            density_list = np.asarray(spot["f_density"])[idx].tolist()
        strength_list = strength[idx].tolist()
        gap0_list = gap[idx, 0].tolist()
        gap_diff_list = np.minimum(gap[idx, 1] - gap[idx, 0], 50).tolist()
        ambiguous_list = ambiguous[idx].tolist()
        y_list = peaks[idx, 0].astype(np.float64).tolist()
        x_list = peaks[idx, 1].astype(np.float64).tolist()
        radius_list = np.hypot(peaks[idx, 0] - tile_centre,
                               peaks[idx, 1] - tile_centre).tolist()
        for j, i in enumerate(idx.tolist()):
            guide, d = combined.match(codes[i], rule["max_distance"])
            mismatched = ([c for c, (x, yv) in enumerate(zip(codes[i], guide)) if x != yv]
                          if guide else [])
            row = {"field": fi, "obj": int(first_id[i]), "guide": guide,
                   "d": d if guide else -1,
                   "label_read": "TN" if guide is None else ("FP" if guide in decoy_set else "TP"),
                   "q": q_list[j],
                   "q_mean": q_mean_list[j],
                   "strength": strength_list[j], "gap0": gap0_list[j],
                   "gap_diff": gap_diff_list[j],
                   "ambiguous": bool(ambiguous_list[j]),
                   "y": y_list[j], "x": x_list[j],
                   "n_mismatch": len(mismatched),
                   "nearest_other_guide": (nearest_other(guide)
                                           if (guide and guide not in decoy_set) else
                                           (int(((table != np.frombuffer(guide.encode(), np.uint8)).sum(1)).min())
                                            if guide else -1))}
            miss = set(mismatched)
            for c in range(n_cycles):
                row[f"margin_c{c + 1}"] = margin_list[j][c]
                row[f"mismatch_c{c + 1}"] = int(c in miss)
                if spot:
                    row[f"raw_max_c{c + 1}"] = raw_list[j][c]
                    row[f"bg_c{c + 1}"] = bg_list[j][c]
            if spot:
                row["sharpness"] = sharp_list[j]
                row["focus_local"] = focus_list[j]
                row["neighbour_spots"] = int(density_list[j])
            row["field_exact_rate"] = float(exact_rate)
            row["radius_in_tile"] = radius_list[j]
            row["nucleus_area"] = float(areas.get(int(first_id[i]), np.nan))
            row["_seq"] = guide or ""
            if keep_reads:
                all_reads.append(row)
            cells[int(first_id[i])].append(row)
        if cycle_keys is None:
            cycle_keys = ([f"margin_c{i}" for i in range(1, n_cycles + 1)]
                          + [f"mismatch_c{i}" for i in range(1, n_cycles + 1)]
                          + [f"bg_c{i}" for i in range(1, n_cycles + 1)]
                          + [f"raw_max_c{i}" for i in range(1, n_cycles + 1)]
                          + ["sharpness", "focus_local", "neighbour_spots",
                             "radius_in_tile", "y", "x"])
        field_cells = []
        for owner in np.asarray(z["owned_ids"]).tolist():
            reads = cells.get(owner, [])
            matched = [r for r in reads if r["guide"]]
            cell = {"field": fi, "obj": owner, "n_reads": len(reads),
                    "n_matched": len(matched),
                    "nucleus_area": float(areas.get(owner, np.nan)),
                    "field_exact_rate": float(exact_rate)}
            label, guide = "TN", None
            if matched:
                tally = Counter(r["guide"] for r in matched)
                (guide, n), *rest = tally.most_common()
                if not (rest and rest[0][1] == n) and n / len(matched) >= rule["min_fraction"]:
                    label = "FP" if guide in decoy_set else "TP"
                else:
                    guide = None
            cell["label"] = label
            cell["guide"] = guide
            if guide:
                support = [r for r in matched if r["guide"] == guide]
                cell["n_support"] = len(support)
                cell["agreement"] = len(support) / len(matched)
                cell["n_exact_support"] = sum(r["d"] == 0 for r in support)
                cell["tp2"] = label == "TP" and cell["n_exact_support"] >= 2
                cell["min_mismatch"] = min(r["d"] for r in support)
                cell["q_best"] = max(r["q"] for r in support)
                cell["q_mean"] = float(np.mean([r["q_mean"] for r in support]))
                cell["strength_max"] = max(r["strength"] for r in support)
                cell["gap0_min"] = min(r["gap0"] for r in support)
                cell["ambiguous_share"] = float(np.mean([r["ambiguous"] for r in support]))
                cell["nearest_other_guide"] = support[0]["nearest_other_guide"]
                for key in cycle_keys:
                    if key in support[0]:
                        cell[key] = float(np.mean([r[key] for r in support]))
                runs = "".join(ch if i == 0 or guide[i - 1] == ch else "|" + ch
                               for i, ch in enumerate(guide)).split("|")
                cell["_gc"] = (guide.count("G") + guide.count("C")) / len(guide)
                cell["_homopolymer"] = max(len(x) for x in runs)
                for base in "GTAC":
                    cell[f"_frac_{base}"] = guide.count(base) / len(guide)
            field_cells.append(cell)
        cell_frames.append(pd.DataFrame(field_cells))
    reads = pd.DataFrame(all_reads)
    cells = (pd.concat(cell_frames, ignore_index=True, sort=False)
             if cell_frames else pd.DataFrame())
    for frame in (cells, reads):
        if len(frame):
            frame["test"] = test_field[frame["field"].to_numpy()]
            frame["plate"] = plates[frame["field"].to_numpy()]
            frame["decoy_half"] = frame["guide"].map(
                lambda g: "eval" if g in evaluation else ("train" if g in training else ""))
    return reads, cells


def _xgboost_classifier(params: Mapping[str, Any]):
    """An XGBoost classifier, imported only when asked for.

    :param params: the classifier's parameters.
    :returns: the unfitted classifier.
    :raises ImportError: with the install line, when xgboost is absent.
    """
    try:
        import xgboost
    except ImportError as error:
        raise ImportError(
            "ops_decode_qc_model 'xgboost' needs the xgboost package: "
            "pip install xgboost, or set ops_decode_qc_model to 'hist_gbm'"
        ) from error
    return xgboost.XGBClassifier(**params)


def _tree_shap(model, frame) -> Tuple[np.ndarray, float]:
    """Per-call SHAP values of a fitted tree model.

    :param model: a fitted XGBoost or scikit-learn tree classifier.
    :param frame: the features.
    :returns: ``(values, base_value)``. The shap package is used when it is
        installed; for XGBoost without it, the booster's own exact TreeSHAP
        (``pred_contribs``) gives the same values.
    """
    try:
        import shap
    except ImportError:
        shap = None
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(frame)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        return values, float(np.atleast_1d(explainer.expected_value)[-1])
    if hasattr(model, "get_booster"):
        import xgboost

        contributions = model.get_booster().predict(
            xgboost.DMatrix(frame), pred_contribs=True)
        return contributions[:, :-1], float(contributions[0, -1])
    raise ImportError("SHAP values of a hist_gbm model need the shap package: "
                      "pip install shap")


def false_call_model(cells, *, model: str = "xgboost", seed: int = 0,
                     shap_sample: int = 20000, permutation_repeats: int = 10,
                     n_jobs: int = 1, features: Optional[Sequence[str]] = None,
                     boot: int = 300) -> Dict[str, Any]:
    """What separates a decoy call from a library call (item 876).

    Trained on the training fields' library calls and training-half decoys,
    scored on the held-out fields' library calls and evaluation-half
    decoys, so neither a field nor a decoy sequence is both learnt and
    tested. A sequence-only model (:data:`SEQUENCE_FEATURES`) measures how
    much of the separation comes from how the decoys were built; chance is
    an AUROC of 0.5. Mean absolute SHAP (Lundberg and Lee 2017) and
    permutation importance (average-precision drop) rank the features.

    :param cells: from :func:`false_call_tables`.
    :param model: ``xgboost`` (an optional dependency) or ``hist_gbm``.
    :param seed: the model and resampling seed (0 in the study).
    :param shap_sample: how many held-out calls the SHAP ranking is taken
        over; 0 takes them all.
    :param permutation_repeats: permutation-importance repeats.
    :param n_jobs: model threads; 1 is reproducible bit for bit.
    :param features: the main model's features; None takes every column that
        is not a label, key, split or sequence feature.
    :param boot: field-bootstrap replicates for the AUROC and AP intervals.
    :returns: ``metrics`` (a DataFrame: main, sequence-only control, chance),
        ``shap_mean_abs`` and ``permutation_importance`` (DataFrames),
        ``model`` (the fitted main model) and ``features``.
    :raises ValueError: for an unknown model, or no held-out decoy.
    """
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    if model not in ("xgboost", "hist_gbm"):
        raise ValueError("ops_decode_qc_model must be 'xgboost' or 'hist_gbm'; "
                         f"got {model!r}")
    sequence = [c for c in cells.columns if c.startswith("_")] + ["nearest_other_guide"]
    main = list(features) if features else [
        c for c in cells.columns if c not in _QC_DROP and c not in sequence]
    assigned = cells[cells["label"].isin(["TP", "FP"])]
    train = assigned[~assigned["test"] & ((assigned["label"] == "TP")
                                          | (assigned["decoy_half"] == "train"))]
    test = assigned[assigned["test"] & ((assigned["label"] == "TP")
                                        | (assigned["decoy_half"] == "eval"))]
    y_train = (train["label"] == "FP").astype(int)
    y_test = (test["label"] == "FP").astype(int).to_numpy()
    if not y_test.any() or not y_train.any():
        raise ValueError("the false-call model needs decoy calls in the training "
                         "and the held-out fields; decode with ops_decoy_library_k "
                         "above 0 over more fields")

    def make():
        """An unfitted classifier of the chosen kind.

        :returns: the classifier.
        """
        if model == "xgboost":
            return _xgboost_classifier(dict(
                n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, n_jobs=n_jobs, eval_metric="logloss",
                random_state=seed))
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=400, max_depth=5,
                                              learning_rate=0.05, random_state=seed)

    rng = np.random.default_rng(seed)
    test_fields = test["field"].unique()
    field_of = test["field"].to_numpy()

    def interval(labels, p, score):
        """A field-bootstrap 95 % interval.

        :param labels: the held-out labels.
        :param p: the predictions.
        :param score: the metric.
        :returns: ``(low, high)``.
        """
        values = []
        for _ in range(int(boot)):
            picked = rng.choice(test_fields, len(test_fields))
            index = np.concatenate([np.flatnonzero(field_of == x) for x in picked])
            if labels[index].min() == labels[index].max():
                continue
            values.append(score(labels[index], p[index]))
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    rows, fitted = [], {}
    for name, columns in (("main_fp_vs_tp", main), ("control_sequence_only", sequence)):
        classifier = make()
        classifier.fit(train[columns].astype(float), y_train)
        p = classifier.predict_proba(test[columns].astype(float))[:, 1]
        rows.append({"model": name, "n_train": int(len(train)), "n_test": int(len(test)),
                     "fp_test": int(y_test.sum()),
                     "auc": round(roc_auc_score(y_test, p), 4),
                     "auc_ci95": interval(y_test, p, roc_auc_score),
                     "average_precision": round(average_precision_score(y_test, p), 4),
                     "ap_ci95": interval(y_test, p, average_precision_score),
                     "fp_prevalence_test": round(float(y_test.mean()), 4)})
        fitted[name] = classifier
    rows.append({"model": "chance", "n_train": int(len(train)), "n_test": int(len(test)),
                 "fp_test": int(y_test.sum()), "auc": 0.5, "auc_ci95": (0.5, 0.5),
                 "average_precision": round(float(y_test.mean()), 4),
                 "ap_ci95": (round(float(y_test.mean()), 4),) * 2,
                 "fp_prevalence_test": round(float(y_test.mean()), 4)})
    best = fitted["main_fp_vs_tp"]
    held_out = test[main].astype(float)
    sample = held_out.sample(min(int(shap_sample), len(held_out)), random_state=seed) \
        if shap_sample else held_out
    values, base_value = _tree_shap(best, sample)
    shap_mean = pd.Series(np.abs(values).mean(0), index=main).sort_values(ascending=False)
    permutation = permutation_importance(
        best, held_out, y_test,
        scoring=lambda est, x, y: average_precision_score(y, est.predict_proba(x)[:, 1]),
        n_repeats=int(permutation_repeats), random_state=seed, n_jobs=1)
    importance = pd.DataFrame({"feature": main,
                               "permutation_ap_drop_mean": permutation.importances_mean,
                               "permutation_ap_drop_sd": permutation.importances_std})
    return {"metrics": pd.DataFrame(rows),
            "shap_mean_abs": shap_mean.rename("mean_abs_shap").rename_axis("feature").reset_index(),
            "shap_base_value": base_value,
            "permutation_importance": importance.sort_values(
                "permutation_ap_drop_mean", ascending=False).reset_index(drop=True),
            "model": best, "features": main, "sequence_features": sequence}


def _write(frame, path: str) -> str:
    """Write a table by its suffix, column names as they are.

    :param frame: the table.
    :param path: the destination.
    :returns: the path written.
    """
    from .tabular import write_table

    return write_table(frame, path, canonicalise=False)


def write_well_calls(db: str, well: str, decoded: Sequence[Mapping[str, Any]]) -> str:
    """Write every call of a well with its evidence, decoys included.

    :param db: the measurements database.
    :param well: the well.
    :param decoded: the decoded fields, each with ``site`` and ``rule_a``
        carrying ``evidence``.
    :returns: the path of ``results/ops_decode/calls_<well>.parquet``.
    """
    import pandas as pd

    rows = [{"well": well, "site": int(r["site"]), **row}
            for r in sorted(decoded, key=lambda r: r["site"])
            for row in r["rule_a"].get("evidence", ())]
    frame = pd.DataFrame(rows, columns=["well", "site"] + list(EVIDENCE_COLUMNS))
    path = os.path.join(results_folder(db), f"calls_{well}.parquet")
    _write(frame, path)
    return path


def _qc_number(settings: Mapping[str, Any], key: str, default, cast=float):
    """One decode-QC setting, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :param default: the default.
    :param cast: ``float`` or ``int``.
    :returns: the value, or None when the default is None and it is empty.
    :raises ValueError: when it is not a number.
    """
    value = settings.get(key)
    if value is None or value == "":
        return default
    try:
        return cast(float(value)) if cast is int else cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number; got {value!r}") from None


def confidence_outputs(result: Mapping[str, Any], folder: str) -> Dict[str, Any]:
    """Write the genotype-confidence report (item 877).

    :param result: from :func:`genotype_confidence`.
    :param folder: ``results/ops_decode_qc``.
    :returns: ``confidence_metrics.json``'s content, with the paths.
    """
    reliability = result["reliability"]
    chosen = result["metrics"]["chosen"]
    mine = reliability[reliability["model"] == chosen]
    table = mine.rename(columns={"predicted_error": "predicted",
                                 "observed_error": "observed",
                                 "real_calls": "n"})[
        ["bin", "predicted", "observed", "n", "decoy_calls", "observed_lo",
         "observed_hi"]]
    reliability_path = os.path.join(folder, "confidence_reliability.csv")
    _write(table, reliability_path)
    _write(reliability, os.path.join(folder, "confidence_reliability_all_models.csv"))
    ece = float((mine["real_calls"] / max(int(mine["real_calls"].sum()), 1)
                 * (mine["predicted_error"] - mine["observed_error"]).abs()).sum())
    metrics = {"model": chosen, "auc": result["metrics"][chosen]["AUC_decoy"],
               "ece": ece,
               "shuffled_auc": result["metrics"]["control: shuffled decoy labels"]["AUC_decoy"],
               "models": result["metrics"]}
    metrics_path = os.path.join(folder, "confidence_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=1, default=str)
    _write(result["strata"], os.path.join(folder, "confidence_strata.csv"))
    calls_path = os.path.join(folder, "confidence_calls.parquet")
    _write(result["table"], calls_path)
    metrics["paths"] = {"reliability": reliability_path, "metrics": metrics_path,
                        "calls": calls_path}
    return metrics


def _with_confidence(db: str, plate: str, table, minimum) -> Dict[str, Any]:
    """Add ``genotype_confidence`` to ``ops_barcodes``, dropping calls below
    ``minimum``.

    :param db: the measurements database.
    :param plate: the plate.
    :param table: the confidence table: ``well``, ``object_id``,
        ``is_decoy``, ``genotype_confidence``.
    :param minimum: the lowest confidence kept, or None.
    :returns: rows updated and rows dropped.
    """
    from .ops_store import read_table, row_count, write_table

    if row_count(db, "ops_barcodes") is None:
        return {"updated": 0, "dropped": 0}
    barcodes = read_table(db, "ops_barcodes")
    real = table[~table["is_decoy"].astype(bool)][
        ["well", "object_id", "genotype_confidence"]].copy()
    real["plate"] = plate
    barcodes = barcodes.drop(columns=[c for c in ("genotype_confidence",)
                                      if c in barcodes.columns])
    merged = barcodes.merge(real, on=["plate", "well", "object_id"], how="left")
    dropped = 0
    if minimum is not None:
        low = (merged["plate"].astype(str) == str(plate)) & (
            merged["genotype_confidence"] < float(minimum))
        dropped = int(low.sum())
        merged = merged[~low].reset_index(drop=True)
    write_table(db, "ops_barcodes", merged, if_exists="replace")
    return {"updated": int(merged["genotype_confidence"].notna().sum()),
            "dropped": dropped}


def false_call_outputs(result: Mapping[str, Any], cells, folder: str) -> Dict[str, Any]:
    """Write the false-call model's report (item 876).

    :param result: from :func:`false_call_model`.
    :param cells: the cell table it was trained on.
    :param folder: ``results/ops_decode_qc``.
    :returns: the paths and the metric rows.
    """
    paths = {name: os.path.join(folder, name) for name in (
        "model_metrics.csv", "shap_mean_abs.csv", "permutation_importance.csv",
        "model.json", "false_call_cells.parquet")}
    _write(result["metrics"], paths["model_metrics.csv"])
    _write(result["shap_mean_abs"], paths["shap_mean_abs.csv"])
    _write(result["permutation_importance"], paths["permutation_importance.csv"])
    model = result["model"]
    if hasattr(model, "get_booster"):
        model.get_booster().save_model(paths["model.json"])
    else:
        with open(paths["model.json"], "w", encoding="utf-8") as handle:
            json.dump({"model": type(model).__name__,
                       "params": {k: str(v) for k, v in model.get_params().items()},
                       "features": list(result["features"])}, handle, indent=1)
    _write(cells, paths["false_call_cells.parquet"])
    return {"paths": paths, "metrics": result["metrics"].to_dict("records")}


def _cached_fields(destination: str, plate: str, wells: Sequence[str]):
    """The stored field evidence of the wells, in well and site order.

    :param destination: ``dst_root``.
    :param plate: the plate.
    :param wells: the wells.
    :returns: ``{"plate", "well", "site", "z"}`` per field; ``z`` opens lazily.
    """
    fields = []
    for well in sorted(wells):
        folder = os.path.join(destination, well, "decode_cache")
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith("site") and name.endswith(".npz"):
                fields.append({"plate": plate, "well": well, "site": int(name[4:8]),
                               "z": np.load(os.path.join(folder, name))})
    return fields


def run_decode_qc(db: str, plate: str, wells: Sequence[str],
                  settings: Mapping[str, Any], library: Sequence[str]
                  ) -> Dict[str, Any]:
    """The decode QC phase: the false-call model and the genotype confidence.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells decoded.
    :param settings: the run's settings; ``ops_decode_qc``,
        ``ops_genotype_confidence``, ``ops_decode_qc_bleed_panel`` and
        ``ops_decode_qc_depth_curve`` choose what runs.
    :param library: the guide barcodes.
    :returns: one report per part run, under ``false_call_model`` and
        ``genotype_confidence``.
    :raises ValueError: when a part's input is missing.
    """
    import pandas as pd

    from .ops_sbs import decoy_library
    from .tabular import read_table

    settings = resolve_settings(settings)
    rule = decode_rule(settings)
    folder = results_folder(db, "ops_decode_qc")
    report: Dict[str, Any] = {}
    if _flag_or(settings, "ops_genotype_confidence"):
        names = [os.path.join(results_folder(db), f"calls_{w}.parquet")
                 for w in sorted(wells)]
        present = [n for n in names if os.path.isfile(n)]
        if not present:
            raise ValueError("ops_genotype_confidence needs the per-call evidence "
                             "the decode writes under results/ops_decode; decode "
                             "with ops_decode_preset rule_a_s2 first")
        calls = pd.concat([read_table(n, canonicalise=False, report=None) for n in present], ignore_index=True)
        result = genotype_confidence(
            calls, model=str(settings.get("ops_confidence_model") or "hist_gbm"),
            calibration=str(settings.get("ops_confidence_calibration") or "target_decoy"),
            bins=_qc_number(settings, "ops_confidence_bins", 10, int),
            seed=_qc_number(settings, "ops_confidence_seed", CONFIDENCE_SEED, int),
            fdr_scale=rule["fdr_scale"], decoy_k=max(rule["decoy_k"], 1))
        metrics = confidence_outputs(result, folder)
        metrics["ops_barcodes"] = _with_confidence(
            db, plate, result["table"],
            _qc_number(settings, "ops_confidence_min", None))
        report["genotype_confidence"] = metrics
    if _flag_or(settings, "ops_decode_qc"):
        split = str(settings.get("ops_decode_qc_split") or "field").strip().lower()
        if split != "field":
            raise ValueError(f"ops_decode_qc_split must be 'field'; got {split!r}")
        destination = os.path.dirname(os.path.abspath(db))
        fields = _cached_fields(destination, plate, wells)
        if not fields:
            raise ValueError("ops_decode_qc needs the field evidence the decode "
                             "stores with ops_decode_qc on; decode again with it")
        decoys = decoy_library(library, k=max(rule["decoy_k"], 1))
        _, cells = false_call_tables(fields, library, decoys)
        chosen = settings.get("ops_decode_qc_features")
        features = ([f.strip() for f in str(chosen).split(",") if f.strip()]
                    if chosen else None)
        result = false_call_model(
            cells, model=str(settings.get("ops_decode_qc_model") or "xgboost"),
            seed=_qc_number(settings, "ops_decode_qc_seed", 0, int),
            shap_sample=_qc_number(settings, "ops_decode_qc_max_calls", 20000, int),
            features=features)
        report["false_call_model"] = false_call_outputs(result, cells, folder)
    if (_flag_or(settings, "ops_decode_qc_bleed_panel")
            or _flag_or(settings, "ops_decode_qc_depth_curve")):
        from .ops_decode_qc_panels import run_qc_panels

        report["qc_panels"] = run_qc_panels(db, plate, wells, settings, library, folder)
    return report


def _flag_or(settings: Mapping[str, Any], key: str) -> bool:
    """A boolean QC setting, False when empty.

    :param settings: the settings.
    :param key: the key.
    :returns: the value.
    """
    value = settings.get(key)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off")
    return bool(value)


#: How far apart, in pixels, two pipelines' nuclei may lie and still be one
#: cell (item 878); the three-way comparison of the decode-yield study used 4.
COMPARE_MATCH_PX = 4.0

#: How far from the tile border a cell must lie to be compared, in pixels.
COMPARE_EDGE_PX = 10


def match_units(tables: Mapping[str, Any], max_px: float = COMPARE_MATCH_PX
                ) -> List[Dict[str, int]]:
    """Group the cells of several call tables into matched units.

    Every pair of tables is linked by mutual nearest neighbours within
    ``max_px``; the links are joined shortest first, and a link that would
    put two cells of one table into a unit is skipped, so a unit holds at
    most one cell per table.

    :param tables: ``name -> DataFrame`` with ``y`` and ``x`` in one frame.
    :param max_px: the largest distance linked.
    :returns: one ``{name: row index}`` per unit, cells left alone included.
    """
    from itertools import combinations

    from scipy.spatial import cKDTree

    names = list(tables)
    edges = []
    for a, b in combinations(names, 2):
        ta, tb = tables[a], tables[b]
        if not len(ta) or not len(tb):
            continue
        pa, pb = ta[["y", "x"]].to_numpy(), tb[["y", "x"]].to_numpy()
        dab, iab = cKDTree(pb).query(pa)
        dba, iba = cKDTree(pa).query(pb)
        for i, (d, j) in enumerate(zip(dab, iab)):
            if d <= max_px and iba[j] == i:
                edges.append((d, (a, i), (b, j)))
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}
    members: Dict[Tuple[str, int], Dict[str, int]] = {}

    def find(node):
        """The root of a node's unit.

        :param node: ``(name, index)``.
        :returns: the root.
        """
        while parent.get(node, node) != node:
            node = parent[node]
        return node

    for name in names:
        for i in range(len(tables[name])):
            parent[(name, i)] = (name, i)
            members[(name, i)] = {name: i}
    for _, u, v in sorted(edges):
        ru, rv = find(u), find(v)
        if ru == rv or set(members[ru]) & set(members[rv]):
            continue
        parent[rv] = ru
        members[ru] = {**members[ru], **members.pop(rv)}
    return [members[r] for r in members if find(r) == r]


def call_category(calls: Mapping[str, Optional[str]],
                  labels: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """The agreement category of one matched unit.

    :param calls: ``name -> guide or None`` for the tables in the unit.
    :param labels: display names; None uses the table names.
    :returns: ``(category, subcategory)``: ``all3_same``, ``all3_diff``,
        ``two_agree_third_diff``, ``two_agree_third_none``,
        ``two_disagree_third_none``, ``one_only``, ``pair_agree``,
        ``pair_disagree`` or ``none``, and which table it names.
    """
    labels = labels or {name: name for name in calls}
    called = {m: g for m, g in calls.items() if g is not None}
    n = len(called)
    values = list(called.values())
    if n == 0:
        return "none", ""
    if n == 1:
        return "one_only", f"{labels[list(called)[0]]}-only"
    if len(calls) == 2 or n == 2:
        missing = [m for m in calls if calls[m] is None]
        joined = "+".join(labels[m] for m in called)
        if len(set(values)) == 1:
            if len(calls) == 2:
                return "pair_agree", joined
            return "two_agree_third_none", f"{joined} agree, {labels[missing[0]]} none"
        if len(calls) == 2:
            return "pair_disagree", joined
        return "two_disagree_third_none", f"{joined} differ, {labels[missing[0]]} none"
    if len(set(values)) == 1:
        return "all3_same", ""
    if len(set(values)) == 3:
        return "all3_diff", ""
    odd = [m for m in called if values.count(called[m]) == 1][0]
    return "two_agree_third_diff", f"{labels[odd]} differs"


def compare_field(tables: Mapping[str, Any], calls: Mapping[str, Mapping[int, str]],
                  decoys, *, max_px: float = COMPARE_MATCH_PX,
                  labels: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    """The matched units of one field and their categories.

    :param tables: ``name -> DataFrame`` of the field's compared cells, with
        ``cell``, ``y`` and ``x``.
    :param calls: ``name -> {cell: guide}``.
    :param decoys: the decoy set.
    :param max_px: the matching distance.
    :param labels: display names for the categories.
    :returns: one row per unit: ``present``, ``<name>_cell``,
        ``<name>_call``, ``<name>_decoy``, ``category``, ``subcategory``.
    """
    names = list(tables)
    single = all(len(n) == 1 for n in names)
    rows = []
    for unit in match_units(tables, max_px):
        present = ("" if single else "+").join(m for m in names if m in unit)
        row: Dict[str, Any] = {"present": present}
        found = {}
        for m in names:
            if m not in unit:
                row.update({f"{m}_cell": np.nan, f"{m}_call": None})
                continue
            cell = int(tables[m].iloc[unit[m]]["cell"])
            guide = calls[m].get(cell)
            found[m] = guide
            row.update({f"{m}_cell": cell, f"{m}_call": guide,
                        f"{m}_decoy": guide in decoys if guide else False})
        category, sub = call_category(found, labels)
        everyone = len(unit) == len(names)
        row["category"] = category if everyone else f"matched_{present}:{category}"
        row["subcategory"] = sub
        rows.append(row)
    return rows


def _reported_fdr_scale(db, well, entered, method):
    """Read the applied scale from the current well report for never methods.

    :param db: the production measurements database path.
    :param well: the production well identifier.
    :param entered: the scale to keep when no matching report exists.
    :param method: the requested calibration method; legacy methods keep
        their previous comparison and preview behavior.
    :returns: the reported scale for a matching never-method decode, or
        the entered scale. A stale calibration sidecar is never consulted.
    """
    if method not in ('never', 'never+shuffle'):
        return float(entered)
    path = os.path.join(os.path.dirname(os.path.abspath(db)), 'results', 'ops_decode',
                        f'decode_fdr_{well}.json')
    if not os.path.isfile(path):
        return float(entered)
    with open(path, encoding='utf-8') as handle:
        record = json.load(handle)
    if record.get('fdr_calibration') != method:
        return float(entered)
    return float(record.get('fdr_scale', entered))


def compare_summary(units, names: Sequence[str], *, decoy_k: int = 4,
                    fdr_scale: float = 1.268,
                    labels: Optional[Mapping[str, str]] = None):
    """Category counts with a decoy FDR per table, and per-table totals.

    :param units: rows from :func:`compare_field` with ``plate`` and
        ``field`` added. Optional per-row ``fdr_scale`` values carry the
        applied well scales; pooling weights decoy counts by those scales.
    :param names: the table names, in order.
    :param decoy_k: decoys per guide.
    :param fdr_scale: the FDR calibration.
    :param labels: display names for the columns.
    :returns: ``(categories, tables)`` DataFrames: per plate and category,
        ``n``, ``pct_of_matched_triples``, ``fields`` and real calls for each table
        with decoy calls and FDR; per plate and table over the units every
        table holds, real calls, assigned % and FDR.
    """
    import pandas as pd

    labels = labels or {name: name for name in names}
    single = all(len(n) == 1 for n in names)
    full = ("" if single else "+").join(names)
    u = units.copy()
    for m in names:
        if f"{m}_call" not in u:
            u[f"{m}_call"] = None
        u[f"{m}_called"] = u[f"{m}_call"].notna()
        u[f"{m}_decoy"] = (u[f"{m}_decoy"].eq(True) if f"{m}_decoy" in u
                           else False)
        u[f"{m}_real"] = u[f"{m}_called"] & ~u[f"{m}_decoy"]
    u["cat_full"] = np.where(u["subcategory"].fillna("") != "",
                             u["category"] + " | " + u["subcategory"].fillna(""),
                             u["category"])
    def call_fdr(rows, name, real, decoy):
        """Use one shared FDR calculation for either category or table totals.

        :param rows: the category or table rows.
        :param name: the table name.
        :param real: the selected real-call count.
        :param decoy: the selected decoy-call count.
        :returns: the rounded FDR at per-well scales, or the entered fallback.
        """
        if not real:
            return np.nan
        if 'fdr_scale' in rows:
            weighted = float(np.dot(rows[f'{name}_decoy'].to_numpy(float),
                                     np.maximum(rows['fdr_scale'].to_numpy(float), 1.0)))
            return round(fdr(real, weighted, decoy_k, 1.0), 3)
        return round(fdr(real, decoy, decoy_k, fdr_scale), 3)

    rows = []
    for plate, g in list(u.groupby("plate")) + [("ALL", u)]:
        triples = int((g["present"] == full).sum())
        for cat, h in g.groupby("cat_full"):
            present = h["present"].iloc[0]
            row = {"plate": plate, "category": cat, "present": present, "n": len(h),
                   "pct_of_matched_triples": (round(100 * len(h) / max(triples, 1), 3)
                                              if present == full else np.nan),
                   "fields": h["field"].nunique()}
            for m in names:
                if not h["present"].str.contains(m, regex=False).any():
                    continue
                real, decoy = int(h[f"{m}_real"].sum()), int(h[f"{m}_decoy"].sum())
                row.update({f"{labels[m]}_real_calls": real,
                            f"{labels[m]}_decoy_calls": decoy,
                            f"{labels[m]}_decoy_FDR_pct": call_fdr(h, m, real, decoy)})
            rows.append(row)
    triples = u[u["present"] == full]
    per_table = []
    for plate, g in list(triples.groupby("plate")) + [("ALL", triples)]:
        for m in names:
            real, decoy = int(g[f"{m}_real"].sum()), int(g[f"{m}_decoy"].sum())
            per_table.append({"plate": plate, "table": labels[m],
                              "matched_triples": len(g), "real_calls": real,
                              "assigned_pct": round(100 * real / max(len(g), 1), 2),
                              "decoy_calls": decoy,
                              "decoy_FDR_pct": call_fdr(g, m, real, decoy)})
    return pd.DataFrame(rows), pd.DataFrame(per_table)


def compare_region(frame, site: int, origin: Tuple[float, float], centres,
                   sites, *, tile: Tuple[int, int], edge: int = COMPARE_EDGE_PX):
    """The cells of one field that it owns and that lie clear of its border.

    :param frame: cells with ``y`` and ``x`` in the tile frame.
    :param site: the field.
    :param origin: the tile's ``(y, x)`` origin in the well, as
        :func:`spacr.ops_compose.tile_origins` gives it.
    :param centres: ``(N, 2)`` every placed tile's centre in the well.
    :param sites: the site of each centre.
    :param tile: the tile ``(height, width)``.
    :param edge: the border margin.
    :returns: the selected rows, re-indexed.
    """
    from scipy.spatial import cKDTree

    from .ops_compose import tile_to_well

    if not len(frame):
        return frame.reset_index(drop=True)
    well = tile_to_well(np.column_stack([frame["y"].to_numpy(),
                                         frame["x"].to_numpy()]),
                        origin, "float")
    _, nearest = cKDTree(centres).query(well)
    inside = ((frame["y"] >= edge) & (frame["y"] < tile[0] - edge)
              & (frame["x"] >= edge) & (frame["x"] < tile[1] - edge)).to_numpy()
    return frame[(np.asarray(sites)[nearest] == site) & inside].reset_index(drop=True)


def _compare_tables_setting(value) -> Dict[str, str]:
    """``ops_compare_tables`` as ``name -> path``.

    :param value: a dict, a JSON object string, or ``name=path`` pairs
        separated by commas.
    :returns: the mapping.
    :raises ValueError: when it names no table.
    """
    if isinstance(value, Mapping):
        out = {str(k): str(v) for k, v in value.items()}
    else:
        text = str(value or "").strip()
        if text.startswith("{"):
            out = {str(k): str(v) for k, v in json.loads(text).items()}
        else:
            out = dict(part.split("=", 1) for part in text.split(",") if "=" in part)
            out = {k.strip(): v.strip() for k, v in out.items()}
    if not out:
        raise ValueError("ops_compare_tables must name at least one external "
                         "call table, as name=path or a JSON object")
    return out


def run_compare_calls(db: str, plate: str, wells: Sequence[str],
                      settings: Mapping[str, Any], library: Sequence[str]
                      ) -> Dict[str, Any]:
    """The compare_calls phase: spaCR's genotype calls against other pipelines'.

    Each external table holds one row per cell with ``well``, ``site``,
    ``y`` and ``x`` in that site's tile frame (the sequencing cycle-1 tile)
    and ``call`` (or ``guide``), empty for an uncalled cell, optionally
    ``cell``. spaCR's side is every nucleus of the decoded fields with its
    call from ``results/ops_decode/calls_<well>.parquet``, decoys included.
    Cells are compared where their field owns them and at least
    :data:`COMPARE_EDGE_PX` from its border; matched units are categorised
    and each category gets a decoy FDR per table.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells decoded.
    :param settings: read for ``ops_compare_tables``,
        ``ops_compare_match_px``, ``ops_compare_fields`` (``well:site``
        pairs separated by commas, or empty for every site of the tables),
        ``ops_decoy_library_k``, ``ops_fdr_scale`` and
        ``ops_decode_fdr_calibration``. Never methods read applied scales
        from current per-well decode reports; missing reports keep the
        entered scale. Legacy methods retain their earlier tables.
    :param library: the guide barcodes, for the decoy set.
    :returns: the report, with the paths written under
        ``results/ops_compare``.
    :raises ValueError: when a table lacks its columns or the decode
        calls are missing.
    """
    import pandas as pd

    from .ops_compose import (tile_centres, tile_origin_rule, tile_origins,
                              well_to_tile)
    from .ops_sbs import decoy_library
    from .ops_store import read_table as read_ops_table
    from .tabular import read_table

    settings = resolve_settings(settings)
    rule = decode_rule(settings)
    origin_rule = tile_origin_rule(ops_setting(settings, "ops_tile_origin"))
    paths = _compare_tables_setting(settings.get("ops_compare_tables"))
    max_px = _qc_number(settings, "ops_compare_match_px", COMPARE_MATCH_PX)
    decoys = set(decoy_library(library, k=max(rule["decoy_k"], 1)))
    external = {}
    for name, path in paths.items():
        frame = read_table(path, canonicalise=False, report=None)
        if "call" not in frame and "guide" in frame:
            frame = frame.rename(columns={"guide": "call"})
        missing = {"well", "site", "y", "x", "call"} - set(frame.columns)
        if missing:
            raise ValueError(f"ops_compare_tables {name}: {path} lacks columns "
                             f"{sorted(missing)}")
        if "cell" not in frame:
            frame = frame.assign(cell=np.arange(1, len(frame) + 1))
        frame["call"] = frame["call"].where(frame["call"].astype(str).str.len() > 0)
        frame["call"] = frame["call"].where(frame["call"].notna(), None)
        external[name] = frame
    wanted = settings.get("ops_compare_fields")
    if wanted:
        fields = [(w.split(":")[0].strip().upper(), int(w.split(":")[1]))
                  for w in (wanted if isinstance(wanted, (list, tuple))
                            else str(wanted).split(",")) if ":" in str(w)]
    else:
        fields = sorted({(str(w), int(s)) for frame in external.values()
                         for w, s in zip(frame["well"], frame["site"])})
    method = str(ops_setting(settings, "ops_decode_fdr_calibration")).strip().lower()
    scales = {well: _reported_fdr_scale(db, well, rule["fdr_scale"], method)
              for well in {well for well, _site in fields}}
    geometry = read_ops_table(db, "ops_geometry")
    geometry = geometry[geometry["plate"].astype(str) == str(plate)]
    objects = read_ops_table(db, "ops_objects")
    objects = objects[objects["plate"].astype(str) == str(plate)]
    names = ["spaCR"] + list(external)
    units = []
    for well, site in fields:
        calls_path = os.path.join(results_folder(db), f"calls_{well}.parquet")
        if not os.path.isfile(calls_path):
            raise ValueError(f"compare_calls needs {calls_path}; decode well {well} "
                             "with ops_decode_preset rule_a_s2 first")
        spacr = read_table(calls_path, canonicalise=False, report=None)
        placed = geometry[(geometry["well"].astype(str) == well)
                          & (geometry["cycle"] == geometry["cycle"].min())]
        tile = (int(placed["tile_height"].iloc[0]), int(placed["tile_width"].iloc[0]))
        centres = tile_centres(placed[["y", "x"]].to_numpy(float), tile,
                               origin_rule)
        origin_row = placed[placed["site"] == site]
        if origin_row.empty:
            continue
        stored = (float(origin_row["y"].iloc[0]), float(origin_row["x"].iloc[0]))
        origin = tile_origins({site: stored}, origin_rule)[site]
        cells = objects[objects["well"].astype(str) == well]
        local = well_to_tile(np.column_stack([cells["centroid_y"].to_numpy(),
                                              cells["centroid_x"].to_numpy()]),
                             stored, origin_rule)
        mine = pd.DataFrame({"cell": cells["object_id"].to_numpy(),
                             "y": local[:, 0], "x": local[:, 1]})
        mine = mine[(mine["y"] > -30) & (mine["y"] < tile[0] + 30)
                    & (mine["x"] > -30) & (mine["x"] < tile[1] + 30)]
        tables = {"spaCR": compare_region(mine, site, origin, centres,
                                          placed["site"].to_numpy(), tile=tile)}
        hits = spacr[spacr["site"] == site]
        calls = {"spaCR": dict(zip(hits["object_id"].astype(int), hits["mapped_guide"]))}
        for name, frame in external.items():
            part = frame[(frame["well"].astype(str) == well) & (frame["site"] == site)]
            tables[name] = compare_region(part[["cell", "y", "x"]], site, origin,
                                          centres, placed["site"].to_numpy(), tile=tile)
            calls[name] = {int(c): g for c, g in zip(part["cell"], part["call"])
                           if isinstance(g, str)}
        for row in compare_field(tables, calls, decoys, max_px=max_px):
            row.update({"plate": plate, "well": well, "site": site,
                        "field": f"{plate}_{well}_{site}"})
            if method.startswith("never"):
                row["fdr_scale"] = scales[well]
            units.append(row)
    frame = pd.DataFrame(units)
    folder = results_folder(db, "ops_compare")
    if frame.empty:
        return {"units": 0, "fields": len(fields)}
    categories, per_table = compare_summary(frame, names, decoy_k=max(rule["decoy_k"], 1),
                                            fdr_scale=rule["fdr_scale"])
    out = {"units": int(len(frame)), "fields": len(fields),
           "paths": {"units": _write(frame, os.path.join(folder, "units.parquet")),
                     "categories": _write(categories, os.path.join(folder, "categories.csv")),
                     "tables": _write(per_table, os.path.join(folder, "tables.csv"))}}
    return out


#: How PERISCOPE's single-cell metadata names a cell's site, object and
#: call (item 922): the site key is ``<plate>-<well>-<site>``, the object
#: is the CellProfiler ``Cells`` ObjectNumber, which is the label of that
#: cell in ``Cells_Objects.tiff``.
PERISCOPE_CALL_COLUMNS: Dict[str, str] = {
    "site": "Metadata_Foci_site",
    "object": "Metadata_Cells_ObjectNumber",
    "barcode": "Metadata_Foci_Barcode_MatchedTo_Barcode",
    "gene": "Metadata_Foci_Barcode_MatchedTo_GeneCode",
    "quality": "Metadata_Foci_Cell_Quality",
}

#: The mask object of each reference object name in a spaCR phenotype mask
#: file (``<well>/phenotype_masks/site-<n>.npz``).
MASK_OBJECT_KEYS: Dict[str, str] = {"cells": "cells", "nuclei": "nuclei"}

#: The concordance categories of a matched cell pair, in report order.
CONCORDANCE_CATEGORIES = ("same_guide", "same_gene_other_guide", "disagree",
                          "spacr_only", "reference_only", "neither")


def parse_site_key(text: str) -> Tuple[str, str, int]:
    """Split a ``<plate>-<well>-<site>`` key such as ``CP186A-Well3-75``.

    :param text: the key; the plate may itself contain hyphens.
    :returns: ``(plate, well, site)``.
    :raises ValueError: when the key has fewer than three parts or the
        site is not a whole number.
    """
    parts = str(text).rsplit("-", 2)
    if len(parts) != 3 or not parts[2].strip().isdigit():
        raise ValueError(f"site key {text!r} is not <plate>-<well>-<site>")
    return parts[0], parts[1], int(parts[2])


def edge_labels(labels: np.ndarray) -> np.ndarray:
    """The labels of a label image that touch its border.

    :param labels: a 2-D label image, 0 for background.
    :returns: the sorted labels on the first or last row or column.
    """
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    found = np.unique(border)
    return found[found > 0]


def label_contingency(first: np.ndarray, second: np.ndarray):
    """Pixel overlap of every pair of objects of two label images, in one pass.

    :param first: a 2-D label image (spaCR), 0 for background.
    :param second: a label image of the same shape (the reference).
    :returns: ``(pairs, area_first, area_second)``: ``pairs`` has
        ``label_a``, ``label_b``, ``overlap`` (pixels both cover) and
        ``iou`` for every pair that shares a pixel, sorted by ``label_a``
        then ``label_b``; the areas are Series indexed by label.
    :raises ValueError: when the shapes differ.
    """
    import pandas as pd
    from .measure import _label_overlaps

    if first.shape != second.shape:
        raise ValueError(f"label images differ in shape: {first.shape} vs {second.shape}")
    a = np.asarray(first).ravel().astype(np.int64)
    b = np.asarray(second).ravel().astype(np.int64)
    ids_a, n_a = np.unique(a[a > 0], return_counts=True)
    ids_b, n_b = np.unique(b[b > 0], return_counts=True)
    area_a = pd.Series(n_a, index=ids_a, dtype=np.int64)
    area_b = pd.Series(n_b, index=ids_b, dtype=np.int64)
    label_a, label_b, overlap = _label_overlaps(a, b)
    pairs = pd.DataFrame({"label_a": label_a, "label_b": label_b,
                          "overlap": overlap})
    union = (area_a.reindex(pairs["label_a"]).to_numpy()
             + area_b.reindex(pairs["label_b"]).to_numpy() - pairs["overlap"].to_numpy())
    pairs["iou"] = pairs["overlap"].to_numpy() / np.maximum(union, 1)
    return pairs, area_a, area_b


def best_overlaps(pairs) -> Any:
    """Each first-image object's reference object of largest overlap.

    Ties are broken by the higher IoU, then the lower reference label, so
    the result does not depend on pixel order.

    :param pairs: the ``pairs`` of :func:`label_contingency`.
    :returns: one row per ``label_a`` with its chosen ``label_b``,
        ``overlap`` and ``iou``.
    """
    order = pairs.sort_values(["label_a", "overlap", "iou", "label_b"],
                              ascending=[True, False, False, True], kind="mergesort")
    return order.drop_duplicates("label_a").reset_index(drop=True)


def one_to_one(pairs, threshold: float = 0.5) -> Any:
    """Optimal one-to-one matching on IoU of the pairs at or above a threshold.

    The standard instance-segmentation pairing: a Hungarian assignment
    maximising summed IoU over the candidate pairs, which at a threshold
    above 0.5 equals keeping every candidate (two objects cannot both
    overlap a third by more than half).

    :param pairs: rows with ``label_a``, ``label_b`` and ``iou``.
    :param threshold: the smallest IoU a match may have.
    :returns: the matched rows, sorted by ``label_a``.
    """
    from scipy.optimize import linear_sum_assignment

    candidates = pairs[pairs["iou"] >= float(threshold)]
    if candidates.empty:
        return candidates.reset_index(drop=True)
    rows, row_of = np.unique(candidates["label_a"].to_numpy(), return_inverse=True)
    cols, col_of = np.unique(candidates["label_b"].to_numpy(), return_inverse=True)
    if len(rows) == len(candidates) and len(cols) == len(candidates):
        return candidates.sort_values("label_a").reset_index(drop=True)
    weight = np.zeros((len(rows), len(cols)))
    weight[row_of, col_of] = candidates["iou"].to_numpy()
    picked_rows, picked_cols = linear_sum_assignment(weight, maximize=True)
    keep = weight[picked_rows, picked_cols] > 0
    chosen = set(zip(rows[picked_rows[keep]].tolist(), cols[picked_cols[keep]].tolist()))
    mask = [(int(x), int(y)) in chosen for x, y in zip(candidates["label_a"], candidates["label_b"])]
    return candidates[mask].sort_values("label_a").reset_index(drop=True)


def match_masks(first: np.ndarray, second: np.ndarray, *, iou: float = 0.5,
                edge_policy: str = "exclude") -> Dict[str, Any]:
    """Match two segmentations of one field object by object.

    :param first: spaCR's label image.
    :param second: the reference label image, same shape.
    :param iou: the IoU a one-to-one match needs, for F1.
    :param edge_policy: ``"exclude"`` leaves objects that touch the image
        border in either image out of the counts (they are counted
        separately); ``"keep"`` counts them.
    :returns: ``{"matches", "counts"}``. ``matches`` has one row per spaCR
        object: ``spacr_id``, ``reference_id`` (largest overlap, 0 when
        none), ``overlap_px``, ``iou``, ``matched`` (in the one-to-one
        matching), ``matched_reference_id`` (its one-to-one partner, 0 when
        none), ``spacr_edge`` and ``reference_edge``. ``counts`` holds
        ``n_spacr``, ``n_reference``, ``tp``, ``fp``, ``fn``, ``f1``,
        ``median_iou`` (of the one-to-one matches), ``edge_spacr`` and
        ``edge_reference``.
    :raises ValueError: for an unknown edge policy or differing shapes.
    """
    import pandas as pd

    if edge_policy not in ("exclude", "keep"):
        raise ValueError(f"ops_compare_edge_policy must be 'exclude' or 'keep', not {edge_policy!r}")
    pairs, area_a, area_b = label_contingency(first, second)
    edge_a = set(edge_labels(first).tolist()) if edge_policy == "exclude" else set()
    edge_b = set(edge_labels(second).tolist()) if edge_policy == "exclude" else set()
    best = best_overlaps(pairs).set_index("label_a")
    kept = pairs[~pairs["label_a"].isin(edge_a) & ~pairs["label_b"].isin(edge_b)]
    pairs_matched = one_to_one(kept, iou)
    matched = dict(zip(pairs_matched["label_a"].astype(int), pairs_matched["label_b"].astype(int)))
    ids = area_a.index.to_numpy()
    reference = best["label_b"].reindex(ids).fillna(0).astype(np.int64).to_numpy()
    matches = pd.DataFrame({
        "spacr_id": ids.astype(np.int64),
        "reference_id": reference,
        "overlap_px": best["overlap"].reindex(ids).fillna(0).astype(np.int64).to_numpy(),
        "iou": best["iou"].reindex(ids).fillna(0.0).to_numpy(),
        "matched": [int(i) in matched for i in ids],
        "matched_reference_id": [matched.get(int(i), 0) for i in ids],
        "spacr_edge": [int(i) in edge_a for i in ids],
        "reference_edge": [int(r) in edge_b for r in reference]})
    n_a = int(sum(1 for i in ids if int(i) not in edge_a))
    n_b = int(sum(1 for i in area_b.index if int(i) not in edge_b))
    tp = len(pairs_matched)
    fp, fn = n_a - tp, n_b - tp
    counts = {"n_spacr": n_a, "n_reference": n_b, "tp": tp, "fp": fp, "fn": fn,
              "f1": (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else float("nan"),
              "median_iou": float(np.median(pairs_matched["iou"])) if tp else float("nan"),
              "edge_spacr": len(edge_a & set(ids.tolist())),
              "edge_reference": len(edge_b & set(area_b.index.tolist()))}
    return {"matches": matches, "counts": counts}


def centroid_matches(first: np.ndarray, second: np.ndarray, max_px: float = COMPARE_MATCH_PX
                     ) -> Any:
    """Each first-image object's nearest reference centroid within a distance.

    The centroid mode of the compare_calls phase, applied to the same two
    label images, as the control for the mask-overlap matching.

    :param first: spaCR's label image.
    :param second: the reference label image.
    :param max_px: the largest centroid distance of a match.
    :returns: ``spacr_id``, ``centroid_reference_id`` (0 when none) and
        ``distance_px`` per spaCR object.
    """
    import pandas as pd
    from scipy import ndimage
    from scipy.spatial import cKDTree

    def centres(labels):
        """Object ids and centroids of a label image."""
        ids = np.unique(labels)
        ids = ids[ids > 0]
        if not len(ids):
            return ids, np.zeros((0, 2))
        return ids, np.asarray(ndimage.center_of_mass(np.ones_like(labels), labels, ids), float)

    ids_a, xy_a = centres(first)
    ids_b, xy_b = centres(second)
    if not len(ids_a):
        return pd.DataFrame({"spacr_id": [], "centroid_reference_id": [], "distance_px": []})
    if not len(ids_b):
        return pd.DataFrame({"spacr_id": ids_a, "centroid_reference_id": 0, "distance_px": np.inf})
    distance, nearest = cKDTree(xy_b).query(xy_a)
    found = np.where(distance <= float(max_px), ids_b[nearest], 0)
    return pd.DataFrame({"spacr_id": ids_a.astype(np.int64),
                         "centroid_reference_id": found.astype(np.int64), "distance_px": distance})


def reference_cell_calls(table, columns: Optional[Mapping[str, str]] = None) -> Any:
    """A reference pipeline's per-cell calls keyed by plate, well, site and object.

    :param table: the reference call table, one row per called cell.
    :param columns: the column map with ``site`` (a ``<plate>-<well>-
        <site>`` key, or the site number when ``plate`` and ``well`` are
        mapped too), ``object``, ``barcode``, ``gene`` and optionally
        ``quality``, ``plate`` and ``well``; :data:`PERISCOPE_CALL_COLUMNS`
        when None.
    :returns: ``plate``, ``well``, ``site``, ``object``, ``barcode``,
        ``gene`` and ``quality`` (``"all"`` when not mapped); an empty
        barcode is no call.
    :raises ValueError: when a mapped column is missing.
    """
    import pandas as pd

    names = dict(PERISCOPE_CALL_COLUMNS if columns is None else columns)
    missing = [names[k] for k in ("site", "object", "barcode", "gene") if names.get(k) not in table.columns]
    if missing:
        raise ValueError(f"the reference call table lacks columns {missing}")
    if names.get("plate") in table.columns and names.get("well") in table.columns:
        plate = table[names["plate"]].astype(str).to_numpy()
        well = table[names["well"]].astype(str).to_numpy()
        site = table[names["site"]].astype(int).to_numpy()
    else:
        keys = [parse_site_key(k) for k in table[names["site"]].astype(str)]
        plate = np.array([k[0] for k in keys], dtype=object)
        well = np.array([k[1] for k in keys], dtype=object)
        site = np.array([k[2] for k in keys], dtype=np.int64)
    barcode = table[names["barcode"]].where(table[names["barcode"]].notna(), None)
    quality = (table[names["quality"]].astype(str).to_numpy() if names.get("quality") in table.columns
               else "all")
    out = pd.DataFrame({"plate": plate, "well": well, "site": site,
                        "object": table[names["object"]].astype(np.int64).to_numpy(),
                        "barcode": barcode.to_numpy(), "gene": table[names["gene"]].to_numpy(),
                        "quality": quality})
    out.loc[out["barcode"].astype(str).str.len().eq(0), "barcode"] = None
    return out


def cell_calls_from_reads(reads, *, cell: str = "cell", barcode: str = "barcode",
                          keys: Sequence[str] = ("well", "tile")) -> Any:
    """One call per cell from a per-read table: the barcode with most reads.

    Feldman et al. 2019 deposit reads, not cells; the rule pinned here is
    the top barcode by read count, and a tie for the top is no call.

    :param reads: one row per read with the key, cell and barcode columns.
    :param cell: the cell column; 0 or empty is a read outside any cell.
    :param barcode: the barcode column.
    :param keys: the columns that, with ``cell``, name one cell.
    :returns: per cell the keys, ``cell``, ``barcode`` (None on a tie),
        ``reads`` of the top barcode and ``total_reads``.
    """
    keys = [k for k in keys if k in reads.columns]
    inside = reads[reads[cell].notna() & (reads[cell].astype(str) != "0")]
    counts = inside.groupby(keys + [cell, barcode], sort=True).size().rename("reads").reset_index()
    counts = counts.sort_values(keys + [cell, "reads", barcode],
                                ascending=[True] * (len(keys) + 1) + [False, True], kind="mergesort")
    total = counts.groupby(keys + [cell])["reads"].transform("sum")
    second = counts.groupby(keys + [cell])["reads"].transform(lambda r: r.iloc[1] if len(r) > 1 else 0)
    counts = counts.assign(total_reads=total, runner_up=second)
    top = counts.drop_duplicates(keys + [cell]).reset_index(drop=True)
    top.loc[top["reads"] == top["runner_up"], barcode] = None
    return top.drop(columns="runner_up")


def concordance_category(spacr_guide, spacr_gene, ref_guide, ref_gene) -> str:
    """The concordance category of one matched cell pair.

    :param spacr_guide: spaCR's guide call or None.
    :param spacr_gene: spaCR's gene or None.
    :param ref_guide: the reference's guide call or None.
    :param ref_gene: the reference's gene or None.
    :returns: one of :data:`CONCORDANCE_CATEGORIES`.
    """
    def called(value):
        """True for a non-empty call."""
        return isinstance(value, str) and value != ""

    mine, theirs = called(spacr_guide), called(ref_guide)
    if mine and theirs:
        if spacr_guide == ref_guide:
            return "same_guide"
        if called(spacr_gene) and spacr_gene == ref_gene:
            return "same_gene_other_guide"
        return "disagree"
    if mine:
        return "spacr_only"
    if theirs:
        return "reference_only"
    return "neither"


def call_concordance(pairs, *, control: str = "observed") -> Any:
    """Per quality class, how many matched cells fall in each concordance category.

    :param pairs: one row per one-to-one matched cell pair with
        ``spacr_guide``, ``spacr_gene``, ``reference_guide``,
        ``reference_gene`` and ``quality``.
    :param control: the label of this table's rows (``observed`` or
        ``shuffled_seed<k>``).
    :returns: ``control``, ``quality`` (each class and ``all``),
        ``category``, ``n``, ``pct`` of the class's pairs and, on the
        ``both_called`` rows, the share with the same guide.
    """
    import pandas as pd

    cats = [concordance_category(a, b, c, d) for a, b, c, d in zip(
        pairs["spacr_guide"], pairs["spacr_gene"], pairs["reference_guide"], pairs["reference_gene"])]
    frame = pd.DataFrame({"quality": pairs["quality"].astype(str).to_numpy(), "category": cats})
    rows = []
    groups = [(q, g) for q, g in frame.groupby("quality", sort=True)] + [("all", frame)]
    for quality, group in groups:
        total = len(group)
        counts = group["category"].value_counts()
        for cat in CONCORDANCE_CATEGORIES:
            n = int(counts.get(cat, 0))
            rows.append({"control": control, "quality": quality, "category": cat, "n": n,
                         "pct": round(100 * n / total, 4) if total else float("nan")})
        both = int(sum(counts.get(c, 0) for c in ("same_guide", "same_gene_other_guide", "disagree")))
        rows.append({"control": control, "quality": quality, "category": "both_called", "n": both,
                     "pct": round(100 * int(counts.get("same_guide", 0)) / both, 4) if both else float("nan")})
    return pd.DataFrame(rows)


def shuffle_calls_within_site(calls, seed: int = 0) -> Any:
    """Reference calls permuted among the cells of each site.

    :param calls: :func:`reference_cell_calls` output.
    :param seed: the :func:`numpy.random.default_rng` seed.
    :returns: the same rows with ``barcode``, ``gene`` and ``quality``
        permuted within each plate, well and site.
    """
    rng = np.random.default_rng(int(seed))
    out = calls.sort_values(["plate", "well", "site", "object"], kind="mergesort").reset_index(drop=True)
    for _, index in out.groupby(["plate", "well", "site"], sort=True).indices.items():
        order = rng.permutation(len(index))
        for column in ("barcode", "gene", "quality"):
            out.loc[out.index[index], column] = out[column].to_numpy()[index][order]
    return out


def _mask_path(template: str, *, plate: str, well: str, site: int, obj: str, root: str) -> str:
    """Fill a mask path template; ``{tile}`` is a synonym of ``{site}``."""
    return str(template).format(plate=plate, well=well, site=site, tile=site, object=obj,
                                dst_root=root)


def _read_labels(path: str, obj: str) -> np.ndarray:
    """One label image: TIFF, an object's array of a spaCR ``.npz``, or an OME-Zarr field."""
    if os.path.isdir(path):
        from .image_store import read_labels

        return read_labels(path, MASK_OBJECT_KEYS.get(obj.lower(), obj.lower()))
    if path.endswith(".npz"):
        with np.load(path) as saved:
            key = MASK_OBJECT_KEYS.get(obj.lower(), obj.lower())
            if key not in saved:
                raise ValueError(f"{path} holds no {key!r} masks; it has {sorted(saved.files)}")
            return np.asarray(saved[key])
    import tifffile

    return np.asarray(tifffile.imread(path))


def _rescale_labels(labels: np.ndarray, shape: Tuple[int, int], factor) -> np.ndarray:
    """A label image brought to ``shape`` by an integer factor, or refused."""
    if labels.shape == tuple(shape):
        return labels
    if not factor:
        raise ValueError(f"spaCR masks are {labels.shape} and the reference masks {tuple(shape)}; "
                         "set ops_compare_mask_scale to compare them on one grid")
    step = int(factor)
    grown = np.repeat(np.repeat(labels, step, axis=0), step, axis=1)
    if grown.shape != tuple(shape):
        raise ValueError(f"spaCR masks scaled by {step} are {grown.shape}, not {tuple(shape)}")
    return grown


def _spacr_cell_calls(path: str, plate: str):
    """spaCR's per-cell guide calls, keyed like the reference calls."""
    import pandas as pd

    from .tabular import read_table

    table = read_table(path, canonicalise=False, report=None)
    guide = next((c for c in ("sgRNA", "mapped_guide", "guide") if c in table.columns), None)
    if guide is None or not {"well", "site", "label"} <= set(table.columns):
        raise ValueError(f"{path} needs well, site, label and sgRNA (or mapped_guide) columns")
    gene = table["gene"] if "gene" in table.columns else pd.Series([None] * len(table))
    return pd.DataFrame({"plate": table["plate"].astype(str) if "plate" in table.columns else plate,
                         "well": table["well"].astype(str).to_numpy(),
                         "site": table["site"].astype(np.int64).to_numpy(),
                         "object": table["label"].astype(np.int64).to_numpy(),
                         "barcode": table[guide].to_numpy(), "gene": gene.to_numpy(),
                         "quality": "all"})


def _call_pairs(matches, spacr_calls, ref_calls, key):
    """The one-to-one matched pairs of one site with both sides' calls."""
    import pandas as pd

    matched = matches[matches["matched"]]
    mine = {int(o): (b, g) for o, b, g in zip(spacr_calls["object"], spacr_calls["barcode"], spacr_calls["gene"])}
    theirs = {int(o): (b, g, q) for o, b, g, q in zip(ref_calls["object"], ref_calls["barcode"],
                                                     ref_calls["gene"], ref_calls["quality"])}
    rows = []
    for spacr_id, ref_id in zip(matched["spacr_id"].astype(int), matched["matched_reference_id"].astype(int)):
        a = mine.get(spacr_id, (None, None))
        b = theirs.get(ref_id, (None, None, "uncalled"))
        rows.append({**key, "spacr_id": spacr_id, "reference_id": ref_id,
                     "spacr_guide": a[0], "spacr_gene": a[1],
                     "reference_guide": b[0], "reference_gene": b[1], "quality": b[2]})
    return pd.DataFrame(rows, columns=list(key) + ["spacr_id", "reference_id", "spacr_guide",
                                                   "spacr_gene", "reference_guide", "reference_gene",
                                                   "quality"])


def _compare_fields(settings: Mapping[str, Any], ref_calls) -> List[Tuple[str, int]]:
    """The ``(well, site)`` fields of a mask comparison."""
    wanted = settings.get("ops_compare_fields")
    if wanted:
        items = wanted if isinstance(wanted, (list, tuple)) else str(wanted).split(",")
        return [(str(w).split(":")[0].strip(), int(str(w).split(":")[1])) for w in items if ":" in str(w)]
    if ref_calls is not None:
        return sorted({(str(w), int(s)) for w, s in zip(ref_calls["well"], ref_calls["site"])})
    raise ValueError("ops_compare_match 'mask_overlap' needs ops_compare_fields (well:site pairs) "
                     "or ops_compare_reference_calls to know which fields to compare")


def run_compare_masks(settings: Mapping[str, Any], report: Optional[Any] = None) -> Dict[str, Any]:
    """The compare_calls phase in mask-overlap mode (item 922).

    spaCR's label masks and a reference pipeline's are matched object by
    object on each field (:func:`match_masks`); the reference's per-cell
    calls are carried to spaCR's cells through the matches and compared
    with spaCR's (:func:`call_concordance`), next to two controls: the
    reference calls shuffled among the cells of each site, and centroid
    matching of the same cells (:func:`centroid_matches`).

    :param settings: read for ``plate``, ``dst_root``,
        ``ops_compare_reference_masks`` (a path template with ``{plate}``,
        ``{well}``, ``{site}`` or ``{tile}`` and ``{object}``),
        ``ops_compare_spacr_masks`` (the same; ``.npz`` files are spaCR
        phenotype masks and directories may name OME-Zarr fields; when
        omitted, discover committed phenotype fields or legacy
        ``{dst_root}/{well}/phenotype_masks/site-{site}.npz`` files),
        ``ops_compare_reference_objects``, ``ops_compare_iou``,
        ``ops_compare_edge_policy``, ``ops_compare_mask_scale``,
        ``ops_compare_reference_calls``, ``ops_compare_call_columns``,
        ``ops_compare_spacr_calls`` (a spaCR cell table with well, site,
        label and sgRNA, or ``"reference"`` for the self-match control),
        ``ops_compare_control_seed``, ``ops_compare_match_px`` and
        ``ops_compare_fields``.
    :param report: called with progress lines.
    :returns: the summary, with the paths written under
        ``<dst_root>/results/ops_compare``: ``match_cells.parquet``,
        ``mask_agreement.csv``, ``call_concordance.csv``,
        ``centroid_vs_overlap.csv``, ``reference_ids.csv`` and
        ``compare_masks.json``.
    :raises ValueError: for missing settings, files or columns, or masks on
        different grids without a scale factor.
    """
    import pandas as pd

    from .tabular import read_table

    say = report or (lambda text: None)
    root = str(settings.get("dst_root") or "")
    if not root:
        raise ValueError("ops_compare_match 'mask_overlap' needs dst_root, where results/ops_compare goes")
    template = settings.get("ops_compare_reference_masks")
    if not template:
        raise ValueError("ops_compare_match 'mask_overlap' needs ops_compare_reference_masks, a path "
                         "template with {well}, {site} and {object}")
    plate = str(settings.get("plate") or "")
    spacr_template = settings.get("ops_compare_spacr_masks")
    objects = list(settings.get("ops_compare_reference_objects") or ["Cells", "Nuclei"])
    threshold = float(settings.get("ops_compare_iou", 0.5))
    policy = str(settings.get("ops_compare_edge_policy") or "exclude")
    seed = int(settings.get("ops_compare_control_seed") or 0)
    max_px = float(settings.get("ops_compare_match_px") or COMPARE_MATCH_PX)
    ref_calls = None
    if settings.get("ops_compare_reference_calls"):
        columns = settings.get("ops_compare_call_columns")
        if isinstance(columns, str):
            columns = json.loads(columns)
        ref_calls = reference_cell_calls(read_table(str(settings["ops_compare_reference_calls"]),
                                                    canonicalise=False, report=None), columns)
        if plate:
            ref_calls = ref_calls[ref_calls["plate"].astype(str) == plate]
    spacr_source = settings.get("ops_compare_spacr_calls")
    if spacr_source == "reference":
        spacr_calls = ref_calls
    elif spacr_source:
        spacr_calls = _spacr_cell_calls(str(spacr_source), plate)
    else:
        default = os.path.join(root, f"ops_cells_{plate}.parquet")
        spacr_calls = _spacr_cell_calls(default, plate) if os.path.isfile(default) else None
    fields = _compare_fields(settings, ref_calls)
    call_object = next((o for o in objects if o.lower() == "cells"), objects[0])
    shuffled = shuffle_calls_within_site(ref_calls, seed) if ref_calls is not None else None
    matches_all, agreement, pairs_obs, pairs_shuf, centroid_rows, id_rows = [], [], [], [], [], []
    mask_paths = {}
    for number, (well, site) in enumerate(fields, 1):
        if not spacr_template and well not in mask_paths:
            from .ops_engine import _ph_mask_paths

            mask_paths[well] = _ph_mask_paths(root, plate, well, settings)
        for obj in objects:
            ref_path = _mask_path(template, plate=plate, well=well, site=site, obj=obj, root=root)
            mine_path = (_mask_path(spacr_template, plate=plate, well=well, site=site, obj=obj, root=root)
                         if spacr_template else mask_paths[well].get(site, os.path.join(
                             root, well, "phenotype_masks", f"site-{site}.npz")))
            for path in (ref_path, mine_path):
                if not os.path.isfile(path) and not os.path.isdir(path):
                    raise ValueError(f"compare_calls (mask_overlap) needs {path}")
            theirs = _read_labels(ref_path, obj)
            mine = _rescale_labels(_read_labels(mine_path, obj), theirs.shape,
                                   settings.get("ops_compare_mask_scale"))
            result = match_masks(mine, theirs, iou=threshold, edge_policy=policy)
            key = {"plate": plate, "well": well, "site": int(site), "object": obj}
            matches_all.append(result["matches"].assign(**key))
            agreement.append({**key, **result["counts"]})
            centroid = centroid_matches(mine, theirs, max_px)
            joined = result["matches"].merge(centroid, on="spacr_id", how="left")
            same = joined["centroid_reference_id"].fillna(0).astype(np.int64) == joined["reference_id"]
            centroid_rows.append({**key, "spacr_objects": int(len(joined)),
                                  "overlap_matched": int(joined["matched"].sum()),
                                  "centroid_matched": int((joined["centroid_reference_id"].fillna(0) > 0).sum()),
                                  "same_reference_object": int(same.sum()),
                                  "differ": int((~same).sum())})
            if obj != call_object or ref_calls is None:
                continue
            here = ref_calls[(ref_calls["well"].astype(str) == well) & (ref_calls["site"] == site)]
            present = set(np.unique(theirs).tolist())
            id_rows.append({**key, "reference_calls": int(len(here)),
                            "resolved": int(here["object"].isin(present).sum()),
                            "unresolved": int((~here["object"].isin(present)).sum())})
            if spacr_calls is None:
                continue
            mine_calls = spacr_calls[(spacr_calls["well"].astype(str) == well) & (spacr_calls["site"] == site)]
            pairs_obs.append(_call_pairs(result["matches"], mine_calls, here, key))
            here_shuffled = shuffled[(shuffled["well"].astype(str) == well) & (shuffled["site"] == site)]
            pairs_shuf.append(_call_pairs(result["matches"], mine_calls, here_shuffled, key))
        say(f"compare_calls (mask_overlap): field {well}:{site}, {number} of {len(fields)}")
    folder = os.path.join(root, "results", "ops_compare")
    os.makedirs(folder, exist_ok=True)
    per_site = pd.DataFrame(agreement)
    every_match = pd.concat(matches_all, ignore_index=True)
    well_rows = []
    for (well, obj), group in per_site.groupby(["well", "object"], sort=True):
        tp, fp, fn = int(group["tp"].sum()), int(group["fp"].sum()), int(group["fn"].sum())
        ious = every_match[(every_match["well"] == well) & (every_match["object"] == obj)
                           & every_match["matched"] & ~every_match["spacr_edge"] & ~every_match["reference_edge"]]
        ious = ious["iou"]
        well_rows.append({"plate": plate, "well": well, "site": "all", "object": obj,
                          "n_spacr": int(group["n_spacr"].sum()), "n_reference": int(group["n_reference"].sum()),
                          "tp": tp, "fp": fp, "fn": fn,
                          "f1": (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else float("nan"),
                          "median_iou": float(np.median(ious)) if len(ious) else float("nan"),
                          "edge_spacr": int(group["edge_spacr"].sum()),
                          "edge_reference": int(group["edge_reference"].sum())})
    agreement_table = pd.concat([per_site.assign(site=per_site["site"].astype(str)),
                                 pd.DataFrame(well_rows)], ignore_index=True)
    agreement_table["iou_threshold"] = threshold
    agreement_table["edge_policy"] = policy
    paths = {"matches": _write(every_match, os.path.join(folder, "match_cells.parquet")),
             "mask_agreement": _write(agreement_table, os.path.join(folder, "mask_agreement.csv")),
             "centroid_vs_overlap": _write(pd.DataFrame(centroid_rows),
                                           os.path.join(folder, "centroid_vs_overlap.csv"))}
    out: Dict[str, Any] = {"fields": len(fields), "objects": objects,
                           "agreement": [r for r in well_rows]}
    if id_rows:
        ids = pd.DataFrame(id_rows)
        paths["reference_ids"] = _write(ids, os.path.join(folder, "reference_ids.csv"))
        out["reference_calls"] = int(ids["reference_calls"].sum())
        out["reference_ids_unresolved"] = int(ids["unresolved"].sum())
    if pairs_obs:
        observed = pd.concat(pairs_obs, ignore_index=True)
        table = pd.concat([call_concordance(observed),
                           call_concordance(pd.concat(pairs_shuf, ignore_index=True),
                                            control=f"shuffled_seed{seed}")], ignore_index=True)
        paths["call_concordance"] = _write(table, os.path.join(folder, "call_concordance.csv"))
        both = table[(table["category"] == "both_called") & (table["quality"] == "all")]
        out["same_guide_pct_of_both_called"] = dict(zip(both["control"], both["pct"]))
    out["matching"] = ("spaCR and reference objects are matched on the same pixels of the same fields with "
                       f"the same IoU threshold ({threshold}) and edge policy ({policy}); the shuffled row "
                       "permutes reference calls within each site (seed "
                       f"{seed}); the centroid row matches the same objects by centroid within {max_px} px")
    out["paths"] = paths
    with open(os.path.join(folder, "compare_masks.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=str)
    out["paths"]["summary"] = os.path.join(folder, "compare_masks.json")
    return out
