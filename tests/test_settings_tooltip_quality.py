"""Quality gate for spacr.settings tooltips.

A tooltip is what the user reads in the GUI before touching a knob, so it has
to earn its place. The bar these tests enforce:

  * every declared setting has one,
  * it declares its type,
  * it says more than the key name already says (no tautologies such as
    ``amsgrad: "Whether to use AMSGrad optimizer."``),
  * it is long enough to have said what changes when you alter the value,
  * no key is silently shadowed by a duplicate entry.

No tooltip is exempt from these checks. The twelve entries that once sat in
``KNOWN_THIN`` are pinned below to the implementation facts established when
that blanket waiver was retired.
"""
from __future__ import annotations

import ast
import re
from typing import NamedTuple

import pytest

from spacr.settings import expected_types, tooltips

# Concrete facts read from the factories and consumers during the final audit.
# This mapping does not waive quality checks: every key still passes the same
# length and non-tautology rules as every other shipped tooltip.
VERIFIED_TOOLTIP_FACTS = {
    "backgrounds": ("Legacy compatibility", "cell_background", "does not alter"),
    "normalize_plots": ("Legacy compatibility", "do not read", "Default True"),
    "organelle_chann_dim": ("organelle_channel", "organelle_mask_dim", "Default None"),
    "visualize": (
        "always joins measurement tables",
        "crops cell images",
        "Default 'cell'",
    ),
    "from_scratch": ("randomly initialised weights", "pretrained model", "Default False"),
    "width_height": ("target_size", "does not change training", "Default [1000, 1000]"),
    "pathogen_model": ("CPSAM-architecture", "pathogen_model_name", "Default None"),
    "train": ("existing model_path", "without retraining", "Default True"),
    "train_channels": ("'r', 'g' and 'b'", "input tensor", "Default ['r', 'g', 'b']"),
    "rescale": ("30/diameter", "Cellpose", "Default False"),
    "pathogen_limit": ("Maximum pathogens per cell", "varies by module", "1000"),
    "save": ("optional disk artifacts", "three-item list", "varies by module"),
}

TYPE_PREFIX = re.compile(r"^\((?P<type>[^)]+)\)\s*-\s*(?P<body>.+)$", re.S)
DEFAULT_LITERAL = re.compile(
    r"(?i)\bDefault(?:\s+is|:)?\s+"
    r"(?P<value>\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|'[^']*'|\"[^\"]*\"|"
    r"None|True|False|-?\d+(?:\.\d+)?|empty|blank)"
)


class DefaultVariant(NamedTuple):
    """One intentional module/default difference from a shared tooltip."""

    claimed: str
    actual_repr: str
    classification: str
    reason: str


ACCURATE_SHARED = "accurate shared tooltip"
REPAIRED_TOOLTIP = "repaired module-specific tooltip"
CONFIG_DEFECT = "default/config defect"

# These are the complete, explicit dispositions for every mismatch witnessed
# by the mechanical default parser below.  Unlike the former digest, this says
# which module owns each value, what value it resolves to, and why that
# difference is accepted.  Adding, removing, or substituting one row fails an
# ordinary mapping comparison with a useful diff.
DEFAULT_VARIANT_EXPECTATIONS = {
    ("external_masks", "experiment"): DefaultVariant(
        "'experiment'", "'external_masks'", ACCURATE_SHARED,
        "The external-mask import names its own run after itself rather than "
        "taking the generic label, which is more useful in a results folder "
        "than a second directory called experiment; the shared tooltip is "
        "correct for every other module that offers the key.",
    ),
    ("analyze_plaques", "background"): DefaultVariant(
        "100", "200", ACCURATE_SHARED,
        "The tooltip explicitly names 200 for plaque analysis.",
    ),
    ("analyze_plaques", "fill_in"): DefaultVariant(
        "False", "True", REPAIRED_TOOLTIP,
        "Plaque analysis fills mask interiors on its initial run.",
    ),
    ("analyze_plaques", "resize"): DefaultVariant(
        "False", "True", ACCURATE_SHARED,
        "The tooltip explicitly names the plaque-analysis resize override.",
    ),
    ("analyze_plaques", "target_height"): DefaultVariant(
        "None", "1120", ACCURATE_SHARED,
        "The tooltip explicitly names the plaque-analysis height.",
    ),
    ("analyze_plaques", "target_width"): DefaultVariant(
        "None", "1120", ACCURATE_SHARED,
        "The tooltip explicitly names the plaque-analysis width.",
    ),
    ("classify_merged", "cmap"): DefaultVariant(
        "'inferno'", "'viridis'", ACCURATE_SHARED,
        "The classifier carries a plotting-specific viridis override.",
    ),
    ("classify_merged", "coordinate_columns"): DefaultVariant(
        "None", "['cell_id']", REPAIRED_TOOLTIP,
        "The classifier derives one object identifier from object_array.",
    ),
    ("classify_merged", "crop_source"): DefaultVariant(
        "'png'", "'load_images'", ACCURATE_SHARED,
        "The tooltip already distinguishes viewer and training spellings.",
    ),
    ("classify_merged", "loss_type"): DefaultVariant(
        "'focal_loss'", "'auto'", REPAIRED_TOOLTIP,
        "The merged classifier resolves auto from the output-head shape.",
    ),
    ("classify_merged", "min_cell_count"): DefaultVariant(
        "100", "25", ACCURATE_SHARED,
        "The tooltip explicitly names 25 for the screen classifier.",
    ),
    ("classify_merged", "nuclei_limit"): DefaultVariant(
        "None", "True", REPAIRED_TOOLTIP,
        "The classifier initially retains single-nucleus cells only.",
    ),
    ("classify_merged", "plot"): DefaultVariant(
        "False", "True", REPAIRED_TOOLTIP,
        "The classifier produces diagnostics on its initial run.",
    ),
    ("external_masks", "channels"): DefaultVariant(
        "[0,1,2,3]", "[]", REPAIRED_TOOLTIP,
        "An empty importer list deliberately means all detected channels.",
    ),
    ("external_masks", "cytoplasm"): DefaultVariant(
        "False", "True", ACCURATE_SHARED,
        "The importer intentionally enables the derived cytoplasm table.",
    ),
    ("external_masks", "dst"): DefaultVariant(
        "''", "None", ACCURATE_SHARED,
        "Both falsey values select the module-computed destination.",
    ),
    ("external_masks", "normalize"): DefaultVariant(
        "True", "False", REPAIRED_TOOLTIP,
        "The importer inherits Measure's no-normalization initial value.",
    ),
    ("external_masks", "organelle_min_size"): DefaultVariant(
        "10", "0", REPAIRED_TOOLTIP,
        "Existing primary-organelle labels are not size-filtered initially.",
    ),
                ("external_masks", "verbose"): DefaultVariant(
        "True", "False", ACCURATE_SHARED,
        "The tooltip already places Measure-family tools on the quiet path.",
    ),
    ("invasion", "cmap"): DefaultVariant(
        "'inferno'", "'viridis'", ACCURATE_SHARED,
        "Invasion uses its plotting-specific viridis override.",
    ),
    ("invasion", "intensity_statistic"): DefaultVariant(
        "'mean'", "'auto'", REPAIRED_TOOLTIP,
        "Auto prefers stable rim statistics and warns before using mean.",
    ),
    ("invasion", "level"): DefaultVariant(
        "'both'", "'object'", ACCURATE_SHARED,
        "The shared tooltip already explains assay-specific levels.",
    ),
    ("invasion", "pathogen_types"): DefaultVariant(
        "['pathogen_1', 'pathogen_2']", "['pc']", ACCURATE_SHARED,
        "The assay's one-condition plate layout deliberately uses pc.",
    ),
    ("invasion", "treatments"): DefaultVariant(
        "['cm','lovastatin']", "None", REPAIRED_TOOLTIP,
        "The assay begins without an invented treatment annotation.",
    ),
    ("invasion", "verbose"): DefaultVariant(
        "True", "False", REPAIRED_TOOLTIP,
        "Invasion starts without detailed console output.",
    ),
    ("investigate_hit", "score_column"): DefaultVariant(
        "'cv_predictions'", "''", REPAIRED_TOOLTIP,
        "Hit investigation requires the user to select its score field.",
    ),
    ("mask", "nucleus_intensity_threshold_method"): DefaultVariant(
        "75", "'mean'", ACCURATE_SHARED,
        "The parser sees the adjacent percentile value; the prose says mean.",
    ),
    ("measure", "normalize"): DefaultVariant(
        "True", "False", REPAIRED_TOOLTIP,
        "Measure starts off and requires a percentile pair when enabled.",
    ),
    ("measure", "organelle_min_size"): DefaultVariant(
        "10", "0", REPAIRED_TOOLTIP,
        "Measurement does not size-filter existing primary-organelle labels.",
    ),
                ("measure", "verbose"): DefaultVariant(
        "True", "False", ACCURATE_SHARED,
        "The tooltip already explicitly names Measure as quiet initially.",
    ),
    ("recruitment", "channel_of_interest"): DefaultVariant(
        "3", "2", ACCURATE_SHARED,
        "The tooltip already says non-ML modules may start at channel 2.",
    ),
    ("recruitment", "nuclei_limit"): DefaultVariant(
        "None", "1", REPAIRED_TOOLTIP,
        "Recruitment initially retains single-nucleus cells only.",
    ),
    ("recruitment", "plot"): DefaultVariant(
        "False", "True", REPAIRED_TOOLTIP,
        "Recruitment produces diagnostics on its initial run.",
    ),
    ("recruitment", "treatment_plate_metadata"): DefaultVariant(
        "None", "[['r1', 'r2', 'r3'], ['r4', 'r5', 'r6']]",
        REPAIRED_TOOLTIP,
        "Recruitment ships two treatment groups paired by position.",
    ),
    ("regression", "analysis_mode"): DefaultVariant(
        "'regression'", "'guide_permutation'", REPAIRED_TOOLTIP,
        "Nonparametric inference resolves the initial mode to permutation.",
    ),
    ("regression", "control_wells"): DefaultVariant(
        "None", "['c1', 'c2', 'c3']", REPAIRED_TOOLTIP,
        "Regression derives the controls from filter_value and control blocks.",
    ),
    ("regression", "fraction_threshold"): DefaultVariant(
        "None", "0.02", REPAIRED_TOOLTIP,
        "Regression uses a reproducible fixed fraction cutoff initially.",
    ),
    ("regression", "guide_nuisance_columns"): DefaultVariant(
        "[]", "['rowID', 'columnID']", REPAIRED_TOOLTIP,
        "Regression removes row and column position before permutation.",
    ),
    ("regression", "transform"): DefaultVariant(
        "None", "'log'", REPAIRED_TOOLTIP,
        "Regression applies log1p to the initial response.",
    ),
    ("regression", "verbose"): DefaultVariant(
        "True", "False", ACCURATE_SHARED,
        "The tooltip already explicitly names Regression as quiet initially.",
    ),
    ("replication", "cmap"): DefaultVariant(
        "'inferno'", "'viridis'", ACCURATE_SHARED,
        "Replication uses its plotting-specific viridis override.",
    ),
    ("replication", "level"): DefaultVariant(
        "'both'", "'object'", ACCURATE_SHARED,
        "The shared tooltip already explains assay-specific levels.",
    ),
    ("replication", "pathogen_types"): DefaultVariant(
        "['pathogen_1', 'pathogen_2']", "['pc']", ACCURATE_SHARED,
        "The assay's one-condition plate layout deliberately uses pc.",
    ),
    ("replication", "treatments"): DefaultVariant(
        "['cm','lovastatin']", "None", REPAIRED_TOOLTIP,
        "The assay begins without an invented treatment annotation.",
    ),
    ("replication", "verbose"): DefaultVariant(
        "True", "False", REPAIRED_TOOLTIP,
        "Replication starts without detailed console output.",
    ),
    ("umap", "crop_source"): DefaultVariant(
        "'png'", "'auto'", REPAIRED_TOOLTIP,
        "Image UMAP chooses PNGs first and otherwise streams merged arrays.",
    ),
    ("umap", "tables"): DefaultVariant(
        "['cell', 'nucleus', 'pathogen', 'cytoplasm']",
        "['cell', 'cytoplasm', 'nucleus', 'pathogen']", ACCURATE_SHARED,
        "The same four tables differ only in order.",
    ),
}

# Every repaired class-B variant has both a live-default assertion and prose
# fragments that must remain in the shared tooltip.  Repeated app/key rows are
# intentional: they prove each affected module contract, not just each source
# string, including generated organelle-slot tooltips.
REPAIRED_TOOLTIP_FACTS = {
    ("analyze_plaques", "fill_in"): (
        "Plaque Analysis starts with this enabled",
    ),
    ("classify_merged", "coordinate_columns"): (
        "Merged Classifier derives one identifier from object_array",
        "initially ['cell_id']",
    ),
    ("classify_merged", "loss_type"): (
        "Merged Classifier starts at 'auto'",
        "cross_entropy for a multi-class head",
        "binary_cross_entropy_with_logits for a single-logit head",
    ),
    ("classify_merged", "nuclei_limit"): (
        "Merged Classifier starts at True",
    ),
    ("classify_merged", "plot"): (
        "Merged Classifier and Recruitment both start with plotting enabled",
    ),
    ("external_masks", "channels"): (
        "External Masks starts with []",
        "means every detected intensity channel",
    ),
    ("external_masks", "normalize"): (
        "Measure and External Masks start at False",
        "two-number [low, high] percentile pair",
    ),
    ("external_masks", "organelle_min_size"): (
        "Measure and External Masks start every",
        "at 0 because they consume existing labels",
    ),
                ("invasion", "intensity_statistic"): (
        "Invasion starts at 'auto'",
        "chooses periphery_95 when present, otherwise percentile_95",
    ),
    ("invasion", "treatments"): (
        "Invasion and Replication start at None",
        "add no treatment condition",
    ),
    ("invasion", "verbose"): (
        "Invasion and Replication also start with console detail disabled",
    ),
    ("investigate_hit", "score_column"): (
        "Investigate Hit starts blank",
        "select the prediction column",
    ),
    ("measure", "normalize"): (
        "Measure and External Masks start at False",
        "refuses bare True",
    ),
    ("measure", "organelle_min_size"): (
        "Measure and External Masks start every",
        "at 0 because they consume existing labels",
    ),
                ("recruitment", "nuclei_limit"): (
        "Recruitment starts at 1",
    ),
    ("recruitment", "plot"): (
        "Merged Classifier and Recruitment both start with plotting enabled",
    ),
    ("recruitment", "treatment_plate_metadata"): (
        "Recruitment starts with [['r1', 'r2', 'r3'], ['r4', 'r5', 'r6']]",
        "paired with its two initial treatment names",
    ),
    ("regression", "analysis_mode"): (
        "starts with inference='nonparametric'",
        "resolved initial mode is 'guide_permutation'",
    ),
    ("regression", "control_wells"): (
        "Regression initializes it from filter_value plus any declared control blocks",
        "['c1', 'c2', 'c3']",
    ),
    ("regression", "fraction_threshold"): (
        "Regression starts at the reproducible fixed cutoff 0.02",
    ),
    ("regression", "guide_nuisance_columns"): (
        "Regression starts with ['rowID', 'columnID']",
        "before the within-plate permutation",
    ),
    ("regression", "transform"): (
        "Regression starts at 'log'",
        "first fit applies log1p",
    ),
    ("replication", "treatments"): (
        "Invasion and Replication start at None",
        "add no treatment condition",
    ),
    ("replication", "verbose"): (
        "Invasion and Replication also start with console detail disabled",
    ),
    ("umap", "crop_source"): (
        "Image UMAP starts at 'auto'",
        "otherwise streaming from merged arrays",
    ),
}


def _body(text: str) -> str:
    """Return the prose after the ``(type) - `` prefix."""
    m = TYPE_PREFIX.match(text.strip())
    return m.group("body").strip() if m else text.strip()


def _same_default(left, right):
    """Compare defaults without treating bool as an integer."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _all_tooltips():
    """Every setting key -> the tooltip shown beside its widget.

    ``spacr.settings.descriptions`` is deliberately NOT merged in: those 13
    entries are per-APP blurbs ('mask', 'measure', 'classify', 'umap', …)
    rendered on the app screen, not per-setting tooltips, and they are
    multi-line prose with no ``(type)`` prefix by design.
    """
    return {k: v for k, v in tooltips.items() if isinstance(v, str) and v.strip()}


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------

def test_every_tooltip_declares_its_type():
    """House format is ``"(type) - prose"``; the GUI shows the type hint."""
    missing = [k for k, v in _all_tooltips().items() if not TYPE_PREFIX.match(v.strip())]
    assert not missing, f"tooltips with no (type) prefix: {sorted(missing)[:20]}"


def test_no_duplicate_keys_in_the_tooltips_literal():
    """A repeated key in the dict literal silently discards the earlier text.

    This is not hypothetical: the literal carried ten shadowed entries
    before the 2026-07 audit, so ten tooltips were dead text nobody could
    ever see. An eleventh -- `crop_source` -- was found in 2026-08, and it
    was worse than dead text: the shadowed entry documented values the code
    rejects ('png', 'merged'), so the CORRECT tooltip only won by where it
    happened to sit in the file.
    """
    from collections import Counter

    keys = _literal_keys("tooltips")
    dupes = {k: c for k, c in Counter(keys).items() if c > 1}
    assert not dupes, f"duplicate keys shadow earlier tooltips: {dupes}"


def _literal_keys(dict_name: str):
    """Every key as it appears in a top-level dict literal in settings.py.

    Reading the SOURCE rather than the imported dict is the point: Python
    silently keeps only the last of a repeated key, so the live object
    cannot reveal the shadowing.

    Parsed with `ast`, not a regex. The regex version matched quoted names
    inside DESCRIPTION TEXT -- `png_channel_mapping`'s tooltip contains the
    literal ``{'r': 2, 'g': 1, 'b': 0}`` (see INVARIANTS 13), so r, g and b
    were reported as duplicate keys forever. Its brace counter also could
    not tell a `{` in prose from a real one, so the end of the literal
    moved with the wording. An AST cannot be fooled by either.
    """
    import ast
    import inspect

    import spacr.settings as st

    tree = ast.parse(inspect.getsource(st))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if dict_name in names and isinstance(node.value, ast.Dict):
            return [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    raise AssertionError(f"{dict_name} dict literal not found")

def test_no_duplicate_keys_in_expected_types():
    """A repeated key here silently changes a setting's declared type.

    ``src`` was declared ``(str, list)`` and then again as bare ``str`` fifty
    lines later; the second won, so the type contract disagreed with
    ``core.py`` and ``measure.py``, which both iterate a list of folders.
    ``check_settings`` coerces against this mapping, so the shadowing was
    load-bearing, not cosmetic.
    """
    from collections import Counter

    dupes = {k: c for k, c in Counter(_literal_keys("expected_types")).items() if c > 1}
    assert not dupes, f"duplicate keys shadow earlier type declarations: {dupes}"


def test_src_accepts_a_list_of_folders():
    """core.py and measure.py both loop over src when given a list."""
    assert expected_types["src"] == (str, list)


def test_tooltips_are_single_line_plain_text():
    """The Qt tooltip widget renders these verbatim: no newlines, no markdown."""
    bad = [k for k, v in _all_tooltips().items() if "\n" in v or "**" in v or "`" in v]
    assert not bad, f"tooltips containing newlines/markdown: {sorted(bad)[:20]}"


# ---------------------------------------------------------------------------
# informativeness
# ---------------------------------------------------------------------------

def test_no_tooltip_merely_restates_its_key():
    """``amsgrad: "Whether to use AMSGrad optimizer."`` is the rejected shape.

    A tooltip fails if every meaningful word of the key appears in the body and
    the body adds fewer than 12 further words — i.e. it names the setting again
    and stops.
    """
    offenders = []
    for key, text in _all_tooltips().items():
        body = _body(text)
        words = body.split()
        key_words = [w for w in key.lower().split("_") if len(w) > 2]
        if not key_words:
            continue
        low = body.lower()
        restates = all(w in low for w in key_words)
        if restates and len(words) < 12 + len(key_words):
            offenders.append((key, body[:70]))
    assert not offenders, (
        "tooltips that only restate their key:\n  "
        + "\n  ".join(f"{k}: {b}" for k, b in offenders)
    )


def test_tooltips_say_what_changes_when_you_alter_the_value():
    """Enforce a floor on substance.

    A tooltip that clears the tautology bar can still be a bare definition. We
    can't test for insight, but we can require enough prose to have expressed
    it — every audited tooltip comfortably clears 15 words.
    """
    thin = []
    for key, text in _all_tooltips().items():
        if len(_body(text).split()) < 15:
            thin.append((key, _body(text)))
    assert not thin, (
        "tooltips too short to say what changes when you alter them:\n  "
        + "\n  ".join(f"{k}: {b}" for k, b in thin)
    )


def test_unit_named_settings_keep_their_units_in_the_tooltip():
    """Names that encode a physical unit must explain that unit in prose."""
    tips = _all_tooltips()
    diameter_or_radius = [
        key for key in tips
        if key == "diameter"
        or key.endswith("_diameter")
        or key.endswith("_radius")
    ]
    missing = [
        key for key in diameter_or_radius
        if not re.search(
            r"(?i)\b(pixel|pixels|units?|micromet(?:er|re)s?)\b", tips[key]
        )
    ]
    missing.extend(
        key for key in tips
        if key.endswith("_px")
        and not re.search(r"(?i)\bpixels?\b", tips[key])
    )
    missing.extend(
        key for key in tips
        if key.endswith("_um")
        and not re.search(r"(?i)\bmicromet(?:er|re)s?\b", tips[key])
    )
    assert len(diameter_or_radius) == 110
    assert not missing, f"unit-bearing tooltips without units: {sorted(missing)}"


def test_real_default_claims_have_no_unrecorded_drift():
    """Compare parseable tooltip claims with every registered app default.

    Most comparisons are exact.  The 52 variants are compared with an
    explicit app/setting/value contract above, including a reasoned
    classification, so a same-size substitution cannot hide behind a digest.
    """
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import resolve_default_settings

    comparisons = 0
    variants = {}
    for entry in APPS:
        app_key = entry[0]
        defaults = resolve_default_settings(app_key)
        for key, actual in defaults.items():
            matches = list(DEFAULT_LITERAL.finditer(tooltips.get(key, "")))
            parsed = []
            for match in matches:
                raw = match.group("value")
                try:
                    value = (
                        "" if raw.lower() in {"empty", "blank"}
                        else ast.literal_eval(raw)
                    )
                except (SyntaxError, ValueError):
                    continue
                parsed.append((raw, value))
            if not parsed:
                continue
            comparisons += 1
            raw, claimed = parsed[-1]
            if not _same_default(actual, claimed):
                pair = (app_key, key)
                assert pair not in variants
                variants[pair] = (raw, repr(actual))

    expected = {
        pair: (variant.claimed, variant.actual_repr)
        for pair, variant in DEFAULT_VARIANT_EXPECTATIONS.items()
    }
    # 682 since 2026-09-02, and the two are both `experiment`. Its tooltip
    # used to read "Defaults vary by pipeline: 'exp', 'exp.' or
    # 'experiment_1'", which states no parseable default and so was not
    # compared at all. Instruction 337 made the abbreviations one word, so
    # the tooltip can now name a real default and IS compared -- in two
    # apps, against their actual value, and `variants` stayed at 52, which
    # is what says the new claim is true rather than merely new.
    # 670 since 2026-09-02. Instruction 326 removed the fixed floor of four
    # organelle slots, so a default measure or external-mask run no longer
    # carries organelleb/c/d keys and there are twelve fewer tooltip claims
    # to compare -- six settings across two apps. The count is the whole
    # point of the pin, so it moves with the change rather than being
    # widened to a range.
    assert comparisons == 670
    assert len(variants) == 47
    assert variants == expected
    assert {
        classification: sum(
            variant.classification == classification
            for variant in DEFAULT_VARIANT_EXPECTATIONS.values()
        )
        for classification in (ACCURATE_SHARED, REPAIRED_TOOLTIP, CONFIG_DEFECT)
    } == {
        ACCURATE_SHARED: 22,
        REPAIRED_TOOLTIP: 25,
        CONFIG_DEFECT: 0,
    }
    assert all(
        len(variant.reason.split()) >= 5
        for variant in DEFAULT_VARIANT_EXPECTATIONS.values()
    )


def test_repaired_tooltips_state_each_module_value_and_behavior():
    """Every class-B repair names both its live value and its consequence."""
    from spacr.qt.screens.settings_model import resolve_default_settings

    repaired = {
        pair for pair, variant in DEFAULT_VARIANT_EXPECTATIONS.items()
        if variant.classification == REPAIRED_TOOLTIP
    }
    # 25 since 2026-09-02: the six organelleb/c/d min_size entries went with
    # the fixed slot floor removed by instruction 326.
    assert len(REPAIRED_TOOLTIP_FACTS) == 25
    assert set(REPAIRED_TOOLTIP_FACTS) == repaired

    defaults_by_app = {}
    for (app_key, key), facts in REPAIRED_TOOLTIP_FACTS.items():
        if app_key not in defaults_by_app:
            defaults_by_app[app_key] = resolve_default_settings(app_key)
        defaults = defaults_by_app[app_key]
        expected = ast.literal_eval(
            DEFAULT_VARIANT_EXPECTATIONS[(app_key, key)].actual_repr
        )
        assert _same_default(defaults[key], expected), (app_key, key)

        text = tooltips[key]
        missing = [fact for fact in facts if fact not in text]
        assert not missing, f"{app_key}:{key} lost repaired facts: {missing}"


def test_inapplicable_real_defaults_always_explain_which_setting_gated_them():
    """Every inactive dependency encountered in a real app has a reason."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import resolve_default_settings
    from spacr.settings import get_setting_dependencies

    rules = get_setting_dependencies()
    witnessed = []
    failures = []
    for entry in APPS:
        app_key = entry[0]
        defaults = resolve_default_settings(app_key)
        for key, rule in rules.items():
            sources = tuple(rule.get("sources", ()))
            if key not in defaults or not any(s in defaults for s in sources):
                continue
            if rule["predicate"](defaults, {}):
                continue
            reason = str(rule["reason"](defaults, {}))
            witnessed.append((app_key, key))
            if not reason.strip() or not any(source in reason for source in sources):
                failures.append((app_key, key, reason))

    assert len(witnessed) == 49
    assert len({key for _app, key in witnessed}) == 35
    assert not failures


@pytest.mark.parametrize("key", sorted(VERIFIED_TOOLTIP_FACTS))
def test_every_former_thin_waiver_states_its_verified_contract(key):
    """The old exceptions now say the exact behavior found in source."""
    text = _all_tooltips()[key]
    missing = [fact for fact in VERIFIED_TOOLTIP_FACTS[key] if fact not in text]
    assert not missing, f"{key} lost verified tooltip facts: {missing}"


# ---------------------------------------------------------------------------
# coverage of the settings surface
# ---------------------------------------------------------------------------

def test_every_typed_setting_has_a_tooltip():
    """A setting the GUI can render must have text to render beside it."""
    tips = _all_tooltips()
    missing = sorted(k for k in expected_types if k not in tips)
    assert not missing, f"typed settings with no tooltip: {missing}"


@pytest.mark.parametrize("key", ["amsgrad", "cell_diameter", "loss_type"])
def test_representative_keys_are_substantive(key):
    """Spot-check the shapes the audit was commissioned to fix."""
    tips = _all_tooltips()
    if key not in tips:
        pytest.skip(f"{key} is not a declared setting")
    body = _body(tips[key])
    assert len(body.split()) >= 15, f"{key} tooltip is still thin: {body}"


def test_no_tooltip_tells_the_user_a_retired_setting_is_what_is_read():
    """A tooltip that names a setting spaCR no longer has is worse than none.

    `file_type` was split from `png_type`: it is a file FORMAT, and which
    object a crop is of moved to `path_string`. Its tooltip kept saying "in
    the GUI this one field writes both file_type and png_type, and only
    png_type is read downstream" -- an instruction about a setting that had
    been removed, shown on the control whose meaning had changed under it,
    so the retired name went on appearing in the UI in the one place a
    settings-key sweep does not look.

    Naming a retired key as HISTORY ("it was called png_type") is fine and
    `path_string` does exactly that. Naming it as the key that is READ is
    not.
    """
    retired_as_live = re.compile(
        r"(only|both)\s+png_type\b|writes?\s+both\s+file_type",
        re.IGNORECASE)
    offenders = [key for key, text in tooltips.items()
                 if isinstance(text, str) and retired_as_live.search(text)]
    assert not offenders, (
        "these tooltips tell the user a retired setting is the one that "
        f"counts: {sorted(offenders)}")
