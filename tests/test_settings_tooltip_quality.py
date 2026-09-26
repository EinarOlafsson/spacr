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
import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest

from spacr.settings import expected_types, tooltips

# Concrete facts read from the factories and consumers during the final audit.
# This mapping does not waive quality checks: every key still passes the same
# length and non-tautology rules as every other shipped tooltip.
VERIFIED_TOOLTIP_FACTS = {
    "backgrounds": ("Legacy compatibility", "cell_background", "does not alter"),
    # `normalize_plots` and `visualize` were here and are RETIRED (357-Q4,
    # 2026-09-09). Both waivers said the quiet part out loud -- "Legacy
    # compatibility", "do not read" -- which is the audit recording that a
    # control did nothing rather than the control being fixed. Retiring
    # them is the fix, and a waiver for a setting that no longer exists is
    # a fact about a tooltip nobody can read.
    "organelle_chann_dim": ("organelle_channel", "organelle_mask_dim", "Default None"),
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
    # organelle_min_area names its own per-app split in prose: "Default 10 in
    # Mask; Measure and External Masks start at 0 because they consume
    # existing labels rather than segmenting new ones." The parser reads the
    # LAST parseable "Default <value>" claim, which is the Mask figure, so the
    # two modules the sentence explicitly excepts register as variants. The
    # tooltip is accurate for all three; this is the case instruction 364 chose
    # deliberately -- name the difference in the prose rather than record it
    # here as drift -- so both are ACCURATE_SHARED and neither is a defect.
    ("measure", "organelle_min_area"): DefaultVariant(
        "10", "0", ACCURATE_SHARED,
        "Measure consumes existing labels rather than segmenting new ones, so "
        "it applies no area floor; the tooltip says so in the same sentence "
        "that gives Mask its 10.",
    ),
    ("external_masks", "organelle_min_area"): DefaultVariant(
        "10", "0", ACCURATE_SHARED,
        "External Masks imports labels made elsewhere and must not delete "
        "objects its source chose to keep; the tooltip names this exception "
        "alongside Measure.",
    ),
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
    ("classify_merged", "min_cells_per_well"): DefaultVariant(
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
    ("analyze_plaques", "channels"): DefaultVariant(
        "[0,1,2,3]", "[0, 0]", ACCURATE_SHARED,
        "The shared tooltip describes spaCR's list of kept channels, whose "
        "LENGTH fixes where masks land, and it is accurate for every module "
        "that uses that list. The Plaque assay does not: it passes the "
        "value straight to Cellpose, where the pair is cytoplasm channel "
        "then nucleus channel and [0,0] means one grayscale image with no "
        "nucleus channel. Same key, a different library's convention, and "
        "saying so in the shared prose is a change to a translated string "
        "-- deferred to the next catalog rebuild rather than shipped as "
        "nine locales of English.",
    ),
    ("external_masks", "channels"): DefaultVariant(
        "[0,1,2,3]", "[]", REPAIRED_TOOLTIP,
        "An empty importer list deliberately means all detected channels.",
    ),
        ("external_masks", "dst"): DefaultVariant(
        "''", "None", ACCURATE_SHARED,
        "Both falsey values select the module-computed destination.",
    ),
    ("external_masks", "normalize"): DefaultVariant(
        "True", "False", REPAIRED_TOOLTIP,
        "The importer inherits Measure's no-normalization initial value.",
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
    ("measure", "normalize"): DefaultVariant(
        "True", "False", REPAIRED_TOOLTIP,
        "Measure starts off and requires a percentile pair when enabled.",
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
    # ("regression", "control_wells") WAS HERE AND THE SPLIT REMOVED IT.
    # The old key served the invasion assay AND Regression, so its tooltip
    # ended "Default None." -- true for the assay, false for Regression,
    # which derives the list from filter_value and the control blocks. That
    # is the drift this entry recorded. `analysis_excluded_wells` states
    # Regression's own default and claims no other, so there is nothing left
    # to record: the variant is gone because the confusion behind it is.
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
    ("replication", "class_column"): DefaultVariant(
        "'test'", "'predictions'", ACCURATE_SHARED,
        "The shared tooltip names Replication's predictions column explicitly.",
    ),
    ("replication", "nuclei_limit"): DefaultVariant(
        "None", "10", ACCURATE_SHARED,
        "The shared tooltip names Replication's ten-nucleus cap explicitly.",
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
    # ("regression", "control_wells") was here. Its repaired tooltip existed
    # to explain a default that DISAGREED with the printed one, and the
    # split (357-Q6) removed the disagreement rather than the explanation:
    # `analysis_excluded_wells` still carries the same sentence about
    # deriving from filter_value, it simply no longer has a "Default None."
    # from the other meaning to contradict.
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
    # DERIVED FROM THE ORGANELLE CEILING, NOT PINNED TO A NUMBER. This was
    # `== 110` and broke the moment instruction 326 raised MAX_ORGANELLES from
    # 26 to 702: every slot contributes four unit-bearing settings, so the
    # count went to 2,814 and a bare integer could only ever be re-pinned by
    # hand after the fact. The invariant the test exists for -- that a name
    # encoding a physical unit explains that unit -- passed throughout; only
    # the census was stale.
    from spacr.organelle_types import MAX_ORGANELLES

    base_unit_settings = 7          # cell/nucleus/pathogen/seg_qc diameters,
                                    # bare `diameter`, spatial_neighbor_radius,
                                    # and cellpose_diameter -- the OPS
                                    # alignment's nucleus size, which entered
                                    # this census on 2026-09-06 when the OPS
                                    # settings gained tooltips. It is a
                                    # genuine seventh base setting rather than
                                    # a per-organelle one, and its tooltip
                                    # does say pixels, so only the census
                                    # moved.
    per_organelle_unit_settings = 4
    # +2 on 2026-09-25 (item 508): enhance_background_radius and
    # enhance_sharpen_radius, the Make Masks chain's two radii as Mask
    # settings. Both tooltips say pixels.
    enhancement_unit_settings = 2
    assert len(diameter_or_radius) == (
        base_unit_settings + enhancement_unit_settings
        + per_organelle_unit_settings * MAX_ORGANELLES
    )
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
    compared_pairs = set()
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
            compared_pairs.add((app_key, key))
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
    #
    # 666 since 2026-09-02. Instruction 364 retired organelle_min_size and
    # organelle_max_size in favour of the _area pair they duplicated, which
    # removes two tooltips and therefore four comparisons across two apps.
    # 673 since 2026-09-04. Instruction 326 raised MAX_ORGANELLES from 26 to
    # 702 and instruction 364 reshaped the organelle size/area settings; seven
    # more app/setting pairs now carry a parseable default claim. Verified as a
    # census change rather than new drift: the comparison SET is identical to
    # the one a worktree at e1e7fd42a^ produces, so nothing was added today.
    #
    # 680 since 2026-09-08, +7/-0, and all seven are the same app: the
    # Plaque assay now resolves `channels`, `grayscale`, `invert`,
    # `model_name`, `normalize`, `percentiles` and `remove_background`,
    # each of whose tooltips already carried a parseable "Default ..."
    # claim. Diffed against the census a worktree at the previous pin
    # produces rather than inferred from the total: nothing left the set,
    # so no claim stopped being compared while a new one arrived to hide
    # it -- which is the substitution a bare count cannot see.
    #
    # 679 since 2026-09-09, -1/+0, and the one is `gradient_accumulation` in
    # Classify. Instruction 364 retired it into `gradient_accumulation_steps`
    # -- `steps = 1` already says "do not accumulate" -- and 9d31a984f dropped
    # the three `setdefault` calls that put it in a resolved panel, so no app
    # resolves it and nothing compares its claim. Its `expected_types` row
    # went with it today: leaving it behind declared the key live and
    # withdrawn at once, which is what `test_no_retired_name_is_also_a_live_
    # setting` refuses. The pair that left is NAMED rather than inferred from
    # the total, so a claim that quietly stopped being compared could not
    # hide behind a new one arriving.
    # 679 -> 671 on 2026-09-09, -8/-0, and the eight are named because a
    # census that moves by a number nobody can list is a census nobody can
    # check: `denoise`, `load_path_regex`, `mask_array`, `normalization`,
    # `normalization_scope`, `normalize_plots`, `save_to_db` and
    # `visualize`, retired under 357-Q4 because the package read none of
    # them. A retired setting is resolved by no app, so its tooltip claim
    # is compared against nothing.
    # 671 -> 675 on 2026-09-10, +4/-0, and the four are TWO settings seen
    # by TWO apps. Instruction 388's `bystander_measurements` (Default
    # False) and `bystander_reach_in_diameters` (Default 1.0) are declared
    # once, in `get_measure_crop_settings`, and that factory is resolved by
    # both `measure` AND `external_masks` -- so a new measure setting moves
    # this census by two per app, not by one per setting. Anyone adding the
    # next one and expecting +1 will look for a bug that is not there.
    #
    # 675 -> 663 on 2026-09-12, -16/+4, and the arithmetic is worth writing
    # down because "-20 settings" does not give -12. Instruction 391 removed
    # five relative settings at four object roles -- twenty keys -- but only
    # SIXTEEN of them were resolved by an app and carried a parseable
    # "Default X." claim, so only sixteen were ever in this census. The four
    # gained are the one absolute setting that replaces them,
    # `<role>_intensity_threshold`, whose "Default None." parses to None.
    # 663 -> 666 on 2026-09-12, +3/-0 and all three from one finding:
    # 364 declared `remove_background_organelle`, `organelle_background`
    # and `organelle_signal_to_noise`, which io.py had been reading
    # through `.get` fallbacks with nothing declaring them. Each carries
    # a parseable "Default X." so each joins this census. The VALUES are
    # the fallbacks they replace (False / 100 / 10), so no run changes.
    # 666 -> 667 on 2026-09-15, +1/-0, and the one is `segmentation_backend`
    # in Mask -- the setting that chooses Cellpose, DINOCell or SAMCell
    # (items 404 and 405). Its tooltip ends "Default 'cellpose'.", which
    # parses, so it joins this census.
    #
    # +1 AND NOT +2, which is worth saying because the `bystander_*` note
    # above warns that a new measure setting moves this census by TWO. The
    # difference is how many apps resolve the factory: `get_measure_crop_
    # settings` is resolved by both `measure` and `external_masks`, while
    # this one is declared where only `mask` reaches it. The census counts
    # APP/SETTING PAIRS, so the multiplier is the app count, not the
    # setting count.
    #
    # DIFFED, NOT INFERRED, in a worktree at d9eef11d3 (the commit that
    # pinned 666): the arriving set is exactly {mask|segmentation_backend}
    # and the leaving set is EMPTY. That second half is the one that
    # matters -- a claim that quietly stopped being compared, replaced by a
    # new one arriving, moves this number by zero.
    # Feature 418 removes the five split/boundary-merge controls for four
    # roles and adds min/max object-mean bounds for those same roles. Only
    # Mask resolves this factory in APPS: 667 - 20 + 8 = 655 comparisons.
    # Keep the exact census and pin the arriving pairs so another claim
    # cannot disappear unnoticed behind an unrelated new one.
    # 655 -> 673 on 2026-09-21, +19/-1, DIFFED against c0b2c5227 (the commit
    # that pinned 655) rather than inferred. Arriving: nineteen Plaque Assay
    # Figure-mode settings from item 468, each with a parseable "Default X."
    # -- plaque_mode, figure_detector, figure_imgsz, figure_confidence,
    # figure_read_text, confirm_annotations and the thirteen text_* reading
    # controls. Leaving: ("regression", "Toxoplasma"), retired by item 364,
    # so no app resolves it and its claim is compared against nothing.
    # Compared with the actual 0a4f5aa75 package: +43 pairs, none removed.
    # Plaque calibration +4, TTA +6, PSF +15, Host–Pathogen +4,
    # Mask image QC/metadata +6 and restored Replication settings +8.
    # 716 -> 735 on 2026-09-25, +19/-0 (item 508): the nineteen enhance_*
    # settings, resolved by Mask only, each ending in a parseable
    # "Default X.". Pinned by name in the 508 census file below.
    # 735 -> 741 on 2026-09-25, +6/-0 (item 503): the six legacy Cellpose 3
    # settings, each ending in a parseable "Default X.", declared where
    # only `mask` resolves them; pinned by name here.
    item_503 = {("mask", key) for key in (
        "cellpose3_add_nucleus_channel", "cellpose3_size_model",
        "cellpose3_resample", "cellpose3_augment",
        "cellpose3_percentile_low", "cellpose3_percentile_high")}
    assert item_503 <= compared_pairs
    # 741 -> 742 on 2026-09-25, item 511: object_filters states "Default {}."
    assert any(key == "object_filters" for _app, key in compared_pairs)
    # 742 -> 734 on 2026-09-25, -8/+0, item 511 again: the maintainer retired
    # Mask's per-object {obj}_min_area, _max_area, _min_intensity and
    # _max_intensity into object_filters rows, and eight of those twelve
    # tooltips stated a parseable default. None of the twelve may remain.
    from spacr.settings import RETIRED_OBJECT_BOUNDS
    assert not {key for _app, key in compared_pairs} & set(RETIRED_OBJECT_BOUNDS)
    # 734 -> 736 on 2026-09-26, +2/-0 (item 493): mask_parallel states
    # "Default False." and mask_gpu_indices "Default blank.", both resolved
    # by Mask only (Timelapse keeps them hidden and is not in APPS).
    assert {("mask", "mask_parallel"), ("mask", "mask_gpu_indices")} <= compared_pairs
    # 736 -> 744 on 2026-09-26, +8/-0 (item 541): confluency (False),
    # confluency_channel (None), confluency_window (15) and
    # confluency_qc_threshold (0.8), each resolved by Measure and by External
    # Masks, which measures with Measure's defaults. confluency_source says
    # "Default auto.", which is not a literal and so is not compared.
    item_541 = {(app, key) for app in ("measure", "external_masks")
                for key in ("confluency", "confluency_channel",
                            "confluency_window", "confluency_qc_threshold")}
    assert item_541 <= compared_pairs
    assert comparisons == 744
    census_508 = json.loads((Path(__file__).parent / 'data' / 'release_contracts' /
                             '508_default_claim_census_2026-09-25.json').read_text())
    assert census_508['comparisons_before'] == 716
    # + 2: item 493's mask_parallel and mask_gpu_indices, pinned above;
    # + 8: item 541's four confluency claims in two apps, pinned above.
    assert (census_508['comparisons_after'] + len(item_503) + 1 - 8 + 2
            + len(item_541) == comparisons)
    assert census_508['removed_pairs'] == []
    assert {tuple(pair) for pair in census_508['added_pairs']} <= compared_pairs
    assert len(census_508['added_pairs']) == 19
    census = json.loads((Path(__file__).parent / 'data' / 'release_contracts' /
                         '491_default_claim_census_2026-09-23.json').read_text())
    assert census['comparisons_after'] == census_508['comparisons_before']
    assert census['removed_pairs'] == []
    assert len(census['added_pairs']) == 43
    assert {tuple(pair) for pair in census['added_pairs']} <= compared_pairs
    assert ("regression", "Toxoplasma") not in compared_pairs
    for key in ("plaque_mode", "figure_detector", "figure_imgsz",
                "figure_confidence", "figure_read_text",
                "confirm_annotations", "text_ignore", "text_min_confidence",
                "text_order", "text_panel_reach", "text_reach_above",
                "text_reach_below", "text_reach_left", "text_reread",
                "text_reread_scale", "text_separator", "text_use_above",
                "text_use_below", "text_use_left"):
        assert ("analyze_plaques", key) in compared_pairs
    for role in ("cell", "nucleus", "pathogen", "organelle"):
        for suffix in ("min_intensity", "max_intensity"):
            # Item 511: only organelle keeps its own mean-bound settings.
            assert (("mask", f"{role}_{suffix}") in compared_pairs) == (
                role == "organelle")
        for suffix in ("minimum_area_to_split", "min_watershed_distance",
                       "intensity_threshold", "intensity_merge", "intensity_split"):
            assert ("mask", f"{role}_{suffix}") not in compared_pairs
    # 44 since 2026-09-02. Instruction 364 unified organelle's duplicated
    # size/area settings, and the surviving tooltip now NAMES its per-app
    # defaults ("Default 10 in Mask; Measure and External Masks start at 0")
    # instead of leaving the difference to be recorded here as drift. A
    # variant that the tooltip itself explains is not drift.
    # 47 -> 46 on 2026-09-09. `control_wells` was split (357-Q6) into
    # `stain_baseline_wells` and `analysis_excluded_wells`, and the variant
    # it carried went with it: one key documented "Default None." while
    # Regression derived a list from filter_value, so the drift was two
    # meanings sharing a tooltip rather than a wrong default. Each half now
    # states its own, and neither disagrees with itself.
    # 46 -> 45 on 2026-09-12. The entry that went was
    # ("mask", "nucleus_intensity_threshold_method"), whose whole reason for
    # existing -- "the parser sees the adjacent percentile value; the prose
    # says mean" -- was an artefact of a setting whose tooltip had to name a
    # SECOND setting to explain itself. Instruction 391 removed both, so the
    # ambiguity is gone rather than newly tolerated.
    assert len(variants) == 47
    assert variants == expected
    assert {
        classification: sum(
            variant.classification == classification
            for variant in DEFAULT_VARIANT_EXPECTATIONS.values()
        )
        for classification in (ACCURATE_SHARED, REPAIRED_TOOLTIP, CONFIG_DEFECT)
    } == {
        # 24 -> 23 on 2026-09-12, the same one variant as the count above:
        # ("mask", "nucleus_intensity_threshold_method") was classified
        # ACCURATE_SHARED because its tooltip was right and only the parser
        # was confused by the neighbouring percentile. Instruction 391
        # removed the setting and the neighbour both.
        ACCURATE_SHARED: 25,
        # 23 -> 22 on 2026-09-09, and it is the same one variant: the
        # `control_wells` split (357-Q6) took its repaired-tooltip entry
        # with it, because the tooltip it repaired documented two meanings
        # at once and there is now one tooltip per meaning.
        REPAIRED_TOOLTIP: 22,
        CONFIG_DEFECT: 0,
    }
    assert all(
        len(variant.reason.split()) >= 5
        for variant in DEFAULT_VARIANT_EXPECTATIONS.values()
    )


def test_repaired_tooltips_state_each_module_value_and_behavior():
    """Every class-B repair names both its live value and its consequence."""
    # APPS IS IMPORTED FOR ITS SIDE EFFECT, not for its value. Several
    # screens register their defaults lazily -- `investigate_hit` among
    # them -- so `resolve_default_settings` answers without `score_column`
    # in a process where nothing has pulled the registry in yet. Running
    # this file whole hid that behind whichever earlier test did the
    # pulling; running this test alone raised KeyError from the loop
    # below. An assertion that only holds when a neighbour ran first is
    # not an assertion about the package.
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import resolve_default_settings

    assert APPS, "the app registry is empty, so no defaults can resolve"

    repaired = {
        pair for pair, variant in DEFAULT_VARIANT_EXPECTATIONS.items()
        if variant.classification == REPAIRED_TOOLTIP
    }
    # 25 since 2026-09-02: the six organelleb/c/d min_size entries went with
    # the fixed slot floor removed by instruction 326.
    # 23 -> 22 on 2026-09-09, the `control_wells` split again (357-Q6).
    # Three counts move together for one cause, which is the point of
    # keeping all three: a repaired tooltip, the variant it explained, and
    # the fact it carried are one entry seen from three sides, and a change
    # that moved only one of them would be a change nobody had understood.
    assert len(REPAIRED_TOOLTIP_FACTS) == 22
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

    # 49 -> 48 on 2026-09-09. `load_path_regex` carried a dependency rule
    # -- "image_source is 'stream_images', which cuts crops from the merged
    # arrays instead" -- and the setting was retired with it (357-Q4). The
    # rule explained when a control did not apply; nothing read the control
    # in either case.
    assert len(witnessed) == 48
    # 35 -> 34 with it: `load_path_regex` was witnessed in exactly one app,
    # so the pair count and the distinct-key count fall by one together.
    assert len({key for _app, key in witnessed}) == 34
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
