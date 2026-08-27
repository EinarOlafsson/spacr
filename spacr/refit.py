"""Re-fit a screen from its plot with a different statistical model.

The regression plot's context menu can rerun an analysis with a different
model, correction method, or FDR threshold.

A re-fit recalculates results rather than changing plot appearance. The menu
therefore presents it separately from styling actions, and
:func:`spacr.ml._next_results_folder` writes the result beside the original
run without overwriting it.

The saved settings require three adjustments before reuse:

  * ``plot`` is forced to False on the way out (:func:`spacr.utils.save_settings`
    does it so a reload reproduces the run headlessly), so re-running from the
    saved copy produces no figures at all, even though the action starts from
    a figure.
  * the per-backend knobs are still set to what the old backend read. Handing
    ``alpha=0.3`` to OLS does not quietly do nothing: :func:`_reject_unused_settings`
    raises, by design, so a lasso -> ols re-fit dies at the entry point.
  * ``regression_type`` and ``random_row_column_effects`` can contradict each
    other, and the reconciliation between them refuses rather than guesses.

:func:`refit_settings` applies these adjustments and reports each change,
including reset backend-specific parameters.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

#: What a re-fit is allowed to change. Deliberately short: this is "the same
#: screen through a different model", not a second settings panel. Anything
#: else -- the data, the formula, the filters -- would make the two runs
#: incomparable, which is the one thing a side-by-side re-fit is for.
REFITTABLE = ("regression_type", "multiple_testing_method", "fdr_alpha",
              "alpha", "cov_type", "quantile", "huber_t", "l1_ratio",
              "hinge_threshold", "random_row_column_effects")

#: The setting that names the correction. SPELLED THE WAY THE RUN READS IT --
#: :func:`spacr.ml.perform_regression` looks up ``multiple_testing_method``
#: and nothing anywhere reads ``correction_method``, so a re-fit that wrote
#: the second spelling would run with the OLD correction and label its output
#: with the new one. Named here so there is one place to be wrong.
CORRECTION_KEY = "multiple_testing_method"

#: The significance level the correction is applied at. NOT ``alpha``, which
#: is the penalty weight of a penalised fit -- two different numbers, one
#: letter apart, and swapping them silently changes either the hit list or
#: the model.
CORRECTION_ALPHA_KEY = "fdr_alpha"

#: What the run uses when the settings name no level -- `spacr.settings`
#: setdefaults it to this. Restated here only so an absent setting and an
#: explicit 0.05 are not reported as a change from one to the other.
DEFAULT_FDR_ALPHA = 0.05

#: Keys that name where the LAST run wrote. A re-fit must not inherit them:
#: `src` is the run's own output root, and carrying it over would nest the
#: new results inside the old ones. It is rebuilt from the count data, which
#: is the rule the results folder already follows.
RESOLVED_OUTPUT_KEYS = ("results_path", "res_folder", "volcano_path")


def policed_settings() -> Dict[str, object]:
    """``{setting: the value that means "not asked for"}``.

    Values are read from :mod:`spacr.regression_spec`, the same source used by
    the regression implementation, so reset behavior tracks backend settings.
    """
    from .regression_spec import _MODEL_LEVEL_DEFAULTS, _RUN_LEVEL_DEFAULTS

    merged = dict(_MODEL_LEVEL_DEFAULTS)
    merged.update(_RUN_LEVEL_DEFAULTS)
    return merged


def prune_for_type(settings: dict, regression_type) -> Tuple[dict, List[str]]:
    """Reset every knob ``regression_type`` cannot read, and name them.

    :param settings: settings mapping; not mutated.
    :param regression_type: the backend about to be fitted. ``None`` means
        "choose from the data", which reads none of the policed knobs -- so
        it prunes as strictly as a named type, for the reason
        :func:`spacr.ml._reject_unused_run_settings` gives.
    :returns: ``(settings, [what was reset])``.

    Inapplicable settings are reset rather than deleted so the settings panel
    retains explicit default values.
    """
    from .regression_spec import REGRESSION_SETTINGS_USED

    used = REGRESSION_SETTINGS_USED.get(regression_type, ())
    out = dict(settings)
    reset: List[str] = []
    for name, default in policed_settings().items():
        if name in used or name not in out:
            continue
        value = out[name]
        # The same 'auto'/None spelling regression_model treats as "no
        # penalty chosen": not a request, so not something to report having
        # dropped.
        if name == "alpha" and (value is None or value == "auto"):
            out[name] = default
            continue
        if value == default:
            continue
        out[name] = default
        reset.append(f"{name}={value!r}")
    return out, reset


def refit_settings(base: dict, *, regression_type=None,
                   correction_method: Optional[str] = None,
                   fdr_alpha: Optional[float] = None,
                   level: Optional[str] = None,
                   alpha=None) -> Tuple[dict, List[str]]:
    """Return settings for re-running a screen with a new model.

    :param base: the settings the run on screen used.
    :param regression_type: the new backend, or ``None`` to keep the old one.
    :param correction_method: the new multiple-testing correction, or ``None``
        to keep the old one. Written to :data:`CORRECTION_KEY`.
    :param fdr_alpha: the new significance level, or ``None`` to keep it.
    :param level: ``'grna'``, ``'gene'`` or ``'both'``, or ``None`` to keep
        the previous value. Mixed models treat guides as random effects and
        therefore provide shrunken predictions without per-guide p-values;
        a guide-level fixed-effect model provides coefficients and p-values.
    :param alpha: the new penalty weight, where the new backend reads one --
        a different number from ``fdr_alpha`` despite the name.
    :returns: ``(settings, [notes for the user])``.
    :raises ValueError: if ``base`` names no count data.

    Returned notes identify every automatic settings change before execution.
    """
    if not base:
        raise ValueError(
            "There are no settings to re-fit from. The panel knows which "
            "table it is showing but not which settings produced it, which "
            "happens when results were opened from disk and the run's "
            "settings CSV is not beside them.")
    if not base.get("count_data"):
        raise ValueError(
            "These settings name no count data, so there is nothing to "
            "re-fit: a regression needs the counts, not just the "
            "coefficients it produced.")

    settings = dict(base)
    notes: List[str] = []

    for key in RESOLVED_OUTPUT_KEYS:
        settings.pop(key, None)

    if level is not None:
        from .settings import REGRESSION_LEVELS

        level = str(level).strip().lower()
        if level not in REGRESSION_LEVELS:
            raise ValueError(
                f"level={level!r} must be one of {list(REGRESSION_LEVELS)}.")
        old_level = str(settings.get("level", "both")).strip().lower()
        if level != old_level:
            settings["level"] = level
            notes.append(f"level {old_level!r} -> {level!r}")

    old_type = settings.get("regression_type")
    if regression_type is not None and regression_type != old_type:
        settings["regression_type"] = regression_type
        notes.append(f"model {old_type!r} -> {regression_type!r}")
    if correction_method is not None:
        from .multiple_testing import canonical_method

        # Canonicalised here rather than passed through: the run raises on an
        # unknown spelling, and it should do that while the dialog is open
        # rather than twenty minutes into a fit.
        correction_method = canonical_method(correction_method)
        old = settings.get(CORRECTION_KEY)
        if correction_method != old:
            settings[CORRECTION_KEY] = correction_method
            notes.append(f"correction {old!r} -> {correction_method!r}")
    if fdr_alpha is not None:
        # Compared against the run's own default when the settings did not
        # record one, so a dialog whose spin box always holds a number does
        # not report "significance level None -> 0.05" as a change on every
        # single re-fit. A note that fires every time is a note nobody reads,
        # and the notes that matter are in the same sentence.
        old = settings.get(CORRECTION_ALPHA_KEY, DEFAULT_FDR_ALPHA)
        settings[CORRECTION_ALPHA_KEY] = fdr_alpha
        if fdr_alpha != old:
            notes.append(f"significance level {old!r} -> {fdr_alpha!r}")

    chosen = settings.get("regression_type")
    if alpha is not None:
        settings["alpha"] = alpha

    # RANDOM EFFECTS WIN OVER A NAMED MODEL, and the run refuses the
    # combination rather than choosing. Turning the flag off when the user
    # has just asked for a specific backend is what they meant by asking --
    # leaving it on would fit a MixedLM and file it under the name they
    # picked, which is the exact bug _reconcile_random_row_column_effects
    # was written for.
    if settings.get("random_row_column_effects") and chosen not in (
            None, "mixed"):
        settings["random_row_column_effects"] = False
        notes.append("random row/column effects off (they fit a mixed model, "
                     f"and {chosen!r} was asked for)")

    # A GUIDE-LEVEL MIXED FIT STILL HAS NO GUIDE P VALUES, and saying so here
    # is the difference between a re-fit that answers the question and one
    # that returns the same empty volcano under a new folder name. `mixed`
    # makes the guide a RANDOM effect at every level, so asking for
    # level='grna' does not turn its BLUPs into estimates.
    if (str(settings.get("level", "")).lower() == "grna"
            and str(chosen or "").lower() == "mixed"):
        notes.append(
            "level='grna' with regression_type='mixed' still gives BLUPs "
            "and no per-guide p value, because a mixed model makes the "
            "guide a random effect. For per-guide significance choose a "
            "fixed-effect model such as 'ols', 'rlm' or 'quantile', or use "
            "inference='nonparametric'.")

    settings, reset = prune_for_type(settings, chosen)
    if reset:
        notes.append(f"{chosen!r} does not read " + ", ".join(reset)
                     + " — reset to default")

    # THE FIGURES COME BACK ON. save_settings writes plot=False so a reload
    # reproduces the run headlessly; a re-fit asked for FROM a figure that
    # then drew no figures would look like it had failed.
    if settings.get("plot") is False:
        settings["plot"] = True
    settings["test_mode"] = False

    # `src` is rebuilt by the run from the count data unless it is set, and
    # the previous run left it pointing at its own output root. Dropping it
    # is what puts the re-fit in a sibling folder rather than nested inside
    # the run it is being compared with.
    settings.pop("src", None)
    return settings, notes


def destination(settings: dict) -> Optional[str]:
    """Return the output directory a run would use without executing it.

    The result uses the same folder-selection rule as the regression run.
    """
    from .ml import _next_results_folder

    count = settings.get("count_data")
    if isinstance(count, (list, tuple)):
        count = count[0] if count else None
    if not count:
        return None
    src = settings.get("src") or os.path.dirname(str(count))
    kind = ("guide_permutation"
            if settings.get("analysis_mode") == "guide_permutation"
            else settings.get("regression_type") or "auto")
    try:
        return _next_results_folder(os.path.join(src, "results"), str(kind))
    except OSError:
        return None


#: Where a run leaves the settings it actually used, relative to the folder
#: holding its results table. `save_settings` writes under the run's `src`,
#: which is the results ROOT rather than the run's own folder -- so a re-fit
#: started from an old table has to look upwards, and may find a settings
#: file from a LATER run. Named here so the search order is one decision in
#: one place.
SETTINGS_NAMES = ("regression_settings.csv", "regression.csv", "settings.csv")


def settings_of_run(results_path) -> Optional[dict]:
    """The settings a finished run used, read back from beside its results.

    :param results_path: the results CSV, or the folder holding it.
    :returns: the parsed settings, or ``None`` when the run left none.

    Search proceeds from the run's own folder to ``settings/`` beside the
    results root. This prioritizes run-specific settings over the shared copy,
    which later runs may overwrite.
    """
    from .utils import load_settings

    if results_path is None:
        return None
    folder = str(results_path)
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)

    roots: Sequence[str] = (folder, os.path.dirname(folder),
                            os.path.dirname(os.path.dirname(folder)))
    for root in roots:
        if not root:
            continue
        for name in SETTINGS_NAMES:
            for candidate in (os.path.join(root, name),
                              os.path.join(root, "settings", name)):
                if not os.path.isfile(candidate):
                    continue
                try:
                    return load_settings(candidate)
                except Exception:                          # noqa: BLE001
                    continue
    return None


__all__ = ["CORRECTION_ALPHA_KEY", "CORRECTION_KEY", "DEFAULT_FDR_ALPHA",
           "REFITTABLE", "RESOLVED_OUTPUT_KEYS", "SETTINGS_NAMES",
           "destination", "policed_settings", "prune_for_type",
           "refit_settings", "settings_of_run"]
