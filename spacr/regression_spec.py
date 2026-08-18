"""What the regression backends are, and which settings each one reads.

PURE DATA, AND THIS MODULE IMPORTS NOTHING. That is the point of it, not a
coincidence.

These tables used to live in :mod:`spacr.ml`, which imports
:mod:`spacr.plot`, which imports torch, cv2 and IPython. Nothing in the GUI
wanted any of that -- but `get_setting_dependencies` reads
:data:`REGRESSION_SETTINGS_USED` to decide which widgets on a settings panel
apply to each other, so BUILDING ANY SETTINGS PANEL paid 2.2 seconds and
900 MB to look up a dict of strings. There is a test asserting the panel does
not import the plotting stack (`tests/qt/test_gui_responsiveness.py`); it was
written when that import cost 770 ms and was "the whole remaining cost of
opening the first module", and torch made it four times worse.

The same move `convert_settings_dict_for_gui` already made into
`settings_spec.py`, for the same reason.

:mod:`spacr.ml` re-exports every name here, so ``from spacr.ml import
REGRESSION_TYPES`` keeps working and there is still one source of truth --
this file.
"""

from __future__ import annotations

#: Every regression backend :func:`regression_model` can fit, in the order the
#: settings panels list them.
#:
#: This tuple is the SINGLE source of truth for "which model may I ask for".
#: :func:`perform_regression` used to carry its own hand-written whitelist, and
#: the two drifted in both directions at once, silently:
#:
#: * ``'beta'`` and ``'quasi_binomial'`` were fittable by
#:   :func:`regression_model` and returned by :func:`check_distribution` (so
#:   ``regression_type=None`` auto-selected them), yet the entry point refused
#:   them outright — a user could never choose the model spaCR itself picked;
#: * ``'quantile'`` was accepted by the entry point, given its own
#:   ``agg_type`` handling in :func:`spacr.settings.get_perform_regression_default_settings`
#:   and its own volcano-filename rule, and then died at the very last step
#:   with "Unsupported regression type quantile" — after both input CSVs had
#:   been read, the QC plots drawn and ``regression_data.csv`` written;
#: * ``'gls'``, ``'wls'`` and ``'rlm'`` were advertised by the entry point and
#:   by the Tk combo box and had no backend at all.
#:
#: Anything added here must be fittable by :func:`regression_model` AND have a
#: coefficient branch in :func:`process_model_coefficients`; the round trip is
#: pinned by ``tests/test_regression_types.py``.
REGRESSION_TYPES = (
    'ols',
    'wls',
    'rlm',
    'huber',
    'glm',
    'poisson',
    'quasi_binomial',
    'beta',
    'logit',
    'probit',
    'quantile',
    'mixed',
    'lasso',
    'ridge',
    'elasticnet',
    'hinge',
    'horseshoe',
    'group_lasso',
    'rra',
)

#: Names spaCR advertised but has never been able to fit, mapped to the
#: sentence that says what to use instead. Kept by name rather than deleted so
#: an old settings CSV is answered with a migration instruction instead of a
#: bare "unsupported".
UNSUPPORTED_REGRESSION_TYPES = {
    'gls': (
        "GLS needs an error covariance structure spaCR does not estimate; "
        "with the default sigma it is arithmetically identical to 'ols', so "
        "offering it only invited a user to believe they had corrected for "
        "something. Use 'ols' with cov_type='HC3' for heteroscedasticity-"
        "robust standard errors, 'wls' to weight wells by cell count, or "
        "'mixed' for plate/row/column random effects."
    ),
}

#: Which optional model setting each regression type actually READS.
#:
#: A key absent from a type's tuple is not applied by that backend, and
#: :func:`regression_model` refuses it rather than ignoring it: a silently
#: ignored setting is this pipeline's most expensive failure mode, because the
#: run completes and the number looks right. ``cov_type='HC3'`` with
#: ``'lasso'`` is the archetype — sklearn has no covariance estimator at all,
#: so the run would report ordinary p-values under a robust-sounding label.
#:
#: ``alpha`` is only listed for the backends that penalise; ``groups``,
#: ``weights`` and ``exposure`` are supplied by :func:`regression` from the
#: data, not by the user, so they are not policed here.
#:
#: ``random_row_column_effects`` is not in this table because it is not a knob
#: on a backend: it REPLACES the backend with a mixed model. That collision is
#: settled by :func:`_reconcile_random_row_column_effects` before any fitting
#: starts, which then uses this same table to police the knobs the mixed fit
#: cannot read.
#:
#: What each type reads that is NOT a policed setting, so this table is not
#: mistaken for the whole story:
#:
#: * ``wls`` and the three binomial links (``logit``, ``probit``,
#:   ``quasi_binomial``) read the per-well ``weights`` — the cell count. For
#:   the binomial links it is ``var_weights``, which is what tells the variance
#:   function that a fraction measured from 400 cells is firmer evidence than
#:   one from 30. ``glm`` reads it too when it auto-selects a binomial family.
#: * ``poisson``, ``horseshoe``, and ``glm`` when it picks Poisson, read
#:   ``exposure`` as ``offset(log(cell_count))`` — which is what makes their
#:   coefficients effects on a per-cell RATE instead of on the well's headcount.
#: * ``mixed`` reads ``groups``, the plate. :func:`_mixed_model_groups` says
#:   why it is the plate and not the well.
#: * ``hinge`` reads no kernel setting and is not going to get one: a
#:   non-linear kernel has no ``coef_``, so there would be no per-gRNA
#:   coefficient for :func:`process_model_coefficients` to table, no volcano
#:   plot and no hit list. Everything downstream of the fit is built on one
#:   linear coefficient per feature.
#: * ``group_lasso`` and ``rra`` both read the GENE OF EACH DESIGN COLUMN,
#:   which is parsed from the column name rather than set on a panel: the
#:   guides of one gene are the block ``group_lasso`` penalises as a unit and
#:   the ranks ``rra`` aggregates. Neither has a setting for it, because a
#:   grouping the user could mistype is a grouping that would silently split
#:   a gene in two.
#: * ``group_lasso`` does NOT read ``alpha``. Its penalty is
#:   ``group_lasso_lambda``, kept as its own key precisely so it cannot be
#:   confused with the per-coefficient penalty of ``lasso``/``ridge``: the
#:   two are on different scales (``group_lasso_lambda`` is compared against
#:   ``spacr.group_lasso.max_lambda``, which is a property of THIS design),
#:   so one number carried across from a lasso run would mean something else
#:   here.
REGRESSION_SETTINGS_USED = {
    'ols': ('cov_type',),
    'wls': ('cov_type',),
    'rlm': ('huber_t',),
    'huber': ('huber_t',),
    'glm': ('cov_type',),
    'poisson': ('cov_type',),
    'quasi_binomial': ('cov_type',),
    'beta': (),
    'logit': ('cov_type',),
    'probit': ('cov_type',),
    'quantile': ('quantile',),
    'mixed': (),
    'lasso': ('alpha', 'lasso_n_boot', 'lasso_selection_threshold'),
    'ridge': ('alpha',),
    'elasticnet': ('alpha', 'l1_ratio', 'lasso_n_boot',
                   'lasso_selection_threshold'),
    'hinge': ('alpha', 'hinge_threshold', 'hinge_n_boot'),
    'horseshoe': (),
    'group_lasso': ('group_lasso_lambda', 'lasso_n_boot',
                    'lasso_selection_threshold'),
    'rra': ('rra_alpha', 'rra_permutations'),
}

#: The subset of :data:`REGRESSION_SETTINGS_USED` that :func:`regression_model`
#: never sees, because they configure what :func:`perform_regression` does with
#: the coefficients AFTER the fit rather than the fit itself. They are policed
#: at the entry point instead, by :func:`_reject_unused_run_settings`, so a
#: number set on the panel and read by nothing still fails loudly.
RUN_LEVEL_SETTINGS = ('lasso_n_boot', 'lasso_selection_threshold',
                      'hinge_n_boot')

#: The nine knobs that reach the estimator, and the value of each that means
#: "not asked for". Comparing against these is what lets a GUI post every
#: widget on the panel without every widget counting as a request.
#:
#: A MODULE CONSTANT because two things now depend on it and they must not
#: drift apart: :func:`regression_model` REFUSES a knob the chosen backend
#: cannot read, and :mod:`spacr.refit` RESETS those knobs when the user
#: switches backend from the plot. If the reset table were a second copy,
#: re-fitting lasso -> ols would carry ``alpha`` across and raise the very
#: error the reset exists to prevent.
_MODEL_LEVEL_DEFAULTS = {
    # 'auto' and None mean "no penalty chosen, cross-validate it", which is
    # not a value an unpenalised model is being asked to honour, so they
    # count as the default rather than as a request. Handled at the call
    # site, which is the only place that knows `alpha` was spelled that way.
    'alpha': 1.0,
    'l1_ratio': 0.5,
    'cov_type': None,
    'quantile': 0.5,
    'hinge_threshold': None,
    'huber_t': 1.345,
    # Instruction 133's two new backends. Their defaults are the ones
    # `spacr.group_lasso` and `spacr.rra` document for themselves, so a panel
    # that posts the untouched widget posts the value the module would have
    # used anyway and no other backend is refused because of it.
    'group_lasso_lambda': 0.05,
    'rra_alpha': 0.25,
    'rra_permutations': 10000,
}

#: Backends that report a coefficient but no frequentist p-value, so
#: ``p_value <= 0.05`` is not a hit rule for them. :func:`perform_regression`
#: ranks these by bootstrap selection frequency instead.
#:
#: ``group_lasso`` belongs here for exactly the reason ``lasso`` does: the
#: sampling distribution of a penalised coefficient is not the OLS one, so the
#: p-value :func:`process_model_coefficients` attaches to it is mis-specified
#: and only safe to read as "too large". ``spacr.group_lasso`` says the same
#: thing in its own words and offers ``stability_selection`` instead.
#:
#: ``rra`` DOES NOT belong here and putting it here would change its answer.
#: Its P value is a permutation P value -- ``spacr.rra`` draws a null of
#: ``rho`` per distinct guide count and reports ``(1 + #{null <= rho}) /
#: (n + 1)`` -- so it is a real test with a real null, and BH over the fit's
#: gene calls is the correct correction. Routed through the selection-
#: frequency branch it would instead be ranked by a bootstrap it never ran,
#: and ``_call_level_hits`` would refuse the run outright for want of one.
NO_P_VALUE_TYPES = ('lasso', 'elasticnet', 'group_lasso')


#: Defaults for the run-level settings, matched to
#: ``get_perform_regression_default_settings``. A value equal to its default is
#: "the panel posted it", not "the user asked for it" — the same rule
#: :func:`_reject_unused_settings` uses for the model knobs.
_RUN_LEVEL_DEFAULTS = {
    'lasso_n_boot': 200,
    'lasso_selection_threshold': 0.6,
    'hinge_n_boot': 200,
}
