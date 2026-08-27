"""Regression backend vocabulary and setting dependencies.

This dependency-light module contains the data needed to populate settings
panels without importing plotting, machine-learning, or GUI stacks.
:mod:`spacr.ml` re-exports its public names for API compatibility.
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
    # THE ONE NONPARAMETRIC FAMILY THAT ANSWERS IN THIS CURRENCY.
    #
    # Instruction 254 sorts seven methods by what they can honestly report.
    # `spline` belongs here: it fits OLS on a design whose COVARIATES carry
    # a spline basis while every guide column is left alone, so one
    # coefficient and one P value per guide survive and the volcano, the hit
    # list and the attribution all read it unchanged.
    #
    # `isotonic` IS NOT HERE, and that is the same rule applied honestly.
    # It needs an ORDERED SINGLE PREDICTOR and the guide design is unordered
    # categories, so offering it in this menu would be the "method in the
    # wrong category" that instruction says is worse than not offering it --
    # a user picking it would get a coefficient table nobody should read.
    # It is available against a covariate through
    # `spacr.nonparametric_fits.isotonic_fit`, which is where it is true.
    'spline',
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
    # `spline` fits OLS on a design whose COVARIATES carry a spline basis
    # and whose guide columns are untouched, so it reads what ols reads.
    'spline': ('cov_type', 'spline_knots', 'spline_degree'),
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
    # 'auto' means "cross-validate it", the same way `alpha`'s 'auto' does,
    # and it is what the panel posts. A number here would make the posted
    # default a request every other backend then had to refuse.
    'group_lasso_lambda': 'auto',
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


# ---------------------------------------------------------------------------
# WHO fits it (instruction 141). `regression_type` says WHAT is fitted.
# ---------------------------------------------------------------------------

#: The backend every existing result was produced with.
#:
#: A DEFAULT THAT CHANGES THE NUMBERS IS NOT A DEFAULT. Every results.csv,
#: every volcano and every hit list in this project came out of statsmodels
#: and sklearn, so that is what an unconfigured run keeps getting; a faster
#: backend is something a user opts into, per screen, with the cost stated.
DEFAULT_REGRESSION_BACKEND = 'statsmodels'

#: ``types`` value meaning "every family in :data:`REGRESSION_TYPES`".
ALL_REGRESSION_TYPES = '*'

#: What each backend is, what it can fit, and what it costs.
#:
#: PURE DATA, like everything else in this module -- the availability
#: question ("is the package here, is there a GPU, can it fit the chosen
#: type") is :mod:`spacr.regression_backends`, which is the only place that
#: touches the environment. Splitting them is what lets a settings panel read
#: this table without importing torch; see
#: ``tests/test_a_settings_panel_does_not_import_torch.py``.
#:
#: Keys:
#:
#: ``label``       what the combo entry reads. ALWAYS ends in ``(CPU)`` or
#:                 ``(GPU)`` so hardware requirements are visible before a
#:                 user makes a selection.
#: ``device``      ``'cpu'`` or ``'gpu'``. The greying rule reads this.
#: ``package``     the import name to probe for. ``None`` means "already a
#:                 hard dependency of spaCR", which is true of statsmodels
#:                 and, since :mod:`spacr.power_model`, of torch.
#: ``pip``         the command that would provide it, shown ON the greyed-out
#:                 entry rather than in a manual.
#: ``types``       which :data:`REGRESSION_TYPES` it can fit.
#: ``url``         its API documentation.
#: ``summary``     one sentence: what it is for.
#: ``cost``        the measured or stated speed claim. Never "may be faster".
#: ``differs``     ``None`` when the backend must return the SAME numbers as
#:                 statsmodels, otherwise the sentence saying what is
#:                 different about its answer. Unstated numerical differences
#:                 from statsmodels are treated as errors.
#: ``implemented`` whether spaCR routes anything through it TODAY. False
#:                 entries are greyed out as "not wired up yet" rather than
#:                 hidden, so the list is the plan and the plan is visible.
REGRESSION_BACKENDS = {
    'statsmodels': {
        'label': 'statsmodels (CPU)',
        'device': 'cpu',
        'package': None,
        'pip': None,
        'types': ALL_REGRESSION_TYPES,
        'url': 'https://www.statsmodels.org/stable/api.html',
        'summary': ("The default. Inference-first: coefficients, standard "
                    "errors, p-values and the diagnostics the QC suite "
                    "reads."),
        'cost': ("Generic dense linear algebra, which is why 'mixed' is slow "
                 "-- measured at 54x OLS on 40 genes and 67x on 80, the "
                 "ratio rising with screen size."),
        'differs': None,
        'implemented': True,
    },
    'torch': {
        'label': 'torch (GPU)',
        'device': 'gpu',
        # torch is already a hard dependency -- spacr.power_model fits with
        # it -- so this backend adds no package, only a device requirement.
        'package': 'torch',
        'pip': 'pip install torch',
        'types': ('mixed',),
        'url': 'https://docs.pytorch.org/docs/stable/index.html',
        'summary': ("The mixed model's profiled REML objective written out "
                    "and optimised on the GPU (spacr.mixed_gpu). Same model, "
                    "same estimates."),
        'cost': ("Measured on an RTX 3090: the dense Cholesky each iteration "
                 "spends its time in takes 204 ms on the CPU and 7.69 ms on "
                 "the GPU at q=1212, and the whole fit is 6-9x faster end to "
                 "end on screen-sized problems."),
        'differs': None,
        'implemented': True,
    },
    'pymer4': {
        'label': 'pymer4 / lme4 (CPU)',
        'device': 'cpu',
        'package': 'pymer4',
        # THE PIP LINE STAYS SHORT because `backend_status` puts it inside
        # a combo entry; what it does not say -- that the package alone is
        # not enough -- is in `cost` below and in the box, which is on screen
        # whether or not the popup is open.
        'pip': 'pip install pymer4  (plus R, rpy2, lme4)',
        'types': ('mixed',),
        'url': 'https://eshinjolly.com/pymer4/',
        'summary': ("The reference implementation for mixed models. Sparse "
                    "Cholesky over the nested structure rather than dense "
                    "algebra -- an algorithmic win, not a hardware one."),
        # IT STILL NEEDS R, and the earlier note here saying otherwise was
        # read off a metadata gap. pymer4 0.9.2's wheel declares NO
        # dependencies at all -- `importlib.metadata.requires('pymer4')`
        # returns None -- so `pip install --dry-run --report` truthfully said
        # "adds one package, changes nothing" and that was mistaken for "needs
        # no R". Installed and imported on 2026-08-18 it fails at
        # `pymer4/io.py: import polars`, and every model module under it opens
        # with `from rpy2.robjects.packages import importr`; its own README
        # says "This is accomplished using rpy2 to interface between
        # languages". So the maintainer's question -- can lme4 be had without
        # installing or interfacing with R -- is answered NO on this version.
        'cost': ("Needs R, rpy2 and lme4 installed alongside it: measured "
                 "2026-08-18, pymer4 0.9.2 imports rpy2 in every model "
                 "module and polars in its loader, and declares neither, so "
                 "pip reports it as a single additive package and it is not "
                 "one."),
        'differs': None,
        'implemented': False,
    },
    'cuml': {
        'label': 'cuML (GPU)',
        'device': 'gpu',
        'package': 'cuml',
        'pip': "pip install 'spacr[rapids]'",
        'types': ('lasso', 'ridge', 'elasticnet'),
        'url': 'https://docs.rapids.ai/api/cuml/stable/',
        'summary': ("RAPIDS' GPU ridge / lasso / elastic-net, near drop-in "
                    "for scikit-learn. Speeds the PENALISED families."),
        # IT IS NOT ADDITIVE, MEASURED 2026-08-18 and this is the one that
        # matters: `pip install --dry-run --report cuml-cu12` against this
        # environment moves NUMPY 1.26.4 -> 2.2.6, downgrades numba
        # 0.62.1 -> 0.61.2 and llvmlite 0.45.1 -> 0.44.0, and moves eight
        # nvidia-cu12 runtime libraries torch is built against. The
        # dependency table at the top of instruction 141 tested pymer4,
        # gpytorch, numpyro, glum, linearmodels and pyfixest -- cuML was
        # never among them, and "all six are purely additive" was read as
        # covering it. The maintainer's condition was "first test if adding
        # any of those dependencies causes any problems"; for this one the
        # answer is yes, so it is not installed here.
        'cost': ("It has NO mixed model, so it does not touch that "
                 "bottleneck. And it is the one optional backend that is NOT "
                 "a safe install here: resolving cuml-cu12 against this "
                 "environment moves numpy 1.26.4 -> 2.2.6 and the CUDA "
                 "runtime torch is built against, measured 2026-08-18."),
        'differs': ("A penalised path solved to a different tolerance can "
                    "select a different set of coefficients at the same "
                    "alpha."),
        'implemented': False,
    },
    'pyfixest': {
        'label': 'pyfixest (CPU)',
        'device': 'cpu',
        'package': 'pyfixest',
        'pip': "pip install 'spacr[pyfixest]'",
        'types': ('ols', 'wls'),
        'url': 'https://py-econometrics.github.io/pyfixest/',
        'summary': ("Absorbs the rowID and columnID fixed effects by "
                    "alternating projections instead of carrying them as "
                    "dummy columns, then solves the remaining normal "
                    "equations directly."),
        # MEASURED 2026-08-18 on synthetic screens of the shape
        # prepare_formula builds, against sm.OLS on the identical design.
        # See `spacr.ml._fit_absorbed_least_squares` for the table and for
        # why the win is the SOLVER rather than the 5% narrower design --
        # pyfixest's own `feols` was measured too and is slower here.
        'cost': ("Measured on this machine: 1.4x at n=1830/p=736 (the "
                 "TSG101 shape), 5.9x at n=6000/p=2242 and 16.7x at "
                 "n=12000/p=4601, the ratio rising with screen size. "
                 "Coefficients agree with statsmodels to 3.9e-9 absolute "
                 "and standard errors to 1.3e-13 relative."),
        'differs': ("The absorbed terms have no coefficient row, so the "
                    "Intercept is missing from the table. rowID and columnID "
                    "were dropped from it anyway; the Intercept is a "
                    "nuisance term the volcano is better without. Every "
                    "gRNA and gene coefficient is reported and unchanged."),
        'implemented': True,
    },
    'glum': {
        'label': 'glum (CPU)',
        'device': 'cpu',
        'package': 'glum',
        'pip': "pip install 'spacr[glum]'",
        # NOT probit AND NOT quasi_binomial, measured rather than dropped:
        # glum 3.4 ships identity, log, logit, cloglog and Tweedie links and
        # has NO probit, and it has no equivalent of statsmodels'
        # scale='X2', which is the free dispersion that IS quasi-binomial.
        # See `spacr.ml._GLUM_FAMILIES`.
        'types': ('glm', 'poisson', 'logit'),
        'url': 'https://glum.readthedocs.io/',
        'summary': ("Fast GLMs by IRLS over tabmat's column-typed design, "
                    "with the Poisson, binomial and Gaussian families "
                    "statsmodels offers."),
        'cost': ("Measured on this machine: 2.4x on poisson and 3.1x on "
                 "logit at n=6000/p=2242, and SLOWER than statsmodels on a "
                 "small screen -- 0.70x on poisson at n=1830/p=736. Its "
                 "setup is fixed and repays itself only once the design is "
                 "wide."),
        'differs': None,
        'implemented': True,
    },
    'numpyro': {
        'label': 'numpyro (GPU)',
        'device': 'gpu',
        'package': 'numpyro',
        'pip': 'pip install numpyro',
        'types': ('mixed', 'horseshoe'),
        'url': 'https://num.pyro.ai/',
        'summary': "Bayesian, NUTS on the GPU.",
        'cost': "Sampling, so slower per fit and parallel across chains.",
        'differs': ("Gives POSTERIORS rather than point estimates plus "
                    "standard errors -- a different answer, not a faster "
                    "version of the same one."),
        'implemented': False,
    },
    'gpytorch': {
        'label': 'gpytorch (GPU)',
        'device': 'gpu',
        'package': 'gpytorch',
        'pip': 'pip install gpytorch',
        'types': ('mixed',),
        'url': 'https://docs.gpytorch.ai/',
        'summary': ("A linear mixed model IS a Gaussian process with a "
                    "linear kernel plus one kernel per nesting level."),
        'cost': "GPU kernel algebra, and scales with the number of wells.",
        'differs': None,
        'implemented': False,
    },
}

#: Backend display order: the default first, the additional implemented
#: backend next, followed by unavailable or not-yet-integrated choices.
REGRESSION_BACKEND_ORDER = (
    'statsmodels', 'torch', 'pymer4', 'cuml', 'pyfixest', 'glum', 'numpyro',
    'gpytorch',
)

#: ``label -> canonical name``, so a panel may post either. Built from the
#: table rather than written twice, because a second copy is the one that
#: drifts.
REGRESSION_BACKEND_LABELS = {
    spec['label']: name for name, spec in REGRESSION_BACKENDS.items()
}
