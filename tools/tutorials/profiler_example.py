"""Reproducible synthetic OLS teaching example, not a biological screen.

The native profiler imports only the saved coefficient CSV. It does not fit
this model or receive the design through a private application method.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(destination):
    """Fit a real OLS model to explicitly synthetic, bounded teaching rows."""
    import statsmodels.api as sm

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    a, b = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    data = pd.DataFrame({'input_a': a.ravel(), 'input_b': b.ravel()})
    data['response'] = (2 + 4 * data.input_a - 1.5 * data.input_b
                        + np.random.default_rng(53).normal(0, .03, len(data)))
    design = pd.DataFrame({'Intercept': np.ones(len(data)),
                           'input_a': data.input_a, 'input_b': data.input_b})
    fit = sm.OLS(data.response, design).fit()
    independent, _, rank, _ = np.linalg.lstsq(design.to_numpy(), data.response.to_numpy(), rcond=None)
    if rank != 3 or not np.allclose(fit.params, independent, rtol=0, atol=1e-12):
        raise ValueError('Independent least-squares coefficients disagree')
    data.to_csv(destination / 'SYNTHETIC_observations.csv', index=False)
    pd.DataFrame({'feature': fit.params.index, 'coefficient': fit.params.values}).to_csv(
        destination / 'SYNTHETIC_OLS_coefficients.csv', index=False)
    manifest = dict(synthetic=True, biological_claim=False, rows=len(data), seed=53,
        fitting_api='statsmodels.api.OLS(...).fit()', link='identity',
        coefficients={str(k): float(v) for k, v in fit.params.items()},
        generation='response = 2 + 4*input_a - 1.5*input_b + N(0, 0.03); 11 by 11 grid',
        independent_coefficient_max_error=float(np.max(np.abs(fit.params-independent))),
        observed_ranges={k: [float(data[k].min()), float(data[k].max())]
                         for k in ('input_a', 'input_b')},
        held_out_validation=False, fit_in_profiler=False,
        note='Teaching example only. The GUI imports coefficients, assumes 0..1 ranges, '
             'and does not read these observations, refit, or validate a model.',
        files={p.name: digest(p) for p in sorted(destination.glob('*.csv'))})
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def verify_curve(curve, coefficients, *, variable, held, points):
    """Check every native curve value by direct scalar arithmetic."""
    if (curve['variable'] != variable or len(curve['values']) != points
            or len(curve['predictions']) != points or curve['scale'] != 'response'):
        raise ValueError('Native curve identity, size or response scale differs')
    values = np.asarray(curve['values'], dtype=float)
    if not np.allclose(values, np.linspace(0, 1, points), rtol=0, atol=1e-12):
        raise ValueError('Native sweep does not match the declared 0..1 range')
    expected = []
    for x in values:
        terms = {**held, variable: float(x)}
        expected.append(coefficients['Intercept'] + sum(
            coefficients[k] * terms[k] for k in ('input_a', 'input_b')))
    actual = np.asarray(curve['predictions'], dtype=float)
    if not np.allclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError('Native curve differs from saved OLS coefficients')
    for name, value in held.items():
        if not np.isclose(curve['held'].get(name, np.nan), value, rtol=0, atol=1e-12):
            raise ValueError('Native held input differs from the selected control')
    return dict(points=points, maximum_absolute_error=float(np.max(np.abs(actual-expected))),
                first=float(expected[0]), last=float(expected[-1]), checked=True)
