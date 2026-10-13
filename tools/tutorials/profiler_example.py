"""Reproducible synthetic OLS teaching example, not a biological screen.

The native profiler imports only the saved coefficient CSV. It does not fit
this model or receive the design through a private application method.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

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


def package(source, archive):
    """Package checked original teaching inputs without refitting or overwriting."""
    source, archive = Path(source), Path(archive)
    manifest = json.loads((source/'manifest.json').read_text())
    names = {'SYNTHETIC_observations.csv', 'SYNTHETIC_OLS_coefficients.csv'}
    if (manifest.get('synthetic') is not True or set(manifest['files']) != names
            or any(digest(source/name) != manifest['files'][name] for name in names)):
        raise ValueError('Synthetic profiler inputs differ from the fitted manifest')
    files = {name: (source/name).read_bytes() for name in sorted(names | {'manifest.json'})}
    files['prepare_example.py'] = Path(__file__).read_bytes()
    files['README.txt'] = (
        'SYNTHETIC OLS TEACHING EXAMPLE -- NOT EXPERIMENTAL DATA\n\n'
        'Extract into a new folder. In spaCR, open Regression -> Prediction Profiler.\n'
        'Browse to SYNTHETIC_OLS_coefficients.csv. Keep Link = identity.\n'
        'The model was fitted to the 121 supplied synthetic observations before import.\n'
        'The GUI does not refit or read the observations; it assumes input ranges 0..1.\n'
        'The synthetic inputs happen to have those ranges. Do not assume this for your data.\n'
        'The ranked Moves by values use 5th/95th quantiles; the curve uses the full range.\n'
        'Changing Points changes the plotted grid, not the fitted model.\n'
        'No biological, causal, uncertainty-interval or held-out accuracy claim is made.\n\n'
        'Reproduce with NumPy, pandas and statsmodels installed:\n'
        '  python prepare_example.py --destination regenerated_example\n'
        'An existing destination is refused. manifest.json records generation, seed and hashes.\n'
    ).encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
        for name, contents in files.items(): handle.writestr('SYNTHETIC_profiler/'+name, contents)
    with zipfile.ZipFile(archive) as handle:
        if ({name.removeprefix('SYNTHETIC_profiler/'):handle.read(name) for name in handle.namelist()}
                != files or handle.testzip() is not None):
            raise ValueError('Packaged profiler inputs differ from the checked source')
    return dict(file=archive.name, sha256=digest(archive), bytes=archive.stat().st_size,
                members=len(files), source_refitted=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
