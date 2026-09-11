"""Explicit API workaround for the demonstrated all-missing metadata failure.

This is tutorial example code, not a patch to spaCR's GUI exporter. Only
entirely missing object-typed observation columns receive an explicit empty
categorical encoding. Unknown remains missing: no zero, string 'None', voxel
calibration, annotation, or prediction is invented. X is never imputed here.
The encoding is a storage choice, not a claim that voxel size is categorical;
real calibration must be supplied as numeric metadata when known.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def encode_empty_metadata(adata):
    """Return an independent copy plus the explicit storage conversions."""
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError('An empty export is not a completed tutorial example')
    result = adata.copy()
    converted = []
    for name in result.obs:
        series = result.obs[name]
        if series.dtype == object and series.isna().all():
            result.obs[name] = pd.Categorical(series)
            converted.append(name)
    result.uns['tutorial_missing_metadata_encoding'] = {
        'columns': converted, 'encoding': 'empty categorical; all values missing',
        'gui_exporter_fixed': False, 'feature_imputation_performed_here': False,
        'numeric_calibration_supplied': False,
    }
    verify_roundtrip(adata, result)
    return result, converted


def verify_roundtrip(expected, actual):
    """Reject changes in feature identity, values, metadata, or missingness."""
    if (list(expected.obs_names) != list(actual.obs_names)
            or list(expected.var_names) != list(actual.var_names)):
        raise ValueError('Observation or feature identity changed')
    if (expected.shape != actual.shape
            or not np.array_equal(expected.X, actual.X, equal_nan=True)):
        raise ValueError('Feature values changed during metadata encoding or storage')
    if list(expected.obs.columns) != list(actual.obs.columns):
        raise ValueError('Observation metadata columns changed')
    for name in expected.obs:
        source, saved = expected.obs[name], actual.obs[name]
        if not np.array_equal(source.isna().to_numpy(), saved.isna().to_numpy()):
            raise ValueError('Missing observation metadata changed: ' + name)
        valid = ~source.isna()
        if list(source[valid]) != list(saved[valid]):
            raise ValueError('Nonmissing observation metadata changed: ' + name)
    return {'matrix_values_checked': int(actual.X.size),
            'missing_feature_values': int(np.isnan(actual.X).sum()),
            'metadata_columns_checked': len(actual.obs.columns),
            'shape': list(actual.shape)}


def export_example(database, destination, *, single_table=None, nan_policy='keep'):
    """Use the real builder, encode missing metadata, save and verify atomically.

    The caller chooses the real spaCR feature-missingness policy explicitly.
    Only this separate metadata encoding is performed by the example. An
    existing destination is never overwritten; failed temporary files are
    not published as successful exports. The database is not modified.
    """
    import anndata
    from spacr.anndata_export import build_anndata

    database, destination = Path(database).resolve(), Path(destination).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if destination.exists():
        raise FileExistsError(destination)
    source_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    settings = {'anndata_nan_policy': nan_policy, 'anndata_compute_umap': False,
                'anndata_single_table': single_table or '', 'anndata_dtype': 'float32'}
    original, result = build_anndata(database, single_table=single_table,
        nan_policy=nan_policy, dtype='float32', compute_umap=False,
        settings=settings, verbose=False)
    encoded, converted = encode_empty_metadata(original)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.anndata-tutorial-', dir=destination.parent) as temporary:
        working = Path(temporary) / 'pending.h5ad'
        encoded.write_h5ad(working, compression='gzip')
        reopened = anndata.read_h5ad(working)
        checks = verify_roundtrip(original, reopened)
        if hashlib.sha256(database.read_bytes()).hexdigest() != source_hash:
            raise ValueError('Source database changed during the export')
        # Same-filesystem hard link publishes only the verified bytes and
        # atomically refuses a destination created by somebody else meanwhile.
        os.link(working, destination)
    return {'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'source_database_sha256': source_hash, 'source_unchanged': True,
            'empty_metadata_columns': converted, 'roundtrip': checks,
            'single_table': single_table or '', 'nan_policy': nan_policy,
            'settings': settings, 'api_workaround': True, 'gui_defect_fixed': False,
            'result_summary': result.describe(), 'published': False}
