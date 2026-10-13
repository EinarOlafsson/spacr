from pathlib import Path
import sys

import anndata
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anndata_missing_metadata_example import encode_empty_metadata, verify_roundtrip, export_example


def example():
    return anndata.AnnData(
        X=np.array([[1, np.nan], [0, 4]], dtype='float32'),
        obs=pd.DataFrame({'empty': [None, None], 'label': ['one', None],
                          'zero_is_real': [0, 1]}, index=['cell1', 'cell2']),
        var=pd.DataFrame(index=['area', 'intensity']))


def test_real_roundtrip_preserves_missing_features_labels_and_zeros(tmp_path):
    source = example(); saved, columns = encode_empty_metadata(source)
    assert columns == ['empty']
    assert source.obs['empty'].dtype == object
    assert str(saved.obs['empty'].dtype) == 'category'
    assert saved.obs['empty'].isna().all()
    target = tmp_path / 'actual.h5ad'; saved.write_h5ad(target)
    reopened = anndata.read_h5ad(target)
    result = verify_roundtrip(source, reopened)
    assert result['matrix_values_checked'] == 4
    assert result['missing_feature_values'] == 1
    assert list(reopened.obs.zero_is_real) == [0, 1]
    assert reopened.uns['tutorial_missing_metadata_encoding']['gui_exporter_fixed'] == False


@pytest.mark.parametrize('kind,message', [('value', 'Feature values'),
    ('imputed', 'Feature values'), ('row', 'identity'), ('feature', 'identity'),
    ('missing', 'Missing observation'), ('label', 'Nonmissing observation'),
    ('column', 'columns')])
def test_roundtrip_rejects_changed_data(kind, message):
    original = example(); changed = original.copy()
    if kind == 'value': changed.X[0, 0] = 5
    elif kind == 'imputed': changed.X[0, 1] = 0
    elif kind == 'row': changed.obs_names = ['wrong', 'cell2']
    elif kind == 'feature': changed.var_names = ['wrong', 'intensity']
    elif kind == 'missing': changed.obs.loc['cell2', 'label'] = 'None'
    elif kind == 'label': changed.obs.loc['cell1', 'label'] = 'two'
    else: changed.obs['extra'] = 1
    with pytest.raises(ValueError, match=message): verify_roundtrip(original, changed)


def test_nonmissing_object_column_is_not_reencoded():
    original = example(); encoded, columns = encode_empty_metadata(original)
    assert encoded.obs.label.dtype == original.obs.label.dtype
    assert list(encoded.obs.label) == list(original.obs.label)
    assert 'label' not in columns


def test_empty_export_is_not_accepted():
    with pytest.raises(ValueError, match='empty export'):
        encode_empty_metadata(anndata.AnnData(X=np.empty((0, 2))))


def test_existing_destination_is_not_overwritten(tmp_path):
    database = tmp_path / 'exists.db'; database.touch()
    output = tmp_path / 'existing.h5ad'; output.touch()
    before = output.stat()
    with pytest.raises(FileExistsError): export_example(database, output)
    assert output.stat().st_ino == before.st_ino


def test_missing_source_is_not_exported(tmp_path):
    with pytest.raises(FileNotFoundError): export_example(tmp_path / 'missing.db', tmp_path / 'out.h5ad')
