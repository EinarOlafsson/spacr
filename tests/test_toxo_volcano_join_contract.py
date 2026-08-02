"""``custom_volcano_plot``'s metadata join is many-to-one, and says so.

The volcano's left side is the regression result table: one row per *feature*,
and several features share a gene (a gRNA-level fit contributes one row per
guide), so ``gene_nr`` repeats there by design. The right side is a gene
lookup -- ``spacr/resources/data/lopit.csv``, 3832 rows and 3832 distinct
``gene_nr`` -- and the plot paints one point one colour from it.

Left many, right one. Getting that backwards in either direction is a bug:
``one_to_one`` would refuse ordinary gRNA-level results, and no contract at all
lets a duplicated metadata row silently double every affected gene, both on the
figure and in the hit list this function returns -- which then feeds
``plot_gene_phenotypes`` and ``plot_gene_heatmaps``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from spacr import toxo as T


def _results(genes_and_guides):
    """Regression results in the shape ``custom_volcano_plot`` parses.

    ``feature`` -> ``variable`` (the bracketed part) -> ``gene_nr`` (the token
    before the first underscore) is the function's own chain; the fixture goes
    through it rather than around it.
    """
    features = [f"grna[{gene}_{guide}]" for gene, guide in genes_and_guides]
    n = len(features)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'feature': features,
        'coefficient': rng.normal(0, 0.4, n),
        'p_value': np.linspace(1e-4, 0.4, n),
    })


def test_several_guides_of_one_gene_are_all_plotted():
    """The left side repeats gene_nr, and must be allowed to.

    This is the shape ``settings['volcano'] = 'grna'`` produces on every real
    run. ``validate='one_to_one'`` here would raise MergeError on it.
    """
    data = _results([(220001, 1), (220001, 2), (220001, 3), (220002, 1)])
    metadata = pd.DataFrame({
        'gene_nr': ['220001', '220002'],
        'tagm_location': ['cytosol', 'Golgi'],
    })

    hits = T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)

    assert isinstance(hits, list)
    # All four guides cleared p<=0.05 / |coef|>=0 or not; either way the call
    # completed, which is the claim. Check no guide was dropped or duplicated.
    assert len(set(hits)) == len(hits)


def test_duplicated_gene_metadata_is_refused_rather_than_fanned_out():
    """A gene listed twice cannot say which localisation is its colour.

    Without the contract this returned a figure: the gene drawn twice, once in
    each colour, and its name in the hit list twice.
    """
    data = _results([(220001, 1), (220002, 1)])
    metadata = pd.DataFrame({
        'gene_nr': ['220001', '220001', '220002'],
        'tagm_location': ['cytosol', 'Golgi', 'dense granules'],
    })

    with pytest.raises(pd.errors.MergeError) as excinfo:
        T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)

    message = str(excinfo.value)
    # The message has to name what is wrong and what to do, not just repeat
    # pandas' "Merge keys are not unique in right dataset".
    assert 'gene_nr' in message
    assert "'220001'" in message or '220001' in message
    assert 'De-duplicate' in message
    assert 'tagm_location' in message


class _PandasWithBrokenMerge:
    """``pandas``, but ``merge`` raises -- and only for the module it is bound into.

    ``T.pd`` IS the pandas module, so ``monkeypatch.setattr(T.pd, 'merge', ...)``
    rebinds ``pandas.merge`` for the whole interpreter: every other module in
    the process, test helper or production code, gets the broken ``merge`` for
    as long as it is installed. Patching the NAME ``pd`` inside ``spacr.toxo``
    instead keeps the break where the test means it, and leaves the real
    ``pandas.merge`` untouched.
    """

    def __init__(self, real, exc):
        self._real = real
        self._exc = exc

    def merge(self, *args, **kwargs):
        raise self._exc

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_a_merge_error_that_is_not_a_duplicate_is_re_raised_unchanged(monkeypatch):
    """The rewritten message must not describe a failure that did not happen.

    ``MergeError`` also covers colliding suffixes and unmergeable key dtypes.
    Reporting one of those as "the metadata lists 0 gene_nr values more than
    once" would send the reader off to fix a file that is fine, so the handler
    re-raises untouched when the duplicates it would blame are not there.
    """
    data = _results([(220001, 1)])
    metadata = pd.DataFrame({
        'gene_nr': ['220001'], 'tagm_location': ['cytosol']})

    boom = pd.errors.MergeError('columns overlap but no suffix specified')
    monkeypatch.setattr(T, 'pd', _PandasWithBrokenMerge(pd, boom))

    with pytest.raises(pd.errors.MergeError) as excinfo:
        T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)

    assert str(excinfo.value) == 'columns overlap but no suffix specified'
    assert 'De-duplicate' not in str(excinfo.value)


def test_the_broken_merge_stays_inside_the_module_under_test(monkeypatch):
    """The patch above must not reach ``pandas`` itself.

    Asserted rather than trusted, because the spelling that does leak --
    ``monkeypatch.setattr(T.pd, 'merge', ...)`` -- looks identical at a glance
    and was what this module used.
    """
    real_merge = pd.merge
    boom = pd.errors.MergeError('should never escape')
    monkeypatch.setattr(T, 'pd', _PandasWithBrokenMerge(pd, boom))

    assert pd.merge is real_merge
    left = pd.DataFrame({'a': [1], 'x': [10]})
    right = pd.DataFrame({'a': [1], 'y': [20]})
    assert pd.merge(left, right, on='a')['y'].tolist() == [20]
    # ...and the module under test really is broken, so the guard is not vacuous.
    with pytest.raises(pd.errors.MergeError):
        T.pd.merge(left, right, on='a')


def test_the_metadata_frame_the_caller_passed_is_not_retyped_underneath_them():
    """``metadata['gene_nr'] = ...astype(str)`` used to land in the caller's frame.

    ``data_path`` was copied; ``metadata_path`` was not. Plotting two volcanoes
    from one in-memory metadata table therefore handed the second call a frame
    whose gene numbers the first call had already turned into strings.
    """
    data = _results([(220001, 1)])
    metadata = pd.DataFrame({
        'gene_nr': [220001, 220002],
        'tagm_location': ['cytosol', 'Golgi'],
    })
    before = metadata['gene_nr'].dtype

    T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)

    assert metadata['gene_nr'].dtype == before
    assert metadata['gene_nr'].tolist() == [220001, 220002]


def test_the_shipped_lopit_metadata_satisfies_the_contract():
    """The production path must not be the thing the new contract breaks.

    ``ml.py`` hardcodes this file as ``metadata_path`` for all three volcano
    modes, so if it ever gained a duplicate row every toxo regression run would
    start failing here instead of silently double-plotting.
    """
    from pathlib import Path
    import spacr

    lopit = Path(spacr.__file__).parent / 'resources' / 'data' / 'lopit.csv'
    metadata = pd.read_csv(lopit)
    gene_nr = metadata['gene_nr'].astype(str)
    assert not gene_nr.duplicated().any(), (
        'lopit.csv gained a duplicated gene_nr; custom_volcano_plot will now '
        'refuse it, which is correct but needs the file fixed')
