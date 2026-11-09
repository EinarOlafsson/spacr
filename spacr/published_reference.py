"""Compare a screen's per-guide or per-gene table with a published one.

A reanalysis of a public screen is credible when its numbers agree with the
published ones, and when that agreement is set against what chance alone
gives. This module joins a spaCR table with a published per-guide or
per-gene table and reports how well they agree.

* :func:`read_reference` reads a published table (CSV, TSV, Excel or
  Parquet, through :func:`spacr.tabular.read_table`). A wide per-gene table
  with one column per guide (``sgRNA_1``, ``sgRNA_1_interphase_cells``,
  ``sgRNA_2`` and so on, the layout of Funk et al. 2022's
  ``vesuvius_genes.xlsx``) becomes one row per guide, and the guide's cell
  columns are added up into ``cells``.
* :func:`match_reference` joins the two tables. Guides are matched on the
  first bases of their sequence (11 by default: the bases an optical pooled
  screen reads in situ), in upper case with ``U`` read as ``T``. Genes are
  matched on the symbol without regard to case, then on the reference's
  aliases where it lists them.
* :func:`reference_agreement` gives Spearman and Pearson correlations with
  bootstrap confidence intervals, and a control row for each: the same
  correlation after the gene labels are shuffled.
* :func:`plot_reference_agreement` draws one scatter of screen against
  reference values.
* :func:`compare_to_reference` runs all of it from settings and writes the
  joined table, the agreement table, a glossary and the figures;
  ``python -m spacr.published_reference`` does the same from the command
  line.

A REFERENCE ROW CAN BE THE SAME GUIDE TWICE. A per-gene table lists a guide
that targets two paralogs under both genes, with the same cell counts. At
the guide level ``published_duplicates='collapse'`` (the default) counts
such a guide once and names both genes; distinct guides that share a
prefix, which the screen cannot tell apart, have their counts added up.
``'keep'`` uses every reference row as it is, and ``'drop'`` leaves out
every key that more than one reference row has.

Everything runs on the CPU, in one process.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Key the settings of this module are registered under.
APP_KEY = 'published_reference'

#: The levels a comparison can be made at.
LEVELS = ('auto', 'guide', 'gene')

#: How reference rows that share one key are treated.
DUPLICATES = ('collapse', 'keep', 'drop')

#: How a reference key the screen does not have is treated.
UNMATCHED = ('drop', 'zero')

#: What each column of the agreement table means.
GLOSSARY: Dict[str, str] = {
    'pair': 'The screen column and the reference column compared, as screen = reference.',
    'level': 'What one point is: a guide (matched on its first bases) or a gene.',
    'method': 'Spearman (rank) or Pearson (linear) correlation coefficient.',
    'row': 'observed: the matched values; shuffled gene labels (control): the same values '
           'with the screen values shuffled across genes, which is what chance alone gives.',
    'n': 'The number of matched guides or genes with both values.',
    'estimate': 'The correlation; for the control row, the mean over the shuffles.',
    'ci_low': 'Lower end of the 95% interval: bootstrap percentile for the observed row, '
              '2.5th percentile of the shuffles for the control row.',
    'ci_high': 'Upper end of the 95% interval, as ci_low.',
    'p_shuffled': 'The share of shuffles at least as far from 0 as the observed correlation, '
                  '(k + 1) / (shuffles + 1).',
    'shuffled_p95_abs': '95th percentile of the absolute shuffled correlation: an observed '
                        'value below it is not resolved from chance.',
}

_GENE_COLUMNS = ('gene', 'gene_symbol', 'symbol', 'gene_name', 'genes', 'target_gene')
_GUIDE_COLUMNS = ('guide', 'sgrna', 'sgrna_sequence', 'spacer', 'guide_sequence',
                  'barcode', 'prefix')
_ALIAS_COLUMNS = ('aliases', 'alias', 'synonyms', 'gene_synonyms', 'previous_symbols',
                  'alias_symbols')
_SCREEN_COUNTS = ('n_cells', 'cells', 'cell_count', 'count', 'n')
_REFERENCE_COUNTS = ('cells', 'n_cells', 'cell_count', 'count', 'n')
_WIDE_GUIDE = re.compile(r'^(sgrna|guide)_(\d+)$', re.IGNORECASE)

_DEFAULTS: Dict[str, Any] = {
    'published_screen': '',
    'published_reference': '',
    'published_sheet': '',
    'published_level': 'auto',
    'published_screen_key': '',
    'published_reference_key': '',
    'published_prefix_length': 11,
    'published_counts': '',
    'published_values': '',
    'published_duplicates': 'collapse',
    'published_unmatched': 'drop',
    'published_bootstrap': 1000,
    'published_shuffles': 1000,
    'published_seed': 0,
    'published_dst': '',
}

_EXPECTED_TYPES: Dict[str, Any] = {
    'published_screen': str,
    'published_reference': str,
    'published_sheet': str,
    'published_level': str,
    'published_screen_key': str,
    'published_reference_key': str,
    'published_prefix_length': int,
    'published_counts': str,
    'published_values': str,
    'published_duplicates': str,
    'published_unmatched': str,
    'published_bootstrap': int,
    'published_shuffles': int,
    'published_seed': int,
    'published_dst': str,
}

_TOOLTIPS: Dict[str, str] = {
    'published_screen': "(str) - spaCR's per-guide or per-gene table for the screen, such as the OPS aggregate's guide or gene table (CSV, TSV, Excel or Parquet). Its rows are compared with the published table's. Default blank.",
    'published_reference': "(str) - The published per-guide or per-gene table to compare with, such as Funk et al. 2022's vesuvius_genes.xlsx. A wide table with one column per guide (sgRNA_1, sgRNA_1_interphase_cells and so on) is read as one row per guide, its cell columns added up into cells. Default blank.",
    'published_sheet': "(str) - The sheet of an Excel reference to read. Blank reads the first sheet. Default blank.",
    'published_level': "(str) - What one point of the comparison is: 'guide' matches guides on their first bases, 'gene' matches gene symbols (a per-guide table is first reduced to genes: cells added up, other values the median of the guides). 'auto' compares guides when both tables have a guide column, else genes. Default 'auto'.",
    'published_screen_key': "(str) - The screen column holding the guide sequence or gene symbol to match on. Blank finds it: guide, sgRNA, barcode or prefix for guides; gene, gene_symbol or symbol for genes. Default blank.",
    'published_reference_key': "(str) - The reference column holding the guide sequence or gene symbol to match on, found as for the screen when blank. Default blank.",
    'published_prefix_length': "(int) - The number of leading bases guides are matched on: an optical pooled screen reads only the first bases of each guide in situ, 11 in Funk et al. 2022. 0 matches the whole sequence. Default 11.",
    'published_counts': "(str) - The cell-count columns to compare, as screen=reference, for example n_cells=cells. Blank uses n_cells, cells or count on each side. Default blank.",
    'published_values': "(str) - Further columns to compare, comma-separated, as screen=reference or one name both tables use, for example mitotic_index. Blank compares every numeric column both tables share. Default blank.",
    'published_duplicates': "(str) - Reference rows that share one guide key: 'collapse' counts a guide listed under two genes once and adds up the counts of distinct guides that share a prefix; 'keep' uses every row as it is; 'drop' leaves out every shared key. Gene comparisons always pool a gene's rows. Default 'collapse'.",
    'published_unmatched': "(str) - A reference guide or gene the screen has no row for: 'drop' leaves it out; 'zero' compares it as 0 cells, for a screen table that lists only guides with cells. Other values stay missing. Default 'drop'.",
    'published_bootstrap': "(int) - Bootstrap resamples of the matched rows for the 95% intervals of the correlations. 0 reports no interval. Default 1000.",
    'published_shuffles': "(int) - Shuffles of the gene labels for the control row, which shows what agreement chance alone gives. 0 writes no control. Default 1000.",
    'published_seed': "(int) - Random seed for the bootstrap and the shuffles, so a rerun gives the same table. Default 0.",
    'published_dst': "(str) - Folder the joined table, the agreement table, the glossary and the figures are written to. Blank writes to a published_reference folder beside the screen table. Default blank.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Published reference': ['published_screen', 'published_reference', 'published_sheet',
                            'published_level', 'published_screen_key',
                            'published_reference_key', 'published_prefix_length',
                            'published_dst'],
    'Published reference agreement': ['published_counts', 'published_values',
                                      'published_duplicates', 'published_unmatched',
                                      'published_bootstrap', 'published_shuffles',
                                      'published_seed'],
}

_DESCRIPTION = (
    "Compare a screen's per-guide or per-gene table with a published one: "
    "match guides and genes, and report Spearman and Pearson correlations "
    "with bootstrap intervals and a shuffled-gene-label control."
)


def published_reference_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`compare_to_reference`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key of this module.
    """
    merged = dict(settings or {})
    for key, value in _DEFAULTS.items():
        merged.setdefault(key, value)
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam.

    Guarded so a reload of this module in one process is not a second
    registration.
    """
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, published_reference_defaults,
                      expected_types=_EXPECTED_TYPES, tooltips=_TOOLTIPS,
                      categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def _say(text: str, **values: Any) -> None:
    """Print one translated progress line.

    :param text: the English message, with ``str.format`` placeholders.
    :param values: the placeholder values.
    """
    from .qt.i18n import tr

    print(tr(text, **values), flush=True)


def _pick(columns: Sequence[str], names: Sequence[str]) -> Optional[str]:
    """The first column whose name, in lower case, is one of ``names``.

    :param columns: the table's column names.
    :param names: accepted names in lower case, in order of preference.
    :returns: the column as spelled in the table, or None.
    """
    lowered = {str(column).strip().lower(): column for column in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _column(frame: pd.DataFrame, name: str) -> str:
    """The column of ``frame`` called ``name``, matched without regard to case.

    :raises ValueError: when there is none.
    """
    found = _pick(list(frame.columns), (str(name).strip().lower(),))
    if found is None:
        raise ValueError(f"no column {name!r}; the table has {list(frame.columns)[:20]}")
    return found


def _melt_wide_guides(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per guide from a per-gene table with one column group per guide.

    ``sgRNA_<k>`` holds guide k's sequence and ``sgRNA_<k>_<name>`` its
    values; the result has ``sgRNA`` and one ``<name>`` column per value,
    the per-gene columns repeated on every guide row, and ``cells``, the sum
    of the ``*_cells`` columns, unless the table names ``cells`` itself.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    sequences: Dict[str, str] = {}
    for column in frame.columns:
        match = _WIDE_GUIDE.match(str(column))
        if match:
            sequences[match.group(2)] = column
    if len(sequences) < 2:
        return frame
    for number, column in sequences.items():
        prefix = f"{column}_"
        groups[number] = [(other, str(other)[len(prefix):]) for other in frame.columns
                          if str(other).startswith(prefix)]
    wide = {other for members in groups.values() for other, _ in members}
    shared = [c for c in frame.columns if c not in wide and c not in sequences.values()]
    parts = []
    for number in sorted(sequences, key=int):
        part = frame[shared].copy()
        part['sgRNA'] = frame[sequences[number]]
        part['guide_number'] = int(number)
        for other, name in groups[number]:
            part[name] = frame[other]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long = long[long['sgRNA'].notna() & (long['sgRNA'].astype(str).str.strip() != '')]
    cell_columns = [c for c in long.columns if str(c).lower().endswith('_cells')
                    and c not in shared]
    if cell_columns and _pick(list(long.columns), ('cells',)) is None:
        numbers = long[cell_columns].apply(pd.to_numeric, errors='coerce')
        long['cells'] = numbers.sum(axis=1, min_count=1)
    return long.reset_index(drop=True)


def read_reference(source: Any, *, sheet: Any = None) -> pd.DataFrame:
    """Read a published per-guide or per-gene table.

    A path is read with :func:`spacr.tabular.read_table`, its column names
    as written; a frame is copied. A per-gene table with two or more
    ``sgRNA_<k>`` (or ``guide_<k>``) columns becomes one row per guide:
    ``sgRNA``, ``guide_number``, the guide's own ``sgRNA_<k>_<name>``
    values as ``<name>``, every per-gene column, and ``cells``, the sum of
    the guide's ``*_cells`` columns (interphase and mitotic cells in Funk
    et al. 2022).

    :param source: a path, or a :class:`pandas.DataFrame`.
    :param sheet: the Excel sheet to read; None reads the first.
    :returns: the table, one row per guide when it was wide.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        from .tabular import read_table

        kwargs = {}
        path = os.fspath(source)
        if sheet not in (None, '') and Path(path).suffix.lower() in ('.xlsx', '.xls', '.xlsm'):
            kwargs['sheet_name'] = sheet
        frame = read_table(path, canonicalise=False, report=None, **kwargs)
    return _melt_wide_guides(frame)


def _guide_key(values: pd.Series, prefix_length: int) -> pd.Series:
    """Guide sequences as match keys: upper case, ``U`` as ``T``, first bases."""
    text = values.astype('string').str.strip().str.upper().str.replace('U', 'T', regex=False)
    if int(prefix_length) > 0:
        text = text.str.slice(0, int(prefix_length))
    return text.where(text.notna() & (text != ''))


def _gene_key(values: pd.Series) -> pd.Series:
    """Gene symbols as match keys: stripped, upper case."""
    text = values.astype('string').str.strip().str.upper()
    return text.where(text.notna() & (text != ''))


def _pairs(spec: Any, screen: pd.DataFrame, reference: pd.DataFrame
           ) -> List[Tuple[str, str]]:
    """``(screen column, reference column)`` pairs from ``'a=b, c'`` or a list."""
    if spec in (None, ''):
        return []
    items = spec.split(',') if isinstance(spec, str) else list(spec)
    pairs = []
    for item in items:
        if isinstance(item, (tuple, list)):
            left, right = item
        else:
            item = str(item).strip()
            if not item:
                continue
            left, _, right = item.partition('=')
            right = right or left
        pairs.append((_column(screen, left.strip()), _column(reference, right.strip())))
    return pairs


def _resolve_key(frame: pd.DataFrame, given: str, level: str) -> Optional[str]:
    """The key column of ``frame`` for ``level``, given or found."""
    if given:
        return _column(frame, given)
    return _pick(list(frame.columns), _GUIDE_COLUMNS if level == 'guide' else _GENE_COLUMNS)


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``columns`` of ``frame`` as floats, unreadable values missing."""
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors='coerce').astype(float)


def _pool(frame: pd.DataFrame, by: str, counts: Sequence[str], values: Sequence[str],
          labels: Optional[str] = None) -> pd.DataFrame:
    """One row per ``by``: counts added up, values the median.

    The pooling of :func:`spacr.gene_measurement_compare.aggregate_genes`,
    with ``rows`` (rows pooled) and, when ``labels`` is given, the distinct
    labels joined with ``;``.
    """
    from .gene_measurement_compare import aggregate_genes

    columns = list(dict.fromkeys(list(counts) + list(values)))
    work = _numeric(frame, columns)
    work.columns = [f'c{i}' for i in range(len(columns))]
    work.insert(0, 'gene', frame[by].to_numpy())
    work.insert(1, 'n_cells', 0)
    pooled = aggregate_genes(work, [f'c{i}' for i in range(len(columns))])
    grouped = work.groupby('gene', sort=True)
    for i, column in enumerate(columns):
        if column in counts:
            pooled[f'c{i}'] = grouped[f'c{i}'].sum(min_count=1).to_numpy()
    out = pd.DataFrame({by: pooled['gene'].to_numpy(), 'rows': pooled['n_guides'].to_numpy()})
    for i, column in enumerate(columns):
        out[column] = pooled[f'c{i}'].to_numpy()
    if labels is not None:
        joined = frame.groupby(by, sort=True)[labels].agg(
            lambda s: ';'.join(sorted({str(v) for v in s.dropna()})))
        out[labels] = joined.reindex(out[by]).to_numpy()
    return out


def _alias_map(reference: pd.DataFrame, key: str) -> Dict[str, str]:
    """Each alias (upper case) to the one reference symbol that lists it.

    An alias that is itself a reference symbol, or that two symbols list,
    is left out: it does not name one gene.
    """
    column = _pick(list(reference.columns), _ALIAS_COLUMNS)
    if column is None:
        return {}
    symbols = set(reference[key].dropna())
    owners: Dict[str, set] = {}
    for symbol, aliases in zip(reference[key], reference[column]):
        if pd.isna(symbol) or pd.isna(aliases):
            continue
        for alias in re.split(r'[,;|/]', str(aliases)):
            alias = alias.strip().upper()
            if alias and alias not in symbols:
                owners.setdefault(alias, set()).add(symbol)
    return {alias: next(iter(found)) for alias, found in owners.items() if len(found) == 1}


def match_reference(screen: pd.DataFrame, reference: pd.DataFrame, *, level: str = 'auto',
                    screen_key: str = '', reference_key: str = '', prefix_length: int = 11,
                    counts: Any = '', values: Any = '', duplicates: str = 'collapse',
                    unmatched: str = 'drop') -> pd.DataFrame:
    """Join a screen table with a published one, guide by guide or gene by gene.

    :param screen: spaCR's per-guide or per-gene table.
    :param reference: the published table, as :func:`read_reference` returns it.
    :param level: ``'guide'``, ``'gene'`` or ``'auto'`` (guides when both
        tables have a guide column).
    :param screen_key: the screen's key column; blank finds it.
    :param reference_key: the reference's key column; blank finds it.
    :param prefix_length: leading bases guides are matched on; 0 for all.
    :param counts: cell-count pairs, ``'screen=reference'`` or a list of
        pairs; blank takes ``n_cells``, ``cells`` or ``count`` on each side.
    :param values: further pairs, as ``counts``; blank takes every numeric
        column both tables share.
    :param duplicates: ``'collapse'``, ``'keep'`` or ``'drop'`` (see the
        module notes); guides only.
    :param unmatched: ``'drop'``, or ``'zero'`` to compare a reference key
        the screen lacks as 0 cells.
    :returns: one row per matched key: ``key``, ``gene``, ``match`` (how
        the key matched: ``guide``, ``symbol``, ``case``, ``alias`` or
        ``unmatched``), ``reference_rows``, then ``screen_<column>`` and
        ``reference_<column>`` per pair. ``attrs['pairs']`` lists the pairs
        as ``(screen column, reference column, kind)`` and
        ``attrs['match']`` counts what matched.
    :raises ValueError: for an unknown option, a key column that cannot be
        found, or no column to compare.
    """
    level = str(level or 'auto').strip().lower()
    duplicates = str(duplicates or 'collapse').strip().lower()
    unmatched = str(unmatched or 'drop').strip().lower()
    for name, value, allowed in (('level', level, LEVELS), ('duplicates', duplicates, DUPLICATES),
                                 ('unmatched', unmatched, UNMATCHED)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {list(allowed)}; got {value!r}")
    if level == 'auto':
        both = all(_resolve_key(frame, given, 'guide') is not None
                   for frame, given in ((screen, screen_key), (reference, reference_key)))
        level = 'guide' if both else 'gene'
    skey = _resolve_key(screen, screen_key, level)
    rkey = _resolve_key(reference, reference_key, level)
    if skey is None or rkey is None:
        side = 'screen' if skey is None else 'reference'
        raise ValueError(f"the {side} table has no {level} column; name it with "
                         f"published_{side}_key")
    count_pairs = _pairs(counts, screen, reference)
    if not count_pairs:
        found = (_pick(list(screen.columns), _SCREEN_COUNTS),
                 _pick(list(reference.columns), _REFERENCE_COUNTS))
        if None not in found:
            count_pairs = [found]
    value_pairs = _pairs(values, screen, reference)
    if values in (None, ''):
        taken = {skey, rkey, *[c for pair in count_pairs for c in pair]}
        lowered = {str(c).lower(): c for c in reference.columns if c not in taken}
        for column in screen.columns:
            other = lowered.get(str(column).lower())
            if (column in taken or other is None or str(column).lower() in _GUIDE_COLUMNS
                    + _GENE_COLUMNS + ('guide_number', 'rows', 'n_guides')):
                continue
            if (_numeric(screen, [column])[column].notna().any()
                    and _numeric(reference, [other])[other].notna().any()):
                value_pairs.append((column, other))
    pairs = [(s, r, 'count') for s, r in count_pairs] + [
        (s, r, 'value') for s, r in value_pairs if (s, r) not in count_pairs]
    if not pairs:
        raise ValueError("no column to compare; name them with published_counts or "
                         "published_values")
    s_counts = [s for s, _, kind in pairs if kind == 'count']
    r_counts = [r for _, r, kind in pairs if kind == 'count']
    s_values = [s for s, _, kind in pairs if kind == 'value']
    r_values = [r for _, r, kind in pairs if kind == 'value']
    rgene = _pick(list(reference.columns), _GENE_COLUMNS)
    summary: Dict[str, Any] = {'level': level, 'screen_rows': int(len(screen)),
                               'reference_rows': int(len(reference))}

    ref = reference.copy()
    scr = screen.copy()
    if level == 'guide':
        ref['_key'] = _guide_key(ref[rkey], prefix_length)
        scr['_key'] = _guide_key(scr[skey], prefix_length)
    else:
        ref['_key'] = _gene_key(ref[rkey])
        scr['_key'] = _gene_key(scr[skey])
    ref = ref[ref['_key'].notna()]
    scr = scr[scr['_key'].notna()]
    ref['_gene'] = (ref[rgene].astype('string') if rgene is not None
                    else ref['_key']) if level == 'guide' else ref[rkey].astype('string').str.strip()

    if level == 'guide':
        full = _guide_key(ref[rkey], 0)
        summary['reference_rows_repeating_a_guide'] = int(
            full.duplicated(keep=False).sum())
        if duplicates == 'collapse':
            ref['_full'] = full
            columns = list(dict.fromkeys(r_counts + r_values))
            genes = ref.groupby('_full', sort=False)['_gene'].agg(
                lambda s: ';'.join(sorted({str(v) for v in s.dropna()})))
            ref = ref.drop_duplicates('_full').copy()
            ref['_gene'] = genes.reindex(ref['_full']).to_numpy()
            ref = _pool(ref, '_key', r_counts, [c for c in columns if c not in r_counts],
                        labels='_gene')
        elif duplicates == 'drop':
            shared = ref['_key'].duplicated(keep=False)
            summary['reference_keys_dropped_as_shared'] = int(ref.loc[shared, '_key'].nunique())
            ref = ref.loc[~shared, ['_key', '_gene', *dict.fromkeys(r_counts + r_values)]]
            ref = ref.assign(rows=1)
        else:
            ref = ref[['_key', '_gene', *dict.fromkeys(r_counts + r_values)]].assign(rows=1)
    else:
        aliases = _alias_map(reference.assign(_sym=_gene_key(reference[rkey])), '_sym')
        ref = _pool(ref, '_key', r_counts, list(dict.fromkeys(r_values)), labels='_gene')
        symbols = set(ref['_key'])
        raw = scr[skey].astype('string').str.strip()
        how = np.where(raw.isin(symbols), 'symbol', np.where(scr['_key'].isin(symbols), 'case', ''))
        alias_hit = (how == '') & scr['_key'].map(aliases).notna().to_numpy()
        scr.loc[alias_hit, '_key'] = scr.loc[alias_hit, '_key'].map(aliases)
        how = np.where(alias_hit, 'alias', how)
        scr['_match'] = how
        summary['screen_genes_matched_by_case'] = int((how == 'case').sum())
        summary['screen_genes_matched_by_alias'] = int(alias_hit.sum())

    match = scr.groupby('_key')['_match'].first() if '_match' in scr else None
    summary['screen_rows_sharing_a_key'] = int(scr['_key'].duplicated(keep=False).sum())
    pooled_screen = _pool(scr, '_key', list(dict.fromkeys(s_counts)),
                          list(dict.fromkeys(s_values)))
    on_screen = set(pooled_screen['_key'])
    in_reference = set(ref['_key'])
    summary['screen_keys'] = int(len(on_screen))
    summary['reference_keys'] = int(len(in_reference))
    summary['screen_keys_not_in_reference'] = int(len(on_screen - in_reference))
    summary['reference_keys_not_in_screen'] = int(len(in_reference - on_screen))
    how_join = 'left' if unmatched == 'zero' else 'inner'
    joined = ref.merge(pooled_screen, on='_key', how=how_join, suffixes=('_r', '_s'),
                       indicator=True)
    out = pd.DataFrame({'key': joined['_key'].to_numpy()})
    gene = joined['_gene'] if '_gene' in joined else joined['_key']
    out['gene'] = gene.to_numpy()
    if level == 'guide':
        out['match'] = 'guide'
    else:
        out['match'] = joined['_key'].map(match).fillna('').to_numpy()
    missing = (joined['_merge'] == 'left_only').to_numpy()
    out.loc[missing, 'match'] = 'unmatched'
    rows_column = 'rows_r' if 'rows_r' in joined else 'rows'
    out['reference_rows'] = joined[rows_column].to_numpy()

    for s, r, kind in pairs:
        screen_values = _side(joined, s, 's').to_numpy(float).copy()
        if kind == 'count' and unmatched == 'zero':
            screen_values[missing] = 0.0
        out[f'screen_{s}'] = screen_values
        out[f'reference_{r}'] = _side(joined, r, 'r').to_numpy(float)
    summary['matched'] = int((~missing).sum())
    summary['compared'] = int(len(out))
    out.attrs['pairs'] = [(s, r, kind) for s, r, kind in pairs]
    out.attrs['match'] = summary
    return out.sort_values('key', kind='stable').reset_index(drop=True)


def _side(joined: pd.DataFrame, column: str, suffix: str) -> pd.Series:
    """``column`` of a merged frame, under its merge suffix if it has one."""
    return joined[f'{column}_{suffix}'] if f'{column}_{suffix}' in joined else joined[column]


def _ranks(matrix: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis."""
    from scipy.stats import rankdata

    return rankdata(matrix, axis=-1)


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of ``x`` with the same row of ``y``."""
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((x * x).sum(axis=-1) * (y * y).sum(axis=-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denominator > 0, (x * y).sum(axis=-1) / denominator, np.nan)


def _correlations(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Row-wise Spearman or Pearson of two matrices of equal shape."""
    if method == 'spearman':
        return _rowwise_pearson(_ranks(x), _ranks(y))
    return _rowwise_pearson(x, y)


def reference_agreement(joined: pd.DataFrame, pairs: Optional[Sequence[Tuple[str, str]]] = None,
                        *, n_boot: int = 1000, shuffles: int = 1000, seed: int = 0,
                        chunk: int = 100) -> pd.DataFrame:
    """Spearman and Pearson agreement with bootstrap intervals and a shuffled control.

    :param joined: the table :func:`match_reference` returns.
    :param pairs: ``(screen column, reference column)`` pairs, the bare
        names of the joined table's ``screen_`` and ``reference_`` columns;
        None takes ``joined.attrs['pairs']``.
    :param n_boot: bootstrap resamples of the matched rows; 0 for no
        interval.
    :param shuffles: gene-label shuffles for the control; 0 for no
        control row.
    :param seed: the random seed.
    :param chunk: resamples computed at a time, which bounds the memory.
    :returns: per pair and method, an ``observed`` row and a ``shuffled
        gene labels (control)`` row: ``pair, level, method, row, n,
        estimate, ci_low, ci_high, p_shuffled, shuffled_p95_abs`` (see
        :data:`GLOSSARY`).

    The interval is the 2.5th to 97.5th percentile of the correlation over
    resamples of the matched guides or genes, drawn with replacement. The
    control shuffles the screen values across the matched rows, which
    breaks the gene labels and keeps both distributions: a reanalysis that
    agrees only as well as its shuffled control does not reproduce the
    published screen.
    """
    if pairs is None:
        pairs = [(s, r) for s, r, *_ in joined.attrs.get('pairs', [])]
    level = joined.attrs.get('match', {}).get('level', '')
    rng = np.random.default_rng(int(seed))
    chunk = max(1, int(chunk))
    rows = []
    for s, r in pairs:
        x = pd.to_numeric(joined[f'screen_{s}'], errors='coerce').to_numpy(float)
        y = pd.to_numeric(joined[f'reference_{r}'], errors='coerce').to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        n = int(x.size)
        label = f'{s} = {r}'
        for method in ('spearman', 'pearson'):
            observed = float(_correlations(x[None], y[None], method)[0]) if n >= 3 else np.nan
            low = high = np.nan
            if n >= 3 and int(n_boot) > 0:
                draws = []
                for start in range(0, int(n_boot), chunk):
                    size = min(chunk, int(n_boot) - start)
                    index = rng.integers(0, n, size=(size, n))
                    draws.append(_correlations(x[index], y[index], method))
                draws = np.concatenate(draws)
                low, high = np.nanpercentile(draws, [2.5, 97.5])
            rows.append({'pair': label, 'level': level, 'method': method, 'row': 'observed',
                         'n': n, 'estimate': observed, 'ci_low': float(low),
                         'ci_high': float(high), 'p_shuffled': np.nan,
                         'shuffled_p95_abs': np.nan})
            if int(shuffles) <= 0:
                continue
            null = np.full(0, np.nan)
            if n >= 3:
                xs = _ranks(x) if method == 'spearman' else x
                ys = _ranks(y) if method == 'spearman' else y
                parts = []
                for start in range(0, int(shuffles), chunk):
                    size = min(chunk, int(shuffles) - start)
                    order = rng.permuted(np.tile(np.arange(n), (size, 1)), axis=1)
                    parts.append(_rowwise_pearson(xs[order], np.broadcast_to(ys, (size, n))))
                null = np.concatenate(parts)
            finite = null[np.isfinite(null)]
            if finite.size:
                p = float((np.sum(np.abs(finite) >= abs(observed)) + 1) / (finite.size + 1))
                low, high = np.percentile(finite, [2.5, 97.5])
                rows.append({'pair': label, 'level': level, 'method': method,
                             'row': 'shuffled gene labels (control)', 'n': n,
                             'estimate': float(finite.mean()), 'ci_low': float(low),
                             'ci_high': float(high), 'p_shuffled': p,
                             'shuffled_p95_abs': float(np.percentile(np.abs(finite), 95))})
            else:
                rows.append({'pair': label, 'level': level, 'method': method,
                             'row': 'shuffled gene labels (control)', 'n': n,
                             'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                             'p_shuffled': np.nan, 'shuffled_p95_abs': np.nan})
            rows[-2]['p_shuffled'] = rows[-1]['p_shuffled']
            rows[-2]['shuffled_p95_abs'] = rows[-1]['shuffled_p95_abs']
    return pd.DataFrame(rows, columns=['pair', 'level', 'method', 'row', 'n', 'estimate',
                                       'ci_low', 'ci_high', 'p_shuffled', 'shuffled_p95_abs'])


def plot_reference_agreement(joined: pd.DataFrame, screen_column: str, reference_column: str,
                             *, agreement: Optional[pd.DataFrame] = None,
                             path: Optional[str] = None, log: Optional[bool] = None):
    """Scatter of the screen's values against the published ones.

    :param joined: the table :func:`match_reference` returns.
    :param screen_column: the bare screen column of a pair.
    :param reference_column: the bare reference column of a pair.
    :param agreement: :func:`reference_agreement` output; its Spearman and
        Pearson rows, with the control, are printed on the figure.
    :param path: write the figure here with :func:`spacr.plot.save_figure`;
        None only returns it.
    :param log: logarithmic axes; None uses them when both columns are
        positive counts spanning more than a factor of 100.
    :returns: the :class:`matplotlib.figure.Figure`.
    """
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES

    x_name, y_name = f'reference_{reference_column}', f'screen_{screen_column}'
    data = joined[['key', 'gene', x_name, y_name]].dropna(subset=[x_name, y_name])
    fig = Figure(figsize=(4.2, 4.0), dpi=100)
    ax = fig.subplots()
    _register_figure_data(fig, data, x=x_name, y=y_name, kind='scatter',
                          title=f'{screen_column} against the published {reference_column}')
    x = data[x_name].to_numpy(float)
    y = data[y_name].to_numpy(float)
    if log is None:
        positive = x.size > 0 and (x > 0).all() and (y > 0).all()
        log = bool(positive and x.max() / x.min() > 100 and y.max() / y.min() > 100)
    ax.scatter(x, y, s=6, color=ROLES['highlight'], alpha=0.5, linewidths=0, rasterized=True)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(f'published {reference_column}')
    ax.set_ylabel(f'screen {screen_column}')
    if agreement is not None and len(agreement):
        rows = agreement[agreement['pair'] == f'{screen_column} = {reference_column}']
        lines = []
        for method, name in (('spearman', 'Spearman'), ('pearson', 'Pearson')):
            observed = rows[(rows['method'] == method) & (rows['row'] == 'observed')]
            control = rows[(rows['method'] == method) & (rows['row'] != 'observed')]
            if observed.empty:
                continue
            first = observed.iloc[0]
            text = f'{name} {first["estimate"]:.3f}'
            if np.isfinite(first['ci_low']):
                text += f' [{first["ci_low"]:.3f}, {first["ci_high"]:.3f}]'
            if not control.empty and np.isfinite(control.iloc[0]['estimate']):
                text += f'; shuffled {control.iloc[0]["estimate"]:.3f}'
            lines.append(text)
        lines.append(f'n = {len(data)}')
        ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes, ha='left', va='top',
                fontsize='small', color=ROLES['reference'])
    fig.tight_layout()
    if path:
        from .plot import save_figure

        save_figure(fig, path, bbox_inches='tight')
    return fig


def _safe_name(text: str) -> str:
    """A file-name stem from a column name."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_') or 'value'


def compare_to_reference(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare a screen table with a published one and write the results.

    Reads both tables (:func:`read_reference` for the reference), joins
    them (:func:`match_reference`), measures their agreement
    (:func:`reference_agreement`) and draws one scatter per pair
    (:func:`plot_reference_agreement`). Written to ``published_dst``
    through :func:`spacr.tabular.write_table`: ``joined.csv``,
    ``agreement.csv``, ``glossary.csv`` and ``summary.json`` (the match
    counts and the settings), and ``agreement_<column>`` figures.

    :param settings: the keys of :func:`published_reference_defaults`;
        missing keys take their defaults. ``published_screen`` and
        ``published_reference`` may also be frames.
    :returns: dict with ``joined``, ``agreement`` and ``glossary`` (the
        frames), ``match`` (the counts of :func:`match_reference`) and
        ``paths`` (every file written).
    :raises ValueError: without a screen or a reference table, or for
        settings :func:`match_reference` rejects.
    """
    from .plot import save_figure
    from .tabular import write_table

    settings = published_reference_defaults(settings)
    screen_source = settings['published_screen']
    reference_source = settings['published_reference']
    if screen_source is None or (isinstance(screen_source, str) and not screen_source):
        raise ValueError('published_screen is blank: name the screen table to compare')
    if reference_source is None or (isinstance(reference_source, str) and not reference_source):
        raise ValueError('published_reference is blank: name the published table to compare with')
    screen = read_reference(screen_source)
    reference = read_reference(reference_source, sheet=settings['published_sheet'] or None)
    joined = match_reference(
        screen, reference, level=settings['published_level'],
        screen_key=settings['published_screen_key'],
        reference_key=settings['published_reference_key'],
        prefix_length=int(settings['published_prefix_length']),
        counts=settings['published_counts'], values=settings['published_values'],
        duplicates=settings['published_duplicates'], unmatched=settings['published_unmatched'])
    agreement = reference_agreement(joined, n_boot=int(settings['published_bootstrap']),
                                    shuffles=int(settings['published_shuffles']),
                                    seed=int(settings['published_seed']))
    dst = settings['published_dst']
    if not dst:
        base = (Path(os.fspath(screen_source)).parent if not isinstance(screen_source, pd.DataFrame)
                else Path.cwd())
        dst = base / 'published_reference'
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    glossary = pd.DataFrame({'column': list(GLOSSARY), 'meaning': list(GLOSSARY.values())})
    paths = [write_table(joined, dst / 'joined.csv', canonicalise=False),
             write_table(agreement, dst / 'agreement.csv', canonicalise=False),
             write_table(glossary, dst / 'glossary.csv', canonicalise=False)]
    for s, r, _kind in joined.attrs['pairs']:
        figure = plot_reference_agreement(joined, s, r, agreement=agreement)
        paths.append(save_figure(figure, str(dst / f'agreement_{_safe_name(s)}.pdf'),
                                 bbox_inches='tight'))
    recorded = {key: (value if isinstance(value, (str, int, float, bool)) else '<frame>')
                for key, value in settings.items() if key in _DEFAULTS}
    summary = {'match': joined.attrs['match'], 'settings': recorded,
               'agreement': json.loads(agreement.to_json(orient='records'))}
    summary_path = dst / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    paths.append(str(summary_path))
    observed = agreement[(agreement['row'] == 'observed') & (agreement['method'] == 'spearman')]
    for _, row in observed.iterrows():
        _say('{pair}: Spearman {rho:.3f} over {n} matched rows',
             pair=row['pair'], rho=row['estimate'], n=int(row['n']))
    return {'joined': joined, 'agreement': agreement, 'glossary': glossary,
            'match': joined.attrs['match'], 'paths': [str(p) for p in paths]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`compare_to_reference` from the command line.

    ``python -m spacr.published_reference --screen guides.csv --reference
    vesuvius_genes.xlsx --dst out`` compares guide cell counts; ``--level
    gene`` compares genes.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0 when at least one pair was compared, 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog='python -m spacr.published_reference',
                                     description=_DESCRIPTION)
    for key, value in _DEFAULTS.items():
        flag = '--' + key[len('published_'):].replace('_', '-')
        parser.add_argument(flag, dest=key, type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    result = compare_to_reference(args)
    return 0 if len(result['agreement']) else 1


if __name__ == '__main__':
    sys.exit(main())
