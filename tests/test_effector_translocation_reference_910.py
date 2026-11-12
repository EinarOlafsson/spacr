"""Item 910, the parts that were not ported first: the Prism reference reader,
spaCR's own Measure readout binned by pathogen objects and the infection report.

On the HRMAn Protein_translocation pilot (3 fields) the curve read from
Result.pzfx equals the reference hrman_reference.csv (33 rows), and a settings
run with the Measure database beside it gives curves_all_readouts (140 rows,
the spacr_db_integrated_by_parasites curve included), agreement_with_hrman
(27 rows) and spacr_infection_report_per_well equal to the reference scripts'
output. These synthetic files pin the rules and the reader's edge cases.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import host_pathogen as hp


def _field(k, size=60):
    """One nucleus with ``k`` parasites around it and an effector signal rising with k."""
    nuc = np.zeros((size, size), np.int32)
    nuc[20:40, 20:40] = 1
    par = np.zeros_like(nuc)
    spots = [(2, 2), (2, 50), (50, 2), (50, 50), (2, 26), (50, 26), (26, 2), (26, 50)]
    for i in range(k):
        y, x = spots[i]
        par[y:y + 6, x:x + 6] = i + 1
    ha = np.full(nuc.shape, 100.0)
    ha[nuc > 0] += 50.0 * k + 500
    ha[par > 0] = 5000.0
    dna = np.where(nuc > 0, 2000.0, 50.0)
    return np.stack([dna, np.zeros_like(ha), ha, np.zeros_like(ha), nuc, par], axis=-1)


def _pzfx_table(title, signal, column='Nuclear signal', centred=False):
    """One Prism XY table with vacuoles/cell 0..n and the signal column."""
    x_title = (f'<Title><TextAlign align="Center">Vacuoles/cell</TextAlign></Title>'
               if centred else '<Title>Vacuoles/cell</Title>')
    xs = ''.join(f'<d>{k}</d>' for k in range(len(signal)))
    ys = ''.join(f'<d>{v}</d>' for v in signal)
    return (f'<Table ID="T" TableType="XY"><Title>{title}</Title>'
            f'<XColumn>{x_title}<Subcolumn>{xs}</Subcolumn></XColumn>'
            f'<YColumn><Title>{column}</Title><Subcolumn>{ys}</Subcolumn></YColumn></Table>')


def _pzfx(path, *tables):
    """A minimal Prism file holding ``tables`` and a binary template after them."""
    body = ''.join(tables)
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><GraphPadPrismFile PrismXMLVersion="5.00">'
                    '<TableSequence><Ref ID="T"></Ref></TableSequence>'
                    f'{body}<Template>eF7sXQl8FEXWr8</Template></GraphPadPrismFile>',
                    encoding='utf-8')
    return str(path)


def test_the_reference_curve_is_read_per_group_table(tmp_path):
    path = _pzfx(tmp_path / 'Result.pzfx',
                 _pzfx_table('TgIST (Group 1)', [4.0, 5.0, 6.0, 8.0]),
                 _pzfx_table('GRA16 (Group 2)', [2.0, 2.5, 3.0, 3.5], centred=True))
    curve = hp.pzfx_reference_curve(path, kmax=2)
    assert list(curve.columns) == ['effector', 'k', 'hrman_total', 'hrman_ratio']
    assert curve['effector'].tolist() == ['TgIST'] * 3 + ['GRA16'] * 3
    assert curve['k'].tolist() == [0, 1, 2] * 2
    assert curve['hrman_ratio'].tolist() == pytest.approx([1, 1.25, 1.5, 1, 1.25, 1.5])
    tables = hp.pzfx_tables(path)
    assert list(tables) == ['TgIST (Group 1)', 'GRA16 (Group 2)']
    assert list(tables['GRA16 (Group 2)'].columns) == ['Vacuoles/cell', 'Nuclear signal']


def test_empty_cells_ragged_columns_and_repeated_titles(tmp_path):
    table = ('<Table><Title>A</Title>'
             '<XColumn><Title>k</Title><Subcolumn><d>0</d><d>1</d><d>2</d></Subcolumn></XColumn>'
             '<YColumn><Title>s</Title><Subcolumn><d>1</d><d></d><d>x</d></Subcolumn></YColumn>'
             '<YColumn><Title>s</Title><Subcolumn><d>3</d></Subcolumn></YColumn></Table>')
    frame = hp.pzfx_tables(_pzfx(tmp_path / 'a.pzfx', table))['A']
    assert list(frame.columns) == ['k', 's', 's_2']
    assert frame['s'].iloc[0] == 1 and frame['s'].iloc[1:].isna().all()
    assert frame['s_2'].iloc[0] == 3 and frame['s_2'].iloc[1:].isna().all()


def test_a_malformed_or_foreign_file_is_refused(tmp_path):
    broken = tmp_path / 'broken.pzfx'
    broken.write_text('<GraphPadPrismFile><Table><Title>A</Title>', encoding='utf-8')
    with pytest.raises(ValueError, match='could not be read'):
        hp.pzfx_tables(str(broken))
    with pytest.raises(ValueError, match='could not be read'):
        hp.pzfx_tables(str(tmp_path / 'missing.pzfx'))
    other = tmp_path / 'other.pzfx'
    other.write_text('<svg><Table/></svg>', encoding='utf-8')
    with pytest.raises(ValueError, match='not a Prism'):
        hp.pzfx_tables(str(other))
    empty = _pzfx(tmp_path / 'empty.pzfx', '<Table><Title>A</Title></Table>')
    with pytest.raises(ValueError, match='no Prism data table'):
        hp.pzfx_tables(empty)


def test_a_missing_signal_column_or_zero_baseline_is_refused(tmp_path):
    other = _pzfx(tmp_path / 'o.pzfx', _pzfx_table('A (Group 1)', [1, 2], column='Area'))
    with pytest.raises(ValueError, match="no table has a 'Nuclear signal' column"):
        hp.pzfx_reference_curve(other)
    assert hp.pzfx_reference_curve(other, column='Area')['hrman_ratio'].tolist() == [1, 2]
    zero = _pzfx(tmp_path / 'z.pzfx', _pzfx_table('A (Group 1)', [0, 2]))
    with pytest.raises(ValueError, match='no k = 0 value'):
        hp.pzfx_reference_curve(zero)
    blank = _pzfx(tmp_path / 'b.pzfx', _pzfx_table('A (Group 1)', ['', 2]))
    with pytest.raises(ValueError, match='no k = 0 value'):
        hp.pzfx_reference_curve(blank)


def _measure_db(folder, rows):
    """A Measure database with the nucleus and pathogen tables of ``rows``.

    :param rows: ``(row, column, cell, integrated, area, parasites)`` per nucleus.
    """
    os.makedirs(folder)
    nucleus, pathogen = [], []
    for r, c, cell, integrated, area, parasites in rows:
        prcf = f'plate_r{r}_c{c}_1'
        nucleus.append(dict(prcf=prcf, rowID=f'r{r}', columnID=f'c{c}', nucleus_cell_id=cell,
                            nucleus_channel_2_integrated_intensity=integrated,
                            nucleus_channel_2_mean_intensity=integrated / area, nucleus_area=area))
        pathogen.extend(dict(prcf=prcf, pathogen_cell_id=cell) for _ in range(parasites))
    pathogen.append(dict(prcf='plate_r2_c2_1', pathogen_cell_id=0))
    with sqlite3.connect(os.path.join(folder, 'measurements.db')) as con:
        pd.DataFrame(nucleus).to_sql('nucleus', con, index=False)
        pd.DataFrame(pathogen).to_sql('pathogen', con, index=False)
    return os.path.join(folder, 'measurements.db')


ROWS = [(2, 2, 1, 1000.0, 300, 0), (2, 2, 2, 1500.0, 300, 2), (2, 3, 1, 1000.0, 300, 0),
        (2, 3, 2, 2000.0, 300, 2), (2, 3, 3, 9999.0, 100, 1), (4, 2, 1, 800.0, 300, 0),
        (4, 2, 2, 800.0, 300, 1), (5, 2, 1, 50.0, 300, 0)]


def test_the_measured_readout_counts_pathogen_objects_per_cell(tmp_path):
    db = _measure_db(tmp_path / 'measurements', ROWS)
    cells = hp.translocation_measured_cells(db, {'translocation_groups': 'B:A, D:B'})
    assert cells['well'].tolist()[:3] == ['B02', 'B02', 'B03']
    assert cells['n_par'].tolist() == [0, 2, 0, 2, 1, 0, 1, 0]
    assert cells['effector'].tolist()[:7] == ['A'] * 5 + ['B'] * 2
    assert pd.isna(cells['effector'].iloc[7])
    by_well = hp.translocation_measured_cells(db, {'translocation_groups': {'B03': 'C', 'B': 'A'}})
    assert by_well['effector'].tolist()[:5] == ['A', 'A', 'C', 'C', 'C']
    summary = hp.translocation_summary(_one_cell_table(), {'translocation_groups': 'B:A, D:B',
                                                           'translocation_min_nucleus_size': 250,
                                                           'translocation_shuffle_n': 2,
                                                           'translocation_kmax': 3}, None, cells)
    curve = summary['curves'][summary['curves']['readout'] == 'spacr_db_integrated_by_parasites']
    curve = curve.set_index(['effector', 'k'])
    assert curve.loc[('A', 2), 'ratio'] == pytest.approx(1.75)
    assert curve.loc[('B', 1), 'ratio'] == pytest.approx(1.0)
    assert ('A', 1) not in curve.index and curve.loc[('A', 0), 'count'] == 2


def test_a_database_without_the_effector_column_is_refused(tmp_path):
    db = _measure_db(tmp_path / 'measurements', ROWS)
    with pytest.raises(ValueError, match='nucleus_channel_5_integrated_intensity'):
        hp.translocation_measured_cells(db, {'translocation_channel': 5})


def _one_cell_table():
    """translocation_cells rows of two fields in wells B02 and D02."""
    rows = [hp.translocation_cells(_territories(_field(k)), {}, well, 1)
            for well, k in (('B02', 0), ('B02', 2), ('D02', 0), ('D02', 1))]
    return pd.concat(rows, ignore_index=True)


def _territories(stack):
    """The six-plane layout with territory cells."""
    territory, nuclei = hp.nucleus_territories(stack[..., 4], 20, 50)
    stack = stack.copy()
    stack[..., 3], stack[..., 4] = territory, nuclei
    return stack


def test_a_settings_run_reads_the_pzfx_the_database_and_the_infection_report(tmp_path):
    from spacr.tabular import write_table

    merged = tmp_path / 'merged'
    os.makedirs(merged)
    for row in 'BD':
        for k in range(4):
            np.save(merged / f'plate_{row}0{k + 2}_1.npy', _territories(_field(k)))
    _measure_db(tmp_path / 'measurements', ROWS)
    report = pd.DataFrame({'plateID': ['plate'], 'rowID': ['r2'], 'columnID': ['c2'],
                           'metric': ['infection_rate'], 'value': [0.5]})
    write_table(report, str(tmp_path / 'measurements' / 'infection_report.csv'), canonicalise=False)
    pzfx = _pzfx(tmp_path / 'Result.pzfx', _pzfx_table('A (Group 1)', [2.0, 2.2, 2.4, 2.6]),
                 _pzfx_table('B (Group 2)', [1.0, 1.1, 1.2, 1.3]))
    result = hp.analyze_host_pathogen({'src': str(tmp_path), 'translocation': True,
                                       'translocation_groups': 'B:A, D:B',
                                       'translocation_min_nucleus_size': 100,
                                       'translocation_kmax': 3, 'translocation_shuffle_n': 2,
                                       'translocation_reference': pzfx})
    folder = result['folder']
    reference = pd.read_csv(os.path.join(folder, 'hrman_reference.csv'))
    assert len(reference) == 8 and reference['hrman_ratio'].iloc[3] == pytest.approx(1.3)
    curves = pd.read_csv(os.path.join(folder, 'curves_all_readouts.csv'))
    assert 'spacr_db_integrated_by_parasites' in set(curves['readout'])
    assert curves['hrman_ratio'].notna().any()
    agreement = pd.read_csv(os.path.join(folder, 'agreement_with_hrman.csv'))
    assert {'hrman_equiv', 'corrected'} <= set(agreement['readout'])
    assert 'spacr_db_integrated_by_parasites' not in set(agreement['readout'])
    copied = pd.read_csv(os.path.join(folder, 'spacr_infection_report_per_well.csv'))
    pd.testing.assert_frame_equal(copied, report)


def test_without_a_database_there_is_no_measured_curve_or_report(tmp_path):
    merged = tmp_path / 'merged'
    os.makedirs(merged)
    for well, k in (('B02', 0), ('B03', 2)):
        np.save(merged / f'plate_{well}_1.npy', _territories(_field(k)))
    result = hp.nuclear_translocation({'src': str(tmp_path), 'translocation_groups': 'B:A',
                                       'translocation_min_nucleus_size': 100,
                                       'translocation_shuffle_n': 2})
    assert 'spacr_db_integrated_by_parasites' not in set(result['curves']['readout'])
    assert not os.path.exists(os.path.join(result['folder'], 'spacr_infection_report_per_well.csv'))
    assert not os.path.exists(os.path.join(result['folder'], 'hrman_reference.csv'))
