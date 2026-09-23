"""Known host/vacuole identities and explicit missing-data denominators."""
import pandas as pd
import numpy as np
import pytest
from spacr.host_pathogen import summarize_tables


def table(rows):
    return pd.DataFrame([dict(plateID='p1', rowID='A', columnID='1', fieldID='1', **row) for row in rows])


def inputs():
    cells = table([{'object_label': 1}, {'object_label': 2}, {'object_label': 3}])
    vacuoles = table([
        dict(object_label=1, cell_id=1, pathogen_channel_0_mean_intensity=30, pathogen_channel_1_mean_intensity=10),
        dict(object_label=2, cell_id=1, pathogen_channel_0_mean_intensity=10, pathogen_channel_1_mean_intensity=30),
        dict(object_label=3, cell_id=2, pathogen_channel_0_mean_intensity=30, pathogen_channel_1_mean_intensity=30),
        dict(object_label=4, cell_id=2, pathogen_channel_0_mean_intensity=10, pathogen_channel_1_mean_intensity=10)])
    reference = table([dict(object_label=i, cytoplasm_channel_0_mean_intensity=10,
                            cytoplasm_channel_1_mean_intensity=10) for i in (1, 2, 3)])
    settings = dict(hp_marker_channels=[0, 1], hp_marker_thresholds={0: 2, 1: 2})
    return cells, vacuoles, reference, settings


def test_joint_states_and_uninfected_hosts_are_retained():
    cells, vacuoles, ref, settings = inputs()
    out = summarize_tables(cells, vacuoles, ref, settings=settings)
    assert out['cells']['vacuole_count'].tolist() == [2, 2, 0]
    well = out['wells'].iloc[0]
    assert well['host_cells'] == 3 and well['infected_cells'] == 2
    assert well['infection_fraction'] == 2 / 3
    assert well['vacuoles'] == 4 and well['vacuoles_with_counts'] == 0
    assert set(out['vacuoles']['joint_marker_state']) == {
        '0:positive|1:negative', '0:negative|1:positive',
        '0:positive|1:positive', '0:negative|1:negative'}
    assert out['marker_states']['fraction_of_all_vacuoles'].tolist() == [.25] * 4
    assert out['vacuoles']['parasite_count'].isna().all()


def test_direct_counts_use_vacuole_links_not_shared_host():
    cells, vacuoles, ref, settings = inputs()
    parasites = table([dict(object_label=i, pathogen_id=parent, cell_id=host)
                       for i, parent, host in [(1, 1, 1), (2, 2, 1), (3, 2, 1), (4, 99, 2)]])
    out = summarize_tables(cells, vacuoles, ref, parasites, settings=settings)
    assert out['vacuoles']['parasite_count'].tolist() == [1, 2, 0, 0]
    assert len(out['orphan_parasites']) == 1
    assert out['orphan_parasites'].iloc[0]['pathogen_id'] == 99


def test_bad_reference_and_missing_host_are_unknown_not_infinite_or_negative():
    cells, vacuoles, ref, settings = inputs()
    ref.loc[0, 'cytoplasm_channel_0_mean_intensity'] = 0
    vacuoles.loc[3, 'cell_id'] = 99
    out = summarize_tables(cells, vacuoles, ref, settings=settings)
    assert out['vacuoles'].loc[:1, 'channel_0_recruitment_ratio'].isna().all()
    assert out['vacuoles'].loc[3, 'channel_1_state'] == 'unknown'
    assert out['wells'].iloc[0]['linked_vacuoles'] == 3
    assert not np.isinf(out['vacuoles']['channel_0_recruitment_ratio']).any()


def test_duplicate_or_time_blind_identities_are_refused():
    cells, vacuoles, ref, settings = inputs()
    with pytest.raises(ValueError, match='duplicate objects'):
        summarize_tables(pd.concat([cells, cells]), vacuoles, ref, settings=settings)
    vacuoles['timeID'] = 1
    with pytest.raises(ValueError, match='lacks time identities'):
        summarize_tables(cells, vacuoles, ref, settings=settings)


def test_provided_counts_and_missing_counts_remain_distinct():
    cells, vacuoles, ref, settings = inputs()
    vacuoles['count'] = [1, 4, 0, np.nan]
    out = summarize_tables(cells, vacuoles, ref, settings=dict(settings, hp_count_column='count'))
    assert out['wells'].iloc[0]['vacuoles_with_counts'] == 3
    assert out['vacuoles']['parasite_count'].iloc[:3].tolist() == [1, 4, 0]
    assert pd.isna(out['vacuoles']['parasite_count'].iloc[3])


def test_empty_vacuoles_preserve_uninfected_hosts():
    cells, vacuoles, ref, settings = inputs()
    out = summarize_tables(cells, vacuoles.iloc[:0], ref, settings=settings)
    assert out['cells']['infected'].tolist() == [False] * 3
    assert out['wells'].iloc[0]['infection_fraction'] == 0
    assert len(out['marker_states']) == 0


def test_multiple_projects_with_repeated_field_labels_never_collapse(tmp_path):
    import sqlite3
    from spacr.host_pathogen import analyze_host_pathogen
    cells, vacuoles, ref, settings = inputs()
    roots = [tmp_path / 'experiment1', tmp_path / 'experiment2']
    for root in roots:
        (root / 'measurements').mkdir(parents=True)
        with sqlite3.connect(root / 'measurements/measurements.db') as connection:
            for name, frame in [('cell', cells), ('pathogen', vacuoles), ('cytoplasm', ref)]:
                frame.to_sql(name, connection, index=False)
    out = analyze_host_pathogen(dict(settings, src=list(map(str, roots))))
    assert len(out['cells']) == 6 and len(out['vacuoles']) == 8
    assert out['wells']['source_id'].nunique() == 2
    saved = pd.read_csv(roots[0] / 'results/host_pathogen/wells.csv')
    assert saved['host_cells'].tolist() == [3, 3]
    assert (roots[0] / 'results/host_pathogen/settings.json').is_file()


def test_vacuole_overlap_refuses_ties_and_retains_unknown_objects():
    from spacr.host_pathogen import vacuole_links
    child = np.ones((2, 4), dtype=int)
    parent = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    row = vacuole_links(child, parent).iloc[0]
    assert pd.isna(row['pathogen_id']) and row['pathogen_overlap_fraction'] == .5
    parent[:, 2] = 1
    assert vacuole_links(child, parent).iloc[0]['pathogen_id'] == 1
