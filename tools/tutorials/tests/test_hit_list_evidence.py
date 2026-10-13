"""Synthetic guard fixtures only; no synthetic results appear in the lesson."""
import csv
import copy
from pathlib import Path
import sys

import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hit_list_evidence import expected_hits, filtered, check_rows, check_screen, read_rows
from spacr.hits import build_hit_list
from spacr.qt.screens.hit_list import HitListScreen


def save(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def example(tmp_path):
    genes = []
    for gene, effect, p, q in [('100', .5, .01, .03), ('200', -.2, .2, .3), ('300', .05, .6, .6)]:
        genes.append(dict(gene=gene, feature=f'gene_fraction:gene[{gene}]',
            level='gene', multiple_testing_method='fdr_bh', standardized_marginal_effect=effect,
            coefficient=effect, permutation_exceedances=round(p*1000)-1, permutations=999,
            permutation_p_value=p, p_value=p, adjusted_p_value=q, condition='pc' if gene=='100' else 'other'))
    guides = [dict(grna=guide, feature=f'fraction:grna[{guide}]', coefficient=effect,
                   standardized_marginal_effect=effect)
              for guide, effect in [('100_1', .2), ('100_2', .3), ('100_3', -.1), ('200_1', -.2)]]
    save(tmp_path/'results_gene.csv', genes); save(tmp_path/'results_grna.csv', guides)
    return tmp_path, genes, guides


@pytest.fixture
def screen(example):
    app = QApplication.instance() or QApplication([])
    widget = HitListScreen(folder=str(example[0]), threaded=False)
    yield widget
    widget.close(); app.processEvents()


def test_positive_sources_fields_and_real_csv_export(example, tmp_path):
    folder, _, _ = example
    expected = expected_hits(folder); actual = build_hit_list(folder)
    assert check_rows([h.to_dict() for h in actual], expected) == 57
    assert actual.summary()['n_significant'] == 1
    assert expected[0]['agreement'] == 2/3
    target = tmp_path/'export.csv'; actual.write_csv(target)
    assert check_rows(read_rows(target), expected) == 57


@pytest.mark.parametrize('options', [{}, {'max_q':.05}, {'min_guides':2},
    {'min_agreement':1.}, {'min_effect':.1}, {'direction':'up'}, {'direction':'down'},
    {'exclude_controls':True}, {'query':'100'}, {'query':'absent_gene'},
    {'min_guides':2, 'min_agreement':1.}])
def test_independent_filter_matches_actual_backend(example, options):
    folder, _, _ = example
    expected = filtered(expected_hits(folder), **options)
    actual = build_hit_list(folder).filter(**options)
    assert check_rows([h.to_dict() for h in actual], expected) == 19*len(expected)


@pytest.mark.parametrize('kind', ['empty_genes','duplicate_gene','duplicate_guide','guide_effect','family','probability'])
def test_source_guard_follows_positive(example, kind):
    folder, genes, guides = example
    assert len(expected_hits(folder)) == 3
    if kind == 'empty_genes': save(folder/'results_gene.csv', genes); (folder/'results_gene.csv').write_text(','.join(genes[0])+'\n')
    elif kind == 'duplicate_gene': genes.append(dict(genes[0])); save(folder/'results_gene.csv', genes)
    elif kind == 'duplicate_guide': guides.append(dict(guides[0])); save(folder/'results_grna.csv', guides)
    elif kind == 'guide_effect': guides[0]['coefficient']=9.; save(folder/'results_grna.csv', guides)
    elif kind == 'family': genes[0]['level']='guide'; save(folder/'results_gene.csv', genes)
    else: genes[0]['adjusted_p_value']=.9; save(folder/'results_gene.csv', genes)
    with pytest.raises(ValueError): expected_hits(folder)


@pytest.mark.parametrize('kind', ['count','column','effect','q','order','invented_interval'])
def test_row_guard_follows_positive(example, kind):
    folder, _, _ = example
    expected=expected_hits(folder); actual=[h.to_dict() for h in build_hit_list(folder)]
    assert check_rows(actual,expected)==57
    actual=copy.deepcopy(actual)
    if kind=='count':actual.pop()
    elif kind=='column':actual[0]['invented']=1
    elif kind=='effect':actual[0]['effect']=99.
    elif kind=='q':actual[0]['q_value']=.001
    elif kind=='order':actual.reverse()
    else:actual[0]['ci_low']=.4
    with pytest.raises(ValueError):check_rows(actual,expected)


@pytest.mark.parametrize('kind', ['worker','missing','table_count','table_gene','table_text'])
def test_native_screen_guard_follows_positive(example,screen,kind):
    expected=expected_hits(example[0]); assert check_screen(screen,expected)['rows']==3
    if kind=='worker':screen.last_error='actual failure'
    elif kind=='missing':screen._shown=None
    elif kind=='table_count':screen._table.takeTopLevelItem(2)
    elif kind=='table_gene':screen._table.topLevelItem(0).setText(1,'wrong')
    else:screen._table.topLevelItem(0).setText(6,'0.00001')
    with pytest.raises(ValueError):check_screen(screen,expected)
