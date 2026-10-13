"""Pin the scoped no-edge oracle to positive existing/missing files and raw rows."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_graph_data import verify_graph


@pytest.fixture
def example(tmp_path):
    path = tmp_path/'actual.txt'
    path.write_text('actual artifact')
    row = dict(artifact_id='one', project=str(tmp_path), kind='crops', role='crops',
               module='measure', path=str(path), run_id='real', settings_hash='digest',
               spacr_version='1.5.0.4', created_utc='2026-09-09T00:00:00Z', created_ns=1,
               size_bytes=15, n_files=1, status='complete')
    node = SimpleNamespace(**row, state='current', exists=True, inputs=(), depth=0)
    graph = SimpleNamespace(project=str(tmp_path), nodes=(node,), edges=())
    return graph, row, tmp_path


@pytest.mark.parametrize('field,value', [('role','other'),('kind','other'),('module','other'),
    ('path','/other'),('run_id','invented'),('settings_hash','invented'),('spacr_version','invented'),
    ('created_utc','wrong'),('created_ns',2),('size_bytes',16),('n_files',2),('status','failed'),
    ('project','/different')])
def test_node_provenance_is_exactly_the_original_record(example, field, value):
    graph, row, root = example
    assert verify_graph(graph, root, [row])['nodes'] == 1
    setattr(graph.nodes[0], field, value)
    with pytest.raises(ValueError, match='Node provenance'):
        verify_graph(graph, root, [row])


@pytest.mark.parametrize('field,value', [('state','missing'),('exists',False),('inputs',('fake',)),('depth',1)])
def test_existence_and_no_edge_premise_are_measured(example, field, value):
    graph, row, root = example
    assert verify_graph(graph, root, [row])['states'] == {'one':'current'}
    setattr(graph.nodes[0], field, value)
    with pytest.raises(ValueError, match='Node state'):
        verify_graph(graph, root, [row])


def test_missing_is_a_real_removed_path_not_one_that_never_existed(example):
    graph, row, root = example
    assert verify_graph(graph, root, [row])['states'] == {'one':'current'}
    Path(row['path']).unlink()
    with pytest.raises(ValueError, match='Node state'):
        verify_graph(graph, root, [row])
    graph.nodes[0].state, graph.nodes[0].exists = 'missing', False
    assert verify_graph(graph, root, [row])['states'] == {'one':'missing'}


@pytest.mark.parametrize('change', ['project', 'nodes', 'duplicate', 'edge', 'id'])
def test_graph_identity_matches_exact_sql_ids(example, change):
    graph, row, root = example
    assert verify_graph(graph, root, [row])['edges'] == 0
    if change == 'project': graph.project = '/other'
    elif change == 'nodes': graph.nodes = ()
    elif change == 'duplicate': graph.nodes = graph.nodes*2
    elif change == 'edge': graph.edges = ('invented',)
    else: graph.nodes[0].artifact_id = 'invented'
    with pytest.raises(ValueError, match='Graph identity'):
        verify_graph(graph, root, [row])


def test_oracle_does_not_claim_general_staleness_validation(example):
    graph, row, root = example
    assert verify_graph(graph, root, [row])['nodes'] == 1
    with pytest.raises(ValueError, match='requires complete records without input edges'):
        verify_graph(graph, root, [row], recorded_edges=[('a','b')])
    row['status'] = 'partial'
    with pytest.raises(ValueError, match='requires complete records without input edges'):
        verify_graph(graph, root, [row])


def test_no_registry_has_no_nodes_or_edges(tmp_path):
    graph = SimpleNamespace(project=str(tmp_path), nodes=(), edges=())
    assert verify_graph(graph, tmp_path, [])['states'] == {}
    graph.edges = ('invented',)
    with pytest.raises(ValueError, match='Graph identity'):
        verify_graph(graph, tmp_path, [])
