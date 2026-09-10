"""Read a real post-cleanup registry on a private copy, without inventing edges."""
from pathlib import Path
import os
import tempfile
import sqlite3

from capture_report import copy_private_source, snapshot_source, require_unchanged
from manager_data import SOURCE_RELATIVE, registry_rows, verify_bind, verify_original


def prepare(stage):
    stage = Path(stage).resolve()
    if os.environ.get('SPACR_ARTIFACTS_DB'):
        raise ValueError('Pipeline capture requires the actual local registry')
    source = stage / SOURCE_RELATIVE
    archive = stage/'data_manager_runs/real-project-c7j1xuha/archive'
    conversion = stage/'batch_runs/real-channels-u0zjboui/converted_01'
    parent = stage/'pipeline_graph_runs'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='actual-cleanup-', dir=parent))
    clone, original = root/'project', root/'original_readonly'
    original.mkdir()
    archived_files, copied = copy_private_source(archive, clone)
    conversion_files, conversion_copied = copy_private_source(conversion, root/'converted_images')
    rows = registry_rows(archive)
    selected = [r for r in rows if r['project'] == str(source)]
    with sqlite3.connect((archive/'artifacts.db').as_uri()+'?mode=ro&immutable=1', uri=True) as con:
        edges = con.execute('SELECT * FROM artifact_inputs').fetchall()
    if (len(rows) != 4 or len(selected) != 3 or edges or
            {r['kind'] for r in selected} != {'measurements-db', 'resource-log', 'crops'} or
            any(r['status'] != 'complete' for r in selected) or
            (clone/'data').exists() or not (clone/'measurements/measurements.db').is_file()):
        raise ValueError('Real Data Manager cleanup no longer matches the scoped example')
    return dict(source=str(source), root=str(root), clone=str(clone), original_readonly=str(original),
                source_files=snapshot_source(source), clone_files=copied,
                archive=str(archive), archive_files=archived_files, all_rows=rows, artifact_rows=selected,
                conversion_original=str(conversion), conversion_copy=str(root/'converted_images'),
                conversion_files=conversion_files, conversion_copied=conversion_copied,
                recorded_edges=edges, synthetic_artifacts=False, registry_rewritten=False)


def verify_graph(graph, root, rows, *, recorded_edges=()):
    """Exact scoped records, recorded sizes and live path presence, not a generic DAG oracle."""
    expected = {r['artifact_id']: r for r in rows}
    if (recorded_edges or any(r['status'] != 'complete' for r in rows)):
        raise ValueError('This independent oracle requires complete records without input edges')
    if (graph.project != str(root) or len(graph.nodes) != len(expected) or
            {n.artifact_id for n in graph.nodes} != set(expected) or graph.edges):
        raise ValueError('Graph identity or recorded edges differ from raw SQL')
    fields = ('artifact_id', 'project', 'kind', 'role', 'module', 'path', 'run_id',
              'settings_hash', 'spacr_version', 'created_utc', 'created_ns',
              'size_bytes', 'n_files', 'status')
    states = {}
    for node in graph.nodes:
        row = expected[node.artifact_id]
        if any(getattr(node, field) != row[field] for field in fields):
            raise ValueError('Node provenance differs from its unchanged original SQL record')
        exists = Path(row['path']).exists()
        state = 'current' if exists else 'missing'
        if node.exists != exists or node.state != state or node.inputs or node.depth != 0:
            raise ValueError('Node state, existence or dependency depth differs from this scoped evidence')
        states[node.artifact_id] = state
    return {'nodes': len(expected), 'edges': 0, 'states': states,
            'sizes_are_recorded_not_a_fresh_measurement': True}


def verify_originals(inputs):
    result = {'original_project': verify_original(inputs)}
    for path, before in [('archive', 'archive_files'), ('conversion_original', 'conversion_files'),
                         ('conversion_copy', 'conversion_copied')]:
        require_unchanged(inputs[before], snapshot_source(inputs[path]))
        result[path] = {'files': len(inputs[before]), 'byte_identical': True}
    if registry_rows(verify_bind(inputs)) != inputs['all_rows']:
        raise ValueError('Read-only graph inspection changed registry records')
    return result
