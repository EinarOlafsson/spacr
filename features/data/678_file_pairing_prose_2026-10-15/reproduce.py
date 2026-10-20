"""Reproduce pairing behavior from the actual pure helpers, without Qt imports."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RELATIVE_SOURCE = 'spacr/qt/widgets/file_list.py'
BASE = '348c63b842a18663a364a746f3b277bcef5efdc8'
NAMES = {'_pair_tokens', '_database_tokens', '_plate_label', '_best_unique',
         'suggest_file_pairs', '_number_unlabelled_plates', '_attach_databases'}
CHANGED = {'suggest_file_pairs', '_number_unlabelled_plates'}


def execute_helpers(source):
    parsed = ast.parse(source)
    body = [node for node in parsed.body if
            (isinstance(node, ast.FunctionDef) and node.name in NAMES) or
            (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '_DATABASE_GENERIC'
                for target in node.targets))]
    namespace = {'os': os, 're': re}
    exec(compile(ast.Module(body=body, type_ignores=[]), RELATIVE_SOURCE, 'exec'), namespace)
    propose = namespace['suggest_file_pairs']
    tied = propose(['plate1_scores.csv'],
                   ['/first/plate1_counts.csv', '/second/plate1_counts.csv'])
    assert tied[0]['count'] == '/first/plate1_counts.csv'
    assert tied[1]['count'] == '/second/plate1_counts.csv'
    assert tied[0]['plate'] == 'plate 1' and tied[1]['plate'] == ''
    unmatched = propose(['alpha_scores.csv', 'beta_scores.csv'],
                        ['gamma_counts.csv', 'delta_counts.csv'])
    assert [row['count'] for row in unmatched] == ['gamma_counts.csv', 'delta_counts.csv']
    assert [row['plate'] for row in unmatched] == ['plate 1', 'plate 2']
    incomplete = {
        'score_only': propose(['scores.csv'], []),
        'count_only': propose([], ['counts.csv']),
        'database_only': propose([], [], databases=['measurements.db']),
    }
    assert all(rows[0]['plate'] == '' for rows in incomplete.values())
    database_tie = propose(['plate1_scores.csv'], ['plate1_counts.csv'],
                           databases=['/first/plate1/measurements.db',
                                      '/second/plate1/measurements.db'])
    assert database_tie[0]['database'] is None
    assert len(database_tie) == 3
    numbered = namespace['_number_unlabelled_plates']([
        {'score': 'score.csv', 'count': None, 'plate': ''},
        {'score': 'score.csv', 'count': 'count.csv', 'plate': 'plate7'},
        {'score': 'score.csv', 'count': 'count.csv', 'plate': ''},
    ])
    assert [row['plate'] for row in numbered] == ['', 'plate7', 'plate 3']
    return {'count_tie_fallback': tied, 'no_overlap_fallback': unmatched,
            'incomplete_labels': incomplete, 'database_tie': database_tie,
            'row_position_labels': numbered}


class WithoutDocs(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        return node


before = subprocess.check_output(['git', 'show', f'{BASE}:{RELATIVE_SOURCE}'], cwd=ROOT)
after = (ROOT / RELATIVE_SOURCE).read_bytes()
assert ast.dump(WithoutDocs().visit(ast.parse(before))) == ast.dump(WithoutDocs().visit(ast.parse(after)))
before_cases = execute_helpers(before)
after_cases = execute_helpers(after)
assert before_cases == after_cases
old_nodes = {node.name: node for node in ast.parse(before).body if isinstance(node, ast.FunctionDef)}
new_nodes = {node.name: node for node in ast.parse(after).body if isinstance(node, ast.FunctionDef)}
changed = {name for name in old_nodes if ast.get_docstring(old_nodes[name]) != ast.get_docstring(new_nodes[name])}
assert changed == CHANGED
ledger = [{
    'symbol': f'spacr.qt.widgets.file_list.{name}',
    'before': ast.get_docstring(old_nodes[name]),
    'after': ast.get_docstring(new_nodes[name]),
    'finding': ('Counts use input-order fallback after unmatched or tied token selection; databases do not.'
                if name == 'suggest_file_pairs' else
                'Generated labels require both score and count; existing labels persist and numbering uses the row position.'),
} for name in sorted(CHANGED)]
print(json.dumps({
    'date': '2026-10-15', 'base_commit': BASE, 'source': RELATIVE_SOURCE,
    'before_sha256': hashlib.sha256(before).hexdigest(),
    'after_sha256': hashlib.sha256(after).hexdigest(),
    'executable_ast_identical': True, 'before_after_cases_identical': True,
    'cases': after_cases, 'ledger': ledger,
    'public_api_dependencies': ['spacr.qt.widgets.file_list.suggest_file_pairs'],
    'private_docstrings_changed': ['spacr.qt.widgets.file_list._number_unlabelled_plates'],
    'limitations': ['Only two selected pairing-helper docstrings reviewed; not full N678 completion.',
                    'No Qt/native/whole-widget acceptance claimed by pure-helper execution.',
                    'Normal English API refresh and nine-locale paragraph review remain with the API lane.',
                    'No runtime behavior, scientific algorithm, visible caption or command changed.'],
}, indent=2))
