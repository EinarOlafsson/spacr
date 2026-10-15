"""Replay the alignment helper's branches with a minimal layout protocol."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SOURCE = 'spacr/qt/widgets/column_aligned_row.py'
BASE = '93474c76ea0'


class Layout:
    def __init__(self, header, strip, on_invalidate=None):
        self.header = header
        self.strip = strip
        self.callback = on_invalidate
        self.items = []
        self.activations = 0

    def add_over_column(self, widget, column):
        self.items.append((widget, column))

    def activate(self):
        self.activations += 1


class Strip:
    def __init__(self, layout=None, alive=True):
        self.current = layout
        self.alive = alive

    def layout(self):
        return self.current


def run(source):
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'align_row_to_columns')
    namespace = {'ColumnAlignedRow': Layout,
                 '_alive': lambda obj: obj is not None and getattr(obj, 'alive', True)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), SOURCE, 'exec'), namespace)
    align = namespace['align_row_to_columns']
    old_header, new_header = object(), object()
    old_callback, new_callback = object(), object()
    strip = Strip()
    existing = Layout(old_header, strip, on_invalidate=old_callback)
    existing.items = [('old widget', 1)]
    strip.current = existing
    result = align(strip, new_header, [('new widget', 2)], on_invalidate=new_callback)
    assert result is existing and result.header is old_header
    assert result.callback is old_callback and result.items == [('old widget', 1)]
    assert result.activations == 0
    assert align(None, new_header, []) is None
    assert align(Strip(), None, []) is None
    assert align(Strip(alive=False), new_header, []) is None
    assert align(Strip(), SimpleNamespace(alive=False), []) is None
    empty = align(Strip(), new_header, [], on_invalidate=new_callback)
    assert isinstance(empty, Layout) and empty.items == [] and empty.activations == 1
    return {'existing_layout_reused_with_different_header_columns_callback': True,
            'missing_strip_returns_none': True, 'missing_header_returns_none': True,
            'dead_strip_returns_none': True, 'dead_header_returns_none': True,
            'empty_columns_returns_new_layout': True}


class WithoutDocs(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        return node


base = subprocess.check_output(['git', 'rev-parse', BASE], cwd=ROOT, text=True).strip()
before = subprocess.check_output(['git', 'show', f'{base}:{SOURCE}'], cwd=ROOT)
after = (ROOT / SOURCE).read_bytes()
assert ast.dump(WithoutDocs().visit(ast.parse(before))) == ast.dump(WithoutDocs().visit(ast.parse(after)))
assert run(before) == run(after)
before_owner = next(node for node in ast.parse(before).body if isinstance(node, ast.FunctionDef) and node.name == 'align_row_to_columns')
after_owner = next(node for node in ast.parse(after).body if isinstance(node, ast.FunctionDef) and node.name == 'align_row_to_columns')
print(json.dumps({
    'date': '2026-10-15', 'base_commit': base, 'source': SOURCE,
    'before_sha256': hashlib.sha256(before).hexdigest(),
    'after_sha256': hashlib.sha256(after).hexdigest(),
    'symbol': 'spacr.qt.widgets.column_aligned_row.align_row_to_columns',
    'before': ast.get_docstring(before_owner), 'after': ast.get_docstring(after_owner),
    'executable_ast_identical': True, 'same_before_after_branches': run(after),
    'limitations': ['Only this one function docstring reviewed; not full N678 completion.',
                    'Actual source function executed against a minimal protocol; no Qt/native layout acceptance inferred.',
                    'Normal API generation and nine-locale source-bound translation review remain with the API lane.',
                    'No runtime caption, layout behavior, public signature or scientific behavior changed.'],
}, indent=2))
