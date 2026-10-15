import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tokenize

root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from api_visibility import public_symbol

base = 'ac5afd18d244a633ee6e7ff8d6d3f09e567be0b9'
paths = ['spacr/qt/widgets/hint_bar.py', 'spacr/qt/preferences.py']
old_caption = 'Drag this edge to make the help area taller or shorter.'
new_caption = 'Drag this edge or use Up and Down to resize the help area.'
sha = lambda data: hashlib.sha256(data).hexdigest()
proof = {'tested_private_base': base, 'incoming_public_nightly': '8d23aa1d9', 'files': {}, 'private_methods': []}

class RestoreCaption(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == new_caption:
            node.value = old_caption
        return node

for relative in paths:
    before = subprocess.check_output(['git', 'show', f'{base}:{relative}'])
    after = (root / relative).read_bytes()
    old_ast, new_ast = ast.parse(before), ast.parse(after)
    before_comments = [t.string for t in tokenize.generate_tokens(io.StringIO(before.decode()).readline) if t.type == tokenize.COMMENT]
    after_comments = [t.string for t in tokenize.generate_tokens(io.StringIO(after.decode()).readline) if t.type == tokenize.COMMENT]
    assert before_comments == after_comments
    if relative.endswith('hint_bar.py'):
        old_class = next(n for n in old_ast.body if isinstance(n, ast.ClassDef) and n.name == '_HintResizeHandle')
        new_class = next(n for n in new_ast.body if isinstance(n, ast.ClassDef) and n.name == '_HintResizeHandle')
        old_methods = {n.name: n for n in old_class.body if isinstance(n, ast.FunctionDef)}
        new_methods = {n.name: n for n in new_class.body if isinstance(n, ast.FunctionDef)}
        for name in ('mousePressEvent', 'mouseMoveEvent', 'mouseReleaseEvent'):
            assert ast.dump(old_methods[name]) == ast.dump(new_methods[name])
        for name in sorted(new_methods.keys() - old_methods.keys()):
            node = new_methods[name]
            symbol = 'spacr.qt.widgets.hint_bar._HintResizeHandle.' + name
            included = public_symbol(symbol)
            assert not included
            assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(node) if n is not node)
            proof['private_methods'].append({'symbol': symbol, 'actual_public_symbol_result': included, 'direct_class_method': True, 'nested_functions': 0})
        old_ast.body.remove(old_class)
        new_ast.body.remove(new_class)
        for tree in (old_ast, new_ast):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'PySide6.QtGui')]
    restored = RestoreCaption().visit(copy.deepcopy(new_ast))
    assert ast.dump(old_ast) == ast.dump(restored)
    proof['files'][relative] = {'before_sha256': sha(before), 'after_sha256': sha(after), 'existing_comments_identical': True, 'outside_authorized_scope_ast_identical': True}
proof['api_policy_sha256'] = sha((root / 'tools/api_visibility.py').read_bytes())
proof['public_api_effect'] = 'Five actual policy calls exclude the direct methods of the private class; no nested helper definitions or public docstrings changed.'
print(json.dumps(proof, indent=2))
