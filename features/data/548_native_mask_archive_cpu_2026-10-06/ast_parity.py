"""Prove that the native reader changes no downstream scientific generator AST."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ.get('SPACR_PROOF_DIR', Path(__file__).resolve().parent))
before = root / 'before_object.py'
after = Path(os.environ['SPACR_PROOF_REPO']) / 'spacr/object.py'
old_tree, new_tree = (ast.parse(path.read_text()) for path in [before, after])
name = 'generate_cellpose_masks_sam'
old, new = (next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            for tree in [old_tree, new_tree])
old_loop, new_loop = (next(node for node in function.body if isinstance(node, ast.For)
                     and 'enumerate(paths)' in ast.unparse(node.iter)) for function in [old, new])
old_index = next(index for index, node in enumerate(old_loop.body) if isinstance(node, ast.With))
new_index = next(index for index, node in enumerate(new_loop.body) if isinstance(node, ast.With))
assert old_index == new_index
native_scope = new_loop.body[new_index]
assert '_mask_archive_arrays' in ast.unparse(native_scope.items[0].context_expr)
assert len(native_scope.body) == 1 and isinstance(native_scope.body[0], ast.Try)
inner = native_scope.body[0]
assert ast.unparse(inner.finalbody[0]) == 'stack = source_map = None'
body = inner.body
batch_index = next(index for index, node in enumerate(body)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'batch_starts')
old_empty = next(node for node in old_loop.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'len(stack) == 0')
old_batch = next(node for node in old_loop.body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'i')
body[batch_index] = copy.deepcopy(old_empty)
new_batch = body[batch_index + 1]
assert isinstance(new_batch, ast.For) and ast.unparse(new_batch.iter) == 'batch_starts'
new_batch.iter = copy.deepcopy(old_batch.iter)
removed = [node for node in new_batch.body if isinstance(node, ast.Expr)
           and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
           and node.value.func.id == '_release_native_mask_pages']
assert len(removed) == 1
new_batch.body.remove(removed[0])
after_scope = new_loop.body[new_index + 1:]
assert ast.unparse(after_scope[0]) == 'gc.collect()'
body.append(after_scope.pop(0))
new_loop.body = new_loop.body[:new_index] + [copy.deepcopy(old_loop.body[old_index])] + body + after_scope
assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
original_defs = {node.name: node for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
new_defs = {node.name: node for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
assert all(ast.dump(node, include_attributes=False) == ast.dump(new_defs[key], include_attributes=False)
           for key, node in original_defs.items())
result = {'before_sha256': hashlib.sha256(before.read_bytes()).hexdigest(),
          'after_sha256': hashlib.sha256(after.read_bytes()).hexdigest(),
          'downstream_generator_ast_exact_after_undoing_storage_scopes': True,
          'all_other_original_function_and_class_asts_exact': True,
          'intentional_changes': ['private native archive scope and borrowed alias cleanup',
                                  'optional source-page release immediately after unchanged owned np.take',
                                  'empty archive skips batch range but collection/completion follows source closure'],
          'scientific_array_batch_filter_track_model_save_database_statements_changed': 0}
(root / 'ast_parity.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
