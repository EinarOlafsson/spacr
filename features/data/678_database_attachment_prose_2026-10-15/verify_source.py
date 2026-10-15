"""Verify the executable change is exactly one translated status sentence."""
import ast
import copy
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = 'spacr/qt/widgets/file_list.py'
BASE = '348c63b842a18663a364a746f3b277bcef5efdc8'
OLD = " It is the first row with no database: nothing in the file's path named a plate."
NEW = ('This row was the first without a database. Filename matching '
       'did not attach the database to a plate row.')
old_assignment = ast.parse('message += ' + repr(OLD)).body[0]
new_assignment = ast.parse('message += " " + tr(' + repr(NEW) + ')').body[0]


class Normalize(ast.NodeTransformer):
    caption_count = 0

    def visit_AugAssign(self, node):
        if ast.dump(node) == ast.dump(new_assignment):
            self.caption_count += 1
            return copy.deepcopy(old_assignment)
        return self.generic_visit(node)

    def visit(self, node):
        node = super().visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        return node


before = subprocess.check_output(['git', 'show', f'{BASE}:{SOURCE}'], cwd=ROOT)
normalizer = Normalize()
after = normalizer.visit(ast.parse((ROOT / SOURCE).read_bytes()))
assert normalizer.caption_count == 1
assert ast.dump(Normalize().visit(ast.parse(before))) == ast.dump(after)
print('PASS: executable AST unchanged except the one exact translated caption assignment.')
