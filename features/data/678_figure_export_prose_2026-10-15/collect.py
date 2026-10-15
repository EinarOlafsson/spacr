import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n678-figure-prose-20261015')
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_documentation_i18n as docs

paths = ['spacr/qt/widgets/figure_settings.py', 'spacr/qt/widgets/save_figure_dialog.py']
base = '021fd85475e04b4a3041dfd284487f47b26fbc61'

def rendered(path, text):
    module = str(path.with_suffix('')).replace('/', '.')
    tree = ast.parse(text)
    out = {module: docs._clean_doc(tree)}
    out.update(docs._additional_assignment_docs(tree.body, module))
    for node in docs._module_scope_nodes(tree.body):
        if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            key = module + '.' + node.name
            out[key] = docs._autoapi_class_doc(node)
            out.update(docs._additional_assignment_docs(node.body, key))
            for child in docs._module_scope_nodes(node.body):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and docs._is_visible_function_name(child.name, module_is_package=False):
                    out[key + '.' + child.name] = docs._clean_doc(child)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and docs._is_visible_function_name(node.name, module_is_package=False):
            out[module + '.' + node.name] = docs._clean_doc(node)
    return {k: v for k, v in out.items() if v and docs._is_rendered_autoapi_entry(root / path, module, k)}

class BehaviorOnly(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)
    def visit_ImportFrom(self, node):
        if node.module == 'i18n' and all(alias.name == 'tr' for alias in node.names):
            return None
        return self.generic_visit(node)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'tr' and len(node.args) == 1 and not node.keywords:
            return self.visit(node.args[0])
        return self.generic_visit(node)
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.Constant(value='PROSE')
        return node

def constants(text):
    return {node.value for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}

def one(texts, prefix):
    matches = [value for value in texts if value.startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]

ledger = {'item': 'N678', 'base_commit': base, 'claim': 'Bounded factual source corrections only; not whole-item acceptance', 'files': {}, 'api_changes': [], 'runtime_changes': []}
for filename in paths:
    path = Path(filename)
    old = subprocess.check_output(['git', 'show', base + ':' + filename], cwd=root, text=True)
    new = (root / path).read_text()
    compile(new, filename, 'exec')
    equal = ast.dump(BehaviorOnly().visit(ast.parse(old))) == ast.dump(BehaviorOnly().visit(ast.parse(new)))
    assert equal, filename
    before = rendered(path, old)
    after = rendered(path, new)
    assert set(before) == set(after), (filename, 'Unexpected API arrival/removal')
    for key in before:
        if before[key] != after[key]:
            ledger['api_changes'].append({'symbol': key, 'before': before[key], 'after': after[key], 'source_sha256_before': hashlib.sha256(before[key].encode()).hexdigest(), 'source_sha256_after': hashlib.sha256(after[key].encode()).hexdigest()})
    ledger['files'][filename] = {'source_sha256_before': hashlib.sha256(old.encode()).hexdigest(), 'source_sha256_after': hashlib.sha256(new.encode()).hexdigest(), 'rendered_symbols_examined_for_dependencies': len(before), 'behavior_ast_equal_ignoring_prose_and_tr_wrappers': equal}
    if path.name == 'save_figure_dialog.py':
        for old_prefix, new_prefix, evidence in [
            ('PDF and SVG are true vector:', 'PDF and SVG can contain raster images.', 'SaveFigureDialog._format_changed disables DPI; save() still passes the stored DPI to Matplotlib. Matplotlib documents rasterized artists in vector files.'),
            ('Every line in the file: the plotted curves,', 'Line colour affects axes spines', 'style_for_file only calls tick_params(color=...) and spine.set_edgecolor; fast-plot styling uses a separate export_styled protocol.'),
            ('Pen width in pixels for every line.', 'Line width applies only to fast-plot exports,', '_extra_styling includes line_width, but only fast-plot refresh/save call that path; Matplotlib refresh passes no line width.'),
            ('Every piece of text in the file:', 'Text colour affects axes titles,', 'style_for_file recolours axes.title, axis labels/ticks, legend entries and axes.texts; no figure-level text or legend title recolour call.'),
            ('Dots per inch in the written file.', 'Resolution in dots per inch', 'DPI control has range72..1200/default300; there is no journal minimum or display-resolution requirement in the implementation.'),
            ('This figure cannot be previewed, so', 'This figure cannot be previewed. It will', 'save() falls back to _source but still passes dpi, bbox_inches=tight, current facecolor and selected transparency.'),
        ]:
            source = one(constants(new), new_prefix)
            ledger['runtime_changes'].append({'kind': 'factual-error', 'before': one(constants(old), old_prefix), 'after': source, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'code_evidence': evidence})
ledger['retained_scope'] = [
    {'surface': 'FigureSettingsDialog size/DPI/preview controls', 'decision': 'retained', 'evidence': 'Width/height call set_size_inches; DPI calls set_dpi; REDRAW_DELAY_MS remains60; reject retains best-effort snapshot restoration caveat.'},
    {'surface': 'figure_settings graph-style save/load and sidecars', 'decision': 'retained except exact dictionary-filter wording', 'evidence': 'JSON schema validation, optional available data/stats/caption exports, failure isolation and statistics-table correction match implementation.'},
    {'surface': 'derive_replot_recipe', 'decision': 'retained', 'evidence': 'Requires exactly one axes and at least two finite plotted pairs; existing prose already explains summary-artist loss.'},
    {'surface': 'save_figure_dialog source-copy and fast-plot dispatch', 'decision': 'retained', 'evidence': 'Pickle copy for Matplotlib, protocol dispatch for fast plots, cancel/failed-write handling unchanged.'},
]
ledger['removed_inaccurate_comment'] = {'before': 'The formats that hold paths rather than pixels, and so have no resolution.', 'after': None, 'reason': 'Removed the incorrect VECTOR_FORMATS attribute explanation; did not promote it to a new API source. The existing dialog prose now states the format-specific control behavior.'}
ledger['references'] = ['https://matplotlib.org/stable/gallery/misc/rasterization_demo.html', 'https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html']
ledger['remaining_dependencies'] = {'runtime': 'Six source-current reviews in each of nine locales plus coordinated normal runtime generation; stale old review rows must be retired with provenance.', 'api': 'Six rendered symbols need the docs owner batch; no API catalog, help index, or documentation writer run here.', 'publication': 'Rendered/published readback and whole N678 inventory remain outside this bounded source pass.'}
out = root / 'features/data/678_figure_export_prose_2026-10-15'
out.mkdir(parents=True, exist_ok=True)
(out / 'ledger.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n')
assert len(ledger['runtime_changes']) == 6
print('PASS behavior/source scope; changed rendered API symbols:', [row['symbol'] for row in ledger['api_changes']])
print('Runtime source replacements:', len(ledger['runtime_changes']))
