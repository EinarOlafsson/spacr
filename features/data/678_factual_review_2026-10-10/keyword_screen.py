"""N678 screen: find promotional, vague or metaphorical wording per surface.

Run from a spacr checkout. Prints counts per surface and writes hits.json.
A hit is a candidate for review, not a defect: every one is read in context.
"""
import ast, json, re, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
PATTERNS = [
    r'\bseamless(ly)?\b', r'\bpowerful\b', r'\beffortless(ly)?\b', r'\bintuitive(ly)?\b',
    r'\bstate[- ]of[- ]the[- ]art\b', r'\bcutting[- ]edge\b', r'\bunlock', r'\bempower',
    r'\bleverag', r'\bworld[- ]class\b', r'\bbest[- ]in[- ]class\b', r'\brevolution',
    r'\bmagic(al(ly)?)?\b', r'\bjourney\b', r'\bstory\b', r'\btells? (its|the|a) story',
    r'\bpixels to answers\b', r'\bbrings? .{0,20}together\b', r'\bat your fingertips\b',
    r'\bwith ease\b', r'\bin no time\b', r'\bbreeze\b', r'\bsimply\b', r'\bjust\b',
    r'\beasily\b', r'\beasy\b', r'\bsupercharg', r'\bgame[- ]chang', r'\bdelight',
    r'\bbeautiful(ly)?\b', r'\bstunning\b', r'\bamazing\b', r'\bawesome\b',
    r'\bthink of (it|this)\b', r'\bimagine\b', r'\bthe road\b', r'\bheart of\b',
    r'\bsuper\b', r'\bsmart\b', r'\brobust(ly)?\b', r'\bhassle', r'\bpain(less)?\b',
    r'\bout of the box\b', r'\bone[- ]stop\b', r'\bblazing', r'\blightning',
    r'\bwhole new\b', r'\bnext[- ]level\b', r'\bcomprehensive\b', r'\bstreamlin',
]
RX = re.compile('|'.join(PATTERNS), re.I)
EXCLUDE = re.compile(r'(make_masks|gate_editor|undo|shortcut|keybind|theme|ambient|data_art|'
                     r'animation|ripple|spaceout|performance|storage|backend_store|duckdb|'
                     r'postgres|parquet|figure_export|figure_integrity|export_figure|i18n_catalogs)', re.I)

def strings_in_python(path):
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except Exception:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                yield 'docstring', node.body[0].lineno if node.body else 1, doc
        if isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', '')) in ('tr', 'translate', 'setToolTip', 'setText', 'setWhatsThis', 'QLabel', 'setStatusTip', 'information', 'warning', 'critical', 'question', 'setPlaceholderText', 'setWindowTitle'):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield 'ui', arg.lineno, arg.value

hits, totals = [], {}
def record(surface, path, line, text):
    totals[surface] = totals.get(surface, 0) + 1
    for m in RX.finditer(text):
        s = max(0, m.start() - 90); e = min(len(text), m.end() + 90)
        hits.append(dict(surface=surface, path=str(path.relative_to(ROOT)), line=line,
                         term=m.group(0).lower(), context=' '.join(text[s:e].split())))

# 1. setting tooltips
from importlib import util
sys.path.insert(0, str(ROOT))
import spacr.settings as S
for name in dir(S):
    obj = getattr(S, name)
    if isinstance(obj, dict) and len(obj) > 200 and all(isinstance(v, str) for v in list(obj.values())[:50]):
        if any(str(v).startswith('(') for v in list(obj.values())[:50]):
            for k, v in obj.items():
                if not EXCLUDE.search(k):
                    record('setting_tooltip', ROOT / 'spacr/settings.py', k, v)
# 2/3. python UI strings and docstrings
for path in sorted((ROOT / 'spacr').rglob('*.py')):
    rel = str(path.relative_to(ROOT))
    if EXCLUDE.search(rel) or '/tests/' in rel:
        continue
    for kind, line, text in strings_in_python(path):
        record('api_docstring' if kind == 'docstring' else 'app_ui', path, line, text)
# 4. docs
for path in [ROOT / 'README.rst', *sorted((ROOT / 'docs/source').rglob('*.rst'))]:
    rel = str(path.relative_to(ROOT))
    if EXCLUDE.search(rel) or '/api/' in rel:
        continue
    for i, para in enumerate(path.read_text(encoding='utf-8').split('\n\n')):
        record('docs', path, i, para)
# 5. tutorial narration and workflow prose
for path in sorted((ROOT / 'tools/tutorials/lessons').glob('*.json')):
    d = json.loads(path.read_text(encoding='utf-8'))
    for key in ('title', 'description'):
        if isinstance(d.get(key), str):
            record('tutorial_text', path, key, d[key])
    for i, sc in enumerate(d.get('scenes', [])):
        if isinstance(sc, dict) and isinstance(sc.get('narration'), str):
            record('tutorial_narration', path, i, sc['narration'])
wf = json.loads((ROOT / 'spacr/resources/module_workflows.json').read_text())
def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, p + '/' + k)
    elif isinstance(o, list):
        for i, v in enumerate(o): walk(v, f'{p}[{i}]')
    elif isinstance(o, str) and len(o) > 40:
        record('workflow_prose', ROOT / 'spacr/resources/module_workflows.json', p, o)
walk(wf)

by = {}
for h in hits:
    by.setdefault(h['surface'], {}).setdefault(h['term'], 0)
    by[h['surface']][h['term']] += 1
print(json.dumps({'texts_screened': totals, 'hits': {s: sum(v.values()) for s, v in by.items()}}, indent=1))
for s, v in by.items():
    print(s, sorted(v.items(), key=lambda x: -x[1])[:15])
Path('/media/carruthers/mnt3/claude/n678/hits.json').write_text(json.dumps(hits, indent=1, ensure_ascii=False))
