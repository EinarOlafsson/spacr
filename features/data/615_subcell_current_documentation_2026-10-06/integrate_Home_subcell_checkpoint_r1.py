from pathlib import Path
import ast
import re
import subprocess

def nodes(text):
    tree = ast.parse(text)
    return {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
old = nodes(subprocess.check_output(['git', 'show', '58df6f5b9:spacr/embeddings.py'], text=True))
current = nodes(Path('spacr/embeddings.py').read_text())
for name in ('_weights_on_disk', 'encoder_entry', '_retrieval_scorecard'):
    assert current[name] == old[name], name
home = nodes(subprocess.check_output(['git', 'show', '3ce39ac7b:spacr/embeddings.py'], text=True))
for name in set(home) - {'_weights_on_disk', 'encoder_entry', '_retrieval_scorecard'}:
    assert current[name] == home[name], name
note = '\n2026-10-06 workstation private four-channel integration: published Home source 3ce39ac7b is now merged. Both branches complete dated notes are preserved, and all three accepted workstation-owned functions are source-text identical to 58df6f5b9; every other embedding callable/class is source-text identical to Home 3ce39ac7b. Normal API/runtime/Help/settings documentation refresh and four-channel CUDA acceptance are now workstation-owned next actions. Earlier six-model receipts retain their exact source scope. No four-channel GPU, final CI or current-source publication acceptance is claimed yet.\n'
for filename in ('features/325_two_sessions_one_repo_working_protocol.temp', 'features/future/560_foundation_model_embeddings.txt'):
    path = Path(filename)
    text = path.read_text()
    pattern = r'(?m)^<<<<<<< HEAD\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]+\n'
    matches = list(re.finditer(pattern, text, re.S))
    assert len(matches) == 1, filename
    replaced = re.sub(pattern, lambda match: match[2] + match[1], text, flags=re.S)
    assert not re.search(r'(?m)^(<<<<<<<|=======|>>>>>>>)', replaced)
    path.write_text(replaced + note)
print('PASS private Home source integration: both complete note tails retained; all three owned functions and all other Home callables source-text exact')
