import ast
import hashlib
import json
import subprocess
from pathlib import Path
ref = 'b0ba43fa18fe644b4f76f660b6f7a0cc584f8b90'
root = Path.cwd()
def data(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
tool_path = 'tools/build_i18n_catalogs.py'
before, after = data(ref + '^', tool_path), data(ref, tool_path)
after_ast = ast.parse(after)
manual = next(n.value for n in after_ast.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == 'MANUAL_UI')
keys = {'F1 @ IoU 0.5', 'Dice', '{metric}: {value:.4f}', 'Precision'}
selected = [(k, v) for k, v in zip(manual.keys, manual.values) if isinstance(k, ast.Constant) and k.value in keys]
assert {k.value for k, v in selected} == keys
kept = [(k, v) for k, v in zip(manual.keys, manual.values) if not (isinstance(k, ast.Constant) and k.value in keys)]
manual.keys, manual.values = [k for k, v in kept], [v for k, v in kept]
assert ast.dump(ast.parse(before), include_attributes=False) == ast.dump(after_ast, include_attributes=False)
identities = []
record_hashes = {}
for language in ['sv', 'de', 'es', 'fr', 'pt', 'is', 'zh_CN', 'hi', 'ko']:
    path = f'docs/i18n/reviewed/runtime/{language}/2026-10-15-f370-scorecard-runtime-home-b.json'
    raw = data(ref, path)
    record_hashes[path] = sha(raw)
    records = json.loads(raw)['records']
    assert len(records) == 8
    for row in records:
        if row['source'] == row['translation']:
            identities.append([language, row['source']])
            assert row['source'] in keys
            assert row['source'] != 'Precision' or language == 'sv'
            assert row['source_sha256'] == sha(row['source'].encode())
assert len(identities) == 28
base = 'features/data/370_560_runtime_inputs_2026-10-15/'
paths = [tool_path, base + 'scoped-runtime.py', base + 'prepared-generation.sh']
receipt = {
    'status': 'PASS: independent source review; no generator/test execution by reviewer',
    'input_commit': ref,
    'file_sha256': {p: sha(data(ref, p)) for p in paths},
    'identity_pairs': identities,
    'review_file_sha256': record_hashes,
    'identity_semantics': 'F1/IoU notation, the named Dice metric and the numeric placeholder-only template remain identical in all nine locales. Precision is the correct Swedish metric name. No complete prose sentence gains a fallback.',
    'scope_proof': 'Removing exactly four MANUAL_UI data entries restores the entire prior tool AST. No candidate, syntax, script, semantic or eligibility gate changed; AJI remains excluded.',
    'writer_review': [
        'Expected owned model_zoo and embeddings byte hashes are required before the actual normal canonical collection. Four source/tool hashes and all eighteen owned review-file hashes are checked again after generation.',
        'One real canonical snapshot is reused by the full normal reviewed_runtime_translations loaders. All nine locales complete preflight before any catalog write; stale or invalid active reviews refuse.',
        'Unowned current sources require exact existing source hashes and retained targets. Missing, stale, removed or conflicting foreign records refuse instead of being rewritten.',
        'No model decoder is used: _translate_batches immediately refuses, and normal write_english/write_language consume only supplied reviewed or retained targets. No API writer, inference or download path is invoked.',
        'After writing, all retained values/source hashes and model/license/normalizer metadata must match the committed baseline; all nine owned UI targets must match their approved inputs.',
        'Normal audit validates English plus all nine locales. A fresh original canonical collection must match the initial snapshot, and source/review bytes must remain unchanged.',
        'Wrapper holds shared heavy.lock, uses run_capped.sh6G, one thread, hidden CUDA and offscreen Qt.'
    ],
    'limits': [
        'No full collector, generator, test or model execution was repeated. This is prepared-source review, not a terminal generation claim.',
        'Coordinator owns actual final catalog preservation/audit readback. Source edits remain prohibited during its live writer.',
        'Initial static reviewer script assumed an unannotated assignment; MANUAL_UI is AnnAssign. The corrected AST-only check above passed without any production edit.'
    ]
}
out = Path('/mnt/wd4tb/scratch/f370-f560-runtime-writer-peer-20261015')
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'result': 'PASS', 'identity_records': len(identities), 'writer_sha256': receipt['file_sha256'][base + 'scoped-runtime.py']}))
