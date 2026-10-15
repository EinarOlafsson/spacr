import ast, hashlib, importlib.util, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runtime', root / 'tools/build_i18n_catalogs.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
manifest = json.loads((out / 'sources.json').read_text())
packet = json.loads(Path('/mnt/wd4tb/scratch/f560-comparison-contract-20261015/extractor-owner-request.json').read_text())
sources = [r['source'] for r in manifest['sources'] if runtime._looks_translatable(r['source'])]
assert len(sources) == 8 and 'AJI' not in sources
sources += packet['literal_tr_sources']
assert len(sources) == 9
runtime.REVIEWED_RUNTIME_DIR = out / 'combined-inputs'
runtime.canonical_sources = lambda: {'ui': tuple(sources)}
checks = {}
for language in runtime.MODEL_SPECS:
    try:
        result = runtime.reviewed_runtime_translations(language)
        assert len(result) == 9
        checks[language] = {'status': 'PASS', 'count': len(result)}
    except Exception as error:
        checks[language] = {'status': 'FAIL', 'error': str(error)}
        print(language, type(error).__name__, str(error), flush=True)
report = {'scope': 'Actual normal reviewed_runtime_translations schema/source-hash/context/semantic/syntax/script/identity gates over isolated proposed inputs; canonical source argument is the exact scoped AST-owned literal set filtered by normal eligibility. This is not a full canonical inventory or publication acceptance.', 'source_count': len(sources), 'checks': checks}
(out / 'input-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(checks, ensure_ascii=False, indent=2))
sys.exit(any(r['status'] != 'PASS' for r in checks.values()))
