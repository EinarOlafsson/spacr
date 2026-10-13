from argparse import Namespace
from pathlib import Path
import hashlib
import json
import build_documentation_i18n as api
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
inventory = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())
languages = tuple(api.MODEL_SPECS)
args = Namespace(device='cpu', batch_size=24, beams=4, threads=2, force=False, sources_only=False,
                 audit=False, audit_english=False, repair_api_blocks=False, rebuild_readme=False,
                 languages=list(languages), model_root=api.default_model_root())
docs = api.public_docstrings()
assert len(docs) == inventory['current_API_symbols'] == 13182
source_sha = hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest()
assert source_sha == inventory['source_sha256']
report = {}
for language in languages:
    reusable = api._reviewed_api_overlay(docs, api.reusable_api_translations(docs, language), language)
    pending = {key: text for key, text in docs.items() if key not in reusable}
    assert set(pending) <= set(inventory['API_added'] + inventory['API_changed']), (language, set(pending))
    translated = dict(reusable)
    if pending:
        translated.update(api._translate_api_documents(pending, language, args.model_root, args, reuse_history=True))
    api.write_language(docs, language, translated)
    report[language] = {'API': len(docs), 'translated': sorted(pending),
                        'normal_generation_functions_match_existing_CLI_path': True}
    print('PASS normal SubCell API generation', language, len(docs), 'pending', len(pending), flush=True)
api._write_english_api_manifest(docs)
assert hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest() == source_sha
(scratch / 'subcell-rybg-API-generation-r2.json').write_text(json.dumps({'languages': report,
    'source_sha256': source_sha, 'all_nine_full_normal_CLI_audits_required_separately': True}, indent=2) + '\n')
