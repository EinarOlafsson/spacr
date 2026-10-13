from pathlib import Path
import json
import sys

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('tools').resolve()))
import build_documentation_i18n as builder
import spacr
assert Path(spacr.__file__).resolve().is_relative_to(Path.cwd())
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
payload = json.loads((scratch / 'data-art-api-reviewed-targets-r1.json').read_text())
docs = builder.public_docstrings()
failed = []
for language, records in payload['languages'].items():
    for record in records:
        symbol, index = record['label'].rsplit('#', 1)
        blocks, _ = builder.translatable_blocks(docs[symbol])
        assert blocks[int(index)] == record['source']
        assert builder._api_translation_source(record['source']) == record['context']
        for authority in ['source', 'context']:
            if not builder._reviewed_api_block_valid(record[authority], record['translation'], language):
                diagnosis = {
                    'language': language, 'label': record['label'], 'authority': authority,
                    'syntax_preserved': builder._syntax_preserved(record[authority], record['translation']),
                    'degenerate': builder._looks_degenerate(record[authority], record['translation'], language),
                    'false_friends': builder._semantic_false_friends(record[authority], record['translation'], language),
                    'model_artifacts': builder._model_artifact_reasons(record[authority], record['translation'], language),
                }
                failed.append(diagnosis)
                print(json.dumps(diagnosis, ensure_ascii=False), flush=True)
    print(language, 'sixteen current-source private targets checked', flush=True)
(scratch / 'data-art-api-review-input-proof.json').write_text(json.dumps({'accepted': not failed, 'failures': failed, 'records_each_language': 16}, ensure_ascii=False, indent=2) + '\n')
assert not failed
print('PASS: all 144 private targets meet existing source/context review gates', flush=True)
