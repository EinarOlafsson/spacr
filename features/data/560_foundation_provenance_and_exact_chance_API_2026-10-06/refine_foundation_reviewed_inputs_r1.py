from pathlib import Path
import json

import build_i18n_catalogs as runtime
import write_reviewed_api_record as writer

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
document = json.loads((scratch / 'foundation-api-reviewed-inputs-r1.json').read_text())
source = 'No scorecard. Measure retrieval against labelled phenotype controls and attach the results.'
translation = 'Inget utvärderingskort. Mät sökprestanda mot märkta fenotypkontroller och bifoga resultaten.'
entry = writer.runtime_record('ui', source, translation)
writer.write('sv', '2026-10-06-encoder-provenance', [entry])
assert runtime.reviewed_runtime_translations('sv')[source] == translation
rows = document['languages']['sv']['runtime']
index = next(i for i, row in enumerate(rows) if row['source'] == source)
rows[index] = entry
document['Swedish_retrieval_technical_sense_refined_before_generation'] = True
(scratch / 'foundation-api-reviewed-inputs-r2.json').write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
print('PASS Swedish retrieval describes search performance; all other directly reviewed inputs retained', flush=True)
