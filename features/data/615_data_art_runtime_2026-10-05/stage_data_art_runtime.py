from pathlib import Path
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path('tools').resolve()))
import build_i18n_catalogs as builder
from spacr.qt.night_themes import DATA_ART_THEMES
from spacr.qt.i18n_catalogs import en

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
translations = json.loads((scratch / 'data-art-runtime-reviewed-r1.json').read_text())
themes = list(DATA_ART_THEMES.values())
sources = builder.canonical_sources()
new = set(sources['ui']) - set(en.UI_SOURCES)
expected = {text for theme in themes for text in (theme.label, theme.description)}
assert len(new) == 24 and new == expected
assert set(en.UI_SOURCES) <= set(sources['ui'])
assert set(translations) == set(builder.MODEL_SPECS)
problems = []
reviews = {}
for language, pairs in translations.items():
    assert len(pairs) == len(themes) == 12
    records = []
    builder._REVIEWED_RUNTIME_LOADING.add(language)
    try:
        for theme, pair in zip(themes, pairs):
            assert len(pair) == 2
            for source, target in zip((theme.label, theme.description), pair):
                reasons = builder._translation_rejection_reasons(source, target, language, force=True)
                if reasons:
                    problems.append([language, source, target, reasons])
                contextual = builder._contextualize(target, language, source)
                if contextual != target:
                    problems.append([language, source, target, 'contextualization changes target', contextual])
                records.append({'table': 'ui', 'key': source, 'source': source,
                                'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                'translation': target})
    finally:
        builder._REVIEWED_RUNTIME_LOADING.discard(language)
    reviews[language] = {'schema': 1, 'language': language,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'review_method': 'Direct Codex AI technical translation and review; no native-speaker signoff',
        'note': 'Twelve distinct data-art labels and descriptions from the actual replacement registry. These describe decorative artwork, not measurements or biological simulation. Four descriptions retain cursor interaction. The removed flow registry is not reintroduced; all earlier complete catalog records are preserved.',
        'records': records}
if problems:
    print(json.dumps(problems, ensure_ascii=False, indent=2, default=list))
    raise SystemExit(1)
for language, review in reviews.items():
    path = Path('docs/i18n/reviewed/runtime') / language / '2026-10-05-data-art-replacement.json'
    assert not path.exists()
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    print(language, '24 source-bound reviewed entries staged', flush=True)
