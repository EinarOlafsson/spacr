from pathlib import Path
import json

import build_guide_i18n as guides

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
folder = scratch / 'foundation-puncta-guide-worklists-r1'
folder.mkdir(exist_ok=False)
for language in guides.LANGUAGES:
    path = folder / (language + '.json')
    count = guides.export_worklist(language, path)
    assert count == 12, (language, count)
    catalog = guides.read_catalog(guides.LOCALE_DIR / language / 'LC_MESSAGES/make_masks.po')
    matches = [(message.id, message.string) for message in catalog
               if isinstance(message.id, str) and '``secondary_growth``' in message.id]
    print(language, json.dumps(matches, ensure_ascii=False), flush=True)
print('PASS all nine normal worklists contain exactly twelve new puncta reference messages', flush=True)
