from pathlib import Path
import hashlib
import json
import re
import sys

import cv2
sys.path.insert(0, 'tools/tutorials')
from redact_frame_paths import ocr_lines, path_kinds

stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r2')
identities = ['01_pypi_github', '03_pip_install', '04_platform_installers']
rows = []
seen = {}
species = re.compile(r'plasmodium|candida|trypanosoma|leishmania|giardia|mammalian', re.I)
problems = []
for identity in identities:
    folder = stage / 'production' / identity
    visual = json.loads((folder / 'visual.json').read_text())
    for number, scene in enumerate(visual['scenes'], 1):
        source = (folder / scene['image']).resolve()
        assert source.is_relative_to(stage)
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert digest == scene['capture_sha256']
        if digest not in seen:
            pixels = cv2.imread(str(source))
            assert pixels.shape == (2160, 3840, 3)
            words = ocr_lines(pixels)
            path_hits = [{'kinds': sorted(path_kinds(word['text'])), 'box': word['box']}
                         for word in words if path_kinds(word['text'])]
            alpha_hits = [{'text': word['text'], 'box': word['box']}
                          for word in words if species.search(word['text'])]
            seen[digest] = {'source_sha256': digest, 'native_shape': list(pixels.shape),
                            'full_native_and_overlapping_tile_OCR': True,
                            'OCR_row_count': len(words), 'private_path_hits': path_hits,
                            'alpha_species_hits': alpha_hits}
        result = {'lesson': identity, 'scene': number, 'source': str(source), **seen[digest]}
        rows.append(result)
        if result['private_path_hits'] or result['alpha_species_hits']:
            problems.append(result)
        print(identity, number, 'native frame', digest, 'path hits', len(result['private_path_hits']),
              'alpha species hits', len(result['alpha_species_hits']), flush=True)
report = {'passed': not problems, 'read_only_original_capture_OCR': True,
          'no_pixel_editing': True, 'scenes': rows, 'unique_frames': len(seen), 'problems': problems}
(stage / 'current-native-frame-path-and-alpha-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
assert len(rows) == 30 and not problems, 'Native frame admission failed; review recorded problem rows'
print('PASS: all 30 native installation scenes, no old alpha assay names or private paths detected.', flush=True)
