from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

from PIL import Image
from pypdf import PdfReader

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / 'deck-yolo-current-r1'
baseline = Path('docs/source/_static/deck')
candidate = root / 'bounded'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before, after = read(baseline / 'slides.json'), read(candidate / 'slides.json')
assert before['count'] == after['count'] == 57
old_pdf, new_pdf = PdfReader(baseline / 'spacr_deck.pdf'), PdfReader(candidate / 'spacr_deck.pdf')
assert len(old_pdf.pages) == len(new_pdf.pages) == 57
for i, (old, new) in enumerate(zip(before['slides'], after['slides']), 1):
    assert old['animations'] == new['animations']
    assert tuple(old_pdf.pages[i-1].mediabox) == tuple(new_pdf.pages[i-1].mediabox)
    if i != 32:
        assert old == new
        assert old_pdf.pages[i-1].get_contents().get_data() == new_pdf.pages[i-1].get_contents().get_data()
        assert old_pdf.pages[i-1].extract_text() == new_pdf.pages[i-1].extract_text()
    for key in ('image', 'thumb'):
        if i != 32:
            assert (baseline / old[key]).read_bytes() == (candidate / new[key]).read_bytes()
        with Image.open(candidate / new[key]) as image:
            assert image.size == ((3200,1800) if key == 'image' else (320,180))
    assert (candidate / f'pages/{i:02d}.md').is_file()
for path in (baseline / 'anim').glob('*'):
    assert path.read_bytes() == (candidate / 'anim' / path.name).read_bytes()
text = new_pdf.pages[31].extract_text()
assert 'YOLO' in text and 'source images stay unchanged' in text
receipt = {'accepted': True, 'slide_count': 57, 'refreshed_pages': [32],
           'unchanged_slide_assets': 56, 'unchanged_vector_page_content_and_text': 56,
           'all_slide_and_thumbnail_dimensions_verified': True,
           'all_57_navigation_pages_verified': True, 'animation_files_and_placement_unchanged': True,
           'visual_reviewed_pages': [32], 'native_pixels_preserved': True,
           'alpha_page_retained_exactly': True, 'published': False}
(root / 'preservation.json').write_text(json.dumps(receipt, indent=2) + '\n')
dest = Path('features/data/662_yolo_deck_refresh_2026-10-06')
dest.mkdir(exist_ok=True)
for source in (root / 'refresh-spec.json', root / 'spacr_overview.refresh.json', root / 'preservation.json', Path(__file__), scratch / 'refresh_yolo_deck.py'):
    shutil.copyfile(source, dest / source.name)
for name in ('yolo-deck-refresh-r1.log', 'yolo-deck-alpha-r1.log', 'yolo-deck-bounded-r2.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
with zipfile.ZipFile(root / 'spacr_overview.pptx') as source, zipfile.ZipFile(dest / 'source-delta.zip', 'w', zipfile.ZIP_DEFLATED) as target:
    for name in read(root / 'spacr_overview.refresh.json')['changed_members']:
        target.writestr(name, source.read(name))
shutil.copytree(candidate, baseline, dirs_exist_ok=True)
receipt['published_repository_artifacts'] = True
receipt['source_refresh'] = read(root / 'spacr_overview.refresh.json')
receipt['artifacts'] = {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}
Path('features/data/662_yolo_deck_refresh_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation YOLO presentation acceptance: page 32 now describes independent class-labelled boxes, history, project saving and normalized YOLO export beside mask/puncta editing, with the genuine acquired-field save/reload screenshot. Normal source refresh, normal full-deck rendering and bounded page publication pass after the required normal alpha-page refresh; the first geometry refusal is retained, not waived. All other 56 vector page contents/text, slide/thumbnail pixels and animation files/placement are exact, including the alpha page. All 57 page routes and dimensions pass, and page 32 was visually inspected for readable, unclipped text and real box overlays. Receipt 662_yolo_deck_refresh_2026-10-06.json archives the source delta, strict provenance/preservation and completed normal logs. The video candidate independently passes all 85 playback routes and both mutation guards; immutable media upload/deployed readback remain pending. Mask 07 and current Home/Measure/Conda corrections stay unchanged.\n'
for path in ('features/new/662_make_masks_yolo_bounding_box_annotations.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: page 32 accepted and normal generated repository artifacts updated; 56 unrelated pages remain exact.', flush=True)
