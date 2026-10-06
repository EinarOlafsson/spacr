from pathlib import Path
import hashlib
import json
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
source = scratch / 'deck-current/spacr_overview.pptx'
root = scratch / 'deck-yolo-current-r1'
root.mkdir(exist_ok=True)
specification = {
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'slide_count': 57,
    'slides': [{
        'number': 32,
        'text': {
            'Make Masks: fix masks by hand': 'Make Masks: edit labels and boxes',
            'Correct segmentation by hand, and build training data for your own Cellpose model.': 'Correct instance masks, prepare training pairs, or annotate independent YOLO boxes.',
            'Ten tools: brush, erase, erase object, magic wand + / −, draw, divide, zoom, recrop, ruler': 'Mask tools: brush, erase, magic wand, draw, divide, zoom, recrop and ruler',
            'Invert the image so dark objects can be detected too': 'Add classes; draw, move and resize boxes with Undo and Redo',
            'A log of every edit, so curated masks are told apart from model output': 'Saved box projects and normalized YOLO labels; source images stay unchanged',
            'Make Masks in spaCR 1.5.1.3: tools across the top, settings left, shortcuts right, field actions below': 'Native Make Masks: demonstration boxes on an acquired microscopy field',
        },
        'pictures': [{
            'capture': str(scratch / 'tutorial-make-masks-662-yolo-r2-current/captures/make_masks_yolo_662_r2'),
            'frame': 'yolo_18_reloaded_boxes',
            'description_suffix': 'Current native Make Masks editor on an acquired field',
            'label': 'Native class-labelled demonstration boxes after exact save and reload',
        }],
    }],
}
spec = root / 'refresh-spec.json'
spec.write_text(json.dumps(specification, indent=2) + '\n')
for command in [
    ['tools/refresh_deck_captures.py', str(source), str(spec), str(root / 'spacr_overview.pptx')],
    ['tools/build_readme_deck.py', str(root / 'spacr_overview.pptx'), '--out', str(root / 'rendered')],
    ['tools/build_readme_deck.py', '--refresh-pages', '32', '--rendered', str(root / 'rendered'),
     '--baseline', 'docs/source/_static/deck', '--out', str(root / 'bounded')],
]:
    subprocess.run([sys.executable, *command], check=True)
print('PASS: normal isolated source refresh and deck rendering; only page 32 selected, publication and visual readback remain separate.', flush=True)
