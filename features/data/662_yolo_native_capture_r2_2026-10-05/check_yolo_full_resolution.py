from pathlib import Path
import json

import numpy as np
from PIL import Image

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
roots = [
 scratch / 'tutorial-make-masks-662-yolo-r2-current/captures/make_masks_yolo_662_r2',
 scratch / 'tutorial-make-masks-662-yolo-r5-desktop/captures/make_masks_yolo_662_r5_desktop',
 scratch / 'tutorial-make-masks-662-yolo-r5-desktop/captures/make_masks_yolo_662_r6_desktop_repaint',
]
proof = {}
for root in roots:
    native = json.loads((root / 'yolo_acceptance.json').read_text())
    states = {'yolo_06_drawn_box': [native['drawn']],
              'yolo_07_moved_box': [native['moved']],
              'yolo_08_resized_box': [native['resized']],
              'yolo_09_contained_box': native['boxes'],
              'yolo_10_deleted_box': native['boxes'][:1],
              'yolo_11_undo_boxes': native['boxes'],
              'yolo_12_redo_boxes': native['boxes'][:1],
              'yolo_13_saved_boxes': native['boxes'],
              'yolo_15_exported': native['boxes'],
              'yolo_18_reloaded_boxes': native['boxes']}
    report = {}
    for name, boxes in states.items():
        pixels = np.array(Image.open(root / (name + '.png')).convert('RGB'))
        assert pixels.shape == (2160, 3840, 3)
        edges = []
        for _, x0, y0, x1, y1 in boxes:
            x0, x1 = [1573 + round(value / 1994 * 1772) for value in (x0, x1)]
            y0, y1 = [266 + round(value / 1994 * 1772) for value in (y0, y1)]
            lines = [pixels[y0-3:y0+4, x0+5:x1-5].transpose(1, 0, 2),
                     pixels[y1-3:y1+4, x0+5:x1-5].transpose(1, 0, 2),
                     pixels[y0+5:y1-5, x0-3:x0+4],
                     pixels[y0+5:y1-5, x1-3:x1+4]]
            for line in lines:
                rgb = line.astype(int)
                blue = (rgb[:,:,2] > 170) & (rgb[:,:,0] < 140) & (rgb[:,:,1] > 70) & (rgb[:,:,1] < 225) & (rgb[:,:,2] > rgb[:,:,0] + 60)
                coverage = float(np.any(blue, axis=1).mean())
                assert coverage > .98, (root.name, name, coverage)
                edges.append(coverage)
        report[name] = {'expected_boxes': len(boxes), 'full_native_edge_blue_coverage': edges}
        print(root.name, name, 'all full-resolution box edges present', flush=True)
    proof[root.name] = report
(scratch / 'yolo-full-resolution-proof.json').write_text(json.dumps({'accepted': True, 'captures': proof,
 'finding': 'Thin native two-pixel outlines are present. The resized image previews obscured some outlines; no application or recorder defect is proved. Recording procedure changes have been removed.'}, indent=2) + '\n')
print('PASS: all thirty native full-resolution states contain every expected box edge', flush=True)
