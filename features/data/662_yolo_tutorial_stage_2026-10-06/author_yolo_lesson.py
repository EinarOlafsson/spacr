from pathlib import Path
import hashlib
import json
import subprocess

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
path = Path('tools/tutorials/lessons/14_make_masks.json')
before = json.loads(path.read_text())
assert len(before['scenes']) == 45 and not any(scene['visual'].startswith('yolo_') for scene in before['scenes'])
(scratch / 'make-masks-before-yolo-lesson.json').write_bytes(path.read_bytes())
scenes = [
    ('yolo_04_box_controls', 'Choose Box beside Draw to annotate separate YOLO bounding boxes. These boxes have their own classes and history; they do not replace the segmentation mask or change the acquired image. Here the boxes demonstrate interaction, not biological training truth.'),
    ('yolo_05_add_class', 'Click Add class and enter a class name. This example uses demonstration. Select the intended class before drawing; keep class names and their numeric identifiers consistent across your training dataset.'),
    ('yolo_06_drawn_box', 'Drag across the image to draw a box. The outline and class label show the new annotation. For real training data, enclose the target object accurately; this practice rectangle is only a gesture example.'),
    ('yolo_07_moved_box', 'Drag inside an existing box to move it. The box keeps its class while its image coordinates change. Inspect its position before saving.'),
    ('yolo_08_resized_box', 'Drag a corner of the box to resize it. Place the edges around the intended object and check that the rectangle remains inside the image.'),
    ('yolo_09_contained_box', 'Hold Control while dragging to draw another box inside an existing one. This creates a separate annotation rather than moving the outer box. Both boxes keep their own coordinates and class.'),
    ('yolo_10_deleted_box', 'Right-click a box to delete it. Check which outline disappears, especially when boxes overlap. Removing a box does not erase pixels from the segmentation mask.'),
    ('yolo_11_undo_boxes', 'Click Undo to restore the deleted box. The two independent annotations return. Use history to recover a mistaken edit before saving.'),
    ('yolo_12_redo_boxes', 'Redo applies the deletion again. Undo once more to keep both demonstration boxes for the export. Review the final annotations before writing them to disk.'),
    ('yolo_13_saved_boxes', 'Click Save boxes to store the annotations in the project file beside the images. This saves box coordinates and classes separately from the mask. Keep the project file with its corresponding source images.'),
    ('yolo_14_export_picker', 'Click Export YOLO labels and choose the text-file destination. Each row contains a class identifier and normalized center X, center Y, width and height relative to the full image. The class-name companion preserves the identifier mapping.'),
    ('yolo_17_negative_exported', 'A field with no annotated boxes exports an empty label file. This second microscopy field illustrates that file format, not a biological negative. Use negative training examples only after checking that no target objects are present.'),
    ('yolo_18_reloaded_boxes', 'Return to the first image. The saved boxes reload with the same classes and coordinates, while the acquired image and segmentation mask remain unchanged. Keep the YOLO labels, class mapping and source images together for training.'),
]
frames = json.loads((scratch / 'tutorial-make-masks-662-yolo-r2-current/captures/make_masks_yolo_662_r2/frames.json').read_text())
new = []
for visual, narration in scenes:
    assert visual in frames
    scene = {'visual': visual, 'narration': narration, 'hold_after': 0.7}
    if visual in ('yolo_05_add_class', 'yolo_14_export_picker'):
        dialogs = [row['rect'] for row in frames[visual]['dialogs'] if row.get('rect')]
        assert len(dialogs) == 1
        scene['focus'] = dialogs[0]
    elif visual == 'yolo_04_box_controls':
        scene['focus'] = [1060, 150, 820, 130]
    elif visual != 'yolo_17_negative_exported':
        scene['focus'] = [2030, 760, 1050, 880]
    new.append(scene)
index = next(i for i, scene in enumerate(before['scenes']) if scene['visual'] == '11_saved_restored_mask') + 1
after = dict(before)
after['scenes'] = before['scenes'][:index] + new + before['scenes'][index:]
after['description'] += ' Draw, edit and export separate class-labelled YOLO boxes without changing the image or segmentation mask.'
after['objectives'] = before['objectives'] + ['Draw and edit separate class-labelled boxes, save their project and export normalized YOLO labels.']
assert [scene for scene in after['scenes'] if not scene['visual'].startswith('yolo_')] == before['scenes']
path.write_text(json.dumps(after, ensure_ascii=False, indent=2) + '\n')
proof = {'previous_english_sha256': hashlib.sha256(json.dumps(before, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
         'current_english_sha256': hashlib.sha256(json.dumps(after, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
         'previous_scene_objects_preserved': 45, 'new_yolo_scenes': 13,
         'insert_index': index, 'new_description_sentence': after['description'][len(before['description']):].strip(),
         'new_objective': after['objectives'][-1],
         'mask_07_sha256': hashlib.sha256(Path('tools/tutorials/lessons/07_mask.json').read_bytes()).hexdigest(),
         'published': False}
(scratch / 'make-masks-yolo-authoring-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print('Authored thirteen YOLO scenes, preserved all forty-five existing complete scenes; not published', flush=True)
