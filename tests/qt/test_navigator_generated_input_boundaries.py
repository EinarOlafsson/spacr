"""Incomplete generated projects cannot gain processing routes from names alone."""
import hashlib

import numpy as np
from PIL import Image

from spacr.qt.widgets import home


def _snapshot(root):
    return {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file() and not path.is_symlink()}


def test_generated_project_skips_hidden_linked_and_incomplete_artifacts(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    merged = project / 'merged'
    masks = project / 'masks'
    models = project / 'model'
    crops = project / 'data' / 'nested' / 'plate_png'
    for folder in (merged, masks, models, crops):
        folder.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    np.save(outside / 'valid.npy', np.zeros((4, 5, 3), dtype='uint16'))
    (merged / 'linked.npy').symlink_to(outside / 'valid.npy')
    (merged / '.hidden.npy').write_bytes((outside / 'valid.npy').read_bytes())
    (merged / 'partial.npy').write_bytes(b'\x93NUMPY')
    (merged / 'notes.txt').write_text('field.npy is still being written')
    np.save(masks / 'floating.npy', np.zeros((4, 5), dtype='float32'))
    (models / 'unfinished.pth').write_bytes(b'incomplete checkpoint')
    (models / 'linked').symlink_to(outside, target_is_directory=True)
    (crops / 'partial.png').write_bytes(b'\x89PNG\r\n\x1a\n')
    Image.fromarray(np.zeros((4, 5), dtype='uint8')).save(project / 'data' / 'unowned.png')
    before = _snapshot(tmp_path)

    def no_pixels(*args, **kwargs):
        raise AssertionError('Navigator must inspect bounded headers, not decode arrays')

    monkeypatch.setattr(np, 'load', no_pixels)
    for selected in (project, merged, models, project / 'data'):
        result = home._easy_inspect_path(str(selected))
        generated = result['context']['generated_outputs']
        assert generated['artifacts'] == []
        assert generated['routes'] == {}
        assert 'measure' not in {row[0] for row in result['actions']}
    assert _snapshot(tmp_path) == before
