"""A local Darwin policy selects real CPU transport without changing the host."""
import builtins
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import PILToTensor

import spacr.io as images
from spacr.torch_loader import ArrayDataLoader


def test_darwin_loader_policy_preserves_unicode_paths_pixels_and_labels(tmp_path):
    """Exercise policy and real spawn transport, without claiming native Mac behavior."""
    host_platform = sys.platform
    host_loader = images._TorchDataLoader
    host_module = sys.modules['spacr.io']
    source_path = Path(images.__file__)
    scope = {'__name__': 'spacr.io', '__package__': 'spacr', '__file__': str(source_path)}
    real_import = builtins.__import__

    class LocalPlatform:
        platform = 'darwin'

        def __getattr__(self, name):
            return getattr(sys, name)

    platform = LocalPlatform()

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'sys' and level == 0 and globals is scope:
            return platform
        return real_import(name, globals, locals, fromlist, level)

    scope['__builtins__'] = dict(vars(builtins), __import__=local_import)
    exec(compile(source_path.read_text(encoding='utf-8'), str(source_path), 'exec'), scope)
    assert scope['sys'] is platform and scope['_TorchDataLoader'] is ArrayDataLoader
    loader_type = scope['spacrDataLoader']
    assert loader_type.__bases__ == (ArrayDataLoader,)
    assert sys.platform == host_platform and sys.modules['spacr.io'] is host_module
    assert images._TorchDataLoader is host_loader

    paths = [tmp_path / 'first' / 'caf\u00e9.png', tmp_path / 'second' / 'cafe\u0301.png']
    arrays = [np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
              np.arange(60, dtype=np.uint8).reshape(4, 5, 3) + 91]
    for path, pixels in zip(paths, arrays):
        path.parent.mkdir()
        Image.fromarray(pixels).save(path)
    before = {path: path.read_bytes() for path in paths}
    dataset = images.spacrDataset(
        str(tmp_path), ['first', 'second'], transform=PILToTensor(), shuffle=False,
        specific_files=[str(path) for path in paths], specific_labels=[0, 1])
    loader = loader_type(dataset, batch_size=2, shuffle=False, num_workers=1,
                         multiprocessing_context='spawn', pin_memory=False)
    try:
        assert loader._array_transport is True
        batches = list(loader)
        assert len(batches) == 1
        pixels, labels, filenames = batches[0]
        assert pixels.device.type == labels.device.type == 'cpu'
        assert pixels.dtype == torch.uint8
        torch.testing.assert_close(pixels, torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)))
        torch.testing.assert_close(labels, torch.tensor([0, 1], dtype=torch.int64))
        assert list(filenames) == [str(path) for path in paths]
        assert all(path.read_bytes() == content for path, content in before.items())
    finally:
        loader.cleanup()
    assert loader.thread is None or not loader.thread.is_alive()
    assert sys.platform == host_platform and sys.modules['spacr.io'] is host_module
    assert images._TorchDataLoader is host_loader
