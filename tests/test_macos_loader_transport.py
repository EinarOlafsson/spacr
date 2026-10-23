"""Mac worker transport preserves batches and lets the interpreter exit."""

import subprocess
import sys
from collections import defaultdict, namedtuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_array_transport_keeps_nested_values_and_dtypes():
    from torch.utils.data import default_collate

    from spacr.torch_loader import _array_collate, _decode

    row = namedtuple("Row", "pixels label name")
    batch = [row(torch.arange(6).reshape(2, 3) + index, index, str(index)) for index in range(2)]
    expected = default_collate(batch)
    actual = _decode(_array_collate(batch))
    assert type(actual) is type(expected)
    torch.testing.assert_close(actual.pixels, expected.pixels)
    torch.testing.assert_close(actual.label, expected.label)
    assert actual.name == expected.name
    values = [
        defaultdict(
            list,
            image=torch.ones(2, dtype=torch.bfloat16),
            scalar=torch.tensor(1, dtype=torch.bfloat16),
        )
    ] * 2
    actual = _decode(_array_collate(values))
    assert type(actual) is type(default_collate(values))
    torch.testing.assert_close(actual["image"], torch.ones(2, 2, dtype=torch.bfloat16))
    torch.testing.assert_close(actual["scalar"], torch.ones(2, dtype=torch.bfloat16))


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process exit regression")
def test_persistent_tensor_workers_finish_and_exit(tmp_path):
    script = tmp_path / "loader_exit.py"
    script.write_text("""
import torch
from torch.utils.data import TensorDataset
from spacr.resource_log import _parallel_data_loader

if __name__ == '__main__':
    torch.set_num_threads(1)
    dataset = TensorDataset(torch.arange(12).reshape(6, 2), torch.arange(6))
    loader = _parallel_data_loader(dataset, batch_size=2, num_workers=1,
                                  persistent_workers=True)
    for epoch in range(2):
        batches = list(loader)
        torch.testing.assert_close(torch.cat([batch[0] for batch in batches]), dataset.tensors[0])
        torch.testing.assert_close(torch.cat([batch[1] for batch in batches]), dataset.tensors[1])
    print('complete', flush=True)
""")
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "complete" in result.stdout


def test_unbatched_array_transport_retains_tensor_shapes():
    from spacr.torch_loader import _array_convert, _decode

    value = {"image": np.arange(6, dtype="float32").reshape(2, 3), "name": "field"}
    restored = _decode(_array_convert(value))
    torch.testing.assert_close(restored["image"], torch.from_numpy(value["image"]))
    assert restored["name"] == value["name"]
