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


@pytest.mark.parametrize('batch_size', [2, None], ids=['batched', 'unbatched'])
def test_real_cpu_array_workers_restore_batches_and_retire_after_two_epochs(tmp_path, batch_size):
    """Spawned persistent workers retain native values and expose their real lifecycle."""
    import os
    from pathlib import Path

    script = tmp_path / 'array_worker_contract.py'
    script.write_text('''
import os
from collections import defaultdict, namedtuple

import torch
from torch.utils.data import Dataset, default_collate, default_convert

from spacr.torch_loader import ArrayDataLoader

Row = namedtuple('Row', 'pixels nested name worker_pid')


class Records(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return Row(
            torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + index,
            defaultdict(list, scalar=torch.tensor(index, dtype=torch.bfloat16),
                        pair=[torch.tensor(index, dtype=torch.int64), 'field-' + str(index)]),
            'record-' + str(index), os.getpid())


def assert_same(actual, expected):
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(expected, torch.Tensor):
        assert actual.device.type == 'cpu'
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        assert actual.default_factory is expected.default_factory
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


def main(batch_size):
    torch.set_num_threads(1)
    dataset = Records()
    loader = ArrayDataLoader(dataset, batch_size=batch_size, num_workers=1,
                            persistent_workers=True, multiprocessing_context='spawn')
    assert loader._array_transport is True
    workers = []
    iterator = None
    underlying = None
    try:
        for epoch in range(2):
            iterator = iter(loader)
            assert iter(iterator) is iterator
            assert len(iterator) == len(loader) == (3 if batch_size == 2 else 6)
            worker = iterator._workers[0]
            assert worker.pid != os.getpid() and worker.is_alive()
            if epoch == 0:
                workers.append(worker)
                underlying = iterator._iterator
            else:
                assert worker is workers[0]
                assert iterator._iterator is underlying
            actual_batches = list(iterator)
            records = [dataset[index]._replace(worker_pid=worker.pid) for index in range(6)]
            expected_batches = ([default_collate(records[start:start + 2])
                                 for start in range(0, 6, 2)] if batch_size == 2 else
                                [default_convert(record) for record in records])
            assert len(actual_batches) == len(expected_batches)
            for actual, expected in zip(actual_batches, expected_batches):
                assert_same(actual, expected)
                assert actual.pixels.dtype == actual.nested['scalar'].dtype == torch.bfloat16
                assert actual.pixels.shape == ((2, 2, 3) if batch_size == 2 else (2, 3))
                assert actual.nested['scalar'].shape == ((2,) if batch_size == 2 else ())
            assert worker.is_alive()
    finally:
        if iterator is not None:
            iterator._shutdown_workers()
    assert workers and all(not worker.is_alive() for worker in workers)
    print('two epochs restored; owned worker retired', flush=True)


if __name__ == '__main__':
    main(BATCH_SIZE)
'''.replace('BATCH_SIZE', repr(batch_size)), encoding='utf-8')
    environment = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    environment['PYTHONPATH'] = os.pathsep.join(
        value for value in (root, environment.get('PYTHONPATH', '')) if value)
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=60, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'two epochs restored; owned worker retired' in result.stdout
