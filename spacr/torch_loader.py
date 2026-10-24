"""Transport macOS loader batches without PyTorch shared-memory daemons.

On Intel macOS, torch 2.2's file-system sharing helpers inherit Python's
resource-tracker pipe. Python then waits for that tracker while the helpers
wait for Python to exit. Default collation sends NumPy-backed records instead;
the parent restores the same CPU tensor shapes and dtypes before yielding.
"""

import copy
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map, default_convert


@dataclass
class _ArrayTensor:
    """Carry NumPy tensor data with metadata needed to restore bfloat16 shape."""
    array: object
    bfloat16: bool = False
    shape: tuple = ()


def _encode(value):
    """Recursively replace tensors with CPU NumPy records, preserving bfloat16 bits."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        bfloat16 = value.dtype == torch.bfloat16
        shape = tuple(value.shape)
        if bfloat16:
            value = value.reshape(-1).view(torch.uint8)
        return _ArrayTensor(value.numpy(), bfloat16, shape)
    if isinstance(value, dict):
        result = copy.copy(value)
        result.update((key, _encode(item)) for key, item in value.items())
        return result
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_encode(item) for item in value))
    if isinstance(value, (tuple, list)):
        return type(value)(_encode(item) for item in value)
    return value


def _decode(value):
    """Recursively restore encoded NumPy records to CPU tensors."""
    if isinstance(value, _ArrayTensor):
        tensor = torch.from_numpy(value.array)
        return tensor.view(torch.bfloat16).reshape(value.shape) if value.bfloat16 else tensor
    if isinstance(value, dict):
        result = copy.copy(value)
        result.update((key, _decode(item)) for key, item in value.items())
        return result
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_decode(item) for item in value))
    if isinstance(value, (tuple, list)):
        return type(value)(_decode(item) for item in value)
    return value


def _tensor_collate(batch, *, collate_fn_map=None):
    """Stack tensors locally and encode the batch without shared tensor storage."""
    return _encode(torch.stack(batch))


def _array_collate(batch):
    """Apply default collation with local tensor stacking, then encode the result."""
    functions = dict(default_collate_fn_map)
    functions[torch.Tensor] = _tensor_collate
    return _encode(collate(batch, collate_fn_map=functions))


def _array_convert(item):
    """Apply default single-item conversion and encode any resulting tensors."""
    return _encode(default_convert(item))


class _RestoringIterator:
    """Restore encoded CPU tensor batches while delegating iterator attributes."""
    def __init__(self, iterator):
        """Store the underlying iterator that yields encoded batches."""
        self._iterator = iterator

    def __iter__(self):
        """Return this restoring iterator."""
        return self

    def __next__(self):
        """Read and decode the next batch from the underlying iterator."""
        return _decode(next(self._iterator))

    def __len__(self):
        """Return the underlying iterator length."""
        return len(self._iterator)

    def __getattr__(self, name):
        """Return the corresponding attribute from the underlying iterator."""
        return getattr(self._iterator, name)


class ArrayDataLoader(DataLoader):
    """Default CPU batches cross process boundaries as NumPy, then restore.

    Explicit custom collators retain their existing behavior. Single-process
    loaders use ordinary PyTorch collation and iteration unchanged.
    """

    def __init__(self, *args, **kwargs):
        """Use NumPy transport only for worker loaders without a custom collator."""
        workers = kwargs.get("num_workers", args[5] if len(args) > 5 else 0)
        function = kwargs.get("collate_fn", args[6] if len(args) > 6 else None)
        self._array_transport = bool(workers and function is None)
        if self._array_transport:
            size = kwargs.get("batch_size", args[1] if len(args) > 1 else 1)
            sampler = kwargs.get("batch_sampler", args[4] if len(args) > 4 else None)
            function = _array_convert if size is None and sampler is None else _array_collate
            if len(args) > 6:
                args = (*args[:6], function, *args[7:])
            else:
                kwargs["collate_fn"] = function
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Wrap worker transport batches for restoration, otherwise retain native iteration."""
        iterator = super().__iter__()
        return _RestoringIterator(iterator) if self._array_transport else iterator
