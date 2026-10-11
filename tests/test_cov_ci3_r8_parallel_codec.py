"""The spawn-pool task body of ``_parallel_cloudpickle_map``, run in process.

The pool runs ``_cloudpickled_parallel_task`` in spawned workers, whose
coverage the measurement does not always see. Called directly it must still
decode a closure with keyword arguments, call it, and encode what it returns
-- here another closure, which plain pickle cannot carry.
"""
from __future__ import annotations


def test_a_cloudpickled_task_round_trips_a_closure_and_its_result():
    from spacr.resource_log import _cloudpickle_codec, _cloudpickled_parallel_task

    codec = _cloudpickle_codec()
    offset = 5

    def compute(value, *, multiplier):
        return lambda: (value + offset) * multiplier

    payload = codec.dumps((compute, (2,), {"multiplier": 3}))

    result = codec.loads(_cloudpickled_parallel_task(payload))

    assert callable(result) and result() == 21
