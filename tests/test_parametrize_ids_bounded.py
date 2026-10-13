"""A huge parametrize value must not become a huge test id.

Run 37245630937 had shards stuck for hours on tests whose id carried a
one-megabyte string (see ``pytest_make_parametrize_id`` in conftest.py).
"""
import pytest

_HUGE = "x" * (1024 * 1024 + 1)


@pytest.mark.parametrize("value", [_HUGE, _HUGE + "y", "short", b"z" * 500, "caf\xe9" * 40])
def test_long_parameters_get_short_distinct_ids(request, value):
    nodeid = request.node.nodeid
    assert len(nodeid) < 250, len(nodeid)
    if isinstance(value, str) and len(value) <= 64:
        assert f"[{value}]" in nodeid
    else:
        assert f"[{len(value)}:" in nodeid
    assert nodeid.isascii()
