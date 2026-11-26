"""Item 1049 (schema 983): ops_pixel_quality defaults to 0.2; older files replay 0.1."""
from __future__ import annotations

from spacr import ops_decode, ops_engine
from spacr import settings as S


def test_the_default_is_0_2_and_the_row_replays_0_1():
    assert S.SETTINGS_SCHEMA_VERSION >= 983
    assert S.DEFAULT_CHANGES[983] == {"ops_pixel_quality": 0.1}
    assert ops_engine.ops_defaults({})["ops_pixel_quality"] == 0.2
    assert ops_decode.decode_rule(ops_engine.ops_defaults({}))["pixel_quality"] == 0.2
    keys = set(ops_engine.ops_defaults({}))
    old = {"plate": "P1"}
    applied = S._fold_older_defaults(old, 982, known=keys)
    assert applied["ops_pixel_quality"] == (0.1, 983)
    assert old["ops_pixel_quality"] == 0.1
    explicit = {"ops_pixel_quality": 0.3}
    S._fold_older_defaults(explicit, 982, known=keys)
    assert explicit["ops_pixel_quality"] == 0.3
    current = {"plate": "P1"}
    assert "ops_pixel_quality" not in S._fold_older_defaults(current, 983, known=keys)
