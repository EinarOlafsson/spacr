"""Item 1028: the cross-field copy link radius is the setting ops_cross_field_radius_px."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from spacr import ops_engine
from spacr.ops_settings import ops_defaults


def _fields(gap):
    def record(centroid):
        return {"transform": (1.0, np.eye(2), np.asarray(record.origin, float)),
                "shape": (100, 100),
                "areas": pd.Series([80.0], index=[1]),
                "whole": pd.Series([True], index=[1]),
                "centroids": {1: centroid}, "cells": None}

    out = {}
    for site, origin, centroid in ((1, (0, 0), (50.0, 80.0)),
                                   (2, (0, 60), (50.0, 20.0 + gap))):
        record.origin = origin
        out[site] = record(centroid)
    return out


def _edges(gap, radius):
    def refuse(site):
        raise AssertionError("label image read")

    return ops_engine._ph_cross_field_fields(
        _fields(gap), metric="centroid", radius=radius, load=refuse)


def _digest(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def test_the_default_is_24_and_matches_an_explicit_24():
    assert ops_defaults()["ops_cross_field_radius_px"] == 24.0
    options = ops_engine._ph_cross_field_options({})
    assert options["radius"] == 24.0
    default = _edges(7.0, options["radius"])
    explicit = _edges(7.0, 24.0)
    assert _digest(default[0]) == _digest(explicit[0])
    assert _digest(default[1]) == _digest(explicit[1])


def test_a_radius_of_8_links_copies_7_px_apart_and_not_9():
    seven = _edges(7.0, 8.0)
    nine = _edges(9.0, 8.0)
    assert len(seven[1]) == 1 and len(nine[1]) == 0
    assert int(seven[0].is_owner.sum()) == 1
    assert int(nine[0].is_owner.sum()) == 2


def test_the_setting_reaches_the_options():
    options = ops_engine._ph_cross_field_options(
        {"ops_cross_field_radius_px": 8})
    assert options["radius"] == 8.0
