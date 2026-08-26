"""An export registers even when its upstream artifact cannot be read.

``_register`` looks up the ``measurements-db`` artifact so a re-run of
Measure marks the export stale. That lookup is a convenience: a registry
that refuses the query must not cost the finished file its own record.
"""
from __future__ import annotations

import os

import pytest

anndata = pytest.importorskip(
    "anndata", reason="the anndata extra is not installed")

from spacr import anndata_export as ax  # noqa: E402  (after importorskip)
from tests.test_anndata_export import build_project  # noqa: E402


@pytest.fixture()
def project_db(tmp_path):
    return build_project(str(tmp_path / "project"))


def test_an_upstream_lookup_that_raises_still_leaves_a_registered_export(
        project_db, tmp_path, monkeypatch):
    from spacr import artifacts

    def refuse(*args, **kwargs):
        raise RuntimeError("registry locked")

    monkeypatch.setattr(artifacts, "latest", refuse)

    out = str(tmp_path / "out" / "export.h5ad")
    result = ax.export_anndata(project_db, out, verbose=False)

    assert os.path.exists(out), "the file must be written before registration"
    assert result.artifact_id, (
        "an unreadable upstream must not cost the export its artifact")

    root = os.path.dirname(os.path.dirname(project_db))
    records = artifacts.by_kind(ax.ANNDATA_KIND, project=root)
    assert [record.artifact_id for record in records] == [result.artifact_id]
    assert list(records[0].inputs) == [], (
        "the input could not be read, so none is claimed")
