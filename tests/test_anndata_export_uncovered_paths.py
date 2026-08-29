"""The paths ``spacr.anndata_export`` only takes when something goes wrong.

Four of the five behaviours here were excluded from measurement (three by a
``# pragma: no cover - env``, one as an unreachable guard) because they are
reached only by an environment the test machine is not in: the optional
extra absent, a registry that refuses to open, an ``.h5ad`` that will not
re-open. All three are reachable deliberately, and each one carries a
promise worth holding to:

* a missing extra must surface as the install line, not as whatever
  traceback ``anndata``'s import machinery happens to raise, and it must
  name the module that was actually missing;
* an export whose artifact registration fails must still leave a complete,
  readable ``.h5ad`` -- the registry is a convenience, the file is the
  product;
* a child file that cannot be re-opened to record its parent must warn
  rather than take the whole set down with it.

The fifth is the "nothing to impute" path of :func:`_compute_umap`, driven
with PCA so the embedding can be compared with the reducer's own output on
the matrix as handed in.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import anndata_export as ax

requires_anndata = pytest.mark.skipif(
    importlib.util.find_spec("anndata") is None,
    reason="the AnnData export needs `pip install spacr[anndata]`")


def _cell_table(n=4):
    """One object table with the five key columns a spaCR writer leaves."""
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": (["c1", "c2"] * n)[:n],
        "fieldID": ["f1"] * n,
        "object_label": range(1, n + 1),
        "cell_area": np.linspace(100.0, 400.0, n),
        "cell_channel_1_mean_intensity": np.linspace(1.0, 4.0, n),
        "measurement_units": ["um"] * n,
    })


def _write_db(path, tables):
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


class _RefuseAnnData(importlib.abc.MetaPathFinder):
    """A finder that fails ``import anndata`` the way a broken install does.

    ``name`` is the module the raised :class:`ImportError` blames, which is
    not necessarily ``anndata`` itself: a wheel whose ``h5py`` is missing
    fails on ``h5py``, and that is the name the user has to install.
    """

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "anndata":
            return None
        if self.name is None:
            raise ImportError("anndata is not importable here")
        raise ImportError(f"No module named {self.name!r}", name=self.name)


# ---------------------------------------------------------------------------
# the optional extra is not installed
# ---------------------------------------------------------------------------

def test_a_failed_anndata_import_becomes_the_install_line_not_a_traceback(
        monkeypatch):
    """``None`` in ``sys.modules`` is exactly what a stripped environment
    looks like to an ``import`` statement, and the caller must get the
    message telling them what to install rather than the interpreter's."""
    monkeypatch.setitem(sys.modules, "anndata", None)

    with pytest.raises(ax.AnnDataExtraMissing) as caught:
        ax.require_anndata()

    message = str(caught.value)
    assert isinstance(caught.value, ImportError), (
        "a caller guarding with `except ImportError` must keep working")
    assert 'python -m pip install "spacr[anndata]"' in message
    assert "(missing module: anndata)" in message
    assert isinstance(caught.value.__cause__, ImportError), (
        "the original import failure is chained, not swallowed")


@pytest.mark.parametrize("raised, reported", [
    ("h5py", "h5py"),
    ("anndata._core.anndata", "anndata"),
    (None, "anndata"),
])
def test_the_message_names_the_top_level_module_that_was_actually_missing(
        monkeypatch, raised, reported):
    """A dependency's name is the one worth printing -- telling someone to
    install ``anndata`` when ``h5py`` is what is missing sends them round a
    loop -- and a submodule is reduced to the package they can install."""
    monkeypatch.delitem(sys.modules, "anndata", raising=False)
    monkeypatch.setattr(sys, "meta_path",
                        [_RefuseAnnData(raised), *sys.meta_path])

    with pytest.raises(ax.AnnDataExtraMissing) as caught:
        ax.require_anndata()

    assert f"(missing module: {reported})" in str(caught.value)


def test_returning_the_module_is_what_callers_write_the_file_through():
    """The success path hands back the real module; ``require_anndata`` is
    the module's only import site, so a wrong return breaks every write."""
    pytest.importorskip("anndata",
                        reason="the anndata extra is not installed")

    module = ax.require_anndata()

    assert module is sys.modules["anndata"]
    assert callable(module.AnnData)


# ---------------------------------------------------------------------------
# the registry refuses the finished export
# ---------------------------------------------------------------------------

@requires_anndata
def test_an_export_that_cannot_be_registered_still_leaves_a_complete_file(
        tmp_path, monkeypatch):
    """The registry is a convenience; the ``.h5ad`` is the product.

    A project root that is a regular file is what a read-only project or a
    lock-refusing network mount looks like from here -- there is nowhere to
    put ``artifacts.db`` -- and the export must survive it with a warning
    and an empty artifact id, not lose the matrix it just computed.
    """
    import anndata as ad

    monkeypatch.delenv("SPACR_ARTIFACTS_DB", raising=False)
    db = _write_db(tmp_path / "m.db", {"cell": _cell_table()})
    blocked_root = tmp_path / "root_is_a_file"
    blocked_root.write_text("not a directory")
    out = str(tmp_path / "written" / "export.h5ad")

    with pytest.warns(RuntimeWarning) as caught:
        result = ax.export_anndata(db, out, single_table="cell",
                                   project=str(blocked_root), verbose=False)

    assert result.artifact_id == "", (
        "no id may be reported for a record that was never filed")
    assert result.path == out
    warned = "\n".join(str(w.message) for w in caught)
    assert out in warned and "could not be registered" in warned
    assert "uns['spacr']" in warned, (
        "the warning must say where the provenance still is")

    written = ad.read_h5ad(out)
    assert written.shape == (result.n_obs, result.n_vars) == (4, 2)
    np.testing.assert_allclose(
        np.asarray(written[:, "cell_area"].X).ravel(),
        np.linspace(100.0, 400.0, 4).astype(np.float32))
    assert written.uns["spacr"]["source_database"] == db


# ---------------------------------------------------------------------------
# the child file cannot be re-opened to record its parent
# ---------------------------------------------------------------------------

@requires_anndata
def test_a_child_file_that_will_not_reopen_warns_and_is_left_untouched(
        tmp_path):
    """``_stamp_parent_file`` re-opens a file the set export has already
    written. If that re-open fails the set is still on disk and usable
    without the cross-reference, so the failure is a warning naming the
    file -- and the file itself is not truncated on the way out."""
    child = tmp_path / "nucleus.h5ad"
    child.write_bytes(b"this is not an HDF5 file")

    with pytest.warns(RuntimeWarning) as caught:
        ax._stamp_parent_file(str(child), str(tmp_path / "cell.h5ad"),
                              "cell_id")

    warned = "\n".join(str(w.message) for w in caught)
    assert "could not record the parent file in" in warned
    assert str(child) in warned
    assert child.read_bytes() == b"this is not an HDF5 file", (
        "a failed stamp must not damage the file it could not read")


@requires_anndata
def test_a_stamped_child_records_the_sibling_file_holding_its_parents(
        tmp_path):
    """The success path the warning above is the fallback for: the child's
    ``uns`` names the parent file and the ``obs`` column that joins to it."""
    import anndata as ad

    db = _write_db(tmp_path / "m.db", {"cell": _cell_table()})
    child = str(tmp_path / "nucleus.h5ad")
    ax.export_anndata(db, child, single_table="cell", register=False,
                      verbose=False)

    ax._stamp_parent_file(child, str(tmp_path / "cell.h5ad"), "cell_id")

    parent = ad.read_h5ad(child).uns["spacr"]["relationships"]["parent"]
    assert parent["file"] == "cell.h5ad"
    assert parent["obs_column"] == "cell_id"


# ---------------------------------------------------------------------------
# _compute_umap with nothing to impute
# ---------------------------------------------------------------------------

def test_a_matrix_without_missing_values_reaches_the_reducer_unchanged():
    """The imputation is guarded because it has nothing to do on a complete
    matrix: the reducer must see the values as handed in, byte for byte.

    Driven through PCA rather than UMAP so the embedding is deterministic
    and can be compared with the reducer's own output on the same matrix.
    """
    from spacr.utils import reduction_and_clustering

    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(12, 5))
    assert np.isfinite(matrix).all(), "the fixture must have nothing to impute"
    settings = {"reduction_method": "pca", "min_samples": 2}

    embedding = ax._compute_umap(matrix, [f"f{i}" for i in range(5)], settings)

    expected, _labels, _reducer = reduction_and_clustering(
        matrix, n_neighbors=15, min_dist=0.1, metric="euclidean", eps=0.5,
        min_samples=2, clustering="dbscan", reduction_method="pca",
        verbose=False, n_jobs=1)
    assert embedding.shape == (12, 2)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(embedding, np.asarray(expected, np.float32),
                               rtol=1e-6, atol=1e-6)


def test_an_export_writes_into_a_directory_that_does_not_exist_yet(tmp_path):
    """``out_path`` is made absolute first, so its parent is always a real
    directory to create -- three levels of it, if that is what was asked
    for -- and the caller never has to mkdir before exporting."""
    pytest.importorskip("anndata",
                        reason="the anndata extra is not installed")

    db = _write_db(tmp_path / "m.db", {"cell": _cell_table()})
    out = tmp_path / "a" / "b" / "c" / "export.h5ad"
    assert not out.parent.exists()

    result = ax.export_anndata(db, out, single_table="cell", register=False,
                               verbose=False)

    assert os.path.isfile(str(out))
    assert result.path == str(out)
    assert result.n_obs == 4
