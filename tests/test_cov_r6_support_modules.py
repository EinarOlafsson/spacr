"""The last uncovered arc in ten small support modules.

Each of these modules is already above 98%: what is left is one branch
apiece, and the point of this file is to say which branch and why it
matters. Grouped here rather than in ten one-test files because the
modules have nothing in common except being nearly finished.

What is pinned, module by module:

``spacr.portable_paths``
    ``candidate_roots`` never yields the same folder twice, and the
    duplicate guard inside its climb is *proved* unreachable rather than
    silenced.
``spacr.spacr_cellpose``
    ``generate_masks_from_imgs(plot=True)`` shows the operator the stack
    the model actually saw -- the source pixels when ``resize=False``,
    the resized ones when ``resize=True``.
``spacr.figures.plates``
    An unnamed plate gets no title; a named one does.
``spacr.errors``
    ``read_run_status`` closes the connection it opened on every exit,
    and the ``conn is not None`` guard in its ``finally`` is proved to
    be false only while an exception is unwinding.
``spacr.resource_log``
    A platform whose ``memory_full_info()`` carries no integer private
    figure still reports resident memory, labelled ``rss``.
``spacr.external_masks``
    The same file reached through two overlapping input groups is
    imported once, not twice.
``spacr.run_journal``
    ``search_runs`` reads a bare-string warning; the ``elif values:``
    beside it is proved unreachable.
``spacr.resources.home.versions._generators.variants``
    Variant 06 still draws its search screen when there is no logo to
    put above it.
``spacr.control_names``
    A control typed as ``PREFIX_<blank>`` is not retried as an empty
    control.
``spacr.organelle_types``
    ``apply_preset(explain=True)`` prints what it set and what
    it kept.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# spacr.portable_paths -- the climb never revisits a folder
# ---------------------------------------------------------------------------

class TestCandidateRootsClimb:
    """``for _ in range(_MAX_CLIMB + 1): if here and here not in out:``

    The false side of that guard (a repeated or empty ``here``) cannot be
    driven, and this is the proof rather than a contortion:

    * ``here`` starts as ``os.path.abspath(...)``, which never returns
      ``""`` -- an empty argument is refused earlier by ``if not root``,
      and every other input becomes an absolute path.
    * the only assignment to ``here`` inside the loop is ``here = parent``
      where ``parent = os.path.dirname(here)``, reached only after
      ``if parent == here: break``. So each iteration replaces ``here``
      with a *strictly shorter* path and the loop stops at the fixed
      point rather than appending it twice.

    A strictly decreasing sequence has no repeats, so ``here not in out``
    is true on every pass. The test below pins that invariant on both a
    file and a folder, which is the property the guard exists to defend.
    """

    def test_every_candidate_is_a_new_shorter_folder(self, tmp_path):
        deep = tmp_path / "screen" / "plate1" / "measurements"
        deep.mkdir(parents=True)

        from spacr.portable_paths import candidate_roots

        roots = candidate_roots(str(deep))

        assert roots, "a real folder must produce at least itself"
        assert len(set(roots)) == len(roots), \
            f"the climb repeated a folder: {roots}"
        lengths = [len(r) for r in roots]
        assert lengths == sorted(lengths, reverse=True), \
            f"each candidate must be a parent of the last: {roots}"
        assert roots[0] == str(deep)

    def test_a_database_file_is_climbed_from_its_folder(self, tmp_path):
        """The file itself is never a candidate -- its folder is."""
        measurements = tmp_path / "plate1" / "measurements"
        measurements.mkdir(parents=True)
        db = measurements / "measurements.db"
        db.write_bytes(b"")

        from spacr.portable_paths import candidate_roots

        roots = candidate_roots(str(db))

        assert str(db) not in roots
        assert roots[0] == str(measurements)
        assert len(set(roots)) == len(roots)


# ---------------------------------------------------------------------------
# spacr.spacr_cellpose -- what the plot preview is handed
# ---------------------------------------------------------------------------

class _FakeCellposeModel:
    """Enough of ``CellposeModel`` for ``generate_masks_from_imgs``."""

    def __init__(self):
        self.seen = []

    def eval(self, x=None, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None,
             normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=256,
             compute_masks=True, progress=None):
        # THE INSTALLED SIGNATURE, WRITTEN OUT, and no **kwargs: a double
        # that accepts anything cannot fail when spaCR passes an argument
        # cellpose has removed. tests/cellpose_api_contract.py enforces it.
        image = np.asarray(x)
        self.seen.append(image.shape)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[1:3, 1:3] = 1
        flows = [np.zeros((h, w, 3), dtype=np.float32)]
        return mask, flows, None, None


def _image_dir(root: Path, size: int) -> Path:
    import tifffile

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2000, size=(size, size, 3)).astype(np.uint16)
    tifffile.imwrite(str(root / "img_0.tif"), arr)
    return root


@pytest.fixture
def _plot_calls(monkeypatch):
    """Record ``print_mask_and_flows`` instead of drawing it.

    ``generate_masks_from_imgs`` imports it from ``spacr.plot`` inside the
    function body, so patching the module attribute is what the product
    code actually resolves. Recording rather than no-op'ing is the point:
    ``plot=True`` exists to show the operator a picture, and only the
    recorded array says which picture.
    """
    import spacr.plot as PL

    calls = []
    monkeypatch.setattr(
        PL, "print_mask_and_flows",
        lambda stack, mask, flows, **k: calls.append(np.asarray(stack).shape))
    monkeypatch.setattr(PL, "plot_resize", lambda *a, **k: None,
                        raising=False)
    return calls


class TestTheMaskPreviewShowsWhatTheModelSaw:
    """``if plot: if resize: stack = resizescikit(...)``"""

    @staticmethod
    def _run(src, model, *, resize, plot):
        from spacr import spacr_cellpose as SC

        SC.generate_masks_from_imgs(
            str(src), model, "cyto", batch_size=1, diameter=30,
            cellprob_threshold=0.0, flow_threshold=0.4, grayscale=False,
            save=False, normalize=False, channels=[0], percentiles=[2, 98],
            invert=False, plot=plot, resize=resize, target_height=8,
            target_width=8, remove_background=False, background=100,
            Signal_to_noise=5, verbose=False)

    def test_without_resize_the_preview_gets_the_source_pixels(
            self, tmp_path, _plot_calls):
        src = _image_dir(tmp_path / "plain", size=32)
        model = _FakeCellposeModel()

        self._run(src, model, resize=True, plot=False)  # warm nothing

        _plot_calls.clear()
        model = _FakeCellposeModel()
        self._run(_image_dir(tmp_path / "plain2", size=32), model,
                  resize=False, plot=True)

        assert _plot_calls == [(32, 32, 3)], (
            "resize=False must hand the preview the untouched source stack, "
            f"not {_plot_calls}")
        assert model.seen == [(32, 32, 3)]

    def test_with_resize_the_preview_gets_the_resized_stack(
            self, tmp_path, _plot_calls):
        """The contrast that makes the assertion above mean something."""
        src = _image_dir(tmp_path / "shrunk", size=32)
        model = _FakeCellposeModel()

        self._run(src, model, resize=True, plot=True)

        assert model.seen == [(8, 8, 3)], "the model saw the resized stack"
        assert _plot_calls == [(32, 32, 3)], (
            "resize=True puts the preview back at the ORIGINAL dims, so the "
            "operator compares the mask against the image they own")


# ---------------------------------------------------------------------------
# spacr.figures.plates -- a plate is titled only when it has a name
# ---------------------------------------------------------------------------

class TestAnUnnamedPlateHasNoTitle:
    """``if name: ax.set_title(name, ...)`` -- both sides."""

    @staticmethod
    def _draw(name):
        import matplotlib.pyplot as plt

        from spacr.figures.plates import draw_plate

        figure, ax = plt.subplots()
        try:
            matrix = np.arange(8 * 12, dtype=float).reshape(8, 12)
            draw_plate(ax, matrix, vmin=0.0, vmax=95.0, cmap="viridis",
                       ink="#222222", name=name)
            return ax.get_title()
        finally:
            plt.close(figure)

    def test_a_small_multiple_panel_carries_no_descriptor(self):
        assert self._draw("") == "", \
            "an unnamed plate must not invent a title"

    def test_a_named_plate_says_which_plate_it_is(self):
        assert self._draw("plate1") == "plate1"


# ---------------------------------------------------------------------------
# spacr.errors -- the run-status connection is always closed
# ---------------------------------------------------------------------------

class _CountingConnection:
    """A real sqlite connection that counts its own ``close()``."""

    def __init__(self, connection, *, fail=False):
        self._connection = connection
        self.closes = 0
        self._fail = fail

    def execute(self, *args, **kwargs):
        if self._fail:
            raise sqlite3.OperationalError("database is locked")
        return self._connection.execute(*args, **kwargs)

    def close(self):
        self.closes += 1
        self._connection.close()


class TestReadRunStatusAlwaysClosesItsConnection:
    """``finally: if conn is not None: conn.close()``

    The false side of that guard -- falling out of the ``finally`` with
    ``conn`` still ``None`` -- is unreachable, and this is why:

    ``conn = None`` is followed immediately by
    ``from .database_concurrency import connect`` and
    ``conn = connect(...)``. ``connect`` is annotated and implemented to
    return an ``sqlite3.Connection``; it never returns ``None``. So the
    only way to be inside the ``finally`` with ``conn is None`` is for
    the import or the ``connect`` call itself to have raised -- and then
    the ``finally`` is being run while an exception unwinds, so control
    leaves by re-raising and never reaches the statement after it. That
    is the "guard inside ``finally`` that is only false during
    unwinding" case exactly.

    Both reachable exits are driven below, and the unwinding case is
    driven too -- as the exception it is, not as a fall-through.
    """

    @staticmethod
    def _db(tmp_path, *, stamped):
        path = tmp_path / "measurements.db"
        connection = sqlite3.connect(path)
        try:
            if stamped:
                from spacr.errors import RUN_STATUS_TABLE, _STATUS_COLUMNS

                columns = ", ".join(f'"{c}"' for c in _STATUS_COLUMNS)
                connection.execute(
                    f'CREATE TABLE {RUN_STATUS_TABLE} ({columns})')
                values = {
                    "run_id": "run-1", "name": "mask", "status": "complete",
                    "n_attempted": 3, "n_succeeded": 3, "n_failed": 0,
                    "failure_rate": 0.0,
                    "started_utc": "2026-01-01T00:00:00+00:00",
                    "stamped_utc": "2026-01-01T00:01:00+00:00",
                    "failures_json": "[]", "summary": "3 of 3",
                }
                connection.execute(
                    f'INSERT INTO {RUN_STATUS_TABLE} ({columns}) VALUES '
                    f'({", ".join("?" * len(_STATUS_COLUMNS))})',
                    tuple(values[c] for c in _STATUS_COLUMNS))
            else:
                connection.execute("CREATE TABLE cell (id INTEGER)")
            connection.commit()
        finally:
            connection.close()
        return path

    def _patched(self, monkeypatch, path, *, fail=False):
        import spacr.database_concurrency as DC

        opened = []
        real = DC.connect

        def _connect(*args, **kwargs):
            wrapper = _CountingConnection(real(*args, **kwargs), fail=fail)
            opened.append(wrapper)
            return wrapper

        monkeypatch.setattr(DC, "connect", _connect)
        return opened

    def test_a_stamped_database_is_read_and_closed(self, tmp_path,
                                                   monkeypatch):
        from spacr.errors import read_run_status

        path = self._db(tmp_path, stamped=True)
        opened = self._patched(monkeypatch, path)

        records = read_run_status(path)

        assert [r["run_id"] for r in records] == ["run-1"]
        assert [c.closes for c in opened] == [1], \
            "the connection must be closed exactly once on the read path"

    def test_an_unstamped_database_is_closed_too(self, tmp_path, monkeypatch):
        from spacr.errors import read_run_status

        path = self._db(tmp_path, stamped=False)
        opened = self._patched(monkeypatch, path)

        assert read_run_status(path) == []
        assert [c.closes for c in opened] == [1]

    def test_a_locked_database_closes_before_it_complains(self, tmp_path,
                                                          monkeypatch):
        from spacr.errors import RunStatusUnreadable, read_run_status

        path = self._db(tmp_path, stamped=True)
        opened = self._patched(monkeypatch, path, fail=True)

        with pytest.raises(RunStatusUnreadable) as raised:
            read_run_status(path)

        assert "cannot be read" in str(raised.value)
        assert [c.closes for c in opened] == [1]

    def test_a_connect_that_never_returns_leaves_by_the_exception(
            self, tmp_path, monkeypatch):
        """The ONLY way ``conn`` is still None in the ``finally``.

        And it leaves by unwinding: no value comes back, so the statement
        after the ``finally`` is not reached.
        """
        import spacr.database_concurrency as DC
        from spacr.errors import read_run_status

        path = self._db(tmp_path, stamped=True)

        def _refuse(*args, **kwargs):
            raise RuntimeError("no connection was ever made")

        monkeypatch.setattr(DC, "connect", _refuse)

        with pytest.raises(RuntimeError, match="no connection was ever made"):
            read_run_status(path)


# ---------------------------------------------------------------------------
# spacr.resource_log -- a platform with no private memory figures
# ---------------------------------------------------------------------------

class _NoPrivateFigures:
    """``memory_full_info()`` answers, but with nothing integral in it."""

    pid = 4242

    class _Full:
        # Present, but floats: psutil on some platforms reports these as
        # floats, and a float is not the measurement this code accepts.
        uss = 1.5
        pss = 2.5

    class _Info:
        rss = 7_340_032

    def memory_full_info(self):
        return self._Full()

    def memory_info(self):
        return self._Info()


class _PrivateFigures(_NoPrivateFigures):
    class _Full:
        uss = 1_048_576
        pss = 2_097_152


class TestMemoryFallsBackToResident:
    """``for measure in MEASURES:`` running out without returning."""

    def test_no_integer_private_figure_reports_rss(self):
        import psutil

        from spacr.resource_log import _memory

        assert _memory(psutil, _NoPrivateFigures()) == (7_340_032, "rss"), \
            "a float uss/pss is not a figure; the row must say rss"

    def test_an_integer_uss_is_preferred_and_labelled(self):
        """The contrast: with a real uss the loop returns before rss."""
        import psutil

        from spacr.resource_log import _memory

        assert _memory(psutil, _PrivateFigures()) == (1_048_576, "uss")


# ---------------------------------------------------------------------------
# spacr.external_masks -- one file, one import
# ---------------------------------------------------------------------------

def _write_tif(path, array):
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


class TestOverlappingInputGroupsImportEachFileOnce:
    """``if marker not in seen:`` -- the false side is a duplicate."""

    @staticmethod
    def _fields(tmp_path):
        shape = (24, 24)
        yy, xx = np.indices(shape)
        images = tmp_path / "images"
        masks = tmp_path / "cell_masks"
        _write_tif(images / "fov001_C1.tif",
                   (yy * 24 + xx).astype(np.uint16))
        mask = np.zeros(shape, dtype=np.uint16)
        mask[4:-4, 4:-4] = 1
        _write_tif(masks / "fov001_cell_mask.tif", mask)
        return images, masks

    def test_the_same_folder_listed_twice_is_scanned_once(self, tmp_path):
        from spacr import external_masks as em

        images, masks = self._fields(tmp_path)
        detected = em.detect_inputs([str(images), str(masks)])
        image_groups = [g for g in detected if g.role == "image"]
        assert image_groups, "the intensity folder must be detected as images"

        once = em.plan_external_masks({
            "inputs": [g.to_dict() for g in detected],
            "dst": str(tmp_path / "one"), "layout": "flat"})

        # The SAME image group listed a second time: every source it scans
        # is a marker already seen, so the duplicate guard fires for each.
        twice = em.plan_external_masks({
            "inputs": [g.to_dict() for g in detected]
                      + [g.to_dict() for g in image_groups],
            "dst": str(tmp_path / "two"), "layout": "flat"})

        assert len(twice.images) == len(once.images), (
            "listing an image group twice must not import its fields twice: "
            f"{len(once.images)} -> {len(twice.images)}")
        assert once.images, "the single-group plan found no images at all"


# ---------------------------------------------------------------------------
# spacr.run_journal -- a warnings block that is not a list
# ---------------------------------------------------------------------------

class TestASingleStringWarningIsStillOneWarning:
    """``values = manifest.get(key) or []`` then ``elif values:``

    The false side of ``elif values:`` is unreachable, and the ``or []``
    on the line above is the whole proof: whatever ``manifest.get(key)``
    returns, it reaches ``values`` only if it was *truthy*; anything
    falsy is replaced by ``[]``. So by the time the ``elif`` is
    evaluated -- which only happens when ``values`` is not a list or
    tuple, i.e. not the ``[]`` substitute -- ``values`` is a JSON scalar
    already known to be truthy. It is the "defensive re-check after a
    call that already guarantees the condition" case.

    Both reachable outcomes are driven here so the claim is not an
    assertion about an absence.
    """

    @staticmethod
    def _write_run(root, name, manifest):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        # Without it the folder itself is a finding, and the warnings list
        # under test would carry the reader's complaint instead.
        (directory / "settings.json").write_text("{}", encoding="utf-8")
        return directory

    def test_a_string_warning_arrives_whole(self, tmp_path, monkeypatch):
        import spacr.run_journal as RJ

        runs = tmp_path / "runs"
        runs.mkdir()
        monkeypatch.setattr(RJ, "runs_root", lambda: runs)
        self._write_run(runs, "2026-01-01_000000_a__mask", {
            "app_key": "mask", "status": "success",
            "start_utc": "2026-01-01T00:00:00+00:00",
            "warnings": "one thing went wrong",
            "provenance_warnings": ["and one more"],
        })

        record = RJ.search_runs()[0]

        assert "one thing went wrong" in record["warnings"], \
            "a bare string must not be spelled out one character per row"
        assert "and one more" in record["warnings"]

    def test_an_empty_warnings_block_contributes_nothing(self, tmp_path,
                                                         monkeypatch):
        """``0``/``""``/``None`` all become ``[]`` before the ``elif``."""
        import spacr.run_journal as RJ

        runs = tmp_path / "runs"
        runs.mkdir()
        monkeypatch.setattr(RJ, "runs_root", lambda: runs)
        self._write_run(runs, "2026-01-01_000000_b__mask", {
            "app_key": "mask", "status": "success",
            "start_utc": "2026-01-01T00:00:00+00:00",
            "warnings": "", "provenance_warnings": None,
        })

        record = RJ.search_runs()[0]

        assert record["warnings"] == [], \
            f"an empty warnings block invented {record['warnings']}"


# ---------------------------------------------------------------------------
# spacr.control_names -- a prefix with nothing after it
# ---------------------------------------------------------------------------

class TestAControlThatIsAllPrefix:
    """``shorter = resolve_control(tail, ...)`` then ``if shorter is not None``

    ``if tail and unused`` lets whitespace through -- ``"TGGT1_ "`` has a
    truthy tail -- and ``resolve_control`` strips it back to ``""`` and
    returns ``None``. That is the false side of the guard.
    """

    LIBRARY = [f"{gene}_{n}" for gene in ("000000", "233460")
               for n in range(1, 4)]

    def test_a_blank_tail_is_not_retried_as_an_empty_control(self):
        import pandas as pd

        from spacr.control_names import rows_for

        mask, described = rows_for("TGGT1_   ", self.LIBRARY)

        assert isinstance(mask, pd.Series)
        assert int(mask.sum()) == 0, \
            "nothing in the library is named 'TGGT1_   '"
        assert "0 guide(s)" in described, (
            "the description must say the control found nothing rather than "
            f"reporting a retry that never ran; got {described!r}")

    def test_a_real_tail_is_retried_without_the_organism_token(self):
        """The contrast: with something after the ``_`` the retry lands."""
        from spacr.control_names import rows_for

        mask, described = rows_for("TGGT1_233460", self.LIBRARY)

        assert int(mask.sum()) == 3, (
            "'233460' names three guides in this library, and dropping the "
            f"organism token is how they are found; got {int(mask.sum())}")
        assert "233460" in described


# ---------------------------------------------------------------------------
# spacr.organelle_types -- explain=True actually explains
# ---------------------------------------------------------------------------

class TestApplyOrganelleTypeExplains:
    """``if not applied and not kept: return`` -- the printing side."""

    def test_it_names_what_it_set_and_what_it_kept(self, capsys):
        from spacr.organelle_types import apply_preset

        out = apply_preset(
            {"organelle_type": "vesicular", "organelle_diameter": 12},
            explain=True)

        printed = capsys.readouterr().out
        assert "organelle_type =" in printed, \
            f"explain=True printed nothing useful: {printed!r}"
        assert "set    " in printed or "KEPT   " in printed, (
            "the explanation must say which values the preset filled in and "
            f"which were left alone; got {printed!r}")
        # The explanation describes the settings actually returned.
        for line in printed.splitlines():
            if line.startswith("  set    "):
                key = line.split()[1]
                assert key in out, f"{key} was announced but not applied"

    def test_it_says_nothing_without_explain(self, capsys):
        from spacr.organelle_types import apply_preset

        apply_preset(
            {"organelle_type": "vesicular", "organelle_diameter": 12})

        assert capsys.readouterr().out == "", \
            "the default must stay silent"
