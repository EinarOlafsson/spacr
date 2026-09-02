"""Tests for ``spacr.cli_download`` — the ``spacr-download`` entry point.

NOT ONE OF THESE TESTS TOUCHES THE NETWORK, and an autouse fixture makes that
structural rather than hopeful: ``requests.get`` and ``urlopen`` are replaced
with something that raises, so a test that reaches for the hub fails on the
attempt instead of quietly downloading a gigabyte on somebody's laptop.

Two things here are load-bearing.

The first is :func:`test_importing_the_download_cli_pulls_no_gui_or_torch`.
This command exists so a cluster login node can stage the example data before a
batch job runs, and the download primitives it uses used to live in
``spacr.qt.hf_download``, which imports PySide6 at module scope. The moment an
import chain reaches Qt again, the command stops working on the machine it was
written for.

The second is the arithmetic in front of the screen. The published screen is
33 GB in eight pieces, and every guard against spending that by accident --
the opt-in selection, the already-on-disk skip, the size total, the
confirmation -- is a number or a branch that can be got wrong silently. So each
one is asserted on its own: what is selected, what is skipped, what the total
says, and what happens when there is nobody there to confirm.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

from spacr import cli_download
from spacr.example_archives import EXAMPLE_SETS
from spacr.screen_data import SCREEN_ASSETS
from tests.child_env import child_env


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Make a real transfer impossible rather than merely unlikely."""

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "a spacr-download test tried to reach the network")

    monkeypatch.setattr("requests.get", _forbidden)
    monkeypatch.setattr("urllib.request.urlopen", _forbidden)


@pytest.fixture(autouse=True)
def _plenty_of_room(monkeypatch):
    """Report a disk with room on it.

    The free-space check is real and is asserted on in its own test. Every
    other test here would otherwise depend on how full the machine's ``/tmp``
    happens to be, which is not what any of them are about.
    """
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda path: types.SimpleNamespace(
            total=10 ** 14, used=0, free=10 ** 14))


@pytest.fixture
def hub(monkeypatch):
    """A fake publisher: every download writes a real tar, and is recorded.

    Returns a namespace with ``calls`` -- ``(repo, archive, folder)`` per
    download -- and ``members``, a ``{archive: {path: bytes}}`` mapping the
    test can fill in to control what an archive unpacks to.
    """
    state = types.SimpleNamespace(calls=[], members={})

    def fake_download(repo, archive, folder, *, progress=None,
                      chunk_size=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        state.calls.append((repo, archive, folder))
        target = folder / archive
        members = state.members.get(archive, {f"{archive}.txt": b"payload"})
        with tarfile.open(target, "w") as tar:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        if progress is not None:
            size = target.stat().st_size
            progress(size, size)
        return target

    monkeypatch.setattr(cli_download, "download_archive", fake_download)
    return state


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "spacr-data"


def run(argv, dest=None):
    """Call the entry point the way the console script does."""
    if dest is not None:
        argv = list(argv) + ["--dest", str(dest)]
    return cli_download.main(argv)


def _row_for(out, name):
    """The listing row for one piece, marked or not."""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"* {name} ") or stripped.startswith(f"{name} "):
            return line
    raise AssertionError(f"no row for {name} in:\n{out}")


def _make_present(dest, keys=(), screen=()):
    """Put the markers on disk that make a piece count as downloaded."""
    folder = cli_download.example_folder(dest)
    folder.mkdir(parents=True, exist_ok=True)
    for key in keys:
        example = next(s for s in EXAMPLE_SETS if s.key == key)
        for marker in example.markers:
            target = folder / marker.replace("*", "made")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix:
                target.write_bytes(b"x")
            else:
                target.mkdir(exist_ok=True)
    for asset in screen:
        where = cli_download.screen_folder(dest, asset.plate)
        target = where / asset.unpacks_to
        if asset.kind == "measurements":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "one.png").write_bytes(b"x")


# ---------------------------------------------------------------------------
# the import weight, which is the reason this command exists at all
# ---------------------------------------------------------------------------


_HEAVY = ("PySide6", "PyQt5", "PyQt6", "tkinter", "torch", "cellpose",
          "matplotlib")

_PROBE = (
    "import json, sys\n"
    "{body}\n"
    "print(json.dumps({{m: (m in sys.modules) for m in %r}}))\n" % (_HEAVY,)
)


def _subprocess_modules(code: str) -> dict:
    """Run ``code`` in a fresh interpreter; it must print a JSON dict."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=180,
        env=child_env(pythonpath=str(REPO_ROOT)),
    )
    assert proc.returncode == 0, \
        f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_importing_the_download_cli_pulls_no_gui_or_torch():
    """The whole point: this runs where there is no display to give Qt.

    Checked in a fresh interpreter, because another test in this session
    having already imported PySide6 would hide the regression completely.
    """
    loaded = _subprocess_modules(_PROBE.format(body="import spacr.cli_download"))
    offenders = [name for name, present in loaded.items() if present]
    assert not offenders, \
        f"spacr.cli_download imported heavy modules: {offenders}"


def test_the_help_text_answers_without_importing_anything_heavy():
    """``--help`` on a login node with a cold NFS cache has to be instant."""
    body = (
        "import io, contextlib, spacr.cli_download\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    rc = spacr.cli_download.main(['--help'])\n"
        "assert rc == 0, rc\n"
        "assert 'spacr-download' in buf.getvalue()\n"
    )
    loaded = _subprocess_modules(_PROBE.format(body=body))
    offenders = [name for name, present in loaded.items() if present]
    assert not offenders, f"--help imported heavy modules: {offenders}"


def test_the_qt_downloader_still_re_exports_what_moved_out_of_it():
    """The GUI must not notice that the data half moved.

    Read as text rather than imported, so this test needs no display of its
    own. Every name here is one the Qt workers resolve as a module global, or
    one a test in ``tests/qt`` patches to keep the demo flow offline.
    """
    source = (REPO_ROOT / "spacr" / "qt" / "hf_download.py").read_text()
    assert "from ..example_archives import" in source
    for name in ("DATASET_REPO", "DATASET_SUB", "SETTINGS_REPO",
                 "MEASURE_EXAMPLE_REPO", "ANNOTATE_EXAMPLE_REPO",
                 "DATASET_PLACEHOLDER", "EXAMPLE_ARCHIVES", "_download_one",
                 "_list_files", "_content_length", "example_plate_folder",
                 "extract_example_archive", "expand_measure_arrays",
                 "explain_download_failure",
                 "make_the_example_paths_absolute"):
        assert name in source, f"hf_download no longer offers {name}"


# ---------------------------------------------------------------------------
# what a set of arguments selects
# ---------------------------------------------------------------------------


def test_no_arguments_selects_every_example_set_and_no_screen_piece():
    """33 GB must never be what happens when you type the command's name."""
    examples, screen = cli_download.resolve_selection([])
    assert [s.key for s in examples] == [s.key for s in EXAMPLE_SETS]
    assert screen == []


def test_one_named_example_set_selects_only_that_one():
    examples, screen = cli_download.resolve_selection(["measure"])
    assert [s.key for s in examples] == ["measure"]
    assert screen == []


def test_classify_is_an_alias_for_the_annotate_example_set():
    """Annotate and Classify share one published set, and a user following
    the Classify tutorial has no reason to know that."""
    examples, _ = cli_download.resolve_selection(["classify"])
    assert [s.key for s in examples] == ["annotate"]


def test_naming_a_screen_kind_is_by_itself_a_request_for_the_screen():
    """``--screen crops`` with no positional must not download the examples."""
    examples, screen = cli_download.resolve_selection([], screen="crops")
    assert examples == []
    assert {a.kind for a in screen} == {"crops"}
    assert len(screen) == 4


def test_a_screen_filter_beside_an_example_name_takes_both():
    """The filter must not be silently dropped because a positional was also
    given -- that is the same mistake as ignoring it entirely, read from the
    other end."""
    examples, screen = cli_download.resolve_selection(["mask"],
                                                      screen="measurements")
    assert [s.key for s in examples] == ["mask"]
    assert {a.kind for a in screen} == {"measurements"}


def test_a_plate_narrows_the_screen_to_that_plate_alone():
    _, screen = cli_download.resolve_selection([], screen="measurements",
                                               plates=[2])
    assert [(a.plate, a.kind) for a in screen] == [(2, "measurements")]


def test_several_plates_are_taken_together():
    _, screen = cli_download.resolve_selection(["screen"], screen="crops",
                                               plates=[1, 3])
    assert [a.plate for a in screen] == [1, 3]
    assert {a.kind for a in screen} == {"crops"}


def test_asking_for_a_plate_with_no_kind_takes_both_of_its_pieces():
    _, screen = cli_download.resolve_selection(["screen"], plates=[4])
    assert {a.kind for a in screen} == {"measurements", "crops"}
    assert {a.plate for a in screen} == {4}


def test_all_takes_the_examples_and_every_piece_of_the_screen():
    examples, screen = cli_download.resolve_selection(["all"])
    assert len(examples) == len(EXAMPLE_SETS)
    assert len(screen) == len(SCREEN_ASSETS)


def test_a_name_nothing_publishes_is_refused_and_the_real_ones_are_named():
    """A typo that selected nothing would download nothing and report
    success, which is the one outcome a download command must never have."""
    with pytest.raises(cli_download.SelectionError) as caught:
        cli_download.resolve_selection(["mesure"])
    message = str(caught.value)
    assert "measure" in message and "mesure" in message


def test_a_plate_the_screen_does_not_have_is_refused_before_anything_runs():
    with pytest.raises(cli_download.SelectionError) as caught:
        cli_download.resolve_selection(["screen"], plates=[9])
    assert "plate 9" in str(caught.value)


def test_a_screen_kind_that_does_not_exist_is_refused():
    with pytest.raises(cli_download.SelectionError):
        cli_download.resolve_selection(["screen"], screen="pictures")


def test_a_bad_name_on_the_command_line_exits_two_and_downloads_nothing(
        capsys, dest, hub):
    assert run(["nonsense"], dest) == cli_download.EXIT_USAGE
    assert hub.calls == []
    assert "nonsense" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the listing
# ---------------------------------------------------------------------------


def test_the_listing_shows_every_piece_with_its_size_and_downloads_nothing(
        capsys, dest, hub):
    assert run(["--list"], dest) == cli_download.EXIT_OK
    out = capsys.readouterr().out
    for example in EXAMPLE_SETS:
        assert example.key in out
    for asset in SCREEN_ASSETS:
        assert asset.archive in out
    assert "8.9 GB" in out                       # plate 1's crops, priced
    assert hub.calls == []


def test_the_listing_says_which_pieces_are_already_on_disk(capsys, dest, hub):
    _make_present(dest, keys=["annotate"])
    run(["--list"], dest)
    out = capsys.readouterr().out
    assert "present" in _row_for(out, "annotate")
    assert "missing" in _row_for(out, "measure")


def test_the_listing_totals_only_what_was_selected(capsys, dest, hub):
    run(["--list", "--screen", "measurements", "--plate", "1"], dest)
    out = capsys.readouterr().out
    assert "Selected: 1 of 11 pieces" in out
    assert "555.7 MB to download" in out


def test_the_total_leaves_out_what_is_already_on_disk(capsys, dest, hub):
    """The bill is what this run would COST, not what the selection weighs."""
    _make_present(dest, keys=["mask", "measure"])
    run(["--list"], dest)
    out = capsys.readouterr().out
    assert "Selected: 3 of 11 pieces, about 280.0 MB to download." in out
    assert "2 already on disk and skipped" in out


def test_dry_run_prints_the_same_listing_and_downloads_nothing(capsys, dest,
                                                               hub):
    assert run(["--dry-run"], dest) == cli_download.EXIT_OK
    out = capsys.readouterr().out
    assert "Dry run: nothing was downloaded." in out
    assert hub.calls == []


def test_the_default_run_says_how_to_ask_for_the_screen(capsys, dest, hub):
    """The screen is the thing this command deliberately does not do by
    itself; a user who never learns it exists will think spaCR has no screen
    to publish."""
    run([], dest)
    out = capsys.readouterr().out
    assert "--screen measurements" in out
    assert "--screen crops --plate 1" in out


# ---------------------------------------------------------------------------
# skipping, forcing, and where things land
# ---------------------------------------------------------------------------


def test_a_piece_already_on_disk_is_not_downloaded_again(capsys, dest, hub):
    _make_present(dest, keys=["mask", "measure", "annotate"])
    assert run([], dest) == cli_download.EXIT_OK
    assert hub.calls == []
    assert "already in" in capsys.readouterr().out


def test_force_downloads_a_piece_that_is_already_on_disk(dest, hub):
    """A truncated or hand-edited copy is repaired by fetching it again, and
    a row that could not be re-fetched would leave no way to do that."""
    _make_present(dest, keys=["mask", "measure", "annotate"])
    assert run(["--force"], dest) == cli_download.EXIT_OK
    assert len(hub.calls) == len(EXAMPLE_SETS)


def test_only_the_missing_pieces_are_fetched(dest, hub):
    _make_present(dest, keys=["mask"])
    assert run([], dest) == cli_download.EXIT_OK
    assert [call[1] for call in hub.calls] == [
        s.archive for s in EXAMPLE_SETS if s.key != "mask"]


def test_every_example_set_unpacks_into_the_one_shared_plate_folder(dest, hub):
    """The three sets are three stages of one plate, and spaCR is pointed at
    a plate rather than at three folders."""
    run([], dest)
    assert {call[2] for call in hub.calls} == {
        cli_download.example_folder(dest)}


def test_the_summary_names_the_folder_to_point_src_at(capsys, dest, hub):
    """What a user does next is put this path in `src`, and a summary that
    said only how many gigabytes arrived would leave them to work it out."""
    run(["mask"], dest)
    assert f"src: {cli_download.example_folder(dest)}" in capsys.readouterr().out


def test_each_screen_plate_lands_in_a_folder_of_its_own(dest, hub):
    """All four plates unpack to `data/` and `measurements/measurements.db`.
    In one folder the fourth would overwrite the third."""
    run(["--screen", "measurements", "--yes"], dest)
    folders = [call[2] for call in hub.calls]
    assert folders == [cli_download.screen_folder(dest, n)
                       for n in (1, 2, 3, 4)]
    assert len(set(folders)) == 4


def test_a_downloaded_archive_is_unpacked_and_then_removed(dest, hub):
    """The archive is a second copy of everything just written, and for a
    crop plate that is another 8 GB for no reason."""
    hub.members["spacr-example-annotate.tar"] = {
        "measurements/measurements.db": b"sqlite",
        "data/A01/one.png": b"png",
    }
    assert run(["annotate"], dest) == cli_download.EXIT_OK
    folder = cli_download.example_folder(dest)
    assert (folder / "measurements" / "measurements.db").is_file()
    assert (folder / "data" / "A01" / "one.png").is_file()
    assert not list(folder.glob("*.tar"))


def test_a_downloaded_database_has_its_relative_paths_made_absolute(
        dest, hub, tmp_path):
    """A measurements database stores absolute paths to its crops. The
    published copy stores them relative to the dataset root, so it is
    portable and carries no account name; this is what turns them back into
    paths that open."""
    import sqlite3

    made = tmp_path / "made.db"
    connection = sqlite3.connect(str(made))
    connection.execute("create table png_list (png_path text)")
    connection.execute("insert into png_list values ('data/A01/one.png')")
    connection.commit()
    connection.close()

    hub.members["spacr-example-annotate.tar"] = {
        "measurements/measurements.db": made.read_bytes(),
        "data/A01/one.png": b"png",
    }
    assert run(["annotate"], dest) == cli_download.EXIT_OK

    folder = cli_download.example_folder(dest)
    connection = sqlite3.connect(str(folder / "measurements"
                                     / "measurements.db"))
    stored = connection.execute("select png_path from png_list").fetchone()[0]
    connection.close()
    assert stored == str(folder / "data" / "A01" / "one.png")


def test_the_measure_example_expands_its_compressed_arrays(dest, hub):
    """The .npz compression is a transport detail; Measure reads .npy."""
    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(buffer, image=np.zeros((4, 4), dtype="uint16"))
    hub.members["spacr-example-measure.tar"] = {
        "merged/field1.npz": buffer.getvalue()}

    assert run(["measure"], dest) == cli_download.EXIT_OK
    merged = cli_download.example_folder(dest) / "merged"
    assert (merged / "field1.npy").is_file()
    assert not list(merged.glob("*.npz"))


# ---------------------------------------------------------------------------
# the confirmation in front of the 33 GB
# ---------------------------------------------------------------------------


def test_the_three_example_sets_stay_under_the_threshold(dest, hub,
                                                         monkeypatch):
    """The default run must never ask. Being asked to confirm the thing the
    command does with no arguments teaches the reflex that must not be in
    front of the screen."""
    monkeypatch.setattr(cli_download, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        cli_download, "_yes_at_the_prompt",
        lambda question: pytest.fail(f"the default run asked: {question}"))
    assert run([], dest) == cli_download.EXIT_OK
    assert len(hub.calls) == len(EXAMPLE_SETS)


def test_a_large_download_is_refused_when_there_is_nobody_to_confirm_it(
        capsys, dest, hub, monkeypatch):
    """A script, a cron job, a batch submission: no terminal, so no consent."""
    monkeypatch.setattr(cli_download, "_is_interactive", lambda: False)
    assert run(["--screen", "crops", "--plate", "1"], dest) == \
        cli_download.EXIT_USAGE
    assert hub.calls == []
    err = capsys.readouterr().err
    assert "8.9 GB" in err and "--yes" in err


def test_a_large_download_proceeds_when_yes_is_given(dest, hub, monkeypatch):
    monkeypatch.setattr(cli_download, "_is_interactive", lambda: False)
    assert run(["--screen", "crops", "--plate", "1", "--yes"], dest) == \
        cli_download.EXIT_OK
    assert [call[1] for call in hub.calls] == ["plate1-data.tar"]


def test_a_no_at_the_prompt_downloads_nothing_and_is_not_a_failure(
        capsys, dest, hub, monkeypatch):
    asked = []
    monkeypatch.setattr(cli_download, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli_download, "_yes_at_the_prompt",
                        lambda question: asked.append(question) or False)
    assert run(["screen"], dest) == cli_download.EXIT_OK
    assert hub.calls == []
    assert asked and "34.7 GB" in asked[0]
    assert "Nothing was downloaded." in capsys.readouterr().out


def test_a_yes_at_the_prompt_starts_the_download(dest, hub, monkeypatch):
    monkeypatch.setattr(cli_download, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli_download, "_yes_at_the_prompt", lambda _q: True)
    assert run(["--screen", "measurements"], dest) == cli_download.EXIT_OK
    assert len(hub.calls) == 4


def test_the_prompt_states_the_size_before_the_answer_is_given(dest,
                                                              monkeypatch):
    """The one number that has to be in front of the user is the one they
    are agreeing to spend."""
    seen = []
    monkeypatch.setattr("builtins.input", lambda question: seen.append(question)
                        or "n")
    assert cli_download._yes_at_the_prompt("Download about 33.7 GB? [y/N] ") \
        is False
    assert "33.7 GB" in seen[0]


@pytest.mark.parametrize("answer,expected",
                         [("y", True), ("Y", True), ("yes", True),
                          ("", False), ("n", False), ("later", False)])
def test_only_an_explicit_yes_is_taken_for_consent(answer, expected,
                                                   monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _q: answer)
    assert cli_download._yes_at_the_prompt("?") is expected


def test_an_end_of_input_is_a_no(monkeypatch):
    """A pipe that closed is not consent."""
    def _eof(_question):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert cli_download._yes_at_the_prompt("?") is False


# ---------------------------------------------------------------------------
# room on disk
# ---------------------------------------------------------------------------


def test_a_disk_without_room_is_refused_before_anything_is_downloaded(
        capsys, dest, hub, monkeypatch):
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda path: types.SimpleNamespace(total=10 ** 9, used=0, free=10 ** 8))
    assert run([], dest) == cli_download.EXIT_USAGE
    assert hub.calls == []
    assert "not enough room" in capsys.readouterr().err


def test_a_filesystem_that_will_not_answer_is_not_treated_as_a_full_one(
        dest, hub, monkeypatch):
    """Refusing on a failed statvfs would break the command on exactly the
    network filesystems a cluster user has."""
    def _boom(_path):
        raise OSError("no statvfs here")

    monkeypatch.setattr("shutil.disk_usage", _boom)
    assert run([], dest) == cli_download.EXIT_OK
    assert len(hub.calls) == len(EXAMPLE_SETS)


def test_the_room_needed_counts_the_archive_beside_its_unpacked_copy(
        dest, monkeypatch):
    """Each archive is unpacked and then deleted, so the peak is everything
    kept plus the largest archive still on disk -- not the total twice over,
    and not the total on its own."""
    plan = cli_download.build_plan(EXAMPLE_SETS, [], dest)
    needed = sum(p.bytes for p in plan) + max(p.bytes for p in plan)

    def _free(amount):
        monkeypatch.setattr(
            "shutil.disk_usage",
            lambda _path: types.SimpleNamespace(total=amount, used=0,
                                                free=amount))

    _free(needed)
    assert cli_download.room_for(plan, dest) is None
    _free(needed - 1)
    assert cli_download.room_for(plan, dest) is not None


# ---------------------------------------------------------------------------
# failure
# ---------------------------------------------------------------------------


def test_a_failed_download_exits_non_zero_and_explains_itself(capsys, dest,
                                                              monkeypatch):
    """The message is the one `explain_download_failure` makes, not a
    urllib3 dump that never says "you are offline"."""
    def _offline(*_args, **_kwargs):
        raise ConnectionError("Max retries exceeded with url: /api/datasets")

    monkeypatch.setattr(cli_download, "download_archive", _offline)
    assert run(["mask"], dest) == cli_download.EXIT_RUNTIME
    err = capsys.readouterr().err
    assert "Could not reach huggingface.co" in err
    assert "mask was not downloaded" in err


def test_one_failure_does_not_abandon_the_pieces_that_would_have_worked(
        capsys, dest, hub, monkeypatch):
    """A crop plate is an hour of network. Losing the three that would have
    succeeded because the second one's connection dropped means starting the
    hour again."""
    good = cli_download.download_archive

    def _second_one_fails(repo, archive, folder, **kwargs):
        if archive == "spacr-example-measure.tar":
            raise ConnectionError("dropped")
        return good(repo, archive, folder, **kwargs)

    monkeypatch.setattr(cli_download, "download_archive", _second_one_fails)
    assert run([], dest) == cli_download.EXIT_RUNTIME
    fetched = [call[1] for call in hub.calls]
    assert "spacr-example-mask.tar" in fetched
    assert "spacr-example-annotate.tar" in fetched
    assert "1 failed: measure" in capsys.readouterr().err


def test_an_interrupt_stops_the_run_rather_than_skipping_to_the_next_piece(
        capsys, dest, hub, monkeypatch):
    """Ctrl-C on the second of four plates means all four, not "abandon this
    8 GB and start the next 8 GB"."""
    good = cli_download.download_archive

    def _interrupted(repo, archive, folder, **kwargs):
        if archive == "plate2-measurements.tar":
            raise KeyboardInterrupt
        return good(repo, archive, folder, **kwargs)

    monkeypatch.setattr(cli_download, "download_archive", _interrupted)
    assert run(["--screen", "measurements", "--yes"], dest) == \
        cli_download.EXIT_RUNTIME
    assert [call[1] for call in hub.calls] == ["plate1-measurements.tar"]
    err = capsys.readouterr().err
    assert "interrupted" in err
    # And the two plates it never reached are reported as untried rather than
    # counted among the ones that arrived.
    assert "2 were not attempted" in err
