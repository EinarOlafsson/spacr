"""``396`` — Make Masks opens on a folder from a terminal, as a queue.

:mod:`spacr.curation_queue` decides what a session is; this file is about
the command that starts one. Four things have to hold for
``spacr-make-masks`` to be worth having over SSH:

* the arguments the ledger asks for exist and default the way it says --
  ``--folder``, ``--order`` (``easy`` by default), ``--limit``;
* a folder that is not there, or that holds no layout spaCR recognises, is
  refused with a sentence BEFORE Qt is imported. A curator on a login node
  with no display gets an answer, not a Qt crash, and that is asserted in a
  subprocess rather than by reading ``sys.modules`` in a suite where some
  other test may already have imported PySide6;
* what the editor is handed is the session the arguments asked for -- the
  ordering, the limit, and the fields the resume record has not already
  closed;
* the refusal in :data:`spacr.cli.INTERACTIVE_ONLY` has stopped being a
  flat refusal and names the command instead.

Everything here is Qt-free. The screen half of the seam --
:meth:`spacr.qt.screens.make_masks.MakeMasksScreen.open_queue` -- is
``tests/qt/test_the_editor_opens_on_the_queue_it_was_handed.py``.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from spacr import cli, cli_make_masks
from spacr.curation_queue import STATUS_FILENAME, CurationQueue, build_queue

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Folders to point the command at
# ---------------------------------------------------------------------------

def _image(path: Path, seed: int = 0) -> None:
    """Write one small field.

    :param path: where to write it.
    :param seed: which pseudo-random field to write.
    """
    import imageio.v2 as imageio

    rng = np.random.default_rng(seed)
    imageio.imwrite(path, rng.integers(0, 4000, (24, 24), dtype=np.uint16))


def _mask(path: Path, objects: int = 2) -> None:
    """Write a draft mask holding ``objects`` square objects.

    :param path: where to write it.
    :param objects: how many objects to draw.
    """
    import imageio.v2 as imageio

    labels = np.zeros((24, 24), dtype=np.uint16)
    for index in range(objects):
        row = 2 + index * 6
        labels[row:row + 4, 2:6] = index + 1
    imageio.imwrite(path, labels)


@pytest.fixture
def nested(tmp_path: Path) -> Path:
    """Four fields with a ``masks/`` folder beneath them; two have drafts."""
    folder = tmp_path / "nested"
    (folder / "masks").mkdir(parents=True)
    for index in range(4):
        _image(folder / f"f_{index:02d}.tif", seed=index)
    _mask(folder / "masks" / "f_01.tif", objects=2)
    _mask(folder / "masks" / "f_02.tif", objects=3)
    return folder


@pytest.fixture
def sibling(tmp_path: Path) -> Path:
    """``images/`` beside ``masks/`` — the layout the field sets are in."""
    folder = tmp_path / "sibling"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir(parents=True)
    for index in range(3):
        _image(folder / "images" / f"s_{index}.tif", seed=index)
    _mask(folder / "masks" / "s_0.tif")
    return folder


@pytest.fixture
def seg(tmp_path: Path) -> Path:
    """Cellpose ``_seg.npy`` bundles — what the external tool edits."""
    folder = tmp_path / "seg"
    folder.mkdir()
    labels = np.zeros((24, 24), dtype=np.uint16)
    labels[2:8, 2:8] = 1
    for index in range(2):
        np.save(folder / f"b_{index}_seg.npy",
                {"masks": labels, "img": labels}, allow_pickle=True)
    return folder


@pytest.fixture
def handed(monkeypatch):
    """Catch the queue that would have been handed to the editor.

    :returns: a list that receives each :class:`CurationQueue`
        :func:`spacr.cli_make_masks.open_editor` was called with.
    """
    caught = []

    def _fake(queue):
        caught.append(queue)
        return 0

    monkeypatch.setattr(cli_make_masks, "open_editor", _fake)
    return caught


# ---------------------------------------------------------------------------
# The arguments the ledger asks for
# ---------------------------------------------------------------------------

def test_the_parser_asks_for_a_folder_and_defaults_to_easy():
    """``--order easy`` is the default because the external tool settled on it."""
    args = cli_make_masks.build_parser().parse_args(["--folder", "/tmp/x"])
    assert args.folder == "/tmp/x"
    assert args.order == "easy"
    assert args.limit is None
    assert args.dry_run is False


@pytest.mark.parametrize("order", ["easy", "prob", "value", "name"])
def test_every_documented_order_is_accepted(order):
    """The four orders the ledger names all parse."""
    args = cli_make_masks.build_parser().parse_args(
        ["--folder", "/tmp/x", "--order", order])
    assert args.order == order


def test_a_mistyped_order_is_refused_rather_than_silently_reordering():
    """``--order esay`` must not quietly become ``value``.

    The external tool's ``sorted(items, key=_value)`` is also its catch-all,
    so a typo there reorders a 500-field session without a word.
    """
    with pytest.raises(SystemExit) as exit_code:
        cli_make_masks.build_parser().parse_args(
            ["--folder", "/tmp/x", "--order", "esay"])
    assert exit_code.value.code == 2


def test_the_folder_is_required():
    """A session with no folder is not a session."""
    with pytest.raises(SystemExit) as exit_code:
        cli_make_masks.build_parser().parse_args(["--order", "name"])
    assert exit_code.value.code == 2


def test_the_limit_is_a_number_of_fields(nested, handed):
    """``--limit 2`` hands the editor two fields out of four."""
    assert cli_make_masks.main(
        ["--folder", str(nested), "--order", "name", "--limit", "2"]) == 0
    queue, = handed
    assert [item.stem for item in queue.items] == ["f_00", "f_01"]
    assert queue.limit == 2
    assert queue.summary.total == 4


def test_a_session_of_no_fields_is_refused(nested):
    """``--limit 0`` is a typo, not a request; it is answered as one."""
    with pytest.raises(SystemExit) as exit_code:
        cli_make_masks.main(["--folder", str(nested), "--limit", "0"])
    assert exit_code.value.code == 2


def test_a_negative_limit_is_refused(nested):
    """As is ``--limit -3``, which argparse itself accepts as an int."""
    with pytest.raises(SystemExit) as exit_code:
        cli_make_masks.main(["--folder", str(nested), "--limit", "-3"])
    assert exit_code.value.code == 2


# ---------------------------------------------------------------------------
# Refused before Qt
# ---------------------------------------------------------------------------

def test_a_folder_that_is_not_there_is_named_and_refused(tmp_path, capsys,
                                                         handed):
    """The one message somebody over SSH is most likely to see."""
    missing = tmp_path / "not_here"
    assert cli_make_masks.main(["--folder", str(missing)]) == 2
    err = capsys.readouterr().err
    assert f"no such folder: {missing}" in err, (
        "a path that is not there is its own answer, not a layout report")
    assert not handed


def test_a_file_is_not_a_folder(tmp_path, capsys, handed):
    """``--folder`` pointed at an image is a mistake worth naming."""
    path = tmp_path / "field.tif"
    _image(path)
    assert cli_make_masks.main(["--folder", str(path)]) == 2
    assert "not a folder" in capsys.readouterr().err
    assert not handed


def test_a_folder_with_no_layout_says_what_each_layout_needs(tmp_path, capsys,
                                                             handed):
    """Refused rather than opened empty — an empty queue reads as "all done"."""
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / "notes.txt").write_text("nothing to curate here", encoding="utf-8")

    assert cli_make_masks.main(["--folder", str(folder)]) == 2
    err = capsys.readouterr().err
    assert "not a curation queue in any layout" in err
    for layout in ("nested", "sibling", "seg"):
        assert layout in err
    assert not handed


def test_two_layouts_at_once_are_refused_rather_than_guessed(nested, capsys,
                                                             handed):
    """A folder that is both nested and seg loads half of itself, or nothing."""
    np.save(nested / "extra_seg.npy", {"masks": np.zeros((4, 4))},
            allow_pickle=True)
    assert cli_make_masks.main(["--folder", str(nested)]) == 2
    assert "ambiguous" in capsys.readouterr().err
    assert not handed


@pytest.mark.parametrize("case", ["missing", "no_layout"])
def test_nothing_qt_is_imported_before_a_folder_is_refused(tmp_path, case):
    """Asserted in a fresh interpreter, where no other test can have imported it.

    This is the whole reason the command exists as its own module: over SSH
    with no display, a wrong ``--folder`` must cost a sentence, not a Qt
    import and the crash at the end of it.
    """
    folder = tmp_path / case
    if case == "no_layout":
        folder.mkdir()
        (folder / "readme.md").write_text("no images", encoding="utf-8")

    script = textwrap.dedent(f"""
        import sys
        from spacr import cli_make_masks
        code = cli_make_masks.main(["--folder", {str(folder)!r}])
        heavy = [name for name in sys.modules
                 if name.split(".")[0] in
                 ("PySide6", "PyQt5", "PyQt6", "torch", "cellpose", "cv2")]
        print("CODE", code)
        print("HEAVY", sorted(heavy))
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert "CODE 2" in result.stdout, result.stderr
    assert "HEAVY []" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# What the editor is handed
# ---------------------------------------------------------------------------

def test_the_editor_is_handed_the_session_the_arguments_asked_for(nested,
                                                                  handed):
    """Order and limit reach the queue, and the queue reaches the editor."""
    assert cli_make_masks.main(
        ["--folder", str(nested), "--order", "name"]) == 0
    queue, = handed
    assert isinstance(queue, CurationQueue)
    assert queue.order == "name"
    assert queue.folder == nested
    assert [item.stem for item in queue.items] == ["f_00", "f_01", "f_02",
                                                   "f_03"]


def test_a_field_already_done_is_not_offered_again(nested, handed):
    """What "resume" means: the record is read before anything is offered."""
    (nested / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\n"
        "f_00,done,4,2026-09-12T10:00:00\n"
        "f_03,skip,,2026-09-12T10:01:00\n", encoding="utf-8")

    assert cli_make_masks.main(
        ["--folder", str(nested), "--order", "name"]) == 0
    queue, = handed
    assert [item.stem for item in queue.items] == ["f_01", "f_02"]
    assert (queue.summary.done, queue.summary.skip,
            queue.summary.remaining) == (1, 1, 2)


def test_the_summary_is_printed_before_the_editor_opens(nested, monkeypatch,
                                                        capsys):
    """A curator sees what they are resuming into, in the shell they typed in.

    The summary is captured from INSIDE the fake editor launch, so this
    fails if the line is printed after the GUI takes the process over --
    which, on a machine where Qt takes ten seconds to import, is the
    difference between a session summary and a blank terminal.
    """
    (nested / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\n"
        "f_00,done,4,2026-09-12T10:00:00\n", encoding="utf-8")
    seen = {}

    def _fake(queue):
        seen["out"] = capsys.readouterr().out
        return 0

    monkeypatch.setattr(cli_make_masks, "open_editor", _fake)
    assert cli_make_masks.main(
        ["--folder", str(nested), "--order", "name", "--limit", "2"]) == 0

    printed = seen["out"]
    assert "4 bundles" in printed
    assert "1 done" in printed
    assert "0 skip" in printed
    assert "3 remaining" in printed
    assert "2 this session" in printed
    assert str(nested) in printed


def test_a_finished_folder_opens_nothing_and_says_so(nested, capsys, handed):
    """Every field reviewed is a finished session, not an empty editor."""
    (nested / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\n"
        + "".join(f"f_{index:02d},done,2,2026-09-12T10:00:00\n"
                  for index in range(4)), encoding="utf-8")

    assert cli_make_masks.main(["--folder", str(nested)]) == 0
    out = capsys.readouterr().out
    assert "nothing to open" in out
    assert STATUS_FILENAME in out
    assert not handed


def test_dry_run_prints_the_queue_in_order_and_opens_nothing(nested, capsys,
                                                             handed):
    """The form of the command that needs no display at all."""
    assert cli_make_masks.main(
        ["--folder", str(nested), "--order", "name", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert not handed
    stems = [line.split()[-1] for line in out.splitlines()
             if line.startswith(("1 ", "2 ", "3 ", "4 "))]
    assert stems == ["f_00", "f_01", "f_02", "f_03"]


def test_easy_order_announces_its_fallback_rather_than_reordering_silently(
        nested, capsys, handed):
    """``easy`` needs probabilities; without them it says what it did instead."""
    assert cli_make_masks.main(["--folder", str(nested)]) == 0
    out = capsys.readouterr().out
    queue, = handed
    assert queue.order == "easy"
    assert queue.effective_order == "value"
    assert "falling back to value" in out


# ---------------------------------------------------------------------------
# The layouts the editor cannot edit in place
# ---------------------------------------------------------------------------

def test_the_sibling_layout_is_refused_with_what_it_would_have_done_wrong(
        sibling, capsys, handed):
    """Opening it on ``images/`` would orphan every draft mask in ``masks/``."""
    assert cli_make_masks.main(["--folder", str(sibling)]) == 2
    captured = capsys.readouterr()
    assert not handed
    assert "3 bundles" in captured.out, "the queue itself was still read"
    assert str(sibling / "images" / "masks") in captured.err
    assert "--dry-run" in captured.err


def test_the_seg_layout_is_refused_but_still_reads_as_a_queue(seg, capsys,
                                                              handed):
    """A Cellpose bundle is a queue spaCR can read and this editor cannot open."""
    assert cli_make_masks.main(["--folder", str(seg)]) == 2
    assert not handed
    assert "_seg.npy" in capsys.readouterr().err

    assert cli_make_masks.main(["--folder", str(seg), "--dry-run"]) == 0
    assert "b_0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The handover, which is how the queue reaches the screen
# ---------------------------------------------------------------------------

def test_the_handover_is_taken_once_and_then_empty(nested):
    """A screen rebuilt later must open on nothing, not on a finished session."""
    queue = object()
    cli_make_masks.hand_over(queue)
    assert cli_make_masks.take_handover() is queue
    assert cli_make_masks.take_handover() is None


def test_open_editor_hands_the_queue_over_and_then_starts_the_gui(nested,
                                                                  monkeypatch):
    """The queue is in the slot by the time the GUI starts building screens."""
    import spacr.qt

    seen = {}

    def _fake_run(argv):
        seen["argv"] = list(argv)
        seen["queue"] = cli_make_masks.take_handover()
        return 0

    monkeypatch.setattr(spacr.qt, "run", _fake_run)
    monkeypatch.setattr(cli_make_masks, "has_display", lambda: True)
    queue = object()
    assert cli_make_masks.open_editor(queue) == 0
    assert seen["argv"] == ["make_masks"]
    assert seen["queue"] is queue
    assert cli_make_masks.take_handover() is None


def test_a_scores_file_that_cannot_be_read_stops_the_session(nested, capsys,
                                                             handed):
    """A scores file with the wrong columns is a mistake, not "no scores".

    ``easy`` and ``prob`` fall back loudly when there is NO scores file.
    When there is one and it cannot be used, falling back would sort a
    500-field session by something other than what was asked for and say
    only that the file was missing, which it was not.
    """
    (nested / "curate_scores.csv").write_text(
        "stem,score\nf_00,0.9\n", encoding="utf-8")

    assert cli_make_masks.main(["--folder", str(nested)]) == 2
    err = capsys.readouterr().err
    assert "curate_scores.csv" in err
    assert "probability column" in err
    assert not handed


@pytest.mark.skipif(sys.platform in ("win32", "darwin"),
                    reason="DISPLAY is a windowing question only on Unix")
def test_no_display_is_answered_with_a_sentence_not_a_qt_crash(nested, capsys,
                                                               monkeypatch):
    """The SSH session the ledger is about: the queue reads, the editor cannot.

    Without this, Qt answers instead -- "could not load the Qt platform
    plugin xcb" and an abort -- which says nothing about the 105 fields that
    were perfectly readable a moment earlier.
    """
    for variable in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(variable, raising=False)
    queue = build_queue(nested, order="name")

    assert cli_make_masks.open_editor(queue) == cli_make_masks.EXIT_NO_GUI
    err = capsys.readouterr().err
    assert "4 bundles" in err, "it still says what is waiting"
    assert "ssh -X" in err
    assert "--dry-run" in err
    assert cli_make_masks.take_handover() is None, (
        "a queue nobody can open must not be left in the slot")


@pytest.mark.skipif(sys.platform in ("win32", "darwin"),
                    reason="DISPLAY is a windowing question only on Unix")
@pytest.mark.parametrize("variable", ["DISPLAY", "WAYLAND_DISPLAY",
                                      "QT_QPA_PLATFORM"])
def test_each_way_of_having_a_surface_counts_as_one(monkeypatch, variable):
    """X, Wayland, and an explicitly chosen Qt platform are all a display.

    ``QT_QPA_PLATFORM`` is how offscreen rendering, VNC and the embedded
    platforms are asked for; somebody who set it has already said which
    surface Qt is to use, and being told there is no display would be wrong.
    """
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    assert cli_make_masks.has_display() is False

    monkeypatch.setenv(variable, "offscreen" if "QT" in variable else ":0")
    assert cli_make_masks.has_display() is True


# ---------------------------------------------------------------------------
# The refusal that stopped being one
# ---------------------------------------------------------------------------

def test_the_gui_only_note_no_longer_merely_says_use_the_gui():
    """It names the headless route, as image_scatter, pca and hit_list do."""
    note = cli.INTERACTIVE_ONLY["make_masks"]
    assert note != ("Make Masks is a manual mask editor; run it in the GUI "
                    "(spacr-qt).")
    for named in ("spacr-make-masks", "--folder", "--order", "--limit",
                  "curate_status.csv"):
        assert named in note, f"the note does not mention {named}"


def test_spacr_run_make_masks_points_at_the_command(capsys):
    """The refusal is what ``spacr-run make_masks`` prints, so it must carry it."""
    assert cli.main(["make_masks"]) == cli.EXIT_USAGE
    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert "spacr-make-masks --folder" in printed


def test_make_masks_stays_a_gui_only_module():
    """Naming a terminal entry point is not the same as running headless.

    The brush is still the feature; what the command opens is still a
    window. ``spacr-run make_masks`` must go on refusing rather than
    growing a pipeline that does not exist.
    """
    assert "make_masks" in cli.INTERACTIVE_ONLY
    assert "make_masks" not in cli.MODULES


def test_setup_py_installs_the_command():
    """An entry point nobody declared is a command nobody can type."""
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    scripts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if getattr(key, "value", None) == "console_scripts":
                scripts.extend(element.value for element in value.elts)
    assert "spacr-make-masks=spacr.cli_make_masks:main" in scripts
