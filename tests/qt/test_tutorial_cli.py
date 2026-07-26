"""`spacr-tutorial` CLI tests.

render_tutorial itself is stubbed — it boots a MainWindow and shells out
to ffmpeg — but everything the CLI is responsible for (argument parsing,
the 'all' fan-out, the unknown-app exit code, what gets forwarded to the
renderer, and what gets printed) is asserted against real values.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt.tutorial.scripts import AVAILABLE_TUTORIALS   # noqa: E402


@pytest.fixture
def fake_render(monkeypatch):
    """Replace render_tutorial, recording every call."""
    from spacr.qt.tutorial import __main__ as cli
    from spacr.qt.tutorial.engine import RenderResult

    calls = []

    def _render(name, out_dir=None, voice_model=None, length_scale=1.0):
        calls.append({"name": name, "out_dir": out_dir,
                        "voice_model": voice_model,
                        "length_scale": length_scale})
        return RenderResult(mp4=Path(out_dir or "/out") / f"{name}.mp4",
                              srt=Path(out_dir or "/out") / f"{name}.srt",
                              frames=120 + len(calls),
                              duration_s=4.25)

    monkeypatch.setattr(cli, "render_tutorial", _render)
    return calls


def test_main_renders_one_named_tutorial(fake_render, tmp_path, capsys):
    from spacr.qt.tutorial.__main__ import main
    rc = main(["mask", "--out", str(tmp_path)])

    assert rc == 0
    assert len(fake_render) == 1
    assert fake_render[0] == {"name": "mask", "out_dir": tmp_path,
                                "voice_model": None, "length_scale": 1.0}
    out = capsys.readouterr().out
    assert "rendering mask" in out
    assert str(tmp_path / "mask.mp4") in out
    assert str(tmp_path / "mask.srt") in out
    assert "4.2s" in out and "121 frames" in out


def test_main_all_fans_out_over_every_tutorial_in_order(fake_render,
                                                          tmp_path, capsys):
    from spacr.qt.tutorial.__main__ import main
    rc = main(["all", "--out", str(tmp_path)])

    assert rc == 0
    assert [c["name"] for c in fake_render] == AVAILABLE_TUTORIALS
    out = capsys.readouterr().out
    for name in AVAILABLE_TUTORIALS:
        assert f"rendering {name}" in out
        assert str(tmp_path / f"{name}.mp4") in out


def test_main_forwards_voice_and_length_scale(fake_render, tmp_path):
    from spacr.qt.tutorial.__main__ import main
    voice = tmp_path / "voice.onnx"
    rc = main(["home", "--out", str(tmp_path), "--voice", str(voice),
                "--length-scale", "0.75"])

    assert rc == 0
    assert fake_render[0]["voice_model"] == voice
    assert fake_render[0]["length_scale"] == 0.75
    assert isinstance(fake_render[0]["voice_model"], Path)


def test_main_defaults_out_to_home_spacr_tutorials(fake_render, monkeypatch,
                                                     tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    from spacr.qt.tutorial.__main__ import main

    assert main(["measure"]) == 0
    assert fake_render[0]["out_dir"] == fake_home / "spacr-tutorials"


def test_main_rejects_an_unknown_tutorial_without_rendering(fake_render,
                                                              capsys):
    from spacr.qt.tutorial.__main__ import main
    rc = main(["definitely-not-a-tutorial"])

    assert rc == 2
    assert fake_render == [], "nothing may be rendered for a bad name"
    err = capsys.readouterr().err
    assert "unknown tutorial: definitely-not-a-tutorial" in err


def test_verbose_flag_selects_debug_logging(fake_render, tmp_path,
                                              monkeypatch):
    from spacr.qt.tutorial.__main__ import main
    seen = {}

    def fake_basic_config(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    main(["home", "--out", str(tmp_path), "-v"])
    assert seen["level"] == logging.DEBUG
    assert "%(levelname)s" in seen["format"]

    seen.clear()
    main(["home", "--out", str(tmp_path)])
    assert seen["level"] == logging.INFO


def test_parser_help_lists_every_tutorial_and_the_all_sentinel(capsys):
    from spacr.qt.tutorial.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in AVAILABLE_TUTORIALS:
        assert name in out
    assert "'all'" in out
    assert "spacr-tutorial" in out
    assert "--length-scale" in out


def test_missing_positional_arg_is_a_usage_error(capsys):
    from spacr.qt.tutorial.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "app" in capsys.readouterr().err


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch,
                                                                capsys):
    """`python -m spacr.qt.tutorial.__main__` must propagate main()'s
    return code, not swallow it."""
    import runpy

    # runpy re-executes the module in a fresh namespace, so the app name
    # here has to be one that returns before any render is attempted.
    monkeypatch.setattr(sys, "argv", ["spacr-tutorial", "no-such-tutorial"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("spacr.qt.tutorial.__main__", run_name="__main__")
    assert exc.value.code == 2
    assert "unknown tutorial" in capsys.readouterr().err
