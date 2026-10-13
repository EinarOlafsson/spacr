"""The OPS settings check asks for the folder OPS reads, not ``src``.

Item 520. :func:`spacr.ops_engine.run_ops` takes its input from
``genotype_source`` and never reads ``src``, so the generic source check,
which asked for ``src``, printed "[settings] ERROR [src]: src is missing
from the settings." in front of every OPS run -- including every one that
went on to succeed, and the one recorded for the OPS tutorial.
"""
from __future__ import annotations

from spacr.ops_settings import ops_defaults
from spacr.qt.bridge import _say_what_is_wrong_with_the_settings
from spacr.validate import ERROR, validate_settings


def _settings(**overrides):
    """OPS defaults with ``overrides`` applied."""
    settings = dict(ops_defaults())
    settings.update(overrides)
    return settings


def _source_errors(problems):
    """The errors raised against OPS's input folder or against ``src``."""
    return [problem for problem in problems
            if problem.severity == ERROR
            and problem.setting in ("src", "genotype_source")]


def test_valid_ops_inputs_raise_no_source_error(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    problems = validate_settings(
        _settings(genotype_source=str(tiles)), "ops")

    assert _source_errors(problems) == []
    assert not any("src is missing" in problem.message for problem in problems)


def test_a_missing_ops_input_is_still_an_error():
    problems = validate_settings(_settings(genotype_source=None), "ops")

    errors = _source_errors(problems)
    assert [problem.setting for problem in errors] == ["genotype_source"]
    assert "genotype_source" in errors[0].fix


def test_an_ops_input_key_left_out_is_an_error():
    settings = _settings()
    del settings["genotype_source"]

    errors = _source_errors(validate_settings(settings, "ops"))

    assert [problem.setting for problem in errors] == ["genotype_source"]
    assert "missing" in errors[0].message


def test_an_ops_input_that_does_not_exist_is_an_error(tmp_path):
    errors = _source_errors(validate_settings(
        _settings(genotype_source=str(tmp_path / "gone")), "ops"))

    assert [problem.setting for problem in errors] == ["genotype_source"]
    assert "does not exist" in errors[0].message


def test_the_gui_run_prints_no_settings_error_for_a_valid_ops_run(
        tmp_path, capsys):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    seen = []
    run = _say_what_is_wrong_with_the_settings("ops", seen.append)

    run(_settings(genotype_source=str(tiles)))

    out = capsys.readouterr().out
    assert "[settings] ERROR" not in out
    assert seen and seen[0]["genotype_source"] == str(tiles)


def test_the_gui_run_still_prints_an_error_when_ops_has_no_input(capsys):
    run = _say_what_is_wrong_with_the_settings("ops", lambda settings: None)

    run(_settings(genotype_source=None))

    out = capsys.readouterr().out
    assert "[settings] ERROR [genotype_source]" in out
    assert "[settings] ERROR [src]" not in out
