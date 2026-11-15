"""The ``spacr-plugins`` parser defaults to listing and rejects other commands.

The command's output is covered by tests/test_plugins_cli_listing_and_doctor.py.
"""
from __future__ import annotations

import pytest

from spacr import cli_plugins


def test_the_parser_defaults_to_listing():
    """No argument means ``list``, and ``--json`` is off."""
    args = cli_plugins.build_parser().parse_args([])

    assert args.command == "list"
    assert args.json is False
    assert cli_plugins.build_parser().parse_args(["doctor"]).command == "doctor"


def test_an_unknown_command_is_rejected():
    """Only the two documented commands are accepted."""
    with pytest.raises(SystemExit):
        cli_plugins.build_parser().parse_args(["undoctor"])
