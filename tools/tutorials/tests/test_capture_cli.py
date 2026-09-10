"""Recording acceptance checks the named command's status and actual output."""
from pathlib import Path
import sys
from subprocess import CompletedProcess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_cli import accepted_command


@pytest.mark.parametrize('code,text', [(0, 'was not called'), (2, 'no batch equivalent')])
def test_actual_expected_outcome_is_accepted(code, text):
    result = CompletedProcess(['spacr-run'], code, stdout=text)
    assert accepted_command(result, code, text)


@pytest.mark.parametrize('code,text', [(1, 'was not called'), (0, 'analysis started')])
def test_success_words_or_exit_zero_alone_do_not_prove_a_dry_run(code, text):
    result = CompletedProcess(['spacr-run'], code, stdout=text)
    assert not accepted_command(result, 0, 'was not called')


def test_unexpectedly_running_an_interactive_module_is_rejected():
    result = CompletedProcess(['spacr-run', 'annotate'], 0, stdout='no batch equivalent')
    assert not accepted_command(result, 2, 'no batch equivalent')
