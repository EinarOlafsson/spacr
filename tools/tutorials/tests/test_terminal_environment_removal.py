"""Guard the recorded retry after Conda's second confirmation was unanswered."""
from pathlib import Path
import sys
import re
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_terminal_install import require_partial_environment_removal, steps


def test_uninstall_answers_both_real_conda_prompts():
    removal = next(row for row in steps('pip_conda', '1.5.1.3', 'uninstall')
                   if len(row) > 1 and row[1] == 'conda env remove -n spacr-pip')
    assert len(removal[3]) == 2
    assert 'Do you wish to continue' in removal[3][1][0]
    retry = steps('pip_conda', '1.5.1.3', 'remove_environment')
    assert retry[0][1] == removal[1]
    assert retry[0][3] == [removal[3][1]]
    assert retry[1][1] == 'test ! -d /home/user/miniforge3/envs/spacr-pip'
    pattern = removal[3][1][0]
    assert re.search(pattern, 'Do you wish to continue?\r\n (y/[n])? ')
    assert re.search(pattern, 'Do you wish to continue? (y/[n])? ')
    assert not re.search(pattern, 'Do you wish to remove an unrelated environment? (y/[n])? ')


@pytest.mark.parametrize('unsafe', ['existing-package', 'conda-packages', 'no-failed-phase', 'symlink', None])
def test_retry_rejects_other_installation_states(tmp_path, unsafe):
    environment = tmp_path / 'home/miniforge3/envs/spacr-pip'
    (environment / 'conda-meta').mkdir(parents=True)
    (environment / 'conda-meta/history').write_text('retained completed removal transaction')
    proof = {'phases': {'uninstall': {'returncode': 1}}}
    if unsafe == 'existing-package':
        (environment / 'lib/python3.12/site-packages/spacr').mkdir(parents=True)
    elif unsafe == 'conda-packages':
        (environment / 'conda-meta/python.json').write_text('{}')
    elif unsafe == 'no-failed-phase':
        proof['phases']['uninstall']['returncode'] = 0
    elif unsafe == 'symlink':
        moved = tmp_path / 'elsewhere'
        environment.rename(moved)
        environment.symlink_to(moved)
    if unsafe:
        with pytest.raises(ValueError):
            require_partial_environment_removal(tmp_path, proof)
    else:
        require_partial_environment_removal(tmp_path, proof)
