"""On-camera install/removal commands match the scoped environment routes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from capture_terminal_install import steps


def test_pip_routes_use_plain_package_command_and_correct_activation():
    for route,activation in (('pip','source .venv/bin/activate'),
                             ('pip_conda','conda activate spacr-pip')):
        commands=[step[1] for step in steps(route,'1.5.1.3','install') if step[0]=='run']
        assert 'pip install spacr' in commands
        assert commands.index(activation)<commands.index('pip install spacr')
        assert not any('spacr[qt]' in command for command in commands)
    lesson=json.loads((Path(__file__).resolve().parents[1]/'lessons/02_install_spacr.json').read_text())
    assert lesson['scenes'][2]['visual']=='pip_02_activate_venv'
    assert 'Activate dot venv' in lesson['scenes'][2]['narration']
    assert 'Qt dependencies' in lesson['scenes'][3]['narration']


def test_pip_conda_removal_checks_own_prefix_before_uninstalling():
    plan=steps('pip_conda','1.5.1.3','uninstall')
    commands=[step[1] for step in plan if step[0] in ('quiet','run')]
    prefix='test "$CONDA_PREFIX" = /home/user/miniforge3/envs/spacr-pip'
    assert commands.index(prefix)<commands.index('pip uninstall spacr')
    assert commands.index('conda deactivate')<commands.index('conda env remove -n spacr-pip')
    assert 'conda env remove -n spacr' not in commands
    assert [step[1] for step in plan if step[0]=='shot']==[
        'pip_conda_03_uninstall','pip_conda_04_remove_environment']
