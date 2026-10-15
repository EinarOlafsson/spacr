"""Run the docs workflow's target resolver and publication commands in isolation."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
OLD_SPACE = 'einarolafsson/spacr-docs-nightly'
OLD_URL = 'https://einarolafsson-spacr-docs-nightly.static.hf.space/'
NEW_SPACE = 'einarolafsson/spacr-docs-nightly-v2'
NEW_URL = 'https://einarolafsson-spacr-docs-nightly-v2.static.hf.space/'


@pytest.fixture
def workflow():
    return yaml.safe_load((ROOT / '.github/workflows/docs.yml').read_text())


def _resolve(workflow, tmp_path, space, url):
    steps = workflow['jobs']['resolve']['steps']
    matches = [step for step in steps if step.get('id') == 'nightly-target']
    assert len(matches) == 1, 'The actual workflow must resolve one coherent host target'
    step = matches[0]
    assert step['env'] == {
        'CONFIGURED_SPACE': '${{ vars.HF_DOCS_SPACE }}',
        'CONFIGURED_URL': '${{ vars.HF_DOCS_URL }}',
    }
    lines = step['run'].splitlines()
    delimiter = re.search(r"<<['\"]?([A-Za-z_]+)", lines[0]).group(1)
    assert lines[-1] == delimiter
    program = textwrap.dedent('\n'.join(lines[1:-1]))
    output = tmp_path / 'resolved-output'
    env = dict(os.environ, CONFIGURED_SPACE=space, CONFIGURED_URL=url,
               GITHUB_OUTPUT=str(output))
    result = subprocess.run([sys.executable, '-c', program], env=env,
                            capture_output=True, text=True, timeout=5)
    values = dict(line.split('=', 1) for line in output.read_text().splitlines()) \
        if output.exists() else {}
    return result, values


@pytest.mark.parametrize(('space', 'url', 'expected'), [
    ('', '', {'space': OLD_SPACE, 'url': OLD_URL}),
    (NEW_SPACE, NEW_URL, {'space': NEW_SPACE, 'url': NEW_URL}),
    (NEW_SPACE, NEW_URL.rstrip('/'), {'space': NEW_SPACE, 'url': NEW_URL}),
    (NEW_SPACE, '', None),
    ('', NEW_URL, None),
    (NEW_SPACE, OLD_URL, None),
    (NEW_SPACE, NEW_URL.replace('https:', 'http:'), None),
    (NEW_SPACE, NEW_URL.replace('https://', 'https://user@'), None),
    (NEW_SPACE, NEW_URL + '?source=wrong', None),
    ('einarolafsson/../wrong', NEW_URL, None),
])
def test_actual_resolver_accepts_only_a_complete_matching_target(
        workflow, tmp_path, space, url, expected):
    result, values = _resolve(workflow, tmp_path, space, url)
    if expected is None:
        assert result.returncode != 0
        assert values == {}, 'Invalid configuration must not publish either output'
    else:
        assert result.returncode == 0, result.stderr
        assert values == expected


def _publication_env(workflow, job, resolved):
    outputs = workflow['jobs']['resolve']['outputs']
    result = {}
    for name, expression in workflow['jobs'][job]['env'].items():
        if name not in {'HF_DOCS_SPACE', 'HF_DOCS_URL'}:
            continue
        output = re.fullmatch(r'\$\{\{ needs\.resolve\.outputs\.(\w+) \}\}', expression)
        assert output, f'{job}/{name} must use the resolved job output'
        field = re.fullmatch(r'\$\{\{ steps\.nightly-target\.outputs\.(\w+) \}\}',
                             outputs[output[1]])
        assert field, f'{name} must come from the actual target resolver'
        result[name] = resolved[field[1]]
    return result


@pytest.mark.parametrize('configured', [False, True])
def test_actual_publication_steps_share_the_resolved_target(
        workflow, tmp_path, configured):
    result, resolved = _resolve(workflow, tmp_path,
                                NEW_SPACE if configured else '',
                                NEW_URL if configured else '')
    assert result.returncode == 0, result.stderr
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'publish_docs_channels.py').symlink_to(ROOT / 'tools/publish_docs_channels.py')
    runner = tools / 'run_capped.sh'
    runner.write_text(f'#!{sys.executable}\n' + textwrap.dedent('''\
        import json, os, sys
        with open(os.environ['CAPTURED_COMMANDS'], 'a') as output:
            output.write(json.dumps(sys.argv[1:]) + '\\n')
        command = sys.argv[4]
        if command == 'upload-nightly':
            print('a' * 40)
        elif command == 'verify-space-tree':
            print('immutable tree verified')
        elif command == 'verify-nightly':
            print('nightly-sha')
        else:
            os.execvp(sys.argv[2], sys.argv[2:])
    '''))
    runner.chmod(0o755)
    captured = tmp_path / 'commands.jsonl'
    for branch in ('main', 'nightly'):
        site = tmp_path / 'channels' / f'docs-channel-{branch}'
        site.mkdir(parents=True)
        (site / 'index.html').write_text(f'<html><body>{branch}</body></html>')
        (site / 'publication.json').write_text(json.dumps({
            'branch': branch, 'commit': branch + '-sha'}))
    common = dict(os.environ, GITHUB_REPOSITORY='EinarOlafsson/spacr',
                  GITHUB_STEP_SUMMARY=str(tmp_path / 'summary'),
                  CAPTURED_COMMANDS=str(captured))
    common['PATH'] = str(Path(sys.executable).parent) + os.pathsep + common['PATH']
    for job, names in (
            ('deploy-pages', ['Assemble main with the nightly redirect stub and the 900 MB guard']),
            ('deploy-space', ['Assemble the standalone nightly site',
                              'Upload one complete site and verify its public deployment'])):
        env = dict(common, **_publication_env(workflow, job, resolved))
        steps = workflow['jobs'][job]['steps']
        for name in names:
            step = next(step for step in steps if step.get('name') == name)
            run = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', step['run']],
                                 cwd=tmp_path, env=env, capture_output=True,
                                 text=True, timeout=10)
            assert run.returncode == 0, run.stdout + run.stderr
    for folder in ('pages', 'space'):
        receipt = json.loads((tmp_path / folder / 'channels.json').read_text())
        assert receipt['nightly_url'] == resolved['url']
    redirect = (tmp_path / 'pages/nightly/index.html').read_text()
    assert resolved['url'] in redirect
    commands = [json.loads(line)[3:] for line in captured.read_text().splitlines()]
    assert [command[0] for command in commands] == [
        'assemble-main', 'assemble-nightly', 'upload-nightly',
        'verify-space-tree', 'verify-nightly']
    for command in commands[:2]:
        assert command[command.index('--nightly-url') + 1] == resolved['url']
    upload, tree, public = commands[2:]
    for command in (upload, tree):
        assert command[command.index('--space') + 1] == resolved['space']
    assert tree[tree.index('--revision') + 1] == 'a' * 40
    assert public[public.index('--url') + 1] == resolved['url']
