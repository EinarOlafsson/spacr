"""Preflight catalog runs before writing a serial GNOME capture queue.

Usage: make_host_jobs.py QUEUE PREFIX TAG LESSON_ID [LESSON_ID ...]
Catalog entries are checked together before any job is written. This only
checks command prerequisites; it does not certify inputs, clips or lessons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/tutorials')]

DEFAULT_ROOT = Path('/media/carruthers/mnt3/claude')
PATH_OPTIONS = frozenset({
    '--inputs', '--xvfb', '--external-input-root', '--classifier-existing-split',
    '--batch-input-root', '--evaluation-from', '--diagnostics-from',
    '--barcode-reference-source', '--sweep-input-root',
})


class QueuePreflightError(ValueError):
    """The selected queue has unresolved or missing local prerequisites."""


def validate_run(env, run_native, capture, *, cwd: Path):
    """Return all unresolved command tokens and missing concrete input paths.

    Bind destinations are created by the recorder; only bind sources must
    exist. URLs and ordinary labels are not local paths. Explicit input
    options also validate relative paths against the invocation directory.
    """
    errors = []
    for key, value in env.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            errors.append(f'invalid environment key: {key!r}')
        if '<' in value or '>' in value:
            errors.append(f'env {key}: unresolved placeholder {value!r}')
    for section, args in (('run_native', run_native), ('capture', capture)):
        for value in args:
            if '<' in value or '>' in value:
                errors.append(f'{section}: unresolved placeholder {value!r}')

    def require(value, context, *, directory=False):
        if '<' in value or '>' in value:
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = cwd / path
        if not path.exists():
            errors.append(f'{context}: missing local path {path}')
        elif directory and not path.is_dir():
            errors.append(f'{context}: expected input directory {path}')

    for key, value in env.items():
        if value.startswith(('/', '~/')):
            require(value, f'env {key}')
    for section, args in (('run_native', run_native), ('capture', capture)):
        index = 0
        while index < len(args):
            token = args[index]
            option, separator, value = token.partition('=')
            if option == '--ro-bind' or option in PATH_OPTIONS:
                if not separator:
                    index += 1
                    if index >= len(args) or args[index].startswith('--'):
                        errors.append(f'{section} {option}: missing argument')
                        continue
                    value = args[index]
                if option == '--ro-bind':
                    source, binding, target = value.partition('=')
                    relative = PurePosixPath(target)
                    if (not binding or not source or not relative.parts
                            or relative.is_absolute() or '..' in relative.parts):
                        errors.append(f'{section} --ro-bind: expected SOURCE=RELATIVE, got {value!r}')
                    else:
                        require(source, f'{section} --ro-bind source')
                else:
                    require(value, f'{section} {option}', directory=option == '--inputs')
            else:
                local = value if separator and option.startswith('--') else token
                if local.startswith(('/', '~/')):
                    require(local, f'{section} {option if separator else "argument"}')
            index += 1
    return list(dict.fromkeys(errors))


def resolve_existing_mapping(arg, *, machine_root, host_root):
    """Preserve the established explicit input-tree mapping; invent none."""
    inputs = host_root / 'inputs'
    placeholders = {
        '<INPUTS_DIR containing annotate_fresh': inputs / 'annotate_fresh',
        '<INPUTS_DIR with captures/.../run_compare': inputs / 'run_compare',
        '<INPUTS_DIR with pipeline_graph_state': inputs / 'pipeline_graph',
        '<INPUTS_DIR with captures/foreign_release_v2': inputs / 'foreign_release',
        '<INPUTS_DIR with regression_runs/example-9wut6lcv': inputs / 'volcano',
        '<INPUTS_DIR with captures/regression_release': inputs / 'regression_release',
        '<INPUTS_DIR with regression_runs/<project>': inputs / 'regression_runs',
        '<prepared dir containing Cellpose_training': inputs / 'cellpose_training',
        '<prepared dir containing example_data/plate1/measurements': inputs / 'plate_viewer',
    }
    for key, value in placeholders.items():
        if arg.startswith(key):
            return str(value)
    if arg.startswith(('n676_stages/', 'toxoplasma_projects/')):
        return str(machine_root / arg)
    return arg.replace('=n676_stages/', f'={machine_root}/n676_stages/')


def write_jobs(queue, prefix, tag, lessons, *, repo, host_root, prototype_root,
               machine_root=DEFAULT_ROOT, launcher=None):
    """Validate the entire selection, then emit jobs in existing catalog order."""
    tutorials = repo / 'tools/tutorials'
    live = Path(launcher) if launcher is not None else prototype_root / 'idle_retest/live.sh'
    if not live.is_absolute():
        live = repo / live
    catalog = {}
    for path in sorted(tutorials.glob('host_jobs_*.json')):
        for key, runs in json.loads(path.read_text()).items():
            if not key.startswith('_'):
                catalog.setdefault(key, []).extend(runs)
    errors, skipped, jobs = [], [], []
    if not queue.is_dir():
        errors.append(f'queue directory does not exist: {queue}')
    if not live.is_file():
        errors.append(f'capture launcher does not exist: {live}')
    for lesson in lessons:
        source = tutorials / 'lessons' / f'{lesson}.json'
        plan = host_root / 'plans' / f'{lesson}-clips.json'
        if not source.is_file():
            errors.append(f'{lesson}: lesson source does not exist: {source}')
            continue
        if lesson not in catalog:
            errors.append(f'{lesson}: no catalog runs; requires its own workflow')
            continue
        visuals = list(dict.fromkeys(s['visual'] for s in json.loads(source.read_text())['scenes']))
        if not plan.is_file():
            errors.append(f'{lesson}: clip plan does not exist: {plan}')
        for run in catalog[lesson]:
            if run.get('skip') or not run.get('capture'):
                skipped.append((lesson, run['name'], run.get('note', '')[:80]))
                continue
            native = [resolve_existing_mapping(str(a), machine_root=machine_root, host_root=host_root)
                      for a in run.get('run_native', [])]
            env = {k: resolve_existing_mapping(str(v), machine_root=machine_root, host_root=host_root)
                   for k, v in run.get('env', {}).items()}
            capture = [str(a) for a in run['capture']]
            problems = validate_run(env, native, capture, cwd=repo)
            if problems:
                errors.extend(f'{lesson}/{run["name"]}: {problem}' for problem in problems)
                continue
            stage = host_root / f'{tag}-{lesson[:2]}-{run["name"]}'
            path = queue / f'{prefix}{len(jobs) + 1:02d}.job'
            if path.exists():
                errors.append(f'job already exists: {path}')
            line = ' '.join([
                *(f'{k}={shlex.quote(v)}' for k, v in env.items()),
                shlex.quote(str(live)), shlex.quote(str(stage)),
                *map(shlex.quote, native), '--', *map(shlex.quote, capture),
                '--clip-durations', shlex.quote(str(plan)),
                '--record-scenes', shlex.quote(','.join(visuals)),
            ])
            jobs.append((path, line, lesson, run['name']))
    if errors:
        raise QueuePreflightError('Queue preflight failed; no jobs written:\n' + '\n'.join(errors))
    for path, line, lesson, name in jobs:
        with path.open('x') as handle:
            handle.write(line + '\n')
        print(path.stem, lesson, name)
    for item in skipped:
        print('SKIPPED', *item)
    return [path for path, _line, _lesson, _name in jobs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    parser.add_argument('prefix')
    parser.add_argument('tag')
    parser.add_argument('lessons', nargs='+')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--machine-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--host-root', type=Path, default=DEFAULT_ROOT / 'n676_stages/host')
    parser.add_argument('--prototype-root', type=Path, default=DEFAULT_ROOT / 'capture_host_proto')
    parser.add_argument('--launcher', type=Path,
                        help='Explicit capture launcher; defaults to prototype-root/idle_retest/live.sh')
    args = parser.parse_args(argv)
    try:
        write_jobs(args.queue, args.prefix, args.tag, args.lessons, repo=args.repo,
                   host_root=args.host_root, prototype_root=args.prototype_root,
                   machine_root=args.machine_root, launcher=args.launcher)
    except QueuePreflightError as error:
        parser.exit(2, str(error) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
