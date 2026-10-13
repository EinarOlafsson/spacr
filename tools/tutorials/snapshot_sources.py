#!/usr/bin/env python3
"""Checkpoint tutorial authoring text without putting models or media in Git.

Run from the spaCR repository. The external workspace is the editing authority;
this reproducible, reviewable copy protects its sources at each Git checkpoint.
The manifest also reports published/source differences, which must be reconciled
deliberately rather than overwritten by a routine publisher invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
TEXT_SUFFIXES = {'.py', '.json', '.js', '.css', '.html', '.md', '.txt'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(workspace: Path, destination: Path) -> dict:
    files = []
    for directory in ('tools', 'tests', 'catalog', 'web', 'localization/reviewed'):
        for path in sorted((workspace / directory).rglob('*')):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in TEXT_SUFFIXES
                    and not {'__pycache__', '.pytest_cache'} & set(path.parts)):
                files.append(path)
    files.extend(p for p in (workspace / 'TUTORIAL_ROADMAP.md',) if p.is_file())
    records = []
    differences = []
    published = REPO / 'docs/source/_extra/tutorials'
    for source in files:
        relative = source.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != digest(source):
            shutil.copy2(source, target)
        record = {'path': relative.as_posix(), 'sha256': digest(source),
                  'bytes': source.stat().st_size}
        records.append(record)
        if relative.parts[0] in {'web', 'catalog'}:
            compare = (published / Path(*relative.parts[1:])
                       if relative.parts[0] == 'web' else published / relative)
            if compare.is_file() and digest(compare) != record['sha256']:
                differences.append({'source': relative.as_posix(),
                                    'published': str(compare.relative_to(REPO)),
                                    'published_sha256': digest(compare)})
    result = {'schema': 1, 'workspace': str(workspace), 'files': records,
              'published_differences': differences}
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / 'authoring-manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).parent / 'authoring')
    args = parser.parse_args()
    result = snapshot(args.workspace.resolve(), args.destination.resolve())
    print(f"Checkpointed {len(result['files'])} source files; "
          f"{len(result['published_differences'])} source/published differences.")
