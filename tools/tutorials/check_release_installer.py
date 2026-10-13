#!/usr/bin/env python3
"""Download the exact public Linux installer and check its release checksum.

This is preparation for lesson 04, not proof of installation. Never execute
an unverified asset, invent a checksum, or compare against a different channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from stage_lesson import DEFAULT_STAGE, write


def checksum_for(text, name):
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+)', line.strip())
        if match and match[2] == name:
            values.append(match[1].lower())
    if len(values) != 1:
        raise ValueError('Exactly one checksum for this precise installer is required')
    return values[0]


def release_asset(assets, filename, tag):
    matches = [item for item in assets if item.get('name') == filename]
    if len(matches) != 1:
        raise ValueError('The exact release asset is missing or ambiguous')
    item = matches[0]
    expected = f'https://github.com/EinarOlafsson/spacr/releases/download/{tag}/{filename}'
    if item.get('browser_download_url') != expected or not 0 < item.get('size', 0) < 30 * 1024**2:
        raise ValueError('Unexpected release URL or online-installer size')
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--tag', default='v1.5.0.5')
    args = parser.parse_args()
    if not re.fullmatch(r'v\d+(?:\.\d+)+', args.tag):
        parser.error('Use an explicit numbered release tag')
    parent = args.stage / 'installation_runs'
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='linux-installer-', dir=parent))
    receipt = dict(accepted=False, installation_performed=False, folder=str(root), tag=args.tag)
    write(root / 'receipt.json', receipt)
    print('Private release verification: ' + str(root), flush=True)

    def fetch(url):
        if urlparse(url).scheme != 'https':
            raise ValueError('Release data must use HTTPS')
        with urlopen(Request(url, headers={'User-Agent': 'spaCR-tutorial-verification'}), timeout=90) as response:
            return response.read(30 * 1024**2 + 1)

    endpoint = f'https://api.github.com/repos/EinarOlafsson/spacr/releases/tags/{args.tag}'
    raw = fetch(endpoint)
    release = json.loads(raw)
    if release.get('tag_name') != args.tag or release.get('draft'):
        raise ValueError('The named public release is not available')
    write(root / 'github-release.json', release)
    filename = f'spaCR-{args.tag[1:]}-Linux-x86_64-Online.run'
    manifest_asset = release_asset(release['assets'], 'SHA256SUMS.txt', args.tag)
    installer_asset = release_asset(release['assets'], filename, args.tag)
    manifest = fetch(manifest_asset['browser_download_url'])
    expected = checksum_for(manifest.decode(), filename)
    payload = fetch(installer_asset['browser_download_url'])
    actual = hashlib.sha256(payload).hexdigest()
    if len(payload) != installer_asset['size'] or actual != expected:
        raise ValueError('Downloaded installer does not match its exact release checksum')
    target = root / filename
    target.write_bytes(payload)
    (root / 'SHA256SUMS.txt').write_bytes(manifest)
    receipt.update(accepted=True, github_response_sha256=hashlib.sha256(raw).hexdigest(),
                   installer=str(target), installer_bytes=len(payload), sha256=actual,
                   checksum_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                   scope='Downloaded bytes match exact public release; no installation or GUI success claimed')
    # This .run is a plain Bash script, NOT a Makeself archive. Its own
    # --help/--dry-run contract and Bash syntax are the correct preflight.
    env = dict(os.environ)
    for key in ('SPACR_TORCH_BACKEND', 'SPACR_PACKAGE_SPEC', 'SPACR_INSTALL_DRY_RUN', 'SPACR_LAUNCHER_DIR'):
        env.pop(key, None)
    env.update(SPACR_INSTALL_LANGUAGE='en', XDG_DATA_HOME=str(root / 'data'))
    commands = [(['bash', '-n', str(target)], None),
                (['bash', str(target), '--help'], '--torch-backend'),
                (['bash', str(target), '--dry-run', '--install-root', str(root / 'runtime'),
                  '--launcher-dir', str(root / 'bin'), '--skip-system-deps', '--no-launch'], 'DRY RUN:')]
    checks = []
    for command, expected_text in commands:
        result = subprocess.run(command, cwd=root, env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=60)
        checks.append(dict(command=command, returncode=result.returncode,
                           output=result.stdout))
        if result.returncode or (expected_text and expected_text not in result.stdout):
            receipt['accepted'] = False
    receipt['script_preflight_checks'] = checks
    write(root / 'receipt.json', receipt)
    if not receipt['accepted']:
        raise RuntimeError('Script syntax/help/dry-run check failed')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
