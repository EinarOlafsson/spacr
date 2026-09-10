#!/usr/bin/env python3
"""Capture current public installation sources without changing page content.

Fresh anonymous Chromium context, one page at a time, native 4K screenshots.
No text replacement, hidden badges, logged-in profile or inferred releases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

from playwright.sync_api import sync_playwright
from stage_lesson import DEFAULT_STAGE, REPO, write


def public_json(url):
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    return json.loads(raw), {'url': url, 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-name', default='installation_sources_current')
    args = parser.parse_args()
    if Path(args.capture_name).name != args.capture_name or args.capture_name in ('.', '..'):
        raise ValueError('Expected one private capture directory name')
    output = DEFAULT_STAGE / 'captures' / args.capture_name
    if output.exists():
        raise RuntimeError('Preserve the preceding capture; choose a new output identity')
    output.mkdir(parents=True)
    pypi, pypi_source = public_json('https://pypi.org/pypi/spacr/json')
    release, release_source = public_json('https://api.github.com/repos/EinarOlafsson/spacr/releases/latest')
    conda, conda_source = public_json('https://api.anaconda.org/package/conda-forge/spacr')
    proof = {'completed_capture': False, 'app_source_modified': False,
             'page_content_modified': False, 'published': False,
             'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             'pypi': dict(pypi_source, version=pypi['info']['version'],
                          project_links=pypi['info']['project_urls']),
             'github': dict(release_source, tag=release['tag_name'],
                            assets=[{'name': x['name'], 'url': x['browser_download_url']}
                                    for x in release['assets']]),
             'conda': dict(conda_source, version=conda['latest_version'])}
    frames = {}
    write(output / 'provenance.json', proof)
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080},
                                      device_scale_factor=2, locale='en-US', color_scheme='dark')
        page = context.new_page()
        page.set_default_timeout(30000)

        def visit(url):
            response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
            if response is None or not response.ok:
                raise RuntimeError(f'Public page refused: {url}')
            page.evaluate('document.fonts.ready')
            page.wait_for_timeout(1500)

        def capture(name, target):
            target.evaluate("element => element.scrollIntoView({block: 'center', inline: 'nearest'})")
            page.wait_for_timeout(400)
            box = target.bounding_box()
            if not box or min(box['width'], box['height']) <= 0:
                raise RuntimeError('The narrated element is not visibly measurable')
            rect = [round(box[k] * 2) for k in ('x', 'y', 'width', 'height')]
            if rect[0] < 0 or rect[1] < 0 or rect[0] + rect[2] > 3840 or rect[1] + rect[3] > 2160:
                raise RuntimeError(f'The narrated element is clipped: {name} {rect}')
            path = output / f'{name}.png'
            page.screenshot(path=str(path))
            frames[name] = {'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'focus': rect, 'url': page.url, 'visible_text': target.inner_text(), 'buttons': []}
            write(output / 'frames.json', frames)
            print(f'Captured {name}: {page.url}', flush=True)

        try:
            visit('https://pypi.org/project/spacr/')
            header = page.locator('h1').filter(has_text='spacr').first
            if pypi['info']['version'] not in header.inner_text():
                raise RuntimeError('Live PyPI header and JSON metadata disagree')
            capture('01_pypi_current_release', header.locator('..'))
            links = page.get_by_role('heading', name='Project links', exact=True)
            capture('02_pypi_project_links', links.locator('..'))
            visit('https://github.com/EinarOlafsson/spacr/tree/main')
            branch = page.get_by_role('button', name='main branch', exact=False)
            if branch.count() != 1:
                branch = page.locator('button').filter(has_text='main').first
            capture('03_github_main_branch', branch)
            visit(release['html_url'])
            assets = page.locator('summary').filter(has_text='Assets').first
            assets.scroll_into_view_if_needed()
            details = assets.locator('..')
            if details.get_attribute('open') is None:
                assets.click()
            for item in release['assets']:
                page.get_by_role('link', name=item['name'], exact=True).wait_for(state='visible')
            capture('04_github_current_assets', details)
            visit('https://anaconda.org/conda-forge/spacr')
            page.get_by_text(conda['latest_version'], exact=False).first.wait_for(state='visible')
            capture('05_conda_current_package', page.get_by_role('heading').filter(has_text='spacr').first.locator('..'))
            proof['completed_capture'] = True
            proof['frames'] = len(frames)
        except Exception as error:
            proof['failure'] = str(error)
            page.screenshot(path=str(output / '99_capture_failure.png'))
            raise
        finally:
            write(output / 'provenance.json', proof)
            browser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
