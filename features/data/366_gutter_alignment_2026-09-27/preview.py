"""Local candidate preview using GitHub's loaded CSS; NOT live acceptance."""
import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

work = Path(__file__).resolve().parent
root = Path('/mnt/wd4tb/scratch/spacr-completion/worktree')
fragment = (work / 'grid.html').read_text()
receipt = {'scope': 'Local candidate HTML4 fragment in a saved GitHub-page shell; not published GitHub acceptance.',
           'readme_sha256': hashlib.sha256((root / 'README.rst').read_bytes()).hexdigest(),
           'tolerance_css_px': 1.0, 'accepted': False, 'viewports': []}
selector = 'img[alt^="Open the "][alt$=" API"]'
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True,
                                   args=['--disable-gpu', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        page.goto('https://github.com/EinarOlafsson/spacr/tree/nightly', wait_until='domcontentloaded', timeout=60000)
        page.wait_for_selector(selector, timeout=60000)
        # Copy the page shell into a separate local document. Generated
        # markup replaces the article, never any stylesheet or image bytes.
        shell = page.content()
        (work / 'github-shell.html').write_text(shell)
        receipt['github_shell_sha256'] = hashlib.sha256(shell.encode()).hexdigest()
        page.set_content(shell, wait_until='domcontentloaded')
        page.locator('article.markdown-body').first.evaluate('(article, html) => article.innerHTML = html', fragment)
        for tile in page.locator(selector).all():
            src = tile.get_attribute('src')
            data = (root / src).read_bytes()
            import base64
            tile.evaluate('(image, src) => image.src = src', 'data:image/png;base64,' + base64.b64encode(data).decode())
        page.wait_for_function('(s) => [...document.querySelectorAll(s)].every(i => i.complete && i.naturalWidth)', arg=selector)
        page.evaluate('document.fonts.ready')
        for width in (1440, 1024):
            page.set_viewport_size({'width': width, 'height': 1100})
            rows = page.locator(selector).evaluate_all('''images => images.map(i => {
                const b=i.getBoundingClientRect(), c=getComputedStyle(i);
                return {label:i.alt.slice(9,-4), x:b.x+scrollX, y:b.y+scrollY,
                        width:b.width, height:b.height, align:c.verticalAlign};
            })''')
            assert len(rows) == 21
            for row in rows:
                # All source canvases are 512px with identical visible bounds.
                row['visible'] = {'left':row['x']+16*row['width']/512,
                                  'right':row['x']+496*row['width']/512,
                                  'top':row['y']+16*row['height']/512,
                                  'bottom':row['y']+496*row['height']/512}
            for path in (root / 'spacr/resources/icons/workflow/apps').glob('*.png'):
                with Image.open(path) as im:
                    assert im.size == (512,512) and im.getchannel('A').getbbox() == (16,16,496,496)
            horizontal = []
            for a,b in zip(rows,rows[1:]):
                if abs(a['y']-b['y']) <= 1:
                    horizontal.append(b['visible']['left']-a['visible']['right'])
            a,b = rows[6],rows[12]
            assert a['label']=='Import' and b['label']=='QC'
            vertical = b['visible']['top']-a['visible']['bottom']
            values = horizontal + [vertical]
            measured = {'viewport_width':width, 'rectangles':rows,
                        'horizontal_median':statistics.median(horizontal), 'vertical_median':vertical,
                        'spread_css_px':max(values)-min(values),
                        'accepted':min(values)>=0 and max(values)-min(values)<=1}
            receipt['viewports'].append(measured)
            page.locator('article.markdown-body').first.screenshot(path=str(work / f'local-grid-{width}.png'))
        browser.close()
        receipt['accepted']=all(row['accepted'] for row in receipt['viewports'])
        assert receipt['accepted'], receipt['viewports']
finally:
    (work / 'local-preview-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'accepted':receipt['accepted'], 'measurements':[
    {key:row[key] for key in ('viewport_width','horizontal_median','vertical_median','spread_css_px')}
    for row in receipt['viewports']]}), flush=True)
