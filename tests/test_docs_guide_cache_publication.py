"""Guide compiler caches cannot consume the documentation publication budget."""
import json

from tests.test_docs_channels import site
from publish_docs_channels import assemble_main, assemble_nightly, prepare


def test_guides_fit_the_same_budget_without_losing_channel_assets(tmp_path):
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly')
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'api': {}}))
    for branch, root in [('main', main), ('nightly', nightly)]:
        for language in ['sv', 'de', 'es', 'pt', 'fr', 'is', 'zh_CN', 'ko', 'hi']:
            cache = root / ('.doctrees-guide-' + language)
            cache.mkdir()
            (cache / 'environment.pickle').write_bytes(b'x' * 4096)
        assets = root / 'analysis'
        assets.mkdir()
        (assets / 'measurements.csv').write_text('object_id,area\n1,42\n')
        (root / '.doctrees-guide-not-a-directory').write_text('preserve this file')
        prepare(root, report, branch, branch + '-sha')
    pages, space = tmp_path / 'published', tmp_path / 'space'
    receipt = assemble_main(main, pages, limit=16384)
    assert receipt['size_bytes'] <= 16384
    assemble_nightly(nightly, space)
    for root, branch in [(pages, 'main'), (space, 'nightly')]:
        assert (root / 'analysis/measurements.csv').read_text() == 'object_id,area\n1,42\n'
        assert (root / '.doctrees-guide-not-a-directory').read_text() == 'preserve this file'
        assert (root / 'tutorials/lesson_catalog.js').read_text() == branch + ' lessons'
        manifest = json.loads((root / 'tutorials/published-media.json').read_text())
        assert (root / 'tutorials' / manifest['lesson/video.mp4']).read_bytes() == b'same recording'
    assert all((root / '.doctrees-guide-sv/environment.pickle').exists() is False
               for root in [main, nightly])
