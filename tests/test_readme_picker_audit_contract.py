"""The docs audit checks the actual generated HTML picker, inside its markers."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_documentation_i18n as builder

BEGIN = '.. spacr-language-picker-begin'
END = '.. spacr-language-picker-end'
GENERATED = (ROOT / 'docs/i18n/readme/README.sv.rst').read_text(encoding='utf-8')
PICKER = GENERATED[GENERATED.index(BEGIN):GENERATED.index(END) + len(END)]
INDEX = '[English](../../../README.rst)\n'


@pytest.fixture
def audit_readme(tmp_path, monkeypatch, capsys):
    """Keep the real audit gate; isolate unrelated catalog/image inventory work."""
    api = tmp_path / 'api'
    readme = tmp_path / 'readme'
    api.mkdir()
    readme.mkdir()
    (api / 'en.json').write_text(json.dumps({'schema': 2, 'symbols': {}}), encoding="utf-8")
    (api / 'sv.json').write_text(json.dumps({
        'schema': 2, 'symbols': {},
        'secondary_generator': builder.SECONDARY_MODEL,
        'secondary_license': builder.SECONDARY_LICENSE,
    }), encoding="utf-8")
    monkeypatch.setattr(builder, 'API_DIR', api)
    monkeypatch.setattr(builder, 'README_DIR', readme)
    monkeypatch.setattr(builder, 'reviewed_api_block_translations', lambda *_: {})
    monkeypatch.setattr(builder.build_readme_images, 'check_structure', lambda *_: [])
    monkeypatch.setattr(builder.build_readme_images, 'check', lambda *_: [])

    def run(text, index=INDEX):
        (readme / 'README.sv.rst').write_text(text, encoding='utf-8')
        if index is not None:
            (readme / 'README.md').write_text(index, encoding='utf-8')
        result = builder.audit({}, ['sv'])
        captured = capsys.readouterr()
        return result, captured.err

    return run


def test_actual_generated_html_picker_passes_audit(audit_readme):
    assert '<a href="README.md"><picture>' in PICKER
    assert ':target: README.md' not in PICKER
    assert audit_readme(GENERATED) == (0, '')


def test_normal_picker_accepts_equivalent_html_quote_syntax(audit_readme):
    assert audit_readme(PICKER.replace('href="README.md"', "href='README.md'")) == (0, '')


@pytest.mark.parametrize('text', [
    PICKER.replace('href="README.md"', 'href="README.fr.rst"'),
    PICKER.replace('href="README.md"', ''),
    PICKER.replace('href="README.md"', 'href="README.md.backup"'),
    PICKER.replace('href="README.md"', 'data-href="README.md"'),
    PICKER.replace('href="README.md"', 'href="wrong" href="README.md"'),
    PICKER.replace('href="README.md"', 'href="wrong"') + '\n<a href="README.md"><picture><img src="x.svg"></picture></a>',
    PICKER.replace('href="README.md"', 'href="wrong"') + '\n.. image:: outside.svg\n   :target: README.md\n',
    PICKER.replace(BEGIN, ''),
    PICKER.replace(END, ''),
    PICKER + '\n' + PICKER,
    END + '\n' + PICKER.replace(END, ''),
    PICKER.replace(BEGIN, BEGIN + '-not-a-marker'),
    PICKER.replace('.. raw:: html', '.. code-block:: html'),
    PICKER.replace('.. raw:: html', ''),
    PICKER.replace('</a>', ''),
    PICKER.replace('</picture>', ''),
    PICKER.replace('<a href="README.md">', '<![unexpected]><a href="README.md">'),
    PICKER.replace('<a href="README.md">', '&lt;a href="README.md"&gt;').replace('</a>', '&lt;/a&gt;'),
], ids=[
    'wrong-target', 'missing-href', 'misleading-target-substring', 'data-href-is-not-link',
    'duplicate-href', 'only-outside-anchor', 'only-outside-rst-target',
    'missing-begin', 'missing-end', 'duplicate-picker', 'reversed-markers',
    'marker-substring', 'code-instead-of-raw', 'no-raw-directive',
    'unclosed-anchor', 'unclosed-picture', 'unknown_html_declaration', 'escaped-anchor',
])
def test_broken_picker_cannot_pass_on_incidental_text(audit_readme, text):
    code, errors = audit_readme(text)
    assert code == 1
    assert 'sv: the language picker link is broken' in errors


@pytest.mark.parametrize('index', [None, '# Languages\n', '[English](wrong.rst)\n'])
def test_valid_picker_still_requires_the_english_index_destination(audit_readme, index):
    code, errors = audit_readme(PICKER, index)
    assert code == 1
    assert 'sv: the language index no longer reaches the English README' in errors
    assert 'the language picker link is broken' not in errors
