"""Protocol choices state their experiment assumptions in the Navigator description box."""
import re

import pytest
from PySide6.QtCore import Qt

from spacr.qt.widgets.home import _EasyHomePage
from spacr.qt.widgets.sample_project import _protocol_assumptions, _protocol_options, workflow_map
from tests.qt.test_739_navigator_protocols import _choose
from tests.qt.test_739_navigator_protocols import protocol_profile as protocol_profile
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

IMPERATIVES = {
    'add', 'apply', 'check', 'choose', 'click', 'configure', 'convert', 'enter', 'import', 'inspect',
    'make', 'map', 'match', 'open', 'pick', 'press', 'review', 'run', 'select', 'set', 'start',
    'supply', 'test', 'type', 'use', 'verify',
}


def _declarative(sentence):
    """Return why a sentence is not one plain statement, or an empty string."""
    words = re.findall(r"[A-Za-z']+", sentence)
    if not sentence or not sentence[0].isupper() or not sentence.endswith('.'):
        return 'not a capitalised sentence ending in a full stop'
    if '?' in sentence or '!' in sentence:
        return 'question or exclamation'
    if {'you', 'your', "you're", 'yours'}.intersection(w.lower() for w in words):
        return 'addresses the reader'
    if words[0].lower() in IMPERATIVES:
        return 'starts with an imperative verb'
    if re.search(r'\.\s+\S', sentence):
        return 'more than one sentence'
    return ''


def _records():
    """Yield every stored protocol choice and its assumption sentence."""
    wizard = workflow_map()['protocol_wizard']
    for family, record in wizard['families'].items():
        yield family, record.get('assumption', '')
        for key, format_ in record['formats'].items():
            yield f'{family}/{key}', format_.get('assumption', '')
    for source in ('image_formats', 'experiments', 'read_modes', 'sequencers', 'barcodes'):
        for key, row in wizard[source].items():
            yield f'{source}/{key}', row.get('assumption', '')
    for field, rows in wizard['detail_assumptions'].items():
        for key, row in rows.items():
            yield f'{field}/{key}', row.get('assumption', '')


def _paths():
    """Yield complete paths that reach every option of every family, format and dependent field.

    Each path varies one field away from the first option, so every option is chosen at least
    once without walking the full product of image formats, conventions and routes.
    """
    for family in _protocol_options():
        for format_ in _protocol_options(family['key']):
            pending = [({}, False)]
            while pending:
                choices, varied = pending.pop()
                fields = _protocol_options(family['key'], format_['key'], choices)
                open_fields = [field for field in fields if field['name'] not in choices]
                if not open_fields:
                    yield family['key'], format_['key'], choices, fields
                    continue
                field = open_fields[0]
                options = field['options'][:1] if varied else field['options']
                for index, option in enumerate(options):
                    pending.append(({**choices, field['name']: option['key']}, varied or index > 0))


def test_every_stored_choice_states_one_declarative_assumption():
    """Families, formats and every dependent choice carry a plain statement."""
    records = list(_records())
    assert len(records) > 60
    problems = {key: _declarative(text) or 'ok' for key, text in records}
    assert {key: why for key, why in problems.items() if why != 'ok'} == {}
    from spacr.qt.organisms import ORGANISMS
    from spacr.regex_infer import _METADATA_CONVENTIONS

    stated = workflow_map()['protocol_wizard']['detail_assumptions']
    assert {row['key'] for row in _METADATA_CONVENTIONS} == set(stated['metadata_type'])
    assert set(ORGANISMS) | {'other'} == set(stated['organism'])


def test_every_reachable_option_shows_its_assumption_and_the_path_lists_them_in_order():
    """No offered option lacks a sentence; a complete path lists them in choice order."""
    seen = 0
    for family, format_, choices, fields in _paths():
        for field in fields:
            for option in field['options']:
                assert option['assumption'] and not _declarative(option['assumption']), (field['name'], option)
        expected = [next(row['assumption'] for row in _protocol_options() if row['key'] == family),
                    next(row['assumption'] for row in _protocol_options(family) if row['key'] == format_)]
        expected += [next(row['assumption'] for row in field['options'] if row['key'] == choices[field['name']])
                     for field in fields]
        assert _protocol_assumptions(family, format_, choices) == tuple(dict.fromkeys(expected))
        seen += 1
    assert seen > 100


def test_assumptions_follow_what_the_protocol_applies():
    """A Yokogawa format implies the CellVoyager setting; a chosen route is the only starter."""
    from spacr.qt.widgets.sample_project import _protocol_result

    result = _protocol_result('plate_images', 'tiff_yokogawa', {'goal': 'recommended'})
    assert result['action_settings']['mask']['metadata_type'] == 'cellvoyager'
    assert any('Yokogawa CellVoyager' in line for line in result['assumptions'])
    assert any('high-content' in line for line in result['assumptions'])
    routed = _protocol_result('small_images', 'tiff_other', {'metadata_type': 'imagexpress', 'goal': 'overview'})
    assert [row[0] for row in routed['actions']] == ['overview']
    assert routed['action_settings']['overview'] == {}
    assert routed['assumptions'][-2:] == ('Images use the ImageXpress / MetaXpress metadata structure.',
                                          'The analysis starts in Overview.')
    drug = _protocol_result('features', 'database', {'experiment': 'drug'})
    assert drug['action_settings']['regression']['screen_type'] == 'drug_screen'
    assert 'drug screen' in drug['assumptions'][-1]
    reads = _protocol_result('sequencing', 'fastq_gz', {'sequencer': 'ont', 'read_mode': 'single',
                                                         'single_direction': 'R2'})
    assert reads['action_settings']['map_barcodes'] == {'mode': 'single', 'sequencer': 'ont',
                                                         'single_direction': 'R2'}
    assert reads['assumptions'][-3:] == ('Reads come from an Oxford Nanopore sequencer.',
                                         'Reads are single-end.',
                                         'The barcode pattern and references match the R2 mate.')


def _box(page):
    """Return the Navigator description box lines."""
    return page.field.details.toPlainText().splitlines()


@pytest.fixture
def page(qtbot, protocol_profile):
    """Open a real Navigator page with Protocols on."""
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1000, 800)
    page.show()
    qtbot.waitExposed(page)
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    return page


def test_box_states_the_current_choice_then_the_full_list_and_follows_back(qtbot, page):
    """Each step lists the choices made so far; the end lists all of them; Back rewrites the list."""
    family = 'Images are from a mid- or high-content microscope with more than 50 images.'
    tiff = 'Images are TIFF files.'
    imagexpress = 'Images use the ImageXpress / MetaXpress metadata structure.'
    opera = 'Images use the Opera Phenix / Operetta CLS Harmony export structure.'
    yokogawa = 'Images are TIFF files that use the Yokogawa CellVoyager metadata structure.'
    _choose(qtbot, page._protocol_select, 'plate_images')
    assert _box(page) == [family]
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    assert _box(page) == [family]
    _choose(qtbot, page._protocol_select, 'tiff_other')
    assert _box(page) == [family, tiff]
    _choose(qtbot, page._protocol_detail_combos['metadata_type'], 'imagexpress')
    assert _box(page) == [family, tiff, imagexpress]
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: page._protocol_step == 2)
    assert _box(page)[:3] == [family, tiff, imagexpress]
    assert page._protocol_result['assumptions'] == (family, tiff, imagexpress)
    page.window()._protocol_browser_dialog.close()
    qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
    assert _box(page) == [family, tiff, imagexpress]
    _choose(qtbot, page._protocol_detail_combos['metadata_type'], 'opera_phenix')
    assert _box(page) == [family, tiff, opera]
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: page._protocol_step == 2)
    assert _box(page)[:3] == [family, tiff, opera] and imagexpress not in _box(page)
    page.window()._protocol_browser_dialog.close()
    qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'tiff_yokogawa')
    assert _box(page) == [family, yokogawa]
    qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
    assert _box(page) == [family]
    _choose(qtbot, page._protocol_select, 'small_images')
    small = 'Images come from a low-content experiment of about 10–50 images.'
    assert _box(page) == [small]
    _choose(qtbot, page._protocol_select, 'plate_images')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'tiff_yokogawa')
    assert _box(page) == [family, yokogawa]
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: page._protocol_step == 2)
    assert _box(page) == [family, yokogawa]
    page.window()._protocol_browser_dialog.close()


def test_requirements_follow_the_list_after_a_blank_line(qtbot, page):
    """A route that cannot start states its assumptions, then the requirement."""
    _choose(qtbot, page._protocol_select, 'sequencing')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'other')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: page._protocol_step == 2)
    stated, requirement = page.field.details.toPlainText().split('\n\n')
    assert stated.splitlines() == list(page._protocol_result['assumptions'])
    assert requirement == page._protocol_result['requirements'][0]
    assert not page.field.cards
    page.window()._protocol_browser_dialog.close()
