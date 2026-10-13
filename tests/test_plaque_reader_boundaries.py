"""Paper lookup, OCR coordinates and manual review retain their boundaries."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import plaque_papers as pp


@pytest.mark.parametrize('reference,kind,value', [
    ('pmc12345', 'pmcid', 'PMC12345'), ('12345', 'pmid', '12345'),
    ('https://doi.org/10.1234/example.', 'doi', '10.1234/example'),
])
def test_reference_spelling_preserves_the_identifier(reference, kind, value):
    assert pp.parse_reference(reference) == {'kind': kind, 'value': value}


def test_invalid_reference_is_not_sent_to_a_paper_service():
    with pytest.raises(ValueError, match='not a DOI'):
        pp.parse_reference('not a reference')


def test_malformed_search_responses_are_retried_before_returning_a_hit(monkeypatch):
    pauses, queries = [], []
    hit = {'doi': '10.1234/example'}
    def invalid_json():
        raise ValueError('truncated JSON')
    replies = iter((SimpleNamespace(status_code=503),
                    SimpleNamespace(status_code=200, json=invalid_json),
                    SimpleNamespace(status_code=200, json=lambda: {'resultList': {'result': [hit]}})))
    def fetch(url, **kwargs):
        queries.append(kwargs['params']['query'])
        return next(replies)
    monkeypatch.setattr(pp.time, 'sleep', pauses.append)
    assert pp._epmc_hit('DOI:example', fetch) == hit
    assert pauses == [1.0, 2.0] and queries == ['DOI:example'] * 3


def test_search_retry_exhaustion_returns_no_invented_paper(monkeypatch):
    pauses = []
    monkeypatch.setattr(pp.time, 'sleep', pauses.append)
    response = SimpleNamespace(status_code=200, json=lambda: {})
    assert pp._epmc_hit('unknown', lambda *a, **kw: response) is None
    assert pauses == [1.0, 2.0, 3.0]


@pytest.mark.parametrize('status,content', [(404, b''), (200, b'not a zip archive')])
def test_unavailable_or_corrupt_figure_bundle_yields_no_images(tmp_path, status, content):
    paper = pp.Paper(key='PMC123', source='europepmc', pmcid='PMC123')
    replies = iter((SimpleNamespace(status_code=200, content=b'<broken xml'),
                    SimpleNamespace(status_code=status, content=content)))
    assert pp.fetch_figures(paper, tmp_path, get=lambda *a, **kw: next(replies)) == []
    assert list(tmp_path.iterdir()) == []
    assert pp.fetch_figures(pp.Paper(key='unknown', source='europepmc'), tmp_path,
        get=lambda *a, **kw: pytest.fail('request without a PMC identifier')) == []


def test_ocr_boxes_are_sorted_in_image_coordinates_and_paths_are_plain_strings(tmp_path):
    path = tmp_path / 'figure.png'
    def engine(image):
        assert image == str(path)
        return [([[20, 30], [10, 30], [10, 40], [20, 40]], 'lower', .7),
                ([[8, 2], [2, 2], [2, 6], [8, 6]], 'upper', .9)], None
    words = pp.read_words(path, engine=engine)
    assert [(w.text, w.x0, w.y0, w.x1, w.y1, w.confidence) for w in words] == [
        ('upper', 2., 2., 8., 6., .9), ('lower', 10., 30., 20., 40., .7)]
    assert pp.read_words(path, engine=lambda image: (None, None)) == []


def test_enlarged_ocr_returns_to_original_coordinates_without_duplicate_words():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    region = pp.Region(20, 20, 40, 40)
    known = pp.Word('known', 1, 1, 4, 4)
    shapes = []
    def engine(crop):
        shapes.append(crop.shape)
        return [([[3, 3], [12, 3], [12, 12], [3, 12]], 'duplicate', .8),
                ([[60, 30], [75, 30], [75, 45], [60, 45]], 'new', .9)], None
    found = pp.reread_around(image, [region], [known], engine=engine, scale=3)
    assert shapes == [(150, 150, 3)]
    assert found == [known, pp.Word('new', 20, 10, 25, 15, .9)]
    assert pp.reread_around(image, [], [known], engine=engine) == [known]
    assert pp.reread_around(image, [pp.Region(0, 0, 1, 1)], [], engine=engine) == []
    assert len(shapes) == 1


@pytest.mark.parametrize('ready,in_process,expected', [
    (True, False, '/isolated/reader'), (False, False, None), (True, True, None),
])
def test_reader_environment_is_used_only_when_ready_and_separate(monkeypatch, ready, in_process, expected):
    from spacr import _segmentation_backends as backends

    monkeypatch.setattr(backends, '_backend_state', lambda key: SimpleNamespace(
        ready=ready, in_process=in_process, env='/isolated/reader'))
    assert pp.reader_environment() == expected


def test_failed_environment_probe_is_unavailable_not_an_import_crash(monkeypatch):
    from spacr import _segmentation_backends as backends

    def fail(key):
        raise OSError('backend registry unavailable')
    monkeypatch.setattr(backends, '_backend_state', fail)
    assert pp.reader_environment() is None
    with pytest.raises(ImportError, match='Install it'):
        pp._reader_request('read_text', np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize('error', [ImportError, ValueError])
def test_broken_optional_module_specs_are_treated_as_unavailable(monkeypatch, error):
    import importlib.util

    def fail(name):
        raise error('broken module specification')
    monkeypatch.setattr(importlib.util, 'find_spec', fail)
    assert not pp._importable('optional_reader')


def test_ocr_uses_the_isolated_reader_or_reports_installation_needed(monkeypatch):
    monkeypatch.setattr(pp, '_ENGINE', {})
    monkeypatch.setattr(pp, '_importable', lambda name: False)
    monkeypatch.setattr(pp, 'reader_environment', lambda: '/isolated/reader')
    assert pp._rapidocr() is pp._reader_ocr
    monkeypatch.setattr(pp, '_reader_request', lambda op, image: {'words': [['box', 'text', .9]]})
    assert pp._reader_ocr('image') == ([('box', 'text', .9)], None)
    monkeypatch.setattr(pp, 'reader_environment', lambda: None)
    with pytest.raises(ImportError, match='RapidOCR'):
        pp._rapidocr()


def test_local_ocr_engine_is_constructed_once_and_reused(monkeypatch):
    calls, engine = [], object()
    def factory():
        calls.append('created')
        return engine
    monkeypatch.setattr(pp, '_ENGINE', {})
    monkeypatch.setattr(pp, '_importable', lambda name: True)
    monkeypatch.setitem(sys.modules, 'rapidocr_onnxruntime', SimpleNamespace(RapidOCR=factory))
    assert pp._rapidocr() is engine and pp._rapidocr() is engine
    assert calls == ['created']


def test_console_review_distinguishes_accept_skip_and_manual_correction(tmp_path, capsys):
    figure = pp.Figure(path=tmp_path/'figure.png')
    annotations = [pp.Annotation(pp.Region(1, 2, 3, 4), panel='A', condition='original', conflict=True)
                   for _ in range(3)]
    answers = iter(('', 'S', '  corrected condition  '))
    assert pp.console_review(figure, annotations, ask=lambda prompt: next(answers)) is annotations
    assert [a.approved for a in annotations] == [True, False, True]
    assert annotations[0].condition == 'original'
    assert (annotations[2].condition, annotations[2].source, annotations[2].strength) == (
        'corrected condition', 'manual', 'manual')
    assert 'CONFLICT' in capsys.readouterr().out
    assert pp.console_legend_prompt(figure, annotations, ask=lambda prompt: '  legend  ') == 'legend'
    assert pp.console_legend_prompt(figure, [], ask=lambda prompt: '  ') is None
    copied = pp.annotation_as_dict(annotations[2])
    copied['region']['x0'] = 100
    assert annotations[2].region.x0 == 1


def test_command_line_preserves_review_and_inference_choices(tmp_path, monkeypatch, capsys):
    requests = []
    def measure(refs, dest, **options):
        requests.append((refs, dest, options))
        return {'papers': 2, 'regions': 3}
    monkeypatch.setattr(pp, 'measure_plaques_from_papers', measure)
    assert pp.main(['PMC123', '12345', '--dst', str(tmp_path), '--detector', 'detector',
        '--segmenter', 'segmenter', '--imgsz', '320,640,', '--plate-format', '6-well', '--confirm-each']) == 0
    assert requests == [(['PMC123', '12345'], str(tmp_path), dict(detector='detector',
        segmenter='segmenter', imgsz=(320, 640), plate_format='6-well', confirm_each=True))]
    assert json.loads(capsys.readouterr().out) == {'papers': 2, 'regions': 3}
