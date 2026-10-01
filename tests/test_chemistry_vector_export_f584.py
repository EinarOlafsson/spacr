"""Exercise real RDKit vector exports through the existing SAR report action."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from spacr import sp_stats as sp
from tests.test_chemistry_aware_hits import _scored, _screen

pytest.importorskip('rdkit')
SVG = '{http://www.w3.org/2000/svg}'


def test_report_exports_real_vector_bonds_and_exact_compound_metadata(tmp_path, monkeypatch):
    # The PNG writer is unchanged. Keep this vector integration check bounded
    # independently of the user's configured bitmap export size and DPI.
    def save_bitmap(figure, filename, **kwargs):
        Path(filename).write_bytes(b'bitmap export delegated')
        return str(filename)

    monkeypatch.setattr('spacr.plot.save_figure', save_bitmap)
    frame, layout, host = _screen()
    result = sp._structure_activity(_scored(frame), layout, host=host)
    before = result.sar.copy(deep=True)
    paths = sp._write_sar_report(result, tmp_path, target='print')
    root = ET.parse(paths['hit_structures_svg']).getroot()
    assert root.tag == SVG + 'svg'
    assert root.findall('.//' + SVG + 'path')
    assert not root.findall('.//' + SVG + 'image')
    records = json.loads(root.find(SVG + 'metadata').text)['structures']
    expected = result.sar[result.sar.hit].sort_values(['cluster', 'best_rank'])
    assert [row['compound'] for row in records] == expected.compound.tolist()
    assert [row['cluster'] for row in records] == expected.cluster.tolist()
    assert [row['potency'] for row in records] == expected.potency.tolist()
    assert next(row for row in records if row['compound'] == 'cpd3')['cytotoxicity_index'] == 80
    pd.testing.assert_frame_equal(result.sar, before)
    assert Path(paths['hit_structures']).read_bytes() == b'bitmap export delegated'


def test_names_are_xml_safe_and_metadata_keeps_full_unicode_names(tmp_path):
    frame, layout, host = _screen()
    result = sp._structure_activity(_scored(frame), layout, host=host)
    name = 'α <compound> & "quoted" ' + 'a long name ' * 20
    result.sar.loc[result.sar.hit, 'compound'] = name
    path = sp._write_hit_structures_svg(result, tmp_path)
    root = ET.parse(path).getroot()
    records = json.loads(root.find(SVG + 'metadata').text)['structures']
    assert all(row['compound'] == name for row in records)
    assert '<compound>' not in Path(path).read_text()


def test_empty_and_invalid_hits_do_not_create_artwork(tmp_path):
    frame, layout, host = _screen()
    result = sp._structure_activity(_scored(frame), layout, host=host)
    result.sar['smiles_valid'] = False
    assert sp._write_hit_structures_svg(result, tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_sheet_uses_same_limit_and_order_as_display(tmp_path):
    frame, layout, host = _screen()
    result = sp._structure_activity(_scored(frame), layout, host=host)
    result.sar = pd.concat([result.sar[result.sar.hit]] * 10, ignore_index=True)
    path = sp._write_hit_structures_svg(result, tmp_path)
    records = json.loads(ET.parse(path).getroot().find(SVG + 'metadata').text)['structures']
    assert len(records) == sp._STRUCTURE_LIMIT


def test_failed_atomic_replace_preserves_previous_svg(tmp_path, monkeypatch):
    frame, layout, host = _screen()
    result = sp._structure_activity(_scored(frame), layout, host=host)
    path = tmp_path / 'hit_structures.svg'
    path.write_text('previous vector export')

    def denied(*args):
        raise OSError('destination unavailable')

    monkeypatch.setattr('os.replace', denied)
    with pytest.raises(OSError, match='destination unavailable'):
        sp._write_hit_structures_svg(result, tmp_path)
    assert path.read_text() == 'previous vector export'
    assert list(tmp_path.iterdir()) == [path]
