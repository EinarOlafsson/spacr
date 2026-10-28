"""Only the microscope metadata field owns its indirect status captions."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def extract_metadata(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    builder = importlib.import_module('build_i18n_catalogs')
    root = tmp_path / 'spacr' / 'qt' / 'screens'
    root.mkdir(parents=True)
    for name in (
            'model_compare.py', 'embeddings.py', 'settings.py', 'object.py',
            'remote_execution.py', 'install_cleanup.py', 'data_manager.py', 'doctor.py', 'updater.py',
            'model_zoo.py', 'utils.py', 'convert.py', 'image_import.py',
            'attribution.py', 'measure.py', 'channel_sorting.py', 'object_classifier.py', 'core.py',
            'psf_pipeline.py', 'pipeline_v2.py', '_lineage_trees.py', 'plot.py', 'run_journal.py', 'ml.py',
            'regression_layout.py', 'regression_failure.py', 'power_model.py'):
        (tmp_path / 'spacr' / name).write_text('', encoding="utf-8")
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    monkeypatch.setattr(builder, '_indirect_runtime_ui_sources', set)
    def extract(source, name='settings_model.py'):
        path = root / name
        path.write_text(source, encoding="utf-8")
        return set(builder.extract_static_ui_sources())
    return extract


def test_metadata_report_templates_enter_exact_class_owner_with_normal_strip(extract_metadata):
    source = '''
READING_THE_FOLDER = "  Reading the metadata folder… "
NO_IMAGES_IN_THE_FOLDER = "No metadata image names were found."
ALL_FILES_PARSE = "All {total} .{extension} names parse."
SOME_FILES_PARSE = "{matched} of {total}; first: {first}"
class _MetadataTypeField:
    def _start(self):
        self._say(READING_THE_FOLDER)
    def _on_failed(self, message):
        self._say(message or NO_IMAGES_IN_THE_FOLDER)
    def _on_tested(self, total):
        self._say(ALL_FILES_PARSE, total=total)
        self._say(SOME_FILES_PARSE, total=total)
'''
    assert extract_metadata(source) == {
        'Reading the metadata folder…', 'No metadata image names were found.',
        'All {total} .{extension} names parse.', '{matched} of {total}; first: {first}'}


def test_metadata_report_owner_excludes_other_scopes_receivers_and_constants(extract_metadata):
    source = '''
READING_THE_FOLDER = "An unrelated outer-scope metadata string."
UNRELATED = "A non-owner metadata detail."
self._say(READING_THE_FOLDER)
def _MetadataTypeField():
    self._say(READING_THE_FOLDER)
def unrelated():
    class _MetadataTypeField:
        def run(self):
            self._say(READING_THE_FOLDER)
class Unrelated:
    def run(self):
        self._say(READING_THE_FOLDER)
class _MetadataTypeField:
    def run(self):
        other._say(READING_THE_FOLDER)
        self._say(UNRELATED)
        self._say(private_transform(READING_THE_FOLDER))
        def nested():
            self._say(READING_THE_FOLDER)
        class Nested:
            def run(self):
                self._say(READING_THE_FOLDER)
        return lambda: self._say(READING_THE_FOLDER)
'''
    assert extract_metadata(source) == set()


def test_metadata_report_owner_excludes_same_class_in_other_file(extract_metadata):
    assert extract_metadata('''
READING_THE_FOLDER = "Unrelated settings-file caption."
class _MetadataTypeField:
    def run(self):
        self._say(READING_THE_FOLDER)
''', 'unrelated.py') == set()


def test_actual_metadata_field_exposes_all_ten_current_missing_templates(extract_metadata):
    import ast

    path = Path(__file__).resolve().parents[1] / 'spacr/qt/screens/settings_model.py'
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {'PROVISIONAL_SUFFIX', 'LOOKS_LIKE', 'RECONSTRUCTED_NOT_DOCUMENTED',
             'READING_THE_FOLDER', 'NO_IMAGES_IN_THE_FOLDER', 'ALL_FILES_PARSE',
             'SOME_FILES_PARSE', 'NOTHING_FITS', 'RANKING_HEADING', 'ONE_RANKING_ROW'}
    declarations = [node for node in tree.body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == '_MetadataTypeField')
    source = ast.unparse(ast.Module(body=[*declarations, owner], type_ignores=[]))
    expected = {ast.literal_eval(node.value).strip() for node in declarations}
    assert len(expected) == 10
    assert expected <= extract_metadata(source)
