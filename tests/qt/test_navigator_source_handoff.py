"""Navigator opens actual assay forms and preserves input while adapters prepare it."""
from types import SimpleNamespace

from spacr.qt import app, dnd_handlers
from tests.qt.test_733_easy_commands import commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def test_plaque_native_route_opens_assay_not_organism_group(commands, monkeypatch):
    requests = []
    commands._screens = {'analyze_plaques': commands.screen}
    monkeypatch.setattr(commands, 'open_module', requests.append, raising=False)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    assert app.MainWindow._open_easy_named_module(commands, 'analyze_plaques') is commands.screen
    assert requests == ['analyze_plaques']


def test_downstream_module_retains_source_even_if_its_adapter_needs_other_input(commands, monkeypatch, tmp_path):
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4\n%%EOF')
    seeded = []
    commands.screen._settings_model = SimpleNamespace(
        set_value_for_key=lambda key, value: seeded.append((key, value)))
    class Handler:
        def can_accept(self, path):
            return False
        def error_message(self, path):
            return 'Choose a compatible dataset before running.'
    monkeypatch.setattr(dnd_handlers, 'get_handler', lambda key: Handler())
    assert commands._open_easy_input('measure', str(pdf), {})
    assert commands.opened == ['measure']
    assert seeded == [('src', str(pdf))]
    assert pdf.read_bytes() == b'%PDF-1.4\n%%EOF'


def test_pdf_source_is_present_while_figure_extraction_starts(commands, tmp_path):
    from spacr.qt.widgets.plaque_preview import FIGURE_MODE
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4\n%%EOF')
    seeded, fetched = [], []
    model = SimpleNamespace(set_value_for_key=lambda key, value: seeded.append((key, value)))
    screen = SimpleNamespace(_settings_model=model,
        _live_preview=SimpleNamespace(fetch_paper=lambda *args: fetched.append(args) or True))
    screen._plaque_mode_switch = SimpleNamespace(mode=lambda: FIGURE_MODE)
    dnd_handlers.PlaqueDropHandler._take_pdfs([pdf], screen)
    assert seeded == [('src', str(pdf))]
    assert fetched == [(str(pdf), str(tmp_path))]


def test_native_input_preserves_approved_inspection_settings(commands, monkeypatch, tmp_path):
    source = tmp_path / 'images'
    source.mkdir()
    applied = []
    commands._startup._settings_overrides = {'mask': {'regex': '(?P<well>A01)'}}
    commands.screen.apply_settings_dict = lambda values: applied.append(values)
    class Handler:
        def can_accept(self, path):
            return True
        def apply(self, path, screen):
            applied.append({'src': str(path)})
    monkeypatch.setattr(dnd_handlers, 'get_handler', lambda key: Handler())
    assert commands._open_easy_input('mask', str(source), {})
    assert applied == [{'src': str(source)}, {'regex': '(?P<well>A01)'}]
