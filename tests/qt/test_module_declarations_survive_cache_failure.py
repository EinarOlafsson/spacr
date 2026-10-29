"""A failed declarations cache never imports a screen or replaces a good entry."""
import ast
import os

import pytest

from spacr.qt import app


@pytest.fixture
def declaration_source(tmp_path, monkeypatch):
    name = 'private_screen_declarations'
    source = tmp_path / (name + '.py')
    source.write_text("APP_KEY = 'mask'\nraise RuntimeError('must never import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('SPACR_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(app, '_TOP_LEVEL_CACHE', {})
    return name, source


def test_corrupt_disk_cache_is_rebuilt_from_source_without_importing_it(declaration_source):
    name, source = declaration_source
    cache = app._top_level_disk_path(name, source.read_bytes())
    cache.parent.mkdir(parents=True)
    cache.write_text('invalid python !')
    assert app._declared_constant(name, 'APP_KEY') == 'mask'
    assert ast.literal_eval(ast.parse(cache.read_text()).body[0].value) == 'mask'
    app._TOP_LEVEL_CACHE.clear()
    assert app._declared_constant(name, 'APP_KEY') == 'mask'
    assert list(cache.parent.glob('*.part')) == []


def test_invalid_source_does_not_leave_a_cached_declaration(declaration_source):
    name, source = declaration_source
    source.write_text('APP_KEY = (\n')
    assert app._top_level_nodes(name) is None
    assert app._declared_constant(name, 'APP_KEY') is None
    cache = app._top_level_disk_path(name, source.read_bytes())
    assert not cache.exists()


@pytest.mark.parametrize('part_already_gone', [False, True])
def test_failed_atomic_cache_replace_preserves_existing_entry_and_still_reads_source(
        declaration_source, monkeypatch, part_already_gone):
    name, source = declaration_source
    cache = app._top_level_disk_path(name, source.read_bytes())
    cache.parent.mkdir(parents=True)
    cache.write_text('invalid old cache !')
    before = cache.read_bytes()
    attempted = []
    replace = os.replace

    def refuse_owned_cache_part(part, destination):
        if destination == cache:
            attempted.append(part)
            if part_already_gone:
                part.unlink()
            raise OSError('cache replacement denied')
        return replace(part, destination)

    monkeypatch.setattr(os, 'replace', refuse_owned_cache_part)
    assert app._declared_constant(name, 'APP_KEY') == 'mask'
    assert attempted and cache.read_bytes() == before
    assert all(not part.exists() for part in attempted)
    assert app._declared_constant(name, 'APP_KEY') == 'mask'
