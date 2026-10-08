"""Actual Sphinx rendering for canonical API outside star-import exports."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import api_visibility


def test_projection_retains_import_inheritance_and_private_boundaries():
    names = api_visibility.documented_tree({
        'example.Engine.step', 'example.Engine.state', 'example._parent.helper'})
    own = SimpleNamespace(imported=False, inherited=False)
    imported = SimpleNamespace(imported=True, inherited=False)
    inherited = SimpleNamespace(imported=False, inherited=True)
    assert api_visibility.source_page_policy('example.Engine', own, names) is False
    assert api_visibility.source_page_policy('example.Engine.step', own, names) is False
    assert api_visibility.source_page_policy('example.Engine.step', imported, names) is None
    assert api_visibility.source_page_policy('example.Engine.step', inherited, names) is None
    assert api_visibility.source_page_policy('example._other', own, names) is None
    assert api_visibility.source_page_policy('example._parent', own, names) is None


def _build(tmp_path, *, projected):
    pytest.importorskip('sphinx')
    pytest.importorskip('autoapi')
    package = tmp_path / 'source' / 'projection_fixture'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('"""Projection fixture package."""\n')
    (package / 'foreign.py').write_text('''"""Canonical foreign objects."""
class Foreign:
    """A separate canonical class."""
''')
    (package / 'engine.py').write_text('''"""Canonical engine documents."""
from .foreign import Foreign as Imported
__all__ = ['Surface', 'Imported']
class Surface:
    """The star-import interface."""
class Engine:
    """A documented engine outside star-import exports."""
    def step(self, seconds=1.0):
        """Advance the canonical engine."""
        return seconds
    @property
    def state(self):
        """Read the canonical engine state."""
        return 1
    def __bool__(self):
        """Report whether the engine has state."""
        return True
def _parent():
    """Private lexical parent stays hidden."""
    def helper():
        """A separately rendered nested helper."""
        return 1
    return helper
class Undocumented:
    pass
''')
    docs = tmp_path / 'docs'
    docs.mkdir()
    policy = '''
import api_visibility
projected = api_visibility.documented_tree({
    'projection_fixture.engine.Engine',
    'projection_fixture.engine.Engine.step',
    'projection_fixture.engine.Engine.state',
    'projection_fixture.engine.Engine.__bool__',
    'projection_fixture.engine.Imported',
    'projection_fixture.engine._parent.helper',
})
def _policy(app, what, name, obj, skip, options):
    return api_visibility.source_page_policy(name, obj, projected)
def setup(app):
    app.connect('autoapi-skip-member', _policy)
''' if projected else ''
    (docs / 'conf.py').write_text(
        'import sys\n'
        f'sys.path.insert(0, {str(ROOT / "tools")!r})\n'
        "project = 'Projection fixture'\n"
        "extensions = ['autoapi.extension']\n"
        f'autoapi_dirs = [{str(package)!r}]\n'
        "autoapi_options = ['members', 'show-inheritance', 'show-module-summary']\n"
        "autoapi_keep_files = True\n"
        + policy
    )
    (docs / 'index.rst').write_text('Projection fixture\n==================\n\n.. toctree::\n\n   autoapi/index\n')
    output = tmp_path / 'html'
    result = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-q', '-W', '-E', '-b', 'html', str(docs), str(output)],
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return (output / 'autoapi/projection_fixture/engine/index.html').read_text()


def test_actual_projection_renders_existing_source_without_imported_duplicates(tmp_path):
    before = _build(tmp_path / 'before', projected=False)
    after = _build(tmp_path / 'after', projected=True)
    for name in ('Engine', 'Engine.step', 'Engine.state', 'Engine.__bool__'):
        anchor = f'id="projection_fixture.engine.{name}"'
        assert anchor not in before
        assert anchor in after
    assert 'Advance the canonical engine.' in after
    assert 'Read the canonical engine state.' in after
    assert 'id="projection_fixture.engine.Imported"' not in after
    assert 'id="projection_fixture.engine.Undocumented"' not in after
    assert 'id="projection_fixture.engine._parent"' not in after
    assert 'Private lexical parent stays hidden.' not in after


def _build_branches(tmp_path, *, supplemented):
    pytest.importorskip('sphinx')
    pytest.importorskip('autoapi')
    package = tmp_path / 'source' / 'projection_fixture'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('"""Source-branch fixture."""\n')
    (package / 'foreign.py').write_text('class Node:\n    """Canonical input node."""\n')
    (package / 'engine.py').write_text('''"""Optional source branches."""
from .foreign import Node
_AVAILABLE = False
if _AVAILABLE:
    _hidden = 1
    class Optional:
        """Installed implementation.

        :param setting: Source setting.
        """
        def __init__(self, setting='raw'):
            """Keep the source setting."""
            self._setting = setting
        @property
        def mode(self):
            """Read the source setting."""
            return self._setting
        def step(self, seconds=1.0):
            """Advance by a measured second."""
            return seconds
    def branch(values: Node, rate=1 / 3):
        """Use the measured source values."""
        return values
else:
    _hidden = 2
    class Optional:
        """Unavailable implementation."""
    def branch(*args, **kwargs):
        """Refuse unavailable optional processing.

        :raises ImportError: Optional processing is unavailable.
        """
        raise ImportError('optional dependency')
def _private():
    """Private implementation stays private."""
    return None
''')
    docs = tmp_path / 'docs'
    docs.mkdir()
    templates = tmp_path / 'templates'
    if supplemented:
        import shutil
        (templates / 'python').mkdir(parents=True)
        for name in ('module.rst', 'nested_helpers.rst'):
            shutil.copyfile(ROOT / 'docs/source/_autoapi_templates/python' / name,
                            templates / 'python' / name)
    projection = {
        'projection_fixture.engine.Optional':
            'Installed implementation.\n\n:param setting: Source setting.\n'
            'Keep the source setting.\nUnavailable implementation.',
        'projection_fixture.engine.Optional.mode': 'Read the source setting.',
        'projection_fixture.engine.Optional.step': 'Advance by a measured second.',
        'projection_fixture.engine.branch':
            'Use the measured source values.\nRefuse unavailable optional processing.\n\n'
            ':raises ImportError: Optional processing is unavailable.',
    }
    custom = (
        f"documents = {projection!r}\n"
        f"source_root = {str(tmp_path / 'source')!r}\n"
        "def prepare(env):\n"
        "    import nested_helper_docs\n"
        "    env.globals['spacr_nested_helpers'] = {}\n"
        "    env.filters['spacr_helper_docstring'] = nested_helper_docs.rendered_docstring\n"
        "    api_visibility.prepare_jinja(env, root=source_root, documents=documents)\n"
        "autoapi_prepare_jinja_env = prepare\n"
    ) if supplemented else ''
    (docs / 'conf.py').write_text(
        'import sys\n'
        f'sys.path.insert(0, {str(ROOT / "tools")!r})\n'
        'import api_visibility\n'
        "project = 'Source branches'\n"
        "extensions = ['autoapi.extension']\n"
        f'autoapi_dirs = [{str(package)!r}]\n'
        "autoapi_options = ['members', 'show-module-summary']\n"
        "autoapi_keep_files = True\n"
        + (f'autoapi_template_dir = {str(templates)!r}\n'
           if supplemented else '') + custom
    )
    (docs / 'index.rst').write_text(
        'Source branches\n===============\n\n.. toctree::\n\n   autoapi/index\n')
    output = tmp_path / 'html'
    result = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-q', '-W', '-E', '-b', 'html',
         str(docs), str(output)], capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_actual_source_branches_render_once_with_source_defaults(tmp_path):
    import zlib
    before = _build_branches(tmp_path / 'before', supplemented=False)
    after = _build_branches(tmp_path / 'after', supplemented=True)
    relative = 'autoapi/projection_fixture/engine/index.html'
    old = (before / relative).read_text()
    rendered = (after / relative).read_text()
    inventory = zlib.decompress((after / 'objects.inv').read_bytes().split(b'\n', 4)[4]).decode()
    for suffix, kind in (('Optional', 'class'), ('Optional.mode', 'property'),
                         ('Optional.step', 'method'), ('branch', 'function')):
        name = 'projection_fixture.engine.' + suffix
        assert f'id="{name}"' not in old
        assert rendered.count(f'id="{name}"') == 1
        rows = [line for line in inventory.splitlines() if line.startswith(name + ' ')]
        assert len(rows) == 1 and rows[0].split()[1] == 'py:' + kind
    for prose in ('Installed implementation.', 'Unavailable implementation.',
                  'Keep the source setting.', 'Advance by a measured second.',
                  'Refuse unavailable optional processing.'):
        assert prose in rendered
    pytest.importorskip('bs4')
    from bs4 import BeautifulSoup
    page = BeautifulSoup(rendered, 'html.parser')
    assert any(''.join(code.get_text().split()) == 'branch(*args,**kwargs)'
               for code in page.select('code.literal'))
    signature = page.find(id='projection_fixture.engine.branch')
    assert 'rate=1/3' in ''.join(signature.get_text().split())
    assert 'values:projection_fixture.foreign.Node' in ''.join(signature.get_text().split())
    assert 'Source setting.' in rendered
    assert 'projection_fixture/engine.py:' in rendered
    assert 'id="projection_fixture.engine._private"' not in rendered
