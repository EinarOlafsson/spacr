from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import tempfile

root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/f561-classifier-application-20261015')
sys.path.insert(0, str(root / 'tools'))
import build_i18n_catalogs as runtime
runtime.canonical_sources = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('No full collector'))
import build_documentation_i18n as api
import nested_helper_docs as nested
assert 'spacr.embeddings' in nested.ENABLED_MODULES
result = {}
for revision in ('379b68d6bfdfeb6b645c238ef45619fdeb17d629', '03647f0827274824ab26c361259dc022a858d342'):
    with tempfile.TemporaryDirectory(dir=scratch, prefix='api-source-') as directory:
        snapshot = Path(directory)
        package = snapshot / 'spacr'
        package.mkdir()
        hashes = {}
        docs = {}
        for name in ('embeddings.py', 'torch_artifacts.py'):
            raw = subprocess.check_output(['git', 'show', revision + ':spacr/' + name])
            path = package / name
            path.write_bytes(raw)
            hashes['spacr/' + name] = hashlib.sha256(raw).hexdigest()
            module = 'spacr.' + path.stem
            tree = ast.parse(raw)
            for node in api._module_scope_nodes(tree.body):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not api._is_visible_function_name(node.name, module_is_package=False):
                    continue
                key = module + '.' + node.name
                if api._is_rendered_autoapi_entry(path, module, key):
                    docs[key] = api._clean_doc(node)
        definitions = nested.inventory(snapshot, ignore_patterns=api.AUTOAPI_IGNORE)
        for entry in nested.entries(definitions, modules={'spacr.embeddings'}):
            docs[entry.qualified_key] = entry.docstring
        result[revision] = {'source_sha256': hashes, 'docs': docs}
old = result['379b68d6bfdfeb6b645c238ef45619fdeb17d629']['docs']
new = result['03647f0827274824ab26c361259dc022a858d342']['docs']
changes = {}
for key in sorted(new):
    if old.get(key) == new[key]:
        continue
    blocks, _ = api.translatable_blocks(new[key])
    oldblocks, _ = api.translatable_blocks(old.get(key, ''))
    changes[key] = {'source': new[key], 'before': old.get(key), 'new_blocks': [
        {'label': key + '#' + str(index), 'source': block,
         'source_sha256': hashlib.sha256(block.encode()).hexdigest()}
        for index, block in enumerate(blocks)
        if api._api_block_requires_translation(block) and block not in oldblocks]}
assert not api._is_visible_function_name('_dino_classifier_apply', module_is_package=False)
receipt = {'scope': 'Two exact source files only; unchanged normal visibility/clean/block/nested inventory and entries helpers, no full collector/decoder/writer',
           'tools_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [root / 'tools/build_documentation_i18n.py', root / 'tools/nested_helper_docs.py']},
           'source_pins': {revision: record['source_sha256'] for revision, record in result.items()},
           'private_apply_top_level_visible': False, 'changes': changes}
(scratch / 'scoped-api-inventory.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: value['new_blocks'] for key, value in changes.items()}, indent=2))
