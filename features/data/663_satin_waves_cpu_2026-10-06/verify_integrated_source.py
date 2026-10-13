"""Compare accepted private satin and point implementations by exact AST."""

import ast
import gzip
import hashlib
import json
import sys
from pathlib import Path

archive = Path(__file__).resolve().parent
repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else archive.parents[2]
path = repo / 'spacr/qt/widgets/ambient.py'
current = path.read_text()


def nodes(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        result[node.name + '.' + method.name] = ast.dump(
                            method, include_attributes=False)
    return result


actual = nodes(current)
checks = []
for name, selected in [
    ('accepted_renderer.py.gz', ('_warp_satin_columns', '_numpy_satin_columns',
                                '_copy_wave_batch', '_WaveCopyWorker', '_SatinCompiler',
                                '_DataArtEngine._paint_chromatin_ribbon')),
    ('integrated_renderer.py.gz', ('_DataArtEngine._shade', '_DataArtEngine.shade',
                                  '_DataArtEngine._frame_point_atlas',
                                  '_DataArtEngine._frame_impulse_lens',
                                  '_DataArtEngine._frame_genetic_advection',
                                  '_DataArtEngine._point_material')),
]:
    expected = nodes(gzip.decompress((archive / name).read_bytes()).decode())
    for key in selected:
        equal = actual.get(key) == expected[key]
        checks.append({'source': name, 'node': key, 'ast_equal': equal})
assert all(check['ast_equal'] for check in checks), checks
print(json.dumps({'current_renderer_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'checks': checks}, indent=2))
