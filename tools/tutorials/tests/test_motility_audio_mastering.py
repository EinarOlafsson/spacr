"""Pin the measured one-track repair without importing speech/GPU dependencies."""
import ast
from pathlib import Path


def namespace():
    path = Path(__file__).resolve().parents[1] / 'authoring/tools/render_all_voices.py'
    tree = ast.parse(path.read_text())
    names = {'LOUDNESS_FILTERS', 'MAX_RELEASE_TRUE_PEAK_DBFS', 'MASTERING_CONFIG'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mastering_config'
        or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    result = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), result)
    return result, tree


def test_only_the_measured_track_gets_a_fingerprinted_extra_attenuation():
    ns, _ = namespace()
    baseline = ns['MASTERING_CONFIG']
    select = ns['mastering_config']
    for lesson, lang, voice in [('17_timelapse', 'en', 'bf_isabella'),
            ('18_motility', 'en', 'af_heart'), ('18_motility', 'fr', 'bf_isabella')]:
        assert select(lesson, lang, voice) == baseline
    repair = select('18_motility', 'en', 'bf_isabella')
    assert repair['filters'][:-1] == baseline['filters']
    assert repair['filters'][-1] == baseline['filters'][-1] + ',volume=-2dB'
    assert repair['maximum_decoded_true_peak_dbfs'] == -1.0
    assert len(baseline['filters']) == 3
    repair['filters'].clear()
    assert len(select('18_motility', 'en', 'bf_isabella')['filters']) == 4


def test_fingerprint_and_encoder_use_the_same_selected_mastering_not_an_unrecorded_override():
    _, tree = namespace()
    fingerprint = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'track_fingerprint')
    calls = [n for n in ast.walk(fingerprint) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'mastering_config']
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["lesson['id']", 'language', 'voice']
    render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_track')
    loops = [n for n in ast.walk(render) if isinstance(n, ast.For) and ast.unparse(n.target) == 'loudness_filter']
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "fingerprint_inputs['mastering']['filters']"
