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


def test_spanish_overview_repair_keeps_other_lessons_and_voices_unchanged():
    ns, _ = namespace()
    select, baseline = ns['mastering_config'], ns['MASTERING_CONFIG']
    repair = select('41_classify', 'es', 'ef_dora')
    assert repair['filters'][:-1] == baseline['filters']
    assert repair['filters'][-1].endswith(',volume=-2dB')
    assert repair['maximum_decoded_true_peak_dbfs'] == -1.0
    for identity in [('10_classify_cv', 'es', 'ef_dora'),
                     ('11_classify_ml', 'es', 'ef_dora'),
                     ('41_classify', 'es', 'em_alex'),
                     ('41_classify', 'en', 'af_heart')]:
        assert select(*identity) == baseline
    assert len(baseline['filters']) == 3


def test_module_reference_repair_is_limited_to_its_measured_heart_track():
    ns, _ = namespace()
    select, baseline = ns['mastering_config'], ns['MASTERING_CONFIG']
    repaired = select('79_module_inputs_outputs', 'en', 'af_heart')
    assert repaired['filters'] == baseline['filters'] + [baseline['filters'][-1] + ',volume=-2dB']
    assert repaired['maximum_decoded_true_peak_dbfs'] == -1.0
    for identity in [('79_module_inputs_outputs', 'en', 'af_bella'),
                     ('79_module_inputs_outputs', 'es', 'ef_dora'),
                     ('78_spacr_screens', 'en', 'af_heart'),
                     ('80_image_analysis_pathways', 'en', 'af_heart'),
                     ('81_sequencing_pathways', 'en', 'af_heart')]:
        assert select(*identity) == baseline


def test_hindi_run_compare_repair_is_limited_to_its_measured_track():
    ns, _ = namespace()
    select, baseline = ns['mastering_config'], ns['MASTERING_CONFIG']
    repaired = select('50_run_compare', 'hi', 'hf_beta')
    assert repaired['filters'] == baseline['filters'] + [baseline['filters'][-1] + ',volume=-2dB']
    assert repaired['maximum_decoded_true_peak_dbfs'] == -1.0
    for identity in [('50_run_compare', 'hi', 'hf_alpha'),
                     ('50_run_compare', 'es', 'ef_dora'),
                     ('49_methods_results', 'hi', 'hf_beta')]:
        assert select(*identity) == baseline


def test_a_true_peak_failure_is_reported_without_aborting_the_matrix():
    _, tree = namespace()
    render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_track')
    raised = [ast.unparse(n.exc.func) for n in ast.walk(render)
              if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
              and 'true peak' in ast.unparse(n.exc).lower() and 'exceeds' in ast.unparse(n.exc)]
    assert raised == ['TruePeakError']
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    handlers = [h for n in ast.walk(main) if isinstance(n, ast.Try) for h in n.handlers]
    assert [ast.unparse(h.type) for h in handlers] == ['TruePeakError']
    returns = [ast.unparse(n.value) for n in ast.walk(main) if isinstance(n, ast.Return)]
    assert returns[-1] == '1 if peak_failures else 0'
    peak_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TruePeakError')
    assert [ast.unparse(b) for b in peak_class.bases] == ['RuntimeError']
