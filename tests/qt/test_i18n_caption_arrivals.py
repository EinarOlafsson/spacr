"""Actual-source negatives for the immutable caption pin arrival adapter."""
import ast
import copy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import build_i18n_catalogs as builder
from i18n_caption_arrivals import LANGUAGES, load_arrival_contract, pinned_external_sources

@pytest.fixture(scope='module')
def state():
    declarations = {}
    for node in ast.parse((ROOT/'tests/qt/test_i18n_caption_ratchet.py').read_text(encoding="utf-8")).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('EXTERNAL_SOURCE_COUNTS','EXTERNAL_SOURCE_KEY_SHA256'):
                    declarations[target.id] = ast.literal_eval(node.value)
    canonical = builder.canonical_sources()
    external = {table:canonical[name] for table,name in (
        ('SETTING_LABELS','setting_labels'),('SETTING_TOOLTIPS','setting_tooltips'),
        ('CATEGORY_HELP','categories'),('UI','ui'),('MODULE_SUMMARIES','module_summaries'))}
    return external, declarations, load_arrival_contract()

def project(external,declarations,contract=None):
    return pinned_external_sources(external,
        original_counts=declarations['EXTERNAL_SOURCE_COUNTS'],
        original_identity_sha256=declarations['EXTERNAL_SOURCE_KEY_SHA256'],contract=contract)

def test_actual_source_and_reviews_reproduce_original_catalog_pin(state):
    external,declarations,contract = state
    pinned = project(external,declarations)
    assert {table:len(rows) for table,rows in pinned.items()}==declarations['EXTERNAL_SOURCE_COUNTS']
    digest = hashlib.sha256('\0'.join(f'{table}\0{key}' for table,key in sorted(
        (table,key) for table,rows in pinned.items() for key in rows)).encode()).hexdigest()
    assert digest==declarations['EXTERNAL_SOURCE_KEY_SHA256']
    en = importlib.import_module('spacr.qt.i18n_catalogs.en')
    assert set(external['UI'])==set(en.UI_SOURCES)
    assert sum(map(len,external.values()))>sum(map(len,pinned.values()))

@pytest.mark.parametrize('change',('unknown_arrival','unknown_retirement','surviving_body_drift','admitted_body_drift'))
def test_actual_source_rejects_every_unrecorded_arrival_retirement_or_body_change(state,change):
    external,declarations,contract = state
    candidate = copy.deepcopy(external)
    if change=='unknown_arrival':
        candidate['UI'] = (*candidate['UI'],'A future unreviewed caption cannot inherit this proof.')
    elif change=='admitted_body_drift':
        entry = next(r for r in contract['added'] if r['table']=='SETTING_LABELS')
        candidate['SETTING_LABELS'][entry['key']] += ' unreviewed'
    else:
        entry = next(r for r in contract['baseline'] if r['table']=='SETTING_LABELS'
                     and r['key'] in candidate['SETTING_LABELS'])
        if change=='unknown_retirement':
            candidate['SETTING_LABELS'].pop(entry['key'])
        else:
            candidate['SETTING_LABELS'][entry['key']] += ' unreviewed'
    with pytest.raises(AssertionError,match='unreviewed caption'):
        project(candidate,declarations)

@pytest.mark.parametrize('change',('target_digest_drift','missing_review_record','file_hash_drift','record_hash_drift'))
def test_actual_source_rejects_unbound_or_drifted_review_evidence(state,change):
    external,declarations,contract = state
    candidate = copy.deepcopy(contract)
    entry = candidate['added'][0]
    if change=='target_digest_drift':
        entry['reviewed_evidence']['sv']['translation_sha256'] = '0'*64
        message = 'caption arrival target drift'
    elif change=='missing_review_record':
        entry['reviewed_evidence']['sv']['matching_review_records'] = []
        message = 'caption arrival lacks explicit reviewed record'
    elif change=='file_hash_drift':
        entry['reviewed_evidence']['sv']['matching_review_records'][0]['file_sha256'] = '0'*64
        message = 'caption arrival reviewed file bytes drift'
    else:
        entry['reviewed_evidence']['sv']['matching_review_records'][0]['record_sha256'] = '0'*64
        message = 'caption arrival reviewed record drift'
    with pytest.raises(AssertionError,match=message):
        project(external,declarations,candidate)


def test_original_historical_source_body_anchor_cannot_be_rewritten(state):
    external,declarations,contract = state
    candidate = copy.deepcopy(contract)
    entry = next(r for r in candidate['baseline'] if r['table']=='SETTING_LABELS')
    entry['source'] += ' silently repinned'
    entry['source_sha256'] = hashlib.sha256(entry['source'].encode()).hexdigest()
    with pytest.raises(AssertionError,match='historical caption bodies changed'):
        project(external,declarations,candidate)


def test_refresh_refuses_a_rewritten_baseline_before_changing_the_file(tmp_path, monkeypatch):
    import i18n_caption_arrivals as arrivals

    candidate = arrivals.load_arrival_contract()
    entry = next(row for row in candidate['baseline'] if row['table'] == 'SETTING_LABELS')
    entry['source'] += ' silently repinned'
    entry['source_sha256'] = hashlib.sha256(entry['source'].encode()).hexdigest()
    fixture = tmp_path / 'arrivals.json.gz'
    fixture.write_bytes(gzip.compress(json.dumps(candidate).encode(), mtime=0))
    before = fixture.read_bytes()
    monkeypatch.setattr(arrivals, 'FIXTURE', fixture)
    with pytest.raises(AssertionError, match='historical caption bodies changed'):
        arrivals.refresh_arrival_contract()
    assert fixture.read_bytes() == before


def test_proof_file_bytes_are_checked_fresh_on_every_validation_call(state,monkeypatch):
    external,declarations,contract = state
    project(external,declarations)
    proof=contract['added'][0]['reviewed_evidence']['sv']['matching_review_records'][0]
    path=ROOT/proof['path']
    original=Path.read_bytes
    def changed(candidate):
        raw=original(candidate)
        return raw+b'\n' if candidate==path else raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(AssertionError,match='caption arrival reviewed file bytes drift'):
        project(external,declarations)


@pytest.fixture
def verified_reviews():
    """Verify and load the real review inputs before a test fakes their bytes.

    The review-input integrity check and the reviewed-translation loader both
    read the review files through ``Path.read_bytes`` and remember the result
    for the process. A test that substitutes those bytes must therefore run
    them first on the genuine files, or the integrity check (correctly) sees
    the substitute as a local edit and refuses before the arrival adapter is
    reached.
    """
    builder._REVIEW_INPUTS_MODULE.ensure_review_inputs(ROOT, offline=True)
    for language in LANGUAGES:
        builder.reviewed_runtime_translations(language)


def test_duplicate_source_bound_review_identity_is_rejected(state,verified_reviews,monkeypatch):
    external,declarations,contract = state
    candidate=copy.deepcopy(contract)
    proof=candidate['added'][0]['reviewed_evidence']['sv']['matching_review_records'][0]
    path=ROOT/proof['path']
    original=Path.read_bytes
    payload=json.loads(original(path))
    payload['records'].append(copy.deepcopy(payload['records'][0]))
    raw=json.dumps(payload,ensure_ascii=False).encode()
    proof['file_sha256']=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(Path,'read_bytes',lambda p:raw if p==path else original(p))
    with pytest.raises(AssertionError,match='duplicate caption reviewed record'):
        project(external,declarations,candidate)


@pytest.mark.parametrize(
    "target,candidate,raw_failure,expected_reasons", [
        ("Compter les cellules par puits pendant l’exécution.",
         "Compter les cellules par puits pendant l’exécution.", False, frozenset()),
        ("Compter les cellules par bien pendant l’exécution.",
         "Compter les cellules par bien pendant l’exécution.", False, frozenset({"semantic"})),
        ("Compter les cellules par puits pendant la exécution.",
         "Compter les cellules par puits pendant l’exécution.", False, frozenset()),
        ("Compter les cellules par bien pendant la exécution.",
         "Compter les cellules par bien pendant l’exécution.", False, frozenset({"semantic"})),
        ("Compter les cellules par puits pendant l’exécution.",
         "Compter les cellules par puits pendant l’exécution.", True, frozenset({"semantic"})),
    ], ids=("unchanged-valid", "unchanged-bad-well", "changed-valid", "changed-bad-well", "raw-failure-flag"),
)
def test_runtime_review_checks_each_distinct_semantic_candidate_once(
        monkeypatch, target, candidate, raw_failure, expected_reasons):
    source = "Count cells per well during the run."
    monkeypatch.setattr(builder, "_REVIEWED_RUNTIME_LOADING", {"fr"})
    assert builder._contextualize(target, "fr", source) == candidate
    semantic = builder._semantic_false_friends
    checked = []

    def check_actual_semantics(current_source, current_target, language):
        checked.append((current_source, current_target, language))
        return semantic(current_source, current_target, language)

    monkeypatch.setattr(builder, "_semantic_false_friends", check_actual_semantics)
    reasons = builder._translation_rejection_reasons(
        source, target, "fr", force=True, raw_semantic_failure=raw_failure)
    assert reasons == expected_reasons
    expected_targets = [] if raw_failure else [target]
    if candidate != target:
        expected_targets.append(candidate)
    assert checked == [(source, value, "fr") for value in expected_targets]


@pytest.fixture
def repeated_runtime_reviews(tmp_path, monkeypatch):
    source = "Count cells per well during the run."
    target = "Compter les cellules par puits pendant l’exécution."
    directory = tmp_path / "runtime" / "fr"
    directory.mkdir(parents=True)
    sources = {"setting_tooltips": {"first": source, "second": source}}
    paths = []
    for key in ("first", "second"):
        payload = {"schema": 1, "language": "fr", "records": [{
            "table": "setting_tooltips", "key": key, "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "translation": target,
        }]}
        path = directory / f"{key}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    builder.reviewed_runtime_translations.cache_clear()
    monkeypatch.setattr(builder, "REVIEWED_RUNTIME_DIR", directory.parent)
    monkeypatch.setattr(builder, "canonical_sources", lambda: sources)
    try:
        yield source, target, sources, paths
    finally:
        builder.reviewed_runtime_translations.cache_clear()


def test_repeated_runtime_targets_keep_each_bound_identity_and_validate_real_gates_once(
        repeated_runtime_reviews, monkeypatch):
    source, target, sources, paths = repeated_runtime_reviews
    original = {path: path.read_bytes() for path in paths}
    guard = builder._translation_rejection_reasons
    checked = []
    def check(source, target, language, **kwargs):
        checked.append((source, target, language))
        return guard(source, target, language, **kwargs)
    monkeypatch.setattr(builder, "_translation_rejection_reasons", check)
    assert builder.reviewed_runtime_translations("fr") == {source: target}
    assert checked == [(source, target, "fr")]
    assert {path: path.read_bytes() for path in paths} == original
    assert set(sources["setting_tooltips"]) == {"first", "second"}


@pytest.mark.parametrize("target,problem", [
    ("Compter les cellules par bien pendant l’exécution.", "rejected reviewed runtime target"),
    ("Compter les cellules par puits pendant la exécution.", "non-idempotent reviewed runtime target"),
])
def test_repeated_invalid_runtime_targets_still_report_every_bound_record(
        repeated_runtime_reviews, monkeypatch, target, problem):
    source, _target, _sources, paths = repeated_runtime_reviews
    contextualize = builder._contextualize
    guard = builder._translation_rejection_reasons
    contextualized, checked = [], []
    def contextualize_actual(value, language, current_source=""):
        if (current_source, value, language) == (source, target, "fr"):
            contextualized.append((current_source, value, language))
        return contextualize(value, language, current_source)
    def check_actual(current_source, value, language, **kwargs):
        checked.append((current_source, value, language))
        return guard(current_source, value, language, **kwargs)
    monkeypatch.setattr(builder, "_contextualize", contextualize_actual)
    monkeypatch.setattr(builder, "_translation_rejection_reasons", check_actual)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["records"][0]["translation"] = target
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    original = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError) as error:
        builder.reviewed_runtime_translations("fr")
    message = str(error.value)
    assert f"{problem} setting_tooltips/first: {paths[0]}" in message
    assert f"{problem} setting_tooltips/second: {paths[1]}" in message
    assert "...and 1 more reviewed runtime problems in fr:" in message
    assert {path: path.read_bytes() for path in paths} == original
    if problem.startswith("non-idempotent"):
        assert contextualized == [(source, target, "fr")]
        assert checked == []
    else:
        assert checked == [(source, target, "fr")]


@pytest.mark.parametrize("mutation,message", [
    ("source", "stale reviewed runtime source"),
    ("hash", "stale reviewed runtime hash"),
    ("fields", "invalid reviewed runtime record:"),
    ("header", "invalid reviewed runtime evidence header:"),
])
def test_a_duplicate_target_never_bypasses_its_own_source_hash_or_schema(
        repeated_runtime_reviews, mutation, message):
    source, _target, sources, paths = repeated_runtime_reviews
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    if mutation == "source":
        sources["setting_tooltips"]["second"] = source + " Different current meaning."
    elif mutation == "hash":
        payload["records"][0]["source_sha256"] = "0" * 64
    elif mutation == "fields":
        payload["records"][0]["retired"] = "2026-10-28"
    else:
        payload["language"] = "de"
    paths[1].write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    original = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match=message) as error:
        builder.reviewed_runtime_translations("fr")
    assert str(paths[1]) in str(error.value)
    assert {path: path.read_bytes() for path in paths} == original


def test_distinct_valid_runtime_targets_still_conflict_despite_duplicate_sources(
        repeated_runtime_reviews):
    source, _target, _sources, paths = repeated_runtime_reviews
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["records"][0]["translation"] = "Comptez les cellules par puits pendant l’exécution."
    paths[1].write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting reviewed runtime targets") as error:
        builder.reviewed_runtime_translations("fr")
    assert repr(source) in str(error.value)


def test_runtime_target_check_exceptions_still_escape_and_release_the_loading_guard(
        repeated_runtime_reviews, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("target validator unavailable")
    monkeypatch.setattr(builder, "_translation_rejection_reasons", unavailable)
    with pytest.raises(OSError, match="target validator unavailable"):
        builder.reviewed_runtime_translations("fr")
    assert "fr" not in builder._REVIEWED_RUNTIME_LOADING
