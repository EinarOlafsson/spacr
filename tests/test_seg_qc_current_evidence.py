"""QC reports describe current mask evidence without asserting a cause."""
from spacr import seg_qc as q


def card(kind, severity='ok', *, stale=False):
    field = q.FieldQC('plate1_A01_1', kind, 20,
                      flags=[q.FLAG_UNDER] if severity == 'fail' else [], severity=severity)
    return q.Scorecard(kind + '.csv', kind, [field], q.summarize_qc([field]), stale=stale)


def test_stale_failures_cannot_accuse_current_masks_or_pass_incomplete_coverage():
    digest = q._digest_from_cards('project', [card('cell', 'fail', stale=True), card('nucleus')])
    assert digest.verdict == 'warn'
    assert digest.n_fields == 1
    assert not digest.failing_fields
    assert not digest.findings
    assert digest.stale
    text = q.format_digest(digest)
    assert 'previous results are excluded' in text
    assert 'possible merged objects' not in text
    assert 'not a program error' in text


def test_all_stale_cards_require_rescoring_not_a_pass_or_historical_failure():
    digest = q._digest_from_cards('project', [card('cell', 'fail', stale=True)])
    assert digest.verdict == 'missing'
    assert digest.n_fields == 0
    assert not digest.findings
    assert 'out of date' in digest.headline


def test_current_failure_survives_exclusion_of_a_stale_clean_card():
    digest = q._digest_from_cards('project', [card('cell', 'fail'), card('nucleus', stale=True)])
    assert digest.verdict == 'fail'
    assert digest.failing_fields == ('plate1_A01_1',)
    assert digest.n_fields == 1
    assert len(digest.findings) == 1


def test_spatial_counts_alone_do_not_diagnose_illumination_or_rule_out_biology():
    fields = [q.FieldQC(f'plate1_{row}{col:02d}_1', 'pathogen', 3 if col <= 12 else 6)
              for row in 'ABCD' for col in (1, 2, 13, 14)]
    finding = next(f for f in q.diagnose(fields) if f.kind == 'count_gradient')
    assert '2.0x' in finding.headline
    assert 'Treatment layout' in finding.detail
    assert 'cannot establish the cause' in finding.detail
    assert 'controls first' in finding.fix
    assert 'does not guarantee' in q.ILLUMINATION_ADVICE
    assert 'rarely biology' not in q.format_findings([finding])
