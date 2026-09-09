"""Reject tutorial demonstrations with partial output, even after GUI success."""
from __future__ import annotations


def assess_pipeline(outcome, console_blocks, figure_count):
    """Return explicit recording acceptance, keeping ordinary QC warnings visible."""
    reasons = []
    if not outcome.get('finished') or not outcome.get('ok'):
        reasons.append('The pipeline did not report successful completion.')
    if outcome.get('errors'):
        reasons.append('The worker reported errors.')
    output = '\n'.join(console_blocks)
    if 'RUN INCOMPLETE' in output or 'ARTIFACTS FROM THIS RUN ARE INCOMPLETE' in output:
        reasons.append('The console reports partial artifacts despite the GUI completion status.')
    if 'Pipeline worker failed' in output:
        reasons.append('A background worker failed, even though the main pipeline finished.')
    if figure_count < 1:
        reasons.append('Plot was enabled but no inspectable figure was produced.')
    return {'accepted': not reasons, 'reasons': reasons, 'figure_count': figure_count,
            'worker_finished': bool(outcome.get('finished')),
            'worker_ok': bool(outcome.get('ok'))}
