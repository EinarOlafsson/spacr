from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_saved_plots import check_saved_file_run
from barcode_qc_evidence import check_native_run


def actual_scope():
    return dict(outcome=dict(finished=True, ok=True, errors=[]), gui_figure_count=0,
                figures_card_visible=False, settings_errors=[
                    '[settings] ERROR [src]: src is missing from the settings.\n'])


def test_saved_file_scope_does_not_certify_the_broken_gui():
    run = actual_scope()
    check_saved_file_run(run)
    with pytest.raises(ValueError, match='visible in the GUI'):
        check_native_run(run)
    run.update(gui_figure_count=2, figures_card_visible=True, settings_errors=[])
    check_native_run(run)
    check_saved_file_run(run)


@pytest.mark.parametrize('key,value', [('finished', False), ('ok', False), ('errors', ['failed'])])
def test_failed_worker_is_never_accepted(key, value):
    run = actual_scope(); run['outcome'][key] = value
    with pytest.raises(ValueError, match='worker'): check_saved_file_run(run)


@pytest.mark.parametrize('errors', [['another error'], [
    '[settings] ERROR [src]: src is missing from the settings.', 'another error']])
def test_only_the_exact_diagnosed_preflight_error_is_allowed(errors):
    run = actual_scope(); run['settings_errors'] = errors
    with pytest.raises(ValueError, match='Unexpected settings'): check_saved_file_run(run)
