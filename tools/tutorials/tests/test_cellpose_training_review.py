"""The real screen lifecycle API, not a lazily absent worker attribute."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from capture_cellpose_training_review import check_idle


def test_idle_screen_accepts_and_both_actual_job_signals_reject():
    idle=SimpleNamespace(_worker_thread_is_running=lambda:False,active_jobs=lambda:[])
    check_idle(idle)
    for thread,jobs in [(True,[]),(False,['training']),(True,['inference'])]:
        active=SimpleNamespace(_worker_thread_is_running=lambda:thread,active_jobs=lambda:jobs)
        with pytest.raises(ValueError,match='must not start'):
            check_idle(active)
