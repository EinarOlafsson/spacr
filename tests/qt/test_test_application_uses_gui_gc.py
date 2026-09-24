"""The test application must share the runtime's worker-safe GC boundary."""
import gc

import pytest

from spacr.qt import gc_policy

pytestmark = pytest.mark.qt


def test_test_application_keeps_collection_on_its_gui_thread(qapp):
    assert gc_policy.is_installed()
    assert not gc.isenabled()
    assert gc_policy._timer.isActive()
    assert gc_policy._timer.parent() is qapp
    assert gc_policy._timer.thread() is qapp.thread()
