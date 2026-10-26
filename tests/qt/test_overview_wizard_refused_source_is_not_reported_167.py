"""GitHub #167: a refused Overview source is a status line, not an error report.

The metadata wizard probes its source on a worker. A source that was not a
folder (or a pattern without a wellID group) raised ``_OverviewValidationError``
on that worker, so the job failed and the refusal was filed as an error report.
The wizard now shows the refusal and keeps Next disabled, and the job succeeds.
"""
import os

import pytest

from spacr.qt.widgets import qc_field_browser as browser


def _wizard(qtbot, source, settings=None):
    wizard = browser._OverviewMetadataWizard(source, settings)
    qtbot.addWidget(wizard)
    wizard.show()
    failures = []
    wizard._runner.job_failed.connect(failures.append)
    return wizard, failures


def test_a_missing_source_folder_is_shown_in_the_status_line_without_a_failed_job(qtbot, tmp_path):
    missing = tmp_path / "not-there-yet"
    wizard, failures = _wizard(qtbot, str(missing))
    expected = browser.tr("Choose an existing image folder.")

    qtbot.waitUntil(lambda: wizard.status.text() == expected, timeout=10000)
    qtbot.waitUntil(lambda: not wizard._runner._jobs, timeout=10000)

    assert failures == []
    assert not wizard.next.isEnabled()
    assert wizard._info is None and tuple(wizard._fields) == ()
    assert not os.path.exists(missing)


def test_the_wizard_recovers_once_the_typed_source_becomes_a_folder(qtbot, tmp_path):
    import numpy as np
    import tifffile

    source = tmp_path / "plate"
    wizard, failures = _wizard(qtbot, str(source))
    qtbot.waitUntil(lambda: wizard.status.text() == browser.tr("Choose an existing image folder."), timeout=10000)

    source.mkdir()
    for channel in (1, 2):
        tifffile.imwrite(source / f"plate1_E01_T0001F001L01A01Z01C{channel:02}.tif",
                         np.full((12, 16), channel, dtype=np.uint16))
    wizard.source.setText(str(source))
    wizard._scan()

    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert wizard._info is not None
    assert failures == []


def test_a_refused_source_returns_its_message_instead_of_raising(tmp_path):
    import threading

    fields, info, refusal = browser._overview_wizard_scan(str(tmp_path / "missing"), "", "auto",
                                                          threading.Event())

    assert (tuple(fields), info) == ((), None)
    assert refusal == browser.tr("Choose an existing image folder.")
    with pytest.raises(browser._OverviewValidationError):
        browser._overview_wizard_probe(str(tmp_path / "missing"), "", "auto", threading.Event())
