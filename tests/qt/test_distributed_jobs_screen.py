"""Distributed Jobs screen, profile editor and current-settings hand-off."""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from spacr.qt.screens.distributed_jobs import (
    APP_INTRO,
    APP_KEY,
    APP_NAME,
    APP_SECTION,
    DistributedJobsScreen,
    ExecutionProfileDialog,
)
from spacr.remote_execution import (
    CommandResult,
    ExecutionProfile,
    JobStore,
    ProfileStore,
    RemoteJobManager,
)


class Runner:
    def __init__(self, *results):
        self.results = deque(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return self.results.popleft()


def _manager(tmp_path, runner):
    profiles = ProfileStore(tmp_path / "profiles.json")
    profiles.save(ExecutionProfile(
        "cloud", "command",
        submit_command="cloud-submit {module} {settings}",
        status_command="cloud-status {external_id}",
        cancel_command="cloud-cancel {external_id}",
        log_command="cloud-logs {external_id}",
    ))
    return RemoteJobManager(
        profiles, JobStore(tmp_path / "jobs.json"), runner
    )


def test_registration_metadata_matches_app_registry():
    from spacr.qt.app import APPS

    row = next(item for item in APPS if item[0] == APP_KEY)
    assert row[1] == APP_NAME == "Distributed Jobs"
    assert row[3] == APP_SECTION == "Data"
    assert APP_INTRO


def test_profile_dialog_switches_backend_fields_and_validates_inline(
    qtbot, qt_theme_applied
):
    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._name.setText("cluster")
    dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
    dialog._workdir.setText("/project")
    assert dialog._slurm.isEnabled()
    assert not dialog._submit_command.isEnabled()
    assert dialog.profile().backend == "slurm"

    dialog._backend.setCurrentIndex(dialog._backend.findData("command"))
    assert dialog._submit_command.isEnabled()
    assert not dialog._workdir.isEnabled()
    dialog._submit_command.setText("cloud-submit")
    dialog._status_command.setText("cloud-status {external_id}")
    dialog._cancel_command.setText("cloud-cancel {external_id}")
    dialog._validate_and_accept()
    assert "{settings}" in dialog._error.text()


def test_every_profile_setting_has_label_help_and_remote_api_link(
    qtbot, qt_theme_applied
):
    from spacr.qt.widgets.info_link import InfoLink

    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    labels = dialog.findChildren(QLabel, "SettingsLabel")
    links = dialog.findChildren(InfoLink)
    assert len(labels) == len(links) == 14
    assert all(label.property("apiTooltipHtml") for label in labels)
    assert all(
        "/spacr/remote_execution/index.html" in link.url()
        for link in links
    )


def test_current_settings_submit_and_status_refresh(
    qtbot, qt_theme_applied, tmp_path
):
    runner = Runner(
        CommandResult(0, "cloud-12\n"),
        CommandResult(0, "SUCCEEDED\n"),
    )
    screen = DistributedJobsScreen(
        manager=_manager(tmp_path, runner),
        threaded=False,
        auto_poll=False,
    )
    qtbot.addWidget(screen)
    screen.configure_submission(
        "mask", {"src": "/data/plate", "random_state": 11}
    )
    assert screen._settings_snapshot["random_state"] == 11
    assert "current mask settings" in screen._settings_path.text()

    qtbot.mouseClick(screen._submit, Qt.LeftButton)
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 1).text() == "queued"
    assert screen._table.item(0, 2).text() == "mask"
    assert runner.calls[0][0][0] == "cloud-submit"

    screen.refresh()
    assert screen._table.item(0, 1).text() == "success"
    assert "cloud-12" in screen._detail.toPlainText()


def test_file_mode_resolves_settings_and_cancel_is_confirmed(
    qtbot, qt_theme_applied, tmp_path, monkeypatch
):
    runner = Runner(
        CommandResult(0, "job-2\n"),
        CommandResult(0),
    )
    screen = DistributedJobsScreen(
        manager=_manager(tmp_path, runner),
        threaded=False,
        auto_poll=False,
    )
    qtbot.addWidget(screen)
    settings = tmp_path / "settings.json"
    settings.write_text('{"src": "/data/plate"}\n')
    screen._module.setCurrentIndex(screen._module.findData("mask"))
    screen._settings_path.setText(str(settings))
    screen._clear_settings_snapshot()
    screen.submit()
    assert screen._table.item(0, 1).text() == "queued"

    monkeypatch.setattr(
        "spacr.qt.screens.distributed_jobs.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    # Import locally so the monkeypatched enum owner is available in lambda.
    from PySide6.QtWidgets import QMessageBox
    screen.cancel_selected()
    assert screen._table.item(0, 1).text() == "cancelled"


def test_missing_profile_and_settings_are_explicit(
    qtbot, qt_theme_applied, tmp_path
):
    manager = RemoteJobManager(
        ProfileStore(tmp_path / "profiles.json"),
        JobStore(tmp_path / "jobs.json"),
        Runner(),
    )
    screen = DistributedJobsScreen(
        manager=manager, threaded=False, auto_poll=False
    )
    qtbot.addWidget(screen)
    screen.submit()
    assert "profile" in screen._status.text().lower()


def test_programmatic_snapshot_survives_but_user_edit_returns_to_file_mode(
    qtbot, qt_theme_applied, tmp_path
):
    screen = DistributedJobsScreen(
        manager=_manager(tmp_path, Runner()),
        threaded=False,
        auto_poll=False,
    )
    qtbot.addWidget(screen)
    screen.configure_submission("measure", {"src": "/one"})
    assert screen._settings_snapshot == {"src": "/one"}
    screen._settings_path.textEdited.emit("/two/settings.json")
    assert screen._settings_snapshot is None


def test_module_remote_button_emits_an_immutable_settings_snapshot(
    qtbot, qt_theme_applied, monkeypatch
):
    from spacr.qt.screens.app_screen import AppScreen

    source = AppScreen("regression")
    qtbot.addWidget(source)
    settings = {"src": "/plate", "alpha": 0.1}
    monkeypatch.setattr(
        source._settings_model, "collect", lambda: settings
    )
    received = []
    source.remote_submit_requested.connect(
        lambda module, snapshot: received.append((module, snapshot))
    )
    qtbot.mouseClick(source._btn_remote, Qt.LeftButton)
    assert received == [("regression", settings)]
    assert received[0][1] is not settings
