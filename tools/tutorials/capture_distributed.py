"""Record real profile preparation, without connecting or submitting a job.

The launcher uses a fresh application-supported profile store and an isolated
network namespace. Example host/path values are explicitly not real targets.
No successful remote execution or scheduler output is manufactured.
"""
import hashlib
import json
import os
from pathlib import Path


def record_distributed(app, window, stage, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox
    from spacr.qt.screens.distributed_jobs import ExecutionProfileDialog

    state = Path(os.environ['SPACR_REMOTE_STATE_DIR']).resolve()
    if not state.is_relative_to(Path(stage).resolve()):
        raise ValueError('Remote profile storage is not private')
    proof = dict(lesson='38_distributed_jobs', accepted=False,
        scope='Native profile preparation only; no verified remote execution',
        state=str(state), application_modified=False, published=False,
        remote_job_submitted=False, real_credentials_used=False,
        example_host='researcher@workstation.invalid', backend_states={})

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Native distributed control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton,
            pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def fill(widget, text):
        click(widget); QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,text); QTest.keyClick(widget,Qt.Key_Tab); settle(.1)

    def backend(dialog, value):
        index=dialog._backend.findData(value)
        if index<0: raise ValueError('The actual backend is unavailable')
        click(dialog._backend); QTest.keyClick(dialog._backend,Qt.Key_Home)
        for _ in range(index): QTest.keyClick(dialog._backend,Qt.Key_Down)
        QTest.keyClick(dialog._backend,Qt.Key_Return);settle(.2)
        proof['backend_states'][value]={
            name:getattr(dialog,'_'+name).isEnabled() for name in
            ('host','workdir','local_root','remote_root','runner','slurm',
             'submit_command','status_command','cancel_command')}

    def dialog_action(button, action):
        errors=[]; finished=[]
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,ExecutionProfileDialog):
                    raise ValueError('The native execution profile editor did not open')
                dialog.resize(2200,1200)
                bounds=window.geometry()
                dialog.move(bounds.x()+700,bounds.y()+200);settle(.3)
                action(dialog);finished.append(True)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:dialog.reject()
        watchdog=QTimer(window);watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda:app.activeModalWidget().reject()
            if app.activeModalWidget() else None)
        QTimer.singleShot(300,handle);watchdog.start(20000)
        click(button);watchdog.stop()
        if errors or not finished:raise ValueError('; '.join(errors) or 'Profile editor timed out')

    try:
        action=next(a for a in window.menuBar().actions() if a.text().replace('&','')=='Help')
        menu=action.menu()
        choice=next(a for a in menu.actions() if a.text().replace('&','').lower()=='distributed jobs')
        QTest.mouseClick(window.menuBar(),Qt.LeftButton,
            pos=window.menuBar().actionGeometry(action).center());settle(.3)
        capture('01_help_route')
        QTest.mouseClick(menu,Qt.LeftButton,pos=menu.actionGeometry(choice).center());settle(.8)
        screen=window._screens['distributed_jobs']
        if not screen.isVisible() or screen.manager.profiles.path.resolve()!=state/'profiles.json':
            raise ValueError('Actual distributed screen/store differs')
        capture('02_empty_actual_workbench')

        def create(dialog):
            box=dialog.findChild(QDialogButtonBox)
            click(box.button(QDialogButtonBox.Save))
            if not dialog.isVisible() or not dialog._error.text():
                raise ValueError('An empty profile was not refused')
            proof['empty_profile_error']=dialog._error.text();capture('03_empty_profile_refused')
            fill(dialog._name,'Tutorial setup - NOT CONNECTED')
            fill(dialog._host,proof['example_host'])
            fill(dialog._workdir,'/shared/tutorial_project')
            fill(dialog._local_root,'/local/tutorial_project')
            fill(dialog._remote_root,'/shared/tutorial_project')
            backend(dialog,'ssh');capture('04_ssh_profile_example')
            click(box.button(QDialogButtonBox.Save))
        dialog_action(screen._new_profile,create)
        path=state/'profiles.json'
        saved=path.read_bytes();proof['profile_sha256']=hashlib.sha256(saved).hexdigest()
        proof['persisted_profiles']=json.loads(saved)['profiles']
        if len(proof['persisted_profiles'])!=1:raise ValueError('One saved profile expected')
        row=proof['persisted_profiles'][0]
        expected=dict(name='Tutorial setup - NOT CONNECTED',backend='ssh',
            host=proof['example_host'],workdir='/shared/tutorial_project',
            local_root='/local/tutorial_project',remote_root='/shared/tutorial_project',runner='spacr-run')
        if any(row.get(k)!=v for k,v in expected.items()):
            raise ValueError('Persisted profile differs from typed controls')
        capture('05_profile_saved_not_connected')

        def inspect(dialog):
            if any(getattr(dialog,'_'+k).text()!=v for k,v in expected.items() if k!='backend'):
                raise ValueError('Reopened profile differs from persisted values')
            proof['profile_reopened_exactly']=True;capture('06_saved_profile_reopened')
            backend(dialog,'slurm');fill(dialog._slurm,'--partition=YOUR_PARTITION --time=00:10:00')
            capture('07_slurm_controls_not_submitted')
            backend(dialog,'command');capture('08_custom_argument_templates')
            click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel))
        dialog_action(screen._edit_profile,inspect)
        if path.read_bytes()!=saved:raise ValueError('Cancelling changed the saved profile')
        proof['cancel_preserves_profile_bytes']=True
        capture('09_submission_controls_no_job')
        if screen.manager.jobs.list() or screen._table.rowCount():
            raise ValueError('Configuration-only capture unexpectedly contains a job')
        proof.update(accepted=True,job_rows=0,profile_count=1,
            verification='Saved JSON, native reopened controls and cancelled edits agree exactly')
    finally:
        write_json(Path(captures)/'scientific_acceptance.json',proof)
