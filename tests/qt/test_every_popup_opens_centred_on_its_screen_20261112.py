"""Every popup opens centred on its screen, and a moved one stays moved.

Reported 2026-11-12: in Plaque Assay without the plaque model, the
"Download" question opened in the top-left corner of a full-screen window.
The question is a static ``QMessageBox.question``, built by Qt rather than
Python, so the glass filter treats it on its Show; rewriting the flags of a
visible window moved it to its unplaced origin and marked it as placed, so
neither Qt nor spaCR centred it.
"""
import ast
import inspect
from functools import partial
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QDialog, QLabel, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from spacr.qt import dialogs
from spacr.qt.hidpi import screen_for_widget

pytestmark = pytest.mark.qt

TOLERANCE = 4

_REAL_STATICS = {name: inspect.getattr_static(QMessageBox, name) for name in ('question', 'information')}


@pytest.fixture
def answered_statics(monkeypatch):
    for name, real in _REAL_STATICS.items():
        monkeypatch.setattr(QMessageBox, name, real)


@pytest.fixture
def popup_filters(qapp, request):
    from spacr.qt.gil_priority import _stop_watching_application_events
    from spacr.qt.widgets import glass

    saved = dialogs._DETACHER, dialogs._DETACHED_APP
    if dialogs.detach_all_dialogs(qapp):
        installed = dialogs._DETACHER
        request.addfinalizer(lambda: (_stop_watching_application_events(qapp, installed),
                                      setattr(dialogs, '_DETACHER', saved[0]),
                                      setattr(dialogs, '_DETACHED_APP', saved[1])))
    if glass.install_glass_everywhere(qapp):
        request.addfinalizer(lambda: glass.uninstall_glass_everywhere(qapp))
    return qapp


def _owner(qtbot, full=False):
    owner = QMainWindow()
    owner.setCentralWidget(QWidget())
    qtbot.addWidget(owner)
    if full:
        owner.showFullScreen()
    else:
        owner.resize(360, 260)
        owner.move(20, 30)
        owner.show()
    qtbot.waitExposed(owner)
    return owner


def _centre_of_work_area(widget):
    return screen_for_widget(widget).availableGeometry().center()


def _look_for_the_box(qapp, seen, kind):
    boxes = [w for w in qapp.topLevelWidgets() if isinstance(w, kind) and w.isVisible()]
    if not boxes:
        QTimer.singleShot(20, partial(_look_for_the_box, qapp, seen, kind))
        return
    box = boxes[0]
    qapp.processEvents()
    seen.append((box.frameGeometry(), box))
    box.reject()


def _answer_when_shown(qapp, seen, kind=QMessageBox):
    QTimer.singleShot(50, partial(_look_for_the_box, qapp, seen, kind))


def _assert_centred(frame, centre):
    assert (frame.center() - centre).manhattanLength() <= TOLERANCE, (frame, centre)


@pytest.mark.parametrize('full', [True, False])
def test_the_plaque_download_question_opens_centred(qtbot, qapp, popup_filters, answered_statics, full):
    from spacr.qt.model_install import _confirm_checkpoint_download

    owner = _owner(qtbot, full=full)
    seen = []
    _answer_when_shown(qapp, seen)
    assert _confirm_checkpoint_download(owner.centralWidget(), 'plaque_v9') is False
    frame, _box = seen[0]
    _assert_centred(frame, _centre_of_work_area(owner))
    assert screen_for_widget(owner).availableGeometry().contains(frame)


def test_an_unparented_message_box_opens_centred(qtbot, qapp, popup_filters, answered_statics):
    owner = _owner(qtbot)
    seen = []
    _answer_when_shown(qapp, seen)
    QMessageBox.information(None, 'Notice', 'An unparented message box.')
    frame, _box = seen[0]
    _assert_centred(frame, _centre_of_work_area(owner))


def test_a_custom_parented_dialog_opens_centred_on_the_screen(qtbot, qapp, popup_filters):
    owner = _owner(qtbot)
    dialog = QDialog(owner.centralWidget())
    QVBoxLayout(dialog).addWidget(QLabel('A custom dialog', dialog))
    dialog.resize(220, 140)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    qapp.processEvents()
    _assert_centred(dialog.frameGeometry(), _centre_of_work_area(owner))


def test_a_dialog_the_user_moved_stays_where_it_was_moved(qtbot, qapp, popup_filters):
    owner = _owner(qtbot)
    dialog = QDialog(owner)
    QVBoxLayout(dialog).addWidget(QLabel('Moved by the user', dialog))
    dialog.resize(200, 120)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    qapp.processEvents()
    _assert_centred(dialog.frameGeometry(), _centre_of_work_area(owner))
    dialog.move(41, 57)
    qapp.processEvents()
    moved = dialog.pos()
    for _ in range(2):
        dialog.hide()
        qapp.processEvents()
        dialog.show()
        qtbot.waitExposed(dialog)
        qapp.processEvents()
        assert dialog.pos() == moved


def test_a_dialog_resized_after_its_first_show_is_centred_for_its_final_size(qtbot, qapp, popup_filters):
    owner = _owner(qtbot)
    dialog = QDialog(owner)
    dialog.resize(120, 80)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.resize(300, 200)
    qapp.processEvents()
    _assert_centred(dialog.frameGeometry(), _centre_of_work_area(owner))


def test_the_application_installs_the_placement_filter():
    from spacr.qt.app import _DIALOG_FILTERS

    assert ('spacr.qt.dialogs', 'detach_all_dialogs') in _DIALOG_FILTERS


def _moves_itself_to_the_origin(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'move' and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'self' and len(node.args) == 2
                and all(isinstance(arg, ast.Constant) and arg.value == 0 for arg in node.args)):
            return True
    return False


def _dialog_classes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
                (isinstance(base, ast.Name) and base.id.endswith('Dialog'))
                or (isinstance(base, ast.Attribute) and base.attr.endswith('Dialog'))
                for base in node.bases):
            yield node


def test_no_dialog_class_pins_itself_to_the_screen_origin():
    root = Path(dialogs.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for cls in _dialog_classes(tree):
            if _moves_itself_to_the_origin(cls):
                offenders.append(f'{path.relative_to(root)}:{cls.lineno} {cls.name}')
    assert not offenders, offenders
