"""The Measure RAM guard asks once, offers three answers, and closes nothing
without a confirm.

Every process comes from a fake psutil, so no real program is signalled.
"""
from __future__ import annotations

import types

import pytest

from spacr.qt.screens import app_screen
from spacr.qt.screens.app_screen import AppScreen, _FreeRamDialog, _RamGuardDialog

GIB = 1024 ** 3
PLAN = {'per_worker': 4 * GIB, 'available': 40 * GIB, 'total': 64 * GIB,
        'reserve': 8 * GIB, 'max_safe': 8, 'requested': 24, 'exceeds': True}


class _Screen:
    _confirm_ram_guard = AppScreen._confirm_ram_guard


def _plan_for(*plans):
    queue = list(plans)
    return lambda _src, _n: queue.pop(0) if len(queue) > 1 else queue[0]


def test_a_run_that_fits_is_not_interrupted():
    settings = {'n_jobs': 4}
    fits = dict(PLAN, exceeds=False)
    assert _Screen()._confirm_ram_guard(settings, plan_for=_plan_for(fits),
                                        ask=pytest.fail)
    assert settings == {'n_jobs': 4}


def test_use_lowers_n_jobs_to_the_safe_count():
    settings = {'n_jobs': 4}
    assert _Screen()._confirm_ram_guard(
        settings, plan_for=_plan_for(PLAN), ask=lambda plan: 'use')
    assert settings['n_jobs'] == 8


def test_keep_switches_the_clamp_off_and_leaves_n_jobs():
    settings = {'n_jobs': 4}
    assert _Screen()._confirm_ram_guard(
        settings, plan_for=_plan_for(PLAN), ask=lambda plan: 'keep')
    assert settings == {'n_jobs': 4, 'ram_guard': False}


def test_free_then_reestimates_and_closing_the_dialog_cancels():
    freed = []
    settings = {'n_jobs': 4}
    answers = iter(['free', None])
    assert not _Screen()._confirm_ram_guard(
        settings, plan_for=_plan_for(PLAN), ask=lambda plan: next(answers),
        free_ram=lambda: freed.append(1))
    assert freed == [1]


def test_free_ram_that_now_fits_starts_the_run():
    fits = dict(PLAN, exceeds=False)
    assert _Screen()._confirm_ram_guard(
        {'n_jobs': 4}, plan_for=_plan_for(PLAN, fits),
        ask=lambda plan: 'free', free_ram=lambda: None)


def test_the_guard_off_skips_the_dialog():
    assert _Screen()._confirm_ram_guard(
        {'n_jobs': 4, 'ram_guard': False}, plan_for=pytest.fail)


@pytest.mark.parametrize('button, choice', [
    ('use_button', 'use'), ('keep_button', 'keep'), ('free_button', 'free')])
def test_each_dialog_button_returns_its_choice(qapp, button, choice):
    dialog = _RamGuardDialog(None, PLAN)
    assert '8' in dialog.use_button.text()
    assert '24' in dialog.keep_button.text()
    getattr(dialog, button).click()
    assert dialog.choice == choice
    dialog.deleteLater()


class _Proc:
    def __init__(self, pid, name, user, rss):
        self.pid = pid
        self.info = {'pid': pid, 'name': name, 'username': user,
                     'memory_info': types.SimpleNamespace(rss=rss)}
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def username(self):
        return 'me'

    def parents(self):
        return []

    def children(self, recursive=False):
        return []


class _Psutil:
    class NoSuchProcess(Exception):
        pass

    def __init__(self):
        self.procs = {pid: _Proc(pid, name, user, rss) for pid, name, user, rss
                      in ((100, 'python', 'me', GIB),
                          (200, 'firefox', 'me', 3 * GIB),
                          (201, 'slack', 'me', 2 * GIB),
                          (300, 'Xorg', 'root', GIB))}

    def Process(self, pid=None):
        return self.procs[100 if pid is None else pid]

    def process_iter(self, _attrs):
        return list(self.procs.values())


def test_the_process_list_starts_unticked_and_needs_a_confirm(qapp):
    fake = _Psutil()
    confirms = []
    dialog = _FreeRamDialog(None, psutil_module=fake,
                            confirm=lambda names: confirms.append(names) or False)
    assert [box.property('pid') for box in dialog.boxes] == [200, 201]
    assert not any(box.isChecked() for box in dialog.boxes)
    assert 'firefox' in dialog.boxes[0].text()
    dialog.close_button.click()
    assert confirms == []
    dialog.boxes[0].setChecked(True)
    dialog.close_button.click()
    assert len(confirms) == 1
    assert not fake.procs[200].terminated
    dialog.deleteLater()


def test_confirmed_programs_are_asked_to_quit(qapp):
    fake = _Psutil()
    dialog = _FreeRamDialog(None, psutil_module=fake, confirm=lambda names: True)
    dialog.boxes[1].setChecked(True)
    dialog.close_button.click()
    assert dialog.outcomes == {201: 'closed'}
    assert fake.procs[201].terminated and not fake.procs[200].terminated
    dialog.deleteLater()


def test_the_real_psutil_is_never_used_to_close(monkeypatch, qapp):
    monkeypatch.setattr(app_screen.QMessageBox, 'question',
                        lambda *a, **k: app_screen.QMessageBox.No)
    fake = _Psutil()
    dialog = _FreeRamDialog(None, psutil_module=fake)
    dialog.boxes[0].setChecked(True)
    dialog.close_button.click()
    assert dialog.outcomes == {}
    dialog.deleteLater()
