"""Real Qt headers exercise only the changed native settings-tour route."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QScrollArea, QVBoxLayout, QWidget

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/tutorials')]
from spacr.qt.widgets.section import Section
spec=importlib.util.spec_from_file_location('actual_training_capture',ROOT/'tools/tutorials/capture_cellpose_training_gui.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
print('ACTUAL_DRIVER_IMPORT',driver.__file__)


@pytest.fixture
def form(qtbot):
    outer=QWidget();layout=QVBoxLayout(outer);scroll=QScrollArea();scroll.setWidgetResizable(True);layout.addWidget(scroll)
    body=QWidget();rows=QVBoxLayout(body);scroll.setWidget(body)
    wanted=Section('Required');wanted.add_row('Field',QLineEdit())
    unused=Section('Unused',expanded=True);unused.add_row('Other',QLineEdit())
    rows.addWidget(wanted);rows.addWidget(unused);rows.addStretch()
    qtbot.addWidget(outer);outer.resize(800,600);outer.show();qtbot.wait(50)
    mapping={'one':wanted,'two':wanted,'unused':unused}
    bar=SimpleNamespace(section_of=mapping.get,indexed_keys=lambda:list(mapping))
    screen=SimpleNamespace(_settings_search=bar,_settings_scroll=scroll,_test_owner=outer)
    clicks=[]
    def click(widget):
        assert widget.isVisible() and widget.isEnabled() and not widget.visibleRegion().isEmpty()
        clicks.append(widget);QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center())
    def settle(seconds):qtbot.wait(max(20,int(seconds*1000)))
    return screen,wanted,unused,clicks,click,settle,rows,mapping


def test_hidden_unused_is_untouched_and_duplicate_wanted_header_clicked_once(form):
    screen,wanted,unused,clicks,click,settle,_,_=form
    unused.hide()
    driver._tour_sections(None,screen,['one','two'],click,settle)
    assert clicks==[wanted._header] and unused.is_expanded() and wanted.is_expanded()


def test_visible_unused_closes_and_required_nested_ancestors_open_in_order(form):
    screen,parent,unused,clicks,click,settle,_,mapping=form
    child=Section('Nested');child.add_row('Child',QLineEdit());parent.add_widget(child)
    mapping['child']=child
    driver._tour_sections(None,screen,['child'],click,settle)
    assert clicks==[unused._header,parent._header,child._header]
    assert parent.is_expanded() and child.is_expanded() and child._header.isVisible() and not unused.is_expanded()


@pytest.mark.parametrize('state',['hidden','disabled'])
def test_required_unavailable_header_remains_rejected(form,state):
    screen,wanted,_,_,click,settle,_,_=form
    if state=='hidden':wanted.hide()
    else:wanted._header.setEnabled(False)
    with pytest.raises(ValueError,match='Required settings section is unavailable'):
        driver._tour_sections(None,screen,['one'],click,settle)


def test_actual_training_form_keeps_normalization_ancestors_and_required_fields(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL,install
    screen=AppScreen(app_key='train_cellpose');qtbot.addWidget(screen)
    install(screen);screen.resize(1600,1000);screen.show();qtbot.wait(300)
    bar=screen._settings_search;bar.set_level(ALL);bar.set_query('');qtbot.wait(300)
    keys=['normalize','percentiles','scale_range','min_train_masks']
    clicks=[]
    def click(widget):
        assert widget.isVisible() and widget.isEnabled() and not widget.visibleRegion().isEmpty()
        clicks.append(widget.text());QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center())
    def settle(seconds):qtbot.wait(int(seconds*1000))
    driver._tour_sections(None,screen,keys,click,settle)
    print('ACTUAL_NORMALIZATION_HEADERS',clicks)
    assert all(screen._settings_model._widgets[key].isVisible() for key in keys)
    assert len(clicks)==len(set(clicks))
