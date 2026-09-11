"""Recheck the real Train import and show the separate API's actual pair figure."""
from pathlib import Path

from capture_barcode_saved_plots import launch
from capture_cellpose_training import record_training
from capture_saved_plots import show_saved_plots
from stage_lesson import read
from train_cellpose_example import digest


def check_idle(screen):
    if screen._worker_thread_is_running() or screen.active_jobs():
        raise ValueError('The review must not start a training or inference job')


def record(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen

    stage=Path(stage)
    api=read(stage/'captures/train_cellpose_explicit_api_v1/scientific_acceptance.json')
    if api.get('accepted') is not True:
        raise ValueError('The actual explicit training must be independently checked first')
    training=api['training']
    for path,value in training['source_hashes'].items():
        if digest(path)!=value:raise ValueError('An original training input changed')
    plot=Path(training['actual_preview']['path'])
    if digest(plot)!=training['actual_preview']['sha256']:
        raise ValueError('The real training-pair figure changed')
    record_training(app,window,stage,captures,capture,settle,write_json,timeout,
                    settings_override=training['settings'],source_check_only=True)
    native=read(captures/'scientific_acceptance.json')
    if native.get('accepted') is not False or 'hold' not in native:
        raise ValueError('Expected the explicit native-source refusal, not successful GUI training')
    write_json(captures/'native_source_import_hold.json',native)
    panels=[p for p in window.findChildren(CellposeWorkbenchScreen) if p.isVisible()]
    if len(panels)!=1:raise ValueError('The actual workbench is not visible')
    panel=panels[0]
    check_idle(panel.train_screen)
    viewer=show_saved_plots(app,window,stage,capture,settle,[plot])
    QTest.mouseClick(panel._tabs.tabBar(),Qt.LeftButton,pos=panel._tabs.tabBar().tabRect(1).center())
    settle(.5);capture('12_actual_apply_tab_no_inference')
    check_idle(panel.apply_screen)
    proof=dict(accepted=True,scope='Native source refusal plus a separately verified training API figure; no GUI training or Apply inference',
               native_source_import=native,api=api,external_viewer=viewer,
               gui_source_control_fixed=False,held_out_accuracy_validated=False,published=False)
    write_json(captures/'scientific_acceptance.json',proof)


if __name__=='__main__':
    raise SystemExit(launch('train_cellpose','train_cellpose_review_and_pair_viewer_v2',
                           '--cellpose-training-review',200))
