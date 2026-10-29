"""Only the canonical console recording and exact checkpoint-log contract; no training."""
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/tutorials')]
import capture_cellpose_training_gui as driver


def run_node():
    record=next(n for n in ast.parse(inspect.getsource(driver)).body if isinstance(n,ast.FunctionDef) and n.name=='record')
    matches=[n for n in ast.walk(record) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='train_run_console'
        for a in i.context_expr.args) for i in n.items)]
    assert len(matches)==1
    outer=next(n for n in record.body if isinstance(n,ast.Try))
    assert matches[0] in outer.body  # not conditional on the worker still running
    assert 'train_run_started' not in inspect.getsource(driver)
    return matches[0]


@pytest.mark.parametrize('progress,visible,error',[
    ('Training model on 6 native annotated images',True,None),
    ('',True,'actual training progress'),
    ('Training model on 6 native annotated images',False,'actual training console'),
])
def test_actual_canonical_block_records_run_even_when_worker_finishes_fast(monkeypatch,progress,visible,error):
    events=[];clips=[]
    @contextmanager
    def clip(capture,name):
        clips.append(name);yield
    monkeypatch.setattr(driver.capture_ui,'clip',clip)
    monkeypatch.setattr(driver.capture_ui,'console_to_end',lambda *args:events.append('console_end'))
    monkeypatch.setattr(driver.capture_ui,'point_at',lambda *args,**kw:events.append('point_at_console'))
    console=SimpleNamespace(isVisible=lambda:visible,visibleRegion=lambda:SimpleNamespace(isEmpty=lambda:False))
    worker=SimpleNamespace(finished=SimpleNamespace(connect=lambda fn:fn(True)),error=SimpleNamespace(connect=lambda fn:None))
    button=object();folder=object()
    screen=SimpleNamespace(_console=console,_console_folder=folder,_btn_run=button,_worker=worker,_worker_thread_is_running=lambda:False)
    outcome={'finished':False,'ok':False,'errors':[]}
    def click(target):
        assert target is button;events.append('actual_run_click')
    values=dict(capture=None,screen=screen,outcome=outcome,timeout=1800,time=driver.time,
        capture_ui=driver.capture_ui,open_fold=lambda *args:events.append('open_console'),click=click,
        settle=lambda seconds:None,console_text=lambda screen:progress)
    code=compile(ast.Module(body=[run_node()],type_ignores=[]),'actual-record-console-block','exec')
    if error:
        with pytest.raises(ValueError,match=error):exec(code,values)
    else:
        exec(code,values)
        assert events==['open_console','actual_run_click','console_end','point_at_console']
        assert outcome=={'finished':True,'ok':True,'errors':[]}
    assert clips==['train_run_console']


@pytest.mark.parametrize('text,valid',[
    ('Training complete\nModel saved at: /actual/models/tutorial_cells.CP_model\n',True),
    ('',False),
    ('Model saved at: /other/models/tutorial_cells.CP_model',False),
    ('Model saved at: /actual/models/tutorial_cells.CP_model\nModel saved at: /actual/models/tutorial_cells.CP_model',False),
])
def test_saved_checkpoint_console_path_is_exact_and_fail_closed(text,valid):
    checkpoint=Path('/actual/models/tutorial_cells.CP_model')
    if valid:assert driver._checkpoint_console_paths(checkpoint,text)==[str(checkpoint)]
    else:
        with pytest.raises(ValueError,match='actual saved checkpoint'):driver._checkpoint_console_paths(checkpoint,text)


def test_native_info_precondition_reuses_existing_helper_before_opening_modules():
    record=next(n for n in ast.parse(inspect.getsource(driver)).body if isinstance(n,ast.FunctionDef) and n.name=='record')
    body=next(n for n in record.body if isinstance(n,ast.Try)).body
    assert isinstance(body[0],ast.ImportFrom) and body[0].module=='capture_external_masks'
    assert body[0].names[0].name=='_enable_console_information'
    assert ast.unparse(body[1])=='_enable_console_information(app, window, capture)'
    assert 'open_module' in ast.unparse(body[2])
