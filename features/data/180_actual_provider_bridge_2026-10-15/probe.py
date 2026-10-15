from __future__ import annotations
import contextlib, copy, gc, hashlib, json, os, pathlib, signal, sys, time
from unittest.mock import patch

ROOT=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-provider-bridge-20261015')
SCRATCH=pathlib.Path('/mnt/wd4tb/scratch/home-b-n180-provider-bridge-20261015')
sys.path.insert(0,str(ROOT))
os.environ['XDG_CONFIG_HOME']=str(SCRATCH/'config')
os.environ['XDG_CACHE_HOME']=str(SCRATCH/'cache')
os.environ['SPACR_MODEL_DOWNLOAD']='0'
signal.alarm(90)
import pandas as pd
from PySide6.QtCore import QCoreApplication,QEvent
from PySide6.QtWidgets import QApplication
from spacr import workspace
from spacr.qt import gc_policy
from spacr.qt.theme import apply_qpalette,stylesheet
from spacr.qt.screens.app_screen import AppScreen,_REGRESSION_RESULTS
from spacr.qt.widgets import measurement_compare_dialog as mcd
from spacr.qt.widgets.regression_results import RegressionResultsPanel

app=QApplication.instance() or QApplication([])
gc_policy.install(app)
apply_qpalette(app);app.setStyleSheet(stylesheet())
objects=pd.DataFrame({'prc':['p_r_c1','p_r_c1','p_r_c2'],'grna':['g1','g2','g1'],'cell_area':[3.,4.,5.],'cell_perimeter':[10.,12.,14.]})
setup_refresh=[];guard_calls=[];events=[];screens=[]
def fail(name):
 def reject(*a,**k):
  guard_calls.append(name)
  raise AssertionError('Forbidden rendering/join/loading during registry bridge: '+name)
 return reject

def built(screen):
 assert screen._build_owed_part(_REGRESSION_RESULTS)
 result=screen._if_built('_results_panel');montage=screen._if_built('_cell_montage');grid=screen._if_built('_figure_grid')
 assert isinstance(result,RegressionResultsPanel) and montage is not None and grid is not None
 providers=workspace.providers()
 for name,obj in [('regression',result),('montage',montage),('figures',grid)]:assert providers['regression:'+name]() is obj
 with patch.object(montage,'rows_to_compare',lambda:objects),patch.object(montage,'picked_groups',lambda:{'picked':[0,2]}),patch.object(mcd.MeasurementComparePanel,'refresh',lambda self,*a:setup_refresh.append('constructor refresh suppressed')):
  montage._ensure_graph_tab()
 graph=montage._graph_panel
 assert isinstance(graph,mcd.MeasurementComparePanel)
 assert graph._canvas is None and graph._comparison is None
 events.append('explicit user-equivalent deferred panel and Graph construction')
 return result,montage,grid,graph

@contextlib.contextmanager
def guards(result,montage,graph):
 with contextlib.ExitStack() as stack:
  for obj,names in [(graph,['refresh','_draw','_draw_and_report','join_the_tables']), (montage,['build','_redraw_from_cache']), (result,['load','refresh_views'])]:
   for name in names:stack.enter_context(patch.object(obj,name,fail(type(obj).__name__+'.'+name)))
  for name in ['build','with_statistics','join_measurements','save']:stack.enter_context(patch.object(mcd,name,fail('mcd.'+name)))
  stack.enter_context(patch.object(pd,'read_csv',fail('pandas.read_csv')))
  yield

def close(screen):
 screen.close();screen.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents();gc.collect()
 assert not any(name.startswith('regression:') for name in workspace.providers())
 events.append('screen closure unregisters all regression providers')

try:
 workspace.clear_providers()
 first=AppScreen('regression');screens.append(first)
 assert first._part_is_owed(_REGRESSION_RESULTS)
 unvisited=workspace.collect(workspace.providers(),app_key='regression',hash_files=False)
 assert first._part_is_owed(_REGRESSION_RESULTS)
 assert not any('regression:'+name in unvisited['sections'] for name in ['montage','regression','figures'])
 events.append('unvisited actual screen collection remains deferred')
 result,montage,grid,graph=built(first)
 comparison={'measurement':'cell_perimeter','operator':'-','second':'cell_area','level':'cell','kind':'bar','spread':'sd','scope':'wells','contrast':'within_well','controls':'control_a, control_b','chosen_wells':[],'selected_guides':['g1'],'selected_wells':[],'join_png_list':False,'join_dependent':True,'only':'picked'}
 assert graph.apply_workspace_state(comparison)
 expected_graph=graph.workspace_state()
 run_key=str(SCRATCH/'unloaded-run')
 run_state=result.plot_state();run_state.update({'selected_keys':['gene_fraction:gene[g1]','gene_fraction:gene[g2]'],'selected_key':'gene_fraction:gene[g2]','colour_by':None,'symbol_by':None,'opacity_by':'batch','threshold_multiplier':2.5})
 assert result.apply_workspace_state({'runs':{run_key:run_state}})
 grid.set_target_cell_width(320)
 with guards(result,montage,graph):
  document=workspace.collect(workspace.providers(),app_key='regression',hash_files=False)
  assert not document.get('problems'),document.get('problems')
  for name in ['montage','regression','figures']:assert 'regression:'+name in document['sections']
  assert document['sections']['regression:montage']['comparison']==expected_graph
  assert document['sections']['regression:regression']['runs'][run_key]['opacity_by']=='batch'
  path=workspace.save(SCRATCH/'saved-workspace',document,mode='reference');assert path.is_file()
  saved=workspace.load(SCRATCH/'saved-workspace');assert saved is not None
 events.append('all actual providers collected and saved through normal workspace APIs')
 close(first);screens.remove(first)
 second=AppScreen('regression');screens.append(second)
 assert second._part_is_owed(_REGRESSION_RESULTS)
 untouched=copy.deepcopy(saved)
 deferred_report=workspace.restore(workspace.providers(),saved,run_dir=SCRATCH/'saved-workspace')
 assert second._part_is_owed(_REGRESSION_RESULTS),'restore built deferred panels'
 skipped={r['section'] for r in deferred_report['skipped']}
 assert {'regression:montage','regression:regression','regression:figures'}<=skipped,deferred_report
 assert saved==untouched
 events.append('restoring to unvisited screen reports missing built contributors without construction')
 new_result,new_montage,new_grid,new_graph=built(second)
 with guards(new_result,new_montage,new_graph):
  report=workspace.restore(workspace.providers(),saved,run_dir=SCRATCH/'saved-workspace')
  assert {'regression:montage','regression:regression','regression:figures'}<=set(report['restored']),report
  assert new_graph.workspace_state()==expected_graph
  assert new_result.workspace_state()['runs'][run_key]==saved['sections']['regression:regression']['runs'][run_key]
  assert new_grid.workspace_state()['cell_width']==320
  assert saved==untouched
  assert new_graph._canvas is None and new_graph._comparison is None
 assert not guard_calls,guard_calls
 events.append('visited real contributors restore exact Graph controls, queued regression run views and figure-grid size')
 close(second);screens.remove(second)
 receipt={'source_pin':'9e49e23fe2debfc19e52720a6eb53e3ccd6b4242','result':'PASS','events':events,'provider_injection':False,'provider_filtering':False,'render_join_read_guard_calls':guard_calls,'setup_refresh_suppression_count':len(setup_refresh),'visited_restored':report['restored'],'unvisited_report':deferred_report,'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['spacr/qt/screens/app_screen.py','spacr/workspace.py','spacr/qt/widgets/cell_montage_view.py','spacr/qt/widgets/measurement_compare_dialog.py','spacr/qt/widgets/regression_results.py','spacr/qt/widgets/figure_grid_view.py']},'limits':['Registry bridge only; already-tested loaded data/render consumers are not rerun.','Regression run views intentionally have no live data path; restoration retains them without file loading.','Actual comparison controls use a tiny in-memory object fixture; constructor refresh is suppressed and no comparison/canvas is computed.','All real providers and full settings are used without mapping injection/filtering.','No scientific inference, image loading, graph rendering, join, pipeline or native desktop acceptance.']}
 (SCRATCH/'result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'result':'PASS','events':events,'guard_calls':guard_calls,'restored':report['restored']},indent=2),flush=True)
finally:
 for screen in screens:
  try:close(screen)
  except Exception as exc:print('cleanup:',repr(exc),flush=True)
 workspace.clear_providers()
 from spacr.qt.job_runner import shutdown_all
 shutdown_all(1000)
 gc_policy.uninstall()
 signal.alarm(0)
