"""One complete real-pointer worker-dispatch Apply clip; ROOT launches only."""
from pathlib import Path
import sys, os, argparse, json, hashlib, time, threading,subprocess
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--peer',type=Path,required=True);p.add_argument('--peer-sha256',required=True);args=p.parse_args()
R=args.repo.resolve();S=args.stage.resolve()
sys.path[:0]=[str(R),str(R/'tools/tutorials'),str(Path(__file__).resolve().parent)]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
assert os.environ.get('SPACR_CAPTURE_HOST') and os.environ.get('DISPLAY') not in (None,':0',':0.0')
assert Path(os.environ['SPACR_CAPTURE_HOST']).resolve()==S/'gnome-host'
# gnome_capture_host deliberately gives its command a private HOME of its own;
# use the already prepared separate app profile before ANY application import.
os.environ['HOME']=str(S/'profile');os.environ['SPACR_HOME']=str(S/'app-state')
os.environ['XDG_CONFIG_HOME']=str(S/'config/home');os.environ['XDG_STATE_HOME']=str(S/'state/home')
os.environ['XDG_CACHE_HOME']=str(S/'cache');os.environ['SPACR_LOG_DIR']=str(S/'logs')
assert all(Path(os.environ[k]).resolve().is_relative_to(S) for k in ['HOME','SPACR_HOME','XDG_CONFIG_HOME','XDG_STATE_HOME','XDG_CACHE_HOME','SPACR_LOG_DIR'])
REVISION='87ffcb60cab0eca8772cb1ffaccc12f3f4f48be9'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==REVISION
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
SOURCES={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ('spacr/qt/**/*.py','tools/tutorials/capture*.py','tools/tutorials/human_input.py','tools/tutorials/gnome_capture_host.py') for p in R.glob(pattern)}
EXTERNAL_FILES={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.iterdir() if p.is_file() and p.suffix in ('.py','.sh')}
(S/'exact-source.json').write_text(json.dumps({'commit':REVISION,'sha256':SOURCES,'fresh':True,'private_profile':True,'surface':'actual spaCR in private GNOME compositor; worker physical pointer, owning sampler/recorder/gate','monitor':os.environ['SPACR_CAPTURE_HOST_SIZE'],'top_bar':os.environ.get('SPACR_CAPTURE_HOST_TOP_BAR'),'external_recorder_source':EXTERNAL_FILES},indent=2)+'\n')

from human_input import GuiCaller, HumanProcess
from capture_host_adapter import QTestAdapter
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtWidgets import QApplication,QDialog,QDialogButtonBox,QTabWidget,QSlider,QScrollArea,QAbstractButton,QMenu,QMessageBox,QStyle,QStyleOptionSlider,QComboBox,QWidget
import spacr
assert Path(spacr.__file__).resolve()==R/'spacr/__init__.py'
import spacr.qt
from spacr.qt import preferences
from spacr.qt.widgets.section import Section
begin_ns=time.monotonic_ns();deadline=begin_ns+300_000_000_000
MINIMUM_SECONDS=968/30
PREFIX_SHA=json.loads((Path(__file__).resolve().parent/'captured-prefix-source.json').read_text())['prefix']['sha256']
out=S/'worker-capture-result.json';assert not out.exists()
gui=None;state={'status':'not_completed','error':'none','phase':'controller_starting','pre_apply_public_state':None};holder={}

def failure_location(exc):
    frames=[]
    tb=exc.__traceback__
    while tb is not None:
        code=tb.tb_frame.f_code
        frames.append({'file':Path(code.co_filename).name,'function':code.co_name,'line':tb.tb_lineno})
        tb=tb.tb_next
    return {'exception_type':type(exc).__name__,'exception_locations':frames[-12:]}

def left():
    remaining=(deadline-time.monotonic_ns())/1e9
    if remaining<=0:raise TimeoutError('diagnostic budget exhausted')
    return remaining

def call(fn):return gui(fn,timeout=min(2.0,left()))
def wait(fn):
    while left()>0:
        value=call(fn)
        if value:return value
        time.sleep(.03)
    raise TimeoutError('diagnostic budget exhausted')
def wait_initial_window(fn):
    while left()>0:
        request_started=time.monotonic_ns()
        value=gui(fn,timeout=left())
        state['initial_window_dispatch_elapsed_ns']=time.monotonic_ns()-request_started
        if value:return value
        time.sleep(min(.03,left()))
    raise TimeoutError('diagnostic budget exhausted')

def center(widget,pos=None):
    if not widget.isVisible() or not widget.isEnabled():raise RuntimeError('public control unavailable')
    point=widget.mapToGlobal(pos if pos is not None else widget.rect().center())
    if QTestAdapter.reachable(widget,(point.x(),point.y())) is not None:raise RuntimeError('public control occluded')
    return point.x(),point.y()
def click(widget,pos=None):
    xy=call(lambda:center(widget,pos));left();holder['human'].click_at(*xy)
def visible(widget):
    if not widget.isVisible():return False
    point=widget.mapToGlobal(widget.rect().center());ancestor=widget.parentWidget()
    while ancestor is not None:
        if isinstance(ancestor,QScrollArea) and not ancestor.viewport().rect().contains(ancestor.viewport().mapFromGlobal(point)):return False
        ancestor=ancestor.parentWidget()
    return True

def reveal(widget):
    for _ in range(8):
        if call(lambda:visible(widget)):return
        def scroll_target():
            parent=widget.parentWidget()
            while parent is not None and not isinstance(parent,QScrollArea):parent=parent.parentWidget()
            if parent is None:raise RuntimeError('no public scroll route')
            bar=parent.verticalScrollBar();target=bar if bar.isVisible() else parent.viewport()
            point=target.rect().center() if bar.isVisible() else QPoint(target.width()-6,target.height()//2)
            xy=center(target,point);up=widget.mapToGlobal(widget.rect().center()).y()<parent.viewport().mapToGlobal(parent.viewport().rect().center()).y()
            return xy,-2 if up else 2
        xy,steps=call(scroll_target);holder['human'].move_to(*xy);holder['human'].scroll(steps);time.sleep(.04)
    raise RuntimeError('public scroll route did not reveal control')

def show_tab(dialog,title):
    tabs=call(lambda:dialog.findChild(QTabWidget,'PreferencesTabs'))
    holder['tabs']=tabs
    index=call(lambda:next(i for i in range(tabs.count()) if tabs.tabText(i).replace('&','')==title))
    if call(tabs.currentIndex)!=index:click(call(tabs.tabBar),call(lambda:tabs.tabBar().tabRect(index).center()))
    wait(lambda:tabs.currentIndex()==index)
    time.sleep(.6)
    return call(tabs.currentWidget)

def hover(widget):
    def hover_point():
        if not widget.isVisible():raise RuntimeError('hover target hidden')
        point=widget.mapToGlobal(widget.rect().center())
        why=QTestAdapter.reachable(widget,(point.x(),point.y()))
        if why is not None:
            classification=('other_window' if why.startswith('covered by another window') else 'same_window_cover' if why.startswith('covered by ') else 'no_application_hit' if why=='nothing of the app under the point' else 'unmapped_window' if why=='target window is not mapped' else 'hidden_target' if why=='target is hidden' else 'other_geometry_rejection')
            state['pointer_target_failure']={'context':'hover','classification':classification,'inside_ancestor_viewports':visible(widget),'target_enabled':widget.isEnabled()}
            raise RuntimeError('hover target occluded')
        return point.x(),point.y()
    xy=call(hover_point);holder['human'].move_to(*xy)

def wheel(scroll,steps):
    def target():
        bar=scroll.verticalScrollBar()
        widget=bar if bar.isVisible() else scroll.viewport()
        point=widget.rect().center() if bar.isVisible() else QPoint(widget.width()-6,widget.height()//2)
        return center(widget,point)
    xy=call(target);holder['human'].move_to(*xy);holder['human'].scroll(steps);time.sleep(.15)

def section_scroll(section):
    def ancestor():
        parent=section.parentWidget()
        while parent is not None and not isinstance(parent,QScrollArea):parent=parent.parentWidget()
        return parent
    return call(ancestor)

def toggle(section,expanded):
    if call(section.is_expanded)!=expanded:
        header=call(section.header);reveal(header);click(header);wait(lambda:section.is_expanded()==expanded);time.sleep(.4)

def set_slider(slider,value):
    def geometry():
        option=QStyleOptionSlider();slider.initStyleOption(option)
        handle=slider.style().subControlRect(QStyle.CC_Slider,option,QStyle.SC_SliderHandle,slider)
        span=slider.width()-handle.width()
        now=QStyle.sliderPositionFromValue(slider.minimum(),slider.maximum(),slider.value(),span)
        goal=QStyle.sliderPositionFromValue(slider.minimum(),slider.maximum(),value,span)
        return center(slider,handle.center()),goal-now
    xy,delta=call(geometry)
    if abs(delta)>=3:
        holder['human'].move_to(*xy);time.sleep(.12);holder['human'].button_down('left');time.sleep(.12)
        holder['human'].move_to(xy[0]+delta,xy[1]);time.sleep(.1);holder['human'].button_up('left');time.sleep(.2)
    for _ in range(101):
        current=call(slider.value)
        if current==value:return
        holder['human'].press('Right' if current<value else 'Left')
    raise RuntimeError('slider did not reach requested value')

def worker():
    runtime=None
    try:
        human=holder['human']
        def current_window():
            cls=getattr(sys.modules.get('spacr.qt.app'),'MainWindow',None)
            return next((x for x in QApplication.topLevelWidgets() if cls is not None and isinstance(x,cls) and x.isVisible()),None)
        state['phase']='initial_window_readiness';window=wait_initial_window(current_window)
        maximize=call(lambda:window.findChild(QAbstractButton,'MaximiseWindow'))
        if not call(window.isMaximized):
            if maximize is None:raise RuntimeError('normal maximize control absent')
            click(maximize)
        state['phase']='normal_startup_settle';time.sleep(6);left()
        from capture_policy import verify_fresh_start,verify_appearance,verify_visible_paths
        call(lambda:verify_fresh_start(window))
        def init_runtime():
            from capture_host_mode import HostRuntime
            from capture_host_adapter import QTestAdapter,EventPump
            adapter=QTestAdapter(human,pump=EventPump(),strict=True,log=(S/'logs/adapter.jsonl').open('x'))
            return HostRuntime(window,adapter,stage=S,fps=30)
        runtime=call(init_runtime);holder['runtime']=runtime
        state['phase']='public_preferences_navigation'
        bar=call(window.menuBar);action=call(lambda:next(a for a in bar.actions() if a.text().replace('&','')=='spaCR'))
        click(bar,call(lambda:bar.actionGeometry(action).center()))
        menu=wait(lambda:QApplication.activePopupWidget());assert isinstance(menu,QMenu)
        item=call(lambda:next(a for a in menu.actions() if a.text().replace('&','').startswith('Preferences')))
        click(menu,call(lambda:menu.actionGeometry(item).center()))
        dialog=wait(lambda:next((x for x in QApplication.topLevelWidgets() if isinstance(x,QDialog) and x.isVisible() and x.findChild(QTabWidget,'PreferencesTabs') is not None),None))
        time.sleep(.8)
        initial_preferences=call(lambda:{'pane_opacity_pct':round(preferences.get_pane_opacity()*100),'font_scale':preferences.get_font_scale()})
        state['phase']='prior_tour_performance'
        selector=call(lambda:dialog.findChild(QComboBox,'PerformanceLevel'));show_tab(dialog,'Performance')
        hover(selector);time.sleep(1.2);before=call(selector.currentIndex);click(selector)
        view=call(selector.view);wait(view.isVisible);time.sleep(.5)
        row=call(lambda:view.visualRect(selector.model().index(before,selector.modelColumn())).center())
        click(call(view.viewport),row);wait(lambda:not view.isVisible());assert call(selector.currentIndex)==before
        state['phase']='prior_tour_general';page=show_tab(dialog,'General')
        controls=call(lambda:[x for x in page.findChildren(QComboBox) if x.isVisible()][:1]+[x for x in page.findChildren(QSlider) if x.isVisible()][:2])
        for control in controls:hover(control);time.sleep(1.5)
        state['phase']='prior_tour_appearance';page=show_tab(dialog,'Appearance')
        controls=call(lambda:[x for x in page.findChildren(QWidget) if x.isVisible() and x.objectName() in ('TooltipDelay','ShowTooltips')][:2])
        for control in controls:hover(control);time.sleep(1.5)
        sections=call(lambda:{x.title().casefold():x for x in page.findChildren(Section)})
        theme=sections['theme'];animation=sections['animation'];scroll=section_scroll(theme)
        toggle(animation,False);toggle(theme,True)
        if scroll is not None:
            for _ in range(4):wheel(scroll,1);time.sleep(.8)
        toggle(theme,False);toggle(animation,True)
        if scroll is not None:
            for _ in range(3):wheel(scroll,1);time.sleep(1.2)
        independence={}
        animation_controls={}
        for settings_name,main_name in [('PopupBackdropSpeed','AmbientSpeed'),('PopupBackdropSize','AmbientSize'),('PopupBackdropResolution','AmbientResolution'),('PopupBackdropDensity','AmbientDensity')]:
            settings=call(lambda:dialog.findChild(QWidget,settings_name));main=call(lambda:dialog.findChild(QWidget,main_name))
            if settings is None or main is None or not call(settings.isVisible):independence[settings_name]={'shown':False};continue
            animation_controls[settings_name]=(settings,main,call(settings.value),call(main.value))
            if not call(settings.isEnabled):reveal(settings);hover(settings);time.sleep(1);independence[settings_name]={'shown':True,'enabled':False,'public_reveal_before_hover':True};continue
            reveal(settings);hover(settings);original=call(settings.value);main_original=call(main.value)
            human.press('Right');edited=call(settings.value);human.press('Left')
            assert edited!=original and call(settings.value)==original and call(main.value)==main_original
            independence[settings_name]={'before':original,'edited':edited,'main_unchanged':True,'restored':True}
            time.sleep(1)
        if scroll is not None:reveal(call(animation.header))
        toggle(animation,False)
        state['phase']='begin_complete_apply_clip'
        captures=S/'captures/home';captures.mkdir(parents=True,exist_ok=True)
        def preflight():
            appearance=verify_appearance(window,home_preferences=True,default_field=True)
            verify_visible_paths([x for x in QApplication.topLevelWidgets() if x.isVisible()],S)
            return appearance
        appearance=call(preflight)
        gui(lambda:runtime.start_scene(captures,'13h_preferences_apply',MINIMUM_SECONDS,appearance=appearance),timeout=min(30,left()))
        state['phase']='apply_clip_physical_actions'
        toggle(theme,True)
        slider=call(lambda:dialog.findChild(QSlider,'PaneOpacity'));reveal(slider)
        original=call(lambda:round(preferences.get_pane_opacity()*100));assert original==60
        set_slider(slider,55);time.sleep(.6)
        apply=call(lambda:dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply))
        state['phase']='snapshot_before_physical_apply'
        def pre_apply_snapshot():
            tabs=holder['tabs'];tab_index=tabs.currentIndex()
            known_titles={'General','Appearance','Performance','Modules','Storage'}
            title=tabs.tabText(tab_index).replace('&','')
            active=QApplication.activeModalWidget()
            def rectangle(widget):
                top_left=widget.mapToGlobal(QPoint(0,0));return [top_left.x(),top_left.y(),widget.width(),widget.height()]
            animations={name:{'popup_value':a.value(),'main_value':b.value(),'popup_value_restored':a.value()==before,'main_value_unchanged':b.value()==main_before,'enabled':a.isEnabled()} for name,(a,b,before,main_before) in animation_controls.items()}
            return {'schema':1,'phase':'after_opacity_edit_and_0_6s_settle_before_physical_apply','captured_old_runtime_prestate_available':False,'source_prefix':PREFIX_SHA,'equivalence_scope':'Public control/state comparison only, not historical wall holds/cache state','tab':{'index':tab_index,'label':title if title in known_titles else 'other_public_tab'},'folds':{'theme_expanded':theme.is_expanded(),'animation_expanded':animation.is_expanded()},'scroll':({'value':scroll.verticalScrollBar().value(),'minimum':scroll.verticalScrollBar().minimum(),'maximum':scroll.verticalScrollBar().maximum(),'viewport_size':[scroll.viewport().width(),scroll.viewport().height()]} if scroll is not None else None),'geometry':{'main_window':rectangle(window),'preferences_dialog':rectangle(dialog),'pane_opacity_slider':rectangle(slider)},'class_labels':{'window':'MainWindow','dialog':'QDialog','tabs':'QTabWidget','opacity':'QSlider'},'pending_modal':{'active_present':active is not None,'active_is_preferences':active is dialog,'active_is_other':active is not None and active is not dialog,'apply_confirmation_present':getattr(dialog,'_apply_confirmation',None) is not None},'known_preferences':{'initial_opacity_pct':initial_preferences['pane_opacity_pct'],'persisted_opacity_pct':round(preferences.get_pane_opacity()*100),'draft_slider_pct':slider.value(),'initial_font_scale':initial_preferences['font_scale'],'font_scale':preferences.get_font_scale(),'font_scale_unchanged':preferences.get_font_scale()==initial_preferences['font_scale'],'performance_initial_index':before,'performance_current_index':selector.currentIndex(),'performance_unchanged':selector.currentIndex()==before,'animation_controls':animations},'previous_animation_readbacks':independence}
        state['pre_apply_public_state']=call(pre_apply_snapshot)
        state['phase']='physical_apply_dispatch'
        click(apply)
        question=wait(lambda:getattr(dialog,'_apply_confirmation',None))
        assert call(lambda:isinstance(question,QMessageBox) and question.isVisible() and question.windowModality()==Qt.WindowModal and question.defaultButton().text().replace('&','')=='Revert' and question.escapeButton()==question.defaultButton())
        time.sleep(.3);assert call(lambda:preferences.get_pane_opacity())==.55
        def focus():
            a,b=dialog.geometry(),question.geometry();ap=dialog.mapToGlobal(QPoint(0,0));bp=question.mapToGlobal(QPoint(0,0))
            x=min(ap.x(),bp.x());y=min(ap.y(),bp.y());right=max(ap.x()+a.width(),bp.x()+b.width());bottom=max(ap.y()+a.height(),bp.y()+b.height())
            return runtime.to_handoff([x,y,right-x,bottom-y])
        focus_rect=call(focus)
        for button in call(question.buttons):hover(button);time.sleep(1)
        while call(runtime.elapsed)<MINIMUM_SECONDS-2.5:time.sleep(.02);left()
        revert=call(lambda:next(b for b in question.buttons() if b.text().replace('&','')=='Revert'))
        click(revert);wait(lambda:getattr(dialog,'_apply_confirmation',None) is None)
        assert call(lambda:round(preferences.get_pane_opacity()*100))==60
        while call(runtime.elapsed)<MINIMUM_SECONDS:time.sleep(.02);left()
        entry=gui(runtime.end_scene,timeout=min(30,left()));entry['focus_rect']=focus_rect
        holder['frame']=entry
        state.update(status='captured_selected_scene',revert_verified=True,original_opacity_pct=60,preview_opacity_pct=55,settings_controls_independent=independence,prior_tour_controls_completed=True,full_native_no_trim=True)
        cancel=call(lambda:dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel));click(cancel)
        wait(lambda:not dialog.isVisible())
    except BaseException as exc:
        state.update(status='capture_incomplete',failure_phase=state['phase']);state.update(failure_location(exc))
    finally:
        try:gui(QApplication.quit,timeout=1)
        except BaseException:pass

human=None;human_log=None;thread=None;app_rc=None
try:
    state['phase']='human_startup'
    human_log=(S/'logs/human-private.log').open('x')
    human=HumanProcess.__new__(HumanProcess)
    HumanProcess.__init__(human,log=human_log)
    state['phase']='production_shell_escape_and_pointer_placement'
    human.press('Escape')
    human.warp(human.width*.62,human.height*.58)
    holder['human']=human
    state['human_startup_elapsed_ns']=time.monotonic_ns()-begin_ns
    gui=GuiCaller()
    thread=threading.Thread(target=worker,name='native-apply-observer-owner',daemon=True);thread.start()
    app_rc=spacr.qt.run(['--fresh','--no-setup']);thread.join(timeout=1)
except BaseException as exc:
    state.update(status='diagnostic_incomplete',error=type(exc).__name__,failure_phase=state['phase'],elapsed_since_native_begin_ns=time.monotonic_ns()-begin_ns)
    state.update(failure_location(exc))
    if not out.exists():out.write_text(json.dumps(state,indent=2)+'\n')
finally:
    if human is not None and hasattr(human,'_proc'):
        if thread is None or not thread.is_alive():
            human.close()
        else:
            state['human_close_deferred_to_marked_cleanup']=True
    if human_log is not None:human_log.close()
captures=S/'captures/home';captures.mkdir(parents=True,exist_ok=True)
source_modified=any(hashlib.sha256((R/k).read_bytes()).hexdigest()!=v for k,v in SOURCES.items())
external_modified=any(hashlib.sha256((Path(__file__).resolve().parent/k).read_bytes()).hexdigest()!=v for k,v in EXTERNAL_FILES.items())
provenance={'commit':REVISION,'module':'home','completed_capture':False,'app_source_modified':source_modified,'cache_isolated_with_bind_mount':False,'private_profile':True,'selected_scenes':'13h_preferences_apply','external_recorder_source':EXTERNAL_FILES,'external_recorder_source_modified':external_modified,'dispatch_context':'worker HumanProcess with ordinary Qt event loop; lifecycle only uses owning recorder adapter','whole_lesson_fresh':False}
summary={};failed=[];recorder_rc=1
if state['status']=='captured_selected_scene' and app_rc==0 and not source_modified and not external_modified and thread is not None and not thread.is_alive():
    runtime=holder['runtime']
    try:
        scene=runtime.pending.pop()
        final=runtime.post_process(scene,pump=False)
        frame=holder['frame'];frame.update(final)
        (captures/'frames.json').write_text(json.dumps({'13h_preferences_apply':frame},indent=2)+'\n')
        summary=runtime.gates;failed=runtime.failed;recorder_rc=0
        provenance.update(completed_capture=not failed,status='gate_failed' if failed else 'completed',gate_failed_clips=failed)
        (captures/'host_gate_summary.json').write_text(json.dumps({'clips':summary,'failed':failed,'passed':not failed},indent=2)+'\n')
    except BaseException as exc:state.update(status='postprocess_incomplete');state.update(failure_location(exc))
(captures/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
out.write_text(json.dumps(state,indent=2)+'\n')
(S/'host-run.json').write_text(json.dumps({'app_exit_code':app_rc,'recorder_rc':recorder_rc,'failure':None if recorder_rc==0 else state.get('exception_type','capture_incomplete'),'clips':summary,'gate_failed':failed},indent=2)+'\n')
raise SystemExit(1 if recorder_rc else 2 if failed else 0)

