"""Read final calibration captions through actual controls and the preview table."""
import hashlib,html,json,os,sys,subprocess
from pathlib import Path
from unittest.mock import patch
ROOT=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT))
EXPECTED_HEAD=sys.argv[3];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==EXPECTED_HEAD
from PySide6.QtCore import QCoreApplication,QEvent,QSettings
from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPushButton
from spacr.qt import preferences
from spacr.qt.i18n import current_language,retranslate_widget_tree,tr
from spacr.qt.i18n_catalogs import setting_label,setting_tooltip
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import _setting_label_for_field,api_docs_url
from spacr.qt.screens.settings_flow_index import SETTINGS_WITH_A_FLOW_SECTION
from spacr.qt.widgets import calibration_preview as preview
from spacr.qt.widgets.console_panel import ConsolePanel
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
LANGS=['sv','de','es','zh_CN','pt','hi','ko','is','fr'];KEYS=['intensity_calibration_sessions','intensity_calibration_session_column','intensity_calibration_beads','intensity_calibration_reference_session'];NAMES=['IntensityCalibrationSessions','IntensityCalibrationSessionColumn','IntensityCalibrationBeads','IntensityCalibrationReferenceSession'];DEFAULTS=['','session','',''];REVIEW='2026-10-15-f580-session-controls-home-b.json'
OLD_KEYS=['intensity_calibration','intensity_calibration_wells','intensity_calibration_reference_plate','intensity_calibration_statistic']
FILES=['spacr/settings.py','spacr/intensity_rescale.py','spacr/measure.py','spacr/qt/screens/settings_model.py','spacr/qt/widgets/calibration_preview.py','spacr/qt/screens/settings_flow_index.py']+[f'spacr/qt/i18n_catalogs/{l}.py' for l in ['en',*LANGS]]
FILES += ['spacr/qt/screens/app_screen.py','spacr/qt/i18n.py','spacr/qt/i18n_catalogs/__init__.py','spacr/qt/widgets/console_panel.py','docs/source/_generated/settings_flow.rst']+[f'docs/i18n/reviewed/runtime/{l}/{REVIEW}' for l in LANGS]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={n:sha(ROOT/n) for n in FILES};app=QApplication.instance() or QApplication([]);ini=OUT/'preferences.ini';assert not ini.exists();rows=[];diagnostics=[];os.environ.pop('SPACR_LANGUAGE',None);old_tips=[];tables=[];usage_calls=[]
packet=json.loads((ROOT/'features/data/580_session_gui_2026-10-15/dependency-packet.json').read_text());assert all(before[p]==h for p,h in packet['source_sha256'].items())
flow=(ROOT/'docs/source/_generated/settings_flow.rst').read_text()
def skip_usage(self):
 usage_calls.append(type(self).__name__)
def report(snapshot):
 return [{'source':'/inert/merged','calibration':{'version':2,'reference_session':'day2','sessions':{'day1':{'gain':{'0':.5},'reference_statistic':{'0':250},'n_reference_fields':3}}}}]
with patch.object(preferences,'_settings',lambda:QSettings(str(ini),QSettings.IniFormat)),patch.object(ConsolePanel,'refresh_ai_usage',skip_usage),patch.object(preview,'_plan_gains',report):
 preferences.set_language('en');preferences._set_show_alpha_features(True);screen=AppScreen('measure')
 try:
  for key in [*KEYS,*OLD_KEYS]:screen._open_the_heading_of(key)
  screen._refresh_alpha_visibility();model=screen._settings_model
  for language in LANGS:
   records=json.loads((ROOT/f'docs/i18n/reviewed/runtime/{language}/{REVIEW}').read_text())['records'];by={(r['table'],r['key']):r for r in records};assert len(records)==18
   preferences.set_language(language);assert current_language()==language;retranslate_widget_tree(screen,language)
   for key,name,default in zip(KEYS,NAMES,DEFAULTS):
    field=model._widgets[key];label=_setting_label_for_field(screen,field);assert isinstance(field,QLineEdit) and field.objectName()==name;assert isinstance(label,QLabel);assert screen.setting_row_is_visible(key);assert field.text()==default
    assert label.text()==by['setting_labels',key]['translation'];assert '<b>'+html.escape(label.text())+'</b>' in label.toolTip();assert f'.. _setting-flow-{key}:' in flow;canonical=field.property('apiTooltipDescriptionSource');assert canonical.endswith(by['setting_tooltips',key]['source']);assert field.property('apiTooltipDescription')==canonical;assert html.escape(by['setting_tooltips',key]['translation']) in label.toolTip();assert key in SETTINGS_WITH_A_FLOW_SECTION;url=api_docs_url('measure',key,language);assert f'settings_flow.html?lang={language}#setting-flow-{key}' in url;assert html.escape(url,quote=True) in label.toolTip();rows.append({'language':language,'key':key,'default':field.text(),'label':label.text(),'url':url})
   for key in OLD_KEYS:
    field=model._widgets[key];label=_setting_label_for_field(screen,field);assert isinstance(label,QLabel),(language,key,type(label));record=by['setting_tooltips',key];assert html.escape(record['translation']) in label.toolTip(),(language,key,label.toolTip());old_tips.append({'language':language,'key':key,'tooltip':label.toolTip()})
   for r in records:
    if r['table']=='setting_labels':rendered=setting_label(r['key'],r['source'],language,'measure')
    elif r['table']=='setting_tooltips':rendered=setting_tooltip(r['key'],r['source'],language,'measure')
    else:
     assert r['table']=='ui';rendered=tr(r['source'],language)
    assert rendered==r['translation'],(language,r['table'],r['key'],rendered);diagnostics.append({'language':language,'table':r['table'],'key':r['key'],'target_sha256':hashlib.sha256(rendered.encode()).hexdigest()})
   button=model._widgets['intensity_calibration_wells'].findChild(QPushButton,'CalibrationPreviewButton');assert button is not None;button_source='Calculate reference gains without running Measure or changing files.';assert button.toolTip()==by['ui',button_source]['translation']
   dialog=preview._CalibrationPreview(lambda:preview._preview_settings(model),threaded=False)
   try:
    dialog._start();assert dialog.table.horizontalHeaderItem(1).text()==tr('Session',language);assert dialog.table.horizontalHeaderItem(2).text()==by['ui','Reference session']['translation'];assert dialog.table.item(0,1).text()=='day1';assert dialog.table.item(0,2).text()=='day2';assert dialog.table.item(0,4).text()=='0.5';assert dialog.table.rowCount()==1
    actual_row=[dialog.table.item(0,c).text() for c in range(7)];assert actual_row==['/inert/merged','day1','day2','0','0.5','250','3'];texts=[label.text() for label in dialog.findChildren(QLabel)]
    for r in records:
     if r['table']=='ui' and r['key'] not in (button_source,'Reference session'):assert any(r['translation'] in text for text in texts),(language,r['key'])
    tables.append({'language':language,'row':actual_row,'session_header':dialog.table.horizontalHeaderItem(1).text(),'reference_header':dialog.table.horizontalHeaderItem(2).text(),'help_sentences':4,'preview_button_tooltip':button.toolTip()})
   finally:dialog.close();dialog.deleteLater()
   print(language+': 4 actual controls/4 revised tooltip widgets/18 keyed or UI lookups/version2 table PASS',flush=True)
 finally:
  retire_pyqtgraph_menus(screen);screen.close();screen.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
assert all(sha(ROOT/n)==value for n,value in before.items())
(OUT/'receipt.json').write_text(json.dumps({'status':'PASS','tested_commit':EXPECTED_HEAD,'actual_revised_tooltip_cases':len(old_tips),'old_tooltips':old_tips,'preview_tables':tables,'isolated_console_usage_calls':usage_calls,'actual_control_language_cases':len(rows),'compiled_target_cases':len(diagnostics),'actual_preview_languages':9,'controls':rows,'targets':diagnostics,'source_sha256':before,'isolation':'4G/offscreen/hiddenCUDA/privatepreferences; unrelated ConsolePanel.refresh_ai_usage replaced only in harness to avoid hosted allowance startup; inert preview planner produces fixed version2 plan, no backend/scientific inference.','limits':'Actual localized UI consumers; not a global network/native/human/scientific/fullCI/release acceptance claim.'},indent=2,ensure_ascii=False)+'\n')
