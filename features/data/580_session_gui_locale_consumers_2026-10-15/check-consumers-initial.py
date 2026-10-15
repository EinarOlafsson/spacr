"""Read final calibration captions through actual controls and the preview table."""
import hashlib,html,json,os,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();OUT.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QCoreApplication,QEvent,QSettings
from PySide6.QtWidgets import QApplication,QLabel,QLineEdit
from spacr.qt import preferences
from spacr.qt.i18n import current_language,retranslate_widget_tree,tr
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import _setting_label_for_field,api_docs_url
from spacr.qt.screens.settings_flow_index import SETTINGS_WITH_A_FLOW_SECTION
from spacr.qt.widgets import calibration_preview as preview
from spacr.qt.widgets.console_panel import ConsolePanel
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
LANGS=['sv','de','es','zh_CN','pt','hi','ko','is','fr'];KEYS=['intensity_calibration_sessions','intensity_calibration_session_column','intensity_calibration_beads','intensity_calibration_reference_session'];NAMES=['IntensityCalibrationSessions','IntensityCalibrationSessionColumn','IntensityCalibrationBeads','IntensityCalibrationReferenceSession'];DEFAULTS=['','session','',''];REVIEW='2026-10-15-f580-session-controls-home-b.json'
FILES=['spacr/settings.py','spacr/intensity_rescale.py','spacr/measure.py','spacr/qt/screens/settings_model.py','spacr/qt/widgets/calibration_preview.py','spacr/qt/screens/settings_flow_index.py']+[f'spacr/qt/i18n_catalogs/{l}.py' for l in ['en',*LANGS]]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={n:sha(ROOT/n) for n in FILES};app=QApplication.instance() or QApplication([]);ini=OUT/'preferences.ini';assert not ini.exists();rows=[];diagnostics=[];os.environ.pop('SPACR_LANGUAGE',None)
def report(snapshot):
 return [{'source':'/inert/merged','calibration':{'version':2,'reference_session':'day2','sessions':{'day1':{'gain':{'0':.5},'reference_statistic':{'0':250},'n_reference_fields':3}}}}]
with patch.object(preferences,'_settings',lambda:QSettings(str(ini),QSettings.IniFormat)),patch.object(ConsolePanel,'refresh_ai_usage',lambda self:None),patch.object(preview,'_plan_gains',report):
 preferences.set_language('en');preferences._set_show_alpha_features(True);screen=AppScreen('measure')
 try:
  for key in KEYS:screen._open_the_heading_of(key)
  screen._refresh_alpha_visibility();model=screen._settings_model
  for language in LANGS:
   records=json.loads((ROOT/f'docs/i18n/reviewed/runtime/{language}/{REVIEW}').read_text())['records'];by={(r['table'],r['key']):r for r in records};assert len(records)==18
   preferences.set_language(language);assert current_language()==language;retranslate_widget_tree(screen,language)
   for key,name,default in zip(KEYS,NAMES,DEFAULTS):
    field=model._widgets[key];label=_setting_label_for_field(screen,field);assert isinstance(field,QLineEdit) and field.objectName()==name;assert isinstance(label,QLabel);assert screen.setting_row_is_visible(key);assert field.text()==default
    assert label.text()==by['setting_labels',key]['translation'];assert html.escape(by['setting_tooltips',key]['translation']) in label.toolTip();assert key in SETTINGS_WITH_A_FLOW_SECTION;url=api_docs_url('measure',key,language);assert f'settings_flow.html?lang={language}#setting-flow-{key}' in url;assert html.escape(url,quote=True) in label.toolTip();rows.append({'language':language,'key':key,'default':field.text(),'label':label.text(),'url':url})
   for r in records:
    assert tr(r['source'],language)==r['translation'],(language,r['key']);diagnostics.append({'language':language,'table':r['table'],'key':r['key'],'target_sha256':hashlib.sha256(r['translation'].encode()).hexdigest()})
   dialog=preview._CalibrationPreview(lambda:preview._preview_settings(model),threaded=False)
   try:
    dialog._start();assert dialog.table.horizontalHeaderItem(1).text()==tr('Session',language);assert dialog.table.horizontalHeaderItem(2).text()==by['ui','Reference session']['translation'];assert dialog.table.item(0,1).text()=='day1';assert dialog.table.item(0,2).text()=='day2';assert dialog.table.item(0,4).text()=='0.5'
   finally:dialog.close();dialog.deleteLater()
   print(language+': 4actual controls/18compiled targets/real version2 table PASS',flush=True)
 finally:
  retire_pyqtgraph_menus(screen);screen.close();screen.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
assert all(sha(ROOT/n)==value for n,value in before.items())
(OUT/'receipt.json').write_text(json.dumps({'status':'PASS','actual_control_language_cases':len(rows),'compiled_target_cases':len(diagnostics),'actual_preview_languages':9,'controls':rows,'targets':diagnostics,'source_sha256':before,'isolation':'4G/offscreen/hiddenCUDA/privatepreferences; unrelated ConsolePanel.refresh_ai_usage replaced only in harness to avoid hosted allowance startup; inert preview planner produces fixed version2 plan, no backend/scientific inference.','limits':'Actual localized UI consumers, not native/human/scientific/fullCI/release acceptance.'},indent=2,ensure_ascii=False)+'\n')
