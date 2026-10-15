"""Verify generated classifier translations through actual GUI consumers."""
import hashlib,json,os,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QSettings,QObject
from PySide6.QtWidgets import QApplication,QPushButton,QLabel,QProgressBar
from spacr.qt import preferences
from spacr.qt.i18n import tr,current_language
from spacr.qt.widgets.console_panel import ConsolePanel
from spacr.qt.screens.embeddings import EmbeddingsScreen

root=Path.cwd();out=Path(__file__).parent
source=root/'spacr/qt/screens/embeddings.py'
expected='719e985de25d06a72560ae4758eb8c033a8b20aa293bc3bfae58ad3ba0c6baea'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
app=QApplication.instance() or QApplication([])
ini=out/'ui-preferences.ini'
assert not ini.exists()
os.environ.pop('SPACR_LANGUAGE',None)
rows=[];lookups=[]
controls={
'EmbeddingsDinoClassifierButton':'Train classifier…',
'EmbeddingsDinoClassifierStop':'Stop classifier',
'EmbeddingsDinoClassifierApply':'Apply saved classifier…',
'EmbeddingsDinoClassifierApplyCancel':'Cancel classifier result',
'EmbeddingsDinoClassifierResults':'Classifier results…'}
with patch.object(preferences,'_settings',lambda:QSettings(str(ini),QSettings.IniFormat)),patch.object(ConsolePanel,'refresh_ai_usage',lambda self:None):
 preferences._set_show_alpha_features(True)
 for language in ['sv','de','es','pt','hi','ko','is','fr','zh_CN']:
  preferences.set_language(language)
  assert current_language()==language
  widget=EmbeddingsScreen(threaded=False)
  dialog=widget._dino_classifier_dialog()
  try:
   for name,english in controls.items():
    button=widget.findChild(QPushButton,name)
    target=tr(english,language=language)
    assert button is not None and button.text()==target,(language,name)
    assert widget._dino_controls.isAncestorOf(button)
    rows.append({'language':language,'control':name,'text':target})
   assert dialog.windowTitle()==tr('Train an own-DINO classifier',language=language)
   bar=widget.findChild(QProgressBar,'EmbeddingsDinoClassifierProgress')
   assert not bar.isTextVisible()
   assert bar.accessibleName()==tr('Classifier training progress',language=language)
   caption=widget.findChild(QLabel,'EmbeddingsDinoClassifierProgressCaption')
   assert caption.text()==tr('Classifier training progress',language=language)
   for stem in ['f561-classifier-ui','f561-classifier-apply-ui']:
    path=root/'docs/i18n/reviewed/runtime'/language/('2026-10-15-'+stem+'-runtime-home-b.json')
    for record in json.loads(path.read_text())['records']:
     assert tr(record['source'],language=language)==record['translation'],(language,record['source'])
     lookups.append({'language':language,'source':record['source']})
  finally:
   dialog.close()
   widget.close()
   dialog.deleteLater()
   widget.deleteLater()
   app.sendPostedEvents(None,0)
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
receipt={'status':'PASS','source_sha256':expected,'actual_button_cases':len(rows),'dialog_languages':9,'progress_controls':9,'runtime_lookups':len(lookups),'controls':rows,'isolation':'Private preferences, offscreen, CPU only. Only unrelated ConsolePanel allowance refresh is isolated. No trainer/model loading or scientific acceptance replay.'}
(out/'ui-consumers-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('PASS:',len(rows),'actual buttons;9 training dialogs/progress controls;',len(lookups),'runtime translations',flush=True)
