"""Observe actual translated backend refusals without model construction."""
import hashlib,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QCoreApplication,QSettings
from spacr.qt import preferences
from spacr.qt.i18n import current_language
from spacr.embeddings import EmbeddingError,_dino_classifier_from_config,_dino_classifier_apply
root=Path.cwd();out=Path(__file__).parent
source=root/'spacr/embeddings.py';before=hashlib.sha256(source.read_bytes()).hexdigest()
app=QCoreApplication.instance() or QCoreApplication([])
ini=out/'backend-consumer-final-preferences.ini';assert not ini.exists()
os.environ.pop('SPACR_LANGUAGE',None)
cases=[]
with patch.object(preferences,'_settings',lambda:QSettings(str(ini),QSettings.IniFormat)):
 for language in ['sv','de','es','pt','hi','ko','is','fr','zh_CN']:
  preferences.set_language(language);assert current_language()==language
  record=json.loads((root/'docs/i18n/reviewed/runtime'/language/'2026-10-15-f561-classifier-backend-home-b.json').read_text())['records'][0]
  for name,call in [('application',lambda:_dino_classifier_apply(np.zeros((0,1,1,1)),str(out/'must-not-open.pt')))]:
   try:call()
   except EmbeddingError as exc:
    assert str(exc)==record['translation'],(language,name,str(exc))
    cases.append({'language':language,'route':name,'message':str(exc)})
   else:raise AssertionError('Invalid input accepted')
assert hashlib.sha256(source.read_bytes()).hexdigest()==before
(out/'backend-consumers-receipt.json').write_text(json.dumps({'status':'PASS','cases':cases,'case_count':len(cases),'source_sha256':before,'limits':'Actual application early-refusal route on CPU. Configuration route not exercised: optional timm package unavailable, original environment failure retained. No model construction, pretrained download, GPU, training or scientific acceptance.'},ensure_ascii=False,indent=2)+'\n')
print('PASS:',len(cases),'actual backend refusal routes in nine languages',flush=True)
