from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import translation_delta as d
def split(text):return [x.strip()+'.' for x in text.split('.') if x.strip()]
def fixture():
 old={'lessons':[{'id':'a','scenes':[{'visual':'first','narration':'Keep. Old.'},{'visual':'last','narration':'Stay.'}]}]}
 local={'lessons':[{'id':'a','scenes':[{'narration':'Garder. Ancien.'},{'narration':'Rester.'}]}]}
 new=deepcopy(old);new['lessons'][0]['scenes'][0]['narration']='Keep. New.'
 return new,old,local

def test_only_changed_sentence_and_stable_inserted_visual():
 new,old,local=fixture();new['lessons'][0]['scenes'].insert(1,{'visual':'docker','narration':'Docker.'})
 result,strings,slots,plans=d.plan_delta(new,old,local,['a'],split,split)
 assert strings==['New.','Docker.']
 result=d.finish_delta(result,['Nouveau.','Docker traduit.'],slots,plans,'fr')
 assert result['lessons'][0]['scenes']==[{'narration':'Garder. Nouveau.'},{'narration':'Docker traduit.'},{'narration':'Rester.'}]
 assert local['lessons'][0]['scenes'][0]['narration']=='Garder. Ancien.'

def test_uncertain_sentence_alignment_fails_closed():
 new,old,local=fixture();local['lessons'][0]['scenes'][0]['narration']='Fusion.'
 with pytest.raises(ValueError,match='alignment'):d.plan_delta(new,old,local,['a'],split,split)

def test_duplicate_visual_rejected():
 new,old,local=fixture();new['lessons'][0]['scenes'].append(deepcopy(new['lessons'][0]['scenes'][0]))
 with pytest.raises(ValueError,match='Duplicate'):d.plan_delta(new,old,local,['a'],split,split)

def test_missing_translation_rejected():
 new,old,local=fixture();result,strings,slots,plans=d.plan_delta(new,old,local,['a'],split,split)
 with pytest.raises(ValueError,match='omitted'):d.finish_delta(result,[],slots,plans,'fr')

def test_unchanged_scene_object_preserved():
 new,old,local=fixture();local['lessons'][0]['scenes'][1]['custom']={'reviewed':1}
 result,_,_,_=d.plan_delta(new,old,local,['a'],split,split)
 assert result['lessons'][0]['scenes'][1]==local['lessons'][0]['scenes'][1]


def test_appended_metadata_sentence_does_not_retranslate_original():
 new,old,local=fixture()
 for lesson in [new['lessons'][0],old['lessons'][0]]:
  lesson.update(title='Title.',section='Section.',description='Description.',prerequisite='Old setup.')
 local['lessons'][0].update(title='Titre.',section='Section traduite.',description='Description traduite.',prerequisite='Ancienne installation.')
 new['lessons'][0]['prerequisite']='Old setup. Docker setup.'
 result,strings,slots,plans=d.plan_delta(new,old,local,['a'],split,split)
 assert strings==['Docker setup.','New.']
 result=d.finish_delta(result,['Installation Docker.','Nouveau.'],slots,plans,'fr')
 assert result['lessons'][0]['prerequisite']=='Ancienne installation. Installation Docker.'


@pytest.mark.parametrize('language,text',[
 ('hi','Python 3.12 तैयार करें। टर्मिनल खोलें। अलग नाम रखें।'),
 ('ja','Python 3.12を準備します。端末を開きます。名前を分けます。'),
 ('ko','Python 3.12를 준비합니다. 터미널을 여세요. 이름을 구분하세요.')])
def test_native_sentence_boundaries_preserve_decimal(language,text):
 parts=d.localized_sentences(text,language,lambda text:[text] if text else [])
 assert len(parts)==3
 assert '3.12' in parts[0]


def test_explicit_alignment_preserves_text_and_rejects_byte_drift():
 import hashlib
 old={'lessons':[{'id':'a','scenes':[{'visual':'x','narration':'First. Second; third.'}]}]}
 local={'lessons':[{'id':'a','scenes':[{'narration':'Uno. Dos. Tres.'}]}]}
 entry={'language':'es','lesson':'a','visual':'x',
  'english_sha256':hashlib.sha256(b'First. Second; third.').hexdigest(),
  'localized_sha256':hashlib.sha256(b'Uno. Dos. Tres.').hexdigest(),
  'segments':['Uno.','Dos. Tres.']}
 f=d.alignment_splitter(old,local,'es',[entry],split)
 assert f('Uno. Dos. Tres.')==['Uno.','Dos. Tres.']
 entry['segments']=['Uno.','Inventado.']
 with pytest.raises(ValueError,match='cannot change'):d.alignment_splitter(old,local,'es',[entry],split)
 entry['segments']=['Uno.','Dos. Tres.'];entry['localized_sha256']='0'*64
 with pytest.raises(ValueError,match='bytes changed'):d.alignment_splitter(old,local,'es',[entry],split)

def test_wholly_new_scene_needs_no_old_sentence_reuse():
 after,slots=d.sentence_slots('Old. Two.','Entirely different.','Fusion.',split,split)
 assert after==['Entirely different.'] and slots==[None]
