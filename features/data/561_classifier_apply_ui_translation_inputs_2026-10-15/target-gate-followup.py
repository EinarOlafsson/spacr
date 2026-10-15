from pathlib import Path
import json,hashlib,shutil
root=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-apply-ui-20261015');old=root/'targets-before-gates';old.mkdir(exist_ok=True)
shutil.copyfile(root/'input-check.json',root/'input-check-before.json')
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
rep={'sv':('GUI-vyn','vyn i det grafiska gränssnittet'),'de':('GUI-Ansicht','Ansicht der grafischen Oberfläche'),'es':('la vista de la GUI','la vista de la interfaz gráfica'),'is':('GUI-sýninni','sýn grafíska viðmótsins'),'zh_CN':('GUI 视图','图形界面视图'),'hi':('GUI दृश्य','ग्राफ़िकल इंटरफ़ेस दृश्य'),'ko':('GUI 뷰','그래픽 인터페이스 화면')}
changes=[]
for name in ['west','middle','asia']:
 p=root/f'targets/{name}.json';shutil.copyfile(p,old/p.name);d=json.loads(p.read_text())
 for lang,parts in d.items():
  if lang in rep:
   a,b=rep[lang];before=parts['api'][1];assert a in before;after=before.replace(a,b);parts['api'][1]=after
   changes.append({'language':lang,'domain':'api','index':1,'before':before,'after':after,'before_sha256':sha(before),'after_sha256':sha(after),'reason':'Equivalent ordinary GUI-view wording without introducing a protected acronym absent from the source.'})
  if lang=='pt':
   for index in ['0','2']:
    before=parts['runtime'][index];assert 'a {n} recortes' in before;after=before.replace('a {n} recortes','em {n} recortes');parts['runtime'][index]=after
    changes.append({'language':lang,'domain':'runtime','index':int(index),'before':before,'after':after,'before_sha256':sha(before),'after_sha256':sha(after),'reason':'Equivalent Portuguese application preposition avoids false crop-gender match after placeholder removal, preserving both meaning and normal gate.'})
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
assert len(changes)==9
(root/'target-gate-followup.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
