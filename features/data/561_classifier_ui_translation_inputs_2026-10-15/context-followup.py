from pathlib import Path
import ast,json,hashlib,subprocess,sys,shutil
root=Path.cwd();p=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-ui-20261015');sys.path.insert(0,str(root/'tools'));import build_documentation_i18n as a
sha=lambda s:hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
planpath=p/'translation-obligations.json';shutil.copyfile(planpath,p/'translation-obligations-before-context.json');plan=json.loads(planpath.read_text());plan['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();plan['source_sha256']=sha((root/'spacr/qt/screens/embeddings.py').read_bytes())
row=plan['api'][3];row['source']=row['source'].replace('this closed screen','this closed GUI view');row['source_sha256']=sha(row['source']);row['context']=a._api_translation_source(row['source']);row['context_sha256']=sha(row['context']);planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
replacements={
 'sv':('den stängda skärmen','den stängda GUI-vyn'),
 'de':('diesen geschlossenen Bildschirm','diese geschlossene GUI-Ansicht'),
 'es':('esta pantalla cerrada','esta vista cerrada de la GUI'),
 'fr':('cet écran fermé','cette vue GUI fermée'),
 'pt':('esta tela fechada','esta visualização fechada da GUI'),
 'is':('þennan lokaða skjá','þessa lokuðu GUI-sýn'),
 'zh_CN':('此已关闭的界面','此已关闭的 GUI 视图'),
 'hi':('इस बंद स्क्रीन','इस बंद GUI दृश्य'),
 'ko':('이 닫힌 화면','이 닫힌 GUI 뷰'),
}
deltas=[]
for name in ['west','middle','asia']:
 f=p/f'targets/{name}.json';data=json.loads(f.read_text())
 for lang,parts in data.items():
  old=parts['api']['3'];one,two=replacements[lang];assert one in old;new=old.replace(one,two);parts['api']['3']=new
  deltas.append({'language':lang,'label':row['label'],'old_target':old,'old_target_sha256':sha(old),'new_target':new,'new_target_sha256':sha(new),'source':row['source'],'source_sha256':row['source_sha256'],'context':row['context'],'context_sha256':row['context_sha256']})
 f.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(p/'context-target-followup.json').write_text(json.dumps({'source_commit':plan['source_commit'],'source_sha256':plan['source_sha256'],'exact_changed_target_count':9,'other387_targets_unchanged':True,'records':deltas},ensure_ascii=False,indent=2)+'\n')
print(plan['source_commit'],plan['source_sha256'])
