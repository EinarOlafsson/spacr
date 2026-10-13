import functools,hashlib,json,sys
from pathlib import Path
r=Path.cwd();p=Path(__file__).parent;sys.path[:0]=[str(r),str(r/'tools')]
import build_i18n_catalogs as b
b._simplify_chinese_prose=functools.lru_cache(maxsize=65536)(b._simplify_chinese_prose)
sources=json.loads((p/'sources.json').read_text());first=sources[0]
corrections=json.loads((p/'root-shared-corrections.json').read_text())['corrections']
corrections.update({
 'ko':{first:'Measure 모듈을 실행하거나 파일을 변경하지 않고 기준 웰의 보정 배율을 계산합니다.'},
 'is':{first:'Reiknaðu leiðréttingarstuðla út frá viðmiðunarbrunnum án þess að keyra Measure eða breyta skrám.',
 'No eligible merged arrays were found.':'Engin sameinuð fylki sem uppfylla skilyrðin fundust.',
 'Turn off Test mode to preview gains for all source fields.':'Slökktu á prófunarham til að forskoða stuðla fyrir öll sjónsvið í upprunamöppunni.'},
 'de':{first:'Kalibrierungsfaktoren der Referenz-Wells berechnen, ohne Measure auszuführen oder Dateien zu ändern.',
 'Gain':'Faktor',
 'The first plate by name is the reference.':'Die nach Namen sortierte erste Platte dient als Referenz.'},
 'es':{first:'Calcular los factores de ganancia de los pocillos de referencia sin ejecutar Measure ni modificar archivos.'},
 'pt':{first:'Calcular os fatores de ganho dos poços de referência sem executar Measure nem alterar arquivos.',
 'Preview gains':'Pré-visualizar fatores de ganho'},
 'fr':{first:'Calculer les facteurs de gain des puits de référence sans exécuter Measure ni modifier les fichiers.'},
})
checks=[];issues=[]
for lang in b.MODEL_SPECS:
 draft=p/f'{lang}-draft.json';targets=json.loads(draft.read_text());assert set(targets)==set(sources)
 if lang=='de':
  for s,t in targets.items():
   revised=t.replace('Verstärkungsfaktoren','Kalibrierungsfaktoren').replace('Referenzwells','Referenz-Wells')
   if revised!=t:corrections['de'].setdefault(s,revised)
 targets.update(corrections.get(lang,{}));final={}
 for source,target in targets.items():
  normalized=b._contextualize(target,lang,source)
  failures=sorted(b._translation_rejection_reasons(source,normalized,lang))
  item={'language':lang,'source':source,'reviewed_target':target,'effective_target':normalized,'normalization_applied':normalized!=target,'rejections':failures}
  checks.append(item)
  if failures:issues.append(item)
  final[source]=normalized
 (p/f'{lang}-final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
(p/'applied-peer-corrections.json').write_text(json.dumps({'reviewer':'Codex AI root (sv reviewed independently by shared_ui)','method':'Independent AI technical reviews in task messages; unchanged immutable drafts retained; no native-speaker signoff.','corrections':corrections,'draft_sha256':{lang:hashlib.sha256((p/f'{lang}-draft.json').read_bytes()).hexdigest() for lang in b.MODEL_SPECS}},ensure_ascii=False,indent=2)+'\n')
(p/'candidate-gates.json').write_text(json.dumps({'targets':len(checks),'issues':issues,'checks':checks},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checked':len(checks),'issues':issues},ensure_ascii=False),flush=True)
assert not issues
