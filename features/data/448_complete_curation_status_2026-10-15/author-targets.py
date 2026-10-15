from pathlib import Path
import ast
import hashlib
import json

root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n448-complete-status-locales-20261015')
tree=ast.parse((root/'tests/qt/test_n448_complete_curation_status.py').read_text())
values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
keys=[('moved',True),('moved',False),('empty',True),('empty',False),('last',True),('last',False),('other',True),('other',False)]
targets={
'fr':[values['FRENCH'][key] for key in keys],
'sv':['{judged} behållet  —  nu på {now}','{judged} bortvalt — ingenting raderades  —  nu på {now}','{judged} behållet  —  inga ogranskade bildfält finns kvar i den här kön','{judged} bortvalt — ingenting raderades  —  inga ogranskade bildfält finns kvar i den här kön','{judged} behållet  —  det var det sista bildfältet i den här mappen','{judged} bortvalt — ingenting raderades  —  det var det sista bildfältet i den här mappen','{judged} behållet','{judged} bortvalt — ingenting raderades'],
'de':['{judged} beibehalten  —  jetzt bei {now}','{judged} verworfen — nichts wurde gelöscht  —  jetzt bei {now}','{judged} beibehalten  —  in dieser Warteschlange sind keine ungeprüften Bildfelder mehr vorhanden','{judged} verworfen — nichts wurde gelöscht  —  in dieser Warteschlange sind keine ungeprüften Bildfelder mehr vorhanden','{judged} beibehalten  —  dies war das letzte Bildfeld in diesem Ordner','{judged} verworfen — nichts wurde gelöscht  —  dies war das letzte Bildfeld in diesem Ordner','{judged} beibehalten','{judged} verworfen — nichts wurde gelöscht'],
'es':['{judged} conservado  —  ahora en {now}','{judged} descartado — no se eliminó nada  —  ahora en {now}','{judged} conservado  —  no quedan campos sin revisar en esta cola','{judged} descartado — no se eliminó nada  —  no quedan campos sin revisar en esta cola','{judged} conservado  —  ese era el último campo de esta carpeta','{judged} descartado — no se eliminó nada  —  ese era el último campo de esta carpeta','{judged} conservado','{judged} descartado — no se eliminó nada'],
'pt':['{judged} mantido  —  agora em {now}','{judged} descartado — nada foi eliminado  —  agora em {now}','{judged} mantido  —  não restam campos por rever nesta fila','{judged} descartado — nada foi eliminado  —  não restam campos por rever nesta fila','{judged} mantido  —  este era o último campo desta pasta','{judged} descartado — nada foi eliminado  —  este era o último campo desta pasta','{judged} mantido','{judged} descartado — nada foi eliminado'],
'is':['{judged} haldið  —  nú á {now}','{judged} útilokað — engu var eytt  —  nú á {now}','{judged} haldið  —  engin óyfirfarin myndsvið eru eftir í þessari biðröð','{judged} útilokað — engu var eytt  —  engin óyfirfarin myndsvið eru eftir í þessari biðröð','{judged} haldið  —  þetta var síðasta myndsviðið í þessari möppu','{judged} útilokað — engu var eytt  —  þetta var síðasta myndsviðið í þessari möppu','{judged} haldið','{judged} útilokað — engu var eytt'],
'hi':['{judged} रखा गया  —  अब {now} पर','{judged} छोड़ दिया गया — कुछ भी नहीं हटाया गया  —  अब {now} पर','{judged} रखा गया  —  इस कतार में कोई बिना समीक्षा वाला फ़ील्ड नहीं बचा है','{judged} छोड़ दिया गया — कुछ भी नहीं हटाया गया  —  इस कतार में कोई बिना समीक्षा वाला फ़ील्ड नहीं बचा है','{judged} रखा गया  —  यह इस फ़ोल्डर का अंतिम फ़ील्ड था','{judged} छोड़ दिया गया — कुछ भी नहीं हटाया गया  —  यह इस फ़ोल्डर का अंतिम फ़ील्ड था','{judged} रखा गया','{judged} छोड़ दिया गया — कुछ भी नहीं हटाया गया'],
'ko':['{judged} 유지됨  —  현재 {now}','{judged} 제외됨 — 아무것도 삭제되지 않음  —  현재 {now}','{judged} 유지됨  —  이 대기열에 검토하지 않은 필드가 남아 있지 않습니다','{judged} 제외됨 — 아무것도 삭제되지 않음  —  이 대기열에 검토하지 않은 필드가 남아 있지 않습니다','{judged} 유지됨  —  이 폴더의 마지막 필드였습니다','{judged} 제외됨 — 아무것도 삭제되지 않음  —  이 폴더의 마지막 필드였습니다','{judged} 유지됨','{judged} 제외됨 — 아무것도 삭제되지 않음'],
'zh_CN':['{judged} 已保留  —  当前为 {now}','{judged} 已弃用 — 未删除任何内容  —  当前为 {now}','{judged} 已保留  —  此队列中没有尚未评判的视野','{judged} 已弃用 — 未删除任何内容  —  此队列中没有尚未评判的视野','{judged} 已保留  —  这是此文件夹中的最后一个视野','{judged} 已弃用 — 未删除任何内容  —  这是此文件夹中的最后一个视野','{judged} 已保留','{judged} 已弃用 — 未删除任何内容'],
}
out=scratch/'runtime-drafts';out.mkdir(exist_ok=True);manifest={}
for lang,translations in targets.items():
 rows=[]
 for key,target in zip(keys,translations):
  source=values['TEMPLATES'][key]
  rows.append({'table':'ui','key':source,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'translation':target})
 p=out/f'{lang}.json';p.write_text(json.dumps({'schema':1,'language':lang,'review_kind':'AI technical translation and self-review; independent peer pending, no human/native signoff','review':'Complete keep/discard curation status sentences preserve judged/next field identity, nothing-deleted meaning, queue/last/other outcomes and placeholders; no change to stored CSV or navigation.','records':rows},ensure_ascii=False,indent=2)+'\n');manifest[lang]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':len(rows)}
(scratch/'runtime-drafts-manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256((root/'spacr/qt/screens/make_masks.py').read_bytes()).hexdigest(),'targets':72,'files':manifest},indent=2)+'\n')
