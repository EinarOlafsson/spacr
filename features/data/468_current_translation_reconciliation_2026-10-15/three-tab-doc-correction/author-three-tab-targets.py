from pathlib import Path
import ast,hashlib,json,sys,tempfile,subprocess
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n468-current-contract-audit-20261015');sys.path.insert(0,str(root/'tools'));import build_documentation_i18n as api
sha=lambda b:hashlib.sha256(b).hexdigest();source=root/'spacr/qt/widgets/plaque_preview.py';text=source.read_text();tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PlaqueSettingsDialog');key='spacr.qt.widgets.plaque_preview.PlaqueSettingsDialog';doc=api._autoapi_class_doc(cls);blocks,_=api.translatable_blocks(doc);assert len(blocks)==6
original=subprocess.check_output(['git','show','HEAD:spacr/qt/widgets/plaque_preview.py'],text=True)
def docs(t):
 out={}
 def walk(body,prefix=''):
  for n in body:
   if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
    label=prefix+n.name;out[label]=ast.get_docstring(n,clean=False);walk(n.body,label+'.')
 walk(ast.parse(t).body);return out
old_docs=docs(original);new_docs=docs(text);changed=[k for k in old_docs if old_docs[k]!=new_docs[k]];assert changed==['PlaqueSettingsDialog','PlaqueSettingsDialog.__init__']
def nodocs(t):
 tree=ast.parse(t)
 for n in ast.walk(tree):
  if isinstance(n,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body.pop(0)
 return ast.dump(tree)
assert nodocs(original)==nodocs(text)
targets={
'sv':['Inställningarna för förhandsvisningen av Plaque Assay: flikarna Figur, Textdetektering och Plackdetektion.','I läget Plack är flikarna Figur och Textdetektering dolda: ingen av dem gäller en mapp med beskurna plackbilder.','Ordna panelens kontroller på tre flikar.'],
'de':['Die Einstellungen der Vorschau von Plaque Assay: die Registerkarten Abbildung, Texterkennung und Plaque-Erkennung.','Im Modus Plaque sind die Registerkarten Abbildung und Texterkennung ausgeblendet: Beide gelten nicht für einen Ordner mit zugeschnittenen Plaque-Bildern.','Die Steuerelemente des Panels auf drei Registerkarten anordnen.'],
'es':['Los ajustes de la vista previa de Plaque Assay: pestañas Figura, Detección de texto y Detección de placas.','En el modo Placa se ocultan las pestañas Figura y Detección de texto: ninguna se aplica a una carpeta de imágenes recortadas de placas.','Distribuir los controles del panel en tres pestañas.'],
'zh_CN':['Plaque Assay 预览的设置：图、文本检测和蚀斑检测选项卡。','在空斑模式下，图和文本检测选项卡隐藏：两者均不适用于包含裁剪空斑图像的文件夹。','将面板的控件排列在三个选项卡中。'],
'fr':['Les paramètres de l’aperçu de Plaque Assay : onglets Figure, Détection de texte et Détection des plages de lyse.','En mode Plaque, les onglets Figure et Détection de texte sont masqués : aucun des deux ne s’applique à un dossier d’images de plages de lyse recadrées.','Disposer les commandes du panneau dans trois onglets.'],
'pt':['As configurações da prévia do Plaque Assay: abas Figura, Detecção de texto e Detecção de placas de lise.','No modo Placa, as abas Figura e Detecção de texto ficam ocultas: nenhuma delas se aplica a uma pasta de imagens recortadas de placas de lise.','Organizar os controles do painel em três abas.'],
'hi':['Plaque Assay के पूर्वावलोकन की सेटिंग: चित्र, पाठ पहचान और प्लाक पहचान टैब।','प्लाक मोड में चित्र और पाठ पहचान टैब छिपे रहते हैं: दोनों में से कोई भी काटी गई प्लाक छवियों वाले फ़ोल्डर पर लागू नहीं होता।','पैनल के नियंत्रणों को तीन टैब में व्यवस्थित करें।'],
'ko':['Plaque Assay 미리보기의 설정: 그림, 텍스트 검출 및 플라크 검출 탭입니다.','플라크 모드에서는 그림 및 텍스트 검출 탭이 숨겨집니다. 두 탭 모두 잘라낸 플라크 이미지가 담긴 폴더에는 적용되지 않습니다.','패널의 컨트롤을 세 개의 탭에 배치합니다.'],
'is':['Stillingar fyrir forskoðun í Plaque Assay: fliparnir Mynd, Textagreining og Skellugreining.','Í skelluham eru fliparnir Mynd og Textagreining faldir: hvorugur á við um möppu með skornum skellumyndum.','Raða stýringum spjaldsins á þrjá flipa.']}
out=scratch/'three-tab-api-drafts';out.mkdir(exist_ok=True);retirements=[];manifest={'source_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_file':str(source.relative_to(root)),'source_sha256':sha(source.read_bytes()),'source_doc_sha256':api._source_hash(doc),'actual_normal_API_doc':doc,'normal_changed_blocks':[0,2,4],'changed_doc_owners_only':changed,'non_doc_AST_unchanged':True,'drafts':{},'retirement_proposals':retirements,'status':'Private27newtechnicaltargets; rootpeerpending; no installednewrecords/writer/retirements for3tabchange.'}
for lang,values in targets.items():
 records=[{'label':key+'#'+str(i),'source':blocks[i],'context':api._api_translation_source(blocks[i]),'source_sha256':api._source_hash(blocks[i]),'translation':v,'review_basis':'Truthful actual three-tab settings dialog: Figure, Text detection, Plaque detection. In Plaque mode BOTH Figure/Text detection hidden; controls arranged across three tabs. Uses existing localized UI label vocabulary. Sourcebound AI technical review only; independent root peer pending, no native/human signoff.'} for i,v in zip([0,2,4],values)]
 payload={'schema':1,'language':lang,'review_kind':'AI technical translation/source review; independent root peer pending; no native/human signoff','note':'Only current rendered class/constructor blocks0/2/4 correct stale two-tab prose. Behavior/publicsignatures/allotherdocblocks unchanged.','records':records};p=out/f'{lang}.json';p.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 with tempfile.TemporaryDirectory(dir=scratch,prefix=lang+'-three-tab-normalreview-') as d:
  api.REVIEWED_API_DIR=Path(d);q=Path(d)/lang;q.mkdir();(q/'input.json').write_bytes(p.read_bytes());accepted=api.reviewed_api_block_translations({key:doc},lang);assert all(accepted[r['source']]==r['translation'] for r in records)
 retired=[]
 for path in sorted((root/'docs/i18n/reviewed/api'/lang).glob('*.json')):
  data=json.loads(path.read_text())
  for index,row in enumerate(data['records']):
   if not row.get('retired') and row.get('label') in {r['label'] for r in records}:
    retired.append({'file':str(path.relative_to(root)),'file_sha256':sha(path.read_bytes()),'row_index':index,'label':row['label'],'old_source_sha256':row['source_sha256'],'old_record_sha256':sha(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()),'old_record':row})
 assert len(retired)==3,(lang,len(retired));retirements+=retired
 manifest['drafts'][lang]={'path':str(p),'sha256':sha(p.read_bytes()),'targets':[{'label':r['label'],'source_sha256':r['source_sha256'],'target_sha256':sha(r['translation'].encode())} for r in records],'normal_source_context_markup_review_gates':'PASS'}
(scratch/'three-tab-api-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'source_sha256':manifest['source_sha256'],'normal_targets':27,'retire_exact_old_rows':len(retirements),'behaviorASTunchanged':True}),flush=True)
