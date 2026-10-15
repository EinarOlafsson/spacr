import hashlib, importlib.util, json, runpy
from pathlib import Path
root=Path.cwd()
spec=importlib.util.spec_from_file_location("spacr_catalog_builder", root/"tools/build_i18n_catalogs.py")
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
canonical=b.canonical_sources()
rows=[]
english=runpy.run_path(str(root/"spacr/qt/i18n_catalogs/en.py"))
for lang in ["de","es","fr","hi","is","ko","pt","sv","zh_CN"]:
 p=root/"docs/i18n/reviewed/runtime"/lang/"2026-10-15-n197-annotation-page-navigation.json"
 payload=json.loads(p.read_text())
 assert payload["schema"]==1 and payload["language"]==lang
 for record in payload["records"]:
  assert set(record)=={"table","key","source","source_sha256","translation"}
  source=record["source"]; target=record["translation"]
  assert record["key"]==source and source in canonical["ui"]
  assert record["source_sha256"]==hashlib.sha256(source.encode()).hexdigest()
  assert b._contextualize(target,lang,source)==target
  reasons=b._translation_rejection_reasons(source,target,lang,force=True)
  assert not reasons,(lang,source,reasons)
  rows.append({"language":lang,"source":source,"translation":target,"candidate_rejection_reasons":sorted(reasons)})
 locale=runpy.run_path(str(root/"spacr/qt/i18n_catalogs"/(lang+".py")))
 for source in ["Move focus left","Move focus right"]:
  digest=hashlib.sha256(source.encode()).hexdigest()
  assert english["SOURCE_HASHES"][("UI",source)]==digest
  assert locale["SOURCE_HASHES"][("UI",source)]==digest
  assert locale["UI"][source].strip()
output={"schema":1,"scope":"Read-only canonical source, exact-source SHA-256, unchanged contextualization and normal runtime candidate gates for 18 new records; nine existing catalogs retain source-current crop-focus labels.","source_sha256":hashlib.sha256((root/"spacr/qt/shortcuts.py").read_bytes()).hexdigest(),"builder_sha256":hashlib.sha256((root/"tools/build_i18n_catalogs.py").read_bytes()).hexdigest(),"records":rows,"existing_focus_labels_verified":18,"normal_writer_run":False,"native_speaker_review":False}
(root/"features/data/197_shortcut_context_and_678_review_2026-10-15/runtime-review-validation.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
print("PASS:18 new reviewed records;18 existing focus translations; no catalog writes")
