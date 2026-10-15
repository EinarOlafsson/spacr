from pathlib import Path
import ast, hashlib, json, subprocess, sys
from email.parser import Parser
from docutils.core import publish_doctree
from docutils import nodes
root=Path.cwd(); scratch=Path(__file__).resolve().parent
inputs=json.loads((scratch/'inputs.json').read_text()); baseline=inputs['baseline']
sha=lambda b: hashlib.sha256(b).hexdigest()
old=lambda p: subprocess.check_output(['git','show',f'{baseline}:{p}'])
rows=[]; allowed={'pyproject.toml'}
for record in inputs['records']:
    source=record['source']; output=record['output']; line=record['line']
    for path,insert in ((source,'    '+line+'\n\n'),(output,line+'\n\n')):
        now=Path(path).read_bytes(); before=old(path)
        assert now.count(insert.encode())==1,(path,'line count')
        assert now.replace(insert.encode(),b'',1)==before,(path,'unrelated bytes')
        allowed.add(path)
    text=Path(output).read_text(); assert text.splitlines()[2]==line
    tree=publish_doctree(line)
    links=list(tree.findall(nodes.reference)); assert len(links)==1
    assert links[0]['refuri']=='https://einarolafsson.github.io/projects/spacr/'
    assert '|Website|' in text.splitlines()[0]
    rows.append({**record,'source_sha256':sha(Path(source).read_bytes()),'output_sha256':sha(Path(output).read_bytes()),'prominent_plain_link_line':3,'rendered_link_uri':links[0]['refuri'],'all_other_source_and_output_bytes_exact':True})
artifacts=json.loads((scratch/'artifact-before.json').read_text())
for path,digest in artifacts.items():
    assert sha(Path(path).read_bytes())==digest,path
changed=set(subprocess.check_output(['git','diff','--name-only'],text=True).splitlines())
assert changed==allowed,sorted(changed^allowed)
setup_ast=ast.parse(Path('setup.py').read_text())
call=next(n for n in ast.walk(setup_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='setup')
assert not {'url','project_urls'}&{kw.arg for kw in call.keywords}
meta={}
for phase in ('before','after'):
    path=scratch/f'metadata-{phase}'/'spacr.egg-info/PKG-INFO'
    parsed=Parser().parsestr(path.read_text()); urls=dict(x.split(', ',1) for x in parsed.get_all('Project-URL'))
    meta[phase]={'path':str(path),'sha256':sha(path.read_bytes()),'project_urls':urls}
assert meta['before']['project_urls']['Homepage']=='https://github.com/EinarOlafsson/spacr'
assert meta['after']['project_urls']['Homepage']=='https://einarolafsson.github.io/projects/spacr/'
assert {k:v for k,v in meta['before']['project_urls'].items() if k!='Homepage'}=={k:v for k,v in meta['after']['project_urls'].items() if k!='Homepage'}
assert meta['after']['project_urls']['Source']=='https://github.com/EinarOlafsson/spacr'
result={'status':'PASS','baseline':baseline,'rows':rows,'metadata':meta,'setup_py_no_url_overrides':True,'unchanged_artifacts':len(artifacts),'artifact_manifest_sha256':sha((scratch/'artifact-before.json').read_bytes()),'only_expected21_paths_changed':sorted(changed)}
(scratch/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','README_languages':len(rows),'unchanged_artifacts':len(artifacts),'homepage':meta['after']['project_urls']['Homepage']}))
