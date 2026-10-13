import csv,gzip,hashlib,json
from pathlib import Path
root=Path(__file__).parent
rows=[];proof={}
for p in sorted((root/'input').glob('*.csv.gz')):
    proof[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    with gzip.open(p,'rt') as f:
        for r in csv.DictReader(f):
            rows.append({'plateID':r['Metadata_Plate'],'well':r['Metadata_Well'],'well_type':r['Metadata_control_type'] or 'sample','cell_count_proxy':r['Cells_Number_Object_Number'],'nuclear_area':r['Nuclei_AreaShape_Area']})
rows.sort(key=lambda r:(r['plateID'],r['well']))
assert len(rows)==1536
with (root/'screen.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'input-hashes.json').write_text(json.dumps(proof,indent=2)+'\n')
