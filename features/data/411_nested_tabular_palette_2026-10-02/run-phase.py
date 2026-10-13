"""Run one normal Sphinx phase and retain its actual cgroup limits."""
import json,os,subprocess,sys
from pathlib import Path
phase=sys.argv[1]
assert phase in ('dummy','html','html-refresh','html-final')
root=Path('/tmp/spacr-implementation-20261001/suggest-capture')
stage=Path('/tmp/spacr-implementation-20261001/f411-slice-tabular-palette/full')
stage.mkdir(exist_ok=True)
line=next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
cgroup=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
limits={key:(cgroup/key).read_text().strip() for key in ('memory.max','memory.swap.max')}
limits['cpu.max']=(cgroup/'cpu.max').read_text().strip() if (cgroup/'cpu.max').exists() else None
allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,{allowed[0]})
assert len(os.sched_getaffinity(0))==1
limits['cpu_affinity']=sorted(os.sched_getaffinity(0))
limits['systemd_requested_properties']=subprocess.check_output(['systemctl','--user','show',cgroup.name,'-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec'],text=True).splitlines()
assert limits['memory.max']=='4294967296',limits
assert limits['memory.swap.max']=='0',limits
if limits['cpu.max'] is not None:
 quota,period=limits['cpu.max'].split();assert int(quota)==int(period),limits
assert 'CPUQuotaPerSecUSec=1s' in limits['systemd_requested_properties'],limits
(stage/f'{phase}-limits.json').write_text(json.dumps({'cgroup':str(cgroup),'limits':limits},indent=2)+'\n')
os.chdir(root)
command=['/usr/bin/time','-v','/home/carruthers/anaconda3/envs/spacr/bin/python','-m','sphinx','-W',*(['-E'] if phase=='dummy' else ['-a'] if phase in ('html','html-final') else []),'-b','dummy' if phase=='dummy' else 'html','--keep-going','-j','1','-d',str(stage/'doctrees'),'docs/source',str(stage/('dummy' if phase=='dummy' else 'html'))]
(stage/f'{phase}-command.json').write_text(json.dumps(command,indent=2)+'\n')
os.execv(command[0],command)
