"""Private receipt wrapper for bounded guide validation commands."""
import json,os,sys,subprocess
from pathlib import Path
receipt=Path(sys.argv[1]);command=sys.argv[2:]
line=next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
cgroup=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
limits={name:(cgroup/name).read_text().strip() for name in ('memory.max','memory.swap.max')}
limits['cpu_affinity']=sorted(os.sched_getaffinity(0))
limits['cpu.max']=(cgroup/'cpu.max').read_text().strip() if(cgroup/'cpu.max').exists()else None
limits['systemd_properties']=subprocess.check_output(['systemctl','--user','show',cgroup.name,'-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec'],text=True).splitlines()
assert limits['memory.max'] in ('1073741824','4294967296') and limits['memory.swap.max']=='0',limits
assert len(limits['cpu_affinity'])==1 and 'CPUQuotaPerSecUSec=1s' in limits['systemd_properties'],limits
receipt.write_text(json.dumps({'command':command,'cgroup':str(cgroup),'limits':limits},indent=2)+'\n')
os.execv(command[0],command)
