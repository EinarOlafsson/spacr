import json
import os
import sys
import time
import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

target_pid = int(sys.argv[1])
rows = []
for attempt in range(12):
    desktop = Atspi.get_desktop(0)
    apps = []
    for i in range(desktop.get_child_count()):
        try:
            child = desktop.get_child_at_index(i)
            apps.append((child.get_name(), child.get_process_id(), child))
        except Exception as exc:
            rows.append({'lookup_error': str(exc)})
    match = next((item for item in apps if item[1] == target_pid), None)
    if match is not None:
        break
    time.sleep(.5)
else:
    print(json.dumps({'found': False, 'target_pid': target_pid,
                      'visible_apps': [(name,pid) for name,pid,_ in apps],
                      'rows': rows}, indent=2))
    sys.exit(1)

for _ in range(20):
    if match[2].get_child_count() > 0:
        break
    time.sleep(.25)

seen = 0
def walk(node, depth=0):
    global seen
    if depth > 7 or seen >= 1000:
        return
    seen += 1
    try:
        name = node.get_name()
        role = node.get_role_name()
        count = node.get_child_count()
        row = {'depth': depth, 'role': role, 'name': name, 'children': count}
        if depth <= 2 or any(word in role.lower() for word in ('button','table','header','text','combo','check','slider')):
            rows.append(row)
        for index in range(min(count, 250)):
            walk(node.get_child_at_index(index), depth+1)
    except Exception as exc:
        rows.append({'depth': depth, 'error': str(exc)})
walk(match[2])
print(json.dumps({'found': True, 'target_pid': target_pid,
                  'application': match[0], 'visited': seen,
                  'rows': rows}, indent=2))
