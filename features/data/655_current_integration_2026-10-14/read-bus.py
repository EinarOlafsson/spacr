"""External AT-SPI client using bounded D-Bus calls without libatspi caches."""
import json,signal,sys,time
import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Gio, GLib, Atspi
signal.alarm(15)
pid=int(sys.argv[1]); expected=sys.argv[2] if len(sys.argv)>2 else 'External reader control button'
session=Gio.bus_get_sync(Gio.BusType.SESSION,None)
def call(connection,bus,path,interface,method,args=None):
    return connection.call_sync(bus,path,interface,method,args,None,Gio.DBusCallFlags.NONE,1000,None).unpack()
address=call(session,'org.a11y.Bus','/org/a11y/bus','org.a11y.Bus','GetAddress')[0]
connection=Gio.DBusConnection.new_for_address_sync(address,Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT|Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,None,None)
root='/org/a11y/atspi/accessible/root'; interface='org.a11y.atspi.Accessible'
print('CONNECTED',file=sys.stderr,flush=True)
for attempt in range(30):
    roots=call(connection,'org.a11y.atspi.Registry',root,interface,'GetChildren')[0]
    applications=[]
    for bus,path in roots:
        application_pid=call(connection,'org.freedesktop.DBus','/org/freedesktop/DBus','org.freedesktop.DBus','GetConnectionUnixProcessID',GLib.Variant('(s)',(bus,)))[0]
        applications.append({'pid':application_pid,'bus_name':bus,'path':path})
    matches=[a for a in applications if a['pid']==pid]
    if matches:break
    time.sleep(.1)
print('ROOTS '+json.dumps(applications),file=sys.stderr,flush=True)
rows=[];found=[];seen=set()
def walk(bus,path,depth=0):
    if (bus,path) in seen or depth>16 or len(rows)>=3000:return
    seen.add((bus,path))
    props={name:call(connection,bus,path,'org.freedesktop.DBus.Properties','Get',GLib.Variant('(ss)',(interface,name)))[0] for name in ['Name','Description','ChildCount']}
    role=call(connection,bus,path,interface,'GetRoleName')[0]
    children=call(connection,bus,path,interface,'GetChildren')[0]
    row={'depth':depth,'name':props.get('Name'),'description':props.get('Description'),'role':role,'declared_child_count':props.get('ChildCount'),'children':len(children),'bus_name':bus,'path':path}
    rows.append(row)
    if row['name']==expected and row['role'] in ('push button','check box','toggle button'):found.append((bus,path))
    for child_bus,child_path in children:walk(child_bus,child_path,depth+1)
for app in matches:walk(app['bus_name'],app['path'])
result={'target_pid':pid,'applications':applications,'matching_roots':len(matches),'rows':rows,'expected_found':len(found)}
if found:
    bus,path=found[0]
    result['actions']=call(connection,bus,path,'org.a11y.atspi.Action','GetActions')[0]
    if result['actions']:
        result['state_before']=call(connection,bus,path,interface,'GetState')[0]
        result['action_result']=call(connection,bus,path,'org.a11y.atspi.Action','DoAction',GLib.Variant('(i)',(0,)))[0]
        time.sleep(.2)
        result['state_after']=call(connection,bus,path,interface,'GetState')[0]
        for which in ('state_before','state_after'):
            bits=sum(value << (32*index) for index,value in enumerate(result[which]))
            result[which+'_names']=[state.value_nick for value,state in Atspi.StateType.__enum_values__.items() if bits & (1<<value)]
print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if found else 1)
