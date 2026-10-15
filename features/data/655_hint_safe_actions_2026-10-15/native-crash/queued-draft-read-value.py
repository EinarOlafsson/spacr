"""Read native AT-SPI height values and resize through its real external interfaces."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi, Gio, GLib

signal.alarm(18)
pid = int(sys.argv[1])
out = Path(os.environ['N655_SCRATCH'])
iface = 'org.a11y.atspi.Accessible'
root = '/org/a11y/atspi/accessible/root'

def call(connection, bus, path, interface, method, args=None):
    return connection.call_sync(bus, path, interface, method, args, None,
                                Gio.DBusCallFlags.NONE, 1000, None).unpack()

session = Gio.bus_get_sync(Gio.BusType.SESSION, None)
address = call(session, 'org.a11y.Bus', '/org/a11y/bus', 'org.a11y.Bus', 'GetAddress')[0]
connection = Gio.DBusConnection.new_for_address_sync(address,
    Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
    None, None)
for attempt in range(30):
    applications = []
    for bus, path in call(connection, 'org.a11y.atspi.Registry', root, iface, 'GetChildren')[0]:
        actual_pid = call(connection, 'org.freedesktop.DBus', '/org/freedesktop/DBus',
            'org.freedesktop.DBus', 'GetConnectionUnixProcessID', GLib.Variant('(s)', (bus,)))[0]
        applications.append({'pid': actual_pid, 'bus': bus, 'path': path})
    matches = [row for row in applications if row['pid'] == pid]
    if matches:
        break
    time.sleep(.1)

def states(row):
    words = call(connection, row['bus'], row['path'], iface, 'GetState')[0]
    bits = sum(value << (32 * index) for index, value in enumerate(words))
    return [state.value_nick for value, state in Atspi.StateType.__enum_values__.items()
            if bits & (1 << value)]

rows = []
seen = set()
def walk(bus, path, depth=0):
    if (bus, path) in seen or depth > 12 or len(rows) >= 100:
        return
    seen.add((bus, path))
    name = call(connection, bus, path, 'org.freedesktop.DBus.Properties', 'Get',
                GLib.Variant('(ss)', (iface, 'Name')))[0]
    row = {'bus': bus, 'path': path, 'name': name, 'depth': depth,
           'role': call(connection, bus, path, iface, 'GetRoleName')[0]}
    row['states'] = states(row)
    rows.append(row)
    for child_bus, child_path in call(connection, bus, path, iface, 'GetChildren')[0]:
        walk(child_bus, child_path, depth + 1)

for application in matches:
    walk(application['bus'], application['path'])

result = {'pid':pid,'applications':applications,'rows':rows,'steps':[]}
action_iface = 'org.a11y.atspi.Action'
value_iface = 'org.a11y.atspi.Value'
props = 'org.freedesktop.DBus.Properties'
def observation():
    return json.loads((out / 'app-observations.json').read_text())
def values(row):
    return call(connection,row['bus'],row['path'],props,'GetAll',GLib.Variant('(s)',(value_iface,)))[0]
def actions(row):
    return call(connection,row['bus'],row['path'],action_iface,'GetActions')[0]
try:
    candidates = [row for row in rows if row['name'] == 'Drag this edge or use Up and Down to resize the help area.']
    assert len(candidates) == 1, candidates
    row = candidates[0]
    result['control'] = row
    result['numeric_role'] = call(connection,row['bus'],row['path'],iface,'GetRole')[0]
    assert row['role'] == 'grip', row
    assert all(state in row['states'] for state in ('focusable','enabled','visible','showing')), row
    initial = observation()
    current = values(row)
    result['initial_value'] = current
    assert current['CurrentValue'] == initial['height'] == current['MinimumValue']
    assert current['MaximumValue'] == initial['maximum'] > initial['height']
    assert current['MinimumIncrement'] == 1
    assert initial['commits'] == []
    expected = []
    for name, wanted in [('increase',initial['height'] + initial['keyboard_step']),('decrease',initial['height'])]:
        available = actions(row)
        indices = [i for i,a in enumerate(available) if a[0].casefold() == name]
        assert len(indices) == 1, (name, available)
        assert call(connection,row['bus'],row['path'],action_iface,'DoAction',GLib.Variant('(i)',(indices[0],)))[0] is True
        time.sleep(.2)
        expected.append(wanted)
        observed = observation()
        now = values(row)
        assert now['CurrentValue'] == observed['height'] == observed['manual_height'] == wanted, (now,observed)
        assert observed['commits'] == expected, observed
        result['steps'].append({'operation':name,'actions':available,'value':now,'observed':observed})
    wanted = initial['height'] + 1
    call(connection,row['bus'],row['path'],props,'Set',GLib.Variant('(ssv)',(value_iface,'CurrentValue',GLib.Variant('d',wanted))))
    time.sleep(.2)
    expected.append(wanted)
    observed = observation()
    now = values(row)
    assert now['CurrentValue'] == observed['height'] == observed['manual_height'] == wanted
    assert observed['commits'] == expected
    result['steps'].append({'operation':'Value.CurrentValue set','value':now,'observed':observed})
    result['result'] = 'PASS'
except Exception as exc:
    result['result'] = 'FAIL'
    result['exception'] = repr(exc)
    raise
finally:
    (out / 'reader-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
