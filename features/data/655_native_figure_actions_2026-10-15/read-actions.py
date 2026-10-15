"""Read native clear/fold actions through the real Linux accessibility bridge."""
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

signal.alarm(45)
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
    if (bus, path) in seen or depth > 12 or len(rows) >= 300:
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

result = {'pid': pid, 'applications': applications, 'initial_rows': rows,
          'activations': [], 'disabled': []}
action_iface = 'org.a11y.atspi.Action'
def observation():
    return json.loads((out / 'app-observations.json').read_text())
def actions(row):
    return call(connection, row['bus'], row['path'], action_iface, 'GetActions')[0]
def invoke(row, index):
    return call(connection, row['bus'], row['path'], action_iface,
                'DoAction', GLib.Variant('(i)', (index,)))[0]
def find(name):
    found = [row for row in rows if row['name'] == name and row['role'] == 'push button']
    assert len(found) == 1, (name, found)
    return found[0]
def press(row):
    available = actions(row)
    indices = [i for i, action in enumerate(available) if action[0].casefold() == 'press']
    assert len(indices) == 1, (row, available)
    assert invoke(row, indices[0]) is True
    time.sleep(.25)
    result['activations'].append({'name': row['name'], 'actions': available,
                                  'observed': observation()})
try:
    clear = find('Clear figures')
    header = find('Native acceptance run')
    for row in (clear, header):
        assert row['role'] == 'push button', row
        assert all(flag in row['states'] for flag in ['enabled', 'focusable', 'visible', 'showing']), row
    assert 'expandable' in header['states'] and 'expanded' in header['states'], header
    initial = observation()
    assert initial['queue_count'] == initial['list_count'] == 1
    assert initial['clear_clicks'] == initial['fold_calls'] == 0
    assert initial['expanded'] and not initial['collapsed'] and initial['visible_cells'] == 1
    press(clear)
    actual = observation()
    assert actual['clear_clicks'] == 1 and actual['queue_count'] == actual['list_count'] == 0, actual
    assert actual['fold_calls'] == 0 and not actual['collapsed'], actual
    rows = []
    seen = set()
    for application in matches:
        walk(application['bus'], application['path'])
    result['after_clear_rows'] = rows
    header = find('Native acceptance run')
    assert 'expandable' in header['states'] and 'expanded' in header['states'], header
    press(header)
    actual = observation()
    assert actual['fold_calls'] == 1 and actual['clear_clicks'] == 1, actual
    assert actual['collapsed'] and not actual['expanded'] and actual['visible_cells'] == 0, actual
    assert actual['workspace']['collapsed'] == [['Native acceptance run', 0]], actual
    rows = []
    seen = set()
    for application in matches:
        walk(application['bus'], application['path'])
    result['after_fold_rows'] = rows
    clear = find('Clear figures')
    header = find('Native acceptance run')
    assert 'expandable' in header['states'] and 'expanded' not in header['states'], header
    assert header['role'] == 'push button'
    press(find('Disable isolated controls'))
    actual = observation()
    assert not actual['clear_enabled'] and not actual['header_enabled'], actual
    for row in (clear, header):
        assert actions(row) == [], row
        refusal = None
        try:
            disabled_result = invoke(row, 0)
        except GLib.Error as exc:
            refusal = Gio.DBusError.get_remote_error(exc)
            assert refusal == 'org.freedesktop.DBus.Error.UnknownObject', (row, refusal)
            assert "No such object path '" + row['path'] + "'" in str(exc), str(exc)
            disabled_result = None
        else:
            assert disabled_result is False, (row, disabled_result)
        now = states(row)
        assert 'enabled' not in now, (row, now)
        actual = observation()
        assert actual['clear_clicks'] == actual['fold_calls'] == 1, actual
        assert actual['queue_count'] == actual['list_count'] == 0 and actual['collapsed'], actual
        result['disabled'].append({'name': row['name'], 'actions': [],
            'DoAction_result': disabled_result, 'protocol_refusal': refusal,
            'same_accessible_state_readable': True, 'states': now, 'observed': actual})
    result['final_app_observations'] = observation()
    result['result'] = 'PASS'
    finish = find('Finish isolated probe')
    available = actions(finish)
    index = next(i for i, action in enumerate(available) if action[0].casefold() == 'press')
    assert invoke(finish, index) is True
except Exception as exc:
    result['result'] = 'FAIL'
    result['exception'] = repr(exc)
    raise
finally:
    (out / 'reader-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
