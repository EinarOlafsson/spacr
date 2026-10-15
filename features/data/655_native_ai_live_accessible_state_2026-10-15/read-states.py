"""Read native AT-SPI roles/states and invoke its actual accessible actions."""
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
controls = {}
for name in ('AI', 'Live'):
    candidates = [row for row in rows if row['name'] == name and 'focusable' in row['states']]
    assert len(candidates) == 1, {'name': name, 'rows': rows}
    controls[name] = candidates[0]
    assert all(state in candidates[0]['states'] for state in ('enabled', 'visible', 'showing', 'focusable'))

action_iface = 'org.a11y.atspi.Action'

def actions(row):
    return call(connection, row['bus'], row['path'], action_iface, 'GetActions')[0]

def invoke(row, index):
    return call(connection, row['bus'], row['path'], action_iface, 'DoAction',
                GLib.Variant('(i)', (index,)))[0]

def observation():
    return json.loads((out / 'app-observations.json').read_text())

result = {'pid': pid, 'applications': applications, 'rows': rows,
          'activation_observations': [], 'disabled_observations': [],
          'scope': 'Native external AT-SPI actions and state; existing toggled signal boundary only.'}
try:
    initial = observation()
    assert initial['checked'] == {'AI': False, 'Live': False} and not initial['signals'], initial
    expected = []
    for name, row in controls.items():
        assert row['role'] == 'check box', row
        assert 'checkable' in row['states'] and 'checked' not in row['states'], row
        assert 'read-only' not in row['states'], row
        available = actions(row)
        toggle_indices = [index for index, action in enumerate(available)
                          if action[0].casefold() == 'toggle']
        assert len(toggle_indices) == 1, (name, available)
        for wanted in (True, False):
            assert invoke(row, toggle_indices[0]) is True
            time.sleep(.2)
            expected.append({'control': name, 'checked': wanted})
            observed = observation()
            assert observed['signals'] == expected, observed
            assert observed['checked'][name] is wanted, observed
            now = states(row)
            assert ('checked' in now) is wanted, (name, wanted, now)
            assert 'checkable' in now and 'read-only' not in now
            result['activation_observations'].append({
                'control': name, 'action': available[toggle_indices[0]],
                'checked': wanted, 'atspi_states': now,
                'signal_count': len(observed['signals'])})
    disable = next(row for row in rows if row['name'] == 'Disable controls')
    disable_actions = actions(disable)
    press_indices = [index for index, action in enumerate(disable_actions)
                     if action[0].casefold() == 'press']
    assert len(press_indices) == 1, disable_actions
    assert invoke(disable, press_indices[0]) is True
    time.sleep(.2)
    assert observation()['enabled'] == {'AI': False, 'Live': False}
    for name, row in controls.items():
        available = actions(row)
        assert not available, (name, available)
        refusal = None
        try:
            disabled_result = invoke(row, 0)
        except GLib.Error as exc:
            refusal = Gio.DBusError.get_remote_error(exc)
            assert refusal == 'org.freedesktop.DBus.Error.UnknownObject', (name, refusal)
            assert "No such object path '" + row['path'] + "'" in str(exc), str(exc)
            disabled_result = None
        else:
            assert disabled_result is False, (name, disabled_result)
        time.sleep(.1)
        observed = observation()
        now = states(row)
        assert observed['signals'] == expected and observed['checked'][name] is False, observed
        assert 'enabled' not in now and 'checked' not in now, now
        result['disabled_observations'].append({
            'control': name, 'actions': available, 'DoAction_0_result': disabled_result,
            'DoAction_0_refusal': refusal, 'same_accessible_object_readable': True,
            'atspi_states': now, 'signal_count_unchanged': len(expected)})
    result['final_app_observations'] = observation()
    assert not result['final_app_observations']['key_events'], 'No old keyboard cases should run'
    result['result'] = 'PASS'
except Exception as exc:
    result['result'] = 'FAIL'
    result['exception'] = repr(exc)
    raise
finally:
    (out / 'reader-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)

