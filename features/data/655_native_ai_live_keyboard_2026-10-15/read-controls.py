"""Read native AT-SPI focus and exercise real X11 control keys."""
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

x11 = ctypes.CDLL(ctypes.util.find_library('X11'))
xtst = ctypes.CDLL(ctypes.util.find_library('Xtst'))
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
x11.XStringToKeysym.restype = ctypes.c_ulong
x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XKeysymToKeycode.restype = ctypes.c_uint
x11.XFlush.argtypes = [ctypes.c_void_p]
x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
display = x11.XOpenDisplay(None)
assert display
codes = {name: x11.XKeysymToKeycode(display, x11.XStringToKeysym(name.encode()))
         for name in ('Tab', 'Shift_L', 'space', 'Return', 'KP_Enter')}
assert all(codes.values()), codes

def press(name, shift=False):
    if shift:
        assert xtst.XTestFakeKeyEvent(display, codes['Shift_L'], 1, 0)
    assert xtst.XTestFakeKeyEvent(display, codes[name], 1, 0)
    assert xtst.XTestFakeKeyEvent(display, codes[name], 0, 0)
    if shift:
        assert xtst.XTestFakeKeyEvent(display, codes['Shift_L'], 0, 0)
    x11.XFlush(display)
    time.sleep(.2)

def observation():
    return json.loads((out / 'app-observations.json').read_text())

result = {'pid': pid, 'applications': applications, 'rows': rows, 'x11_keycodes': codes,
          'activation_observations': [], 'focus_observations': []}
try:
    initial = observation()
    assert initial['checked'] == {'AI': False, 'Live': False} and not initial['signals'], initial
    press('Tab')
    assert 'focused' in states(controls['AI'])
    result['focus_observations'].append({'action': 'Tab from Before', 'control': 'AI', 'states': states(controls['AI'])})
    expected_signals = []
    for name in ('AI', 'Live'):
        if name == 'Live':
            press('Tab')
            assert 'focused' in states(controls[name])
            assert 'focused' not in states(controls['AI'])
            result['focus_observations'].append({'action': 'Tab from AI', 'control': name, 'states': states(controls[name])})
        for key, checked in [('space', True), ('Return', False), ('KP_Enter', True)]:
            press(key)
            observed = observation()
            expected_signals.append({'control': name, 'checked': checked})
            assert observed['signals'] == expected_signals, observed
            assert observed['checked'][name] is checked, observed
            assert 'focused' in states(controls[name])
            event = observed['key_events'][-1]
            assert event['control'] == name and not event['auto_repeat'], event
            if key == 'KP_Enter':
                assert event['key'] == 16777221 and event['modifiers'] == 0x20000000, event
            result['activation_observations'].append({'control': name, 'native_key': key,
                'actual_qt_event': event, 'checked': checked, 'signal_count': len(expected_signals),
                'atspi_states': states(controls[name])})
    press('Tab', shift=True)
    assert 'focused' in states(controls['AI']) and 'focused' not in states(controls['Live'])
    result['focus_observations'].append({'action': 'Shift+Tab from Live', 'control': 'AI', 'states': states(controls['AI'])})
    press('Tab')
    assert 'focused' in states(controls['Live'])
    press('Tab')
    assert 'focused' not in states(controls['Live'])
    after = next(row for row in rows if row['name'] == 'After controls')
    assert 'focused' in states(after)
    result['focus_observations'].append({'action': 'Tab from Live', 'control': 'After controls', 'states': states(after)})
    result['final_app_observations'] = observation()
    assert result['final_app_observations']['signals'] == expected_signals
finally:
    x11.XCloseDisplay(display)
result['result'] = 'PASS'
print(json.dumps(result, indent=2), flush=True)
