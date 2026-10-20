"""Read the real AT-SPI tree, then move focus using native X11 Tab events."""
import ctypes
import ctypes.util
import json
import signal
import sys
import time
import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi, Gio, GLib

signal.alarm(15)
pid = int(sys.argv[1])
expected = 'Run 2026-10-15 native acceptance'
interface = 'org.a11y.atspi.Accessible'
root = '/org/a11y/atspi/accessible/root'

def call(connection, bus, path, iface, method, args=None):
    return connection.call_sync(bus, path, iface, method, args, None,
                                Gio.DBusCallFlags.NONE, 1000, None).unpack()

session = Gio.bus_get_sync(Gio.BusType.SESSION, None)
address = call(session, 'org.a11y.Bus', '/org/a11y/bus', 'org.a11y.Bus', 'GetAddress')[0]
connection = Gio.DBusConnection.new_for_address_sync(address,
    Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
    None, None)
applications = []
for attempt in range(30):
    applications = []
    roots = call(connection, 'org.a11y.atspi.Registry', root, interface, 'GetChildren')[0]
    for bus, path in roots:
        actual_pid = call(connection, 'org.freedesktop.DBus', '/org/freedesktop/DBus',
                          'org.freedesktop.DBus', 'GetConnectionUnixProcessID',
                          GLib.Variant('(s)', (bus,)))[0]
        applications.append({'pid': actual_pid, 'bus': bus, 'path': path})
    matches = [row for row in applications if row['pid'] == pid]
    if matches:
        break
    time.sleep(.1)

def states(bus, path):
    words = call(connection, bus, path, interface, 'GetState')[0]
    bits = sum(value << (32 * index) for index, value in enumerate(words))
    return [state.value_nick for value, state in Atspi.StateType.__enum_values__.items()
            if bits & (1 << value)]

rows = []
seen = set()
def walk(bus, path, depth=0):
    if (bus, path) in seen or depth > 15 or len(rows) >= 200:
        return
    seen.add((bus, path))
    name = call(connection, bus, path, 'org.freedesktop.DBus.Properties', 'Get',
                GLib.Variant('(ss)', (interface, 'Name')))[0]
    row = {'depth': depth, 'bus': bus, 'path': path, 'name': name,
           'role': call(connection, bus, path, interface, 'GetRoleName')[0],
           'states': states(bus, path)}
    rows.append(row)
    for child_bus, child_path in call(connection, bus, path, interface, 'GetChildren')[0]:
        walk(child_bus, child_path, depth + 1)

for application in matches:
    walk(application['bus'], application['path'])
headings = [row for row in rows if row['name'] == expected and 'focusable' in row['states']]
result = {'pid': pid, 'applications': applications, 'rows': rows, 'focusable_heading_count': len(headings)}
assert len(headings) == 1, result
heading = headings[0]
assert all(state in heading['states'] for state in ['visible', 'showing', 'enabled', 'focusable']), heading
assert 'focused' not in heading['states'], heading

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
tab = x11.XKeysymToKeycode(display, x11.XStringToKeysym(b'Tab'))
shift = x11.XKeysymToKeycode(display, x11.XStringToKeysym(b'Shift_L'))

def press_tab(reverse=False):
    if reverse:
        assert xtst.XTestFakeKeyEvent(display, shift, 1, 0)
    assert xtst.XTestFakeKeyEvent(display, tab, 1, 0)
    assert xtst.XTestFakeKeyEvent(display, tab, 0, 0)
    if reverse:
        assert xtst.XTestFakeKeyEvent(display, shift, 0, 0)
    x11.XFlush(display)
    time.sleep(.25)

try:
    focus_steps = []
    for index in range(6):
        press_tab()
        state = states(heading['bus'], heading['path'])
        focus_steps.append({'tab_count': index + 1, 'heading_states': state})
        if 'focused' in state:
            break
    result['native_tab_steps'] = focus_steps
    assert 'focused' in focus_steps[-1]['heading_states'], result
    time.sleep(.4)
    press_tab()
    result['after_tab_away'] = states(heading['bus'], heading['path'])
    assert 'focused' not in result['after_tab_away'], result
    press_tab(reverse=True)
    result['after_shift_tab_back'] = states(heading['bus'], heading['path'])
    assert 'focused' in result['after_shift_tab_back'], result
finally:
    x11.XCloseDisplay(display)
result['result'] = 'PASS'
print(json.dumps(result, indent=2), flush=True)
