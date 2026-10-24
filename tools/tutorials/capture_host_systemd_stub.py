#!/usr/bin/python3
"""A stand-in ``org.freedesktop.systemd1`` for the private capture-host bus.

Run with the SYSTEM ``/usr/bin/python3`` (it needs PyGObject). Only the
private session bus of :mod:`gnome_capture_host` ever sees it.

Why it exists: gnome-shell 42 starts its X11 services by asking the *user*
systemd on the session bus to start ``gnome-session-x11-services-ready.target``.
Our shell runs inside the maintainer's systemd user scope (so it counts as
"systemd managed") but on a private bus where no user systemd answers; the
call fails and mutter then refuses to initialise its X11 window manager, so
Xwayland windows (the spaCR app on xcb) would not be managed or composited.

This stub acknowledges unit start/stop requests (logging each), which is all
the shell waits for. No unit is actually started: the GNOME settings daemons
are not needed for recording a Qt application.
"""
import sys

import gi
gi.require_version('Gio', '2.0')
from gi.repository import Gio, GLib  # noqa: E402

XML = """
<node>
  <interface name="org.freedesktop.systemd1.Manager">
    <method name="StartUnit"><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="o" direction="out"/></method>
    <method name="StopUnit"><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="o" direction="out"/></method>
    <method name="RestartUnit"><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="o" direction="out"/></method>
    <method name="GetUnit"><arg type="s" direction="in"/><arg type="o" direction="out"/></method>
    <method name="Subscribe"/>
    <method name="Unsubscribe"/>
    <signal name="JobRemoved"><arg type="u"/><arg type="o"/><arg type="s"/><arg type="s"/></signal>
  </interface>
</node>
"""
_job = [0]


def on_call(connection, sender, path, interface, method, params, invocation):
    print(f'systemd-stub: {method}{tuple(params.unpack())}', flush=True)
    if method in ('StartUnit', 'StopUnit', 'RestartUnit'):
        _job[0] += 1
        job = f'/org/freedesktop/systemd1/job/{_job[0]}'
        invocation.return_value(GLib.Variant('(o)', (job,)))
        unit = params.unpack()[0]
        GLib.idle_add(lambda: connection.emit_signal(
            None, '/org/freedesktop/systemd1', 'org.freedesktop.systemd1.Manager', 'JobRemoved',
            GLib.Variant('(uoss)', (_job[0], job, unit, 'done'))) and False)
    elif method == 'GetUnit':
        invocation.return_dbus_error('org.freedesktop.systemd1.NoSuchUnit',
                                     'capture host: no units are managed here')
    else:
        invocation.return_value(None)


def main():
    loop = GLib.MainLoop()
    info = Gio.DBusNodeInfo.new_for_xml(XML)

    def acquired(connection, name):
        print('systemd-stub: owns', name, flush=True)

    def bus(connection, name):
        connection.register_object('/org/freedesktop/systemd1', info.interfaces[0], on_call)

    Gio.bus_own_name(Gio.BusType.SESSION, 'org.freedesktop.systemd1', Gio.BusNameOwnerFlags.NONE,
                     bus, acquired, lambda *a: loop.quit())
    loop.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
