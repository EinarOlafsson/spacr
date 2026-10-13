import os
import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk, GLib
window = Gtk.Window(title='n655-atspi-positive-control')
window.add(Gtk.Button(label='Positive control button'))
window.show_all()
print('READY', os.getpid(), flush=True)
GLib.timeout_add_seconds(8, Gtk.main_quit)
Gtk.main()
