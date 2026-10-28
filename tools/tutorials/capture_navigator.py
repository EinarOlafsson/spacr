"""Navigator's native local-input tour; never execute an analysis or AI request.

External Files drops use the owned GNOME desktop and a reviewed entry position.
The first Files preflight records its real window and exits before any drop.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

import capture_ui
from capture_ui_drv_a import Rec


SCENES = ('01_navigator_start', '02_help_navigator', '03_dock_right_panels',
          '04_enter_case_query', '05_native_file_drop', '06_native_folder_drop',
          '07_typed_project_path', '08_measured_evidence_cards', '09_annotation_only_scope',
          '10_saved_settings_preview', '11_module_workflow', '12_incremental_next_steps',
          '13_toxoplasma_network', '14_back_escape', '15_native_source_handoff',
          '16_running_inspect', '17_ai_consent_cancel', '18_home_fresh_reset')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(stage):
    """Make exclusive local demonstration copies; never rewrite a source input."""
    stage = Path(stage).resolve()
    project = stage / 'annotate_fresh/example_data/plate1'
    image = project / 'data/multiple_nucleus/multiple_pathogens/plate1_E01/cell_png/plate1_E01_10_1_4.png'
    database = project / 'measurements/measurements.db'
    settings = project / 'settings/annotate_settings.csv'
    for path in (image, database, settings):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(stage):
            raise ValueError(f'Missing private real Annotate input: {path}')
    original = {str(path): digest(path) for path in (image, database, settings)}
    root = stage / 'navigator-inputs'
    root.mkdir(exist_ok=False)
    file_tray, folder_tray = root / 'file-tray', root / 'folder-tray'
    file_tray.mkdir()
    folder_tray.mkdir()
    file_copy = file_tray / image.name
    folder_copy = folder_tray / 'plate1_E01_cell_png'
    folder_copy.mkdir()
    for target in (file_copy, folder_copy / image.name):
        shutil.copy2(image, target)
        if digest(target) != original[str(image)]:
            raise ValueError('The real PNG changed while making its private drop tray')
    annotation = root / 'annotation_history_only_example/measurements.db'
    annotation.parent.mkdir()
    # These are the real annotation/history schema categories, not measured features.
    with sqlite3.connect(annotation) as connection:
        connection.execute('CREATE TABLE image_annotations (plateID TEXT, rowID TEXT, columnID TEXT, image_id INTEGER)')
        connection.execute('CREATE TABLE image_annotations_meta (revision INTEGER)')
        connection.execute('CREATE TABLE image_annotations_history (revision INTEGER)')
    return dict(project=project, image=image, database=database, settings=settings,
                file_tray=file_tray, folder_tray=folder_tray, file_copy=file_copy,
                folder_copy=folder_copy, annotation=annotation, original=original)


def checked_input_routes(inputs):
    """Read the actual bounded inspector, including the explicitly labelled example."""
    from spacr.qt.widgets.home import _easy_inspect_path
    rows = {key: _easy_inspect_path(str(inputs[key]))
            for key in ('database', 'settings', 'annotation')}
    if 'graph_builder' not in {row[0] for row in rows['database']['actions']}:
        raise ValueError('The real measurements do not offer Graph Builder')
    saved = rows['settings']['context'].get('saved_settings', {})
    if saved.get('module') != 'annotate' or [row[0] for row in rows['settings']['actions']] != ['annotate']:
        raise ValueError('The retained CSV does not offer the actual Annotate settings preview')
    annotation = rows['annotation']
    if ([row[0] for row in annotation['actions']] != ['db_browser']
            or annotation['context']['analysis_inputs']
            or not annotation['context']['database_schemas'][0]['annotation_only']):
        raise ValueError('The labelled history-only example was misclassified as measurements')
    return rows


def _query(*args):
    return subprocess.check_output(args, text=True, timeout=5)


def _x_rectangle(stats):
    def number(label):
        match = re.search(r'(?m)^\s*' + re.escape(label) + r':\s*(-?\d+)\s*$', stats)
        if not match:
            raise RuntimeError('Missing actual X geometry: ' + label)
        return int(match.group(1))
    return [number(label) for label in ('Absolute upper-left X', 'Absolute upper-left Y', 'Width', 'Height')]


def _visible_files_geometry(client_rect, properties, monitor_rect):
    """Inset GTK's invisible CSD shadows; NET extents describe added WM borders.

    See Gdk.Window.set_shadow_width and EWMH section5.17. The original X client
    rectangle and both properties remain in the capture metadata unmodified.
    """
    def extents(name):
        line = next((line for line in properties.splitlines()
                     if line.startswith(name + '(') or line.startswith(name + ':')), None)
        if line is None:
            raise RuntimeError('The requested Files frame property was not returned: ' + name)
        if re.fullmatch(re.escape(name) + r':\s*(?:not found\.|no such atom on any window\.)\s*', line):
            return None
        match = re.fullmatch(re.escape(name) + r'\(CARDINAL\)\s*=\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*', line)
        if not match:
            raise RuntimeError('Malformed actual Files frame extents: ' + line)
        return [int(value) for value in match.groups()]
    gtk = extents('_GTK_FRAME_EXTENTS')
    net = extents('_NET_FRAME_EXTENTS')
    x, y, width, height = client_rect
    left, right, top, bottom = gtk if gtk is not None else [0, 0, 0, 0]
    visible = [x + left, y + top, width - left - right, height - top - bottom]
    nl, nr, nt, nb = net if net is not None else [0, 0, 0, 0]
    wm_frame = [x - nl, y - nt, width + nl + nr, height + nt + nb]
    mx, my, mw, mh = monitor_rect
    def inside(rect):
        rx, ry, rw, rh = rect
        return (rw > 0 and rh > 0 and mx <= rx and my <= ry
                and rx + rw <= mx + mw and ry + rh <= my + mh)
    if (mw <= 0 or mh <= 0 or visible[2] < 400 or visible[3] < 300 or not inside(visible)
            or (net is not None and any(net) and not inside(wm_frame))):
        raise RuntimeError(f'Files visible geometry is outside its actual monitor or unusable: '
                           f'client={client_rect} gtk={gtk} net={net} visible={visible} monitor={monitor_rect}')
    return dict(rect=visible, client_rect=list(client_rect), gtk_frame_extents=gtk,
                net_frame_extents=net, wm_frame_rect=wm_frame, monitor_rect=list(monitor_rect),
                rect_kind='visible_frame_after_gtk_shadow', entry_geometry_verified=False)


def files_window(stage, tray):
    """Read a single mapped Files window on this stage's owned private display."""
    from capture_saved_plots import _inspection_display
    display = _inspection_display(stage)
    monitor = _x_rectangle(_query('xwininfo', '-root', '-stats'))
    clients = _query('xprop', '-root', '_NET_CLIENT_LIST_STACKING')
    found = []
    for wid in dict.fromkeys(re.findall(r'0x[0-9a-fA-F]+', clients)):
        props = _query('xprop', '-id', wid, '_NET_WM_PID', '_NET_WM_NAME', 'WM_CLASS',
                       '_GTK_FRAME_EXTENTS', '_NET_FRAME_EXTENTS')
        match = re.search(r'_NET_WM_PID\([^)]*\)\s*=\s*(\d+)', props)
        if not match or 'nautilus' not in props.casefold():
            continue
        pid = int(match.group(1))
        process = Path('/proc') / str(pid)
        try:
            argv = [os.fsdecode(arg) for arg in (process / 'cmdline').read_bytes().split(b'\0') if arg]
            env = dict(row.split('=', 1) for row in (process / 'environ').read_text().split('\0') if '=' in row)
            owned = (process.stat().st_uid == os.getuid() and argv
                     and Path(argv[0]).name == 'nautilus' and env.get('DISPLAY') == display
                     and Path(env.get('SPACR_CAPTURE_HOST', '')).samefile(Path(stage) / 'gnome-host')
                     and Path(env.get('XDG_RUNTIME_DIR', '')).samefile(Path(stage) / 'gnome-host/run')
                     and env.get('DBUS_SESSION_BUS_ADDRESS') == os.environ.get('DBUS_SESSION_BUS_ADDRESS')
                     and bool(env.get('DBUS_SESSION_BUS_ADDRESS')))
        except (OSError, ValueError):
            owned = False
        if not owned:
            raise RuntimeError(f'Files is not owned by this private host: pid={pid} properties={props!r}')
        stats = _query('xwininfo', '-id', wid, '-stats')
        if 'Map State: IsViewable' not in stats:
            continue
        geometry = _visible_files_geometry(_x_rectangle(stats), props, monitor)
        found.append(dict(window_id=wid, pid=pid, **geometry, properties=props, xwininfo=stats,
                          argv=argv, tray=str(tray), display=display, private_host_verified=True))
    if not found:
        return None
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one owned mapped Files window, found {found!r}')
    return found[0]


def checked_source_point(review, actual, target):
    """A point is a Root-reviewed real entry, never inferred from a window center."""
    if (review.get('window_rect') != actual['rect'] or review.get('entry_name') != target.name
            or review.get('tray_name') != target.parent.name):
        raise ValueError('The Files entry review does not match the actual tray/window geometry')
    point = review.get('source_point')
    if (not isinstance(point, list) or len(point) != 2
            or any(type(value) is not int for value in point)):
        raise ValueError('A reviewed integer native entry point is required')
    x, y, width, height = actual['rect']
    if not (x + 8 < point[0] < x + width - 8 and y + 60 < point[1] < y + height - 8):
        raise ValueError('The reviewed entry point is outside the real Files content window')
    return point


def drop_text_point(edit, files_rect):
    """Choose actual visible editor text, excluding inset controls and Files."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QRegion
    from PySide6.QtWidgets import QAbstractButton, QLineEdit

    if (not isinstance(edit, QLineEdit) or not edit.isVisible() or not edit.isEnabled()
            or edit.isReadOnly()):
        raise RuntimeError('The actual Navigator text editor is unavailable for a native drop')
    text = edit.contentsRect().marginsRemoved(edit.textMargins()).adjusted(4, 4, -4, -4)
    available = QRegion(text) & edit.visibleRegion()
    for child in edit.findChildren(QAbstractButton):
        if child.isVisible():
            rect = QRect(child.mapTo(edit, QPoint(0, 0)), child.size()).adjusted(-4, -4, 4, 4)
            available -= QRegion(rect)
    x, y, width, height = files_rect
    blocked = QRect(edit.mapFromGlobal(QPoint(x, y)), QRect(0, 0, width, height).size())
    available -= QRegion(blocked.adjusted(-4, -4, 4, 4))
    if available.isEmpty():
        raise RuntimeError('The actual Files window leaves no uncovered Navigator text area')
    point = available.boundingRect().center()
    if not available.contains(point):
        raise RuntimeError('The actual uncovered Navigator text area has no safe central point')
    return point


def _close_files(adapter, actual):
    x, y, width, _height = actual['rect']
    adapter.call('move_to', x + width // 2, y + 20)
    adapter.call('click', 'left')
    active = _query('xprop', '-root', '_NET_ACTIVE_WINDOW')
    if int(re.search(r'0x[0-9a-fA-F]+', active).group(), 16) != int(actual['window_id'], 16):
        raise RuntimeError('Refusing Alt+F4: the owned Files window is not active')
    adapter.call('key_down', 'Alt_L')
    try:
        adapter.call('press', '#keysym:0xffc1')
    finally:
        adapter.call('key_up', 'Alt_L')


def native_drop(capture, window, page, stage, tray, target, review, write_json, captures, *, preflight=False):
    """Select the real single entry in Files and drag it with the real pointer."""
    from capture_saved_plots import _inspection_display
    h = capture_ui._h()
    adapter = capture.host.adapter
    _inspection_display(stage)
    if len(list(tray.iterdir())) != 1 or next(tray.iterdir()) != target:
        raise ValueError('Native drop trays must contain exactly their named real entry')
    log = open(captures / (tray.name + '-files.log'), 'x', encoding='utf-8')
    env = dict(os.environ, GDK_BACKEND='x11')
    subprocess.Popen(['/usr/bin/nautilus', '--new-window', str(tray)], env=env,
                     stdout=log, stderr=subprocess.STDOUT)
    log.close()
    actual = h.wait_until(capture, lambda: files_window(stage, tray), 30,
                          'The actual owned Files window did not become mapped')
    # Focus a measured titlebar with real input; never XActivate or move the window by code.
    x, y, width, _height = actual['rect']
    adapter.call('move_to', x + width // 2, y + 20)
    adapter.call('click', 'left')
    h.pause(capture, .4)
    active = _query('xprop', '-root', '_NET_ACTIVE_WINDOW')
    if int(re.search(r'0x[0-9a-fA-F]+', active).group(), 16) != int(actual['window_id'], 16):
        raise RuntimeError('The owned Files window did not receive actual native focus')
    # Normal GNOME keyboard tiling makes the private source window reproducible.
    adapter.call('key_down', 'Super_L')
    try:
        adapter.call('press', 'Left')
    finally:
        adapter.call('key_up', 'Super_L')
    h.pause(capture, .6)
    actual = h.wait_until(capture, lambda: files_window(stage, tray), 30,
                          'The actual owned Files window did not become mapped after native tiling')
    # Ctrl+L is Files' normal location entry; no internal file-manager state is injected.
    adapter.call('key_down', 'Control_L')
    try:
        adapter.call('press', 'l')
    finally:
        adapter.call('key_up', 'Control_L')
    adapter.call('type_text', str(tray))
    adapter.call('press', 'Return')
    h.pause(capture, 1)
    adapter.call('press', 'Home')
    h.pause(capture, .5)
    actual = h.wait_until(capture, lambda: files_window(stage, tray), 30,
                          'The actual owned Files window did not become mapped after native location entry')
    write_json(captures / (tray.name + '-preflight.json'), actual)
    if preflight:
        h.pause(capture, 2)
        _close_files(adapter, actual)
        h.pause(capture, .5)
        return actual
    point = checked_source_point(review, actual, target)
    text_point = drop_text_point(page.path_edit, actual['rect'])
    destination = adapter.target_point(page.path_edit, text_point)
    # Dropping onto an area hidden by the source window would hit Files, not Navigator.
    x, y, width, height = actual['rect']
    if x <= destination[0] < x + width and y <= destination[1] < y + height:
        raise RuntimeError('The real Files window covers the Navigator drop target; review native placement first')
    adapter.call('move_to', *point)
    adapter.call('click', 'left')
    adapter.call('button_down', 'left')
    try:
        adapter.call('move_to', *destination, duration=2.5)
    finally:
        adapter.call('button_up', 'left')
    h.wait_until(capture, lambda: page.path_edit.text() == str(target), 10,
                 'The genuine Files drop did not deliver exactly its local path')
    adapter.note('navigator-native-files-drop', source_window=actual, source_point=point,
                 destination=destination, text_area_point=[text_point.x(), text_point.y()],
                 accepted_path=page.path_edit.text())
    _close_files(adapter, actual)
    h.pause(capture, .5)
    return actual


def record_navigator(app, window, stage, captures, capture, settle, write_json, timeout,
                     *, files_preflight=False, files_points=None, files_points_sha256=None):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox, QLineEdit, QMenu, QPlainTextEdit
    from spacr.qt import preferences
    from spacr.qt.bridge import registry
    from capture_home_live import inside_next_dialog

    if not capture_ui.live(capture) or capture.host is None:
        raise RuntimeError('Navigator requires the owned native GNOME capture host')
    if preferences.get_show_alpha():
        raise RuntimeError('Navigator tutorial must keep alpha modules off')
    page = window._startup
    if not hasattr(page, '_navigator_aside_split') or window._stack.currentWidget() is not page:
        raise RuntimeError('The actual fresh Home is not Navigator')
    if page.path_edit.ai.isChecked() or page._ai_provider:
        raise RuntimeError('Navigator AI must start off without a configured provider')
    h = capture_ui._h()
    inputs = prepare_inputs(stage)
    input_routes = checked_input_routes(inputs)
    write_json(captures / 'navigator-input-routes.json', input_routes)
    reviews = {}
    if files_points is not None:
        if not files_points_sha256 or digest(files_points) != files_points_sha256:
            raise ValueError('Files source-point review SHA must match its actual bytes')
        reviews = json.loads(Path(files_points).read_text())
    checks = []

    def wait(predicate, what):
        return h.wait_until(capture, predicate, timeout, what)

    def usable(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError(f'The actual Navigator control is unavailable: {widget.objectName()}')
        return widget

    def click(widget):
        h.click(usable(widget), widget.visibleRegion().boundingRect().center())
        settle(.5)

    def enter(value):
        edit = page.findChild(QLineEdit, 'EasyModePath')
        if edit is None:
            raise RuntimeError('Navigator has no actual native path edit')
        capture_ui.set_text(capture, usable(edit), str(value), commit=False)
        h.key(edit, Qt.Key_Return)
        wait(lambda: page.field.cards and all(not card._growing for card in page.field.cards),
             'Navigator did not reveal its actual cards on Enter')
        return list(page.field.cards)

    def card(key):
        rows = [row for row in page.field.cards if row._key == key]
        if len(rows) != 1:
            raise RuntimeError(f'Expected one actual Navigator card {key}, got {[r._key for r in page.field.cards]}')
        row = rows[0]
        bar = page.field._scrollbar
        if bar.maximum():
            wanted = max(bar.minimum(), min(bar.maximum(), bar.value()
                         + row.geometry().center().x() - page.field._row_rect.center().x()))
            if wanted != bar.value():
                h.drag_scroll_bar(capture, bar, wanted)
                settle(.3)
        return usable(row)

    def home():
        edge = window._dock_edge
        if edge.is_collapsed():
            click(edge)
        rows = [row for row in window._sidebar._rows if row.key == '__home__' and row.isVisible()]
        if len(rows) != 1:
            raise RuntimeError('No unique visible native Home row')
        click(rows[0])
        wait(lambda: window._stack.currentWidget() is page and not page.path_edit.text()
             and page._workflow is None and not page.field.cards, 'Home did not reset Navigator')

    def modal_cancel(match, assertion):
        def steps(dialog):
            assertion(dialog)
            box = dialog.findChild(QDialogButtonBox)
            if box is None:
                raise RuntimeError('The actual dialog has no Cancel control')
            click(box.button(QDialogButtonBox.Cancel))
        return inside_next_dialog(app, match, steps)

    with Rec(capture, SCENES[0], settle):
        capture_ui.point_at(capture, page.path_edit, dwell=2)
    if files_preflight:
        for index, tray, target in ((4, inputs['file_tray'], inputs['file_copy']),
                                    (5, inputs['folder_tray'], inputs['folder_copy'])):
            with Rec(capture, SCENES[index], settle, desktop=True):
                native_drop(capture, window, page, stage, tray, target,
                            {}, write_json, captures, preflight=True)
        write_json(captures / 'navigator-preflight.json', dict(preflight_only=True,
                   drop_executed=False, analysis_executed=False, source_preserved=True))
        return
    if set(reviews) != {'file', 'folder'}:
        raise ValueError('Both actual Files entry positions must be reviewed before the full tour')
    with Rec(capture, SCENES[1], settle):
        bar = window.menuBar()
        help_action = next(action for action in bar.actions() if action.text().replace('&', '') == 'Help')
        # Checked QAction toggles off first; enable it again through the same real menu.
        for expected in (False, True):
            h.click(bar, bar.actionGeometry(help_action).center())
            menu = wait(lambda: app.activePopupWidget(), 'Help menu did not open')
            if not isinstance(menu, QMenu) or window._act_easy_mode not in menu.actions():
                raise RuntimeError('Help has no actual Navigator action')
            h.click(menu, menu.actionGeometry(window._act_easy_mode).center())
            wait(lambda: preferences.get_easy_mode() == expected, 'Help did not apply the actual Navigator choice')
        page = window._startup
        wait(lambda: window._stack.currentWidget() is page, 'Help Navigator did not return Home')
    with Rec(capture, SCENES[2], settle):
        capture_ui.open_module(capture, window, 'measure', settle, prefer='sidebar')
        home()
        handle = page._navigator_aside_split.handle(1)
        if page._navigator_aside_split.is_collapsed('Widgets'):
            click(handle)
        news = page._navigator_aside._navigator_folders['News']
        if not news.shut:
            click(news.heading)
        wait(lambda: news.shut and not news.body.isVisible(),
             'The native News header did not hide its release-history body')
        capture_ui.point_at(capture, news.heading, dwell=1.5)
        click(handle)
        wait(lambda: page._navigator_aside_split.is_collapsed('Widgets'),
             'The native Widgets handle did not fold the side panel')
        checks.append(dict(scene=SCENES[2], dock_native=True, right_panel_native=True))
    with Rec(capture, SCENES[3], settle):
        enter('mEaSuRe')
        card('measure')
    home()
    for index, kind, tray, target in ((4, 'file', inputs['file_tray'], inputs['file_copy']),
                                      (5, 'folder', inputs['folder_tray'], inputs['folder_copy'])):
        with Rec(capture, SCENES[index], settle, desktop=True):
            native_drop(capture, window, page, stage, tray, target, reviews[kind], write_json, captures)
            wait(lambda: page._last_result and page._last_result['path'] == str(target),
                 'Navigator did not inspect the exact dropped local input')
            checks.append(dict(scene=SCENES[index], accepted_path=str(target), genuine_files_drop=True))
        home()
    with Rec(capture, SCENES[6], settle):
        enter(inputs['project'])
        wait(lambda: page._last_result and page._last_result['path'] == str(inputs['project']),
             'Navigator did not inspect the typed project')
    with Rec(capture, SCENES[7], settle):
        enter(inputs['database'])
        card('graph_builder')
        capture_ui.point_at(capture, card('graph_builder'), dwell=2)
    home()
    with Rec(capture, SCENES[8], settle):
        enter(inputs['annotation'])
        if [row._key for row in page.field.cards] != ['db_browser']:
            raise RuntimeError('The history-only example incorrectly offered measured-feature analysis')
        checks.append(dict(scene=SCENES[8], example=True, measured_features=False))
    home()
    with Rec(capture, SCENES[9], settle, desktop=True):
        enter(inputs['settings'])
        def preview(dialog):
            text = dialog.findChild(QPlainTextEdit, 'EasySettingsPreview')
            if text is None or not text.isReadOnly() or 'annotation_column' not in text.toPlainText():
                raise RuntimeError('The real saved-settings preview did not appear')
            settle(2)
        click(card('annotate'))
        wait(lambda: page._workflow is not None, 'Saved settings did not reveal its module workflow')
        hook = modal_cancel(lambda dialog: dialog.findChild(QPlainTextEdit, 'EasySettingsPreview') is not None, preview)
        click(page._workflow.central)
        hook.finish(capture, timeout)
    home()
    with Rec(capture, SCENES[10], settle):
        enter('Mask Generation')
        click(card('mask'))
        wait(lambda: page._workflow is not None, 'The real module workflow did not open')
    with Rec(capture, SCENES[11], settle):
        workflow = page._workflow
        old_edges = {(edge.edge['from'], edge.edge['to']) for edge in workflow.view.edges}
        node = workflow.view.nodes.get('measure')
        if node is None or not node.isVisible():
            raise RuntimeError('The actual Mask workflow has no visible Measure next step')
        point = workflow.view.mapFromScene(node.sceneBoundingRect().center())
        if not workflow.view.viewport().rect().contains(point):
            raise RuntimeError('The actual next-step node is outside the visible viewport')
        h.click(workflow.view.viewport(), point)
        wait(lambda: workflow._chosen_key == 'measure', 'The actual next-step click did not reveal Measure')
        if workflow.central._key != 'mask' or window._stack.currentWidget() is not page:
            raise RuntimeError('Exploration replaced its original root or executed a module')
        if not old_edges <= {(edge.edge['from'], edge.edge['to']) for edge in workflow.view.edges}:
            raise RuntimeError('Incremental exploration discarded an earlier branch')
    home()
    with Rec(capture, SCENES[12], settle):
        enter('Toxoplasma')
        click(card('toxoplasma'))
        wait(lambda: page._workflow is not None, 'Toxoplasma did not reveal its real network')
    with Rec(capture, SCENES[13], settle):
        click(page._workflow.back)
        wait(lambda: page._workflow is None, 'Back did not restore suggestions')
        click(card('toxoplasma'))
        h.key(page, Qt.Key_Escape)
        wait(lambda: page._workflow is None, 'Escape did not restore suggestions')
    home()
    with Rec(capture, SCENES[14], settle):
        enter(inputs['database'])
        click(card('db_browser'))
        wait(lambda: page._workflow is not None, 'The actual Database Browser workflow did not open')
        click(page._workflow.central)
        wait(lambda: window._screens.get('db_browser') is not None
             and window._stack.currentWidget() is window._screens['db_browser'], 'Native Source handoff failed')
        screen = window._screens['db_browser']
        wait(lambda: getattr(screen, '_db', None) is not None, 'The native browser did not load its database')
        if Path(screen._db.path).resolve() != inputs['database'] or screen._edit_check.isChecked():
            raise RuntimeError('Native Source changed the path or enabled database editing')
    home()
    with Rec(capture, SCENES[15], settle):
        enter('running jobs')
        click(card('__easy_command__:Running'))
        wait(lambda: not page._navigator_aside_split.is_collapsed('Widgets')
             and not page._navigator_aside._navigator_folders['Running'].shut, 'Running panel did not open')
    home()
    with Rec(capture, SCENES[16], settle, desktop=True):
        hook = modal_cancel(lambda dialog: dialog.objectName() == 'NavigatorAiProviderDialog',
                            lambda dialog: settle(2))
        click(page.path_edit.ai)
        hook.finish(capture, timeout)
        if page._ai_provider or page.path_edit.ai.isChecked() or page._ai_thread is not None:
            raise RuntimeError('Cancelling AI consent changed provider state or started a request')
    with Rec(capture, SCENES[17], settle):
        enter('Toxoplasma')
        home()
        if page._last_result is not None or page._settings_overrides:
            raise RuntimeError('Explicit Home retained stale input evidence or settings')
    unchanged = {path: digest(path) == sha for path, sha in inputs['original'].items()}
    if not all(unchanged.values()) or [handle for handle in registry().active() if getattr(handle, 'user_visible', True)]:
        raise RuntimeError('The Navigator tour modified source inputs or left an analysis job active')
    write_json(captures / 'navigator-workflow.json', dict(accepted=True, scenes=list(SCENES),
               checks=checks, source_preservation=unchanged, analysis_executed=False,
               ai_request_sent=False, alpha_enabled=False, native_drop_review_sha256=files_points_sha256))
