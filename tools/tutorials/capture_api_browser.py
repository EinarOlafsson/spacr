"""Capture real Help-to-browser navigation in an isolated Chromium profile.

The application's opener stays unchanged. Python's browser preference selects
the installed browser for this recording process; DevTools verifies the actual
URL and symbol anchor before the desktop frame is accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


class CaptureApiBrowser:
    def __init__(self, app, executable, profile):
        self.app = app
        self.profile = Path(profile).resolve()
        if self.profile.exists():
            raise ValueError('Use a fresh private browser profile for this capture')
        if not Path(executable).is_file():
            raise ValueError('The capture browser executable does not exist')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        command = [str(executable), '--no-sandbox', '--disable-gpu', '--test-type',
                   '--force-device-scale-factor=2',
                   '--no-first-run', '--no-default-browser-check',
                   f'--user-data-dir={self.profile}', f'--remote-debugging-port={self.port}',
                   '--remote-debugging-address=127.0.0.1', '--window-size=1920,1080',
                   '--window-position=0,0',
                   '--start-fullscreen', '%s']
        webbrowser.register('spacr-capture-browser', None,
                            webbrowser.BackgroundBrowser(command), preferred=True)

    def pages(self):
        try:
            with urlopen(f'http://127.0.0.1:{self.port}/json/list', timeout=.3) as response:
                return json.load(response)
        except (URLError, TimeoutError, OSError):
            return []

    def find(self, url):
        matches = [page for page in self.pages()
                   if page.get('type') == 'page' and page.get('url') == url]
        if len(matches) > 1:
            raise ValueError('More than one browser tab matches the requested API URL')
        return matches[0] if matches else None

    def command(self, page, method, params=None, timeout=10):
        from PySide6.QtCore import QUrl
        from PySide6.QtNetwork import QAbstractSocket
        from PySide6.QtWebSockets import QWebSocket

        connection = QWebSocket()
        replies = []
        connection.textMessageReceived.connect(lambda text: replies.append(json.loads(text)))
        connection.open(QUrl(page['webSocketDebuggerUrl']))
        deadline = time.monotonic() + timeout
        try:
            while connection.state() != QAbstractSocket.ConnectedState:
                if time.monotonic() >= deadline:
                    raise TimeoutError('The recording browser did not connect to DevTools')
                self.app.processEvents()
                time.sleep(.02)
            connection.sendTextMessage(json.dumps({'id': 1, 'method': method, 'params': params or {}}))
            while not any(reply.get('id') == 1 for reply in replies):
                if time.monotonic() >= deadline:
                    raise TimeoutError('The recording browser did not answer ' + method)
                self.app.processEvents()
                time.sleep(.02)
            reply = next(reply for reply in replies if reply.get('id') == 1)
            if 'error' in reply:
                raise ValueError(f'Browser command failed: {reply["error"]}')
            return reply['result']
        finally:
            connection.close()
            self.app.processEvents()

    def prepare_frame(self, page, symbol):
        expression = ('({ready: document.readyState, url: location.href, title: document.title, '
                      'anchor: !!document.getElementById(' + json.dumps(symbol) + ')})')
        deadline = time.monotonic() + 45
        while True:
            result = self.command(page, 'Runtime.evaluate',
                                  {'expression': expression, 'returnByValue': True})
            value = result.get('result', {}).get('value', {})
            if value.get('ready') == 'complete' and value.get('anchor') is True:
                break
            if time.monotonic() >= deadline:
                raise ValueError('The real browser page did not load the requested API anchor')
            self.app.processEvents()
            time.sleep(.1)
        if value['url'] != page['url']:
            raise ValueError('The API browser unexpectedly navigated to another page')
        window = self.command(page, 'Browser.getWindowForTarget', {'targetId': page['id']})
        self.command(page, 'Browser.setWindowBounds',
                     {'windowId': window['windowId'], 'bounds': {'windowState': 'fullscreen'}})
        self.command(page, 'Page.bringToFront')
        self.command(page, 'Runtime.evaluate', {'expression':
                     'document.getElementById(' + json.dumps(symbol) + ').scrollIntoView({block:"center"})'})
        return {'url': value['url'], 'title': value['title'], 'symbol_anchor_present': True,
                'browser_device_scale_factor': 2}

    def close(self):
        pages = self.pages()
        if pages:
            with urlopen(f'http://127.0.0.1:{self.port}/json/version', timeout=1) as response:
                browser = json.load(response)
            try:
                self.command(browser, 'Browser.close', timeout=5)
            except TimeoutError:
                if self.pages():
                    raise
