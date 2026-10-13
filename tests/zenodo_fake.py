"""A local stand-in for the Zenodo deposition API, for tests only.

It serves, on 127.0.0.1, the endpoints spaCR's Zenodo deposit uses:
``POST /api/deposit/depositions``, ``PUT /api/files/<bucket>/<name>``,
``PUT /api/deposit/depositions/<id>``,
``POST /api/deposit/depositions/<id>/actions/publish`` and
``GET /api/deposit/depositions/<id>/files``. Like Zenodo it wants
``Authorization: Bearer <token>``, answers 401 without it, checks the
required metadata on update (400 when missing) and refuses to publish a
deposition without files. Nothing leaves this machine.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeZenodo:
    """The fake server: ``url`` is its root, ``depositions`` what it holds."""

    def __init__(self, token: str = "test-token"):
        self.token = token
        self.depositions: dict = {}
        self.requests: list = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.api = f"{self.url}/api"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5)

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return None

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _route(self, method):
                body = self._body()
                fake.requests.append({"method": method, "path": self.path,
                                      "auth": self.headers.get("Authorization")})
                if self.headers.get("Authorization") != f"Bearer {fake.token}":
                    return self._send(401, {"status": 401, "message":
                                            "The server could not verify "
                                            "that you are authorized."})
                path = self.path
                if method == "POST" and path == "/api/deposit/depositions":
                    ident = len(fake.depositions) + 1
                    dep = {"id": ident, "files": {}, "metadata": {
                        "prereserve_doi": {"doi": f"10.5072/zenodo.{ident}",
                                           "recid": ident}},
                        "submitted": False, "bucket": f"b{ident}"}
                    fake.depositions[ident] = dep
                    return self._send(201, self._view(dep))
                match = re.fullmatch(r"/api/files/b(\d+)/(.+)", path)
                if method == "PUT" and match:
                    dep = fake.depositions.get(int(match.group(1)))
                    if dep is None:
                        return self._send(404, {"message": "No bucket."})
                    from urllib.parse import unquote
                    name = unquote(match.group(2))
                    dep["files"][name] = body
                    return self._send(201, {
                        "key": name, "size": len(body),
                        "checksum": "md5:" + hashlib.md5(body).hexdigest()})
                match = re.fullmatch(r"/api/deposit/depositions/(\d+)(/.*)?",
                                     path)
                dep = fake.depositions.get(int(match.group(1))) if match else None
                if dep is None:
                    return self._send(404, {"message": "Not found."})
                rest = match.group(2) or ""
                if method == "PUT" and not rest:
                    meta = json.loads(body or b"{}").get("metadata", {})
                    missing = [k for k in ("upload_type", "title", "creators",
                                           "description") if not meta.get(k)]
                    if missing:
                        return self._send(400, {"status": 400, "message":
                                                "Validation error.",
                                                "errors": missing})
                    dep["metadata"].update(meta)
                    return self._send(200, self._view(dep))
                if method == "POST" and rest == "/actions/publish":
                    if not dep["files"] or "title" not in dep["metadata"]:
                        return self._send(400, {"message": "Validation error."})
                    dep["submitted"] = True
                    dep["doi"] = dep["metadata"]["prereserve_doi"]["doi"]
                    return self._send(202, self._view(dep))
                if method == "GET" and rest == "/files":
                    return self._send(200, [
                        {"filename": n, "filesize": len(b),
                         "checksum": hashlib.md5(b).hexdigest()}
                        for n, b in dep["files"].items()])
                return self._send(405, {"message": "Method not allowed."})

            def _view(self, dep):
                view = {k: v for k, v in dep.items()
                        if k not in ("files", "bucket")}
                view["links"] = {
                    "bucket": f"{fake.api}/files/{dep['bucket']}",
                    "html": f"{fake.url}/deposit/{dep['id']}"}
                if dep["submitted"]:
                    view["links"]["record_html"] = f"{fake.url}/records/{dep['id']}"
                return view

            def do_POST(self):
                self._route("POST")

            def do_PUT(self):
                self._route("PUT")

            def do_GET(self):
                self._route("GET")

        return Handler
