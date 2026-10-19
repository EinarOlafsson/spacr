"""Every text-mode open, read_text and write_text in spacr/ names its encoding.

Without ``encoding=`` Python uses the locale code page: cp1252 on Windows, so
a non-Latin path, name or note raises or is silently mangled. Files older spaCR
wrote on Windows are read with ``_legacy_text_encoding``, which tries UTF-8 and
falls back to cp1252. Binary opens, archive/dialog ``open`` methods and
``importlib.metadata`` reads are not text files and are not checked.
"""
from __future__ import annotations

import ast
from pathlib import Path

from spacr.logging_util import _legacy_text_encoding

ROOT = Path(__file__).resolve().parents[1]


def _binary(mode):
    return (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
            and "b" in mode.value)


def _mode(call, index):
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return keyword.value
    return call.args[index] if len(call.args) > index else None


def _unnamed(call):
    keywords = {k.arg for k in call.keywords}
    if "encoding" in keywords or None in keywords:
        return None
    function = call.func
    if isinstance(function, ast.Name):
        if function.id == "open" and not _binary(_mode(call, 1)):
            return "open"
        return None
    if not isinstance(function, ast.Attribute):
        return None
    receiver = ast.unparse(function.value)
    if function.attr in ("read_text", "write_text"):
        if "dist" in receiver or "md." in receiver:
            return None
        if function.attr == "read_text" and call.args:
            return None
        if function.attr == "write_text" and len(call.args) > 1:
            return None
        return function.attr
    if function.attr == "open" and receiver.endswith("path"):
        return None if _binary(_mode(call, 0)) else "path.open"
    return None


def test_no_text_open_in_the_package_lacks_an_encoding():
    offenders = []
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _unnamed(node):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (
        "text file access without encoding= (Windows would use cp1252):\n  "
        + "\n  ".join(offenders[:30]))


def test_a_file_older_spacr_wrote_in_cp1252_is_still_read(tmp_path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_bytes("name\nMüller\n".encode("cp1252"))
    modern = tmp_path / "modern.csv"
    modern.write_bytes("name\nMüller 日本\n".encode("utf-8"))
    assert _legacy_text_encoding(legacy) == "cp1252"
    assert _legacy_text_encoding(modern) == "utf-8"
    assert open(legacy, encoding=_legacy_text_encoding(legacy)).read() == "name\nMüller\n"
    assert _legacy_text_encoding(tmp_path / "missing.csv") == "utf-8"
