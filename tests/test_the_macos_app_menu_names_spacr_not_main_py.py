"""The macOS application menu says "Quit spaCR", not "Quit __main__.py".

2026-10-19 macOS acceptance (installed 1.5.1.5, Intel Mac): Qt names the
application menu's About, Hide and Quit items from argv[0], which under
``python -m spacr.qt`` is ``.../spacr/qt/__main__.py``; the menu read "About
__main__.py", "Hide __main__.py", "Quit __main__.py". With the application name
as argv[0] the same native menu reads "About spaCR", "Hide spaCR", "Quit spaCR".
"""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "spacr" / "qt" / "app.py"


def test_launch_hands_qt_the_application_name_as_argv0_on_macos():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    launch = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "launch")
    calls = [node for node in ast.walk(launch) if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "QApplication"]
    assert len(calls) == 1
    source = ast.unparse(calls[0].args[0])
    assert source == "[_app_name] if sys.platform == 'darwin' else sys.argv[:1]"
