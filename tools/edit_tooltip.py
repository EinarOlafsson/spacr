"""Replace one tooltip in spacr/settings.py, safely.

Three failures on 2026-08-09 produced this, and each is now impossible
rather than merely discouraged:

* a line-based regex "fixed" 203 files because `"` is a legal rST underline
  character and it matched closing triple-quotes;
* a replacement containing an unescaped `"` inside a double-quoted string
  broke the module, because the script called write_text() BEFORE
  ast.parse() -- the gate existed but ran too late to help;
* a pattern anchored on `"key":` silently matched nothing for the half of
  the table that is written `'key':`, and reported success.

So: the new text is passed as a Python VALUE and this module does the
quoting with repr(), the file is parsed BEFORE it is written, and both key
quotings are accepted.
"""
import ast
import re
import sys
from pathlib import Path

SETTINGS = Path(__file__).resolve().parent.parent / "spacr" / "settings.py"


def replace(key: str, text: str, path: Path = SETTINGS) -> bool:
    """Point ``key``'s tooltip at ``text``. Returns False and writes nothing
    if the key is not found or the result would not compile."""
    src = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'''(^[ \t]*['"]{re.escape(key)}['"]\s*:\s*)"(?:[^"\\]|\\.)*",[ \t]*$''',
        re.M)
    match = pattern.search(src)
    if not match:
        print(f"{key}: not found (checked both quote styles)")
        return False
    # repr() does the quoting and escaping. Hand-written quotes in the
    # replacement are what broke this file once already.
    new = src[:match.start()] + match.group(1) + repr(text) + "," + src[match.end():]
    try:
        ast.parse(new)                      # BEFORE the write, not after
    except SyntaxError as exc:
        print(f"{key}: refused, would not compile (line {exc.lineno})")
        return False
    path.write_text(new, encoding="utf-8")
    return True


if __name__ == "__main__":
    key, text = sys.argv[1], sys.argv[2]
    sys.exit(0 if replace(key, text) else 1)
