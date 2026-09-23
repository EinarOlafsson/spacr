"""Formatting-only compatibility for documentation on older release branches.

The main 1.5.0.9 branch has an extra leading space in the ambient timing table.
Nightly already fixes its source. Apply that exact whitespace correction while
rendering main; never copy newer API prose or application code into a release.
"""

import re


def normalize_ambient_table(text: str) -> str:
    """Correct the known table without changing any words, values or other text."""
    border = "=========  =====================  ====================="
    pattern = re.compile(
        r"(?m)^(?P<indent> *)" + re.escape(border) + r"\n"
        r"(?P=indent) theme      shading \(moved\)        soften \+ blit \(stays\)\n"
        r"(?P=indent)" + re.escape(border) + r"\n"
        r"(?P<rows>(?:(?P=indent) (?:blobs|aurora|ripple|bokeh|cells|drift|resonance) +[^\n]+\n){7})"
        r"(?P=indent)" + re.escape(border) + r"$"
    )

    def replace(match):
        indent = match["indent"]
        lines = match[0].splitlines()
        return "\n".join(
            line if line.strip() == border else indent + line[len(indent) + 1:]
            for line in lines
        )

    return pattern.sub(replace, text)


def source_read(app, docname, source):
    if docname == "api/spacr/qt/widgets/ambient/index":
        source[0] = normalize_ambient_table(source[0])


def setup(app):
    app.connect("source-read", source_read)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
