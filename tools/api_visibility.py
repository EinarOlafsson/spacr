"""Explicit public documentation for services in selected private modules.

The service module stays private for Python packaging, while its documented
entry points are deliberately exposed in Sphinx, localization and Help search.
This policy does not expose private helpers within those modules.
"""

EXPLICIT_MODULES = frozenset({"spacr._starplast"})


def public_symbol(name: str) -> bool:
    """Accept ordinary public names and public members of explicit modules."""
    for module in EXPLICIT_MODULES:
        if name == module:
            return True
        if name.startswith(module + "."):
            name = name[len(module) + 1:]
            break
    return all(part == "__main__" or not part.startswith("_")
               for part in name.split("."))


def explicit_page_policy(name: str):
    """Unskip only an explicit module; retain member and import filtering."""
    if name in EXPLICIT_MODULES:
        return False
    return None
