"""Check exported gate membership without confusing count with identity."""


def check_exported_rows(expected, exported):
    """Require each marked plate/row/column/field/object/type key exactly once."""
    expected, exported = list(expected), list(exported)
    if len(exported) != len(set(exported)):
        raise ValueError('Exported gate contains duplicate object identities')
    if set(exported) != set(expected):
        raise ValueError('Exported gate object identities or types differ from the drawn population')
    return len(exported)
