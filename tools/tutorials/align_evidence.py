"""Validate identities as well as geometry in the known-crop tutorial export."""


def check_coordinate_rows(tiles, rows, stack):
    """Require one exact source/field/origin record per emulated tile."""
    expected = {tile['path']: tile for tile in tiles}
    actual = {row['source']: row for row in rows}
    if len(actual) != len(rows):
        raise ValueError('Duplicate exported tile source')
    if set(actual) != set(expected):
        raise ValueError('Exported source files differ from the real tile inputs')
    for path, tile in expected.items():
        row = actual[path]
        identity = (row['plateID'], row['rowID'], row['columnID'], row['fieldID'])
        if identity != ('tutorial', 'r1', 'c1', f"f{tile['field']}"):
            raise ValueError('Exported tile identities differ from the explicit tutorial naming')
        if (row['canvas_y'], row['canvas_x']) != tuple(tile['source_yx']):
            raise ValueError('Exported canvas coordinates differ from the known source crop')
        if row['stack_path'] != str(stack):
            raise ValueError('Exported coordinates reference a different output stack')
    return len(rows)
