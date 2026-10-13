"""Filename grouping for merged timelapse arrays, without analysis imports."""

import os


def parse_merged_filename(fname):
    """Parse a merged filename for grouping and numeric timepoint sorting.

    These are legacy display/grouping fields, not canonical database keys.
    Row IDs are letters, column/time IDs are integers, and ``prcf`` uses
    the well rather than separate row and column keys. Use
    :func:`spacr.schema.parse_field_stem` for database identities.

    :param fname: path or filename in ``plate_well_field_time.npy`` form.
    :returns: plateID, wellID, rowID, columnID, fieldID, timeID, prcf, prcft
        and filename. Missing field/time values default to ``1``/``0``.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    plateID = parts[0] if len(parts) > 0 else ""
    wellID = parts[1] if len(parts) > 1 else ""
    fieldID = parts[2] if len(parts) > 2 else "1"
    time_str = parts[3] if len(parts) > 3 else "0"
    digits = "".join(ch for ch in time_str if ch.isdigit())
    timeID = int(digits) if digits else 0
    rowID = wellID[0] if wellID else ""
    col_part = "".join(ch for ch in wellID[1:] if ch.isdigit())
    columnID = int(col_part) if col_part else 0
    prcf = f"{plateID}_{wellID}_{fieldID}"
    return dict(
        plateID=plateID, wellID=wellID, rowID=rowID, columnID=columnID,
        fieldID=fieldID, timeID=timeID, prcf=prcf, prcft=f"{prcf}_{timeID}",
        filename=os.path.basename(fname),
    )
