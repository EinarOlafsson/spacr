"""Reading a barcode reference from a mapping, a FASTA, or a CSV.

``_read_reference`` accepts all three because a screen's barcodes arrive in all
three -- a dict from a settings panel, a FASTA from a vendor, a CSV the user
made. Every path upper-cases the sequence, which is what makes a lower-case
FASTA compare equal to an upper-case CSV; without it the same library read two
ways would collide with nothing.

The uncovered arc is a FASTA that ends without a final record to flush, which
is what an empty or header-only file is.
"""
from __future__ import annotations

import pytest


def test_a_mapping_is_taken_as_written_but_upper_cased():
    """The first route, which a settings panel supplies directly."""
    from spacr.sequencing_qc import _read_reference

    assert _read_reference({"g1": "acgt", "g2": "TTTT"}) == {
        "g1": "ACGT", "g2": "TTTT"}


def test_a_fasta_is_read_record_by_record(tmp_path):
    """The flush at the top of each header AND the final flush after the loop.

    Both matter: without the first, every record but the last is lost; without
    the second, only the last is.
    """
    from spacr.sequencing_qc import _read_reference

    fasta = tmp_path / "library.fasta"
    fasta.write_text(">g1 some description\nacgt\nAACC\n>g2\ntttt\n")

    assert _read_reference(str(fasta)) == {"g1": "ACGTAACC", "g2": "TTTT"}


def test_a_fasta_name_is_its_first_word():
    """The ``.split()[0]``: a description after the name is not part of it."""
    from spacr.sequencing_qc import _read_reference
    import tempfile, os

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "library.fa")
        with open(path, "w") as handle:
            handle.write(">g1 TGGT1_231640 guide 1\nACGT\n")
        assert list(_read_reference(path)) == ["g1"]


@pytest.mark.parametrize("content", ["", "\n\n\n", "   \n"])
def test_a_fasta_with_no_records_reads_as_empty(tmp_path, content):
    """Arc 542 -> 544: ``name`` is still None, so nothing is flushed.

    An empty reference file is what a failed download or an interrupted export
    leaves. Flushing a None name would key the table on the string 'None' and
    the caller would compare every barcode against one nonsense entry.
    """
    from spacr.sequencing_qc import _read_reference

    fasta = tmp_path / "empty.fasta"
    fasta.write_text(content)

    assert _read_reference(str(fasta)) == {}


def test_a_fasta_with_a_header_and_no_sequence_still_records_the_name(tmp_path):
    """The final flush taken with empty chunks, which is a truncated record.

    An empty sequence is recorded rather than dropped, so the caller can see
    the name is present and the sequence is not -- which is the difference
    between a truncated file and a missing guide.
    """
    from spacr.sequencing_qc import _read_reference

    fasta = tmp_path / "truncated.fasta"
    fasta.write_text(">g1\n")

    assert _read_reference(str(fasta)) == {"g1": ""}
