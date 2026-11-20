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


@pytest.mark.parametrize('filename, content, expected', [
    pytest.param(
        'library.fasta', '>g1 some description\nacgt\nAACC\n>g2\ntttt\n', {'g1': 'ACGTAACC', 'g2': 'TTTT'},
        id='test_a_fasta_is_read_record_by_record',
    ),
    pytest.param(
        'truncated.fasta', '>g1\n', {'g1': ''},
        id='test_a_fasta_with_a_header_and_no_sequence_still_records_the_name',
    ),
])
def test_fasta_files_keep_their_original_barcode_references(tmp_path, filename, content, expected):
    """Keep each original FASTA file paired with the same barcode reference."""
    from spacr.sequencing_qc import _read_reference

    fasta = tmp_path / filename
    fasta.write_text(content, encoding="utf-8")

    assert _read_reference(str(fasta)) == expected


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
    fasta.write_text(content, encoding="utf-8")

    assert _read_reference(str(fasta)) == {}
