"""How ``spacr.regex_infer`` turns a filename family into slots and a pattern.

``_proposal_for`` decides what varies and builds the regex the importer runs;
``_merge_wells`` folds an adjacent letter slot and digit slot into one well
only when every joined value reads as a well; ``_assign_roles`` never hands a
role out twice.
"""

from __future__ import annotations

from spacr import regex_infer
from spacr.regex_infer import FieldEvidence, _assign_roles, _merge_wells, _proposal_for


def _summary(proposal):
    return (proposal.pattern,
            {name: (slot.values, slot.role, slot.numeric, slot.before,
                    slot.fixed_tail, slot.because)
             for name, slot in proposal.fields.items()},
            proposal.matched, proposal.total, proposal.unmatched,
            proposal.suffix)


def test_a_filename_family_becomes_an_exact_proposal_or_none(subtests,
                                                             monkeypatch):
    """Each family gives its exact pattern, slot evidence and match counts."""
    cases = [
        ("no names", [], [], None),
        ("nothing varies", ["A01.tif", "A01.tif"], ["A01.tif"], None),
        ("literal between two numeric slots", ["1x2.tif", "3x4.tif"],
         ["1x2.tif", "3x4.tif", "zz.png"],
         (r"(?P<chanID>\d+)x(?P<fieldID>\d+)\.tif",
          {"chanID": (("1", "3"), "chanID", True, "", "",
                      "only 2 distinct values"),
           "fieldID": (("2", "4"), "fieldID", True, "x", "",
                       "2 distinct numbers")},
          2, 3, ("zz.png",), "tif")),
        ("constant number after letters that are not a well",
         ["ab12.tif", "cd12.tif"], ["ab12.tif", "cd12.tif"],
         (r"(?P<group0>[A-Za-z]+)12\.tif",
          {"group0": (("ab", "cd"), "", False, "", "", "")},
          2, 2, (), "tif")),
        ("number against a word is two fields", ["1a.tif", "2b.tif"],
         ["1a.tif", "2b.tif"],
         (r"(?P<chanID>\d+)(?P<group1>[A-Za-z]+)\.tif",
          {"chanID": (("1", "2"), "chanID", True, "", "",
                      "only 2 distinct values"),
           "group1": (("a", "b"), "", False, "", "", "")},
          2, 2, (), "tif")),
        ("letter and varying digits merge to a well", ["A01.tif", "B02.tif"],
         ["A01.tif", "B02.tif"],
         (r"(?P<wellID>[A-Za-z0-9]+)\.tif",
          {"wellID": (("A01", "B02"), "wellID", False, "", "",
                      "a letter and digits that read as a well")},
          2, 2, (), "tif")),
        ("constant well number is absorbed as a fixed tail",
         ["A01.tif", "B01.tif"], ["A01.tif", "B01.tif"],
         (r"(?P<wellID>[A-Za-z]+01)\.tif",
          {"wellID": (("A01", "B01"), "wellID", False, "", "01",
                      "a letter and a well number that never changes")},
          2, 2, (), "tif")),
    ]
    for case_id, names, all_names, expected in cases:
        with subtests.test(msg=case_id):
            proposal = _proposal_for(names, all_names)
            if expected is None:
                assert proposal is None
            else:
                assert _summary(proposal) == expected

    with subtests.test(msg="a slot left empty after factoring becomes literal"):
        monkeypatch.setattr(regex_infer, "_factor",
                            lambda values: ("head", "tail", ("", "")))
        assert _proposal_for(["ab.tif", "cd.tif"], ["ab.tif", "cd.tif"]) is None


def test_slot_evidence_is_merged_into_wells_and_given_unique_roles(subtests):
    """Only joins that read as wells merge; one constant slot is the plate."""
    merges = [
        ("two letters and three digits stay apart",
         [FieldEvidence(index=0, values=("ab", "cd"), numeric=False),
          FieldEvidence(index=1, values=("123", "456"), numeric=True)],
         [FieldEvidence(index=0, values=("ab", "cd"), numeric=False),
          FieldEvidence(index=1, values=("123", "456"), numeric=True)]),
        ("a letter and two digits fold into one well",
         [FieldEvidence(index=0, values=("A", "B"), numeric=False),
          FieldEvidence(index=1, values=("01", "02"), numeric=True)],
         [FieldEvidence(index=0, values=("A01", "B02"), numeric=False,
                        role="wellID",
                        because="a letter and digits that read as a well")]),
    ]
    for case_id, slots, expected in merges:
        with subtests.test(msg=case_id):
            pieces = [{"slot": slot} for slot in slots]
            _merge_wells(pieces)
            assert [piece["slot"] for piece in pieces] == expected

    with subtests.test(msg="a role is never handed out twice"):
        first = FieldEvidence(index=0, values=("exp", "exp"), numeric=False)
        second = FieldEvidence(index=1, values=("run", "run"), numeric=False)
        _assign_roles([first, second])
        assert (first.role, first.because) == ("plateID",
                                               "the same in every file")
        assert (second.role, second.because) == ("", "")
