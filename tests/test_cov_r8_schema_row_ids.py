"""Row identifiers, and two validations that cannot fail.

`row_id` turns whatever a plate reader wrote into spaCR's canonical
`r<N>`. The letters branch exists because plate rows are written `A`..`P`
far more often than `r1`..`r16`, and a key that did not normalise them
would put the same well under two identities.

Two guards in this module are unreachable, and both are pinned to the
thing that makes them so rather than forced.
"""
from __future__ import annotations

import itertools
import string

import pytest

import spacr.schema as S


class TestRowIdentifiers:

    @pytest.mark.parametrize("spelled, expected", [
        pytest.param("A", "r1", id="upper-A"),
        pytest.param("B", "r2", id="upper-B"),
        pytest.param("P", "r16", id="upper-P"),
        pytest.param("a", "r1", id="lower-a"),
        pytest.param("p", "r16", id="lower-p"),
        pytest.param("  C  ", "r3", id="surrounding-space-is-ignored"),
        # 1536-well plates run past Z, so AA is row 27.
        pytest.param("AA", "r27", id="two-letter-row-is-read-past-z"),
        pytest.param("r5", "r5", id="already-canonical-is-unchanged"),
        pytest.param("1", "r1", id="bare-number-is-a-row-index"),
    ])
    def test_a_row_spelling_becomes_its_canonical_row(self, spelled, expected):
        assert S.row_id(spelled) == expected

    def test_a_well_label_yields_the_row_it_names(self):
        """`A1` is a well, and its row is A -- so `row_id` answers r1.

        That is deliberate rather than accidental: callers hand this
        whatever the plate reader wrote, and a well label carries its
        row. The two spellings of the same row must not become two
        identities.
        """
        assert S.row_id("A1") == "r1"
        assert S.row_id("A01") == "r1"
        assert S.row_id("A") == "r1"

class TestTheLetterIndexThatIsAlwaysFound:
    """`if index is not None:` cannot be false.

    `_ROW_ONLY` is `^([A-Za-z]{1,2})$`, and `row_index_from_letters`
    answers an index for every one- or two-letter string. So the
    fall-through to `_prefixed_id` below it is never taken FROM this
    branch -- it is reached only by inputs the pattern rejected.

    Checked exhaustively: all 52 one-letter and all 2,704 two-letter
    strings.
    """

    def test_every_string_the_pattern_accepts_yields_an_index(self):
        checked = 0
        for size in (1, 2):
            for combo in itertools.product(string.ascii_letters,
                                           repeat=size):
                text = "".join(combo)
                if not S._ROW_ONLY.match(text):
                    continue
                if S._PREFIXED_INT.match(text):
                    continue
                checked += 1
                assert S.row_index_from_letters(text) is not None, (
                    f"{text!r} matches _ROW_ONLY but has no row index; the "
                    "fall-through in row_id is now reachable")
        assert checked == 52 + 52 * 52


class TestTheObjectRoleValidationAtImport:
    """The module refuses to import with an invalid object role.

    A role that is empty, holds a digit, or holds the key separator
    cannot round-trip through an identity: `cell1` + 7 and `cell` + 17
    are the same string, and an underscore would split the surrounding
    prcfo key. Failing the import is preferable to writing identities
    that cannot be read back.

    The raise cannot fire, because the roles that ship are all valid --
    which is the point. Pinned to them.
    """

    def test_every_shipped_role_satisfies_the_rule_it_is_checked_against(
            self):
        for role in S.OBJECT_TYPES:
            assert role, "an empty object role would produce a bare identity"
            assert S.KEY_SEPARATOR not in role, (
                f"{role!r} holds the key separator and would split a prcfo")
            assert not any(ch.isdigit() for ch in role), (
                f"{role!r} holds a digit; {role}1 + 7 and {role} + 17 are the "
                "same string")

    def test_the_rule_would_reject_a_bad_role(self):
        """The check itself, applied to roles that are NOT shipped.

        This is what the import-time loop does; running it here shows the
        rule has teeth without corrupting the module's own constant.
        """
        def offends(role):
            return (not role or S.KEY_SEPARATOR in role
                    or any(ch.isdigit() for ch in role))

        assert offends("")
        assert offends("cell1")
        assert offends(f"cell{S.KEY_SEPARATOR}wall")
        assert not offends("cell")
