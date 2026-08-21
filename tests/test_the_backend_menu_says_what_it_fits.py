"""An unavailable backend says what it fits and whether it is installed.

Reported 2026-08-21: with `regression_type` left to be chosen from the
response, every optional backend read

    torch (GPU) API -- unavailable: needs an explicit regression type
    pymer4 / lme4 (CPU) API -- unavailable: needs an explicit regression type
    ... seven identical lines

which says what is MISSING and nothing about what any of them does, or
whether it is even on the machine.

    "write the explisit regression type and what needs to be done if it is
     not installed. if it is intalled write installed."

BOTH FACTS BELONG THERE BECAUSE THEY ARE ANSWERED DIFFERENTLY. The types say
which choice would make the row selectable; the install state says whether
making that choice would be enough.
"""
from __future__ import annotations

import pytest

from spacr.regression_backends import backend_menu, package_installed


def _rows(regression_type=None):
    return {r["name"]: r for r in backend_menu(regression_type)}


class TestItNamesTheTypes:

    def test_no_row_is_the_old_bare_message(self):
        for row in backend_menu(None).__iter__():
            assert row["short_reason"] != "needs an explicit regression type"

    @pytest.mark.parametrize("name,expected", [
        ("pyfixest", "ols"),
        ("glum", "glm"),
        ("cuml", "lasso"),
    ])
    def test_each_names_a_family_it_can_fit(self, name, expected):
        row = _rows(None)[name]
        assert not row["enabled"]
        assert expected in row["short_reason"]
        assert "fits" in row["short_reason"]

    def test_the_long_reason_says_how_to_select_it(self):
        row = _rows(None)["pyfixest"]
        assert "regression_type" in row["reason"]


class TestItNamesTheInstallState:

    def test_an_installed_backend_says_installed(self):
        row = _rows(None)["pyfixest"]
        if not package_installed("pyfixest"):
            pytest.skip("pyfixest is not installed in this environment")
        assert "installed" in row["short_reason"]
        assert "not installed" not in row["short_reason"]

    def test_a_missing_backend_says_how_to_get_it(self):
        row = _rows(None)["gpytorch"]
        if package_installed("gpytorch"):
            pytest.skip("gpytorch is installed in this environment")
        assert "install" in row["short_reason"]

    def test_every_short_reason_fits_the_dropdown_entry(self):
        """`short_reason` is appended to the entry's own text, so it has a
        hard budget. Asserted here as well as in the backend suite because
        this file is what added the types and the install state to it."""
        for regression_type in ("mixed", "ols", "lasso", None):
            for row in backend_menu(regression_type):
                if row["enabled"]:
                    continue
                assert len(row["short_reason"]) <= 80, (
                    regression_type, row["name"], row["short_reason"])

    def test_pymer4_still_names_r(self):
        """pip alone cannot make it work -- R is a system package."""
        row = _rows(None)["pymer4"]
        if package_installed("pymer4"):
            pytest.skip("pymer4 is installed in this environment")
        assert "R" in row["reason"] or "R" in row["short_reason"]

    def test_not_wired_up_is_said_alongside_not_installed(self):
        """Installing the package alone would not make it choosable, and
        wiring it up alone would not either -- so both facts are carried.

        THE SHORT FORM SPELLS "NOT INSTALLED" AS THE PIP COMMAND. It has to
        fit inside a dropdown entry and there is an 80-character test
        holding it there; the first version of this message ran to 94 and
        broke it. The long `reason` has room to say both in words."""
        row = _rows(None)["numpyro"]
        if package_installed("numpyro"):
            pytest.skip("numpyro is installed in this environment")

        short = row["short_reason"]
        assert "not wired up" in short
        # HOW TO GET IT, in whatever spelling -- the exact wording of this
        # message is being tuned, so this asserts the SUBSTANCE. A test that
        # pins prose turns every improvement into a failure.
        assert "install" in short
        assert len(short) <= 80

        assert "not installed" in row["reason"]
        assert "does not route any fit" in row["reason"]


class TestTheCapabilityMessageIsUnchanged:
    """With an explicit type, the reason is about the CAPABILITY, which is a
    different and better message -- it must not have been replaced."""

    def test_a_backend_with_no_mixed_model_says_so(self):
        row = _rows("mixed")["pyfixest"]
        assert not row["enabled"]
        assert "cannot fit" in row["reason"]
        assert "ols" in row["reason"]

    def test_the_two_that_can_fit_mixed_are_enabled_or_explained(self):
        rows = _rows("mixed")
        assert rows["statsmodels"]["enabled"]
        # torch can fit mixed; whether it is enabled depends on CUDA, and if
        # it is refused the reason must be about the DEVICE, not the family.
        torch_row = rows["torch"]
        if not torch_row["enabled"]:
            assert "CUDA" in torch_row["reason"]

    def test_no_row_claims_a_family_it_cannot_fit(self):
        for regression_type in ("mixed", "ols", "lasso"):
            for row in backend_menu(regression_type):
                if row["enabled"]:
                    continue
                assert "fits" in row["reason"] or "not installed" in \
                    row["reason"] or "wired" in row["reason"] or \
                    "CUDA" in row["reason"], (regression_type, row["name"])
