"""The arcs in `regression_annotation` the rest of the suite never drove.

Measured for instruction 352 rather than guessed: `coverage run` over all
ten test files that import the module put it at 99.06 %, four statements
and three branches short. Three of the four are behaviours a reader would
name -- a strategy declared but not built, an entry passed where a key is
expected, and a comparison whose denominator is zero -- and they are here.

THE MEASUREMENT IS AGAINST THE SUITE, NOT AGAINST ONE FILE. Asked of the
five files whose names carry the module's name, `guide_attribution` looked
twenty-five items worse than it was; the same trap applies here, and the
ten files these numbers come from include several whose names say nothing
about annotation.

THE FOURTH IS NOT HERE AND SHOULD NOT BE. `_run_self_training` ends with

    if best is None:
        raise NotEnoughLabels("No self-training round could be fitted.")

and `best` cannot be None there. The loop is
`range(max(1, int(request.rounds)))`, so it runs at least once whatever
the request asks for, and the first pass takes the `best is None` arm and
assigns it. The only way past the assignment is `_fit_report` raising,
which propagates rather than reaching this line.

It is a guard over an invariant the `max(1, ...)` establishes. Reaching it
would mean removing that floor -- letting a request for zero rounds run
zero rounds -- which would trade an unreachable line for a real way to get
no fit at all. The line stays; this note is what it is worth instead of a
test. 99.57 %, and the rest of the module is behaviour.
"""

# --- four arcs the whole suite still never drove ---------------------------

class TestAStrategyThatIsDeclaredButNotBuilt:
    """`Strategy.describe()` says so, and the menu is where a user reads it.

    Every shipped strategy is implemented, so this sentence has never been
    produced by the real menu -- but the field exists precisely so a
    strategy can be OFFERED for discussion before it is built, and a menu
    entry that reads like a working feature is the failure it prevents.
    """

    def test_the_sentence_is_appended(self):
        from spacr.regression_annotation import Strategy

        entry = Strategy(key="k", title="T", purpose="P.", cost="C.",
                         implemented=False)
        said = entry.describe()
        assert said.startswith("T — P. Limitations: C.")
        assert "declared but is not yet implemented" in said

    def test_an_implemented_one_does_not_claim_to_be_unbuilt(self):
        from spacr.regression_annotation import Strategy

        entry = Strategy(key="k", title="T", purpose="P.", cost="C.")
        assert "not yet implemented" not in entry.describe()


class TestResolvingAStrategyThatIsAlreadyOne:
    """`strategy()` takes a key OR an entry, and hands the entry back.

    The caller that matters is one passing a resolved entry through a
    second layer: making it look the key up again would fail for a
    strategy that is not in `STRATEGIES` at all, which is exactly the
    case a caller holding the object is in.
    """

    def test_an_entry_is_returned_unchanged(self):
        from spacr.regression_annotation import STRATEGIES, strategy

        entry = STRATEGIES[0]
        assert strategy(entry) is entry

    def test_even_one_the_menu_does_not_carry(self):
        from spacr.regression_annotation import Strategy, strategy

        mine = Strategy(key="not_on_the_menu", title="T", purpose="P.",
                        cost="C.")
        assert strategy(mine) is mine


class TestAnInclusiveFitAtChance:
    """`LeakageReport.summary()` when lift cannot be divided by zero.

    `survival` is the fraction of the inclusive fit's lift over chance
    that survives removing the score's own inputs. When the INCLUSIVE fit
    is itself at chance there is no lift to retain and the fraction is
    undefined -- so the report says that in words rather than printing a
    ratio nobody can interpret or, worse, dividing by zero.
    """

    @staticmethod
    def _fit(auc):
        from spacr.regression_annotation import FitReport

        return FitReport(model="m", features=("a",), n_train=10, n_test=10,
                         accuracy=0.5, balanced_accuracy=0.5, roc_auc=auc,
                         positive_share_train=0.5, positive_share_test=0.5,
                         label_source="test", split_summary="10/10")

    def test_it_says_so_instead_of_printing_a_ratio(self):
        from spacr.regression_annotation import LeakageReport

        report = LeakageReport(mode="strict", dropped=("score",),
                               with_score_inputs=self._fit(0.5),
                               without_score_inputs=self._fit(0.6),
                               survival=None)
        said = report.summary()
        assert "at chance, so retained lift cannot be calculated" in said
        assert "% of the fit's lift" not in said

    def test_a_real_survival_is_reported_as_a_percentage(self):
        """The control: the sentence is the exception, not the rule."""
        from spacr.regression_annotation import LeakageReport

        report = LeakageReport(mode="strict", dropped=("score",),
                               with_score_inputs=self._fit(0.9),
                               without_score_inputs=self._fit(0.7),
                               survival=0.5)
        said = report.summary()
        assert "50% of the fit's lift over chance survives" in said
        assert "cannot be calculated" not in said

    def test_one_fit_alone_says_neither(self):
        """Both fits are needed before the two can be compared at all."""
        from spacr.regression_annotation import LeakageReport

        report = LeakageReport(mode="strict", dropped=(),
                               with_score_inputs=self._fit(0.5),
                               without_score_inputs=None, survival=None)
        said = report.summary()
        assert "cannot be calculated" not in said
        assert "Including score inputs:" in said
