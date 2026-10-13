"""Regression guards for precise, scientific user-facing descriptions."""

from __future__ import annotations

from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]


def test_setting_tooltips_reject_reviewed_figurative_phrases():
    """Notebook-bound setting help must not regain conversational metaphors."""

    from spacr.settings import tooltips

    text = "\n".join(map(str, tooltips.values())).casefold()
    rejected = (
        "one object's opinion",
        "chattier",
        "faster and blinder",
        "lucky split",
        "flattering the model",
        "slam probabilities",
        "the honest one",
        "nobody can scroll",
        "nobody opens",
        "quietly delete the effect",
        "wrong whenever shape",
        "only say anyone had",
        "merely for existing",
        "the wrong trade",
        "keep intensity differences honest",
        "honest negative distribution",
        "the honest choice",
        "welded",
        "shattering",
        "shards",
        "signature itself",
        "baked into",
        "blowouts",
        "what every existing run does",
        "segmentation blow-up",
        "shows it to you first",
        "costs you real objects",
        "died on a message",
        "a healthy field",
        "throws away the object's edges",
        "being thrown away",
        "what you want while",
        "worth its cost",
        "silently breaks it",
        "buys safety",
        "what the model actually uses",
        "one bad field can",
        "how much thinking",
        "gets corrected away",
        "objects land near",
        "masks come back",
        "allowed to look at",
        "brings its relationships",
        "rule claimed",
        "goes stale",
        "auto-pick",
        "guessing it",
        "walks bins",
        "turn invaded back",
        "hits land",
        "confident explanation of the wrong thing",
        "pick iou",
        "lands closest",
        "land well off",
        "feed spacr",
        "spacr walks",
        "blowing out",
        "pick up fainter",
        "pick up thick",
        "being shredded",
        "glue neighbours",
        "beat those keep flags",
        "straight off the acquisition settings",
        "eyeball segmentation",
        "there is no third option",
        "nothing to read",
        "tops the cell tracking challenge leaderboard",
        "wins on densely packed",
        "all-round 2d model",
        "does not get you that model",
        "existing figures do not change under you",
        "eyeballing heatmaps",
        "costs disk and buys nothing",
        "costs real time per image",
        "switching it on can backfire",
        "cellpose cut in half",
        "split down the middle",
        "spreading across wells or plates buys coverage",
        "pushes gradient onto hard ones",
        "the model gets over-confident",
        "this is a path filter and nothing more",
        "local density is the dominant confounder",
        "about twice as fast",
        "uses about half the memory",
        "refused out loud",
        "scores move a little",
        "cannot inherit a half-written result",
        "whatever is left is no longer the experiment",
        "difference between minutes and hours",
        "feature ranking is stable long before",
        "rather than biology",
        "smallest fusion worth catching",
        "collapse of a monolayer into slabs",
        "larger than most real effects",
        "perfect two-population mixture",
        "demand cleaner separation",
        "the p value is",
        "when nothing exceeded",
        "flat row of guides",
        "memory-hungry",
        "loss spikes or flatlines",
        "training crawls",
        "squashed against the axis",
        "the column the phenotype lives in",
        "the number the whole figure shows",
        "solid blobby",
        "worth enabling",
        "polarisation in one number",
        "without the stain distracting",
        "a nan loss several minutes into training",
    )
    found = [phrase for phrase in rejected if phrase in text]
    assert not found, f"figurative setting descriptions returned: {found}"


def test_setting_tooltips_reject_reviewed_emphatic_copy():
    """Reviewed tooltip prose must not regain editorial all-caps emphasis."""

    from spacr.settings import tooltips

    text = "\n".join(map(str, tooltips.values()))
    rejected = (
        "changes the PICTURE only",
        "Computed BEFORE",
        "Show EVERY cell",
        "ON BY DEFAULT",
        "from EVERY measurement",
        "Applied AFTER",
        "is SKIPPED with a warning",
        "DEPRECATED. Expected object",
        "BEWARE which channel",
        "changes WHICH channels",
        "crops ARE; 'classes' is what they MEAN",
        "model_name is STILL read",
        "ADJUSTED P value",
        "the one the RUN uses",
        "re-labelling IGNORES",
        "plate design to NAME",
        "leave OFF",
        "INERT where",
        "entry 0 becomes BLUE",
        "Add the CORRECT Manders",
        "RAISE for less clipping",
        "segments MORE",
        "Do NOT pass False",
        "name is NOT fatal",
        "INCLUDING 'organelle'",
        "run WOULD do",
        "Turn OFF to apply",
        "OR, not AND",
        "INERT. spaCR",
        "is IGNORED",
        "Affects the FIGURES",
        "SMALLER label's perimeter",
        "MEDIAN pathogen area",
        "SELECTS annotation mode",
        "the IDENTICAL ranking",
        "tensorboard --logdir PATH",
        "takes ONE bound",
        "saved crop FILE",
        "a STORAGE choice",
        "does NOT change training precision",
        "and NOT the same",
        "from a DATABASE instead",
        "gene NUMBER",
    )
    found = [phrase for phrase in rejected if phrase in text]
    assert not found, f"emphatic setting descriptions returned: {found}"


def test_reviewed_qt_tooltips_reject_emphatic_or_conversational_copy():
    """Reviewed widget help must retain its neutral scientific wording."""

    paths = (
        "spacr/qt/widgets/cell_montage_view.py",
        "spacr/qt/widgets/measurement_compare_dialog.py",
        "spacr/qt/widgets/sweep_panel.py",
        "spacr/qt/widgets/regression_results.py",
        "spacr/qt/widgets/feature_rank.py",
    )
    text = "\n".join((REPO / path).read_text(encoding="utf-8")
                     for path in paths)
    rejected = (
        "The LOADED coefficient table",
        "COLOUR LETTERS the annotation",
        "NUMBERS also work",
        "for anyone who knows them",
        "are the better picture",
        "quietly giving you a box",
        "how a real effect is told",
        "ONE NUMBER FOR THE WHOLE SCREEN",
        "widening it to rescue one gene",
        "the objects' own answer",
        "FIELDS TOUCHED, not crops cut",
        "dropped from BOTH sides",
        "WHICH OBJECTS are plotted",
        "their WELL-MATES",
        "into ONE folder",
        "SUM of its guides",
        "Matched at BOTH levels",
        "Up to THREE columns",
        "A SECOND column",
        "marker SHAPE rather than",
        "A THIRD column",
        "SHAPE, NOT SHIFT",
    )
    found = [phrase for phrase in rejected if phrase in text]
    assert not found, f"reviewed Qt prose returned: {found}"


def test_home_descriptions_reject_reviewed_colloquial_phrases():
    """Every registered Home tile summary must describe its operation directly."""

    import spacr.qt
    from spacr.qt.app import APPS

    spacr.qt.register_self_registering_modules()
    text = "\n".join(str(row[2]) for row in APPS).casefold()
    rejected = (
        "someone else's images",
        "see what a project costs",
        "run them overnight",
        "watch the prediction move",
        "separate layers in one world",
    )
    found = [phrase for phrase in rejected if phrase in text]
    assert not found, f"colloquial Home descriptions returned: {found}"


def test_registered_app_copy_uses_professional_project_neutral_language():
    """App summaries, introductions and CLI notes must address the task."""

    import spacr.qt
    from spacr.qt.app import APPS, APP_META

    spacr.qt.register_self_registering_modules()
    text = "\n".join(
        str(value)
        for key, _name, description, _section in APPS
        for value in (
            description,
            APP_META.get(key, {}).get("intro", ""),
            APP_META.get(key, {}).get("cli_note", ""),
        )
    )
    conversational = re.compile(
        r"\b(?:we|our|ours|i think|i want|just|simply|honest|obvious(?:ly)?|"
        r"somebody|someone)\b",
        re.IGNORECASE,
    )
    found = sorted({match.group(0) for match in conversational.finditer(text)})
    assert not found, f"conversational registered-app copy returned: {found}"


def test_preferences_and_preview_help_reject_reviewed_conversational_copy():
    """Indirect tooltip constants retain the scientific wording from review."""

    from spacr.qt.preferences import PREFERENCE_TIPS
    from spacr.qt.widgets.preview_controls import MAX_SETS_TOOLTIP

    text = "\n".join((*PREFERENCE_TIPS.values(), MAX_SETS_TOOLTIP)).casefold()
    rejected = (
        "what spacr thinks you should look at",
        "the level most people want",
        "two is the usual answer",
        "below this, a spinner is a flicker",
        "how hard the highlight follows",
        "a large experiment is never listed whole",
        "simply re-rendering",
    )
    found = [phrase for phrase in rejected if phrase in text]
    assert not found, f"conversational preference or preview help returned: {found}"


def test_terms_setup_instructions_reject_conversational_phrasing():
    """Terms chrome must state the required setup actions directly."""

    from spacr.qt.terms import SCROLL_HINT, WHY_NOT_YET
    from spacr.qt.widgets.setup_slides import SLIDES

    terms_blurb = next(blurb for title, blurb, _keys in SLIDES
                       if title == "Terms of use")
    text = "\n".join((terms_blurb, SCROLL_HINT, WHY_NOT_YET)).casefold()
    rejected = (
        "whole licence is one click away",
        "stays greyed",
        "asks again next time",
    )
    found = [phrase for phrase in rejected if phrase in text]
    assert not found, f"conversational terms instructions returned: {found}"
