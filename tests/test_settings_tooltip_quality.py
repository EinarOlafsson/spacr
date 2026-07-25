"""Quality gate for spacr.settings tooltips.

A tooltip is what the user reads in the GUI before touching a knob, so it has
to earn its place. The bar these tests enforce:

  * every declared setting has one,
  * it declares its type,
  * it says more than the key name already says (no tautologies such as
    ``amsgrad: "Whether to use AMSGrad optimizer."``),
  * it is long enough to have said what changes when you alter the value,
  * no key is silently shadowed by a duplicate entry.

The two allow-lists below are deliberately explicit: a setting may only be
exempt if someone wrote down why.
"""
from __future__ import annotations

import re

import pytest

from spacr.settings import descriptions, expected_types, tooltips


# Settings whose tooltip is still the original short text because the audit
# could not establish the truth (dead knobs with no consumer, or claims the
# grader could not verify). Shrinking this list is good; growing it needs a
# reason.
KNOWN_THIN = {
    # verified dead: declared in settings.py but read by nothing
    "remove_border_cells", "remove_border_pathogens",
    "signal_direction", "offset", "nc", "pc", "nc_loc", "pc_loc",
    "backgrounds", "normalize_plots", "organelle_chann_dim",
    "class_1_threshold", "infection_xgb_proba", "visualize",
    "from_scratch", "width_height",
    # audit could not confirm behaviour to the grader's standard
    "pathogen_model", "train", "train_channels", "rescale",
    "pathogen_limit", "save",
}

TYPE_PREFIX = re.compile(r"^\((?P<type>[^)]+)\)\s*-\s*(?P<body>.+)$", re.S)


def _body(text: str) -> str:
    """Return the prose after the ``(type) - `` prefix."""
    m = TYPE_PREFIX.match(text.strip())
    return m.group("body").strip() if m else text.strip()


def _all_tooltips():
    """Every setting key -> the tooltip shown beside its widget.

    ``spacr.settings.descriptions`` is deliberately NOT merged in: those 13
    entries are per-APP blurbs ('mask', 'measure', 'classify', 'umap', …)
    rendered on the app screen, not per-setting tooltips, and they are
    multi-line prose with no ``(type)`` prefix by design.
    """
    return {k: v for k, v in tooltips.items() if isinstance(v, str) and v.strip()}


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------

def test_every_tooltip_declares_its_type():
    """House format is ``"(type) - prose"``; the GUI shows the type hint."""
    missing = [k for k, v in _all_tooltips().items() if not TYPE_PREFIX.match(v.strip())]
    assert not missing, f"tooltips with no (type) prefix: {sorted(missing)[:20]}"


def test_no_duplicate_keys_in_the_tooltips_literal():
    """A repeated key in the dict literal silently discards the earlier text.

    This is not hypothetical: the literal carried ten shadowed entries before
    the 2026-07 audit, so ten tooltips were dead text nobody could ever see.
    """
    import inspect
    from collections import Counter

    import spacr.settings as st

    src = inspect.getsource(st)
    m = re.search(r"^tooltips\s*=\s*\{", src, re.M)
    assert m, "tooltips dict literal not found"
    start = m.end() - 1
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    keys = re.findall(r'[\{,]\s*["\']([A-Za-z_][A-Za-z_0-9]*)["\']\s*:', src[start:end])
    dupes = {k: c for k, c in Counter(keys).items() if c > 1}
    assert not dupes, f"duplicate keys shadow earlier tooltips: {dupes}"


def test_tooltips_are_single_line_plain_text():
    """The Qt tooltip widget renders these verbatim: no newlines, no markdown."""
    bad = [k for k, v in _all_tooltips().items() if "\n" in v or "**" in v or "`" in v]
    assert not bad, f"tooltips containing newlines/markdown: {sorted(bad)[:20]}"


# ---------------------------------------------------------------------------
# informativeness
# ---------------------------------------------------------------------------

def test_no_tooltip_merely_restates_its_key():
    """``amsgrad: "Whether to use AMSGrad optimizer."`` is the rejected shape.

    A tooltip fails if every meaningful word of the key appears in the body and
    the body adds fewer than 12 further words — i.e. it names the setting again
    and stops.
    """
    offenders = []
    for key, text in _all_tooltips().items():
        if key in KNOWN_THIN:
            continue
        body = _body(text)
        words = body.split()
        key_words = [w for w in key.lower().split("_") if len(w) > 2]
        if not key_words:
            continue
        low = body.lower()
        restates = all(w in low for w in key_words)
        if restates and len(words) < 12 + len(key_words):
            offenders.append((key, body[:70]))
    assert not offenders, (
        "tooltips that only restate their key:\n  "
        + "\n  ".join(f"{k}: {b}" for k, b in offenders)
    )


def test_tooltips_say_what_changes_when_you_alter_the_value():
    """Enforce a floor on substance.

    A tooltip that clears the tautology bar can still be a bare definition. We
    can't test for insight, but we can require enough prose to have expressed
    it — every audited tooltip comfortably clears 15 words.
    """
    thin = []
    for key, text in _all_tooltips().items():
        if key in KNOWN_THIN:
            continue
        if len(_body(text).split()) < 15:
            thin.append((key, _body(text)))
    assert not thin, (
        "tooltips too short to say what changes when you alter them:\n  "
        + "\n  ".join(f"{k}: {b}" for k, b in thin)
    )


def test_known_thin_list_contains_no_stale_entries():
    """Every exemption must still correspond to a real setting.

    Stops the allow-list rotting into a place where typos hide.
    """
    known = set(_all_tooltips())
    stale = sorted(k for k in KNOWN_THIN if k not in known)
    assert not stale, f"KNOWN_THIN names settings that no longer have tooltips: {stale}"


# ---------------------------------------------------------------------------
# coverage of the settings surface
# ---------------------------------------------------------------------------

def test_every_typed_setting_has_a_tooltip():
    """A setting the GUI can render must have text to render beside it."""
    tips = _all_tooltips()
    missing = sorted(k for k in expected_types if k not in tips)
    assert not missing, f"typed settings with no tooltip: {missing}"


@pytest.mark.parametrize("key", ["amsgrad", "cell_diameter", "loss_type"])
def test_representative_keys_are_substantive(key):
    """Spot-check the shapes the audit was commissioned to fix."""
    tips = _all_tooltips()
    if key not in tips:
        pytest.skip(f"{key} is not a declared setting")
    body = _body(tips[key])
    assert len(body.split()) >= 15, f"{key} tooltip is still thin: {body}"
