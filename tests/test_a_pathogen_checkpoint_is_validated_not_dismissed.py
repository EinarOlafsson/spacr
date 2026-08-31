"""``pathogen_model`` names a checkpoint that loads, and validation says so.

For a long time it did not: the pre-SAM toxo checkpoints were Cellpose-3
and were removed, Cellpose 4 ships only ``cpsam``, and the validator warned
that anything set here was IGNORED.

That stopped being true. ``spacr/object.py`` reads ``pathogen_model`` for
``object_type == 'pathogen'`` and hands it to
``_resolve_cellpose_pretrained``, which returns an existing file as-is, so a
cpsam fine-tune loads. With a model picker now writing paths into that
field, a warning saying the setting is discarded would be actively wrong at
the moment it starts working.

Two opposite things are still worth saying, and this file is about telling
them apart:

* a MISSING file is an ERROR. The resolver raises ``FileNotFoundError``
  rather than falling back to stock weights -- segmenting silently with the
  wrong model is worse than stopping -- but it raises inside the run, after
  the images are batched. Catching it here is what a validator is for.
* a LEGACY NAME resolves to ``cpsam`` with only a log line, so it is the
  case that really does pass unnoticed.
"""
from __future__ import annotations

import pytest

from spacr.validate import ERROR, WARNING, validate_settings


def _problems(settings, field="pathogen_model"):
    """Every problem reported against ``field``.

    ARGUMENT ORDER IS (settings, app_key), and the key is ``setting``,
    not ``field``. Written the other way round
    first, which validated the string "mask" as though it were a settings
    dict -- so every assertion about this field compared against an empty
    list. Two of them failed and said so; the one asserting SILENCE passed,
    green, against a validator that had never seen the setting.

    That is why `_reaches_the_check` below exists: this helper returning
    [] has to mean "nothing was wrong", never "nothing ran".
    """
    return [p for p in validate_settings(dict(settings), "mask")
            if p.setting == field]


def _reaches_the_check(settings):
    """Whether the pathogen_model branch ran at all for ``settings``.

    Driven by asking for a value the branch is GUARANTEED to complain
    about. If this returns False, an empty `_problems` result proves
    nothing about the setting under test.
    """
    return bool(_problems({**settings, "pathogen_model": "definitely_bogus"}))


BASE = {"src": ".", "pathogen_channel": 1, "cell_channel": 0}


def test_a_checkpoint_that_exists_is_accepted_in_silence(tmp_path):
    """THE CASE THAT MUST PRODUCE NOTHING.

    This is the whole point of the change and the easiest one to lose:
    the picker writes a real path here, and a validator that greets it
    with a warning teaches users to ignore the validator.
    """
    checkpoint = tmp_path / "toxo_pv_v1.pth"
    checkpoint.write_bytes(b"weights")
    # The check is REACHED -- otherwise the silence below is the silence
    # of a branch that never ran, which is how this test first passed
    # against a validator called with its arguments swapped.
    assert _reaches_the_check(BASE), "the pathogen_model branch is not running"
    assert _problems({**BASE, "pathogen_model": str(checkpoint)}) == []


def test_a_checkpoint_that_is_not_there_is_an_error(tmp_path):
    """Reported BEFORE the run, not from inside it.

    The resolver raises for this case anyway; the value here is when.
    """
    missing = tmp_path / "never_trained.pth"
    problems = _problems({**BASE, "pathogen_model": str(missing)})
    assert len(problems) == 1
    assert problems[0].severity == ERROR
    assert "not there" in problems[0].message


def test_a_legacy_name_warns_that_stock_weights_will_be_used(tmp_path):
    """The case that DOES pass silently, which is why it is the warning.

    ``toxo_pv_lumen`` resolves to cpsam with a log line nobody reads, and
    the run then segments with stock weights while the settings file says
    otherwise.
    """
    problems = _problems({**BASE, "pathogen_model": "toxo_pv_lumen"})
    assert len(problems) == 1
    assert problems[0].severity == WARNING
    assert "cpsam" in problems[0].message


def test_naming_the_stock_model_explicitly_is_fine():
    """'cpsam' is what the resolver would use anyway; saying so is not an
    error."""
    assert _reaches_the_check(BASE)
    assert _problems({**BASE, "pathogen_model": "cpsam"}) == []


def test_nothing_is_said_when_no_pathogen_channel_is_segmented():
    """A model for an object that is not being segmented is not a problem
    with the model."""
    assert _problems({"src": ".", "cell_channel": 0,
                      "pathogen_model": "whatever"}) == []


def test_the_validator_and_the_resolver_agree_on_the_stock_name():
    """One spelling of 'cpsam', imported rather than repeated.

    A second copy of that string in the validator is the one that goes
    stale when Cellpose renames its stock model.
    """
    import spacr.validate as validate_module
    from spacr.utils import CPSAM_MODEL

    source = __import__("pathlib").Path(
        validate_module.__file__).read_text(encoding="utf-8")
    assert f'"{CPSAM_MODEL}"' not in source and f"'{CPSAM_MODEL}'" not in source
