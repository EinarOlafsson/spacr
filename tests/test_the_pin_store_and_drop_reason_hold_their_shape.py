"""A pin file that is the wrong shape, a module with one pin left, and the
message a drop gives when it found nothing.

The pin store's own comment states the trade it makes: "A lost pin costs one
re-typed path; refusing to open the screen would cost the whole session." The
branches that keep that promise are the ones a hand-edited or half-written
file takes, and none of them had run.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# PinStore._load — a file that parses but is not a mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["[]", '["a", "b"]', '"a string"', "42",
                                     "null"])
def test_a_pin_file_that_is_not_a_mapping_loads_as_empty(tmp_path, content):
    """Arc 234 -> 238: valid JSON of the wrong shape.

    The comment above it covers "no file, an unreadable one, or one someone
    hand-edited into invalid JSON" -- but JSON that PARSES and is a list is a
    fourth case, and the one an older format leaves behind. It must load as no
    pins rather than raising, on exactly the trade the comment states.
    """
    from spacr.chaining import PinStore

    path = tmp_path / "pins.json"
    path.write_text(content)

    store = PinStore(str(path))

    # `_load` is the whole mapping; `pins` is one module's slice of it.
    assert store._load() == {}
    assert store.pins("measure") == {}


def test_a_pin_file_of_the_wrong_inner_shape_keeps_only_what_it_can_read(tmp_path):
    """The inner ``isinstance`` too: one bad module must not lose the others."""
    from spacr.chaining import PinStore

    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"measure": {"src": "/data/plate1"},
                                "mask": ["not", "a", "mapping"]}))

    store = PinStore(str(path))

    assert store.pins("measure").get("src") == "/data/plate1"
    assert store.pins("mask") == {}


def test_an_unreadable_pin_file_loads_as_empty(tmp_path):
    """The except above, so the shape checks are reached deliberately."""
    from spacr.chaining import PinStore

    path = tmp_path / "pins.json"
    path.write_text("{not json at all")

    assert PinStore(str(path))._load() == {}


# ---------------------------------------------------------------------------
# PinStore.unpin — a module that still has other pins
# ---------------------------------------------------------------------------

def test_unpinning_one_setting_leaves_the_module_and_its_other_pins(tmp_path):
    """Arc 301 -> 303: the module entry is NOT dropped.

    Dropping it would take the module's other pins with it, so clearing one
    field would silently clear the rest -- and the user would find out on the
    next run, when automatic chaining picked different paths.
    """
    from spacr.chaining import PinStore

    store = PinStore(str(tmp_path / "pins.json"))
    store.pin("measure", "src", "/data/plate1")
    store.pin("measure", "dst", "/data/out")

    assert store.unpin("measure", "src") is True

    pins = store.pins("measure")
    assert pins["dst"] == "/data/out"
    assert "src" not in pins


def test_unpinning_the_last_setting_drops_the_module_entry(tmp_path):
    """The taken side: an empty module entry is removed rather than kept."""
    from spacr.chaining import PinStore

    store = PinStore(str(tmp_path / "pins.json"))
    store.pin("measure", "src", "/data/plate1")

    assert store.unpin("measure", "src") is True
    assert store.pins("measure") == {}
    assert "measure" not in store._load()


def test_unpinning_something_that_was_never_pinned_answers_false(tmp_path):
    """The early return, which is what makes the return value meaningful."""
    from spacr.chaining import PinStore

    store = PinStore(str(tmp_path / "pins.json"))

    assert store.unpin("measure", "src") is False


# ---------------------------------------------------------------------------
# ports_for_kinds — a kind nothing produces or consumes
# ---------------------------------------------------------------------------

def test_a_kind_no_module_declares_is_passed_over():
    """Arc 1309 -> 1307: the loop goes round rather than appending None.

    Callers iterate the result and read ``port.path``. A None in the tuple
    would be an AttributeError at the call site, far from the unknown kind
    that caused it.
    """
    from spacr.chaining import ports_for_kinds

    found = ports_for_kinds(["definitely_not_a_port_kind"])

    assert found == ()


def test_a_real_kind_comes_back_with_its_port():
    """The taken side, so the skip above is visibly a decision."""
    from spacr.chaining import layout_directories, ports_for_kinds
    from spacr import ports as _ports

    kinds = [k for k in dir(_ports)
             if k.isupper() and isinstance(getattr(_ports, k), str)]
    found = []
    for name in kinds:
        found = ports_for_kinds([getattr(_ports, name)])
        if found:
            break

    assert found, "no declared port kind resolved to a port"
    assert all(getattr(p, "kind", None) for p in found)


# ---------------------------------------------------------------------------
# DropResolution.reason — nothing found and nothing wrong
# ---------------------------------------------------------------------------

def test_a_drop_with_no_targets_and_no_errors_names_the_folder():
    """Arc 1483 -> 1485: the final sentence.

    Reached when a folder is perfectly readable and simply holds nothing this
    module consumes. Naming the root is the whole of the message's value --
    "nothing was found" without saying where is a sentence the user cannot act
    on.
    """
    from spacr.chaining import DropResolution

    resolution = DropResolution(module="measure", dropped="/data/empty",
                                root="/data/empty")

    reason = resolution.reason

    assert "/data/empty" in reason
    assert "nothing this module reads" in reason


def test_a_drop_with_an_error_reports_the_error_and_its_fix():
    """The taken side, so the plain sentence above is visibly the fallback."""
    from spacr.chaining import DropResolution

    class _Problem:
        is_error = True
        message = "no merged folder"
        fix = "run Mask first"

    resolution = DropResolution(module="measure", dropped="/data/x",
                                root="/data/x", problems=(_Problem(),))

    reason = resolution.reason

    assert "no merged folder" in reason
    assert "run Mask first" in reason
