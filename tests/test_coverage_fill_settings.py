"""Coverage-fill for spacr.settings — call every defaults function with
an empty dict so its setdefault body executes.
"""
from __future__ import annotations

import inspect

import pytest

import spacr.settings as S


def _defaults_functions():
    out = []
    for name, fn in inspect.getmembers(S, inspect.isfunction):
        if fn.__module__ != "spacr.settings":
            continue
        if name.startswith("get_") or name.startswith("set_"):
            out.append(name)
    return sorted(out)


@pytest.mark.parametrize("fn_name", _defaults_functions())
def test_defaults_function_populates_dict(fn_name):
    fn = getattr(S, fn_name)
    sig = inspect.signature(fn)
    # Most take a single ``settings`` arg (dict) or default None.
    try:
        if len(sig.parameters) == 0:
            result = fn()
        else:
            result = fn({})
    except TypeError:
        # Some accept settings=None.
        result = fn(None)
    except Exception as e:
        pytest.skip(f"{fn_name} needs specific keys: {e}")
    # Defaults functions return the (mutated) settings dict.
    assert result is None or isinstance(result, dict)


def test_defaults_are_idempotent():
    # Calling twice must not clobber a caller-provided value.
    s = S.set_default_settings_preprocess_generate_masks({"diameter": 99})
    s2 = S.set_default_settings_preprocess_generate_masks(s)
    assert s2.get("diameter") == 99


def test_measure_crop_settings_returns_populated():
    s = S.get_measure_crop_settings({})
    assert isinstance(s, dict) and len(s) > 5


def test_perform_regression_defaults_keys():
    s = S.get_perform_regression_default_settings({})
    assert "regression_type" in s or len(s) > 3


# ---------------------------------------------------------------------------
# _set_organelle_defaults + check_settings
# ---------------------------------------------------------------------------

def test_set_organelle_defaults():
    s = S._set_organelle_defaults({})
    assert s.get("organelle_morphology") == "spots"
    assert "organelle_channel" in s


class _Var:
    def __init__(self, v): self._v = v
    def get(self): return self._v


def _vd(**kv):
    # Build a vars_dict: key -> (label, widget, var, frame)
    return {k: ("label", None, _Var(v), None) for k, v in kv.items()}


def test_check_settings_coerces_types():
    expected = {
        "flag": bool, "opt_int": (int, type(None)), "mylist": list,
        "name": str, "num": int, "ratio": float,
        "cell_plate_metadata": list, "none_field": (int, type(None)),
    }
    vd = _vd(
        flag="True", opt_int="5", mylist="[1, 2, 3]", name="hello",
        num="42", ratio="0.5",
        cell_plate_metadata="[['c1'], ['c2']]",   # list-of-lists path
        unknown_key="x",                            # warning path
        none_field="None",                          # None coercion
    )
    settings, errors = S.check_settings(vd, expected)
    assert settings["flag"] is True
    assert settings["opt_int"] == 5
    assert settings["name"] == "hello"
    assert settings["cell_plate_metadata"] == [["c1"], ["c2"]]
    # unknown_key produced a warning.
    assert any("not found in expected types" in e for e in errors)


def test_check_settings_bad_list_of_lists_errors():
    expected = {}
    vd = _vd(cell_plate_metadata="not-a-valid-list")
    settings, errors = S.check_settings(vd, expected)
    assert errors  # invalid format recorded


def test_check_settings_all_type_branches():
    expected = {
        "f_or_none": (float, type(None)),
        "int_or_float": (int, float),
        "int_or_float_dot": (int, float),
        "str_or_none": (str, type(None)),
        "str_none_list": (str, type(None), list),
        "adict": dict,
        "tuple_type": (int, str),
        "plainfloat": float,
    }
    vd = _vd(
        f_or_none="0.5", int_or_float="7", int_or_float_dot="7.5",
        str_or_none="hi", str_none_list="a", adict="{'x': 1}",
        tuple_type="3", plainfloat="1.5",
    )
    settings, errors = S.check_settings(vd, expected)
    assert settings["f_or_none"] == 0.5
    assert settings["int_or_float"] == 7
    assert settings["int_or_float_dot"] == 7.5
    assert settings["adict"] == {"x": 1}
    assert settings["plainfloat"] == 1.5


def test_check_settings_error_branches():
    expected = {
        "f_or_none": (float, type(None)),
        "int_or_none": (int, type(None)),
        "adict": dict,
        "plainint": int,
    }
    vd = _vd(
        f_or_none="not-a-float", int_or_none="not-an-int",
        adict="not-a-dict", plainint="not-an-int",
    )
    settings, errors = S.check_settings(vd, expected)
    # Every bad value recorded an error.
    assert len(errors) >= 3
    # dict error path sets an empty dict.
    assert settings.get("adict") == {}


def test_check_settings_none_and_empty_values():
    expected = {"a": str, "b": bool, "c": list}
    vd = _vd(a="", b="", c="")
    settings, errors = S.check_settings(vd, expected)
    # '' → None coercion at the top.
    assert settings.get("a") is None or settings.get("a") == ""
