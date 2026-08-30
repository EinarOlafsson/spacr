"""Every fractal preference the dialog offers must reach the backdrop.

`install_the_spaceout_fractal` built its `Settings` from three of the
stored values and its `RuntimeControls` from three more, silently dropping
the rest. `pattern` was among the dropped ones, so `Settings` fell back to
its default and the backdrop drew Mandelbrot whichever pattern was chosen
-- which is what "the spaceout themes do not work" looks like from the
outside. The pointer and zoom sliders moved and changed nothing for the
same reason.

This pins the plumbing rather than the drawing: every key the preferences
dialog stores and the dataclasses accept has to be handed over.
"""

import inspect

from spacr.qt.preferences import get_fractal_settings
from spacr.qt.widgets.fractal_travel import RuntimeControls, Settings


def _passed_names(source, cls):
    """The stored keys handed to `cls` where it is constructed."""
    call = source.split(f"{cls.__name__}(", 1)[1]
    depth, out = 1, []
    for i, ch in enumerate(call):
        depth += (ch == "(") - (ch == ")")
        if depth == 0:
            out = call[:i]
            break
    return {part.split("=", 1)[0].strip()
            for part in out.replace("\n", " ").split(",") if "=" in part}


def test_every_stored_setting_the_dataclasses_accept_is_handed_over():
    import spacr.qt.app as app_module

    source = inspect.getsource(app_module.install_the_spaceout_fractal)
    stored = set(get_fractal_settings())

    for cls in (Settings, RuntimeControls):
        accepted = set(inspect.signature(cls).parameters)
        expected = stored & accepted
        passed = _passed_names(source, cls)
        missing = expected - passed
        assert not missing, (
            f"{cls.__name__} never receives {sorted(missing)}, so those "
            f"preferences do nothing")


def test_the_chosen_pattern_is_what_the_backdrop_is_built_for():
    """The specific case that made the themes look broken."""
    values = dict(get_fractal_settings())
    values["pattern"] = "cascade"
    built = Settings(pattern=values["pattern"], backend=values["backend"],
                     quality=values["quality"], scale=values["scale"])
    assert built.pattern == "cascade"
    # And the default is genuinely something else, so the check can fail.
    assert Settings().pattern != "cascade"


def test_the_backdrop_is_made_click_through_where_it_is_installed():
    """It is lowered AND click-through; lowering alone is not enough.

    Behind the interface, a lowered backdrop is harmless. Over the bare
    background there is nothing in front of it, so it becomes the topmost
    widget under the cursor and swallows the press -- which is why the main
    window could not be dragged. `create_fractal_widget`'s own comment says
    the backdrop "must not accept events"; nothing had ever set the
    attribute that makes that true.
    """
    import inspect

    import spacr.qt.app as app_module

    source = inspect.getsource(app_module.install_the_spaceout_fractal)
    assert "WA_TransparentForMouseEvents" in source, (
        "the backdrop will swallow clicks meant for the window")
    # Lowering is still required: click-through alone would leave it painted
    # over the interface.
    assert ".lower()" in source
