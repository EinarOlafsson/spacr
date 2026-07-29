"""Compatibility coverage for the corrected vision-interpretation API names."""


def test_ml_exposes_correctly_spelled_interpret_api():
    from spacr import ml

    assert ml.interperate_vision_model is ml.interpret_vision_model
    assert ml.interpret_vision_model.__name__ == "interpret_vision_model"


def test_submodules_exposes_correctly_spelled_interpret_api():
    from spacr import submodules

    assert submodules.interperate_vision_model is submodules.interpret_vision_model
    assert submodules.interpret_vision_model.__name__ == "interpret_vision_model"


def test_settings_defaults_keep_the_legacy_name_as_an_alias():
    from spacr import settings

    assert (
        settings.set_interperate_vision_model_defaults
        is settings.set_interpret_vision_model_defaults
    )
    configured = settings.set_interpret_vision_model_defaults({})
    assert configured["score_column"] == "cv_predictions"
