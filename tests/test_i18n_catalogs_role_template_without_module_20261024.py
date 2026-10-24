"""A numbered background switch is only localised when a language catalogue exists."""
import pytest

from spacr.organelle_types import (
    CATALOGUED_ORGANELLE_SLOTS, _background_switch_role, organelle_number,
)
from spacr.qt.i18n_catalogs import _english, _numbered_background_value


def _numbered_source(table_name, slot):
    key = f"remove_background_organelle_{slot}"
    role = _background_switch_role(key)
    template = getattr(_english(), table_name)["remove_background_organelle_2"]
    number = organelle_number(role)
    assert number == slot > CATALOGUED_ORGANELLE_SLOTS
    return key, template.replace("organelleb", role).replace(
        "organelle 2", f"organelle {number}")


@pytest.mark.parametrize("table_name", ["SETTING_LABELS", "SETTING_TOOLTIPS"])
@pytest.mark.parametrize("language", ["en", "", "xx"])
def test_exact_numbered_source_without_a_language_module_is_left_alone(table_name, language):
    key, source = _numbered_source(table_name, 9)
    assert _numbered_background_value(key, source, language, table_name) is None


@pytest.mark.parametrize("table_name", ["SETTING_LABELS", "SETTING_TOOLTIPS"])
def test_exact_numbered_source_in_a_real_language_is_localised_with_its_number(table_name):
    key, source = _numbered_source(table_name, 9)
    localized = _numbered_background_value(key, source, "de", table_name)
    assert localized is not None and localized != source
    assert "9" in localized
