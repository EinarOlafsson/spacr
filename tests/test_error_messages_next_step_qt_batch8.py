"""Overview errors name a correction before any preview or source state changes."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spacr.qt import i18n
from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _english_messages(monkeypatch):
    """Read the exact source-language error contract through the real localizer."""
    def translate(template, **values):
        """Use the existing English localization path with unchanged format values."""
        return i18n.tr(template, language='en', **values)

    monkeypatch.setattr(browser, 'tr', translate)


@pytest.mark.parametrize('options, prefix, recovery', [
    ({'gamma': 0}, 'Gamma must be positive', 'Enter finite values with gamma greater than zero'),
    ({'low': 99, 'high': 1}, 'Percentiles must satisfy', 'Enter increasing lower and upper percentiles'),
    ({'method': 'manual', 'low': 3, 'high': 2}, 'Manual limits must be finite', 'Enter finite display limits'),
    ({'method': 'ratio', 'reference_channel': -1}, 'Reference channel must be nonnegative', 'Choose an existing reference channel'),
], ids=['gamma', 'percentiles', 'manual-limits', 'reference-channel'])
def test_invalid_display_options_explain_the_correction_without_changing_inputs(options, prefix, recovery):
    """Each invalid display control remains refused and names a usable correction."""
    before = copy.deepcopy(options)
    with pytest.raises(ValueError) as caught:
        browser._overview_channel_normalization(options, 0)
    message = str(caught.value)
    assert message.startswith(prefix)
    assert recovery in message
    assert options == before


def test_an_unselected_reference_is_explained_before_any_source_read():
    """An invalid display reference cannot reach pixel sampling or mutate its options."""
    options = {'reference_channel': 1}
    before = copy.deepcopy(options)
    read_planes = Mock(side_effect=AssertionError('Validation must precede source reads.'))
    with pytest.raises(browser._OverviewValidationError) as caught:
        browser._overview_normalization_plan((), (0,), options, read_planes)
    assert 'Select the reference channel for display or turn off channel-reference scaling' in str(caught.value)
    read_planes.assert_not_called()
    assert options == before


def test_ambiguous_field_files_explain_the_source_choice():
    """Ambiguous image metadata remains refused before decoding any source file."""
    field = SimpleNamespace(files=())
    with pytest.raises(browser._OverviewValidationError) as caught:
        browser._overview_read_channel_info(field)
    message = str(caught.value)
    assert message.startswith('This field has ambiguous image files.')
    assert 'Choose a source with one unambiguous image per field and channel' in message
    assert field.files == ()
