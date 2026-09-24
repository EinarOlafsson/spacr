"""Make Masks exposes clickable API help and concept-matched animations."""
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QWidget
from spacr.qt.screens.make_masks import MakeMasksScreen
from spacr.qt.screens.settings_model import _sibling_label_for
from spacr.qt.widgets.hover_tooltip import HoverTooltip, _DERIVE
from spacr.qt.widgets.make_masks_help import ANIMATIONS

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made.close_folded()
    HoverTooltip.instance().hide()


def test_all_existing_help_has_api_metadata_on_its_visible_anchor(screen):
    anchors = [w for w in screen.findChildren(QWidget) if w.toolTip()]
    assert len(anchors) > 70
    for anchor in anchors:
        assert anchor.property('apiTooltipHtml'), (type(anchor).__name__, anchor.toolTip())
        assert 'href=' in anchor.property('apiTooltipHtml')
        assert anchor.property('settingsAppKey') == 'make_masks'
    for attribute in ('_btn_apply', '_btn_compare', '_wand_max', '_cp_diameter', '_enh_gamma'):
        field = getattr(screen, attribute)
        anchor = _sibling_label_for(field) if field.property('apiTooltipDisplayRole') == 'metadata' else field
        assert anchor.toolTip()
        assert 'href=' in anchor.toolTip()


def test_matching_animations_are_offered_without_replacing_original_descriptions(screen):
    popup = HoverTooltip.instance()
    for attribute, animation_key in ANIMATIONS.items():
        field = getattr(screen, attribute)
        anchor = _sibling_label_for(field) if field.property('apiTooltipDisplayRole') == 'metadata' else field
        assert anchor.property('settingAnimationKey') == animation_key
        animation = popup._resolve_animation(anchor, _DERIVE)
        assert animation is not None
        assert animation_key in animation.settings
        assert animation.path.is_file()
    assert 'Expected object diameter' in screen._cp_diameter.property('apiTooltipDescriptionSource')
    assert 'raw' in screen._filter_min_int.property('apiTooltipDescriptionSource')


def test_make_masks_animation_reveal_keeps_separate_state_for_each_control(screen, qtbot):
    popup = HoverTooltip.instance()
    low = _sibling_label_for(screen._norm_lo)
    high = _sibling_label_for(screen._norm_hi)
    popup.show_for(low, low.toolTip())
    assert popup._offered_animation is not None
    assert popup._api_url
    popup.toggle_animation()
    assert popup.animation() is not None
    popup.show_for(high, high.toolTip())
    assert popup.animation() is None
    assert low.property('settingKey') != high.property('settingKey')
