"""Bind Make Masks' custom controls to linked help and matching animations."""
from __future__ import annotations

from PySide6.QtWidgets import QWidget


ANIMATIONS = {
    '_norm_lo': 'normalization_percentiles',
    '_norm_hi': 'normalization_percentiles',
    '_detect_normalized': 'normalization_percentiles',
    '_cp_normalize': 'normalization_percentiles',
    '_min_area': 'object_filters',
    '_otsu_fill_holes': 'organelle_fill_holes',
    '_secondary_fill_holes': 'organelle_fill_holes',
    '_otsu_split': 'organelle_watershed_spots',
    '_enh_split': 'organelle_watershed_spots',
    '_otsu_exclude_border': 'cell_remove_border_objects',
    '_mag_exclude_border': 'cell_remove_border_objects',
    '_cp_cellprob': 'cell_cellprob_threshold',
    '_cp_flow': 'cell_flow_threshold',
    '_cp_diameter': 'cell_diameter',
    '_enh_background_radius': 'organelle_rolling_ball_radius',
    '_enh_clahe': 'organelle_clahe',
    '_enh_clahe_clip': 'organelle_clahe_clip_limit',
}
"""Exact concept matches; unrelated controls do not borrow an illustration."""


def install_make_masks_help(screen):
    """Attach persistent API help and available animations to custom controls.

    :param screen: constructed MakeMasksScreen, before tooltip retargeting.
    :returns: number of controls receiving linked help. Labels receive the
        same metadata through the standard settings retargeting function.
        Existing prose is retained instead of using another module's help.
    """
    from ..screens.settings_model import (
        _ApiTooltipFilter, attach_api_tooltip, retarget_field_tooltips,
    )

    names = {}
    for name, widget in vars(screen).items():
        if isinstance(widget, QWidget):
            names.setdefault(id(widget), name)
        elif isinstance(widget, dict):
            for key, field in widget.items():
                if isinstance(field, QWidget):
                    names.setdefault(id(field), name + '_' + str(key))
    count = 0
    for widget in screen.findChildren(QWidget):
        source = widget.toolTip()
        if not source or widget.property('apiTooltipHtml'):
            continue
        name = names.get(id(widget), widget.objectName() or f'control_{count}')
        key = 'make_masks_' + name.lstrip('_')
        attach_api_tooltip(widget, 'make_masks', key, source, _descriptions={})
        animation = ANIMATIONS.get(name)
        if animation:
            widget.setProperty('settingAnimationKey', animation)
        count += 1
    event_filter = getattr(screen, '_api_tooltip_filter', None)
    if event_filter is None:
        event_filter = _ApiTooltipFilter(screen)
        screen._api_tooltip_filter = event_filter
    retarget_field_tooltips(screen)
    for widget in screen.findChildren(QWidget):
        if widget.property('apiTooltipHtml') and widget.property('apiTooltipDisplayRole') != 'metadata':
            widget.removeEventFilter(event_filter)
            widget.installEventFilter(event_filter)
    return count
