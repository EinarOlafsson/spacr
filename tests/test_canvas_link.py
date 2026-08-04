"""``B16`` — N canvases held on one world window.

The model under the comparison grid. Its whole job is that a difference
between two panels is a difference in the data rather than a difference in
where they happen to be looking, so the tests are about the world: that
panning one panel moves the others by the same distance ACROSS THE SAMPLE (not
by the same number of pixels), that a zoom keeps the point under the cursor
under the cursor in every panel, and that a panel of a different pixel size
still shares the magnification rather than the field of view.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.layers import Canvas, CanvasLink, LayerError, LayerStack, Spacing


def field(size=64, step=1.0, units='px'):
    stack = LayerStack()
    stack.add_image(np.zeros((size, size), np.uint16), name='image',
                    spacing=Spacing.isotropic(2, step, units=units))
    return stack


def canvas(height=100, width=100, source=None):
    return Canvas.covering(source or field(), height=height, width=width)


def link_of(*shapes):
    """A link over panels of the given ``(height, width)`` pixel sizes."""
    link = CanvasLink()
    for index, (height, width) in enumerate(shapes):
        link.add(f'p{index}', canvas(height, width))
    return link


# ---------------------------------------------------------------------------
# 1. what is shared, and what is not
# ---------------------------------------------------------------------------

def test_the_panels_share_the_window_and_keep_their_own_pixel_size():
    link = link_of((100, 100), (100, 60), (50, 100))
    windows = {(c.origin, c.step) for c in link.canvases().values()}
    shapes = {c.shape for c in link.canvases().values()}
    assert len(windows) == 1, 'the panels are not looking at the same place'
    assert shapes == {(100, 100), (100, 60), (50, 100)}


def test_a_panel_added_later_starts_where_the_others_are():
    """Adding a fifth channel must not throw away the view."""
    link = link_of((100, 100))
    link.zoom(4.0)
    link.pan(20, 20)
    moved = link['p0']
    link.add('late', canvas(80, 80))
    assert link['late'].origin == pytest.approx(moved.origin)
    assert link['late'].step == pytest.approx(moved.step)
    assert link['late'].shape == (80, 80)


def test_a_panel_added_unlocked_keeps_its_own_view():
    link = link_of((100, 100))
    link.zoom(4.0)
    free = canvas(80, 80)
    link.add('free', free, locked=False)
    assert link['free'].step == pytest.approx(free.step)
    assert link.is_locked('free') is False


def test_a_panel_cannot_be_added_twice_or_asked_for_when_absent():
    link = link_of((100, 100))
    with pytest.raises(LayerError, match='already in this link'):
        link.add('p0', canvas())
    with pytest.raises(LayerError, match='no panel'):
        link['nope']
    with pytest.raises(LayerError, match='a linked panel needs a Canvas'):
        link.add('bad', 'not a canvas')
    assert 'p0' in link and len(link) == 1
    assert link.keys == ('p0',)


# ---------------------------------------------------------------------------
# 2. moving them together
# ---------------------------------------------------------------------------

def test_panning_one_panel_moves_them_all_by_the_same_world_distance():
    link = link_of((100, 100), (100, 100))
    before = link['p1'].origin
    link.pan(10, 5, key='p0')
    step = link['p0'].step
    assert link['p1'].origin[0] == pytest.approx(before[0] + 10 * step[0])
    assert link['p1'].origin[1] == pytest.approx(before[1] + 5 * step[1])
    assert link['p0'].origin == pytest.approx(link['p1'].origin)


def test_a_pan_is_a_world_distance_even_when_the_panels_are_at_odd_zooms():
    """The driving panel's pixels, converted to world before they travel."""
    link = link_of((100, 100), (100, 100))
    link.unlock('p1')
    link.zoom(4.0, key='p0')            # p0 is now 4x closer than p1
    link.lock('p1')                     # ...and p1 is brought onto it
    assert link['p1'].step == pytest.approx(link['p0'].step)
    before = link['p1'].origin[0]
    link.pan(10, 0, key='p0')
    assert link['p1'].origin[0] == pytest.approx(
        before + 10 * link['p0'].step[0])


def test_zooming_keeps_the_point_under_the_cursor_under_the_cursor():
    link = link_of((100, 100), (100, 60))
    anchor = link['p0'].world_at(30.0, 40.0)
    link.zoom(2.0, key='p0', centre=(30.0, 40.0))

    assert link['p0'].pixel_at(anchor) == pytest.approx((30.0, 40.0))
    # The other panel is zoomed about the SAME WORLD POINT, so the grid does
    # not slide sideways under a wheel over one cell.
    assert link['p1'].pixel_at(anchor) == pytest.approx((30.0, 40.0))


def test_a_depth_change_reaches_every_locked_panel():
    link = link_of((100, 100), (100, 100))
    link.at_depth(z=12.0)
    assert all(c.depth['z'] == 12.0 for c in link.canvases().values())


def test_setting_one_panel_s_canvas_brings_the_others():
    link = link_of((100, 100), (100, 100))
    replacement = link['p0'].zoomed(3.0).panned(5, 5)
    link.set('p0', replacement)
    assert link['p1'].step == pytest.approx(replacement.step)
    assert link['p1'].origin == pytest.approx(replacement.origin)
    assert link['p1'].shape == (100, 100)
    with pytest.raises(LayerError, match='needs a Canvas'):
        link.set('p0', None)


def test_moving_a_link_with_no_panels_says_so():
    with pytest.raises(LayerError, match='no panels to move'):
        CanvasLink().pan(1, 1)


# ---------------------------------------------------------------------------
# 3. letting one go
# ---------------------------------------------------------------------------

def test_an_unlocked_panel_stays_where_it_was_while_the_others_move():
    link = link_of((100, 100), (100, 100), (100, 100))
    link.unlock('p2')
    parked = link['p2'].origin
    link.pan(10, 10, key='p0')
    assert link['p0'].origin == pytest.approx(link['p1'].origin)
    assert link['p2'].origin == pytest.approx(parked)


def test_driving_from_an_unlocked_panel_moves_only_that_panel():
    link = link_of((100, 100), (100, 100))
    link.unlock('p0')
    others = link['p1'].origin
    link.pan(10, 10, key='p0')
    assert link['p1'].origin == pytest.approx(others)
    assert link['p0'].origin != pytest.approx(others)


def test_relocking_brings_a_panel_back_to_where_the_others_are():
    link = link_of((100, 100), (100, 100))
    link.unlock('p1')
    link.zoom(4.0, key='p0')
    link.lock('p1')
    assert link['p1'].origin == pytest.approx(link['p0'].origin)
    assert link['p1'].step == pytest.approx(link['p0'].step)
    assert link.is_locked('p1') is True


def test_a_link_of_only_unlocked_panels_moves_nothing_it_should_not():
    link = link_of((100, 100), (100, 100))
    link.unlock('p0')
    link.unlock('p1')
    parked = link['p1'].origin
    link.pan(10, 10, key='p0')
    assert link['p1'].origin == pytest.approx(parked)


# ---------------------------------------------------------------------------
# 4. resizing — magnification, not field of view
# ---------------------------------------------------------------------------

def test_a_resized_panel_shows_more_of_the_sample_at_the_same_scale():
    """The opposite of Canvas.resized, and the whole reason for the method."""
    link = link_of((100, 100), (100, 100))
    step = link['p0'].step
    link.resize('p0', 100, 200)
    assert link['p0'].step == pytest.approx(step), (
        'the resized panel changed magnification; the two cells are no longer '
        'comparable')
    assert link['p0'].shape == (100, 200)
    assert link['p1'].shape == (100, 100)
    # ...and it now spans twice as much world as its neighbour.
    assert (link['p0'].step[1] * link['p0'].shape[1]
            == pytest.approx(2 * link['p1'].step[1] * link['p1'].shape[1]))


def test_fitting_puts_every_locked_panel_back_on_the_whole_field():
    stack = field()
    link = link_of((100, 100), (100, 100))
    link.unlock('p1')
    link.zoom(8.0, key='p0')
    zoomed_free = link['p1'].step
    link.reset(stack)
    assert link['p0'].step == pytest.approx(canvas(100, 100, stack).step)
    assert link['p1'].step == pytest.approx(zoomed_free)


# ---------------------------------------------------------------------------
# 5. bookkeeping
# ---------------------------------------------------------------------------

def test_every_change_is_announced_with_the_panel_that_moved():
    link = CanvasLink()
    seen = []
    assert link.subscribe(seen.append) is not None
    link.add('p0', canvas())
    link.add('p1', canvas())
    link.pan(1, 1, key='p1')
    link.unlock('p0')
    link.remove('p1')
    assert seen == ['p0', 'p1', 'p1', 'p0', 'p1']
    with pytest.raises(LayerError, match='must be callable'):
        link.subscribe('not callable')


def test_a_listener_can_be_taken_off_again():
    link = CanvasLink({'p0': canvas()})
    seen = []

    def listener(key):
        seen.append(key)

    link.subscribe(listener)
    link.subscribe(listener)   # idempotent
    link.pan(1, 1)
    assert link.unsubscribe(listener) is True
    link.pan(1, 1)
    assert seen == ['p0']


def test_removing_a_panel_takes_it_out_of_the_link():
    link = link_of((100, 100), (100, 100))
    removed = link.remove('p1')
    assert isinstance(removed, Canvas)
    assert link.keys == ('p0',)
    with pytest.raises(LayerError, match='no panel'):
        link.remove('p1')


def test_the_description_says_what_each_panel_shows_and_whether_it_follows():
    link = link_of((100, 100), (60, 80))
    link.unlock('p1')
    lines = link.describe().splitlines()
    assert lines[0].startswith('p0: 100×100 at ') and lines[0].endswith('linked')
    assert lines[1].startswith('p1: 80×60 at ') and lines[1].endswith('free')
    assert CanvasLink().describe() == 'no panels'
