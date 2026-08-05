"""``B13`` — counting by hand, with the clicks kept.

The point of the feature is that the tally is *derived* from markers that are
data, not a number somebody typed. So the tests assert on the markers: that a
click lands at the world coordinate it was given (which is what makes a count
made at 8× zoom agree with one made at 1×), that the tally always equals what
is on the canvas, that undo puts a removed marker back where it was rather than
where the cursor now is, and that the export can be read back into a session
and produce the same picture.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.counting import CountClass, CountingSession, LAYER_PREFIX
from spacr.layers import (Canvas, FieldKey, LayerError, LayerStack,
                          PointsLayer, Spacing)


def stack(size=64, step=1.0, units='px'):
    """A stack with one image layer, which is where the spacing comes from."""
    out = LayerStack()
    out.add_image(np.zeros((size, size), np.uint16), name='image',
                  spacing=Spacing.isotropic(2, step, units=units))
    return out


def session(**kwargs):
    kwargs.setdefault('size', 6.0)
    return CountingSession(stack(), **kwargs)


# ---------------------------------------------------------------------------
# classes and their layers
# ---------------------------------------------------------------------------

def test_a_session_starts_with_a_marker_layer_per_class():
    counting = session()
    assert counting.class_names == ('infected', 'uninfected')
    assert counting.active == 'infected'
    names = counting.layer('infected').stack.names
    assert names == ('image', f'{LAYER_PREFIX}infected',
                     f'{LAYER_PREFIX}uninfected')
    assert isinstance(counting.layer(), PointsLayer)


def test_each_class_keeps_its_own_colour_and_shortcut():
    counting = CountingSession(stack(), classes=[
        ('mitotic', 'yellow'), CountClass('dead', 'red', shortcut='d')])
    assert counting.layer('mitotic').face_color == (1.0, 1.0, 0.0, 1.0)
    assert counting.layer('dead').face_color == (1.0, 0.0, 0.0, 1.0)
    assert counting.class_for_shortcut('1') == 'mitotic'
    assert counting.class_for_shortcut('d') == 'dead'
    assert counting.class_for_shortcut('9') is None


def test_a_bare_name_is_given_the_next_colour_and_the_next_number():
    counting = CountingSession(stack(), classes=['a', 'b', 'c'])
    colours = {counting.layer(name).face_color for name in ('a', 'b', 'c')}
    assert len(colours) == 3, 'two classes were given the same colour'
    assert [c.shortcut for c in counting.classes] == ['1', '2', '3']


def test_two_classes_cannot_share_a_name():
    counting = session()
    with pytest.raises(LayerError, match='already counts'):
        counting.add_class('infected')


def test_counting_something_the_session_does_not_have_names_what_it_has():
    counting = session()
    with pytest.raises(LayerError, match="counts \\['infected', 'uninfected'\\]"):
        counting.add({'y': 1.0, 'x': 1.0}, 'mitotic')
    with pytest.raises(LayerError, match='does not count'):
        counting.active = 'mitotic'


def test_a_session_counts_on_a_layer_stack():
    with pytest.raises(LayerError, match='LayerStack'):
        CountingSession('not a stack')


# ---------------------------------------------------------------------------
# the clicks
# ---------------------------------------------------------------------------

def test_a_marker_lands_on_the_world_point_it_was_given():
    counting = session()
    counting.add({'y': 12.5, 'x': 30.25})
    np.testing.assert_allclose(counting.layer().world[0], [12.5, 30.25])


def test_a_marker_placed_on_a_micrometre_stack_is_stored_in_micrometres():
    """The count and the image agree because both are in world units."""
    counting = CountingSession(stack(step=0.65, units='um'), size=3.0)
    counting.add({'y': 13.0, 'x': 6.5})
    np.testing.assert_allclose(counting.layer().world[0], [13.0, 6.5])
    # ...and in the layer's own data coordinates that is voxel 20, 10.
    np.testing.assert_allclose(counting.layer().data[0], [20.0, 10.0])


def test_the_tally_is_read_from_the_markers_not_kept_beside_them():
    counting = session()
    counting.add({'y': 1.0, 'x': 1.0})
    counting.add({'y': 2.0, 'x': 2.0})
    counting.active = 'uninfected'
    counting.add({'y': 3.0, 'x': 3.0})
    assert counting.counts() == {'infected': 2, 'uninfected': 1}
    assert counting.total == 3

    # Removed through the layer rather than through the session: the tally
    # still agrees with the canvas, because it is not stored.
    counting.layer('infected').remove(0)
    assert counting.counts()['infected'] == 1


def test_clicking_a_marker_again_takes_it_away():
    counting = session()
    counting.add({'y': 10.0, 'x': 10.0})
    action, name, _index = counting.toggle({'y': 11.0, 'x': 10.0})
    assert (action, name) == ('removed', 'infected')
    assert counting.total == 0
    action, name, _index = counting.toggle({'y': 40.0, 'x': 40.0})
    assert (action, name) == ('added', 'infected')


def test_a_marker_is_removed_whatever_class_it_was_scored_as():
    """Otherwise the wrong marker is left where it is."""
    counting = session()
    counting.add({'y': 10.0, 'x': 10.0}, 'uninfected')
    counting.active = 'infected'
    assert counting.remove_at({'y': 10.0, 'x': 10.0}) == ('uninfected', 0)
    assert counting.total == 0


def test_a_click_on_nothing_removes_nothing():
    counting = session()
    counting.add({'y': 10.0, 'x': 10.0})
    assert counting.remove_at({'y': 50.0, 'x': 50.0}) is None
    assert counting.find({'y': 50.0, 'x': 50.0}) is None
    assert counting.total == 1


def test_a_marker_is_found_within_its_own_radius_and_not_beyond_it():
    counting = CountingSession(stack(), size=10.0)
    counting.add({'y': 20.0, 'x': 20.0})
    assert counting.find({'y': 24.0, 'x': 20.0}) == ('infected', 0)
    assert counting.find({'y': 26.0, 'x': 20.0}) is None


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------

def test_undo_takes_back_an_add():
    counting = session()
    counting.add({'y': 1.0, 'x': 1.0})
    assert counting.undo() == ('add', 'infected')
    assert counting.total == 0
    assert counting.undo() is None


def test_undo_puts_a_removed_marker_back_where_it_was():
    """Not where the cursor is now — which is the whole difficulty."""
    counting = session()
    counting.add({'y': 12.0, 'x': 34.0})
    counting.remove_at({'y': 12.0, 'x': 34.0})
    assert counting.undo() == ('remove', 'infected')
    np.testing.assert_allclose(counting.layer().world[0], [12.0, 34.0])


def test_undo_walks_back_through_both_classes_in_order():
    counting = session()
    counting.add({'y': 1.0, 'x': 1.0}, 'infected')
    counting.add({'y': 2.0, 'x': 2.0}, 'uninfected')
    counting.add({'y': 3.0, 'x': 3.0}, 'infected')
    for expected in (('add', 'infected'), ('add', 'uninfected'),
                     ('add', 'infected')):
        assert counting.undo() == expected
    assert counting.total == 0


def test_undo_after_a_clear_does_not_resurrect_what_was_cleared():
    counting = session()
    counting.add({'y': 1.0, 'x': 1.0})
    counting.add({'y': 2.0, 'x': 2.0}, 'uninfected')
    assert counting.clear('infected') == 1
    assert counting.counts() == {'infected': 0, 'uninfected': 1}
    assert counting.undo() == ('add', 'uninfected')
    assert counting.total == 0


def test_clearing_everything_empties_every_class():
    counting = session()
    for i in range(3):
        counting.add({'y': float(i), 'x': 1.0})
    counting.add({'y': 9.0, 'x': 9.0}, 'uninfected')
    assert counting.clear() == 4
    assert counting.total == 0
    assert counting.undo() is None


# ---------------------------------------------------------------------------
# the number the count is for
# ---------------------------------------------------------------------------

def test_the_fraction_is_computed_rather_than_divided_by_hand():
    counting = session()
    for i in range(4):
        counting.add({'y': float(i), 'x': 1.0}, 'infected')
    for i in range(6):
        counting.add({'y': float(i), 'x': 2.0}, 'uninfected')
    assert counting.fraction('infected') == pytest.approx(0.4)
    assert counting.describe() == (
        'infected 4 (40%) · uninfected 6 (60%) · 10 total')


def test_an_empty_session_has_a_fraction_rather_than_a_zero_division():
    counting = session()
    assert counting.fraction('infected') == 0.0
    assert counting.describe() == 'nothing counted yet'


# ---------------------------------------------------------------------------
# export — one row per click, and the unit beside the coordinates
# ---------------------------------------------------------------------------

def test_the_export_is_one_row_per_click_with_its_units():
    counting = CountingSession(stack(step=0.65, units='um'), size=3.0)
    counting.add({'y': 6.5, 'x': 13.0}, 'infected')
    counting.add({'y': 19.5, 'x': 26.0}, 'uninfected')
    frame = counting.to_frame()

    assert list(frame.columns) == ['class', 'y', 'x', 'units']
    assert len(frame) == 2
    assert set(frame['units']) == {'um'}
    np.testing.assert_allclose(
        frame.loc[frame['class'] == 'infected', ['y', 'x']].to_numpy(),
        [[6.5, 13.0]])


def test_the_export_carries_the_field_key_so_a_count_can_be_joined():
    field = FieldKey(values=dict(zip(FieldKey.columns(),
                                     ('plate1', 'A', '1', '1'))))
    counting = CountingSession(stack(), size=6.0, field=field)
    counting.add({'y': 1.0, 'x': 1.0})
    frame = counting.to_frame()
    for column, value in field.values.items():
        assert list(frame[column]) == [value]
    assert list(counting.summary().columns)[:len(field.values)] == \
        list(field.values)


def test_the_summary_is_one_row_per_class_including_the_empty_ones():
    counting = session()
    counting.add({'y': 1.0, 'x': 1.0})
    summary = counting.summary()
    assert list(summary['class']) == ['infected', 'uninfected']
    assert list(summary['count']) == [1, 0]
    assert list(summary['fraction']) == [1.0, 0.0]
    assert set(summary['total']) == {1}


def test_the_count_is_written_where_it_was_asked_for(tmp_path):
    counting = session()
    counting.add({'y': 1.0, 'x': 2.0})
    path = counting.to_csv(str(tmp_path / 'deeper' / 'counts.csv'))
    import pandas as pd

    reloaded = pd.read_csv(path)
    assert list(reloaded['class']) == ['infected']
    summary = counting.to_csv(str(tmp_path / 'tally.csv'), summary=True)
    assert list(pd.read_csv(summary)['count']) == [1, 0]


def test_an_exported_count_reloads_onto_the_same_world_points(tmp_path):
    counting = session()
    counting.add({'y': 4.0, 'x': 8.0}, 'infected')
    counting.add({'y': 16.0, 'x': 32.0}, 'uninfected')
    frame = counting.to_frame()

    reopened = session()
    assert reopened.load_frame(frame) == 2
    np.testing.assert_allclose(reopened.layer('infected').world, [[4.0, 8.0]])
    np.testing.assert_allclose(reopened.layer('uninfected').world,
                               [[16.0, 32.0]])
    assert reopened.counts() == counting.counts()


def test_reloading_a_count_declares_classes_it_has_never_heard_of():
    counting = CountingSession(stack(), classes=['a'], size=6.0)
    counting.add({'y': 1.0, 'x': 1.0}, 'a')
    frame = counting.to_frame()
    frame.loc[0, 'class'] = 'somebody elses class'

    reopened = CountingSession(stack(), classes=['a'], size=6.0)
    reopened.load_frame(frame)
    assert 'somebody elses class' in reopened.class_names


def test_reloading_a_count_made_in_other_units_refuses():
    """The markers would be somewhere plausible and wrong."""
    micrometres = CountingSession(stack(step=0.65, units='um'), size=3.0)
    micrometres.add({'y': 6.5, 'x': 13.0})
    with pytest.raises(LayerError, match='plausible and wrong'):
        session().load_frame(micrometres.to_frame())


def test_reloading_a_frame_that_is_not_a_count_says_what_is_missing():
    import pandas as pd

    with pytest.raises(LayerError, match=r"\['class', 'y'\]"):
        session().load_frame(pd.DataFrame({'x': [1.0]}))


# ---------------------------------------------------------------------------
# what the markers look like
# ---------------------------------------------------------------------------

def test_the_markers_are_drawn_in_their_class_colour():
    """A rendered pixel, so this fails if the colour never reaches the canvas."""
    counting = CountingSession(stack(size=32), classes=[('a', 'red')],
                               size=8.0)
    counting.add({'y': 16.0, 'x': 16.0})
    canvas = Canvas.for_grid(counting.layer().spacing, (32, 32))
    rgb = counting.layer().stack.render(canvas)
    np.testing.assert_allclose(rgb[16, 16], [1.0, 0.0, 0.0], atol=1e-6)
    # ...and nowhere near the marker, nothing was drawn.
    np.testing.assert_allclose(rgb[2, 2], [0.0, 0.0, 0.0], atol=1e-6)


def test_hiding_a_class_hides_only_its_markers():
    counting = CountingSession(stack(size=32),
                               classes=[('a', 'red'), ('b', 'green')],
                               size=8.0)
    counting.add({'y': 8.0, 'x': 8.0}, 'a')
    counting.add({'y': 24.0, 'x': 24.0}, 'b')
    canvas = Canvas.for_grid(counting.layer().spacing, (32, 32))
    counting.layer('a').visible = False
    rgb = counting.layer().stack.render(canvas)
    np.testing.assert_allclose(rgb[8, 8], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(rgb[24, 24], [0.0, 1.0, 0.0], atol=1e-6)
    # The tally is unaffected: hidden is not deleted.
    assert counting.counts() == {'a': 1, 'b': 1}


def test_detaching_leaves_the_tally_readable():
    counting = session()
    counting.add({'y': 1.0, 'x': 1.0})
    counting.detach()
    assert counting.layer().stack is None
    assert counting.counts() == {'infected': 1, 'uninfected': 0}
