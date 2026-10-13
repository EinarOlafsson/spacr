"""Positive actual compositing, then corrupt the result or requested state."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from layer_viewer_evidence import percentile_limits,verify_layer_state,verify_pixels
from spacr.layers import LayerStack,Canvas


@pytest.fixture
def example():
    image=(np.arange(360).reshape(18,20)*173).astype(np.uint16)
    mask=np.zeros_like(image);mask[3:11,4:13]=17;mask[12:17,12:19]=29
    stack=LayerStack();stack.add_image(image.copy(),name='image')
    stack.add_labels(mask.copy(),name='mask',opacity=.5)
    wanted=[dict(name='image',kind='image',visible=True,opacity=1.,blending='translucent',colormap='gray'),
            dict(name='mask',kind='labels',visible=True,opacity=.5,blending='translucent')]
    canvas=Canvas(origin=(-1.,-1.),step=(.7,.6),shape=(29,37))
    return image,mask,stack,wanted,canvas


def test_full_histogram_percentile_matches_known_linear_values(example):
    image,*_=example
    assert percentile_limits(image)==pytest.approx((1242.14,60864.86))


@pytest.mark.parametrize('mode',['translucent','additive','opaque','multiply','minimum'])
def test_actual_all_five_blends_against_independent_samples(example,mode):
    image,mask,stack,wanted,canvas=example
    stack[1].blending=mode;wanted[1]['blending']=mode
    assert verify_layer_state(stack,image,mask,wanted)['layers']==2
    p=verify_pixels(stack.render_uint8(canvas),canvas,image,mask,wanted,samples=19)
    assert p['sampled_screen_pixels']==361 and p['samples_on_source_labels']>0 and p['outside_source_samples']>0


@pytest.mark.parametrize('kind',['hidden','zero_opacity','cyan','below'])
def test_real_display_changes_preserve_pixels(example,kind):
    image,mask,stack,wanted,canvas=example
    original=stack.render_uint8(canvas)
    if kind=='hidden': stack[1].visible=False;wanted[1]['visible']=False
    elif kind=='zero_opacity': stack[1].opacity=0;wanted[1]['opacity']=0.
    elif kind=='cyan': stack[0].set_colormap('cyan');wanted[0]['colormap']='cyan'
    else: stack.lower_layer(stack[1]);wanted.reverse()
    assert verify_layer_state(stack,image,mask,wanted)['source_arrays_unchanged']
    changed=stack.render_uint8(canvas);assert not np.array_equal(changed,original)
    assert verify_pixels(changed,canvas,image,mask,wanted,samples=19)['max_rgb_byte_error']<=1


@pytest.mark.parametrize('kind',['count','order','pixels','contrast','identity','geometry'])
def test_state_corruption_after_positive(example,kind):
    image,mask,stack,wanted,canvas=example
    assert verify_layer_state(stack,image,mask,wanted)['layers']==2
    if kind=='count': stack.add_points(name='points',ndim=2)
    elif kind=='order': wanted.reverse()
    elif kind=='pixels': stack[0].data[0,0]=123
    elif kind=='contrast': stack[0].set_contrast_limits(0,65535)
    elif kind=='identity': stack[1].selected_label=17
    else:
        from spacr.layers import Spacing
        stack[1].spacing=Spacing.isotropic(2,2)
    with pytest.raises(ValueError):verify_layer_state(stack,image,mask,wanted)


def test_corrupt_painted_points_after_positive(example):
    image,mask,stack,wanted,canvas=example
    painted=stack.render_uint8(canvas)
    assert verify_pixels(painted,canvas,image,mask,wanted,samples=19)['sampled_screen_pixels']==361
    painted[:]=255
    with pytest.raises(ValueError,match='Painted pixels'):
        verify_pixels(painted,canvas,image,mask,wanted,samples=19)


def test_empty_layer_guard_with_actual_nonempty_counterpart(example):
    image,mask,stack,wanted,canvas=example
    layer=stack.add_points(name='points',ndim=2)
    wanted.append(dict(name='points',kind='points',visible=True,opacity=1.,blending='translucent'))
    assert verify_layer_state(stack,image,mask,wanted)['layers']==3
    layer.data=np.array([[3.,4.]])
    assert len(layer.data)==1
    with pytest.raises(ValueError,match='invented data'):
        verify_layer_state(stack,image,mask,wanted)


def test_empty_shapes_guard_with_actual_rectangle_counterpart(example):
    image,mask,stack,wanted,canvas=example
    layer=stack.add_shapes(name='shapes',ndim=2)
    wanted.append(dict(name='shapes',kind='shapes',visible=True,opacity=1.,blending='translucent'))
    assert verify_layer_state(stack,image,mask,wanted)['layers']==3
    layer.add_rectangle((2.,3.),(7.,8.))
    assert len(layer)==1
    with pytest.raises(ValueError,match='invented data'):
        verify_layer_state(stack,image,mask,wanted)
