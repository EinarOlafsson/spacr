"""Independent, bounded grayscale/label-overlay proof for tutorial 57.

Single uint16 image, aligned integer labels, pixel units and empty auxiliary
layers only. Compositing is checked at deterministic actual screen pixels;
this is not a proof of all image/channel/spacing or shape-rendering policies.
"""
import colorsys
import math
import numpy as np


def percentile_limits(image):
    """Linear 2/98 percentiles from the complete uint16 histogram."""
    if image.dtype != np.uint16 or image.ndim != 2 or not image.size:
        raise ValueError('Expected a nonempty single-channel uint16 image')
    counts = np.bincount(image.ravel(), minlength=65536).cumsum()
    def at(q):
        position = q*(image.size-1); left = math.floor(position)
        lo = np.searchsorted(counts, left+1); hi = np.searchsorted(counts, math.ceil(position)+1)
        return float(lo+(hi-lo)*(position-left))
    result = at(.02), at(.98)
    if result[1] <= result[0]:
        raise ValueError('Degenerate contrast is outside this proof')
    return result


def verify_layer_state(stack, image, mask, expected):
    if len(stack) != len(expected):
        raise ValueError('Actual layer count differs')
    lohi = percentile_limits(image)
    for actual, wanted in zip(stack, expected):
        state = {key:getattr(actual,key) for key in ('name','kind','visible','opacity','blending')}
        if state != {key:wanted[key] for key in state}:
            raise ValueError('Actual layer order or requested properties differ')
        sp = actual.spacing
        if (sp.axes,sp.scale,sp.translate,sp.units) != (('y','x'),(1.,1.),(0.,0.),'px'):
            raise ValueError('Unexpected layer geometry or physical-unit claim')
        if actual.kind in ('image','labels'):
            data = image if actual.kind == 'image' else mask
            if not np.array_equal(actual.data, data):
                raise ValueError('Loaded source pixels differ')
        if actual.kind == 'image':
            if (actual.n_channels != 1 or actual.data.dtype != image.dtype
                    or actual.colormap.name != wanted['colormap']
                    or tuple(actual.contrast_limits()) != lohi):
                raise ValueError('Actual image channels, colour or contrast differ')
        elif actual.kind == 'labels':
            if actual.field is not None or actual.selected_label != 0:
                raise ValueError('Unexpected object identity or selected-label claim')
        elif actual.kind == 'points':
            if len(actual.data):
                raise ValueError('An empty auxiliary layer gained invented data')
        elif actual.kind == 'shapes':
            if len(actual):
                raise ValueError('An empty auxiliary layer gained invented data')
        else: raise ValueError('Unsupported layer kind')
    return dict(layers=len(expected),source_arrays_unchanged=True,contrast_limits=list(lohi),
                physical_calibration=False,field_key_assigned=False)


def label_rgb(label):
    if not label: return (0.,0.,0.)
    h = (label*0.6180339887498949)%1
    s = .65+.35*((label*7)%5)/4
    v = .75+.25*((label*13)%4)/3
    return colorsys.hsv_to_rgb(h,s,v)


def blend(background, foreground, alpha, mode):
    if mode == 'additive': return [min(1.,d+s*alpha) for d,s in zip(background,foreground)]
    if mode == 'multiply': return [d*(1-alpha)+d*s*alpha for d,s in zip(background,foreground)]
    if mode == 'minimum': return [d*(1-alpha)+min(d,s)*alpha for d,s in zip(background,foreground)]
    if mode not in ('translucent','opaque'):
        raise ValueError('Unsupported blend mode')
    return [d*(1-alpha)+s*alpha for d,s in zip(background,foreground)]


def verify_pixels(painted, canvas, image, mask, expected, *, samples=71):
    """Check a reproducible grid of real canvas pixels, including background."""
    if (painted.shape != tuple(canvas.shape)+(3,) or painted.dtype != np.uint8
            or canvas.axes != ('y','x') or canvas.units != 'px'):
        raise ValueError('Painted canvas shape or coordinate units differ')
    lo,hi = percentile_limits(image)
    coordinates = [(r,c) for r in np.linspace(0,canvas.shape[0]-1,samples,dtype=int)
                   for c in np.linspace(0,canvas.shape[1]-1,samples,dtype=int)]
    ramp = {'gray':(1,1,1),'cyan':(0,1,1),'magenta':(1,0,1)}
    worst=0; foreground_count=0; outside=0
    for row,col in coordinates:
        y = round(canvas.origin[0]+canvas.step[0]*int(row))
        x = round(canvas.origin[1]+canvas.step[1]*int(col))
        inside = 0 <= y < image.shape[0] and 0 <= x < image.shape[1]
        rgb = [0.,0.,0.]
        if inside:
            label = int(mask[y,x]); foreground_count += bool(label)
            for layer in expected:
                if not layer['visible'] or not layer['opacity']: continue
                if layer['kind'] == 'image':
                    value = min(1.,max(0.,(int(image[y,x])-lo)/(hi-lo)))
                    color = [value*v for v in ramp[layer['colormap']]]
                elif layer['kind'] == 'labels' and label:
                    color = label_rgb(label)
                else: continue
                rgb = blend(rgb,color,layer['opacity'],layer['blending'])
        else: outside += 1
        wanted = [int(min(255,max(0,v*255+.5))) for v in rgb]
        worst = max(worst,max(abs(int(a)-b) for a,b in zip(painted[row,col],wanted)))
    if worst > 1:
        raise ValueError('Painted pixels differ from independent source sampling and blending')
    return dict(sampled_screen_pixels=len(coordinates),samples_on_source_labels=foreground_count,
                outside_source_samples=outside,max_rgb_byte_error=worst,tolerance_rgb_bytes=1)
