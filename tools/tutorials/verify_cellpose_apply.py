#!/usr/bin/env python3
"""Check saved tutorial masks and figure arrays against direct Cellpose calls.

This bypasses spaCR's loader, shuffling, batching and file naming. It proves
input/output attribution and implementation consistency, NOT mask accuracy.
Run only after the native capture has stopped, under the memory guard.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

from build_evaluation_example import sha
from cellpose_apply_evidence import normalize_field,require_pixels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args()
    proof=json.loads((args.capture/'scientific_acceptance.json').read_text())
    work=Path(proof['private_folder'])
    settings=json.loads((args.capture/'configured_settings.json').read_text())
    expected=dict(model_name='cpsam',custom_model=None,normalize=True,
        percentiles=[2,99],diameter=30,CP_prob=0,flow_threshold=.4,
        rescale=False,resample=False,fill_in=False,resize=False,invert=False,
        remove_background=False,channels=[0])
    if any(settings.get(k)!=v for k,v in expected.items()):
        raise ValueError('The saved settings do not describe the bounded reference case')
    import torch
    from cellpose.models import CellposeModel
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise ValueError('This recorded reference requires the same CUDA device')
    torch.cuda.set_per_process_memory_fraction(.25)
    model=CellposeModel(pretrained_model='cpsam',gpu=True,device=torch.device('cuda'))
    result=dict(accepted=False,scope='input/output attribution, not segmentation accuracy',
        checkpoint=str(model.pretrained_model),checkpoint_sha256=sha(model.pretrained_model),
        files=[],figures=[])
    references={}
    for filename,source_hash in proof['original_inputs'].items():
        image_path=work/Path(filename).name
        if sha(filename)!=source_hash or sha(image_path)!=source_hash:
            raise ValueError('A source image changed')
        image=tifffile.imread(image_path)
        normalized=normalize_field(image)
        masks,flows,*_=model.eval(normalized,normalize=False,channel_axis=None,
            diameter=30,flow_threshold=.4,cellprob_threshold=0,rescale=None,resample=False,progress=True)
        saved=tifffile.imread(work/'masks'/image_path.name)
        pixels=require_pixels(saved,masks)
        base=np.repeat((normalized*255).astype(np.uint8)[...,None],3,axis=2)
        contours,_=cv2.findContours(saved.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base,contours,-1,(255,0,0),2)
        references[image_path.name]=dict(original=normalized,overlay=base,flow=flows[0][...,0])
        result['files'].append(dict(name=image_path.name,checked_pixels=pixels,
            objects=int(np.count_nonzero(np.unique(saved))),exact_mask_equal=True,
            source_sha256=source_hash,saved_mask_sha256=sha(work/'masks'/image_path.name)))
    seen=set()
    for path in sorted(args.capture.glob('batch_figure_*_arrays.npz')):
        with np.load(path,allow_pickle=False) as data:
            matches=[name for name,expected in references.items() if np.array_equal(data['original'],expected['original'])]
            if len(matches)!=1:raise ValueError('A figure original does not identify exactly one source')
            name=matches[0]
            if name in seen:raise ValueError('The figures duplicate an input and omit another')
            seen.add(name)
            checked={key:require_pixels(data[key],references[name][key]) for key in ('original','overlay','flow')}
            result['figures'].append(dict(file=path.name,source_image=name,exact_values_checked=checked))
    if seen!=set(references):raise ValueError('Not every saved mask has its matching three-panel figure')
    # Current live preview uses Cellpose's own default preprocessing, not the
    # Apply form's explicit normalize=False after its 2/99 loader stretch.
    image=tifffile.imread(work/'cell_pair_02.tif')
    mask,*_=model.eval(image,diameter=30,flow_threshold=.4,cellprob_threshold=0)
    preview=np.load(args.capture/'preview_cell.npy',allow_pickle=False)
    result['preview_default_reference']=dict(checked_pixels=require_pixels(preview,mask),
        objects=int(np.count_nonzero(np.unique(mask))),exact_mask_equal=True,
        differs_from_batch=not np.array_equal(mask,tifffile.imread(work/'masks/cell_pair_02.tif')),
        meaning='Matches raw-image Cellpose defaults, not the Apply preprocessing settings.')
    result['accepted']=True
    (args.capture/'independent_reference.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
