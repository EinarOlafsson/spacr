"""The authoritative Funk card-v3 six-tile sample must match exact source pixels and MD5s."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path('/media/carruthers/mnt2/spacr/card_v3/card_v3_sample_plateA')


def test_all_six_card_v3_golden_tiles_match_exactly(tmp_path, monkeypatch):
    """Use default nuclei for three tiles and explicit cell-mask source controls for the other three."""
    if not SAMPLE.is_dir():
        pytest.skip('The read-only card_v3_sample_plateA dataset is not installed.')
    from spacr import ops_cell_cycle as cc, ops_engine
    from spacr.image_store import read_image
    from spacr.tabular import read_table

    params=json.loads((SAMPLE/'golden/golden_params.json').read_text())
    assert len(params['tiles']) == 6
    index=read_table(str(SAMPLE/'crop_index.csv'),canonicalise=False,report=None)
    identities={(int(item['tile'].split('_')[0][4:]),int(item['tile'].split('_')[1][1:]))
                for item in params['tiles']}
    queue=index[[*cc.KEY,'cy','cx','model_stage']]
    queue=queue[[ (int(row.site),int(row.label)) in identities for row in queue.itertuples()]].copy()
    queue['rank']=range(len(queue))
    assert len(queue) == 3

    def masked_shards_are_forbidden(*args, **kwargs):
        """Fail if the default field adapter ever attempts to reuse a zero-filled legacy shard."""
        raise AssertionError('The validated source uses full fields, not the zero-filled mini shard.')

    monkeypatch.setattr(ops_engine,'_ph_read_crop',masked_shards_are_forbidden)
    default=tmp_path/'default_nuclei'
    cc.write_label_queue(queue,str(SAMPLE),str(default),labels=str(tmp_path/'unannotated.csv'))
    metadata=json.loads((default/'queue.json').read_text())
    assert metadata['object_mask'] == 'nucleus'
    assert metadata['ranges'] == [[550,23039],[1084,5002]]
    assert metadata['radius'] == 30 and metadata['window'] == 96
    assert metadata['pixel_size_um'] == [params['um_per_px']]*3
    nucleus_rows=read_table(str(default/'queue.csv'),canonicalise=False,report=None)

    class CellMaskControl:
        """Explicit own-cell-mask controls; the shipping Funk adapter still selects nuclei."""
        context=True
        object_mask='cell'
        pixel_size_um=params['um_per_px']

        def __init__(self):
            """Start an empty cache of unmodified supplied full fields and labels."""
            self.fields={}

        def get_raw(self,row):
            """Read via existing shared readers and select only the requested cell mask."""
            site=int(row['site'])
            if site not in self.fields:
                self.fields[site]=(read_image(SAMPLE/f'fields/site-{site}_DAPI_GFP_flatcorrected_uint16.tif'),
                    ops_engine._ph_read_masks(str(SAMPLE/f'masks/site-{site}.npz')))
            pixels,masks=self.fields[site]
            return pixels,masks['cells']==int(row['label'])

    controls=tmp_path/'explicit_cells'
    cc.write_label_queue(queue,str(SAMPLE),str(controls),labels=str(tmp_path/'unannotated.csv'),
                         crop_source=CellMaskControl(),ranges=params['ranges_raw_uint16'])
    cell_rows=read_table(str(controls/'queue.csv'),canonicalise=False,report=None)
    results=[]
    for expected in params['tiles']:
        parts=expected['tile'].split('_')
        site,label=int(parts[0][4:]),int(parts[1][1:])
        mapping=nucleus_rows if expected['object_mask']=='nucleus' else cell_rows
        selected=mapping[(mapping.site==site)&(mapping.label==label)]
        assert len(selected)==1
        actual=read_image(selected.iloc[0].png_path)
        reference=read_image(SAMPLE/'golden'/expected['tile'])
        np.testing.assert_array_equal(actual,reference,err_msg=expected['tile'])
        digest=hashlib.md5(actual.tobytes()).hexdigest()
        assert digest==expected['png_md5']
        assert actual.shape==(288,872,3)
        assert int(actual.astype(np.uint64).sum())==expected['png_sum']
        results.append({'tile':expected['tile'],'object_mask':expected['object_mask'],
                        'actual_pixel_md5':digest,'expected_pixel_md5':expected['png_md5'],'exact_pixels':True})
    assert len(results)==6 and sum(row['object_mask']=='nucleus' for row in results)==3
    (tmp_path/'golden-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    assert not (tmp_path/'unannotated.csv').exists()
