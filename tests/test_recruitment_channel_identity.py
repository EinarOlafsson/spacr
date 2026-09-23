"""Channel identity survives repeated recruitment calculations."""
import numpy as np
import pandas as pd


def test_distinct_channel_ratios_survive_and_no_slope_is_invented():
    from spacr.utils import _calculate_recruitment

    frame = pd.DataFrame(index=['vacuole_a', 'vacuole_b'])
    for channel, signal in ((2, 12.), (3, 100.)):
        for statistic in ('mean_intensity', 'percentile_75', 'outside_mean',
                          'outside_percentile_75', 'periphery_mean'):
            frame[f'pathogen_channel_{channel}_{statistic}'] = [signal, signal * 2]
        for compartment in ('cell', 'cytoplasm', 'nucleus'):
            frame[f'{compartment}_channel_{channel}_mean_intensity'] = [4., 5.]
    original = frame.copy()
    for channel in (2, 3, 2):
        _calculate_recruitment(frame, channel)
    np.testing.assert_allclose(frame['pathogen_channel_2_cytoplasm_mean_ratio'], [3., 4.8])
    np.testing.assert_allclose(frame['pathogen_channel_3_cytoplasm_mean_ratio'], [25., 40.])
    assert sum(column.endswith('_ratio') for column in frame) == 30
    assert not any('slope_channel_' in column for column in frame)
    assert 'pathogen_cytoplasm_mean_mean' not in frame
    pd.testing.assert_frame_equal(frame[original.columns], original)
