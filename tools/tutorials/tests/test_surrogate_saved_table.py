from io import StringIO
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_explain_cv_capture import read_saved_table


def test_saved_float_survives_the_actual_five_digit_rounding_boundary():
    frame=read_saved_table(StringIO('index,mean\n242,4049.4500000000003\n'))
    assert frame.index.tolist()==[242]
    assert frame.loc[242,'mean']==float('4049.4500000000003')
    assert f"{frame.loc[242,'mean']:.5g}"=='4049.5'


def test_text_identity_and_column_order_survive_reading():
    frame=read_saved_table(StringIO('object,value,class\np_r1_c2_o3,0.125,1\n'))
    assert frame.index.tolist()==['p_r1_c2_o3']
    assert frame.columns.tolist()==['value','class']
    assert frame.loc['p_r1_c2_o3'].tolist()==[.125,1]
