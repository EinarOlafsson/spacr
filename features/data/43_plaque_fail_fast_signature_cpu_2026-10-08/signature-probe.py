import ast
import inspect
import json
from pathlib import Path

import cellpose
from cellpose import models

from tests.cellpose_api_contract import assert_declares_installed_eval_signature
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

path = Path('tests/test_plaque_segmentation_diagnostics.py')
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_incompatible_flow_and_metric_returns_refuse_before_model_evaluation')
model = next(n for n in node.body if isinstance(n, ast.ClassDef))
namespace = {'MISSING_CHANNEL_AXIS': MISSING_CHANNEL_AXIS, 'check_cellpose_eval_call': check_cellpose_eval_call}
exec(compile(ast.Module(body=[model], type_ignores=[]), str(path), 'exec'), namespace)
eval_method = namespace['Model'].eval
assert_declares_installed_eval_signature(eval_method)
native = inspect.signature(models.CellposeModel.eval)
double = inspect.signature(eval_method)
assert list(native.parameters) == list(double.parameters)
for name in native.parameters:
    if name == 'channel_axis':
        assert double.parameters[name].default is MISSING_CHANNEL_AXIS
        assert native.parameters[name].default is None
    else:
        assert double.parameters[name] == native.parameters[name]
print(json.dumps({'cellpose': cellpose.version, 'double_signature': str(double), 'native_signature': str(native), 'exact_names_order_defaults_except_channel_axis_sentinel': True}))
