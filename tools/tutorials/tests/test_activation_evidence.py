"""Positive counterparts and deliberate corruption of saliency evidence."""
from pathlib import Path
import sys
import copy
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from activation_evidence import encode_gradient, check_pixels, check_outputs, check_database, check_native_runs, check_saved_runs, require_cuda_reference, observe_cuda_run, check_cuda_run


def gradients():
    return np.array([[[0,1],[2,3]],[[1,1],[1,1]],[[4,2],[1,0]]],dtype=np.float32)


def test_channel_minmax_preserves_rgb_and_flat_plane():
    actual=encode_gradient(gradients(),'saliency_channel')
    expected=np.array([[[0,0,255],[85,0,127]],[[170,0,63],[255,0,0]]],dtype=np.uint8)
    np.testing.assert_array_equal(actual,expected)


def test_image_sums_raw_derivatives_before_rescaling():
    # Raw sums 5,4,4,4 become 255,0,0,0. Summing individually normalised
    # channels first would instead create a different map.
    np.testing.assert_array_equal(encode_gradient(gradients(),'saliency_image'),
                                  np.array([[255,0],[0,0]],dtype=np.uint8))


@pytest.mark.parametrize('method',['saliency_image','saliency_channel'])
def test_flat_gradient_is_black_not_nan(method):
    result=encode_gradient(np.ones((3,2,2),dtype=np.float32),method)
    assert result.dtype==np.uint8 and not result.any()


@pytest.mark.parametrize('bad',[np.ones((2,2)),np.ones((2,2,2)),
    np.full((3,2,2),np.nan),np.full((3,2,2),np.inf),np.full((3,2,2),-1.)])
def test_invalid_gradient_refused_after_positive(bad):
    encode_gradient(gradients(),'saliency_image')
    with pytest.raises(ValueError,match='three-channel'):encode_gradient(bad,'saliency_image')


def test_unmeasured_method_refused_after_positive():
    encode_gradient(gradients(),'saliency_channel')
    with pytest.raises(ValueError,match='measured'):encode_gradient(gradients(),'gradcam')


def test_pixels_allow_only_measured_one_level_rounding():
    expected=np.array([[0,100],[200,255]],dtype=np.uint8)
    assert check_pixels(expected,expected)==dict(pixels=4,max_absolute_error=0,differing_pixels=0,tolerance=1)
    actual=expected.copy();actual[0,0]=1
    assert check_pixels(actual,expected)['max_absolute_error']==1
    actual[0,0]=2
    with pytest.raises(ValueError,match='disagree'):check_pixels(actual,expected)


@pytest.mark.parametrize('bad',[np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2),dtype=float)])
def test_shape_and_type_refused_after_positive(bad):
    expected=np.zeros((2,2),dtype=np.uint8);check_pixels(expected,expected)
    with pytest.raises(ValueError,match='shape or pixel type'):check_pixels(bad,expected)


def outputs():
    return [Path('maps/class_0/plate1/E01/a.png'),Path('maps/class_1/plate1/E01/b.png')]


def test_complete_source_identity_and_predicted_class():
    check_outputs(outputs(),['a.png','b.png'],{'a.png':0,'b.png':1})


@pytest.mark.parametrize('kind',['missing','extra','duplicate','substitute','class'])
def test_false_completion_refused_after_positive(kind):
    paths=outputs();predictions={'a.png':0,'b.png':1}
    check_outputs(paths,['a.png','b.png'],predictions)
    if kind=='missing':paths.pop()
    if kind=='extra':paths.append(Path('maps/class_0/plate1/E01/c.png'))
    if kind=='duplicate':paths.append(paths[0])
    if kind=='substitute':paths[1]=paths[1].with_name('c.png')
    if kind=='class':predictions['b.png']=0
    with pytest.raises(ValueError):check_outputs(paths,['a.png','b.png'],predictions)


def database_case():
    source=dict(plateID='plate1',rowID='r5',columnID='c1',fieldID='f10',
                prcfo='plate1_r5_c1_f10_o4',cell_id='o4')
    path='maps/class_0/plate1/E01/a.png'
    row={k:v for k,v in source.items() if k!='cell_id'}
    row.update(file_name='a.png',png_path=path,object='o4')
    return [row],[dict(name='a.png',identity=source)],[path]


@pytest.mark.parametrize('field',['plateID','rowID','columnID','fieldID','prcfo','object','file_name','png_path'])
def test_wrong_cell_or_database_file_refused_after_positive(field):
    rows,crops,paths=database_case();check_database(rows,crops,paths)
    rows[0][field]='not the source value'
    with pytest.raises(ValueError):check_database(rows,crops,paths)


def test_missing_database_record_refused_after_positive():
    rows,crops,paths=database_case();check_database(rows,crops,paths)
    with pytest.raises(ValueError):check_database([],crops,paths)


def cuda_case(method='saliency_channel'):
    return dict(method=method, model_path='/bound/model.pth',
                observer='loaded_classifier_forward_pre_hook', errors=[],
                loads=[dict(model_path='/bound/model.pth', forwards=[dict(
                    parameter_devices=['cuda:0'], buffer_devices=['cuda:0'],
                    input_devices=['cuda:0'], input_shapes=[[4,3,128,128]])])])


def good_runs():
    return [dict(method=method,outcome=dict(finished=True,ok=True,errors=[]),
                 gui_figure_count=1,figures_card_visible=True,settings_errors=[],cuda_execution=cuda_case(method))
            for method in ('saliency_channel','saliency_image')]


def test_native_completion_positive():
    check_native_runs(good_runs())


@pytest.mark.parametrize('kind',['missing','duplicate_method','unfinished','failed','error',
    'no_figure','hidden_card','settings_error'])
def test_false_native_completion_refused_after_positive(kind):
    runs=good_runs();check_native_runs(copy.deepcopy(runs))
    if kind=='missing':runs.pop()
    if kind=='duplicate_method':runs[1]['method']=runs[0]['method']
    if kind=='unfinished':runs[1]['outcome']['finished']=False
    if kind=='failed':runs[1]['outcome']['ok']=False
    if kind=='error':runs[1]['outcome']['errors']=['traceback']
    if kind=='no_figure':runs[1]['gui_figure_count']=0
    if kind=='hidden_card':runs[1]['figures_card_visible']=False
    if kind=='settings_error':runs[1]['settings_errors']=['[settings] ERROR [src]: missing']
    with pytest.raises(ValueError):check_native_runs(runs)


@pytest.mark.parametrize('name', ['cpu', 'mps'])
def test_cuda_reference_rejects_fallback_before_model_execution(name):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='CUDA reference'):
        require_cuda_reference(SimpleNamespace(type=name))


def test_cuda_reference_accepts_cuda_device_description_without_allocating():
    class Device:
        type = 'cuda'
        def __str__(self): return 'cuda:0'
    assert require_cuda_reference(Device()) == 'cuda:0'


@pytest.mark.parametrize('fault', ['missing', 'other_method', 'other_model', 'no_forward',
    'cpu_parameters', 'cpu_inputs', 'cpu_buffer', 'empty_parameters', 'wrong_crops', 'observer_error',
    'foreign_parameter_index', 'foreign_input_index', 'foreign_buffer_index'])
def test_cuda_run_rejects_unobserved_or_foreign_execution(fault):
    run = good_runs()[0]
    evidence = run['cuda_execution']; loaded = evidence['loads'][0]
    row = loaded['forwards'][0]
    if fault == 'missing': run.pop('cuda_execution')
    elif fault == 'other_method': evidence['method'] = 'saliency_image'
    elif fault == 'other_model': loaded['model_path'] = '/foreign/model.pth'
    elif fault == 'no_forward': loaded['forwards'] = []
    elif fault == 'cpu_parameters': row['parameter_devices'] = ['cpu']
    elif fault == 'cpu_inputs': row['input_devices'] = ['cpu']
    elif fault == 'cpu_buffer': row['buffer_devices'] = ['cpu']
    elif fault == 'empty_parameters': row['parameter_devices'] = []
    elif fault == 'wrong_crops': row['input_shapes'] = [[3,3,128,128]]
    elif fault == 'foreign_parameter_index': row['parameter_devices'] = ['cuda:1']
    elif fault == 'foreign_input_index': row['input_devices'] = ['cuda:1']
    elif fault == 'foreign_buffer_index': row['buffer_devices'] = ['cuda:1']
    else: evidence['errors'] = ['observer failed']
    with pytest.raises(ValueError, match='CUDA|classifier'):
        check_cuda_run(run)


@pytest.mark.parametrize('index', [0, 1])
def test_cuda_run_accepts_matching_observed_device_index(index):
    run = good_runs()[0]
    row = run['cuda_execution']['loads'][0]['forwards'][0]
    for key in ('parameter_devices', 'buffer_devices', 'input_devices'):
        row[key] = [f'cuda:{index}']
    assert check_cuda_run(run) is run['cuda_execution']


def test_cuda_saved_route_cannot_accept_cpu_even_with_saved_pixel_matches():
    run = good_runs()[0]
    run['cuda_execution']['loads'][0]['forwards'][0]['input_devices'] = ['cpu']
    second = good_runs()[1]
    with pytest.raises(ValueError, match='CUDA'):
        check_saved_runs([run, second])


def test_cuda_observer_records_real_cpu_forward_without_changing_output(tmp_path):
    import torch
    from types import SimpleNamespace
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3,2))
    image = torch.ones((4,3,128,128))
    expected = model(image).detach().clone()
    artifact = tmp_path/'model.pth'; metadata = {'unchanged':True}; calls=[]
    def loader(path, device):
        calls.append((path,device))
        return model, metadata
    owner = SimpleNamespace(_load_inference_model=loader)
    with observe_cuda_run(owner, artifact, 'saliency_channel') as observed:
        actual_model, actual_metadata = owner._load_inference_model(artifact, 'cpu')
        assert actual_model is model and actual_metadata is metadata
        actual = actual_model(image)
        assert torch.equal(actual, expected)
    assert owner._load_inference_model is loader and len(calls)==1
    assert not model._forward_pre_hooks
    row = observed['loads'][0]['forwards'][0]
    assert row['parameter_devices']==['cpu'] and row['input_devices']==['cpu']
    assert row['input_shapes']==[[4,3,128,128]] and not observed['errors']
    with pytest.raises(ValueError, match='CUDA'):
        check_cuda_run(dict(method='saliency_channel', cuda_execution=observed))


def test_cuda_observer_restores_loader_and_hook_on_worker_failure(tmp_path):
    import torch
    from types import SimpleNamespace
    model = torch.nn.Linear(1,1)
    def loader(path, device): return model, {}
    owner=SimpleNamespace(_load_inference_model=loader)
    with pytest.raises(RuntimeError, match='worker failure'):
        with observe_cuda_run(owner, tmp_path/'model.pth', 'saliency_image'):
            owner._load_inference_model(tmp_path/'model.pth', 'cpu')
            assert model._forward_pre_hooks
            raise RuntimeError('worker failure')
    assert owner._load_inference_model is loader and not model._forward_pre_hooks
