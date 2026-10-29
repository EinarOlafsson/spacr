"""New interpreter requirements retain the established older environments."""
import pytest
from packaging.requirements import Requirement

from tests.test_packaging_metadata import _core_dependencies, _extras, _requires_python


@pytest.mark.parametrize('group', ['core', 'qt', 'flowview', 'tutorial', 'all'])
@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_python315_selects_one_compatible_qt_requirement(group, platform):
    values = _core_dependencies() if group == 'core' else _extras()[group]
    requirements = [Requirement(value) for value in values]
    qt = [requirement for requirement in requirements
          if requirement.name.lower() == 'pyside6' and requirement.marker.evaluate(
              {'python_version': '3.15', 'sys_platform': platform})]
    assert len(qt) == 1
    assert '6.12.0' in qt[0].specifier
    assert '6.12.1' in qt[0].specifier
    assert '6.11.3' not in qt[0].specifier
    assert '7.0' not in qt[0].specifier


@pytest.mark.parametrize('name,extra', [('pyfixest', 'pyfixest'), ('catboost', 'boosting'),
                                       ('pymc', 'pymc')])
def test_python315_does_not_force_unavailable_optional_accelerators(name, extra):
    requirements = [Requirement(value) for value in _core_dependencies()]
    matching = [requirement for requirement in requirements if requirement.name == name]
    assert len(matching) == 1
    assert matching[0].marker.evaluate({'python_version': '3.14', 'sys_platform': 'linux'})
    assert not matching[0].marker.evaluate({'python_version': '3.15', 'sys_platform': 'linux'})
    optional = [Requirement(value) for value in _extras()[extra]]
    assert any(requirement.name == name for requirement in optional)


@pytest.mark.parametrize('group', ['core', 'tutorial', 'all'])
def test_speech_does_not_force_unsupported_onnxruntime_on_python315(group):
    values = _core_dependencies() if group == 'core' else _extras()[group]
    matching = [Requirement(value) for value in values
                if Requirement(value).name == 'piper-tts']
    assert len(matching) == 1
    assert matching[0].marker.evaluate({'python_version': '3.14', 'sys_platform': 'linux'})
    assert not matching[0].marker.evaluate({'python_version': '3.15', 'sys_platform': 'linux'})


def test_python315_uses_the_numpy_wheel_era_without_changing_python314():
    numpy = [Requirement(value) for value in _core_dependencies()
             if Requirement(value).name == 'numpy']
    for python, accepted, refused in [('3.14', '1.26.4', '3.0'),
                                       ('3.15', '2.5.2', '2.4.6')]:
        applicable = [requirement for requirement in numpy
                      if requirement.marker.evaluate({'python_version': python,
                                                      'sys_platform': 'linux'})]
        assert len(applicable) == 1
        assert accepted in applicable[0].specifier
        assert refused not in applicable[0].specifier


@pytest.mark.parametrize('family', [None, 'ols'])
def test_missing_pyfixest_on_python315_does_not_offer_an_incompatible_install(
        monkeypatch, family):
    import sys

    from spacr import regression_backends as backends
    if sys.version_info[:2] != (3, 15):
        pytest.skip('actual Python 3.15 environment control')
    monkeypatch.setattr(backends, 'package_installed', lambda name: False)
    status = backends.backend_status('pyfixest', family)
    assert not status['enabled'] and not status['pip']
    assert 'Python 3.14' in status['reason']
    offer = backends.backend_install_offer('pyfixest', family)
    assert offer.action == 'elsewhere'
    assert 'Python 3.14' in offer.message


def test_manually_installed_optional_backend_remains_usable(monkeypatch):
    from spacr import regression_backends as backends
    monkeypatch.setattr(backends, 'package_installed', lambda name: True)
    assert backends.backend_status('pyfixest', 'ols')['enabled']
    assert backends.backend_install_offer('pyfixest', 'ols').action == 'ready'


def test_doctor_fallback_matches_the_declared_interpreter_range():
    from spacr import doctor
    assert doctor.FALLBACK_REQUIRES_PYTHON == _requires_python()


def test_named_missing_pymc_keeps_its_explicit_backend_identity(monkeypatch):
    import sys

    from spacr import power_model
    if sys.version_info[:2] != (3, 15):
        pytest.skip('actual Python 3.15 environment control')
    monkeypatch.setattr(power_model, '_module_installed', lambda name: False)
    with pytest.raises(power_model.PowerFitError, match='pymc.*Python 3.15.*Python 3.14'):
        power_model.resolve_backend('pymc')
    monkeypatch.setattr(power_model, '_module_installed', lambda name: True)
    assert power_model.resolve_backend('pymc') == 'pymc'


def test_python315_retains_the_central_scientific_stack():
    requirements = [Requirement(value) for value in _core_dependencies()]
    selected = {requirement.name.lower() for requirement in requirements
                if requirement.marker is None or requirement.marker.evaluate({
                    'python_version': '3.15', 'sys_platform': 'linux',
                    'platform_machine': 'x86_64', 'platform_system': 'Linux'})}
    assert {'numpy', 'pandas', 'scipy', 'scikit-image', 'scikit-learn',
            'torch', 'torchvision', 'cellpose', 'numba', 'llvmlite',
            'statsmodels', 'pywavelets', 'opencv-python-headless'} <= selected


def test_modeled_new_interpreter_keeps_optional_backend_refusals_consistent(monkeypatch):
    """The status, install offer and both classifier routes share one policy.

    Model the policy input locally; actual-interpreter controls above remain
    separate and are still required by the Python 3.15 compatibility job.
    """
    import sys
    from collections import namedtuple
    from types import SimpleNamespace

    from spacr import hyperparam, ml, power_model, regression_backends as backends
    from tests.test_cov_ml_ml_scores import COMMON, _feature_df

    version = namedtuple('Version', 'major minor micro releaselevel serial')
    monkeypatch.setattr(backends, 'sys', SimpleNamespace(
        version_info=version(3, 15, 0, 'final', 0)))
    monkeypatch.setattr(backends, 'package_installed', lambda name: False)
    monkeypatch.setattr(power_model, '_module_installed', lambda name: False)
    monkeypatch.setitem(sys.modules, 'catboost', None)
    expected = 'Python 3.15.*Python 3.14'
    for family in (None, 'ols'):
        status = backends.backend_status('pyfixest', family)
        assert not status['enabled'] and not status['pip']
        assert 'Python 3.15' in status['reason'] and 'Python 3.14' in status['reason']
        offer = backends.backend_install_offer('pyfixest', family)
        assert offer.action == 'elsewhere'
        assert 'Python 3.14' in offer.message
    with pytest.raises(power_model.PowerFitError, match='pymc.*' + expected):
        power_model.resolve_backend('pymc')
    with pytest.raises(ImportError, match='catboost.*' + expected):
        hyperparam.build_sklearn_model('catboost', {'n_estimators': 5}, n_jobs=1)
    frame = _feature_df(per_class=20)
    before = frame.copy(deep=True)
    with pytest.raises(ImportError, match='catboost.*' + expected):
        ml.ml_analysis(frame, positive_control='c2', negative_control='c1',
                       model_type='catboost', n_estimators=5, verbose=False, **COMMON)
    import pandas as pd
    pd.testing.assert_frame_equal(frame, before)
