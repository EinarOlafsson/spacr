"""Exact reviewed Home spelling variant; captions and Latin rejection retained."""
import ast
from pathlib import Path
import re
import pytest

@pytest.fixture
def speech():
    source=Path(__file__).resolve().parents[1]/'tools/render_all_voices.py'
    if not source.is_file():source=Path(__file__).with_name('render_all_voices.py')
    module=ast.parse(source.read_text())
    function=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='track_speech_text')
    namespace={'re':re}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['track_speech_text']

@pytest.mark.parametrize('language,expected',[('ja','ドーズ レスポンス'),('zh-CN','剂量反应')])
def test_reviewed_spaced_and_hyphenated_dose_response_have_same_speech(speech,language,expected):
    for caption in ['Dose Response','Dose-Response']:
        assert speech('05_home',language,None,caption,caption)==expected

@pytest.mark.parametrize('language',['ja','zh-CN'])
def test_unreviewed_latin_still_refused(speech,language):
    with pytest.raises(ValueError,match='Unreviewed Latin'):
        speech('05_home',language,None,'Dose ResponseNew','Dose ResponseNew')

@pytest.mark.parametrize('language',['ja','zh-CN'])
def test_spoken_premise_must_match_display(speech,language):
    with pytest.raises(ValueError,match='premise changed'):
        speech('05_home',language,None,'Dose Response','Dose-Response')
