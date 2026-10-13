"""The optional caption GPU path moves both weights and inputs; CPU stays default."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('caption_device',
    Path(__file__).resolve().parents[3] / 'tools/translate_caption_catalogs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('device', [None, 'cpu', 'cuda'])
def test_model_and_inputs_move_together_with_cpu_default(monkeypatch, device):
    calls = []

    class Tensor:
        def to(self, target):
            calls.append(('input', target))
            return self

    class Model:
        def to(self, target):
            calls.append(('model', target))
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs['forced_bos_token_id'] == 42
            return ['translated']

    class Tokenizer:
        def __call__(self, batch, **kwargs):
            assert batch == ['A real source sentence.']
            return {'input_ids': Tensor()}

        def convert_tokens_to_ids(self, code):
            assert code == 'deu_Latn'
            return 42

        def batch_decode(self, values, **kwargs):
            return ['Ein wirklicher Ausgangssatz.']

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda count: calls.append(('threads', count)),
        inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    source = {'lessons': [{'id': 'example', 'scenes': [{'narration': 'A real source sentence.'}]}]}
    kwargs = {} if device is None else {'device': device}
    result = module.translate(source, 'de', Path('local-model'), 2, 2, **kwargs)
    assert calls == [('threads', 2), ('model', device or 'cpu'), ('input', device or 'cpu')]
    assert result['lessons'] == [{'id': 'example', 'scenes': [{'narration': 'Ein wirklicher Ausgangssatz.'}]}]
