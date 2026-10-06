from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import sys
import urllib.request

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
out = scratch / '567-videomae-preparation-r1'
out.mkdir(exist_ok=False)
model_dir = out / 'model'
model_dir.mkdir()
assert not torch.cuda.is_available()
assert importlib.metadata.version('transformers') == '4.48.3'
assert Path(sys.prefix) == scratch / '567-video-model-env-py313-r1'
sha = lambda data: hashlib.sha256(data).hexdigest()
url = 'https://huggingface.co/api/models/MCG-NJU/videomae-base-finetuned-kinetics?blobs=true'
metadata_bytes = urllib.request.urlopen(url, timeout=60).read()
(out / 'primary-model-metadata.json').write_bytes(metadata_bytes)
metadata = json.loads(metadata_bytes)
revision = metadata['sha']
assert revision == '488eb9a0565f257b32866000305c8178965eb9f6'
assert metadata.get('gated') is False
assert metadata['cardData']['license'] == 'cc-by-nc-4.0'
files = {row['rfilename']: row for row in metadata['siblings']}
acquired = {}
for name in ('config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors'):
    source_url = 'https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/resolve/' + revision + '/' + name
    digest = hashlib.sha256()
    count = 0
    target = model_dir / name
    with urllib.request.urlopen(source_url, timeout=120) as response, target.open('xb') as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            digest.update(block)
            count += len(block)
    assert count == files[name]['size'], name
    if files[name].get('lfs'):
        assert digest.hexdigest() == files[name]['lfs']['sha256'], name
    else:
        payload = target.read_bytes()
        blob = hashlib.sha1(('blob ' + str(len(payload)) + '\0').encode() + payload).hexdigest()
        assert blob == files[name]['blobId'], (name, blob, files[name])
    acquired[name] = {'primary_URL': source_url, 'bytes': count, 'sha256': digest.hexdigest()}
    print('PASS acquired exact pinned official model file', name, count, digest.hexdigest(), flush=True)

config = json.loads((model_dir / 'config.json').read_text())
assert config['num_frames'] == 16 and config['num_channels'] == 3 and config['image_size'] == 224
assert config['hidden_size'] == 768 and config['patch_size'] == 16
assert len(config['id2label']) == 400
processor = VideoMAEImageProcessor.from_pretrained(model_dir, local_files_only=True)
model, loading = VideoMAEForVideoClassification.from_pretrained(model_dir, local_files_only=True, use_safetensors=True, output_loading_info=True)
assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'], loading
model = model.cpu().eval()
state_digest = hashlib.sha256()
for name, tensor in sorted(model.state_dict().items()):
    array = tensor.detach().cpu().contiguous().numpy()
    state_digest.update(name.encode() + b'\0' + str(array.dtype).encode() + b'\0' + json.dumps(array.shape).encode() + b'\0' + array.tobytes())

torch.set_num_threads(2)
y, x = np.mgrid[0:224, 0:224]
video = [np.stack([((x + 2 * y + i * 7 + channel * 19) % 256).astype(np.uint8) for channel in range(3)], axis=-1) for i in range(16)]
original = np.stack(video)
np.save(out / 'synthetic-16-frame-RGB-uint8.npy', original, allow_pickle=False)
inputs = processor(video, return_tensors='pt')
assert tuple(inputs['pixel_values'].shape) == (1, 16, 3, 224, 224)
with torch.inference_mode():
    reply = model(**inputs, output_hidden_states=True)
    features = model.fc_norm(reply.hidden_states[-1].mean(dim=1))
    reconstructed = model.classifier(features)
assert tuple(reply.logits.shape) == (1, 400) and tuple(features.shape) == (1, 768)
assert torch.isfinite(reply.logits).all() and torch.isfinite(features).all()
assert torch.equal(reply.logits, reconstructed)
np.save(out / 'synthetic-CPU-pooled-features.npy', features.numpy(), allow_pickle=False)
np.save(out / 'synthetic-CPU-kinetics-logits.npy', reply.logits.numpy(), allow_pickle=False)
np.save(out / 'synthetic-CPU-processor-pixels.npy', inputs['pixel_values'].numpy(), allow_pickle=False)
report = {'item':567, 'preparation_only':True, 'passed':True, 'finished_UTC':datetime.now(timezone.utc).isoformat(),
    'official_model':'MCG-NJU/videomae-base-finetuned-kinetics', 'actual_revision':revision,
    'model_weights_license':'cc-by-nc-4.0', 'acquired_original_files':acquired,
    'actual_strict_loading_info':loading, 'ordered_loaded_model_state_sha256':state_digest.hexdigest(),
    'private_environment_prefix':sys.prefix, 'shared_base_torch_used_without_modification':True,
    'actual_package_versions':{name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','safetensors','tokenizers','numpy')},
    'CPU_preflight_input_scope':'Explicit deterministic synthetic 16-frame RGB fixture; no acquired microscopy input or human annotation.',
    'actual_CPU_input_shape':list(inputs['pixel_values'].shape), 'actual_CPU_features_shape':list(features.shape), 'actual_CPU_logits_shape':list(reply.logits.shape),
    'exact_reconstruction_of_real_model_pool_norm_classifier_logits':True,
    'current_application_timelapse_source_sha256':sha(Path('spacr/timelapse.py').read_bytes()),
    'no_application_backend_feature_integration_GPU_training_or_biological_event_accuracy_claim':True,
    'Home_source_integration_and_real_annotated_microscopy_GPU_acceptance_remain_required':True,
    'script_sha256':sha(Path(__file__).read_bytes()),
    'outputs':{path.name:{'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())} for path in sorted(out.glob('*.npy'))}}
(out / 'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS pinned real pretrained VideoMAE strict CPU load, finite 768 features/400 logits and exact classifier reconstruction; synthetic preflight only.',flush=True)
