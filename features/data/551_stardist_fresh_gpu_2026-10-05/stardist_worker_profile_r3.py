import os
from pathlib import Path
import runpy

profile = Path(os.environ['SPACR_STARDIST_PROFILE_ROOT'])
profile.mkdir(parents=True, exist_ok=False)
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                         ' --xla_dump_hlo_as_text --xla_dump_to=' + str(profile / 'hlo')).strip()
import tensorflow as tf

tf.debugging.set_log_device_placement(True)
tf.profiler.experimental.start(str(profile / 'trace'), options=tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=2, python_tracer_level=0, device_tracer_level=1))
try:
    runpy.run_path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005/spacr/_segmentation_backends.py',
                  run_name='__main__')
finally:
    tf.profiler.experimental.stop()
