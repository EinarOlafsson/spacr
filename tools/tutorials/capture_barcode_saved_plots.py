"""Launch Barcode QC with an explicit, real saved-plot viewer workflow."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from stage_lesson import DEFAULT_STAGE


def main():
    stage = DEFAULT_STAGE.resolve()
    name = 'barcode_qc_saved_plot_viewer'
    if (stage / 'captures' / name).exists():
        raise FileExistsError('Preserve the previous recording; use a new named run')
    env = dict(os.environ); root = stage / 'desktop' / name
    for key, folder in [('XDG_CONFIG_HOME','config'), ('XDG_DATA_HOME','data'),
                        ('XDG_CACHE_HOME','cache')]:
        path = root / folder; path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    runtime = root / 'runtime'; runtime.mkdir(parents=True, exist_ok=True)
    env['XDG_RUNTIME_DIR'] = tempfile.mkdtemp(prefix='capture-', dir=runtime)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GVFS_DISABLE_FUSE='1', GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0',
               QT_QPA_PLATFORMTHEME='', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', GDK_SCALE='2', GDK_DPI_SCALE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = ['xvfb-run','-a','-s','-screen 0 3840x2160x24','dbus-run-session','--',
               sys.executable,str(Path(__file__).with_name('capture_refresh.py')),
               '--module','barcode_qc','--capture-name',name,'--barcode-saved-plots',
               '--stage',str(stage),'--platform','xcb','--timeout','200']
    return subprocess.run(command, env=env, timeout=260).returncode


if __name__ == '__main__':
    raise SystemExit(main())
