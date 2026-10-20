"""Real offline fake-wheel updates exercise staging and atomic rollback."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import venv
import zipfile

import pytest

from spacr import install_cleanup as ic


def _wheel(folder, version, *, broken=False):
    folder.mkdir(exist_ok=True)
    info = f'spacr-{version}.dist-info'
    files = {
        'spacr/__init__.py': 'from ._version import __version__\n',
        'spacr/_version.py': f'__version__ = "{version}"\n',
        'spacr/qt.py': ('raise RuntimeError("broken staged Qt import")\n' if broken else
                         f'def main():\n    print("{version}")\n'),
        info+'/METADATA': f'Metadata-Version: 2.1\nName: spacr\nVersion: {version}\n',
        info+'/WHEEL': 'Wheel-Version: 1.0\nGenerator: real-offline-703-contract\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        info+'/entry_points.txt': '[console_scripts]\nspacr-probe = spacr.qt:main\n',
    }
    rows = []
    for name,value in files.items():
        data=value.encode();digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        rows.append(f'{name},sha256={digest},{len(data)}')
    files[info+'/RECORD']='\n'.join(rows+[info+'/RECORD,,'])+'\n'
    path=folder/f'spacr-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path,'w') as archive:
        for name,value in files.items():archive.writestr(name,value)
    return path


def acknowledging_spawn(launched):
    """A relaunch stand-in that answers as the verified new desktop would.

    The helper now waits for the restarted GUI to confirm its main window
    (:func:`spacr.install_cleanup._launch_updated_desktop`); this records the
    launch and writes the receipt the real window writes, echoing the request.
    """
    def spawn(argv, environment, cwd):
        launched.append(argv)
        folder, binding = json.loads(environment["SPACR_UPDATE_LAUNCH"])
        channel = ic._UpdateLaunchHandshake(folder, binding)
        expected = channel._read("launch-request.json")["details"]
        channel._write("launch-result.json", "ready", details={
            "version": expected["version"], "executable": expected["executable"],
            "module": os.path.join(expected["root"], "spacr", "__init__.py")})
        return None
    return spawn


def _snapshot(root):
    return {str(path.relative_to(root)): ('link',os.readlink(path)) if path.is_symlink()
            else ('file',hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob('*') if path.is_symlink() or path.is_file()}


@pytest.fixture
def installation(tmp_path,monkeypatch):
    monkeypatch.delenv('PYTHONPATH',raising=False)
    prefix=tmp_path/'installation'/'venv'
    venv.EnvBuilder(with_pip=False).create(prefix)
    python=prefix/'bin/python'
    old=_wheel(tmp_path/'old-wheels','1.5.1.4')
    result=subprocess.run([sys.executable,'-I','-B','-m','pip','--python',str(python),
                           '--isolated','--no-cache-dir','install','--no-index','--no-deps',str(old)],
                           capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    site=Path(subprocess.check_output([str(python),'-I','-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip())
    import packaging
    shutil.copytree(Path(packaging.__file__).parent,site/'packaging',ignore=shutil.ignore_patterns('__pycache__'))
    assert (site/'spacr/qt.py').read_text(encoding="utf-8")=='def main():\n    print("1.5.1.4")\n'
    identity=ic._retiring_runtime_identity(str(prefix))
    assert identity['verified'] is True and identity['version']=='1.5.1.4'
    assert Path(identity['module'])==(site/'spacr/__init__.py').resolve()
    retained=prefix/'my extra package data'/'original.bin';retained.parent.mkdir();retained.write_bytes(b'keep exactly')
    record=ic.InstallRecord(kind='installer',layout='macos-runtime',platform='macos',root=str(prefix.parent),
                            python=str(python),version='1.5.1.4',running=True)
    machine=ic._Machine(platform='macos',environ={'HOME':str(tmp_path/'private-user')},
                         fs_root=str(tmp_path),executable=str(python),running_prefix=str(prefix))
    return record,machine,prefix,site


def _installer(tmp_path,wheel,*,fail_install=False,fail_check=False):
    script=tmp_path/'fake_uv.py'
    script.write_text('import subprocess,sys\n'
        +'args=sys.argv[1:]\npython=args[args.index("--python")+1]\n'
        +f'failed_install={fail_install!r}; failed_check={fail_check!r}\n'
        +'if args[1]=="install" and failed_install: sys.exit(9)\n'
        +'if args[1]=="check" and failed_check: sys.exit(8)\n'
        +f'base={[sys.executable,"-I","-B","-m","pip"]!r}+["--python",python,"--isolated","--no-cache-dir"]\n'
        +f'command=base+(["check"] if args[1]=="check" else ["install","--no-index","--no-deps","--upgrade",{str(wheel)!r}])\n'
        +'sys.exit(subprocess.run(command).returncode)\n', encoding="utf-8")
    uv=tmp_path/'uv'
    uv.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' -I -B '+shlex.quote(str(script))+' "$@"\n', encoding="utf-8")
    uv.chmod(0o755)
    return uv


@pytest.mark.posix_only
@pytest.mark.parametrize('failure',[None,'install','dependencies','import','final_identity','restart','stage_identity','stage_json','final_json','final_version','final_import'])
def test_offline_staging_keeps_previous_runtime_and_rolls_back_after_activation(installation,tmp_path,failure):
    record,machine,prefix,_=installation
    if failure is None:
        os.symlink(str(prefix/'my extra package data/original.bin'), prefix/'bin/retained-data-link')
    before=_snapshot(prefix)
    wheel=_wheel(tmp_path/'new-wheels','1.5.1.5',broken=failure=='import')
    uv=_installer(tmp_path,wheel,fail_install=failure=='install',fail_check=failure=='dependencies')
    command=[str(uv),'pip','install','--upgrade','--python',record.python,'spacr']
    calls=[]
    def run(argv):
        calls.append(argv)
        identity = len(argv)>3 and argv[1:3]==['-I','-c'] and 'from spacr._version' in argv[-2]
        if identity and argv[0]!=record.python:
            if failure=='stage_identity': return 1,'forced staged identity failure'
            if failure=='stage_json': return 0,'not JSON'
        if identity and argv[0]==record.python:
            if failure=='final_identity': return 1,'forced final identity failure'
            if failure=='final_json': return 0,'not JSON'
            if failure=='final_version': return 0,json.dumps({'verified':True,'version':'1.5.1.6'})
        if failure=='final_import' and argv[0]==record.python and argv[-1]=='import spacr.qt':
            return 1,'forced activated import failure'
        return ic._run(argv,timeout=30)
    def activate(version):
        assert version=='1.5.1.5'
        if failure=='restart':raise OSError('forced restart failure')
    if failure:
        with pytest.raises((RuntimeError,ValueError,OSError)):
            ic._transactional_runtime_update(record,'1.5.1.5',command,machine=machine,runner=run,activate=activate)
        assert _snapshot(prefix)==before
        assert not list(prefix.parent.glob('.spacr-runtime-update-*'))
    else:
        version,output=ic._transactional_runtime_update(record,'1.5.1.5',command,machine=machine,runner=run,activate=None)
        assert version=='1.5.1.5' and 'Previous runtime retained' in output
        folders=list(prefix.parent.glob('.spacr-runtime-update-*'));assert len(folders)==1
        state=json.loads((folders[0]/'transaction.json').read_text(encoding="utf-8"))
        assert state['state']=='committed' and state['verified_version']=='1.5.1.5'
        assert _snapshot(Path(state['rollback_runtime']))==before
        assert (prefix/'my extra package data'/'original.bin').read_bytes()==b'keep exactly'
        probe=subprocess.run([str(prefix/'bin/spacr-probe')],capture_output=True,text=True,timeout=10)
        assert probe.returncode==0 and probe.stdout.strip()=='1.5.1.5',probe.stderr
    assert not (prefix.parent/'.venv.spacr-update-lock').exists()
    installs=[argv for argv in calls if len(argv)>2 and argv[1:3]==['pip','install']]
    assert installs and all(argv[argv.index('--python')+1]!=record.python for argv in installs)


@pytest.mark.posix_only
@pytest.mark.parametrize('unsupported',['system-site','conda','absolute-pth','editable','user-data','egg-link'])
def test_unsupported_environment_is_refused_before_any_package_mutation(installation,unsupported,monkeypatch):
    record,machine,prefix,site=installation
    if unsupported=='system-site':(prefix/'pyvenv.cfg').write_text('include-system-site-packages = true\n', encoding="utf-8")
    elif unsupported=='conda':(prefix/'conda-meta').mkdir()
    elif unsupported=='absolute-pth':(site/'old-path.pth').write_text(str(prefix)+'\n', encoding="utf-8")
    elif unsupported=='egg-link':(site/'linked.egg-link').write_text('/external/source\n', encoding="utf-8")
    elif unsupported=='editable':(site/'spacr-1.5.1.4.dist-info/direct_url.json').write_text('{"dir_info":{"editable":true}}', encoding="utf-8")
    else:monkeypatch.setattr(ic,'_user_data',lambda machine:[str(prefix/'my extra package data')])
    before=_snapshot(prefix)
    with pytest.raises(ValueError):
        ic._transactional_runtime_update(record,'1.5.1.5',['unused','pip','install','--python',record.python,'spacr'],
            machine=machine,runner=lambda args:pytest.fail('No package operation for unsupported environment'))
    assert _snapshot(prefix)==before
    assert not list(prefix.parent.glob('.spacr-runtime-update-*'))


@pytest.mark.posix_only
def test_unknown_target_is_refused_without_package_mutation(installation):
    record,machine,prefix,_=installation
    before=_snapshot(prefix)
    with pytest.raises(ValueError,match='does not target'):
        ic._transactional_runtime_update(record,'1.5.1.5',['unused','pip','install','--python','/unrelated/python','spacr'],
            machine=machine,runner=lambda argv:pytest.fail('Wrong target cannot invoke package tool'))
    assert _snapshot(prefix)==before
    assert not list(prefix.parent.glob('.spacr-runtime-update-*'))


@pytest.mark.posix_only
def test_failed_restoration_keeps_original_bytes_journal_and_lock(installation,tmp_path,monkeypatch):
    record,machine,prefix,_=installation
    before=_snapshot(prefix)
    uv=_installer(tmp_path,_wheel(tmp_path/'new-wheels','1.5.1.5'))
    exchange=ic._runtime_exchange
    operations=[]
    def faulty_exchange(first,second):
        operations.append((first,second))
        if len(operations)==2:raise OSError('forced restoration failure')
        exchange(first,second)
    monkeypatch.setattr(ic,'_runtime_exchange',faulty_exchange)
    def run(argv):
        if argv[0]==record.python and 'from spacr._version' in argv[-2]:return 1,'forced final failure'
        return ic._run(argv,timeout=30)
    with pytest.raises(RuntimeError,match='previous runtime retained'):
        ic._transactional_runtime_update(record,'1.5.1.5',[str(uv),'pip','install','--python',record.python,'spacr'],
            machine=machine,runner=run)
    folders=list(prefix.parent.glob('.spacr-runtime-update-*'));assert len(folders)==1
    state=json.loads((folders[0]/'transaction.json').read_text(encoding="utf-8"))
    assert state['state']=='verifying'
    assert _snapshot(Path(state['rollback_runtime']))==before
    assert (prefix.parent/'.venv.spacr-update-lock').is_dir()
    exchange(str(prefix),state['rollback_runtime'])
    assert _snapshot(prefix)==before


def test_darwin_atomic_binding_uses_swap_flag_without_emulating_native_acceptance(monkeypatch,tmp_path):
    import ctypes
    calls=[]
    class Operation:
        def __call__(self,*args):calls.append(args);return 0
    class Library:
        renamex_np=Operation()
    monkeypatch.setattr(ctypes,'CDLL',lambda *args,**kwargs:Library())
    monkeypatch.setattr(ic.sys,'platform','darwin')
    ic._runtime_exchange(str(tmp_path/'first'),str(tmp_path/'second'))
    assert calls==[(os.fsencode(tmp_path/'first'),os.fsencode(tmp_path/'second'),2)]


def test_unsupported_atomic_exchange_has_no_nonatomic_fallback(monkeypatch,tmp_path):
    monkeypatch.setattr(ic.sys,'platform','unsupported')
    with pytest.raises(OSError,match='not available'):
        ic._runtime_exchange(str(tmp_path/'first'),str(tmp_path/'second'))
    assert not list(tmp_path.iterdir())


def test_failed_atomic_exchange_leaves_the_current_prefix_in_place(tmp_path):
    current=tmp_path/'current';current.mkdir();(current/'original').write_bytes(b'preserved')
    with pytest.raises(OSError):ic._runtime_exchange(str(current),str(tmp_path/'missing'))
    assert (current/'original').read_bytes()==b'preserved'
