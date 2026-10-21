import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux proc identities and POSIX update leases")

def dead_owner():
    child=subprocess.Popen([sys.executable,'-c','import sys;print("ready",flush=True);sys.stdin.read()'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,start_new_session=True)
    assert child.stdout.readline().strip()=='ready'
    owner=ic._transaction_owner(child.pid)
    child.communicate('',timeout=10)
    return owner


@pytest.fixture
def retired(tmp_path):
    home=tmp_path/'home'
    root=home/'.local/share/spacr'
    root.mkdir(parents=True)
    tx=root/'.spacr-runtime-update-real'
    backup=tx/'runtime'
    (backup/'bin').mkdir(parents=True)
    (backup/'bin/python').write_text('private executable')
    (backup/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    meta=backup/'lib/python3.12/site-packages/spacr-1.5.1.5.dist-info/METADATA'
    meta.parent.mkdir(parents=True)
    meta.write_text('Name: spacr\nVersion: 1.5.1.5\n')
    tx.chmod(0o700)
    details=backup.stat()
    data={'schema':2,'family':'posix-runtime','prefix':str(root/'venv'),'rollback_runtime':str(backup),
          'owner':dead_owner(),'old_identity':[details.st_dev,details.st_ino],
          'old_inventory':ic._runtime_recovery_inventory(str(backup)),'state':'committed',
          'new_identity':[details.st_dev,details.st_ino+1], 'new_inventory':{'.':{'kind':'directory','mode':0o755}},
          'requested_version':'1.5.1.5','verified_version':'1.5.1.5'}
    journal=tx/'transaction.json';journal.write_text(json.dumps(data));journal.chmod(0o600)
    keep=root/'keep-user-file.txt';keep.write_text('unknown user data')
    machine=ic._Machine(platform='linux',environ={'HOME':str(home)},fs_root=str(tmp_path),running_prefix='')
    return root,tx,backup,journal,data,keep,machine


def test_orphan_committed_backup_is_temporary_and_purged_without_root_deletion(retired):
    root,tx,backup,journal,data,keep,machine=retired
    assert not (root/'venv').exists()
    assert ('temporary',str(tx),True) in ic._uninstall_data_candidates(machine)
    assert str(tx) in ic._purge_targets(machine)
    rows=ic._uninstall_preview('strict-retired',system=machine)['cleanup']
    assert any(row['path']==str(tx) and row['category']=='temporary' and row['selectable'] for row in rows)
    report=ic._purge(machine,[str(tx)])
    assert report.ok and not tx.exists()
    assert keep.read_text()=='unknown user data' and root.exists()
    assert not (root/'.venv.spacr-update-lock').exists()


@pytest.mark.parametrize('problem',['uncommitted','live_owner','malformed_owner','missing_owner','wrong_prefix','wrong_family','wrong_inode','changed_inventory','extra_file','unverified_launch','missing_new_inventory','public_journal','linked_journal','linked_backup','linked_transaction'])
def test_invalid_or_live_transaction_is_never_authorized(retired,problem,tmp_path):
    root,tx,backup,journal,data,keep,machine=retired
    if problem=='uncommitted':data['state']='launch_unconfirmed'
    elif problem=='live_owner':data['owner']=ic._transaction_owner()
    elif problem=='malformed_owner':data['owner']['started']=['invalid']
    elif problem=='missing_owner':data.pop('owner')
    elif problem=='wrong_prefix':data['prefix']=str(root/'other')
    elif problem=='wrong_family':data['family']='windows-private-pip'
    elif problem=='wrong_inode':data['old_identity'][1]+=1
    elif problem=='changed_inventory':(backup/'bin/python').write_text('changed')
    elif problem=='extra_file':(tx/'unknown.txt').write_text('user data')
    elif problem=='unverified_launch':data.pop('verified_version')
    elif problem=='missing_new_inventory':data.pop('new_inventory')
    elif problem=='linked_backup':
        moved=tmp_path/'outside-backup';backup.rename(moved);backup.symlink_to(moved,target_is_directory=True)
    elif problem=='linked_transaction':
        moved=tmp_path/'outside-transaction';tx.rename(moved);tx.symlink_to(moved,target_is_directory=True)
    if problem=='linked_journal':
        moved=tmp_path/'outside-journal';journal.rename(moved);journal.symlink_to(moved)
    else:
        journal.write_text(json.dumps(data))
        if problem=='public_journal':journal.chmod(0o644)
    assert str(tx) not in ic._retired_update_data(machine)
    report=ic.RemovalReport(record=None)
    assert ic._delete_retired_update(str(tx),[],report,machine)
    assert not report.ok and tx.exists() and keep.exists()


def test_live_process_using_backup_working_directory_is_kept(retired):
    root,tx,backup,journal,data,keep,machine=retired
    child=subprocess.Popen(['/bin/cat'],cwd=backup,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        assert str(tx) not in ic._retired_update_data(machine)
    finally:child.communicate(b'',timeout=10)
    assert str(tx) in ic._retired_update_data(machine)


def test_opted_out_backup_is_untouched(retired):
    root,tx,backup,journal,data,keep,machine=retired
    unrelated=root/'unrelated-cache';unrelated.mkdir()
    report=ic._purge(machine,[str(unrelated)])
    assert report.ok and tx.exists() and keep.exists()


def test_authentic_dead_lock_is_removed_but_live_and_unknown_lease_is_kept(retired):
    root,tx,backup,journal,data,keep,machine=retired
    lock=root/'.venv.spacr-update-lock'
    program='import importlib.util,sys; s=importlib.util.spec_from_file_location("lease_source",sys.argv[1]); m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m._take_lease_dir(sys.argv[2]);print("ready",flush=True);sys.stdin.read()'
    child=subprocess.Popen([sys.executable,'-c',program,str(ic.__file__),str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,start_new_session=True)
    assert child.stdout.readline().strip()=='ready'
    assert str(lock) not in ic._retired_update_data(machine)
    child.communicate('',timeout=10)
    assert str(lock) in ic._retired_update_data(machine)
    report=ic._purge(machine,[str(lock)])
    assert report.ok and not lock.exists() and tx.exists()
    lock.mkdir();(lock/'unknown').write_text('never remove')
    assert str(lock) not in ic._retired_update_data(machine)
    assert (lock/'unknown').read_text()=='never remove'


@pytest.mark.parametrize('candidate,relative',[(False,False),(False,True),(True,False),(True,True)])
def test_unreadable_unrelated_cwd_does_not_block_but_candidate_command_is_protected(retired,candidate,relative,tmp_path):
    root,tx,backup,journal,data,keep,machine=retired
    program='import ctypes,sys;assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0;print("ready",flush=True);sys.stdin.read()'
    argv0=('python' if candidate else '(sd-pam)') if relative else str(backup/'bin/python') if candidate else sys.executable
    child=subprocess.Popen([argv0,'-c',program],executable=sys.executable,cwd=tmp_path,
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        with pytest.raises(PermissionError):os.readlink('/proc/'+str(child.pid)+'/cwd')
        assert (str(tx) in ic._retired_update_data(machine)) is not candidate
    finally:child.communicate('',timeout=10)


def test_durable_backup_binding_survives_removed_private_base_interpreter(retired,tmp_path):
    root,tx,backup,journal,data,keep,machine=retired
    base=tmp_path/'installer-private-python'
    base.write_text('original private interpreter')
    executable=backup/'bin/python';executable.unlink();executable.symlink_to(base)
    data['old_inventory']=ic._runtime_recovery_inventory(str(backup))
    journal.write_text(json.dumps(data));journal.chmod(0o600)
    base.unlink()
    assert executable.is_symlink() and not executable.exists()
    assert str(tx) in ic._retired_update_data(machine)
    report=ic._purge(machine,[str(tx)])
    assert report.ok and not tx.exists() and keep.read_text()=='unknown user data'


def test_custom_root_losing_discovery_never_falls_through_to_generic_deletion(retired):
    root,tx,backup,journal,data,keep,machine=retired
    custom=root.parents[2]/"custom-spacr"
    root.rename(custom)
    tx=custom/tx.name
    journal=tx/"transaction.json"
    data["prefix"]=str(custom/"venv")
    data["rollback_runtime"]=str(tx/"runtime")
    journal.write_text(json.dumps(data))
    bootstrap=custom/"bootstrap/uv"
    bootstrap.parent.mkdir()
    bootstrap.write_text("installer bootstrap marker")
    launcher=Path(machine.home)/".local/bin/spacr"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('#!/bin/sh\nexec "'+str(custom/'venv/bin/python')+'" "$@"\n')
    assert str(tx) in ic._retired_update_data(machine)
    launcher.unlink()
    bootstrap.unlink()
    bootstrap.parent.rmdir()
    assert str(custom) not in ic._retired_update_roots(machine)
    report=ic._purge(machine,[str(tx)])
    assert not report.ok and report.failed
    assert tx.exists() and (custom/"keep-user-file.txt").read_text()=="unknown user data"
