from pathlib import Path
import ast,difflib,hashlib,json,subprocess
root=Path.cwd();scratch=Path(__file__).parent;base='20a4313bcd9d58bc9267c1b80d3b51c02206635d';rel='tests/test_api_i18n_extractor.py';raw=subprocess.check_output(['git','show',base+':'+rel]);before=raw.decode();after=before
needle='    zoo_arrivals = {**zoo_arrivals, **late_reviewed_arrivals}\n'
insert='''    classifier_arrivals = json.loads((Path(__file__).resolve().parent
        / "data/release_contracts"
        / "561_classifier_api_arrivals_2026-10-15.json").read_text())
    assert len(classifier_arrivals) == 19
    assert {key: docs[key] for key in classifier_arrivals} == classifier_arrivals
    assert not classifier_arrivals.keys() & (
        zoo_arrivals.keys() | hidden_close.keys() | worker_arrivals.keys()
        | home_ui_arrivals.keys() | volume_arrivals.keys()
    )
    zoo_arrivals = {**zoo_arrivals, **classifier_arrivals}
'''
assert before.count(needle)==1;after=after.replace(needle,needle+insert)
needle2='        "spacr.embeddings._dino_network.Head.__init__",\n';assert before.count(needle2)==1
after=after.replace(needle2,'        "spacr.embeddings._dino_classifier_from_config.Classifier.__init__",\n'+needle2)
ast.parse(after)
patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel,n=3))
assert sum(line.startswith('@@') for line in patch.splitlines())==2
out=root/'features/data/561_api_arrival_owner_packet_2026-10-15';out.mkdir(parents=True,exist_ok=True)
(out/'extractor-test-owner.patch').write_text(patch)
fixture=(scratch/'561_classifier_api_arrivals_2026-10-15.json').read_bytes();(out/'561_classifier_api_arrivals_2026-10-15.json').write_bytes(fixture)
plan=json.loads((scratch/'run-plan.json').read_text());sha=lambda v:hashlib.sha256(v).hexdigest()
r={'status':'UNAPPLIED owner proposal; authoritative test and fixture paths untouched','owner':'Original CI','base_commit':base,'test_path':rel,'test_preimage_sha256':sha(raw),'test_proposed_sha256':sha(after.encode()),'patch':'extractor-test-owner.patch','patch_sha256':sha(patch.encode()),'fixture':'561_classifier_api_arrivals_2026-10-15.json','fixture_destination':'tests/data/release_contracts/561_classifier_api_arrivals_2026-10-15.json','fixture_sha256':sha(fixture),'normal_source_files':plan['source_files'],'eligible_arrivals':plan['new_symbols'],'unchanged_historical_surface_floor':13218,'unchanged_nonrendered_boundary':211,'new_nested_constructor':'spacr.embeddings._dino_classifier_from_config.Classifier.__init__','validation':'Two patch hunks; proposed Python AST parses; fixture19 exact normal rendered bodies. No full collector/test run or runtime behavior claim.'}
(out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
(out/'README.txt').write_text('UNAPPLIED exact-source owner proposal\n\nOriginal CI owns tests/test_api_i18n_extractor.py. No authoritative test or\nrelease-contract file was modified by this packet. The patch targets immutable\npublic20a; verify manifest preimage or review any owner-local drift first.\n\nAfter owner acceptance, copy the included JSON to\ntests/data/release_contracts/561_classifier_api_arrivals_2026-10-15.json, then\napply extractor-test-owner.patch in the owner private tree. Use git apply\n--check before git apply. If the owner test changed, reconcile these two named\nadditions normally, preserving every newer owner assertion.\n\nThe first hunk validates exactly19 source-bound helper bodies and disjoint\nidentities before subtracting them as named arrivals. The second admits only\nthe new normal-rendered local Classifier.__init__. Neither the historical\n13218 floor nor the211 excluded-symbol boundary changes.\n\nThe actual API23/all10 catalogs and Help19 arrivals are separately generated\nby the normal tools. This packet does not claim full extractor acceptance.\n')
(out/'prepare-arrival-owner-patch.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(r,indent=2))
