"""Pure protocol regression; no Qt, model, recorder or biological acceptance."""
from pathlib import Path
import importlib.util
import os
from types import SimpleNamespace
import unittest

path=Path(os.environ.get('PLAQUE_ADAPTER_PROPOSAL',str(Path(__file__).with_name('adapter-v3.py'))))
spec=importlib.util.spec_from_file_location('proposal',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Labels:
    shape=(1,1)
    dtype='int32'
    def __init__(self,body=b'\x01\x00\x00\x00'):self.body=body
    def tobytes(self):return self.body


class Signal:
    def __init__(self):self.slots=[]
    def connect(self,f):self.slots.append(f)
    def disconnect(self,f):self.slots.remove(f)
    def emit(self,r):
        for f in self.slots:f(dict(r))


class Protocol(unittest.TestCase):
    def panel(self):
        button=lambda:SimpleNamespace(isVisible=lambda:True,isEnabled=lambda:True)
        p=SimpleNamespace(_run_token=0,_plaque_result=None,busy=False,preview_ready=Signal(),_run_btn=button(),_cancel_btn=button(),_status=SimpleNamespace(text=lambda:'Actual guarded failure'))
        p.settings={'src':'/private','plaque_model':'declared-model','diameter':30,'flow_threshold':.4,'CP_prob':0,'colony_counting':False};p.current_path=lambda:'/private/declared.tif';p.current_settings=lambda:dict(p.settings);p.mode=lambda:'plaque';p.preview_running=lambda:p.busy
        self.addCleanup(lambda:self.assertEqual(p.preview_ready.slots,[]))
        return p

    def result(self,p,path='/private/declared.tif'):
        r={'path':path,'labels':Labels(),'count':0};p._plaque_result=r;p.preview_ready.emit(r);return r

    def observe(self,p,click,settle=None):
        now=[0.]
        def tick(delay):
            now[0]+=delay
            if settle:settle()
        return module.observe_current_plaque_preview(p,click,tick,1,clock=lambda:now[0])

    def test_synchronous_completion_before_click_returns(self):
        p=self.panel()
        def click(_):p._run_token+=1;self.result(p)
        self.assertIs(self.observe(p,click),p._plaque_result)

    def test_deferred_start_waits_for_actual_token_then_retirement(self):
        p=self.panel();steps=[0]
        def settle():
            steps[0]+=1
            if steps[0]==1:p._run_token+=1;p.busy=True
            if steps[0]==2:self.result(p)
            if steps[0]==3:p.busy=False
        self.assertIs(self.observe(p,lambda _:None,settle),p._plaque_result)
        self.assertEqual(steps[0],3)

    def test_unrequested_replacement_result_is_not_accepted(self):
        p=self.panel()
        def click(_):p._run_token+=2;self.result(p)
        with self.assertRaisesRegex(ValueError,'replaced or cancelled'):self.observe(p,click)

    def test_preexisting_result_without_new_pass_times_out_and_cancels(self):
        p=self.panel();p._plaque_result={'path':p.current_path(),'labels':Labels(),'count':0};buttons=[]
        with self.assertRaises(TimeoutError):self.observe(p,buttons.append)
        self.assertEqual(buttons,[p._run_btn,p._cancel_btn])

    def test_new_pass_without_accepted_result_reports_actual_status(self):
        p=self.panel()
        def click(_):p._run_token+=1
        with self.assertRaisesRegex(ValueError,'Actual guarded failure'):self.observe(p,click)

    def test_other_image_completion_is_refused(self):
        p=self.panel()
        def click(_):p._run_token+=1;self.result(p,'/private/other.tif')
        with self.assertRaisesRegex(ValueError,'current Plaque pass'):self.observe(p,click)

    def test_cancelled_token_cannot_become_completion(self):
        p=self.panel()
        def click(_):p._run_token+=1;p.busy=True
        def settle():p._run_token+=1;p.busy=False
        with self.assertRaisesRegex(ValueError,'replaced or cancelled'):self.observe(p,click,settle)


    def test_signal_payload_with_different_label_values_is_refused(self):
        p=self.panel()
        def click(_):
            p._run_token+=1
            p._plaque_result={'path':p.current_path(),'labels':Labels(),'count':1}
            p.preview_ready.emit({'path':p.current_path(),'labels':Labels(b'other'),'count':1})
        with self.assertRaisesRegex(ValueError,'completion masks differ'):self.observe(p,click)

    def test_in_place_recorded_model_identity_change_is_refused(self):
        p=self.panel();p.settings['model_metadata']={'schema_version':1,'settings':{'plaque_model':{'requested_value':'declared-model','key':'declared-model','version':1,'version_status':'pinned','sha256':'a'*64,'kind':'cellpose','filename':'model.pt'}}}
        def click(_):p._run_token+=1;p.busy=True
        def settle():p.settings['model_metadata']['settings']['plaque_model']['sha256']='b'*64;p.busy=False;self.result(p)
        with self.assertRaisesRegex(ValueError,'settings changed'):self.observe(p,click,settle)

    def test_input_source_change_is_refused(self):
        p=self.panel()
        def click(_):p._run_token+=1;p.settings['src']='/other';self.result(p)
        with self.assertRaisesRegex(ValueError,'settings changed'):self.observe(p,click)

    def test_all_four_model_and_inference_controls_are_bound(self):
        for key,value in [('plaque_model','another-model'),('diameter',31),('flow_threshold',.5),('CP_prob',1)]:
            with self.subTest(control=key):
                p=self.panel()
                def click(_):p._run_token+=1;p.settings[key]=value;self.result(p)
                with self.assertRaisesRegex(ValueError,'settings changed'):self.observe(p,click)

    def test_colony_branch_is_refused_before_any_click(self):
        p=self.panel();p.settings['colony_counting']=True;clicks=[]
        with self.assertRaisesRegex(ValueError,'ordinary Plaque mode only'):self.observe(p,clicks.append)
        self.assertEqual(clicks,[])


if __name__=='__main__':unittest.main()
