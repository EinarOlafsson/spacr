import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'tools/tutorials/capture_illumination.py'
module=ast.parse(source.read_text())
helper=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_viewer_window')
namespace={}
exec(compile(ast.Module(body=[helper],type_ignores=[]),str(source),'exec'),namespace)
find=namespace['_viewer_window']

class ViewerDetection(unittest.TestCase):
    def query(self,props,mapped=True):
        return lambda argv: props[argv[2]] if argv[0]=='xprop' else ('Map State: IsViewable' if mapped else 'Map State: IsUnMapped')
    def properties(self,pid=123,title='illumination_qc_plate1.png',klass='Eog'):
        return f'_NET_WM_PID(CARDINAL) = {pid}\n_NET_WM_NAME(UTF8_STRING) = "{title}"\nWM_CLASS(STRING) = "eog", "{klass}"\n'
    def test_real_owner_with_unnamed_tree_uses_actual_utf8_title(self):
        self.assertEqual(find('  0xA001 (has no name): ()\n',123,self.query({'0xA001':self.properties()})),('0xA001','illumination_qc_plate1.png'))
    def test_same_named_other_process_cannot_satisfy_detection(self):
        self.assertIsNone(find(' 0xa001 "illumination_qc_plate1.png"\n',123,self.query({'0xa001':self.properties(pid=987)})))
    def test_unmapped_window_rejected(self):
        self.assertIsNone(find(' 0xa001 "image"\n',123,self.query({'0xa001':self.properties()},False)))
    def test_viewer_own_title_is_observed_not_synthesized(self):
        self.assertEqual(find(' 0xa001 "old title"\n',123,self.query({'0xa001':self.properties(title='Image Viewer')})),('0xa001','Image Viewer'))
    def test_missing_owner_or_title_rejected(self):
        self.assertIsNone(find(' 0xa001 "image"\n',123,self.query({'0xa001':'WM_CLASS(STRING) = "eog", "Eog"'})))
        self.assertIsNone(find(' 0xa001 "image"\n',123,self.query({'0xa001':self.properties(title='')})))
    def test_wrong_application_same_pid_rejected(self):
        props=self.properties().replace('"eog", "Eog"','"other", "Other"')
        self.assertIsNone(find(' 0xa001 "image"\n',123,self.query({'0xa001':props})))

if __name__=='__main__':unittest.main()
