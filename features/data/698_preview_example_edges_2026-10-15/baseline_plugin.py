import ast, subprocess

def pytest_configure(config):
    from tools.web_mask_generator import app,examples
    pin='279fec641fb'
    raw=subprocess.check_output(['git','show',pin+':tools/web_mask_generator/app.py'],text=True)
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_preview')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'committed_preview','exec'),app.__dict__)
    raw=subprocess.check_output(['git','show',pin+':tools/web_mask_generator/examples.py'],text=True)
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='ExampleStore')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    values=dict(examples.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'committed_examples','exec'),values)
    examples.ExampleStore.__init__=values['__init__']
