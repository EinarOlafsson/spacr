import inspect
from spacr.qt import theme
source=inspect.getsource(theme.apply_stylesheet_per_window)
guard='''        if (_sheets_itself_before_it_shows(window)
                and not window.isVisible()):
            continue
'''
assert source.count(guard)==1
exec(compile(source.replace(guard,''),'frozen-baseline-apply','exec'),vars(theme))
