"""Standard-library import guards on every supported Python version."""
import pkgutil
import sys
import sysconfig


def stdlib_names():
    """Return builtins and stdlib modules without importing their contents."""
    names = getattr(sys, 'stdlib_module_names', None)
    if names is not None:
        return set(names)
    paths = [path for path in (sysconfig.get_path('stdlib'),
                              sysconfig.get_config_var('DESTSHARED')) if path]
    return (set(sys.builtin_module_names) | {'__future__'} |
            {module.name for module in pkgutil.iter_modules(paths)
             if module.name not in ('site-packages', 'dist-packages')})
