"""Explicit public documentation for services in selected private modules.

The service module stays private for Python packaging, while its documented
entry points are deliberately exposed in Sphinx, localization and Help search.
This policy does not expose private helpers within those modules.
"""

EXPLICIT_MODULES = frozenset({"spacr._starplast"})


def documented_tree(names):
    """Include the ancestors required to reach source-documented API objects.

    This is a documentation projection, independent of Python's star-import
    export list. An undocumented ancestor may need a signature, but gains no
    invented prose or translation record.
    """
    result = set()
    for name in names:
        parts = name.split(".")
        result.update(".".join(parts[:length]) for length in range(1, len(parts) + 1))
    return frozenset(result)


def source_page_policy(name, obj, projected):
    """Render canonical source documents without exposing imported copies.

    Keep AutoAPI's import and inheritance boundaries. Ordinary source-defined
    objects in the translation inventory must remain reachable even when a
    module's star-import export list does not name them.
    """
    if name not in projected or obj.imported or obj.inherited:
        return None
    leaf = name.rsplit(".", 1)[-1]
    if (getattr(obj, "type", "") not in {"module", "package"}
            and leaf.startswith("_") and not leaf.endswith("__")):
        return None
    return False


def public_symbol(name: str) -> bool:
    """Accept ordinary public names and public members of explicit modules."""
    for module in EXPLICIT_MODULES:
        if name == module:
            return True
        if name.startswith(module + "."):
            name = name[len(module) + 1:]
            break
    return all(part == "__main__" or not part.startswith("_")
               for part in name.split("."))


def explicit_page_policy(name: str):
    """Unskip only an explicit module; retain member and import filtering."""
    if name in EXPLICIT_MODULES:
        return False
    return None


def source_supplements(app, module, *, root, documents):
    """Render source-defined API branches absent from AutoAPI's object map.

    Only current canonical documentation keys are eligible. Signatures and
    locations come from the original AST, never imports or guessed aliases.
    Nested helpers retain their separate existing directive and inventory.
    """
    import ast
    import copy
    import re
    from pathlib import Path
    import build_documentation_i18n as builder

    path = Path(root).joinpath(*module.split('.')).with_suffix('.py')
    if not path.is_file():
        path = Path(root).joinpath(*module.split('.'), '__init__.py')
    if not path.is_file():
        return ''
    mapped = app.env.autoapi_all_objects
    candidates = {key for key in documents
                  if key.startswith(module + '.') and key not in mapped}
    if not candidates:
        return ''
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    entries = {}
    imports = {}
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    for node in builder._module_scope_nodes(tree.body):
        if isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                parents = package.split('.')
                prefix = '.'.join(parents[:len(parents) - node.level + 1]
                                  + ([prefix] if prefix else []))
            for alias in node.names:
                if alias.name != '*':
                    imports[alias.asname or alias.name] = prefix + '.' + alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = (
                    alias.name if alias.asname else alias.name.split('.')[0])

    class AnnotationNames(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in imports:
                return ast.copy_location(ast.parse(imports[node.id], mode='eval').body, node)
            return node

    def remember(node, owner, kind):
        key = owner + '.' + node.name
        if key in candidates:
            entries.setdefault(key, []).append((node, kind))

    for node in builder._module_scope_nodes(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            remember(node, module, 'function')
        elif isinstance(node, ast.ClassDef):
            remember(node, module, 'class')
            owner = module + '.' + node.name
            for child in builder._module_scope_nodes(node.body):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorators = {ast.unparse(value) for value in child.decorator_list}
                    kind = 'property' if 'property' in decorators else 'method'
                    remember(child, owner, kind)
    if not entries:
        return ''
    rendered = []
    for key, variants in sorted(entries.items()):
        kinds = {kind for node, kind in variants}
        if len(kinds) != 1:
            raise ValueError(f'Conditional API changes object kind: {key}')
        kind = next(iter(kinds))
        relative = key[len(module) + 1:]
        signatures = []
        source_bodies = {}
        for node, _kind in variants:
            source = builder._autoapi_class_doc(node) if kind == 'class' else builder._clean_doc(node)
            body = source
            if kind == 'class':
                constructors = [child for child in builder._module_scope_nodes(node.body)
                                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
                constructor_docs = list(dict.fromkeys(builder._clean_doc(child)
                                                      for child in constructors
                                                      if child.name == '__init__'
                                                      and builder._clean_doc(child)))
                if not constructor_docs:
                    constructor_docs = list(dict.fromkeys(builder._clean_doc(child)
                                                          for child in constructors
                                                          if child.name == '__new__'
                                                          and builder._clean_doc(child)))
                body = '\n\n'.join(part for part in [builder._clean_doc(node), *constructor_docs]
                                   if part)
            if source:
                source_bodies.setdefault(source, body)
            function = node
            if kind == 'class':
                constructors = [child for child in builder._module_scope_nodes(node.body)
                                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and child.name in {'__init__', '__new__'}]
                if not constructors:
                    continue
                function = next((child for child in constructors
                                 if child.name == '__init__'), constructors[0])
            if kind == 'property':
                signatures.append((0, relative))
                continue
            arguments = copy.deepcopy(function.args)
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                             *([arguments.vararg] if arguments.vararg else []),
                             *([arguments.kwarg] if arguments.kwarg else [])):
                if argument.annotation is not None:
                    argument.annotation = AnnotationNames().visit(argument.annotation)
            if kind in {'method', 'class'}:
                positional = arguments.posonlyargs or arguments.args
                if positional and positional[0].arg in {'self', 'cls'}:
                    positional.pop(0)
            signature = relative + '(' + ast.unparse(arguments) + ')'
            if kind != 'class' and function.returns is not None:
                signature += ' -> ' + ast.unparse(AnnotationNames().visit(copy.deepcopy(function.returns)))
            signatures.append((len(arguments.posonlyargs) + len(arguments.args)
                               + len(arguments.kwonlyargs), signature))
        if not signatures:
            signatures.append((0, relative))
        primary = max(signatures, key=lambda item: item[0])[1]
        body = '\n\n'.join(source_bodies.values())
        if ' '.join(body.split()) != ' '.join(documents[key].split()):
            raise ValueError(f'Source-branch prose differs from canonical documentation: {key}')
        body = re.sub(r':(class|func|meth|attr|exc):`(~?)([A-Za-z_]\w*)`',
                      lambda match: (f':{match[1]}:`~{imports[match[3]]}`'
                                     if match[3] in imports else match[0]), body)
        lines = body.splitlines() + ['']
        if 'autodoc-process-docstring' in app.events.events:
            app.emit('autodoc-process-docstring', kind, key, None, None, lines)
        rendered.extend([f'.. py:{kind}:: {primary}', f'   :module: {module}', ''])
        rendered.extend('   ' + line if line else '' for line in lines)
        rendered.append('')
        alternatives = dict.fromkeys(signature for _rank, signature in signatures
                                     if signature != primary)
        rendered.extend(f'   ``{signature}``' for signature in alternatives)
        rendered.extend(f'   ``{path.relative_to(root).as_posix()}:{node.lineno}``'
                        for node, _kind in variants)
        rendered.append('')
    return '\n'.join(rendered)


def prepare_jinja(env, *, root, documents):
    """Give normal module templates the canonical source-branch renderer."""
    from functools import partial

    env.filters['spacr_source_supplements'] = partial(
        source_supplements, root=root, documents=documents)
