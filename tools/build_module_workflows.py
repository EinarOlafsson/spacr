"""Generate workflow documentation from the bundled module input/output map.

Run with --check in validation. The JSON is the editorial source; this tool
checks its live navigation and port declarations before producing any pages.
It never imports a pipeline entry point or opens a user's project.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = Path("spacr/resources/module_workflows.json")
MODES = {"classify": "classify_merged", "ml_analyze": "classify_merged",
         "parameter_sweep": "regression"}


def load(root=ROOT):
    """Read the sole editorial source for pathways and module handoffs."""
    return json.loads((root / MAP_PATH).read_text(encoding="utf-8"))


def validate(data, root=ROOT, *, live=True):
    """Reject broken links, missing sources and drift from shipped contracts."""
    if data.get("schema_version") != 1:
        raise ValueError("unsupported module workflow schema")
    modules = data["modules"]
    artifacts = data["artifacts"]
    if not modules or not artifacts or not data["connections"]:
        raise ValueError("workflow inventory is empty")
    for key, module in modules.items():
        for field in ("inputs", "outputs"):
            if not module[field] or set(module[field]) - artifacts.keys():
                raise ValueError(f"{key}: invalid {field}")
        api = root / (module["api_module"].replace(".", "/") + ".py")
        if not api.is_file() and not (api.with_suffix("") / "__init__.py").is_file():
            raise ValueError(f"{key}: missing API source {api}")
        if not module["guidance"].strip():
            raise ValueError(f"{key}: missing handoff guidance")
        if module.get("api_entry"):
            prefix, _, name = module["api_entry"].rpartition(".")
            if (prefix != module["api_module"] or module["home"] is not None
                    or module["parent"] is not None):
                raise ValueError(f"{key}: invalid API-only route")
            if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == name for node in ast.parse(api.read_text()).body):
                raise ValueError(f"{key}: missing API entry point")
    seen = set()
    for edge in data["connections"]:
        pair = edge["from"], edge["to"]
        if pair in seen or pair[0] not in modules or pair[1] not in modules:
            raise ValueError(f"invalid or duplicate connection {pair}")
        seen.add(pair)
        if not edge["artifacts"] or set(edge["artifacts"]) - artifacts.keys():
            raise ValueError(f"invalid connection artifacts: {pair}")
        if not edge["handoff"].strip():
            raise ValueError(f"connection needs explicit handoff: {pair}")
    for key, route in data["pathways"].items():
        if route["home_app"] not in modules or not route["steps"]:
            raise ValueError(f"{key}: missing start")
        if modules[route["steps"][0]["module"]]["home"] != route["home_app"]:
            raise ValueError(f"{key}: first step disagrees with Home entry")
        previous = set()
        for step in route["steps"]:
            module = step["module"]
            if module not in modules or not step["action"].strip():
                raise ValueError(f"{key}: invalid step {module}")
            if set(step["after"]) - previous:
                raise ValueError(f"{key}: dependency precedes its input")
            previous.add(module)
        if set(route.get("alternatives", [])) - modules.keys():
            raise ValueError(f"{key}: unknown alternative")
    for key, lesson in data["tutorials"].items():
        for field in ("description", "introduction", "conclusion"):
            if not str(lesson.get(field, "")).strip():
                raise ValueError(f"{key}: missing tutorial {field}")
        if set(lesson["modules"]) - modules.keys():
            raise ValueError(f"{key}: unknown module")
        if set(lesson["pathways"]) - data["pathways"].keys():
            raise ValueError(f"{key}: unknown pathway")
    for source in data["source_contracts"]:
        text = (root / source["path"]).read_text(encoding="utf-8")
        for literal in source["literals"]:
            if literal not in text:
                raise ValueError(f"I/O contract moved in {source['path']}: {literal}")
    if not live:
        return
    sys.path.insert(0, str(root))
    import spacr
    if Path(spacr.__file__).resolve().parent != root / "spacr":
        raise ValueError(f"wrong checkout imported: {spacr.__file__}")
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr import ports
    from spacr.validate import APP_FUNCTIONS
    from spacr.qt import app
    from spacr.qt.widgets.fold_strip import folded_modules
    from spacr.qt.screens.settings_model import _APP_API_MODULE
    names = {row[0]: row[1] for row in app.APPS}
    names.update({key: row[0] for key, row in folded_modules().items()})
    if set(modules) != set(names) | set(MODES) | set(APP_FUNCTIONS) | set(ports.PORTS):
        raise ValueError("workflow modules differ from the live registry")
    parents = {child: host for host, children in app.folded_children().items()
               for child in children}
    parents.update(MODES)
    tiles = {row[0] for row in app.tiled_apps()}
    for key, module in modules.items():
        if key not in names and key not in MODES:
            if not module.get("api_entry") or module["api_entry"] != APP_FUNCTIONS.get(key):
                raise ValueError(f"{key}: API-only entry point drift")
        elif module.get("api_entry"):
            raise ValueError(f"{key}: live GUI route marked API-only")
        if key in names and module["name"] != names[key]:
            raise ValueError(f"{key}: display name drift")
        if module["parent"] != parents.get(key):
            raise ValueError(f"{key}: nested route drift")
        expected_home = parents.get(key, key if key in tiles else None)
        if module["home"] != expected_home:
            raise ValueError(f"{key}: Home route drift")
        if key in _APP_API_MODULE:
            api = "spacr." + _APP_API_MODULE[key].replace("/", ".")
            if module["api_module"] != api:
                raise ValueError(f"{key}: Home API target drift")
        current = (json.loads(json.dumps(asdict(ports.PORTS[key])))
                   if key in ports.PORTS else None)
        if module.get("declared_ports") != current:
            raise ValueError(f"{key}: declared input/output ports drift")
    for key, route in data["pathways"].items():
        if route["home_app"] not in tiles:
            raise ValueError(f"{key}: first module has no Home tile")


def _heading(text, marker):
    return f"{text}\n{marker * len(text)}\n\n"


def module_rst(data, key):
    """Render the same input/output facts for overview and API pages."""
    module = data["modules"][key]
    parts = [f".. _workflow-module-{key}:\n\n",
             _heading(module["name"], "~"), module["guidance"] + "\n\n"]
    parent = module["parent"]
    if module.get("api_entry"):
        parts.append(f"**Use from Python:** :func:`{module['api_entry']}`. "
                     "This API-only workflow has no Home tile or menu entry.\n\n")
    elif parent:
        host = data['modules'][parent]
        prefix = "Help search → " if host['home'] is None else ""
        parts.append(f"**Open:** {prefix}{host['name']} → {module['name']}.\n\n")
    elif module["home"]:
        parts.append(f"**Open:** Home → {module['name']}.\n\n")
    else:
        parts.append("**Open:** the application's Help/tools menus.\n\n")
    parts.append("Inputs and outputs below include conditional alternatives. "
                 "The guidance and handoff notes say which route applies.\n\n")
    for field, title in (("inputs", "Inputs"), ("outputs", "Outputs")):
        parts.append(f"**{title}**\n\n")
        for artifact in module[field]:
            item = data["artifacts"][artifact]
            location = item["location"].replace("*", "\\*")
            parts.append(f"* **{item['title']}** — {location}\n")
            if item["tables"]:
                parts.append("  Relevant tables, depending on the route: " + ", ".join(
                    f"``{name}``" for name in item["tables"]) + ".\n")
            if item["columns"]:
                parts.append("  Relevant columns, depending on the route: " + ", ".join(
                    f"``{name}``" for name in item["columns"]) + ".\n")
        parts.append("\n")
    for direction, title, peer in (("to", "Before this module", "from"),
                                    ("from", "After this module", "to")):
        edges = [edge for edge in data["connections"] if edge[direction] == key]
        if edges:
            parts.append(f"**{title}**\n\n")
            for edge in edges:
                other = edge[peer]
                name = data["modules"][other]["name"]
                parts.append(f"* :ref:`{name} <workflow-module-{other}>`: {edge['handoff']}\n")
            parts.append("\n")
    api_path = module["api_module"].replace(".", "/")
    parts.append(f":doc:`API reference </api/{api_path}/index>`.\n\n")
    if module["lesson"]:
        parts.append(f"`Module tutorial <tutorials/#lesson={module['lesson']}>`__.\n\n")
    return "".join(parts)


def lesson_document(data, key, *, translate=None):
    """Write narration from the same artifacts and handoffs used by the API."""
    tr = translate if translate is not None else lambda text: text
    tutorial = data["tutorials"][key]
    number, slug = key.split("_", 1)
    scenes = [{"visual": "home", "narration": tr(tutorial["introduction"]),
               "hold_after": 0.7, "related_lessons": ["05_home"]}]
    detailed = key == "79_module_inputs_outputs"
    included = set(tutorial["modules"])
    introduced_artifacts = set()
    for module_key in tutorial["modules"]:
        module = data["modules"][module_key]
        parent = module["parent"]
        if module.get("api_entry"):
            opening = tr(
                "Use {name} from Python through {api}. "
                "This API-only workflow has no Home tile or menu entry."
            ).format(name=module['name'], api=module['api_entry'])
        elif parent:
            host = data['modules'][parent]
            template = ("Open {host} from Home, then choose {name}." if host['home']
                        else "Open {host} through Help search, then choose {name}.")
            opening = tr(template).format(host=host['name'], name=module['name'])
        elif module["home"]:
            opening = tr("Open {name} from Home.").format(name=module['name'])
        else:
            opening = tr("Find {name} in the application's Help or tools menus.").format(
                name=module['name'])
        parts = [opening, tr(module["guidance"])]
        for field, title in (("inputs", "Input data"), ("outputs", "Output data")):
            parts.append(tr(title) + ": " + "; ".join(
                tr(data["artifacts"][artifact]["title"]) for artifact in module[field]) + ".")
            if detailed:
                for artifact in module[field]:
                    if artifact in introduced_artifacts:
                        continue
                    introduced_artifacts.add(artifact)
                    item = data["artifacts"][artifact]
                    parts.append(tr(item["title"]) + ": " + tr(item["location"]))
                    if item["tables"]:
                        parts.append(tr("Relevant tables depend on the selected route: {tables}.").format(
                            tables=", ".join(item["tables"])))
                    if item["columns"]:
                        parts.append(tr("Relevant columns depend on the selected route: {columns}.").format(
                            columns=", ".join(item["columns"])))
        links = {module["lesson"]} if module.get("lesson") else set()
        for edge in data["connections"]:
            if module_key not in (edge["from"], edge["to"]):
                continue
            if not detailed and not (
                {edge["from"], edge["to"]} <= included or edge["to"] == "regression"
            ):
                continue
            producer = data["modules"][edge["from"]]
            consumer = data["modules"][edge["to"]]
            parts.append(tr("{producer} to {consumer}: {handoff}").format(
                producer=producer['name'], consumer=consumer['name'], handoff=tr(edge['handoff'])))
            links.update(row["lesson"] for row in (producer, consumer) if row.get("lesson"))
        scenes.append({"visual": "module_" + module_key,
                       "narration": " ".join(parts), "hold_after": 0.7,
                       "related_lessons": sorted(links)})
    scenes.append({"visual": "home_summary", "narration": tr(tutorial["conclusion"]),
                   "hold_after": 0.7})
    return {"id": key, "number": int(number), "slug": slug,
            "title": tr(tutorial["title"]), "series": 1, "app_key": None,
            "section": tr("Workflows"), "description": tr(tutorial["description"]),
            "objectives": [tr("Choose a starting module from the data you already have."),
                           tr("Identify what each module reads and writes."),
                           tr("Follow the linked module lessons and API contracts for the next step.")],
            "prerequisite": tr("Install spaCR and open Home. This lesson explains navigation and data handoffs; the linked module lessons provide the worked data examples. Inputs and outputs include conditional alternatives, as explained for each module."),
            "scenes": scenes}


def outputs(data):
    """Return deterministic documentation and tutorial handoff artifacts."""
    intro = (_heading("Choose a workflow after installation", "=")
             + "Start on Home with the first module for your experiment. "
               "Use that module's example-data control when available, inspect "
               "the inputs and preview, then run a bounded example before your own data.\n\n"
             + "These routes and the API handoffs share the bundled module workflow map. "
               "A walkthrough explains the steps; it does not run an experiment automatically.\n\n")
    sections = [intro]
    for key, route in data["pathways"].items():
        sections += [f".. _workflow-{key}:\n\n", _heading(route["title"], "-")]
        for step in route["steps"]:
            module = data["modules"][step["module"]]
            sections.append(f"#. :ref:`{module['name']} <workflow-module-{step['module']}>`: {step['action']}\n")
        sections.append("\n")
        if route.get("alternatives"):
            links = [f":ref:`{data['modules'][key]['name']} <workflow-module-{key}>`"
                     for key in route["alternatives"]]
            sections.append("Other routes for the appropriate question: " + ", ".join(links) + ".\n\n")
        if route.get("note"):
            sections.append(route["note"] + "\n\n")
    sections.append(_heading("Module inputs, outputs and next steps", "-"))
    sections.extend(module_rst(data, key) for key in data["modules"])
    result = {Path("docs/source/workflows.rst"): "".join(sections)}
    by_api = {}
    for key, module in data["modules"].items():
        by_api.setdefault(module["api_module"], []).append(key)
    for api, keys in by_api.items():
        text = _heading("Workflow inputs and outputs", "-")
        for key in keys:
            text += module_rst(data, key).split("\n\n", 1)[1].replace(
                "<tutorials/", "<https://einarolafsson.github.io/spacr/tutorials/")
        result[Path(f"docs/source/_generated/module_workflows/{api}.rst")] = text
    for key, lesson in data["tutorials"].items():
        payload = dict(lesson, schema_version=1, source=str(MAP_PATH),
                       module_contracts={k: data["modules"][k] for k in lesson["modules"]},
                       pathway_steps={k: data["pathways"][k] for k in lesson["pathways"]},
                       connections=[e for e in data["connections"]
                                    if e["from"] in lesson["modules"] and e["to"] in lesson["modules"]])
        result[Path(f"tools/tutorials/workflows/{key}.json")] = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        result[Path(f"tools/tutorials/lessons/{key}.json")] = json.dumps(
            lesson_document(data, key), ensure_ascii=False, indent=2) + "\n"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = load()
    validate(data)
    stale = []
    for path, content in outputs(data).items():
        destination = ROOT / path
        if args.check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
                stale.append(str(path))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
    if stale:
        print("Workflow artifacts need regeneration:\n" + "\n".join(stale))
        return 1
    print(f"Workflow map: {len(data['modules'])} modules, {len(data['pathways'])} pathways verified")
    return 0


def prepare_jinja(env, root=ROOT):
    """Check generated prose and expose its include paths to AutoAPI."""
    data = load(root)
    validate(data, root, live=False)
    for path, content in outputs(data).items():
        destination = root / path
        if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
            raise ValueError(f"stale workflow artifact: {path}; run build_module_workflows.py")
    env.globals["spacr_workflow_includes"] = {
        row["api_module"]: f"/_generated/module_workflows/{row['api_module']}.rst"
        for row in data["modules"].values()
    }


if __name__ == "__main__":
    raise SystemExit(main())
