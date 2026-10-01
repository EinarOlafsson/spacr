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
        input_ids = set()
        for source in route.get("inputs", []):
            identity = source.get("id")
            if (not isinstance(identity, str) or not identity.strip()
                    or identity in input_ids
                    or not str(source.get("title", "")).strip()
                    or not str(source.get("description", "")).strip()
                    or not source.get("artifacts")
                    or set(source["artifacts"]) - artifacts.keys()
                    or not source.get("targets")
                    or set(source["targets"]) - previous):
                raise ValueError(f"{key}: invalid external input {identity!r}")
            input_ids.add(identity)
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
        _validate_stories(data)
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
    _validate_stories(data)


def _validate_stories(data):
    """Check written stories after route drift, which names its own cause."""
    for key, lesson in data["tutorials"].items():
        if "scenes" in lesson:
            _validate_story(data, key)


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


DETAILED_LESSON = "79_module_inputs_outputs"
_SENT_ON = ("{peer} picks up {artifacts} next.",
            "You can also take {artifacts} on to {peer}.",
            "{peer} reads {artifacts} as well.")
_BROUGHT_IN = ("{peer} can supply {artifacts}.",
               "If you start in {peer}, it provides {artifacts}.",
               "{artifacts_from} {peer} work here too.")
_SENT_AGAIN = ("{peer} uses them too.", "They also feed {peer}.",
               "{peer} is another place they go.")
_BROUGHT_AGAIN = ("{peer} can supply them as well.", "They can also come from {peer}.")


def _scene_edges(data, key, module_key):
    """Return the handoffs a lesson scene must tell, in map order."""
    included = set(data["tutorials"][key]["modules"])
    return [edge for edge in data["connections"]
            if module_key in (edge["from"], edge["to"])
            and (key == DETAILED_LESSON or {edge["from"], edge["to"]} <= included
                 or edge["to"] == "regression")]


def _validate_story(data, key):
    """A written story must still name every route, host and handoff peer."""
    lesson = data["tutorials"][key]
    story = lesson["scenes"]
    if set(story) != set(lesson["modules"]):
        raise ValueError(f"{key}: story scenes differ from the lesson modules")
    for module_key, text in story.items():
        module = data["modules"][module_key]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{key}: empty story scene {module_key}")
        required = [module["name"]]
        if module.get("api_entry"):
            required.append(module["api_entry"])
        if module["parent"]:
            required.append(data["modules"][module["parent"]]["name"])
        for edge in _scene_edges(data, key, module_key):
            peer = edge["to"] if edge["from"] == module_key else edge["from"]
            required.append(data["modules"][peer]["name"])
        missing = [name for name in required if name.lower() not in text.lower()]
        if missing:
            raise ValueError(f"{key}: story scene {module_key} omits {missing}")


def _spoken(title):
    """Lower a title's first word for mid-sentence speech, keeping acronyms."""
    first = title.split()[0]
    if first[:1].isupper() and first[1:] == first[1:].lower():
        return title[:1].lower() + title[1:]
    return title


def _items(keys, data, tr):
    names = [tr("the " + _spoken(data["artifacts"][key]["title"])) for key in keys]
    if len(names) == 1:
        return names[0]
    template = ("{items}, and {last}" if any(" and " in data["artifacts"][key]["title"]
                                            for key in keys) else "{items} and {last}")
    return tr(template).format(items=", ".join(names[:-1]), last=names[-1])


def _opening(data, module, tr):
    parent = module["parent"]
    if module.get("api_entry"):
        return tr(
            "Use {name} from Python through {api}. "
            "This API-only workflow has no Home tile or menu entry."
        ).format(name=module['name'], api=module['api_entry'])
    if parent:
        host = data['modules'][parent]
        template = ("Open {host} from Home, then choose {name}." if host['home']
                    else "Open {host} through Help search, then choose {name}.")
        return tr(template).format(host=host['name'], name=module['name'])
    if module["home"]:
        return tr("Open {name} from Home.").format(name=module['name'])
    return tr("Find {name} in the application's Help or tools menus.").format(
        name=module['name'])


def _told_scene(data, key, module_key, tr, introduced):
    """Tell one module as a step in the data's journey, from the shared map."""
    module = data["modules"][module_key]
    detailed = key == DETAILED_LESSON
    parts = [_opening(data, module, tr), tr(module["guidance"])]
    for field, template, detail in (
            ("inputs", "{name} reads {items}.", "Look for {artifact} here:"),
            ("outputs", "From those, {name} produces {items}.", "You'll find {artifact} here:")):
        parts.append(tr(template).format(name=module["name"],
                                         items=_items(module[field], data, tr)))
        if not detailed:
            continue
        for artifact in module[field]:
            if artifact in introduced:
                continue
            introduced.add(artifact)
            item = data["artifacts"][artifact]
            parts.append(tr(detail).format(artifact=_items([artifact], data, tr))
                         + " " + tr(item["location"]))
            if item["tables"]:
                parts.append(tr("Depending on the route, the tables that matter are {tables}.").format(
                    tables=", ".join(item["tables"])))
            if item["columns"]:
                parts.append(tr("The columns to keep an eye on are {columns}.").format(
                    columns=", ".join(item["columns"])))
    counts = {"sent": 0, "again": 0, "brought": 0, "brought_again": 0}
    previous = None
    for edge in _scene_edges(data, key, module_key):
        artifacts = _items(edge["artifacts"], data, tr)
        direction = "sent" if edge["from"] == module_key else "brought"
        peer = data["modules"][edge["to" if direction == "sent" else "from"]]["name"]
        # "them" may only point back to the handoff spoken just before.
        if previous == (direction, edge["artifacts"]):
            kind = "again" if direction == "sent" else "brought_again"
        else:
            kind = direction
        previous = direction, edge["artifacts"]
        variants = {"sent": _SENT_ON, "again": _SENT_AGAIN, "brought": _BROUGHT_IN,
                    "brought_again": _BROUGHT_AGAIN}[kind]
        template = variants[counts[kind] % len(variants)]
        counts[kind] += 1
        spoken = tr(template).format(peer=peer, artifacts=artifacts,
                                     artifacts_from=tr("{artifacts} from").format(
                                         artifacts=artifacts))
        parts.append(spoken[:1].upper() + spoken[1:] + " " + tr(edge["handoff"]))
    return " ".join(parts)


def lesson_document(data, key, *, translate=None):
    """Write narration from the same artifacts and handoffs used by the API.

    A lesson may carry a written story for each module scene; validation keeps
    that story naming the same routes and handoff peers as the map. Otherwise
    the scene is told from the module's guidance, artifacts and handoffs.
    """
    tr = translate if translate is not None else lambda text: text
    tutorial = data["tutorials"][key]
    number, slug = key.split("_", 1)
    scenes = [{"visual": "home", "narration": tr(tutorial["introduction"]),
               "hold_after": 0.7, "related_lessons": ["05_home"]}]
    story = tutorial.get("scenes", {})
    introduced_artifacts = set()
    for module_key in tutorial["modules"]:
        module = data["modules"][module_key]
        if module_key in story:
            narration = tr(story[module_key])
        else:
            narration = _told_scene(data, key, module_key, tr, introduced_artifacts)
        links = {module["lesson"]} if module.get("lesson") else set()
        for edge in _scene_edges(data, key, module_key):
            links.update(data["modules"][end]["lesson"] for end in (edge["from"], edge["to"])
                         if data["modules"][end].get("lesson"))
        scenes.append({"visual": "module_" + module_key,
                       "narration": narration, "hold_after": 0.7,
                       "related_lessons": sorted(links)})
    scenes.append({"visual": "home_summary", "narration": tr(tutorial["conclusion"]),
                   "hold_after": 0.7})
    return {"id": key, "number": int(number), "slug": slug,
            "title": tr(tutorial["title"]), "series": 1, "app_key": None,
            "section": tr("Workflows"), "description": tr(tutorial["description"]),
            "objectives": [tr("Choose a starting module from the data you already have."),
                           tr("Identify what each module reads and writes."),
                           tr("Follow the linked module lessons and API contracts for the next step.")],
            "prerequisite": tr("Install spaCR and open Home. Choose the starting data for your workflow, then follow the linked module lessons for controls and example data."),
            "scenes": scenes}


def outputs(data):
    """Return deterministic documentation and tutorial handoff artifacts."""
    intro = (_heading("Choose a workflow after installation", "=")
             + "Start on Home with the first module for your experiment. "
               "Use that module's example-data control when available, inspect "
               "the inputs and preview, then run a bounded example before your own data.\n\n"
             + "These routes and the API handoffs share the bundled module workflow map. "
               "A walkthrough explains the steps; it does not run an experiment automatically.\n\n"
             + "Open **Pipeline overviews** to choose a pathway. Its graph includes external "
               "input nodes that explain the files you supply; these are not runnable modules. "
               "Persistent explanation cards describe modules and connections below the graph, "
               "with API links at the end of module cards. Select a graph element to outline its "
               "card, and drag the blue divider to adjust the space between graph and explanations. "
               "**Start example** opens the pathway's first real module; **Walkthrough** keeps "
               "the route available afterward. Alternative input and annotation branches do not "
               "require you to execute every listed step.\n\n")
    sections = [intro]
    for key, route in data["pathways"].items():
        sections += [f".. _workflow-{key}:\n\n", _heading(route["title"], "-")]
        if route.get("description"):
            sections.append(route["description"] + "\n\n")
        if route.get("inputs"):
            sections.append("**Choose an input route:**\n\n")
            for source in route["inputs"]:
                targets = ", ".join(
                    f":ref:`{data['modules'][target]['name']} <workflow-module-{target}>`"
                    for target in source["targets"])
                sections.append(f"* **{source['title']}**: {source['description']} "
                                f"Continue with {targets}.\n")
            sections.append("\n**Module steps and alternatives:**\n\n")
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
        tutorial_root = "../" * (len(api.split(".")) + 1) + "tutorials/"
        for key in keys:
            text += module_rst(data, key).split("\n\n", 1)[1].replace(
                "<tutorials/", "<" + tutorial_root)
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
