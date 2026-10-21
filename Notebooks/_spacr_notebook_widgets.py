import ast
import copy
from pprint import pformat
from IPython.display import display

def show_advanced_settings(values, essential, defaults=None):
    try:
        import ipywidgets as widgets
    except ImportError:
        print("Expand an advanced reference group for keys/defaults; edit advanced_overrides above.")
        return None
    from spacr.settings import categories
    panels = []
    titles = []
    remaining = set(values) - set(essential)
    groups = []
    for title, members in categories.items():
        selected = [key for key in members if key in remaining]
        if selected:
            remaining.difference_update(selected)
            groups.append((title, selected))
    groups.append(("Additional settings", sorted(remaining)))
    for title, members in groups:
        if not members:
            continue
        original = copy.deepcopy({key: (defaults or values)[key] for key in members})
        editor = widgets.Textarea(value=pformat({key: values[key] for key in members}, sort_dicts=False),
                                  layout=widgets.Layout(width="100%", height="260px"))
        apply = widgets.Button(description="Apply changes")
        reset = widgets.Button(description="Reset group")
        message = widgets.Output()
        def update(_button, editor=editor, message=message, members=tuple(members), values=values):
            with message:
                message.clear_output()
                try:
                    candidate = ast.literal_eval(editor.value)
                    if not isinstance(candidate, dict) or set(candidate) != set(members):
                        raise ValueError("Keep the displayed setting names; edit their values only.")
                    for key, value in candidate.items():
                        previous = values[key]
                        if previous is not None and value is not None and not isinstance(value, type(previous)):
                            if not (isinstance(previous, float) and isinstance(value, (int, float))):
                                raise ValueError(f"{key}: preserve its value type or use None.")
                except (ValueError, SyntaxError, TypeError) as error:
                    print(error)
                else:
                    values.update(candidate)
                    print("Changes applied to the settings used by the run cell.")
        def restore(_button, editor=editor, original=original, values=values, message=message):
            values.update(copy.deepcopy(original))
            editor.value = pformat(original, sort_dicts=False)
            with message:
                message.clear_output()
                print("Group defaults restored.")
        apply.on_click(update)
        reset.on_click(restore)
        panels.append(widgets.VBox([editor, widgets.HBox([apply, reset]), message]))
        titles.append(title)
    groups_widget = widgets.Accordion(children=panels, selected_index=None)
    for index, title in enumerate(titles):
        groups_widget.set_title(index, title)
    advanced = widgets.Accordion(children=[groups_widget], selected_index=None)
    advanced.set_title(0, "Advanced settings — click to expand")
    display(advanced)
    return advanced


def show_workflow_settings_editor(settings, module):
    import json
    import ipywidgets as widgets
    from spacr.cli import apply_overrides, SettingsError
    editor = widgets.Textarea(value=json.dumps(settings, indent=2, default=str),
                              layout=widgets.Layout(width="100%", height="300px"))
    apply_button = widgets.Button(description="Apply settings")
    messages = widgets.Output()
    editable_module = module
    editable_settings = settings

    def apply_settings(_button, editor=editor, messages=messages,
                       editable_module=editable_module, editable_settings=editable_settings):
        with messages:
            messages.clear_output()
            try:
                values = json.loads(editor.value)
                if not isinstance(values, dict):
                    raise ValueError("Settings must be a JSON object.")
                if set(values) != set(editable_settings):
                    raise ValueError("Keep the current setting names; edit their values only.")
                candidate = dict(editable_settings)
                changed = [f"{key}={value!r}" if not isinstance(value, str)
                           else f"{key}={value}" for key, value in values.items()
                           if value != editable_settings[key]]
                apply_overrides(candidate, changed, editable_module)
            except (ValueError, TypeError, SettingsError) as error:
                print(error)
            else:
                editable_settings.update(candidate)
                print("Settings applied. Review them before enabling the run cell.")

    apply_button.on_click(apply_settings)
    advanced = widgets.Accordion(children=[widgets.VBox([editor, apply_button, messages])])
    advanced.set_title(0, "Advanced settings — click to expand")
    advanced.selected_index = None
    display(advanced)
    return advanced, editor, apply_button


def show_help_menu(reference):
    """Load and display only a requested section, never on initial display."""
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError:
        print("Open the linked reference for help.")
        return None
    picker = widgets.Dropdown(options=[("Choose help…", ""),
        ("Workflow and outputs", "overview"), ("API calls", "api"),
        ("Essential inputs", "essential"), ("All parameter groups", "parameters")],
        description="Help:", layout=widgets.Layout(width="95%"))
    button = widgets.Button(description="Show selected help")
    clear = widgets.Button(description="Close help")
    output = widgets.Output()

    def show(_):
        if not picker.value:
            return
        with output:
            output.clear_output()
            try:
                from pathlib import Path
                path = Path(reference)
                if path.is_file():
                    text = path.read_text(encoding="utf-8")
                else:
                    from urllib.request import urlopen
                    url = ("https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/Notebooks/"
                           + reference.removeprefix("../"))
                    text = urlopen(url, timeout=20).read().decode("utf-8")
                import re
                sections = re.split(r"(?m)^(?=## )", text)
                if picker.value == "overview":
                    selected = [sections[0]] + [part for part in sections if part.startswith("## Outputs")]
                elif picker.value == "api":
                    selected = [part for part in sections if part.startswith("## 2.")]
                elif picker.value == "essential":
                    parameters = next(part for part in sections if part.startswith("## 3."))
                    selected = []
                    for function in parameters.split("\n### ")[1:]:
                        heading, _, body = function.partition("\n")
                        essential = body.partition("#### Essential inputs")[2].partition("<details>")[0]
                        selected.append("### " + heading + "\n\n" + essential)
                else:
                    selected = [part for part in sections if part.startswith("## 3.")]
                display(Markdown("\n\n".join(selected)))
            except (OSError, ValueError, StopIteration) as error:
                print(f"Help unavailable: {error}. Open the linked reference instead.")

    button.on_click(show)
    clear.on_click(lambda _: output.clear_output())
    panel = widgets.Accordion(children=[widgets.VBox([picker, widgets.HBox([button, clear]), output])])
    panel.set_title(0, "Help — click to choose a topic")
    panel.selected_index = None
    display(panel)
    return panel


def preview_outputs(folder, limit=12):
    """List a bounded set of output files; preview only an explicitly chosen file."""
    from pathlib import Path
    import ipywidgets as widgets
    from IPython.display import display, FileLink, Image
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError("Choose an existing output folder.")
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path)
        if len(files) >= limit:
            break
    picker = widgets.Dropdown(options=[(str(path.relative_to(root)), str(path)) for path in files], description="Output:")
    button = widgets.Button(description="Preview / link")
    output = widgets.Output()

    def preview(_):
        if not picker.value:
            return
        path = Path(picker.value)
        with output:
            output.clear_output()
            if path.stat().st_size <= 5_000_000 and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                display(Image(filename=str(path), width=640))
            elif path.stat().st_size <= 5_000_000 and path.suffix.lower() == ".csv":
                import pandas as pd
                display(pd.read_csv(path, nrows=10))
            display(FileLink(str(path)))
    button.on_click(preview)
    display(widgets.VBox([picker, button, output]))
    return picker, button, output
