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
