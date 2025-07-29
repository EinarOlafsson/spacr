spacr.gui_elements
==================

.. py:module:: spacr.gui_elements








Module Contents
---------------

.. py:data:: fig
   :value: None


.. py:function:: restart_gui_app(root)

   Restarts the SpaCr GUI application by destroying the current root window
   and launching a fresh instance.

   :param root: The current Tkinter root window to be destroyed.
   :type root: tk.Tk

   .. note:: The new instance is launched by importing and invoking `gui_app()`.


.. py:function:: create_menu_bar(root)

   Creates a top-level menu bar for the SpaCr GUI containing shortcuts to all
   major application modules and help resources.

   :param root: The root window where the menu bar will be attached.
   :type root: tk.Tk

   Adds:
       - A 'SpaCr Applications' menu with links to:
         'Mask', 'Measure', 'Classify', 'ML Analyze', 'Map Barcodes',
         'Regression', 'Activation', and 'Recruitment'.
       - A Help option linking to the online documentation.
       - An Exit option to quit the application.


.. py:function:: set_element_size()

   Calculates and returns standardized UI element dimensions
   based on the current screen size.

   :returns:

             A dictionary with element dimensions including:
                 - 'btn_size' (int): Size of buttons.
                 - 'bar_size' (int): Height of progress bars.
                 - 'settings_width' (int): Width of the settings panel.
                 - 'panel_width' (int): Width of the plotting panel.
                 - 'panel_height' (int): Height of the bottom control panel.
   :rtype: dict


.. py:function:: set_dark_style(style, parent_frame=None, containers=None, widgets=None, font_family='OpenSans', font_size=12, bg_color='black', fg_color='white', active_color='blue', inactive_color='dark_gray')

   Applies a dark theme to the SpaCr GUI using the provided styling options.

   :param style: The ttk style instance to configure.
   :type style: ttk.Style
   :param parent_frame: The top-level container to apply styles to.
   :type parent_frame: tk.Widget, optional
   :param containers: Additional containers (ttk.Frame or tk.Frame) to style.
   :type containers: list, optional
   :param widgets: List of individual widgets to apply colors and fonts.
   :type widgets: list, optional
   :param font_family: Font family for all labels and buttons.
   :type font_family: str
   :param font_size: Font size for all text elements.
   :type font_size: int
   :param bg_color: Background color.
   :type bg_color: str
   :param fg_color: Foreground/text color.
   :type fg_color: str
   :param active_color: Highlight or selected color.
   :type active_color: str
   :param inactive_color: Secondary background color.
   :type inactive_color: str

   :returns: Style parameters used, including resolved font and color values.
   :rtype: dict


.. py:class:: spacrFont(font_name, font_style, font_size=12)

   
   Initializes the FontLoader class.

   Args:
   - font_name: str, the name of the font (e.g., 'OpenSans').
   - font_style: str, the style of the font (e.g., 'Regular', 'Bold').
   - font_size: int, the size of the font (default: 12).


   .. py:attribute:: font_name


   .. py:attribute:: font_style


   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_path


   .. py:method:: get_font_path(font_name, font_style)

      Returns the font path based on the font name and style.

      Args:
      - font_name: str, the name of the font.
      - font_style: str, the style of the font.

      Returns:
      - str, the path to the font file.



   .. py:method:: load_font()

      Loads the font into Tkinter.



   .. py:method:: get_font(size=None)

      Returns the font in the specified size.

      Args:
      - size: int, the size of the font (optional).

      Returns:
      - tkFont.Font object.



.. py:class:: spacrContainer(parent, orient=tk.VERTICAL, bg=None, *args, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom container widget that manages multiple resizable panes arranged
   either vertically or horizontally, separated by draggable sashes.

   :param parent: The parent widget.
   :type parent: tk.Widget
   :param orient: Orientation of the layout ('tk.VERTICAL' or 'tk.HORIZONTAL'). Default is vertical.
   :type orient: str
   :param bg: Background color of the container and sashes. Defaults to 'lightgrey'.
   :type bg: str

   Initialize the spacrContainer with the specified orientation and background color.

   :param parent: Parent widget.
   :type parent: tk.Widget
   :param orient: Layout orientation (tk.VERTICAL or tk.HORIZONTAL).
   :type orient: str
   :param bg: Background color. Defaults to 'lightgrey'.
   :type bg: str, optional


   .. py:attribute:: orient
      :value: 'vertical'



   .. py:attribute:: bg
      :value: 'lightgrey'



   .. py:attribute:: sash_thickness
      :value: 10



   .. py:attribute:: panes
      :value: []



   .. py:attribute:: sashes
      :value: []



   .. py:method:: add(widget, stretch='always')

      Add a new widget as a pane to the container.

      :param widget: Widget to add.
      :type widget: tk.Widget
      :param stretch: Stretch policy (currently unused).
      :type stretch: str



   .. py:method:: create_sash()

      Create a draggable sash between panes.



   .. py:method:: reposition_panes()

      Reposition panes and sashes within the container based on orientation.



   .. py:method:: on_configure(event)

      Event handler triggered on container resize.

      :param event: Tkinter event object.
      :type event: tk.Event



   .. py:method:: on_enter_sash(event)

      Change sash color on mouse enter.

      :param event: Tkinter event object.
      :type event: tk.Event



   .. py:method:: on_leave_sash(event)

      Reset sash color on mouse leave.

      :param event: Tkinter event object.
      :type event: tk.Event



   .. py:method:: start_resize(event)

      Initiate resizing behavior when mouse press begins on a sash.

      :param event: Tkinter event object.
      :type event: tk.Event



   .. py:method:: perform_resize(event)

      Adjust pane sizes during mouse drag on a sash.

      :param event: Tkinter event object.
      :type event: tk.Event



.. py:class:: spacrEntry(parent, textvariable=None, outline=False, width=None, *args, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom Tkinter entry widget with rounded corners, dark theme styling, and active/inactive color handling.

   :param parent: Parent widget.
   :type parent: tk.Widget
   :param textvariable: Tkinter textvariable to bind to the entry.
   :type textvariable: tk.StringVar, optional
   :param outline: Whether to show an outline. Currently unused. Defaults to False.
   :type outline: bool, optional
   :param width: Width of the entry widget. Defaults to 220 if not provided.
   :type width: int, optional
   :param \*args: Additional arguments passed to the parent Frame.
   :param \*\*kwargs: Additional arguments passed to the parent Frame.

   Initialize the custom entry widget with dark theme and rounded styling.


   .. py:attribute:: bg_color
      :value: 'dark_gray'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: outline
      :value: False



   .. py:attribute:: font_family
      :value: 'OpenSans'



   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_loader


   .. py:attribute:: canvas_height
      :value: 40



   .. py:attribute:: canvas


   .. py:method:: draw_rounded_rectangle(color)

      Draws a rounded rectangle with the given color as background.

      :param color: Fill color for the rounded rectangle.
      :type color: str



   .. py:method:: on_focus_in(event)

      Event handler for focus in. Changes the background to the active color.



   .. py:method:: on_focus_out(event)

      Event handler for focus out. Reverts the background to the inactive color.



.. py:class:: spacrCheck(parent, text='', variable=None, *args, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom checkbox widget with rounded square appearance and dark style.

   :param parent: Parent widget.
   :type parent: tk.Widget
   :param text: Label text (currently unused).
   :type text: str, optional
   :param variable: Tkinter variable to bind the checkbox state.
   :type variable: tk.BooleanVar
   :param \*args: Additional arguments passed to the parent Frame.
   :param \*\*kwargs: Additional arguments passed to the parent Frame.

   Initializes the custom checkbox widget and binds visual updates to variable state.


   .. py:attribute:: bg_color
      :value: 'black'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: variable
      :value: None



   .. py:attribute:: canvas_width
      :value: 20



   .. py:attribute:: canvas_height
      :value: 20



   .. py:attribute:: canvas


   .. py:method:: draw_rounded_square(color)

      Draws a rounded square with border and fill.

      :param color: The fill color based on the current checkbox state.
      :type color: str



   .. py:method:: update_check(*args)

      Redraws the checkbox based on the current value of the associated variable.



   .. py:method:: toggle_variable(event)

      Toggles the value of the associated variable when the checkbox is clicked.

      :param event: The mouse click event.
      :type event: tk.Event



.. py:class:: spacrCombo(parent, textvariable=None, values=None, width=None, *args, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom styled combo box widget with rounded edges and dropdown functionality.

   This widget mimics a modern dropdown menu with dark-themed styling, allowing
   users to select from a list of values in a visually appealing interface.

   :param parent: Parent widget.
   :type parent: tk.Widget
   :param textvariable: Variable linked to the combo box selection.
   :type textvariable: tk.StringVar, optional
   :param values: List of selectable values. Defaults to empty list.
   :type values: list, optional
   :param width: Width of the combo box in pixels. Defaults to 220.
   :type width: int, optional

   Initialize the combo box UI and style settings.

   :param parent: The parent widget.
   :type parent: tk.Widget
   :param textvariable: A Tkinter StringVar linked to the selected value.
   :type textvariable: tk.StringVar, optional
   :param values: List of values to populate the dropdown.
   :type values: list, optional
   :param width: Width of the combo box. Defaults to 220 pixels.
   :type width: int, optional


   .. py:attribute:: bg_color
      :value: 'black'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: font_family
      :value: 'OpenSans'



   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_loader


   .. py:attribute:: values
      :value: []



   .. py:attribute:: canvas_width
      :value: None



   .. py:attribute:: canvas_height
      :value: 40



   .. py:attribute:: canvas


   .. py:attribute:: var


   .. py:attribute:: selected_value


   .. py:attribute:: dropdown_menu
      :value: None



   .. py:method:: draw_rounded_rectangle(color)

      Draw a rounded rectangle on the canvas with the specified background color.

      :param color: The fill color for the rounded rectangle.
      :type color: str



   .. py:method:: on_click(event)

      Handle click event on the combo box to toggle the dropdown menu.

      :param event: The mouse click event.
      :type event: tk.Event



   .. py:method:: open_dropdown()

      Display the dropdown menu with available values.



   .. py:method:: close_dropdown()

      Close the dropdown menu if it is open.



   .. py:method:: on_select(value)

      Update the displayed label and internal variable when a value is selected.

      :param value: The selected value from the dropdown.
      :type value: str



   .. py:method:: set(value)

      Programmatically set the combo box selection to the specified value.

      :param value: The value to set in the combo box.
      :type value: str



.. py:class:: spacrDropdownMenu(parent, variable, options, command=None, font=None, size=50, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom dark-themed dropdown menu widget with rounded edges and hover interaction.

   This widget displays a labeled button that reveals a menu of selectable options
   when clicked. It supports external callback functions, styling updates, and dynamic
   highlighting of active menu items.

   :param parent: Parent widget in which the dropdown menu is placed.
   :type parent: tk.Widget
   :param variable: A Tkinter variable to store the selected option.
   :type variable: tk.StringVar
   :param options: A list of option labels to populate the dropdown menu.
   :type options: list
   :param command: A function to call when an option is selected.
   :type command: callable, optional
   :param font: Font used for the button label.
   :type font: tuple or tk.Font, optional
   :param size: Height of the button in pixels. Defaults to 50.
   :type size: int, optional
   :param \*\*kwargs: Additional keyword arguments passed to the `tk.Frame` base class.

   Initialize the spacrDropdownMenu with a canvas-based button and popup menu.

   Sets up the button appearance, binds mouse interaction events,
   and constructs the dropdown menu with the given options.

   :param parent: Parent container.
   :type parent: tk.Widget
   :param variable: Variable that stores the selected option.
   :type variable: tk.StringVar
   :param options: List of strings representing the dropdown options.
   :type options: list
   :param command: Callback function when an option is selected.
   :type command: callable, optional
   :param font: Font used for the button text.
   :type font: tk.Font or tuple, optional
   :param size: Button height in pixels. Defaults to 50.
   :type size: int, optional
   :param \*\*kwargs: Additional keyword arguments for the Frame.


   .. py:attribute:: variable


   .. py:attribute:: options


   .. py:attribute:: command
      :value: None



   .. py:attribute:: text
      :value: 'Settings'



   .. py:attribute:: size
      :value: 50



   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_loader


   .. py:attribute:: button_width
      :value: 150



   .. py:attribute:: canvas_width
      :value: 154



   .. py:attribute:: canvas_height
      :value: 54



   .. py:attribute:: canvas


   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: bg_color
      :value: 'black'



   .. py:attribute:: button_bg


   .. py:attribute:: button_text


   .. py:attribute:: menu


   .. py:method:: create_rounded_rectangle(x1, y1, x2, y2, radius=20, **kwargs)

      Draw a rounded rectangle polygon on the internal canvas.

      :param x1: Top-left x coordinate.
      :type x1: int
      :param y1: Top-left y coordinate.
      :type y1: int
      :param x2: Bottom-right x coordinate.
      :type x2: int
      :param y2: Bottom-right y coordinate.
      :type y2: int
      :param radius: Radius of the corners. Defaults to 20.
      :type radius: int, optional
      :param \*\*kwargs: Canvas polygon configuration options (fill, outline, etc.).

      :returns: Canvas item ID of the created polygon.
      :rtype: int



   .. py:method:: on_enter(event=None)

      Handle mouse enter event by updating the button's background color.

      :param event: The event object. Defaults to None.
      :type event: tk.Event, optional



   .. py:method:: on_leave(event=None)

      Handle mouse leave event by resetting the button's background color.

      :param event: The event object. Defaults to None.
      :type event: tk.Event, optional



   .. py:method:: on_click(event=None)

      Handle button click event to display the dropdown menu.

      :param event: The event object. Defaults to None.
      :type event: tk.Event, optional



   .. py:method:: post_menu()

      Display the dropdown menu below the button using screen coordinates.



   .. py:method:: on_select(option)

      Callback when an option is selected from the dropdown menu.

      :param option: The selected option label.
      :type option: str



   .. py:method:: update_styles(active_categories=None)

      Update the styles of the dropdown menu entries based on active categories.

      :param active_categories: List of option labels to highlight
                                with the active color. If None, all entries are styled as inactive.
      :type active_categories: list or None



.. py:class:: spacrCheckbutton(parent, text='', variable=None, command=None, *args, **kwargs)

   Bases: :py:obj:`tkinter.ttk.Checkbutton`


   A dark-themed styled Checkbutton widget for use in the SpaCr GUI.

   This class wraps a `ttk.Checkbutton` with a custom style and binds it to a
   `BooleanVar`, allowing it to integrate seamlessly into dark-themed interfaces.

   :param parent: The parent widget in which to place the checkbutton.
   :type parent: tk.Widget
   :param text: The label text displayed next to the checkbutton.
   :type text: str, optional
   :param variable: Variable linked to the checkbutton's state.
                    If None, a new `BooleanVar` is created.
   :type variable: tk.BooleanVar, optional
   :param command: A callback function to execute when the checkbutton is toggled.
   :type command: callable, optional
   :param \*args: Additional positional arguments passed to `ttk.Checkbutton`.
   :param \*\*kwargs: Additional keyword arguments passed to `ttk.Checkbutton`.

   Construct a Ttk Checkbutton widget with the parent master.

   STANDARD OPTIONS

       class, compound, cursor, image, state, style, takefocus,
       text, textvariable, underline, width

   WIDGET-SPECIFIC OPTIONS

       command, offvalue, onvalue, variable


   .. py:attribute:: text
      :value: ''



   .. py:attribute:: variable


   .. py:attribute:: command
      :value: None



.. py:class:: spacrProgressBar(parent, label=True, *args, **kwargs)

   Bases: :py:obj:`tkinter.ttk.Progressbar`


   A dark-themed progress bar widget with optional progress label display.

   This class extends `ttk.Progressbar` and applies a dark visual theme consistent
   with SpaCr GUI styling. It also provides an optional label that displays real-time
   progress, operation type, and additional information.

   :param parent: The parent widget in which to place the progress bar.
   :type parent: tk.Widget
   :param label: Whether to show a label below the progress bar. Defaults to True.
   :type label: bool, optional
   :param \*args: Additional positional arguments passed to `ttk.Progressbar`.
   :param \*\*kwargs: Additional keyword arguments passed to `ttk.Progressbar`.

   Initialize the progress bar and optional label with dark theme styling.

   Sets the initial value to 0, applies custom style attributes, and creates
   a label for displaying progress information.

   :param parent: Parent container for the widget.
   :type parent: tk.Widget
   :param label: Whether to show a label for progress updates. Defaults to True.
   :type label: bool, optional


   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: bg_color
      :value: 'black'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_loader


   .. py:attribute:: style


   .. py:attribute:: label
      :value: True



   .. py:attribute:: operation_type
      :value: None



   .. py:attribute:: additional_info
      :value: None



   .. py:method:: set_label_position()

      Place the progress label one row below the progress bar in the grid layout.

      Should be called after the progress bar has been placed with `.grid(...)`.



   .. py:method:: update_label()

      Update the progress label with current progress, operation type, and extra info.

      Constructs a single-line status message with:
      - Current progress value
      - Operation type (if set)
      - Additional info (if set)



.. py:class:: spacrSlider(master=None, length=None, thickness=2, knob_radius=10, position='center', from_=0, to=100, value=None, show_index=False, command=None, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom slider widget with dark-themed styling, optional numeric entry, and mouse interaction.

   This slider is designed for GUI applications where numeric control is needed,
   supporting dynamic resizing, labeled value entry, and a callback on release.

   :param master: Parent widget.
   :type master: tk.Widget
   :param length: Fixed pixel length of the slider. If None, adapts to canvas width.
   :type length: int, optional
   :param thickness: Thickness of the slider bar in pixels. Defaults to 2.
   :type thickness: int, optional
   :param knob_radius: Radius of the slider knob in pixels. Defaults to 10.
   :type knob_radius: int, optional
   :param position: Alignment of slider within the frame. One of "left", "center", "right".
   :type position: str, optional
   :param from_: Minimum slider value.
   :type from_: float
   :param to: Maximum slider value.
   :type to: float
   :param value: Initial value. Defaults to `from_`.
   :type value: float, optional
   :param show_index: Whether to show an entry for numeric value. Defaults to False.
   :type show_index: bool, optional
   :param command: Function to call with the final value upon knob release.
   :type command: Callable, optional
   :param \*\*kwargs: Additional options passed to `tk.Frame`.

   Initialize a custom dark-themed slider widget.

   This slider supports mouse interaction, optional direct numeric input via an Entry,
   and dynamically adapts its layout based on container resizing unless a fixed `length` is specified.

   :param master: Parent widget.
   :type master: tk.Widget, optional
   :param length: Fixed length of the slider in pixels. If None, dynamically resizes with the container.
   :type length: int, optional
   :param thickness: Thickness of the slider bar in pixels. Default is 2.
   :type thickness: int, optional
   :param knob_radius: Radius of the draggable knob in pixels. Default is 10.
   :type knob_radius: int, optional
   :param position: Alignment of the slider within the frame. One of {"left", "center", "right"}. Default is "center".
   :type position: str, optional
   :param from_: Minimum value of the slider.
   :type from_: float
   :param to: Maximum value of the slider.
   :type to: float
   :param value: Initial value of the slider. Defaults to `from_` if not specified.
   :type value: float, optional
   :param show_index: If True, displays a text entry box to manually input the slider value. Default is False.
   :type show_index: bool, optional
   :param command: Optional function to be called with the final value when the knob is released.
   :type command: Callable, optional
   :param \*\*kwargs: Additional keyword arguments passed to the `tk.Frame` initializer.


   .. py:attribute:: specified_length
      :value: None



   .. py:attribute:: knob_radius
      :value: 10



   .. py:attribute:: thickness
      :value: 2



   .. py:attribute:: knob_position
      :value: 10



   .. py:attribute:: slider_line
      :value: None



   .. py:attribute:: knob
      :value: None



   .. py:attribute:: position
      :value: ''



   .. py:attribute:: offset
      :value: 0



   .. py:attribute:: from_
      :value: 0



   .. py:attribute:: to
      :value: 100



   .. py:attribute:: value
      :value: None



   .. py:attribute:: show_index
      :value: False



   .. py:attribute:: command
      :value: None



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: bg_color
      :value: 'black'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: canvas


   .. py:attribute:: length
      :value: None



   .. py:method:: resize_slider(event)

      Recalculate slider dimensions and redraw upon resizing.

      :param event: Resize event.
      :type event: tk.Event



   .. py:method:: value_to_position(value)

      Convert a numerical slider value to a pixel position on the canvas.

      :param value: The numerical value to convert.
      :type value: float

      :returns: Corresponding position in pixels on the slider.
      :rtype: float



   .. py:method:: position_to_value(position)

      Convert a pixel position on the slider to a numerical value.

      :param position: Pixel position on the slider.
      :type position: float

      :returns: Corresponding numerical slider value.
      :rtype: float



   .. py:method:: draw_slider(inactive=False)

      Draw the slider bar and knob on the canvas.

      :param inactive: If True, draw knob in inactive color. Otherwise, use active color.
      :type inactive: bool



   .. py:method:: move_knob(event)

      Move the knob in response to mouse drag, updating internal value and position.

      :param event: Motion event.
      :type event: tk.Event



   .. py:method:: activate_knob(event)

      Highlight knob and respond to click by positioning knob at mouse location.

      :param event: Click event.
      :type event: tk.Event



   .. py:method:: release_knob(event)

      Finalize knob movement and call the `command` callback with final value.

      :param event: Mouse release event.
      :type event: tk.Event



   .. py:method:: set_to(new_to)

      Change the maximum value (`to`) of the slider.

      :param new_to: New upper bound of the slider.
      :type new_to: float



   .. py:method:: get()

      Get the current slider value.

      :returns: Current value of the slider.
      :rtype: float



   .. py:method:: set(value)

      Set the slider to a specific value and redraw.

      :param value: New value to set (clipped to bounds).
      :type value: float



   .. py:method:: jump_to_click(event)

      Move the knob directly to the mouse click position.

      :param event: Click event.
      :type event: tk.Event



   .. py:method:: update_slider_from_entry(event)

      Update the slider from a value entered manually in the index entry.

      :param event: Return key press event.
      :type event: tk.Event



.. py:function:: spacrScrollbarStyle(style, inactive_color, active_color)

   Applies a custom vertical scrollbar style using the given colors.

   This function defines a new ttk scrollbar style named 'Custom.Vertical.TScrollbar'.
   It reuses the base elements from the 'clam' theme and sets the colors for active
   and inactive states accordingly. If the required elements do not exist, it creates
   them from the base theme.

   :param style: The ttk Style object to configure.
   :type style: ttk.Style
   :param inactive_color: Hex or color name for the scrollbar in its default (inactive) state.
   :type inactive_color: str
   :param active_color: Hex or color name for the scrollbar when hovered or active.
   :type active_color: str


.. py:class:: spacrFrame(container, width=None, *args, bg='black', radius=20, scrollbar=True, textbox=False, **kwargs)

   Bases: :py:obj:`tkinter.ttk.Frame`


   A styled frame with optional rounded background, vertical scrollbar, and embedded content area (text or widgets).

   This frame supports both scrollable `ttk.Frame` containers and scrollable `tk.Text` areas, with a dark custom theme.

   .. attribute:: scrollable_frame

      The inner content widget.

      :type: Union[ttk.Frame, tk.Text]

   Initialize the spacrFrame.

   :param container: The parent container for this frame.
   :type container: tk.Widget
   :param width: Width of the frame in pixels. Defaults to 1/4 of screen width if None.
   :type width: int, optional
   :param \*args: Additional positional arguments for ttk.Frame.
   :param bg: Background color of the frame. Defaults to 'black'.
   :type bg: str
   :param radius: Radius of the rounded rectangle background. Defaults to 20.
   :type radius: int
   :param scrollbar: Whether to include a vertical scrollbar. Defaults to True.
   :type scrollbar: bool
   :param textbox: If True, use a scrollable `tk.Text` widget. Otherwise, use a `ttk.Frame`. Defaults to False.
   :type textbox: bool
   :param \*\*kwargs: Additional keyword arguments for ttk.Frame.


   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:method:: rounded_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs)

      Draw a rounded rectangle on a canvas.

      :param canvas: The canvas to draw on.
      :type canvas: tk.Canvas
      :param x1: Left coordinate.
      :type x1: int
      :param y1: Top coordinate.
      :type y1: int
      :param x2: Right coordinate.
      :type x2: int
      :param y2: Bottom coordinate.
      :type y2: int
      :param radius: Radius of the rounded corners. Defaults to 20.
      :type radius: int
      :param \*\*kwargs: Options passed to the canvas `create_polygon` method.

      :returns: ID of the created polygon.
      :rtype: int



.. py:class:: spacrLabel(parent, text='', font=None, style=None, align='right', height=None, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom label widget with optional dark styling, alignment options, and support for both `ttk.Label` and `Canvas`-rendered text.

   The label adapts to screen size or a given height, and can display text either centered or right-aligned.

   Initialize the spacrLabel widget.

   :param parent: The parent widget.
   :type parent: tk.Widget
   :param text: The text to display on the label. Defaults to "".
   :type text: str
   :param font: A custom font to use if not using the default style. Defaults to None.
   :type font: tkFont.Font, optional
   :param style: A ttk style name to apply to the label. If set, uses a `ttk.Label` instead of `Canvas` text.
   :type style: str, optional
   :param align: Text alignment, either "right" or "center". Defaults to "right".
   :type align: str
   :param height: Height of the label. If None, scales based on screen height.
   :type height: int, optional
   :param \*\*kwargs: Additional keyword arguments for the outer frame (excluding font/background/anchor-specific ones).


   .. py:attribute:: text
      :value: ''



   .. py:attribute:: align
      :value: 'right'



   .. py:attribute:: style_out


   .. py:attribute:: font_style
      :value: 'OpenSans'



   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_family
      :value: 'OpenSans'



   .. py:attribute:: font_loader


   .. py:attribute:: canvas


   .. py:attribute:: style
      :value: None



   .. py:method:: set_text(text)

      Update the label text.

      :param text: The new text to display.
      :type text: str



.. py:class:: spacrButton(parent, text='', command=None, font=None, icon_name=None, size=50, show_text=True, outline=False, animation=True, *args, **kwargs)

   Bases: :py:obj:`tkinter.Frame`


   A custom animated button widget with icon and optional text, styled with dark mode and zoom animation on hover.

   The button is rendered using a Canvas to support rounded corners and icon embedding. Optional description
   display is supported via parent methods `show_description` and `clear_description`.

   Initialize the spacrButton.

   :param parent: The parent container.
   :type parent: tk.Widget
   :param text: Button text to display.
   :type text: str
   :param command: Function to call when button is clicked.
   :type command: callable, optional
   :param font: Font to use if font loader is unavailable.
   :type font: tkFont.Font or tuple, optional
   :param icon_name: Name of icon (without extension) to load from resources/icons.
   :type icon_name: str, optional
   :param size: Button height (and icon size reference). Defaults to 50.
   :type size: int
   :param show_text: Whether to show text next to the icon. Defaults to True.
   :type show_text: bool
   :param outline: Whether to draw a border around the button. Defaults to False.
   :type outline: bool
   :param animation: Whether to animate the icon on hover. Defaults to True.
   :type animation: bool
   :param \*args: Additional positional arguments for the Frame.
   :param \*\*kwargs: Additional keyword arguments for the Frame.


   .. py:attribute:: text
      :value: ''



   .. py:attribute:: command
      :value: None



   .. py:attribute:: icon_name
      :value: ''



   .. py:attribute:: size
      :value: 50



   .. py:attribute:: show_text
      :value: True



   .. py:attribute:: outline
      :value: False



   .. py:attribute:: animation
      :value: True



   .. py:attribute:: font_size
      :value: 12



   .. py:attribute:: font_loader


   .. py:attribute:: canvas


   .. py:attribute:: inactive_color
      :value: 'dark_gray'



   .. py:attribute:: bg_color
      :value: 'dark_gray'



   .. py:attribute:: active_color
      :value: 'blue'



   .. py:attribute:: fg_color
      :value: 'white'



   .. py:attribute:: is_zoomed_in
      :value: False



   .. py:method:: load_icon()

      Load and resize the icon from the icon folder.

      Falls back to 'default.png' if the specified icon cannot be found.



   .. py:method:: get_icon_path(icon_name)

      Get the full path to the icon file.

      :param icon_name: Icon name without extension.
      :type icon_name: str

      :returns: Full path to the icon file.
      :rtype: str



   .. py:method:: on_enter(event=None)

      Handle mouse hover enter event.

      Changes button color, shows description, and animates zoom-in if enabled.



   .. py:method:: on_leave(event=None)

      Handle mouse hover leave event.

      Resets button color, clears description, and animates zoom-out if enabled.



   .. py:method:: on_click(event=None)

      Trigger the button's command callback when clicked.



   .. py:method:: create_rounded_rectangle(x1, y1, x2, y2, radius=20, **kwargs)

      Create a rounded rectangle on the canvas.

      :param x1: Coordinates of the rectangle.
      :type x1: int
      :param y1: Coordinates of the rectangle.
      :type y1: int
      :param x2: Coordinates of the rectangle.
      :type x2: int
      :param y2: Coordinates of the rectangle.
      :type y2: int
      :param radius: Radius of the corners.
      :type radius: int
      :param \*\*kwargs: Passed to `create_polygon`.

      :returns: Canvas item ID of the rounded rectangle.
      :rtype: int



   .. py:method:: update_description(event)

      Call parent container’s `show_description()` if available,
      passing the description based on `main_buttons` or `additional_buttons` maps.



   .. py:method:: clear_description(event)

      Call parent container’s `clear_description()` if available.



   .. py:method:: animate_zoom(target_scale, steps=10, delay=10)

      Animate zoom effect by resizing icon incrementally.

      :param target_scale: Final scale factor relative to base icon size.
      :type target_scale: float
      :param steps: Number of animation steps. Defaults to 10.
      :type steps: int
      :param delay: Delay between steps in milliseconds. Defaults to 10.
      :type delay: int



   .. py:method:: zoom_icon(scale_factor)

      Resize and update the icon image on the canvas.

      :param scale_factor: Scaling factor relative to base icon size.
      :type scale_factor: float



.. py:class:: spacrSwitch(parent, text='', variable=None, command=None, *args, **kwargs)

   Bases: :py:obj:`tkinter.ttk.Frame`


   A custom toggle switch widget with animated transitions and label, styled using the spacr dark theme.

   This switch mimics a physical toggle with animated motion of the switch knob and changes in color.

   Initialize the spacrSwitch widget.

   :param parent: Parent container.
   :type parent: tk.Widget
   :param text: Label displayed to the left of the switch.
   :type text: str
   :param variable: Tkinter BooleanVar linked to the switch state.
   :type variable: tk.BooleanVar, optional
   :param command: Function to call when the switch is toggled.
   :type command: callable, optional
   :param \*args: Additional positional arguments for the Frame.
   :param \*\*kwargs: Additional keyword arguments for the Frame.


   .. py:attribute:: text
      :value: ''



   .. py:attribute:: variable


   .. py:attribute:: command
      :value: None



   .. py:attribute:: canvas


   .. py:attribute:: switch_bg


   .. py:attribute:: switch


   .. py:attribute:: label


   .. py:method:: toggle(event=None)

      Toggle the state of the switch.

      Updates the linked variable, animates the movement, and calls the `command` callback if defined.



   .. py:method:: update_switch()

      Immediately update the switch position and color based on the current value of the variable.



   .. py:method:: animate_switch()

      Trigger an animated transition of the switch knob between on and off states.



   .. py:method:: animate_movement(start_x, end_x, final_color)

      Animate the horizontal movement of the switch knob.

      :param start_x: Starting x-coordinate of the knob.
      :type start_x: int
      :param end_x: Ending x-coordinate of the knob.
      :type end_x: int
      :param final_color: Fill color of the knob at the end of the animation.
      :type final_color: str



   .. py:method:: get()

      Get the current Boolean value of the switch.

      :returns: True if switch is on, False otherwise.
      :rtype: bool



   .. py:method:: set(value)

      Set the switch to a given Boolean value.

      :param value: New state for the switch.
      :type value: bool



   .. py:method:: create_rounded_rectangle(x1, y1, x2, y2, radius=9, **kwargs)

      Draw a rounded rectangle polygon on the canvas.

      :param x1: Coordinates of the rectangle bounds.
      :type x1: int
      :param y1: Coordinates of the rectangle bounds.
      :type y1: int
      :param x2: Coordinates of the rectangle bounds.
      :type x2: int
      :param y2: Coordinates of the rectangle bounds.
      :type y2: int
      :param radius: Radius of corner curvature.
      :type radius: int
      :param \*\*kwargs: Options passed to `create_polygon`.

      :returns: ID of the created polygon item on the canvas.
      :rtype: int



.. py:class:: spacrToolTip(widget, text)

   A simple tooltip widget for displaying hover text in a Tkinter application using spacr dark styling.

   Initialize the tooltip for a given widget.

   :param widget: The widget to attach the tooltip to.
   :type widget: tk.Widget
   :param text: The text to display in the tooltip.
   :type text: str


   .. py:attribute:: widget


   .. py:attribute:: text


   .. py:attribute:: tooltip_window
      :value: None



   .. py:method:: show_tooltip(event)

      Display the tooltip near the cursor when mouse enters the widget.



   .. py:method:: hide_tooltip(event)

      Hide and destroy the tooltip when mouse leaves the widget.



.. py:function:: standardize_figure(fig)

   Apply standardized appearance settings to a matplotlib figure using spaCR GUI style preferences.

   This includes:
   - Setting font size and font family based on spaCR's theme
   - Setting text and spine colors to match spaCR foreground color
   - Applying OpenSans font via `font_loader`
   - Removing top and right spines
   - Setting line and spine widths
   - Adjusting background and grid colors

   :param fig: The matplotlib figure to standardize.
   :type fig: matplotlib.figure.Figure

   :returns: The updated figure with standardized style.
   :rtype: matplotlib.figure.Figure


.. py:function:: modify_figure_properties(fig, scale_x=None, scale_y=None, line_width=None, font_size=None, x_lim=None, y_lim=None, grid=False, legend=None, title=None, x_label_rotation=None, remove_axes=False, bg_color=None, text_color=None, line_color=None)

   Modifies the properties of the figure, including scaling, line widths, font sizes, axis limits, x-axis label rotation, background color, text color, line color, and other common options.

   Args:
   - fig: The Matplotlib figure object to modify.
   - scale_x: Scaling factor for the width of subplots (optional).
   - scale_y: Scaling factor for the height of subplots (optional).
   - line_width: Desired line width for all lines (optional).
   - font_size: Desired font size for all text (optional).
   - x_lim: Tuple specifying the x-axis limits (min, max) (optional).
   - y_lim: Tuple specifying the y-axis limits (min, max) (optional).
   - grid: Boolean to add grid lines to the plot (optional).
   - legend: Boolean to show/hide the legend (optional).
   - title: String to set as the title of the plot (optional).
   - x_label_rotation: Angle to rotate the x-axis labels (optional).
   - remove_axes: Boolean to remove or show the axes labels (optional).
   - bg_color: Color for the figure and subplot background (optional).
   - text_color: Color for all text in the figure (optional).
   - line_color: Color for all lines in the figure (optional).


.. py:function:: save_figure_as_format(fig, file_format)

   Opens a file dialog to save a matplotlib figure in the specified format.

   Prompts the user to choose a save location and filename using a file dialog.
   The figure is saved using the provided format if a valid path is selected.

   :param fig: The figure to save.
   :type fig: matplotlib.figure.Figure
   :param file_format: The file format to save as (e.g., 'png', 'pdf', 'svg').
   :type file_format: str

   :returns: None


.. py:function:: modify_figure(fig)

   Opens a GUI window for interactively modifying various properties of a matplotlib figure.

   This function allows users to:
   - Rescale the X and Y axes
   - Change line width and font size
   - Set axis limits and title
   - Customize colors (background, text, line)
   - Rotate x-axis labels
   - Enable/disable grid, legend, and axes
   - Toggle spine visibility ("spleens")

   Once modifications are entered and applied, the figure is updated and re-rendered via `display_figure`.

   :param fig: The matplotlib figure to modify.
   :type fig: matplotlib.figure.Figure

   :returns: None


.. py:function:: generate_dna_matrix(output_path='dna_matrix.gif', canvas_width=1500, canvas_height=1000, duration=30, fps=20, base_size=20, transition_frames=30, font_type='arial.ttf', enhance=[1.1, 1.5, 1.2, 1.5], lowercase_prob=0.3)

   Generates an animated matrix-style DNA sequence visual and saves it as a GIF or video.

   The animation simulates vertical streams of random DNA bases ('A', 'T', 'C', 'G') cascading down the screen.
   Each column has independently scrolling strings of bases. The latest base in each stream is highlighted,
   and fading effects, coloring, and random lowercase transformations are applied for visual flair.

   :param output_path: Path to the output file (should end in .gif, .mp4, or .avi).
   :type output_path: str
   :param canvas_width: Width of the canvas in pixels.
   :type canvas_width: int
   :param canvas_height: Height of the canvas in pixels.
   :type canvas_height: int
   :param duration: Duration of the animation in seconds.
   :type duration: int
   :param fps: Frames per second of the animation.
   :type fps: int
   :param base_size: Font size (in pixels) for the bases.
   :type base_size: int
   :param transition_frames: Number of frames for the looping transition.
   :type transition_frames: int
   :param font_type: Path to a .ttf font to use. Defaults to Arial.
   :type font_type: str
   :param enhance: List of four enhancement multipliers for [brightness, sharpness, contrast, saturation].
   :type enhance: list
   :param lowercase_prob: Probability that a given base will be drawn in lowercase.
   :type lowercase_prob: float

   :returns: None


