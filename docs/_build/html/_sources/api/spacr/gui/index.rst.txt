spacr.gui
=========

.. py:module:: spacr.gui






Module Contents
---------------

.. py:class:: MainApp(default_app=None)

   Bases: :py:obj:`tkinter.Tk`


   The main graphical user interface (GUI) window for launching various SpaCr modules.

   This class creates a full-screen, themed Tkinter application with buttons to launch
   sub-GUIs for image mask generation, measurement, classification, and advanced analysis
   tools like regression, barcode mapping, and model activation visualization.

   Initialize the SpaCr main application window.

   Sets up the full-screen GUI, applies dark theme styling, registers available
   sub-applications, and optionally launches a specific sub-application.

   :param default_app: If specified, directly launches the corresponding
                       application view. Should match a key in `main_gui_apps` or `additional_gui_apps`.
   :type default_app: str, optional


   .. py:attribute:: color_settings


   .. py:attribute:: main_buttons


   .. py:attribute:: additional_buttons


   .. py:attribute:: main_gui_apps


   .. py:attribute:: additional_gui_apps


   .. py:attribute:: selected_app


   .. py:method:: create_widgets()

      Create and place all GUI widgets and layout frames.

      This includes the content and inner frames, sets dark style themes,
      and triggers the startup screen.



   .. py:method:: create_startup_screen()

      Create the initial screen with buttons to launch different SpaCr modules.

      This function generates main and additional app buttons with tooltips and
      handles layout and styling.



   .. py:method:: update_description()

      Update the description label based on the currently active (highlighted) button.

      If no button is active, the label is cleared.



   .. py:method:: show_description(description)

      Show a specific description text in the description label.

      :param description: The text to be displayed.
      :type description: str



   .. py:method:: clear_description()

      Clear the content of the description label.



   .. py:method:: load_app(app_name, app_func)

      Load and display the selected application GUI in a new frame.

      :param app_name: The name of the app being loaded.
      :type app_name: str
      :param app_func: The function that initializes the app.
      :type app_func: Callable



   .. py:method:: clear_frame(frame)

      Remove all widgets from a given Tkinter frame.

      :param frame: The frame to clear.
      :type frame: tk.Frame



.. py:function:: gui_app()

   Launch the main SpaCr GUI.

   This function initializes and runs the `MainApp` class with no default app specified,
   allowing users to select functionality from the startup screen.


