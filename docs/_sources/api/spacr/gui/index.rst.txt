spacr.gui
=========

.. py:module:: spacr.gui






Module Contents
---------------

.. py:class:: MainApp(default_app=None)

   Bases: :py:obj:`tkinter.Tk`


   Toplevel widget of Tk which represents mostly the main window
   of an application. It has an associated Tcl interpreter.


   .. py:attribute:: color_settings


   .. py:attribute:: main_buttons


   .. py:attribute:: additional_buttons


   .. py:attribute:: main_gui_apps


   .. py:attribute:: additional_gui_apps


   .. py:attribute:: selected_app


   .. py:method:: create_widgets()


   .. py:method:: create_startup_screen()


   .. py:method:: update_description()


   .. py:method:: show_description(description)


   .. py:method:: clear_description()


   .. py:method:: load_app(app_name, app_func)


   .. py:method:: clear_frame(frame)


.. py:function:: gui_app()

