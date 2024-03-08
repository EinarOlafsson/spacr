def process_fig_queue_v1():
    while not fig_queue.empty():
        update_task = fig_queue.get()
        try:
            update_task()
        except Exception as e:
            print(f"Error processing fig_queue: {e}")
    root.after(100, process_fig_queue)

def update_figure_in_gui_v1(fig):
    

    def task():
        global canvas, canvas_widget, fig_queue
        disconnect_all_event_handlers(fig)
        canvas.figure = fig
        fig.set_size_inches(10, 10, forward=True)
        for axis in fig.axes:
            axis.set_visible(False)        
        canvas.draw()
        canvas_widget.draw()  
           
    fig_queue.put(task)
    
def my_show_v1():
    fig = plt.gcf()
    update_figure_in_gui(fig)
    
    def disconnect_all_event_handlers(fig):
    canvas = fig.canvas
    if canvas.callbacks.callbacks:
        for event, callback_list in list(canvas.callbacks.callbacks.items()):
            for cid in list(callback_list.keys()):
                canvas.mpl_disconnect(cid)
    return canvas

def resize_figure_to_canvas(fig, canvas):
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Convert pixels to inches for matplotlib
    fig_width = canvas_width / fig.dpi
    fig_height = canvas_height / fig.dpi

    # Resizing the figure
    fig.set_size_inches(fig_width, fig_height, forward=True)

    # Optionally, hide axes
    for ax in fig.axes:
        ax.set_visible(False)
        
    return fig
    
    def process_fig_queue_v1():
    global canvas
    while not fig_queue.empty():
        try:
            fig = fig_queue.get_nowait()
            canvas.figure = fig  
            canvas.draw()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing fig_queue: {e}")
            traceback.print_exc()
    root.after(100, process_fig_queue)
    
def process_fig_queue():
    while not fig_queue.empty():
        try:
            fig = fig_queue.get_nowait()
            # Signal the main thread to update the GUI with the new figure
            root.after_idle(update_canvas_with_figure, fig)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing fig_queue: {e}")
            traceback.print_exc()
    # Reschedule itself to run again
    root.after(100, process_fig_queue)
    
def update_canvas_with_figure(fig):
    global canvas
    # Resize the figure to fit the canvas
    canvas_width = canvas.get_tk_widget().winfo_width()
    canvas_height = canvas.get_tk_widget().winfo_height()
    fig_width = canvas_width / fig.dpi
    fig_height = canvas_height / fig.dpi
    fig.set_size_inches(fig_width, fig_height, forward=True)
    # Hide the axes if needed
    for ax in fig.axes:
        ax.set_visible(False)
    # Update the canvas with the new figure
    canvas.figure = fig
    canvas.draw_idle()  # Use draw_idle for efficiency and thread safety

def run_mask_gui(q):
    global vars_dict
    try:
        settings = check_mask_gui_settings(vars_dict)
        settings = add_mask_gui_defaults(settings)
        preprocess_generate_masks_wrapper(settings['src'], settings=settings, advanced_settings={})
    except Exception as e:
        q.put(f"Error during processing: {e}\n")
