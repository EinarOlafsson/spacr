import traceback, matplotlib, spacr
import matplotlib.pyplot as plt
matplotlib.use('Agg')

fig_queue = None

def spacrFigShow_v1():
    """
    Replacement for plt.show() that queues figures instead of displaying them.
    """
    global fig_queue
    fig = plt.gcf()
    fig_queue.put(fig)
    plt.close(fig)

def spacrFigShow(fig_queue=None):
    """
    Replacement for plt.show() that queues figures instead of displaying them.
    """
    fig = plt.gcf()
    if fig_queue:
        fig_queue.put(fig)
    else:
        fig.show()
    plt.close(fig)

def preprocess_generate_masks_wrapper(settings, q, fig_queue):
    """
    Wraps the measure_crop function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the measure_crop function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        spacr.core.preprocess_generate_masks(src=settings['src'], settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show

def measure_crop_wrapper(settings, q, fig_queue):
    """
    Wraps the measure_crop function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the measure_crop function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        print('start')
        spacr.measure.measure_crop(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show  

def sequencing_wrapper(settings, q, fig_queue):

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        spacr.sequencing.analyze_reads(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show

def umap_wrapper(settings, q, fig_queue):

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        spacr.core.generate_image_umap(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)
        traceback.print_exc()
    finally:
        plt.show = original_show

def train_test_model_wrapper(settings, q, fig_queue):
    """
    Wraps the measure_crop function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the measure_crop function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        spacr.core.train_test_model(settings['src'], settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)  # Send the error message to the GUI via the queue
        traceback.print_exc()
    finally:
        plt.show = original_show  # Restore the original plt.show function
        
        
def run_multiple_simulations_wrapper(settings, q, fig_queue):
    """
    Wraps the run_multiple_simulations function to integrate with GUI processes.
    
    Parameters:
    - settings: dict, The settings for the run_multiple_simulations function.
    - q: multiprocessing.Queue, Queue for logging messages to the GUI.
    - fig_queue: multiprocessing.Queue, Queue for sending figures to the GUI.
    """

    # Temporarily override plt.show
    original_show = plt.show
    plt.show = lambda: spacrFigShow(fig_queue)

    try:
        spacr.sim.run_multiple_simulations(settings=settings)
    except Exception as e:
        errorMessage = f"Error during processing: {e}"
        q.put(errorMessage)  # Send the error message to the GUI via the queue
        traceback.print_exc()
    finally:
        plt.show = original_show  # Restore the original plt.show function