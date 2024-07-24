import traceback
from .gui_wrappers import measure_crop_wrapper, preprocess_generate_masks_wrapper, train_test_model_wrapper, umap_wrapper, sequencing_wrapper

def run_mask_gui(settings, q, fig_queue, stop_requested):
    from .gui_utils import process_stdout_stderr
    process_stdout_stderr(q)
    try:
        preprocess_generate_masks_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_sequencing_gui(settings, q, fig_queue, stop_requested):
    from .gui_utils import process_stdout_stderr
    process_stdout_stderr(q)
    try:
        sequencing_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_umap_gui(settings, q, fig_queue, stop_requested):
    from .gui_utils import process_stdout_stderr
    process_stdout_stderr(q)
    try:
        umap_wrapper(settings, q, fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_measure_gui(settings, q, fig_queue, stop_requested):
    from .gui_utils import process_stdout_stderr
    process_stdout_stderr(q)
    try:
        settings['input_folder'] = settings['src']
        measure_crop_wrapper(settings=settings, q=q, fig_queue=fig_queue)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1

def run_classify_gui(settings, q, fig_queue, stop_requested):
    from .gui_utils import process_stdout_stderr
    process_stdout_stderr(q)
    try:
        train_test_model_wrapper(settings['src'], settings)
    except Exception as e:
        q.put(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        stop_requested.value = 1