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

#@log_function_call   
def main_thread_update_function(root, q, fig_queue, canvas_widget, progress_label):
    try:
        while not q.empty():
            message = q.get_nowait()
            if message.startswith("Progress"):
                progress_label.config(text=message)
            elif message.startswith("Processing"):
                progress_label.config(text=message)
            elif message == "" or message == "\r":
                pass
            elif message.startswith("/"):
                pass
            elif message.startswith("\\"):
                pass
            elif message.startswith(""):
                pass
            else:
                print(message)
    except Exception as e:
        print(f"Error updating GUI canvas: {e}")
    #try:    
    #    while not fig_queue.empty():
    #        fig = fig_queue.get_nowait()
    #        #if hasattr(canvas_widget, 'figure'):
    #        #clear_canvas(canvas_widget)
    #        canvas_widget.figure = fig
    #except Exception as e:
    #    print(f"Error updating GUI figure: {e}")
    finally:
        root.after(100, lambda: main_thread_update_function(root, q, fig_queue, canvas_widget, progress_label))
        
    class MPNN(MessagePassing):
    def __init__(self, node_in_features, edge_in_features, out_features):
        super(MPNN, self).__init__(aggr='mean')  # 'mean' aggregation.
        self.message_mlp = Sequential(
            Linear(node_in_features + edge_in_features, 128),
            ReLU(),
            Linear(128, out_features)
        )
        self.update_mlp = Sequential(
            Linear(out_features, out_features),
            ReLU(),
            Linear(out_features, out_features)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: Node features [N, node_in_features]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge attributes/features [E, edge_in_features]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: Input features of neighbors [E, node_in_features]
        # edge_attr: Edge attributes [E, edge_in_features]
        tmp = torch.cat([x_j, edge_attr], dim=-1)  # Concatenate node features with edge attributes
        return self.message_mlp(tmp)

    def update(self, aggr_out):
        # aggr_out: Aggregated messages [N, out_features]
        return self.update_mlp(aggr_out)
    
def weighted_mse_loss(output, target, score_threshold=0.8, high_score_weight=10):
    # Assumes output and target are the predicted and true scores, respectively
    weights = torch.ones_like(target)
    high_score_mask = target >= score_threshold
    weights[high_score_mask] = high_score_weight
    return ((output - target) ** 2 * weights).mean()

def generate_single_graph(sequencing, scores):
    # Load and preprocess sequencing data
    gene_df = pd.read_csv(sequencing)
    gene_df = gene_df.rename(columns={"prc": "well_id", "grna": "gene_id", "count": "read_count"})
    total_reads_per_well = gene_df.groupby('well_id')['read_count'].sum().reset_index(name='total_reads')
    gene_df = gene_df.merge(total_reads_per_well, on='well_id')
    gene_df['well_read_fraction'] = gene_df['read_count']/gene_df['total_reads']

    # Load and preprocess cell score data
    cell_df = pd.read_csv(scores)
    cell_df = cell_df[['prcfo', 'prc', 'pred']].rename(columns={'prcfo': 'cell_id', 'prc': 'well_id', 'pred': 'score'})

    # Initialize mappings
    gene_id_to_index = {gene: i for i, gene in enumerate(gene_df['gene_id'].unique())}
    cell_id_to_index = {cell: i + len(gene_id_to_index) for i, cell in enumerate(cell_df['cell_id'].unique())}

    # Initialize edge indices and attributes
    edge_index = []
    edge_attr = []

    # Associate each cell with all genes in the same well
    for well_id, group in gene_df.groupby('well_id'):
        if well_id in cell_df['well_id'].values:
            cell_indices = cell_df[cell_df['well_id'] == well_id]['cell_id'].map(cell_id_to_index).values
            gene_indices = group['gene_id'].map(gene_id_to_index).values
            fractions = group['well_read_fraction'].values
            
            for cell_idx in cell_indices:
                for gene_idx, fraction in zip(gene_indices, fractions):
                    edge_index.append([cell_idx, gene_idx])
                    edge_attr.append([fraction])

    # Convert lists to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    cell_scores = torch.tensor(cell_df['score'].values, dtype=torch.float)

    # One-hot encoding for genes, and zero features for cells (could be replaced with real features if available)
    gene_features = torch.eye(len(gene_id_to_index))
    cell_features = torch.zeros(len(cell_id_to_index), gene_features.size(1))

    # Combine features
    x = torch.cat([cell_features, gene_features], dim=0)

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=cell_scores)

    return data, gene_id_to_index, len(gene_id_to_index)

# in _normalize_and_outline
    outlines = []
    
        overlayed_image = rgb_image.copy()
        for i, mask_dim in enumerate(mask_dims):
            mask = np.take(image, mask_dim, axis=2)
            outline = np.zeros_like(mask)
            # Find the contours of the objects in the mask
            for j in np.unique(mask)[1:]:
                contours = find_contours(mask == j, 0.5)
                for contour in contours:
                    contour = contour.astype(int)
                    outline[contour[:, 0], contour[:, 1]] = j
            # Make the outline thicker
            outline = dilation(outline, square(outline_thickness))
            outlines.append(outline)
            # Overlay the outlines onto the RGB image
            for j in np.unique(outline)[1:]:
                overlayed_image[outline == j] = outline_colors[i % len(outline_colors)]

def _extract_filename_metadata(filenames, src, images_by_key, regular_expression, metadata_type='cellvoyager', pick_slice=False, skip_mode='01'):
    for filename in filenames:
        match = regular_expression.match(filename)
        if match:
            try:
                try:
                    plate = match.group('plateID')
                except:
                    plate = os.path.basename(src)

                well = match.group('wellID')
                field = match.group('fieldID')
                channel = match.group('chanID')
                mode = None

                if well[0].isdigit():
                    well = str(_safe_int_convert(well))
                if field[0].isdigit():
                    field = str(_safe_int_convert(field))
                if channel[0].isdigit():
                    channel = str(_safe_int_convert(channel))

                if metadata_type =='cq1':
                    orig_wellID = wellID
                    wellID = _convert_cq1_well_id(wellID)
                    clear_output(wait=True)
                    print(f'Converted Well ID: {orig_wellID} to {wellID}', end='\r', flush=True)

                if pick_slice:
                    try:
                        mode = match.group('AID')
                    except IndexError:
                        sliceid = '00'

                    if mode == skip_mode:
                        continue      
                        
                key = (plate, well, field, channel, mode)
                with Image.open(os.path.join(src, filename)) as img:
                    images_by_key[key].append(np.array(img))
            except IndexError:
                print(f"Could not extract information from filename {filename} using provided regex")
        else:
            print(f"Filename {filename} did not match provided regex")
            continue
        
    return images_by_key

def compare_cellpose_masks_v1(src, verbose=False, save=False):
    
    from .io import _read_mask
    from .plot import visualize_masks, plot_comparison_results, visualize_cellpose_masks
    from .utils import extract_boundaries, boundary_f1_score, compute_segmentation_ap, jaccard_index

    import os
    import numpy as np
    from skimage.measure import label

    # Collect all subdirectories in src
    dirs = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]

    dirs.sort()  # Optional: sort directories if needed

    # Get common files in all directories
    common_files = set(os.listdir(dirs[0]))
    for d in dirs[1:]:
        common_files.intersection_update(os.listdir(d))
    common_files = list(common_files)

    results = []
    conditions = [os.path.basename(d) for d in dirs]

    for index, filename in enumerate(common_files):
        print(f'Processing image {index+1}/{len(common_files)}', end='\r', flush=True)
        paths = [os.path.join(d, filename) for d in dirs]

        # Check if file exists in all directories
        if not all(os.path.exists(path) for path in paths):
            print(f'Skipping {filename} as it is not present in all directories.')
            continue

        masks = [_read_mask(path) for path in paths]
        boundaries = [extract_boundaries(mask) for mask in masks]

        if verbose:
            visualize_cellpose_masks(masks, titles=conditions, comparison_title=f"Masks Comparison for {filename}", save=save, src=src)

        # Initialize data structure for results
        file_results = {'filename': filename}

        # Compare each mask with each other
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                condition_i = conditions[i]
                condition_j = conditions[j]
                mask_i = masks[i]
                mask_j = masks[j]

                # Compute metrics
                boundary_f1 = boundary_f1_score(mask_i, mask_j)
                jaccard = jaccard_index(mask_i, mask_j)
                average_precision = compute_segmentation_ap(mask_i, mask_j)

                # Store results
                file_results[f'jaccard_{condition_i}_{condition_j}'] = jaccard
                file_results[f'boundary_f1_{condition_i}_{condition_j}'] = boundary_f1
                file_results[f'average_precision_{condition_i}_{condition_j}'] = average_precision

        results.append(file_results)

    fig = plot_comparison_results(results)

    save_results_and_figure(src, fig, results)

    return results, fig