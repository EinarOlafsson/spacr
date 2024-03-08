import umap
import random
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from IPython.display import display

from .logger import log_function_call

# Create a function to check if images overlap
def check_overlap(current_position, other_positions, threshold):
    for other_position in other_positions:
        distance = np.linalg.norm(np.array(current_position) - np.array(other_position))
        if distance < threshold:
            return True
    return False

def remove_highly_correlated_columns(df, threshold):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(to_drop, axis=1)

def hyperparameter_search(db_path, tables, filter_by=None, sample_size=None, umap_params=None, dbscan_params=None, pointsize=2, save=False, remove_highly_correlated=False, log_data=False, verbose=True):
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_path)

    # Read the tables into a list of DataFrames
    dfs = [pd.read_sql_query(f"SELECT * FROM {table_name}", conn) for table_name in tables]
    
    # Concatenate the DataFrames along the columns (axis=1)
    df = pd.concat(dfs, axis=1)
    
    if verbose:
        print(df.columns)
        display(df)
    
    # Filter the DataFrame if filter_by is specified
    if filter_by is not None:
        if filter_by !='morphology':
            cols_to_include = [col for col in df.columns if filter_by in str(col)]
        else:
            cols_to_include = [col for col in df.columns if 'channel' not in str(col)]
        df = df[cols_to_include]

    if sample_size is not None:
        df = df.sample(n=sample_size)

    #Remove non-numerical data
    numeric_data = df.select_dtypes(include=['number'])    
    
    # Remove highly correlated columns if required
    if remove_highly_correlated:
        numeric_data = remove_highly_correlated_columns(df=numeric_data, threshold=95)
    
    if verbose:
        print(f'Columns included in UMAP')
        print(numeric_data.columns.tolist())
        display(numeric_data)
    
    #Log transform data
    if log_data:
        numeric_data = np.log(numeric_data + 1e-6)
    
    #Fill NaN values with columns mean
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Scale the numeric data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True) 
    numeric_data = scaler.fit_transform(numeric_data)
    
    if verbose:
        print(numeric_data)

    # Calculate the grid size
    grid_rows = len(umap_params)
    grid_cols = len(dbscan_params)

    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(20, 20))
    
    # Iterate through the Cartesian product of UMAP and DBSCAN hyperparameters
    for i, umap_param in enumerate(umap_params):
        for j, dbscan_param in enumerate(dbscan_params):
            ax = axs[i, j]

            reducer = umap.UMAP(**umap_param)
            embedding = reducer.fit_transform(numeric_data)

            clustering = DBSCAN(**dbscan_param).fit(embedding)
            labels = clustering.labels_

            # Get unique labels to create a custom legend
            unique_labels = np.unique(labels)
            for label in unique_labels:
                ax.scatter(embedding[labels == label, 0], embedding[labels == label, 1], 
                           s=pointsize, label=f"Cluster {label}")

            ax.set_title(f'UMAP {umap_param}\nDBSCAN {dbscan_param}')
            #ax.set_xlabel('UMAP Dimension 1') # x-axis label
            #ax.set_ylabel('UMAP Dimension 2') # y-axis label
            ax.legend() # Add legend

    plt.tight_layout()
    if save:
        plt.savefig('hyperparameter_search.png')
    else:
        plt.show()

    # Close the database connection
    conn.close()
    return

# Create a function to check if images overlap
def check_overlap(current_position, other_positions, threshold):
    for other_position in other_positions:
        distance = np.linalg.norm(np.array(current_position) - np.array(other_position))
        if distance < threshold:
            return True
    return False

# Define a function to try random positions around a given point
def find_non_overlapping_position(x, y, image_positions, threshold, max_attempts=100):
    offset_range = 10  # Adjust the range for random offsets
    attempts = 0
    while attempts < max_attempts:
        random_offset_x = random.uniform(-offset_range, offset_range)
        random_offset_y = random.uniform(-offset_range, offset_range)
        new_x = x + random_offset_x
        new_y = y + random_offset_y
        if not check_overlap((new_x, new_y), image_positions, threshold):
            return new_x, new_y
        attempts += 1
    return x, y  # Return the original position if no suitable position found


def smooth_hull_lines(cluster_data):
    hull = ConvexHull(cluster_data)

    # Extract vertices of the hull
    vertices = hull.points[hull.vertices]

    # Close the loop
    vertices = np.vstack([vertices, vertices[0, :]])

    # Parameterize the vertices
    tck, u = splprep(vertices.T, u=None, s=0.0)

    # Evaluate spline at new parameter values
    new_points = splev(np.linspace(0, 1, 100), tck)

    return new_points[0], new_points[1]

def generate_image_umap(db_paths, tables=['cell'], visualize='cell', image_nr=100, dot_size=50, n_neighbors=30, min_dist=0.1, metric='cosine', eps=0.5, min_samples=5, filter_by=None, img_zoom=0.3, plot_by_cluster=False, plot_cluster_grids=False, remove_cluster_noise=False, figuresize=20, remove_highly_correlated=True, log_data=False, black_background=False, remove_image_canvas=False, plot_outlines=False, plot_points=True, smooth_lines=False, row_limit=None, verbose=False):    
    
    from .annotate_app import check_for_duplicates
    
    if not isinstance(db_paths, list):
        print(f'Warning: Variable db_paths is not a list. db_paths:{db_paths}')
        return
    
    all_df = pd.DataFrame()
    for db_path in db_paths:
        check_for_duplicates(db_path)
        if verbose:
            print(f'database:{db_path}')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        df = pd.DataFrame()
        for table in tables:
            if table == 'cell':
                object_name = 'object_label'
            if table == 'cytoplasm':
                object_name = 'object_label'
            if table == 'nucleus':
                object_name = 'cell_id'
            if table == 'parasite':
                object_name = 'cell_id'

            print(f'{table}:{object_name}')
            
            # Fetch all data
            c.execute(f'SELECT * FROM {table}')
            data = c.fetchall()
            columns_info = c.execute(f'PRAGMA table_info({table})').fetchall()
            column_names = [col_info[1] for col_info in columns_info]

            # Create a DataFrame from the data
            df_temp = pd.DataFrame(data, columns=column_names)
            df_temp = df_temp.dropna(subset=[object_name])

            if object_name in df_temp.columns:
                if df_temp[object_name].dtype == float:
                    df_temp[object_name] = df_temp[object_name].astype(int)

            df_temp = df_temp.assign(object_label=lambda x: 'o' + x[object_name].astype(int).astype(str))
            
            if verbose:
                display(df_temp)
                
            if 'prfco' in df_temp.columns:
                df_temp = df_temp.drop(columns=['prfco'])

            df_temp = df_temp.assign(prcfo = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col']+ '_' +x['field']+ '_' +x['object_label'])
            df_temp = df_temp.drop(columns=[object_name])
            df = pd.concat([df, df_temp],axis=1)

        #Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        if row_limit != None:
            df = df.sample(n=row_limit, replace=False, random_state=1)

        # Fetch image paths
        c.execute(f'SELECT * FROM png_list')
        data = c.fetchall()
        columns_info = c.execute(f'PRAGMA table_info(png_list)').fetchall()
        column_names = [col_info[1] for col_info in columns_info]
        #column_names = ['png_path', 'file_name', 'plate', 'row', 'col','field','cell_id','prcfo']
        column_names_keep = ['png_path','prcfo']
        image_paths_df = pd.DataFrame(data, columns=column_names)
        image_paths_df = image_paths_df.loc[:, image_paths_df.columns.isin(column_names_keep)]
        
        if visualize is not None:
            object_visualize = visualize+'_png'
            image_paths_df = image_paths_df[image_paths_df['png_path'].str.contains(object_visualize)]

        image_paths_df.set_index('prcfo', inplace=True)
        df.set_index('prcfo', inplace=True)
        df = image_paths_df.merge(df, left_index=True, right_index=True)

        if verbose:
            display(df)
        
        all_df = pd.concat([all_df, df],axis=0)
        df.reset_index(inplace=True)
        image_paths = all_df['png_path'].to_list()
        
        conn.close()
    
    if verbose:
        display(all_df)
    
    # Filter the DataFrame if filter_by is specified
    if filter_by is not None:
        if filter_by !='morphology':
            cols_to_include = [col for col in df.columns if filter_by in str(col)]
        else:
            cols_to_include = [col for col in df.columns if 'channel' not in str(col)]
        df = df[cols_to_include]

    #Remove non-numerical data
    numeric_data = all_df.select_dtypes(include=['number'])    
    
    # Remove highly correlated columns if required
    if remove_highly_correlated:
        numeric_data = remove_highly_correlated_columns(df=numeric_data, threshold=95)
    
    if verbose:
        print(f'Columns included in UMAP')
        print(numeric_data.columns.tolist())
        display(numeric_data)
    
    #Log transform data
    if log_data:
        numeric_data = np.log(numeric_data + 1e-6)
    
    #Fill NaN values with columns mean
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Scale the numeric data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    numeric_data = scaler.fit_transform(numeric_data)
    
    if verbose:
        print(numeric_data)
    
    # Perform UMAP analysis
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        n_components=2,
                        metric=metric, #default='euclidean'
                        output_metric='euclidean', #default='euclidean'
                        n_epochs=10, #default=None
                        learning_rate=0.1,
                        init='spectral',
                        min_dist=min_dist,
                        spread=1.0,
                        low_memory=False,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1.0,
                        repulsion_strength=1.0,
                        negative_sample_rate=5,
                        transform_queue_size=4.0,
                        a=None,
                        b=None,
                        random_state=None,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        target_n_neighbors=-1,
                        target_metric='categorical',
                        target_metric_kwds=None,
                        target_weight=0.5,
                        transform_seed=42,
                        verbose=False)
    
    embedding = reducer.fit_transform(numeric_data)
    
    clustering = DBSCAN(eps=eps,
                        min_samples=min_samples,
                        metric='euclidean',
                        metric_params=None,
                        algorithm='auto',
                        leaf_size=30,
                        p=None,
                        n_jobs=None).fit(embedding)
    
    labels = clustering.labels_
    
    if remove_cluster_noise:
        non_noise_indices = labels != -1
        embedding = embedding[non_noise_indices]
        labels = labels[non_noise_indices]
    
    # Create random integer RGB colors
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_clusters + 1, 4)
    random_colors[:, 3] = 1  # Set alpha channel

    # Set specific colors for the first four clusters
    specific_colors = [
        [155/255, 55/255, 155/255, 1],
        [55/255, 155/255, 155/255, 1],
        [55/255, 155/255, 255/255, 1],
        [255/255, 55/255, 155/255, 1]]
    
    random_colors = np.vstack((specific_colors, random_colors[len(specific_colors):]))

    if remove_cluster_noise == False:
        random_colors = np.vstack(([0, 0, 0, 1], random_colors))

    # Normalize colors to [0, 1]
    normalized_colors = random_colors / 255
    colors_img = [tuple(color) for color in normalized_colors]
    colors = [tuple(color) for color in random_colors]
    
    # Get cluster centers to place the labels
    cluster_centers = [np.mean(embedding[labels == cluster_label], axis=0) for cluster_label in unique_labels]

    # Create mapping from cluster labels to color indices
    label_to_color_index = {label: index for index, label in enumerate(unique_labels)}

    #Generate matplotlib figure
    if black_background:
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
    else:
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        
    fig, ax = plt.subplots(1, 1, figsize=(figuresize,figuresize))
    fontsize = int(figuresize*0.75)
    handles = []
    
    # Plot all points in the embedding
    for cluster_label, color, center in zip(unique_labels, colors, cluster_centers):
        cluster_data = embedding[labels == cluster_label]
        
        if smooth_lines:
            # Check if the cluster has more than 2 points to create a Convex Hull
            if cluster_data.shape[0] > 2:
                x_smooth, y_smooth = smooth_hull_lines(cluster_data)
                if plot_outlines:
                    plt.plot(x_smooth, y_smooth, color=color, linewidth=2)
        else:
            if cluster_data.shape[0] > 2:
                hull = ConvexHull(cluster_data)
                for simplex in hull.simplices:
                    if plot_outlines:
                        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], color=color, linewidth=4) #w =white, k=black
        if plot_points:
            scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=dot_size, c=[color], alpha=0.5, label=f'Cluster {cluster_label if cluster_label != -1 else "Noise"}')
        else:
            scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=dot_size, c=[color], alpha=0, label=f'Cluster {cluster_label if cluster_label != -1 else "Noise"}')
        handles.append(scatter)
        
        # Annotate the cluster center with the cluster label
        if cluster_label != -1:  # Skip noise labeled as -1
            ax.text(center[0], center[1], str(cluster_label), fontsize=12, ha='center', va='center')
    
    # Create a dictionary to track indices for each cluster
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels if label != -1}
    
    if visualize is not None:
        if not plot_by_cluster:
            # Plot images replacing random points
            indices = random.sample(range(len(embedding)), image_nr)
            sampled_embedding = embedding[indices]
            #sampled_image_paths = [image_paths[i] for i in sample_indices]

            for i, index in enumerate(indices):
                x, y = embedding[index]
                img_array = Image.open(image_paths[index])
                img = np.array(img_array)

                if remove_image_canvas:
                    #Get the images for these indices 
                    for index in indices:
                        x, y = embedding[index]
                        img_array = Image.open(image_paths[index])

                        if img_array.mode in ['L', 'I']:  # Grayscale image
                            img_data = np.array(img_array)
                            img_data = img_data / np.max(img_data)  # Normalize to [0, 1]
                            alpha_channel = (img_data > 0).astype(float)  # Create alpha channel
                            img_data_rgb = np.stack([img_data] * 3, axis=-1)  # Convert to RGB
                            img_data_with_alpha = np.dstack([img_data_rgb, alpha_channel])
                        elif img_array.mode == 'RGB':  # RGB image
                            img_data = np.array(img_array)
                            img_data = img_data / 255.0  # Normalize to [0, 1]
                            alpha_channel = (np.sum(img_data, axis=-1) > 0).astype(float)  # Non-black pixels
                            img_data_with_alpha = np.dstack([img_data, alpha_channel])
                        else:
                            raise ValueError(f"Unsupported image mode: {img_array.mode}")

                        imagebox = OffsetImage(img_data_with_alpha, zoom=img_zoom)
                        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                        ax.add_artist(ab)
                else:
                    for i, index in enumerate(indices):
                        x, y = embedding[index]
                        img_array = Image.open(image_paths[index])
                        img = np.array(img_array)
                        imagebox = OffsetImage(img, zoom=img_zoom, cmap='gray')
                        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                        ax.add_artist(ab)

        if plot_by_cluster:

            # Create a dictionary to track indices for each cluster
            cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels if label != -1}

            # Plot images replacing random points for each cluster
            for cluster_label, color, center in zip(unique_labels, colors, cluster_centers):
                if cluster_label == -1:  # Skip noise labeled as -1
                    continue

                # Get 10 random indices for this cluster
                indices = cluster_indices.get(cluster_label, [])
                if len(indices) > image_nr:
                    indices = random.sample(list(indices), image_nr)
                elif len(indices) > 1:
                    indices = random.sample(list(indices), 1)

                if remove_image_canvas:
                    #Get the images for these indices 
                    for index in indices:
                        x, y = embedding[index]
                        img_array = Image.open(image_paths[index])

                        if img_array.mode in ['L', 'I']:  # Grayscale image
                            img_data = np.array(img_array)
                            img_data = img_data / np.max(img_data)  # Normalize to [0, 1]
                            alpha_channel = (img_data > 0).astype(float)  # Create alpha channel
                            img_data_rgb = np.stack([img_data] * 3, axis=-1)  # Convert to RGB
                            img_data_with_alpha = np.dstack([img_data_rgb, alpha_channel])
                        elif img_array.mode == 'RGB':  # RGB image
                            img_data = np.array(img_array)
                            img_data = img_data / 255.0  # Normalize to [0, 1]
                            alpha_channel = (np.sum(img_data, axis=-1) > 0).astype(float)  # Non-black pixels
                            img_data_with_alpha = np.dstack([img_data, alpha_channel])
                        else:
                            raise ValueError(f"Unsupported image mode: {img_array.mode}")

                        imagebox = OffsetImage(img_data_with_alpha, zoom=img_zoom)
                        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                        ax.add_artist(ab)
                else:
                    for i, index in enumerate(indices):
                        x, y = embedding[index]
                        img_array = Image.open(image_paths[index])
                        img = np.array(img_array)
                        imagebox = OffsetImage(img, zoom=img_zoom, cmap='gray')
                        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                        ax.add_artist(ab)

    plt.legend(handles=handles, loc='best', fontsize=fontsize)
    plt.xlabel('UMAP Dimension 1', fontsize=fontsize)
    plt.ylabel('UMAP Dimension 2', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.show()

    if plot_cluster_grids:
        # Determine the number of clusters
        num_clusters = len(unique_labels[unique_labels != -1])

        # Dictionary to keep track of images for each cluster
        cluster_images = {label: [] for label in unique_labels if label != -1}

        # Collect the images for each cluster based on previously selected indices
        for cluster_label, indices in cluster_indices.items():
            if cluster_label == -1:
                continue

            if len(indices) > image_nr:
                indices = random.sample(list(indices), image_nr)
            elif len(indices) > 1:
                indices = random.sample(list(indices), 1)

            for index in indices:
                img_path = image_paths[index]
                img_array = Image.open(img_path)
                img = np.array(img_array)
                cluster_images[cluster_label].append(img)

        # Create a new figure for the cluster grids
        grid_fig, grid_axes = plt.subplots(1, num_clusters, figsize=(figuresize * num_clusters, figuresize), gridspec_kw={'wspace': 0.2, 'hspace': 0})

        # Iterate through the clusters and plot the grids

        if len(cluster_images.keys()) >1:
            for cluster_label, axes in zip(cluster_images.keys(), grid_axes):
                images = cluster_images[cluster_label]
                num_images = len(images)
                grid_size = int(np.ceil(np.sqrt(num_images)))  # Calculate grid size (both rows and columns)
                image_size = 0.9 / grid_size  # Adjusting this value will control the whitespace
                whitespace = (1 - grid_size * image_size) / (grid_size + 1)

                color = colors[label_to_color_index[cluster_label]]  # Retrieve the color for this cluster

                # Fill the entire axes with the cluster color
                axes.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes.transAxes, color=color[:3]))

                axes.set_title(f'Cluster {cluster_label}', fontsize=fontsize*3)
                axes.axis('off')

                for i, img in enumerate(images):
                    row = i // grid_size
                    col = i % grid_size
                    x_pos = (col + 1) * whitespace + col * image_size
                    y_pos = 1 - ((row + 1) * whitespace + (row + 1) * image_size)
                    ax_img = axes.inset_axes([x_pos, y_pos, image_size, image_size], transform=axes.transAxes)
                    ax_img.imshow(img, cmap='gray', aspect='auto')
                    ax_img.axis('off')
                    ax_img.set_aspect('equal') # Ensure that the aspect ratio is equal
                    ax_img.set_facecolor(color[:3]) # Set the inset axes background color
            plt.show()
        else:
            cluster_label = list(cluster_images.keys())[0]
            images = cluster_images[cluster_label]
            num_images = len(images)
            grid_size = int(np.ceil(np.sqrt(num_images)))  # Calculate grid size (both rows and columns)

            fig, axes = plt.subplots(grid_size, grid_size, figsize=(figuresize, figuresize))

            if grid_size == 1:
                # Special case for one image
                axes.imshow(images[0], cmap='gray', aspect='auto')
                axes.axis('off')
            else:
                for i, ax in enumerate(axes.flat):
                    if i < num_images:
                        ax.imshow(images[i], cmap='gray', aspect='auto')
                        ax.set_aspect('equal') # Ensure that the aspect ratio is equal
                        ax.axis('off')
                    else:  # Turn off any remaining empty subplots
                        ax.axis('off')

            plt.suptitle(f'Cluster {cluster_label}', fontsize=fontsize*3, y=0.95)  # Adjust the y-position
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
            plt.show()
    return
    
db_paths = ['/mnt/data/CellVoyager/20x/tsg101/crispr_screen/all/measurements/measurements.db']

generate_image_umap(db_paths=db_paths,
                row_limit = 1000,
                tables=['cytoplasm'],
                visualize='cell',
                image_nr=36,
                dot_size=50,
                n_neighbors=1000,
                min_dist=0.1,
                metric='euclidean',
                eps=0.5,
                min_samples=1000,
                filter_by='channel_0',
                img_zoom=0.3,
                plot_by_cluster=True,
                plot_cluster_grids=True,
                remove_cluster_noise=True,
                remove_highly_correlated=True,
                log_data=True,
                figuresize=60,
                black_background=False,
                remove_image_canvas=False,
                plot_outlines=False,
                plot_points=True,
                smooth_lines=False,
                verbose=True)

generate_image_umap(db_paths=db_paths,
                tables=['cytoplasm'],
                visualize='cytoplasm',
                image_nr=36,
                dot_size=50,
                n_neighbors=1000,
                min_dist=0.1,
                metric='euclidean',
                eps=0.5,
                min_samples=1000,
                filter_by='channel_0',
                img_zoom=0.3,
                plot_by_cluster=True,
                plot_cluster_grids=True,
                remove_cluster_noise=True,
                remove_highly_correlated=True,
                log_data=True,
                figuresize=60,
                black_background=False,
                remove_image_canvas=False,
                plot_outlines=False,
                plot_points=True,
                smooth_lines=False,
                verbose=False)
                
                
db_path = '/mnt/data/CellVoyager/63x/mack/CRCR2P2_20230721_162734/PECCU/measurements/measurements.db'
db = db_path
channels = ['channel_0','channel_1','channel_2', 'channel_3', None]

for channel in channels:
    generate_image_umap(db,
                    tables=['cell','cytoplasm', 'nucleus'],
                    image_nr=36,
                    dot_size=50,
                    n_neighbors=50,
                    min_dist=0.1,
                    metric='euclidean',
                    eps=0.3,
                    min_samples=100,
                    filter_by=channel,
                    img_zoom=0.2,
                    plot_by_cluster=True,
                    plot_cluster_grids=True,
                    remove_cluster_noise=True,
                    remove_highly_correlated=True,
                    log_data=True,
                    figuresize=60,
                    verbose=False)
                    
#db_path = '/mnt/data/CellVoyager/63x/mack/CRCR2P2_20230721_162734/PECCU/measurements/measurements.db'
db_path = '/mnt/data/CellVoyager/20x/tsg101/crispr_screen/all/measurements/measurements.db'
tables = ['cell','cytoplasm','parasite']

# UMAP hyperparameters
umap_params = [{'n_neighbors': 20, 'min_dist': 0.01, 'metric': 'euclidean'},
               {'n_neighbors': 40, 'min_dist': 0.1, 'metric': 'euclidean'}]

# DBSCAN hyperparameters
dbscan_params = [{'eps': 0.3, 'min_samples': 100},
                {'eps': 0.3, 'min_samples': 100}]

hyperparameter_search(db_path,
                      tables=tables,
                      filter_by = 'channel_0',
                      sample_size=5000,
                      umap_params=umap_params,
                      dbscan_params=dbscan_params,
                      remove_highly_correlated=True,
                      log_data=True,
                      pointsize=2,
                      verbose=False)
