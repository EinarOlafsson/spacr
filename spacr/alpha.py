from skimage import measure, feature
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, hmean, gmean, mode
import pywt

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim import Adam
import os
import shutil
import random

# Step 1: Data Preparation

# Load images
def load_images(image_dir):
    images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(image_dir, filename))
            images[filename] = img
    return images

# Load sequencing data
def load_sequencing_data(seq_file):
    seq_data = pd.read_csv(seq_file)
    return seq_data

# Step 2: Data Representation

# Image Representation (Using a simple CNN for feature extraction)
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 128)  # Assuming input images are 64x64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Graph Representation
def create_graph(wells, sequencing_data):
    nodes = []
    edges = []
    node_features = []
    
    for well in wells:
        # Add node for each well
        nodes.append(well)
        
        # Get sequencing data for the well
        seq_info = sequencing_data[sequencing_data['well'] == well]
        
        # Create node features based on gene knockouts and abundances
        features = torch.tensor(seq_info['abundance'].values, dtype=torch.float)
        node_features.append(features)
        
        # Define edges (for simplicity, assume fully connected graph)
        for other_well in wells:
            if other_well != well:
                edges.append((wells.index(well), wells.index(other_well)))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.stack(node_features)
    
    data = Data(x=x, edge_index=edge_index)
    return data

# Step 3: Model Architecture

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=8)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Apply attention mechanism
        x, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Step 4: Training

# Training Loop
def train(model, data_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data in data_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            out = model(data.x, data.edge_index, data.batch)
            _, predicted = torch.max(out, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

def spacr_transformer(image_dir, seq_file, nr_grnas=1350, lr=0.001, mode='train'):
    images = load_images(image_dir)
    
    sequencing_data = load_sequencing_data(seq_file)
    wells = sequencing_data['well'].unique()
    graph_data = create_graph(wells, sequencing_data)
    model = GraphTransformer(in_channels=nr_grnas, hidden_channels=128, out_channels=nr_grnas)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    data_list = [graph_data]
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    if mode == 'train':
        train(model, loader, criterion, optimizer)
    elif mode == 'eval':
        evaluate(model, loader)
    else:
        raise ValueError('Invalid mode. Use "train" or "eval".')

from skimage.feature import greycomatrix

from skimage.feature import greycoprops

def _calculate_glcm_features(intensity_image):
    glcm = greycomatrix(img_as_ubyte(intensity_image), distances=[1, 2, 3, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        for i, distance in enumerate([1, 2, 3, 4]):
            for j, angle in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
                features[f'glcm_{prop}_d{distance}_a{angle}'] = greycoprops(glcm, prop)[i, j]
    return features

from skimage.feature import local_binary_pattern

def _calculate_lbp_features(intensity_image, P=8, R=1):
    lbp = local_binary_pattern(intensity_image, P, R, method='uniform')
    lbp_hist, _ = np.histogram(lbp, density=True, bins=np.arange(0, P + 3), range=(0, P + 2))
    return {f'lbp_{i}': val for i, val in enumerate(lbp_hist)}

def _calculate_wavelet_features(intensity_image, wavelet='db1'):
    coeffs = pywt.wavedec2(intensity_image, wavelet=wavelet, level=3)
    features = {}
    for i, coeff in enumerate(coeffs):
        if isinstance(coeff, tuple):
            for j, subband in enumerate(coeff):
                features[f'wavelet_coeff_{i}_{j}_mean'] = np.mean(subband)
                features[f'wavelet_coeff_{i}_{j}_std'] = np.std(subband)
                features[f'wavelet_coeff_{i}_{j}_energy'] = np.sum(subband**2)
        else:
            features[f'wavelet_coeff_{i}_mean'] = np.mean(coeff)
            features[f'wavelet_coeff_{i}_std'] = np.std(coeff)
            features[f'wavelet_coeff_{i}_energy'] = np.sum(coeff**2)
    return features


from .measure import _estimate_blur, _calculate_correlation_object_level, _calculate_homogeneity, _periphery_intensity, _outside_intensity, _calculate_radial_distribution, _create_dataframe, _extended_regionprops_table, _calculate_correlation_object_level

def _intensity_measurements(cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[3, 6, 12, 24], periphery=True, outside=True):
    radial_dist = settings['radial_dist']
    calculate_correlation = settings['calculate_correlation']
    homogeneity = settings['homogeneity']
    distances = settings['homogeneity_distances']
    
    intensity_props = ["label", "centroid_weighted", "centroid_weighted_local", "max_intensity", "mean_intensity", "min_intensity", "integrated_intensity"]
    additional_props = ["standard_deviation_intensity", "median_intensity", "sum_intensity", "intensity_range", "mean_absolute_deviation_intensity", "skewness_intensity", "kurtosis_intensity", "variance_intensity", "mode_intensity", "energy_intensity", "entropy_intensity", "harmonic_mean_intensity", "geometric_mean_intensity"]
    col_lables = ['region_label', 'mean', '5_percentile', '10_percentile', '25_percentile', '50_percentile', '75_percentile', '85_percentile', '95_percentile']
    cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs = [], [], [], []
    ls = ['cell','nucleus','pathogen','cytoplasm']
    labels = [cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask]
    dfs = [cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs]
    
    for i in range(0,channel_arrays.shape[-1]):
        channel = channel_arrays[:, :, i]
        for j, (label, df) in enumerate(zip(labels, dfs)):
            
            if np.max(label) == 0:
                empty_df = pd.DataFrame()
                df.append(empty_df)
                continue
                
            mask_intensity_df = _extended_regionprops_table(label, channel, intensity_props)
            
            # Additional intensity properties
            region_props = measure.regionprops_table(label, intensity_image=channel, properties=['label'])
            intensity_values = [channel[region.coords[:, 0], region.coords[:, 1]] for region in measure.regionprops(label)]
            additional_data = {prop: [] for prop in additional_props}
            
            for values in intensity_values:
                if len(values) == 0:
                    continue
                additional_data["standard_deviation_intensity"].append(np.std(values))
                additional_data["median_intensity"].append(np.median(values))
                additional_data["sum_intensity"].append(np.sum(values))
                additional_data["intensity_range"].append(np.max(values) - np.min(values))
                additional_data["mean_absolute_deviation_intensity"].append(np.mean(np.abs(values - np.mean(values))))
                additional_data["skewness_intensity"].append(skew(values))
                additional_data["kurtosis_intensity"].append(kurtosis(values))
                additional_data["variance_intensity"].append(np.var(values))
                additional_data["mode_intensity"].append(mode(values)[0][0])
                additional_data["energy_intensity"].append(np.sum(values**2))
                additional_data["entropy_intensity"].append(entropy(values))
                additional_data["harmonic_mean_intensity"].append(hmean(values))
                additional_data["geometric_mean_intensity"].append(gmean(values))
            
            for prop in additional_props:
                region_props[prop] = additional_data[prop]
            
            additional_df = pd.DataFrame(region_props)
            mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), additional_df.reset_index(drop=True)], axis=1)
            
            if homogeneity:
                homogeneity_df = _calculate_homogeneity(label, channel, distances)
                mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), homogeneity_df], axis=1)

            if periphery:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    periphery_intensity_stats = _periphery_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(periphery_intensity_stats, columns=[f'periphery_{stat}' for stat in col_lables])],axis=1)

            if outside:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    outside_intensity_stats = _outside_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(outside_intensity_stats, columns=[f'outside_{stat}' for stat in col_lables])], axis=1)

            # Adding GLCM features
            glcm_features = _calculate_glcm_features(channel)
            for k, v in glcm_features.items():
                mask_intensity_df[f'{ls[j]}_channel_{i}_{k}'] = v

            # Adding LBP features
            lbp_features = _calculate_lbp_features(channel)
            for k, v in lbp_features.items():
                mask_intensity_df[f'{ls[j]}_channel_{i}_{k}'] = v
            
            # Adding Wavelet features
            wavelet_features = _calculate_wavelet_features(channel)
            for k, v in wavelet_features.items():
                mask_intensity_df[f'{ls[j]}_channel_{i}_{k}'] = v

            blur_col = [_estimate_blur(channel[label == region_label]) for region_label in mask_intensity_df['label']]
            mask_intensity_df[f'{ls[j]}_channel_{i}_blur'] = blur_col

            mask_intensity_df.columns = [f'{ls[j]}_channel_{i}_{col}' if col != 'label' else col for col in mask_intensity_df.columns]
            df.append(mask_intensity_df)
    
    if radial_dist:
        if np.max(nucleus_mask) != 0:
            nucleus_radial_distributions = _calculate_radial_distribution(cell_mask, nucleus_mask, channel_arrays, num_bins=6)
            nucleus_df = _create_dataframe(nucleus_radial_distributions, 'nucleus')
            dfs[1].append(nucleus_df)
            
        if np.max(nucleus_mask) != 0:
            pathogen_radial_distributions = _calculate_radial_distribution(cell_mask, pathogen_mask, channel_arrays, num_bins=6)
            pathogen_df = _create_dataframe(pathogen_radial_distributions, 'pathogen')
            dfs[2].append(pathogen_df)
        
    if calculate_correlation:
        if channel_arrays.shape[-1] >= 2:
            for i in range(channel_arrays.shape[-1]):
                for j in range(i+1, channel_arrays.shape[-1]):
                    chan_i = channel_arrays[:, :, i]
                    chan_j = channel_arrays[:, :, j]
                    for m, mask in enumerate(labels):
                        coloc_df = _calculate_correlation_object_level(chan_i, chan_j, mask, settings)
                        coloc_df.columns = [f'{ls[m]}_channel_{i}_channel_{j}_{col}' for col in coloc_df.columns]
                        dfs[m].append(coloc_df)
    
    return pd.concat(cell_dfs, axis=1), pd.concat(nucleus_dfs, axis=1), pd.concat(pathogen_dfs, axis=1), pd.concat(cytoplasm_dfs, axis=1)

def sample_and_copy_images(folder_list, nr_of_images, dst):

    if isinstance(folder_list, str):
        folder_list = [folder_list]
        
    # Create the destination folder if it does not exist
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    # Calculate the number of images to sample from each folder
    nr_of_images_per_folder = nr_of_images // len(folder_list)
    
    print(f"Sampling {nr_of_images_per_folder} images from {len(folder_list)} folders...")
    # Initialize a list to hold the paths of the images to be copied
    images_to_copy = []
    
    for folder in folder_list:
        # Get a list of all files in the current folder
        all_files = [os.path.join(folder, file) for file in os.listdir(folder)]
        
        # Filter out non-image files
        image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'))]
        
        # Sample images randomly from the list
        sampled_images = random.sample(image_files, min(nr_of_images_per_folder, len(image_files)))
        
        # Add the sampled images to the list of images to copy
        images_to_copy.extend(sampled_images)
    
    # Copy the sampled images to the destination folder
    for image in images_to_copy:
        shutil.copy(image, os.path.join(dst, os.path.basename(image)))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim

def generate_graphs(sequencing, scores, cell_min, gene_min_read):
    # Load and preprocess sequencing (gene) data
    gene_df = pd.read_csv(sequencing)
    gene_df = gene_df.rename(columns={'prc': 'well_id', 'grna': 'gene_id', 'count': 'read_count'})
    # Filter out genes with read counts less than gene_min_read
    gene_df = gene_df[gene_df['read_count'] >= gene_min_read]
    total_reads_per_well = gene_df.groupby('well_id')['read_count'].sum().reset_index(name='total_reads')
    gene_df = gene_df.merge(total_reads_per_well, on='well_id')
    gene_df['well_read_fraction'] = gene_df['read_count'] / gene_df['total_reads']

    # Load and preprocess cell score data
    cell_df = pd.read_csv(scores)
    cell_df = cell_df[['prcfo', 'prc', 'pred']].rename(columns={'prcfo': 'cell_id', 'prc': 'well_id', 'pred': 'score'})

    # Create a global mapping of gene IDs to indices
    unique_genes = gene_df['gene_id'].unique()
    gene_id_to_index = {gene_id: index for index, gene_id in enumerate(unique_genes)}

    graphs = []
    for well_id in pd.unique(gene_df['well_id']):
        well_genes = gene_df[gene_df['well_id'] == well_id]
        well_cells = cell_df[cell_df['well_id'] == well_id]

        # Skip wells with no cells or genes or with fewer cells than threshold
        if well_cells.empty or well_genes.empty or len(well_cells) < cell_min:
            continue

        # Initialize gene features tensor with zeros for all unique genes
        gene_features = torch.zeros((len(gene_id_to_index), 1), dtype=torch.float)

        # Update gene features tensor with well_read_fraction for genes present in this well
        for _, row in well_genes.iterrows():
            gene_index = gene_id_to_index[row['gene_id']]
            gene_features[gene_index] = torch.tensor([[row['well_read_fraction']]])

        # Prepare cell features (scores)
        cell_features = torch.tensor(well_cells['score'].values, dtype=torch.float).view(-1, 1)

        num_genes = len(gene_id_to_index)
        num_cells = cell_features.size(0)
        num_nodes = num_genes + num_cells

        # Create adjacency matrix connecting each cell to all genes in the well
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for _, row in well_genes.iterrows():
            gene_index = gene_id_to_index[row['gene_id']]
            adj[num_genes:, gene_index] = 1

        graph = {
            'adjacency_matrix': adj,
            'gene_features': gene_features,
            'cell_features': cell_features,
            'num_cells': num_cells,
            'num_genes': num_genes
        }
        graphs.append(graph)

    print(f'Generated dataset with {len(graphs)} graphs')
    return graphs, gene_id_to_index

def print_graphs_info(graphs, gene_id_to_index):
    # Invert the gene_id_to_index mapping for easy lookup
    index_to_gene_id = {v: k for k, v in gene_id_to_index.items()}

    for i, graph in enumerate(graphs, start=1):
        print(f"Graph {i}:")
        num_genes = graph['num_genes']
        num_cells = graph['num_cells']
        gene_features = graph['gene_features']
        cell_features = graph['cell_features']

        print(f"  Number of Genes: {num_genes}")
        print(f"  Number of Cells: {num_cells}")

        # Identify genes present in the graph based on non-zero feature values
        present_genes = [index_to_gene_id[idx] for idx, feature in enumerate(gene_features) if feature.item() > 0]
        print("  Genes present in this Graph:", present_genes)

        # Display gene features for genes present in the graph
        print("  Gene Features:")
        for gene_id in present_genes:
            idx = gene_id_to_index[gene_id]
            print(f"    {gene_id}: {gene_features[idx].item()}")

        # Display a sample of cell features, for brevity
        print("  Cell Features (sample):")
        for idx, feature in enumerate(cell_features[:min(5, len(cell_features))]):
            print(f"Cell {idx+1}: {feature.item()}")

        print("-" * 40)

class Attention(nn.Module):
    def __init__(self, feature_dim, attn_dim, dropout_rate=0.1):
        super(Attention, self).__init__()
        self.query = nn.Linear(feature_dim, attn_dim)
        self.key = nn.Linear(feature_dim, attn_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = 1.0 / (attn_dim ** 0.5)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, gene_features, cell_features):
        # Queries come from the cell features
        q = self.query(cell_features)
        # Keys and values come from the gene features
        k = self.key(gene_features)
        v = self.value(gene_features)
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)  

        # Apply attention weights to the values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights

class GraphTransformer(nn.Module):
    def __init__(self, gene_feature_size, cell_feature_size, hidden_dim, output_dim, attn_dim, dropout_rate=0.1):
        super(GraphTransformer, self).__init__()
        self.gene_transform = nn.Linear(gene_feature_size, hidden_dim)
        self.cell_transform = nn.Linear(cell_feature_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Attention layer to let each cell attend to all genes
        self.attention = Attention(hidden_dim, attn_dim)

        # This layer is used to transform the combined features after attention
        self.combine_transform = nn.Linear(2 * hidden_dim, hidden_dim)

        # Output layer for predicting cell scores, ensuring it matches the number of cells
        self.cell_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, adjacency_matrix, gene_features, cell_features):
        # Apply initial transformation to gene and cell features
        transformed_gene_features = F.relu(self.gene_transform(gene_features))
        transformed_cell_features = F.relu(self.cell_transform(cell_features))

        # Incorporate attention mechanism
        attn_output, attn_weights = self.attention(transformed_gene_features, transformed_cell_features)

        # Combine the transformed cell features with the attention output features
        combined_cell_features = torch.cat((transformed_cell_features, attn_output), dim=1)
        
        # Apply dropout here as well
        combined_cell_features = self.dropout(combined_cell_features)  

        combined_cell_features = F.relu(self.combine_transform(combined_cell_features))

        # Combine gene and cell features for message passing
        combined_features = torch.cat((transformed_gene_features, combined_cell_features), dim=0)

        # Apply message passing via adjacency matrix multiplication
        message_passed_features = torch.matmul(adjacency_matrix, combined_features)

        # Predict cell scores from the post-message passed cell features
        cell_scores = self.cell_output(message_passed_features[-cell_features.size(0):])

        return cell_scores, attn_weights
    
def train_graph_transformer(graphs, lr=0.01, dropout_rate=0.1, weight_decay=0.00001, epochs=100, save_fldr='', acc_threshold = 0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphTransformer(gene_feature_size=1, cell_feature_size=1, hidden_dim=256, output_dim=1, attn_dim=128, dropout_rate=dropout_rate).to(device)

    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_log = []
    
    accumulate_grad_batches=1
    threshold=acc_threshold
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        optimizer.zero_grad()
        batch_count = 0  # Initialize batch_count
        
        for graph in graphs:
            adjacency_matrix = graph['adjacency_matrix'].to(device)
            gene_features = graph['gene_features'].to(device)
            cell_features = graph['cell_features'].to(device)
            num_cells = graph['num_cells']
            predictions, attn_weights = model(adjacency_matrix, gene_features, cell_features)
            predictions = predictions.squeeze()
            true_scores = cell_features[:num_cells, 0]
            loss = criterion(predictions, true_scores) / accumulate_grad_batches
            loss.backward()

            # Calculate "accuracy"
            with torch.no_grad():
                correct_predictions = (torch.abs(predictions - true_scores) / true_scores <= threshold).sum().item()
                total_correct += correct_predictions
                total_samples += num_cells

            batch_count += 1  # Increment batch_count
            if batch_count % accumulate_grad_batches == 0 or batch_count == len(graphs):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulate_grad_batches
        
        accuracy = total_correct / total_samples
        training_log.append({"Epoch": epoch+1, "Average Loss": total_loss / len(graphs), "Accuracy": accuracy})
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(graphs)}, Accuracy: {accuracy}", end="\r", flush=True)
    
    # Save the training log and model as before
    os.makedirs(save_fldr, exist_ok=True)
    log_path = os.path.join(save_fldr, 'training_log.csv')
    training_log_df = pd.DataFrame(training_log)
    training_log_df.to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")
    
    model_path = os.path.join(save_fldr, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model
        
def annotate_cells_with_genes(graphs, model, gene_id_to_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    annotated_data = []

    with torch.no_grad():
        for graph in graphs:
            adjacency_matrix = graph['adjacency_matrix'].to(device)
            gene_features = graph['gene_features'].to(device)
            cell_features = graph['cell_features'].to(device)

            predictions, attn_weights = model(adjacency_matrix, gene_features, cell_features)
            predictions = np.atleast_1d(predictions.squeeze().cpu().numpy())
            attn_weights = np.atleast_2d(attn_weights.squeeze().cpu().numpy())

            # This approach assumes all genes in gene_id_to_index are used in the model.
            # Create a list of gene IDs present in this specific graph.
            present_gene_ids = [key for key, value in gene_id_to_index.items() if value < gene_features.size(0)]

            for cell_idx in range(cell_features.size(0)):
                true_score = cell_features[cell_idx, 0].item()
                predicted_score = predictions[cell_idx]
                
                # Find the index of the most probable gene. 
                most_probable_gene_idx = attn_weights[cell_idx].argmax()

                if len(present_gene_ids) > most_probable_gene_idx:  # Ensure index is within the range
                    most_probable_gene_id = present_gene_ids[most_probable_gene_idx]
                    most_probable_gene_score = attn_weights[cell_idx, most_probable_gene_idx] if attn_weights.ndim > 1 else attn_weights[most_probable_gene_idx]

                    annotated_data.append({
                        "Cell ID": cell_idx,
                        "Most Probable Gene": most_probable_gene_id,
                        "Cell Score": true_score,
                        "Predicted Cell Score": predicted_score,
                        "Probability Score for Highest Gene": most_probable_gene_score
                    })
                else:
                    # Handle the case where the index is out of bounds - this should not happen but is here for robustness
                    print("Error: Gene index out of bounds. This might indicate a mismatch in the model's output.")

    return pd.DataFrame(annotated_data)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Let's assume that the feature embedding part and the dataset loading part
# has already been taken care of, and your data is already in the format
# suitable for PyTorch (i.e., Tensors).

class FeatureEmbedder(nn.Module):
    def __init__(self, vocab_sizes, embedding_size):
        super(FeatureEmbedder, self).__init__()
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(num_embeddings=vocab_size+1, 
                              embedding_dim=embedding_size, 
                              padding_idx=vocab_size)
            for key, vocab_size in vocab_sizes.items()
        })
        # Adding the 'visit' embedding
        self.embeddings['visit'] = nn.Parameter(torch.zeros(1, embedding_size))

    def forward(self, feature_map, max_num_codes):
        # Implementation will depend on how you want to handle sparse data
        # This is just a placeholder
        embeddings = {}
        masks = {}
        for key, tensor in feature_map.items():
            embeddings[key] = self.embeddings[key](tensor.long())
            mask = torch.ones_like(tensor, dtype=torch.float32)
            masks[key] = mask.unsqueeze(-1)
        
        # Batch size hardcoded for simplicity in example
        batch_size = 1  # Replace with actual batch size
        embeddings['visit'] = self.embeddings['visit'].expand(batch_size, -1, -1)
        masks['visit'] = torch.ones(batch_size, 1)
        
        return embeddings, masks

class GraphConvolutionalTransformer(nn.Module):
    def __init__(self, embedding_size=128, num_attention_heads=1, **kwargs):
        super(GraphConvolutionalTransformer, self).__init__()
        # Transformer Blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_attention_heads,
                batch_first=True) 
            for _ in range(kwargs.get('num_transformer_stack', 3))
        ])
        # Output Layer for Classification
        self.output_layer = nn.Linear(embedding_size, 1)

    def feedforward(self, features, mask=None, training=None):
        # Implement feedforward logic (placeholder)
        pass

    def forward(self, embeddings, masks, mask=None, training=False):
        features = embeddings
        attentions = []  # Storing attentions if needed
        
        # Pass through each Transformer block
        for layer in self.layers:
            features = layer(features)  # Apply transformer encoding here
            
            if mask is not None:
                features = features * mask
            
        logits = self.output_layer(features[:, 0, :])  # Using the 'visit' embedding for classification
        return logits, attentions

# Usage Example
#vocab_sizes = {'dx_ints':3249, 'proc_ints':2210}
#embedding_size = 128
#gct_params = {
#    'embedding_size': embedding_size,
#    'num_transformer_stack': 3,
#    'num_attention_heads': 1
#}
#feature_embedder = FeatureEmbedder(vocab_sizes, embedding_size)
#gct_model = GraphConvolutionalTransformer(**gct_params)
#
# Assume `feature_map` is a dictionary of tensors, and `max_num_codes` is provided
#embeddings, masks = feature_embedder(feature_map, max_num_codes)
#logits, attentions = gct_model(embeddings, masks)

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import umap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal
from sklearn.cluster import KMeans
from scipy import stats

def load_image(image_path):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def extract_features(image_paths, resnet=resnet50):
    """Extract features from images using a pre-trained ResNet model."""
    model = resnet(pretrained=True)
    model = model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last classification layer

    features = []
    for image_path in image_paths:
        image = load_image(image_path)
        with torch.no_grad():
            feature = model(image).squeeze().numpy()
        features.append(feature)

    return np.array(features)

def check_normality(series):
    """Helper function to check if a feature is normally distributed."""
    k2, p = stats.normaltest(series)
    alpha = 0.05
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        return False
    return True

def random_forest_feature_importance(all_df, cluster_col='cluster'):
    """Random Forest feature importance."""
    numeric_features = all_df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_features:
        numeric_features.remove(cluster_col)

    X = all_df[numeric_features]
    y = all_df[cluster_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    return importance_df

def perform_statistical_tests(all_df, cluster_col='cluster'):
    """Perform ANOVA or Kruskal-Wallis tests depending on normality of features."""
    numeric_features = all_df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_features:
        numeric_features.remove(cluster_col)
    
    anova_results = []
    kruskal_results = []

    for feature in numeric_features:
        groups = [all_df[all_df[cluster_col] == label][feature] for label in np.unique(all_df[cluster_col])]
        
        if check_normality(all_df[feature]):
            stat, p = f_oneway(*groups)
            anova_results.append((feature, stat, p))
        else:
            stat, p = kruskal(*groups)
            kruskal_results.append((feature, stat, p))
    
    anova_df = pd.DataFrame(anova_results, columns=['Feature', 'ANOVA_Statistic', 'ANOVA_pValue'])
    kruskal_df = pd.DataFrame(kruskal_results, columns=['Feature', 'Kruskal_Statistic', 'Kruskal_pValue'])

    return anova_df, kruskal_df

def combine_results(rf_df, anova_df, kruskal_df):
    """Combine the results into a single DataFrame."""
    combined_df = rf_df.merge(anova_df, on='Feature', how='left')
    combined_df = combined_df.merge(kruskal_df, on='Feature', how='left')
    return combined_df

def cluster_feature_analysis(all_df, cluster_col='cluster'):
    """
    Perform Random Forest feature importance, ANOVA for normally distributed features,
    and Kruskal-Wallis for non-normally distributed features. Combine results into a single DataFrame.
    """
    rf_df = random_forest_feature_importance(all_df, cluster_col)
    anova_df, kruskal_df = perform_statistical_tests(all_df, cluster_col)
    combined_df = combine_results(rf_df, anova_df, kruskal_df)
    return combined_df