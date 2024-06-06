from skimage import measure, feature
from skimage.filters import gabor
from skimage.color import rgb2gray
#from skimage.feature.texture import greycomatrix, greycoprops, local_binary_pattern
from skimage.util import img_as_ubyte
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, hmean, gmean, mode
import pywt

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
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

def _calculate_glcm_features(intensity_image):
    glcm = greycomatrix(img_as_ubyte(intensity_image), distances=[1, 2, 3, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        for i, distance in enumerate([1, 2, 3, 4]):
            for j, angle in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
                features[f'glcm_{prop}_d{distance}_a{angle}'] = greycoprops(glcm, prop)[i, j]
    return features

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