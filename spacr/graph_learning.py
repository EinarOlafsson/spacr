import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from collections import defaultdict
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree, add_self_loops, softmax
from torch_geometric.loader import DataLoader, NeighborSampler
from sklearn.metrics import mean_squared_error
from torch_geometric.nn import SAGEConv, global_mean_pool, Linear, TransformerConv, GCNConv, GATConv, MessagePassing
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn import Linear, Module
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import MessagePassing

def collate(batch):
    data_list = [data for _, data in batch]
    return Batch.from_data_list(data_list)


def generate_well_graphs(sequencing, scores):
    # Load and preprocess sequencing data
    gene_df = pd.read_csv(sequencing)
    gene_df = gene_df.rename(columns={'prc': 'well_id', 'grna': 'gene_id', 'count': 'read_count'})
    total_reads_per_well = gene_df.groupby('well_id')['read_count'].sum().reset_index(name='total_reads')
    gene_df = gene_df.merge(total_reads_per_well, on='well_id')
    gene_df['well_read_fraction'] = gene_df['read_count'] / gene_df['total_reads']

    # Load and preprocess cell score data
    cell_df = pd.read_csv(scores)
    cell_df = cell_df[['prcfo', 'prc', 'pred']].rename(columns={'prcfo': 'cell_id', 'prc': 'well_id', 'pred': 'score'})

    # Initialize mappings
    gene_id_to_index = {gene: i for i, gene in enumerate(gene_df['gene_id'].unique())}
    cell_id_to_index = {cell: i + len(gene_id_to_index) for i, cell in enumerate(cell_df['cell_id'].unique())}

    # Initialize a dictionary to store edge information for each well subgraph
    wells_subgraphs = defaultdict(lambda: {'edge_index': [], 'edge_attr': []})

    # Associate each cell with all genes in the same well
    for well_id, group in gene_df.groupby('well_id'):
        if well_id in cell_df['well_id'].values:
            cell_indices = cell_df[cell_df['well_id'] == well_id]['cell_id'].map(cell_id_to_index).values
            gene_indices = group['gene_id'].map(gene_id_to_index).values
            fractions = group['well_read_fraction'].values

            for cell_idx in cell_indices:
                for gene_idx, fraction in zip(gene_indices, fractions):
                    wells_subgraphs[well_id]['edge_index'].append([cell_idx, gene_idx])
                    wells_subgraphs[well_id]['edge_attr'].append([fraction])

    # Process well subgraphs into PyTorch Geometric Data objects
    well_data_list = []
    for well_id, subgraph in wells_subgraphs.items():
        edge_index = torch.tensor(subgraph['edge_index'], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(subgraph['edge_attr'], dtype=torch.float)
        num_nodes = max(max(edge) for edge in subgraph['edge_index']) + 1
        x = torch.ones((num_nodes, 1))  # Feature matrix with a single feature set to 1 for each node

        # Retrieve cell scores for the current well
        cell_scores = cell_df[cell_df['well_id'] == well_id]['score'].values
        # Create a tensor for cell scores, ensuring the order matches that of the nodes in the graph
        y = torch.tensor(cell_scores, dtype=torch.float)
        
        subgraph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        well_data_list.append((well_id, subgraph_data))
    
    return well_data_list, gene_id_to_index, len(gene_id_to_index), cell_id_to_index

class CustomTransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, beta=False, dropout=0.0, edge_dim=None):
        super().__init__(node_dim=0, aggr='add')  # Specify 'add' as the aggregation method
        # Initialize the layers and parameters...
        # Rest of init...
        
        # Ensure that the scale is a tensor and properly moved to the device during initialization
        self.scale = torch.sqrt(torch.tensor(out_channels / heads, dtype=torch.float))
        
    def reset_parameters(self):
        # Reset parameters...
        self.scale.data = torch.sqrt(torch.tensor(self.out_channels / self.heads, dtype=torch.float))

    def forward(self, x, edge_index, edge_attr=None):
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        
        # Propagate the messages
        out = self.propagate(edge_index, x=(query, key, value), edge_attr=edge_attr, size=None)
        
        # Reshape and concatenate head outputs if required
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Apply root node transformation with skip connection if required
        if self.root_weight:
            out = out + self.lin_root(x[:out.size(0), :])
        
        return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        # Compute messages
        # This needs to be implemented based on your model's specifics
        query, key, value = x_i[0], x_j[1], x_j[2]
        # Compute the attention scores
        alpha = (query * key).sum(dim=-1) / self.scale
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Apply attention scores to the values
        out = value * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.heads * self.out_channels)


class GraphTransformer(torch.nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.1):
        super(GraphTransformer, self).__init__()
        # Assuming you want to predict a single value per graph, adjust the out_channels as needed.
        num_heads = 4  # Example: 4 attention heads
        out_channels = 1  # Example: predicting a single score per graph
        self.conv1 = CustomTransformerConv(num_node_features, 128, heads=num_heads, dropout=dropout_rate, edge_dim=1)
        self.conv2 = CustomTransformerConv(128 * num_heads, 256, heads=num_heads, dropout=dropout_rate, edge_dim=1)
        self.lin = Linear(256 * num_heads, out_channels)  # Adjusted for a single output feature

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Assuming you want to pool graph features to predict a single value per graph
        x = global_mean_pool(x, batch)  # Pool to get one graph-level representation
        x = self.lin(x)  # Predict a single value per graph

        return x

def train_graph_network(graph_data_list, feature_size, model_path, batch_size=8, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphTransformer(num_node_features=feature_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    data_loader = TorchDataLoader(graph_data_list, batch_size=batch_size, shuffle=True, collate_fn=collate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.view(-1), data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}')
    
    torch.save(model.state_dict(), model_path)