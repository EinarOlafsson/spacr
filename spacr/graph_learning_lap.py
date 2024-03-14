import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
from sklearn.preprocessing import LabelEncoder

class BasicGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(BasicGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        return x
    
def create_graph(cells_df, wells_df, well_id='prc', infer_id='gene'):
    # Nodes: Combine cell and well nodes
    num_cells = cells_df.shape[0]
    num_wells = len(wells_df[well_id].unique())
    num_genes = len(wells_df[infer_id].unique())
    
    # Edges: You need to define edges based on your cells' and wells' relationships
    edge_index = [...]  # Fill with (source, target) index pairs
    
    # Node features: Depending on your data, this might include measurements for cells, and gene fractions for wells
    x = [...]  # Feature matrix
    
    # Labels: If you're predicting something specific, like gene knockouts
    y = [...]  # Target labels for nodes
    
    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(y))
    
    return data

def create_graph(cells_df, wells_df, well_id='prc', infer_id='gene'):
    # Assume cells_df and wells_df are preprocessed to include 'well_id' and 'gene_id' as encoded fields
    
    # Node feature creation (this is highly data-dependent; consider cells_df features like cell area, intensity, etc.)
    cell_features = [...]  # Extract cell features into a matrix
    well_features = [...]  # Optional: Aggregate or represent well features
    gene_features = [...]  # Optional: Represent gene features
    
    x = np.concatenate([cell_features, well_features, gene_features], axis=0)
    
    # Edge index construction
    edge_index = [...]  # You'll need to construct this based on your data relationships
    
    # Labels (assuming you have a column 'label' in cells_df for cell-level labels)
    y = cells_df['label'].values
    
    # If y needs to include well and gene nodes, you'll have to expand it appropriately, possibly with dummy labels
    
    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(y, dtype=torch.long))  # Adjust dtype as needed
    
    return data


def train_gnn(cell_data_loc, well_data_loc, well_id='prc', infer_id='gene', lr=0.01):

    # Example loading step
    cells_df = pd.read_csv(cell_data_loc)
    wells_df = pd.read_csv(well_data_loc)

    well_encoder = LabelEncoder()
    cells_df['well_id'] = well_encoder.fit_transform(cells_df[well_id])

    gene_encoder = LabelEncoder()
    wells_df['gene_id'] = gene_encoder.fit_transform(wells_df[infer_id])

    graph_data = create_graph(cells_df, wells_df)


    # Example instantiation and use
    model = BasicGNN(num_node_features=..., num_classes=...)

    # Assuming binary classification for simplicity
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = criterion(out, graph_data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')