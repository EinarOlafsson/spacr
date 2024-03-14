import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#1. Cell Nodes: Represent individual cells. Each cell node could have attributes like cell area, nuclear area, and a CNN-based phenotype score. These nodes could be visualized as circles labeled with "C".
#2. Well Nodes: Represent wells in the 384-well plates. Wells are intermediary nodes that link cells to genes based on the experimental setup. Each well could contain multiple cells and be associated with certain gene knockouts. These nodes might not have direct attributes in the schematic but serve to connect cell nodes to gene nodes. These can be visualized as squares labeled with "W".
#3. Gene Nodes: Represent genes that have been knocked out. Gene nodes are connected to well nodes, indicating which genes are knocked out in each well. Attributes might include the fraction of sequencing reads for that gene, indicating its relative abundance or importance in the well. These nodes can be visualized as diamonds labeled with "G".
    
# Define a simple GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 16)  # Assume node features are 1-dimensional for simplicity
        self.conv2 = GCNConv(16, 32)
        self.out = Linear(32, 1)  # Predicting a single score for each cell/well

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Two layers of GCN convolution
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global mean pooling
        x = global_mean_pool(x, batch=torch.tensor([0, 0, 0]))  # Assume all nodes belong to the same graph
        x = self.out(x)
        return x

def construct_graph(cell_data_loc, well_data_loc, well_id='prc', infer_id='gene', features=[]):
    
    # Example loading step
    cells_df = pd.read_csv(cell_data_loc)
    wells_df = pd.read_csv(well_data_loc)
    
    # Encode categorical data
    well_encoder = LabelEncoder()
    gene_encoder = LabelEncoder()
    
    cells_df['well_id'] = well_encoder.fit_transform(cells_df[well_id])
    wells_df['gene_id'] = gene_encoder.fit_transform(wells_df[infer_id])
    
    # Assume cell features are in columns ['feature1', 'feature2', ...]
    cell_features = torch.tensor(cells_df[[features]].values, dtype=torch.float)
    
    # Creating nodes for cells and assigning phenotype scores as labels
    y = torch.tensor(cells_df['phenotype_score'].values, dtype=torch.float).unsqueeze(1)
    
    # Constructing edges (this is simplified; you should define edges based on your data structure)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long).t().contiguous()
    
    graphdata = Data(x=cell_features, edge_index=torch.tensor(edge_index, dtype=torch.long), y=y)
    return graphdata, len(features)


def train_gnn(cell_data_loc, well_data_loc, well_id='prc', infer_id='gene', lr=0.01, epochs=100):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, nr_of_features = construct_graph(cell_data_loc, well_data_loc).to(device)
    
    model = GNN(feature_size=nr_of_features).to(device)

    # Assuming binary classification for simplicity
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            
    return model