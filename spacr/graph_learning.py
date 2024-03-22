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
            print(f"    Cell {idx+1}: {feature.item()}")

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