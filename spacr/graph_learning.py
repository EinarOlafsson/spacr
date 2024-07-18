import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch, dgl
import pandas as pd
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from PIL import Image
import dgl.nn.pytorch as dglnn
from sklearn.datasets import make_classification
from .utils import SelectChannels
from IPython.display import display

# approach outline
#
#    1. Data Preparation:
#        Test Mode: Load MNIST data and generate synthetic gRNA data.
#        Real Data: Load image paths and sequencing data as fractions.
#
#    2. Graph Construction:
#        Each well is represented as a graph.
#        Each graph has cell nodes (with image features) and gRNA nodes (with gRNA fraction features).
#        Each cell node is connected to each gRNA node within the same well.
#
#    3. Model Training:
#        Use an encoder-decoder architecture with the Graph Transformer model.
#        The encoder processes the cell and gRNA nodes.
#        The decoder outputs the phenotype score for each cell node.
#        The model is trained on all wells (including positive and negative controls).
#        The model learns to score the gRNA in column 1 (negative control) as 0 and the gRNA in column 2 (positive control) as 1 based on the cell features.
#
#    4. Model Application:
#        Apply the trained model to all wells to get classification probabilities.
#
#    5. Evaluation:
#        Evaluate the model's performance using the control wells.
#
#    6. Association Analysis:
#        Analyze the association between gRNAs and the classification scores.
#
# The model learns the associations between cell features and phenotype scores based on the controls and then generalizes this learning to the screening wells.

# Load MNIST data for testing
def load_mnist_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return mnist_train, mnist_test

# Generate synthetic gRNA data
def generate_synthetic_grna_data(n_samples, n_features):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=0, n_classes=2, random_state=42)
    synthetic_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    synthetic_data['label'] = y
    return synthetic_data

# Preprocess image
def preprocess_image(image_path, image_size=224, channels=[1,2,3], normalize=True):

    if normalize:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            SelectChannels(channels),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=(image_size, image_size)),
            SelectChannels(channels)])
    
    image = Image.open(image_path).convert('RGB')
    return preprocess(image)

def extract_metadata_from_path(path):
    """
    Extract metadata from the image path.
    The path format is expected to be plate_well_field_objectnumber.png

    Parameters:
    path (str): The path to the image file.

    Returns:
    dict: A dictionary with the extracted metadata.
    """
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)

    # Ensure the file has the correct extension
    if ext.lower() != '.png':
        raise ValueError("Expected a .png file")

    # Split the name by underscores
    parts = name.split('_')
    if len(parts) != 4:
        raise ValueError("Expected filename format: plate_well_field_objectnumber.png")

    plate, well, field, object_number = parts

    return {'plate': plate, 'well': well,'field': field, 'object_number': object_number}

# Load images
def load_images(image_paths, image_size=224, channels=[1,2,3], normalize=True):
    images = []
    metadata_list = []
    for path in image_paths:
        image = preprocess_image(path, image_size, channels, normalize)
        images.append(image)
        metadata = extract_metadata_from_path(path)  # Extract metadata from image path or database
        metadata_list.append(metadata)
    return torch.stack(images), metadata_list

# Normalize sequencing data
def normalize_sequencing_data(sequencing_data):
    scaler = StandardScaler()
    sequencing_data.iloc[:, 2:] = scaler.fit_transform(sequencing_data.iloc[:, 2:])
    return sequencing_data

# Construct graph for each well
def construct_well_graph(images, image_metadata, grna_data):
    cell_nodes = len(images)
    grna_nodes = grna_data.shape[0]
    
    graph = dgl.DGLGraph()
    graph.add_nodes(cell_nodes + grna_nodes)

    cell_features = torch.stack(images)
    grna_features = torch.tensor(grna_data).float()

    features = torch.cat([cell_features, grna_features], dim=0)
    graph.ndata['features'] = features

    for i in range(cell_nodes):
        for j in range(cell_nodes, cell_nodes + grna_nodes):
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    
    return graph

def create_graphs_for_wells(images, metadata_list, sequencing_data):
    graphs = []
    labels = []

    for well in sequencing_data['well'].unique():
        well_images = [img for img, meta in zip(images, metadata_list) if meta['well'] == well]
        well_metadata = [meta for meta in metadata_list if meta['well'] == well]
        well_grna_data = sequencing_data[sequencing_data['well'] == well].iloc[:, 2:].values

        graph = construct_well_graph(well_images, well_metadata, well_grna_data)
        graphs.append(graph)

        if well_metadata[0]['column'] == 1:  # Negative control
            labels.append(0)
        elif well_metadata[0]['column'] == 2:  # Positive control
            labels.append(1)
        else:
            labels.append(-1)  # Screen wells, will be used for evaluation

    return graphs, labels

# Define Encoder-Decoder Transformer Model
class Encoder(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(Encoder, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, hidden_feats)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_feats, out_feats):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_feats, out_feats)

    def forward(self, x):
        return self.linear(x)

class GraphTransformer(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphTransformer, self).__init__()
        self.encoder = Encoder(in_feats, hidden_feats)
        self.decoder = Decoder(hidden_feats, out_feats)

    def forward(self, g, features):
        x = self.encoder(g, features)
        with g.local_scope():
            g.ndata['h'] = x
            hg = dgl.mean_nodes(g, 'h')
        return self.decoder(hg)

def train(graphs, labels, model, loss_fn, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for graph, label in zip(graphs, labels):
            if label == -1:
                continue  # Skip screen wells for training
            
            features = graph.ndata['features']
            logits = model(graph, features)
            loss = loss_fn(logits, torch.tensor([label]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == label).sum().item()
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f'Epoch {epoch}, Loss: {total_loss / total:.4f}, Accuracy: {accuracy * 100:.2f}%')

def apply_model(graphs, model):
    model.eval()
    results = []

    with torch.no_grad():
        for graph in graphs:
            features = graph.ndata['features']
            logits = model(graph, features)
            probabilities = torch.softmax(logits, dim=1)
            results.append(probabilities[:, 1].item())
    
    return results

def analyze_associations(probabilities, sequencing_data):
    # Analyze associations between gRNAs and classification scores
    sequencing_data['positive_prob'] = probabilities
    return sequencing_data.groupby('gRNA').positive_prob.mean().sort_values(ascending=False)

def process_sequencing_df(seq):

    if isinstance(seq, pd.DataFrame):
        sequencing_df = seq
    elif isinstance(seq, str):
        sequencing_df = pd.read_csv(seq)

    # Check if 'plate_row' column exists and split into 'plate' and 'row'
    if 'plate_row' in sequencing_df.columns:
        sequencing_df[['plate', 'row']] = sequencing_df['plate_row'].str.split('_', expand=True)

    # Check if 'plate', 'row' and 'col' or 'plate', 'row' and 'column' exist
    if {'plate', 'row', 'col'}.issubset(sequencing_df.columns) or {'plate', 'row', 'column'}.issubset(sequencing_df.columns):
        if 'col' in sequencing_df.columns:
            sequencing_df['prc'] = sequencing_df[['plate', 'row', 'col']].agg('_'.join, axis=1)
        elif 'column' in sequencing_df.columns:
            sequencing_df['prc'] = sequencing_df[['plate', 'row', 'column']].agg('_'.join, axis=1)

    # Check if 'count', 'total_reads', 'read_fraction', 'grna' exist and create new dataframe
    if {'count', 'total_reads', 'read_fraction', 'grna'}.issubset(sequencing_df.columns):
        new_df = sequencing_df[['grna', 'prc', 'count', 'total_reads', 'read_fraction']]
        return new_df
    
    return sequencing_df

def train_graph_transformer(src, lr=0.01, epochs=100, hidden_feats=128, n_classes=2, row_limit=None, image_size=224, channels=[1,2,3], normalize=True, test_mode=False):
    if test_mode:
        # Load MNIST data
        mnist_train, mnist_test = load_mnist_data()
        
        # Generate synthetic gRNA data
        synthetic_grna_data = generate_synthetic_grna_data(len(mnist_train), 10)  # 10 synthetic features
        sequencing_data = synthetic_grna_data
        
        # Load MNIST images and metadata
        images = [] 
        metadata_list = []
        for idx, (img, label) in enumerate(mnist_train):
            images.append(img)
            metadata_list.append({'index': idx, 'plate': 'plate1', 'well': idx, 'column': label})
        images = torch.stack(images)

        # Normalize synthetic sequencing data
        sequencing_data = normalize_sequencing_data(sequencing_data)
    else:
        from .io import _read_and_join_tables
        from .utils import get_db_paths, get_sequencing_paths, correct_paths

        db_paths = get_db_paths(src)
        seq_paths = get_sequencing_paths(src)

        if isinstance(src, str):
            src = [src]

        sequencing_data = pd.DataFrame()
        for seq in seq_paths:
            sequencing_df = pd.read_csv(seq)
            sequencing_df = process_sequencing_df(sequencing_df)
            sequencing_data = pd.concat([sequencing_data, sequencing_df], axis=0)

        all_df = pd.DataFrame()
        image_paths = []
        for i, db_path in enumerate(db_paths):
            df = _read_and_join_tables(db_path, table_names=['png_list'])
            df, image_paths_tmp = correct_paths(df, src[i])
            all_df = pd.concat([all_df, df], axis=0)
            image_paths.extend(image_paths_tmp)

        if row_limit is not None:
            all_df = all_df.sample(n=row_limit, random_state=42)

        images, metadata_list = load_images(image_paths, image_size, channels, normalize)
        sequencing_data = normalize_sequencing_data(sequencing_data)

    # Step 1: Create graphs for each well
    graphs, labels = create_graphs_for_wells(images, metadata_list, sequencing_data)

    # Step 2: Train Graph Transformer Model
    in_feats = graphs[0].ndata['features'].shape[1]
    model = GraphTransformer(in_feats, hidden_feats, n_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train(graphs, labels, model, loss_fn, optimizer, epochs)

    # Step 3: Apply the model to all wells (including screen wells)
    screen_graphs = [graph for graph, label in zip(graphs, labels) if label == -1]
    probabilities = apply_model(screen_graphs, model)

    # Step 4: Analyze associations between gRNAs and classification scores
    associations = analyze_associations(probabilities, sequencing_data)
    print("Top associated gRNAs with positive control phenotype:")
    print(associations.head())

    return model, associations
