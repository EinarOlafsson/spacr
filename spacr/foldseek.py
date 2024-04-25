import os, shutil, subprocess, tarfile, glob, requests, time, random
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_command(command):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stdout)
        print(result.stderr)
        return False
    return True

def add_headers_and_save_csv(input_tsv_path, output_csv_path, results_dir):
    
    headers = [
        'query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen',
        'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits'
    ]

    # Rename the aln_tmscore file to have a .tsv extension if it doesn't already
    input_tsv_path = f"{results_dir}/aln_tmscore"
    if not input_tsv_path.endswith('.tsv'):
        os.rename(input_tsv_path, input_tsv_path + '.tsv')
        input_tsv_path += '.tsv'
    
    # Read the TSV file into a DataFrame
    df = pd.read_csv(input_tsv_path, sep='\t', header=None)

    # Assign headers to the DataFrame
    df.columns = headers

    # Save the DataFrame as a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"File saved as {output_csv_path}")
    
def generate_database(path, base_dir, mode='file'):
    structures_dir = f'{base_dir}/structures'
    os.makedirs(structures_dir, exist_ok=True)
    
    if mode == 'tar':
        if os.path.exists(structures_dir) and not os.listdir(structures_dir):
            if not os.path.exists(path):
                print(f"Structure tar file {path} not found.")
            else:
                tar = tarfile.open(path)
                tar.extractall(path=structures_dir)
                tar.close()
                if not run_command(f"foldseek createdb {structures_dir} {structures_dir}/structures_db"):
                    raise Exception("Failed to create structures database.")

    if mode == 'file':
        if os.path.exists(structures_dir) and not os.listdir(structures_dir):
            if not os.path.exists(path):
                print(f"Structure folder {path} not found.")
            else:
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    new_path = os.path.join(structures_dir, file)
                    #print(path)
                    #print(structures_dir)
                    shutil.copy(file_path, new_path)

                if not run_command(f"foldseek createdb {structures_dir} {structures_dir}/structures_db"):
                    raise Exception("Failed to create structures database.")
    return structures_dir

def align_to_database(structure_fldr_path, base_dir='/home/carruthers/foldseek', cores=25):

    databases_dir = f'{base_dir}/foldseek_databases'
    results_dir = f'{base_dir}/results'
    tmp_dir = f'{base_dir}/tmp'

    os.makedirs(databases_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True) 

    # Check and download PDB database if not exists
    pdb_db_path = os.path.join(databases_dir, "pdb")
    if not os.path.exists(pdb_db_path):
        print("Downloading PDB database...")
        if not run_command(f"foldseek databases PDB {pdb_db_path} {tmp_dir}"):
            raise Exception("Failed to download PDB database.")

    # Check and download AlphaFold database if not exists
    afdb_db_path = os.path.join(databases_dir, "afdb")
    if not os.path.exists(afdb_db_path):
        print("Downloading AlphaFold database...")
        if not run_command(f"foldseek databases Alphafold/Proteome {afdb_db_path} {tmp_dir}"):
            raise Exception("Failed to download AlphaFold database.")

    structures_dir = generate_database(structure_fldr_path, base_dir, mode='file')
            
    for i, targetDB in enumerate([pdb_db_path, afdb_db_path]):
        
        if i == 0:
            results_dir = os.path.join(base_dir, 'results', "pdb")
            os.makedirs(results_dir, exist_ok=True)
            print("Running Foldseek on PDB...")
        if i == 1:
            results_dir = os.path.join(base_dir, 'results', "afdb")
            os.makedirs(results_dir, exist_ok=True)
            print("Running Foldseek on AFdb...")
        
        aln_tmscore = f"{results_dir}/aln_tmscore"
        aln_tmscore_tsv = f"{results_dir}/aln_tmscore.tsv"

        queryDB = f"{structures_dir}/structures_db"
        targetDB = pdb_db_path
        aln = f"{results_dir}/results"    

        if not run_command(f"foldseek search {queryDB} {targetDB} {aln} {tmp_dir} -a --threads {cores}"):
            raise Exception("Foldseek search against PDB failed.")

        if not run_command(f"foldseek aln2tmscore {queryDB} {targetDB} {aln} {aln_tmscore} --threads {cores}"):
            raise Exception("Foldseek aln2tmscore against PDB failed.")

        if not run_command(f"foldseek createtsv {queryDB} {targetDB} {aln} {aln_tmscore} {aln_tmscore_tsv}"):
            raise Exception("Foldseek createtsv against PDB failed.")

        input_tsv_path = f"{results_dir}/aln_tmscore"
        output_csv_path = f"{results_dir}/aln_tmscore.csv"

        # Call the function with the path to your TSV file and the output CSV file path
        add_headers_and_save_csv(input_tsv_path, output_csv_path, results_dir)

def check_uniprot_structure(uniprot_id):
    import requests
    base_url = "https://www.ebi.ac.uk/proteins/api/proteins"
    headers = {"Accept": "application/json"}
    response = requests.get(f"{base_url}/{uniprot_id}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(data)  # Print the whole JSON to examine its structure
    else:
        print(f"Failed to retrieve data for {uniprot_id}: {response.status_code}")

def get_ec_numbers(data):
    try:
        # Navigate through the nested structure with checks at each step
        protein_info = data.get('protein', {})
        recommended_name = protein_info.get('recommendedName', {})
        ec_numbers = recommended_name.get('ecNumber', [])

        # Extract the 'value' field from each EC number entry
        return ", ".join(ec['value'] for ec in ec_numbers if 'value' in ec)
    except Exception as e:
        print(f"Failed to extract EC numbers: {str(e)}")
        return ""  

def process_protein_data(data, verbose=False):
    if data is None:
        return None
    
    uniprot_id = data.get('accession')
    protein_data = {}
    protein_data[uniprot_id] = {
                    'UniProt ID': uniprot_id,
                    'Entry Name': data.get('id'),
                    'Organism': next((name['value'] for name in data.get('organism', {}).get('names', []) if name['type'] == 'scientific'), None),
                    #'Taxonomic Lineage': ", ".join(data.get('organism', {}).get('lineage', [])),
                    'Taxonomy ID': data.get('organism', {}).get('taxonomy'),
                    'Sequence Length': data.get('sequence', {}).get('length'),
                    #'EC Number': ", ".join([ec['value'] for ec in data.get('protein', {}).get('recommendedName', {}).get('ecNumber', [])]),
                    'EC Number': get_ec_numbers(data),
                    'Function': "; ".join([func['text'][0]['value'] for func in data.get('comments', []) if func['type'] == 'FUNCTION']),
                    'Recommended Name': data.get('protein', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                    'Alternative Names': "; ".join([alt['fullName']['value'] for alt in data.get('protein', {}).get('alternativeName', [])]),
                    'GO Biological Process': [],
                    'GO Cellular Component': [],
                    'GO Molecular Function': [],
                    'GO IDs': [],
                    'KEGG': [],
                    'OrthoDB': [],
                    'Sequence': data.get('sequence', {}).get('sequence', ''),
                    'Family and Domains': {},
                    'Catalytic Activity': "; ".join([cat['reaction']['name'] for cat in data.get('comments', []) if cat['type'] == 'CATALYTIC_ACTIVITY']),
                    'Cofactor': "; ".join([cof['cofactors'][0]['name'] for cof in data.get('comments', []) if cof['type'] == 'COFACTOR']),
                    'Enzyme Regulation': "; ".join([reg['text'][0]['value'] for reg in data.get('comments', []) if reg['type'] == 'ENZYME_REGULATION']),
                    'Disease Association': "; ".join([dis['text'][0]['value'] for dis in data.get('comments', []) if dis['type'] == 'DISEASE']),
                    'Interaction Partners': "; ".join([inter['id'] for inter in data.get('dbReferences', []) if inter['type'] == 'InterPro'])
                }

    # Subcellular Location processing
    protein_data[uniprot_id].update({
        'sub_loc_Intermembrane': "",
        'sub_loc_Topological_Domain': "",
        'sub_loc_Subcellular_Location': "",
        'sub_loc_Transmembrane': ""
    })

    for loc in data.get('comments', []):
        if loc['type'] == 'SUBCELLULAR_LOCATION':
            for component in loc.get('locations', []):
                if 'topology' in component:
                    protein_data[uniprot_id]['sub_loc_Topological_Domain'] += component['topology']['value'] + "; "
                if 'orientation' in component:
                    protein_data[uniprot_id]['sub_loc_Intermembrane'] += component['orientation']['value'] + "; "
                if 'location' in component:
                    protein_data[uniprot_id]['sub_loc_Subcellular_Location'] += component['location']['value'] + "; "
                if 'subcellularLocation' in component:
                    protein_data[uniprot_id]['sub_loc_Transmembrane'] += component['subcellularLocation']['value'] + "; "

    # Initialize PTM/Processing details
    ptms = set(ptm['type'] for ptm in data.get('features', []) if ptm['category'] == 'PTM')
    for ptm in ptms:
        protein_data[uniprot_id][ptm] = []

    # Process each PTM type
    for ptm in data.get('features', []):
        if ptm['category'] == 'PTM' and ptm['type'] in protein_data[uniprot_id]:
            ptm_description = ptm.get('description', '')
            ptm_details = f"{ptm_description} (positions {ptm.get('begin')} to {ptm.get('end')})"
            protein_data[uniprot_id][ptm['type']].append(ptm_details)

    # Gene Ontology Annotations
    for go in data.get('dbReferences', []):
        if go['type'] == 'GO' and 'properties' in go:
            go_term = go['properties']['term']
            if go_term.startswith('P:'):
                protein_data[uniprot_id]['GO Biological Process'].append(go_term[2:])
            elif go_term.startswith('C:'):
                protein_data[uniprot_id]['GO Cellular Component'].append(go_term[2:])
            elif go_term.startswith('F:'):
                protein_data[uniprot_id]['GO Molecular Function'].append(go_term[2:])
            protein_data[uniprot_id]['GO IDs'].append(go['id'])

    # External sources
    for xref in data.get('dbReferences', []):
        if xref['type'] == 'KEGG':
            protein_data[uniprot_id]['KEGG'].append(xref['id'])
        elif xref['type'] == 'OrthoDB':
            protein_data[uniprot_id]['OrthoDB'].append(xref['id'])

    # Initialize Family and Domains from 'features'
    for feature in data.get('features', []):
        if feature['type'] in ['DOMAIN', 'MOTIF', 'REGION']:
            domain_key = f"{feature['type']} {feature.get('description', 'N/A')}"
            if domain_key not in protein_data[uniprot_id]:
                protein_data[uniprot_id][domain_key] = f"Positions {feature.get('begin')} to {feature.get('end')}"
    if verbose:
        print(protein_data)
    return protein_data

def fetch_data_for_uniprot_id(uniprot_id):
    """ Fetch data for a single UniProt ID from the UniProt API. """
    base_url = "https://www.ebi.ac.uk/proteins/api/proteins"
    headers = {"Accept": "application/json"}
    request_url = f"{base_url}/{uniprot_id}"
    response = requests.get(request_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data for {uniprot_id}: {response.status_code}")
        return None

def fetch_and_aggregate_functional_data(uniprot_ids, num_workers=4):
    """
    Fetch and process functional data for a list of UniProt IDs using multiple processes.
    """
    # Create a process pool to fetch data asynchronously
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map each UniProt ID to a future object responsible for fetching and processing its data
        future_to_uniprot = {executor.submit(fetch_data_for_uniprot_id, uid): uid for uid in uniprot_ids}

        # Dictionary to hold processed protein data
        protein_data = {}
        
        # Collect results as they are completed
        for future in as_completed(future_to_uniprot):
            data = future.result()
            if data:
                processed_data = process_protein_data(data)
                if processed_data:
                    # Each key in processed_data should be a UniProt ID and the value a dictionary of attributes
                    protein_data.update(processed_data)  # Merge the processed data into the main dictionary

    # Convert the accumulated dictionary into a pandas DataFrame
    df = pd.DataFrame.from_dict(protein_data, orient='index')
    
    # Handle multi-value fields that are lists
    #for col in df.columns:
    #    if any(isinstance(x, list) for x in df[col]):
    #        df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    #for col in df.columns:
    #    if col in df.columns:
    #        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            
    return df

def get_unique_uniprot_ids(mapping):
    # Extract all UniProt IDs from the mapping and flatten the list
    all_uniprot_ids = set()
    for uniprot_list in mapping.values():
        all_uniprot_ids.update(uniprot_list)
    return list(all_uniprot_ids)

def send_chunked_request(pdb_ids, base_url, headers, chunk_size=10000):
    print(f'Fetching Uniprot IDs for {len(pdb_ids)} PDB files', end='\r', flush=True)
    uniprot_mapping = {}
    query_template = '''{
        entries(entry_ids: [%s]) {
            rcsb_id
            polymer_entities {
                uniprots {
                    rcsb_id
                }
            }
        }
    }'''

    retry_delay = 1 # Start with a 1 second delay
    max_retries = 5 # Maximum number of retries before giving up on a chunk

    for i in range(0, len(pdb_ids), chunk_size):
        retries = 0
        while retries < max_retries:
            chunk = pdb_ids[i:i + chunk_size]
            query_ids = ', '.join(f'"{pdb_id.upper()}"' for pdb_id in chunk)
            query = query_template % query_ids

            response = requests.post(base_url, json={'query': query}, headers=headers)

            if response.status_code == 200:
                response_json = response.json()
                entries = response_json.get('data', {}).get('entries', [])
                for entry in entries:
                    pdb_id = entry.get('rcsb_id')
                    uniprots = [u['rcsb_id'] for entity in entry.get('polymer_entities', []) if entity.get('uniprots') for u in entity['uniprots'] if u]
                    if uniprots: 
                        uniprot_mapping[pdb_id] = uniprots
                break  # Success, break out of the retry loop
            else:
                if response.status_code == 504:
                    retries += 1
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential back-off
                    print(f"Retrying chunk ({retries}/{max_retries}). Delay: {retry_delay} seconds")
                else:
                    print(f"Failed to retrieve data for chunk; HTTP {response.status_code}")
                    print("Response content:", response.text)
                    break  # For non-504 errors, break out of the retry loop

    return uniprot_mapping

def pdb_to_uniprot(pdb_ids):
    base_url = "https://data.rcsb.org/graphql"
    headers = {'Content-Type': 'application/json'}
    # Call the chunked request function
    return send_chunked_request(pdb_ids, base_url, headers)

def functionally_annotate_foldseek_hits(csv_file_path, num_workers=25, limit=None, threshold=None):
    
    foldseek_df = pd.read_csv(csv_file_path)
    
    if not threshold is None:
        foldseek_df = foldseek_df[foldseek_df['evalue'] < threshold] #0.0001

    # Extract PDB IDs and convert them to uppercase
    foldseek_df['target_pdbID'] = foldseek_df['target'].str.split('-').str[0].str.upper()
    
    # Get unique PDB IDs from the DataFrame
    unique_pdb_ids = foldseek_df['target_pdbID'].unique().tolist()
    
    if not limit is None:
        #unique_pdb_ids = unique_pdb_ids[:limit]
        unique_pdb_ids = random.sample(unique_pdb_ids, limit)
        
    print(f'Found {len(unique_pdb_ids)} unique target proteins')

    # Fetch UniProt mapping for the unique PDB IDs
    unique_pdb_mapping = pdb_to_uniprot(unique_pdb_ids)

    # Map the target PDB IDs to UniProt IDs using the unique_pdb_mapping
    foldseek_df['target_uniprotID'] = foldseek_df['target_pdbID'].apply(
        lambda pdb_id: unique_pdb_mapping.get(pdb_id, [pd.NA])[0])
    
    unique_pdb_ids = get_unique_uniprot_ids(unique_pdb_mapping)
    target_metadata_df = fetch_and_aggregate_functional_data(unique_pdb_ids, num_workers=20)
    
    #display(target_metadata_df)
    #display(foldseek_df)
    
    merged_df = pd.merge(foldseek_df, target_metadata_df, left_on='target_uniprotID', right_on='UniProt ID')
    
    return merged_df

def perform_enrichment_analysis_v1(df, feature_columns):
    results = []
    # Isolating each query
    for query, group in df.groupby('query'):
        total = df.shape[0]
        group_total = group.shape[0]
        
        # Checking each feature category
        for feature in feature_columns:
            all_features = set(df[feature].explode().dropna().unique())
            query_features = set(group[feature].explode().dropna().unique())
            
            for specific_feature in all_features:
                observed_present = group[feature].apply(lambda x: specific_feature in x).sum()
                observed_absent = group_total - observed_present
                expected_present = df[feature].apply(lambda x: specific_feature in x).sum()
                expected_absent = total - expected_present
                
                # Constructing the contingency table
                contingency_table = [[observed_present, observed_absent],
                                     [expected_present, expected_absent]]
                
                # Performing Fisher's Exact Test
                odds_ratio, p_value = fisher_exact(contingency_table, 'greater')
                
                results.append({
                    'query': query,
                    'feature': specific_feature,
                    'p_value': p_value,
                    'category': feature
                })
    
    # Converting results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Adjusting p-values for multiple testing
    correction_method = 'fdr_bh'  # Benjamini/Hochberg (non-negative)
    p_adjust = multipletests(results_df['p_value'], method=correction_method)
    results_df['adjusted_p_value'] = p_adjust[1]
    
    return results_df

def perform_enrichment_analysis(df, num_workers=4):
    
    def _analyze_group(args):
        group, total, feature_columns, query = args
        results = []
        group_total = group.shape[0]
        for feature in feature_columns:
            try:
                all_features = set(group[feature].explode().dropna().unique())
            except TypeError:
                all_features = set(group[feature].dropna().apply(lambda x: x if isinstance(x, list) else [x]).explode().unique())

            for specific_feature in all_features:
                observed_present = group[feature].apply(lambda x: specific_feature in x if isinstance(x, list) else specific_feature == x).sum()
                observed_absent = group_total - observed_present
                expected_present = group[feature].apply(lambda x: specific_feature in x if isinstance(x, list) else specific_feature == x).sum()
                expected_absent = total - expected_present

                contingency_table = [[observed_present, observed_absent], [expected_present, expected_absent]]
                odds_ratio, p_value = fisher_exact(contingency_table, 'greater')

                results.append({
                    'query': query,
                    'feature': specific_feature,
                    'p_value': p_value,
                    'category': feature
                })
        return results

    exclude_columns = [
        'query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 
        'tstart', 'tend', 'evalue', 'bits', 'target_pdbID', 'target_uniprotID', 'UniProt ID',
        'Entry Name', 'Organism', 'Taxonomy ID', 'Sequence Length', 'Sequence', 'EC Number', 'Function',
        'Recommended Name', 'Alternative Names'
    ]
    feature_columns = df.columns.difference(exclude_columns)
    total = df.shape[0]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_group = {executor.submit(_analyze_group, (group, total, feature_columns, query)): query for query, group in df.groupby('query')}
        results = []
        for future in as_completed(future_to_group):
            results.extend(future.result())

    results_df = pd.DataFrame(results)
    correction_method = 'fdr_bh'
    p_adjust = multipletests(results_df['p_value'], method=correction_method)
    results_df['adjusted_p_value'] = p_adjust[1]
    
    return results_df

def visualize_heatmap(data, pivot_index, pivot_columns, values):
    # Pivoting the data for heatmap
    heatmap_data = data.pivot_table(index=pivot_index, columns=pivot_columns, values=values, aggfunc='first')
    
    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap on the specified axes
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2g", linewidths=.5, ax=ax)
    
    ax.set_title('Heatmap of Enriched Features Across Queries')
    ax.set_ylabel('Query')
    ax.set_xlabel('Feature')

    # Return the figure object
    return fig

def visualize_bar_chart(data):
    # Counting occurrences of significant features
    feature_counts = data['feature'].value_counts().reset_index()
    feature_counts.columns = ['feature', 'counts']
    
    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the bar plot on the specified axes
    bar_plot = sns.barplot(x='counts', y='feature', data=feature_counts.head(20), ax=ax)
    
    # Optional: set color palette manually if needed
    #bar_plot.set_palette(sns.color_palette("viridis", n_colors=20))
    
    ax.set_title('Top Enriched Features Across All Queries')
    ax.set_xlabel('Counts of Significant Enrichment')
    ax.set_ylabel('Features')
    
    # Properly setting the x-ticks and rotating them
    ax.set_xticks(ax.get_xticks())  # This ensures the ticks are explicitly set
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Return the figure object
    return fig

def visualize_dot_plot(data):
    # Adjusting data for visualization
    data['-log10(p_value)'] = -np.log10(data['adjusted_p_value'])

    # Create a figure object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the plot on the specified axes
    sns.scatterplot(data=data, x='feature', y='query', size='-log10(p_value)', 
                    legend=None, sizes=(20, 200), hue='category', ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Dot Plot of Feature Enrichment Across Queries')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Query')
    ax.grid(True)

    # Return the figure object
    return fig

def compare_features(enrichment_results, verbose=False):
    
    # Check feature matches
    def check_feature_match(row):
        category = row['category']
        feature = row['feature']
        # Check if the category column exists in the DataFrame
        if category in protein_data_df.columns:
            # Flatten the list if it's not already a scalar
            values = row[category]
            if verbose:
                print(f'category:{category}, feature:{feature}, values:{values}')
            if isinstance(values, list) or isinstance(values, np.ndarray):
                if any(pd.isna(values)):
                    return np.nan
                else:
                    # Check if the feature is within the list of values
                    return 1 if feature in values else 0
            else:
                # Direct comparison if it's scalar
                if pd.isna(values):
                    return np.nan
                return 1 if feature == values else 0
        else:
            print(f'Could not find {category} in columns')
        return np.nan

    # Assuming the format 'something-UniProtID' in the 'query' column
    enrichment_results['UniProt ID'] = enrichment_results['query'].str.split('-').str[1]

    # Get unique UniProt IDs
    uniprot_ids = enrichment_results['UniProt ID'].unique().tolist()

    # Fetch data for these UniProt IDs
    protein_data_df = fetch_and_aggregate_functional_data(uniprot_ids)

    # Assuming the fetched protein_data_df is indexed by 'UniProt ID', merge it
    comparison_df = pd.merge(enrichment_results, protein_data_df, on='UniProt ID', how='left')

    # Filter significant features
    significant_features = comparison_df[comparison_df['adjusted_p_value'] < 0.05]

    # Apply the checking function to each row
    significant_features['comparison'] = significant_features.apply(check_feature_match, axis=1)

    return significant_features

def calculate_feature_metrics(comparison_df):
    # Drop rows where comparison is NaN
    filtered_df = comparison_df.dropna(subset=['comparison'])

    # Convert 'comparison' to integer for metrics calculation
    filtered_df['comparison'] = filtered_df['comparison'].astype(int)

    # Initialize dictionary to store metrics by category and feature
    metrics = []

    # Group by category and feature for detailed metrics
    grouped = filtered_df.groupby(['category', 'feature'])
    for (category, feature), group in grouped:
        # True labels are 'comparison', predictions assume 1 if 'comparison' > 0 (already true for 1 and 0)
        true_labels = group['comparison']
        pred_labels = (group['comparison'] > 0).astype(int)  # Prediction: 1 if comparison > 0, else 0

        # Calculating precision, recall, F1-score, and accuracy
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)

        # Append results to metrics list
        metrics.append({
            'category': category,
            'feature': feature,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        })

    # Convert list of metrics to DataFrame
    metrics_df = pd.DataFrame(metrics)

    return metrics_df
    

# Set up directories
structure_fldr_path = "/home/carruthers/Downloads/ME49_proteome/cif"
base_dir='/home/carruthers/foldseek/me49'

align_to_database(structure_fldr_path, base_dir, cores=25)

foldseek_csv_path = f'{base_dir}/results/pdb/aln_tmscore.csv'

results = functionally_annotate_foldseek_hits(foldseek_csv_path, limit=1000)
#display(results)

feature_columns = ['GO Biological Process', 'GO Cellular Component', 'GO Molecular Function']
enrichment_results = perform_enrichment_analysis(results, feature_columns)
filtered_results = enrichment_results[enrichment_results['adjusted_p_value'] < 0.05]
#display(filtered_results)

fldr = os.path.dirname(foldseek_csv_path)

heatmap_path = os.path.join(fldr, 'heatmap.pdf')
bar_path = os.path.join(fldr, 'bar.pdf')
dot_path = os.path.join(fldr, 'dot.pdf')

heatmap_fig = visualize_heatmap(filtered_results, 'query', 'feature', 'adjusted_p_value')
bar_fig = visualize_bar_chart(filtered_results)
dot_fig = visualize_dot_plot(filtered_results)

heatmap_fig.savefig(heatmap_path, bbox_inches='tight')
bar_fig.savefig(bar_path, bbox_inches='tight')
dot_fig.savefig(dot_path, bbox_inches='tight')

comparison_results = compare_features(filtered_results)
#display(comparison_results)
feature_metrics_results = calculate_feature_metrics(comparison_results)
#display(feature_metrics_results)

fldr = os.path.dirname(foldseek_csv_path)

merged_path = os.path.join(fldr, 'merged.csv')
enrichment_path = os.path.join(fldr, 'enrichment.csv')
comparison_path = os.path.join(fldr, 'comparison.csv')

results.to_csv(merged_path, index=False)
filtered_results.to_csv(enrichment_path, index=False)
comparison_results.to_csv(comparison_path, index=False)

print(f'saved to results to {merged_path}')
print(f'saved to enrichment results to {enrichment_path}')
print(f'saved to comparison results to {comparison_path}')

#display(functional_data_df)