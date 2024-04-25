import os, shutil, subprocess, tarfile, glob, requests
import pandas as pd

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

def fetch_and_aggregate_functional_data(uniprot_ids):
    base_url = "https://www.ebi.ac.uk/proteins/api/proteins"
    headers = {"Accept": "application/json"}
    protein_data = {}

    for uniprot_id in uniprot_ids:
        request_url = f"{base_url}/{uniprot_id}"
        response = requests.get(request_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve data for {uniprot_id}: {response.status_code}")
            continue

        data = response.json()

        # Initialize dictionary entry for each UniProt ID if not already present
        if uniprot_id not in protein_data:
            protein_data[uniprot_id] = {
                'Type': [],
                'ID': [],
                'Description': [],
                'Sequence': '',  # Initialize sequence as an empty string
                'PDB': [],       # Initialize PDB as an empty list
                'Subcellular Location': [],
                'Propeptide': [],
                'Transmembrane': [],
                'Signal Peptide': [],
                'Family and Domains': []
            }

        # Collect Subcellular Location information
        subcellular_locations = data.get('comments', [])
        subcell_locations_info = [
            comment.get('text', [{}])[0].get('value')
            for comment in subcellular_locations if comment['type'] == 'SUBCELLULAR_LOCATION'
        ]
        protein_data[uniprot_id]['Subcellular Location'] = subcell_locations_info

        # Collect Propeptide, Transmembrane, and Signal Peptide information
        features = data.get('features', [])
        propeptides = [
            {'Description': feature['description'], 'Start': feature['begin'], 'End': feature['end']}
            for feature in features if feature['type'] == 'propeptide'
        ]
        transmembranes = [
            {'Description': feature['description'], 'Start': feature['begin'], 'End': feature['end']}
            for feature in features if feature['type'] == 'transmembrane region'
        ]
        signal_peptides = [
            {'Description': feature['description'], 'Start': feature['begin'], 'End': feature['end']}
            for feature in features if feature['type'] == 'signal peptide'
        ]
        protein_data[uniprot_id]['Propeptide'] = propeptides
        protein_data[uniprot_id]['Transmembrane'] = transmembranes
        protein_data[uniprot_id]['Signal Peptide'] = signal_peptides

        # Collect Family and Domains information
        family_domains = data.get('comments', [])
        family_domains_info = [
            {'Type': comment['type'], 'Value': comment.get('text', [{}])[0].get('value')}
            for comment in family_domains if comment['type'] in ['DOMAIN', 'SIMILARITY']
        ]
        protein_data[uniprot_id]['Family and Domains'] = family_domains_info

        # Collect protein sequence
        sequence = data.get('sequence', {}).get('sequence', '')
        protein_data[uniprot_id]['Sequence'] = sequence

        # Collect PDB information
        pdb_entries = [entry for entry in data.get('dbReferences', []) if entry['type'] == 'PDB']
        pdb_names_ids = [{'PDB Name': entry.get('id'), 'PDB ID': entry.get('id')} for entry in pdb_entries]
        protein_data[uniprot_id]['PDB'] = pdb_names_ids

        # Collect GO annotations
        go_annotations = [entry for entry in data.get('dbReferences', []) if entry['type'] == 'GO']
        for term in go_annotations:
            properties = term.get('properties', {})
            protein_data[uniprot_id]['Type'].append('GO')
            protein_data[uniprot_id]['ID'].append(term.get('id'))
            protein_data[uniprot_id]['Description'].append(properties.get('term'))
        
        # Collect function annotations
        function_annotations = data.get('comments', [])
        for comment in function_annotations:
            if comment['type'] == 'FUNCTION':
                protein_data[uniprot_id]['Type'].append('Function')
                protein_data[uniprot_id]['ID'].append(None)  # Functions typically don't have a unique ID like GO terms
                protein_data[uniprot_id]['Description'].append(comment.get('text', [{}])[0].get('value'))

    # Convert the dictionary to a DataFrame
    rows = []
    for uni_id, attributes in protein_data.items():
        row = {
            'UniProt ID': uni_id,
            'Types': attributes['Type'],
            'IDs': attributes['ID'],
            'Descriptions': attributes['Description'],
            'Sequence': attributes['Sequence'],
            'PDB Info': attributes['PDB'],
            'Subcellular Location': attributes['Subcellular Location'],
            'Propeptide': attributes['Propeptide'],
            'Transmembrane': attributes['Transmembrane'],
            'Signal Peptide': attributes['Signal Peptide'],
            'Family and Domains': attributes['Family and Domains']
        }
        rows.append(row)

    return pd.DataFrame(rows)

def merge_metadata(csv):
    foldseek_df = pd.read_csv(csv)
    foldseek_df['query_uniprotID'] = foldseek_df['query'].str.split('-').str[1]
    foldseek_df['target_pdbID'] = foldseek_df['target'].str.split('-').str[0]
    unique_uniprot_ids = foldseek_df['query'].str.split('-').str[1].unique().tolist()
    unique_pdb_ids = foldseek_df['target'].str.split('-').str[0].unique().tolist()
    return [foldseek_df, unique_uniprot_ids, unique_pdb_ids]

# Set up directories
structure_fldr_path = "/home/carruthers/Downloads/ME49_proteome/cif"
base_dir='/home/carruthers/foldseek/me49'

#align_to_database(structure_fldr_path, base_dir, cores=25)

foldseek_csv_path = f'{base_dir}/results/pdb/aln_tmscore.csv'
results = merge_metadata(foldseek_csv_path)

uniprot_ids = results[1]
#uniprot_ids = uniprot_ids[:100]
print(uniprot_ids)
functional_data_df = fetch_and_aggregate_functional_data(uniprot_ids)

merged_df = pd.merge(results[0], functional_data_df, left_on='query_uniprotID', right_on='UniProt ID')

fldr = os.path.dirname(foldseek_csv_path)
merged_path = os.path.join(fldr, 'merged.csv')
functional_data_df.to_csv(merged_path, index=False)
print(f'saved to {merged_path}')

#display(functional_data_df)