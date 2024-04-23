import os
import subprocess
import tarfile
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

# Set up directories
cores = 30
base_dir = 'path'
databases_dir = f'{base_dir}/foldseek_databases'
structures_dir = f'{base_dir}/structures'
results_dir = f'{base_dir}/results'
tmp_dir = f'{base_dir}/tmp'  # Temporary directory for Foldseek operations
os.makedirs(databases_dir, exist_ok=True)
os.makedirs(structures_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)  # Ensure the temporary directory exists

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

# Create structure database
structure_tar_path = 'path'
if os.path.exists(structures_dir) and not os.listdir(structures_dir):
    if not os.path.exists(structure_tar_path):
        print(f"Structure tar file {structure_tar_path} not found.")
    else:
        tar = tarfile.open(structure_tar_path)
        tar.extractall(path=structures_dir)
        tar.close()
        if not run_command(f"foldseek createdb {structures_dir} {structures_dir}/structures_db"):
            raise Exception("Failed to create structures database.")

# Running Foldseek
print("Running Foldseek...")
if not run_command(f"foldseek search {structures_dir}/structures_db {pdb_db_path} {results_dir}/results {tmp_dir} --threads {cores}"):
    raise Exception("Foldseek search against PDB failed.")
if not run_command(f"foldseek convertalis {structures_dir}/structures_db {pdb_db_path} {results_dir}/results.m8 {results_dir}/results_readable.m8"):
    raise Exception("Failed to convert PDB results.")

if not run_command(f"foldseek search {structures_dir}/structures_db {afdb_db_path} {results_dir}/results_afdb {tmp_dir} --threads {cores}"):
    raise Exception("Foldseek search against AlphaFold/Proteome failed.")
if not run_command(f"foldseek convertalis {structures_dir}/structures_db {afdb_db_path} {results_dir}/results_afdb.m8 {results_dir}/results_readable_afdb.m8"):
    raise Exception("Failed to convert AlphaFold/Proteome results.")

# Process and save results
def process_results(results_file):
    try:
        results = pd.read_csv(results_file, sep='\t', header=None, names=['query', 'target', 'score', 'other_info'])
        top_hits = results.sort_values(by='score', ascending=False).groupby('query').head(3)
        return top_hits
    except Exception as e:
        print(f"Failed to process results from {results_file}: {str(e)}")
        return pd.DataFrame()

pdb_hits = process_results(f"{results_dir}/results_readable.m8")
afdb_hits = process_results(f"{results_dir}/results_readable_afdb.m8")
pdb_hits.to_csv(f"{results_dir}/top_hits_pdb.csv", index=False)
afdb_hits.to_csv(f"{results_dir}/top_hits_afdb.csv", index=False)

print("Top hits saved to CSV.")

