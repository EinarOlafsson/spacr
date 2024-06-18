import csv
import os
import requests

def download_alphafold_structures(tsv_location, dst, version="4"):
    # Create the destination directory if it does not exist
    dst_pdb = os.path.join(dst,'pdb')
    dst_cif = os.path.join(dst,'cif')
    dst_pae = os.path.join(dst,'pae')
    
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not os.path.exists(dst_pdb):
        os.makedirs(dst_pdb)
    if not os.path.exists(dst_cif):
        os.makedirs(dst_cif)
    if not os.path.exists(dst_pae):
        os.makedirs(dst_pae)

    failed_downloads = []  # List to keep track of failed downloads

    # Open the TSV file and read entries
    with open(tsv_location, 'r') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in reader:
            entry = row['Entry']
            af_link = f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v{version}.pdb"
            cif_link = f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v{version}.cif"
            pae_link = f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-predicted_aligned_error_v{version}.json"

            try:
                response_pdb = requests.get(af_link, stream=True)
                response_cif = requests.get(cif_link, stream=True)
                response_pae = requests.get(pae_link, stream=True)
                if response_pdb.status_code == 200:
                    
                    # Save the PDB file
                    with open(os.path.join(dst_pdb, f"AF-{entry}-F1-model_v{version}.pdb"), 'wb') as pdb_file:
                        pdb_file.write(response_pdb.content)
                    print(f"Downloaded: AF-{entry}-F1-model_v{version}.pdb")
                    
                    # Save the CIF file
                    with open(os.path.join(dst_cif, f"AF-{entry}-F1-model_v{version}.cif"), 'wb') as cif_file:
                        cif_file.write(response_cif.content)
                    print(f"Downloaded: AF-{entry}-F1-model_v{version}.cif")
                    
                    # Save the PAE file
                    with open(os.path.join(dst_pae, f"AF-{entry}-F1-predicted_aligned_error_v{version}.json"), 'wb') as pdb_file:
                        pdb_file.write(response_pae.content)
                    print(f"Downloaded: AF-{entry}-F1-predicted_aligned_error_v{version}.json")
                    
                else:
                    # If the file could not be downloaded, record the entry
                    failed_downloads.append(entry)
                    print(f"Failed to download structure for: {entry}")
            except Exception as e:
                print(f"Error downloading structure for {entry}: {e}")
                failed_downloads.append(entry)

    # Save the list of failed downloads to a CSV file in the destination folder
    if failed_downloads:
        with open(os.path.join(dst, 'failed_downloads.csv'), 'w', newline='') as failed_file:
            writer = csv.writer(failed_file)
            writer.writerow(['Entry'])
            for entry in failed_downloads:
                writer.writerow([entry])
        print(f"Failed download entries saved to: {os.path.join(dst, 'failed_downloads.csv')}")

# Example usage:
#tsv_location = '/home/carruthers/Downloads/GT1_proteome/GT1_proteins_uniprot.tsv'  # Replace with the path to your TSV file containing a list of UniProt entries
#dst_folder = '/home/carruthers/Downloads/GT1_proteome'  # Replace with your destination folder
#download_alphafold_structures(tsv_location, dst_folder)
