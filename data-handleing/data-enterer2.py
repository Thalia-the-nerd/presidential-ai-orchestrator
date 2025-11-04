# upgrade_to_abide2.py

import os
import requests
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuration ---
PHENOTYPIC_URL = "https://www.nitrc.org/frs/download.php/8181/ABIDEII_Composite_Phenotypic.csv"
# === THIS IS THE FIX: Updated the URL to the new, correct S3 path ===
DATA_URL_TEMPLATE = "https://s3.amazonaws.com/fcp-indi/ABIDE2/Outputs/cpac/filt_noglobal/rois_cc200/{file_id}_rois_cc200.1D"
# ===================================================================
DATASET_DIR = os.path.join("..", "server", "dataset") 

# --- Helper Functions ---
def download_file(url, local_filename, is_pheno=False):
    headers = {}
    if is_pheno:
        headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

def process_matrix_file(filepath):
    try:
        time_series = np.loadtxt(filepath)
        if time_series.ndim != 2 or time_series.shape[0] < 2: return None
        correlation_matrix = np.corrcoef(time_series.T)
        np.nan_to_num(correlation_matrix, copy=False, nan=0.0)
        return correlation_matrix
    except Exception:
        return None

# --- Main Script Logic ---
def main():
    print("--- ABIDE II Definitive Upgrade Script (Corrected URL) ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    original_x_path = os.path.join(DATASET_DIR, "X_train.npy")
    original_y_path = os.path.join(DATASET_DIR, "y_train.npy")
    if not os.path.exists(original_x_path) or not os.path.exists(original_y_path):
        print(f"Error: Could not find original dataset in '{os.path.abspath(DATASET_DIR)}'.")
        return

    abide2_csv = "ABIDEII_Composite_Phenotypic.csv"

    if not os.path.exists(abide2_csv):
        print("Downloading ABIDE II master subject list...")
        if not download_file(PHENOTYPIC_URL, abide2_csv, is_pheno=True):
            print("CRITICAL: Failed to download the ABIDE II phenotypic file.")
            return
    else:
        print("Found local ABIDE II master subject list. Using it.")
    
    print("Finding all high-quality subjects in ABIDE II...")
    try:
        df = pd.read_csv(abide2_csv, encoding='latin-1', header=0)
    except Exception as e:
        print(f"CRITICAL: Could not read the CSV file. Error: {e}")
        return
        
    df = df.dropna(subset=['DX_GROUP', 'SUB_ID'])
    df.columns = df.columns.str.strip()
    df['FILE_ID'] = df['SITE_ID'] + '_' + df['SUB_ID'].astype(str).str.zfill(7)
    
    subjects = df[['FILE_ID', 'DX_GROUP']].to_dict('records')
    print(f"Found {len(subjects)} potential subjects in the ABIDE II master list.")

    temp_dir = "temp_abide2_downloads"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    X_abide2 = []; y_abide2 = []
    
    print("Attempting to download and process all available subject files...")
    for subject in tqdm(subjects, desc="Processing ABIDE II Subjects"):
        file_id = subject['FILE_ID']
        label = 1 if subject['DX_GROUP'] == 1 else 0
        data_url = DATA_URL_TEMPLATE.format(file_id=file_id)
        local_path = os.path.join(temp_dir, f"{file_id}.1D")
        
        if download_file(data_url, local_path):
            matrix = process_matrix_file(local_path)
            if matrix is not None and matrix.shape == (200, 200):
                X_abide2.append(matrix)
                y_abide2.append(label)
            os.remove(local_path)

    print(f"\nSuccessfully processed and added {len(X_abide2)} new subjects from ABIDE II.")

    if not X_abide2:
        print("Warning: No new subjects were downloaded. Please check the server's internet connection.")
        shutil.rmtree(temp_dir)
        return

    backup_dir = os.path.join(DATASET_DIR, "backup_abide1")
    if not os.path.exists(backup_dir): os.makedirs(backup_dir)
    print(f"Backing up original dataset...")
    shutil.copy(original_x_path, backup_dir); shutil.copy(original_y_path, backup_dir)
    print("Backup complete.")

    print("Loading original data and merging...")
    X1 = np.load(original_x_path); y1 = np.load(original_y_path)
    X2 = np.array(X_abide2, dtype=np.float32); y2 = np.array(y_abide2, dtype=np.int64)
    X_combined = np.concatenate((X1, X2), axis=0); y_combined = np.concatenate((y1, y2), axis=0)
    print(f"Combined dataset shape: X={X_combined.shape}")

    print(f"Saving new combined dataset...")
    np.save(original_x_path, X_combined); np.save(original_y_path, y_combined)
    shutil.rmtree(temp_dir)

    print("\n--- ✅ SUCCESS! ---")
    print("Your dataset has been successfully upgraded with all available ABIDE II C-PAC data.")

if __name__ == "__main__":
    main()
