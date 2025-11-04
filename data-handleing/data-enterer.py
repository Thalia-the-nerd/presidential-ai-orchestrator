import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm # For a nice progress bar!

# --- Configuration ---
# URL for the main CSV file with all participant info and labels
PHENOTYPIC_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed.csv"

# URL template for the data files we want
# We'll use:
# pipeline = cpac
# strategy = filt_noglobal (a good standard choice)
# derivative = rois_cc200 (200x200 connectome matrix)
DATA_URL_TEMPLATE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/{file_id}_rois_cc200.1D"

# Output directory (relative to this script)
OUTPUT_DIR = "../server/api/dataset"
X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "X_train.npy")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "y_train.npy")

# --- Main Script ---
def download_file(url, local_filename):
    """Downloads a file from a URL with retry logic."""
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to download {url}. Error: {e}")
        return False

# --- MODIFIED: process_matrix_file ---
def process_matrix_file(filepath):
    """
    Loads a .1D time series file, computes the correlation matrix, and returns it.
    """
    try:
        # Load the time series data (timepoints, regions)
        time_series = np.loadtxt(filepath)
        
        # We need at least 2 timepoints to calculate correlation
        if time_series.ndim != 2 or time_series.shape[0] < 2:
            print(f"Warning: Skipping {filepath}, not a valid 2D time series. Shape: {time_series.shape}")
            return None
            
        # --- THIS IS THE FIX ---
        # Calculate the Pearson correlation matrix (regions, regions)
        # np.corrcoef expects (features, observations), so we transpose the (timepoints, regions) matrix
        # to (regions, timepoints)
        correlation_matrix = np.corrcoef(time_series.T)
        
        # The result is a (regions, regions) square matrix, e.g., (200, 200)
        
        # Handle NaN values:
        # NaNs can occur if a region's time series is constant (zero variance).
        # We'll just set these NaNs to 0 (no correlation).
        if np.isnan(correlation_matrix).any():
            np.nan_to_num(correlation_matrix, copy=False, nan=0.0)
            
        return correlation_matrix
        
    except Exception as e:
        print(f"Warning: Error processing {filepath}: {e}")
        return None

def main():
    print("--- ABIDE Data Preparation Script ---")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
        
    # --- 1. Download Phenotypic CSV file ---
    local_csv_file = "phenotypic.csv"
    print(f"Downloading main phenotypic file: {PHENOTYPIC_URL}...")
    if not download_file(PHENOTYPIC_URL, local_csv_file):
        print("CRITICAL: Failed to download phenotypic file. Exiting.")
        return
    print("Download complete.")

    # --- 2. Load CSV and find subjects ---
    try:
        df = pd.read_csv(local_csv_file)
    except Exception as e:
        print(f"CRITICAL: Failed to read {local_csv_file}: {e}")
        return

    # Filter out any subjects that failed quality control (QC)
    # 'qc_rater_1' is a good column to check. 'OK' means pass.
    # We also need to drop subjects with no DX_GROUP
    df = df.dropna(subset=['DX_GROUP', 'qc_rater_1'])
    df = df[df['qc_rater_1'] == 'OK']

    # Get the file IDs and labels
    # DX_GROUP: 1 = Autism, 2 = Control
    # We will convert this to 1 = Autism, 0 = Control
    subjects = df[['FILE_ID', 'DX_GROUP']].to_dict('records')
    print(f"Found {len(subjects)} high-quality subjects to download.")

    # --- 3. Download and process all data files ---
    X_data = [] # List to hold all the 2D matrix arrays
    y_data = [] # List to hold all the labels (0 or 1)
    
    temp_dir = "temp_data_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print(f"Downloading and processing {len(subjects)} matrix files...")
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        file_id = subject['FILE_ID']
        
        # 1 = Autism, 2 = Control. We map 2 -> 0.
        label = 1 if subject['DX_GROUP'] == 1 else 0
        
        # Construct URL and local file path
        data_url = DATA_URL_TEMPLATE.format(file_id=file_id)
        local_data_file = os.path.join(temp_dir, f"{file_id}.1D")
        
        # Download the file
        if download_file(data_url, local_data_file):
            # Process the file
            matrix = process_matrix_file(local_data_file)
            
            # If processing was successful
            if matrix is not None:
                # Check that our new matrix is the correct shape
                if matrix.shape == (200, 200):
                    X_data.append(matrix)
                    y_data.append(label)
                else:
                    # This might happen if a different atlas was used
                    print(f"Warning: Skipping {file_id}, processed matrix is wrong shape: {matrix.shape}. Expected (200, 200)")
            
            # Clean up the temp file
            os.remove(local_data_file)

    print("All files processed.")
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        # Clean up any failed downloads that might be left
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    if os.path.exists(local_csv_file):
        os.remove(local_csv_file)

    # --- 4. Convert to NumPy and Save ---
    if not X_data:
        print("CRITICAL: No data was successfully processed. Exiting.")
        return

    print("Converting data to NumPy arrays...")
    X_train = np.array(X_data, dtype=np.float32)
    y_train = np.array(y_data, dtype=np.int64)

    print(f"Final data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    print(f"Saving files to {OUTPUT_DIR}...")
    np.save(X_TRAIN_FILE, X_train)
    np.save(Y_TRAIN_FILE, y_train)

    print("--- 🥳 Data preparation complete! ---")
    print(f"Your files '{os.path.basename(X_TRAIN_FILE)}' and '{os.path.basename(Y_TRAIN_FILE)}' are ready in:")
    print(f"{os.path.abspath(OUTPUT_DIR)}")
    print("You can now start your orchestrator server.")

if __name__ == "__main__":
    main()


