# upgrade_to_abide2.py - DIAGNOSTIC SCRIPT

import os
import pandas as pd

def main_diagnostic():
    print("--- ABIDE II CSV Diagnostic Script ---")
    
    # Navigate to the script's directory to ensure it finds the CSV
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # This handles cases where the script is run in an interactive interpreter
        print("Could not change directory. Assuming CSV is in the current directory.")
        pass

    abide2_csv = "ABIDEII_Composite_Phenotypic.csv"

    if not os.path.exists(abide2_csv):
        print(f"CRITICAL: The file '{abide2_csv}' was not found in the current directory.")
        print("Please ensure the ABIDE II phenotypic CSV is present before running again.")
        return

    print(f"Found '{abide2_csv}'. Running diagnostics...")

    try:
        # --- DIAGNOSTIC STEP 1: READ THE RAW TOP OF THE FILE ---
        # We will read the first 15 rows without assuming any header.
        # This lets us visually inspect the file's structure.
        print("\n--- [RAW CONTENT] First 15 lines of the CSV file ---")
        df_raw = pd.read_csv(
            abide2_csv,
            encoding='latin-1',
            header=None,  # Treat every line as data
            nrows=15      # Only read the top of the file
        )
        print(df_raw.to_string())
        print("--- [RAW CONTENT] End of raw content ---\n")

        # --- DIAGNOSTIC STEP 2: TEST DIFFERENT HEADER ROWS ---
        # We will now loop through the first 12 rows and try to use each one as a header.
        # We will print the column names pandas finds for each attempt.
        print("--- [HEADER TEST] Trying to find the correct header row ---")
        for i in range(12):
            try:
                # Read only the header and the first row of data
                df_test = pd.read_csv(abide2_csv, encoding='latin-1', header=i, nrows=1)
                print(f"\n>>> Success with header on row index {i} (Line {i + 1}):")
                print("Detected Columns:")
                # We print the list of columns to check for extra spaces or typos
                print(list(df_test.columns))
            except Exception as e:
                print(f"\n>>> Failed with header on row index {i} (Line {i + 1}): {type(e).__name__}")
        print("--- [HEADER TEST] End of header test ---\n")

    except Exception as e:
        print(f"CRITICAL: A fatal error occurred during the diagnostic process.")
        print(f"Error details: {e}")

    print("--- DIAGNOSTICS COMPLETE ---")
    print("Please copy the entire output from this script and paste it in your next response.")

if __name__ == "__main__":
    main_diagnostic()
