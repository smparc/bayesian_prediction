import pandas as pd
import pickle
from collections import defaultdict

def build_cpts(csv_path, network_structure):
    """
    Builds the Conditional Probability Tables (CPTs) from the SCDB CSV file.

    This function implements the counting logic from the paper:
    P(X=x) = Count(X=x) / Total cases
    P(X=x|Parents=p) = Count(X=x, Parents=p) / Count(Parents=p)

    Args:
        csv_path (str): Path to the SCDB_..._Citation.csv file.
        network_structure (list): A list of (var_name, [parent_names]) tuples,
                                  in topological order.

    Returns:
        dict: The CPTs in the nested dictionary format.
    """
    
    print(f"Loading SCDB data from {csv_path}...")
    # Load the dataset.
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
        
    print(f"Loaded {len(df)} cases.")

    print("Building CPTs...")
    cpts = {}
    total_cases = len(df)

    # Handle NaN values. A simple strategy is to fill with a placeholder.
    for var_name, _ in network_structure:
        if var_name not in df.columns:
            raise KeyError(f"Column '{var_name}' from network structure not found in CSV.")
        
        if pd.api.types.is_numeric_dtype(df[var_name]):
            df[var_name] = df[var_name].fillna(-1) # For numeric
        else:
            df[var_name] = df[var_name].fillna('MISSING') # For categorical
            
    for var_name, parents in network_structure:
        cpts[var_name] = {}
        
        if not parents:
            # --- Root Node --- [cite: 35-36]
            counts = df[var_name].value_counts()
            for value, count in counts.items():
                cpts[var_name][value] = count / total_cases
        
        else:
            # --- Child Node --- [cite: 37-38]
            parent_counts = df.groupby(parents).size()
            child_parent_counts = df.groupby(parents + [var_name]).size()
            
            for (parent_vals, child_val), count in child_parent_counts.items():
                if not isinstance(parent_vals, tuple):
                    parent_vals = (parent_vals,) # Ensure it's a tuple
                
                total_parent_count = parent_counts.get(parent_vals, 0)
                
                # CPT key format: (parent1_val, ..., child_val)
                cpt_key = parent_vals + (child_val,)
                
                if total_parent_count > 0:
                    cpts[var_name][cpt_key] = count / total_parent_count
                else:
                    cpts[var_name][cpt_key] = 0.0

    print("CPT construction complete.")
    return cpts

if __name__ == "__main__":
    
    # --- ACTION REQUIRED ---
    # This is the network structure from your diagram [cite: 18-29]
    # You MUST inspect 'SCDB_2025_01_justiceCentered_Citation.csv'
    # and replace these placeholder names with the *actual column names*.
    #
    # Example: 'inf_justi' might be 'chief' in the CSV.
    # 'dispos' might be 'disposition' in the CSV.
    #
    NETWORK_STRUCTURE = [
        ('inf_justi', []), # Placeholder
        ('sue_are', []),    # Placeholder
        ('sw_type', []),    # Placeholder
        ('supplan', []),    # Placeholder
        ('ourt_dis', []),   # Placeholder
        ('position', []),   # Placeholder
        ('mas', []),        # Placeholder
        ('UNCONS', []),     # Placeholder
        ('vahn_k', ['sue_are', 'position']), # Placeholder
        ('tent_ve', ['sw_type', 'inf_justi', 'supplan']), # Placeholder
        ('bill_vet', ['ourt_dis', 'mas', 'UNCONS']), # Placeholder
        ('dispos', ['vahn_k', 'tent_ve', 'bill_vet']) # Placeholder
    ]
    
    CSV_FILE_PATH = 'SCDB_2025_01_justiceCentered_Citation.csv'
    CPT_OUTPUT_FILE = 'scdb_cpts.pkl'

    try:
        final_cpts = build_cpts(CSV_FILE_PATH, NETWORK_STRUCTURE)
        
        with open(CPT_OUTPUT_FILE, 'wb') as f:
            pickle.dump((final_cpts, NETWORK_STRUCTURE), f)
            
        print(f"Successfully built and saved CPTs to {CPT_OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Error: Could not find data file '{CSV_FILE_PATH}'")
    except KeyError as e:
        print(f"Error: A column name in your NETWORK_STRUCTURE was not found in the CSV.")
        print(f"Missing column: {e}")
        print("Please edit the 'NETWORK_STRUCTURE' list in 'build_cpts.py'.")
