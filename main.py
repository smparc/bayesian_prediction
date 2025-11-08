import pickle
from bayes_net import rejection_sampling

def run_prediction():
    """
    Loads pre-built CPTs and runs rejection sampling.
    """
    
    CPT_FILE = 'scdb_cpts.pkl'

    # --- Load CPTs ---
    try:
        with open(CPT_FILE, 'rb') as f:
            cpts, network_structure = pickle.load(f)
        print(f"Successfully loaded CPTs and network structure from {CPT_FILE}")
    except FileNotFoundError:
        print(f"Error: CPT file '{CPT_FILE}' not found.")
        print("Please run 'python build_cpts.py' first to generate it.")
        return
    except Exception as e:
        print(f"An error occurred loading the CPT file: {e}")
        return

    # --- !!! ACTION REQUIRED !!! ---
    # 1. Define your observations
    #    You must replace these with *actual values* from your CSV.
    observations = {
        'inf_justi': 'Roberts', # Placeholder: Replace with a real value
        'sue_are': 10           # Placeholder: Replace with a real value
    }
    
    # 2. Define your query
    #    This should be your 'dispos' variable and one of its
    #    possible outcomes (e.g., 2 for 'affirmed') [cite: 88, 93]
    target_variable = 'dispos'
    target_value = 2
    
    # 3. Set sample size
    N_samples = 30000 
    
    print(f"Running Rejection Sampling with {N_samples} samples...")
    print(f"Query: P({target_variable}={target_value} | {observations})")
    
    probability = rejection_sampling(
        cpts,
        network_structure,
        observations,
        target_variable,
        target_value,
        N_samples
    )
    
    print(f"Estimated Probability: {probability:.4f}")

if __name__ == "__main__":
    run_prediction()
