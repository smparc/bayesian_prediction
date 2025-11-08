import random
import numpy as np

def generate_sample(cpts, network_structure):
    """
    Generates a single complete sample from the Bayesian network.
    
    Args:
        cpts (dict): The Conditional Probability Tables.
        network_structure (list): A list of (var_name, [parent_names]) tuples,
                                  in topological order.
        
    Returns:
        dict: A single sample {variable_name: value}.
    """
    sample = {}
    
    for var_name, parents in network_structure:
        var_cpt = cpts[var_name]
        
        if not parents:
            # --- Root Node ---
            if not var_cpt:
                sample[var_name] = None
                continue
            
            values = list(var_cpt.keys())
            probabilities = list(var_cpt.values())
            prob_sum = sum(probabilities)
            if prob_sum == 0:
                chosen_value = random.choice(values)
            else:
                probabilities = [p / prob_sum for p in probabilities]
                chosen_value = random.choices(values, weights=probabilities, k=1)[0]
            sample[var_name] = chosen_value
        
        else:
            # --- Child Node ---
            try:
                parent_key_part = tuple(sample[p] for p in parents)
            except KeyError:
                sample[var_name] = None
                continue
            
            child_dist = {}
            for full_key, prob in var_cpt.items():
                if full_key[:-1] == parent_key_part:
                    child_val = full_key[-1]
                    child_dist[child_val] = prob
            
            if not child_dist:
                all_child_values = {k[-1] for k in var_cpt.keys()}
                if not all_child_values:
                    chosen_value = None
                else:
                    chosen_value = random.choice(list(all_child_values))
            else:
                total_prob = sum(child_dist.values())
                if total_prob == 0:
                    chosen_value = random.choice(list(child_dist.keys()))
                else:
                    values = list(child_dist.keys())
                    probabilities = [p / total_prob for p in child_dist.values()]
                    chosen_value = random.choices(values, weights=probabilities, k=1)[0]
            
            sample[var_name] = chosen_value
            
    return sample


def rejection_sampling(cpts, network_structure, observations, target_var, target_val, N):
    """
    Performs Rejection Sampling to estimate a conditional probability.
    This is a direct implementation of Algorithm 1 from your paper .
    
    Args:
        cpts (dict): The Conditional Probability Tables for the network.
        network_structure (list): A list of (var_name, [parent_names]) tuples,
                                  in topological order.
        observations (dict): A dictionary of observed variables and their values.
        target_var (str): The name of the variable we are querying.
        target_val (any): The specific value of the target variable we are
                          interested in.
        N (int): The total number of samples to generate.
    
    Returns:
        float: The estimated probability P(target_var = target_val | observations).
    """
    
    target_event = 0
    total_accepted = 0
    
    for i in range(N):
        sample = generate_sample(cpts, network_structure) 
        
        is_consistent = True
        for var, value in observations.items():
            if sample.get(var) != value:
                is_consistent = False
                break
        
        if is_consistent:
            total_accepted += 1
            if sample.get(target_var) == target_val:
                target_event += 1
    
    if total_accepted == 0:
        return 0.0
    else:
        return target_event / total_accepted
