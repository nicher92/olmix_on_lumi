import yaml
from generate_variants import get_configs, parse_yaml

def main():
    # 1. Load the configs using your existing function
    _, _, datasets_config = get_configs()
    
    # 2. Calculate the tokens using your existing function
    leaf_tokens, _, _ = parse_yaml(datasets_config)
    
    # 3. Clean up and calculate relative sizes
    leaf_tokens = {k: v for k, v in leaf_tokens.items() if v > 0}
    total_tokens = sum(leaf_tokens.values())
    
    relative_sizes = {k: float(v / total_tokens) for k, v in leaf_tokens.items()}
    
    # 4. Construct the YAML block
    priors_block = {
        "priors": {
            "relative_sizes": relative_sizes,
            "token_counts": leaf_tokens
        }
    }
    
    print("\n" + "="*50)
    print("📋 COPY AND PASTE THIS BLOCK INTO YOUR FIT CONFIG")
    print("="*50 + "\n")
    print(yaml.dump(priors_block, default_flow_style=False, sort_keys=True))

if __name__ == "__main__":
    main()
