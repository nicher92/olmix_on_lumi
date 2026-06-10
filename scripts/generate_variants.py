import os
import json
import yaml
import numpy as np

from olmix.aliases import SourceConfig, TopicConfig, QualityConfig
from olmix.generate.synthesize_mixture import generate_weights_dirichlet

def get_configs():
    """Reads local config"""
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["settings"], config.get("swarm", {}), config["datasets"]

def calculate_num_tokens(bin_path):
    """Calculates num tokens in a nemotron .bin path"""
    if not os.path.exists(bin_path):
        print(f"⚠️ Warning: File not found - {bin_path}")
        return 0
    header_offset = 74 # nemotron header
    bytes_per_token = 4 # assume int32
    file_size = os.path.getsize(bin_path)
    return (file_size - header_offset) // bytes_per_token

def parse_yaml(datasets_config):
    """Reads source datasets, applies multipliers, and calculates tokens per .bin file"""
    sources = []
    leaf_tokens = {}
    prefix_map = {}  # Stores [(prefix, token_count)] for proportional weight distribution

    # Parse YAML, count tokens (appending .bin), and build SourceConfigs
    for source_name, source_data in datasets_config.items():
        # Get the specific multiplier for this dataset, default to 1 (Full)
        multiplier = source_data.get("multiplier", 1)

        if "quality" in source_data:
            qualities = []
            for q_name, q_prefixes in source_data["quality"].items():
                total_tokens = 0
                q_bin_paths = []
                shard_info = []
                
                # Loop through ALL paths, not just index [0]
                for prefix in q_prefixes:
                    bin_path = f"{prefix}.bin"
                    q_bin_paths.append(bin_path)
                    tokens = calculate_num_tokens(bin_path)
                    total_tokens += tokens
                    shard_info.append((prefix, tokens))

                qualities.append(QualityConfig(name=q_name, paths=q_bin_paths))

                leaf_name = f"{source_name}:{q_name}"
                prefix_map[leaf_name] = shard_info
                
                # Apply the sampling multiplier to scale up the perceived token count
                leaf_tokens[leaf_name] = total_tokens * multiplier

            sources.append(SourceConfig(name=source_name, quality=qualities))

        else:
            prefixes = source_data.get("paths", [])
            bin_paths = []
            total_tokens = 0
            shard_info = []
            
            # Loop through ALL paths, not just index [0]
            for prefix in prefixes:
                bin_path = f"{prefix}.bin"
                bin_paths.append(bin_path)
                tokens = calculate_num_tokens(bin_path)
                total_tokens += tokens
                shard_info.append((prefix, tokens))

            sources.append(SourceConfig(name=source_name, paths=bin_paths))

            prefix_map[source_name] = shard_info
            
            # Apply the sampling multiplier to scale up the perceived token count
            leaf_tokens[source_name] = total_tokens * multiplier
    
    for k,v in leaf_tokens.items():
        print(k)
        print(v)
        print()
        print("------------------------------------")

    return leaf_tokens, sources, prefix_map

def calculate_priors_and_variants(leaf_tokens):
    """Calculate priors and variant counts"""
    leaf_tokens = {k: v for k, v in leaf_tokens.items() if v > 0}
    total_tokens = sum(leaf_tokens.values())
    leaf_dist = {name: count / total_tokens for name, count in leaf_tokens.items()}
    domains = list(leaf_dist.keys())
    return domains, leaf_dist, leaf_tokens, total_tokens

def write_mixes_to_json(mixtures, domains):
    """Writes variant combinations to a JSON file"""
    lumi_variants = []
    for idx, mix in enumerate(mixtures):
        variant_config = {}
        for i, leaf_name in enumerate(domains):
            variant_config[leaf_name] = {
                "weight": round(float(mix[0][i]), 6),
                "repetition_factor": round(float(mix[1][i]), 3)
            }
        lumi_variants.append({
            "variant_id": f"nested-swarm-{idx:04d}",
            "mix": variant_config
        })

    os.makedirs("./data", exist_ok=True)
    with open("./data/lumi_nested_variants.json", "w") as f:
        json.dump(lumi_variants, f, indent=2)

    return lumi_variants

def make_megatron_text_files_and_bash_script(lumi_variants, prefix_map):
    """Generate Megatron .txt mix files and the bash launcher"""
    os.makedirs("./data/mixes", exist_ok=True)
    

    for variant in lumi_variants:
        variant_id = variant["variant_id"]
        mix_file_path = f"./data/mixes/{variant_id}.txt"

        with open(mix_file_path, "w") as f:
            for domain, config in variant["mix"].items():
                domain_weight = config["weight"]
                if domain_weight > 0:
                    shard_info = prefix_map[domain]
                    total_actual_tokens = sum(tokens for _, tokens in shard_info)
                    
                    for prefix, tokens in shard_info:
                        if total_actual_tokens > 0:
                            # Distribute the domain's weight across its shards proportionally
                            shard_weight = domain_weight * (tokens / total_actual_tokens)
                            f.write(f"{shard_weight:.6f} {prefix}\n")
        
    
    launch_script = f"sbatch --array=0-{len(lumi_variants) - 1}%5 scripts/train-0.05B.sh"
    # Create master launcher script
    launcher_script = "launch_all_swarms.sh"
    with open(launcher_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(launch_script + "\n")

    os.chmod(launcher_script, 0o755)
    return launcher_script

if __name__ == "__main__":
    settings, swarm_config, datasets_config = get_configs()
    leaf_tokens, sources, prefix_map = parse_yaml(datasets_config)
    domains, leaf_dist, leaf_tokens, total_tokens = calculate_priors_and_variants(leaf_tokens)

    NUM_LEAVES = len(domains)
    NUM_VARIANTS = 3 * (NUM_LEAVES + 1)

    print(f"📊 Config loaded. Found {NUM_LEAVES} total leaf datasets. Scaled total tokens: {total_tokens:,}")

    # Generate Mixtures
    mixtures = generate_weights_dirichlet(
        sources=sources,
        leaf_dist=leaf_dist,
        num_samples_out=NUM_VARIANTS,
        leaf_tokens=leaf_tokens,
        max_tokens=settings["max_tokens"],
        repetition_factor=settings["repetition_factor"],

        minimum_source_weight=swarm_config.get("minimum_source_weight", 0.002),
        minimum_topic_weight=swarm_config.get("minimum_topic_weight", 0.002),
        source_temperature=swarm_config.get("source_temperature", 1.0),
        topic_temperature=swarm_config.get("topic_temperature", 1.0),
        min_source_strength=swarm_config.get("min_source_strength", 0.1),
        max_source_strength=swarm_config.get("max_source_strength", 5.0),
        min_topic_strength=swarm_config.get("min_topic_strength", 0.1),
        max_topic_strength=swarm_config.get("max_topic_strength", 5.0),
        
        sample_multiplier=20,
        
        enable_bound=swarm_config.get("enable_bound", True),
        manual_prior=swarm_config.get("manual_prior", None),
        manual_topic_prior=swarm_config.get("manual_topic_prior", None),
        nonzero_weight=swarm_config.get("nonzero_weight", None),
        existing_mix_file=swarm_config.get("existing_mix_file", None),
    )

    lumi_variants = write_mixes_to_json(mixtures, domains)
    launcher_script = make_megatron_text_files_and_bash_script(lumi_variants, prefix_map)

    print(f"✅ Generated {NUM_VARIANTS} Megatron mix files in the mixes/ directory.")
    print(f"✅ Generated {launcher_script}. Run it with: ./{launcher_script}")
