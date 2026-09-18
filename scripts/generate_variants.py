import os
import json
import yaml
import numpy as np
from datetime import datetime

from olmix.aliases import SourceConfig, QualityConfig
from olmix.generate.synthesize_mixture import generate_weights_dirichlet

def get_configs():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["settings"], config.get("swarm", {}), config["datasets"]

def calculate_num_tokens(bin_path):
    if not os.path.exists(bin_path):
        print(f"⚠️ Warning: File not found - {bin_path}")
        return 0
    header_offset = 74 
    bytes_per_token = 4 
    file_size = os.path.getsize(bin_path)
    return (file_size - header_offset) // bytes_per_token

def parse_yaml(datasets_config):
    sources = []
    leaf_tokens = {}
    prefix_map = {} 

    for source_name, source_data in datasets_config.items():
        multiplier = source_data.get("multiplier", 1)

        if "quality" in source_data:
            qualities = []
            for q_name, q_prefixes in source_data["quality"].items():
                total_tokens = 0
                q_bin_paths = []
                shard_info = []

                for prefix in q_prefixes:
                    bin_path = f"{prefix}.bin"
                    q_bin_paths.append(bin_path)
                    tokens = calculate_num_tokens(bin_path)
                    total_tokens += tokens
                    shard_info.append((prefix, tokens))

                qualities.append(QualityConfig(name=q_name, paths=q_bin_paths))
                leaf_name = f"{source_name}:{q_name}"
                prefix_map[leaf_name] = shard_info
                leaf_tokens[leaf_name] = total_tokens * multiplier

            sources.append(SourceConfig(name=source_name, quality=qualities))

        else:
            prefixes = source_data.get("paths", [])
            bin_paths = []
            total_tokens = 0
            shard_info = []

            for prefix in prefixes:
                bin_path = f"{prefix}.bin"
                bin_paths.append(bin_path)
                tokens = calculate_num_tokens(bin_path)
                total_tokens += tokens
                shard_info.append((prefix, tokens))

            sources.append(SourceConfig(name=source_name, paths=bin_paths))
            prefix_map[source_name] = shard_info
            leaf_tokens[source_name] = total_tokens * multiplier

    return leaf_tokens, sources, prefix_map

def calculate_priors_and_variants(leaf_tokens):
    leaf_tokens = {k: v for k, v in leaf_tokens.items() if v > 0}
    total_tokens = sum(leaf_tokens.values())
    leaf_dist = {name: count / total_tokens for name, count in leaf_tokens.items()}
    domains = list(leaf_dist.keys())
    return domains, leaf_dist, leaf_tokens, total_tokens


def write_mixes_to_json(mixtures, domains):
    lumi_variants = []

    # Create a unique prefix using the current date and time
    run_prefix = datetime.now().strftime("stage2_mix_%Y%m%d_%H%M")

    for idx, mix in enumerate(mixtures):
        variant_config = {}
        for i, leaf_name in enumerate(domains):
            variant_config[leaf_name] = {
                "weight": round(float(mix[0][0][i]), 6),
                "repetition_factor": round(float(mix[1][0][i]), 3)
            }
        lumi_variants.append({
            "variant_id": f"{run_prefix}-{idx:04d}", # <--- Injects the unique name here!
            "mix": variant_config
        })

    os.makedirs("./data", exist_ok=True)
    with open("./data/lumi_nested_variants.json", "w") as f:
        json.dump(lumi_variants, f, indent=2)

    return lumi_variants


def make_megatron_text_files_and_bash_script(lumi_variants, prefix_map):
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
                            shard_weight = domain_weight * (tokens / total_actual_tokens)
                            f.write(f"{shard_weight:.6f} {prefix}\n")

    launch_script = f"sbatch --array=0-{len(lumi_variants) - 1} scripts/train-0.05B.sh"
    launcher_script = "launch_all_swarms.sh"
    with open(launcher_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(launch_script + "\n")

    os.chmod(launcher_script, 0o755)
    return launcher_script

if __name__ == "__main__":
    settings, swarm_config, datasets_config = get_configs()
    leaf_tokens, sources, prefix_map = parse_yaml(datasets_config)
    domains, leaf_dist, _, total_tokens = calculate_priors_and_variants(leaf_tokens)

    existing_mix_path = swarm_config.get("existing_mix_file")
    
    # Defaults for standard run
    target_sources = sources
    target_leaf_dist = leaf_dist
    target_leaf_tokens = leaf_tokens
    effective_leaves = len(domains)
    frozen_domains = []
    frozen_ratios = {}
    collapsed_domains = domains

    if existing_mix_path and os.path.exists(existing_mix_path):
        with open(existing_mix_path, "r") as f:
            old_mix = json.load(f)

        frozen_domains = [d for d in domains if d in old_mix]
        new_domains = [d for d in domains if d not in old_mix]
        
        if frozen_domains:
            # 1. Calculate the locked internal ratios of the frozen block
            old_mix_sum = sum(old_mix[d] for d in frozen_domains)
            frozen_ratios = {d: old_mix[d] / old_mix_sum for d in frozen_domains}

            # 2. Package the Virtual Domain tokens and priors
            target_leaf_tokens = {"VIRTUAL_DOMAIN": sum(leaf_tokens[d] for d in frozen_domains)}
            for d in new_domains:
                target_leaf_tokens[d] = leaf_tokens[d]

            collapsed_domains, target_leaf_dist, _, _ = calculate_priors_and_variants(target_leaf_tokens)

            # 3. Package the SourceConfigs for Olmix
            target_sources = [SourceConfig(name="VIRTUAL_DOMAIN", paths=["virtual"])]
            for src in sources:
                is_new = False
                if hasattr(src, 'quality') and src.quality:
                    if any(f"{src.name}:{q.name}" in new_domains for q in src.quality):
                        is_new = True
                elif src.name in new_domains:
                    is_new = True
                    
                if is_new:
                    target_sources.append(src)

            effective_leaves = len(new_domains) + 1
            print(f"♻️  Mixture Reuse: Packaged {len(frozen_domains)} frozen datasets into 1 VIRTUAL_DOMAIN.")
            print(f"♻️  Optimizing over {effective_leaves} actual dimensions.")

    NUM_VARIANTS = 3 * (effective_leaves + 1)
    print(f"📊 Config loaded. Generating {NUM_VARIANTS} variants. Scaled total tokens: {total_tokens:,}")

    # Generate Mixtures (We pass existing_mix_file=None to bypass the buggy backend features)
    raw_mixtures = generate_weights_dirichlet(
        sources=target_sources,
        leaf_dist=target_leaf_dist,
        num_samples_out=NUM_VARIANTS,
        leaf_tokens=target_leaf_tokens,
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
        existing_mix_file=None,
        manual_prior=swarm_config.get("manual_prior", None),
        manual_topic_prior=swarm_config.get("manual_topic_prior", None)
    )

    # UNPACK the Virtual Domain back into the 27 original domains
    unpacked_mixtures = []
    for mix in raw_mixtures:
        # Flatten arrays to guarantee they are 1D, avoiding any shape mismatches
        flat_weights = np.array(mix[0]).flatten()
        flat_reps = np.array(mix[1]).flatten()
        
        new_weights = []
        new_reps = []
        
        for d in domains:
            if d in frozen_domains:
                v_idx = collapsed_domains.index("VIRTUAL_DOMAIN")
                w = float(flat_weights[v_idx]) * frozen_ratios[d]
                new_weights.append(w)
                new_reps.append(float(flat_reps[v_idx]))
            else:
                n_idx = collapsed_domains.index(d)
                new_weights.append(float(flat_weights[n_idx]))
                new_reps.append(float(flat_reps[n_idx]))
                
        # Package back into the nested shape write_mixes_to_json expects
        unpacked_mixtures.append((np.array([new_weights]), np.array([new_reps])))

    lumi_variants = write_mixes_to_json(unpacked_mixtures, domains)
    launcher_script = make_megatron_text_files_and_bash_script(lumi_variants, prefix_map)

    print(f"✅ Generated {NUM_VARIANTS} Megatron mix files in the mixes/ directory.")
    print(f"✅ Generated {launcher_script}. Run it with: ./{launcher_script}")

