import json, numpy as np, sys
sys.path.insert(0, "scripts")
from generate_variants import get_configs, parse_yaml, calculate_priors_and_variants

_, _, datasets = get_configs()
leaf_tokens, sources, _ = parse_yaml(datasets)
domains, leaf_dist, _, _ = calculate_priors_and_variants(leaf_tokens, sources)

v = json.load(open("data/lumi_nested_variants.json"))
for d in domains:
    mean = np.mean([x["mix"][d]["weight"] for x in v])
    print(f"{d:30s} prior={leaf_dist[d]:.4f}  mean={mean:.4f}")
