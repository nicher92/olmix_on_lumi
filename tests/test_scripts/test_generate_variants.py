#!/usr/bin/env python3
"""Tests for scripts/generate_variants.py.

Run from the repository root with the olmix venv:
    $OLMIX_PYTHON -m pytest tests

The important one here is test_weights_are_labelled_with_their_own_dataset.
olmix returns weight vectors ordered by source name, so if leaf_dist is built
in config order every dataset is given another dataset's weight. That bug cost
us a whole swarm and is invisible without a check like this one.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_variants as gv  # noqa: E402

# Must match calculate_num_tokens() in generate_variants.py. Worth promoting to
# a module constant there so the two cannot drift apart.
HEADER_OFFSET = 74


# Deliberately NOT in alphabetical order, and zeta dominates the prior.
# If the ordering regresses, zeta's weight lands on another dataset.
TOKENS = {"zeta": 9_000_000, "mid_lo": 450_000, "mid_hi": 150_000,
          "alpha_a": 200_000, "alpha_b": 100_000, "beta": 100_000}

DATASETS = """
  zeta:
    paths: [{d}/zeta]
  mid:
    multiplier: 2
    quality:
      lo: [{d}/mid_lo]
      hi: [{d}/mid_hi]
  alpha:
    paths: [{d}/alpha_a, {d}/alpha_b]
  beta:
    paths: [{d}/beta]
"""

# leaf -> tokens after multipliers, which is what the prior is built from
SCALED = {"alpha": 300_000, "beta": 100_000, "mid:hi": 300_000,
          "mid:lo": 900_000, "zeta": 9_000_000}


def write_config(tmp_path, datasets=DATASETS, extra_swarm=""):
    """Create fake .bin shards of known size plus a config pointing at them."""
    for name, tokens in TOKENS.items():
        with open(tmp_path / f"{name}.bin", "wb") as f:
            # the only thing that matters is the file's size
            f.truncate(HEADER_OFFSET + 4 * tokens)
    path = tmp_path / "config.yaml"
    path.write_text(
        "settings: {max_tokens: 1000000, repetition_factor: 5.0, seed: 7}\n"
        # a high Dirichlet strength keeps every sample close to the prior,
        # so a mislabelled weight shows up as a large, obvious error
        "swarm: {min_source_strength: 1000, max_source_strength: 10000,"
        " min_topic_strength: 1000, max_topic_strength: 10000,"
        f" enable_bound: false{extra_swarm}}}\n"
        "datasets:" + datasets.format(d=tmp_path)
    )
    return path


def parse(config_path):
    _, swarm, datasets = gv.get_configs(str(config_path))
    leaf_tokens, sources, prefix_map = gv.parse_yaml(datasets)
    return swarm, leaf_tokens, sources, prefix_map


# --------------------------------------------------------------------------
# Leaf ordering: the contract with olmix
# --------------------------------------------------------------------------

def test_leaf_order_is_sorted_by_source_then_quality(tmp_path):
    _, _, sources, _ = parse(write_config(tmp_path))
    assert gv.olmix_leaf_order(sources) == ["alpha", "beta", "mid:hi", "mid:lo", "zeta"]


def test_priors_are_built_in_that_order(tmp_path):
    _, leaf_tokens, sources, _ = parse(write_config(tmp_path))
    domains, leaf_dist, _, total = gv.calculate_priors_and_variants(leaf_tokens, sources)

    assert list(leaf_dist) == domains, "leaf_dist must be in olmix's leaf order"
    assert total == sum(SCALED.values())
    for leaf, tokens in SCALED.items():
        assert leaf_dist[leaf] == pytest.approx(tokens / total)


def test_missing_shard_is_an_error_not_a_silent_drop(tmp_path):
    datasets = DATASETS + "  ghost:\n    paths: [{d}/does_not_exist]\n"
    _, leaf_tokens, sources, _ = parse(write_config(tmp_path, datasets))
    with pytest.raises(SystemExit, match="ghost"):
        gv.calculate_priors_and_variants(leaf_tokens, sources)


# --------------------------------------------------------------------------
# End to end through olmix: does a sampled weight reach the right dataset?
# --------------------------------------------------------------------------

def sample_swarm(tmp_path, config_path, num_variants=15):
    """Run the sampler the way __main__ does and return the variants."""
    import random

    import numpy as np
    from olmix.generate.synthesize_mixture import generate_weights_dirichlet

    swarm, leaf_tokens, sources, prefix_map = parse(config_path)
    domains, leaf_dist, leaf_tokens, _ = gv.calculate_priors_and_variants(leaf_tokens, sources)

    random.seed(7)
    np.random.seed(7)
    mixtures = generate_weights_dirichlet(
        sources=sources, leaf_dist=leaf_dist, leaf_tokens=leaf_tokens,
        num_samples_out=num_variants, max_tokens=1_000_000, repetition_factor=5.0,
        minimum_source_weight=1e-4, minimum_topic_weight=1e-4,
        source_temperature=1.0, topic_temperature=1.0,
        min_source_strength=swarm["min_source_strength"],
        max_source_strength=swarm["max_source_strength"],
        min_topic_strength=swarm["min_topic_strength"],
        max_topic_strength=swarm["max_topic_strength"],
        sample_multiplier=20, enable_bound=False,
        manual_prior=None, manual_topic_prior=None, nonzero_weight=None,
    )
    wrapped = [(np.array([np.asarray(m[0]).flatten()]),
                np.array([np.asarray(m[1]).flatten()])) for m in mixtures]

    run_dir = tmp_path / "run"
    (run_dir / "mixes").mkdir(parents=True)
    variants = gv.write_mixes_to_json(wrapped, domains, "testrun", str(run_dir))
    return variants, prefix_map, run_dir


def test_weights_are_labelled_with_their_own_dataset(tmp_path):
    """Each dataset's average weight should sit near its own prior.

    With a high Dirichlet strength the samples hug the prior, so a swapped
    label shows up as a dataset receiving a completely different share.
    """
    variants, _, _ = sample_swarm(tmp_path, write_config(tmp_path))
    total = sum(SCALED.values())
    for leaf, tokens in SCALED.items():
        mean = sum(v["mix"][leaf]["weight"] for v in variants) / len(variants)
        assert mean == pytest.approx(tokens / total, abs=0.01), f"{leaf} got the wrong share"


def test_same_seed_gives_the_same_swarm(tmp_path):
    config = write_config(tmp_path)
    first, _, _ = sample_swarm(tmp_path / "a", config)
    second, _, _ = sample_swarm(tmp_path / "b", config)
    assert [v["mix"] for v in first] == [v["mix"] for v in second]


# --------------------------------------------------------------------------
# Mix files: what Megatron actually reads
# --------------------------------------------------------------------------

def test_mix_file_splits_a_dataset_over_its_shards_by_size(tmp_path):
    variants, prefix_map, run_dir = sample_swarm(tmp_path, write_config(tmp_path))
    gv.make_megatron_text_files_and_bash_script(variants, prefix_map, "testrun", str(run_dir))

    lines = (run_dir / "mixes" / f"{variants[0]['variant_id']}.txt").read_text().split("\n")
    weights = {Path(p).name: float(w) for w, p in (l.split() for l in lines if l)}

    # alpha has twice as many tokens in shard a as in shard b
    assert weights["alpha_a"] == pytest.approx(2 * weights["alpha_b"], rel=1e-3)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_variants_json_is_written_into_the_run_directory(tmp_path):
    _, _, run_dir = sample_swarm(tmp_path, write_config(tmp_path))
    written = json.loads((run_dir / "variants.json").read_text())
    assert [v["variant_id"] for v in written][:2] == ["testrun-0000", "testrun-0001"]
