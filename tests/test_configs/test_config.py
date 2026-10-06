#!/usr/bin/env python3
"""Checks on the swarm configs in configs/.

These test the config files, not the code that reads them. Every check here
corresponds to a mistake that has actually cost us something:

  - a duplicate key made swallow_math_v2 point at OpenWebMath for a whole swarm
  - a dataset with neither paths nor quality is silently invisible
  - a name in the frozen file that is not a dataset freezes nothing, and the
    swarm quietly becomes an ordinary one

Run from the repository root:
    $OLMIX_PYTHON -m pytest tests                  # structure only
    $OLMIX_PYTHON -m pytest tests -m data          # also check the .bin files exist
"""

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SWARM_CONFIGS = sorted(
    p for p in (REPO_ROOT / "configs").glob("*.yaml")
    # run_config.yaml and fit configs are not swarm configs
    if "datasets" in yaml.safe_load(p.read_text() or "{}") or {}
)


class UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML keeps the last of two duplicate keys without complaining."""

    def construct_mapping(self, node, deep=False):
        keys = [str(self.construct_object(k, deep=deep)) for k, _ in node.value]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise AssertionError(
                f"duplicate key(s) {dupes} near line {node.start_mark.line + 1}"
            )
        return super().construct_mapping(node, deep=deep)


def leaf_names_of(datasets):
    """What the generator calls a leaf: a dataset, or dataset:bucket."""
    leaves = []
    for name, spec in datasets.items():
        if spec.get("quality"):
            leaves += [f"{name}:{bucket}" for bucket in spec["quality"]]
        else:
            leaves.append(name)
    return leaves


def prefixes_of(spec):
    if spec.get("paths"):
        return list(spec["paths"])
    return [p for bucket in (spec.get("quality") or {}).values() for p in bucket]


@pytest.fixture(params=SWARM_CONFIGS, ids=lambda p: p.name)
def config(request):
    return request.param


def test_config_dir_is_not_empty():
    assert SWARM_CONFIGS, "no swarm configs found in configs/"


def test_loads_without_duplicate_keys(config):
    yaml.load(config.read_text(), Loader=UniqueKeyLoader)


def test_has_the_blocks_the_generator_reads(config):
    cfg = yaml.safe_load(config.read_text())
    assert "settings" in cfg and "datasets" in cfg
    for key in ("max_tokens", "repetition_factor"):
        assert isinstance(cfg["settings"].get(key), (int, float)), f"settings.{key}"
        assert cfg["settings"][key] > 0


def test_every_dataset_has_paths_or_quality(config):
    datasets = yaml.safe_load(config.read_text())["datasets"]
    assert datasets, "no datasets"
    for name, spec in datasets.items():
        has_paths = bool(spec.get("paths"))
        has_quality = bool(spec.get("quality"))
        assert has_paths != has_quality, f"{name}: needs exactly one of paths/quality"
        assert prefixes_of(spec), f"{name}: no path prefixes"


def test_multipliers_are_positive_numbers(config):
    datasets = yaml.safe_load(config.read_text())["datasets"]
    for name, spec in datasets.items():
        mult = spec.get("multiplier", 1)
        assert isinstance(mult, (int, float)) and not isinstance(mult, bool), name
        assert mult > 0, f"{name}: multiplier must be > 0"


def test_paths_are_prefixes_without_a_suffix(config):
    """The generator appends .bin itself, so a suffix here means a missing file."""
    datasets = yaml.safe_load(config.read_text())["datasets"]
    for name, spec in datasets.items():
        for prefix in prefixes_of(spec):
            assert not prefix.endswith((".bin", ".idx")), f"{name}: {prefix}"


def test_no_path_is_used_by_two_datasets(config):
    datasets = yaml.safe_load(config.read_text())["datasets"]
    seen = {}
    for name, spec in datasets.items():
        for prefix in prefixes_of(spec):
            assert prefix not in seen, f"{prefix} used by both {seen[prefix]} and {name}"
            seen[prefix] = name


def test_frozen_mix_names_match_dataset_names(config):
    """A name in the frozen file that is not a dataset freezes nothing."""
    cfg = yaml.safe_load(config.read_text())
    frozen_path = (cfg.get("swarm") or {}).get("existing_mix_file")
    if not frozen_path:
        pytest.skip("no existing_mix_file")

    frozen_file = REPO_ROOT / frozen_path
    assert frozen_file.is_file(), f"existing_mix_file not found: {frozen_path}"

    frozen = json.loads(frozen_file.read_text())
    leaves = set(leaf_names_of(cfg["datasets"]))
    unknown = sorted(set(frozen) - leaves)
    assert not unknown, f"not leaves in this config: {unknown}"
    assert set(frozen) != leaves, "every leaf is frozen; nothing would vary"


def test_manual_prior_names_are_sources_or_existing(config):
    cfg = yaml.safe_load(config.read_text())
    prior = (cfg.get("swarm") or {}).get("manual_prior")
    if not prior:
        pytest.skip("no manual_prior")
    allowed = set(cfg["datasets"]) | {"existing"}  # manual_prior takes sources, not leaves
    assert not set(prior) - allowed, f"unknown names: {sorted(set(prior) - allowed)}"


@pytest.mark.data
def test_every_bin_file_exists(config):
    """Only meaningful on LUMI with the data mounted: run with -m data."""
    datasets = yaml.safe_load(config.read_text())["datasets"]
    missing = [f"{p}.bin" for spec in datasets.values() for p in prefixes_of(spec)
               if not Path(f"{p}.bin").is_file()]
    assert not missing, f"{len(missing)} missing .bin file(s), e.g. {missing[0]}"
