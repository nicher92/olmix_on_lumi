#!/usr/bin/env python3
"""Audit the paths in a config: how many .bin files sit next to each one,
and how many tokens each holds.

Usage: python audit_paths.py configs/config_flagmix.yaml
"""
import json
import sys
from pathlib import Path

import yaml

HEADER_OFFSET = 74  # only used when there is no .info.json sidecar

def tokens_for(bin_path: Path) -> tuple[int, str]:
    name = bin_path.name
    if name.endswith("_text_document.bin"):
        info = bin_path.with_name(name.replace("_text_document.bin", ".info.json"))
        if info.is_file() and info.stat().st_size < 1_000_000:
            try:
                return json.load(open(info))["idx"]["total_tokens"], "info.json"
            except (KeyError, ValueError, OSError):
                pass
    return (bin_path.stat().st_size - HEADER_OFFSET) // 4, "size"

def main(config_path: str) -> None:
    datasets = yaml.safe_load(open(config_path))["datasets"]
    for name, spec in datasets.items():
        prefixes = spec.get("paths") or [p for q in spec["quality"].values() for p in q]
        listed = {Path(f"{p}.bin") for p in prefixes}
        by_dir: dict[Path, set[Path]] = {}
        for p in listed:
            by_dir.setdefault(p.parent, set()).add(p)

        total, sources = 0, set()
        for p in sorted(listed):
            if not p.is_file():
                print(f"{name:28s} MISSING {p}")
                continue
            n, how = tokens_for(p)
            total += n
            sources.add(how)

        extra = []
        for d, used in by_dir.items():
            if d.is_dir():
                extra += sorted(set(d.glob("*.bin")) - used)
        note = f"  (+{len(extra)} more .bin in same dir: {extra[0].name}...)" if extra else ""
        print(f"{name:28s} {len(listed)} file(s)  {total/1e9:9.2f}B tokens [{','.join(sources) or '-'}]{note}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
