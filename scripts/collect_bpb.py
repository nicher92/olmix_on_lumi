#!/usr/bin/env python3
"""Turn lm-eval BPB results into the ratios.csv / metrics.csv pair that
``olmix fit`` consumes.

Reads lm-eval's raw result JSONs directly rather than going through
``oellm-eval collect``, which drops any task whose name starts with ``mmlu_``
and so would discard every ``mmlu_*_bpb`` result.

Each result JSON records the checkpoint path it was run against. A variant is
identified by finding which id from the variant file appears in that path, so
the checkpoint directory layout does not matter.

Usage:
    python scripts/collect_bpb.py --results-dir /path/to/eval-output/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, type=Path,
                   help="Directory of lm-eval result JSONs (searched recursively).")
    p.add_argument("--variants-json", type=Path, default=Path("data/lumi_nested_variants.json"),
                   help="Variant file written by scripts/generate_variants.py.")
    p.add_argument("--output-dir", type=Path, default=Path("data/olmix_fit"),
                   help="Where to write ratios.csv and metrics.csv.")
    return p.parse_args()


def checkpoint_of(payload: dict) -> str | None:
    """The model path an lm-eval run used, from its recorded config."""
    model_args = (payload.get("config") or {}).get("model_args")
    if isinstance(model_args, str):
        for part in model_args.split(","):
            if part.startswith("pretrained="):
                return part.split("=", 1)[1]
    if isinstance(model_args, dict):
        return model_args.get("pretrained")
    return None


def bpb_of(task_results: dict) -> float | None:
    """The bits_per_byte value for one task.

    lm-eval suffixes metric keys with the filter name ('bits_per_byte,none').
    The matching stderr key is always 'N/A' -- BPB is a corpus-level ratio, not
    a mean over documents, so it has no bootstrap stderr -- and is skipped.
    """
    for key, value in task_results.items():
        if key.startswith("bits_per_byte") and "stderr" not in key:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def read_results(results_dir: Path, variant_ids: list[str]) -> tuple[dict, list[str]]:
    """Return {variant: {task: bpb}} and a list of warnings."""
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    checkpoints: dict[str, set[str]] = defaultdict(set)
    warnings: list[str] = []

    for path in sorted(results_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "results" not in payload:
            continue  # not a results file (e.g. a samples dump)

        checkpoint = checkpoint_of(payload)
        if not checkpoint:
            warnings.append(f"{path.name}: no checkpoint path recorded")
            continue

        # Longest match wins, so 'swarm-1' does not shadow 'swarm-11'.
        matches = sorted((v for v in variant_ids if v in checkpoint), key=len, reverse=True)
        if not matches:
            warnings.append(f"{path.name}: '{checkpoint}' matches no known variant")
            continue
        variant = matches[0]
        checkpoints[variant].add(checkpoint)

        for task, task_results in payload["results"].items():
            if not isinstance(task_results, dict) or not task.endswith("_bpb"):
                continue
            bpb = bpb_of(task_results)
            if bpb is None or not math.isfinite(bpb):
                warnings.append(f"{variant}/{task}: no usable bits_per_byte value")
            elif not 0.0 < bpb < 12.0:
                warnings.append(f"{variant}/{task}: implausible BPB {bpb:.4f}")
                scores[variant][task] = bpb
            else:
                scores[variant][task] = bpb

    # One variant, two checkpoints means a stale copy is still around. Say so
    # rather than picking one by directory order.
    ambiguous = {v: paths for v, paths in checkpoints.items() if len(paths) > 1}
    if ambiguous:
        detail = "\n".join(f"  {v}:\n" + "\n".join(f"    {p}" for p in sorted(paths))
                           for v, paths in ambiguous.items())
        raise SystemExit(
            "These variants have results from more than one checkpoint:\n\n"
            f"{detail}\n\nEvaluate only the checkpoint you want to keep."
        )

    return scores, warnings


def main() -> int:
    args = parse_args()

    if not args.variants_json.exists():
        raise SystemExit(f"Variant file not found: {args.variants_json}")
    variants = json.loads(args.variants_json.read_text())
    variant_ids = [v["variant_id"] for v in variants]

    scores, warnings = read_results(args.results_dir, variant_ids)
    if not scores:
        raise SystemExit("No BPB results found. Did the eval jobs finish?")

    # Average the MMLU STEM subjects into one column, as in the Olmix README.
    for by_task in scores.values():
        subjects = {t: v for t, v in by_task.items() if re.match(r"^mmlu_.+_bpb$", t)}
        if subjects:
            for task in subjects:
                del by_task[task]
            by_task["mmlu_stem_bpb"] = sum(subjects.values()) / len(subjects)

    index = {v["variant_id"]: i for i, v in enumerate(variants)}
    keep = sorted(scores, key=lambda v: index[v])

    ratios = pd.DataFrame([
        {"run": v["variant_id"], "name": v["variant_id"], "index": index[v["variant_id"]],
         **{domain: float(cfg["weight"]) for domain, cfg in v["mix"].items()}}
        for v in variants if v["variant_id"] in scores
    ])
    metrics = pd.DataFrame([
        {"run": v, "name": v, "index": index[v], **scores[v]} for v in keep
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(args.output_dir / "ratios.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    metric_cols = [c for c in metrics.columns if c not in ("run", "name", "index")]
    missing = [v for v in variant_ids if v not in scores]
    incomplete = metrics[metrics[metric_cols].isna().any(axis=1)]["name"].tolist()

    print(f"{len(metrics)} runs x {len(metric_cols)} metrics -> {args.output_dir}")
    print(f"metrics: {', '.join(metric_cols)}")
    if missing:
        print(f"\n[warn] {len(missing)} variant(s) have a mixture but no results: "
              + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if incomplete:
        print(f"\n[warn] {len(incomplete)} variant(s) missing some metrics (olmix fit drops these): "
              + ", ".join(incomplete[:8]))
    for w in warnings[:10]:
        print(f"[warn] {w}")
    if len(warnings) > 10:
        print(f"[warn] ... and {len(warnings) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
