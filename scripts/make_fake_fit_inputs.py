#!/usr/bin/env python3
"""Build fit inputs from a variants.json, with invented BPB scores.

ratios.csv holds the real mixture weights; metrics.csv holds fake scores
generated from a known rule, so you can check whether the fit recovers it:

    bpb(task) = base(task) + slope(task) * weight(TARGET) + noise

Two tasks are given a negative slope (more TARGET data -> lower BPB, i.e.
better), one a positive slope, and one a slope of zero. A working fit should
therefore recommend MORE of TARGET for the first two, less for the third, and
be indifferent on the fourth.

Usage:
    python make_fake_fit_inputs.py data/runs/<run>/variants.json out_dir/
"""

import json
import random
import sys
from pathlib import Path

TARGET = "open_web_math"

# task -> (base bpb, slope, noise size)
TASKS = {
    "arc_easy_bpb":       (1.80, -1.2, 0.01),   # likes TARGET
    "openbookqa_bpb":     (1.95, -0.8, 0.01),   # likes TARGET a bit
    "copa_bpb":           (1.70, +0.9, 0.01),   # dislikes TARGET
    "social_iqa_bpb":     (1.88,  0.0, 0.01),   # indifferent
}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    variants = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    if TARGET not in variants[0]["mix"]:
        print(f"ERROR: '{TARGET}' is not a dataset in this variants file")
        return 1

    rng = random.Random(0)
    domains = sorted(variants[0]["mix"])

    ratio_rows, metric_rows = [], []
    for i, v in enumerate(variants):
        run = v["variant_id"]
        weights = {d: v["mix"][d]["weight"] for d in domains}
        ratio_rows.append([run, run, i] + [weights[d] for d in domains])

        w = weights[TARGET]
        scores = [base + slope * w + rng.uniform(-n, n)
                  for base, slope, n in TASKS.values()]
        metric_rows.append([run, run, i] + [round(s, 6) for s in scores])

    def write(path, header, rows):
        with open(path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"wrote {path}")

    write(out_dir / "ratios.csv", ["run", "name", "index"] + domains, ratio_rows)
    write(out_dir / "metrics.csv", ["run", "name", "index"] + list(TASKS), metric_rows)

    print(f"\n{len(variants)} runs, {len(domains)} datasets, {len(TASKS)} tasks")
    print(f"{TARGET} ranges {min(r[3 + domains.index(TARGET)] for r in ratio_rows):.4f} "
          f"to {max(r[3 + domains.index(TARGET)] for r in ratio_rows):.4f}")
    print("\nExpected from a working fit: more " + TARGET + " for arc_easy and "
          "openbookqa, less for copa, no clear preference for social_iqa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
