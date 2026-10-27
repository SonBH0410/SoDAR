"""Report validation mean ± sample SD across seeds, never silently test-peek."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    groups = defaultdict(list)
    for f in Path(args.runs_dir).glob("*/summary.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (d["experiment"], d["model"])
        groups[key].append(d["best_validation_macro_dice"])
    if not groups:
        raise SystemExit("No run summaries found")
    print("experiment,model,n_seeds,validation_macro_dice_mean,validation_macro_dice_sd")
    for (ex, model), scores in sorted(groups.items()):
        print(f"{ex},{model},{len(scores)},{statistics.mean(scores):.5f},"
              f"{statistics.stdev(scores):.5f}" if len(scores) > 1 else
              f"{ex},{model},1,{scores[0]:.5f},NA")


if __name__ == "__main__":
    main()
