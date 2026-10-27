"""Sequential, predeclared 3-seed experiment matrix."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run a comparable SoDAR experiment matrix")
    parser.add_argument("--experiments", nargs="+", default=["B0", "B1", "B2", "S1", "S2", "S4"])
    parser.add_argument("--models", nargs="+", default=["unet", "sovasegnet", "segformer_b2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--layout", choices=["manifest", "split"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    for ex in args.experiments:
        for model in args.models:
            for seed in args.seeds:
                cmd = [sys.executable, "-m", "scripts.train", "--experiment", ex,
                       "--model", model, "--seed", str(seed), "--data-root", args.data_root]
                if args.config:
                    cmd += ["--config", args.config]
                if args.generated_root:
                    cmd += ["--generated-root", args.generated_root]
                if args.layout:
                    cmd += ["--layout", args.layout]
                if args.epochs is not None:
                    cmd += ["--epochs", str(args.epochs)]
                if args.no_pretrained:
                    cmd += ["--no-pretrained"]
                print("RUN:", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
