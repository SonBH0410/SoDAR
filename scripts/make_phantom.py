"""Create small synthetic ovarian-like ellipses; no patient data is used."""

import argparse
import json

from sodar.phantom import make_phantom


def main():
    parser = argparse.ArgumentParser(description="Generate SoDAR fake data for IO smoke test")
    parser.add_argument("--output", default="/tmp/sodar-phantom")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(make_phantom(args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
