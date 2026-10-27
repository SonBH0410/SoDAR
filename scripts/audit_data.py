"""Audit original split and optional DDPM pairs. No PyTorch required."""

import argparse
import json

from sodar.config import configure
from sodar.io import audit


def main():
    parser = argparse.ArgumentParser(description="Check SoDAR manifest and source file integrity")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--layout", choices=["manifest", "split"], default=None)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--allow-phantom", action="store_true", help="Allow non-850/150/469 fixture counts")
    args = parser.parse_args()
    conf = configure("S2" if not args.real_only else "B0", config_path=args.config,
                     data_root=args.data_root, generated_root=args.generated_root, layout=args.layout)
    if args.allow_phantom:
        conf["data"]["enforce_counts"] = False
    report = audit(conf, include_synthetic=not args.real_only)
    print(json.dumps(report, indent=2))
    if report["patient_disjoint"] == "unverified":
        print("WARNING: Patient disjointness cannot be verified without patient_id metadata.")


if __name__ == "__main__":
    main()
