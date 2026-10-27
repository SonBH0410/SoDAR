"""Run `python -m scripts.train --experiment B0|...|S5`."""

import argparse

from sodar.config import configure


def get_parser():
    parser = argparse.ArgumentParser(description="Train SoDAR without test-set peeking")
    parser.add_argument("--experiment", choices=["B0", "B1", "B2", "S1", "S2", "S3", "S4", "S5"], required=True)
    parser.add_argument("--config", default=None, help="Custom config YAML overriding base.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--layout", choices=["manifest", "split"], default=None)
    parser.add_argument("--model", choices=["unet", "sovasegnet", "segformer_b2"], default="unet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    conf = configure(args.experiment, config_path=args.config, data_root=args.data_root,
                     generated_root=args.generated_root, layout=args.layout,
                     model=args.model, seed=args.seed, epochs=args.epochs,
                     pretrained=False if args.no_pretrained else None, save_root=args.save_root)
    from sodar.engine import train
    print("Saved to", train(conf))


if __name__ == "__main__":
    main()
