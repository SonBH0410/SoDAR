"""Train the joint DDPM from scratch on manifest train-850 ONLY."""

import argparse

from sodar.diffusion.data import ddpm_config
from sodar.diffusion.runner import train_generator


def main():
    parser = argparse.ArgumentParser(description="Train joint image/mask DDPM on train-850")
    parser.add_argument("--config", default="configs/ddpm.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", default=None, help="Trusted SoDAR checkpoint to resume")
    args = parser.parse_args()
    conf = ddpm_config(path=args.config, data_root=args.data_root,
                       manifest=args.manifest, steps=args.steps, batch_size=args.batch_size,
                       grad_accum=args.grad_accum, save_root=args.save_root,
                       seed=args.seed, device=args.device,
                       amp=False if args.no_amp else None)
    print("Saved DDPM checkpoint to", train_generator(conf, resume=args.resume))


if __name__ == "__main__":
    main()
