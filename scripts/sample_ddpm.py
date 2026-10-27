"""DDIM export: JPEG ultrasound images + raw 2-channel logits, no softmax."""

import argparse
import json

from sodar.diffusion.runner import sample_generator


def main():
    parser = argparse.ArgumentParser(description="Sample images + raw logits from a trusted DDPM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1177)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-output", action="store_true", help="Resume only if provenance matches")
    args = parser.parse_args()
    result = sample_generator(args.checkpoint, output=args.output,
                              num_samples=args.num_samples, batch_size=args.batch_size,
                              seed=args.seed, device_name=args.device,
                              resume_output=args.resume_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
