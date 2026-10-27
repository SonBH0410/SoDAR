"""Separate held-out test evaluation for already finalized checkpoints."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="One-shot evaluation on test-469")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    import torch
    from sodar.data import RealDataset
    from sodar.engine import assess, build_loader, device_for
    from sodar.models import build_model
    # Only open checkpoints produced by this repository or otherwise trusted.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conf = checkpoint["config"]
    device = device_for(conf)
    model = build_model(conf, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = build_loader(RealDataset(conf, "test"), conf, seed_offset=7)
    result = assess(model, loader, device)
    print(json.dumps(result, indent=2))
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "test_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test report saved to {output}")


if __name__ == "__main__":
    main()
