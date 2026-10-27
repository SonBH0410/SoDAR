"""Tiny END-TO-END check: train DDPM -> checkpoint -> DDIM -> S4 student."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from sodar.config import configure
from sodar.diffusion.data import ddpm_config
from sodar.diffusion.runner import sample_generator, train_generator
from sodar.phantom import make_phantom
from scripts.audit_generated import run_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/tmp/sodar-ddpm-demo")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty; use a fresh --root to avoid overwriting outputs")
    fixture = make_phantom(root)
    conf = ddpm_config(path="configs/ddpm_phantom.yaml",
                       data_root=str(root / "real"), manifest=str(root / "manifest.json"),
                       save_root=str(root / "ddpm_runs"))
    checkpoint = train_generator(conf)
    generated_root = root / "ddpm_generated"
    samples = sample_generator(checkpoint, output=generated_root, num_samples=4,
                               batch_size=2, seed=123)
    generated_audit = run_audit(generated_root, expected_count=4)
    student = configure("S4", config_path="configs/phantom.yaml",
                        data_root=str(root / "real"), generated_root=str(generated_root),
                        save_root=str(root / "student_runs"), model="unet",
                        epochs=1, pretrained=False)
    student["data"]["manifest"] = str(root / "manifest.json")
    from sodar.engine import train
    student_dir = train(student)
    subprocess.run([sys.executable, "-m", "scripts.evaluate", "--checkpoint",
                    str(student_dir / "best.pt")], check=True)
    print(json.dumps({"result": "PASS", "fixture": fixture, "samples": samples,
                      "generated_audit": generated_audit,
                      "student_checkpoint": str(student_dir / "best.pt"),
                      "scope": "Synthetic phantom execution only; NO medical validity claim"}, indent=2))


if __name__ == "__main__":
    main()
