"""Inspect the active Python, package imports and GPU compatibility."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Inspect current SoDAR environment")
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-segformer", action="store_true")
    args = parser.parse_args()
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "repo": str(Path(__file__).resolve().parents[1]), "packages": {}}
    ok = True
    for name in ("numpy", "Pillow", "PyYAML", "tqdm", "torch", "torchvision", "transformers"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        report["packages"][name] = {"version": version}

    for name in ("numpy", "Pillow", "PyYAML", "tqdm"):
        if report["packages"][name]["version"] is None:
            report["packages"][name]["error"] = "required core package is missing"
            ok = False

    for name, required in (("torch", args.require_torch),
                           ("torchvision", args.require_torch),
                           ("transformers", args.require_segformer)):
        if report["packages"][name]["version"] is None:
            if required:
                report["packages"][name]["error"] = "missing"
                ok = False
            continue
        try:
            mod = importlib.import_module(name)
            if name == "torch":
                report["torch_cuda_available"] = mod.cuda.is_available()
                report["torch_cuda_runtime"] = mod.version.cuda
                if mod.cuda.is_available():
                    report["gpu_name"] = mod.cuda.get_device_name(0)
            elif name == "transformers" and args.require_segformer:
                getattr(mod, "SegformerModel")
        except Exception as exc:
            report["packages"][name]["error"] = f"{type(exc).__name__}: {exc}"
            ok = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit("Some required packages are missing or cannot be imported; see errors above.")


if __name__ == "__main__":
    main()
