"""Dependency-light phantom flow; optional actual torch one-epoch train/eval."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from sodar.config import configure
from sodar.io import (audit, generated_pairs, load_manifest, preprocess_image,
                      probs_from_generated, read_two_channel_array, real_pair_paths)
from sodar.phantom import make_phantom


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(np.asarray(x, dtype=np.float64), -30, 30)))


def toy_real_loss(theta, x, y):
    logits = theta[0]*np.asarray(x, dtype=np.float64) + theta[1]
    target = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - target*logits, dtype=np.float64))


def toy_real_grad(theta, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    errors = sigmoid(theta[0]*x + theta[1]) - y
    return np.array([np.mean(errors*x, dtype=np.float64),
                     np.mean(errors, dtype=np.float64)], dtype=np.float64)


def toy_soft_grad(theta, x, q):
    x = np.asarray(x, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    errors = sigmoid(theta[0]*x + theta[1]) - q
    return np.array([np.mean(errors*x, dtype=np.float64),
                     np.mean(errors, dtype=np.float64)], dtype=np.float64)


def run_numpy_flow(phantom_root: Path) -> dict:
    config = configure("S4", config_path=str(Path(__file__).resolve().parents[1]/"configs"/"phantom.yaml"),
                       data_root=str(phantom_root/"real"), generated_root=str(phantom_root/"generated"))
    config["data"]["manifest"] = str(phantom_root/"manifest.json")
    report = audit(config)
    assert report["patient_disjoint"] == "yes"
    assert [report["real"][s]["count"] for s in ["train", "val", "test"]] == [16, 8, 8]
    assert report["synthetic"]["count"] == 12
    manifest = load_manifest(config)

    def real_arrays(record):
        image_path, mask_path = real_pair_paths(config, "train", record)
        with Image.open(image_path) as img:
            x = preprocess_image(img, 64, "imagenet")[0]
        with Image.open(mask_path) as mask:
            y = (np.asarray(mask) > 0).astype(np.float64)
        return x, y

    xr, yr = real_arrays(manifest["train"][0])
    xm, ym = real_arrays(manifest["train"][1])
    syn_img_path, syn_logits_path = generated_pairs(config)[0]
    with Image.open(syn_img_path) as img:
        xs = preprocess_image(img, 64, "imagenet")[0]
    logits = read_two_channel_array(syn_logits_path, expected_format="logits")
    q = probs_from_generated(logits, temperature=2.0, fmt="logits")
    assert q.shape == (2, 64, 64)
    assert np.allclose(q.sum(axis=0), 1.0, atol=1e-6)
    assert q.min() >= 0 and q.max() <= 1
    hard = (logits[1] > logits[0]).astype(np.float32)
    assert hard.shape == yr.shape

    # Tiny reference student (two parameters, binary pixel classifier):
    # the only purpose is to validate the *data → soft supervision → virtual
    # update → real meta objective* flow without needing PyTorch.
    theta = np.array([0.15, -0.03], dtype=np.float64)
    eta, lam = 0.02, 0.10
    grad_real = toy_real_grad(theta, xr, yr)
    grad_syn = toy_soft_grad(theta, xs, q[1])

    def virtual_meta_loss(weight):
        virtual_theta = theta - eta*(grad_real + weight*grad_syn)
        return toy_real_loss(virtual_theta, xm, ym)

    # Double-precision input/loss is intentional. In some NumPy/Python builds
    # float32 intermediates make a finite-difference check of an O(1e-2)
    # hypergradient spuriously fail even when the analytic gradient is correct.
    eps = 1e-3
    finite_difference = (virtual_meta_loss(lam+eps)-virtual_meta_loss(lam-eps))/(2*eps)
    updated = theta - eta*(grad_real+lam*grad_syn)
    hypergrad_approx = -eta*np.dot(toy_real_grad(updated, xm, ym), grad_syn)
    error = abs(finite_difference-hypergrad_approx)
    if not (np.isfinite(finite_difference) and np.isfinite(hypergrad_approx)
            and np.isclose(finite_difference, hypergrad_approx, rtol=1e-4, atol=1e-6)):
        raise RuntimeError(
            "Phantom hypergradient mismatch: "
            f"finite_difference={finite_difference:.12g}, "
            f"analytic={hypergrad_approx:.12g}, absolute_error={error:.12g}, "
            f"numpy={np.__version__}, python={sys.version.split()[0]}. "
            "Regenerate the phantom in a fresh directory and share this message if it persists."
        )
    return {"result": "PASS", "real": report["real"], "synthetic": report["synthetic"],
            "q_shape": list(q.shape), "q_sum_max_error": float(np.max(np.abs(q.sum(axis=0)-1))),
            "lambda_initial": lam, "meta_gradient_numeric": float(finite_difference),
            "meta_gradient_analytic": float(hypergrad_approx),
            "meta_gradient_absolute_error": float(error),
            "meta_step_loss": virtual_meta_loss(lam),
            "scope": "IO + NumPy reference; not a real PyTorch model forward"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phantom-root", default="/tmp/sodar-phantom")
    parser.add_argument("--with-torch", action="store_true", help="Run real U-Net S4 one-epoch train/evaluation")
    args = parser.parse_args()
    root = Path(args.phantom_root)
    if not (root/"manifest.json").exists():
        make_phantom(root)
    result = run_numpy_flow(root)
    print(json.dumps(result, indent=2))
    if args.with_torch:
        if importlib.util.find_spec("torch") is None:
            raise SystemExit("--with-torch requires PyTorch. Install matching torch/torchvision "
                             "for your CUDA from https://pytorch.org/get-started/locally/")
        commands = [
            [sys.executable, "-m", "scripts.train", "--experiment", "S4", "--config", "configs/phantom.yaml",
             "--model", "unet", "--no-pretrained", "--epochs", "1", "--data-root", str(root/"real"),
             "--generated-root", str(root/"generated"), "--save-root", str(root/"runs")],
            [sys.executable, "-m", "scripts.evaluate", "--checkpoint", str(root/"runs"/"S4_unet_seed42"/"best.pt")]
        ]
        # Override the fixture manifest path if the requested phantom root differs.
        if root.resolve() != Path("/tmp/sodar-phantom").resolve():
            raise SystemExit("Full torch smoke expects /tmp/sodar-phantom, matching configs/phantom.yaml")
        for cmd in commands:
            print("Executing:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        print("PASS: full torch one-epoch S4 train and test-evaluation flow")


if __name__ == "__main__":
    main()
