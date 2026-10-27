# Executed phantom validation report

## Updated v0.2 — end-to-end DDPM generator included

- New `scripts.smoke_ddpm --root /tmp/sodar-ddpm-full-smoke` passed on a
  Python 3.12 + PyTorch 2.5.1+cpu environment.
- The actual joint DDPM trained 2 steps on 16 phantom train cases, saved
  checkpoints, sampled 4 paired images + two-channel raw logits with DDIM,
  and passed the full generated-pair/provenance audit.
- The freshly generated output fed S4/U-Net for one epoch; the student
  checkpoint saved and loaded successfully for a separate test evaluation.
- Checkpoint resume from step 2 to step 3 and safe sample-output resume were
  exercised. **Synthetic provenance is checked against train-list digest**.
- The phantom generator's foreground occupancy (~0.94) is **not** realistic:
  two training steps are only a smoke test; no claim of synthetic image quality.
- Real OTU_2D DDPM training at 256×256 and workstation CUDA throughput have
  **not** been tested here. No medical Dice improvement has been established.

## Updated v0.1.1 SIGM verification

- The phantom reference computes its two-parameter numerical derivative in
  explicit float64. With NumPy 2.3.5 its analytic-versus-numeric absolute
  error was **7.04e-14**. This is a numerical sanity check, not a result.
- End-to-end one-epoch S4/U-Net training and phantom test-evaluation passed
  again using PyTorch 2.5.1+cpu / torchvision 0.20.1+cpu / transformers 4.46.3.
- **7/7** unit tests passed in the torch-enabled environment.
- v0.1.1 separates base package dependencies from PyTorch/CUDA installation,
  adds `configs/sigm.yaml` and an environment diagnostic CLI.
- **Real OTU_2D images are not mounted here**. Their filesystem paths and
  any real ultrasound segmentation performance remain unverified.

## Original v0.1.0 report

Test environment: 2026-10-27, Python 3.12, **PyTorch 2.5.1+cpu**,
torchvision 0.20.1+cpu, Transformers 4.46.3, synthetic-only 64×64 phantoms.
No original patient data or original DDPM weights were available.

## Verified outcomes

| Check | Outcome |
|---|---|
| Python compileall | PASS |
| Contract unit tests | 3/3 PASS |
| Torch unit test (U-Net loss + meta update) | PASS |
| Phantom manifest | 16/8/8; all patient IDs disjoint |
| Synthetic pairs | 12/12; image + 2×64×64 logits |
| Softmax sum error | 1.19 × 10⁻⁷ maximum |
| First-order hypergradient vs finite differences | PASS (absolute error ≈ 5.7 × 10⁻¹³) |
| B0, B1, B2, S1, S2, S3, S4, S5 | One complete U-Net epoch each: PASS |
| S4 end-to-end checkpoint + held-out phantom evaluate | PASS |
| SovaSeg-Net forward and backward | PASS (random initialization, 64×64) |
| SegFormer-B2 forward and backward | PASS (random initialization, 64×64) |
| Real OTU_2D train / clinical results | NOT TESTED: source images absent |
| Pretrained weight download/inference | NOT TESTED: external model weights absent |

In the one-epoch phantom run, S3's naive lambda decreased from 0.100 to
approximately 0.0988, while S4's meta lambda increased to approximately
0.1020. This is a **pipeline illustration only**, not evidence that SoDAR
helps on real ultrasound images. Every one-epoch phantom run had validation
macro Dice near zero; the four minibatches and randomly initialized model
were not intended to converge. Real efficacy remains unknown until the
850/150/469 experiment is run.

Reproduce:

```bash
python -m scripts.make_phantom --output /tmp/sodar-phantom
python -m scripts.smoke_test --phantom-root /tmp/sodar-phantom --with-torch
python -m unittest discover -s tests -v
```

The `--with-torch` step needs PyTorch installed. All eight configuration
smokes can be run on the phantom using `RUN.md` and `--config
configs/phantom.yaml --no-pretrained --epochs 1`.
