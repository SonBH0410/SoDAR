# RUN — reproducible SoDAR commands

Run commands from the repository root. Do **not** change the test partition in
the included JSON, and do not run test evaluation while selecting hyperparameters.

## 0. Environment

```bash
python --version
python -m pip install -e .
python -m pip install -e '.[segformer]'   # required by segformer_b2
```

For CUDA, install matching PyTorch/torchvision builds from official PyTorch
instructions before installing this package.
If Hugging Face's default cache is not writable, use
`export HF_HOME=/kaggle/working/hf-cache` before loading MiT-B2 pretrained.

## 1. Local original data and DDPM data

Original layout:

```bash
python -m scripts.audit_data \
  --data-root '/kaggle/input/otu-2d-dataset/OTU_2D' \
  --generated-root '/kaggle/input/your-ddpm/generated_dataset_full'
```

Kaggle `dataset_split` layout:

```bash
python -m scripts.audit_data \
  --layout split \
  --data-root '/kaggle/input/otu2d-dataset/dataset_split' \
  --generated-root '/kaggle/input/your-ddpm/generated_dataset_full'
```

The Kaggle paths above are **examples**; adapt to your dataset slug. A missing
file is a hard error. Expected real counts: 850/150/469. An audit of paths
does not substitute for patient ID verification.

## 2. Run one baseline

```bash
python -m scripts.train --experiment B0 --model unet \
  --data-root /path/to/OTU_2D --seed 42
python -m scripts.train --experiment B1 --model unet \
  --data-root /path/to/OTU_2D --seed 42
python -m scripts.train --experiment B2 --model unet \
  --data-root /path/to/OTU_2D --seed 42
```

## 3. Run SoDAR experiments (without Flip)

```bash
python -m scripts.train --experiment S1 --model unet \
  --data-root /path/to/OTU_2D --generated-root /path/to/generated_dataset_full
python -m scripts.train --experiment S2 --model unet \
  --data-root /path/to/OTU_2D --generated-root /path/to/generated_dataset_full
python -m scripts.train --experiment S3 --model unet \
  --data-root /path/to/OTU_2D --generated-root /path/to/generated_dataset_full
python -m scripts.train --experiment S4 --model unet \
  --data-root /path/to/OTU_2D --generated-root /path/to/generated_dataset_full
python -m scripts.train --experiment S5 --model unet \
  --data-root /path/to/OTU_2D --generated-root /path/to/generated_dataset_full
```

## 4. Three-seed matrix, SegFormer-B2 pretrained

```bash
python -m scripts.run_experiments --experiments B0 B1 B2 S1 S2 S4 \
  --models segformer_b2 --seeds 42 123 2026 \
  --data-root /path/to/OTU_2D \
  --generated-root /path/to/generated_dataset_full
```

Use `--models unet sovasegnet segformer_b2` for all architectures. Models are
run sequentially; choose sensible batch sizes for the available GPU. For
budgeted pilot runs, use `--epochs 5`, then return to 50 epochs for the paper.

## 5. Test selected checkpoints

```bash
python -m scripts.evaluate --checkpoint runs/S4_segformer_b2_seed42/best.pt
python -m scripts.summarize --runs-dir runs
```

Summarization reads the **validation** metric from each run; testing is
explicit. For publication, only evaluate final selected runs on the 469-image
test set and report mean ± SD and paired seed-level differences.

## 6. Phantom dataset — executable without real patient data

```bash
python -m scripts.make_phantom --output /tmp/sodar-phantom
python -m scripts.audit_data --config configs/phantom.yaml \
  --data-root /tmp/sodar-phantom/real \
  --generated-root /tmp/sodar-phantom/generated --allow-phantom
python -m scripts.smoke_test --phantom-root /tmp/sodar-phantom
```

The smoke test exercises JSON split membership, image/mask loading, DDPM
`.npy` contracts, temperature conversion, binary soft KL, hard BCE and a
NumPy first-order meta-weight reference check. If PyTorch is available,
`--with-torch` additionally runs a **one-epoch actual U-Net S4 training** on
the phantom; this is not a clinical validation or a speed benchmark.

## 7. Paths, formats, reproducibility

- Edit `configs/base.yaml` or pass CLI `--data-root`, `--generated-root`,
  `--layout`.
- DDPM must be trained on **train-850 only**, despite possible older notebooks
  using another split. Patient-disjoint manifests should be verified against
  patient metadata when available.
- `generated/images/000000.jpg` must pair with
  `generated/logits_npy/000000.npy` (or `soft_labels_npy/soft_label_000000.npy`).
- `data.synthetic_format: logits` is the default. If `.npy` already contains
  probabilities, set `probabilities` to prevent double softmax.
- Temperature (T) is applied symmetrically to teacher DDPM logits and student
  logits; test `1, 2, 4` as a dedicated ablation. Do not reuse the old 12/2
  mismatch.
- Use `--no-pretrained` only for a diagnostic sanity test, not as a published
  pretrained SegFormer-B2 baseline.
- Synthetic batch size and real batch size are fixed across S1–S5; each real
  minibatch anchors exactly one optimizer update.

## 8. Interpretation

If S4 learns `lambda -> 0` and matches B0, it has *suppressed negative
transfer*, not shown that synthetic augmentation helps. A positive contribution
requires reproducible improvement over B1 and, ideally, B2, plus S4 > S2 and
S2 > S1 on the same synthetic samples.
