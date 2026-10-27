# RUN — reproducible SoDAR commands

Run commands from the repository root. Do **not** change the test partition in
the included JSON, and do not run test evaluation while selecting hyperparameters.

## NEW in v0.2: train the missing DDPM generator first

**No old weights or generated samples are required.** The project now trains
and samples its own joint DDPM from scratch, using only the manifest train-850.
The full phantom test checks actual DDPM training, checkpoint, sampling,
soft logits, S4 student training and test-evaluation:

```bash
python -m scripts.smoke_ddpm --root /tmp/sodar-ddpm-demo
```

Choose a fresh `--root` for a repeat; the smoke refuses to overwrite outputs.
For real OTU_2D on SIGM:

```bash
# A. Confirm all 1469 original pairs exist
python -m scripts.audit_data --config configs/sigm.yaml --real-only

# B. Engineering pilot: NOT a converged generator
python -m scripts.train_ddpm --config configs/ddpm_sigm.yaml \
  --steps 1000 --batch-size 2 --grad-accum 4 \
  --save-root runs/ddpm_pilot

# C. Test export of a SMALL pilot sample; this is not paper data
python -m scripts.sample_ddpm --checkpoint runs/ddpm_pilot/latest.pt \
  --output runs/ddpm_pilot_generated --num-samples 16 --batch-size 4
python -m scripts.audit_generated \
  --generated-root runs/ddpm_pilot_generated --expected-count 16

# D. Controlled study: train generator from scratch with predeclared budget.
# IMPORTANT: Do not mix pilot & final checkpoints in the same directory.
python -m scripts.train_ddpm --config configs/ddpm_sigm.yaml \
  --steps 100000 --batch-size 2 --grad-accum 4 \
  --save-root runs/ddpm_full

# E. Export a fixed final synthetic corpus from EMA checkpoint
python -m scripts.sample_ddpm --checkpoint runs/ddpm_full/latest.pt \
  --output /mnt/nvme0/home/sonbh/Dataset/SoDAR-DDPM \
  --num-samples 1177 --batch-size 4
python -m scripts.audit_generated \
  --generated-root /mnt/nvme0/home/sonbh/Dataset/SoDAR-DDPM \
  --expected-count 1177

# F. Compare fixed synthetic teacher vs learned lambda, WITHOUT Flip
python -m scripts.train --experiment S2 --config configs/sigm.yaml \
  --model unet --seed 42 \
  --generated-root /mnt/nvme0/home/sonbh/Dataset/SoDAR-DDPM
python -m scripts.train --experiment S4 --config configs/sigm.yaml \
  --model unet --seed 42 \
  --generated-root /mnt/nvme0/home/sonbh/Dataset/SoDAR-DDPM
```

**Resume** an interrupted DDPM run without changing architecture, dataset,
optimizer, batch size or accumulation. Only raise `--steps` beyond the saved
step if extending training:

```bash
python -m scripts.train_ddpm --config configs/ddpm_sigm.yaml \
  --steps 120000 --batch-size 2 --grad-accum 4 \
  --save-root runs/ddpm_full --resume runs/ddpm_full/latest.pt
```

Sampling supports `--resume-output` only for an interrupted output directory
whose checkpoint SHA, seed, sample count and batch size match `provenance.json`.
It never silently overwrites existing pairs. See [docs/DDPM.md](docs/DDPM.md)
for the loss, clean split, EMA, and reproducibility limitations. Never run the
old Kaggle notebook's `pip uninstall torch` cells in `(SonBH-env)`.

If GPU memory is insufficient, start a **new run** with `--batch-size 1
--grad-accum 8`; do not alter batch-size parameters when resuming an existing
checkpoint. At real 256×256 resolution, model runtime and generated quality
must be measured rather than inferred from the fast 32×32 phantom.

## 0. Environment

On the SIGM server, use the existing activated `(SonBH-env)` environment.
Install only the base dependencies first; this does **not** replace an existing
PyTorch/CUDA installation:

```bash
cd /mnt/nvme0/home/sonbh/Code/SoDAR
python --version
python -m pip install -e .
python -m scripts.check_env
```

`torch` and `torchvision` are deliberately optional in the package's default
install. If they are **already installed**, do not overwrite them. If missing,
install a matching pair for the machine's NVIDIA driver using the official
[PyTorch installer](https://pytorch.org/get-started/locally/). Never assume
the server needs CPU-only wheels. After installation:

```bash
python -m scripts.check_env --require-torch
python -m pip install -e '.[segformer]'  # for SegFormer-B2 only
python -m scripts.check_env --require-torch --require-segformer
python -m pip check
```

An example **only if** the selected PyTorch/CUDA combination is 2.5.1/cu121:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Confirm `nvidia-smi`, `python -m scripts.check_env --require-torch` and the
official compatibility selector **before** choosing a wheel index. Python
3.10/3.11 are convenient environments for these tested library versions.
If Hugging Face's model cache is not writable, set `HF_HOME` to a writable
directory such as `/mnt/nvme0/home/sonbh/.cache/huggingface` before loading
pretrained MiT-B2.

### Existing repo vs new ZIP

Do not unpack the new ZIP over a modified old working tree. For an isolated
copy, from the directory containing the downloaded archive:

```bash
mkdir -p /mnt/nvme0/home/sonbh/Code/SoDAR-v0.2
unzip -q SoDAR-v0.2-complete.zip -d /mnt/nvme0/home/sonbh/Code/SoDAR-v0.2
cd /mnt/nvme0/home/sonbh/Code/SoDAR-v0.2/SoDAR
```

### SIGM-specific dataset paths (supplied by SonBH)

```text
/mnt/nvme0/home/sonbh/Dataset/OTU_2D/
  images/393.JPG
  annotations/393.PNG
```

Their **common parent** is the `data.root`. The included
`configs/sigm.yaml` sets this for you, without copying or changing the 850/
150/469 manifest. No need to pass `/images` or `/annotations` separately.
First check all 1,469 image/mask pairs (DDPM not needed):

```bash
python -m scripts.audit_data --config configs/sigm.yaml --real-only
```

Then run a short real-data I/O and training pilot:

```bash
python -m scripts.train --config configs/sigm.yaml \
  --experiment B0 --model unet --epochs 1 --seed 42
```

For S1–S5, the **generated DDPM folder is an additional input**, not the
original `OTU_2D/annotations` directory. Set it after locating the real
folder containing `images/` and `logits_npy/`:

```bash
python -m scripts.audit_data --config configs/sigm.yaml \
  --generated-root /ACTUAL/PATH/TO/generated_dataset_full

python -m scripts.train --config configs/sigm.yaml \
  --experiment S4 --model unet --epochs 1 \
  --generated-root /ACTUAL/PATH/TO/generated_dataset_full
```

Replace `/ACTUAL/PATH/TO/...` with the real generated-data directory. A
dummy path is intentionally left in `configs/sigm.yaml`; SoDAR does not
generate DDPM samples during the student-training step. Instead, run
`scripts.train_ddpm` and `scripts.sample_ddpm` **before** the student experiment.
`configs/sigm.yaml` requires `provenance.json` from a generator trained on
this exact train-850 list, preventing accidental use of the old overlapping
split or a mismatched generator checkpoint.

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
python -m scripts.smoke_test --phantom-root /tmp/sodar-phantom --with-torch
```

The smoke test exercises JSON split membership, image/mask loading, DDPM
`.npy` contracts, temperature conversion, binary soft KL, hard BCE and a
NumPy first-order meta-weight reference check. If PyTorch is available,
`--with-torch` additionally runs a **one-epoch actual U-Net S4 training** on
the phantom; this is not a clinical validation or a speed benchmark.

The old v0.1.0 smoke test used a strict finite-difference assertion that could
fail on a different NumPy environment **before PyTorch starts**. In v0.1.1,
the two-parameter reference uses explicit float64, a more stable difference
step and a diagnostic error reporting the numerical and analytical gradients.
If you still see `Phantom hypergradient mismatch`, run a fresh fixture:

```bash
python -m scripts.make_phantom --output /tmp/sodar-phantom-clean
python -m scripts.smoke_test --phantom-root /tmp/sodar-phantom-clean
```

The full `--with-torch` smoke still uses the default `/tmp/sodar-phantom`
path. The phantom has **no medical data** and is only an execution sanity test.

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
