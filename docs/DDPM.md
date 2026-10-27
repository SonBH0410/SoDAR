# Joint DDPM generator — implementation notes and research safeguards

## What was recovered

The `us-dakter (1).ipynb` attachment still contained U-Net, diffusion,
training and DDIM sampling source, but **no saved weights or generated outputs**.
SoDAR v0.2 is a from-scratch, PyTorch-only reimplementation of its five-channel
joint diffusion design, **not** a byte-for-byte reproduction and **not**
compatible with its old `model-35.pt` state dictionary.

The attached notebook's final sampling setup was 256×256, 1000 diffusion
timesteps, 200 DDIM sampling steps, `objective='pred_x0'`, cosine schedule,
`snr_balance_b=0.684`, 1177 generated pairs. These are the v0.2 defaults.
The U-Net preserves its main pattern: time-embedded residual blocks, RMSNorm,
linear attention at high resolutions, full bottleneck attention and a combined
3-channel image + 2-channel mask output. Implementation choices (attention
kernel, weight names, data loader) are modernized for PyTorch >=2.2.

## Joint training objective

For image \(I\in[0,1]^{3\times H\times W}\) and binary mask \(M\), define
\(I_0=2I-1\), \(Y=\operatorname{onehot}(M)\), and
\(Y_0=b(2Y-1)\). The 5-channel DDPM state is
\(X_0=[I_0,Y_0]\), \(b=0.684\) by default. A cosine noise schedule gives

\[
X_t=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\varepsilon.
\]

The shared U-Net \(f_\phi(X_t,t)=[\widehat I_0,Z_M]\) outputs image prediction
and two **raw segmentation logits**. The training objective is

\[
\mathcal L_{DDPM}=\|\widehat I_0-I_0\|_2^2+
 \mathrm{CE}(Z_M,M).
\]

During DDIM, the mask component of the estimated \(X_0\) is reconstructed as
\(b(2\operatorname{softmax}(Z_M)-1)\), while *unsoftmaxed* \(Z_M\) is exported
as student supervision. The `snr_balance_b` scalar scales the mask state;
it **does not** weight the image-MSE/mask-CE loss. The old notebook calculated
a theoretical SNR `loss_weight` but did **not** multiply its final loss by it;
SoDAR intentionally mirrors that behavior and does not claim min-SNR loss
weighting. No teacher calibration or uncertainty interpretation is presumed.

## Explicit split and provenance contract

- `JointDiffusionDataset` loads ONLY `manifest["train"]` for pixel training.
  It neither samples nor evaluates the 150 validation or 469 test images.
- A full path audit checks original files before training; this is NOT proof
  of patient-disjointness without patient IDs. The user reports fixing the
  original patient overlap. Do not change the supplied JSON without review.
- Checkpoints store the exact ordered train-member SHA-256 digest, settings,
  model, EMA, optimizer, GradScaler and random state; `--resume` refuses a
  changed train list or model/training configuration.
- Sampling records checkpoint SHA-256, step, train-list digest, seed, sample
  count, image size and logits format in `provenance.json`. Student training
  using `configs/sigm.yaml` REQUIRES the provenance to match current train-850.
- A manifest digest does not hash image pixels or prove patient identities.
  Document the original raw dataset version and patient-split audit separately.

## Practical limits

- Image quality / anatomical plausibility is **not** established by passing
  phantom or contract tests. Visually inspect/generated-case review and report
  quality/diversity quantitatively before claiming augmented segmentation.
- A checkpoint from 2 or 1000 DDPM steps is an **engineering pilot**, not a
  trained generator. The default 100000 steps mirrors the notebook's first
  training budget; it is not a proven optimum for the corrected split.
- `--resume` restores model, optimizer, scaler and random states, but the
  shuffled DataLoader iterator restarts; it is NOT bitwise-identical training
  continuation at microbatch level. Avoid interruption for exact reruns.
- EMA is updated at `ema_update_every`; sampling always uses EMA weights.
- Image generation is compute-heavy. An RTX Ada card does not guarantee that
  a particular batch size or 100k-step runtime will fit. Use a 100–1000 step
  pilot, watch memory with `nvidia-smi`, and start from small microbatches.
- Generator training defaults to no Flip. Changing `data.flip_probability`
  would be a separate generator-level augmentation experiment. All S student
  branches continue to use **no conventional Flip augmentation**.
- The same synthetic folder, number of samples, generator checkpoint and
  train budget must be held fixed across S1–S5. Neither pass of `audit` nor a
  generated foreground ratio is evidence of accurate annotation.

## File structure and outputs

```text
runs/ddpm_full/
  audit.json, config.json, history.jsonl
  step-0002000.pt, step-0004000.pt, ..., latest.pt, latest.json

Dataset/SoDAR-DDPM/
  images/000000.jpg
  logits_npy/000000.npy       # float32 (2,256,256), raw logits
  hard_masks/000000.png       # preview only, NOT student S2 target
  provenance.json             # complete only after all samples exported
```

## Phantom verification

```bash
python -m scripts.smoke_ddpm --root /tmp/sodar-ddpm-demo
python -m unittest discover -s tests -v
```

The end-to-end phantom test actually trains the DDPM, checkpoints, samples,
checks soft logits, trains S4/U-Net and evaluates a phantom checkpoint. A zero
phantom Dice after one student epoch is expected and is not clinical evidence.
