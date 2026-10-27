# SoDAR method (implementation specification)

## Frozen generator and paired logits

The pretrained DAKTer-style joint diffusion generator outputs generated
ultrasound image \(x_s\) and two-channel mask logits \(z_g\). This repository
does **not** train DDPM. It reads immutable outputs from the generator. Only
850 training patients/images may have contributed to generator training.

Binary hard ground truth \(y_r\) accompanies real images \(x_r\). The eight
values in the manifest's `label` column are image-level categories for
stratified evaluation, **not** eight foreground segmentation classes.

## Real loss

For a one-logit binary student: \(p_r = \sigma(z_\theta(x_r))\).

\[
 L_r = \mathrm{BCEWithLogits}(z_\theta(x_r),y_r)
      + \beta_d\,[1-\mathrm{Dice}_{soft}(p_r,y_r)].
\]

All configurations use the same real loss and the same resizing/normalization.

## Synthetic loss: controlled hard / soft comparison

Teacher distribution with the **same** temperature as student:

\[
q_g^T=\operatorname{softmax}(z_g/T), \qquad
p_s^T=[1-\sigma(z_\theta(x_s)/T),\sigma(z_\theta(x_s)/T)].
\]

Primary soft loss: \(L_s = T^2 D_{KL}(q_g^T\|p_s^T)\), averaged over pixels
and samples. The teacher is a frozen generator, **not** a separately trained
segmentation teacher. Soft targets are not claimed to be calibrated clinical
uncertainty without independent evidence. Do not apply hard argmax Dice to S2.

Hard control S1: \(\hat y_s=\arg\max_c z_{g,c}\); compute the same hard BCE
+ Dice as on real images. All S experiments use identical \(x_s,z_g\).

Optional soft loss ablations: soft BCE without temperature, soft BCE + soft
Dice. They are *not* silently substituted for the primary KL objective.

## Weight learning

\[L(\theta,\lambda)=L_r(\theta)+\lambda L_s(\theta).\]

Bounded weight \(\lambda=\lambda_{max}\sigma(\alpha)\).

**S3, naive learnable:** ordinary SGD/Adam on \(\alpha\) and \(\theta\).
Because \(L_s\ge0\), \(\partial_\alpha L=\lambda_{max}\sigma(\alpha)
(1-\sigma(\alpha))L_s\ge0\); the weight tends to zero. It is an explicit
negative control, **not** the SoDAR proposal.

**S4, first-order meta-learned:** on each `meta_every`-th training iteration,
compute inner loss gradients \(g_r=\nabla_\theta L_r\) and
\(g_s=\nabla_\theta L_s\) and perform a virtual update

\[
\tilde\theta=\theta-\eta_\theta(g_r+\lambda g_s).
\]

Optimize \(\alpha\) through real-only loss
\(L_{meta-real}(\tilde\theta)\) on **another real minibatch**. The code uses
the first-order approximation (detached \(g_r,g_s\)) to avoid an expensive
full second-order unroll; this still propagates a hypergradient from the
virtual network to \(\lambda\). The alternate batch is sampled from the
train-850 distribution rather than the official 150 validation examples, so
it is a *within-training meta signal*, not an unbiased held-out estimator.
For stronger inference, pre-register a fixed patient-disjoint inner-meta
subset from train and compare all methods under the same effective label
budget; keep official validation untouched.

**S5, exploratory:** multiply each synthetic KL by
\(r_j=\max(r_{min},\exp[-\gamma\bar H(q_j)])\) and normalize by sum of
sample weights. This is a sensitivity analysis, not proof that entropy equals
label correctness. The scalar lambda remains meta-learned.

## Hypothesis and falsification

H1: S2 outperforms S1 on identical pairs (benefit of soft information).

H2: S4 outperforms S2 and strong real-only baselines (benefit of adaptive
synthetic weighting).

H3: S4 learns a nontrivial \(\lambda(t)\) schedule rather than merely
rejecting all synthetic supervision.

If these fail across predeclared seeds, do not claim positive augmentation.
Report negative-transfer analysis instead or revisit the generator labels.
