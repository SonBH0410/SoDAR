# OTU_2D protocol and integrity checklist

- [ ] The 1,469 manifest image paths are unique: 850/150/469.
- [ ] Patient IDs are verified to be disjoint externally (file stem alone does
      not prove this). Any duplicate/similar frame audit should use metadata.
- [ ] DDPM checkpoint provenance confirms training on 850 train *only*.
- [ ] `scripts.audit_generated` checks generated-pair files, raw two-channel
      logits, preview hard masks and `provenance.json`.
- [ ] SIGM's student config enforces generator train-record hash equal to the
      current manifest train-850 hash; synthetic labels are not double-softmaxed.
- [ ] Validate both real path styles, and image-mask pairing by basename.
- [ ] Every synthetic image has matching 2-channel `.npy` logits.
- [ ] The same 1,177 or other fixed number of pairs is used in S1–S5.
- [ ] Synthetic sample IDs and their order are deterministic across methods.
- [ ] No synthetic samples or augmentations enter validation/test.
- [ ] Real preprocessing, optimizer, updates, and seeds are matched.
- [ ] All training checkpoints are selected by real-validation macro-Dice.
- [ ] Real-only B1 (flip) and B2 (combined conventional) are independent
      baselines; synthetic groups have real augmentation `none`.
- [ ] Three predeclared seeds, mean ± sample SD, paired differences.
- [ ] Classwise image-level category metrics (0..7), macro and micro scores.
- [ ] DDPM training/sampling time and student training time reported.
- [ ] Test-469 used only after final method selection.

On OTU_2D, the normal ovary image category is **not** a negative/empty mask
class: its annotated region is the ovary. Read the official MMOTU task
definition before inventing class-conditional mask processing.
