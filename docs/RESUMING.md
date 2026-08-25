# Resuming or extending this project later

The baseline run is complete (see `docs/RESULTS_baseline.md`: 95.97% LFW),
and its final checkpoint is saved locally so future work doesn't depend on
any particular Google/Colab account staying accessible.

## What's saved where

- `checkpoints/baseline/epoch_29.pt` -- the completed baseline model
  (MobileFaceNet, 256-d embedding, epoch 29/30). Verified to load
  correctly and contains `model_state`, `head_state`, `optimizer_state`,
  and `scheduler_state`, so it supports a real resume (continuing
  optimizer momentum/LR schedule), not just inference from the weights.
- This one checkpoint is committed to the repo (~28MB, small enough to be
  fine in git -- see the `!checkpoints/baseline/epoch_29.pt` exception in
  `.gitignore`) so the demo works out-of-the-box. Every other checkpoint
  (other epochs, other runs) stays gitignored as usual.

## To continue training past epoch 30

Training needs a real GPU to be practical -- use
`notebooks/colab_train.ipynb` (Colab T4, ~800+ img/s), the same path used
for the actual baseline run.

Either way, upload/copy `epoch_29.pt` into that environment, then:

```bash
cd src
python train.py --config ../configs/baseline.yaml --resume path/to/epoch_29.pt --epochs 40
```

`--epochs 40` (or whatever higher number) is required -- resuming with the
original `epochs: 30` would immediately exit, since epoch 29 already
satisfies that config's stopping point. Note the LR schedule
(`lr_milestones: [16, 24, 28]`) won't have any more scheduled drops past
epoch 28, so extending training this way trains longer at the final
(smallest) LR rather than following a newly-extended schedule -- edit
`configs/baseline.yaml`'s milestones first if you want a real extended
schedule rather than just more epochs at a flat LR.

## Pushing past 95.97% LFW

Directions worth trying to chase a higher number, not pursued to a
conclusion here:

- **Pretrained backbones** -- the current direction (decided 2026-08-23,
  after this from-scratch baseline plateaued): swap in an
  ImageNet-pretrained MobileNetV2/V3 or ResNet backbone in place of the
  randomly-initialized MobileFaceNet. The project's original "train from
  scratch, no pretrained weights" constraint (`context.md`) no longer
  applies -- see that file's "Amendment" section.
- **Matching a published recipe more closely** -- larger batch size and
  embedding dimension than this baseline's 128/256 are the most likely
  levers, based on other reported results for this architecture/dataset
  combination, though not verified here.

## Re-evaluating this checkpoint

Internal splits (fast, no extra downloads needed beyond `data/splits/`,
regenerate via `scripts/make_identity_split.py` if needed):

```bash
cd src
python eval.py --config ../configs/baseline.yaml --checkpoint ../checkpoints/baseline/epoch_29.pt
```

Real external benchmarks (LFW/AgeDB-30/CPLFW) require downloading the
Kaggle `yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112` mirror first
(not stored locally, only used inside Colab so far):

```bash
cd src
python eval.py --config ../configs/baseline.yaml --checkpoint ../checkpoints/baseline/epoch_29.pt \
    --split none --pairs-file <path>/lfw_ann.txt --pairs-root <path> --pairs-name LFW
```
