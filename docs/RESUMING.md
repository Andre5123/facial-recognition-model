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
- This file is gitignored (per `checkpoints/` in `.gitignore`) -- it only
  exists on this machine. Back it up yourself if you want extra safety
  (it's not in git or in any cloud storage other than the original
  training account's Drive, which this local copy is now independent of).

## To continue training past epoch 30

Training needs a real GPU to be practical -- see `docs/ENVIRONMENT_SETUP.md`
for the local ROCm setup (works, but slow: ~55-65 img/s) or
`notebooks/colab_train.ipynb` for the much faster Colab T4 path (~800+
img/s) used for the actual baseline run.

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

## Other experiments already built but not run to completion

- **`configs/mobilenet_pretrained.yaml`** -- ImageNet-pretrained MobileNetV2
  fine-tuned with ArcFace, as an alternative to from-scratch MobileFaceNet.
  Was run partway (stopped at epoch 8) and was behind the baseline at every
  checkpoint observed, but never reached the first LR milestone (epoch 16)
  where the baseline got its biggest jump -- inconclusive, not disproven.
  Notebook section "8b" runs this.
- **`configs/airface_recipe.yaml`** -- batch size 512 / embedding dim 512
  instead of the baseline's 128/256, an experiment aimed at closing the
  gap to published ~99% LFW results. Built and smoke-tested locally but
  never actually run for real -- see chat history from 2026-08-23/24 for
  why this specific change's justification turned out to be shakier than
  first thought (the "same setup gets 99%" comparison didn't hold up under
  scrutiny). Worth treating as a speculative experiment, not a confirmed fix.

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
