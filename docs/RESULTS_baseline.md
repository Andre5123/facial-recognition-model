# Baseline training run results

## Configuration

`configs/baseline.yaml` (seed 42, unchanged from the values committed to the repo):

- Model: MobileFaceNet, 256-d embedding, randomly initialized
- Loss: ArcFace, margin 0.5, scale 64.0
- Optimizer: SGD, lr 0.1, momentum 0.9, weight decay 5e-4
- LR schedule: multistep, milestones [16, 24, 28], gamma 0.1
- Batch size: 128, 30 epochs, AMP (fp16) enabled
- Augmentation: horizontal flip only
- Data: CASIA-WebFace (Kaggle `webface-112x112` mirror), identity-disjoint
  split, seed 42 -- 10,043 train identities / 419,680 images, 529 unseen
  identities / 23,926 images, 47,017 held-out images of train identities
  (see `docs/DATASET.md`)

## Hardware / environment

Trained on Google Colab (free tier), Tesla T4 GPU, PyTorch 2.11.0+cu128.
Interrupted once by a Colab usage-limit cutoff after epoch 19; resumed on a
different Google account from the `epoch_19.pt` checkpoint (downloaded from
the first account's Drive, re-uploaded into the second session) -- see
chat history for the account-switch procedure. Training throughput was
~800-880 images/sec once warmed up; ~480-580s/epoch.

Note: this run was NOT done on the local AMD Radeon 880M setup documented
in `docs/ENVIRONMENT_SETUP.md` -- that GPU measured ~55-65 img/s for this
same model (see that doc / project chat history), which is why Colab's T4
was used for the actual training run instead.

## Final results (epoch 29 of 30, i.e. after the full schedule)

Verification evaluated via `src/eval.py`: 6,000 sampled pairs per split
(3,000 genuine / 3,000 impostor), 10-fold cross-validated accuracy (LFW-style
protocol -- threshold chosen on 9 folds, evaluated on the held-out fold).

| Split | Accuracy | ROC-AUC | TAR@FAR=0.01 | TAR@FAR=0.001 |
|---|---|---|---|---|
| Seen identity / unseen image | 89.57% ± 0.81% | 0.9451 | 0.6693 | 0.5123 |
| Unseen identity (never trained on) | 88.70% ± 1.35% | 0.9394 | 0.7007 | 0.3817 |

Seen and unseen accuracy tracked closely throughout training (e.g. 89.0%
vs 88.3% at epoch 24, 89.6% vs 88.7% at epoch 29), indicating the model is
learning a generalizable face representation rather than memorizing
training identities.

## Training trajectory (epoch-end eval, seen split)

| Epoch | Loss | Accuracy | ROC-AUC | Note |
|---|---|---|---|---|
| 0 | 47.42 | 75.78% | 0.855 | first epoch |
| 8 | 22.24 | 83.02% | 0.906 | pre-milestone plateau beginning |
| 15 | 22.24 | 81.88% | 0.895 | plateau, right before LR drop |
| 16 | 20.10 | 87.12% | 0.931 | **LR milestone 1** (0.1 -> 0.01) |
| 23 | 19.97 | 88.47% | 0.941 | pre-milestone-2 |
| 24 | 19.30 | 89.05% | 0.944 | **LR milestone 2** (0.01 -> 0.001) |
| 28 | 18.91 | 89.45% | 0.945 | **LR milestone 3** (0.001 -> 0.0001) |
| 29 | 18.89 | 89.57% | 0.945 | final |

Each scheduled LR decay produced a visible jump in both loss and
verification accuracy, most pronounced at the first drop (epoch 16: +5-6
points of accuracy). Between milestones, loss and accuracy both plateaued
(and accuracy occasionally dipped slightly) -- expected behavior for
margin-based softmax training at a fixed LR, not a sign of a broken run.

## Real LFW evaluation (added 2026-08-23)

Downloaded the standard aligned-LFW benchmark (Kaggle
`yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112` mirror, official
6,000-pair protocol via `lfw_ann.txt`) and evaluated the baseline `best.pt`
checkpoint against it using `src/eval.py --pairs-file` (added specifically
to consume this fixed-pairs format rather than our own sampled pairs):

| Benchmark | Accuracy | ROC-AUC | TAR@FAR=0.01 | TAR@FAR=0.001 |
|---|---|---|---|---|
| **LFW** (real, 6,000 official pairs) | **95.97% ± 1.17%** | **0.9918** | 0.9010 | 0.6560 |
| **AgeDB-30** (cross-age) | 82.25% ± 3.20% | 0.8972 | 0.2490 | 0.0640 |
| **CPLFW** (cross-pose) | 76.85% ± 2.47% | 0.8361 | 0.2597 | 0.0590 |
| Our seen-identity split (for reference) | 89.57% ± 0.81% | 0.9451 | 0.6693 | 0.5123 |
| Our unseen-identity split (for reference) | 88.70% ± 1.35% | 0.9394 | 0.7007 | 0.3817 |

AgeDB-30 and CPLFW are meaningfully harder than LFW (cross-age and
cross-pose verification specifically, vs. LFW's more typical same-era
photos), so the drop-off is expected, not a red flag -- published
reference models show the same pattern (e.g. a small MobileNetV1_0.25
trained on MS1MV2 gets 98.76% LFW but only 82.37% CPLFW). CFP-FP wasn't
evaluated -- not available in the downloaded benchmark mirror.

The real LFW number is meaningfully higher than either internal metric --
confirming those internal splits were a harder/different benchmark than
LFW, not a reliable proxy for it. Published results for this exact
combination (MobileFaceNet + ArcFace + CASIA-WebFace) report ~99.18% LFW
(AirFace, ICCVW 2019), using a larger batch size (512 vs our 128) and
embedding dimension (512 vs our 256) -- the gap to our 95.97% is plausibly
explained by those hyperparameter differences rather than anything
fundamentally wrong with the approach. 95.97% is a legitimate, solid
result for a from-scratch model on CASIA-WebFace, and much closer to the
project's aspirational ~99% target than the internal-split numbers alone
suggested.

## Honest comparison to project goals

The aspirational target in `context.md` is ~99% **LFW** verification
accuracy. With real LFW evaluation now done (above), the honest gap is
~99% target vs 95.97% actual -- a real but modest remaining gap, not the
large one the internal-split-only comparison implied.

## Checkpoints

`best.pt` / `final.pt` / per-epoch checkpoints were saved to Google Drive
under the training account used (`facial-recognition-checkpoints/`), not
committed to this repo (see `.gitignore`). Download a local copy before
relying on this run's weights for anything further.

## Next steps (not yet done)

- Consider closing the ~3-point gap to the published 99.18% reference by
  matching their batch size (512) and embedding dim (512), if pursued.
- Embedding space visualization (PCA/t-SNE/UMAP) and the unseen-identity
  demonstration described in context.md's "Future Work" section.
- Decide whether further training (more epochs, tuned augmentation) is
  worthwhile before or after LFW results are in hand.
