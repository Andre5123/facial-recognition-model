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

## Honest comparison to project goals

The aspirational target in `context.md` is ~99% **LFW** verification
accuracy. The ~89% figures above are on held-out CASIA-WebFace pairs, not
LFW -- a different (likely easier, in-domain) benchmark, since LFW has not
yet been downloaded/evaluated. This baseline result should not be
interpreted as "89% of the way to the LFW target"; LFW evaluation is
still outstanding.

## Checkpoints

`best.pt` / `final.pt` / per-epoch checkpoints were saved to Google Drive
under the training account used (`facial-recognition-checkpoints/`), not
committed to this repo (see `.gitignore`). Download a local copy before
relying on this run's weights for anything further.

## Next steps (not yet done)

- Evaluate on actual LFW once downloaded, using the proper verification
  protocol (context.md's stated benchmark target).
- AgeDB-30 / CFP-FP / CPLFW evaluation.
- Embedding space visualization (PCA/t-SNE/UMAP) and the unseen-identity
  demonstration described in context.md's "Future Work" section.
- Decide whether further training (more epochs, tuned augmentation) is
  worthwhile before or after LFW results are in hand.
