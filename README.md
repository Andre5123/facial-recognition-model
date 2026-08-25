# Face Recognition

A face embedding model (used for a facial recognition system) 
trained on CASIA-WebFace, plus an interactive demo
for trying it on your own photos or webcam. The current baseline below was
trained from scratch (randomly initialized weights, no pretrained
face-recognition or generic-pretrained weights); pretrained-backbone
experiments (e.g. an ImageNet-pretrained MobileNet/ResNet) are the current
direction for pushing past it.

**Pipeline:** CASIA-WebFace &rarr; MobileFaceNet &rarr; ArcFace loss &rarr;
256-dimensional L2-normalized face embeddings &rarr; cosine similarity for
verification/recognition.

## Results

| Benchmark | Accuracy | ROC-AUC |
|---|---|---|
| **LFW** (real, official 6,000-pair protocol) | **95.97% ± 1.17%** | 0.9918 |
| AgeDB-30 (cross-age) | 82.25% ± 3.20% | 0.8972 |
| CPLFW (cross-pose) | 76.85% ± 2.47% | 0.8361 |
| Unseen-identity split (held-out CASIA-WebFace identities) | 88.70% ± 1.35% | 0.9394 |

Full methodology, training trajectory, and honest discussion of these
numbers in [`docs/RESULTS_baseline.md`](docs/RESULTS_baseline.md). The
unseen-identity numbers matter as much as LFW here: they measure whether
the model generalizes to people it never trained on, not just how well it
memorized training identities -- see
[`docs/VISUALIZATIONS.md`](docs/VISUALIZATIONS.md) for a t-SNE embedding
plot and similarity-distribution demonstration of exactly that.

## Try it: interactive demo

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or a CUDA build for GPU inference
pip install -r requirements/base.txt
pip install -r requirements/demo.txt
cd src
python app.py --config ../configs/baseline.yaml --checkpoint ../checkpoints/baseline/epoch_29.pt
```

Opens a local Gradio app at http://127.0.0.1:7860 with two modes:

- **Verification (1:1)** -- upload or webcam-capture two photos, get a
  same/different verdict and the raw cosine similarity.
- **Recognition (1:N)** -- enroll a gallery of people (one photo at a time,
  or bulk-enroll a folder), then identify a new photo against it, in
  open-set (can say "not recognized") or closed-set mode.

See [`docs/DEMO.md`](docs/DEMO.md) for details, including how to fetch a
ready-made test dataset if you don't have photos on hand. The command
above points at `checkpoints/baseline/epoch_29.pt`, the trained checkpoint
committed in this repo (~28MB) -- no separate download needed.

## Project structure

```
src/
  models/mobilefacenet.py    MobileFaceNet architecture
  losses/arcface.py          ArcFace additive angular margin loss
  data/dataset.py            dataset loading + augmentation
  train.py                   training loop
  eval.py                    verification evaluation (internal splits + external benchmarks)
  visualize_embeddings.py    PCA/t-SNE embedding space plots
  unseen_identity_demo.py    unseen-identity genuine/impostor similarity demo
  face_detect.py             face detection/alignment for the demo (MediaPipe)
  app.py                     interactive Gradio demo
scripts/
  make_identity_split.py     builds the identity-disjoint train/val/test split
  inspect_dataset.py         verifies raw dataset format/stats
  fetch_demo_test_photos.py  downloads a small LFW subset for trying the demo
configs/baseline.yaml        training hyperparameters
notebooks/colab_train.ipynb  Colab notebook used for the actual (GPU) training run
docs/                        results, dataset, demo, and visualization writeups
```

## Reproducing the training run

1. **Environment**: the baseline run used
   [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) on a free
   Colab T4 GPU (~800-880 img/s) -- see `requirements/base.txt` for the
   non-training-hardware-specific dependencies.
2. **Dataset**: [`docs/DATASET.md`](docs/DATASET.md) covers CASIA-WebFace's
   source, license (non-commercial research use only -- not committed to
   this repo), and format.
3. **Train**:
   ```bash
   cd src
   python train.py --config ../configs/baseline.yaml
   ```
4. **Evaluate**:
   ```bash
   python eval.py --config ../configs/baseline.yaml --checkpoint ../checkpoints/baseline/best.pt
   ```
   See [`docs/RESULTS_baseline.md`](docs/RESULTS_baseline.md) for
   evaluating against real external benchmarks (LFW/AgeDB-30/CPLFW).

## License note

This project's own code is free to use. The CASIA-WebFace training data
is **not** included and is licensed for non-commercial research/
educational use only -- see [`docs/DATASET.md`](docs/DATASET.md) before
using this pipeline with that dataset for anything else.
