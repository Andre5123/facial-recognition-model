# Project Context: From-Scratch Face Embedding Model

## Goal

I want to build my own face-recognition system by training a face-embedding model from scratch.

The primary goal is to achieve the highest-quality face embeddings reasonably possible given my available hardware and the CASIA-WebFace training dataset. A target of approximately **99% LFW verification accuracy** would be excellent if achievable, but benchmark performance should be reported honestly rather than treated as a guaranteed target.

## Hardware

My GPU is:

**AMD Ryzen AI 9 365 with integrated Radeon 880M graphics.**

I want to use the GPU for PyTorch training rather than defaulting to CPU training.

Before setting up the environment, **inspect my operating system, Python version, AMD graphics driver, and available system RAM**, then consult the **current official PyTorch and AMD ROCm documentation** to determine the correct supported installation.

Do not assume that a generic ROCm installation is appropriate. Verify the current hardware/software compatibility first.

PyTorch's ROCm backend uses the `torch.cuda` API semantics, so GPU availability should ultimately be verified using PyTorch itself rather than assuming that successful installation means the GPU is actually being used.

## Dataset

I am currently downloading **CASIA-WebFace**.

Here is the link to where I am downloading it from kaggle: 
https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112?resource=download

Do not include the CASIA-WebFace images in the GitHub repository or otherwise redistribute the dataset.

When the dataset is available, inspect its actual format, contents, and applicable license/usage terms before deciding how it should be incorporated into the project.

The repository should contain code and documentation necessary to reproduce the training pipeline, but the dataset itself should remain external.

For example, the repository may contain:

* model architecture
* training code
* ArcFace implementation
* dataset loader
* preprocessing code
* evaluation code
* configuration files
* requirements/environment files
* documentation
* scripts for obtaining/preparing the dataset locally

but **not the CASIA-WebFace image files themselves**.

## Model

I want to train the face embedding model **from scratch**, meaning the model should use randomly initialized weights.

Recreate the **MobileFaceNet architecture** rather than downloading a pretrained face-recognition model.

Using a published architecture is acceptable; I am specifically interested in training the weights myself.

The model should produce a compact face embedding, initially targeting **256 dimensions**, unless experimentation indicates that another embedding size is preferable.

## Training

Use **ArcFace loss** as the primary training objective.

The intended architecture is approximately:

CASIA-WebFace
→ face images/preprocessing
→ MobileFaceNet with randomly initialized weights
→ 256-dimensional embedding
→ ArcFace classification head during training

The ArcFace classification head is a training mechanism and should not be part of the final embedding model used for inference.

After training, the desired inference pipeline is:

Face image
→ MobileFaceNet
→ normalized face embedding
→ cosine similarity / nearest-neighbor comparison

Do not use pretrained face-recognition weights.

## Identity Split / Generalization

A key goal is to determine whether the model can recognize **identities it never saw during training**.

Therefore, before training, create a proper **identity-disjoint split** of CASIA-WebFace.

Do not randomly split images from the same identities between training and validation/test sets.

Instead, reserve a subset of complete identities for validation/testing.

The identities in the held-out set must never appear in the training set.

This allows evaluation of whether the learned embedding generalizes to unseen people.

Document the exact split and random seed so that the experiment is reproducible.

## Identity and Image Generalization

Evaluation should test **two different types of generalization**:

### 1. Seen identities, unseen images

For identities that were present in the training set, hold out some images that were **never shown to the model during training**.

This tests whether the model can recognize a person from new photographs rather than simply memorizing the specific images used during training.

The same identity may therefore appear in both the training and evaluation sets, but **the specific evaluation images must never be used during training**.

### 2. Completely unseen identities

Reserve a subset of complete identities before training.

No images of these identities may appear anywhere in the training set.

This tests whether the learned embedding generalizes to people the model has never encountered.

The evaluation should therefore contain both:

* **Seen identity / unseen image** evaluations
* **Unseen identity / unseen image** evaluations

These should be reported separately where practical, since they measure different capabilities.

For all splits, avoid accidental image duplication or near-duplicate leakage between training and evaluation sets.

## Evaluation

After training, evaluate the model using:

1. A held-out identity-disjoint portion of CASIA-WebFace.
2. **LFW** when it is downloaded later.
3. Potentially:

   * AgeDB-30
   * CFP-FP
   * CPLFW

For LFW and the other external benchmarks, use the appropriate **verification protocol**, rather than treating them as ordinary closed-set classification datasets.

Report metrics such as:

* verification accuracy
* ROC curve / ROC-AUC where appropriate
* TAR at specified FAR values where appropriate
* cosine-similarity distributions for genuine vs. impostor pairs

Do not evaluate only on identities that appeared in training.

## Benchmark Target

The aspirational target is approximately:

**≥99% LFW verification accuracy**

if this is achievable with the chosen architecture, dataset, and training procedure.

However, do not artificially optimize for LFW at the expense of generalization.

Once performance approaches ~99% on LFW, place greater emphasis on harder benchmarks such as AgeDB-30, CFP-FP, and CPLFW.

Clearly distinguish between:

* training performance
* validation performance
* held-out identity performance
* external benchmark performance

## Training Experiments

Start with a sensible baseline rather than immediately trying to optimize every possible hyperparameter.

Record:

* batch size
* learning rate
* optimizer
* weight decay
* ArcFace margin
* ArcFace scale
* number of epochs
* learning-rate schedule
* image resolution
* embedding dimension
* augmentation strategy
* training/validation identities
* random seed
* GPU utilization
* training throughput
* training time

Because the Radeon 880M is an integrated GPU, prioritize a configuration that is realistically trainable on the available hardware.

If a particular configuration is too computationally expensive, reduce model/batch/training complexity rather than silently falling back to CPU.

## Reproducibility

The project should be reproducible.

Use configuration files or clearly defined command-line arguments for important hyperparameters.

Save:

* model checkpoints
* optimizer state when appropriate
* training configuration
* validation results
* best checkpoint
* final checkpoint

Do not commit large generated artifacts or the CASIA dataset to Git.

A `.gitignore` should exclude datasets, caches, checkpoints, and other large/generated files where appropriate.

## Future Work

After obtaining a working baseline, I would like to add:

### Embedding visualization

Visualize the learned embedding space using methods such as:

* PCA
* t-SNE
* UMAP

Show whether identities naturally form clusters.

### Unseen identity demonstrations

Demonstrate the model using identities that were **never present during training**.

For example:

1. Enroll a previously unseen person's face embedding.
2. Generate embeddings from additional images of that person.
3. Compare them using cosine similarity.
4. Compare them against embeddings from other unseen people.
5. Visualize the genuine/impostor similarity distributions.

This should demonstrate that the model is learning a generalizable face representation rather than simply memorizing the training identities.

## Important Constraints

* Do not use pretrained face-recognition weights.
* MobileFaceNet may be used as an architecture, but its weights should start randomly initialized.
* Do not include CASIA-WebFace in the GitHub repository.
* Verify the dataset's actual licensing/usage terms rather than assuming it is open-source.
* Do not redistribute CASIA-WebFace images.
* Do not claim benchmark performance that has not actually been measured.
* Keep training and evaluation identities strictly separated.
* Prefer official PyTorch/AMD documentation when configuring ROCm/PyTorch for the Radeon 880M.
* Record the experimental configuration and results so the final project is reproducible.

## Overall Objective

Build a technically sound, reproducible, from-scratch face embedding system using:

**CASIA-WebFace → randomly initialized MobileFaceNet → ArcFace → 256D normalized embeddings**

with the goal of achieving strong generalization to identities not seen during training and approaching **99% LFW verification accuracy** if the available hardware and dataset make that feasible.
