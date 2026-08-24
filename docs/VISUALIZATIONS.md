# Embedding space visualization and unseen-identity demonstration

Generated from `checkpoints/baseline/epoch_29.pt` (the completed baseline
run, see `docs/RESULTS_baseline.md`), per context.md's "Future Work"
section.

## Embedding space clustering

`src/visualize_embeddings.py` samples 10 seen identities (from
`val_seen.csv`) and 10 unseen identities the model never trained on (from
`test_unseen.csv`), 8 images each, embeds them, and projects to 2D.

- `embeddings_pca.png` -- PCA (linear projection). Noisier, as expected
  for a 2D linear projection of a 256-d space, but still shows real
  identity grouping.
- `embeddings_tsne.png` -- t-SNE (nonlinear). Clean, tight, well-separated
  clusters -- and critically, the unseen identities (triangles) cluster
  just as distinctly as the seen ones (circles), visual confirmation the
  model learned a generalizable representation rather than memorizing
  training identities.

Regenerate: `python src/visualize_embeddings.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/epoch_29.pt --splits-dir data/splits --output-dir docs`

## Unseen-identity verification demonstration

`src/unseen_identity_demo.py` enrolls 30 identities never seen during
training, using one reference image each, then compares that reference
against the person's other photos (genuine) and against other unseen
people's photos (impostor).

Result (`unseen_identity_similarity.png`): genuine mean cosine similarity
0.9716 (std 0.0144) vs. impostor mean 0.9309 (std 0.0130) -- a real
but visually narrow-looking gap on a raw [-1, 1] cosine scale. Note: the
plot's x-axis is zoomed to the actual data range (~0.88-1.0), not the
full [-1, 1] range -- ArcFace-trained embeddings characteristically
occupy a narrow high-similarity band on the hypersphere (sometimes called
the "narrow cone effect"), so a full-range plot would visually compress
this real separation into what looks like near-total overlap even though
it isn't. Zoomed in, the two distributions are cleanly bimodal with only
a small overlap region -- consistent with the ~96-97% verification
accuracy measured elsewhere.

Regenerate: `python src/unseen_identity_demo.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/epoch_29.pt --splits-dir data/splits --output-dir docs`
