# Interactive demo

`src/app.py` is a small Gradio app for trying the trained model directly --
verification (1:1: "are these the same person?") and recognition (1:N:
"who is this, out of the people I've enrolled?", with an open-set mode
that can say "not recognized" instead of forcing a match).

## Why this exists, and its real limitation

The model was trained entirely on pre-aligned 112x112 CASIA-WebFace crops
(see `docs/DATASET.md`). A raw photo or webcam frame isn't aligned like
that, so the demo runs every image through a face detector/aligner first
(`src/face_detect.py`, MediaPipe's BlazeFace) before handing anything to
the model. That alignment step is a best-effort approximation of whatever
process produced the original training crops -- it won't be pixel-
identical, so expect real-world accuracy on casual photos/webcam captures
to run somewhat below the benchmark numbers in `docs/RESULTS_baseline.md`,
which were measured on properly pre-aligned data.

Every mode shows the actual aligned crop it fed to the model, specifically
so this gap is visible rather than hidden -- if a result looks wrong,
check the crop first.

## Running it

```bash
pip install -r requirements/demo.txt   # gradio + mediapipe, on top of the usual environment
cd src
python app.py --config ../configs/baseline.yaml --checkpoint ../checkpoints/baseline/epoch_29.pt
```

Opens at http://127.0.0.1:7860. The face-detector model (224KB) auto-
downloads on first run to `~/.cache/facial-recognition-demo/`.

Pass `--share` for a temporary public Gradio link if you want to let
someone else try it remotely.

## Modes

- **Verification**: upload or webcam-capture two photos, get a same/
  different verdict plus the raw cosine similarity, against an adjustable
  threshold.
- **Recognition**: enroll one or more people (photo + name, repeatable per
  person for multiple reference images), then identify a new photo against
  the gallery. The gallery lives only in the current browser session (a
  Gradio `State`, not saved to disk) and starts empty each run.
  - **Closed-set** (open-set unchecked): always returns the closest match,
    even if it's a poor one.
  - **Open-set** (default): returns "not recognized" if the best match's
    similarity falls below the threshold -- the more realistic mode, since
    a real gallery usually doesn't contain everyone who might show up.

## Default threshold

The default (0.95) comes from the genuine/impostor similarity stats
measured on real unseen identities (`docs/VISUALIZATIONS.md`: genuine mean
0.972, impostor mean 0.931) -- roughly the midpoint. It's a starting point,
not a calibrated operating point; adjust it live via the slider.
