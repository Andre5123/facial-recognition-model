# Interactive demo

`src/app.py` is a small Gradio app for trying the trained model directly --
verification (1:1: "are these the same person?") and recognition (1:N:
"who is this, out of the people I've enrolled?", with an open-set mode
that can say "not recognized" instead of forcing a match).

## Preprocessing

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

## Bulk enrolling a folder of people

Enrolling one photo at a time through the UI doesn't scale if you want to
test with more than a couple of people. The "Bulk enroll from a folder"
box takes a local folder path shaped like:

```
some_folder/
  Alice/
    photo1.jpg
    photo2.jpg
  Bob/
    photo1.jpg
```

One subfolder per person (folder name = enrolled name), any number of
photos inside. This is the same one-folder-per-identity convention used
throughout the rest of this project (`webface_112x112/`, `data/splits/`,
`scripts/make_identity_split.py`) -- deliberately just the one simple,
already-established format rather than trying to auto-detect various
dataset layouts. Point it at a folder you've organized yourself (your own
photos, or a small hand-picked subset of any dataset reorganized into this
shape) and it enrolls everyone in one click, skipping any individual photo
where no face is detected rather than failing the whole batch.

Don't have a folder of photos handy? `scripts/fetch_demo_test_photos.py`
downloads a small, well-photographed subset of LFW and saves it already in
this exact folder-per-person shape:

```bash
python scripts/fetch_demo_test_photos.py --output-dir demo_test_photos
```

Then point the bulk-enroll box at `demo_test_photos`. These are real faces
from the public LFW benchmark and none of them were in the model's
training data (CASIA-WebFace) -- a genuine test of generalization, not a
memorization check.

## Default threshold

The default (0.95) comes from the genuine/impostor similarity stats
measured on real unseen identities (`docs/VISUALIZATIONS.md`: genuine mean
0.972, impostor mean 0.931) -- roughly the midpoint. It's a starting point,
not a calibrated operating point; adjust it live via the slider.
