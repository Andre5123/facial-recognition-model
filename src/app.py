"""Interactive demo: verification (1:1) and recognition (1:N, open/closed-set)
on top of the trained face embedding model.

Every mode shows the actual 112x112 aligned crop that was fed to the model,
not just the final result -- the model was trained entirely on pre-aligned
crops (see docs/DATASET.md), so what the detector/aligner produces from an
arbitrary photo or webcam frame directly explains the quality of the result.

Usage:
    python src/app.py --config configs/baseline.yaml --checkpoint ../checkpoints/baseline/epoch_29.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image

from data.dataset import build_transform
from eval import build_model
from face_detect import FaceAligner, NoFaceDetected

# Default similarity threshold for open-set rejection. Chosen from the
# genuine/impostor similarity stats measured in docs/VISUALIZATIONS.md
# (genuine mean 0.972, impostor mean 0.931 on unseen identities) -- roughly
# the midpoint, adjustable live via the UI slider since the right value
# depends on how strict you want false-accepts to be.
DEFAULT_THRESHOLD = 0.95


class Engine:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.device = device
        self.model = build_model(self.cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"loaded checkpoint from epoch {ckpt['epoch']}: {checkpoint_path}")

        self.transform = build_transform(self.cfg["data"]["image_size"], train=False, horizontal_flip=False)
        self.aligner = FaceAligner()

    @torch.no_grad()
    def embed(self, image: Image.Image):
        """Returns (embedding vector, aligned 112x112 crop) or raises NoFaceDetected."""
        aligned = self.aligner.align(image)
        tensor = self.transform(aligned).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().numpy()[0]
        return embedding, aligned


engine: Engine | None = None


def _no_face_result(*extra_outputs):
    return ("No face detected in this image -- try a clearer, more front-facing photo.", None, *extra_outputs)


def do_verify(image_a, image_b, threshold):
    if image_a is None or image_b is None:
        return "Provide two images.", None, None
    try:
        emb_a, crop_a = engine.embed(image_a)
    except NoFaceDetected:
        return "No face detected in the first image.", None, None
    try:
        emb_b, crop_b = engine.embed(image_b)
    except NoFaceDetected:
        return "No face detected in the second image.", crop_a, None

    similarity = float(np.dot(emb_a, emb_b))
    same = similarity >= threshold
    verdict = "SAME PERSON" if same else "DIFFERENT PEOPLE"
    result = f"{verdict}  (cosine similarity: {similarity:.4f}, threshold: {threshold:.4f})"
    return result, crop_a, crop_b


def do_enroll(image, name, gallery):
    gallery = dict(gallery or {})
    if image is None or not name or not name.strip():
        return gallery, _gallery_summary(gallery), None, "Provide both an image and a name."
    try:
        emb, crop = engine.embed(image)
    except NoFaceDetected:
        return gallery, _gallery_summary(gallery), None, "No face detected -- try a clearer photo."

    name = name.strip()
    gallery.setdefault(name, [])
    gallery[name].append(emb.tolist())
    return gallery, _gallery_summary(gallery), crop, f"Enrolled '{name}' ({len(gallery[name])} reference image(s))."


def do_clear_gallery():
    return {}, _gallery_summary({}), "Gallery cleared."


def _gallery_summary(gallery: dict) -> str:
    if not gallery:
        return "(no one enrolled yet)"
    return "\n".join(f"- {name}: {len(embs)} reference image(s)" for name, embs in gallery.items())


def do_identify(image, gallery, open_set, threshold):
    if image is None:
        return "Provide an image.", None
    if not gallery:
        return "Gallery is empty -- enroll at least one person first.", None
    try:
        probe_emb, crop = engine.embed(image)
    except NoFaceDetected:
        return _no_face_result_identify()

    best_name, best_sim = None, -1.0
    for name, embs in gallery.items():
        for emb in embs:
            sim = float(np.dot(probe_emb, np.array(emb)))
            if sim > best_sim:
                best_sim, best_name = sim, name

    if open_set and best_sim < threshold:
        result = f"NOT RECOGNIZED  (closest was '{best_name}' at {best_sim:.4f}, below threshold {threshold:.4f})"
    else:
        result = f"MATCH: {best_name}  (cosine similarity: {best_sim:.4f})"
    return result, crop


def _no_face_result_identify():
    return "No face detected in this image -- try a clearer, more front-facing photo.", None


def build_ui():
    # Gradio already deletes cached uploads/webcam captures when the browser
    # tab closes cleanly, but that relies on the close event actually firing.
    # delete_cache adds a guarantee that doesn't depend on that: every 5
    # minutes, delete any cached file older than 5 minutes, regardless of
    # how the session ended (crash, Ctrl+C, force-closed tab, etc.).
    with gr.Blocks(title="Face Recognition Demo", delete_cache=(300, 300)) as demo:
        gr.Markdown(
            "# Face Recognition Demo\n"
            "From-scratch MobileFaceNet + ArcFace, 95.97% LFW verification accuracy "
            "(see `docs/RESULTS_baseline.md`). Every mode shows the actual aligned "
            "112x112 crop the model sees -- that's what it was trained on, and "
            "alignment quality directly affects the result."
        )

        with gr.Tab("Verification (1:1)"):
            gr.Markdown("Are these two photos the same person?")
            with gr.Row():
                with gr.Column():
                    img_a = gr.Image(label="Person A", sources=["upload", "webcam"], type="pil")
                    crop_a = gr.Image(label="Aligned crop fed to model", interactive=False)
                with gr.Column():
                    img_b = gr.Image(label="Person B", sources=["upload", "webcam"], type="pil")
                    crop_b = gr.Image(label="Aligned crop fed to model", interactive=False)
            verify_threshold = gr.Slider(0.80, 1.00, value=DEFAULT_THRESHOLD, step=0.005, label="Same-person threshold (cosine similarity)")
            verify_btn = gr.Button("Compare", variant="primary")
            verify_result = gr.Textbox(label="Result", interactive=False)
            verify_btn.click(do_verify, [img_a, img_b, verify_threshold], [verify_result, crop_a, crop_b])

        with gr.Tab("Recognition (1:N)"):
            gr.Markdown("Enroll one or more people, then identify a new photo against the gallery.")
            gallery_state = gr.State({})

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Enroll")
                    enroll_img = gr.Image(label="Photo", sources=["upload", "webcam"], type="pil")
                    enroll_crop = gr.Image(label="Aligned crop fed to model", interactive=False)
                    enroll_name = gr.Textbox(label="Name")
                    enroll_btn = gr.Button("Enroll")
                    enroll_status = gr.Textbox(label="Status", interactive=False)
                    clear_btn = gr.Button("Clear gallery")
                    gallery_display = gr.Textbox(label="Enrolled", value="(no one enrolled yet)", interactive=False)

                with gr.Column():
                    gr.Markdown("### Identify")
                    probe_img = gr.Image(label="Photo to identify", sources=["upload", "webcam"], type="pil")
                    probe_crop = gr.Image(label="Aligned crop fed to model", interactive=False)
                    open_set = gr.Checkbox(
                        value=True,
                        label="Open-set (reject as \"not recognized\" if below threshold; uncheck for closed-set: always return the best match)",
                    )
                    id_threshold = gr.Slider(0.80, 1.00, value=DEFAULT_THRESHOLD, step=0.005, label="Recognition threshold (open-set only)")
                    identify_btn = gr.Button("Identify", variant="primary")
                    identify_result = gr.Textbox(label="Result", interactive=False)

            enroll_btn.click(
                do_enroll,
                [enroll_img, enroll_name, gallery_state],
                [gallery_state, gallery_display, enroll_crop, enroll_status],
            )
            clear_btn.click(do_clear_gallery, [], [gallery_state, gallery_display, enroll_status])
            identify_btn.click(
                do_identify,
                [probe_img, gallery_state, open_set, id_threshold],
                [identify_result, probe_crop],
            )

    return demo


def main():
    global engine
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../checkpoints/baseline/epoch_29.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = parser.parse_args()

    engine = Engine(args.config, args.checkpoint, args.device)
    demo = build_ui()
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
