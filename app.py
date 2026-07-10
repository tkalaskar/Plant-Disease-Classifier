import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plant_disease_classifier.data import build_transforms
from plant_disease_classifier.models import build_model


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def choose_device() -> torch.device:
    if os.getenv("USE_CUDA", "0") != "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = choose_device()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def get_grad_cam_target_layers(model, model_name: str):
    model_name = model_name.lower()
    if model_name in {"efficientnet_b0", "mobilenet_v3_large"}:
        return [model.features[-1]]
    if model_name == "resnet50":
        return [model.layer4[-1]]
    raise ValueError(f"Unsupported model_name for Grad-CAM: {model_name}")


def load_model():
    if not MODEL_PATH.exists():
        return None, None, f"Missing checkpoint: {MODEL_PATH}"

    checkpoint = load_checkpoint(MODEL_PATH)
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    class_names = checkpoint["class_names"]

    model = build_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    metadata = {
        "model_name": model_name,
        "class_names": class_names,
        "image_size": checkpoint.get("image_size", 224),
        "target_layers": get_grad_cam_target_layers(model, model_name),
    }
    return model, metadata, None


MODEL, METADATA, LOAD_ERROR = load_model()


def predict(image: Image.Image):
    if image is None:
        return "Upload a leaf image first.", 0.0, None

    if LOAD_ERROR:
        message = (
            f"{LOAD_ERROR}\n\n"
            "Train the model first, then upload the checkpoint to this Space as "
            "`models/best_model.pth`, or set the `MODEL_PATH` environment variable."
        )
        return message, 0.0, None

    image = image.convert("RGB")
    image_size = METADATA["image_size"]
    _, eval_transform = build_transforms(image_size)
    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_index = probabilities.max(dim=0)

    predicted_index = int(predicted_index.item())
    predicted_class = METADATA["class_names"][predicted_index]
    confidence_percent = float(confidence.item() * 100)
    overlay = build_grad_cam_overlay(image, input_tensor, predicted_index)

    return predicted_class, round(confidence_percent, 2), overlay


def build_grad_cam_overlay(image: Image.Image, input_tensor: torch.Tensor, target_index: int) -> Image.Image:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ModuleNotFoundError as exc:
        raise gr.Error("Install Grad-CAM with `pip install grad-cam`.") from exc

    image_size = METADATA["image_size"]
    resized_image = image.resize((image_size, image_size))
    rgb_float = np.asarray(resized_image, dtype=np.float32) / 255.0

    MODEL.zero_grad(set_to_none=True)
    with GradCAM(model=MODEL, target_layers=METADATA["target_layers"]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_index)],
        )[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return Image.fromarray(overlay)


def build_demo():
    description = (
        "Upload a plant leaf photo to classify the likely disease and view a "
        "Grad-CAM heatmap. The heatmap helps check whether the model focuses on "
        "leaf lesions instead of background clutter, pots, hands, or labels."
    )

    with gr.Blocks(title="Plant Disease Classifier") as demo:
        gr.Markdown("# Plant Disease Classifier")
        gr.Markdown(description)

        if LOAD_ERROR:
            gr.Markdown(
                "**Model checkpoint not found yet.** Add `models/best_model.pth` "
                "to the Space after training, or set `MODEL_PATH`."
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload leaf photo",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "webcam", "clipboard"],
                )
                predict_button = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                disease_output = gr.Textbox(label="Predicted disease")
                confidence_output = gr.Number(label="Confidence (%)", precision=2)
                grad_cam_output = gr.Image(label="Grad-CAM overlay", type="pil")

        predict_button.click(
            fn=predict,
            inputs=input_image,
            outputs=[disease_output, confidence_output, grad_cam_output],
            api_name="predict",
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
