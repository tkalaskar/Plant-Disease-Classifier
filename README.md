---
title: Plant Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Plant Disease Identification Using Transfer Learning

An end-to-end computer vision project that identifies plant leaf diseases from images using transfer learning in PyTorch. The project uses an ImageNet-pretrained EfficientNet-B0 model, a reproducible PlantVillage training pipeline, evaluation reports, Grad-CAM explainability, and a Streamlit interface for local prediction.

**Result:** Achieved **96.94% accuracy on a held-out PlantVillage test set of 4,345 images**, with **95.96% macro F1**. Best validation accuracy was **97.15% at epoch 9**. These results measure performance on PlantVillage's controlled images; accuracy on real field photos has not been established.

## Problem Statement

Plant diseases can reduce crop quality and yield if symptoms are not identified early. Manual inspection can be slow, inconsistent, and difficult to scale. This project explores whether a transfer learning based CNN can classify plant leaf diseases from images and provide interpretable visual feedback about the regions used for prediction.

The goal is not only to train a high-accuracy classifier, but to show a complete ML workflow that is understandable to internship recruiters: data preparation, model selection, training, evaluation, explainability, and deployment.

## Demo Screenshot

> Placeholder: add a screenshot of the Streamlit app after running it locally.

Suggested path:

```text
docs/assets/demo_screenshot.png
```

```md
![Streamlit demo screenshot](docs/assets/demo_screenshot.png)
```

## Features

- 38-class plant disease classification from leaf images.
- Transfer learning with an ImageNet-pretrained EfficientNet-B0 backbone.
- Stratified 80/10/10 train, validation, and test split.
- Train-only augmentation with horizontal flips, rotations, and color jitter.
- ImageNet normalization for compatibility with pretrained CNN weights.
- Two-stage fine-tuning: train classifier head first, then unfreeze the last backbone blocks.
- Mixed precision training with AMP on CUDA.
- Cosine learning rate scheduling.
- Early stopping based on validation loss.
- Best checkpointing based on validation accuracy.
- CSV training logs for reproducibility.
- Per-class precision, recall, and F1 evaluation.
- Confusion matrix and most-confused class pair analysis.
- Grad-CAM heatmaps to inspect whether the model focuses on leaf symptoms.
- Streamlit app for local image upload and prediction.
- Gradio app for Hugging Face Spaces deployment.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, Torchvision |
| Model | EfficientNet-B0 with ImageNet pretrained weights |
| Data Handling | Torchvision ImageFolder, scikit-learn, pandas |
| Visualization | Matplotlib, Seaborn |
| Explainability | pytorch-grad-cam |
| App | Streamlit, Gradio |
| Deployment | Hugging Face Spaces |
| Experiment Outputs | CSV logs, confusion matrix, Grad-CAM image artifacts |

## Dataset Description

This project uses the PlantVillage dataset from Kaggle.

- Dataset: PlantVillage
- Source: `emmarex/plantdisease` on Kaggle
- Size: about 54K leaf images
- Classes: 38 plant and disease categories
- Format: image folders grouped by class name
- Local expected structure: ImageFolder-compatible class folders under `data/PlantVillage` or `data/raw/plantvillage`

The dataset is useful for learning transfer learning workflows, but it is mostly made of clean, lab-style images with controlled backgrounds. This means test performance on PlantVillage may be higher than performance on real field images.

## Preprocessing

The data pipeline is implemented in `src/plant_disease_classifier/data.py`.

- Images are resized to `224 x 224`.
- Splits are stratified by class:
  - 80% training
  - 10% validation
  - 10% test
- Training augmentations:
  - random horizontal flip
  - random rotation up to 15 degrees
  - color jitter for brightness, contrast, and saturation
- Validation and test transforms:
  - resize
  - tensor conversion
  - normalization only
- Normalization uses ImageNet statistics:
  - mean: `[0.485, 0.456, 0.406]`
  - std: `[0.229, 0.224, 0.225]`

Before training, class balance can be checked with:

```bash
python scripts/plot_class_counts.py --data-dir data/PlantVillage
```

Output:

```text
reports/figures/class_counts.png
```

## Model Architecture

The selected backbone is `torchvision.models.efficientnet_b0`.

EfficientNet-B0 was chosen because it gives a strong accuracy-to-compute tradeoff for a portfolio project and can be trained on limited GPU resources such as a free Google Colab T4 session.

Architecture summary:

| Component | Description |
|---|---|
| Input | RGB leaf image resized to `224 x 224` |
| Backbone | ImageNet-pretrained EfficientNet-B0 feature extractor |
| Original head | ImageNet 1000-class classifier |
| New head | Linear classifier for 38 PlantVillage classes |
| Loss | Cross entropy loss |
| Optimizer | AdamW |
| Scheduler | Cosine annealing learning rate schedule |

Parameter summary:

| Stage | Total Parameters | Trainable Parameters |
|---|---:|---:|
| Full EfficientNet-B0 model | 4,056,226 | 4,056,226 |
| Frozen backbone, classifier head only | 4,056,226 | 48,678 |
| Last 2 backbone blocks unfrozen | 4,056,226 | 1,178,070 |

## Transfer Learning Approach

The model is trained in two stages.

Stage 1: Frozen Backbone

The EfficientNet-B0 backbone is frozen and only the new 38-class classifier head is trained. This lets the randomly initialized head learn PlantVillage class boundaries while preserving useful pretrained visual features such as edges, shapes, textures, and color patterns.

Stage 2: Fine-Tuning

After the classifier head has learned a stable mapping, the last two EfficientNet backbone blocks are unfrozen. These deeper layers are more task-specific, so fine-tuning them helps adapt ImageNet features to leaf disease patterns without retraining the entire network.

The backbone uses a lower learning rate than the classifier head because pretrained weights already contain useful visual representations. Updating them too aggressively can cause catastrophic forgetting, where the model overwrites useful general features learned from ImageNet.

## Training Process

Training is implemented in `scripts/train.py`.

Key training settings:

- Model: EfficientNet-B0
- Image size: `224`
- Batch size: `32`
- Head training epochs: `5`
- Fine-tuning epochs: `5`
- Unfrozen backbone blocks: `2`
- Initial classifier learning rate: `1e-3`
- Fine-tuning classifier learning rate: `1e-4`
- Fine-tuning backbone learning rate: `1e-5`
- Weight decay: `1e-4`
- Early stopping patience: `3`
- Best checkpoint metric: validation accuracy
- Training log: `models/training_log.csv`
- Best model checkpoint: `models/best_model.pth`

Best validation accuracy from the current training log:

```text
97.15% at epoch 9
```

## Evaluation Metrics

Evaluation is implemented in `scripts/evaluate.py`.

The project reports:

- test accuracy
- per-class precision
- per-class recall
- per-class F1-score
- macro precision, recall, and F1
- weighted precision, recall, and F1
- confusion matrix
- 10 most-confused class pairs
- Grad-CAM overlays for selected test images

Evaluation outputs:

```text
reports/classification_report.csv
reports/confusion_matrix.csv
reports/most_confused_pairs.csv
reports/test_predictions.csv
reports/figures/confusion_matrix.png
reports/figures/grad_cam/
```

## Results

The model correctly classified **4,212 of 4,345 held-out test images**, giving **96.94% test accuracy**. Accuracy is the number of correct predictions divided by the total number of labeled images evaluated.

The best checkpoint was selected by validation accuracy (**97.15% at epoch 9**). Test accuracy is reported separately to assess the selected model on the held-out test split.

| Model | Training Strategy | Val Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 5 frozen-head epochs + last 2 blocks fine-tuned | 97.15% | 96.94% | 96.48% | 95.63% | 95.96% | 96.91% |

### Where Accuracy Is Reported

- `models/training_log.csv`: training and validation accuracy for each epoch (`train_acc` and `val_acc`, stored as fractions). Generated locally during training.
- `reports/test_predictions.csv`: per-image test predictions and `is_correct` values used to calculate overall test accuracy.
- `reports/classification_report.csv`: per-class precision, recall, F1, and support; macro F1 gives each class equal weight, which helps assess performance when class sizes differ.

The app's **confidence score** is the model's softmax score for one prediction. It is not test accuracy or a guarantee that the uploaded image was classified correctly; confidence can be high even for an incorrect prediction.

These metrics describe the evaluated PlantVillage split, not field performance. An independent test set of real agricultural photos is needed to measure generalization beyond controlled backgrounds.

Most confused class pairs:

| Rank | Class A | Class B | Total Confusions |
|---:|---|---|---:|
| 1 | Corn Cercospora leaf spot / Gray leaf spot | Corn Northern Leaf Blight | 19 |
| 2 | Tomato Early blight | Tomato Late blight | 9 |
| 3 | Tomato Spider mites | Tomato healthy | 7 |
| 4 | Tomato Bacterial spot | Tomato Septoria leaf spot | 6 |
| 5 | Tomato Target Spot | Tomato healthy | 5 |

## Confusion Matrix

The confusion matrix is generated during evaluation.

```text
reports/figures/confusion_matrix.png
```

```md
![Confusion matrix](reports/figures/confusion_matrix.png)
```

## Grad-CAM Explainability

Grad-CAM is included to make the model more inspectable. Instead of only returning a class label, Grad-CAM highlights the image regions that contributed most strongly to the prediction.

For this project, Grad-CAM is useful because it helps answer an important question:

> Is the model looking at actual disease symptoms on the leaf, or is it learning shortcuts from the background, pot labels, lighting, or image borders?

Generated Grad-CAM files are saved in:

```text
reports/figures/grad_cam/
```

Example placeholder:

```md
![Grad-CAM example](reports/figures/grad_cam/example.png)
```

In the final portfolio version, I would include three example predictions showing:

- original uploaded leaf image
- predicted disease
- confidence score
- Grad-CAM overlay
- short interpretation of whether the highlighted area makes sense

## Streamlit App Usage

The Streamlit app is located at:

```text
app/streamlit_app.py
```

It lets a user:

- upload a leaf image
- select or enter the checkpoint path
- view the predicted disease class
- view the prediction confidence

Run it locally:

```bash
streamlit run app/streamlit_app.py
```

Default checkpoint path:

```text
models/best_model.pth
```

If the checkpoint does not exist yet, train the model first or copy the trained checkpoint into `models/best_model.pth`.

## Installation Instructions

Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd Plant-Disease-Classifier

python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download or prepare the dataset:

```bash
python scripts/download_dataset.py --dataset plantvillage --copy-to data/raw/plantvillage
```

If the dataset is already downloaded, make sure it is arranged like:

```text
data/PlantVillage/
  Apple___Apple_scab/
  Apple___Black_rot/
  ...
```

## How to Run Training

Run the full two-stage transfer learning pipeline:

```bash
python scripts/train.py \
  --data-dir data/PlantVillage \
  --model-name efficientnet_b0 \
  --head-epochs 5 \
  --fine-tune-epochs 5 \
  --unfreeze-blocks 2 \
  --batch-size 32
```

Expected key outputs:

```text
models/best_model.pth
models/training_log.csv
models/history.json
models/class_names.json
```

For Google Colab, enable:

```text
Runtime > Change runtime type > T4 GPU
```

Then run the same training command after installing dependencies and placing the dataset in the expected folder.

## How to Run Evaluation

```bash
python scripts/evaluate.py \
  --data-dir data/PlantVillage \
  --checkpoint models/best_model.pth \
  --num-grad-cam 10 \
  --grad-cam-misclassified-first
```

This generates the classification report, confusion matrix, most-confused pairs, prediction CSV, and Grad-CAM overlays.

## How to Run Prediction

Option 1: Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL, upload a leaf image, and use:

```text
models/best_model.pth
```

as the checkpoint path.

Option 2: Gradio app for Hugging Face Spaces

```bash
python app.py
```

The Gradio app returns:

- predicted disease
- confidence score
- Grad-CAM overlay

Minimum files needed for Hugging Face Spaces:

```text
app.py
requirements.txt
src/plant_disease_classifier/
models/best_model.pth
```

The app runs on CPU by default. For ZeroGPU Spaces, `app.py` includes a `@spaces.GPU` decorated prediction function. To actually use CUDA on ZeroGPU, set this Space environment variable:

```text
USE_CUDA=1
```

## Future Improvements

- Validate the model on real field images with cluttered backgrounds and varied lighting.
- Add a small external test set separate from PlantVillage.
- Compare EfficientNet-B0 with MobileNetV3, ResNet-50, and ConvNeXt-Tiny.
- Add experiment tracking with MLflow or Weights & Biases.
- Add top-k predictions so users can see alternative likely diseases.
- Improve calibration so confidence scores are more reliable.
- Export the model to ONNX or TorchScript for lighter deployment.
- Add a command-line prediction script for batch inference.
- Add automated tests for data splitting, transforms, and checkpoint loading.

## What I Learned

- How transfer learning reduces training cost compared with training a CNN from scratch.
- Why freezing the backbone first can stabilize training on a smaller dataset.
- Why the classifier head and backbone should use different learning rates during fine-tuning.
- How to build stratified train, validation, and test splits for image classification.
- How to evaluate a model beyond accuracy using precision, recall, F1, and confusion matrices.
- How Grad-CAM can reveal whether a model is focusing on meaningful disease regions or learning dataset shortcuts.
- How to package an ML project for recruiters with training code, evaluation artifacts, and an interactive app.

## Limitations

PlantVillage is a clean lab-style dataset. Real agricultural images can include multiple leaves, complex backgrounds, shadows, occlusion, blur, soil, tools, hands, and different camera qualities. Because of this domain gap, the model should be treated as a learning and portfolio project, not a production plant disease diagnosis system.
