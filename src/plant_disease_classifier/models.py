import os

from torch import nn
from torchvision import models


FEATURE_BACKBONES = {"efficientnet_b0", "mobilenet_v3_large"}


def configure_ssl_certificates() -> None:
    """Point urllib/torch downloads at certifi's CA bundle when available."""
    try:
        import certifi
    except ModuleNotFoundError:
        return

    certificate_path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", certificate_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certificate_path)


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if pretrained:
        configure_ssl_certificates()

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Torchvision could not download the pretrained EfficientNet-B0 weights. "
                    "If this is an SSL certificate error on macOS, run "
                    "`python -m pip install -U certifi` inside your venv and retry. "
                    "You can also train without pretrained weights by changing the code to "
                    "call build_model(..., pretrained=False), but transfer learning quality "
                    "will be much worse."
                ) from exc
            raise

        # Torchvision's EfficientNet-B0 classifier ends with a Linear layer at index 1.
        # Swap that ImageNet 1000-class layer for the PlantVillage 38-class problem.
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_classifier_module(model: nn.Module, model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name in FEATURE_BACKBONES:
        return model.classifier
    if model_name == "resnet50":
        return model.fc
    raise ValueError(f"Unsupported model_name: {model_name}")


def get_classifier_parameters(model: nn.Module, model_name: str) -> list[nn.Parameter]:
    return list(get_classifier_module(model, model_name).parameters())


def freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    # Start frozen because the pretrained backbone already contains useful generic
    # visual features such as edges, textures, shapes, and leaf-like patterns. With
    # only the new classifier head trainable, the random head can learn stable class
    # boundaries before we risk changing the transferred feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model_name = model_name.lower()
    if model_name in FEATURE_BACKBONES:
        # Keep only the new classifier head trainable for the first training stage.
        # If we fine-tuned the whole network from epoch 1 on a relatively small
        # dataset, early noisy gradients from the randomly initialized head could
        # overwrite useful pretrained representations. That overwrite is called
        # catastrophic forgetting: the model "forgets" broadly useful ImageNet
        # features while trying to adapt too aggressively to the new dataset.
        for parameter in get_classifier_parameters(model, model_name):
            parameter.requires_grad = True
    elif model_name == "resnet50":
        for parameter in get_classifier_parameters(model, model_name):
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


def unfreeze_last_backbone_blocks(model: nn.Module, model_name: str, num_blocks: int = 2) -> nn.Module:
    if num_blocks < 1:
        raise ValueError("num_blocks must be at least 1.")

    model_name = model_name.lower()
    if model_name in FEATURE_BACKBONES:
        blocks = list(model.features.children())
    elif model_name == "resnet50":
        blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    for block in blocks[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    return model


def get_trainable_backbone_parameters(model: nn.Module, model_name: str) -> list[nn.Parameter]:
    classifier_parameter_ids = {id(parameter) for parameter in get_classifier_parameters(model, model_name)}
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_parameter_ids
    ]


def build_frozen_efficientnet_b0(num_classes: int = 38) -> nn.Module:
    model = build_model("efficientnet_b0", num_classes=num_classes, pretrained=True)
    return freeze_backbone(model, "efficientnet_b0")
