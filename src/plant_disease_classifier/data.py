from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _contains_direct_images(path: Path) -> bool:
    return any(file.suffix.lower() in IMG_EXTENSIONS for file in path.iterdir() if file.is_file())


def find_imagefolder_root(data_dir: str | Path, min_classes: int = 2) -> Path:
    """Find the directory whose immediate children are image class folders."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for candidate in candidates:
        class_dirs = [path for path in candidate.iterdir() if path.is_dir() and _contains_direct_images(path)]
        if len(class_dirs) >= min_classes:
            return candidate

    raise ValueError(f"Could not find an ImageFolder-style dataset under: {root}")


def count_images_by_class(data_dir: str | Path) -> tuple[dict[str, int], Path]:
    dataset_root = find_imagefolder_root(data_dir)
    counts = {}

    for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        image_count = sum(
            1 for file in class_dir.iterdir() if file.is_file() and file.suffix.lower() in IMG_EXTENSIONS
        )
        if image_count:
            counts[class_dir.name] = image_count

    return counts, dataset_root


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_split_indices(
    targets: list[int],
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> dict[str, list[int]]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    indices = list(range(len(targets)))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=targets,
    )
    train_val_targets = [targets[index] for index in train_val_indices]
    relative_val_size = val_size / (1.0 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_targets,
    )

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    val_size: float = 0.10,
    test_size: float = 0.10,
    num_workers: int = 2,
    seed: int = 42,
):
    dataset_root = find_imagefolder_root(data_dir)
    train_transform, eval_transform = build_transforms(image_size)

    base_dataset = datasets.ImageFolder(dataset_root)
    split_indices = build_split_indices(base_dataset.targets, val_size=val_size, test_size=test_size, seed=seed)

    train_dataset = datasets.ImageFolder(dataset_root, transform=train_transform)
    eval_dataset = datasets.ImageFolder(dataset_root, transform=eval_transform)

    loaders = {
        "train": DataLoader(
            Subset(train_dataset, split_indices["train"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            Subset(eval_dataset, split_indices["val"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            Subset(eval_dataset, split_indices["test"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders, base_dataset.classes, dataset_root
