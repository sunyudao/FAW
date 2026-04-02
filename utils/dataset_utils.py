"""Dataset utilities for loading and configuring various datasets."""

import os
import zipfile
import requests
import shutil
from tqdm import tqdm

from torchvision import transforms
from torchvision import datasets

# Dataset configurations
DATASET_CONFIG = {
    "mnist": {
        "in_channels": 3,
        "num_classes": 10,
        "train_transform": transforms.Compose([transforms.ToTensor()]),
        "test_transform": transforms.Compose([transforms.ToTensor()]),
    },
    "gtsrb": {
        "in_channels": 3,
        "num_classes": 43,
        "train_transform": transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        ),
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "train_transform": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "test_transform": transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    },
    "cifar10": {
        "in_channels": 3,
        "num_classes": 10,
        "train_transform": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ]
        ),
        "test_transform": transforms.Compose([transforms.ToTensor()]),
    },
    "imagenet": {
        "num_classes": 1000,
        "train_transform": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "test_transform": transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        ),
    },
}


def download_tiny_imagenet(data_dir):
    """
    Download and organize the Tiny ImageNet dataset.

    Downloads the dataset, extracts it, and reorganizes the validation set
    to match the ImageFolder directory structure.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        Path to the organized dataset directory.
    """
    os.makedirs(data_dir, exist_ok=True)

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")

    # Check if dataset is already organized
    if os.path.exists(extract_path) and len(os.listdir(extract_path)) == 3:
        print("Tiny ImageNet dataset already exists and is organized.")
        return extract_path

    # Download dataset
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Organize validation set structure
    print("Organizing validation set...")
    val_dir = os.path.join(extract_path, "val")
    val_images_dir = os.path.join(val_dir, "images")

    # Read validation annotations
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    with open(val_annotations_file, "r") as f:
        val_annotations = f.readlines()

    # Create class directories and move images
    for line in val_annotations:
        filename, class_name, *_ = line.strip().split("\t")

        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        src_path = os.path.join(val_images_dir, filename)
        dst_path = os.path.join(class_dir, filename)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    # Clean up original files
    if os.path.exists(val_images_dir):
        shutil.rmtree(val_images_dir)
    if os.path.exists(val_annotations_file):
        os.remove(val_annotations_file)

    print("Tiny ImageNet dataset download and organization complete!")
    return extract_path


def get_dataset(dataset_name, data_dir, download=True):
    """
    Load and configure a dataset.

    Args:
        dataset_name: Name of the dataset ('mnist', 'gtsrb', 'tiny_imagenet', 'cifar10', 'imagenet').
        data_dir: Directory to store/load the dataset.
        download: Whether to download the dataset if not present.

    Returns:
        Tuple of (train_dataset, test_dataset, num_classes).
    """
    config = DATASET_CONFIG[dataset_name]

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=download, transform=config["train_transform"]
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=download, transform=config["test_transform"]
        )

    elif dataset_name == "gtsrb":
        train_dataset = datasets.GTSRB(
            root=data_dir, split="train", download=download, transform=config["train_transform"]
        )
        test_dataset = datasets.GTSRB(
            root=data_dir, split="test", download=download, transform=config["test_transform"]
        )

    elif dataset_name == "tiny_imagenet":
        dataset_root = download_tiny_imagenet(data_dir)
        train_dir = os.path.join(dataset_root, "train")
        val_dir = os.path.join(dataset_root, "val")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Tiny ImageNet data not found at '{dataset_root}'. "
                "Please ensure the dataset is available."
            )

        train_dataset = datasets.ImageFolder(root=train_dir, transform=config["train_transform"])
        test_dataset = datasets.ImageFolder(root=val_dir, transform=config["test_transform"])

    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=config["train_transform"]
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=download, transform=config["test_transform"]
        )

    elif dataset_name == "imagenet":
        train_dir = os.path.join(data_dir, "Imagenet2012", "train")
        val_dir = os.path.join(data_dir, "Imagenet2012", "val")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"ImageNet data not found at '{data_dir}'. "
                "Please ensure 'train' and 'val' folders exist."
            )

        train_dataset = datasets.ImageFolder(root=train_dir, transform=config["train_transform"])
        test_dataset = datasets.ImageFolder(root=val_dir, transform=config["test_transform"])

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from ['mnist', 'gtsrb', 'tiny_imagenet', 'cifar10', 'imagenet']."
        )

    return train_dataset, test_dataset, config["num_classes"]
