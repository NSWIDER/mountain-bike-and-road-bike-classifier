"""
evaluate_pyTorch.py

Loads (or trains) a PyTorch CNN model and evaluates images from test
directories, outputting a JSON report to json/pyTorch/<modelname>.json.

The CNN architecture mirrors train.py exactly:
    Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
    Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
    Flatten → Dense(128, ReLU) → Dense(1, Sigmoid)

Preprocessing matches the Keras ImageDataGenerator used in train.py:
    • Resize to 64×64 pixels
    • Rescale to [0, 1]  (÷255)

Usage:
    python evaluate_pyTorch.py --model bike_classifier.keras \
        --test-dir images/mountain --test-dir images/road \
        --train-dir images/trainset

Exposes:
    run_evaluation(model_path, test_dirs, train_dir, threshold=0.5)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ── Model definition ─────────────────────────────────────────────────────────

class BikeClassifierCNN(nn.Module):
    """
    PyTorch equivalent of the Keras CNN defined in train.py.

    Input shape  : (N, 3, 64, 64)

    Architecture:
        Conv2d(3→32, 3×3)  → ReLU → MaxPool(2×2)   # 64 → 62 → 31
        Conv2d(32→32, 3×3) → ReLU → MaxPool(2×2)   # 31 → 29 → 14
        Flatten                                      # 32×14×14 = 6272
        Linear(6272 → 128) → ReLU
        Linear(128  → 1)   → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)   # no padding (valid)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)  # no padding (valid)
        self.fc1   = nn.Linear(32 * 14 * 14, 128)
        self.fc2   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → (N, 32, 31, 31)
        x = self.pool(F.relu(self.conv2(x)))   # → (N, 32, 14, 14)
        x = x.view(x.size(0), -1)             # → (N, 6272)
        x = F.relu(self.fc1(x))               # → (N, 128)
        x = torch.sigmoid(self.fc2(x))        # → (N, 1)
        return x


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_class_names(train_dir):
    """
    Return class names from training directory subdirectories, sorted
    alphabetically (mirrors Keras ImageDataGenerator class ordering).
    """
    subdirs = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return subdirs


def _get_pth_path(model_path):
    """
    Derive the .pth filename from the supplied model_path.
    E.g. 'bike_classifier.keras' → 'bike_classifier.pth'
    """
    return str(Path(model_path).with_suffix('.pth'))


def _build_eval_transform():
    """
    Preprocessing pipeline for inference – matches Keras rescale=1./255
    with target_size=(64, 64), but without training-time augmentations.
    """
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),     # HWC uint8 [0,255] → CHW float32 [0,1]
    ])


# ── Simple dataset for training ───────────────────────────────────────────────

class FolderDataset(Dataset):
    """
    Minimal image dataset that mirrors Keras flow_from_directory:
    class labels are assigned alphabetically from subdirectory names.
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    def __init__(self, root, transform=None):
        self.transform  = transform
        self.class_names = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = []
        for cls in self.class_names:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if Path(fname).suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def _train_model(pth_path, train_dir, epochs=10):
    """
    Train a BikeClassifierCNN using the same data pipeline as train.py
    (with augmentation) and save the weights to *pth_path*.
    """
    print(f"[PT] Training new PyTorch model from '{train_dir}' → '{pth_path}' ...")

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, shear=11.5),   # shear_range≈0.2 rad
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # zoom_range=0.2
        transforms.ToTensor(),
    ])

    dataset   = FolderDataset(train_dir, transform=train_transform)
    loader    = DataLoader(dataset, batch_size=32, shuffle=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = BikeClassifierCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds    = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += images.size(0)

        print(f"  Epoch {epoch:02d}/{epochs}  "
              f"loss={total_loss/total:.4f}  "
              f"acc={correct/total:.4f}")

    # Save model state dict
    pth_dir = os.path.dirname(os.path.abspath(pth_path))
    os.makedirs(pth_dir, exist_ok=True)
    torch.save(model.state_dict(), pth_path)
    print(f"[PT] Model saved to '{pth_path}'.")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(model_path, test_dirs, train_dir, threshold=0.5):
    """
    Evaluate images in *test_dirs* using a PyTorch CNN model.

    Parameters
    ----------
    model_path : str
        Path given by the user (e.g. 'bike_classifier.keras').
        The PyTorch weights file is derived from it by swapping the
        extension to '.pth' (e.g. 'bike_classifier.pth').  If that
        file does not exist the model is trained from scratch and saved.
    test_dirs  : list[str]
        Directories whose images will be evaluated.  The directory
        *name* (basename) is used as the ground-truth label.
    train_dir  : str
        Root of the training dataset (subdirectory names → class names).
    threshold  : float, optional
        Sigmoid decision threshold.  Defaults to 0.5.

    Returns
    -------
    list[dict]
        Each dict contains: filename, predicted_class, confidence_score,
        ground_truth.  Also written to json/pyTorch/<model_stem>.json.
    """
    pth_path = _get_pth_path(model_path)

    # ── Load or train model ──────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = BikeClassifierCNN().to(device)

    if not os.path.exists(pth_path):
        print(f"[PT] Model file '{pth_path}' not found – training a new one.")
        model = _train_model(pth_path, train_dir)
    else:
        print(f"[PT] Loading model weights from '{pth_path}' ...")
        state = torch.load(pth_path, map_location=device)
        model.load_state_dict(state)

    model.eval()

    # ── Determine class names from training directory ────────────────────────
    class_names = get_class_names(train_dir)
    if len(class_names) < 2:
        raise ValueError(
            f"Expected at least 2 class subdirectories in '{train_dir}', "
            f"found: {class_names}"
        )

    transform = _build_eval_transform()
    results   = []

    for test_dir in test_dirs:
        ground_truth = os.path.basename(os.path.normpath(test_dir))

        files = sorted([
            f for f in os.listdir(test_dir)
            if os.path.isfile(os.path.join(test_dir, f))
        ])

        for filename in files:
            filepath = os.path.join(test_dir, filename)

            img   = Image.open(filepath).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 64, 64)

            with torch.no_grad():
                output = model(tensor)                        # (1, 1)

            confidence = float(output[0][0].item())
            predicted_class = class_names[1] if confidence >= threshold else class_names[0]

            results.append({
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence_score': round(confidence, 6),
                'ground_truth': ground_truth,
            })

            print(f"  {filename:30s}  pred={predicted_class:20s}  "
                  f"conf={confidence:.4f}  gt={ground_truth}")

    # ── Write JSON report ────────────────────────────────────────────────────
    model_stem  = Path(model_path).stem
    output_dir  = os.path.join('json', 'pyTorch')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_stem}.json')

    with open(output_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[PT] Report written to '{output_path}'  ({len(results)} records).")
    return results


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a PyTorch image-classification model and output a JSON report.'
    )
    parser.add_argument(
        '--model', default='bike_classifier.keras',
        help='Model reference (default: bike_classifier.keras).  '
             'The PyTorch weights are stored/loaded as <stem>.pth.'
    )
    parser.add_argument(
        '--test-dir', action='append', dest='test_dirs',
        metavar='DIR',
        help='Directory of test images (may be given multiple times). '
             'Defaults to images/mountain and images/road.'
    )
    parser.add_argument(
        '--train-dir', default='images/trainset',
        help='Training-set root directory (default: images/trainset)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Sigmoid decision threshold (default: 0.5)'
    )

    args = parser.parse_args()

    if args.test_dirs is None:
        args.test_dirs = ['images/mountain', 'images/road']

    run_evaluation(
        model_path=args.model,
        test_dirs=args.test_dirs,
        train_dir=args.train_dir,
        threshold=args.threshold,
    )
    sys.exit(0)
