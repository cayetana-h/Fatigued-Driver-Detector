"""
Train EyeStateCNN on the MRL Eye Dataset.

Dataset layout (akashshingha850/mrl-eye-dataset from Kaggle)
-------------------------------------------------------------
<data_dir>/
    train/
        awake/
        sleepy/
    val/
        awake/
        sleepy/
    test/
        awake/
        sleepy/

Usage
-----
    python scripts/train_eye_classifier.py data/raw
    python scripts/train_eye_classifier.py data/raw --epochs 30
    python scripts/train_eye_classifier.py data/raw --epochs 30 --out models/eye_classifier.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fatigue.dl import EyeStateCNN


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EyeDataset(Dataset):
    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    # awake = 0 (open), sleepy = 1 (closed)
    _LABEL_MAP = {"awake": 0, "sleepy": 1}

    def __init__(self, split_dir: Path, size: int = EyeStateCNN.INPUT_SIZE) -> None:
        self.size = size
        self.samples: list[tuple[Path, int]] = []

        for folder, label in self._LABEL_MAP.items():
            folder_path = split_dir / folder
            if not folder_path.is_dir():
                raise FileNotFoundError(
                    f"Expected folder '{folder_path}'. "
                    "Make sure the dataset has awake/ and sleepy/ subdirectories."
                )
            for path in sorted(folder_path.iterdir()):
                if path.suffix.lower() in self._EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under '{split_dir}'.")

        awake  = sum(1 for _, l in self.samples if l == 0)
        sleepy = sum(1 for _, l in self.samples if l == 1)
        print(f"  {split_dir.name:<6}  awake={awake:>6}  sleepy={sleepy:>6}  total={len(self.samples):>6}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size))
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, torch.tensor(float(label))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(data_dir: str, epochs: int, batch_size: int, lr: float, out: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    root = Path(data_dir)
    print("Loading splits...")
    train_ds = EyeDataset(root / "train")
    val_ds   = EyeDataset(root / "val")
    test_ds  = EyeDataset(root / "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model     = EyeStateCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"\nTraining for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = (model(imgs).squeeze(1) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc  = correct / total
        avg_loss = running_loss / len(train_loader)
        scheduler.step(1.0 - val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            marker = "  <- saved"

        print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}{marker}")

    # final evaluation on held-out test set
    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"Model saved to    : {out_path}")

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    model.eval()
    correct = total = tp = fp = fn = tn = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (model(imgs).squeeze(1) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Test accuracy : {correct / total:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1            : {f1:.4f}")
    print(f"  Confusion     : TP={tp} FP={fp} FN={fn} TN={tn}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EyeStateCNN on MRL Eye Dataset")
    parser.add_argument(
        "data_dir",
        help="Dataset root containing train/, val/, test/ subdirectories (e.g. data/raw)",
    )
    parser.add_argument("--epochs",     type=int,   default=20,                         help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int,   default=64,                         help="Batch size (default: 64)")
    parser.add_argument("--lr",         type=float, default=1e-3,                       help="Learning rate (default: 0.001)")
    parser.add_argument("--out",        type=str,   default="models/eye_classifier.pt", help="Output path for best model weights")
    args = parser.parse_args()

    train(
        data_dir   = args.data_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        out        = args.out,
    )
