"""
Evaluation script for Dense-SwinV2.

Loads a trained checkpoint and computes:
    - Top-1 accuracy
    - Macro F1-score
    - ROC-AUC (one-vs-rest, macro)
    - Per-class precision / recall / F1
    - Confusion matrix

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir ./data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from model import DenseSwinV2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_test_loader(data_dir: str, batch_size: int, num_workers: int = 4):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dir = Path(data_dir) / "val"
    if not test_dir.exists():
        test_dir = Path(data_dir) / "test"
    ds = datasets.ImageFolder(str(test_dir), transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, ds.classes


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    for images, targets in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(1).cpu().numpy())
        all_targets.append(targets.numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return probs, preds, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a checkpoint saved by train.py")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir containing val/ or test/ subfolder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", 5)
    print(f"Checkpoint epoch {ckpt.get('epoch', '?')} | "
          f"best val acc {ckpt.get('val_acc', 0.0):.4f}")

    # Build model
    model = DenseSwinV2(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Data
    loader, class_names = build_test_loader(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Evaluating on {len(loader.dataset)} samples, {len(class_names)} classes")

    # Inference
    probs, preds, targets = evaluate(model, loader, device)

    # Metrics
    acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average="macro")
    try:
        auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Top-1 accuracy : {acc:.4f}")
    print(f"  Macro F1       : {f1_macro:.4f}")
    print(f"  ROC-AUC (macro): {auc:.4f}")
    print("=" * 60)

    print("\nPer-class report:")
    print(classification_report(targets, preds, target_names=class_names,
                                digits=4))

    print("Confusion matrix:")
    cm = confusion_matrix(targets, preds)
    print(cm)


if __name__ == "__main__":
    main()
