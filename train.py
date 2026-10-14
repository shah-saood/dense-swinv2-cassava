"""
Training script for Dense-SwinV2 on the Cassava Leaf Disease dataset.

Usage:
    python train.py --data_dir /path/to/data --epochs 30 --batch_size 16

Features:
    - Focal loss for class imbalance
    - Cosine annealing learning rate schedule
    - Mixed precision training (AMP)
    - Automatic checkpointing of best model
    - Per-epoch validation metrics
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms
from tqdm import tqdm

from model import DenseSwinV2


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """
    Expects an ImageFolder layout:
        data_dir/
            train/class_0/...
            train/class_1/...
            val/class_0/...
            val/class_1/...
    """

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="  train", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="  val  ", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir containing train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, n_classes = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    if n_classes != args.num_classes:
        print(f"[warn] dataset has {n_classes} classes, overriding --num_classes")
        args.num_classes = n_classes
    print(f"Classes: {n_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Model
    model = DenseSwinV2(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.2f} M")

    # Loss / optimizer / scheduler
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(args.out_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": args.num_classes,
            }, ckpt_path)
            print(f"  -> saved {ckpt_path} (val_acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
