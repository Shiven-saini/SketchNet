"""
train.py — Training pipeline for SketchNet on 50 Quick Draw classes

Uses SketchNet: a CNN tuned specifically for 50 classes.
  - Block3: 192 channels  (vs 128 in 20-cls, 256 in 345-cls)
  - Block4: 384 channels  (vs 256 in 20-cls, 512 in 345-cls)
  - Classifier: 384→192→50  (192 >> 50 = healthy separation margin)
  - Extra ResidualBlock in block2 for richer mid-level features
  - ~2.1M parameters

Usage:
    cd kalakari/
    python train.py
    python train.py --epochs 40 --batch_size 512
    python train.py --max_per_class 30000
    python train.py --resume checkpoints/best_model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

# allow imports from parent directory if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from model import build_model


# 50 categories

CATEGORIES_20 = [
    "cat", "dog", "car", "house", "tree",
    "fish", "bird", "airplane", "bicycle", "clock",
    "sun", "moon", "star", "flower", "apple",
    "banana", "pizza", "guitar", "hat", "shoe",
]

CATEGORIES_50 = CATEGORIES_20 + [
    "elephant", "giraffe", "penguin", "dolphin", "butterfly",
    "strawberry", "pineapple", "watermelon", "grapes", "camera",
    "telephone", "laptop", "television", "couch", "chair",
    "bed", "door", "picture frame", "ladder", "bridge",
    "sailboat", "bus", "train", "helicopter", "hot air balloon",
    "sword", "crown", "diamond", "hourglass", "candle",
]


# Mixup data for much better idenitification among confusing stroke set

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B   = x.size(0)
    idx = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Metrics

def topk_accuracy(output, target, topk=(1, 3, 5)):
    with torch.no_grad():
        maxk = max(topk)
        B    = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / B).item()
                for k in topk]


# Train epoch

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, use_mixup):
    model.train()
    total_loss = top1_sum = top3_sum = n = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            logits = model(imgs)
            loss   = (mixup_criterion(criterion, logits, y_a, y_b, lam)
                      if use_mixup else criterion(logits, labels))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        top1, top3, _ = topk_accuracy(logits, labels)
        B = labels.size(0)
        total_loss += loss.item() * B
        top1_sum   += top1 * B
        top3_sum   += top3 * B
        n          += B

    return total_loss / n, top1_sum / n, top3_sum / n


# Val / test epoch

@torch.no_grad()
def val_epoch(model, loader, criterion, device, collect_preds=False):
    model.eval()
    total_loss = top1_sum = top3_sum = n = 0
    all_preds  = []
    all_labels = []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        top1, top3, _ = topk_accuracy(logits, labels)
        B = labels.size(0)
        total_loss += loss.item() * B
        top1_sum   += top1 * B
        top3_sum   += top3 * B
        n          += B

        if collect_preds:
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    result = (total_loss / n, top1_sum / n, top3_sum / n)
    return (result, all_preds, all_labels) if collect_preds else result


# Confusion analysis

def save_confusion(preds, labels, categories, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(18, 16))
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=categories)
        disp.plot(ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues",
                  values_format=".2f")
        ax.set_title("SketchNet — Normalized Confusion Matrix", fontsize=14, pad=16)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved → {path}")

        np.fill_diagonal(cm, 0)
        confused = sorted(
            [(cm[i, j], categories[i], categories[j])
             for i in range(len(categories))
             for j in range(len(categories)) if cm[i, j] > 0],
            reverse=True,
        )
        print(f"\n  Top confused pairs:")
        for count, tc, pc in confused[:10]:
            print(f"    {tc:<14} → {pc:<14}  ({count} samples)")

    except ImportError:
        print("  (Install matplotlib + scikit-learn to generate confusion matrix)")


# Main

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═'*62}")
    print(f"  SketchNet: Quick Draw 50-Class Training")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM     : {mem:.1f} GB")
    print(f"{'═'*62}\n")

    categories  = CATEGORIES_50[:args.categories]
    num_classes = len(categories)
    print(f"  Classes  : {num_classes}\n")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Data
    train_loader, val_loader, test_loader, label2idx = get_dataloaders(
        data_dir=args.data_dir,
        categories=categories,
        img_size=args.img_size,
        max_per_class=args.max_per_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    idx2label = {str(v): k for k, v in label2idx.items()}
    label_map_path = f"checkpoints/label_map_{num_classes}.json"
    with open(label_map_path, "w") as f:
        json.dump(idx2label, f, indent=2)
    print(f"  Label map → {label_map_path}")

    # ── Model
    model = build_model(num_classes=num_classes, device=device, dropout=args.dropout)

    # ── Resume
    start_epoch   = 1
    best_val_top1 = 0.0
    best_ckpt     = f"checkpoints/best_model_{num_classes}.pt"

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_top1 = ckpt.get("val_top1", 0.0)
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, "
              f"val_top1={best_val_top1:.2f}%)\n")

    # ── Optimizer: SGD + OneCycleLR Scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler("cuda")

    history    = []
    no_improve = 0

    print(f"{'Epoch':>6} | {'T-Loss':>7} | {'T-Acc1':>7} | {'T-Acc3':>7} | "
          f"{'V-Loss':>7} | {'V-Acc1':>7} | {'V-Acc3':>7} | {'LR':>9} | {'Time':>5}")
    print("─" * 90)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_top1, train_top3 = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler,
            device, use_mixup=args.mixup,
        )
        val_loss, val_top1, val_top3 = val_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {train_loss:>7.4f} | {train_top1:>6.2f}% | {train_top3:>6.2f}% | "
              f"{val_loss:>7.4f} | {val_top1:>6.2f}% | {val_top3:>6.2f}% | "
              f"{lr:>9.2e} | {elapsed:>4.0f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_top1": train_top1, "train_top3": train_top3,
            "val_loss":   val_loss,   "val_top1":   val_top1,   "val_top3":   val_top3,
        })

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            no_improve    = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_top1":        val_top1,
                "val_top3":        val_top3,
                "args":            vars(args),
                "categories":      categories,
                "num_classes":     num_classes,
                "arch":            "SketchNet50",
            }, best_ckpt)
            print(f"         ✓ Best model saved → {best_ckpt}  (val_top1={val_top1:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping — no improvement for {args.patience} epochs.")
                break

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       f"checkpoints/epoch_{epoch:03d}_{num_classes}cls.pt")

    log_path = f"logs/history_{num_classes}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history → {log_path}")

    # ── Final test evaluation
    print(f"\n{'═'*62}")
    print("  Loading best model for final test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    (test_loss, test_top1, test_top3), preds, labels_list = val_epoch(
        model, test_loader, criterion, device, collect_preds=True
    )

    print(f"  Test Top-1 Accuracy : {test_top1:.2f}%")
    print(f"  Test Top-3 Accuracy : {test_top3:.2f}%")
    print(f"  Test Loss           : {test_loss:.4f}")
    print(f"  Classes             : {num_classes}")
    print(f"{'═'*62}\n")

    save_confusion(preds, labels_list, categories,
                   path=f"logs/confusion_matrix_{num_classes}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train SketchNet-50 on 50 Quick Draw classes"
    )

    # Data
    p.add_argument("--data_dir",      default="data/npy")
    p.add_argument("--categories",    type=int,   default=50,
                   help="First N categories from CATEGORIES_50 (default: 50)")
    p.add_argument("--img_size",      type=int,   default=64)
    p.add_argument("--max_per_class", type=int,   default=50_000,
                   help="Samples per class cap (default: 50k → 2.5M total)")

    # Model
    p.add_argument("--dropout",       type=float, default=0.4)

    # Training
    p.add_argument("--epochs",        type=int,   default=35)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=0.05)
    p.add_argument("--mixup",         action="store_true", default=True)
    p.add_argument("--no_mixup",      action="store_false", dest="mixup")
    p.add_argument("--patience",      type=int,   default=8)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--resume",        default=None)

    args = p.parse_args()
    main(args)

    
