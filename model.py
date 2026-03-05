"""
model_50.py — SketchNet-50: CNN tuned for 50-class Quick Draw recognition

Why a separate architecture for 50 classes vs 20 or 345?

  20 classes : narrow backbone (32→64→128→256), 1.2M params
               128-dim embedding → 20 output — ratio 6.4× (very comfortable)

  50 classes : medium backbone (32→64→128→256→384), 2.1M params  ← THIS FILE
               384-dim embedding → 50 output — ratio 7.7× (healthy separation margin)

  345 classes: wide backbone  (32→64→128→256→512), 4.5M params
               512-dim embedding → 345 output — ratio ~1.5× (tight but workable)

Key changes vs the 20-class model:
  1. Block 3: 128 → 192 channels  (was 128→128, added capacity)
  2. Block 4: 192 → 384 channels  (was 128→256, widens embedding)
  3. Classifier: 384 → 192 → 50  (penultimate 192 >> 50 → clean separation)
  4. Extra ResidualBlock in block2 (was ×2, now ×3 — more shape detail)
  5. Slightly higher block3 dropout (0.15 → 0.2) to counteract added capacity

Input : (B, 1, 64, 64) — grayscale sketch
Output: (B, 50)
Params: ~2.1M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks (identical to base model) ─────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch,  kernel=3, stride=stride, padding=1, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, kernel=1, stride=1,      padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch, kernel=3, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class MultiScaleBlock(nn.Module):
    """
    Inception-style multi-scale block.
    out_ch must be divisible by 4 (split equally across 4 branches).
    """
    def __init__(self, in_ch, out_ch):
        assert out_ch % 4 == 0
        branch_ch = out_ch // 4
        super().__init__()
        self.b1 = ConvBNReLU(in_ch, branch_ch, kernel=1, padding=0)
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel=1, padding=0),
            ConvBNReLU(branch_ch, branch_ch, kernel=3, padding=1),
        )
        self.b5 = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel=1, padding=0),
            ConvBNReLU(branch_ch, branch_ch, kernel=3, padding=1),
            ConvBNReLU(branch_ch, branch_ch, kernel=3, padding=1),
        )
        self.bp = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, branch_ch, kernel=1, padding=0),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


# ── SketchNet-50 ──────────────────────────────────────────────────────────────

class SketchNet50(nn.Module):
    """
    Forward pass shapes:

    Input  (B,   1, 64, 64)
      ↓ block1      → (B,  32, 32, 32)   Conv + ResBlock×1   + MaxPool2
      ↓ block2      → (B,  64, 16, 16)   Conv + ResBlock×3   + MaxPool2  ← +1 residual vs 20-cls
      ↓ multiscale  → (B, 128, 16, 16)   Inception 4-branch
      ↓ block3      → (B, 192,  8,  8)   DW-Sep 128→192 + ResBlock + MaxPool2
      ↓ block4      → (B, 384,  4,  4)   DW-Sep 192→384 + ResBlock×2 + MaxPool2
      ↓ GAP         → (B, 384,  1,  1)
      ↓ classifier  → (B,  50)           Linear 384→192→50
    """

    def __init__(self, num_classes=50, dropout=0.4):
        super().__init__()

        # ── Block 1: edge detection (same as 20-cls, shallow is fine here)
        self.block1 = nn.Sequential(
            ConvBNReLU(1, 32, kernel=3, stride=1, padding=1),
            ResidualBlock(32, dropout=0.05),
            nn.MaxPool2d(2, 2),           # 64 → 32
        )

        # ── Block 2: shape assembly — extra residual for richer mid-level features
        self.block2 = nn.Sequential(
            ConvBNReLU(32, 64, kernel=3, stride=1, padding=1),
            ResidualBlock(64, dropout=0.1),
            ResidualBlock(64, dropout=0.1),
            ResidualBlock(64, dropout=0.1),   # +1 vs 20-cls model
            nn.MaxPool2d(2, 2),               # 32 → 16
        )

        # ── Multi-scale: stroke-width invariance (unchanged)
        self.multiscale = MultiScaleBlock(64, 128)

        # ── Block 3: part-level features — widened to 192 channels
        self.block3 = nn.Sequential(
            DepthwiseSeparable(128, 192),
            ResidualBlock(192, dropout=0.2),
            nn.MaxPool2d(2, 2),           # 16 → 8
        )

        # ── Block 4: abstract class features — widened to 384
        self.block4 = nn.Sequential(
            DepthwiseSeparable(192, 384),
            ResidualBlock(384, dropout=0.25),
            DepthwiseSeparable(384, 384),
            ResidualBlock(384, dropout=0.25),
            nn.MaxPool2d(2, 2),           # 8 → 4
        )

        # ── Global Average Pool
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B, 384, 4, 4) → (B, 384, 1, 1)

        # ── Classifier: 384 → 192 → num_classes
        #    192 >> 50 ensures plenty of separable dimensions per class
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)       # (B,  32, 32, 32)
        x = self.block2(x)       # (B,  64, 16, 16)
        x = self.multiscale(x)   # (B, 128, 16, 16)
        x = self.block3(x)       # (B, 192,  8,  8)
        x = self.block4(x)       # (B, 384,  4,  4)
        x = self.gap(x)          # (B, 384,  1,  1)
        return self.classifier(x)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(num_classes=50, device="cpu", dropout=0.4):
    model = SketchNet50(num_classes=num_classes, dropout=dropout).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : SketchNet-50 (from scratch, tuned for 50 classes)")
    print(f"Params  : {total:,} total | {trainable:,} trainable")
    print(f"Device  : {device}")

    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, 64, 64, device=device)
        print(f"\nForward pass shape trace (batch=1):")
        x = model.block1(x);     print(f"  After block1     : {tuple(x.shape)}")
        x = model.block2(x);     print(f"  After block2     : {tuple(x.shape)}")
        x = model.multiscale(x); print(f"  After multiscale : {tuple(x.shape)}")
        x = model.block3(x);     print(f"  After block3     : {tuple(x.shape)}")
        x = model.block4(x);     print(f"  After block4     : {tuple(x.shape)}")
        x = model.gap(x);        print(f"  After GAP        : {tuple(x.shape)}  (384-dim embedding)")
    print()

    model.train()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_model(num_classes=50, device=device)
