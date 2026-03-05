# SketchNet (Quick Draw 50-Class Classifier)

SketchNet is a lightweight CNN for classifying hand-drawn sketches from the Quick Draw dataset.
It is trained on 50 categories and uses 64x64 grayscale bitmap inputs.

## Architecture (SketchNet50)

- Input: `(B, 1, 64, 64)`
- Block 1: `ConvBNReLU(1->32)` + `ResidualBlock(32)` + MaxPool
- Block 2: `ConvBNReLU(32->64)` + `ResidualBlock(64) x3` + MaxPool
- Multi-scale block: Inception-style 4-branch fusion `64->128`
- Block 3: `DepthwiseSeparable(128->192)` + `ResidualBlock(192)` + MaxPool
- Block 4: `DepthwiseSeparable(192->384)` + `ResidualBlock(384)` + `DepthwiseSeparable(384->384)` + `ResidualBlock(384)` + MaxPool
- Head: Global Average Pool + `Linear(384->192->50)`
- Parameter count: ~2.1M

## Project Files

- `model.py`: SketchNet50 architecture and model factory
- `dataset.py`: Quick Draw `.npy` loading, augmentation, dataloaders
- `train.py`: training pipeline and checkpointing
- `inference.py`: FastAPI inference server and sanity CLI
- `download_dataset.py`: downloader for the 50 Quick Draw classes
- `convert_to_onnx.py`: export trained model to ONNX

## Quick Start

```bash
uv sync
uv run download_dataset.py --out data/npy
uv run train.py --data_dir data/npy
uv run inference.py --checkpoint checkpoints/best_model.pt --host 0.0.0.0 --port 8000
```

## Contact

Shiven Saini  
shiven.career@proton.me
