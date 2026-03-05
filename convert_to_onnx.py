#!/usr/bin/env python3
"""Convert SketchNet-50 PyTorch checkpoint → ONNX for browser inference."""

import sys, json, torch, onnx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import SketchNet50

def main():
    root = Path(__file__).resolve().parent.parent
    ckpt = torch.load(root / "best_model_50.pt", map_location="cpu", weights_only=False)
    categories = ckpt["categories"]
    num_classes = len(categories)

    model = SketchNet50(num_classes=num_classes, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(__file__).resolve().parent / "public" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 1, 64, 64)
    onnx_path = out_dir / "sketchnet50.onnx"

    # Use opset 18 — the dynamo exporter requires >= 18.
    # dynamic_shapes replaces the deprecated dynamic_axes for dynamo.
    batch = torch.export.Dim("batch")
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=18,
        do_constant_folding=True,
    )

    # Inline external data (sketchnet50.onnx.data) into the .onnx file so
    # that browsers can load a single file via onnxruntime-web.
    data_file = onnx_path.with_suffix(".onnx.data")
    m = onnx.load(str(onnx_path))
    onnx.save_model(m, str(onnx_path), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()
        print("Removed sidecar .data file — weights are now inline.")

    with open(out_dir / "labels.json", "w") as f:
        json.dump(categories, f, indent=2)

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"ONNX model : {onnx_path}  ({size_mb:.1f} MB)")
    print(f"Labels      : {out_dir / 'labels.json'}  ({num_classes} classes)")

if __name__ == "__main__":
    main()
