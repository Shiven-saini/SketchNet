"""
inference.py — CNN inference for Kalakari — 50 Quick Draw classes

Uses SketchNet (block3=192ch, block4=384ch) — must match train.py checkpoint.

Train with:
    python train.py

Run backend:
    uvicorn inference:app --host 0.0.0.0 --port 8000 --reload

Sanity check (no server):
    python inference.py --sanity
"""

import sys
from pathlib import Path

# allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from model import SketchNet50

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

DEFAULT_CHECKPOINT = "best_model_50.pt"


# Stroke → bitmap

def render_strokes_to_tensor(strokes, img_size=64):
    """
    Convert raw canvas strokes to a normalized CNN input tensor.

    Args:
        strokes  : [[xs, ys], ...]  raw canvas pixel coordinates (any scale)
        img_size : must match --img_size used during training (default 64)

    Returns:
        tensor   : (1, 1, img_size, img_size) float32 in [-1, 1]
    """
    img = np.zeros((img_size, img_size), dtype=np.float32)

    if not strokes:
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    all_x = [x for s in strokes for x in s[0]]
    all_y = [y for s in strokes for y in s[1]]

    if not all_x:
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    span  = max(max_x - min_x, max_y - min_y, 1.0)
    pad   = img_size * 0.1
    scale = (img_size - 2 * pad) / span

    for stroke in strokes:
        xs = [int(np.clip((x - min_x) * scale + pad, 0, img_size - 1)) for x in stroke[0]]
        ys = [int(np.clip((y - min_y) * scale + pad, 0, img_size - 1)) for y in stroke[1]]
        for i in range(len(xs) - 1):
            cv2.line(img, (xs[i], ys[i]), (xs[i+1], ys[i+1]),
                     color=1.0, thickness=2)

    img = img * 2.0 - 1.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


# Predictor

class QuickDraw50Predictor:
    """
    Loads a SketchNet checkpoint and runs inference on canvas strokes.
    Checkpoint must have been saved by train_50/train_50.py (arch=SketchNet50).
    """

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)

        arch = ckpt.get("arch", "")
        if arch and arch != "SketchNet50":
            raise ValueError(
                f"Checkpoint arch='{arch}' — this inference file requires "
                f"a SketchNet50 checkpoint produced by train_50/train_50.py"
            )

        self.categories = ckpt["categories"]
        self.idx2label  = {i: c for i, c in enumerate(self.categories)}
        self.img_size   = ckpt.get("args", {}).get("img_size", 64)
        num_classes     = len(self.categories)

        self.model = SketchNet50(
            num_classes=num_classes,
            dropout=0.0,           # no dropout at inference
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Kalakari SketchNet Predictor ready")
        print(f"  Classes    : {num_classes}")
        print(f"  Img size   : {self.img_size}×{self.img_size}")
        print(f"  Device     : {self.device}")
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    @torch.no_grad()
    def predict(self, strokes, top_k=10):
        """
        Args:
            strokes : [[xs, ys], ...] raw canvas strokes
            top_k   : number of top predictions to return

        Returns:
            list of {"label": str, "prob": float} sorted descending
        """
        tensor  = render_strokes_to_tensor(strokes, self.img_size).to(self.device)
        probs   = F.softmax(self.model(tensor), dim=-1)[0].cpu().numpy()
        top_k   = min(top_k, len(self.categories))
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [{"label": self.idx2label[i], "prob": float(probs[i])}
                for i in top_idx]

    @torch.no_grad()
    def predict_batch(self, stroke_list, top_k=10):
        """
        Batch inference — one prediction list per drawing.

        Args:
            stroke_list : list of stroke sequences
            top_k       : top-k per drawing
        """
        tensors = torch.cat([
            render_strokes_to_tensor(s, self.img_size) for s in stroke_list
        ], dim=0).to(self.device)

        probs_batch = F.softmax(self.model(tensors), dim=-1).cpu().numpy()
        top_k = min(top_k, len(self.categories))

        results = []
        for probs in probs_batch:
            top_idx = np.argsort(probs)[::-1][:top_k]
            results.append([
                {"label": self.idx2label[i], "prob": float(probs[i])}
                for i in top_idx
            ])
        return results


# FastAPI app

app = FastAPI(title="Kalakari 50-Class CNN API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = QuickDraw50Predictor()


class StrokeRequest(BaseModel):
    strokes: List[List[List[float]]]   # [[xs, ys], ...]
    top_k: int = 10


class BatchStrokeRequest(BaseModel):
    drawings: List[List[List[List[float]]]]
    top_k: int = 10


@app.post("/predict")
def predict(req: StrokeRequest):
    results = predictor.predict(req.strokes, req.top_k)
    return {"predictions": results}


@app.post("/predict_batch")
def predict_batch(req: BatchStrokeRequest):
    if len(req.drawings) > 64:
        raise HTTPException(status_code=400, detail="Max 64 drawings per batch.")
    results = predictor.predict_batch(req.drawings, req.top_k)
    return {"predictions": results}


@app.get("/categories")
def get_categories():
    return {"count": len(predictor.categories), "categories": predictor.categories}


@app.get("/health")
def health():
    return {"status": "ok", "model": "SketchNet", "classes": len(predictor.categories)}


# CLI / direct run

if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="Kalakari 50-class inference server")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--port",       type=int, default=8000)
    p.add_argument("--sanity",     action="store_true",
                   help="Run a quick prediction test and exit")
    args = p.parse_args()

    pred = QuickDraw50Predictor(checkpoint_path=args.checkpoint)

    if args.sanity:
        test_strokes = [
            [list(range(10, 90, 2)), [50] * 40],
            [[50, 50], [10, 90]],
        ]
        print("\nSanity check (top 10):")
        for r in pred.predict(test_strokes, top_k=10):
            bar = "█" * int(r["prob"] * 40)
            print(f"  {r['label']:<16} {r['prob']*100:5.1f}%  {bar}")
        print()
    else:
        print(f"\nStarting server → http://{args.host}:{args.port}")
        print(f"Docs             → http://localhost:{args.port}/docs\n")
        uvicorn.run("inference:app", host=args.host, port=args.port, reload=False)
