"""
download_dataset.py — Parallel Quick Draw .npy downloader for the 50 SketchNet classes

Downloads exactly the 50 categories used by train.py.
Saves to data/npy/<category>.npy  (white-spaces preserved, e.g. "hot dog.npy")
to match dataset.py's loader convention.

Usage:
    python download_dataset.py                 # 16 parallel workers
    python download_dataset.py --workers 32     # faster on high-speed internet
    python download_dataset.py --out data/npy   # custom output dir
"""

import argparse
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Exactly the 50 categories used by train.py

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

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


# Workers

def download_one(category: str, out_dir: Path) -> tuple[str, str, float]:
    """
    Download one .npy file. Filename preserves spaces to match dataset.py:
        "hot dog" → data/npy/hot dog.npy
    Returns (category, status, elapsed_seconds).
    """
    filename = category + ".npy"
    out_path = out_dir / filename
    url      = f"{BASE_URL}/{urllib.parse.quote(category)}.npy"

    if out_path.exists() and out_path.stat().st_size > 0:
        return category, "skip", 0.0

    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(out_path, "wb") as f:
            while chunk := resp.read(1 << 16):   # 64 KB chunks
                f.write(chunk)
        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / 1e6
        return category, f"ok  ({size_mb:.1f} MB, {elapsed:.1f}s)", elapsed
    except Exception as exc:
        if out_path.exists():
            out_path.unlink()
        return category, f"error: {exc}", time.time() - t0


# Main

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total   = len(CATEGORIES_50)
    pending = [c for c in CATEGORIES_50
               if not (out_dir / (c + ".npy")).exists()
               or (out_dir / (c + ".npy")).stat().st_size == 0]
    already = total - len(pending)

    print(f"\n{'═'*60}")
    print(f"  Quick Draw Downloader — SketchNet-50 dataset")
    print(f"  Output dir  : {out_dir.resolve()}")
    print(f"  Workers     : {args.workers}")
    print(f"  Total       : {total} categories")
    print(f"  Already done: {already}")
    print(f"  To download : {len(pending)}")
    print(f"{'═'*60}\n")

    if not pending:
        print("All 50 files already present. Nothing to download.\n")
        _print_summary(out_dir)
        return

    t_start  = time.time()
    errors   = []
    done     = 0
    max_name = max(len(c) for c in pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, cat, out_dir): cat for cat in pending}

        for fut in as_completed(futures):
            cat, status, _ = fut.result()
            done += 1

            tag = ("SKIP" if status.startswith("skip") else
                   "FAIL" if status.startswith("error") else " OK ")

            if status.startswith("error"):
                errors.append((cat, status))

            elapsed = time.time() - t_start
            pct     = done / len(pending) * 100
            filled  = int(20 * done / len(pending))
            bar     = "█" * filled + "░" * (20 - filled)
            eta     = (elapsed / done) * (len(pending) - done) if done > 0 else 0

            print(f"[{tag}] {cat:<{max_name}}  |{bar}| {pct:5.1f}%  "
                  f"ETA {eta:4.0f}s  {status}", flush=True)

    print(f"\n{'═'*60}")
    print(f"  Finished in {time.time() - t_start:.1f}s")
    print(f"  Downloaded  : {len(pending) - len(errors)}")
    print(f"  Skipped     : {already}")
    print(f"  Errors      : {len(errors)}")

    if errors:
        print(f"\n  Failed:")
        for cat, msg in errors:
            print(f"    ✗ {cat:<20}  {msg}")
        print()

    _print_summary(out_dir)

    if errors:
        print("Re-run the script to retry failed downloads.\n")
        sys.exit(1)


def _print_summary(out_dir: Path):
    present = [c for c in CATEGORIES_50 if (out_dir / (c + ".npy")).exists()]
    missing = [c for c in CATEGORIES_50 if c not in present]
    total_mb = sum((out_dir / (c + ".npy")).stat().st_size
                   for c in present) / 1e6

    print(f"\n  Files on disk : {len(present)}/{len(CATEGORIES_50)}")
    print(f"  Total size    : {total_mb:.0f} MB  ({total_mb/1024:.2f} GB)")
    if missing:
        print(f"\n  Still missing ({len(missing)}):")
        for c in missing:
            print(f"    • {c}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download 50 Quick Draw .npy files for SketchNet-50"
    )
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel download threads (default: 16)")
    p.add_argument("--out",     type=str, default="data/npy",
                   help="Output directory (default: data/npy)")
    main(p.parse_args())

