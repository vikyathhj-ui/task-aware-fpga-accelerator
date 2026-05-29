"""
utils.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Utility Functions
Team Vulcan 629 | DSATM Bengaluru

Shared utilities used across all 10 modules:
  - StageTimer   : measures per-stage wall-clock time
  - EmbeddingCache: disk + in-memory cache for embeddings
  - MemoryMonitor : tracks RAM usage
  - Logger        : coloured stage-wise logging
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import pickle
import logging
import functools
import psutil
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────
# STAGE TIMER
# ─────────────────────────────────────────────────────────────────────

@dataclass
class StageTimer:
    """
    Tracks wall-clock time for each pipeline stage.
    Usage:
        timer = StageTimer()
        with timer.measure("yolo"):
            run_yolo(...)
        print(timer.report())
    """
    _records: Dict[str, float] = field(default_factory=dict)
    _start:   Dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self._start[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        elapsed = (time.perf_counter() - self._start[stage]) * 1000  # ms
        self._records[stage] = elapsed
        return elapsed

    def get(self, stage: str) -> float:
        return self._records.get(stage, 0.0)

    def total_ms(self) -> float:
        return sum(self._records.values())

    def report(self) -> str:
        lines = ["  Stage Timings:"]
        for stage, ms in self._records.items():
            bar = "█" * min(30, int(ms / 10))
            lines.append(f"    {stage:<28} {ms:>8.1f} ms  {bar}")
        lines.append(f"    {'TOTAL':<28} {self.total_ms():>8.1f} ms")
        return "\n".join(lines)

    class _Context:
        def __init__(self, timer: "StageTimer", name: str):
            self._t, self._n = timer, name
        def __enter__(self):
            self._t.start(self._n)
            return self
        def __exit__(self, *_):
            self._t.stop(self._n)

    def measure(self, stage: str) -> "_Context":
        return self._Context(self, stage)


# ─────────────────────────────────────────────────────────────────────
# EMBEDDING CACHE
# ─────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Two-level cache for text embeddings:
      L1: in-process dict  (zero I/O, fastest)
      L2: disk pickle file (survives restarts)

    Key = SHA-256 of the input text string.
    This ensures that the same affordance description always
    returns the same precomputed vector without re-running the model.
    """

    def __init__(self, cache_dir: str = ".cache"):
        self._mem:   Dict[str, np.ndarray] = {}
        self._dir    = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hits   = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _disk_path(self, key: str) -> Path:
        return self._dir / f"{key}.pkl"

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)
        # L1
        if key in self._mem:
            self._hits += 1
            return self._mem[key]
        # L2
        dp = self._disk_path(key)
        if dp.exists():
            with open(dp, "rb") as f:
                vec = pickle.load(f)
            self._mem[key] = vec
            self._hits += 1
            return vec
        self._misses += 1
        return None

    def put(self, text: str, vec: np.ndarray) -> None:
        key = self._key(text)
        self._mem[key] = vec
        with open(self._disk_path(key), "wb") as f:
            pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)

    def stats(self) -> str:
        total = self._hits + self._misses
        rate  = self._hits / total * 100 if total else 0
        return (f"EmbeddingCache: hits={self._hits} misses={self._misses} "
                f"hit_rate={rate:.1f}% entries_in_mem={len(self._mem)}")


# ─────────────────────────────────────────────────────────────────────
# MEMORY MONITOR
# ─────────────────────────────────────────────────────────────────────

class MemoryMonitor:
    """Snapshots process RAM usage at key pipeline stages."""

    def __init__(self):
        self._proc    = psutil.Process(os.getpid())
        self._records: Dict[str, float] = {}

    def snapshot(self, label: str) -> float:
        mb = self._proc.memory_info().rss / 1e6
        self._records[label] = mb
        return mb

    def report(self) -> str:
        lines = ["  Memory Usage (RSS MB):"]
        for label, mb in self._records.items():
            lines.append(f"    {label:<30} {mb:>8.1f} MB")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    If both are already L2-normalised this equals their dot product.
    Returns float in [-1, 1].
    """
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def scale_to_unit(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Linearly rescale val from [lo, hi] to [0, 1]."""
    return float(np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax over a 1-D array."""
    x = np.asarray(x, dtype=np.float64) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def uncertainty(scores: np.ndarray) -> float:
    """
    Uncertainty estimate based on score entropy.
    High entropy → the model is unsure which object to pick.
    Low entropy  → confident about the top object.
    Returns value in [0, 1] where 1 = maximally uncertain.
    """
    if len(scores) < 2:
        return 0.0
    p    = softmax(scores)
    eps  = 1e-8
    H    = -np.sum(p * np.log(p + eps))
    Hmax = np.log(len(scores) + eps)
    return float(H / Hmax)


def bbox_iou(a: list, b: list) -> float:
    """
    Intersection over Union for two bounding boxes.
    Each box is [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def bbox_centre(bbox: list) -> tuple:
    """Return (cx, cy) centre of a bounding box [x1,y1,x2,y2]."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_area(bbox: list) -> float:
    """Return pixel area of a bounding box [x1,y1,x2,y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def centre_distance(a: list, b: list) -> float:
    """Euclidean distance between centres of two bounding boxes."""
    ca = bbox_centre(a)
    cb = bbox_centre(b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def aspect_ratio(bbox: list) -> float:
    """Width / Height of bounding box. >1 = wide, <1 = tall."""
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    return w / h