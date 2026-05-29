"""
multimodal_embeddings.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Multimodal Embedding Engine
Team Vulcan 629 | DSATM Bengaluru

Pipeline Stage: 6

Two embedding engines in one module:

1. TextEmbeddingEngine
   Model : all-MiniLM-L6-v2  (384-dim, 23 MB, fast on CPU)
   Input : text strings (task queries, affordance descriptions)
   Output: L2-normalised float32 vectors

2. CLIPEmbeddingEngine
   Model : openai/clip-vit-base-patch32  (151 MB)
   Input : image crops (object regions) + text strings
   Output: L2-normalised float32 vectors (512-dim)
   Purpose: visual similarity between image region and task text

Both engines use EmbeddingCache to avoid re-computing identical texts.
CLIP is lazy-loaded on first use to save memory if not needed.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import warnings
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image

warnings.filterwarnings("ignore")

from config import TEXT_EMBED_MODEL, CLIP_MODEL_NAME
from utils import (
    get_logger, EmbeddingCache, l2_normalize,
    cosine_similarity, scale_to_unit
)
from detector import Detection

log = get_logger("Embeddings")


# ─────────────────────────────────────────────────────────────────────
# TEXT EMBEDDING ENGINE
# ─────────────────────────────────────────────────────────────────────

class TextEmbeddingEngine:
    """
    Converts text to 384-dimensional L2-normalised float32 vectors.
    Uses sentence-transformers/all-MiniLM-L6-v2.

    Embedding cache ensures each unique affordance description
    is only encoded once per process lifetime (plus disk persistence).
    """

    def __init__(self, cache_dir: str = ".cache/text"):
        log.info("Loading text embedding model: %s", TEXT_EMBED_MODEL)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(TEXT_EMBED_MODEL)
        self._cache = EmbeddingCache(cache_dir)
        self._dim   = 384
        log.info("TextEmbeddingEngine ready (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string.
        Returns L2-normalised float32 ndarray of shape (384,).
        Uses cache — will NOT re-run the model for repeated inputs.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        vec = self._model.encode(text, normalize_embeddings=True)
        vec = vec.astype(np.float32)
        self._cache.put(text, vec)
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts efficiently.
        Cache-aware: only encodes texts not already cached.
        Returns array of shape (N, 384).
        """
        results  = [None] * len(texts)
        uncached_idx   = []
        uncached_texts = []

        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is not None:
                results[i] = v
            else:
                uncached_idx.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            vecs = self._model.encode(
                uncached_texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype(np.float32)
            for j, idx in enumerate(uncached_idx):
                self._cache.put(uncached_texts[j], vecs[j])
                results[idx] = vecs[j]

        return np.stack(results)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text strings (cached)."""
        return cosine_similarity(self.encode(text_a), self.encode(text_b))

    def cache_stats(self) -> str:
        return self._cache.stats()


# ─────────────────────────────────────────────────────────────────────
# CLIP EMBEDDING ENGINE
# ─────────────────────────────────────────────────────────────────────

class CLIPEmbeddingEngine:
    """
    Multimodal vision-language model (CLIP ViT-B/32).
    Encodes:
      - image crops (object regions from bounding boxes)
      - text strings (task queries)
    Into the same 512-dimensional embedding space.

    This means we can compute how similar an IMAGE REGION looks
    to a task description — pure visual-semantic alignment.

    FPGA note: in production the CLIP vision encoder would be
    quantised to INT8 and run on the systolic array. Here we
    run it in FP32 on CPU for functional correctness.
    """

    def __init__(self, cache_dir: str = ".cache/clip"):
        self._loaded   = False
        self._cache    = EmbeddingCache(cache_dir)
        self._dim      = 512
        self._model    = None
        self._processor = None
        log.info("CLIPEmbeddingEngine created (lazy load on first use)")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        log.info("Loading CLIP model: %s", CLIP_MODEL_NAME)
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self._model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self._model.eval()
        self._loaded = True
        log.info("CLIP model loaded (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text → 512-dim CLIP text embedding.
        Cached per unique string.
        """
        key = f"clip_text::{text}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._ensure_loaded()
        import torch
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        vec = l2_normalize(feats[0].numpy().astype(np.float32))
        self._cache.put(key, vec)
        return vec

    def encode_image_crop(
        self,
        image_path: str,
        bbox: List[int],
    ) -> np.ndarray:
        """
        Encode a cropped image region → 512-dim CLIP image embedding.
        bbox = [x1, y1, x2, y2] in pixels.

        Cache key = image_path + bbox, so the same crop won't
        be re-processed if the pipeline runs multiple tasks on one image.
        """
        key = f"clip_img::{image_path}::{bbox}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._ensure_loaded()
        import torch

        img = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        crop = img.crop((x1, y1, max(x1+1, x2), max(y1+1, y2)))

        inputs = self._processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        vec = l2_normalize(feats[0].numpy().astype(np.float32))
        self._cache.put(key, vec)
        return vec

    def image_text_similarity(
        self,
        image_path: str,
        bbox: List[int],
        text: str,
    ) -> float:
        """
        Compute cosine similarity between an image crop and a text string.
        Returns float in [0, 1] (scaled from [-1,1]).

        This is the core visual-semantic alignment score.
        """
        img_vec  = self.encode_image_crop(image_path, bbox)
        text_vec = self.encode_text(text)
        raw_sim  = cosine_similarity(img_vec, text_vec)
        return scale_to_unit(raw_sim, lo=-1.0, hi=1.0)

    def batch_image_text_similarity(
        self,
        image_path: str,
        detections: List["Detection"],
        text: str,
    ) -> List[float]:
        """
        Compute visual similarity for all detected objects at once.
        Returns list of float scores aligned with detections list.
        """
        text_vec = self.encode_text(text)
        scores   = []
        for det in detections:
            img_vec = self.encode_image_crop(image_path, det.bbox)
            raw     = cosine_similarity(img_vec, text_vec)
            scores.append(scale_to_unit(raw, lo=-1.0, hi=1.0))
        return scores

    def cache_stats(self) -> str:
        return self._cache.stats()