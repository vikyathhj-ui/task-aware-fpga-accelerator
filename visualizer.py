"""
visualizer.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Visualization Engine
Team Vulcan 629 | DSATM Bengaluru

Pipeline Stage: 9

Renders annotated output images showing:
  ██ GREEN  (thick)  : selected winner object
  ██ RED    (medium) : detected but score below rejection threshold
  ██ YELLOW (medium) : contextually relevant neighbour (context anchor)
  ██ GREY   (thin)   : detected but irrelevant to task

Overlays:
  - Task query banner at top
  - Per-object: class name, final score, prior label
  - Score breakdown bar (sem / vis / phys / ctx) as mini progress bars
  - Result banner at bottom (green=selected / red=none found)
  - Uncertainty indicator if uncertain decision was made
  - Hardware latency stats in corner

Output: saves output_task{N}.jpg to working directory.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import os
import numpy as np
from typing import List, Optional

from config import (
    REJECTION_THRESHOLD,
    VIZ_GREEN, VIZ_RED, VIZ_GREY, VIZ_YELLOW, VIZ_WHITE, VIZ_BLACK,
)
from scene_context import TASK_CONTEXT_ANCHORS
from scoring import ScoredObject, SelectionResult
from utils import get_logger

log = get_logger("Visualizer")


# ─────────────────────────────────────────────────────────────────────
# SCORE BAR RENDERER
# ─────────────────────────────────────────────────────────────────────

def _draw_score_bar(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    value: float,
    color: tuple,
    label: str,
    height: int = 5,
):
    """
    Draw a horizontal progress bar showing a score value in [0,1].
    Used to render the sem/vis/phys/ctx breakdown under each bbox.
    """
    # Background track
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Filled portion
    filled = max(1, int(width * np.clip(value, 0, 1)))
    cv2.rectangle(img, (x, y), (x + filled, y + height), color, -1)
    # Label
    cv2.putText(
        img, f"{label}:{value:.2f}",
        (x + width + 4, y + height),
        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
        VIZ_WHITE, 1, cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN VISUALIZER
# ─────────────────────────────────────────────────────────────────────

class Visualizer:
    """
    Renders the annotated output image for Stage 9.
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LABEL = 0.42
    FONT_SCALE_BANNER = 0.65
    FONT_SCALE_STATS  = 0.35

    def render(
        self,
        image_path:   str,
        result:       SelectionResult,
        task_id:      int,
        task_query:   str,
        latency_ms:   float,
        fpga_us:      float,
        output_path:  Optional[str] = None,
    ) -> str:
        """
        Render and save the annotated image.

        Args:
            image_path  : path to original input image
            result      : SelectionResult from RejectionGate
            task_id     : task number 1-14
            task_query  : task question string
            latency_ms  : total software pipeline latency (ms)
            fpga_us     : FPGA estimated similarity latency (µs)
            output_path : override output filename (default: output_task{N}.jpg)

        Returns:
            path to saved output image
        """
        img = cv2.imread(image_path)
        if img is None:
            log.error("Could not load image: %s", image_path)
            return ""

        h, w = img.shape[:2]

        # ── Top banner (task query) ────────────────────────────────────
        self._draw_top_banner(img, task_id, task_query, w)

        # ── Context anchors for this task ──────────────────────────────
        ctx_anchor_names = set(TASK_CONTEXT_ANCHORS.get(task_id, []))

        # ── Bounding boxes for all scored objects ──────────────────────
        winner = result.winner

        for obj in result.all_scored:
            is_winner   = (
                winner is not None
                and obj.class_name == winner.class_name
                and obj.bbox       == winner.bbox
            )
            is_ctx_anchor = obj.class_name in ctx_anchor_names and not is_winner
            is_rejected   = obj.final_score < result.threshold_used

            color, thickness = self._box_style(is_winner, is_ctx_anchor, is_rejected)
            self._draw_object(img, obj, color, thickness, result.threshold_used, w, h)

        # ── Bottom result banner ───────────────────────────────────────
        self._draw_bottom_banner(img, result, w, h)

        # ── Hardware stats overlay (top-right corner) ──────────────────
        self._draw_hw_stats(img, latency_ms, fpga_us, w)

        # ── Uncertainty badge ──────────────────────────────────────────
        if result.uncertain and not result.rejected:
            self._draw_uncertainty_badge(img, result.uncertainty, w)

        # ── Save ───────────────────────────────────────────────────────
        if output_path is None:
            output_path = f"output_task{task_id}.jpg"

        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.info("Visualization saved → %s", output_path)
        log.info(
            "Legend: Green=selected | Red=rejected | Yellow=context | Grey=ignored"
        )
        return output_path

    # ──────────────────────────────────────────────────────────────────

    def _box_style(
        self,
        is_winner:    bool,
        is_ctx_anchor: bool,
        is_rejected:   bool,
    ):
        if is_winner:
            return VIZ_GREEN, 3
        if is_ctx_anchor:
            return VIZ_YELLOW, 2
        if is_rejected:
            return VIZ_RED, 2
        return VIZ_GREY, 1

    def _draw_top_banner(self, img, task_id: int, task_query: str, w: int):
        """Semi-transparent dark bar at top with task query text."""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.putText(
            img,
            f"Task {task_id}: {task_query}",
            (10, 32),
            self.FONT, self.FONT_SCALE_BANNER,
            VIZ_WHITE, 2, cv2.LINE_AA,
        )

    def _draw_object(
        self,
        img:       np.ndarray,
        obj:       ScoredObject,
        color:     tuple,
        thickness: int,
        threshold: float,
        img_w:     int,
        img_h:     int,
    ):
        x1, y1, x2, y2 = obj.bbox
        # Clamp to image bounds
        x1, x2 = max(0, x1), min(img_w - 1, x2)
        y1, y2 = max(0, y1), min(img_h - 1, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # ── Label text ────────────────────────────────────────────────
        label = (
            f"{obj.class_name}  {obj.final_score:.2f}  {obj.prior_label}"
        )
        (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE_LABEL, 1)
        label_y = max(th + 10, y1 - 2)

        # Label background
        cv2.rectangle(
            img,
            (x1, label_y - th - 6),
            (x1 + tw + 6, label_y),
            color, -1,
        )
        cv2.putText(
            img, label,
            (x1 + 3, label_y - 3),
            self.FONT, self.FONT_SCALE_LABEL,
            VIZ_WHITE, 1, cv2.LINE_AA,
        )

        # ── Score breakdown bars (only for non-grey objects) ──────────
        if obj.final_score >= threshold * 0.5:
            bar_x  = x1
            bar_w  = min(80, x2 - x1 - 4)
            bar_y0 = y2 + 4

            if bar_y0 + 30 < img_h:
                bar_colors = [
                    ((100, 200, 255), obj.sem_score,  "S"),
                    ((255, 180, 80),  obj.vis_score,  "V"),
                    ((80, 255, 160),  obj.phys_score, "P"),
                    ((220, 80, 220),  obj.ctx_boost,  "C"),
                ]
                for k, (bc, val, lbl) in enumerate(bar_colors):
                    _draw_score_bar(
                        img,
                        bar_x, bar_y0 + k * 9,
                        bar_w, val, bc, lbl, height=5,
                    )

    def _draw_bottom_banner(
        self,
        img:    np.ndarray,
        result: SelectionResult,
        w:      int,
        h:      int,
    ):
        """Solid colour bar at bottom with selection result."""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 42), (w, h), VIZ_BLACK, -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

        if result.rejected:
            text  = "NO SUITABLE OBJECT FOUND FOR THIS TASK"
            color = VIZ_RED
        else:
            w_obj = result.winner
            text  = (
                f"SELECTED: {w_obj.class_name.upper()}"
                f"  score={w_obj.final_score:.3f}"
                f"  [{w_obj.prior_label}]"
            )
            color = VIZ_GREEN

        cv2.putText(
            img, text,
            (10, h - 14),
            self.FONT, self.FONT_SCALE_BANNER,
            color, 2, cv2.LINE_AA,
        )

    def _draw_hw_stats(
        self,
        img:        np.ndarray,
        latency_ms: float,
        fpga_us:    float,
        w:          int,
    ):
        """Small hardware stats panel at top-right corner."""
        lines = [
            f"CPU: {latency_ms:.0f} ms",
            f"FPGA(sim): {fpga_us:.1f} us",
        ]
        for i, line in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(line, self.FONT, self.FONT_SCALE_STATS, 1)
            x = w - tw - 6
            y = 60 + i * 16
            cv2.putText(
                img, line, (x, y),
                self.FONT, self.FONT_SCALE_STATS,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

    def _draw_uncertainty_badge(
        self,
        img: np.ndarray,
        U:   float,
        w:   int,
    ):
        """Yellow ⚠ badge for uncertain decisions."""
        badge = f"! UNCERTAIN (U={U:.2f})"
        (tw, th), _ = cv2.getTextSize(badge, self.FONT, self.FONT_SCALE_STATS, 1)
        cv2.rectangle(img, (w - tw - 12, 90), (w - 4, 110), (0, 180, 220), -1)
        cv2.putText(
            img, badge,
            (w - tw - 8, 106),
            self.FONT, self.FONT_SCALE_STATS,
            VIZ_BLACK, 1, cv2.LINE_AA,
        )
