"""
pipeline.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Main Pipeline Orchestrator
Team Vulcan 629 | DSATM Bengaluru

Entry point for the full 9-stage task-aware object detection system.

PIPELINE STAGES:
  Stage 1  : Image loading and preprocessing
  Stage 2  : YOLO Conv1  — edge detection backbone layer
  Stage 3  : YOLO Conv2  — texture detection backbone layer
  Stage 4  : YOLO Conv3  — part/object detection backbone layer
  Stage 5  : Detection head + NMS filtering
  Stage 6A : Text affordance embedding (TextEmbeddingEngine)
  Stage 6B : Physical affordance scoring (PhysicalAffordanceScorer)
  Stage 6C : CLIP visual-semantic similarity (CLIPEmbeddingEngine)
  Stage 6D : Affordance fusion (AffordanceFusionEngine)
  Stage 7  : INT8 systolic array similarity (SemanticSystolicArray)
  Stage 8A : Scene graph construction (SceneGraphBuilder)
  Stage 8B : Contextual boosting (ContextualBooster)
  Stage 8C : Physical constraint filtering (PhysicalConstraintFilter)
  Stage 8D : Hierarchical scoring (HierarchicalScorer)
  Stage 8E : Adaptive rejection gate (RejectionGate)
  Stage 9  : Visualization (Visualizer)

USAGE:
  python pipeline.py image.jpg --task 10
  python pipeline.py image.jpg --task 3 --threshold 0.30 --clip
  python pipeline.py image.jpg --task 7 --backend yolow

FLAGS:
  --task        INT   Task ID 1-14 (required)
  --threshold   FLOAT Override rejection threshold (default 0.35)
  --clip              Enable CLIP visual scoring (adds ~600ms on CPU)
  --backend     STR   yolov8n (default) or yolow (YOLO-World)
  --output      STR   Custom output image path
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from config import TASKS, FPGA, REJECTION_THRESHOLD
from utils import get_logger, StageTimer, MemoryMonitor
from detector import YOLODetector
from multimodal_embeddings import TextEmbeddingEngine, CLIPEmbeddingEngine
from affordance_engine import AffordanceFusionEngine
from systolic_array import SemanticSystolicArray
from scene_context import SceneGraphBuilder, ContextualBooster, PhysicalConstraintFilter
from scoring import HierarchicalScorer, RejectionGate, SelectionResult
from visualizer import Visualizer

log = get_logger("Pipeline")


# ─────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────

class TaskAwarePipeline:
    """
    Orchestrates all 9 stages of the DVCon India 2026 pipeline.

    Key improvements over the monolithic task14.py prototype:
      ✔ Five-component hierarchical scoring (not just semantic + conf)
      ✔ CLIP visual-semantic region scoring (optional)
      ✔ Physical affordance reasoning (generalises beyond COCO names)
      ✔ Contextual scene graph boost
      ✔ Adaptive rejection threshold based on entropy
      ✔ Physical constraint filter (safety rejection)
      ✔ Realistic INT8 systolic array with latency/GOPS model
      ✔ Embedding cache (disk + memory) — repeated images are fast
      ✔ Full modular decomposition across 10 separate Python files
    """

    def __init__(self, use_clip: bool = False, backend: str = "yolov8n"):
        self._banner()
        log.info("Initializing all pipeline stages...")

        self.detector        = YOLODetector()
        self.text_engine     = TextEmbeddingEngine()
        self.clip_engine     = CLIPEmbeddingEngine() if use_clip else None
        self.afford_engine   = AffordanceFusionEngine()
        self.systolic        = SemanticSystolicArray()
        self.scene_builder   = SceneGraphBuilder()
        self.ctx_booster     = ContextualBooster()
        self.constraint_filt = PhysicalConstraintFilter()
        self.scorer          = HierarchicalScorer()
        self.gate            = RejectionGate()
        self.viz             = Visualizer()

        self.use_clip = use_clip
        log.info("All stages ready.%s",
                 " (CLIP visual scoring ENABLED)" if use_clip else
                 " (CLIP disabled — use --clip to enable)")

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC RUN METHOD
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        image_path:    str,
        task_id:       int,
        threshold:     float = REJECTION_THRESHOLD,
        output_path:   str   = None,
    ) -> SelectionResult:
        """
        Run the full 9-stage pipeline on one image for one task.

        Args:
            image_path  : path to input image (.jpg or .png)
            task_id     : task ID 1-14
            threshold   : rejection threshold override
            output_path : output image path (default: output_task{N}.jpg)

        Returns:
            SelectionResult containing winner (or None) and all details.
        """

        if task_id not in TASKS:
            raise ValueError(f"Task ID must be 1–14, got {task_id}")

        task_name  = TASKS[task_id]["name"]
        task_query = TASKS[task_id]["query"]

        print("\n" + "═" * 65)
        print(f"  TASK {task_id}: {task_name.upper()}")
        print(f"  Query  : {task_query}")
        print("═" * 65)

        timer  = StageTimer()
        mem    = MemoryMonitor()
        mem.snapshot("start")

        # ── STAGE 1: Image Validation ──────────────────────────────────
        if not os.path.exists(image_path):
            log.error("Image not found: %s", image_path)
            return None

        img_h, img_w = self._image_hw(image_path)
        log.info("Stage 1: image %s  size=%dx%d", image_path, img_w, img_h)

        # ── STAGES 2-5: YOLO Detection ────────────────────────────────
        with timer.measure("Stage 2-5 YOLO detection"):
            detections = self.detector.detect(image_path)

        if not detections:
            log.warning("No objects detected. Exiting.")
            return SelectionResult(
                winner=None, all_scored=[], threshold_used=threshold,
                uncertainty=1.0, uncertain=False, rejected=True,
            )

        log.info(
            "Stage 2-5: %d objects detected in %.0f ms",
            len(detections), timer.get("Stage 2-5 YOLO detection"),
        )
        mem.snapshot("after_yolo")

        # ── STAGE 6A: Text Affordance Embeddings ──────────────────────
        with timer.measure("Stage 6A text embeddings"):
            task_vec  = self.text_engine.encode(task_query)
            obj_texts = [det.affordance for det in detections]
            obj_vecs  = self.text_engine.encode_batch(obj_texts)

        log.info("Stage 6A: encoded task + %d object affordances", len(detections))
        mem.snapshot("after_text_embed")

        # ── STAGE 6B: Physical Affordance Scoring ─────────────────────
        #   (done inside AffordanceFusionEngine below, needs sem scores first)

        # ── STAGE 7: Systolic Array Similarity ────────────────────────
        with timer.measure("Stage 7 systolic similarity"):
            sim_scores, hw_report = self.systolic.compute_similarity(
                task_vec, obj_vecs
            )

        log.info(
            "Stage 7: similarity done | FPGA est. %.2f µs | SW %.1f ms",
            hw_report.latency_us, hw_report.sw_latency_ms,
        )
        hw_report.print()

        # ── STAGE 6B+D: Affordance Fusion (sem + phys) ────────────────
        with timer.measure("Stage 6B+D affordance fusion"):
            fused_scores, phys_scores = self.afford_engine.fuse(
                text_semantic_scores = sim_scores.tolist(),
                detections           = detections,
                task_id              = task_id,
            )

        # ── STAGE 6C: CLIP Visual-Semantic Similarity (optional) ──────
        if self.use_clip and self.clip_engine is not None:
            with timer.measure("Stage 6C CLIP visual"):
                vis_scores = self.clip_engine.batch_image_text_similarity(
                    image_path, detections, task_query
                )
            log.info(
                "Stage 6C: CLIP visual scores computed in %.0f ms",
                timer.get("Stage 6C CLIP visual"),
            )
        else:
            # Without CLIP, use semantic score as stand-in for visual
            # (slightly discounted to reflect missing visual modality)
            vis_scores = [s * 0.85 for s in sim_scores.tolist()]

        mem.snapshot("after_embeddings")

        # ── STAGE 8A: Scene Graph ──────────────────────────────────────
        with timer.measure("Stage 8A scene graph"):
            graph = self.scene_builder.build(detections, (img_h, img_w))

        # ── STAGE 8B: Contextual Boosts ───────────────────────────────
        with timer.measure("Stage 8B context boost"):
            ctx_boosts = self.ctx_booster.compute_boosts(
                detections, graph, task_id
            )

        # ── STAGE 8C: Physical Constraint Filter ──────────────────────
        with timer.measure("Stage 8C constraints"):
            c_mults, c_reasons = self.constraint_filt.apply(
                detections, task_id
            )

        # ── STAGE 8D: Hierarchical Scoring ────────────────────────────
        with timer.measure("Stage 8D scoring"):
            self.gate.BASE_THRESHOLD = threshold
            scored = self.scorer.score_all(
                detections          = detections,
                task_id             = task_id,
                fused_sem_scores    = fused_scores,
                vis_scores          = vis_scores,
                phys_scores         = phys_scores,
                ctx_boosts          = ctx_boosts,
                constraint_mults    = c_mults,
                constraint_reasons  = c_reasons,
            )

        self.scorer.print_breakdown(scored, task_id)

        # ── STAGE 8E: Rejection Gate ───────────────────────────────────
        with timer.measure("Stage 8E rejection gate"):
            result = self.gate.evaluate(scored)

        total_ms = timer.total_ms()
        self.gate.print_result(result, task_query)

        print(timer.report())
        print(mem.report())
        print(f"\n  Total latency : {total_ms:.0f} ms")
        print(f"  Cache stats   : {self.text_engine.cache_stats()}\n")

        # ── STAGE 9: Visualization ─────────────────────────────────────
        with timer.measure("Stage 9 visualization"):
            self.viz.render(
                image_path  = image_path,
                result      = result,
                task_id     = task_id,
                task_query  = task_query,
                latency_ms  = total_ms,
                fpga_us     = hw_report.latency_us,
                output_path = output_path,
            )

        return result

    # ──────────────────────────────────────────────────────────────────

    def _image_hw(self, image_path: str):
        """Return (height, width) of image without full load."""
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return 480, 640
        return img.shape[:2]

    def _banner(self):
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  DVCon India 2026 — Task-Aware Object Detection (14 Tasks)     ║
║  Team Vulcan 629 | DSATM Bengaluru                             ║
╠══════════════════════════════════════════════════════════════════╣
║  PIPELINE  (9 Stages + Physical + Context + Uncertainty)       ║
║  Stage 1   → Image Input                                       ║
║  Stage 2-4 → YOLO Backbone Conv1/Conv2/Conv3                   ║
║  Stage 5   → Detection Head + NMS                              ║
║  Stage 6A  → Text Affordance Embedding (MiniLM-L6)            ║
║  Stage 6B  → Physical Affordance Scoring (7-dim vector)        ║
║  Stage 6C  → CLIP Visual-Semantic Similarity (optional)        ║
║  Stage 6D  → Affordance Fusion                                 ║
║  Stage 7   → INT8 Systolic Array Similarity (FPGA sim)         ║
║  Stage 8A  → Scene Graph Construction                          ║
║  Stage 8B  → Contextual Scene Boost                            ║
║  Stage 8C  → Physical Constraint Filter                        ║
║  Stage 8D  → Hierarchical 5-Component Scoring                  ║
║  Stage 8E  → Adaptive Rejection Gate (entropy-based)           ║
║  Stage 9   → Visualization                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  HARDWARE  : Genesys-2 (Artix-7) + VEGA AS1061 RISC-V         ║
║  FPGA SIM  : 8×8 INT8 systolic array, 150 MHz, 9.6 GOPS        ║
║  CPU ONLY  : No GPU required                                   ║
╚══════════════════════════════════════════════════════════════════╝
        """)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DVCon India 2026 — Task-Aware Object Detection Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "image",
        help="Path to input image (.jpg or .png)",
    )
    p.add_argument(
        "--task", "-t",
        type=int,
        required=True,
        choices=range(1, 15),
        metavar="TASK_ID",
        help=(
            "Task ID 1–14:\n"
            "  1=step_on       2=sit           3=place_flowers\n"
            "  4=potatoes_fire 5=water_plant   6=lemon_from_tea\n"
            "  7=dig_hole      8=open_beer     9=open_parcel\n"
            " 10=serve_wine   11=pour_sugar   12=smear_butter\n"
            " 13=extinguish   14=pound_carpet"
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=REJECTION_THRESHOLD,
        help=f"Rejection threshold [0-1] (default: {REJECTION_THRESHOLD}). "
             "Objects scoring below this are rejected.",
    )
    p.add_argument(
        "--clip",
        action="store_true",
        default=False,
        help="Enable CLIP visual-semantic scoring (Stage 6C). "
             "Adds ~500-800 ms on CPU but improves accuracy.",
    )
    p.add_argument(
        "--backend",
        choices=["yolov8n", "yolow"],
        default="yolov8n",
        help="Detection backend: yolov8n (default) or yolow (YOLO-World open-vocab).",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Custom output image path (default: output_task{N}.jpg).",
    )
    p.add_argument(
        "--tasks",
        action="store_true",
        default=False,
        help="List all 14 tasks and exit.",
    )
    return p


def list_tasks():
    print("\n  Available tasks:\n")
    for tid, info in TASKS.items():
        print(f"    {tid:>2}  {info['name']:<30}  {info['query']}")
    print()


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.tasks:
        list_tasks()
        sys.exit(0)

    # Override DETECTOR_BACKEND in config at runtime
    import config
    config.DETECTOR_BACKEND = args.backend

    pipeline = TaskAwarePipeline(
        use_clip = args.clip,
        backend  = args.backend,
    )

    result = pipeline.run(
        image_path  = args.image,
        task_id     = args.task,
        threshold   = args.threshold,
        output_path = args.output,
    )

    # Exit code: 0 = winner found, 1 = rejected
    sys.exit(0 if (result and not result.rejected) else 1)


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
