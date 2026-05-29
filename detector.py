"""
detector.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Object Detection Module
Team Vulcan 629 | DSATM Bengaluru

Pipeline Stages covered: 2 (Conv1), 3 (Conv2), 4 (Conv3), 5 (Head+NMS)

Backend: YOLOv8-nano (CPU-only)
  - 3 convolution backbone layers (run on FPGA systolic array in Stage 3)
  - Detection head + NMS (runs on VEGA CPU)
  - 80 COCO classes + confidence + bounding boxes

Open-vocabulary expansion:
  - YOLO-World available as an upgrade (set DETECTOR_BACKEND="yolow")
  - Falls back to YOLOv8-nano if YOLO-World unavailable
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import cv2
from typing import List, Dict, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")

from config import (
    DETECTOR_BACKEND, YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD, YOLO_DEVICE, YOLO_IMAGE_SIZE,
    COCO_AFFORDANCES
)
from utils import get_logger, bbox_area, aspect_ratio

log = get_logger("Detector")


# ─────────────────────────────────────────────────────────────────────
# DETECTION RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One detected object with all metadata."""
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        List[int]         # [x1, y1, x2, y2] in pixels
    area:        float             # pixel area
    aspect:      float             # width / height
    affordance:  str               # rich text description for embedding

    def centre(self):
        return ((self.bbox[0]+self.bbox[2])//2,
                (self.bbox[1]+self.bbox[3])//2)

    def as_dict(self) -> Dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox":       self.bbox,
            "area":       self.area,
            "aspect":     self.aspect,
            "affordance": self.affordance,
        }


# ─────────────────────────────────────────────────────────────────────
# YOLO DETECTOR
# ─────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Wraps YOLOv8-nano (or YOLO-World) for CPU-only object detection.

    FPGA Architecture Note:
      Stages 2-4 (3 conv layers) will be offloaded to the 8×8 INT8
      systolic array on Genesys-2 in Stage 2B/3.
      Stage 5 (detection head + NMS) runs on VEGA RISC-V CPU
      via AXI4-Stream interface from the FPGA fabric.

    Convolution layer profile (YOLOv8-nano, 640×640 input):
      Conv1: [3×640×640]  →  [32×640×640]   354M MACs  S=1 P=1 K=3
      Conv2: [32×640×640] →  [64×320×320]  1887M MACs  S=2 P=1 K=3
      Conv3: [64×320×320] → [128×160×160]  1887M MACs  S=2 P=1 K=3
      Total backbone: 4,128M MACs → accelerated by systolic array
    """

    def __init__(self):
        log.info("Loading YOLO detector backend: %s", DETECTOR_BACKEND)
        self._model  = None
        self._backend = DETECTOR_BACKEND
        self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        if self._backend == "yolow":
            try:
                self._model = YOLO("yolov8s-world.pt")
                log.info("YOLO-World loaded (open-vocabulary mode)")
            except Exception as e:
                log.warning("YOLO-World unavailable (%s), falling back to YOLOv8n", e)
                self._backend = "yolov8n"

        if self._backend == "yolov8n":
            self._model = YOLO("yolov8n.pt")
            log.info("YOLOv8-nano loaded (80-class COCO mode)")

    def set_classes(self, class_names: List[str]) -> None:
        """
        Open-vocabulary: set detection targets dynamically.
        Only works with YOLO-World backend.
        """
        if self._backend == "yolow" and hasattr(self._model, "set_classes"):
            self._model.set_classes(class_names)
            log.info("YOLO-World classes set: %s", class_names)

    def detect(
        self,
        image_path: str,
        task_classes: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Run YOLO on an image. Returns list of Detection objects
        sorted by confidence descending.

        Args:
            image_path   : path to input image
            task_classes : optional list of class names for YOLO-World
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Optionally restrict detection to task-relevant classes
        if task_classes and self._backend == "yolow":
            self.set_classes(task_classes)

        t0      = time.perf_counter()
        results = self._model(
            image_path,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=YOLO_DEVICE,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                cls_id    = int(box.cls.item())
                cls_name  = results[0].names.get(cls_id, f"class_{cls_id}")
                conf      = float(box.conf.item())
                bbox      = [int(v) for v in box.xyxy[0].tolist()]
                area      = bbox_area(bbox)
                asp       = aspect_ratio(bbox)
                afford    = COCO_AFFORDANCES.get(
                    cls_name,
                    f"{cls_name} object found in the scene"
                )

                detections.append(Detection(
                    class_id   = cls_id,
                    class_name = cls_name,
                    confidence = conf,
                    bbox       = bbox,
                    area       = area,
                    aspect     = asp,
                    affordance = afford,
                ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        log.info(
            "Detection done: %d objects in %.0f ms (backend=%s)",
            len(detections), elapsed_ms, self._backend
        )
        return detections

    def detect_image_array(self, img: np.ndarray) -> List[Detection]:
        """
        Detect on a numpy array (HWC uint8 BGR) instead of file path.
        Used in batch evaluation mode.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name
        cv2.imwrite(tmp, img)
        dets = self.detect(tmp)
        os.unlink(tmp)
        return dets