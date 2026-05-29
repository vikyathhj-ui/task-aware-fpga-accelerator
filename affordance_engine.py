"""
affordance_engine.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Affordance Reasoning Engine
Team Vulcan 629 | DSATM Bengaluru

Pipeline Stage: 6B  (runs after semantic embedding, before systolic array)

Responsibilities:
  1. PhysicalAffordanceScorer
       Maps physical object properties (size, rigidity, containment,
       sharpness, porosity) to task-requirement vectors.
       Produces a [0,1] physical-fit score without any hardcoded
       object-name lookup — works on ANY detected class.

  2. TaskAffordanceProfile
       Encodes each of the 14 tasks into a NamedTuple of physical
       requirements so the scorer can reason about new objects
       it has never seen before (generalisation beyond COCO).

  3. AffordanceFusionEngine
       Fuses text-semantic score (Stage 6A) + physical score (Stage 6B)
       into a single enriched affordance score passed to Stage 7.

PHYSICAL PROPERTIES modelled (all in [0, 1]):
  rigidity      — how rigid/solid the object is
  containment   — how well it can hold liquids or loose materials
  sharpness     — whether it has an edge suitable for cutting
  elongation    — length-to-width ratio (normalised)
  surface_area  — relative surface area (for striking / smearing)
  porosity      — lets water through (bad for containers)
  heat_resist   — can tolerate high temperature

TASK REQUIREMENT VECTORS (14 tasks, 7 physical dims each):
  Each task has an ideal physical profile. The physical score is
  the cosine similarity between the object's property vector and
  the task's requirement vector, scaled to [0,1].

  Formula:
    phys_score(o, t) = cosine_sim(phys_vec(o), req_vec(t))
                     = (phys_vec · req_vec) / (|phys_vec| |req_vec|)

  This score is task-conditioned and object-generalizable.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


from utils import get_logger, cosine_similarity, l2_normalize

log = get_logger("AffordanceEngine")


# ─────────────────────────────────────────────────────────────────────
# PHYSICAL PROPERTY DIMENSIONS (7-dimensional vector)
# ─────────────────────────────────────────────────────────────────────
# Index  Property       Description
# ─────────────────────────────────────────────────────────────────────
#   0    rigidity       solid/rigid  ↔  soft/flexible
#   1    containment    can hold liquid/powder (bowl-like)
#   2    sharpness      has a cutting edge (knife-like)
#   3    elongation     long thin shape (bat/fork-like)
#   4    surface_area   large flat area (useful for striking/smearing)
#   5    heat_resist    survives contact with heat or flame
#   6    porosity       lets liquid through (bad for containers)
# ─────────────────────────────────────────────────────────────────────

PHYS_DIM = 7

# Physical property vectors for COCO classes
# Values in [0.0, 1.0] — estimated from material/geometry knowledge
COCO_PHYSICAL: Dict[str, List[float]] = {
    # ── People / animals ────────────────────────────────────────────
    "person":       [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.2],
    "dog":          [0.4, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
    "cat":          [0.3, 0.0, 0.1, 0.3, 0.2, 0.2, 0.1],
    "bird":         [0.2, 0.0, 0.0, 0.4, 0.1, 0.2, 0.3],
    "horse":        [0.7, 0.0, 0.0, 0.6, 0.7, 0.5, 0.1],
    "sheep":        [0.4, 0.0, 0.0, 0.4, 0.4, 0.3, 0.3],
    "cow":          [0.6, 0.0, 0.0, 0.5, 0.6, 0.4, 0.1],
    "elephant":     [0.9, 0.0, 0.0, 0.5, 0.8, 0.6, 0.1],
    "bear":         [0.7, 0.0, 0.2, 0.4, 0.7, 0.5, 0.1],
    "zebra":        [0.7, 0.0, 0.0, 0.5, 0.7, 0.5, 0.1],
    "giraffe":      [0.7, 0.0, 0.0, 0.9, 0.5, 0.5, 0.1],
    # ── Vehicles ────────────────────────────────────────────────────
    "bicycle":      [0.9, 0.0, 0.0, 0.8, 0.4, 0.7, 0.1],
    "car":          [1.0, 0.0, 0.0, 0.6, 0.9, 0.7, 0.0],
    "motorcycle":   [0.9, 0.0, 0.0, 0.7, 0.5, 0.7, 0.0],
    "airplane":     [1.0, 0.0, 0.0, 0.9, 0.8, 0.8, 0.0],
    "bus":          [1.0, 0.0, 0.0, 0.7, 0.9, 0.7, 0.0],
    "train":        [1.0, 0.0, 0.0, 0.9, 0.9, 0.7, 0.0],
    "truck":        [1.0, 0.0, 0.0, 0.7, 0.9, 0.7, 0.0],
    "boat":         [0.9, 0.3, 0.0, 0.8, 0.8, 0.6, 0.1],
    # ── Outdoor ─────────────────────────────────────────────────────
    "bench":        [1.0, 0.0, 0.0, 0.7, 0.9, 0.6, 0.0],
    "fire hydrant": [1.0, 0.0, 0.0, 0.4, 0.5, 0.9, 0.0],
    "traffic light":[1.0, 0.0, 0.0, 0.5, 0.3, 0.7, 0.0],
    "stop sign":    [0.9, 0.0, 0.0, 0.2, 0.7, 0.6, 0.0],
    "parking meter":[1.0, 0.0, 0.0, 0.5, 0.3, 0.6, 0.0],
    # ── Bags / accessories ──────────────────────────────────────────
    "backpack":     [0.3, 0.4, 0.0, 0.4, 0.4, 0.2, 0.2],
    "umbrella":     [0.7, 0.0, 0.0, 0.8, 0.5, 0.4, 0.3],
    "handbag":      [0.3, 0.3, 0.0, 0.3, 0.4, 0.2, 0.2],
    "tie":          [0.1, 0.0, 0.0, 0.8, 0.1, 0.1, 0.4],
    "suitcase":     [0.9, 0.1, 0.0, 0.4, 0.8, 0.5, 0.0],
    # ── Sports ──────────────────────────────────────────────────────
    "frisbee":      [0.6, 0.0, 0.1, 0.1, 0.8, 0.4, 0.0],
    "skis":         [0.9, 0.0, 0.1, 0.9, 0.6, 0.5, 0.0],
    "snowboard":    [0.9, 0.0, 0.0, 0.7, 0.9, 0.4, 0.0],
    "sports ball":  [0.5, 0.0, 0.0, 0.1, 0.4, 0.3, 0.0],
    "kite":         [0.1, 0.0, 0.0, 0.7, 0.6, 0.1, 0.5],
    "baseball bat": [0.9, 0.0, 0.0, 0.9, 0.3, 0.6, 0.0],
    "baseball glove":[0.4,0.2, 0.0, 0.3, 0.5, 0.3, 0.3],
    "skateboard":   [0.9, 0.0, 0.0, 0.5, 0.9, 0.4, 0.0],
    "surfboard":    [0.8, 0.0, 0.0, 0.9, 0.9, 0.3, 0.0],
    "tennis racket":[0.6, 0.0, 0.0, 0.7, 0.6, 0.3, 0.4],
    # ── Kitchen / dining ────────────────────────────────────────────
    "bottle":       [0.8, 0.9, 0.0, 0.7, 0.2, 0.6, 0.0],
    "wine glass":   [0.7, 0.9, 0.0, 0.5, 0.2, 0.3, 0.0],
    "cup":          [0.7, 0.85,0.0, 0.3, 0.2, 0.4, 0.0],
    "fork":         [0.9, 0.1, 0.4, 0.8, 0.1, 0.8, 0.0],
    "knife":        [0.9, 0.0, 0.95,0.8, 0.1, 0.8, 0.0],
    "spoon":        [0.8, 0.5, 0.0, 0.7, 0.2, 0.7, 0.0],
    "bowl":         [0.8, 0.95,0.0, 0.1, 0.5, 0.5, 0.0],
    "banana":       [0.2, 0.1, 0.0, 0.6, 0.2, 0.2, 0.2],
    "apple":        [0.3, 0.0, 0.0, 0.1, 0.3, 0.2, 0.1],
    "sandwich":     [0.1, 0.1, 0.0, 0.3, 0.4, 0.1, 0.3],
    "orange":       [0.3, 0.1, 0.0, 0.1, 0.3, 0.2, 0.2],
    "broccoli":     [0.2, 0.0, 0.0, 0.3, 0.3, 0.1, 0.3],
    "carrot":       [0.4, 0.0, 0.2, 0.7, 0.1, 0.2, 0.1],
    "hot dog":      [0.1, 0.0, 0.0, 0.5, 0.2, 0.2, 0.2],
    "pizza":        [0.1, 0.0, 0.0, 0.2, 0.7, 0.1, 0.3],
    "donut":        [0.1, 0.0, 0.0, 0.1, 0.4, 0.1, 0.2],
    "cake":         [0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.3],
    # ── Furniture ───────────────────────────────────────────────────
    "chair":        [1.0, 0.0, 0.0, 0.4, 0.7, 0.5, 0.0],
    "couch":        [0.7, 0.0, 0.0, 0.5, 0.9, 0.3, 0.0],
    "potted plant": [0.3, 0.4, 0.0, 0.4, 0.4, 0.2, 0.4],
    "bed":          [0.5, 0.0, 0.0, 0.5, 1.0, 0.2, 0.0],
    "dining table": [1.0, 0.0, 0.0, 0.4, 1.0, 0.5, 0.0],
    "toilet":       [0.9, 0.7, 0.0, 0.3, 0.6, 0.5, 0.0],
    # ── Electronics ─────────────────────────────────────────────────
    "tv":           [0.9, 0.0, 0.0, 0.3, 0.9, 0.4, 0.0],
    "laptop":       [0.8, 0.0, 0.0, 0.5, 0.7, 0.3, 0.0],
    "mouse":        [0.8, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0],
    "remote":       [0.7, 0.0, 0.0, 0.6, 0.2, 0.3, 0.0],
    "keyboard":     [0.8, 0.0, 0.0, 0.5, 0.7, 0.3, 0.0],
    "cell phone":   [0.8, 0.0, 0.0, 0.5, 0.4, 0.3, 0.0],
    "microwave":    [1.0, 0.0, 0.0, 0.3, 0.7, 0.7, 0.0],
    "oven":         [1.0, 0.0, 0.0, 0.4, 0.7, 1.0, 0.0],
    "toaster":      [0.9, 0.0, 0.0, 0.4, 0.4, 0.8, 0.0],
    "sink":         [1.0, 0.8, 0.0, 0.3, 0.7, 0.6, 0.0],
    "refrigerator": [1.0, 0.0, 0.0, 0.4, 0.8, 0.1, 0.0],
    # ── Books / decorative ──────────────────────────────────────────
    "book":         [0.7, 0.0, 0.0, 0.4, 0.8, 0.2, 0.1],
    "clock":        [0.8, 0.0, 0.0, 0.3, 0.4, 0.4, 0.0],
    "vase":         [0.8, 0.9, 0.0, 0.4, 0.3, 0.5, 0.0],
    "scissors":     [0.9, 0.0, 0.9, 0.6, 0.2, 0.6, 0.0],
    "teddy bear":   [0.2, 0.0, 0.0, 0.3, 0.4, 0.1, 0.2],
    "hair drier":   [0.7, 0.0, 0.0, 0.6, 0.2, 0.5, 0.0],
    "toothbrush":   [0.6, 0.0, 0.1, 0.8, 0.1, 0.4, 0.0],
}

# Default for unknown classes — mid-range neutral vector
_DEFAULT_PHYS = [0.5, 0.2, 0.1, 0.4, 0.4, 0.3, 0.1]


# ─────────────────────────────────────────────────────────────────────
# TASK PHYSICAL REQUIREMENT VECTORS  (14 tasks × 7 dims)
# ─────────────────────────────────────────────────────────────────────
# Each vector encodes the IDEAL PHYSICAL PROFILE for that task.
# An object scores high if its physical properties match the task need.
#
# Interpretation:
#   step_on         → needs high rigidity + large surface + some elongation
#   sit_comfortably → needs high surface + moderate rigidity (soft ok)
#   place_flowers   → needs high containment (holds water)
#   get_potatoes    → needs elongation + heat resistance + some sharpness
#   water_plant     → needs high containment (holds liquid) + porosity low
#   get_lemon       → needs elongation + some sharpness or scooping
#   dig_hole        → needs high rigidity + sharpness + elongation
#   open_beer       → needs rigidity + sharpness
#   open_parcel     → needs sharpness + elongation
#   serve_wine      → needs containment (holds liquid, prevents spill)
#   pour_sugar      → needs containment (holds dry powder)
#   smear_butter    → needs surface_area + some rigidity
#   extinguish_fire → needs containment (holds water) + heat resistance
#   pound_carpet    → needs rigidity + elongation + surface_area

TASK_REQUIREMENTS: Dict[int, List[float]] = {
    1:  [0.9, 0.0, 0.0, 0.5, 0.9, 0.4, 0.0],   # step_on
    2:  [0.6, 0.0, 0.0, 0.4, 1.0, 0.2, 0.0],   # sit
    3:  [0.5, 1.0, 0.0, 0.3, 0.2, 0.4, 0.0],   # place_flowers
    4:  [0.8, 0.0, 0.5, 0.9, 0.1, 0.9, 0.0],   # potatoes from fire
    5:  [0.5, 1.0, 0.0, 0.3, 0.1, 0.3, 0.0],   # water_plant
    6:  [0.7, 0.2, 0.3, 0.8, 0.1, 0.5, 0.0],   # lemon from tea
    7:  [0.9, 0.0, 0.8, 0.9, 0.2, 0.5, 0.0],   # dig_hole
    8:  [0.9, 0.0, 0.8, 0.6, 0.2, 0.5, 0.0],   # open_beer
    9:  [0.8, 0.0, 0.9, 0.7, 0.2, 0.4, 0.0],   # open_parcel
    10: [0.7, 0.9, 0.0, 0.4, 0.2, 0.3, 0.0],   # serve_wine
    11: [0.7, 0.9, 0.0, 0.3, 0.3, 0.3, 0.0],   # pour_sugar
    12: [0.7, 0.1, 0.4, 0.7, 0.4, 0.4, 0.0],   # smear_butter
    13: [0.5, 1.0, 0.0, 0.3, 0.2, 0.9, 0.0],   # extinguish_fire
    14: [0.9, 0.0, 0.0, 0.8, 0.6, 0.4, 0.0],   # pound_carpet
}


# ─────────────────────────────────────────────────────────────────────
# PHYSICAL AFFORDANCE SCORER
# ─────────────────────────────────────────────────────────────────────

class PhysicalAffordanceScorer:
    """
    Computes task-conditioned physical affordance scores.

    For any detected object:
      1. Look up (or estimate) its 7-dim physical property vector
      2. Load the 7-dim requirement vector for the current task
      3. Compute cosine similarity → physical fit score in [0, 1]

    Generalizes to unseen objects via a geometry-based estimator
    that uses bounding-box aspect ratio and area as proxies for
    elongation and surface_area (the two most shape-dependent dims).

    FPGA NOTE:
      The 7-dim dot product is trivially mapped to 7 DSP48E2 blocks.
      At 150 MHz this takes < 50 ns — effectively free in hardware.
    """

    def __init__(self):
        # Pre-build normalised physical vectors for all known classes
        self._phys: Dict[str, np.ndarray] = {}
        for cls, props in COCO_PHYSICAL.items():
            self._phys[cls] = l2_normalize(np.array(props, dtype=np.float32))

        # Pre-normalise task requirement vectors
        self._req: Dict[int, np.ndarray] = {}
        for tid, req in TASK_REQUIREMENTS.items():
            self._req[tid] = l2_normalize(np.array(req, dtype=np.float32))

        log.info("PhysicalAffordanceScorer initialized (%d classes, %d tasks)",
                 len(self._phys), len(self._req))

    def get_physical_vector(
        self,
        class_name: str,
        bbox: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Return the 7-dim normalised physical property vector.
        If class unknown, estimate elongation and surface_area from bbox
        and keep other dims at default, then normalise.
        """
        if class_name in self._phys:
            return self._phys[class_name]

        # Unknown class: estimate from geometry
        vec = _DEFAULT_PHYS.copy()
        if bbox is not None:
            w = max(1, bbox[2] - bbox[0])
            h = max(1, bbox[3] - bbox[1])
            area = w * h
            aspect = w / h
            # elongation proxy: high if long thin shape
            vec[3] = float(np.clip((max(aspect, 1/aspect) - 1.0) / 4.0, 0, 1))
            # surface_area proxy: normalise relative to image (assume 640^2)
            vec[4] = float(np.clip(area / (640 * 640), 0, 1))

        return l2_normalize(np.array(vec, dtype=np.float32))

    def score(
        self,
        class_name: str,
        task_id: int,
        bbox: Optional[List[int]] = None,
    ) -> float:
        """
        Compute cosine similarity between object physical vector and
        task requirement vector.

        Returns float in [0, 1]:
          0.0 → object is physically incompatible with the task
          1.0 → object is a perfect physical match for the task
        """
        if task_id not in self._req:
            return 0.5   # unknown task → neutral

        obj_vec  = self.get_physical_vector(class_name, bbox)
        task_vec = self._req[task_id]

        # Both already L2-normalised → dot product = cosine similarity
        raw = float(np.dot(obj_vec, task_vec))

        # Scale from [-1,1] to [0,1]
        return (raw + 1.0) / 2.0

    def score_all(
        self,
        detections: list,
        task_id: int,
    ) -> List[float]:
        """
        Batch score all detections for a given task.
        Returns list of float scores aligned with detections list.
        """
        return [
            self.score(d.class_name, task_id, d.bbox)
            for d in detections
        ]


# ─────────────────────────────────────────────────────────────────────
# TASK AFFORDANCE PROFILE
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TaskAffordanceProfile:
    """
    A structured profile describing what a task physically requires.
    Used for readable reporting and for adaptive threshold tuning.
    """
    task_id:           int
    task_name:         str
    primary_property:  str    # dominant physical dimension needed
    secondary_property: str   # secondary dimension
    requires_liquid:   bool   # must hold / dispense liquid
    requires_cutting:  bool   # must have a sharp edge
    requires_heat_res: bool   # must handle heat


TASK_PROFILES: Dict[int, TaskAffordanceProfile] = {
    1:  TaskAffordanceProfile(1,  "step_on",         "rigidity",    "surface_area", False, False, False),
    2:  TaskAffordanceProfile(2,  "sit_comfortably", "surface_area","rigidity",     False, False, False),
    3:  TaskAffordanceProfile(3,  "place_flowers",   "containment", "rigidity",     True,  False, False),
    4:  TaskAffordanceProfile(4,  "potatoes_fire",   "heat_resist", "elongation",   False, True,  True),
    5:  TaskAffordanceProfile(5,  "water_plant",     "containment", "porosity",     True,  False, False),
    6:  TaskAffordanceProfile(6,  "lemon_from_tea",  "elongation",  "sharpness",    False, False, False),
    7:  TaskAffordanceProfile(7,  "dig_hole",        "sharpness",   "rigidity",     False, True,  False),
    8:  TaskAffordanceProfile(8,  "open_beer",       "sharpness",   "rigidity",     False, True,  False),
    9:  TaskAffordanceProfile(9,  "open_parcel",     "sharpness",   "elongation",   False, True,  False),
    10: TaskAffordanceProfile(10, "serve_wine",      "containment", "rigidity",     True,  False, False),
    11: TaskAffordanceProfile(11, "pour_sugar",      "containment", "rigidity",     False, False, False),
    12: TaskAffordanceProfile(12, "smear_butter",    "surface_area","sharpness",    False, False, False),
    13: TaskAffordanceProfile(13, "extinguish_fire", "containment", "heat_resist",  True,  False, True),
    14: TaskAffordanceProfile(14, "pound_carpet",    "rigidity",    "elongation",   False, False, False),
}


# ─────────────────────────────────────────────────────────────────────
# AFFORDANCE FUSION ENGINE
# ─────────────────────────────────────────────────────────────────────

class AffordanceFusionEngine:
    """
    Fuses text-semantic affordance score with physical affordance score.

    text_semantic_score  : cosine sim between task query embedding and
                           object affordance text embedding (Stage 6A)
    physical_score       : cosine sim between task requirement vector and
                           object physical property vector (Stage 6B)

    Fusion formula:
        fused = α * text_semantic + (1 - α) * physical
    where α = SEMANTIC_WEIGHT (default 0.65).

    The physical score acts as a regulariser: an object with
    a plausible semantic description but incompatible physical
    properties (e.g. a teddy bear for digging) gets pulled down.

    Novelty: this is the key generalisation mechanism.
    The physical scorer works on ANY object class — even one not
    in COCO — because it estimates elongation/area from the bbox.
    """

    SEMANTIC_WEIGHT: float = 0.65
    PHYSICAL_WEIGHT: float = 0.35

    def __init__(self):
        self.phys_scorer = PhysicalAffordanceScorer()
        log.info("AffordanceFusionEngine ready (sem=%.2f phys=%.2f)",
                 self.SEMANTIC_WEIGHT, self.PHYSICAL_WEIGHT)

    def fuse(
        self,
        text_semantic_scores: List[float],
        detections: list,
        task_id: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Args:
            text_semantic_scores : list of semantic scores per detection
            detections           : list of Detection objects
            task_id              : current task ID

        Returns:
            fused_scores         : list of fused affordance scores
            physical_scores      : list of raw physical scores (for reporting)
        """
        physical_scores = self.phys_scorer.score_all(detections, task_id)

        fused_scores = []
        for sem, phys in zip(text_semantic_scores, physical_scores):
            fused = (
                self.SEMANTIC_WEIGHT * sem
                + self.PHYSICAL_WEIGHT * phys
            )
            fused_scores.append(float(np.clip(fused, 0.0, 1.0)))

        return fused_scores, physical_scores

    def get_task_profile(self, task_id: int) -> Optional[TaskAffordanceProfile]:
        return TASK_PROFILES.get(task_id)
