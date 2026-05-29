"""
scene_context.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Scene Context and Physical Reasoning
Team Vulcan 629 | DSATM Bengaluru

Pipeline Stage: 8A  (runs before scoring, after affordance fusion)

Responsibilities:
  1. SceneGraphBuilder
       Builds a lightweight spatial relation graph between detected
       objects. Nodes = detected objects. Edges = spatial relations
       (NEAR, ON_TOP_OF, INSIDE, TOUCHING, FAR).

  2. ContextualBooster
       Uses the scene graph to modulate object scores.
       Objects that are spatially co-located with task-relevant
       context objects receive a small boost.
       Example: a cup near a potted plant → boosted for water_plant task.

  3. PhysicalConstraintFilter
       Hard-rejects objects that are physically impossible for the task.
       Example: a glass bottle rejected for "dig_hole" even if semantics
       score it moderately.

CONTEXTUAL BOOST FORMULA:
  context_score(o) = Σ_{n ∈ neighbours(o)} relevance(n, task) × proximity_weight(o, n)
  where proximity_weight = exp(−dist / σ)   σ = 0.3 × image_diagonal
  Final boost = W_CONTEXT × tanh(context_score)   clipped to [0, 0.15]

WHY THIS MATTERS:
  A knife near a parcel gets a stronger boost for "open_parcel" than
  the same knife on the other side of a kitchen scene. Context
  encodes real-world co-occurrence and spatial plausibility.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set

from config import TASKS, W_CONTEXT, NEVER_A_TOOL
from utils import get_logger, centre_distance, bbox_iou

log = get_logger("SceneContext")


# ─────────────────────────────────────────────────────────────────────
# SPATIAL RELATIONS
# ─────────────────────────────────────────────────────────────────────

class SpatialRelation(str, Enum):
    NEAR      = "NEAR"        # centres within 30% of image diagonal
    FAR       = "FAR"         # centres beyond 60% of image diagonal
    OVERLAPPING = "OVERLAPPING"  # IoU > 0.05
    ON_TOP_OF = "ON_TOP_OF"   # object A's bottom > object B's centre y (A on B)
    INSIDE    = "INSIDE"      # one bbox largely contained in another


@dataclass
class SceneEdge:
    """One directed spatial relation between two detected objects."""
    src_idx:   int
    dst_idx:   int
    relation:  SpatialRelation
    proximity: float   # 0=far, 1=touching/overlapping


@dataclass
class SceneGraph:
    """
    Lightweight scene graph for one image.
    Nodes are Detection indices. Edges encode spatial relations.
    """
    n_nodes:    int
    edges:      List[SceneEdge] = field(default_factory=list)
    adjacency:  Dict[int, List[int]] = field(default_factory=dict)   # node→neighbours

    def neighbours(self, node_idx: int) -> List[int]:
        return self.adjacency.get(node_idx, [])

    def edge_between(self, i: int, j: int) -> Optional[SceneEdge]:
        for e in self.edges:
            if e.src_idx == i and e.dst_idx == j:
                return e
        return None


# ─────────────────────────────────────────────────────────────────────
# TASK CONTEXT ANCHORS
# ─────────────────────────────────────────────────────────────────────
# Objects that, when detected nearby, boost relevance for a task.
# These are "context anchors" — not the target object, but companions.
#
# Example:
#   Task 5 (water_plant): if a potted plant is detected, any container
#   object near it gets a context boost (it's likely being used to water it).
#
# This is learned in principle but here encoded as semantic knowledge
# because the 14 tasks are fixed and closed-world.

TASK_CONTEXT_ANCHORS: Dict[int, List[str]] = {
    1:  ["bench", "skateboard", "suitcase"],            # step_on: things near a step
    2:  ["table", "dining table", "tv", "laptop"],      # sit: near a table/TV
    3:  ["potted plant", "vase", "dining table"],       # flowers: near containers
    4:  ["oven", "microwave", "dining table"],          # potatoes: near kitchen
    5:  ["potted plant", "sink", "bottle"],             # water_plant: near plant
    6:  ["cup", "dining table", "bowl"],                # lemon: near tea cup
    7:  ["potted plant", "bench"],                      # dig: outdoor scene
    8:  ["bottle", "dining table", "cup"],              # open_beer: near bottle
    9:  ["dining table", "laptop", "keyboard"],         # parcel: on table
    10: ["bottle", "dining table"],                     # wine: near bottle/table
    11: ["bowl", "dining table", "cup"],                # sugar: near bowl
    12: ["dining table", "knife", "bowl"],              # butter: near table
    13: ["oven", "microwave", "bottle"],                # fire: near heat source
    14: ["chair", "couch", "dining table"],             # carpet: near furniture
}


# ─────────────────────────────────────────────────────────────────────
# SCENE GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────

class SceneGraphBuilder:
    """
    Builds a spatial scene graph from a list of Detection objects.

    NEAR threshold    : dist < 0.30 × image_diagonal
    OVERLAPPING       : bbox IoU > 0.05
    ON_TOP_OF         : src bottom < 10px above dst centre, src above dst
    FAR threshold     : dist > 0.60 × image_diagonal
    """

    NEAR_FACTOR      = 0.30
    FAR_FACTOR       = 0.60
    OVERLAP_THRESH   = 0.05
    SIGMA_FACTOR     = 0.30   # σ = SIGMA_FACTOR × image_diagonal (for proximity weight)

    def build(
        self,
        detections: list,
        image_hw: Tuple[int, int],
    ) -> SceneGraph:
        """
        Args:
            detections : list of Detection objects
            image_hw   : (height, width) of the image in pixels

        Returns:
            SceneGraph with all spatial edges
        """
        h, w    = image_hw
        diag    = np.hypot(w, h)
        sigma   = self.SIGMA_FACTOR * diag
        n       = len(detections)
        graph   = SceneGraph(n_nodes=n, adjacency={i: [] for i in range(n)})

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                di, dj = detections[i], detections[j]
                dist   = centre_distance(di.bbox, dj.bbox)
                iou    = bbox_iou(di.bbox, dj.bbox)

                # Determine spatial relation
                if iou > self.OVERLAP_THRESH:
                    relation = SpatialRelation.OVERLAPPING
                elif dist < self.NEAR_FACTOR * diag:
                    relation = SpatialRelation.NEAR
                elif dist > self.FAR_FACTOR * diag:
                    relation = SpatialRelation.FAR
                else:
                    continue   # medium distance — no strong relation

                # ON_TOP_OF: di's centre y < dj's centre y (di is above dj in image)
                ci_y = (di.bbox[1] + di.bbox[3]) / 2
                cj_y = (dj.bbox[1] + dj.bbox[3]) / 2
                if relation == SpatialRelation.NEAR and ci_y < cj_y - 20:
                    relation = SpatialRelation.ON_TOP_OF

                # Proximity weight: Gaussian decay with distance
                proximity = float(np.exp(-dist / (sigma + 1e-6)))

                edge = SceneEdge(
                    src_idx  = i,
                    dst_idx  = j,
                    relation = relation,
                    proximity = proximity,
                )
                graph.edges.append(edge)
                if j not in graph.adjacency[i]:
                    graph.adjacency[i].append(j)

        log.info(
            "SceneGraph: %d nodes, %d edges (image %dx%d diag=%.0f)",
            n, len(graph.edges), w, h, diag,
        )
        return graph


# ─────────────────────────────────────────────────────────────────────
# CONTEXTUAL BOOSTER
# ─────────────────────────────────────────────────────────────────────

class ContextualBooster:
    """
    Computes per-object contextual boost scores using the scene graph.

    For each candidate object o:
      1. Find all neighbours in the scene graph
      2. Check if any neighbour is a task context anchor
      3. Weight the boost by spatial proximity (Gaussian decay)
      4. Apply tanh saturation to prevent runaway boosts

    context_score(o) = Σ_{n ∈ neighbours(o)} is_anchor(n) × proximity(o,n)
    boost(o)         = W_CONTEXT × tanh(context_score(o))
                       clipped to [0.0, 0.15]

    EXAMPLE:
      Task 5 (water_plant), anchors = ["potted plant", "sink", "bottle"]
      Object = cup (idx=2), neighbour = potted plant (idx=5)
      proximity(cup, potted_plant) = 0.72 (they're near each other)
      context_score = 0.72
      boost = 0.05 × tanh(0.72) = 0.033

    This is small but enough to break ties in favour of contextually
    coherent selections.
    """

    MAX_BOOST: float = 0.15

    def compute_boosts(
        self,
        detections: list,
        graph: SceneGraph,
        task_id: int,
    ) -> List[float]:
        """
        Compute contextual boost for every detected object.
        Returns list of float boosts aligned with detections list.
        """
        anchors: Set[str] = set(TASK_CONTEXT_ANCHORS.get(task_id, []))
        boosts   = []

        for i, det in enumerate(detections):
            ctx_score = 0.0
            for j in graph.neighbours(i):
                neighbour_name = detections[j].class_name
                if neighbour_name in anchors:
                    # Find the edge to get proximity weight
                    edge = graph.edge_between(i, j)
                    prox = edge.proximity if edge else 0.5
                    ctx_score += prox

            # Saturate and scale
            boost = min(
                self.MAX_BOOST,
                W_CONTEXT * float(np.tanh(ctx_score))
            )
            boosts.append(boost)

        return boosts


# ─────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTRAINT FILTER
# ─────────────────────────────────────────────────────────────────────

# Hard physical constraints per task.
# Objects failing these are penalised regardless of semantic score.
# Format: {task_id: {class_name: reason_string}}

HARD_CONSTRAINTS: Dict[int, Dict[str, str]] = {
    5:  {   # water_plant: nothing brittle that shatters near water
        "wine glass": "fragile near water, likely to break",
        "laptop":     "electronics must not contact water",
        "cell phone": "electronics must not contact water",
        "keyboard":   "electronics must not contact water",
    },
    13: {   # extinguish_fire: nothing that melts or combusts
        "laptop":     "electronics — fire hazard",
        "cell phone": "electronics — fire hazard",
        "book":       "paper — combustible",
        "knife":      "metal conducts heat — dangerous",
    },
    4:  {   # potatoes from fire: nothing that melts
        "wine glass": "glass shatters near open flame",
        "cup":        "ceramic may crack near open flame",
        "cell phone": "electronics melt near fire",
        "laptop":     "electronics melt near fire",
    },
}

# Multiplier applied when hard constraint fires
CONSTRAINT_PENALTY: float = 0.08


class PhysicalConstraintFilter:
    """
    Applies hard physical constraint penalties to object scores.
    This is the safety layer — prevents recommending physically
    dangerous or impossible object-task combinations.

    Returns:
        constraint_multipliers: list of floats (1.0 = no penalty, <1 = penalised)
        constraint_reasons    : list of str (empty if no constraint fired)
    """

    def apply(
        self,
        detections: list,
        task_id: int,
    ) -> Tuple[List[float], List[str]]:
        """
        Args:
            detections : list of Detection objects
            task_id    : current task ID

        Returns:
            multipliers: float per detection (1.0 or CONSTRAINT_PENALTY)
            reasons    : str per detection (empty if no constraint)
        """
        constraints = HARD_CONSTRAINTS.get(task_id, {})
        multipliers = []
        reasons     = []

        for det in detections:
            cls = det.class_name
            if cls in constraints:
                multipliers.append(CONSTRAINT_PENALTY)
                reasons.append(f"PHYSICAL_CONSTRAINT: {constraints[cls]}")
            elif cls in NEVER_A_TOOL:
                multipliers.append(0.02)
                reasons.append("NEVER_A_TOOL: biological agent")
            else:
                multipliers.append(1.0)
                reasons.append("")

        return multipliers, reasons
