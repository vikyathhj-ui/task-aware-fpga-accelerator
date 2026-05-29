"""
scoring.py
─────────────────────────────────────────────────────────────────────
DVCon India 2026 — Hierarchical Scoring and Rejection Gate
Team Vulcan 629 | DSATM Bengaluru

Pipeline Stage: 8

This module implements the five-component hierarchical scoring system
and the adaptive rejection gate. It is the intellectual core of the
task-aware selection pipeline.

═══════════════════════════════════════════════════════════════════════
SCORING FORMULA
═══════════════════════════════════════════════════════════════════════

Given detected object o and task t, the final score is computed as:

  Step 1 — Component scores (all in [0, 1]):
    S_sem   = affordance-fused semantic score (Stage 6A + 6B)
    S_vis   = CLIP visual-semantic similarity (Stage 6C)
    S_prior = task-prior knowledge bonus (config.TASK_PRIORS)
    S_phys  = physical affordance score (affordance_engine)
    S_ctx   = contextual scene boost (scene_context)

  Step 2 — Weighted combination:
    raw(o) = W_sem   × S_sem(o)
           + W_vis   × S_vis(o)
           + W_prior × S_prior(o)
           + W_phys  × S_phys(o)
           + W_ctx   × S_ctx(o)

    Weights from config.py:
      W_sem = 0.40,  W_vis = 0.25,  W_prior = 0.20,
      W_phys = 0.10, W_ctx = 0.05   (sum = 1.00)

  Step 3 — Prior multiplier:
    final(o) = raw(o) × prior_mult(o, t)
    where prior_mult ∈ {2.2 PREFERRED, 1.35 ACCEPTABLE,
                        1.00 NEUTRAL, 0.05 PENALISED}

  Step 4 — Physical constraint multiplier:
    final(o) = final(o) × constraint_mult(o, t)
    (1.0 normally, 0.08 if a hard physical constraint fires)

  Step 5 — Clip to [0, 1]:
    final(o) = clip(final(o), 0.0, 1.0)

═══════════════════════════════════════════════════════════════════════
ADAPTIVE REJECTION GATE
═══════════════════════════════════════════════════════════════════════

  Static threshold  : REJECTION_THRESHOLD = 0.35 (from config)
  Adaptive override : if uncertainty is HIGH (entropy > 0.8),
                      raise threshold by 0.05 to be more conservative.

  Rule:
    threshold = BASE_THRESHOLD + 0.05 × (uncertainty > 0.80)

  Decision:
    if best_score < threshold → REJECT ("no suitable object found")
    if margin(1st, 2nd) < UNCERTAINTY_MARGIN → flag as UNCERTAIN
    else                      → ACCEPT best object

═══════════════════════════════════════════════════════════════════════
UNCERTAINTY ESTIMATION
═══════════════════════════════════════════════════════════════════════

  Entropy-based uncertainty over final scores:
    p_i = softmax(final_scores / τ)   τ = 0.5 (temperature)
    H   = −Σ p_i log(p_i)
    U   = H / log(N)                  (normalised, in [0,1])

  U → 0 : model is confident about one object
  U → 1 : model is equally uncertain about all objects

  This is reported in the output but does NOT cause a hard reject
  unless the adaptive threshold is crossed.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from config import (
    W_SEMANTIC, W_VISUAL, W_PRIOR, W_PHYSICAL, W_CONTEXT,
    PRIOR_PREFERRED, PRIOR_ACCEPTABLE, PRIOR_NEUTRAL, PRIOR_PENALISED,
    REJECTION_THRESHOLD, UNCERTAINTY_MARGIN, TASK_PRIORS,
)
from utils import get_logger, softmax, uncertainty, scale_to_unit

log = get_logger("Scoring")


# ─────────────────────────────────────────────────────────────────────
# SCORED OBJECT DATACLASS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ScoredObject:
    """All scores and metadata for one detected object after Stage 8."""
    class_name:          str
    class_id:            int
    confidence:          float
    bbox:                List[int]

    # Component scores
    sem_score:           float   # fused text+physical semantic
    vis_score:           float   # CLIP visual similarity
    phys_score:          float   # raw physical affordance
    prior_score:         float   # task prior bonus [0..1]
    ctx_boost:           float   # contextual scene boost

    # Aggregation
    raw_score:           float   # weighted sum before multipliers
    prior_mult:          float   # multiplier from TASK_PRIORS
    prior_label:         str     # PREFERRED / ACCEPTABLE / NEUTRAL / PENALISED
    constraint_mult:     float   # physical constraint multiplier
    constraint_reason:   str     # why constraint fired (or empty)
    final_score:         float   # after all multipliers, clipped [0,1]
    rejection_reason:    str     # why rejected (or empty if accepted)
    uncertainty:         float   # local uncertainty contribution

    def is_valid_candidate(self, threshold: float) -> bool:
        return (
            self.final_score >= threshold
            and self.constraint_mult > 0.1
            and self.prior_label != "PENALISED"
        )

    def as_dict(self) -> dict:
        return {
            "class_name":        self.class_name,
            "class_id":          self.class_id,
            "confidence":        self.confidence,
            "bbox":              self.bbox,
            "sem_score":         round(self.sem_score,  4),
            "vis_score":         round(self.vis_score,  4),
            "phys_score":        round(self.phys_score, 4),
            "prior_label":       self.prior_label,
            "final_score":       round(self.final_score, 4),
            "rejection_reason":  self.rejection_reason,
        }


# ─────────────────────────────────────────────────────────────────────
# PRIOR LOOKUP
# ─────────────────────────────────────────────────────────────────────

def get_prior_multiplier(task_id: int, class_name: str) -> Tuple[float, float, str]:
    """
    Returns (multiplier, bonus_score, label) for a task–object pair.
    bonus_score is the direct component score contribution.
    """
    prior = TASK_PRIORS.get(task_id, {})

    if class_name in prior.get("preferred", []):
        return PRIOR_PREFERRED, 1.0, "PREFERRED"

    if class_name in prior.get("acceptable", []):
        return PRIOR_ACCEPTABLE, 0.6, "ACCEPTABLE"

    if class_name in prior.get("penalised", []):
        return PRIOR_PENALISED, 0.0, "PENALISED"

    return PRIOR_NEUTRAL, 0.3, "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────
# HIERARCHICAL SCORER
# ─────────────────────────────────────────────────────────────────────

class HierarchicalScorer:
    """
    Combines five component scores into one final score per object.
    Applies prior multipliers and physical constraint multipliers.

    Inputs per object:
      - fused_sem_score  : affordance-fused semantic score [0,1]
      - vis_score        : CLIP visual similarity [0,1]
      - phys_score       : physical affordance [0,1]
      - ctx_boost        : context boost [0, 0.15]
      - constraint_mult  : hard constraint multiplier
      - constraint_reason: str

    Output: ScoredObject with all breakdowns filled in.
    """

    def score_all(
        self,
        detections: list,
        task_id: int,
        fused_sem_scores:   List[float],
        vis_scores:         List[float],
        phys_scores:        List[float],
        ctx_boosts:         List[float],
        constraint_mults:   List[float],
        constraint_reasons: List[str],
    ) -> List[ScoredObject]:
        """
        Score all detections. Returns list of ScoredObject sorted
        by final_score descending.
        """
        scored = []

        for i, det in enumerate(detections):
            cls = det.class_name

            S_sem   = float(fused_sem_scores[i])
            S_vis   = float(vis_scores[i])
            S_phys  = float(phys_scores[i])
            S_ctx   = float(ctx_boosts[i])

            # Task prior lookup
            prior_mult, prior_bonus, prior_label = get_prior_multiplier(
                task_id, cls
            )
            S_prior = prior_bonus

            # Weighted sum of five components
            raw_score = (
                W_SEMANTIC * S_sem
                + W_VISUAL  * S_vis
                + W_PRIOR   * S_prior
                + W_PHYSICAL * S_phys
                + W_CONTEXT * S_ctx
            )

            # Apply prior multiplier
            after_prior = raw_score * prior_mult

            # Apply physical constraint multiplier
            c_mult   = float(constraint_mults[i])
            c_reason = constraint_reasons[i]
            after_constraint = after_prior * c_mult

            # Clip to [0, 1]
            final_score = float(np.clip(after_constraint, 0.0, 1.0))

            # Rejection reason (for display)
            if c_reason:
                rej_reason = c_reason
            elif prior_label == "PENALISED":
                rej_reason = f"PENALISED for task {task_id}"
            else:
                rej_reason = ""

            scored.append(ScoredObject(
                class_name        = cls,
                class_id          = det.class_id,
                confidence        = det.confidence,
                bbox              = det.bbox,
                sem_score         = S_sem,
                vis_score         = S_vis,
                phys_score        = S_phys,
                prior_score       = S_prior,
                ctx_boost         = S_ctx,
                raw_score         = raw_score,
                prior_mult        = prior_mult,
                prior_label       = prior_label,
                constraint_mult   = c_mult,
                constraint_reason = c_reason,
                final_score       = final_score,
                rejection_reason  = rej_reason,
                uncertainty       = 0.0,   # filled in by RejectionGate
            ))

        # Sort by final score descending
        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored

    def print_breakdown(self, scored: List[ScoredObject], task_id: int):
        """Print the full per-object scoring table to stdout."""
        print(
            f"\n  {'Object':<22} {'Sem':>5}  {'Vis':>5}  {'Phys':>5}"
            f"  {'Ctx':>5}  {'Raw':>5}  {'×Prior':>7}"
            f"  {'Final':>6}  Status"
        )
        print("  " + "─" * 90)
        for obj in scored:
            status = obj.rejection_reason or obj.prior_label
            marker = "★" if obj == scored[0] else " "
            print(
                f"{marker} {obj.class_name:<22}"
                f" {obj.sem_score:>5.3f}"
                f"  {obj.vis_score:>5.3f}"
                f"  {obj.phys_score:>5.3f}"
                f"  {obj.ctx_boost:>5.3f}"
                f"  {obj.raw_score:>5.3f}"
                f"  ×{obj.prior_mult:>5.2f}"
                f"  {obj.final_score:>6.3f}"
                f"  {status}"
            )
        print()


# ─────────────────────────────────────────────────────────────────────
# REJECTION GATE
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SelectionResult:
    """
    Final output of the rejection gate.
    winner   = None if no object passes the threshold.
    uncertain = True if top-2 scores are within UNCERTAINTY_MARGIN.
    """
    winner:        Optional[ScoredObject]
    all_scored:    List[ScoredObject]
    threshold_used: float
    uncertainty:   float           # entropy-based [0, 1]
    uncertain:     bool            # True if margin between 1st and 2nd is small
    rejected:      bool            # True if no winner


class RejectionGate:
    """
    Adaptive rejection gate for the task-aware selection pipeline.

    1. Computes entropy-based uncertainty over final scores.
    2. Raises threshold slightly if uncertainty is high.
    3. Accepts or rejects the best object.
    4. Flags uncertain decisions (close race between 1st and 2nd).

    ADAPTIVE THRESHOLD FORMULA:
      threshold = BASE + 0.05 × (U > 0.80)
    where U = normalised entropy of softmax(final_scores / τ), τ=0.5.
    """

    BASE_THRESHOLD:     float = REJECTION_THRESHOLD
    HIGH_UNCERTAINTY:   float = 0.80   # entropy above this → raise threshold
    THRESHOLD_RAISE:    float = 0.05
    TEMPERATURE:        float = 0.50   # softmax temperature for entropy

    def evaluate(self, scored: List[ScoredObject]) -> SelectionResult:
        """
        Args:
            scored : list of ScoredObject sorted by final_score descending

        Returns:
            SelectionResult with winner (or None) and all diagnostics
        """
        if not scored:
            return SelectionResult(
                winner=None, all_scored=[], threshold_used=self.BASE_THRESHOLD,
                uncertainty=1.0, uncertain=False, rejected=True,
            )

        final_scores = np.array([s.final_score for s in scored], dtype=np.float64)

        # Entropy-based uncertainty
        U = uncertainty(final_scores)

        # Adaptive threshold
        threshold = self.BASE_THRESHOLD
        if U > self.HIGH_UNCERTAINTY:
            threshold += self.THRESHOLD_RAISE
            log.info(
                "High uncertainty (U=%.3f) → threshold raised to %.2f",
                U, threshold,
            )

        # Annotate uncertainty into each ScoredObject
        probs = softmax(final_scores, temperature=self.TEMPERATURE)
        for i, obj in enumerate(scored):
            obj.uncertainty = float(probs[i])

        best      = scored[0]
        rejected  = best.final_score < threshold

        # Uncertainty flag: top-2 too close?
        uncertain = False
        if len(scored) >= 2:
            margin = best.final_score - scored[1].final_score
            uncertain = margin < UNCERTAINTY_MARGIN

        return SelectionResult(
            winner         = None if rejected else best,
            all_scored     = scored,
            threshold_used = threshold,
            uncertainty    = U,
            uncertain      = uncertain,
            rejected       = rejected,
        )

    def print_result(self, result: SelectionResult, task_query: str):
        """Print the final selection result to stdout."""
        print("═" * 65)

        if result.rejected:
            best = result.all_scored[0] if result.all_scored else None
            print(f"  RESULT: NO SUITABLE OBJECT FOUND FOR THIS TASK")
            if best:
                print(f"  Best candidate   : {best.class_name}")
                print(f"  Best score       : {best.final_score:.3f}")
                print(f"  Required (thresh): ≥ {result.threshold_used:.2f}")
                reason = best.rejection_reason or "score too low"
                print(f"  Reason           : {reason}")
            print(f"  Uncertainty (U)  : {result.uncertainty:.3f}")
        else:
            w = result.winner
            print(f"  FINAL SELECTION: {w.class_name.upper()}")
            print(f"  ─────────────────────────────────────────────────────── ")
            print(f"  Task             : {task_query}")
            print(f"  Selected object  : {w.class_name}")
            print(f"  ├ Semantic score : {w.sem_score:.3f}")
            print(f"  ├ Visual score   : {w.vis_score:.3f}")
            print(f"  ├ Physical score : {w.phys_score:.3f}")
            print(f"  ├ Context boost  : {w.ctx_boost:.3f}")
            print(f"  ├ Prior status   : {w.prior_label}")
            print(f"  ├ Raw score      : {w.raw_score:.3f}")
            print(f"  ├ × Prior mult   : {w.prior_mult:.2f}")
            print(f"  └ Final score    : {w.final_score:.3f}  "
                  f"(threshold {result.threshold_used:.2f})")
            print(f"  Uncertainty (U)  : {result.uncertainty:.3f}"
                  f"  {'⚠ UNCERTAIN' if result.uncertain else '✓ confident'}")
            print(f"  Bounding box     : {w.bbox}")

            # UART-format output line (for VEGA RISC-V console)
            uart = (
                f"{w.class_id} {w.class_name} {w.final_score:.3f} "
                f"{w.bbox[0]} {w.bbox[1]} {w.bbox[2]} {w.bbox[3]}"
            )
            print(f"\n  UART: {uart}")

        print("═" * 65 + "\n")
