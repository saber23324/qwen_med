"""
GRPO reward plugin for Phase 3 RL.

Loaded by ms-swift via `--external_plugins Qwen3_VL/grpo_plugin.py`.

Design contract
---------------
The Phase 3 multi-turn rollout scheduler (`Qwen3_VL/phase3_rl_rollout.py`)
populates `RolloutOutput.rollout_infos` with a dict that already carries
every reward component pre-computed from the executed trajectory:

    {
        # Outcome (§9.3)
        "r_dice":        float,   # mean per-slice Dice over sampled Z, post-refine
        "r_key_dice":    float,   # Dice on the ordinal committed in run_medsam2
        # Process shaping (§9.4)
        "r_bbox_iou":    float,   # mean IoU over add_bbox calls vs GT bbox
        "r_dice_gain":   float,   # sum of clipped per-step refinement gains
        "r_coverage":    float,   # lesion-ordinal recall − 0.5·FP penalty
        "r_nav_style":   float,   # signed credit for tool-choice/Δ pattern
        "r_efficiency":  float,   # 0 within budget, negative beyond cap
        # Format (§9.5)
        "gate":          0 or 1,  # multiplicative validity gate
        # Diagnostics
        "num_turns":     int,
        "n_add_bbox":    int,
        "n_add_point":   int,
        "n_get_slice":   int,
        "n_scroll":      int,
        "boundary_clamp_count": int,
        "revisit_no_change_count": int,
        # ... any other diagnostic fields
    }

Reward functions just look these up. If `rollout_infos` is missing a
component (e.g. teacher-forced sanity check or rollout failure), they
return `nan` — ms-swift treats nan as "skip this row for this func".

Curriculum (§9.7) is applied on the launcher side via stage-specific
shaping weights, not here. This plugin only provides the components.
"""

from typing import Any, Dict, List, Optional

from swift.rewards.orm import ORM, orms

# §9.6 baseline shaping weights — used by `phase3_combined` to assemble the
# scalar policy reward. Stage-specific scaling lives in the launcher
# (override via `--phase3_shaping_scale 2.0` for stage 1, etc., env var below).
import os

SHAPING_SCALE = float(os.environ.get('PHASE3_SHAPING_SCALE', '1.0'))

# Soft gate during early training: gate∈{0,1} hard by default; if
# PHASE3_GATE_SOFT=1 (stage 1), invalid trajectories get 0.1× instead of 0×.
GATE_SOFT = os.environ.get('PHASE3_GATE_SOFT', '0') == '1'


def _get_infos(kwargs: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    """Pull `rollout_infos` from ms-swift's batched reward kwargs.

    `RowPreprocessor.rows_to_batched` turns each row's `rollout_infos` field
    into a per-row entry in a list keyed `rollout_infos`. Some code paths use
    `trajectory_inputs` instead. Fall back to None when neither is present
    (e.g. accidentally invoked outside an actual rollout).
    """
    if 'rollout_infos' in kwargs and kwargs['rollout_infos'] is not None:
        return list(kwargs['rollout_infos'])
    if 'trajectory_inputs' in kwargs and kwargs['trajectory_inputs'] is not None:
        return [
            (t.get('rollout_infos') if isinstance(t, dict) else None)
            for t in kwargs['trajectory_inputs']
        ]
    return [None] * len(kwargs.get('completions', []) or [])


def _safe(infos: Optional[Dict[str, Any]], key: str, default: float = float('nan')) -> float:
    if not isinstance(infos, dict):
        return default
    v = infos.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ────────────────────────────────────────────────────────────────────────────
# Per-component diagnostic ORMs
# ────────────────────────────────────────────────────────────────────────────

class _RolloutInfoORM(ORM):
    """Base class: each subclass declares the rollout_infos key it surfaces."""
    info_key: str = ''
    default: float = float('nan')

    def __call__(self, completions, **kwargs) -> List[float]:
        infos_list = _get_infos({'completions': completions, **kwargs})
        out: List[float] = []
        for infos in infos_list:
            out.append(_safe(infos, self.info_key, self.default))
        # Pad/truncate just in case (length mismatch is a bug in the rollout):
        if len(out) < len(completions):
            out.extend([float('nan')] * (len(completions) - len(out)))
        return out[:len(completions)]


class Phase3DiceReward(_RolloutInfoORM):       info_key = 'r_dice'
class Phase3KeyDiceReward(_RolloutInfoORM):    info_key = 'r_key_dice'
class Phase3BboxIoUReward(_RolloutInfoORM):    info_key = 'r_bbox_iou'
class Phase3DiceGainReward(_RolloutInfoORM):   info_key = 'r_dice_gain'
class Phase3CoverageReward(_RolloutInfoORM):   info_key = 'r_coverage'
class Phase3NavStyleReward(_RolloutInfoORM):   info_key = 'r_nav_style'
class Phase3EfficiencyReward(_RolloutInfoORM): info_key = 'r_efficiency'


class Phase3FormatGate(_RolloutInfoORM):
    """Surfaces the binary gate (1 valid / 0 invalid) for monitoring."""
    info_key = 'gate'
    default = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Combined policy reward — §9.6
# ────────────────────────────────────────────────────────────────────────────

class Phase3CombinedReward(ORM):
    """Implements:

        R = gate · [
              1.00 · (0.7·R_dice + 0.3·R_key_dice)
            + s · 0.20 · R_bbox_iou
            + s · 0.15 · R_dice_gain
            + s · 0.10 · R_coverage
            + s · 0.05 · R_nav_style
            + 1.00 · R_efficiency
        ]

    where `s = SHAPING_SCALE` controls the curriculum (stage 1 = 2.0,
    stage 2 = 1.0, stage 3 = 0.0 except R_bbox_iou which stays).

    `gate` is multiplicative; if `PHASE3_GATE_SOFT=1`, invalid trajectories
    receive 0.1× instead of 0× (used during stage-1 warmup if entire groups
    collapse).
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        infos_list = _get_infos({'completions': completions, **kwargs})
        out: List[float] = []
        for infos in infos_list:
            if not isinstance(infos, dict):
                out.append(float('nan'))
                continue

            r_dice      = _safe(infos, 'r_dice',      0.0)
            r_key_dice  = _safe(infos, 'r_key_dice',  0.0)
            r_bbox_iou  = _safe(infos, 'r_bbox_iou',  0.0)
            r_dice_gain = _safe(infos, 'r_dice_gain', 0.0)
            r_coverage  = _safe(infos, 'r_coverage',  0.0)
            r_nav_style = _safe(infos, 'r_nav_style', 0.0)
            r_eff       = _safe(infos, 'r_efficiency', 0.0)
            gate        = _safe(infos, 'gate', 0.0)

            s = SHAPING_SCALE
            r = (
                1.00 * (0.7 * r_dice + 0.3 * r_key_dice)
                + s * 0.20 * r_bbox_iou
                + s * 0.15 * r_dice_gain
                + s * 0.10 * r_coverage
                + s * 0.05 * r_nav_style
                + 1.00 * r_eff
            )
            if gate <= 0:
                r = 0.1 * r if GATE_SOFT else 0.0
            out.append(float(r))

        if len(out) < len(completions):
            out.extend([float('nan')] * (len(completions) - len(out)))
        return out[:len(completions)]


# ────────────────────────────────────────────────────────────────────────────
# Registry — names referenced from the training launcher via `--reward_funcs`
# ────────────────────────────────────────────────────────────────────────────

orms['phase3_combined']      = Phase3CombinedReward
orms['phase3_dice']          = Phase3DiceReward
orms['phase3_key_dice']      = Phase3KeyDiceReward
orms['phase3_bbox_iou']      = Phase3BboxIoUReward
orms['phase3_dice_gain']     = Phase3DiceGainReward
orms['phase3_coverage']      = Phase3CoverageReward
orms['phase3_nav_style']     = Phase3NavStyleReward
orms['phase3_efficiency']    = Phase3EfficiencyReward
orms['phase3_format_gate']   = Phase3FormatGate
