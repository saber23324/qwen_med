"""
Phase 3 RL — reward-computation helpers, vLLM-free.

Imported by both the multi-turn rollout scheduler (`phase3_rl_rollout.py`,
which needs vLLM at the server side) and the dry-run / unit-test path
(this file alone, no swift / vllm needed).

Provides:
  • `RewardTrackingExecutor`  — NavAgentExecutor subclass that records the
    per-call traces required by §9.4.
  • `compute_reward_components` — produces the §9.6 rollout_infos dict.
  • `collect_gate_violations`  — implements the §9.5 hard format gate.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as maskUtils

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_phase3 import NavAgentExecutor, dice_score


# ── Reward-tuning knobs (env-overridable; mirrors phase3_rl_rollout) ────────
DICE_GAIN_CLIP      = float(os.environ.get('PHASE3_DICE_GAIN_CLIP', '0.3'))
COVERAGE_FP_LAMBDA  = float(os.environ.get('PHASE3_COVERAGE_FP_LAMBDA', '0.5'))
NAV_STYLE_REWARD    = float(os.environ.get('PHASE3_NAV_REWARD', '0.01'))
NAV_STYLE_CLAMP     = float(os.environ.get('PHASE3_NAV_CLAMP_PEN', '0.02'))
NAV_STYLE_REVISIT   = float(os.environ.get('PHASE3_NAV_REVISIT_PEN', '0.05'))
EFFICIENCY_PEN_RATE = float(os.environ.get('PHASE3_EFF_PEN', '0.01'))
EFFICIENCY_CAP      = int(os.environ.get('PHASE3_EFFICIENCY_CAP', '45'))


# ────────────────────────────────────────────────────────────────────────────
# RewardTrackingExecutor — wraps NavAgentExecutor to collect reward telemetry
# ────────────────────────────────────────────────────────────────────────────

class RewardTrackingExecutor(NavAgentExecutor):
    """Adds per-call traces required to compute §9.4 shaping rewards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bbox_iou_calls: List[float] = []
        self.bbox_calls_on_lesion: List[bool] = []
        self.dice_gain_calls: List[float] = []
        self.nav_decisions: List[Dict[str, Any]] = []
        self.key_z_chosen: Optional[int] = None
        self.bbox_iou_at_run: Optional[float] = None
        self.finish_called: bool = False
        self.run_medsam2_called: bool = False
        self._last_read_overlay_state: Dict[int, str] = {}

    def _is_lesion_ord(self, ord_idx: int) -> bool:
        z = self.sampled[ord_idx]
        if not (0 <= z < len(self.masks)):
            return False
        m = self.masks[z]
        if m is None:
            return False
        try:
            return bool(maskUtils.decode(m).any())
        except Exception:
            return False

    def exec_get_slice(self, z_index):
        prev_ord = self.cur_ord
        if z_index in self.sampled:
            target_ord = self.sampled.index(z_index)
            delta = abs(target_ord - prev_ord) if prev_ord is not None else 99
            overlay_changed = (
                target_ord not in self._last_read_overlay_state
                or self._last_read_overlay_state[target_ord]
                != self.overlay_state.get(target_ord, 'raw')
            )
            self.nav_decisions.append({
                'tool': 'get_slice', 'prev_ord': prev_ord,
                'target_ord': target_ord, 'delta': delta,
                'clamped': False, 'overlay_changed': overlay_changed,
            })
        out = super().exec_get_slice(z_index)
        if self.cur_ord is not None:
            self._last_read_overlay_state[self.cur_ord] = self.overlay_state.get(
                self.cur_ord, 'raw')
        return out

    def exec_scroll(self, delta):
        prev_ord = self.cur_ord
        if prev_ord is not None:
            target_raw = prev_ord + int(delta)
            target_ord = max(0, min(len(self.sampled) - 1, target_raw))
            clamped = (target_ord != target_raw)
            overlay_changed = (
                target_ord not in self._last_read_overlay_state
                or self._last_read_overlay_state[target_ord]
                != self.overlay_state.get(target_ord, 'raw')
            )
            self.nav_decisions.append({
                'tool': 'scroll', 'prev_ord': prev_ord,
                'target_ord': target_ord, 'delta': abs(int(delta)),
                'clamped': clamped, 'overlay_changed': overlay_changed,
            })
        out = super().exec_scroll(delta)
        if self.cur_ord is not None:
            self._last_read_overlay_state[self.cur_ord] = self.overlay_state.get(
                self.cur_ord, 'raw')
        return out

    def exec_add_bbox(self, z_index, bbox):
        if self.cur_ord is not None and self.sampled[self.cur_ord] == z_index:
            iou = float(self._iou_vs_gt(bbox, z_index))
            self.bbox_iou_calls.append(iou)
            self.bbox_calls_on_lesion.append(self._is_lesion_ord(self.cur_ord))
        return super().exec_add_bbox(z_index, bbox)

    def exec_run_medsam2(self, key_z, bbox):
        self.run_medsam2_called = True
        self.key_z_chosen = int(key_z)
        if 0 <= key_z < len(self.masks):
            self.bbox_iou_at_run = float(self._iou_vs_gt(bbox, key_z))
        return super().exec_run_medsam2(key_z, bbox)

    def exec_add_point(self, z_index, points, labels):
        if 0 <= z_index < len(self.masks) and self.masks[z_index] is not None:
            gt = maskUtils.decode(self.masks[z_index]).astype(bool)
            prev = self.final_masks.get(z_index)
            dice_before = float(dice_score(prev, gt)) if prev is not None else 0.0
        else:
            dice_before = 0.0
        out = super().exec_add_point(z_index, points, labels)
        dice_after = float(self.refined_dice.get(z_index, dice_before))
        gain = max(0.0, min(DICE_GAIN_CLIP, dice_after - dice_before))
        self.dice_gain_calls.append(gain)
        return out


# ────────────────────────────────────────────────────────────────────────────
# Reward-component computation (§9.6)
# ────────────────────────────────────────────────────────────────────────────

def compute_reward_components(
    ex: RewardTrackingExecutor,
    *,
    lesion_ordinals: List[int],
    non_lesion_ordinals: List[int],
    n_tool_calls: int,
    gate_violations: List[str],
) -> Dict[str, Any]:
    sampled = ex.sampled
    gt_masks = ex.masks

    # Outcome: per-slice Dice over sampled Z, post-refine
    dices = []
    for z in sampled:
        if 0 <= z < len(gt_masks) and gt_masks[z] is not None:
            gt = maskUtils.decode(gt_masks[z]).astype(bool)
        else:
            gt = None
        pred = ex.final_masks.get(z)
        if gt is None and pred is None:
            continue
        if pred is None:
            pred = np.zeros_like(gt) if gt is not None else None
        if gt is None:
            gt = np.zeros_like(pred)
        dices.append(dice_score(pred, gt))
    r_dice = float(np.mean(dices)) if dices else 0.0

    # Outcome: key-slice Dice
    if ex.key_z_chosen is not None and 0 <= ex.key_z_chosen < len(gt_masks):
        gt_k = gt_masks[ex.key_z_chosen]
        if gt_k is not None:
            gt_arr = maskUtils.decode(gt_k).astype(bool)
            pred_k = ex.final_masks.get(ex.key_z_chosen)
            if pred_k is None:
                pred_k = np.zeros_like(gt_arr)
            r_key_dice = float(dice_score(pred_k, gt_arr))
        else:
            r_key_dice = 0.0
    else:
        r_key_dice = 0.0

    # Process: bbox IoU (mean over add_bbox calls)
    r_bbox_iou = float(np.mean(ex.bbox_iou_calls)) if ex.bbox_iou_calls else 0.0

    # Process: dice gain (sum of clipped per-step gains)
    r_dice_gain = float(np.sum(ex.dice_gain_calls)) if ex.dice_gain_calls else 0.0

    # Process: coverage
    annotated = set(ex.annotated)
    lesion_set = set(lesion_ordinals)
    non_lesion_set = set(non_lesion_ordinals)
    if lesion_set:
        recall = len(annotated & lesion_set) / len(lesion_set)
    else:
        recall = 1.0
    fp = len(annotated & non_lesion_set)
    r_coverage = float(recall - COVERAGE_FP_LAMBDA * fp)

    # Process: nav style
    nav_score = 0.0
    boundary_clamp_count = 0
    revisit_no_change = 0
    seen_ords: set = set()
    for d in ex.nav_decisions:
        target = d['target_ord']
        if d['tool'] == 'scroll':
            if d['clamped']:
                nav_score -= NAV_STYLE_CLAMP
                boundary_clamp_count += 1
            elif d['delta'] <= 2:
                nav_score += NAV_STYLE_REWARD
        elif d['tool'] == 'get_slice':
            if d['delta'] >= 3:
                nav_score += NAV_STYLE_REWARD
            if target in seen_ords and not d['overlay_changed']:
                nav_score -= NAV_STYLE_REVISIT
                revisit_no_change += 1
        seen_ords.add(target)
    r_nav_style = float(nav_score)

    # Process: efficiency — soft cap
    excess = max(0, n_tool_calls - EFFICIENCY_CAP)
    r_efficiency = float(-EFFICIENCY_PEN_RATE * excess)

    gate = 1 if not gate_violations else 0

    return {
        'r_dice': r_dice,
        'r_key_dice': r_key_dice,
        'r_bbox_iou': r_bbox_iou,
        'r_dice_gain': r_dice_gain,
        'r_coverage': r_coverage,
        'r_nav_style': r_nav_style,
        'r_efficiency': r_efficiency,
        'gate': gate,
        'gate_violations': gate_violations,
        'num_turns': n_tool_calls,
        'n_add_bbox': len(ex.bboxes_by_ord),
        'n_add_point': len(ex.dice_gain_calls),
        'n_run_medsam2': int(ex.run_medsam2_called),
        'n_finish': int(ex.finish_called),
        'boundary_clamp_count': boundary_clamp_count,
        'revisit_no_change_count': revisit_no_change,
        'key_z_chosen': ex.key_z_chosen,
        'bbox_iou_at_run': ex.bbox_iou_at_run,
    }


# ────────────────────────────────────────────────────────────────────────────
# Format-gate validation (§9.5)
# ────────────────────────────────────────────────────────────────────────────

def collect_gate_violations(
    parsed_calls: List[Tuple[str, Dict[str, Any]]],
    sampled_z: List[int],
) -> List[str]:
    """Return human-readable gate violations; empty list = pass."""
    violations: List[str] = []
    cur_ord: Optional[int] = None
    seen_get_slice = False
    n_run_medsam2 = 0
    n_finish = 0
    finish_idx = None

    for idx, (name, args) in enumerate(parsed_calls):
        if name == 'get_slice':
            seen_get_slice = True
            z = args.get('z_index')
            if isinstance(z, int) and z in sampled_z:
                cur_ord = sampled_z.index(z)
        elif name == 'scroll':
            if not seen_get_slice:
                violations.append('scroll_before_first_get_slice')
            if cur_ord is not None:
                cur_ord = max(0, min(len(sampled_z) - 1,
                                      cur_ord + int(args.get('delta', 0))))
        elif name in ('add_bbox', 'add_point'):
            z = args.get('z_index')
            if cur_ord is None or sampled_z[cur_ord] != z:
                violations.append(f'{name}_z_mismatch_at_{idx}')
        elif name == 'run_medsam2':
            n_run_medsam2 += 1
            key_z = args.get('key_z')
            if not isinstance(key_z, int) or key_z not in sampled_z:
                violations.append('run_medsam2_key_z_not_in_sampled')
        elif name == 'finish_3d_segmentation':
            n_finish += 1
            finish_idx = idx

    if n_run_medsam2 != 1:
        violations.append(f'run_medsam2_count={n_run_medsam2}')
    if n_finish != 1:
        violations.append(f'finish_count={n_finish}')
    elif finish_idx != len(parsed_calls) - 1:
        violations.append('finish_not_last')
    return violations
