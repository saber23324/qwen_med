#!/usr/bin/env python3
"""
Convert M3D data to ms-swift Phase-3 agent trajectory JSONL.

Phase 3 extends Phase 2 with two navigation tools — `get_slice(z_index)` and
`scroll(delta)` — so the agent loads sampled MRI slices ON DEMAND, one at a
time, rather than receiving all 10 upfront. The segmentation scope is the SAME
10 sampled Z slices as Phase 2; only the perceptual framing changes: the agent
must build its spatial model by reading slices sequentially and reasoning from
conversational context (Z indices, accumulated CoT, history block).

Trajectory structure (variable length, typically ~40 turns):

    Turn 1   [user]         Task + sampled Z list (NO slice images)
    Turn 2   [assistant]    Phase-A opening CoT
    Phase A  (5 reads)      get_slice(mid) + scroll zigzag across {mid-2..mid+2}
    Turn X   [assistant]    Phase-A summary CoT (tentative key slice)
    Phase B  (≥10 reads)    spiral walk 5→4→6→3→7→2→8→1→9→0
                            scroll if |Δ|≤2 else get_slice, add_bbox per lesion slice
    Turn Y   [assistant]    Phase-B summary + run_medsam2 announcement
             [tool_call]    run_medsam2(key_z, bbox)
             [tool_response] all N mask images + Dice per slice
    Turn Z   [assistant]    Phase-C opening CoT (mask review sweep)
    Phase C  (≥10 reads)    get_slice(Z[0]) + scroll(+1)×9 sweep
                            add_point at ordinals with Dice < 0.70
    Turn N   [assistant]    Final CoT
             [tool_call]    finish_3d_segmentation()

Tool set: add_bbox, run_medsam2, add_point, scroll, get_slice,
          finish_3d_segmentation.

Usage:
    # Quick validation (3 samples, CPU only, GT-derived masks):
    conda run -n qwen3 python3 Qwen3_VL/convert_to_agent_trajectory_phase3.py \\
        --data_root  /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \\
        --output_dir /tmp/phase3_test --max_samples 3 --device cpu

    # Full run with MedSAM2:
conda run -n dtos_test python3 Qwen3_VL/convert_to_agent_trajectory_phase3.py \
--data_root  /BDSZ6/private/user/yxd/data/M3D/data_18-22/train \
--output_dir /BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22 \
--ckpt  /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
--cfg   /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
--device cuda:4
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── MedSAM2 in sibling directory ─────────────────────────────────────────────
MEDSAM2_ROOT = Path(__file__).resolve().parents[1] / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

N_SLICES             = 10
JITTER_RATIO         = 0.05
FAILURE_INJECT_RATIO = 0.10
FAILURE_OFFSET_RATIO = 0.30
IOU_PASS_THRESHOLD   = 0.75
DICE_PASS_THRESHOLD  = 0.70
MIN_LESION_FRAMES    = 3
N_FG_POINTS          = 3
N_BG_POINTS          = 2
RENDER_SIZE          = 256   # 256x256 renders (~84 visual tokens each)

# ────────────────────────────────────────────────────────────────────────────
# Tool schema (Phase 3 = Phase 2 + scroll + get_slice)
# ────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_slice",
            "description": (
                "Jump directly to a sampled Z slice and read it. Use for non-adjacent moves — "
                "initial exploration, jumping to a slice identified in review, or re-checking "
                "after refinement. z_index must be one of the 10 sampled Z values echoed in the "
                "opening user turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "z_index": {
                        "type": "integer",
                        "description": "Target Z index; must be a member of the sampled Z list."
                    }
                },
                "required": ["z_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": (
                "Move the slice pointer by `delta` positions within the sampled Z list. "
                "delta=+1 moves to the next (higher-Z) sampled slice, delta=-1 to the previous. "
                "Use for adjacent moves (|delta| <= 2); for jumps of 3+ use get_slice. "
                "Out-of-bounds scrolls are clamped, not rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "delta": {
                        "type": "integer",
                        "description": "Step in sampled-list ordering; allowed range ±9, excluding 0."
                    }
                },
                "required": ["delta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_bbox",
            "description": (
                "Annotate the 2D bounding box of a lesion on the CURRENT slice. "
                "z_index must match the slice currently being viewed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "z_index": {"type": "integer", "description": "Z-axis index of the current slice."},
                    "bbox": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] pixel coordinates (top-left, bottom-right)."
                    }
                },
                "required": ["z_index", "bbox"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_medsam2",
            "description": (
                "Trigger MedSAM2 3D segmentation seeded from a single key slice. "
                "Propagates forward and backward through the volume. Call exactly once, "
                "after all lesion-containing slices have been annotated with add_bbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key_z": {"type": "integer",
                              "description": "Z index of the key (largest cross-section) slice."},
                    "bbox": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] bbox on the key slice."
                    }
                },
                "required": ["key_z", "bbox"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_point",
            "description": (
                "Refine the segmentation mask on the CURRENT slice using foreground / "
                "background point prompts. z_index must match the slice currently being viewed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "z_index": {"type": "integer", "description": "Z-axis index of the current slice."},
                    "points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"},
                                  "minItems": 2, "maxItems": 2},
                        "description": "List of [x, y] pixel coordinates."
                    },
                    "labels": {
                        "type": "array", "items": {"type": "integer", "enum": [0, 1]},
                        "description": "Per-point label: 1=foreground, 0=background."
                    }
                },
                "required": ["z_index", "points", "labels"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_3d_segmentation",
            "description": "Confirm that 3D segmentation and all refinements are complete. Terminates the task.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

TOOLS_JSON = json.dumps(TOOLS)


# ────────────────────────────────────────────────────────────────────────────
# Expression classifier (same as Phase 1/2)
# ────────────────────────────────────────────────────────────────────────────

_ORGAN_KW  = {'organ', 'kidney', 'liver', 'spleen', 'filtering', 'responsible',
               'pair of', 'essential for', 'producing urine', 'eliminating waste'}
_LESION_KW = {'tumor', 'tumour', 'cancer', 'mass', 'abnormal', 'malignant',
               'carcinoma', 'lesion', 'metastasis', 'cluster of cells',
               'uncontrolled cell', 'arising from', 'tissue formed'}
_CYST_KW   = {'cyst', 'fluid', 'pouch', 'cystic', 'encapsulated', 'non-malignant'}


def classify_expression(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _LESION_KW): return 'lesion'
    if any(k in t for k in _CYST_KW):   return 'cyst'
    if any(k in t for k in _ORGAN_KW):  return 'organ'
    return 'other'


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def mask_to_bbox(gt_2d):
    if gt_2d.sum() == 0: return None
    rows, cols = np.where(gt_2d > 0)
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def add_jitter(bbox, img_w, img_h, jitter_ratio=JITTER_RATIO):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    def j(v, scale, lo, hi):
        return int(np.clip(v + np.random.uniform(-scale, scale), lo, hi))
    nx1 = j(x1, bw * jitter_ratio, 0, img_w - 1)
    ny1 = j(y1, bh * jitter_ratio, 0, img_h - 1)
    nx2 = j(x2, bw * jitter_ratio, nx1 + 1, img_w - 1)
    ny2 = j(y2, bh * jitter_ratio, ny1 + 1, img_h - 1)
    return [nx1, ny1, nx2, ny2]


def inject_failure(bbox, img_w, img_h, offset_ratio=FAILURE_OFFSET_RATIO):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    ox, oy = int(bw * offset_ratio), int(bh * offset_ratio)
    nx1 = min(x1 + ox, img_w - bw - 1)
    ny1 = min(y1 + oy, img_h - bh - 1)
    return [nx1, ny1, nx1 + bw, ny1 + bh]


def compute_iou(bbox, gt_2d):
    if gt_2d.sum() == 0: return 1.0
    x1, y1, x2, y2 = bbox
    gt_rows, gt_cols = np.where(gt_2d > 0)
    gx1, gy1 = int(gt_cols.min()), int(gt_rows.min())
    gx2, gy2 = int(gt_cols.max()), int(gt_rows.max())
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (x2 - x1)*(y2 - y1) + (gx2 - gx1)*(gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else float(pred.sum() == 0 and gt.sum() == 0)


# ────────────────────────────────────────────────────────────────────────────
# Point prompt generation (same as Phase 2)
# ────────────────────────────────────────────────────────────────────────────

def sample_correction_points(pred_mask, gt_mask, n_fg=N_FG_POINTS, n_bg=N_BG_POINTS):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    fn_mask = gt & ~pred
    fp_mask = pred & ~gt
    points, labels = [], []

    fn_coords = np.argwhere(fn_mask)
    if len(fn_coords) >= n_fg:
        chosen = fn_coords[np.random.choice(len(fn_coords), n_fg, replace=False)]
        for r, c in chosen: points.append([int(c), int(r)]); labels.append(1)
    elif len(fn_coords) > 0:
        for r, c in fn_coords: points.append([int(c), int(r)]); labels.append(1)

    fp_coords = np.argwhere(fp_mask)
    if len(fp_coords) >= n_bg:
        chosen = fp_coords[np.random.choice(len(fp_coords), n_bg, replace=False)]
        for r, c in chosen: points.append([int(c), int(r)]); labels.append(0)
    elif len(fp_coords) > 0:
        for r, c in fp_coords: points.append([int(c), int(r)]); labels.append(0)

    return points, labels


# ────────────────────────────────────────────────────────────────────────────
# Image rendering
# ────────────────────────────────────────────────────────────────────────────

BBOX_COLOR     = (255, 80, 80)
POINT_FG_COLOR = (255, 50, 50)
POINT_BG_COLOR = (50, 100, 255)


def render_raw(img_path: str, save_path: str):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)
    img.save(save_path)


def render_bbox_overlay(img_path: str, bbox: list, save_path: str):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=2)
    img = img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)
    img.save(save_path)


def render_mask_overlay(img_path: str, mask: np.ndarray, save_path: str,
                         points=None, labels=None, bbox=None):
    img = Image.open(img_path).convert('RGBA')
    H, W = mask.shape
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    mask_px = np.zeros((H, W, 4), dtype=np.uint8)
    mask_px[mask.astype(bool)] = [0, 200, 100, 120]
    overlay.paste(Image.fromarray(mask_px, 'RGBA'))
    img = Image.alpha_composite(img, overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=2)
    if points and labels:
        for (x, y), lbl in zip(points, labels):
            color = POINT_FG_COLOR if lbl == 1 else POINT_BG_COLOR
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline='white')
    img = img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)
    img.save(save_path)


# ────────────────────────────────────────────────────────────────────────────
# MedSAM2 wrapper (reused verbatim from Phase 2)
# ────────────────────────────────────────────────────────────────────────────

import torch

IMG_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMG_STD  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def load_and_preprocess_volume(jpeg_dir: str, device) -> tuple:
    files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith('.jpg'))
    slices = [np.array(Image.open(os.path.join(jpeg_dir, f)).convert('L')) for f in files]
    vol = np.stack(slices)
    D, H, W = vol.shape
    out = np.zeros((D, 3, 512, 512), dtype=np.float32)
    for i in range(D):
        arr = np.array(Image.fromarray(vol[i]).convert('RGB')).transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = arr
    tensor = torch.from_numpy(out).to(device)
    mean = IMG_MEAN[:, None, None].to(device)
    std  = IMG_STD[:, None, None].to(device)
    tensor = (tensor - mean) / std
    return tensor, H, W


def run_medsam2_propagation(predictor, img_tensor, orig_H, orig_W,
                             key_z: int, bbox_512: np.ndarray, device) -> dict:
    result = {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=key_z, obj_id=1, box=bbox_512
        )
        key_mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        result[key_z] = key_mask.astype(bool)
        predictor.reset_state(state)

        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=False):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)

        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=True):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ────────────────────────────────────────────────────────────────────────────

def build_ordered_annos(meta, mask_dict):
    annos, anno_idx = [], 0
    for vid in sorted(meta.keys()):
        vd = meta[vid]
        frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):
            exp = vd['expressions'][eid]
            masks = mask_dict[str(anno_idx)]
            non_none_z = [i for i, m in enumerate(masks) if m is not None]
            annos.append({
                'vid': vid, 'eid': eid, 'caption': exp['exp'],
                'anno_id': str(anno_idx), 'frames': frames,
                'img_w': vd['width'], 'img_h': vd['height'],
                'masks': masks, 'non_none_z': non_none_z,
                'category': classify_expression(exp['exp']),
            })
            anno_idx += 1
    return annos


def sample_slices(non_none_z, n=N_SLICES):
    z_min, z_max = non_none_z[0], non_none_z[-1]
    return np.linspace(z_min, z_max, n, dtype=int).tolist()


# ────────────────────────────────────────────────────────────────────────────
# Navigation plan (Phase A / B / C)
# ────────────────────────────────────────────────────────────────────────────

def build_navigation_plan(sampled, areas, dice_per_ord):
    """
    Return (plan, key_ord) where plan is a list of ('op', *args) tuples.
    Navigation ops: ('get_slice', z_target, ordinal), ('scroll', delta, new_ordinal)
    Action ops:     ('add_bbox', z, ordinal), ('add_point', z, ordinal),
                    ('run_medsam2', key_z, key_ord),
                    ('phase_boundary', name), ('finish',)
    """
    mid = N_SLICES // 2   # ordinal 5 for a 10-slice list
    key_ord = int(np.argmax(areas))
    plan = []

    # ── Phase A: Exploration (5 reads) ─────────────────────────────────────
    # get_slice(mid) then zigzag scroll (-2, +1, +2, +1) → visits {mid-2..mid+2}
    plan.append(('phase_boundary', 'A'))
    plan.append(('get_slice', sampled[mid], mid))
    cur = mid
    for d in (-2, +1, +2, +1):
        new = cur + d
        plan.append(('scroll', d, new))
        cur = new
    plan.append(('phase_boundary', 'A_done'))

    # ── Phase B: Annotation walk (spiral outward from mid) ─────────────────
    # Visit every ordinal in |i-mid| order, add_bbox if lesion present.
    plan.append(('phase_boundary', 'B'))
    order = sorted(range(N_SLICES), key=lambda i: abs(i - mid))
    visited_b = set()
    for i in order:
        if i in visited_b: continue
        step = i - cur
        if step != 0:
            if abs(step) >= 3:
                plan.append(('get_slice', sampled[i], i))
            else:
                plan.append(('scroll', step, i))
            cur = i
        visited_b.add(i)
        if areas[i] > 0:
            plan.append(('add_bbox', sampled[i], i))
    plan.append(('phase_boundary', 'B_done'))

    # ── Transition: run_medsam2 ─────────────────────────────────────────────
    plan.append(('run_medsam2', sampled[key_ord], key_ord))
    plan.append(('phase_boundary', 'C'))

    # ── Phase C: Mask review sweep (get_slice(Z[0]) + scroll +1 × 9) ───────
    plan.append(('get_slice', sampled[0], 0))
    cur = 0
    if dice_per_ord.get(0, 1.0) < DICE_PASS_THRESHOLD:
        plan.append(('add_point', sampled[0], 0))
    for i in range(1, N_SLICES):
        plan.append(('scroll', +1, i))
        cur = i
        if dice_per_ord.get(i, 1.0) < DICE_PASS_THRESHOLD:
            plan.append(('add_point', sampled[i], i))
    plan.append(('phase_boundary', 'C_done'))
    plan.append(('finish',))
    return plan, key_ord


# ────────────────────────────────────────────────────────────────────────────
# Overlay-state-aware image selection
# ────────────────────────────────────────────────────────────────────────────

def pick_render_for_view(ordinal, overlay_state, renders):
    """
    Given the current overlay state of an ordinal, return the render path to
    bake into the navigation tool_response. Priority: refined_mask > mask > bbox > raw.
    """
    state = overlay_state.get(ordinal, 'raw')
    if state == 'refined_mask' and ordinal in renders['refined_mask']:
        return renders['refined_mask'][ordinal]
    if state == 'mask' and ordinal in renders['mask']:
        return renders['mask'][ordinal]
    if state == 'bbox' and ordinal in renders['bbox']:
        return renders['bbox'][ordinal]
    return renders['raw'][ordinal]


# ────────────────────────────────────────────────────────────────────────────
# CoT generators
# ────────────────────────────────────────────────────────────────────────────

def cot_phase_a_open():
    return (
        "Starting Phase A (exploration). I have no image yet — only the sampled Z list. "
        "I'll open from the middle slice to locate the likely key (largest cross-section), "
        "then probe the ±2 neighborhood with scroll to build an initial spatial model."
    )


def cot_phase_a_summary(sampled, areas, key_ord_tentative):
    key_z = sampled[key_ord_tentative]
    # Summarise the 5 central ordinals probed in Phase A (indices mid-2..mid+2)
    mid = N_SLICES // 2
    lines = ["Phase A complete. Cross-section summary across probed ordinals:"]
    for i in range(max(0, mid - 2), min(N_SLICES, mid + 3)):
        a = areas[i]
        tag = "lesion" if a > 0 else "empty"
        lines.append(f"  ord {i} (Z={sampled[i]}): area={a} px² ({tag})")
    lines.append(
        f"\nTentative key slice: Z={key_z} (ord {key_ord_tentative}). "
        f"Moving to Phase B — full annotation walk covering all 10 sampled slices."
    )
    return "\n".join(lines)


def cot_phase_b_summary(sampled, areas, bboxes_by_ord, key_ord):
    n_anno = sum(1 for a in areas if a > 0)
    return (
        f"Phase B complete. Annotated {n_anno}/{N_SLICES} lesion-containing slices "
        f"via add_bbox; {N_SLICES - n_anno} slice(s) had no visible target. "
        f"Confirmed key slice: Z={sampled[key_ord]} (ord {key_ord}) — largest cross-section. "
        f"Initiating 3D segmentation with run_medsam2."
    )


def cot_mask_review_open():
    return (
        "MedSAM2 propagation complete. Entering Phase C — scroll-based mask review. "
        "I'll sweep from the lowest sampled Z (ord 0) forward via scroll(+1), "
        "inspecting each per-slice Dice and calling add_point on any slice below 0.70."
    )


def cot_final(needs_point, dice_per_z, dice_after):
    if not needs_point:
        return (
            f"Sweep complete. All {N_SLICES} sampled slices met Dice ≥ {DICE_PASS_THRESHOLD}. "
            f"Segmentation finalised."
        )
    improved = sum(1 for z in needs_point if dice_after.get(z, 0) > dice_per_z.get(z, 0))
    return (
        f"Sweep complete. Refined {len(needs_point)} slice(s) via add_point "
        f"(Z={sorted(needs_point.keys())}); {improved} improved above threshold. "
        f"Segmentation finalised."
    )


# ────────────────────────────────────────────────────────────────────────────
# Trajectory builder
# ────────────────────────────────────────────────────────────────────────────

def build_trajectory(anno, jpeg_root, render_dir, predictor, device):
    """
    Build one Phase-3 agent trajectory. Returns dict {tools, messages, images}
    or None if the sample is unusable.
    """
    vid     = anno['vid']
    caption = anno['caption']
    frames  = anno['frames']
    masks   = anno['masks']
    img_w   = anno['img_w']
    img_h   = anno['img_h']
    non_none = anno['non_none_z']
    category = anno['category']

    if len(non_none) < MIN_LESION_FRAMES:
        return None

    sampled = sample_slices(non_none)
    jpeg_dir = os.path.join(jpeg_root, vid)

    def slice_path(z):
        return os.path.join(jpeg_dir, frames[z] + '.jpg')

    # ── Ground truth areas per ordinal ────────────────────────────────────
    areas = []
    gt_bboxes = {}       # ord -> gt bbox
    for i, z in enumerate(sampled):
        m = masks[z]
        if m is None:
            areas.append(0)
            continue
        gt_2d = maskUtils.decode(m)
        areas.append(int(gt_2d.sum()))
        bb = mask_to_bbox(gt_2d)
        if bb is not None:
            gt_bboxes[i] = bb

    if not gt_bboxes:
        return None

    key_ord = int(np.argmax(areas))
    key_z = sampled[key_ord]

    # ── Bbox with jitter (+optional failure on one ordinal) ───────────────
    bboxes_by_ord = {}
    for i, bb in gt_bboxes.items():
        bboxes_by_ord[i] = add_jitter(bb, img_w, img_h)

    if random.random() < FAILURE_INJECT_RATIO and bboxes_by_ord:
        fail_ord = random.choice(list(bboxes_by_ord.keys()))
        bboxes_by_ord[fail_ord] = inject_failure(gt_bboxes[fail_ord], img_w, img_h)

    # ── MedSAM2 propagation (or GT-derived fallback) ──────────────────────
    if predictor is not None:
        sx, sy = 512 / img_w, 512 / img_h
        kbb = bboxes_by_ord[key_ord]
        bbox_512 = np.array([kbb[0]*sx, kbb[1]*sy, kbb[2]*sx, kbb[3]*sy], dtype=np.float32)
        img_tensor, orig_H, orig_W = load_and_preprocess_volume(jpeg_dir, device)
        propagated_masks = run_medsam2_propagation(
            predictor, img_tensor, orig_H, orig_W, key_z, bbox_512, device
        )
    else:
        propagated_masks = {}
        from scipy import ndimage
        for z in sampled:
            m = masks[z]
            if m is None:
                propagated_masks[z] = np.zeros((512, 512), dtype=bool)
                continue
            gt_2d = maskUtils.decode(m).astype(bool)
            if np.random.random() < 0.5:
                propagated_masks[z] = ndimage.binary_dilation(gt_2d, iterations=np.random.randint(1, 4))
            else:
                propagated_masks[z] = ndimage.binary_erosion(gt_2d, iterations=np.random.randint(1, 3))

    dice_per_z   = {}
    dice_per_ord = {}
    for i, z in enumerate(sampled):
        pred = propagated_masks.get(z)
        m = masks[z]
        if pred is None or m is None:
            continue
        gt_2d = maskUtils.decode(m).astype(bool)
        d = round(dice_score(pred, gt_2d), 3)
        dice_per_z[z] = d
        dice_per_ord[i] = d

    # ── Point prompts for below-threshold slices ──────────────────────────
    needs_point = {}    # z -> {'points', 'labels', 'ordinal'}
    corrected_masks = {}
    corrected_dice  = {}
    from scipy import ndimage
    for i, z in enumerate(sampled):
        d = dice_per_z.get(z)
        m = masks[z]
        if d is None or m is None:
            continue
        if d >= DICE_PASS_THRESHOLD:
            continue
        gt_2d = maskUtils.decode(m).astype(bool)
        pred  = propagated_masks[z]
        pts, lbls = sample_correction_points(pred, gt_2d)
        if not pts:
            continue
        needs_point[z] = {'points': pts, 'labels': lbls, 'ordinal': i}
        iters = np.random.randint(0, 2)
        corrected = ndimage.binary_dilation(gt_2d, iterations=iters) if iters > 0 else gt_2d.copy()
        corrected_masks[z] = corrected
        corrected_dice[z]  = round(dice_score(corrected, gt_2d), 3)

    # ── Build the navigation plan ─────────────────────────────────────────
    plan, _ = build_navigation_plan(sampled, areas, dice_per_ord)

    # ── Pre-render all image variants we might need ───────────────────────
    os.makedirs(render_dir, exist_ok=True)
    renders = {'raw': {}, 'bbox': {}, 'mask': {}, 'refined_mask': {}}

    # raw renders (one per ordinal, always needed)
    for i, z in enumerate(sampled):
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_raw.png")
        if not os.path.exists(rp):
            render_raw(slice_path(z), rp)
        renders['raw'][i] = rp

    # bbox renders (one per lesion ordinal)
    for i, bb in bboxes_by_ord.items():
        z = sampled[i]
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_bbox.png")
        if not os.path.exists(rp):
            render_bbox_overlay(slice_path(z), bb, rp)
        renders['bbox'][i] = rp

    # mask renders (one per sampled ordinal with a propagated mask)
    for i, z in enumerate(sampled):
        pred = propagated_masks.get(z)
        if pred is None: continue
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_mask.png")
        if not os.path.exists(rp):
            render_mask_overlay(slice_path(z), pred, rp, bbox=bboxes_by_ord.get(i))
        renders['mask'][i] = rp

    # refined_mask renders (per ord that gets add_point)
    for z, info in needs_point.items():
        i = info['ordinal']
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_refined.png")
        if not os.path.exists(rp):
            render_mask_overlay(slice_path(z), corrected_masks[z], rp,
                                points=info['points'], labels=info['labels'])
        renders['refined_mask'][i] = rp

    # ── Walk the plan: emit messages + images ─────────────────────────────
    messages = []
    images = []

    _task_desc = {
        'organ':  "annotate the organ structure",
        'lesion': "annotate all lesion regions",
        'cyst':   "annotate all cyst regions",
    }.get(category, "annotate the target structure")

    # Turn 1: user — task + sampled Z list (NO slice images)
    user_text = (
        f"Task: Perform 3D segmentation on an MRI volume. The target structure is: "
        f"\"{caption}\". For each slice containing the target, {_task_desc} using add_bbox. "
        f"You will read slices ONE AT A TIME — use get_slice(z) to jump to any sampled Z, "
        f"or scroll(delta) to move by delta positions within the sampled list. Adjacent "
        f"moves (|delta| ≤ 2) should use scroll; jumps of 3+ should use get_slice. "
        f"After all lesion slices are annotated, call run_medsam2(key_z, bbox) exactly once "
        f"to propagate the mask through the volume, then scroll through the masks and "
        f"call add_point on any slice below Dice 0.70. Conclude with finish_3d_segmentation.\n\n"
        f"Sampled Z list ({N_SLICES} values, ordinal 0 → {N_SLICES - 1}): {sampled}\n"
        f"Lesion Z range in the full volume: Z={non_none[0]}~{non_none[-1]}."
    )
    messages.append({"role": "user", "content": user_text})

    # State for walking the plan
    overlay_state = {}                # ord -> 'raw'|'bbox'|'mask'|'refined_mask'
    visited = []                      # ordered list of ordinals visited
    annotated = set()                 # ordinals with add_bbox applied
    cur_ord = None
    have_mask_phase = False           # True after run_medsam2

    def history_block():
        return {
            'visited_ordinals': list(visited),
            'annotated_ordinals': sorted(annotated),
            'unvisited_ordinals': [i for i in range(N_SLICES) if i not in set(visited)],
        }

    def navigation_response(new_ord, applied_delta=None):
        """Emit the tool_response dict for a navigation read at `new_ord`."""
        z = sampled[new_ord]
        state = overlay_state.get(new_ord, 'raw')
        ov = {'has_bbox': new_ord in renders['bbox']
                            and state in ('bbox', 'mask', 'refined_mask'),
              'has_mask': state in ('mask', 'refined_mask'),
              'has_refined_mask': state == 'refined_mask'}
        if ov['has_bbox']:
            ov['bbox'] = bboxes_by_ord.get(new_ord)
        if ov['has_mask']:
            ov['mask_dice'] = dice_per_z.get(z, 0.0)
        if ov['has_refined_mask']:
            ov['refined_dice'] = corrected_dice.get(z, 0.0)
        return {
            'z_index': z,
            'ordinal': new_ord,
            'slice_image': '<image>',
            'sampled_z_list': list(sampled),
            'overlays': ov,
            'boundary': {
                'at_start': new_ord == 0,
                'at_end':   new_ord == N_SLICES - 1,
                'clamped':  False,
            },
            'history': history_block(),
        }

    for op in plan:
        tag = op[0]

        if tag == 'phase_boundary':
            phase = op[1]
            if phase == 'A':
                messages.append({"role": "assistant", "content": cot_phase_a_open()})
            elif phase == 'A_done':
                messages.append({"role": "assistant",
                                 "content": cot_phase_a_summary(sampled, areas, key_ord)})
            elif phase == 'B':
                messages.append({"role": "assistant",
                                 "content": "Entering Phase B — spiral walk outward from the "
                                            "middle, annotating every sampled slice and "
                                            "switching to get_slice for jumps of 3+ ordinals."})
            elif phase == 'B_done':
                messages.append({"role": "assistant",
                                 "content": cot_phase_b_summary(sampled, areas,
                                                                bboxes_by_ord, key_ord)})
            elif phase == 'C':
                messages.append({"role": "assistant", "content": cot_mask_review_open()})
            elif phase == 'C_done':
                messages.append({"role": "assistant",
                                 "content": cot_final(needs_point, dice_per_z, corrected_dice)})

        elif tag == 'get_slice':
            _, z_target, new_ord = op
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "get_slice", "arguments": {"z_index": z_target}
            })})
            cur_ord = new_ord
            visited.append(new_ord)
            images.append(pick_render_for_view(new_ord, overlay_state, renders))
            messages.append({"role": "tool_response",
                             "content": json.dumps(navigation_response(new_ord))})

        elif tag == 'scroll':
            _, delta, new_ord = op
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "scroll", "arguments": {"delta": delta}
            })})
            cur_ord = new_ord
            visited.append(new_ord)
            images.append(pick_render_for_view(new_ord, overlay_state, renders))
            messages.append({"role": "tool_response",
                             "content": json.dumps(navigation_response(new_ord))})

        elif tag == 'add_bbox':
            _, z_target, i = op
            bb = bboxes_by_ord[i]
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "add_bbox",
                "arguments": {"z_index": z_target, "bbox": bb}
            })})
            m = masks[z_target]
            gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
            iou = round(compute_iou(bb, gt_2d), 3)
            messages.append({"role": "tool_response", "content": json.dumps({
                "z_index": z_target, "bbox_image": "<image>", "iou_with_gt": iou
            })})
            images.append(renders['bbox'][i])
            annotated.add(i)
            overlay_state[i] = 'bbox'

        elif tag == 'run_medsam2':
            _, kz, kord = op
            kbb = bboxes_by_ord[kord]
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "run_medsam2",
                "arguments": {"key_z": kz, "bbox": kbb}
            })})
            slices_out = []
            for i, z in enumerate(sampled):
                d = dice_per_z.get(z, -1.0)
                if i in renders['mask']:
                    slices_out.append({"z_index": z, "ordinal": i,
                                       "mask_image": "<image>", "dice_with_gt": d})
                    images.append(renders['mask'][i])
                else:
                    slices_out.append({"z_index": z, "ordinal": i,
                                       "mask_image": None, "dice_with_gt": d})
            messages.append({"role": "tool_response", "content": json.dumps({
                "status": "propagation_complete",
                "key_z": kz,
                "slices": slices_out,
            })})
            # All slices with propagated masks are now in 'mask' state.
            for i in range(N_SLICES):
                if i in renders['mask']:
                    overlay_state[i] = 'mask'
            have_mask_phase = True

        elif tag == 'add_point':
            _, z_target, i = op
            info = needs_point[z_target]
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "add_point",
                "arguments": {
                    "z_index": z_target,
                    "points": info['points'],
                    "labels": info['labels'],
                }
            })})
            d = corrected_dice.get(z_target, 0.0)
            messages.append({"role": "tool_response", "content": json.dumps({
                "z_index": z_target, "mask_image": "<image>", "dice_with_gt": d
            })})
            images.append(renders['refined_mask'][i])
            overlay_state[i] = 'refined_mask'

        elif tag == 'finish':
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "finish_3d_segmentation", "arguments": {}
            })})

    return {"tools": TOOLS_JSON, "messages": messages, "images": images}


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

VALID_TOOL_NAMES = {"get_slice", "scroll", "add_bbox", "run_medsam2",
                    "add_point", "finish_3d_segmentation"}


def validate_entry(entry):
    msgs = entry["messages"]
    images = entry["images"]

    # <image> tokens == images length
    n_img_tokens = sum(m["content"].count("<image>") for m in msgs if m["content"])
    if n_img_tokens != len(images):
        return False, f"<image> tokens ({n_img_tokens}) != images ({len(images)})"

    tool_calls = [m for m in msgs if m["role"] == "tool_call"]
    parsed_calls = []
    for tc in tool_calls:
        try:
            obj = json.loads(tc["content"])
        except Exception as e:
            return False, f"tool_call JSON error: {e}"
        name = obj.get("name")
        if name not in VALID_TOOL_NAMES:
            return False, f"Unknown tool: {name}"
        parsed_calls.append(obj)

    if not parsed_calls:
        return False, "no tool calls"
    if parsed_calls[-1]["name"] != "finish_3d_segmentation":
        return False, f"last tool_call must be finish_3d_segmentation, got {parsed_calls[-1]['name']}"

    # Exactly one run_medsam2
    n_medsam2 = sum(1 for c in parsed_calls if c["name"] == "run_medsam2")
    if n_medsam2 != 1:
        return False, f"run_medsam2 must appear exactly once, got {n_medsam2}"

    # scroll: delta != 0
    for c in parsed_calls:
        if c["name"] == "scroll":
            if c["arguments"].get("delta") == 0:
                return False, "scroll delta must not be 0"
        elif c["name"] == "add_bbox":
            bb = c["arguments"].get("bbox", [])
            if len(bb) != 4 or not (bb[0] < bb[2] and bb[1] < bb[3]):
                return False, f"invalid bbox: {bb}"
        elif c["name"] == "add_point":
            pts = c["arguments"].get("points", [])
            lbls = c["arguments"].get("labels", [])
            if len(pts) != len(lbls) or not pts:
                return False, "points/labels mismatch or empty"

    # First navigation-expecting tool must be get_slice (scroll requires initialised pointer)
    for c in parsed_calls:
        if c["name"] in ("get_slice", "scroll"):
            if c["name"] == "scroll":
                return False, "first navigation call must be get_slice (pointer not initialised)"
            break

    # Invariant: add_bbox and add_point z_index must match last navigation ordinal's Z.
    # We walk the call sequence and track i_cur via sampled_z_list from context.
    # (We can't recover sampled_z_list from entry without parsing tool_responses, so just
    #  verify the z_index of each add_bbox/add_point matches the immediately preceding
    #  navigation tool_call's target Z.)
    last_nav_z = None
    for c in parsed_calls:
        name = c["name"]
        if name == "get_slice":
            last_nav_z = c["arguments"].get("z_index")
        elif name == "scroll":
            # scroll doesn't give z directly; rely on tool_response instead (relaxed check)
            last_nav_z = None
        elif name in ("add_bbox", "add_point"):
            target_z = c["arguments"].get("z_index")
            if last_nav_z is not None and target_z != last_nav_z:
                return False, (
                    f"{name}(z={target_z}) does not match last get_slice target "
                    f"z={last_nav_z}"
                )
        elif name == "run_medsam2":
            last_nav_z = None

    # Every tool_call except finish_3d_segmentation must have exactly one tool_response
    n_responses = sum(1 for m in msgs if m["role"] == "tool_response")
    n_calls_needing_response = sum(
        1 for c in parsed_calls if c["name"] != "finish_3d_segmentation"
    )
    if n_responses != n_calls_needing_response:
        return False, (f"responses ({n_responses}) != calls expecting a response "
                       f"({n_calls_needing_response})")

    return True, "ok"


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert M3D data to Phase-3 agent trajectory JSONL "
                    "(navigation via scroll + get_slice)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_root',   default='/BDSZ6/private/user/yxd/data/M3D/data_6-13/train')
    p.add_argument('--output_dir',  default='/BDSZ6/private/user/yxd/data/qwen/agent_phase3')
    p.add_argument('--ckpt',        default=None)
    p.add_argument('--cfg',         default=None)
    p.add_argument('--device',      default='cuda:4')
    p.add_argument('--train_ratio', type=float, default=0.9)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--max_samples', type=int,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    predictor = None
    device = torch.device('cpu')
    if args.ckpt and args.cfg:
        from sam2.build_sam import build_sam2_video_predictor_npz
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        cfg_path = args.cfg
        if os.path.isabs(cfg_path) and not cfg_path.startswith('//'):
            cfg_path = '//' + cfg_path
        torch.set_float32_matmul_precision('high')
        predictor = build_sam2_video_predictor_npz(
            cfg_path, args.ckpt, device=device, apply_postprocessing=False
        )
        print(f"[MedSAM2] Loaded: {args.ckpt} on {device}")
    else:
        print("[MedSAM2] No checkpoint — using GT-derived approximate masks.")

    meta_path  = os.path.join(args.data_root, 'meta_expressions.json')
    mask_path  = os.path.join(args.data_root, 'mask_dict.pkl')
    jpeg_root  = os.path.join(args.data_root, 'JPEGImages')
    render_dir = os.path.join(args.output_dir, 'renders')

    with open(meta_path) as f: meta = json.load(f)['videos']
    with open(mask_path, 'rb') as f: mask_dict = pickle.load(f)

    annos = build_ordered_annos(meta, mask_dict)
    valid_annos = [a for a in annos
                   if a['category'] in {'organ', 'lesion', 'cyst'}
                   and len(a['non_none_z']) >= MIN_LESION_FRAMES]
    print(f"Total expressions: {len(annos)} → valid: {len(valid_annos)}")

    if args.max_samples:
        valid_annos = valid_annos[:args.max_samples]
        print(f"Capped to {args.max_samples} samples.")

    all_vids   = sorted(meta.keys())
    split_idx  = int(len(all_vids) * args.train_ratio)
    train_vids = set(all_vids[:split_idx])

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'agent_train.jsonl')
    val_path   = os.path.join(args.output_dir, 'agent_val.jsonl')

    stats = {'train': {'ok': 0, 'skip': 0, 'invalid': 0},
             'val':   {'ok': 0, 'skip': 0, 'invalid': 0}}

    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for anno in tqdm(valid_annos, desc='Building Phase-3 trajectories'):
            split = 'train' if anno['vid'] in train_vids else 'val'
            f_out = f_train if split == 'train' else f_val
            try:
                entry = build_trajectory(anno, jpeg_root, render_dir, predictor, device)
            except Exception as e:
                tqdm.write(f"  SKIP {anno['vid']} exp{anno['eid']}: {e}")
                stats[split]['skip'] += 1
                continue
            if entry is None:
                stats[split]['skip'] += 1
                continue
            ok, reason = validate_entry(entry)
            if not ok:
                tqdm.write(f"  INVALID {anno['vid']} exp{anno['eid']}: {reason}")
                stats[split]['invalid'] += 1
                continue
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            stats[split]['ok'] += 1

    print("\n=== Phase-3 Conversion Summary ===")
    for sp in ('train', 'val'):
        s = stats[sp]
        print(f"{sp.capitalize():>6}: written={s['ok']}  skipped={s['skip']}  invalid={s['invalid']}")
    print(f"\nOutput:  {train_path}\n         {val_path}")
    print(f"Renders: {render_dir}/")


if __name__ == '__main__':
    main()
