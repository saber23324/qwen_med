#!/usr/bin/env python3
"""
Convert M3D data to ms-swift Phase-2 agent trajectory JSONL.

Phase 2 extends Phase 1 (bbox annotation) with MedSAM2-based 3D segmentation
and optional point-based refinement. Each trajectory follows this structure:

  Turn 1   [user]           Task instruction + 10 Z-indexed MRI slices
  Turn 2   [assistant]      Global spatial analysis CoT
  Turn 3   [tool_call ×N]   Parallel add_bbox (one per lesion-containing slice)
  Turn 4   [tool_resp ×N]   Rendered bbox feedback (with IoU)
  Turn 5   [assistant]      Bbox review CoT
  Turn 6   [tool_call ×M]   Correction add_bbox (M slices with IoU < threshold)
  Turn 7   [tool_resp ×M]   Correction bbox feedback
  Turn 8   [assistant]      "Verified. Initiating 3D segmentation from key slice Z=K ..."
  Turn 9   [tool_call]      run_medsam2(key_z=K, bbox=[...])
  Turn 10  [tool_response]  Initial masks for all N sampled slices ({z, mask_image, dice})
  Turn 11  [assistant]      Mask review CoT (flag slices with Dice < DICE_PASS)
  Turn 12  [tool_call ×P]   Parallel add_point for poor slices (P ≤ N)
  Turn 13  [tool_resp ×P]   Corrected mask + Dice per refined slice
  Turn 14  [assistant]      Final CoT → "Segmentation complete."
  Turn 15  [tool_call]      finish_3d_segmentation

The tool_response for run_medsam2 packs all N mask images into one response
(inline "mask_images" list) so that the single run_medsam2 call maps to exactly
one tool_response, satisfying the ms-swift 1:1 pairing requirement.

Usage:
    # Quick test (3 samples, no GPU needed for data-only path):
    python Qwen3_VL/convert_to_agent_trajectory_phase2.py \\
        --data_root  /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \\
        --output_dir /BDSZ6/private/user/yxd/data/qwen/agent_phase2 \\
        --max_samples 3 --device cpu

    # Full run with MedSAM2 on GPU:
    python Qwen3_VL/convert_to_agent_trajectory_phase2.py \\
        --data_root  /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \\
        --output_dir /BDSZ6/private/user/yxd/data/qwen/agent_phase2 \\
        --ckpt  /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \\
        --cfg   /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \\
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

N_SLICES            = 10
JITTER_RATIO        = 0.05
FAILURE_INJECT_RATIO = 0.10
FAILURE_OFFSET_RATIO = 0.30
IOU_PASS_THRESHOLD  = 0.75   # bbox quality gate
DICE_PASS_THRESHOLD = 0.70   # mask quality gate → triggers add_point
MIN_LESION_FRAMES   = 3
N_FG_POINTS         = 3      # foreground correction points per slice
N_BG_POINTS         = 2      # background correction points per slice

# ────────────────────────────────────────────────────────────────────────────
# Tool schema (Phase 2)
# ────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_bbox",
            "description": (
                "Annotate the 2D bounding box of a lesion on the specified Z slice. "
                "Must be called for every slice containing a lesion; do not call if no lesion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "z_index": {"type": "integer", "description": "Z-axis index of the target slice"},
                    "bbox": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] pixel coordinates (top-left, bottom-right)"
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
                "Propagates the segmentation forward and backward through the entire volume. "
                "Returns one mask per sampled slice. "
                "Call exactly once, after all bbox annotations are verified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key_z": {
                        "type": "integer",
                        "description": "Z index of the key (largest cross-section) slice"
                    },
                    "bbox": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[x1, y1, x2, y2] bbox on the key slice to seed MedSAM2"
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
                "Refine the segmentation mask on a specific Z slice using foreground / "
                "background point prompts. Call for each slice whose Dice score is below "
                f"{DICE_PASS_THRESHOLD}. Returns the updated mask image and Dice score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "z_index": {"type": "integer", "description": "Z-axis index of the slice to refine"},
                    "points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                        "description": "List of [x, y] pixel coordinates for point prompts"
                    },
                    "labels": {
                        "type": "array", "items": {"type": "integer", "enum": [0, 1]},
                        "description": "Label for each point: 1 = foreground, 0 = background"
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
            "description": (
                "Confirm that 3D segmentation and all point refinements are complete. "
                "Terminates the task."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

TOOLS_JSON = json.dumps(TOOLS)


# ────────────────────────────────────────────────────────────────────────────
# Expression classifier (same as Phase 1)
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
# Point prompt generation (for add_point corrections)
# ────────────────────────────────────────────────────────────────────────────

def sample_correction_points(pred_mask: np.ndarray, gt_mask: np.ndarray,
                             n_fg: int = N_FG_POINTS, n_bg: int = N_BG_POINTS):
    """
    Sample foreground and background point prompts from the error regions.

    Foreground points: GT=1 but pred=0 (false negatives — missed lesion area)
    Background points: GT=0 but pred=1 (false positives — over-segmented area)

    Returns:
        points: list of [x, y]
        labels: list of int (1=foreground, 0=background)
    """
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    fn_mask = gt & ~pred   # missed foreground
    fp_mask = pred & ~gt   # over-predicted background

    points, labels = [], []

    # Sample foreground corrections
    fn_coords = np.argwhere(fn_mask)
    if len(fn_coords) >= n_fg:
        chosen = fn_coords[np.random.choice(len(fn_coords), n_fg, replace=False)]
        for r, c in chosen:
            points.append([int(c), int(r)])   # [x, y]
            labels.append(1)
    elif len(fn_coords) > 0:
        for r, c in fn_coords:
            points.append([int(c), int(r)])
            labels.append(1)

    # Sample background corrections
    fp_coords = np.argwhere(fp_mask)
    if len(fp_coords) >= n_bg:
        chosen = fp_coords[np.random.choice(len(fp_coords), n_bg, replace=False)]
        for r, c in chosen:
            points.append([int(c), int(r)])
            labels.append(0)
    elif len(fp_coords) > 0:
        for r, c in fp_coords:
            points.append([int(c), int(r)])
            labels.append(0)

    return points, labels


# ────────────────────────────────────────────────────────────────────────────
# Image rendering
# ────────────────────────────────────────────────────────────────────────────

BBOX_COLOR = (255, 80, 80)     # red
MASK_COLOR = (0, 200, 100, 100) # green with alpha
POINT_FG_COLOR = (255, 50, 50)  # red dot = foreground point
POINT_BG_COLOR = (50, 100, 255) # blue dot = background point


RENDER_SIZE = 256  # render images at 256×256 to reduce visual tokens (~84 vs ~333 for 512×512)


def render_bbox_overlay(img_path: str, bbox: list, save_path: str):
    img = Image.open(img_path).convert('RGB')
    orig_W, orig_H = img.size
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=2)
    img = img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)
    img.save(save_path)


def render_mask_overlay(img_path: str, mask: np.ndarray, save_path: str,
                        points=None, labels=None):
    """
    Draw a semi-transparent green mask + optional correction points on the image.
    Saved at RENDER_SIZE×RENDER_SIZE to reduce visual token count during training.
    """
    img = Image.open(img_path).convert('RGBA')
    H, W = mask.shape

    # Create mask overlay
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    mask_px = np.zeros((H, W, 4), dtype=np.uint8)
    mask_px[mask.astype(bool)] = [0, 200, 100, 120]
    overlay.paste(Image.fromarray(mask_px, 'RGBA'))
    img = Image.alpha_composite(img, overlay).convert('RGB')

    # Draw correction points (scale to original size before resizing)
    if points and labels:
        draw = ImageDraw.Draw(img)
        for (x, y), lbl in zip(points, labels):
            color = POINT_FG_COLOR if lbl == 1 else POINT_BG_COLOR
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline='white')

    img = img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)
    img.save(save_path)


# ────────────────────────────────────────────────────────────────────────────
# MedSAM2 wrapper (lazy-import; skipped when predictor=None)
# ────────────────────────────────────────────────────────────────────────────

import torch

IMG_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMG_STD  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def load_and_preprocess_volume(jpeg_dir: str, device) -> tuple:
    """Load all JPEG slices → normalised (D,3,512,512) tensor. Returns (tensor, H, W)."""
    files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith('.jpg'))
    slices = [np.array(Image.open(os.path.join(jpeg_dir, f)).convert('L')) for f in files]
    vol = np.stack(slices)             # (D, H, W)
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
                             key_z: int, bbox_512: np.ndarray,
                             device) -> dict[int, np.ndarray]:
    """
    Seed MedSAM2 with bbox on key_z, propagate forward+backward.
    Returns {frame_idx: bool mask (512,512)}.
    """
    result = {}

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        # Get key-slice mask from bbox
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=key_z, obj_id=1, box=bbox_512
        )
        key_mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        result[key_z] = key_mask.astype(bool)
        predictor.reset_state(state)

        # Forward propagation
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=False):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)

        # Backward propagation
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=True):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)

    return result


# ────────────────────────────────────────────────────────────────────────────
# Annotation helpers (same as Phase 1)
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
# CoT generators
# ────────────────────────────────────────────────────────────────────────────

def generate_spatial_cot(sampled, masks, caption):
    areas = []
    for z in sampled:
        m = masks[z]
        areas.append(int(maskUtils.decode(m).sum()) if m is not None else 0)
    has_lesion = [z for z, a in zip(sampled, areas) if a > 0]
    no_lesion  = [z for z, a in zip(sampled, areas) if a == 0]
    if not has_lesion:
        return None, None
    key_z = sampled[int(np.argmax(areas))]
    peak_idx = int(np.argmax(areas))
    if abs(peak_idx - (N_SLICES // 2 - 1)) <= 1:
        trend = "gradually increases from both ends toward the middle — near-symmetric ellipsoid"
    elif peak_idx >= N_SLICES // 2:
        trend = "continuously grows from bottom to top — lesion skewed toward upper slices"
    else:
        trend = "largest at the bottom, shrinking upward — lesion skewed toward lower slices"
    cot = (
        f"Global spatial analysis:\n\n"
        f"Target: {caption}\n\n"
        f"Target visible at Z={has_lesion[0]}~{has_lesion[-1]} "
        f"({len(has_lesion)} of {N_SLICES} sampled slices contain the target). "
        f"Z={key_z} has the largest cross-section (key slice for MedSAM2 seeding). "
        f"Cross-sectional area trend: {trend}.\n"
    )
    if no_lesion:
        cot += f"Z={no_lesion} — target not visible, skipping annotation.\n"
    cot += "\nProceeding to annotate all target-containing slices:"
    return cot, key_z


def generate_bbox_review_cot(initial_bboxes, masks, needs_correction):
    lines = ["Bbox review:"]
    for z, pred_bb in sorted(initial_bboxes.items()):
        m = masks[z]
        gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
        iou = compute_iou(pred_bb, gt_2d)
        status = f"FAIL (IoU={iou:.2f})" if z in needs_correction else f"PASS (IoU={iou:.2f})"
        lines.append(f"  Z={z}: {status}")
    if needs_correction:
        lines.append(f"\nRe-annotating {len(needs_correction)} slice(s): Z={sorted(needs_correction.keys())}.")
    return "\n".join(lines)


def generate_mask_review_cot(sampled, dice_per_z, needs_point):
    lines = ["MedSAM2 segmentation review:"]
    for z in sampled:
        d = dice_per_z.get(z, -1.0)
        if d < 0:
            lines.append(f"  Z={z}: no GT (skip)")
        elif z in needs_point:
            lines.append(f"  Z={z}: FAIL (Dice={d:.2f} < {DICE_PASS_THRESHOLD}) — adding correction points.")
        else:
            lines.append(f"  Z={z}: PASS (Dice={d:.2f})")
    if needs_point:
        lines.append(f"\nAdding point corrections for Z={sorted(needs_point.keys())}.")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Trajectory builder
# ────────────────────────────────────────────────────────────────────────────

def build_trajectory(anno, jpeg_root, render_dir, predictor, device):
    """
    Build one complete Phase-2 agent trajectory JSONL entry.
    Returns dict {tools, messages, images} or None if unusable.
    """
    vid      = anno['vid']
    caption  = anno['caption']
    frames   = anno['frames']
    masks    = anno['masks']
    img_w    = anno['img_w']
    img_h    = anno['img_h']
    non_none = anno['non_none_z']

    if len(non_none) < MIN_LESION_FRAMES:
        return None

    # ── 1. Sample slices + spatial CoT ──────────────────────────────────────
    sampled = sample_slices(non_none)
    cot, key_z = generate_spatial_cot(sampled, masks, caption)
    if cot is None:
        return None

    jpeg_dir = os.path.join(jpeg_root, vid)

    def slice_path(z):
        return os.path.join(jpeg_dir, frames[z] + '.jpg')

    # ── 2. Phase-1 bbox annotation ───────────────────────────────────────────
    gt_bboxes = {}
    initial_bboxes = {}
    for z in sampled:
        m = masks[z]
        if m is None: continue
        gt_2d = maskUtils.decode(m)
        gt_bb = mask_to_bbox(gt_2d)
        if gt_bb is None: continue
        gt_bboxes[z]      = gt_bb
        initial_bboxes[z] = add_jitter(gt_bb, img_w, img_h)

    if not initial_bboxes:
        return None

    # Optionally inject one bbox failure
    if random.random() < FAILURE_INJECT_RATIO:
        fail_z = random.choice(list(initial_bboxes.keys()))
        initial_bboxes[fail_z] = inject_failure(gt_bboxes[fail_z], img_w, img_h)

    # Identify bbox corrections
    needs_correction = {}
    for z, pred_bb in initial_bboxes.items():
        m = masks[z]
        if m is None: continue
        gt_2d = maskUtils.decode(m)
        if compute_iou(pred_bb, gt_2d) < IOU_PASS_THRESHOLD:
            needs_correction[z] = add_jitter(gt_bboxes[z], img_w, img_h, jitter_ratio=0.02)

    # Final bboxes (after correction)
    final_bboxes = {**initial_bboxes, **needs_correction}

    # ── 3. Run MedSAM2 propagation ───────────────────────────────────────────
    if predictor is not None:
        # Scale key-z bbox from img space to 512 space
        sx, sy = 512 / img_w, 512 / img_h
        kbb = final_bboxes[key_z]
        bbox_512 = np.array([kbb[0]*sx, kbb[1]*sy, kbb[2]*sx, kbb[3]*sy], dtype=np.float32)
        img_tensor, orig_H, orig_W = load_and_preprocess_volume(jpeg_dir, device)
        propagated_masks = run_medsam2_propagation(
            predictor, img_tensor, orig_H, orig_W, key_z, bbox_512, device
        )
    else:
        # Fallback: derive approximate masks from GT with erosion jitter
        propagated_masks = {}
        for z in sampled:
            m = masks[z]
            if m is None:
                propagated_masks[z] = np.zeros((512, 512), dtype=bool)
            else:
                gt_2d = maskUtils.decode(m).astype(bool)
                # Simulate slight imperfection: dilate/erode randomly
                from scipy import ndimage
                if np.random.random() < 0.5:
                    propagated_masks[z] = ndimage.binary_dilation(gt_2d, iterations=np.random.randint(1, 4))
                else:
                    propagated_masks[z] = ndimage.binary_erosion(gt_2d, iterations=np.random.randint(1, 3))

    # Compute Dice per sampled slice
    dice_per_z = {}
    for z in sampled:
        pred = propagated_masks.get(z)
        m = masks[z]
        if pred is None or m is None:
            continue
        gt_2d = maskUtils.decode(m).astype(bool)
        dice_per_z[z] = round(dice_score(pred, gt_2d), 3)

    # Identify slices needing add_point correction
    needs_point = {}
    for z in sampled:
        d = dice_per_z.get(z)
        m = masks[z]
        if d is None or m is None:
            continue
        if d < DICE_PASS_THRESHOLD:
            gt_2d = maskUtils.decode(m).astype(bool)
            pred  = propagated_masks[z]
            pts, lbls = sample_correction_points(pred, gt_2d)
            if pts:  # only add if we have valid points
                needs_point[z] = {'points': pts, 'labels': lbls}

    # Corrected masks (GT with slight jitter, simulating perfect point response)
    corrected_masks = {}
    corrected_dice  = {}
    for z, info in needs_point.items():
        m = masks[z]
        if m is None: continue
        gt_2d = maskUtils.decode(m).astype(bool)
        # Simulate a "near-perfect" corrected mask (GT + tiny random jitter)
        from scipy import ndimage
        iters = np.random.randint(0, 2)
        if iters > 0:
            corrected = ndimage.binary_dilation(gt_2d, iterations=iters)
        else:
            corrected = gt_2d.copy()
        corrected_masks[z] = corrected
        corrected_dice[z]  = round(dice_score(corrected, gt_2d), 3)

    # ── 4. Render images ─────────────────────────────────────────────────────
    os.makedirs(render_dir, exist_ok=True)

    # Bbox renders (initial)
    init_bbox_render = {}
    for z, bb in initial_bboxes.items():
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_bbox_init.png")
        if not os.path.exists(rp):
            render_bbox_overlay(slice_path(z), bb, rp)
        init_bbox_render[z] = rp

    # Bbox renders (corrected)
    corr_bbox_render = {}
    for z, bb in needs_correction.items():
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_bbox_corr.png")
        if not os.path.exists(rp):
            render_bbox_overlay(slice_path(z), bb, rp)
        corr_bbox_render[z] = rp

    # Mask renders (initial MedSAM2 output for sampled slices)
    init_mask_render = {}
    for z in sampled:
        pred = propagated_masks.get(z)
        if pred is None: continue
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_mask_init.png")
        if not os.path.exists(rp):
            render_mask_overlay(slice_path(z), pred, rp)
        init_mask_render[z] = rp

    # Mask renders (after add_point correction)
    corr_mask_render = {}
    for z, mask in corrected_masks.items():
        pts  = needs_point[z]['points']
        lbls = needs_point[z]['labels']
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_mask_corr.png")
        if not os.path.exists(rp):
            render_mask_overlay(slice_path(z), mask, rp, points=pts, labels=lbls)
        corr_mask_render[z] = rp

    # ── 5. Assemble messages + images ────────────────────────────────────────
    messages = []
    images   = []

    # Turn 1: user — task + slices
    category   = anno['category']
    _task_desc = {
        'organ':  "annotate the organ structure",
        'lesion': "annotate all lesion regions",
        'cyst':   "annotate all cyst regions",
    }.get(category, "annotate the target structure")

    user_lines = [
        f"Task: Examine the following {N_SLICES} MRI slices sampled uniformly from a 3D volume. "
        f"The target structure is: \"{caption}\". "
        f"For each slice containing the target, {_task_desc} using add_bbox. "
        f"After bbox review, call run_medsam2 to generate the 3D segmentation, "
        f"then refine any poorly-segmented slices with add_point. "
        f"Conclude with finish_3d_segmentation.\n"
    ]
    for z in sampled:
        user_lines.append(f"Z={z}: <image>")
        images.append(slice_path(z))
    messages.append({"role": "user", "content": "\n".join(user_lines)})

    # Turn 2: assistant — spatial CoT
    messages.append({"role": "assistant", "content": cot})

    # Turn 3: tool_calls — parallel add_bbox
    for z in sampled:
        if z not in initial_bboxes: continue
        messages.append({"role": "tool_call", "content": json.dumps({
            "name": "add_bbox",
            "arguments": {"z_index": z, "bbox": initial_bboxes[z]}
        })})

    # Turn 4: tool_responses — initial bbox feedback
    for z in sampled:
        if z not in initial_bboxes: continue
        m = masks[z]
        gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
        iou = compute_iou(initial_bboxes[z], gt_2d)
        messages.append({"role": "tool_response", "content": json.dumps({
            "z_index": z, "bbox_image": "<image>", "iou_with_gt": round(iou, 3)
        })})
        images.append(init_bbox_render[z])

    # Turn 5+6+7: bbox review CoT + optional corrections
    bbox_review = generate_bbox_review_cot(initial_bboxes, masks, needs_correction)

    if needs_correction:
        messages.append({"role": "assistant", "content": bbox_review})
        for z in sorted(needs_correction.keys()):
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "add_bbox",
                "arguments": {"z_index": z, "bbox": needs_correction[z]}
            })})
        for z in sorted(needs_correction.keys()):
            m = masks[z]
            gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
            iou = compute_iou(needs_correction[z], gt_2d)
            messages.append({"role": "tool_response", "content": json.dumps({
                "z_index": z, "bbox_image": "<image>", "iou_with_gt": round(iou, 3)
            })})
            images.append(corr_bbox_render[z])

    # Turn 8: assistant — bbox verified, initiating MedSAM2
    kbb = final_bboxes[key_z]
    medsam2_cot = (
        f"{bbox_review}\n\n"
        if not needs_correction else
        f"Corrections applied. All {len(initial_bboxes)} lesion slices verified.\n\n"
    )
    medsam2_cot += (
        f"Bbox annotation complete. Initiating 3D segmentation seeded from key slice Z={key_z} "
        f"(largest cross-section, bbox {kbb})."
    )
    messages.append({"role": "assistant", "content": medsam2_cot})

    # Turn 9: tool_call — run_medsam2
    messages.append({"role": "tool_call", "content": json.dumps({
        "name": "run_medsam2",
        "arguments": {"key_z": key_z, "bbox": kbb}
    })})

    # Turn 10: tool_response — initial masks for all sampled slices
    # All N mask images are packed into a single response to maintain 1:1 tool pairing.
    mask_entries = []
    for z in sampled:
        d = dice_per_z.get(z, -1.0)
        if z in init_mask_render:
            mask_entries.append({"z_index": z, "mask_image": "<image>", "dice_with_gt": d})
            images.append(init_mask_render[z])
        else:
            mask_entries.append({"z_index": z, "mask_image": None, "dice_with_gt": d})

    messages.append({"role": "tool_response", "content": json.dumps({
        "status": "propagation_complete",
        "key_z": key_z,
        "slices": mask_entries
    })})

    # Turn 11: assistant — mask review CoT
    mask_review = generate_mask_review_cot(sampled, dice_per_z, needs_point)
    messages.append({"role": "assistant", "content": mask_review})

    # Turns 12+13: add_point corrections (if any)
    if needs_point:
        for z in sorted(needs_point.keys()):
            info = needs_point[z]
            messages.append({"role": "tool_call", "content": json.dumps({
                "name": "add_point",
                "arguments": {
                    "z_index": z,
                    "points": info['points'],
                    "labels": info['labels']
                }
            })})
        for z in sorted(needs_point.keys()):
            d = corrected_dice.get(z, 0.0)
            messages.append({"role": "tool_response", "content": json.dumps({
                "z_index": z,
                "mask_image": "<image>",
                "dice_with_gt": d
            })})
            images.append(corr_mask_render[z])

        # Turn 14: final CoT
        improved = [z for z in sorted(needs_point.keys())
                    if corrected_dice.get(z, 0) > dice_per_z.get(z, 0)]
        final_cot = (
            f"Point corrections applied to Z={sorted(needs_point.keys())}. "
            f"{len(improved)}/{len(needs_point)} slices improved. "
            f"3D segmentation complete. Finalising."
        )
    else:
        # No point corrections needed
        final_cot = (
            f"All {len(sampled)} sampled slices meet the Dice ≥ {DICE_PASS_THRESHOLD} threshold. "
            f"3D segmentation complete. Finalising."
        )

    messages.append({"role": "assistant", "content": final_cot})

    # Turn 15: finish_3d_segmentation
    messages.append({"role": "tool_call", "content": json.dumps({
        "name": "finish_3d_segmentation", "arguments": {}
    })})

    return {"tools": TOOLS_JSON, "messages": messages, "images": images}


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

def validate_entry(entry):
    msgs   = entry["messages"]
    images = entry["images"]
    valid_names = {"add_bbox", "run_medsam2", "add_point", "finish_3d_segmentation"}

    # Image token count
    n_img_tokens = sum(m["content"].count("<image>") for m in msgs if m["content"])
    if n_img_tokens != len(images):
        return False, f"<image> tokens ({n_img_tokens}) != images ({len(images)})"

    # Tool call JSON + names
    tool_calls = [m for m in msgs if m["role"] == "tool_call"]
    for tc in tool_calls:
        try:
            obj = json.loads(tc["content"])
        except Exception as e:
            return False, f"tool_call JSON error: {e}"
        name = obj.get("name")
        if name not in valid_names:
            return False, f"Unknown tool: {name}"
        if name == "add_bbox":
            bbox = obj.get("arguments", {}).get("bbox", [])
            if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                return False, f"Invalid bbox: {bbox}"
        if name == "add_point":
            args = obj.get("arguments", {})
            pts, lbls = args.get("points", []), args.get("labels", [])
            if len(pts) != len(lbls) or len(pts) == 0:
                return False, f"points/labels mismatch or empty"

    # Last tool_call must be finish_3d_segmentation
    if tool_calls:
        last = json.loads(tool_calls[-1]["content"]).get("name")
        if last != "finish_3d_segmentation":
            return False, f"Last tool_call must be finish_3d_segmentation, got {last}"

    # add_bbox + add_point calls must each have a paired tool_response
    # run_medsam2 has one paired response
    # finish_3d_segmentation has no response
    tc_with_response = sum(
        1 for m in msgs if m["role"] == "tool_call"
        and json.loads(m["content"]).get("name") != "finish_3d_segmentation"
    )
    n_responses = sum(1 for m in msgs if m["role"] == "tool_response")
    if tc_with_response != n_responses:
        return False, f"tool_calls expecting response ({tc_with_response}) != tool_responses ({n_responses})"

    return True, "ok"


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert M3D data to Phase-2 agent trajectory JSONL (bbox → MedSAM2 → point refine)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_root',   default='/BDSZ6/private/user/yxd/data/M3D/data_6-13/train')
    p.add_argument('--output_dir',  default='/BDSZ6/private/user/yxd/data/qwen/agent_phase2')
    p.add_argument('--ckpt',        default=None, help='MedSAM2 checkpoint; if omitted uses GT-derived masks')
    p.add_argument('--cfg',         default=None, help='MedSAM2 Hydra config YAML')
    p.add_argument('--device',      default='cuda:4')
    p.add_argument('--train_ratio', type=float, default=0.9)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--max_samples', type=int,   default=None, help='Cap for quick testing')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load MedSAM2 predictor (optional) ────────────────────────────────────
    import torch
    predictor = None
    device    = torch.device('cpu')
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
        print(f"[MedSAM2] Loaded checkpoint: {args.ckpt}  device={device}")
    else:
        print("[MedSAM2] No checkpoint provided — using GT-derived approximate masks.")

    # ── Load data ─────────────────────────────────────────────────────────────
    meta_path  = os.path.join(args.data_root, 'meta_expressions.json')
    mask_path  = os.path.join(args.data_root, 'mask_dict.pkl')
    jpeg_root  = os.path.join(args.data_root, 'JPEGImages')
    render_dir = os.path.join(args.output_dir, 'renders')

    with open(meta_path) as f:
        meta = json.load(f)['videos']
    with open(mask_path, 'rb') as f:
        mask_dict = pickle.load(f)

    annos = build_ordered_annos(meta, mask_dict)
    valid_annos = [
        a for a in annos
        if a['category'] in {'organ', 'lesion', 'cyst'}
        and len(a['non_none_z']) >= MIN_LESION_FRAMES
    ]
    print(f"Total expressions: {len(annos)} → valid: {len(valid_annos)}")

    if args.max_samples:
        valid_annos = valid_annos[:args.max_samples]
        print(f"Capped to {args.max_samples} samples.")

    # Train/val split by video
    all_vids   = sorted(meta.keys())
    split_idx  = int(len(all_vids) * args.train_ratio)
    train_vids = set(all_vids[:split_idx])
    val_vids   = set(all_vids[split_idx:])

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'agent_train.jsonl')
    val_path   = os.path.join(args.output_dir, 'agent_val.jsonl')

    stats = {'train': {'ok': 0, 'skip': 0, 'invalid': 0},
             'val':   {'ok': 0, 'skip': 0, 'invalid': 0}}

    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for anno in tqdm(valid_annos, desc='Building Phase-2 trajectories'):
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

    print("\n=== Phase-2 Conversion Summary ===")
    for sp in ['train', 'val']:
        s = stats[sp]
        print(f"\n{sp.capitalize()}: written={s['ok']}  skipped={s['skip']}  invalid={s['invalid']}")
    print(f"\nOutput:  {train_path}")
    print(f"         {val_path}")
    print(f"Renders: {render_dir}/")


if __name__ == '__main__':
    main()
