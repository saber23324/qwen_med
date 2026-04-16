#!/usr/bin/env python3
"""
Convert M3D data_4_ori to ms-swift agent trajectory JSONL format (Phase 1).

Each JSONL entry is a complete SFT trajectory for one 3D MRI volume + lesion expression:

  [user]              Task instruction + 10 Z-indexed MRI slices
  [assistant]         Global spatial analysis CoT
  [tool_call × N]     Parallel add_bbox calls (one per lesion-containing slice)
  [tool_response × N] Rendered bbox feedback (with IoU)
  [assistant]         Review CoT  ← merges with finish / correction below
  [tool_call × M]     Corrected add_bbox (only for ~10 % of samples with injected failure)
  [tool_response × M] Correction feedback
  [assistant]         Completion CoT
  [tool_call]         finish_bbox_annotation

Usage:
    python Qwen3_VL/convert_to_agent_trajectory.py \\
        --data_root /BDSZ6/private/user/yxd/data/M3D/data_4_ori/train \\
        --output_dir /BDSZ6/private/user/yxd/data/qwen/agent_phase1 \\
        [--n_slices 10] [--train_ratio 0.9] [--seed 42]

Outputs:
    {output_dir}/agent_train.jsonl
    {output_dir}/agent_val.jsonl
    {output_dir}/renders/           PNG files of bbox overlays

Expression selection:
    - lesion / tumour expressions only (skip organ + zero-frame annotations)
    - cyst expressions are also included when they have >= 3 masked frames
    - minimum 3 non-null Z-frames required for a meaningful 10-slice trajectory
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image, ImageDraw
from scipy import ndimage
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

N_SLICES = 10
JITTER_RATIO = 0.05          # normal prediction jitter
FAILURE_INJECT_RATIO = 0.10  # 10 % of samples get one bad bbox
FAILURE_OFFSET_RATIO = 0.30  # bbox offset that guarantees IoU < IOU_THRESHOLD
IOU_PASS_THRESHOLD = 0.75    # below this → trigger correction
MIN_LESION_FRAMES = 3        # skip expressions with fewer non-null frames

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
                    "z_index": {
                        "type": "integer",
                        "description": "Z-axis index of the target slice"
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
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
            "name": "finish_bbox_annotation",
            "description": (
                "Confirm all slice bbox annotations are complete (including review). "
                "Terminates the task."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

TOOLS_JSON = json.dumps(TOOLS)

SYSTEM_MSG = (
    "You are a medical imaging assistant specialising in 3D MRI analysis. "
    "Given multiple Z-indexed MRI slices, you identify lesions and annotate "
    "them with precise bounding boxes."
)

# Keyword sets used to classify expression text
_ORGAN_KW  = {'organ', 'kidney', 'liver', 'spleen', 'filtering', 'responsible',
               'pair of', 'essential for', 'producing urine', 'eliminating waste'}
_LESION_KW = {'tumor', 'tumour', 'cancer', 'mass', 'abnormal', 'malignant',
               'carcinoma', 'lesion', 'metastasis', 'cluster of cells',
               'uncontrolled cell', 'arising from', 'tissue formed'}
_CYST_KW   = {'cyst', 'fluid', 'pouch', 'cystic', 'encapsulated', 'non-malignant'}


# ────────────────────────────────────────────────────────────────────────────
# Expression classifier
# ────────────────────────────────────────────────────────────────────────────

def classify_expression(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _LESION_KW):
        return 'lesion'
    if any(k in t for k in _CYST_KW):
        return 'cyst'
    if any(k in t for k in _ORGAN_KW):
        return 'organ'
    return 'other'


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def mask_to_bbox(gt_2d):
    """Tight [x1,y1,x2,y2] bbox for a binary mask, or None if empty."""
    if gt_2d.sum() == 0:
        return None
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
    """Shift bbox to force IoU < IOU_PASS_THRESHOLD."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    ox, oy = int(bw * offset_ratio), int(bh * offset_ratio)
    nx1 = min(x1 + ox, img_w - bw - 1)
    ny1 = min(y1 + oy, img_h - bh - 1)
    return [nx1, ny1, nx1 + bw, ny1 + bh]


def compute_iou(bbox, gt_2d):
    """Compute IoU between predicted bbox and GT binary mask's bbox."""
    if gt_2d.sum() == 0:
        return 1.0
    x1, y1, x2, y2 = bbox
    gt_rows, gt_cols = np.where(gt_2d > 0)
    gx1, gy1 = int(gt_cols.min()), int(gt_rows.min())
    gx2, gy2 = int(gt_cols.max()), int(gt_rows.max())
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Annotation ordering (must match mask_dict key assignment)
# ────────────────────────────────────────────────────────────────────────────

def build_ordered_annos(meta, mask_dict):
    annos = []
    anno_idx = 0
    for vid in sorted(meta.keys()):
        vd = meta[vid]
        frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):
            exp = vd['expressions'][eid]
            masks = mask_dict[str(anno_idx)]
            non_none_z = [i for i, m in enumerate(masks) if m is not None]
            annos.append({
                'vid':          vid,
                'eid':          eid,
                'caption':      exp['exp'],
                'anno_id':      str(anno_idx),
                'frames':       frames,
                'img_w':        vd['width'],
                'img_h':        vd['height'],
                'masks':        masks,
                'non_none_z':   non_none_z,
                'category':     classify_expression(exp['exp']),
            })
            anno_idx += 1
    return annos


# ────────────────────────────────────────────────────────────────────────────
# Slice sampling
# ────────────────────────────────────────────────────────────────────────────

def sample_slices(non_none_z, n=N_SLICES):
    """Uniformly sample n Z indices within the lesion Z range."""
    z_min, z_max = non_none_z[0], non_none_z[-1]
    return np.linspace(z_min, z_max, n, dtype=int).tolist()


# ────────────────────────────────────────────────────────────────────────────
# CoT generators
# ────────────────────────────────────────────────────────────────────────────

def generate_spatial_cot(sampled_indices, masks, caption):
    """Spatial analysis CoT: lesion range, key slice, morphology trend."""
    areas = []
    for z in sampled_indices:
        m = masks[z]
        areas.append(int(maskUtils.decode(m).sum()) if m is not None else 0)

    has_lesion = [z for z, a in zip(sampled_indices, areas) if a > 0]
    no_lesion  = [z for z, a in zip(sampled_indices, areas) if a == 0]

    if not has_lesion:
        return None

    key_z = sampled_indices[int(np.argmax(areas))]
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
        f"Lesion present at Z={has_lesion[0]}~{has_lesion[-1]} "
        f"({len(has_lesion)} of {N_SLICES} sampled slices contain the lesion). "
        f"Z={key_z} has the largest cross-section (key slice). "
        f"Cross-sectional area trend: {trend}.\n"
    )
    if no_lesion:
        cot += f"Z={no_lesion} — no visible lesion on these slices, skipping annotation.\n"
    cot += "\nProceeding to annotate all lesion-containing slices:"
    return cot


def generate_review_cot(initial_bboxes, masks, needs_correction):
    lines = ["Review complete. Checking bbox quality:"]
    for z, pred_bbox in sorted(initial_bboxes.items()):
        m = masks[z]
        gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
        iou = compute_iou(pred_bbox, gt_2d)
        if z in needs_correction:
            lines.append(f"  Z={z}: FAIL (IoU={iou:.2f} < {IOU_PASS_THRESHOLD}) — re-annotating.")
        else:
            lines.append(f"  Z={z}: PASS (IoU={iou:.2f})")
    if needs_correction:
        fail_zs = sorted(needs_correction.keys())
        lines.append(f"\nCorrecting {len(fail_zs)} slice(s): Z={fail_zs}.")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Image rendering
# ────────────────────────────────────────────────────────────────────────────

def render_bbox_overlay(img_path, bbox, save_path):
    """Draw a red bbox rectangle on the image and save as PNG."""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    img.save(save_path)


# ────────────────────────────────────────────────────────────────────────────
# Trajectory builder
# ────────────────────────────────────────────────────────────────────────────

def build_trajectory(anno, jpeg_root, render_dir, inject_fail: bool):
    """
    Build one complete agent trajectory JSONL entry.

    Returns a dict with keys: tools, messages, images
    Returns None if the annotation is not usable.
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

    # ── 1. Sample 10 Z indices ──────────────────────────────────────────────
    sampled = sample_slices(non_none)

    # Paths for the 10 input slices
    def slice_path(z):
        return os.path.join(jpeg_root, vid, frames[z] + '.jpg')

    # ── 2. Spatial CoT ──────────────────────────────────────────────────────
    cot = generate_spatial_cot(sampled, masks, caption)
    if cot is None:
        return None

    # ── 3. Extract GT bboxes + jittered predictions ─────────────────────────
    gt_bboxes      = {}   # z → [x1,y1,x2,y2]
    initial_bboxes = {}   # z → predicted (jittered) bbox

    for z in sampled:
        m = masks[z]
        if m is None:
            continue
        gt_2d = maskUtils.decode(m)
        gt_bb = mask_to_bbox(gt_2d)
        if gt_bb is None:
            continue
        gt_bboxes[z] = gt_bb
        initial_bboxes[z] = add_jitter(gt_bb, img_w, img_h)

    if not initial_bboxes:
        return None

    # ── 4. Optionally inject one failure ────────────────────────────────────
    fail_z = None
    if inject_fail:
        fail_z = random.choice(list(initial_bboxes.keys()))
        initial_bboxes[fail_z] = inject_failure(
            gt_bboxes[fail_z], img_w, img_h
        )

    # ── 5. Identify slices needing correction ───────────────────────────────
    needs_correction = {}   # z → corrected bbox
    for z, pred_bb in initial_bboxes.items():
        m = masks[z]
        if m is None:
            continue
        gt_2d = maskUtils.decode(m)
        iou = compute_iou(pred_bb, gt_2d)
        if iou < IOU_PASS_THRESHOLD:
            needs_correction[z] = add_jitter(gt_bboxes[z], img_w, img_h,
                                             jitter_ratio=0.02)

    # ── 6. Render bbox overlay images ───────────────────────────────────────
    os.makedirs(render_dir, exist_ok=True)

    init_render = {}    # z → render path
    for z, bb in initial_bboxes.items():
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_init.png")
        if not os.path.exists(rp):
            render_bbox_overlay(slice_path(z), bb, rp)
        init_render[z] = rp

    corr_render = {}    # z → render path
    for z, bb in needs_correction.items():
        rp = os.path.join(render_dir, f"{vid}_z{z:03d}_corr.png")
        if not os.path.exists(rp):
            render_bbox_overlay(slice_path(z), bb, rp)
        corr_render[z] = rp

    # ── 7. Assemble messages and images list ────────────────────────────────
    messages = []
    images   = []   # ordered list of all image paths; <image> tokens consume in order

    # ── Turn 1: user — task description + 10 Z-indexed slices ───────────────
    user_lines = [
        f"Task: Examine the following {N_SLICES} MRI slices sampled uniformly from a 3D volume. "
        f"The target structure is: \"{caption}\". "
        f"For each slice that contains a lesion, call add_bbox with the precise 2-D bounding box "
        f"[x1, y1, x2, y2] in pixel coordinates. "
        f"After all slices are annotated and reviewed, call finish_bbox_annotation to complete.\n"
    ]
    for z in sampled:
        user_lines.append(f"Z={z}: <image>")
        images.append(slice_path(z))

    messages.append({"role": "user", "content": "\n".join(user_lines)})

    # ── Turn 2+3: assistant — spatial CoT then parallel add_bbox calls ───────
    # In ms-swift: consecutive assistant + tool_call entries form one turn.
    messages.append({"role": "assistant", "content": cot})

    for z in sampled:
        if z not in initial_bboxes:
            continue
        messages.append({
            "role": "tool_call",
            "content": json.dumps({
                "name": "add_bbox",
                "arguments": {"z_index": z, "bbox": initial_bboxes[z]}
            })
        })

    # ── Turn 4: tool_responses — rendered initial bbox feedback ─────────────
    for z in sampled:
        if z not in initial_bboxes:
            continue
        m = masks[z]
        gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
        iou = compute_iou(initial_bboxes[z], gt_2d)
        messages.append({
            "role": "tool_response",
            "content": json.dumps({
                "z_index": z,
                "bbox_image": "<image>",
                "iou_with_gt": round(iou, 3)
            })
        })
        images.append(init_render[z])

    # ── Turns 5+6+7+8: review CoT → corrections (if any) → finish ───────────
    review_cot = generate_review_cot(initial_bboxes, masks, needs_correction)

    if needs_correction:
        # ── Turn 5+6: review CoT + parallel correction tool_calls ────────────
        messages.append({"role": "assistant", "content": review_cot})

        for z in sorted(needs_correction.keys()):
            messages.append({
                "role": "tool_call",
                "content": json.dumps({
                    "name": "add_bbox",
                    "arguments": {"z_index": z, "bbox": needs_correction[z]}
                })
            })

        # ── Turn 7: correction tool_responses ────────────────────────────────
        for z in sorted(needs_correction.keys()):
            m = masks[z]
            gt_2d = maskUtils.decode(m) if m is not None else np.zeros((512, 512), dtype=np.uint8)
            iou = compute_iou(needs_correction[z], gt_2d)
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "z_index": z,
                    "bbox_image": "<image>",
                    "iou_with_gt": round(iou, 3)
                })
            })
            images.append(corr_render[z])

        # ── Turn 8: completion CoT + finish_bbox_annotation ──────────────────
        completion_cot = (
            f"Corrections applied to Z={sorted(needs_correction.keys())}. "
            f"All {len(initial_bboxes)} lesion slices now annotated and verified. "
            f"Finalising annotation."
        )
        messages.append({"role": "assistant", "content": completion_cot})
        messages.append({
            "role": "tool_call",
            "content": json.dumps({"name": "finish_bbox_annotation", "arguments": {}})
        })

    else:
        # No corrections needed — review CoT leads directly to finish
        finish_cot = (
            review_cot
            + f"\n\nAll {len(initial_bboxes)} lesion slices annotated and verified. "
            f"Finalising annotation."
        )
        messages.append({"role": "assistant", "content": finish_cot})
        messages.append({
            "role": "tool_call",
            "content": json.dumps({"name": "finish_bbox_annotation", "arguments": {}})
        })

    return {
        "tools":    TOOLS_JSON,
        "messages": messages,
        "images":   images,
    }


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

def validate_entry(entry):
    """Basic sanity checks. Returns (ok: bool, reason: str)."""
    msgs   = entry["messages"]
    images = entry["images"]

    # Count <image> tokens across all message contents
    n_image_tokens = sum(
        msg["content"].count("<image>") for msg in msgs
    )
    if n_image_tokens != len(images):
        return False, (
            f"<image> token count ({n_image_tokens}) != "
            f"images list length ({len(images)})"
        )

    # Verify tool_call JSON and names
    valid_names = {"add_bbox", "finish_bbox_annotation"}
    tool_calls = [m for m in msgs if m["role"] == "tool_call"]
    for tc in tool_calls:
        try:
            obj = json.loads(tc["content"])
        except json.JSONDecodeError as e:
            return False, f"tool_call JSON parse error: {e}"
        if obj.get("name") not in valid_names:
            return False, f"Unknown tool name: {obj.get('name')}"
        if obj["name"] == "add_bbox":
            args = obj.get("arguments", {})
            bbox = args.get("bbox", [])
            if len(bbox) != 4:
                return False, f"bbox must have 4 elements, got {len(bbox)}"
            x1, y1, x2, y2 = bbox
            if not (x1 < x2 and y1 < y2):
                return False, f"Invalid bbox coords: {bbox}"

    # Every tool_call must have a following tool_response (in order)
    roles = [m["role"] for m in msgs]
    tc_count = roles.count("tool_call")
    tr_count = roles.count("tool_response")
    if tc_count != tr_count + 1:   # +1 because finish_bbox_annotation has no response
        # actually finish_bbox_annotation has no tool_response — count add_bbox only
        add_bbox_calls = sum(
            1 for m in msgs
            if m["role"] == "tool_call"
            and json.loads(m["content"]).get("name") == "add_bbox"
        )
        if add_bbox_calls != tr_count:
            return False, (
                f"add_bbox calls ({add_bbox_calls}) != "
                f"tool_response entries ({tr_count})"
            )

    # finish_bbox_annotation must be last tool_call
    last_tc = [m for m in msgs if m["role"] == "tool_call"][-1]
    if json.loads(last_tc["content"]).get("name") != "finish_bbox_annotation":
        return False, "Last tool_call must be finish_bbox_annotation"

    return True, "ok"


# ────────────────────────────────────────────────────────────────────────────
# Main conversion loop
# ────────────────────────────────────────────────────────────────────────────

def convert(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    meta_path = os.path.join(args.data_root, 'meta_expressions.json')
    mask_path = os.path.join(args.data_root, 'mask_dict.pkl')
    jpeg_root = os.path.join(args.data_root, 'JPEGImages')
    render_dir = os.path.join(args.output_dir, 'renders')

    print(f"Loading {meta_path} ...")
    with open(meta_path) as f:
        meta = json.load(f)['videos']

    print(f"Loading {mask_path} ...")
    with open(mask_path, 'rb') as f:
        mask_dict = pickle.load(f)

    print(f"Building annotation list ...")
    annos = build_ordered_annos(meta, mask_dict)
    assert len(annos) == len(mask_dict), (
        f"Annotation/mask mismatch: {len(annos)} vs {len(mask_dict)}"
    )

    # Filter to lesion + cyst expressions with enough frames
    target_cats = {'lesion', 'cyst'}
    valid_annos = [
        a for a in annos
        if a['category'] in target_cats
        and len(a['non_none_z']) >= MIN_LESION_FRAMES
    ]
    print(f"Total expressions: {len(annos)}")
    print(f"Lesion/cyst with >={MIN_LESION_FRAMES} frames: {len(valid_annos)}")

    # Train/val split by video (deterministic)
    all_vids  = sorted(meta.keys())
    split_idx = int(len(all_vids) * args.train_ratio)
    train_vids = set(all_vids[:split_idx])
    val_vids   = set(all_vids[split_idx:])
    print(f"Videos: {len(all_vids)} (train={len(train_vids)}, val={len(val_vids)})")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'agent_train.jsonl')
    val_path   = os.path.join(args.output_dir, 'agent_val.jsonl')

    stats = {
        'train': {'ok': 0, 'skip': 0, 'fail_inject': 0, 'invalid': 0},
        'val':   {'ok': 0, 'skip': 0, 'fail_inject': 0, 'invalid': 0},
    }

    # Decide upfront which samples get an injected failure
    n_fail = max(1, int(len(valid_annos) * FAILURE_INJECT_RATIO))
    fail_indices = set(random.sample(range(len(valid_annos)), n_fail))

    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for idx, anno in enumerate(tqdm(valid_annos, desc='Building trajectories')):
            split = 'train' if anno['vid'] in train_vids else 'val'
            f_out = f_train if split == 'train' else f_val

            inject_fail = (idx in fail_indices)

            try:
                entry = build_trajectory(
                    anno, jpeg_root, render_dir, inject_fail=inject_fail
                )
            except Exception as e:
                stats[split]['skip'] += 1
                tqdm.write(f"  SKIP {anno['vid']} exp{anno['eid']}: {e}")
                continue

            if entry is None:
                stats[split]['skip'] += 1
                continue

            ok, reason = validate_entry(entry)
            if not ok:
                stats[split]['invalid'] += 1
                tqdm.write(f"  INVALID {anno['vid']} exp{anno['eid']}: {reason}")
                continue

            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            stats[split]['ok'] += 1
            if inject_fail:
                stats[split]['fail_inject'] += 1

    # Summary
    print("\n=== Conversion Summary ===")
    total_ok = 0
    for sp in ['train', 'val']:
        s = stats[sp]
        total = s['ok'] + s['skip'] + s['invalid']
        print(f"\n{sp.capitalize()} set:")
        print(f"  Written      : {s['ok']}")
        print(f"  With failure : {s['fail_inject']}")
        print(f"  Skipped      : {s['skip']}")
        print(f"  Invalid      : {s['invalid']}")
        total_ok += s['ok']
    print(f"\nTotal written: {total_ok}")
    print(f"Output: {train_path}")
    print(f"        {val_path}")
    print(f"Renders: {render_dir}/")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert M3D data_4_ori to Phase-1 agent trajectory JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--data_root',
        default='/BDSZ6/private/user/yxd/data/M3D/data_4_ori/train',
        help='Root containing JPEGImages/, meta_expressions.json, mask_dict.pkl',
    )
    p.add_argument(
        '--output_dir',
        default='/BDSZ6/private/user/yxd/data/qwen/agent_phase1',
        help='Directory to write agent_train.jsonl / agent_val.jsonl / renders/',
    )
    p.add_argument('--n_slices',    type=int,   default=10,  help='Number of Z slices to sample')
    p.add_argument('--train_ratio', type=float, default=0.9, help='Train/val split ratio by video')
    p.add_argument('--seed',        type=int,   default=42,  help='Random seed')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(args)
