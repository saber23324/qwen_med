#!/usr/bin/env python3
"""
Phase 2 inference and evaluation script.

Runs the trained Qwen3-VL agent on the validation set through the full
multi-turn agent loop (bbox → MedSAM2 3D segmentation → point refinement),
then evaluates predicted masks against GT annotations.

Metrics reported per-case and overall:
  Dice, HD95, Precision, Recall

Mask overlay images are saved to --output_dir.

Usage:
    conda run -n qwen3 python Qwen3_VL/infer_phase2.py \\
        --model  /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \\
        --ckpt   /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase2/v0-xxx/checkpoint-500 \\
        --val_jsonl /BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl \\
        --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \\
        --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \\
        --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \\
        --output_dir /tmp/phase2_eval \\
        --device cuda:4
"""

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

# ── ms-swift engine (same as infer.py) ───────────────────────────────────────
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

# ── MedSAM2 ──────────────────────────────────────────────────────────────────
MEDSAM2_ROOT = Path(__file__).resolve().parents[1] / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
N_SLICES       = 10
RENDER_SIZE    = 256
IMG_MEAN       = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMG_STD        = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
MAX_AGENT_TURNS = 12        # safety limit for the agent loop
DICE_PASS      = 0.70       # threshold: below triggers add_point

BBOX_COLOR     = (255, 80, 80)
POINT_FG_COLOR = (255, 50, 50)
POINT_BG_COLOR = (50, 100, 255)

TOOL_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)

# ── Phase 2 tool schema (must match training) ─────────────────────────────────
TOOLS = [
    {"type": "function", "function": {"name": "add_bbox", "description":
        "Annotate the 2D bounding box of a lesion on the specified Z slice.",
        "parameters": {"type": "object", "properties": {
            "z_index": {"type": "integer"},
            "bbox":    {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
        }, "required": ["z_index", "bbox"]}}},
    {"type": "function", "function": {"name": "run_medsam2", "description":
        "Trigger MedSAM2 3D segmentation seeded from a single key slice.",
        "parameters": {"type": "object", "properties": {
            "key_z": {"type": "integer"},
            "bbox":  {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
        }, "required": ["key_z", "bbox"]}}},
    {"type": "function", "function": {"name": "add_point", "description":
        "Refine the mask on a specific Z slice using point prompts.",
        "parameters": {"type": "object", "properties": {
            "z_index": {"type": "integer"},
            "points":  {"type": "array"},
            "labels":  {"type": "array"}
        }, "required": ["z_index", "points", "labels"]}}},
    {"type": "function", "function": {"name": "finish_3d_segmentation",
        "description": "Confirm 3D segmentation is complete.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]
TOOLS_JSON = json.dumps(TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if (pred.sum() == 0 and gt.sum() == 0) else 0.0
    return float(2 * inter / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hausdorff distance at 95th percentile (2D or 3D)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float('inf')
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)
    dt_gt   = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)
    d1 = dt_gt[pred_surf]
    d2 = dt_pred[gt_surf]
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))


# ─────────────────────────────────────────────────────────────────────────────
# Image rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_bbox_overlay(img_path: str, bbox: list, save_path: str):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=BBOX_COLOR, width=2)
    img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS).save(save_path)


def render_mask_overlay(img_path: str, mask: np.ndarray, save_path: str,
                        points=None, labels=None):
    img  = Image.open(img_path).convert('RGBA')
    H, W = mask.shape
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    px = np.zeros((H, W, 4), dtype=np.uint8)
    px[mask.astype(bool)] = [0, 200, 100, 120]
    overlay.paste(Image.fromarray(px, 'RGBA'))
    img = Image.alpha_composite(img, overlay).convert('RGB')
    if points and labels:
        draw = ImageDraw.Draw(img)
        for (x, y), lbl in zip(points, labels):
            color = POINT_FG_COLOR if lbl == 1 else POINT_BG_COLOR
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline='white')
    img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS).save(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_volume(jpeg_dir: str, device) -> tuple:
    """Load all JPEG slices → normalised (D,3,512,512) tensor + (H, W)."""
    files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith('.jpg'))
    slices = [np.array(Image.open(os.path.join(jpeg_dir, f)).convert('L')) for f in files]
    arr = np.stack(slices, axis=0)  # (D, H, W) uint8
    orig_H, orig_W = arr.shape[1], arr.shape[2]
    arr_f = arr.astype(np.float32) / 255.0
    # Resize to 512×512 if needed
    if orig_H != 512 or orig_W != 512:
        resized = []
        for s in arr_f:
            img = Image.fromarray((s * 255).astype(np.uint8))
            resized.append(np.array(img.resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255.0)
        arr_f = np.stack(resized, axis=0)
    # (D,H,W) → (D,3,H,W) and normalise
    t = torch.from_numpy(arr_f).unsqueeze(1).expand(-1, 3, -1, -1)  # (D,3,512,512)
    t = (t - IMG_MEAN[None, :, None, None]) / IMG_STD[None, :, None, None]
    return t.to(device), orig_H, orig_W


def run_medsam2(predictor, img_tensor, orig_H, orig_W, key_z: int,
                bbox_512: np.ndarray, device) -> dict:
    """Bidirectional MedSAM2 propagation. Returns {frame_idx: bool mask (512,512)}."""
    result = {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        # Key slice mask from bbox
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=key_z, obj_id=1, box=bbox_512)
        key_mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        result[key_z] = key_mask.astype(bool)
        predictor.reset_state(state)
        # Forward
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=False):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)
        # Backward
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=True):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)
    return result


def run_medsam2_point(predictor, img_tensor, orig_H, orig_W, frame_idx: int,
                      points_512: np.ndarray, labels: np.ndarray, device) -> np.ndarray:
    """Single-slice MedSAM2 with point prompts. Returns bool mask (512,512)."""
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=1,
            points=points_512, labels=labels)
        mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(bool)
        predictor.reset_state(state)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_tool_calls(text: str) -> list:
    """Extract all <tool_call>...</tool_call> blocks from model output."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append({'name': obj.get('name', ''), 'arguments': obj.get('arguments', {})})
        except Exception:
            pass
    return calls


def split_cot_and_calls(text: str):
    """
    Split model output into (cot_text, list_of_raw_call_strings).
    cot_text is the text before the first <tool_call> block.
    """
    first = text.find('<tool_call>')
    cot = text[:first].strip() if first >= 0 else text.strip()
    raw_calls = TOOL_CALL_RE.findall(text)
    return cot, raw_calls


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

class AgentExecutor:
    """Executes Phase 2 tool calls and manages conversation state."""

    def __init__(self, predictor, device, jpeg_dir: str, img_tensor,
                 orig_H: int, orig_W: int, frames: list,
                 masks: list, sampled: list, render_dir: str, vid: str):
        self.predictor  = predictor
        self.device     = device
        self.jpeg_dir   = jpeg_dir
        self.img_tensor = img_tensor
        self.orig_H     = orig_H
        self.orig_W     = orig_W
        self.frames     = frames      # frame filename (no ext) at each Z
        self.masks      = masks       # RLE masks[z]
        self.sampled    = sampled     # the 10 sampled Z indices
        self.render_dir = render_dir
        self.vid        = vid

        # Scale factors: image space → 512
        self.sx = 512 / orig_W
        self.sy = 512 / orig_H

        # State accumulated during the loop
        self.propagated_masks: dict = {}  # {z: bool mask (512,512)}
        self.final_masks: dict      = {}  # after point refinement
        self.messages: list         = []
        self.images: list           = []

    def slice_path(self, z: int) -> str:
        return os.path.join(self.jpeg_dir, self.frames[z] + '.jpg')

    # ── individual tool handlers ──────────────────────────────────────────────

    def exec_add_bbox(self, z_index: int, bbox: list) -> str:
        """Render bbox overlay. Returns tool_response JSON string."""
        rp = os.path.join(self.render_dir, f"{self.vid}_z{z_index:03d}_bbox.png")
        render_bbox_overlay(self.slice_path(z_index), bbox, rp)
        self.images.append(rp)

        # IoU vs GT for feedback (mirrors training)
        iou = 0.0
        m = self.masks[z_index] if z_index < len(self.masks) else None
        if m is not None:
            gt_2d = maskUtils.decode(m)
            x1, y1, x2, y2 = bbox
            gx1, gy1 = int(np.where(gt_2d > 0)[1].min()), int(np.where(gt_2d > 0)[0].min())
            gx2, gy2 = int(np.where(gt_2d > 0)[1].max()), int(np.where(gt_2d > 0)[0].max())
            ix1, iy1 = max(x1, gx1), max(y1, gy1)
            ix2, iy2 = min(x2, gx2), min(y2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (x2 - x1)*(y2 - y1) + (gx2 - gx1)*(gy2 - gy1) - inter
            iou   = round(inter / union, 3) if union > 0 else 0.0

        return json.dumps({"z_index": z_index, "bbox_image": "<image>", "iou_with_gt": iou})

    def exec_run_medsam2(self, key_z: int, bbox: list) -> str:
        """Run MedSAM2 propagation. Returns tool_response JSON string."""
        if self.predictor is None:
            # Fallback: use GT masks as approximate propagation result
            for z in self.sampled:
                m = self.masks[z] if z < len(self.masks) else None
                self.propagated_masks[z] = maskUtils.decode(m).astype(bool) if m is not None \
                                           else np.zeros((512, 512), dtype=bool)
        else:
            kbb = bbox
            bbox_512 = np.array([kbb[0]*self.sx, kbb[1]*self.sy,
                                  kbb[2]*self.sx, kbb[3]*self.sy], dtype=np.float32)
            self.propagated_masks = run_medsam2(
                self.predictor, self.img_tensor, self.orig_H, self.orig_W,
                key_z, bbox_512, self.device
            )
        self.final_masks = dict(self.propagated_masks)

        # Render masks for each sampled slice
        slices_info = []
        for z in self.sampled:
            pred = self.propagated_masks.get(z)
            if pred is None:
                pred = np.zeros((512, 512), dtype=bool)
            rp = os.path.join(self.render_dir, f"{self.vid}_z{z:03d}_mask_init.png")
            render_mask_overlay(self.slice_path(z), pred, rp)
            self.images.append(rp)

            dice = 0.0
            m = self.masks[z] if z < len(self.masks) else None
            if m is not None:
                gt_2d = maskUtils.decode(m).astype(bool)
                dice  = round(dice_score(pred, gt_2d), 3)

            slices_info.append({"z_index": z, "mask_image": "<image>", "dice_with_gt": dice})

        return json.dumps({
            "status": "propagation_complete",
            "key_z": key_z,
            "slices": slices_info,
        })

    def exec_add_point(self, z_index: int, points: list, labels: list) -> str:
        """Refine mask with point prompts. Returns tool_response JSON string."""
        if self.predictor is not None:
            pts_512 = np.array([[p[0] * self.sx, p[1] * self.sy] for p in points],
                               dtype=np.float32)
            lbls = np.array(labels, dtype=np.int32)
            refined = run_medsam2_point(
                self.predictor, self.img_tensor, self.orig_H, self.orig_W,
                z_index, pts_512, lbls, self.device
            )
        else:
            # Fallback GT mask
            m = self.masks[z_index] if z_index < len(self.masks) else None
            refined = maskUtils.decode(m).astype(bool) if m is not None \
                      else np.zeros((512, 512), dtype=bool)

        self.final_masks[z_index] = refined

        rp = os.path.join(self.render_dir, f"{self.vid}_z{z_index:03d}_mask_refined.png")
        render_mask_overlay(self.slice_path(z_index), refined, rp,
                            points=points, labels=labels)
        self.images.append(rp)

        dice = 0.0
        m = self.masks[z_index] if z_index < len(self.masks) else None
        if m is not None:
            dice = round(dice_score(refined, maskUtils.decode(m).astype(bool)), 3)

        return json.dumps({"z_index": z_index, "mask_image": "<image>", "dice_with_gt": dice})

    # ── main agent loop ───────────────────────────────────────────────────────

    def run_loop(self, engine: TransformersEngine, request_cfg: RequestConfig,
                 user_content: str, slice_images: list):
        """
        Execute the full agent conversation loop.
        Returns the accumulated final_masks dict {z: bool mask}.
        """
        self.messages = [{"role": "user", "content": user_content}]
        self.images   = list(slice_images)

        for turn in range(MAX_AGENT_TURNS):
            req  = InferRequest(
                messages=self.messages,
                tools=TOOLS,
                images=self.images,
            )
            resp = engine.infer([req], request_config=request_cfg)[0]
            asst_text = resp.choices[0].message.content or ''

            tool_calls = parse_tool_calls(asst_text)
            cot, raw_calls = split_cot_and_calls(asst_text)

            # Add CoT text to history (if non-empty)
            if cot:
                self.messages.append({"role": "assistant", "content": cot})

            # Check for finish before executing anything
            if any(tc['name'] == 'finish_3d_segmentation' for tc in tool_calls):
                # Add the finish tool_call, loop ends
                for raw in raw_calls:
                    self.messages.append({"role": "tool_call", "content": raw.strip()})
                break

            if not tool_calls:
                # Model chose to stop on its own
                break

            # Execute each tool call and build paired responses
            tool_responses = []
            for tc, raw in zip(tool_calls, raw_calls):
                self.messages.append({"role": "tool_call", "content": raw.strip()})
                name = tc['name']
                args = tc['arguments']

                if name == 'add_bbox':
                    resp_json = self.exec_add_bbox(args['z_index'], args['bbox'])
                elif name == 'run_medsam2':
                    resp_json = self.exec_run_medsam2(args['key_z'], args['bbox'])
                elif name == 'add_point':
                    resp_json = self.exec_add_point(args['z_index'], args['points'], args['labels'])
                else:
                    resp_json = json.dumps({"error": f"unknown tool: {name}"})

                tool_responses.append(resp_json)

            # Add paired tool_response messages
            for resp_json in tool_responses:
                self.messages.append({"role": "tool_response", "content": resp_json})

        return self.final_masks


# ─────────────────────────────────────────────────────────────────────────────
# Per-case evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(final_masks: dict, masks: list, non_none_z: list) -> dict:
    """
    Compare predicted masks against GT for every annotated frame.
    Returns dict with per-frame metrics and volume-level aggregates.
    """
    per_frame = []
    vol_pred  = []
    vol_gt    = []

    for z in non_none_z:
        m = masks[z]
        if m is None:
            continue
        gt_2d   = maskUtils.decode(m).astype(bool)
        pred_2d = final_masks.get(z)
        if pred_2d is None:
            pred_2d = np.zeros_like(gt_2d)

        d  = dice_score(pred_2d, gt_2d)
        h  = hd95(pred_2d, gt_2d)
        p, r = precision_recall(pred_2d, gt_2d)
        per_frame.append({'z': z, 'dice': round(d, 4), 'hd95': round(h, 4),
                          'precision': round(p, 4), 'recall': round(r, 4)})
        vol_pred.append(pred_2d)
        vol_gt.append(gt_2d)

    if not vol_pred:
        return {'per_frame': [], 'dice': 0.0, 'hd95': float('inf'),
                'precision': 0.0, 'recall': 0.0}

    # Volume-level metrics (stack into 3D, compute once)
    vol_pred_3d = np.stack(vol_pred, axis=0)
    vol_gt_3d   = np.stack(vol_gt,   axis=0)
    vd = dice_score(vol_pred_3d, vol_gt_3d)
    vh = hd95(vol_pred_3d, vol_gt_3d)
    vp, vr = precision_recall(vol_pred_3d, vol_gt_3d)

    return {
        'per_frame': per_frame,
        'dice':      round(vd, 4),
        'hd95':      round(vh, 4),
        'precision': round(vp, 4),
        'recall':    round(vr, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_val_samples(val_jsonl: str, max_samples=None) -> list:
    samples = []
    with open(val_jsonl) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def extract_metadata(rec: dict):
    """
    From a val JSONL record extract: sampled Z indices, first user message,
    and the list of input slice image paths.
    """
    user_content = rec['messages'][0]['content']
    # Parse Z indices from "Z=15: <image>" lines
    sampled = [int(m) for m in re.findall(r'Z=(\d+):', user_content)]
    # First N_SLICES images are input slices; rest are renders
    n_input = user_content.count('<image>')
    slice_images = rec['images'][:n_input]
    return user_content, sampled, slice_images


def load_mask_dict_and_meta(data_root: str):
    with open(os.path.join(data_root, 'mask_dict.pkl'), 'rb') as f:
        mask_dict = pickle.load(f)
    with open(os.path.join(data_root, 'meta_expressions.json')) as f:
        meta = json.load(f)['videos']
    return mask_dict, meta


def build_anno_index(meta: dict, mask_dict: dict) -> list:
    """Build flat annotation list matching convert_to_agent_trajectory_phase2 ordering."""
    annos, idx = [], 0
    for vid in sorted(meta.keys()):
        vd = meta[vid]
        frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):
            masks      = mask_dict[str(idx)]
            non_none_z = [i for i, m in enumerate(masks) if m is not None]
            annos.append({
                'vid': vid, 'eid': eid,
                'caption': vd['expressions'][eid]['exp'],
                'anno_id': str(idx),
                'frames':  frames,
                'img_w':   vd['width'],
                'img_h':   vd['height'],
                'masks':   masks,
                'non_none_z': non_none_z,
            })
            idx += 1
    return annos


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',        default='/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--ckpt',         required=True, help='Adapter checkpoint dir')
    p.add_argument('--val_jsonl',    default='/BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl')
    p.add_argument('--data_root',    default='/BDSZ6/private/user/yxd/data/M3D/data_6-13/train')
    p.add_argument('--medsam2_ckpt', default=None, help='MedSAM2 checkpoint; if omitted uses GT masks')
    p.add_argument('--medsam2_cfg',  default=None, help='MedSAM2 Hydra config YAML')
    p.add_argument('--output_dir',   default='/tmp/phase2_eval')
    p.add_argument('--device',       default='cuda:4')
    p.add_argument('--max_samples',  type=int, default=None)
    p.add_argument('--max_tokens',   type=int, default=2048)
    p.add_argument('--temperature',  type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    render_dir = os.path.join(args.output_dir, 'renders')
    os.makedirs(render_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load LLM engine ───────────────────────────────────────────────────────
    print(f"[LLM] Loading {args.model} + adapter {args.ckpt}")
    engine = TransformersEngine(args.model, adapters=[args.ckpt])
    request_cfg = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)

    # ── Load MedSAM2 ─────────────────────────────────────────────────────────
    predictor = None
    if args.medsam2_ckpt and args.medsam2_cfg:
        from sam2.build_sam import build_sam2_video_predictor_npz
        cfg_path = args.medsam2_cfg
        if os.path.isabs(cfg_path) and not cfg_path.startswith('//'):
            cfg_path = '//' + cfg_path
        torch.set_float32_matmul_precision('high')
        predictor = build_sam2_video_predictor_npz(
            cfg_path, args.medsam2_ckpt, device=device, apply_postprocessing=False
        )
        print(f"[MedSAM2] Loaded {args.medsam2_ckpt}")
    else:
        print("[MedSAM2] No checkpoint provided — GT-derived masks used for tool execution.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[Data] Loading validation JSONL …")
    val_samples = load_val_samples(args.val_jsonl, max_samples=args.max_samples)
    mask_dict, meta = load_mask_dict_and_meta(args.data_root)
    all_annos = build_anno_index(meta, mask_dict)

    # Build (vid, caption) → anno lookup for robust matching.
    # Index-based matching fails because build_trajectory skips some annotations
    # (< MIN_LESION_FRAMES), so the JSONL may have fewer entries than the val split.
    anno_lookup = {(a['vid'], a['caption']): a for a in all_annos}

    jpeg_root = os.path.join(args.data_root, 'JPEGImages')

    _caption_re = re.compile(r'The target structure is:\s*"([^"]+)"')

    def match_anno(rec: dict):
        """Recover vid and caption from JSONL record, then look up annotation."""
        # vid: parent directory of the first input slice image
        vid = Path(rec['images'][0]).parent.name
        # caption: extracted from user message text
        user_text = rec['messages'][0]['content']
        m = _caption_re.search(user_text)
        caption = m.group(1) if m else ''
        anno = anno_lookup.get((vid, caption))
        if anno is None:
            raise KeyError(f"No annotation found for vid={vid!r} caption={caption!r}")
        return anno

    if args.max_samples:
        val_samples = val_samples[:args.max_samples]

    # ── Main evaluation loop ──────────────────────────────────────────────────
    all_results = []

    for rec in tqdm(val_samples, desc='Evaluating'):
        anno = match_anno(rec)
        vid         = anno['vid']
        frames      = anno['frames']
        masks       = anno['masks']
        non_none_z  = anno['non_none_z']
        jpeg_dir    = os.path.join(jpeg_root, vid)
        img_w, img_h = anno['img_w'], anno['img_h']

        user_content, sampled, slice_images = extract_metadata(rec)

        # Load and preprocess volume for MedSAM2
        img_tensor = orig_H = orig_W = None
        if predictor is not None:
            img_tensor, orig_H, orig_W = load_volume(jpeg_dir, device)
        else:
            orig_H, orig_W = img_h, img_w

        executor = AgentExecutor(
            predictor=predictor, device=device,
            jpeg_dir=jpeg_dir, img_tensor=img_tensor,
            orig_H=orig_H, orig_W=orig_W,
            frames=frames, masks=masks,
            sampled=sampled, render_dir=render_dir, vid=vid,
        )

        try:
            final_masks = executor.run_loop(
                engine, request_cfg, user_content, slice_images
            )
        except Exception as e:
            print(f"\n[WARN] Agent loop failed for {vid}: {e}")
            final_masks = {}

        metrics = evaluate_case(final_masks, masks, non_none_z)
        metrics['vid']     = vid
        metrics['caption'] = anno['caption']
        all_results.append(metrics)

        tqdm.write(
            f"  {vid}  Dice={metrics['dice']:.3f}  "
            f"HD95={metrics['hd95']:.1f}  "
            f"Prec={metrics['precision']:.3f}  "
            f"Recall={metrics['recall']:.3f}"
        )

    # ── Save per-case results ─────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'results.jsonl')
    with open(results_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    finite_hd95 = [r['hd95'] for r in all_results if r['hd95'] != float('inf')]

    print("\n" + "=" * 50)
    print(f"{'Metric':<20} {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("=" * 50)
    for key in ('dice', 'precision', 'recall'):
        vals = [r[key] for r in all_results]
        print(f"{key:<20} {np.mean(vals):>10.4f}  {np.std(vals):>10.4f}"
              f"  {np.min(vals):>10.4f}  {np.max(vals):>10.4f}")
    # HD95 separately (may have inf values)
    if finite_hd95:
        print(f"{'hd95 (finite)':<20} {np.mean(finite_hd95):>10.2f}  "
              f"{np.std(finite_hd95):>10.2f}  "
              f"{np.min(finite_hd95):>10.2f}  "
              f"{np.max(finite_hd95):>10.2f}")
    inf_count = len(all_results) - len(finite_hd95)
    if inf_count:
        print(f"  ({inf_count} cases with HD95=inf — empty pred or GT)")
    print("=" * 50)
    print(f"Total cases: {len(all_results)}")
    print(f"Results saved to: {results_path}")
    print(f"Mask renders in:  {render_dir}/")


if __name__ == '__main__':
    main()
