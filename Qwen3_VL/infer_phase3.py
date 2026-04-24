#!/usr/bin/env python3
"""
Phase 3 inference and evaluation script.

Runs the trained Qwen3-VL agent on the Phase-3 validation set through a
navigation-aware agent loop. Supported tools:

    get_slice(z_index)                 # absolute jump to a sampled Z
    scroll(delta)                      # relative pointer move (clamped)
    add_bbox(z_index, bbox)            # annotate the CURRENT slice
    run_medsam2(key_z, bbox)           # 3D propagation (once per trajectory)
    add_point(z_index, points, labels) # refine CURRENT slice
    finish_3d_segmentation()           # terminate

Key differences from Phase 2:
  - Turn 1 carries NO slice images; only the task text + sampled Z list.
  - Navigation tools lazily render the correct overlay (raw / bbox / mask /
    refined_mask) for each slice view.
  - Every navigation tool_response includes `overlays`, `boundary`, and
    `history` (visited / annotated / unvisited ordinals).
  - add_bbox / add_point are REJECTED unless the supplied z_index matches the
    slice currently being viewed (z == sampled_z_list[cur_ord]).
  - scroll before the pointer is initialised (before any get_slice) returns an
    error tool_response.

Metrics: Dice, HD95, Precision, Recall — same as Phase 2.

Usage:
    conda run -n qwen3 
CUDA_VISIBLE_DEVICES=4,5 \
QWENVL_BBOX_FORMAT='new' \
python Qwen3_VL/infer_phase3.py \
    --model  /BDSZ6/private/user/yxd/models/Qwen3-VL-8B-Instruct \
    --ckpt   /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3_18-22/v2-20260422-230040/checkpoint-4050 \
    --val_jsonl /BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22/agent_val_subset.jsonl \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_18-22/train \
    --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3_18-22/output
    --device cuda:0
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

try:
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
except ImportError:
    InferRequest = RequestConfig = TransformersEngine = None  # teacher-forced mode only

MEDSAM2_ROOT = Path(__file__).resolve().parents[1] / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))

# ── Constants ───────────────────────────────────────────────────────────────
N_SLICES        = 10
RENDER_SIZE     = 256
IMG_MEAN        = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMG_STD         = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
MAX_AGENT_TURNS = 60
DICE_PASS       = 0.70

BBOX_COLOR     = (255, 80, 80)
POINT_FG_COLOR = (255, 50, 50)
POINT_BG_COLOR = (50, 100, 255)

TOOL_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
SAMPLED_Z_RE = re.compile(r'Sampled Z list[^:]*:\s*\[([^\]]+)\]')
VID_FROM_RENDER = re.compile(r'/([^/]+?)_z\d+_(?:raw|bbox|mask|refined)\.png$')

# ── Tool schema (must match convert_to_agent_trajectory_phase3.py) ──────────
TOOLS = [
    {"type": "function", "function": {"name": "get_slice",
        "description": "Jump directly to a sampled Z slice.",
        "parameters": {"type": "object", "properties": {
            "z_index": {"type": "integer"}
        }, "required": ["z_index"]}}},
    {"type": "function", "function": {"name": "scroll",
        "description": "Move the slice pointer by `delta` in sampled-list order.",
        "parameters": {"type": "object", "properties": {
            "delta": {"type": "integer"}
        }, "required": ["delta"]}}},
    {"type": "function", "function": {"name": "add_bbox",
        "description": "Annotate bbox on the current slice.",
        "parameters": {"type": "object", "properties": {
            "z_index": {"type": "integer"},
            "bbox":    {"type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4}
        }, "required": ["z_index", "bbox"]}}},
    {"type": "function", "function": {"name": "run_medsam2",
        "description": "Trigger MedSAM2 3D segmentation seeded from key slice.",
        "parameters": {"type": "object", "properties": {
            "key_z": {"type": "integer"},
            "bbox":  {"type": "array", "items": {"type": "integer"},
                      "minItems": 4, "maxItems": 4}
        }, "required": ["key_z", "bbox"]}}},
    {"type": "function", "function": {"name": "add_point",
        "description": "Refine mask on the current slice via point prompts.",
        "parameters": {"type": "object", "properties": {
            "z_index": {"type": "integer"},
            "points":  {"type": "array"},
            "labels":  {"type": "array"}
        }, "required": ["z_index", "points", "labels"]}}},
    {"type": "function", "function": {"name": "finish_3d_segmentation",
        "description": "Terminate the task.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


# ── Geometry / metric helpers ───────────────────────────────────────────────

def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if (pred.sum() == 0 and gt.sum() == 0) else 0.0
    return float(2 * inter / denom)


def precision_recall(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)


def hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any(): return 0.0
    if not pred.any() or not gt.any(): return float('inf')
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)
    dt_gt   = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)
    all_d = np.concatenate([dt_gt[pred_surf], dt_pred[gt_surf]])
    return float(np.percentile(all_d, 95))


def bbox_iou(bbox, gt_2d):
    if gt_2d.sum() == 0: return 1.0
    x1, y1, x2, y2 = bbox
    gt_rows, gt_cols = np.where(gt_2d > 0)
    gx1, gy1 = int(gt_cols.min()), int(gt_rows.min())
    gx2, gy2 = int(gt_cols.max()), int(gt_rows.max())
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter
    return inter / union if union > 0 else 0.0


# ── Rendering ────────────────────────────────────────────────────────────────

def render_raw(img_path, save_path):
    Image.open(img_path).convert('RGB').resize(
        (RENDER_SIZE, RENDER_SIZE), Image.LANCZOS).save(save_path)


def render_bbox_overlay(img_path, bbox, save_path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=BBOX_COLOR, width=2)
    img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS).save(save_path)


def render_mask_overlay(img_path, mask, save_path,
                         points=None, labels=None, bbox=None):
    img = Image.open(img_path).convert('RGBA')
    H, W = mask.shape
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    px = np.zeros((H, W, 4), dtype=np.uint8)
    px[mask.astype(bool)] = [0, 200, 100, 120]
    overlay.paste(Image.fromarray(px, 'RGBA'))
    img = Image.alpha_composite(img, overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        draw.rectangle(bbox, outline=BBOX_COLOR, width=2)
    if points and labels:
        for (x, y), lbl in zip(points, labels):
            color = POINT_FG_COLOR if lbl == 1 else POINT_BG_COLOR
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline='white')
    img.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS).save(save_path)


# ── MedSAM2 helpers (mirror Phase 2) ────────────────────────────────────────

def load_volume(jpeg_dir, device):
    files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith('.jpg'))
    slices = [np.array(Image.open(os.path.join(jpeg_dir, f)).convert('L')) for f in files]
    arr = np.stack(slices, axis=0)
    orig_H, orig_W = arr.shape[1], arr.shape[2]
    arr_f = arr.astype(np.float32) / 255.0
    if orig_H != 512 or orig_W != 512:
        resized = []
        for s in arr_f:
            img = Image.fromarray((s * 255).astype(np.uint8))
            resized.append(np.array(img.resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255.0)
        arr_f = np.stack(resized, axis=0)
    t = torch.from_numpy(arr_f).unsqueeze(1).expand(-1, 3, -1, -1)
    t = (t - IMG_MEAN[None, :, None, None]) / IMG_STD[None, :, None, None]
    return t.to(device), orig_H, orig_W


def run_medsam2_propagate(predictor, img_tensor, orig_H, orig_W,
                           key_z, bbox_512, device):
    result = {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=key_z, obj_id=1, box=bbox_512)
        key_mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        result[key_z] = key_mask.astype(bool)
        predictor.reset_state(state)

        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(
                state, start_frame_idx=key_z, reverse=False):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)

        state = predictor.init_state(img_tensor, orig_H, orig_W)
        predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
        for fidx, _, lg in predictor.propagate_in_video(
                state, start_frame_idx=key_z, reverse=True):
            result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
        predictor.reset_state(state)
    return result


def run_medsam2_point(predictor, img_tensor, orig_H, orig_W,
                      frame_idx, points_512, labels, device):
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=1,
            points=points_512, labels=labels)
        mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(bool)
        predictor.reset_state(state)
    return mask


# ── Tool call parsing ───────────────────────────────────────────────────────

def parse_tool_calls(text):
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append({'name': obj.get('name', ''),
                          'arguments': obj.get('arguments', {})})
        except Exception:
            pass
    return calls


def split_cot_and_calls(text):
    first = text.find('<tool_call>')
    cot = text[:first].strip() if first >= 0 else text.strip()
    raw_calls = TOOL_CALL_RE.findall(text)
    return cot, raw_calls


# ── Phase 3 navigation-aware agent executor ─────────────────────────────────

class NavAgentExecutor:
    """Executes Phase 3 tool calls; manages navigation pointer + overlay state."""

    def __init__(self, predictor, device, jpeg_dir, img_tensor,
                 orig_H, orig_W, frames, masks, sampled,
                 render_dir, vid, img_w, img_h):
        self.predictor  = predictor
        self.device     = device
        self.jpeg_dir   = jpeg_dir
        self.img_tensor = img_tensor
        self.orig_H     = orig_H
        self.orig_W     = orig_W
        self.frames     = frames
        self.masks      = masks
        self.sampled    = list(sampled)
        self.render_dir = render_dir
        self.vid        = vid
        self.img_w      = img_w
        self.img_h      = img_h
        self.sx = 512 / img_w
        self.sy = 512 / img_h

        self.cur_ord = None
        self.visited = []
        self.annotated = set()
        self.overlay_state = {i: 'raw' for i in range(len(sampled))}
        self.bboxes_by_ord = {}
        self.dice_per_ord = {}
        self.refined_dice = {}

        self.final_masks = {}        # z -> mask (512,512)
        self.messages = []
        self.images = []

    # ── path helpers ────────────────────────────────────────────────────────
    def slice_path(self, z):
        if z < 0 or z >= len(self.frames):
            return None
        return os.path.join(self.jpeg_dir, self.frames[z] + '.jpg')

    def _render_path(self, z, suffix):
        return os.path.join(self.render_dir, f"{self.vid}_z{z:03d}_{suffix}.png")

    # ── metric helpers ──────────────────────────────────────────────────────
    def _iou_vs_gt(self, bbox, z):
        m = self.masks[z] if 0 <= z < len(self.masks) else None
        if m is None: return 0.0
        gt = maskUtils.decode(m)
        return round(bbox_iou(bbox, gt), 3)

    def _dice_vs_gt(self, pred, z):
        m = self.masks[z] if 0 <= z < len(self.masks) else None
        if m is None: return 0.0
        gt = maskUtils.decode(m).astype(bool)
        return round(dice_score(pred, gt), 3)

    # ── navigation response builder ─────────────────────────────────────────
    def _render_for_current_state(self, ord_idx):
        """Return the render path matching the current overlay state for this ord."""
        z = self.sampled[ord_idx]
        state = self.overlay_state[ord_idx]
        if state == 'refined_mask':
            return self._render_path(z, 'refined')
        if state == 'mask':
            return self._render_path(z, 'mask')
        if state == 'bbox':
            return self._render_path(z, 'bbox')
        # raw: lazily render if not yet cached
        rp = self._render_path(z, 'raw')
        if not os.path.exists(rp):
            render_raw(self.slice_path(z), rp)
        return rp

    def _emit_nav_response(self, ord_idx, clamped=False):
        z = self.sampled[ord_idx]
        state = self.overlay_state[ord_idx]
        rp = self._render_for_current_state(ord_idx)
        self.images.append(rp)

        ov = {
            'has_bbox': ord_idx in self.bboxes_by_ord,
            'has_mask': state in ('mask', 'refined_mask'),
            'has_refined_mask': state == 'refined_mask',
        }
        if ov['has_bbox']:
            ov['bbox'] = self.bboxes_by_ord[ord_idx]
        if ov['has_mask']:
            ov['mask_dice'] = self.dice_per_ord.get(ord_idx, 0.0)
        if ov['has_refined_mask']:
            ov['refined_dice'] = self.refined_dice.get(z, 0.0)

        history = {
            'visited_ordinals': list(self.visited),
            'annotated_ordinals': sorted(self.annotated),
            'unvisited_ordinals': [i for i in range(len(self.sampled))
                                    if i not in set(self.visited)],
        }
        return json.dumps({
            'z_index': z,
            'ordinal': ord_idx,
            'slice_image': '<image>',
            'sampled_z_list': list(self.sampled),
            'overlays': ov,
            'boundary': {
                'at_start': ord_idx == 0,
                'at_end': ord_idx == len(self.sampled) - 1,
                'clamped': clamped,
            },
            'history': history,
        })

    # ── tool handlers ───────────────────────────────────────────────────────
    def exec_get_slice(self, z_index):
        if z_index not in self.sampled:
            return json.dumps({"error": "z_index not in sampled set",
                                "sampled_z_list": list(self.sampled)})
        ord_idx = self.sampled.index(z_index)
        self.cur_ord = ord_idx
        self.visited.append(ord_idx)
        return self._emit_nav_response(ord_idx, clamped=False)

    def exec_scroll(self, delta):
        if self.cur_ord is None:
            return json.dumps({"error": "pointer not initialized — call get_slice first"})
        if delta == 0:
            return json.dumps({"error": "scroll delta must not be 0"})
        target = self.cur_ord + int(delta)
        new_ord = max(0, min(len(self.sampled) - 1, target))
        clamped = (new_ord != target)
        self.cur_ord = new_ord
        self.visited.append(new_ord)
        return self._emit_nav_response(new_ord, clamped=clamped)

    def exec_add_bbox(self, z_index, bbox):
        if self.cur_ord is None or self.sampled[self.cur_ord] != z_index:
            cur_z = self.sampled[self.cur_ord] if self.cur_ord is not None else None
            return json.dumps({"error": f"must view slice Z={z_index} before add_bbox "
                                         f"(current Z={cur_z})"})
        ord_idx = self.cur_ord
        self.bboxes_by_ord[ord_idx] = list(bbox)
        self.annotated.add(ord_idx)
        self.overlay_state[ord_idx] = 'bbox'
        rp = self._render_path(z_index, 'bbox')
        render_bbox_overlay(self.slice_path(z_index), bbox, rp)
        self.images.append(rp)
        return json.dumps({
            "z_index": z_index,
            "bbox_image": "<image>",
            "iou_with_gt": self._iou_vs_gt(bbox, z_index),
        })

    def exec_run_medsam2(self, key_z, bbox):
        if self.predictor is None:
            # Fallback: GT masks
            propagated = {}
            for z in self.sampled:
                m = self.masks[z] if 0 <= z < len(self.masks) else None
                propagated[z] = maskUtils.decode(m).astype(bool) if m is not None \
                                 else np.zeros((512, 512), dtype=bool)
        else:
            bbox_512 = np.array([bbox[0]*self.sx, bbox[1]*self.sy,
                                  bbox[2]*self.sx, bbox[3]*self.sy], dtype=np.float32)
            propagated = run_medsam2_propagate(
                self.predictor, self.img_tensor, self.orig_H, self.orig_W,
                key_z, bbox_512, self.device
            )
        self.final_masks = dict(propagated)

        slices_info = []
        for i, z in enumerate(self.sampled):
            pred = propagated.get(z)
            if pred is None:
                slices_info.append({"z_index": z, "ordinal": i,
                                     "mask_image": None, "dice_with_gt": -1.0})
                continue
            rp = self._render_path(z, 'mask')
            render_mask_overlay(self.slice_path(z), pred, rp,
                                bbox=self.bboxes_by_ord.get(i))
            self.images.append(rp)
            d = self._dice_vs_gt(pred, z)
            self.dice_per_ord[i] = d
            self.overlay_state[i] = 'mask'
            slices_info.append({"z_index": z, "ordinal": i,
                                 "mask_image": "<image>", "dice_with_gt": d})
        return json.dumps({
            "status": "propagation_complete",
            "key_z": key_z,
            "slices": slices_info,
        })

    def exec_add_point(self, z_index, points, labels):
        if self.cur_ord is None or self.sampled[self.cur_ord] != z_index:
            cur_z = self.sampled[self.cur_ord] if self.cur_ord is not None else None
            return json.dumps({"error": f"must view slice Z={z_index} before add_point "
                                         f"(current Z={cur_z})"})
        ord_idx = self.cur_ord
        if self.predictor is not None:
            pts_512 = np.array([[p[0]*self.sx, p[1]*self.sy] for p in points],
                                dtype=np.float32)
            lbls = np.array(labels, dtype=np.int32)
            refined = run_medsam2_point(
                self.predictor, self.img_tensor, self.orig_H, self.orig_W,
                z_index, pts_512, lbls, self.device
            )
        else:
            m = self.masks[z_index] if 0 <= z_index < len(self.masks) else None
            refined = maskUtils.decode(m).astype(bool) if m is not None \
                       else np.zeros((512, 512), dtype=bool)

        self.final_masks[z_index] = refined
        self.overlay_state[ord_idx] = 'refined_mask'
        rp = self._render_path(z_index, 'refined')
        render_mask_overlay(self.slice_path(z_index), refined, rp,
                            points=points, labels=labels,
                            bbox=self.bboxes_by_ord.get(ord_idx))
        self.images.append(rp)
        d = self._dice_vs_gt(refined, z_index)
        self.refined_dice[z_index] = d
        return json.dumps({
            "z_index": z_index,
            "mask_image": "<image>",
            "dice_with_gt": d,
        })

    # ── main agent loop ─────────────────────────────────────────────────────
    def run_loop(self, engine, request_cfg, user_content):
        self.messages = [{"role": "user", "content": user_content}]
        self.images   = []           # Turn 1 has no images in Phase 3
        done = False

        for turn in range(MAX_AGENT_TURNS):
            req = InferRequest(
                messages=self.messages,
                tools=TOOLS,
                images=self.images,
            )
            resp = engine.infer([req], request_config=request_cfg)[0]
            asst_text = resp.choices[0].message.content or ''

            tool_calls = parse_tool_calls(asst_text)
            cot, raw_calls = split_cot_and_calls(asst_text)
            if cot:
                self.messages.append({"role": "assistant", "content": cot})

            if any(tc['name'] == 'finish_3d_segmentation' for tc in tool_calls):
                for raw in raw_calls:
                    self.messages.append({"role": "tool_call", "content": raw.strip()})
                done = True
                break
            if not tool_calls:
                break

            # Execute sequentially — each call may depend on state of prior one.
            for tc, raw in zip(tool_calls, raw_calls):
                self.messages.append({"role": "tool_call", "content": raw.strip()})
                name, args = tc['name'], tc['arguments']
                try:
                    if name == 'get_slice':
                        resp_json = self.exec_get_slice(int(args['z_index']))
                    elif name == 'scroll':
                        resp_json = self.exec_scroll(int(args['delta']))
                    elif name == 'add_bbox':
                        resp_json = self.exec_add_bbox(int(args['z_index']),
                                                       list(args['bbox']))
                    elif name == 'run_medsam2':
                        resp_json = self.exec_run_medsam2(int(args['key_z']),
                                                          list(args['bbox']))
                    elif name == 'add_point':
                        resp_json = self.exec_add_point(int(args['z_index']),
                                                        list(args['points']),
                                                        list(args['labels']))
                    else:
                        resp_json = json.dumps({"error": f"unknown tool: {name}"})
                except Exception as e:
                    resp_json = json.dumps({"error": f"tool exec failed: {type(e).__name__}: {e}"})
                self.messages.append({"role": "tool_response", "content": resp_json})

        return self.final_masks, done


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_case(final_masks, masks, non_none_z):
    per_frame, vol_pred, vol_gt = [], [], []
    for z in non_none_z:
        m = masks[z] if z < len(masks) else None
        if m is None: continue
        gt = maskUtils.decode(m).astype(bool)
        pr = final_masks.get(z)
        if pr is None: pr = np.zeros_like(gt)
        d = dice_score(pr, gt); h = hd95(pr, gt); p, r = precision_recall(pr, gt)
        per_frame.append({'z': z, 'dice': round(d, 4), 'hd95': round(h, 4),
                           'precision': round(p, 4), 'recall': round(r, 4)})
        vol_pred.append(pr); vol_gt.append(gt)
    if not vol_pred:
        return {'per_frame': [], 'dice': 0.0, 'hd95': float('inf'),
                 'precision': 0.0, 'recall': 0.0}
    vp = np.stack(vol_pred); vg = np.stack(vol_gt)
    return {
        'per_frame': per_frame,
        'dice':      round(dice_score(vp, vg), 4),
        'hd95':      round(hd95(vp, vg), 4),
        'precision': round(precision_recall(vp, vg)[0], 4),
        'recall':    round(precision_recall(vp, vg)[1], 4),
    }


# ── Data loading ────────────────────────────────────────────────────────────

def load_val_samples(val_jsonl, max_samples=None):
    samples = []
    with open(val_jsonl) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples: break
            samples.append(json.loads(line))
    return samples


def extract_metadata_phase3(rec):
    """Pull the user message + parse the sampled Z list from it."""
    user_content = rec['messages'][0]['content']
    m = SAMPLED_Z_RE.search(user_content)
    if not m:
        raise ValueError("Could not parse 'Sampled Z list' from user content")
    sampled = [int(x.strip()) for x in m.group(1).split(',')]
    return user_content, sampled


def load_mask_dict_and_meta(data_root):
    with open(os.path.join(data_root, 'mask_dict.pkl'), 'rb') as f:
        mask_dict = pickle.load(f)
    with open(os.path.join(data_root, 'meta_expressions.json')) as f:
        meta = json.load(f)['videos']
    return mask_dict, meta


def build_anno_index(meta, mask_dict):
    annos, idx = [], 0
    for vid in sorted(meta.keys()):
        vd = meta[vid]
        frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):
            masks = mask_dict[str(idx)]
            annos.append({
                'vid': vid, 'eid': eid,
                'caption': vd['expressions'][eid]['exp'],
                'anno_id': str(idx), 'frames': frames,
                'img_w': vd['width'], 'img_h': vd['height'],
                'masks': masks,
                'non_none_z': [i for i, m in enumerate(masks) if m is not None],
            })
            idx += 1
    return annos


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',        default='/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--ckpt',         required=True)
    p.add_argument('--val_jsonl',    default='/BDSZ6/private/user/yxd/data/qwen/agent_phase3/agent_val.jsonl')
    p.add_argument('--data_root',    default='/BDSZ6/private/user/yxd/data/M3D/data_18-22/train')
    p.add_argument('--medsam2_ckpt', default=None)
    p.add_argument('--medsam2_cfg',  default=None)
    p.add_argument('--output_dir',   default='/tmp/phase3_eval')
    p.add_argument('--device',       default='cuda:0')
    p.add_argument('--max_samples',  type=int, default=None)
    p.add_argument('--max_tokens',   type=int, default=2048)
    p.add_argument('--temperature',  type=float, default=0.0)
    p.add_argument('--teacher_forced', action='store_true',
                   help='Replay the GT tool_call sequence from --val_jsonl instead '
                        'of running the model. Reproduces the oracle pipeline ceiling '
                        '(matches visualize_phase3.py). Skips LLM loading.')
    return p.parse_args()


def replay_trajectory(rec, executor):
    """Teacher-forced: execute each GT tool_call from the record in order.
    Returns (final_masks, done) matching run_loop's contract."""
    done = False
    for m in rec['messages']:
        if m['role'] != 'tool_call':
            continue
        try:
            tc = json.loads(m['content'])
        except Exception:
            continue
        name, args = tc.get('name'), tc.get('arguments', {}) or {}
        try:
            if name == 'get_slice':
                executor.exec_get_slice(int(args['z_index']))
            elif name == 'scroll':
                executor.exec_scroll(int(args['delta']))
            elif name == 'add_bbox':
                executor.exec_add_bbox(int(args['z_index']), list(args['bbox']))
            elif name == 'run_medsam2':
                executor.exec_run_medsam2(int(args['key_z']), list(args['bbox']))
            elif name == 'add_point':
                executor.exec_add_point(int(args['z_index']),
                                        list(args['points']),
                                        list(args['labels']))
            elif name == 'finish_3d_segmentation':
                done = True
                break
        except Exception as e:
            print(f"\n[WARN] replay step {name} failed: {type(e).__name__}: {e}")
    return executor.final_masks, done


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    render_dir = os.path.join(args.output_dir, 'renders')
    os.makedirs(render_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.teacher_forced:
        print("[Mode] Teacher-forced replay (no LLM loaded).")
        engine = request_cfg = None
    else:
        if TransformersEngine is None:
            raise RuntimeError(
                "ms-swift is required for autoregressive inference; install it or use --teacher_forced")
        print(f"[LLM] Loading {args.model} + adapter {args.ckpt}")
        engine = TransformersEngine(args.model, adapters=[args.ckpt])
        request_cfg = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)

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
        print("[MedSAM2] No checkpoint — GT-derived masks used for tool execution.")

    print("[Data] Loading validation JSONL …")
    val_samples = load_val_samples(args.val_jsonl, max_samples=args.max_samples)
    mask_dict, meta = load_mask_dict_and_meta(args.data_root)
    anno_lookup = {(a['vid'], a['caption']): a
                    for a in build_anno_index(meta, mask_dict)}
    jpeg_root = os.path.join(args.data_root, 'JPEGImages')

    _caption_re = re.compile(r'The target structure is:\s*"([^"]+)"')

    def match_anno(rec):
        user_text = rec['messages'][0]['content']
        m = _caption_re.search(user_text)
        caption = m.group(1) if m else ''
        # Phase-3 Turn 1 has no images, but render paths written by
        # convert_to_agent_trajectory_phase3.py (and by this script) encode the
        # vid as `{vid}_z{zzz}_{raw|bbox|mask|refined}.png`. Recover vid from
        # the first such filename; identical strategy to visualize_phase3.py.
        vid = None
        for img in rec.get('images', []):
            p = img.get('path') if isinstance(img, dict) else str(img)
            if not p: continue
            mm = VID_FROM_RENDER.search(p)
            if mm: vid = mm.group(1); break
        if vid is not None:
            anno = anno_lookup.get((vid, caption))
            if anno is not None:
                return anno
        raise KeyError(f"No annotation match for vid={vid!r} caption={caption!r}")

    all_results = []
    for rec in tqdm(val_samples, desc='Evaluating'):
        try:
            anno = match_anno(rec)
        except KeyError as e:
            print(f"[WARN] {e}")
            continue
        vid, frames, masks = anno['vid'], anno['frames'], anno['masks']
        non_none_z = anno['non_none_z']
        jpeg_dir = os.path.join(jpeg_root, vid)
        img_w, img_h = anno['img_w'], anno['img_h']

        user_content, sampled = extract_metadata_phase3(rec)

        img_tensor = orig_H = orig_W = None
        if predictor is not None:
            img_tensor, orig_H, orig_W = load_volume(jpeg_dir, device)
        else:
            orig_H, orig_W = img_h, img_w

        executor = NavAgentExecutor(
            predictor=predictor, device=device,
            jpeg_dir=jpeg_dir, img_tensor=img_tensor,
            orig_H=orig_H, orig_W=orig_W,
            frames=frames, masks=masks, sampled=sampled,
            render_dir=render_dir, vid=vid, img_w=img_w, img_h=img_h,
        )
        try:
            if args.teacher_forced:
                final_masks, done = replay_trajectory(rec, executor)
            else:
                final_masks, done = executor.run_loop(engine, request_cfg, user_content)
        except Exception as e:
            print(f"\n[WARN] Agent loop failed for {vid}: {type(e).__name__}: {e}")
            final_masks, done = {}, False

        metrics = evaluate_case(final_masks, masks, non_none_z)
        metrics.update(vid=vid, caption=anno['caption'], finished=done,
                        n_turns=len(executor.visited))
        all_results.append(metrics)
        tqdm.write(
            f"  {vid}  Dice={metrics['dice']:.3f}  HD95={metrics['hd95']:.1f}  "
            f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
            f"finished={done}  reads={metrics['n_turns']}"
        )

    results_path = os.path.join(args.output_dir, 'results.jsonl')
    with open(results_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    finite_hd95 = [r['hd95'] for r in all_results if r['hd95'] != float('inf')]
    print("\n" + "=" * 50)
    print(f"{'Metric':<20} {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("=" * 50)
    for key in ('dice', 'precision', 'recall'):
        vals = [r[key] for r in all_results]
        if not vals: continue
        print(f"{key:<20} {np.mean(vals):>10.4f}  {np.std(vals):>10.4f}"
              f"  {np.min(vals):>10.4f}  {np.max(vals):>10.4f}")
    if finite_hd95:
        print(f"{'hd95 (finite)':<20} {np.mean(finite_hd95):>10.2f}  "
              f"{np.std(finite_hd95):>10.2f}  "
              f"{np.min(finite_hd95):>10.2f}  {np.max(finite_hd95):>10.2f}")
    inf_count = len(all_results) - len(finite_hd95)
    if inf_count:
        print(f"  ({inf_count} cases with HD95=inf)")
    finished = sum(1 for r in all_results if r['finished'])
    print(f"\nFinished cleanly: {finished}/{len(all_results)}")
    print(f"Results saved to: {results_path}")
    print(f"Renders in:       {render_dir}/")


if __name__ == '__main__':
    main()
