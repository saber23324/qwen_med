#!/usr/bin/env python3
"""
Visualize Phase-3 inference results.

Same metric/figure format as ``visualize_phase2.py``, but adapted for Phase-3
trajectories in which Turn-1 carries no slice images — the sampled Z list is
written in plain text, and the volume id has to be recovered from a render
filename.

Usage:
    conda run -n dtos_test python3 Qwen3_VL/visualize_phase3.py \
        --infer_jsonl /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase3/v0-20260419-225116/checkpoint-200/infer_result \
        --data_root   /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
        --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
        --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
        --output_dir  /home/yxd/medagent/output/phase3_vis \
        --device cuda:4
"""

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

MEDSAM2_ROOT = Path(__file__).resolve().parents[1] / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMG_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

CAPTION_RE      = re.compile(r'The target structure is:\s*"([^"]+)"')
SAMPLED_LIST_RE = re.compile(r'Sampled Z list[^:]*:\s*\[([^\]]+)\]')
VID_FROM_RENDER = re.compile(r'/([^/]+?)_z\d+_(?:raw|bbox|mask|refined)\.png$')


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(2 * inter / denom)


def precision_recall(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    pp = pred.sum()
    ap = gt.sum()
    return (float(tp / pp) if pp > 0 else 0.0,
            float(tp / ap) if ap > 0 else 0.0)


def hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float('inf')
    ps = pred ^ binary_erosion(pred)
    gs = gt   ^ binary_erosion(gt)
    dt_g = distance_transform_edt(~gs)
    dt_p = distance_transform_edt(~ps)
    return float(np.percentile(np.concatenate([dt_g[ps], dt_p[gs]]), 95))


def compute_metrics(pred_masks: dict, masks: list, non_none_z: list) -> dict:
    vol_pred, vol_gt, per_frame = [], [], []
    for z in non_none_z:
        m = masks[z]
        if m is None:
            continue
        gt_2d   = maskUtils.decode(m).astype(bool)
        pred_2d = pred_masks.get(z, np.zeros_like(gt_2d))
        d       = dice_score(pred_2d, gt_2d)
        p, r    = precision_recall(pred_2d, gt_2d)
        h       = hd95(pred_2d, gt_2d)
        per_frame.append({'z': z, 'dice': round(d,4), 'hd95': round(h,4),
                          'precision': round(p,4), 'recall': round(r,4)})
        vol_pred.append(pred_2d)
        vol_gt.append(gt_2d)

    if not vol_pred:
        return {'per_frame': [], 'dice': 0., 'hd95': float('inf'),
                'precision': 0., 'recall': 0.}

    vp = np.stack(vol_pred); vg = np.stack(vol_gt)
    vd = dice_score(vp, vg)
    vh = hd95(vp, vg)
    vpr, vrec = precision_recall(vp, vg)
    return {'per_frame': per_frame,
            'dice':      round(vd, 4),
            'hd95':      round(vh, 4),
            'precision': round(vpr, 4),
            'recall':    round(vrec, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# MedSAM2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_volume(jpeg_dir: str, device):
    files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith('.jpg'))
    slices = [np.array(Image.open(os.path.join(jpeg_dir, f)).convert('L')) for f in files]
    arr = np.stack(slices, axis=0).astype(np.float32) / 255.0
    orig_H, orig_W = arr.shape[1], arr.shape[2]
    if orig_H != 512 or orig_W != 512:
        resized = [np.array(Image.fromarray((s*255).astype(np.uint8))
                            .resize((512,512), Image.LANCZOS)).astype(np.float32)/255.
                   for s in arr]
        arr = np.stack(resized)
    t = torch.from_numpy(arr).unsqueeze(1).expand(-1, 3, -1, -1)
    t = (t - IMG_MEAN[None,:,None,None]) / IMG_STD[None,:,None,None]
    return t.to(device), orig_H, orig_W


def medsam2_propagate(predictor, img_tensor, orig_H, orig_W,
                      key_z, bbox_512, device) -> dict:
    result = {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, lg = predictor.add_new_points_or_box(
            state, frame_idx=key_z, obj_id=1, box=bbox_512)
        key_mask = (lg[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        result[key_z] = key_mask.astype(bool)
        predictor.reset_state(state)
        for reverse in [False, True]:
            state = predictor.init_state(img_tensor, orig_H, orig_W)
            predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
            for fidx, _, lg in predictor.propagate_in_video(
                    state, start_frame_idx=key_z, reverse=reverse):
                result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
            predictor.reset_state(state)
    return result


def medsam2_point(predictor, img_tensor, orig_H, orig_W,
                  frame_idx, pts_512, lbls, device) -> np.ndarray:
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        state = predictor.init_state(img_tensor, orig_H, orig_W)
        _, _, lg = predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=1,
            points=pts_512, labels=lbls)
        mask = (lg[0] > 0.0).squeeze(0).cpu().numpy().astype(bool)
        predictor.reset_state(state)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Parse inference JSONL record
# ─────────────────────────────────────────────────────────────────────────────

def _extract_vid(images: list) -> str | None:
    for img in images:
        p = img.get('path') if isinstance(img, dict) else str(img)
        if not p:
            continue
        m = VID_FROM_RENDER.search(p)
        if m:
            return m.group(1)
    return None


def parse_record(rec: dict) -> dict:
    """Extract everything the model predicted from one Phase-3 inference record."""
    user_text = rec['messages'][0]['content']
    vid       = _extract_vid(rec.get('images', []))
    caption_m = CAPTION_RE.search(user_text)
    sampled_m = SAMPLED_LIST_RE.search(user_text)
    caption   = caption_m.group(1) if caption_m else ''
    sampled   = ([int(x.strip()) for x in sampled_m.group(1).split(',')]
                 if sampled_m else [])

    add_bboxes   = {}
    medsam2_call = None
    add_points   = {}

    for m in rec['messages']:
        if m['role'] != 'tool_call':
            continue
        try:
            tc = json.loads(m['content'])
        except Exception:
            continue
        name = tc.get('name')
        args = tc.get('arguments', {}) or {}
        if name == 'add_bbox':
            add_bboxes[int(args['z_index'])] = list(args['bbox'])
        elif name == 'run_medsam2':
            medsam2_call = {'key_z': int(args['key_z']), 'bbox': list(args['bbox'])}
        elif name == 'add_point':
            add_points[int(args['z_index'])] = {
                'points': args['points'], 'labels': args['labels']}

    return {
        'vid':         vid,
        'caption':     caption,
        'sampled':     sampled,
        'add_bboxes':  add_bboxes,
        'medsam2':     medsam2_call,
        'add_points':  add_points,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_root: str):
    with open(os.path.join(data_root, 'mask_dict.pkl'), 'rb') as f:
        mask_dict = pickle.load(f)
    with open(os.path.join(data_root, 'meta_expressions.json')) as f:
        meta = json.load(f)['videos']
    return mask_dict, meta


def build_anno_lookup(meta, mask_dict) -> dict:
    lookup = {}
    idx = 0
    for vid in sorted(meta.keys()):
        vd     = meta[vid]
        frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):
            masks       = mask_dict[str(idx)]
            non_none_z  = [i for i, m in enumerate(masks) if m is not None]
            caption     = vd['expressions'][eid]['exp']
            lookup[(vid, caption)] = {
                'vid': vid, 'eid': eid, 'caption': caption,
                'frames': frames,
                'img_w': vd['width'], 'img_h': vd['height'],
                'masks': masks, 'non_none_z': non_none_z,
            }
            idx += 1
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _overlay(gray_img: np.ndarray, mask: np.ndarray, color_rgb, alpha=0.45) -> np.ndarray:
    if gray_img.ndim == 2:
        rgb = np.stack([gray_img]*3, axis=-1).astype(np.float32)
    else:
        rgb = gray_img.astype(np.float32)
    c = np.array(color_rgb, dtype=np.float32)
    m = mask.astype(bool)
    rgb[m] = rgb[m] * (1 - alpha) + c * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _overlap_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred, gt = pred.astype(bool), gt.astype(bool)
    rgb = np.full((*pred.shape, 3), 30, dtype=np.uint8)
    rgb[~pred & ~gt]  = [30,   30,  30]
    rgb[ pred &  gt]  = [255, 255, 255]
    rgb[~pred &  gt]  = [30,  100, 255]
    rgb[ pred & ~gt]  = [255, 140,   0]
    return rgb


def make_case_figure(case_info: dict, pred_masks: dict, anno: dict,
                     metrics: dict, jpeg_root: str) -> plt.Figure:
    sampled   = case_info['sampled']
    masks     = anno['masks']
    frames    = anno['frames']
    vid       = anno['vid']
    jpeg_dir  = os.path.join(jpeg_root, vid)
    per_frame = {f['z']: f for f in metrics['per_frame']}

    rows = [z for z in sampled if z < len(masks) and masks[z] is not None]
    if not rows:
        rows = sampled

    n_rows = len(rows)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2 + 2.5, n_rows * 3.0 + 1.2),
                             squeeze=False)

    bboxes = case_info.get('add_bboxes', {})
    key_z  = case_info.get('medsam2', {}).get('key_z') if case_info.get('medsam2') else None
    refined = set(case_info.get('add_points', {}).keys())

    fig.suptitle(
        f"{vid}  |  target: {case_info['caption']}\n"
        f"Vol Dice={metrics['dice']:.3f}  "
        f"HD95={metrics['hd95']:.1f}  "
        f"Prec={metrics['precision']:.3f}  "
        f"Recall={metrics['recall']:.3f}  "
        f"|  key_z={key_z}  bboxes={len(bboxes)}  refined={len(refined)}",
        fontsize=11, fontweight='bold', y=0.995
    )

    col_titles = ['MRI (input)', 'GT mask', 'Predicted mask', 'Overlap map']
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, pad=4)

    for row, z in enumerate(rows):
        frame_name = frames[z] + '.jpg'
        mri_path = os.path.join(jpeg_dir, frame_name)
        mri = np.array(Image.open(mri_path).convert('L'))

        m = masks[z] if z < len(masks) else None
        gt_2d = maskUtils.decode(m).astype(bool) if m is not None \
                else np.zeros(mri.shape, dtype=bool)

        pred_2d = pred_masks.get(z, np.zeros(mri.shape, dtype=bool))

        def _resize(arr_2d):
            if arr_2d.shape != (512, 512):
                return np.array(Image.fromarray(arr_2d.astype(np.uint8) * 255)
                                .resize((512,512), Image.NEAREST)).astype(bool)
            return arr_2d

        gt_2d   = _resize(gt_2d)
        pred_2d = _resize(pred_2d)
        if mri.shape != (512, 512):
            mri = np.array(Image.fromarray(mri).resize((512,512), Image.LANCZOS))

        panels = [
            mri,
            _overlay(mri, gt_2d,   (30, 100, 255)),
            _overlay(mri, pred_2d, (0,  200, 100)),
            _overlap_map(pred_2d, gt_2d),
        ]
        for c, panel in enumerate(panels):
            ax = axes[row, c]
            if panel.ndim == 2:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.axis('off')

        # Phase-3 extras: draw predicted bbox on the MRI panel; flag key_z / refined z
        bb = bboxes.get(z)
        if bb:
            sx = 512 / anno['img_w']
            sy = 512 / anno['img_h']
            rect = mpatches.Rectangle(
                (bb[0]*sx, bb[1]*sy),
                (bb[2]-bb[0])*sx, (bb[3]-bb[1])*sy,
                linewidth=1.5, edgecolor='yellow', facecolor='none')
            axes[row, 0].add_patch(rect)

        tags = []
        if z == key_z: tags.append('★key')
        if z in refined: tags.append('◆refined')
        label = f'Z={z}' + (f'  {" ".join(tags)}' if tags else '')
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0,
                                labelpad=34, va='center')

        pf = per_frame.get(z)
        if pf:
            color = 'green' if pf['dice'] >= 0.7 else 'red'
            txt = (f"Dice={pf['dice']:.3f}\n"
                   f"HD95={pf['hd95']:.1f}\n"
                   f"Prec={pf['precision']:.3f}\n"
                   f"Rec={pf['recall']:.3f}")
            axes[row, n_cols - 1].text(
                1.05, 0.5, txt,
                transform=axes[row, n_cols - 1].transAxes,
                fontsize=7.5, va='center', ha='left',
                color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor=color, alpha=0.85))

    legend_patches = [
        mpatches.Patch(color=(1,1,1),       label='TP'),
        mpatches.Patch(color=(0.12,0.39,1), label='FN (missed)'),
        mpatches.Patch(color=(1,0.55,0),    label='FP (over-pred)'),
        mpatches.Patch(color=(1,1,0),       label='Predicted bbox'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def make_summary_figure(all_metrics: list) -> plt.Figure:
    n = len(all_metrics)
    vids   = [m['vid'].replace('_image','')[-20:] for m in all_metrics]
    dices  = [m['dice']      for m in all_metrics]
    hd95s  = [min(m['hd95'], 100) for m in all_metrics]
    precs  = [m['precision'] for m in all_metrics]
    recs   = [m['recall']    for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle('Phase-3 Val-set Segmentation Metrics (per case)', fontsize=13)
    x = np.arange(n)
    bar_kw = dict(edgecolor='black', linewidth=0.4)

    def _bar(ax, vals, title, color, ylim=(0,1)):
        ax.bar(x, vals, color=color, alpha=0.75, **bar_kw)
        ax.axhline(np.mean(vals), color='red', linewidth=1.5,
                   linestyle='--', label=f'mean={np.mean(vals):.3f}')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(vids, rotation=55, ha='right', fontsize=5.5)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    _bar(axes[0,0], dices, 'Dice',      '#4c9be8')
    _bar(axes[0,1], precs, 'Precision', '#59c278')
    _bar(axes[1,0], recs,  'Recall',    '#f0a030')
    _bar(axes[1,1], hd95s, 'HD95 (capped @ 100)', '#d9534f',
         ylim=(0, max(hd95s)*1.1+1 if hd95s else 1))

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--infer_jsonl', required=True)
    p.add_argument('--data_root',   default='/BDSZ6/private/user/yxd/data/M3D/data_6-13/train')
    p.add_argument('--medsam2_ckpt', default=None)
    p.add_argument('--medsam2_cfg',  default=None)
    p.add_argument('--output_dir',   default='/tmp/phase3_vis')
    p.add_argument('--device',       default='cuda:4')
    p.add_argument('--max_samples',  type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load MedSAM2 ─────────────────────────────────────────────────────────
    predictor = None
    if args.medsam2_ckpt and args.medsam2_cfg:
        from sam2.build_sam import build_sam2_video_predictor_npz
        cfg = args.medsam2_cfg
        if os.path.isabs(cfg) and not cfg.startswith('//'): cfg = '//' + cfg
        torch.set_float32_matmul_precision('high')
        predictor = build_sam2_video_predictor_npz(
            cfg, args.medsam2_ckpt, device=device, apply_postprocessing=False)
        print(f"[MedSAM2] loaded  device={device}")
    else:
        print("[MedSAM2] not provided — pred masks will be empty "
              "(metrics effectively compare GT vs all-zero).")

    # ── Load data ─────────────────────────────────────────────────────────────
    mask_dict, meta = load_data(args.data_root)
    anno_lookup     = build_anno_lookup(meta, mask_dict)
    jpeg_root       = os.path.join(args.data_root, 'JPEGImages')

    with open(args.infer_jsonl) as f:
        records = [json.loads(l) for l in f]
    if args.max_samples:
        records = records[:args.max_samples]

    all_metrics = []

    for rec in tqdm(records, desc='Processing cases'):
        info = parse_record(rec)
        vid, caption = info['vid'], info['caption']
        if not vid or not caption:
            print(f"[WARN] skipped: missing vid or caption (vid={vid!r}, cap={caption!r})")
            continue

        anno = anno_lookup.get((vid, caption))
        if anno is None:
            print(f"[WARN] annotation not found: {vid!r} / {caption!r}")
            continue

        masks      = anno['masks']
        non_none_z = anno['non_none_z']
        jpeg_dir   = os.path.join(jpeg_root, vid)
        img_w, img_h = anno['img_w'], anno['img_h']

        # ── Re-run MedSAM2 using the model's predicted bbox + add_points ─────
        pred_masks: dict = {}

        if predictor is not None and info['medsam2'] is not None:
            try:
                img_tensor, orig_H, orig_W = load_volume(jpeg_dir, device)
                sx, sy = 512 / img_w, 512 / img_h

                ms  = info['medsam2']
                kbb = ms['bbox']
                bbox_512 = np.array(
                    [kbb[0]*sx, kbb[1]*sy, kbb[2]*sx, kbb[3]*sy],
                    dtype=np.float32)
                pred_masks = medsam2_propagate(
                    predictor, img_tensor, orig_H, orig_W,
                    ms['key_z'], bbox_512, device)

                for z, pt_info in info['add_points'].items():
                    pts_512 = np.array(
                        [[p[0]*sx, p[1]*sy] for p in pt_info['points']],
                        dtype=np.float32)
                    lbls = np.array(pt_info['labels'], dtype=np.int32)
                    pred_masks[z] = medsam2_point(
                        predictor, img_tensor, orig_H, orig_W,
                        z, pts_512, lbls, device)
            except Exception as e:
                print(f"[ERR] MedSAM2 inference failed for {vid}: {e}")
                pred_masks = {}

        if not pred_masks:
            for z in info['sampled']:
                m = masks[z] if z < len(masks) else None
                shape = maskUtils.decode(m).shape if m is not None else (512, 512)
                pred_masks[z] = np.zeros(shape, dtype=bool)

        metrics = compute_metrics(pred_masks, masks, non_none_z)
        metrics['vid']     = vid
        metrics['caption'] = caption
        all_metrics.append(metrics)

        tqdm.write(
            f"  {vid}  Dice={metrics['dice']:.3f}  "
            f"HD95={metrics['hd95']:.1f}  "
            f"Prec={metrics['precision']:.3f}  "
            f"Recall={metrics['recall']:.3f}"
        )

        fig = make_case_figure(info, pred_masks, anno, metrics, jpeg_root)
        cap_slug = re.sub(r'[^a-z0-9]+', '-', caption.lower()).strip('-')[:40] or 'x'
        fig_path = os.path.join(fig_dir, f"{vid}__{cap_slug}.png")
        fig.savefig(fig_path, dpi=130, bbox_inches='tight')
        plt.close(fig)

    if all_metrics:
        sfig = make_summary_figure(all_metrics)
        sfig.savefig(os.path.join(args.output_dir, 'summary.png'),
                     dpi=130, bbox_inches='tight')
        plt.close(sfig)

    finite_hd = [m['hd95'] for m in all_metrics if m['hd95'] != float('inf')]
    print("\n" + "=" * 56)
    print(f"{'Metric':<16} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("=" * 56)
    for key in ('dice', 'precision', 'recall'):
        v = [m[key] for m in all_metrics]
        if v:
            print(f"{key:<16} {np.mean(v):>8.4f}  {np.std(v):>8.4f}"
                  f"  {np.min(v):>8.4f}  {np.max(v):>8.4f}")
    if finite_hd:
        print(f"{'hd95 (finite)':<16} {np.mean(finite_hd):>8.2f}  "
              f"{np.std(finite_hd):>8.2f}  "
              f"{np.min(finite_hd):>8.2f}  {np.max(finite_hd):>8.2f}")
        inf_n = len(all_metrics) - len(finite_hd)
        if inf_n: print(f"  ({inf_n} cases HD95=inf)")
    print("=" * 56)
    print(f"Cases: {len(all_metrics)}")

    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('vid,caption,dice,hd95,precision,recall\n')
        for m in all_metrics:
            hd = '' if m['hd95'] == float('inf') else f"{m['hd95']:.4f}"
            f.write(f"{m['vid']},{m['caption']},{m['dice']:.4f},"
                    f"{hd},{m['precision']:.4f},{m['recall']:.4f}\n")

    print(f"\nFigures : {fig_dir}/")
    print(f"Summary : {os.path.join(args.output_dir, 'summary.png')}")
    print(f"CSV     : {csv_path}")


if __name__ == '__main__':
    main()
