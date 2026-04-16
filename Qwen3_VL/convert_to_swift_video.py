#!/usr/bin/env python3
"""
Convert Stage 2 MRI data to ms-swift video-frame grounding JSONL format.

Each JSONL entry = one expression (one organ in one MRI volume).
All frame paths for that volume are fed as a video (`"videos"` key) so
Qwen3-VL receives the full temporal context of the scan.

The assistant response lists only frames that contain a lesion:
    Frame 14: <ref-object><bbox>
    Frame 15: <ref-object><bbox>
    Frame 16: <ref-object><bbox> and <ref-object><bbox>

Usage:
python Qwen3_VL/convert_to_swift_video.py \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_18-22/train \
    --output_dir /BDSZ6/private/user/yxd/data/qwen/data_18-22_video \
    --max_frames 10
    [--train_ratio 0.8] \
    [--max_frames 30]   # 0 or -1 = use all 100 frames (memory-intensive)

Output:
    {output_dir}/mri_video_train.jsonl
    {output_dir}/mri_video_val.jsonl
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

import pycocotools.mask as maskUtils

SYSTEM_MSG = "You are a medical imaging assistant specializing in MRI analysis."
USER_MSG   = (
    "The following video contains {n_frames} consecutive MRI slices from a single volume. "
    "localize <ref-object> region for For each slice. Only localize frames with visible lesions."
)
NEG_RESP   = "No pathological regions detected in this volume."


# ---------------------------------------------------------------------------
# Geometry helpers (identical to rvos.py static methods)
# ---------------------------------------------------------------------------

def bounding_boxes_per_component(mask):
    """Return list of (y1, y2, x1, x2) tuples, one per connected component."""
    labeled, num = ndimage.label(mask > 0)
    boxes = []
    for c in range(1, num + 1):
        rows = np.any(labeled == c, axis=1)
        cols = np.any(labeled == c, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        boxes.append((int(rmin), int(rmax), int(cmin), int(cmax)))
    return boxes


def extract_bboxes(mask_rle, scale_x=1.0, scale_y=1.0, img_w=512, img_h=512):
    """
    Decode COCO RLE mask → per-component bboxes in image pixel coords.
    Returns list of [x1, y1, x2, y2] (empty if mask is all zeros).
    """
    mask = maskUtils.decode(mask_rle)     # (mask_h, mask_w) uint8
    if mask.sum() == 0:
        return []
    comp_boxes = bounding_boxes_per_component(mask)
    result = []
    for (y1, y2, x1, x2) in comp_boxes:
        sx1 = int(min(max(round(x1 * scale_x), 0), img_w - 1))
        sy1 = int(min(max(round(y1 * scale_y), 0), img_h - 1))
        sx2 = int(min(max(round(x2 * scale_x), 0), img_w - 1))
        sy2 = int(min(max(round(y2 * scale_y), 0), img_h - 1))
        result.append([sx1, sy1, sx2, sy2])
    return result


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(pos_indices, none_indices, frames, max_frames):
    """
    Return a sampled (ordered) subset of frame indices covering both
    positive and, if capacity allows, some negative frames.
    Positive frames are always prioritised.

    Returns: (sampled_frame_indices_sorted, mapping_to_original_frame_names)
    """
    all_indices = sorted(pos_indices + none_indices)   # all 100 indices

    if max_frames <= 0 or max_frames >= len(all_indices):
        return all_indices

    # Always keep all positive frames if they fit; else subsample them too
    if len(pos_indices) <= max_frames:
        # Fill remaining slots with evenly-spaced negative frames
        n_neg = max_frames - len(pos_indices)
        if n_neg > 0 and none_indices:
            step_neg = np.round(
                np.linspace(0, len(none_indices) - 1, n_neg)
            ).astype(int)
            chosen_neg = [none_indices[i] for i in step_neg]
        else:
            chosen_neg = []
        return sorted(pos_indices + chosen_neg)
    else:
        # More positives than max_frames — subsample positives evenly
        step = np.round(
            np.linspace(0, len(pos_indices) - 1, max_frames)
        ).astype(int)
        return sorted([pos_indices[i] for i in step])


# ---------------------------------------------------------------------------
# Annotation ordering (must match load_refytvos_json in refytvos_utils.py)
# ---------------------------------------------------------------------------

def build_ordered_annos(meta):
    """
    Replicate load_refytvos_json() ordering so anno_id indices correctly
    map to mask_dict keys '0', '1', ..., '1092'.
    """
    annos = []
    anno_idx = 0
    for vid in sorted(meta.keys()):                              # lexicographic
        vd = meta[vid]
        vid_frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):   # numeric
            exp_dict = vd['expressions'][eid]
            annos.append({
                'video':   vid,
                'exp_id':  eid,
                'caption': exp_dict['exp'],
                'anno_id': str(anno_idx),
                'frames':  vid_frames,
                'img_h':   vd['height'],
                'img_w':   vd['width'],
            })
            anno_idx += 1
    return annos


def get_image_size(jpeg_root, video, frame):
    path = os.path.join(jpeg_root, video, frame + '.jpg')
    with Image.open(path) as img:
        return img.size   # (width, height)


# ---------------------------------------------------------------------------
# JSONL entry builders
# ---------------------------------------------------------------------------

def make_video_entry(frame_paths, n_total_frames, caption, per_frame_data):
    """
    Build one JSONL entry for a positive expression.

    per_frame_data: list of (frame_idx, bboxes) pairs, ordered by frame_idx.
                   bboxes is a list of [x1,y1,x2,y2] for each component.

    Returns the entry dict and total number of <ref-object>/<bbox> pairs used.
    """
    assistant_lines = []
    all_refs  = []
    all_bboxes = []

    for frame_idx, bboxes in per_frame_data:
        n = len(bboxes)
        parts = " and ".join(["<bbox>"] * n)
        assistant_lines.append(f"Frame {frame_idx}: {parts}")
        all_refs.extend([caption] * n)
        all_bboxes.extend(bboxes)

    assistant_content = "\n".join(assistant_lines)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": f"<video>{USER_MSG.format(n_frames=n_total_frames)}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "videos":  [frame_paths],      # single "video" = list of frame paths
        "objects": {
            "ref":  all_refs,
            "bbox": all_bboxes,
        },
        "channel": "grounding",
    }


def make_negative_video_entry(frame_paths, n_total_frames):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": f"<video>{USER_MSG.format(n_frames=n_total_frames)}"},
            {"role": "assistant", "content": NEG_RESP},
        ],
        "videos":  [frame_paths],
        "objects": {"ref": [], "bbox": []},
        "channel": "negative",
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args):
    meta_path = os.path.join(args.data_root, 'meta_expressions.json')
    mask_path = os.path.join(args.data_root, 'mask_dict.pkl')

    print(f'Loading {meta_path} ...')
    with open(meta_path, 'r') as f:
        meta = json.load(f)['videos']

    print(f'Loading {mask_path} ...')
    with open(mask_path, 'rb') as f:
        mask_dict = pickle.load(f)

    annos = build_ordered_annos(meta)
    assert len(annos) == len(mask_dict), (
        f'Annotation/mask count mismatch: {len(annos)} vs {len(mask_dict)}. '
        'Ordering logic may not match load_refytvos_json().'
    )
    print(f'Total expressions: {len(annos)}')

    # Train/val split by video (deterministic)
    all_videos = sorted(meta.keys())
    split_idx  = int(len(all_videos) * args.train_ratio)
    train_vids = set(all_videos[:split_idx])
    val_vids   = set(all_videos[split_idx:])
    print(f'Videos: {len(all_videos)} total  (train: {len(train_vids)}, val: {len(val_vids)})')

    jpeg_root   = os.path.join(args.data_root, 'JPEGImages')
    img_size_cache = {}   # video -> (img_w, img_h)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'mri_video_train.jsonl')
    val_path   = os.path.join(args.output_dir, 'mri_video_val.jsonl')

    stats = {'train': {'pos': 0, 'neg': 0}, 'val': {'pos': 0, 'neg': 0}}

    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for anno in tqdm(annos, desc='Converting'):
            vid      = anno['video']
            caption  = anno['caption']
            frames   = anno['frames']        # sorted list of 100 frame name strings
            anno_id  = anno['anno_id']
            split    = 'train' if vid in train_vids else 'val'
            f_out    = f_train if split == 'train' else f_val

            # Detect image dimensions once per video
            if vid not in img_size_cache:
                img_w, img_h = get_image_size(jpeg_root, vid, frames[0])
                img_size_cache[vid] = (img_w, img_h)
            else:
                img_w, img_h = img_size_cache[vid]

            mask_h, mask_w = 512, 512
            scale_x = img_w / mask_w
            scale_y = img_h / mask_h

            frame_masks  = mask_dict[anno_id]
            pos_indices  = [i for i, m in enumerate(frame_masks) if m is not None]
            none_indices = [i for i, m in enumerate(frame_masks) if m is None]

            if len(pos_indices) == 0:
                # All-None expression — use all frames as video, respond "no lesion"
                all_paths = [
                    os.path.join(jpeg_root, vid, frames[i] + '.jpg')
                    for i in range(len(frames))
                ]
                entry = make_negative_video_entry(all_paths, len(all_paths))
                f_out.write(json.dumps(entry) + '\n')
                stats[split]['neg'] += 1
                continue

            # Choose which frame indices to include in this video entry
            sampled = sample_frames(pos_indices, none_indices, frames, args.max_frames)

            # Build frame paths for the video
            frame_paths = [
                os.path.join(jpeg_root, vid, frames[i] + '.jpg')
                for i in sampled
            ]

            # Collect per-frame bbox annotations (only for positive frames)
            per_frame_data = []
            for local_idx, orig_idx in enumerate(sampled):
                mask_entry = frame_masks[orig_idx]
                if mask_entry is None:
                    continue   # negative frame — not annotated in response
                bboxes = extract_bboxes(
                    mask_entry,
                    scale_x=scale_x, scale_y=scale_y,
                    img_w=img_w, img_h=img_h,
                )
                if bboxes:
                    # Use the original frame index so the model sees real slice numbers
                    per_frame_data.append((orig_idx, bboxes))

            if not per_frame_data:
                # All positive masks decoded to empty — treat as negative
                entry = make_negative_video_entry(frame_paths, len(frame_paths))
                f_out.write(json.dumps(entry) + '\n')
                stats[split]['neg'] += 1
                continue

            entry = make_video_entry(frame_paths, len(frame_paths), caption, per_frame_data)
            f_out.write(json.dumps(entry) + '\n')
            stats[split]['pos'] += 1

    # Summary
    print('\n=== Conversion Summary ===')
    print(f'Videos: {len(all_videos)} total  (train: {len(train_vids)}, val: {len(val_vids)})')
    for split_name in ['train', 'val']:
        pos   = stats[split_name]['pos']
        neg   = stats[split_name]['neg']
        total = pos + neg
        ratio = f'{100 * pos / total:.2f}%' if total > 0 else 'N/A'
        print(f'\n{split_name.capitalize()} set:')
        print(f'  Positive entries : {pos}')
        print(f'  Negative entries : {neg}')
        print(f'  Total            : {total}')
        print(f'  Positive ratio   : {ratio}')
    print(f'\nOutput:')
    print(f'  {train_path}')
    print(f'  {val_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert M3D Stage 2 MRI data to ms-swift video-frame grounding JSONL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_root', default='/BDSZ6/private/user/yxd/data/M3D/data_6/train',
        help='Stage 2 data root (contains JPEGImages/, meta_expressions.json, mask_dict.pkl)',
    )
    parser.add_argument(
        '--output_dir', default='/BDSZ6/private/user/yxd/data/qwen/data_6',
        help='Directory for output JSONL files',
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Fraction of videos for training (rest → val). Split is by video, deterministic.',
    )
    parser.add_argument(
        '--max_frames', type=int, default=30,
        help=(
            'Max frames to include per video entry. '
            'Positive frames are always prioritised; negatives fill remaining slots. '
            'Use 0 or -1 to include all 100 frames (very memory intensive).'
        ),
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    convert(args)
