#!/usr/bin/env python3
"""
Convert Stage 2 MRI segmentation data to ms-swift grounding JSONL format for Qwen3-VL training.

Usage:
python Qwen3_VL/convert_to_swift_grounding2.py \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_18-22/train \
    --output_dir /BDSZ6/private/user/yxd/data/qwen/data_18-22_video \
        [--train_ratio 0.8] \
        [--include_negatives] \
        [--max_frames_per_expr 20] \
        [--seed 42]

Output:
    {output_dir}/mri_grounding_train.jsonl
    {output_dir}/mri_grounding_val.jsonl

Each line is a ms-swift grounding entry with:
  - messages: system/user/assistant conversation
  - images: list of absolute image paths
  - objects: {"ref": [...captions...], "bbox": [[x1,y1,x2,y2],...]}
             bbox_type='real' (default) — ms-swift handles Qwen3-VL thousandth normalization.
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
NEG_RESP   = "No match for the target."


def _user_msg_pos(slice_idx, total_slices):
    return (f"<image>This image is slice {slice_idx} out of {total_slices} from an MRI volume. "
            f"Identify and localize <ref-object> regions in this slice.")


def _user_msg_neg(slice_idx, total_slices):
    return (f"<image>This image is slice {slice_idx} out of {total_slices} from an MRI volume. "
            f"Identify and localize pathological regions in this slice.")


# ---------------------------------------------------------------------------
# Helpers
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
    Decode a COCO RLE mask and return per-component bboxes in image pixel coords.
    Returns list of [x1, y1, x2, y2] (empty if mask decodes to all zeros).
    """
    mask = maskUtils.decode(mask_rle)  # (mask_h, mask_w) uint8
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


def make_positive_entry(image_path, caption, bboxes, slice_idx, total_slices):
    """Build one JSONL dict for a frame with 1+ lesion bboxes."""
    n = len(bboxes)
    assistant_content = "".join(["<bbox>"] * n)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": _user_msg_pos(slice_idx, total_slices)},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
        "objects": {
            "ref":  [caption],
            "bbox": bboxes,
        },
    }


def make_negative_entry(image_path, slice_idx, total_slices):
    """Build one JSONL dict for a frame with no visible lesion."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": _user_msg_neg(slice_idx, total_slices)},
            {"role": "assistant", "content": NEG_RESP},
        ],
        "images": [image_path],
        "objects": {"ref": [], "bbox": []},
    }


def build_ordered_annos(meta):
    """
    Replicate the exact annotation ordering used by load_refytvos_json() in
    mllm/dataset/utils/refytvos_utils.py (lines 107-133) so that sequential
    anno_id values correctly map to mask_dict keys.

    Returns list of dicts: video, exp_id, caption, anno_id (str), frames, img_h, img_w.
    """
    annos = []
    anno_idx = 0
    for vid in sorted(meta.keys()):                                  # lexicographic sort
        vd = meta[vid]
        vid_frames = sorted(vd['frames'])
        for eid in sorted(vd['expressions'].keys(), key=int):        # numeric sort
            exp_dict = vd['expressions'][eid]
            annos.append({
                'video':    vid,
                'exp_id':   eid,
                'caption':  exp_dict['exp'],
                'anno_id':  str(anno_idx),
                'frames':   vid_frames,
                'img_h':    vd['height'],
                'img_w':    vd['width'],
            })
            anno_idx += 1
    return annos


def get_image_size(jpeg_root, video, frame):
    """Open a JPEG and return (width, height). Cached by caller."""
    path = os.path.join(jpeg_root, video, frame + '.jpg')
    with Image.open(path) as img:
        return img.size  # PIL returns (width, height)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args):
    # 1. Load metadata and masks
    meta_path = os.path.join(args.data_root, 'meta_expressions.json')
    mask_path = os.path.join(args.data_root, 'mask_dict.pkl')

    print(f'Loading {meta_path} ...')
    with open(meta_path, 'r') as f:
        meta = json.load(f)['videos']

    print(f'Loading {mask_path} ...')
    with open(mask_path, 'rb') as f:
        mask_dict = pickle.load(f)

    # 2. Build ordered annotation list (must match mask_dict key assignment)
    annos = build_ordered_annos(meta)
    assert len(annos) == len(mask_dict), (
        f'Annotation/mask count mismatch: {len(annos)} annos vs {len(mask_dict)} mask entries. '
        'The ordering logic may not match load_refytvos_json().'
    )
    print(f'Total expressions: {len(annos)}')

    # 3. Train/val split by video (deterministic, no shuffle)
    all_videos = sorted(meta.keys())
    split_idx  = int(len(all_videos) * args.train_ratio)
    train_vids = set(all_videos[:split_idx])
    val_vids   = set(all_videos[split_idx:])
    print(f'Videos: {len(all_videos)} total  (train: {len(train_vids)}, val: {len(val_vids)})')

    # 4. Image size cache (avoid re-opening the same JPEG repeatedly)
    img_size_cache = {}   # video -> (img_w, img_h)

    jpeg_root = os.path.join(args.data_root, 'JPEGImages')

    # 5. Open output files
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'mri_grounding_train.jsonl')
    val_path   = os.path.join(args.output_dir, 'mri_grounding_val.jsonl')

    stats = {
        'train': {'pos': 0, 'neg': 0},
        'val':   {'pos': 0, 'neg': 0},
    }

    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for anno in tqdm(annos, desc='Converting'):
            vid      = anno['video']
            caption  = anno['caption']
            frames   = anno['frames']       # sorted list of 100 frame name strings
            anno_id  = anno['anno_id']
            split    = 'train' if vid in train_vids else 'val'
            f_out    = f_train if split == 'train' else f_val

            # Determine image dimensions (cached per video)
            if vid not in img_size_cache:
                img_w, img_h = get_image_size(jpeg_root, vid, frames[0])
                img_size_cache[vid] = (img_w, img_h)
            else:
                img_w, img_h = img_size_cache[vid]

            # Mask resolution is always 512×512 for this dataset
            mask_h, mask_w = 512, 512
            scale_x = img_w / mask_w
            scale_y = img_h / mask_h

            frame_masks  = mask_dict[anno_id]   # list of 100 (COCO RLE or None)
            pos_indices  = [i for i, m in enumerate(frame_masks) if m is not None]
            none_indices = [i for i, m in enumerate(frame_masks) if m is None]

            total_slices = len(frames)

            if len(pos_indices) == 0:
                # All-None expression: one negative entry using middle frame
                mid_idx   = len(frames) // 2
                mid_frame = frames[mid_idx]
                img_path  = os.path.join(jpeg_root, vid, mid_frame + '.jpg')
                entry = make_negative_entry(img_path, mid_idx + 1, total_slices)
                f_out.write(json.dumps(entry) + '\n')
                stats[split]['neg'] += 1

            else:
                # Possibly subsample positive frames
                chosen_pos = pos_indices
                if args.max_frames_per_expr and len(pos_indices) > args.max_frames_per_expr:
                    idxs = np.round(
                        np.linspace(0, len(pos_indices) - 1, args.max_frames_per_expr)
                    ).astype(int)
                    chosen_pos = [pos_indices[i] for i in idxs]

                for frame_idx in chosen_pos:
                    frame_name = frames[frame_idx]
                    img_path   = os.path.join(jpeg_root, vid, frame_name + '.jpg')
                    bboxes = extract_bboxes(
                        frame_masks[frame_idx],
                        scale_x=scale_x, scale_y=scale_y,
                        img_w=img_w, img_h=img_h,
                    )
                    slice_idx = frame_idx + 1   # 1-based
                    if not bboxes:
                        # Decoded mask was all-zeros — treat as negative
                        entry = make_negative_entry(img_path, slice_idx, total_slices)
                        stats[split]['neg'] += 1
                    else:
                        entry = make_positive_entry(img_path, caption, bboxes, slice_idx, total_slices)
                        stats[split]['pos'] += 1
                    f_out.write(json.dumps(entry) + '\n')

                if args.include_negatives:
                    for frame_idx in none_indices:
                        frame_name = frames[frame_idx]
                        img_path   = os.path.join(jpeg_root, vid, frame_name + '.jpg')
                        entry = make_negative_entry(img_path, frame_idx + 1, total_slices)
                        f_out.write(json.dumps(entry) + '\n')
                        stats[split]['neg'] += 1

    # 6. Summary
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
        description='Convert M3D Stage 2 MRI data to ms-swift grounding JSONL for Qwen3-VL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_root', default='/BDSZ6/private/user/yxd/data/M3D/train',
        help='Path to Stage 2 data root (contains JPEGImages/, meta_expressions.json, mask_dict.pkl)',
    )
    parser.add_argument(
        '--output_dir', default='/home/yxd/OPEN-DTOS-LMM/Qwen3_VL',
        help='Directory to write mri_grounding_train.jsonl and mri_grounding_val.jsonl',
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.9,
        help='Fraction of videos used for training (split is by video, deterministic)',
    )
    parser.add_argument(
        '--include_negatives', action='store_true', default=False,
        help='Also emit per-frame negative entries for None-mask frames within positive expressions',
    )
    parser.add_argument(
        '--max_frames_per_expr', type=int, default=None,
        help='Cap on positive frames per expression (None = no cap, subsample via linspace if exceeded)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (used for numpy; split itself is deterministic)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    convert(args)
