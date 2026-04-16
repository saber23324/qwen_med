#!/usr/bin/env python3
"""
Convert Stage 2 MRI segmentation data to ms-swift grounding JSONL format for Qwen3-VL training.

Usage:
python Qwen3_VL/convert_to_swift_grounding.py \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_18-22/train \
    --output_dir /BDSZ6/private/user/yxd/data/qwen/data_18-22_one \
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


def _user_msg_neg(slice_idx, total_slices):
    return (f"<image>This image is slice {slice_idx} out of {total_slices} from an MRI volume. "
            f"Identify and localize pathological regions in this slice.")


def _user_msg_pos(slice_idx, total_slices, unique_captions):
    """
    Build user prompt embedding one <ref-object> tag per unique category,
    joined with ' and ':
        "... Identify and localize <ref-object> regions ..."          (1 category)
        "... Identify and localize <ref-object> and <ref-object> ..." (2 categories)
    """
    ref_str = " and ".join(["<ref-object>"] * len(unique_captions))
    return (f"<image>This image is slice {slice_idx} out of {total_slices} from an MRI volume. "
            f"Identify and localize {ref_str} regions in this slice.")


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


def extract_bboxes(
    mask_rle,
    scale_x=1.0,
    scale_y=1.0,
    img_w=512,
    img_h=512,
    min_area=20,          # 面积阈值（像素）
    min_side=3,           # 最小边长（像素）
    normalize=False        # 是否做千分位归一化
):
    """
    Decode a COCO RLE mask and return per-component bboxes.

    Returns:
        list of [x1, y1, x2, y2]
        - pixel coords if normalize=False
        - [0,1000] normalized ints if normalize=True
    """
    mask = maskUtils.decode(mask_rle)  # (H, W)

    if mask.sum() == 0:
        return []

    comp_boxes = bounding_boxes_per_component(mask)
    result = []

    for (y1, y2, x1, x2) in comp_boxes:
        # scale到目标尺寸
        
        sx1 = int(min(max(round(x1 * scale_x), 0), img_w - 1))
        sy1 = int(min(max(round(y1 * scale_y), 0), img_h - 1))
        sx2 = int(min(max(round(x2 * scale_x), 0), img_w - 1))
        sy2 = int(min(max(round(y2 * scale_y), 0), img_h - 1))
        
        # --- 1. 过滤小bbox ---
        w = max(0, sx2 - sx1)
        h = max(0, sy2 - sy1)
        area = w * h

        if area < min_area or w < min_side or h < min_side:
            continue

        # --- 2. 千分位归一化 ---
        if normalize:
            nx1 = int(round(sx1 / img_w * 1000))
            ny1 = int(round(sy1 / img_h * 1000))
            nx2 = int(round(sx2 / img_w * 1000))
            ny2 = int(round(sy2 / img_h * 1000))

            # 保证边界
            nx1 = min(max(nx1, 0), 1000)
            ny1 = min(max(ny1, 0), 1000)
            nx2 = min(max(nx2, 0), 1000)
            ny2 = min(max(ny2, 0), 1000)

            result.append([nx1, ny1, nx2, ny2])
        else:
            result.append([sx1, sy1, sx2, sy2])

    return result


def make_positive_entry(image_path, captions, bboxes, slice_idx, total_slices):
    """
    Build one JSONL dict for a frame with 1+ lesion bboxes, possibly from
    multiple expressions/categories.

    captions : list of str, one per bbox (same caption repeated for multiple
               connected components of the same expression).
    bboxes   : list of [x1,y1,x2,y2], same length as captions.

    objects.ref layout
    ------------------
    ms-swift replaces <ref-object> tokens in message order across all roles.
    The user turn contains one <ref-object> per *unique* category; the
    assistant turn contains one <ref-object> per bbox.  Both draw from
    objects.ref in sequence, so the list is:
        [unique_cap_1, ..., unique_cap_K,   <- consumed by user turn
         bbox_cap_1,   ..., bbox_cap_N]     <- consumed by assistant turn
    """
    # Unique captions in first-appearance order (preserves expression ordering)
    seen, unique_captions = set(), []
    for cap in captions:
        if cap not in seen:
            seen.add(cap)
            unique_captions.append(cap)

    user_content      = _user_msg_pos(slice_idx, total_slices, unique_captions)
    assistant_content = "".join(["<bbox>"] * len(bboxes))

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
        "objects": {
            "ref":  unique_captions,   # user refs then assistant refs
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

    # Group annotations by video so we can merge all expressions per frame
    from collections import defaultdict
    video_to_annos = defaultdict(list)
    for anno in annos:
        video_to_annos[anno['video']].append(anno)

    with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
        for vid in tqdm(sorted(video_to_annos.keys()), desc='Converting'):
            vid_annos = video_to_annos[vid]
            frames    = vid_annos[0]['frames']   # same frame list for all expressions in vid
            split     = 'train' if vid in train_vids else 'val'
            f_out     = f_train if split == 'train' else f_val
            total_slices = len(frames)

            # Determine image dimensions (cached per video)
            if vid not in img_size_cache:
                img_w, img_h = get_image_size(jpeg_root, vid, frames[0])
                img_size_cache[vid] = (img_w, img_h)
            else:
                img_w, img_h = img_size_cache[vid]

            mask_w, mask_h = 512, 512
            scale_x = img_w / mask_w
            scale_y = img_h / mask_h

            # Determine which frames to include: union of positive frames across
            # all expressions, then subsample if requested.
            all_pos_frame_indices = sorted({
                i
                for anno in vid_annos
                for i, m in enumerate(mask_dict[anno['anno_id']])
                if m is not None
            })

            if not all_pos_frame_indices:
                # No expression in this video has any mask — emit one negative
                # entry for the middle frame.
                mid_idx  = len(frames) // 2
                img_path = os.path.join(jpeg_root, vid, frames[mid_idx] + '.jpg')
                entry = make_negative_entry(img_path, mid_idx + 1, total_slices)
                f_out.write(json.dumps(entry) + '\n')
                stats[split]['neg'] += 1
                continue

            # Subsample positive frames if requested
            chosen_pos = all_pos_frame_indices
            if args.max_frames_per_expr and len(all_pos_frame_indices) > args.max_frames_per_expr:
                idxs = np.round(
                    np.linspace(0, len(all_pos_frame_indices) - 1, args.max_frames_per_expr)
                ).astype(int)
                chosen_pos = [all_pos_frame_indices[i] for i in idxs]
            chosen_pos_set = set(chosen_pos)

            for frame_idx, frame_name in enumerate(frames):
                img_path  = os.path.join(jpeg_root, vid, frame_name + '.jpg')
                slice_idx = frame_idx + 1   # 1-based

                if frame_idx not in chosen_pos_set:
                    # Frame not selected as positive — optionally emit negative
                    if args.include_negatives:
                        entry = make_negative_entry(img_path, slice_idx, total_slices)
                        f_out.write(json.dumps(entry) + '\n')
                        stats[split]['neg'] += 1
                    continue

                # Collect bboxes from ALL expressions active on this frame,
                # preserving expression order (already sorted by anno_id).
                frame_captions = []
                frame_bboxes   = []
                for anno in vid_annos:
                    mask_list = mask_dict[anno['anno_id']]
                    if mask_list[frame_idx] is None:
                        continue
                    bboxes = extract_bboxes(
                        mask_list[frame_idx],
                        scale_x=scale_x, scale_y=scale_y,
                        img_w=img_w, img_h=img_h,
                    )
                    # Each connected component gets its own <ref-object><bbox> pair
                    for bbox in bboxes:
                        frame_captions.append(anno['caption'])
                        frame_bboxes.append(bbox)

                if frame_bboxes:
                    entry = make_positive_entry(
                        img_path, frame_captions, frame_bboxes, slice_idx, total_slices
                    )
                    stats[split]['pos'] += 1
                else:
                    # All masks decoded to zero — treat as negative
                    entry = make_negative_entry(img_path, slice_idx, total_slices)
                    stats[split]['neg'] += 1
                f_out.write(json.dumps(entry) + '\n')

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
