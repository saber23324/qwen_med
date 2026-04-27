#!/usr/bin/env python3
"""
Convert Phase 3 SFT JSONL into GRPO input JSONL.

Each output row keeps only the user prompt + GT side-channel needed by the
multi-turn rollout scheduler (`Qwen3_VL/phase3_rl_rollout.py`) and reward
plugin (`Qwen3_VL/grpo_plugin.py`). Assistant CoT and tool-call/response
turns are stripped — RL re-rolls the trajectory from scratch.

Output schema (one JSON object per line):

    {
      "messages": [{"role": "user", "content": "Task: ...\\nSampled Z = ..."}],
      "images":   [],
      "vid":      "volume_001",
      "caption":  "the liver tumor visible on MRI",
      "anno_id":  "42",
      "data_root": "/.../M3D/data_18-22/train",
      "sampled_z": [40, 43, 46, 50, 53, 56, 60, 63, 66, 70],
      "lesion_ordinals":     [3, 4, 5, 6, 7],
      "non_lesion_ordinals": [0, 1, 2, 8, 9],
      "oracle_key_z": 53
    }

`vid` is recovered from the rendered image filename pattern in the SFT
record (same strategy as `infer_phase3.match_anno`). `caption` is parsed
from the user content. Everything else is computed from the GT mask via
`mask_dict.pkl` + `meta_expressions.json` under `--data_root`.

Usage
-----
    conda run -n qwen3 python3 Qwen3_VL/convert_to_grpo_dataset.py \
        --sft_jsonl  /BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22/agent_val.jsonl \
        --data_root  /BDSZ6/private/user/yxd/data/M3D/data_18-22/train \
        --output     /BDSZ6/private/user/yxd/data/qwen/agent_phase3_18-22/grpo_val.jsonl

    # And again for val:
    conda run -n qwen3 python3 Qwen3_VL/convert_to_grpo_dataset.py \\
        --sft_jsonl  /BDSZ6/.../agent_phase3_18-22/agent_val.jsonl \\
        --data_root  /BDSZ6/.../M3D/data_18-22/train \\
        --output     /BDSZ6/.../agent_phase3_18-22/grpo_val.jsonl
"""

import argparse
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as maskUtils
from tqdm import tqdm

VID_FROM_RENDER = re.compile(r'/([^/]+?)_z\d+_(?:raw|bbox|mask|refined)\.png$')
SAMPLED_Z_RE    = re.compile(r'Sampled Z list[^:]*:\s*\[([^\]]+)\]')
CAPTION_RE      = re.compile(r'The target structure is:\s*"([^"]+)"')


def load_mask_meta(data_root: str) -> Tuple[Dict, Dict]:
    with open(os.path.join(data_root, 'mask_dict.pkl'), 'rb') as f:
        mask_dict = pickle.load(f)
    with open(os.path.join(data_root, 'meta_expressions.json')) as f:
        meta = json.load(f)['videos']
    return mask_dict, meta


def build_caption_to_anno(meta: Dict, mask_dict: Dict) -> Dict[Tuple[str, str], Dict]:
    """Same indexing as infer_phase3.build_anno_index, keyed by (vid, caption)."""
    out: Dict[Tuple[str, str], Dict] = {}
    idx = 0
    for vid in sorted(meta.keys()):
        vd = meta[vid]
        for eid in sorted(vd['expressions'].keys(), key=int):
            caption = vd['expressions'][eid]['exp']
            out[(vid, caption)] = {
                'anno_id': str(idx),
                'masks':   mask_dict[str(idx)],
            }
            idx += 1
    return out


def parse_user_meta(user_text: str) -> Tuple[Optional[str], List[int]]:
    cap_m = CAPTION_RE.search(user_text)
    caption = cap_m.group(1) if cap_m else None
    z_m = SAMPLED_Z_RE.search(user_text)
    sampled = [int(x.strip()) for x in z_m.group(1).split(',')] if z_m else []
    return caption, sampled


def vid_from_images(rec: Dict) -> Optional[str]:
    for img in rec.get('images', []):
        p = img.get('path') if isinstance(img, dict) else str(img)
        if not p:
            continue
        m = VID_FROM_RENDER.search(p)
        if m:
            return m.group(1)
    return None


def lesion_partition(masks: List, sampled_z: List[int]) -> Tuple[List[int], List[int]]:
    lesion, non_lesion = [], []
    for ord_idx, z in enumerate(sampled_z):
        if 0 <= z < len(masks) and masks[z] is not None:
            try:
                arr = maskUtils.decode(masks[z])
                if arr.any():
                    lesion.append(ord_idx)
                    continue
            except Exception:
                pass
        non_lesion.append(ord_idx)
    return lesion, non_lesion


def oracle_key_z(masks: List, sampled_z: List[int],
                 lesion_ords: List[int]) -> Optional[int]:
    if not lesion_ords:
        return None
    best_ord, best_area = lesion_ords[0], -1
    for ord_idx in lesion_ords:
        z = sampled_z[ord_idx]
        try:
            area = int(maskUtils.decode(masks[z]).sum())
        except Exception:
            area = 0
        if area > best_area:
            best_area, best_ord = area, ord_idx
    return sampled_z[best_ord]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sft_jsonl', required=True,
                   help='Phase 3 SFT JSONL produced by convert_to_agent_trajectory_phase3.py')
    p.add_argument('--data_root', required=True,
                   help='M3D data root containing mask_dict.pkl + meta_expressions.json')
    p.add_argument('--output',    required=True, help='Output GRPO JSONL path')
    p.add_argument('--max_samples', type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mask_dict, meta = load_mask_meta(args.data_root)
    cap_to_anno = build_caption_to_anno(meta, mask_dict)

    n_in = n_out = n_skip = 0
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.sft_jsonl) as fin, open(args.output, 'w') as fout:
        for line in tqdm(fin, desc='Converting'):
            n_in += 1
            if args.max_samples and n_in > args.max_samples:
                break
            rec = json.loads(line)
            user_msg = rec['messages'][0]
            user_text = user_msg.get('content', '') or ''
            caption, sampled_z = parse_user_meta(user_text)
            vid = vid_from_images(rec)
            if not (caption and sampled_z and vid):
                n_skip += 1
                continue
            anno = cap_to_anno.get((vid, caption))
            if anno is None:
                n_skip += 1
                continue

            lesion, non_lesion = lesion_partition(anno['masks'], sampled_z)
            key_z = oracle_key_z(anno['masks'], sampled_z, lesion)

            row = {
                'messages':            [{'role': 'user', 'content': user_text}],
                'images':              [],
                'vid':                 vid,
                'caption':             caption,
                'anno_id':             anno['anno_id'],
                'data_root':           args.data_root,
                'sampled_z':           sampled_z,
                'lesion_ordinals':     lesion,
                'non_lesion_ordinals': non_lesion,
                'oracle_key_z':        key_z,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            n_out += 1

    print(f'[done] in={n_in}  out={n_out}  skipped={n_skip}')
    print(f'       wrote: {args.output}')


if __name__ == '__main__':
    main()
