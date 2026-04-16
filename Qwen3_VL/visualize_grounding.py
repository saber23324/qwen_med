#!/usr/bin/env python3
"""
Visualize entries from a ms-swift grounding JSONL file (mri_grounding_*.jsonl).

For each sampled entry the script draws all bounding boxes on the MRI slice
and annotates each box with its caption.  Results are saved as PNGs.

Usage
-----
python Qwen3_VL/visualize_grounding.py \
    --jsonl  /BDSZ6/private/user/yxd/data/qwen/data_4_one/mri_grounding_val.jsonl \
    --output /BDSZ6/private/user/yxd/data/qwen/data_4_one/vis \
    --n 20
    [--n        20]      # number of entries to visualise (random sample)
    [--seed     42]
    [--all]              # visualise every entry (overrides --n)
    [--neg]              # include negative (no-bbox) entries in the sample
    [--index    5]       # visualise a single specific line index (0-based)
"""

import argparse
import json
import os
import random
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Palette — one colour per unique caption (cycles if more than len(PALETTE))
PALETTE = [
    (255,  80,  80),   # red
    ( 80, 160, 255),   # blue
    ( 80, 220,  80),   # green
    (255, 200,  50),   # yellow
    (200,  80, 255),   # purple
    ( 50, 220, 200),   # teal
    (255, 140,  50),   # orange
    (220, 220,  80),   # lime
]

BOX_WIDTH   = 2      # bbox outline thickness (px)
FONT_SIZE   = 14
LABEL_PAD   = 3      # padding around caption text
MAX_CAPTION = 60     # chars before wrapping caption label


def _assign_colors(captions):
    """Map unique captions to palette colours (deterministic within entry)."""
    seen, mapping = {}, {}
    idx = 0
    for cap in captions:
        if cap not in seen:
            seen[cap] = PALETTE[idx % len(PALETTE)]
            idx += 1
        mapping[cap] = seen[cap]
    return mapping


def _get_font(size=FONT_SIZE):
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_entry(entry, font):
    """
    Render one JSONL entry.  Returns a PIL Image with bboxes and captions
    drawn, or None if the image file is missing.
    """
    image_paths = entry.get('images', []) or entry.get('videos', [[]])[0]
    if not image_paths:
        return None
    img_path = image_paths[0]
    if not os.path.exists(img_path):
        print(f'  [SKIP] image not found: {img_path}')
        return None

    img  = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size

    objects  = entry.get('objects', {})
    captions = objects.get('ref',  [])
    bboxes   = objects.get('bbox', [])
    color_map = _assign_colors(captions)

    for cap, bbox in zip(captions, bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = color_map[cap]
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

        # Caption label — wrap long text
        label = '\n'.join(textwrap.wrap(cap, MAX_CAPTION))
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        # Place label above the box; flip below if out of frame
        lx = max(0, x1)
        ly = y1 - th - 2 * LABEL_PAD - BOX_WIDTH
        if ly < 0:
            ly = y2 + BOX_WIDTH + LABEL_PAD

        # Background rectangle for readability
        bg = [lx - LABEL_PAD, ly - LABEL_PAD,
              lx + tw + LABEL_PAD, ly + th + LABEL_PAD]
        draw.rectangle(bg, fill=(*color, 180))
        draw.text((lx, ly), label, fill=(255, 255, 255), font=font)

    # Overlay slice info from user message
    user_content = next(
        (m['content'] for m in entry.get('messages', []) if m['role'] == 'user'),
        ''
    )
    # Extract "slice X out of Y" substring for the title bar
    import re
    m = re.search(r'slice\s+(\d+)\s+out of\s+(\d+)', user_content)
    header = f'Slice {m.group(1)}/{m.group(2)}' if m else ''
    if not captions:
        header += '  [NEGATIVE]'
    else:
        header += f'  ({len(bboxes)} bbox{"es" if len(bboxes)>1 else ""})'

    # Draw header bar at top of image
    if header:
        hb = draw.textbbox((0, 0), header, font=font)
        hh = hb[3] - hb[1] + 2 * LABEL_PAD + 4
        bar = Image.new('RGB', (W, hh), (30, 30, 30))
        bar_draw = ImageDraw.Draw(bar)
        bar_draw.text((LABEL_PAD, LABEL_PAD), header, fill=(220, 220, 220), font=font)
        combined = Image.new('RGB', (W, H + hh))
        combined.paste(bar, (0, 0))
        combined.paste(img, (0, hh))
        img = combined

    return img


def load_entries(jsonl_path, include_neg, sample_n, seed, all_entries, index):
    with open(jsonl_path) as f:
        lines = f.readlines()

    if index is not None:
        return [json.loads(lines[index])], [index]

    entries, indices = [], []
    for i, line in enumerate(lines):
        e = json.loads(line)
        has_bbox = bool(e.get('objects', {}).get('bbox'))
        if not has_bbox and not include_neg:
            continue
        entries.append(e)
        indices.append(i)

    if not all_entries:
        rng = random.Random(seed)
        paired = list(zip(entries, indices))
        rng.shuffle(paired)
        paired = paired[:sample_n]
        entries, indices = zip(*paired) if paired else ([], [])

    return list(entries), list(indices)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--jsonl',   required=True,
                    help='Path to mri_grounding_*.jsonl')
    ap.add_argument('--output',  default='vis_output',
                    help='Directory to save visualised PNGs')
    ap.add_argument('--n',       type=int, default=20,
                    help='Number of entries to visualise (random sample)')
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--all',     action='store_true',
                    help='Visualise every entry (ignores --n)')
    ap.add_argument('--neg',     action='store_true',
                    help='Include negative (no-bbox) entries in the sample')
    ap.add_argument('--index',   type=int, default=None,
                    help='Visualise a single entry by its 0-based line index')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    font = _get_font(FONT_SIZE)

    entries, indices = load_entries(
        args.jsonl,
        include_neg=args.neg,
        sample_n=args.n,
        seed=args.seed,
        all_entries=args.all,
        index=args.index,
    )
    print(f'Visualising {len(entries)} entries → {args.output}')

    saved = 0
    for entry, line_idx in zip(entries, indices):
        vis = draw_entry(entry, font)
        if vis is None:
            continue
        out_name = f'line{line_idx:06d}.png'
        vis.save(os.path.join(args.output, out_name))
        saved += 1

    print(f'Saved {saved} images to {args.output}/')


if __name__ == '__main__':
    main()
