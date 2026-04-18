"""
Phase 2: MedSAM2 mask generation from agent-predicted bboxes.

Pipeline
--------
Step 1 – Core (always runs):
    For each sample, pick the key slice (highest final IoU) from the agent's
    predictions and use its bbox to seed MedSAM2 on that single slice.

Step 2 – Propagation (--propagate flag):
    Starting from the key-slice mask, propagate forward then backward through
    the full volume so every slice gets a mask.

Usage
-----
conda run -n dtos_test python Qwen3_VL/medsam2_phase2.py \
    --infer_jsonl  /BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase6-13/v0-20260415-230620/checkpoint-380/infer_result/20260416-143743.jsonl \
    --ckpt         /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --cfg          /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir   /BDSZ6/private/user/yxd/dtos_output/qwen/medsam2_phase2 \
    --propagate \
    --device       cuda:4
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── MedSAM2 lives in the sibling directory ──────────────────────────────────
MEDSAM2_ROOT = Path(__file__).resolve().parents[1] / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))
from sam2.build_sam import build_sam2_video_predictor_npz  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Image preprocessing (same convention as existing MedSAM2 scripts)
# ────────────────────────────────────────────────────────────────────────────
IMG_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMG_STD  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
TARGET_SIZE = 512


def load_volume(jpeg_dir: str) -> np.ndarray:
    """Load all JPEG slices in a case directory into a uint8 (D, H, W) array."""
    files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith(".jpg"))
    slices = [np.array(Image.open(os.path.join(jpeg_dir, f)).convert("L")) for f in files]
    return np.stack(slices, axis=0)  # (D, H, W)


def preprocess_volume(vol: np.ndarray, device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Convert (D, H, W) uint8 → normalised (D, 3, 512, 512) float tensor.

    Returns (tensor, orig_H, orig_W).
    """
    D, H, W = vol.shape
    out = np.zeros((D, 3, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    for i in range(D):
        img = Image.fromarray(vol[i]).convert("RGB")
        if H != TARGET_SIZE or W != TARGET_SIZE:
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = arr

    tensor = torch.from_numpy(out).to(device)
    mean = IMG_MEAN[:, None, None].to(device)
    std  = IMG_STD[:, None, None].to(device)
    tensor = (tensor - mean) / std
    return tensor, H, W


def scale_bbox(bbox: list[int], orig_H: int, orig_W: int) -> np.ndarray:
    """Scale bbox from original image space to 512×512 if needed."""
    x1, y1, x2, y2 = bbox
    sx = TARGET_SIZE / orig_W
    sy = TARGET_SIZE / orig_H
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# JSONL parsing helpers
# ────────────────────────────────────────────────────────────────────────────

def parse_sample(rec: dict) -> dict:
    """Extract case metadata, final bboxes, and IoU scores from one record.

    Returns a dict with:
        case_dir   : path to JPEG slices
        ann_dir    : path to annotation PNGs
        category   : target category string
        z_bboxes   : {z_index: [x1,y1,x2,y2]}  (final / corrected bboxes)
        z_ious     : {z_index: float}            (final IoU with GT)
        key_z      : z_index with highest final IoU
        all_z      : ordered list of all sampled Z indices from user message
    """
    msgs = rec["messages"]
    imgs = rec["images"]

    # ── derive paths from first input image ─────────────────────────────────
    first_path = imgs[0]["path"]
    case_dir   = os.path.dirname(first_path)
    ann_dir    = case_dir.replace("JPEGImages", "Annotations").replace(".jpg", "")
    # fix: ann_dir may still end with a filename; make sure it's a directory
    if not os.path.isdir(ann_dir):
        ann_dir = os.path.dirname(ann_dir)

    # ── category ────────────────────────────────────────────────────────────
    user_text = msgs[0]["content"]
    m = re.search(r'target structure is: "([^"]+)"', user_text)
    category = m.group(1) if m else "unknown"

    # ── all sampled Z indices (from user message) ────────────────────────────
    all_z = [int(z) for z in re.findall(r"Z=(\d+)", user_text)]

    # ── pair tool_calls with tool_responses to build z→[iou…] history ───────
    queue: list[tuple[str, int | None]] = []
    z_iou_history: dict[int, list[float]] = defaultdict(list)
    z_bbox_history: dict[int, list[list[int]]] = defaultdict(list)

    for msg in msgs:
        role = msg["role"]
        if role == "tool_call":
            try:
                d = json.loads(msg["content"])
                if d["name"] == "add_bbox":
                    z = int(d["arguments"]["z_index"])
                    bbox = [int(v) for v in d["arguments"]["bbox"]]
                    queue.append(("add_bbox", z))
                    z_bbox_history[z].append(bbox)
                else:
                    queue.append(("other", None))
            except Exception:
                queue.append(("error", None))
        elif role == "tool_response" and queue:
            call_type, z = queue.pop(0)
            if call_type == "add_bbox":
                try:
                    d = json.loads(msg["content"])
                    z_iou_history[z].append(float(d["iou_with_gt"]))
                except Exception:
                    pass

    # final (corrected) values
    z_bboxes = {z: bboxes[-1] for z, bboxes in z_bbox_history.items()}
    z_ious   = {z: ious[-1]   for z, ious   in z_iou_history.items()}

    key_z = max(z_ious, key=z_ious.get) if z_ious else (all_z[len(all_z) // 2] if all_z else 0)

    return dict(
        case_dir=case_dir,
        ann_dir=ann_dir,
        category=category,
        z_bboxes=z_bboxes,
        z_ious=z_ious,
        key_z=key_z,
        all_z=all_z,
    )


# ────────────────────────────────────────────────────────────────────────────
# GT mask loading
# ────────────────────────────────────────────────────────────────────────────

def load_gt_mask_2d(ann_dir: str, z: int, label_id: int | None) -> np.ndarray | None:
    """Load 2-D GT mask (512×512 bool) for slice z in ann_dir.

    If label_id is None every non-zero pixel is treated as foreground.
    """
    path = os.path.join(ann_dir, f"{z:05d}.png")
    if not os.path.exists(path):
        return None
    ann = np.array(Image.open(path))
    if label_id is None:
        return ann > 0
    return ann == label_id


def category_to_label(category: str, ann_dir: str) -> int | None:
    """Heuristically infer the integer label from a category string.

    Checks the meta.json next to the Annotations/ directory.
    Falls back to scanning non-zero values if meta.json is absent.
    """
    # try meta.json  (sits two levels up: data_root/train/meta.json)
    meta_path = Path(ann_dir).parents[1] / "meta.json"
    case_name = Path(ann_dir).name
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        objs = meta.get("videos", {}).get(case_name, {}).get("objects", {})
        for lid, info in objs.items():
            if info.get("category", "").lower() == category.lower():
                return int(lid)
    return None


# ────────────────────────────────────────────────────────────────────────────
# Dice / IoU
# ────────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * inter / denom) if denom > 0 else float(pred.sum() == 0 and gt.sum() == 0)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter / union) if union > 0 else float(pred.sum() == 0 and gt.sum() == 0)


# ────────────────────────────────────────────────────────────────────────────
# MedSAM2 inference helpers
# ────────────────────────────────────────────────────────────────────────────

def get_key_mask(predictor, inference_state, frame_idx: int, bbox_512: np.ndarray) -> np.ndarray:
    """Seed MedSAM2 with a bbox on frame_idx; return the uint8 (H,W) mask for that frame.

    Uses add_new_points_or_box, which is the standard seeding entry point.
    The returned uint8 mask can be directly passed to add_new_mask for propagation.
    """
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=1,
        box=bbox_512,
    )
    # out_mask_logits[0] shape: (1, H, W)
    return (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────
# Per-sample processing
# ────────────────────────────────────────────────────────────────────────────

def process_sample(
    predictor,
    sample_meta: dict,
    device: torch.device,
    propagate: bool,
    output_dir: str,
) -> dict:
    """Run MedSAM2 on one case.  Returns a dict of per-slice metrics."""

    case_dir  = sample_meta["case_dir"]
    ann_dir   = sample_meta["ann_dir"]
    category  = sample_meta["category"]
    z_bboxes  = sample_meta["z_bboxes"]
    key_z     = sample_meta["key_z"]

    case_name = Path(case_dir).name

    # ── load & preprocess volume ─────────────────────────────────────────────
    vol = load_volume(case_dir)           # (D, H, W) uint8
    D, orig_H, orig_W = vol.shape
    img_tensor, _, _ = preprocess_volume(vol, device)  # (D,3,512,512)

    # ── get label id for GT comparison ──────────────────────────────────────
    label_id = category_to_label(category, ann_dir)

    # ── scale key-slice bbox to 512 space ───────────────────────────────────
    bbox_raw   = z_bboxes[key_z]
    bbox_512   = scale_bbox(bbox_raw, orig_H, orig_W)

    result_masks: dict[int, np.ndarray] = {}

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        # ── Step 1 + Step 2 forward pass ─────────────────────────────────────
        # Follow the reference script pattern exactly:
        #   init_state → add_new_points_or_box → get key_mask
        #              → add_new_mask (same state) → propagate forward
        #   reset_state → init_state → add_new_mask → propagate backward
        inference_state = predictor.init_state(img_tensor, orig_H, orig_W)

        # Seed with bbox to get the key-slice mask
        key_mask = get_key_mask(predictor, inference_state, key_z, bbox_512)
        result_masks[key_z] = key_mask.astype(bool)

        if propagate:
            # Forward: overwrite key frame with clean mask, then propagate
            _, _, fwd_logits = predictor.add_new_mask(
                inference_state, frame_idx=key_z, obj_id=1, mask=key_mask
            )
            result_masks[key_z] = (fwd_logits[0] > 0.0).cpu().numpy()[0].astype(bool)
            for fidx, _, logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=key_z, reverse=False
            ):
                result_masks[fidx] = (logits[0] > 0.0).cpu().numpy()[0].astype(bool)

        predictor.reset_state(inference_state)

        if propagate:
            # Backward pass: fresh state seeded only with key_mask (no box)
            inference_state = predictor.init_state(img_tensor, orig_H, orig_W)
            predictor.add_new_mask(inference_state, frame_idx=key_z, obj_id=1, mask=key_mask)
            for fidx, _, logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=key_z, reverse=True
            ):
                result_masks[fidx] = (logits[0] > 0.0).cpu().numpy()[0].astype(bool)
            predictor.reset_state(inference_state)

    # ── save masks ───────────────────────────────────────────────────────────
    if output_dir:
        case_out = os.path.join(output_dir, "masks", case_name)
        os.makedirs(case_out, exist_ok=True)
        for z, mask in result_masks.items():
            out_path = os.path.join(case_out, f"{z:05d}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(out_path)

    # ── compute per-slice metrics ────────────────────────────────────────────
    metrics: dict[str, list] = {"z": [], "dice": [], "iou": [], "gt_positive": []}
    for z, pred_mask in sorted(result_masks.items()):
        gt_mask = load_gt_mask_2d(ann_dir, z, label_id)
        if gt_mask is None:
            continue
        dice = dice_score(pred_mask, gt_mask)
        iou  = iou_score(pred_mask, gt_mask)
        metrics["z"].append(z)
        metrics["dice"].append(dice)
        metrics["iou"].append(iou)
        metrics["gt_positive"].append(int(gt_mask.any()))

    return {
        "case": case_name,
        "category": category,
        "key_z": key_z,
        "key_z_iou_bbox": float(sample_meta["z_ious"].get(key_z, -1)),
        "num_slices_masked": len(result_masks),
        "metrics": metrics,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 MedSAM2 integration")
    p.add_argument("--infer_jsonl", required=True, help="Agent inference result JSONL")
    p.add_argument("--ckpt",   required=True, help="MedSAM2 checkpoint .pt")
    p.add_argument("--cfg",    required=True, help="MedSAM2 config YAML")
    p.add_argument("--output_dir", default="./medsam2_phase2_output", help="Where to save masks + metrics")
    p.add_argument("--propagate", action="store_true", help="Run bidirectional propagation (Step 2)")
    p.add_argument("--device",  default="cuda:4", help="Torch device")
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of samples for quick testing")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Config] ckpt    = {args.ckpt}")
    print(f"[Config] cfg     = {args.cfg}")
    print(f"[Config] device  = {device}")
    print(f"[Config] propagate = {args.propagate}")

    # ── build predictor ──────────────────────────────────────────────────────
    # Hydra requires an absolute config path to be prefixed with "//" so it
    # resolves against the filesystem rather than the sam2 package search path.
    cfg_path = args.cfg
    if os.path.isabs(cfg_path) and not cfg_path.startswith("//"):
        cfg_path = "//" + cfg_path

    torch.set_float32_matmul_precision("high")
    # apply_postprocessing=False disables fill_holes_in_mask_scores (fill_hole_area=8),
    # which triggers a CUDA illegal memory access on this platform.
    # Disabling it does not affect segmentation quality in practice.
    predictor = build_sam2_video_predictor_npz(
        cfg_path, args.ckpt, device=device, apply_postprocessing=False
    )

    # ── load inference JSONL ─────────────────────────────────────────────────
    with open(args.infer_jsonl) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"[Data]  {len(records)} samples to process")

    # ── run ──────────────────────────────────────────────────────────────────
    all_results = []
    all_dice: list[float] = []
    all_iou:  list[float] = []
    all_dice_gt_pos: list[float] = []  # only slices where GT has a lesion

    for rec in tqdm(records, desc="MedSAM2"):
        meta = parse_sample(rec)
        try:
            result = process_sample(predictor, meta, device, args.propagate, args.output_dir)
        except Exception as e:
            print(f"  [WARN] {meta['case_dir']} failed: {e}")
            continue

        all_results.append(result)
        dices = result["metrics"]["dice"]
        ious  = result["metrics"]["iou"]
        gt_pos = result["metrics"]["gt_positive"]
        all_dice.extend(dices)
        all_iou.extend(ious)
        all_dice_gt_pos.extend(d for d, g in zip(dices, gt_pos) if g)

    # ── save per-sample results ──────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "results.jsonl")
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # ── print summary ────────────────────────────────────────────────────────
    def arr_stats(vals: list[float]) -> str:
        if not vals:
            return "N/A"
        a = np.array(vals)
        return f"mean={a.mean():.4f}  median={np.median(a):.4f}  std={a.std():.4f}  n={len(a)}"

    mode = "key-slice only" if not args.propagate else "bidirectional propagation"
    print(f"\n{'='*60}")
    print(f"Phase 2 MedSAM2 Segmentation  [{mode}]")
    print(f"{'='*60}")
    print(f"Samples processed : {len(all_results)}")
    print(f"Total slices eval : {len(all_dice)}")
    print(f"\nAll slices (including empty GT):")
    print(f"  Dice  {arr_stats(all_dice)}")
    print(f"  IoU   {arr_stats(all_iou)}")
    print(f"\nGT-positive slices only:")
    print(f"  Dice  {arr_stats(all_dice_gt_pos)}")

    for thresh in [0.5, 0.7, 0.8]:
        hits = sum(1 for d in all_dice_gt_pos if d >= thresh)
        pct = hits / len(all_dice_gt_pos) * 100 if all_dice_gt_pos else 0
        print(f"  Dice≥{thresh}: {pct:.1f}%  ({hits}/{len(all_dice_gt_pos)})")

    print(f"\nSaved masks  → {args.output_dir}/masks/")
    print(f"Saved results→ {results_path}")


if __name__ == "__main__":
    main()
