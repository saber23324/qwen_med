# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 3D MRI lesion segmentation agent based on **Qwen3-VL** + **ms-swift** parallel tool calling. The model observes 10 Z-indexed MRI slices sampled from the lesion range, performs global spatial analysis, annotates bounding boxes (bbox) slice-by-slice, and optionally self-corrects before finalizing. All dependencies are installed in the conda `base` environment.

## Phased Roadmap

| Phase | Task | Input | Output |
|---|---|---|---|
| **Phase 1 (current)** | Predict bbox per slice across 10 sampled slices | 10 slices + Z indices | `[x1,y1,x2,y2]` per slice |
| Phase 2 | Use MedSAM2 to generate masks from bbox, refine with points | Same + bbox results | Per-slice segmentation mask |
| Phase 3 | Dynamic scroll/get_slice interactive segmentation | Single slice + history | Full 3D segmentation |

## Repository Structure

```
medagent/
├── CLAUDE.md                            # This file
├── Agent-support.md                     # ms-swift agent framework docs
└── Qwen3_VL/
    ├── train.sh                         # Standard LoRA training (CUDA 4,5)
    ├── train_mri.sh                     # LoRA training with custom MRI bbox loss
    ├── eval.sh                          # Inference + VLMEvalKit evaluation
    ├── infer.py                         # Batch inference with IoU/mAP metrics
    ├── custom_loss.py                   # MRIBboxLoss: weighted CE + Smooth-L1
    ├── convert_to_swift_grounding.py    # Convert raw M3D data → ms-swift JSONL
    ├── convert_to_swift_grounding2.py   # Alternative conversion pipeline
    ├── visualize_grounding.py           # Visualize bbox predictions on slices
    ├── mri_grounding_train.jsonl        # Training dataset
    ├── mri_grounding_val.jsonl          # Validation dataset
    ├── loss/                            # Loss function module (base, causal_lm, etc.)
    └── custom/                          # Custom model/dataset registration for ms-swift
```

## Common Commands
CUDA_VISIBLE_DEVICES=4,5,6,7
you can only use GPU 4,5,6,7
### Training

```bash
# Standard LoRA fine-tuning (GPUs 4,5; bfloat16; DeepSpeed Zero-2)
bash Qwen3_VL/train.sh

# Training with custom MRI bbox loss (weighted CE + L1 coordinate regression)
bash Qwen3_VL/train_mri.sh
```

### Inference and Evaluation

```bash
# Batch inference with automatic IoU/AP metrics
python Qwen3_VL/infer.py \
    --model /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --ckpt /path/to/adapter \
    --jsonl Qwen3_VL/mri_grounding_val.jsonl \
    --batch-size 4 \
    --coord-mode norm1000

# ms-swift streaming inference
swift infer --adapters <ckpt> --stream true

# VLMEvalKit evaluation (see eval.sh for full config)
swift eval --eval_backend VLMEvalKit
```

### Data Preparation

```bash
# Convert raw M3D data to ms-swift grounding JSONL
python Qwen3_VL/convert_to_swift_grounding.py \
    --data_root /path/to/M3D/data/train \
    --output_dir /path/to/output \
    --train_ratio 0.8 \
    --include_negatives

# Visualize bbox annotations on validation set (first 20 samples)
python Qwen3_VL/visualize_grounding.py \
    --jsonl Qwen3_VL/mri_grounding_val.jsonl \
    --output /tmp/viz \
    --n 20
```

### Format Validation

```bash
python Qwen3_VL/test_grouding.py
python Qwen3_VL/test_dataset.py
```

## Key Server Paths

| Resource | Path |
|---|---|
| Base model | `/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct` |
| Data root | `/BDSZ6/private/user/yxd/data/qwen/` |
| Training output | `/BDSZ6/private/user/yxd/dtos_output/qwen/` |
| MRI bbox output | `/dtos_output/qwen/qwen3vl_mri_bbox` |

---

## Phase 1 Design Spec

### Goal

Show the model 10 Z-annotated MRI slices. It must first perform a spatial analysis, then output an accurate bbox for each slice containing a lesion, explicitly declare "no lesion" for empty slices, and optionally self-correct one bbox if the initial prediction is inaccurate.

### Slice Sampling

Uniformly sample 10 Z indices within the lesion range from the 3D GT mask:

```python
indices = np.linspace(z_min, z_max, 10, dtype=int)
# Example (lesion Z=40~70): [40, 43, 46, 50, 53, 56, 60, 63, 66, 70]
```

Each slice is presented in the `user` message as `Z=xx: <image>`. The Z index must be explicit so the model builds an image ↔ spatial-position binding.

### Action Space (Phase 1 only)

```python
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
            "description": "Confirm all slice bbox annotations are complete (including review). Terminates the task.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
```

Tool evolution across phases:
```
Phase 1  add_bbox  finish_bbox_annotation
Phase 2  add_bbox  add_point  finish_3d_segmentation
Phase 3  add_bbox  add_point  scroll  get_slice  finish_3d_segmentation
```
Phase 1 JSONL format is fully compatible with later phases — only the `tools` field needs to be extended.

### SFT Trajectory Structure

```
Turn 1  [user]       Task instruction + 10 slices (with Z indices)
Turn 2  [assistant]  Global spatial analysis CoT (no tool calls)
Turn 3  [assistant]  Per-slice bbox annotation (multiple parallel tool_calls)
Turn 4  [user]       Rendered bbox feedback per slice (tool_response × N)
Turn 5  [assistant]  Review CoT (assess which slices need correction)
Turn 6  [assistant]  Corrected add_bbox calls (skip if none needed)
Turn 7  [user]       Correction feedback (tool_response, if applicable)
Turn 8  [assistant]  Completion CoT → finish_bbox_annotation
```

**Turn 2 — Global Spatial Analysis CoT (required).** Purpose: build spatial awareness before annotating, not blind per-slice selection. Must include:
1. **Lesion Z range**: which slices have a lesion, which do not
2. **Key slice**: the slice with the largest cross-sectional area (Phase 2 will start segmentation here)
3. **Morphology trend**: how lesion size changes from bottom to top (growing / shrinking / peak in middle)
4. **No-lesion declaration**: explicitly list any sampled slices with no lesion

**Turn 3 — Parallel bbox annotation.** Emit multiple `tool_call`s in a single assistant turn; ms-swift automatically treats them as parallel calls:

```
[assistant]
(CoT text, e.g. "Annotating all slices now:")
tool_call: add_bbox(z=40, bbox=[...])
tool_call: add_bbox(z=43, bbox=[...])
...
tool_call: add_bbox(z=70, bbox=[...])
```

**Turn 5 — Review CoT.**
- **Pass**: bbox covers lesion body; boundary does not exceed normal tissue by more than 10px
- **Fail**: obvious offset (IoU_with_GT < 0.7) or lesion edge clipped

If all slices pass, Turn 5 outputs the completion confirmation directly (no Turn 6).

### Loss Computation Scope

| Content | Compute loss? |
|---|---|
| Spatial analysis CoT (assistant) | Yes |
| Per-slice `add_bbox` tool_calls | Yes |
| Review CoT (assistant) | Yes |
| Correction `add_bbox` tool_calls | Yes |
| `finish_bbox_annotation` tool_call | Yes |
| `tool_response` (rendered feedback) | No |
| `user` (instructions and images) | No |

### ms-swift Training Command

```bash
swift sft \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset path/to/phase1_bbox_train.jsonl \
  --agent_template hermes \
  --loss_scale default \
  --max_length 8192
```

### Custom Loss — MRIBboxLoss (`custom_loss.py`)

- **Weighted CE**: tokens inside `<|box_start|>...<|box_end|>` are upweighted 3.0×
- **Smooth-L1 regression**: compares decoded GT vs. predicted bbox in thousandths-normalized coordinates
- **Formula**: `loss = CE_loss + 0.05 × L1_loss`
- **Enable**: `--loss_type mri_bbox` (configured in `train_mri.sh`)

---

## JSONL Data Format

Training data is stored as JSONL with three top-level fields: `tools`, `messages`, `images`.

```jsonl
{
  "tools": "[...]",
  "messages": [
    {"role": "user",          "content": "Task: annotate liver tumor...\nZ=40: <image>\n..."},
    {"role": "assistant",     "content": "Global spatial analysis:\n..."},
    {"role": "tool_call",     "content": "{\"name\": \"add_bbox\", \"arguments\": {\"z_index\": 40, \"bbox\": [148,112,172,138]}}"},
    ...
    {"role": "tool_response", "content": "{\"z_index\": 40, \"bbox_image\": \"<image>\", \"iou_with_gt\": 0.88}"},
    ...
  ],
  "images": [
    "volumes/case_001/slice_040.png",
    ...
  ]
}
```

The `images` list must contain paths in the exact order that `<image>` tokens appear across all messages (input slices → bbox renders → correction renders).

### Format Validation Constraints

1. `<image>` token count == `images` list length (10 inputs + N renders + correction renders)
2. All `tool_call` content is valid JSON; `name` is only `add_bbox` or `finish_bbox_annotation`
3. `z_index` in every `add_bbox` call must be one of the 10 sampled Z values
4. bbox coordinates satisfy `x1 < x2`, `y1 < y2`, all values within `[0, image_width/height]`
5. Every `tool_call` has a paired `tool_response` in order; none may be missing
6. Turn 2 (first assistant turn) contains no `tool_call` — pure CoT only
7. `finish_bbox_annotation` is present and is the last `tool_call`

---

## Trajectory Generation Script Logic

### Step 1 — Sample slices

```python
def sample_slices(gt_mask_3d, n=10):
    z_with_lesion = np.where(gt_mask_3d.sum(axis=(1, 2)) > 0)[0]
    z_min, z_max = z_with_lesion.min(), z_with_lesion.max()
    return np.linspace(z_min, z_max, n, dtype=int).tolist()
```

### Step 2 — Generate spatial analysis CoT

```python
def generate_spatial_cot(sampled_indices, gt_mask_3d):
    areas = [int(gt_mask_3d[z].sum()) for z in sampled_indices]
    key_z = sampled_indices[int(np.argmax(areas))]
    has_lesion = [z for z, a in zip(sampled_indices, areas) if a > 0]
    no_lesion  = [z for z, a in zip(sampled_indices, areas) if a == 0]

    peak_idx = int(np.argmax(areas))
    if abs(peak_idx - 4) <= 1:
        trend = "gradually increases from both ends toward the middle — near-symmetric ellipsoid"
    elif peak_idx > 5:
        trend = "continuously grows from bottom to top — lesion skewed toward upper slices"
    else:
        trend = "largest at the bottom, shrinking upward — lesion skewed toward lower slices"

    cot = (
        f"Global spatial analysis:\n\n"
        f"Lesion present at Z={has_lesion[0]}~{has_lesion[-1]}, "
        f"{len(has_lesion)} sampled slices contain lesion. "
        f"Z={key_z} has the largest cross-section (key slice). "
        f"Cross-sectional area trend: {trend}.\n"
    )
    if no_lesion:
        cot += f"Z={no_lesion}: no lesion visible — skipping annotation.\n"
    cot += "\nProceeding to annotate all lesion-containing slices:"
    return cot
```

### Step 3 — Extract bbox with jitter

```python
def extract_bbox(gt_2d, jitter_ratio=0.05):
    """Extract GT bounding rectangle and add random jitter to simulate prediction error."""
    rows, cols = np.where(gt_2d > 0)
    x1, y1, x2, y2 = cols.min(), rows.min(), cols.max(), rows.max()
    w, h = x2 - x1, y2 - y1
    def j(v, scale):
        return int(np.clip(v + np.random.uniform(-scale, scale), 0, 512))
    return [j(x1, w*jitter_ratio), j(y1, h*jitter_ratio),
            j(x2, w*jitter_ratio), j(y2, h*jitter_ratio)]
```

### Step 4 — Simulate review logic

```python
def simulate_review(initial_bboxes, gt_mask_3d, iou_threshold=0.75):
    """Identify slices needing correction. In real training, iou_with_gt is returned by the environment."""
    needs_correction = []
    for z, bbox in initial_bboxes.items():
        gt_2d = gt_mask_3d[z]
        if gt_2d.sum() == 0:
            continue
        if compute_bbox_iou(bbox, gt_2d) < iou_threshold:
            needs_correction.append(z)
    return needs_correction

def compute_bbox_iou(bbox, gt_2d):
    x1, y1, x2, y2 = bbox
    gt_rows, gt_cols = np.where(gt_2d > 0)
    gx1, gy1 = gt_cols.min(), gt_rows.min()
    gx2, gy2 = gt_cols.max(), gt_rows.max()
    ix1, iy1 = max(x1, gx1), max(y1, gy1)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (x2-x1)*(y2-y1) + (gx2-gx1)*(gy2-gy1) - inter
    return inter / union if union > 0 else 0.0
```

### Step 5 — Inject 10% failure samples

For 10% of samples, deliberately inject a large bbox offset (IoU < 0.6) on one slice so the review turn triggers a correction:

```python
def inject_bbox_failure(initial_bboxes, target_z, offset_ratio=0.25):
    x1, y1, x2, y2 = initial_bboxes[target_z]
    w, h = x2-x1, y2-y1
    initial_bboxes[target_z] = [
        int(x1 + w*offset_ratio), int(y1 + h*offset_ratio),
        int(x2 + w*offset_ratio), int(y2 + h*offset_ratio),
    ]
    return initial_bboxes
```
