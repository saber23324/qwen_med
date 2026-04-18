# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 3D MRI lesion segmentation agent based on **Qwen3-VL** + **ms-swift** parallel tool calling. The model observes 10 Z-indexed MRI slices sampled from the lesion range, performs global spatial analysis, annotates bounding boxes (bbox) slice-by-slice, and optionally self-corrects before finalizing. All dependencies are installed in the conda `base` environment.

## Phased Roadmap

| Phase | Task | Input | Output | Status |
|---|---|---|---|---|
| **Phase 1** | Predict bbox per slice across 10 sampled slices | 10 slices + Z indices | `[x1,y1,x2,y2]` per slice | **Done** |
| **Phase 2** | Use MedSAM2 to generate masks from bbox, refine with points | Same + bbox results | Per-slice segmentation mask | **Done** |
| **Phase 3** | Dynamic scroll/get_slice interactive segmentation | Single slice + history | Full 3D segmentation | Next |

## Repository Structure

```
medagent/
├── CLAUDE.md                                   # This file
├── Agent-support.md                            # ms-swift agent framework docs
├── MedSAM2/                                    # MedSAM2 submodule
│   ├── checkpoints/MedSAM2/MedSAM2_latest.pt  # Model checkpoint
│   └── sam2/configs/sam2.1_hiera_t512.yaml     # Hydra config (512×512 input)
├── agent/
│   ├── qwen3_mcore.sh                          # Phase 2 training (2-GPU, no DeepSpeed)
│   └── eval.sh                                 # swift infer evaluation script
└── Qwen3_VL/
    ├── train.sh                                # Phase 1 LoRA training (GPU 4,5)
    ├── train_phase2.sh                         # Phase 2 LoRA training (GPU 4,5,6,7)
    ├── infer.py                                # Phase 1 batch inference (IoU/mAP)
    ├── infer_phase2.py                         # Phase 2 agent inference (Dice/HD95)
    ├── visualize_phase2.py                     # Phase 2 result visualizer
    ├── custom_loss.py                          # MRIBboxLoss: weighted CE + Smooth-L1
    ├── convert_to_swift_grounding.py           # Phase 1 data: M3D → ms-swift JSONL
    ├── convert_to_agent_trajectory.py          # Phase 1 agent trajectory generator
    ├── convert_to_agent_trajectory_phase2.py   # Phase 2 trajectory generator
    ├── medsam2_phase2.py                       # Standalone MedSAM2 inference + Dice
    ├── PHASE2_WORKFLOW.md                      # Full Phase 2 documentation
    ├── mri_grounding_train.jsonl               # Phase 1 training dataset
    ├── mri_grounding_val.jsonl                 # Phase 1 validation dataset
    ├── loss/                                   # Loss function module
    └── custom/                                 # Custom model/dataset for ms-swift
```

## Common Commands

> **GPU constraint**: only GPUs 0,7 are available (`CUDA_VISIBLE_DEVICES=0,7`).
> **Conda environments**: `qwen3` for training/ms-swift and MedSAM2 inference.

### Phase 1 Training

```bash
bash Qwen3_VL/train.sh          # LoRA, GPU 4,5, DeepSpeed Zero-2
```

### Phase 2 Training

```bash
bash Qwen3_VL/train_phase2.sh   # LoRA, GPU 4,5,6,7, DeepSpeed Zero-2
# or
bash agent/qwen3_mcore.sh       # LoRA, GPU 3,7, no DeepSpeed
```

### Phase 2 Dataset Generation

```bash
# Full run with real MedSAM2 masks (dtos_test env):
conda run -n dtos_test python3 Qwen3_VL/convert_to_agent_trajectory_phase2.py \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --output_dir /BDSZ6/private/user/yxd/data/qwen/agent_phase2 \
    --ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --device cuda:4

# Quick validation (3 samples, no GPU):
conda run -n qwen3 python3 Qwen3_VL/convert_to_agent_trajectory_phase2.py \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --output_dir /tmp/phase2_test --max_samples 3 --device cpu
```

### Phase 2 Inference and Evaluation

```bash
# Agent-loop inference with Dice/HD95/Precision/Recall (qwen3 env):
conda run -n qwen3 python3 Qwen3_VL/infer_phase2.py \
    --model  /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --ckpt   /path/to/checkpoint \
    --val_jsonl /BDSZ6/private/user/yxd/data/qwen/agent_phase2/agent_val.jsonl \
    --data_root /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir /tmp/phase2_eval --device cuda:4

# Swift batch inference (eval.sh pattern):
bash agent/eval.sh   # uses --load_data_args true to read val set from training args

# Visualize swift infer results (re-runs MedSAM2 with model's predicted bboxes):
conda run -n dtos_test python3 Qwen3_VL/visualize_phase2.py \
    --infer_jsonl /path/to/infer_result/*.jsonl \
    --data_root   /BDSZ6/private/user/yxd/data/M3D/data_6-13/train \
    --medsam2_ckpt /home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt \
    --medsam2_cfg  /home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
    --output_dir /tmp/phase2_vis --device cuda:4
```

### Phase 1 Inference

```bash
python Qwen3_VL/infer.py \
    --model /BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct \
    --ckpt /path/to/adapter \
    --jsonl Qwen3_VL/mri_grounding_val.jsonl \
    --batch-size 4 --coord-mode norm1000
```

## Key Server Paths

| Resource | Path |
|---|---|
| Base model | `/BDSZ6/private/user/yxd/models/Qwen/Qwen3-VL-8B-Instruct` |
| M3D data (phases 1–2) | `/BDSZ6/private/user/yxd/data/M3D/data_6-13/train` |
| Phase 2 dataset | `/BDSZ6/private/user/yxd/data/qwen/agent_phase2/` |
| Phase 2 renders | `/BDSZ6/private/user/yxd/data/qwen/agent_phase2/renders/` |
| Training output | `/BDSZ6/private/user/yxd/dtos_output/qwen/` |
| Phase 2 checkpoints | `/BDSZ6/private/user/yxd/dtos_output/qwen/agent_phase2/` |
| MedSAM2 checkpoint | `/home/yxd/medagent/MedSAM2/checkpoints/MedSAM2/MedSAM2_latest.pt` |
| MedSAM2 config | `/home/yxd/medagent/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml` |

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

---

## Phase 2 Design Spec

> Full documentation: `Qwen3_VL/PHASE2_WORKFLOW.md`

### Goal

Extend Phase 1 to full 3D segmentation. After the agent annotates bboxes on 10 sampled slices and verifies them, it seeds MedSAM2 from the key (largest cross-section) slice and propagates the mask forward and backward through the entire volume. Poorly-segmented slices (Dice < 0.70) are refined with point prompts.

### Tool Schema (Phase 2)

```
add_bbox(z_index, bbox)                       # same as Phase 1
run_medsam2(key_z, bbox)                      # seeds 3D propagation from key slice
add_point(z_index, points, labels)            # refine one slice with fg/bg points
finish_3d_segmentation()                      # terminal tool
```

`run_medsam2` makes **one** tool_call → **one** tool_response that packs ALL N mask images into a `slices` array, satisfying ms-swift's 1:1 pairing requirement.

### Trajectory Structure (15 turns)

```
Turn 1   [user]            Task + 10 Z-indexed MRI slices
Turn 2   [assistant]       Global spatial analysis CoT
Turn 3   [tool_call × N]   Parallel add_bbox (N = lesion slices)
Turn 4   [tool_resp × N]   Bbox render + IoU per slice
Turn 5   [assistant]       Bbox review CoT (PASS/FAIL per slice)
Turn 6   [tool_call × M]   Correction add_bbox (M = failing slices, skip if 0)
Turn 7   [tool_resp × M]   Corrected bbox renders
Turn 8   [assistant]       "Verified. Initiating 3D segmentation from Z=K …"
Turn 9   [tool_call]       run_medsam2(key_z=K, bbox=[...])
Turn 10  [tool_response]   All N mask images + Dice per slice
Turn 11  [assistant]       Mask review CoT
Turn 12  [tool_call × P]   Parallel add_point (P = Dice < 0.70 slices)
Turn 13  [tool_resp × P]   Refined mask + Dice per slice
Turn 14  [assistant]       "All slices pass. Segmentation complete."
Turn 15  [tool_call]       finish_3d_segmentation()
```

### MedSAM2 Propagation Protocol

Three separate init/propagate/reset cycles to avoid state contamination:

```python
# Phase A: get key-slice mask from bbox
state = predictor.init_state(img_tensor, orig_H, orig_W)
_, _, logits = predictor.add_new_points_or_box(state, frame_idx=key_z, obj_id=1, box=bbox_512)
key_mask = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
predictor.reset_state(state)

# Phase B: forward propagation seeded with key_mask
state = predictor.init_state(img_tensor, orig_H, orig_W)
predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=False):
    result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
predictor.reset_state(state)

# Phase C: backward propagation
state = predictor.init_state(img_tensor, orig_H, orig_W)
predictor.add_new_mask(state, frame_idx=key_z, obj_id=1, mask=key_mask)
for fidx, _, lg in predictor.propagate_in_video(state, start_frame_idx=key_z, reverse=True):
    result[fidx] = (lg[0] > 0.0).cpu().numpy()[0].astype(bool)
predictor.reset_state(state)
```

**Critical**: always pass `apply_postprocessing=False` to `build_sam2_video_predictor_npz`. The default `fill_holes_in_mask_scores` CUDA kernel causes illegal memory access on this hardware.

**Hydra config path fix**: absolute paths need `//` prefix:
```python
if os.path.isabs(cfg_path) and not cfg_path.startswith('//'):
    cfg_path = '//' + cfg_path
```

### Dataset Statistics (Phase 2)

| Split | Samples | add_bbox | run_medsam2 | add_point | finish |
|---|---|---|---|---|---|
| Train | 298 | 2900 | 298 | 705 | 298 |
| Val | 33 | 326 | 33 | 55 | 33 |

- Mean tokens/sample: ~7,110 (text ~1,924 + ~32 images)
- Input slices: 512×512 (~333 tokens each)
- Render images (bbox overlays + mask overlays): **256×256** (~84 tokens each)

### Token Length Issue and Fix

**Problem**: with `max_length=8192`, only 4/298 samples survived — trajectories have ~32 images and at 333 tokens/image exceed 8192 before any text.

**Fix**:
1. Render images (bbox/mask overlays) saved at **256×256** instead of 512×512 → ~84 tokens each
2. `max_length` increased to **12288** in training scripts

### Training Parameters (Phase 2)

```bash
--agent_template hermes   # required for Qwen3 parallel tool calling
--max_length 12288        # was 8192; needed for 30+ images per trajectory
--lora_rank 8
--freeze_vit true
--freeze_aligner true
--packing true
--attn_impl flash_attn
--deepspeed zero2         # (train_phase2.sh only)
```

### Evaluation Results (checkpoint-300, val set n=33)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Dice | 0.826 | 0.127 | 0.303 | 0.953 |
| Precision | 0.849 | 0.151 | 0.186 | 0.991 |
| Recall | 0.826 | 0.110 | 0.530 | 0.978 |
| HD95 (px) | 9.50 | 12.91 | 1.0 | 48.6 |

Evaluated by re-running MedSAM2 with the model's predicted `run_medsam2(key_z, bbox)` args on all annotated Z frames (not just the 10 sampled). Visualizations saved to `vis/figures/{vid}.png` (per-case grids) and `vis/summary.png`.

### Annotation Matching (val JSONL → mask_dict)

Index-based split matching is unreliable because `build_trajectory` skips some annotations. Always match by `(vid, caption)`:

```python
vid     = Path(rec['images'][0]['path']).parent.name
caption = re.search(r'The target structure is:\s*"([^"]+)"', user_text).group(1)
anno    = anno_lookup[(vid, caption)]
```

---

## Phase 3 Design Notes (upcoming)

Phase 3 adds interactive scrolling: the agent can call `scroll(delta)` and `get_slice(z)` to navigate the volume on demand, rather than being shown all 10 slices upfront. This enables finer-grained control for difficult cases and is compatible with the Phase 2 tool set (just add `scroll` and `get_slice`).

Planned tool set:
```
add_bbox   run_medsam2   add_point   scroll(delta)   get_slice(z)   finish_3d_segmentation
```

The Phase 2 dataset JSONL format is already forward-compatible — only the `tools` field needs extension.
